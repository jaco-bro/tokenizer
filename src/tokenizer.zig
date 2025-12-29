//! tokenizer.zig - BPE tokenizer compatible with tiktoken and huggingface models
//!
//! Copyright 2025 Joe

const std = @import("std");
const stdout = std.io.getStdOut().writer();
const pcre2 = @cImport({
    @cDefine("PCRE2_CODE_UNIT_WIDTH", "8");
    @cInclude("pcre2.h");
});

pub const Regex = struct {
    const Self = @This();
    pattern: ?*pcre2.pcre2_code_8,
    match_data: ?*pcre2.pcre2_match_data_8,

    pub fn init(pattern_str: []const u8) !Self {
        var error_code: c_int = undefined;
        var error_offset: usize = undefined;
        const options = pcre2.PCRE2_UTF | pcre2.PCRE2_UCP;
        const pattern = pcre2.pcre2_compile_8(pattern_str.ptr, pattern_str.len, options, &error_code, &error_offset, null);
        if (pattern == null) {
            var error_message: [256]u8 = undefined;
            _ = pcre2.pcre2_get_error_message_8(error_code, &error_message, error_message.len);
            std.debug.print("PCRE2 compilation error at offset {d}: {s}\n", .{ error_offset, &error_message });
            return error.RegexCompilationFailed;
        }
        const match_data = pcre2.pcre2_match_data_create_from_pattern_8(pattern, null);
        if (match_data == null) {
            pcre2.pcre2_code_free_8(pattern);
            return error.MatchDataCreationFailed;
        }
        return Self{
            .pattern = pattern,
            .match_data = match_data,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.match_data) |md| {
            pcre2.pcre2_match_data_free_8(md);
            self.match_data = null;
        }
        if (self.pattern) |p| {
            pcre2.pcre2_code_free_8(p);
            self.pattern = null;
        }
    }

    pub fn match(self: *Self, text: []const u8, start_pos: usize) !?struct { start: usize, end: usize } {
        const rc = pcre2.pcre2_match_8(self.pattern, text.ptr, text.len, start_pos, 0, self.match_data, null);
        if (rc < 0) {
            if (rc == pcre2.PCRE2_ERROR_NOMATCH) return null;
            return error.MatchingError;
        }
        const ovector = pcre2.pcre2_get_ovector_pointer_8(self.match_data);
        const match_start = ovector[0];
        const match_end = ovector[1];
        if (match_end <= match_start) return null;
        return .{ .start = match_start, .end = match_end };
    }
};

pub const Tokenizer = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    pattern_regex: ?Regex,
    special_regex: ?Regex,
    vocab: std.StringHashMap(u32),
    id_to_token: std.AutoHashMap(u32, []const u8),
    specials: []const []const u8,
    verbose: bool,

    pub fn init(allocator: std.mem.Allocator, repo_name: []const u8, model_name: []const u8, verbose: bool) !Self {
        if (!std.mem.eql(u8, repo_name, "local")) try download(allocator, repo_name, model_name, null);
        const json_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_name});
        defer allocator.free(json_path);
        const json_content = try std.fs.cwd().readFileAlloc(allocator, json_path, 100 * 1024 * 1024);
        defer allocator.free(json_content);

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_content, .{});
        defer parsed.deinit();

        var pattern_regex: ?Regex = null;
        const pre_tokenizer = parsed.value.object.get("pre_tokenizer");
        if (pre_tokenizer != null) {
            const pretokenizers = pre_tokenizer.?.object.get("pretokenizers");
            if (pretokenizers != null and pretokenizers.?.array.items.len > 0) {
                const pattern_obj = pretokenizers.?.array.items[0].object.get("pattern");
                if (pattern_obj != null and pattern_obj.?.object.get("Regex") != null) {
                    const pattern = pattern_obj.?.object.get("Regex").?.string;
                    pattern_regex = try Regex.init(pattern);
                }
            }
        }

        var vocab = std.StringHashMap(u32).init(allocator);
        errdefer vocab.deinit();
        var id_to_token = std.AutoHashMap(u32, []const u8).init(allocator);
        errdefer id_to_token.deinit();

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();
        var decoder_map = createDecoderMap(arena_allocator);

        const Replacement = struct { bytes: []const u8, replacement: u8 };
        const explicit_replacements = [_]Replacement{
            .{ .bytes = "\xE2\x96\x81", .replacement = ' ' },
            .{ .bytes = "\xC4\xA0", .replacement = ' ' },
            .{ .bytes = "\xC4\x8A", .replacement = '\n' },
            .{ .bytes = "\xC4\x89", .replacement = '\t' },
        };

        var vocab_iter = parsed.value.object.get("model").?.object.get("vocab").?.object.iterator();
        while (vocab_iter.next()) |entry| {
            const token_str = entry.key_ptr.*;
            const id = @as(u32, @intCast(entry.value_ptr.*.integer));

            var buf = std.ArrayList(u8).init(allocator);
            defer buf.deinit();

            var i: usize = 0;
            while (i < token_str.len) {
                var replaced = false;
                for (explicit_replacements) |repl| {
                    if (std.mem.startsWith(u8, token_str[i..], repl.bytes)) {
                        try buf.append(repl.replacement);
                        i += repl.bytes.len;
                        replaced = true;
                        break;
                    }
                }
                if (replaced) continue;

                const char_len = std.unicode.utf8ByteSequenceLength(token_str[i]) catch 1;
                const char_slice = token_str[i .. i + char_len];
                const cp = std.unicode.utf8Decode(char_slice) catch 0;

                if (decoder_map.get(cp)) |byte_val| {
                    try buf.append(byte_val);
                } else {
                    try buf.appendSlice(char_slice);
                }
                i += char_len;
            }

            const processed_token = try buf.toOwnedSlice();
            try vocab.put(processed_token, id);
            try id_to_token.put(id, processed_token);
        }

        var special_tokens = std.ArrayList([]const u8).init(allocator);
        defer special_tokens.deinit();
        const added_tokens = parsed.value.object.get("added_tokens").?.array;
        for (added_tokens.items) |token| {
            if (token != .object) continue;
            const content = token.object.get("content") orelse continue;
            const id = token.object.get("id") orelse continue;
            if (content != .string or id != .integer) continue;
            const token_id = @as(u32, @intCast(id.integer));
            const token_str = content.string;

            if (vocab.get(token_str)) |old_id| {
                if (old_id != token_id and verbose) {
                    std.debug.print("Duplicate: {s} {d} -> {d}\n", .{ token_str, old_id, token_id });
                }
            }

            const token_copy = try allocator.dupe(u8, token_str);
            try vocab.put(token_copy, token_id);
            try id_to_token.put(token_id, token_copy);

            if (!std.mem.startsWith(u8, token_str, "<unused")) {
                try special_tokens.append(token_copy);
            }
        }

        const specials = try allocator.dupe([]const u8, special_tokens.items);
        const special_regex = try createSpecialRegex(allocator, specials);
        return Self{
            .allocator = allocator,
            .pattern_regex = pattern_regex,
            .special_regex = special_regex,
            .vocab = vocab,
            .id_to_token = id_to_token,
            .specials = specials,
            .verbose = verbose,
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.vocab.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.vocab.deinit();
        self.id_to_token.deinit();
        if (self.special_regex) |*regex| {
            regex.deinit();
        }
        if (self.pattern_regex) |*regex| {
            regex.deinit();
        }
        self.allocator.free(self.specials);
    }

    fn createDecoderMap(allocator: std.mem.Allocator) std.AutoHashMap(u21, u8) {
        var map = std.AutoHashMap(u21, u8).init(allocator);
        var n: u21 = 0;
        var b: u16 = 0;
        while (b < 256) : (b += 1) {
            const byte_val = @as(u8, @intCast(b));
            var cp: u21 = 0;
            if ((b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)) {
                cp = byte_val;
            } else {
                cp = 256 + n;
                n += 1;
            }
            map.put(cp, byte_val) catch {};
        }
        return map;
    }

    fn createSpecialRegex(allocator: std.mem.Allocator, specials: []const []const u8) !?Regex {
        if (specials.len == 0) return null;

        const sorted_specials = try allocator.dupe([]const u8, specials);
        defer allocator.free(sorted_specials);
        std.mem.sort([]const u8, sorted_specials, {}, struct {
            fn lessThan(_: void, a: []const u8, b: []const u8) bool {
                return a.len > b.len;
            }
        }.lessThan);

        var pattern = std.ArrayList(u8).init(allocator);
        defer pattern.deinit();
        var first = true;

        for (sorted_specials) |special| {
            if (!first) {
                try pattern.append('|');
            } else {
                first = false;
            }
            for (special) |char| {
                if (std.mem.indexOfScalar(u8, "\\^$.|?*+()[{", char) != null) try pattern.append('\\');
                try pattern.append(char);
            }
        }
        if (pattern.items.len == 0) return null;
        return try Regex.init(pattern.items);
    }

    fn splitWithSpecials(self: *Self, text: []const u8) !std.ArrayList([]const u8) {
        var result = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (result.items) |item| self.allocator.free(item);
            result.deinit();
        }
        if (self.special_regex == null) {
            try result.append(try self.allocator.dupe(u8, text));
            return result;
        }
        var last_end: usize = 0;
        var start: usize = 0;
        while (start < text.len) {
            const match_result = try self.special_regex.?.match(text, start);
            if (match_result == null) break;
            const match = match_result.?;
            if (match.start > last_end) {
                const non_match = try self.allocator.dupe(u8, text[last_end..match.start]);
                try result.append(non_match);
            }
            const matched_special = try self.allocator.dupe(u8, text[match.start..match.end]);
            try result.append(matched_special);
            last_end = match.end;
            start = match.end;
        }
        if (last_end < text.len) {
            const remaining = try self.allocator.dupe(u8, text[last_end..]);
            try result.append(remaining);
        }
        return result;
    }

    fn splitWithPattern(self: *Self, text: []const u8) !std.ArrayList([]const u8) {
        var result = std.ArrayList([]const u8).init(self.allocator);
        errdefer {
            for (result.items) |item| self.allocator.free(item);
            result.deinit();
        }
        if (self.pattern_regex == null) {
            try result.append(try self.allocator.dupe(u8, text));
            return result;
        }
        var start: usize = 0;
        while (start < text.len) {
            const match_result = try self.pattern_regex.?.match(text, start);
            if (match_result == null) {
                if (start < text.len) {
                    const remaining = try self.allocator.dupe(u8, text[start..]);
                    try result.append(remaining);
                }
                break;
            }
            const match = match_result.?;

            if (match.start > start) {
                const gap = try self.allocator.dupe(u8, text[start..match.start]);
                try result.append(gap);
            }

            const matched_text = try self.allocator.dupe(u8, text[match.start..match.end]);
            try result.append(matched_text);
            start = match.end;
        }
        return result;
    }

    fn bpeMerges(self: *Self, token: []const u8) !std.ArrayList(u32) {
        var result = std.ArrayList(u32).init(self.allocator);
        errdefer result.deinit();
        if (self.vocab.get(token)) |id| {
            try result.append(id);
            return result;
        }
        if (token.len == 0) return result;

        var boundaries = std.ArrayList(usize).init(self.allocator);
        defer boundaries.deinit();
        try boundaries.append(0);
        for (1..token.len + 1) |i| {
            try boundaries.append(i);
        }

        var did_merge = true;
        while (did_merge and boundaries.items.len > 2) {
            did_merge = false;
            var best_idx: ?usize = null;
            var best_rank: u32 = std.math.maxInt(u32);
            for (0..boundaries.items.len - 2) |i| {
                const start = boundaries.items[i];
                const end = boundaries.items[i + 2];
                const pair = token[start..end];
                if (self.vocab.get(pair)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }
            if (best_idx) |i| {
                did_merge = true;
                _ = boundaries.orderedRemove(i + 1);
            }
        }

        const num_tokens = boundaries.items.len - 1;
        for (0..num_tokens) |i| {
            const start = boundaries.items[i];
            const end = boundaries.items[i + 1];
            const segment = token[start..end];
            if (self.vocab.get(segment)) |id| {
                try result.append(id);
            } else {
                if (self.verbose) std.debug.print("Warning: No ID found for segment: '{s}' (Bytes: {any})\n", .{ segment, segment });
                try result.append(0);
            }
        }
        return result;
    }

    pub fn encode(self: *Self, text: []const u8) ![]const u32 {
        if (self.verbose) std.debug.print("\nStr: {s} -> ", .{text});
        var result = std.ArrayList(u32).init(self.allocator);
        errdefer result.deinit();
        var parts = try self.splitWithSpecials(text);
        defer {
            for (parts.items) |part| self.allocator.free(part);
            parts.deinit();
        }
        for (parts.items) |part| {
            if (self.vocab.get(part)) |id| {
                if (self.verbose) std.debug.print("{s}(special) ", .{part});
                try result.append(id);
                continue;
            }
            var tokens = try self.splitWithPattern(part);
            defer {
                for (tokens.items) |token| self.allocator.free(token);
                tokens.deinit();
            }
            for (tokens.items) |token| {
                if (self.verbose) std.debug.print("{s}(regular) ", .{token});
                var token_ids = try self.bpeMerges(token);
                defer token_ids.deinit();
                try result.appendSlice(token_ids.items);
            }
        }
        if (self.verbose) std.debug.print("\n\nIDs: ", .{});
        if (self.verbose) try stdout.print("{any}", .{result.items});
        if (self.verbose) std.debug.print("\n\n", .{});
        return result.toOwnedSlice();
    }

    pub fn decode(self: *Self, token_ids: []const u32) ![]const u8 {
        if (self.verbose) std.debug.print("\nIDs: {any} -> ", .{token_ids});
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();
        for (token_ids) |id| {
            if (self.id_to_token.get(id)) |token| {
                if (self.verbose) std.debug.print("{s}({d}) ", .{ token, id });
                try result.appendSlice(token);
            } else {
                if (self.verbose) std.debug.print("Warning: Unknown token ID: {d}\n", .{id});
            }
        }
        if (self.verbose) std.debug.print("\n\nStr: ", .{});
        if (self.verbose) try stdout.print("{s}", .{result.items});
        if (self.verbose) std.debug.print("\n\n", .{});
        return result.toOwnedSlice();
    }

    pub fn encodeChat(self: *Self, chat_format: ?[]const u8, replacements: []const []const u8) ![]const u32 {
        if (self.verbose) std.debug.print("\nReplacements #: {d}\nChat format: ", .{replacements.len});
        if (chat_format == null) {
            if (self.verbose) std.debug.print("null\n", .{});
            return self.encode(replacements[0]);
        }
        if (self.verbose) std.debug.print("{s}\n", .{chat_format.?});
        const format = chat_format.?;
        const formatted = try formatDynamic(self.allocator, format, replacements);
        defer self.allocator.free(formatted);
        return self.encode(formatted);
    }
};

fn formatDynamic(allocator: std.mem.Allocator, chat_fmt: []const u8, replacements: []const []const u8) ![]const u8 {
    var segments = std.ArrayList([]const u8).init(allocator);
    defer segments.deinit();
    var splitter = std.mem.splitSequence(u8, chat_fmt, "{s}");
    while (splitter.next()) |segment| {
        try segments.append(segment);
    }
    const expected_replacements = segments.items.len - 1;
    if (replacements.len != expected_replacements) {
        return error.ReplacementCountMismatch;
    }
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();
    for (segments.items[0 .. segments.items.len - 1], replacements) |segment, replacement| {
        try result.appendSlice(segment);
        try result.appendSlice(replacement);
    }
    try result.appendSlice(segments.items[segments.items.len - 1]);
    return try result.toOwnedSlice();
}

fn download(allocator: std.mem.Allocator, repo_name: []const u8, model_name: []const u8, file_names: ?[]const []const u8) !void {
    const default_filenames = [_][]const u8{ "tokenizer_config.json", "tokenizer.json" };
    const filenames = if (file_names) |f| f else &default_filenames;
    var args = std.ArrayList([]const u8).init(allocator);
    defer args.deinit();
    try args.append("curl");
    try args.append("--location");
    try args.append("--parallel");
    var paths_to_free = std.ArrayList([]const u8).init(allocator);
    defer {
        for (paths_to_free.items) |path| allocator.free(path);
        paths_to_free.deinit();
    }
    var all_exist = true;
    for (filenames) |filename| {
        const local_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_name, filename });
        try paths_to_free.append(local_path);
        std.debug.print("File '{s}'", .{local_path});
        if (fileExists(local_path)) {
            std.debug.print(" already exists. Skipping download.\n", .{});
        } else {
            all_exist = false;
            std.debug.print(" is missing.\n", .{});
            const url_path = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/{s}/resolve/main/{s}", .{ repo_name, model_name, filename });
            try paths_to_free.append(url_path);
            try args.append(url_path);
            try args.append("-o");
            try args.append(local_path);
        }
    }
    if (all_exist) return;
    try mkdir(model_name);
    var proc = std.process.Child.init(args.items, allocator);
    try proc.spawn();
    const result = try proc.wait();
    switch (result) {
        .Exited => |code| {
            if (code == 0) std.debug.print("Download successful.\n", .{}) else {
                std.debug.print("Download failed with exit code: {d}\n", .{code});
                return error.DownloadFailed;
            }
        },
        else => {
            std.debug.print("Download process terminated abnormally.\n", .{});
            return error.DownloadFailed;
        },
    }
}

fn mkdir(dir: []const u8) !void {
    std.fs.cwd().makeDir(dir) catch |err| {
        if (err == error.PathAlreadyExists) {
            std.debug.print("Directory '{s}' already exists.\n", .{dir});
            return;
        } else {
            return err;
        }
    };
    std.debug.print("Directory '{s}' created successfully.\n", .{dir});
}

fn fileExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn printUsage(program_name: []const u8) void {
    std.debug.print(
        \\Usage: {s} [--model MODEL_NAME] COMMAND INPUT
        \\
        \\Commands:
        \\  --encode TEXT    Encode text to tokens
        \\  --decode TOKENS  Decode tokens to text
        \\
        \\Options:
        \\  --model MODEL 
        \\    Specify the model name (default: Qwen2.5-Coder-1.5B-4bit)
        \\
        \\Examples:
        \\  {s} --encode "hello world"
        \\  {s} --decode "[14990, 1879]"
        \\  {s} --model "phi-4-4bit" --encode "hello world"
        \\  {s} --model "phi-4-4bit" --decode "15339 1917"
        \\
    , .{ program_name, program_name, program_name, program_name, program_name });
}

pub const TokenizerHandle = ?*anyopaque;

export fn create_tokenizer(repo_name: [*:0]const u8, model_name: [*:0]const u8, verbose: bool) TokenizerHandle {
    const allocator = std.heap.c_allocator;
    const repo_name_slice = std.mem.span(repo_name);
    const model_name_slice = std.mem.span(model_name);
    const tokenizer = allocator.create(Tokenizer) catch return null;
    tokenizer.* = Tokenizer.init(allocator, repo_name_slice, model_name_slice, verbose) catch {
        allocator.destroy(tokenizer);
        return null;
    };
    return @ptrCast(tokenizer);
}

export fn encode_text(handle: TokenizerHandle, text: [*:0]const u8, out_tokens: [*]u32, max_tokens: usize) usize {
    if (handle == null) return 0;
    const tokenizer = @as(*Tokenizer, @ptrCast(@alignCast(handle.?)));
    const text_slice = std.mem.span(text);
    const token_ids = tokenizer.encode(text_slice) catch return 0;
    defer tokenizer.allocator.free(token_ids);
    const tokens_to_copy = @min(token_ids.len, max_tokens);
    if (tokens_to_copy > 0) {
        @memcpy(out_tokens[0..tokens_to_copy], token_ids[0..tokens_to_copy]);
    }
    return token_ids.len;
}

export fn decode_tokens(handle: TokenizerHandle, tokens: [*]const u32, token_count: usize, out_text: [*]u8, max_text_len: usize) usize {
    if (handle == null) return 0;
    const tokenizer = @as(*Tokenizer, @ptrCast(@alignCast(handle.?)));
    const tokens_slice = tokens[0..token_count];
    const text = tokenizer.decode(tokens_slice) catch return 0;
    defer tokenizer.allocator.free(text);
    const chars_to_copy = @min(text.len, max_text_len - 1);
    if (chars_to_copy > 0) {
        @memcpy(out_text[0..chars_to_copy], text[0..chars_to_copy]);
    }
    out_text[chars_to_copy] = 0;
    return text.len;
}

export fn free_tokenizer(handle: TokenizerHandle) void {
    if (handle == null) return;
    const tokenizer = @as(*Tokenizer, @ptrCast(@alignCast(handle.?)));
    tokenizer.deinit();
    std.heap.c_allocator.destroy(tokenizer);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Changed to 'var' to allow modification via arguments
    var repo_name: []const u8 = "mlx-community";
    var model_name: []const u8 = "Qwen2.5-Coder-1.5B-4bit";

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage(args[0]);
        return error.InvalidArguments;
    }

    var i: usize = 1;
    var command: ?[]const u8 = null;
    var input: ?[]const u8 = null;

    while (i < args.len) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --model requires a model name\n", .{});
                printUsage(args[0]);
                return error.MissingModelName;
            }
            model_name = args[i];
        } else if (std.mem.eql(u8, arg, "--repo")) {
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: --repo requires a repository name\n", .{});
                printUsage(args[0]);
                return error.MissingRepoName;
            }
            repo_name = args[i];
        } else if (std.mem.eql(u8, arg, "--encode") or std.mem.eql(u8, arg, "--decode")) {
            command = arg;
            i += 1;
            if (i >= args.len) {
                std.debug.print("Error: {s} requires input\n", .{arg});
                printUsage(args[0]);
                return error.MissingInput;
            }
            input = args[i];
        } else {
            std.debug.print("Unknown argument: {s}\n", .{arg});
            printUsage(args[0]);
            return error.UnknownArgument;
        }

        i += 1;
    }

    if (command == null or input == null) {
        printUsage(args[0]);
        return error.MissingRequiredArguments;
    }

    std.debug.print("Using repo: {s}\n", .{repo_name});
    std.debug.print("Using model: {s}\n", .{model_name});

    var tokenizer = try Tokenizer.init(allocator, repo_name, model_name, true);
    defer tokenizer.deinit();

    if (std.mem.eql(u8, command.?, "--encode")) {
        const encode_result = try tokenizer.encode(input.?);
        defer allocator.free(encode_result);
    } else if (std.mem.eql(u8, command.?, "--decode")) {
        var tokenList = std.ArrayList(u32).init(allocator);
        defer tokenList.deinit();
        var iterator = std.mem.tokenizeAny(u8, input.?, "{}[], ");
        while (iterator.next()) |token_str| {
            const token = try std.fmt.parseUnsigned(u32, token_str, 10);
            try tokenList.append(token);
        }
        const decode_result = try tokenizer.decode(tokenList.items);
        defer allocator.free(decode_result);
    } else {
        std.debug.print("Invalid command: {s}\n", .{command.?});
        printUsage(args[0]);
        return error.InvalidCommand;
    }
}

test "Tokenizer round-trip" {
    const allocator = std.testing.allocator;
    const repo_name = "mlx-community";
    const model_name = "Qwen2.5-Coder-1.5B-4bit";
    var tokenizer = try Tokenizer.init(allocator, repo_name, model_name, true);
    defer tokenizer.deinit();
    const text =
        \\<|fim_prefix|>def quicksort(arr):
        \\    if len(arr) <= 1:
        \\        return arr
        \\    pivot = arr[len(arr) // 2]
        \\    <|fim_suffix|>
        \\    middle = [x for x in arr if x == pivot]
        \\    right = [x for x in arr if x > pivot]
        \\    return quicksort(left) + middle + quicksort(right)<|fim_middle|>
    ;
    std.debug.print("Original text: \"{s}\"\n\n", .{text});
    std.debug.print("1. Encoding text to token IDs...\n", .{});
    const token_ids = try tokenizer.encode(text);
    defer allocator.free(token_ids);
    std.debug.print("   Result: {d} tokens\n", .{token_ids});
    for (token_ids, 0..) |id, i| {
        if (tokenizer.id_to_token.get(id)) |token| {
            std.debug.print("   Token {d}: ID {d} = '{s}'\n", .{ i, id, token });
        } else {
            std.debug.print("   Token {d}: ID {d} = UNKNOWN\n", .{ i, id });
        }
    }
    std.debug.print("\n2. Decoding token IDs back to text...\n", .{});
    const decoded_text = try tokenizer.decode(token_ids);
    defer allocator.free(decoded_text);
    std.debug.print("   Result: \"{s}\"\n", .{decoded_text});
    std.debug.print("\n3. Verifying round-trip accuracy...\n", .{});
    const match = std.mem.eql(u8, text, decoded_text);
    std.debug.print("   Original and decoded text match: {}\n", .{match});
    try std.testing.expect(match);
}
