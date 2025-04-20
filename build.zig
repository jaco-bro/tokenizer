const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    // const optimize = b.standardOptimizeOption(.{});
    const optimize: std.builtin.OptimizeMode = .ReleaseFast;
    const pcre2_dep = b.dependency("pcre2", .{
        .target = target,
        .optimize = optimize,
    });
    // Exe
    const exe = b.addExecutable(.{
        .name = "tokenizer_exe",
        .root_source_file = b.path("src/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.linkLibrary(pcre2_dep.artifact("pcre2-8"));
    const exe_install = b.addInstallArtifact(exe, .{});
    const exe_step = b.step("exe", "Install executable");
    exe_step.dependOn(&exe_install.step);
    // Run exe
    const exe_run = b.addRunArtifact(exe);
    if (b.args) |args| exe_run.addArgs(args);
    const run_step = b.step("run", "Run tokenizer app");
    run_step.dependOn(&exe_run.step);
    // Tests
    const tst = b.addTest(.{
        .root_source_file = b.path("src/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    tst.linkLibrary(pcre2_dep.artifact("pcre2-8"));
    // Run test
    const run_test = b.addRunArtifact(tst);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_test.step);
    // Lib
    const lib = b.addSharedLibrary(.{
        .name = "tokenizer",
        .root_source_file = b.path("src/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    lib.linkLibrary(pcre2_dep.artifact("pcre2-8"));
    const lib_install = b.addInstallArtifact(lib, .{});
    const lib_step = b.step("lib", "Install library only");
    lib_step.dependOn(&lib_install.step);
    b.getInstallStep().dependOn(&lib_install.step);
}
