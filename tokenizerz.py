import os
import platform
import ctypes
import sys
from ctypes import c_char_p, c_uint32, c_size_t, POINTER, c_void_p, c_bool
import json
import jinja2

class Tokenizer:
    def __init__(self, repo_name="Qwen", model_name="Qwen2.5-Coder-0.5B", verbose=False):
        system = platform.system()
        if system == "Darwin": 
            lib_ext = ".dylib"
        elif system == "Linux":
            lib_ext = ".so"
        elif system == "Windows":
            lib_ext = ".dll"
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")
        lib_name = f"libtokenizer{lib_ext}"
        package_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(package_dir, "zig-out", "lib", lib_name)
        try:
            self.lib = ctypes.CDLL(lib_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load the tokenizer library at {lib_path}: {e}")
        self.lib.create_tokenizer.argtypes = [c_char_p, c_char_p, c_bool]
        self.lib.create_tokenizer.restype = c_void_p
        self.lib.encode_text.argtypes = [c_void_p, c_char_p, POINTER(c_uint32), c_size_t]
        self.lib.encode_text.restype = c_size_t
        self.lib.decode_tokens.argtypes = [c_void_p, POINTER(c_uint32), c_size_t, c_char_p, c_size_t]
        self.lib.decode_tokens.restype = c_size_t
        self.lib.free_tokenizer.argtypes = [c_void_p]
        self.lib.free_tokenizer.restype = None
        self.handle = self.lib.create_tokenizer(repo_name.encode('utf-8'), model_name.encode('utf-8'), verbose)
        if not self.handle:
            raise RuntimeError(f"Failed to initialize tokenizer with model: {model_name}")
        try:
            with open(f'{model_name}/tokenizer_config.json', 'r') as f:
                self.config = json.load(f)
            self.chat_template = self.config['chat_template']
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            self.chat_template = None
            print(f"No chat template found: {e}")
    
    def __call__(self, text, use_chat_template=False, pad_token_id=1, pad_position_id=1, **kwargs):
        if use_chat_template:
            text = [self.apply_chat_template(text, **kwargs)]
        elif isinstance(text, str):
            text = [text]
        tokens = [self.encode(t) for t in text]
        input_ids, position_ids, padding_mask = self.pad_token_sequences(tokens, pad_token_id, pad_position_id)
        return text, input_ids, position_ids, padding_mask

    def encode(self, text):
        text_bytes = text.encode('utf-8')
        max_tokens = max(128, len(text_bytes) * 16)
        token_buffer = (c_uint32 * max_tokens)()
        actual_tokens = self.lib.encode_text(self.handle, text_bytes, token_buffer, max_tokens)
        if actual_tokens == 0:
            raise RuntimeError(f"Encoding failed for {text}")
        if actual_tokens <= max_tokens:
            return [token_buffer[i] for i in range(actual_tokens)]
        token_buffer = (c_uint32 * actual_tokens)()
        actual_tokens = self.lib.encode_text(self.handle, text_bytes, token_buffer, actual_tokens)
        return [token_buffer[i] for i in range(actual_tokens)]
    
    def decode(self, tokens):
        token_array = (ctypes.c_uint32 * len(tokens))(*tokens)
        initial_capacity = 4096 
        out_buffer = ctypes.create_string_buffer(initial_capacity)
        required_len = self.lib.decode_tokens(
            self.handle, 
            token_array, 
            len(tokens), 
            out_buffer, 
            initial_capacity
        )
        if required_len >= initial_capacity:
            out_buffer = ctypes.create_string_buffer(required_len + 1)
            self.lib.decode_tokens(
                self.handle, 
                token_array, 
                len(tokens), 
                out_buffer, 
                required_len + 1
            )
        return out_buffer.value.decode('utf-8', errors='replace')

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.lib.free_tokenizer(self.handle)

    def apply_chat_template(self, messages, **kwargs):
        if not self.chat_template:
            raise ValueError("No chat template available")
        if not (isinstance(messages, list) and all(isinstance(m, dict) for m in messages)):
            raise ValueError(f"Messages must be a list of dictionaries but received {messages}")
        env = jinja2.Environment(autoescape=False)
        template = env.from_string(self.chat_template)
        try:
            return template.render(messages=messages, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to render chat template: {e}")

    @staticmethod
    def pad_token_sequences(sequences, pad_token_id, pad_position_id):
        max_len = max(len(seq) for seq in sequences)
        input_ids = []
        position_ids = []
        padding_mask = []
        for seq in sequences:
            pad_len = max_len - len(seq)
            padded = [pad_token_id] * pad_len + seq
            input_ids.append(padded)
            pos_ids = [pad_position_id] * pad_len + list(range(len(seq)))
            position_ids.append(pos_ids)
            mask = [False] * pad_len + [True] * len(seq)
            padding_mask.append(mask)
        return input_ids, position_ids, padding_mask

def run():
    import argparse
    parser = argparse.ArgumentParser(description="Command-line interface for the tokenizer")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B", help="Model to use for tokenization (default: Qwen/Qwen2.5-Coder-0.5B)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--encode", help="Text to encode into tokens")
    group.add_argument("--decode", help="Tokens to decode into text")
    args = parser.parse_args()
    repo_name, model_name = args.model.split('/')
    tokenizer = Tokenizer(repo_name=repo_name, model_name=model_name)
    if args.encode:
        try:
            tokens = tokenizer.encode(args.encode)
            print(tokens)
        except Exception as e:
            print(f"Error encoding text: {e}")
            sys.exit(1)
    elif args.decode:
        try:
            text = args.decode.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            elif text.startswith("{") and text.endswith("}"):
                text = text[1:-1]
            tokens = []
            for part in text.replace(",", " ").split():
                if part.strip():
                    tokens.append(int(part))
            if not tokens:
                print("Error: No valid tokens found")
                sys.exit(1)
            decoded = tokenizer.decode(tokens)
            print(decoded)
        except ValueError as e:
            print(f"Error: Failed to parse tokens. Please provide tokens as space or comma-separated integers: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            sys.exit(1)

def demo():
    print("===Demo===")
    tokenizer = Tokenizer(repo_name="Qwen", model_name="Qwen2.5-Coder-0.5B")
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    print(f"Encoded: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    _, iids, pids, mask = tokenizer(["#write a quick sort algorithm", "#hello world"])
    print(f"Batch:\n{iids=}\n{pids=}\n{mask=}")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    _, iids, pids, mask = tokenizer(messages, use_chat_template=True, add_generation_prompt=True, tools=None)
    print(f"Format:\n{iids=}\n{pids=}\n{mask=}")
    print(tokenizer.decode(iids[0]))

if __name__ == "__main__":
    demo()
