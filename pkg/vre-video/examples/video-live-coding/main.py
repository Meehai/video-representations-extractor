#!/usr/bin/env python3
from vre_video import VREVideo
import sys
import importlib.util

def load_function_from_module(module_path, function_name):
    module_name = module_path.split("/")[-1].replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)

def main():
    video = VREVideo(sys.argv[1])

    i = 0
    while True:
        frame = video[i % len(video)]
        try:
            fn = load_function_from_module("src.py", "fn")
            frame = fn(frame)
        except Exception as e:
            print(e, file=sys.stderr)
        print(f"Frame {i}: shape={frame.shape}", file=sys.stderr)  # Debugging info

        sys.stdout.buffer.write(frame.reshape(-1).tobytes())
        sys.stdout.flush()

        i += 1

if __name__ == "__main__":
    main()
