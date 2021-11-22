from argparse import ArgumentParser
from difflib import diff_bytes, unified_diff
from pathlib import Path
from typing import Generator, Optional
from io import FileIO, BytesIO
from clang_helper import find_src_files
import subprocess as sp
import sys


def clang_format_corrections(src_path: Path) -> Optional[bytes]:
    if not src_path.is_file():
        return None

    # -style=file requires .clang-format in directory for src_path or parent
    result = sp.run(("clang-format", "-style=file", src_path), stdout=sp.PIPE)
    if result.returncode == 0:
        # clang-format outputs format result to stdout by default
        return result.stdout

    return None


def clang_format_fix(src_path: Path) -> bool:
    return (
        src_path.is_file()
        and sp.run(("clang-format", "-style=file", "-i", src_path)).returncode == 0
    )


def get_diff(src_path: Path, to_bytes: bytes) -> Optional[Generator]:
    if not src_path.is_file():
        return None

    with FileIO(src_path) as from_file, BytesIO(to_bytes) as to_file:
        return diff_bytes(
            unified_diff,
            from_file.readlines(),
            to_file.readlines(),
            fromfile=bytes(src_path),
            tofile=bytes(src_path) + b".clang-format",
        )


if __name__ == "__main__":
    argp = ArgumentParser(
        description="show diff between original and clang-format output"
    )
    argp.add_argument(
        "-r", "--recurse", action="store_true", help="recurse into subdirectories",
    )
    argp.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="write clang-format suggestions to source files",
    )
    args = argp.parse_args()

    for src_path in find_src_files(args.recurse):
        src_path = src_path.strip()

        # first apply clang-format on the source file
        clang_formatted = clang_format_corrections(Path(src_path))
        if not clang_formatted:
            continue

        diff = get_diff(Path(src_path), clang_formatted)
        if not diff:
            continue

        print(f"clang-format {src_path}")
        # now show unified diff between original file and clang-formatted
        for line in diff:
            print(line.decode("utf-8"), end="")

        # actually modify file
        if args.write and not clang_format_fix(Path(src_path)):
            sys.exit(f"clang-format failed to modify {src_path}")
