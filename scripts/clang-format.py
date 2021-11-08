from argparse import ArgumentParser
from difflib import diff_bytes, unified_diff
from pathlib import Path
from typing import Generator, Optional
from io import FileIO, BytesIO
import subprocess as sp


def clang_format(src_path: Path) -> Optional[bytes]:
    # -style=file requirest .clang-format in directory for src_path or parent
    result = sp.run(("clang-format", "-style=file", src_path), stdout=sp.PIPE)
    if result.returncode == 0:
        # clang-format outputs format result to stdout by default
        return result.stdout

    return None


def get_diff(src_path: Path, to_bytes: bytes) -> Generator:
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
        "-s", "--src-files", help=";-separated of paths to files to check"
    )
    args = argp.parse_args()

    for src_path in args.src_files.split(";"):
        src_path = src_path.strip()
        # first apply clang-format on the source file
        clang_formatted = clang_format(Path(src_path))
        # now show unified diff between original file and clang-formatted
        for line in get_diff(Path(src_path), clang_formatted):
            print(line.decode("utf-8"), end="")
