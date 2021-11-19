from argparse import ArgumentParser
from difflib import diff_bytes, unified_diff
from pathlib import Path
from typing import Generator, List, Optional
from io import FileIO, BytesIO
import subprocess as sp
import os


CLANG_FORMAT_SRC_FILES = "clang-format-srcs"


def clang_format(src_path: Path) -> Optional[bytes]:
    if not src_path.is_file():
        return None

    # -style=file requirest .clang-format in directory for src_path or parent
    result = sp.run(("clang-format", "-style=file", src_path), stdout=sp.PIPE)
    if result.returncode == 0:
        # clang-format outputs format result to stdout by default
        return result.stdout

    return None


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


def find_src_files(recurse: bool) -> List[str]:
    listing_files: List[Path] = []
    with os.scandir(os.curdir) as scan:
        for i in scan:
            if i.name == CLANG_FORMAT_SRC_FILES:
                # always read clang-format-srcs in current directory
                listing_files.append(i.path)
            # find clang-format-srcs in any subdirectory
            elif recurse and i.is_dir():
                for walk in os.walk(i.path):
                    if CLANG_FORMAT_SRC_FILES in walk[-1]:
                        listing_files.append(
                            os.path.join(walk[0], CLANG_FORMAT_SRC_FILES)
                        )

    src_files: List[str] = []
    # each clang-format-srcs has source file paths per line
    for listing in listing_files:
        with open(listing) as file:
            for line in file:
                src_files.append(line.strip())

    return src_files


if __name__ == "__main__":
    argp = ArgumentParser(
        description="show diff between original and clang-format output"
    )
    argp.add_argument(
        "-r", "--recurse",
        action="store_true",
        help="recurse into subdirectories",
    )
    argp.add_argument(
        "-w", "--write",
        action="store_true",
        help="write clang-format suggestions to source files",
    )
    args = argp.parse_args()

    for src_path in find_src_files(args.recurse):
        src_path = src_path.strip()
        # first apply clang-format on the source file
        clang_formatted = clang_format(Path(src_path))
        if not clang_formatted:
            continue

        # now show unified diff between original file and clang-formatted
        diff = get_diff(Path(src_path), clang_formatted)
        if diff:
            for line in diff:
                print(line.decode("utf-8"), end="")
