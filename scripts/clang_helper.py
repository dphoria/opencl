from typing import List
from pathlib import Path
import os

CLANG_FORMAT_SRC_FILES = "clang-format-srcs"


def find_src_files(recurse: bool) -> List[str]:
    """
    file paths listed in clang-format-srcs

    Paramters
    ---------
    recurse: bool
        read clang-format-srcs in subdirectories

    Returns
    -------
    file_paths: List[str]
        source file paths listed in clang-format-srcs
    """
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
