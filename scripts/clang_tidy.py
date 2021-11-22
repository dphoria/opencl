from argparse import ArgumentParser
from pathlib import Path
from sys import stderr
from clang_helper import find_src_files
import subprocess as sp


def clang_tidy(src_path: Path, *nargs) -> None:
    """
    Run clang-tidy on src_path

    Parameters
    ----------
    src_path: Path
        source file fpath
    *nargs
        pass to clang-tidy

    Notes
    -----
    clang-tidy [*nargs] <src_path>
    """
    if not src_path.is_file():
        print(src_path, "is not a file", file=stderr)
        return

    result = sp.run(
        ("clang-tidy", *nargs, src_path), stdout=sp.PIPE, stderr=sp.PIPE
    )
    if result.returncode != 0:
        print(
            f"clang-tidy {' '.join(nargs)} {src_path} failed:\n"
            f"{result.stderr.decode('utf-8')}",
            file=stderr,
            end="",
        )
        return

    print(f"clang-tidy {src_path}\n{result.stdout.decode('utf-8')}", end="")


if __name__ == "__main__":
    argp = ArgumentParser(
        description="run clang-tidy on files listed in clang-format-srcs"
    )
    argp.add_argument(
        "-r", "--recurse",
        action="store_true",
        help="recurse into subdirectories",
    )
    argp.add_argument(
        "-w", "--write",
        action="store_true",
        help="write clang-tidy suggestions to source files",
    )
    argp.add_argument(
        "-p",
        help="build directory containing compile_commands.json",
    )
    args = argp.parse_args()

    for src_path in find_src_files(args.recurse):
        if args.write:
            clang_tidy(Path(src_path.strip()), "-p", args.p, "--fix")
        else:
            clang_tidy(Path(src_path.strip()), "-p", args.p)
