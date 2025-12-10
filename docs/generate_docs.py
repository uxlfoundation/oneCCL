"""
 Copyright (C) 2023-2024 Intel Corporation
 
 Under the Apache License v2.0 with LLVM Exceptions. See LICENSE.TXT.
 SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

from pathlib import Path
from shutil import rmtree, copytree
import subprocess  # nosec B404
import time


def _check_cwd() -> None:
    cwd = Path.cwd()
    include_dir = Path(cwd, "../include")
    # Verify if include dir is one level up (as defined in Doxyfile)
    if not include_dir.exists():
        print(
            f"Include directory {include_dir.resolve()} not found! "
            "Please run this script from <repo_root_dir>/build!",
            flush=True,
        )
        exit(1)


def _clear_docs_dir(docs_path: Path) -> None:
    if docs_path.exists():
        try:
            rmtree(docs_path)
        except:
            print(f"Failed to remove docs directory {docs_path.resolve()}", flush=True)


def _create_docs_dir(docs_path: Path) -> None:
    docs_path.mkdir()


def _prepare_docs_dir(docs_path: Path) -> None:
    _clear_docs_dir(docs_path)
    _create_docs_dir(docs_path)


def _generate_xml(config_path: Path, docs_path: Path) -> None:
    print("Generating XML files with doxygen...", flush=True)
    try:
        subprocess.run(
            ["doxygen", Path(config_path, "Doxyfile")], text=True
        ).check_returncode()  # nosec B603, B607
        print(f"All XML files generated in {docs_path}", flush=True)
    except subprocess.CalledProcessError as ex:
        print("Generating XML files failed!")
        print(ex)
        exit(1)


def _generate_html(config_path: Path, docs_path: Path) -> None:
    print("Generating HTML pages with sphinx...", flush=True)
    try:
        subprocess.run(
            ["sphinx-build", config_path, Path(docs_path, "html")], text=True
        ).check_returncode()  # nosec B603, B607
        print(f"All HTML files generated in {docs_path}", flush=True)
    except subprocess.CalledProcessError as ex:
        print("Generating HTML pages failed!")
        print(ex)
        exit(1)


def main() -> None:
    _check_cwd()

    script_dir = Path(__file__).resolve().parent
    docs_build_path = Path("docs_build").resolve()

    # Sphinx and breathe require access to a Doxygen generated dir ('doxyxml')
    # so we copy the whole content of the 'docs' dir to the build dir.
    copytree(Path(script_dir), docs_build_path, dirs_exist_ok=True)

    config_path = Path(docs_build_path, "config").resolve()
    docs_path = Path(docs_build_path, "generated").resolve()

    start = time.time()
    _prepare_docs_dir(docs_path)
    _generate_xml(config_path, docs_path)
    _generate_html(config_path, docs_path)
    print(f"Pages generated in {time.time() - start:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
