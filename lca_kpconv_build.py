"""Setuptools hooks: compile cpp_wrappers before packaging (see pyproject cmdclass)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from setuptools.command.build_py import build_py

_ROOT = Path(__file__).resolve().parent
_WRAPPERS = _ROOT / "cpp_wrappers"
_SCRIPT = _WRAPPERS / "compile_wrappers.sh"


def _run_compile_wrappers() -> None:
    if not _SCRIPT.is_file():
        return
    env = {**os.environ, "PYTHON": sys.executable}
    subprocess.check_call(["/bin/bash", str(_SCRIPT)], cwd=str(_WRAPPERS), env=env)


class build_py_with_cpp(build_py):
    def run(self) -> None:
        _run_compile_wrappers()
        super().run()
