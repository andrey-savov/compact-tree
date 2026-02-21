"""Generate compile_commands.json for _marisa_ext.c.

Run once (or after changing the C source) to give the VS Code C/C++
extension accurate IntelliSense without hard-coding any paths:

    python generate_compile_commands.py
"""

import json
import os
import shutil
import subprocess
import sys
import sysconfig

SRC = "_marisa_ext.c"
OUT = "compile_commands.json"


def msvc_cl_flags() -> list[str]:
    """Return the MSVC cl.exe flags used by distutils/setuptools on Windows."""
    return ["/nologo", "/W3", "/MD", "/O2", f"/I{sysconfig.get_path('include')}"]


def gcc_flags() -> list[str]:
    return ["-O2", f"-I{sysconfig.get_path('include')}"]


def main() -> None:
    src_abs = os.path.abspath(SRC)
    cwd = os.path.dirname(src_abs)

    if sys.platform == "win32":
        # Prefer cl.exe if available, otherwise fall back to gcc/clang.
        cl = shutil.which("cl")
        if cl:
            compiler = cl
            flags = msvc_cl_flags()
            command = f'"{compiler}" {" ".join(flags)} /c "{src_abs}"'
        else:
            compiler = shutil.which("gcc") or shutil.which("clang") or "cc"
            flags = gcc_flags()
            command = f'"{compiler}" {" ".join(flags)} -c "{src_abs}"'
    else:
        compiler = shutil.which("gcc") or shutil.which("clang") or "cc"
        flags = gcc_flags()
        command = f"{compiler} {' '.join(flags)} -c \"{src_abs}\""

    entry = {
        "directory": cwd,
        "file": src_abs,
        "command": command,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump([entry], f, indent=2)

    print(f"Written {OUT}")
    print(f"  compiler : {compiler}")
    print(f"  include  : {sysconfig.get_path('include')}")


if __name__ == "__main__":
    main()
