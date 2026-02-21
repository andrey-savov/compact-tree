"""Build script — adds the optional _marisa_ext C extension.

The C extension accelerates MarisaTrie.lookup() by ~5-10x for uncached
lookups.  If a C compiler is unavailable the package still works correctly
via the pure-Python fallback in marisa_trie.py.
"""

import os
import sys
from setuptools import setup, Extension


def _compile_args() -> list:
    """Return compiler flags for the extension.

    The default flags are portable (-O3 on gcc/clang, /O2 /Ob3 on MSVC).
    Set the environment variable COMPACT_TREE_MARCH_NATIVE=1 to add
    -march=native (Linux/macOS) for a locally-optimised build.  Do NOT
    set this when producing redistributable wheels — the resulting binary
    will crash with 'Illegal instruction' on CPUs that lack the native
    instruction-set extensions.
    """
    native = os.environ.get("COMPACT_TREE_MARCH_NATIVE", "").strip() == "1"
    if sys.platform == "win32":
        # MSVC: /O2 is max standard optimisation; /Ob3 = more aggressive
        # inlining (VS 2019+).  There is no portable equivalent of -march=native.
        return ["/O2", "/Ob3"]
    args = ["-O3"]
    if native:
        args.append("-march=native")
    return args


ext = Extension(
    "_marisa_ext",
    sources=["_marisa_ext.c"],
    optional=True,          # don't fail install if compiler is missing
    extra_compile_args=_compile_args(),
)

setup(ext_modules=[ext])
