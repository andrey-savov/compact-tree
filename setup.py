"""Build script — adds the optional _marisa_ext C extension.

The C extension accelerates MarisaTrie.lookup() by ~5-10x for uncached
lookups.  If a C compiler is unavailable the package still works correctly
via the pure-Python fallback in marisa_trie.py.
"""

from setuptools import setup, Extension

ext = Extension(
    "_marisa_ext",
    sources=["_marisa_ext.c"],
    optional=True,          # don't fail install if compiler is missing
    # MSVC: /O2 is already max standard optimisation (no /O3 exists).
    #   /Ob3  = more aggressive inlining than /O2's default /Ob2 (VS 2019+)
    #   /arch:AVX2 = enable AVX2 vectorisation (equivalent to -march=native AVX2)
    #               only safe when building for local use; omit for portable wheels.
    # gcc/clang: -O3 -march=native adds loop vectorisation and CPU-specific
    # codegen on top of -O2; helps on Linux/macOS but makes the wheel non-portable.
    extra_compile_args=["-O3", "-march=native"] if __import__("sys").platform != "win32"
                        else ["/O2", "/Ob3"],
)

setup(ext_modules=[ext])
