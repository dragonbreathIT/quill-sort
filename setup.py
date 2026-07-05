"""Optional C-extension build shim for QuillSort v7.

The canonical project metadata lives in ``pyproject.toml`` (PEP 621, setuptools
backend). This ``setup.py`` exists solely to augment that build with two
*optional* numpy-backed C extensions:

    - quill._listconv_ext   (quill/_listconv_ext.c)
    - quill._counting_ext   (quill/_counting_ext.c)

Design constraints (NEVER-LOSE applies at install time too):

  * The package MUST install cleanly as pure-Python whenever the C path is
    unavailable. The Python wrappers in :mod:`quill` already fall back to numpy
    (and ultimately ``np.sort`` / ``list.sort``) when the compiled module is
    missing, so an unbuilt extension is *not* an error condition.

  * No C compiler? Install proceeds with no extensions — setuptools will simply
    skip them at build time, and pip falls through to the pure-Python wheel.

  * No numpy at build time? We cannot resolve ``numpy.get_include()`` and so
    we cannot declare the Extensions safely. We swallow the ImportError and
    install with ``ext_modules=[]``. This is the common case for ``pip install
    quill-sort`` on a fresh venv where numpy is an extra, not a hard dep.

  * Unsupported platform / build failure? Same story — silently drop to pure
    Python. The runtime layer transparently uses numpy paths instead.

Environment-variable escape hatches (for CI and power users):

  * ``QUILL_NO_C_EXT=1``     — Force a pure-Python install. Skip both
                                extensions even when everything is available.
                                Useful for building the universal sdist /
                                py3-none-any wheel published to PyPI.

  * ``QUILL_FORCE_C_EXT=1``  — Inverse: refuse to silently fall back. If numpy
                                is missing or extension construction fails,
                                re-raise so the build is loud. Intended for CI
                                wheel-building jobs where a missing extension
                                is a real bug.

Compiler flags follow the project convention used for the Rust ``_fastsort``
backend: aggressive optimization on every supported toolchain.

  * POSIX (gcc/clang):  -O3 -fPIC
  * MSVC (Windows cl):  /O2

The pyproject.toml is the source of truth for metadata; this file passes only
``ext_modules`` to :func:`setup`, leaving name/version/deps/etc to pyproject.
"""

from __future__ import annotations

import os
import sys

from setuptools import setup


def _compile_args() -> list[str]:
    # WHY: MSVC rejects -O3/-fPIC outright; gcc/clang ignore /O2. Branch on
    # platform rather than compiler because setuptools doesn't expose the
    # compiler choice at Extension-construction time.
    if sys.platform.startswith("win"):
        return ["/O2"]
    return ["-O3", "-fPIC"]


def _cpp_args() -> list[str]:
    # The bundled native backends (quill/_native/native.cpp) are C++17 and use
    # std::thread, so they need the standard flag + -pthread on POSIX.
    if sys.platform.startswith("win"):
        return ["/O2", "/std:c++17", "/EHsc"]
    return ["-O3", "-fPIC", "-std=c++17", "-pthread"]


def _cpp_link() -> list[str]:
    return [] if sys.platform.startswith("win") else ["-pthread"]


def _c_thread_args() -> list[str]:
    # Spectre (spectre_sort.c) is C11 and multi-threaded (POSIX pthreads / Win32).
    # It needs the C optimizer flags plus -pthread on POSIX; MSVC links its
    # threading (Win32) implicitly, so /O2 alone suffices there.
    if sys.platform.startswith("win"):
        return ["/O2"]
    return ["-O3", "-fPIC", "-pthread"]


def _c_thread_link() -> list[str]:
    return [] if sys.platform.startswith("win") else ["-pthread"]


def _build_ext_modules() -> list:
    """Return the C/C++ Extensions to compile, or [] for pure-Python.

    Honors QUILL_NO_C_EXT (skip everything) and QUILL_FORCE_C_EXT (re-raise /
    fail loudly instead of silently falling back — used by the wheel-building
    CI, where a missing extension is a real bug).

    NEVER-LOSE: outside of QUILL_FORCE_C_EXT, every extension is marked
    ``optional=True`` so a compile failure on an exotic toolchain simply skips
    that extension and the pure-Python paths take over — ``pip install
    quill-sort`` can never fail because a C++ compiler misbehaved.
    """
    if os.environ.get("QUILL_NO_C_EXT") == "1":
        # Explicit opt-out for building the universal py3-none-any wheel.
        return []

    force = os.environ.get("QUILL_FORCE_C_EXT") == "1"
    optional = not force

    try:
        from setuptools import Extension
    except ImportError:
        if force:
            raise
        return []

    exts: list = []

    # 1) Bundled native CPU backends (quill._native). Pure C++17 via the buffer
    #    protocol — NO numpy headers required, so it builds even on a numpy-less
    #    build env. This is the fold-in of the former quill-fastsort* companions.
    try:
        exts.append(Extension(
            "quill._native",
            sources=["quill/_native_src/native.cpp"],
            include_dirs=["quill/_native_src"],
            language="c++",
            extra_compile_args=_cpp_args(),
            extra_link_args=_cpp_link(),
            optional=optional,
        ))
    except BaseException:
        if isinstance(sys.exc_info()[1], (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        if force:
            raise

    # 1b) Spectre — bundled parallel integer radix sort (quill._spectre). Pure
    #    C11 via the buffer protocol (no numpy headers), compiled as its OWN
    #    extension because it is C, not C++, and so cannot share quill._native's
    #    -std=c++17 flags. Ships in the same binary wheels. Kept optional=True
    #    unconditionally: it's a newer backend, and a compile hiccup on an exotic
    #    toolchain must never fail the wheel — SpectreBackend simply reports
    #    unavailable and dispatch falls through to the other backends / np.sort.
    try:
        exts.append(Extension(
            "quill._spectre",
            sources=["quill/_native_src/spectre_sort.c",
                     "quill/_native_src/spectre_py.c"],
            include_dirs=["quill/_native_src"],
            extra_compile_args=_c_thread_args(),
            extra_link_args=_c_thread_link(),
            optional=True,
        ))
    except BaseException:
        if isinstance(sys.exc_info()[1], (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        if force:
            raise

    # 2) numpy-backed C accelerators (list<->ndarray conversion, fused counting
    #    sort). These DO need numpy headers, so they're only declared when numpy
    #    is importable at build time.
    try:
        import numpy as np  # noqa: F401 — only for get_include()
        include_dirs = [np.get_include()]
        extra = _compile_args()
        exts.append(Extension(
            "quill._listconv_ext", sources=["quill/_listconv_ext.c"],
            include_dirs=include_dirs, extra_compile_args=extra, optional=optional,
        ))
        exts.append(Extension(
            "quill._counting_ext", sources=["quill/_counting_ext.c"],
            include_dirs=include_dirs, extra_compile_args=extra, optional=optional,
        ))
    except BaseException:
        if isinstance(sys.exc_info()[1], (KeyboardInterrupt, SystemExit, GeneratorExit)):
            raise
        # The numpy-backed exts are a SECONDARY list-path optimization. When numpy
        # isn't importable at build time — e.g. cibuildwheel's isolated build env,
        # which has no numpy — we skip them even under QUILL_FORCE_C_EXT. The
        # binary wheel must carry the numpy-free native backend (quill._native,
        # forced above); these two are a bonus built only where numpy is present
        # (source installs, or a build env that provides it).

    return exts


setup(ext_modules=_build_ext_modules())
