"""Mechanical writing-style checks over csrc/.

clang-format owns whitespace and line breaks, but it cannot order CUDA
execution-space modifiers: `QualifierOrder` only knows the C++ keywords, so
`__forceinline__ __device__` and `__device__ __forceinline__` both survive a
format pass. The same goes for `__inline__` (a GNU spelling of `inline`, i.e. a
hint) sitting next to `__forceinline__` (which carries always_inline) -- the two
read alike and mean different things.

This file is a plain text scan with no torch/CUDA dependency, so it runs in
under a second on any checkout.
"""

import re
from pathlib import Path

import pytest

CSRC = Path(__file__).resolve().parent.parent / "csrc"

# Vendored third-party trees keep their upstream style.
EXCLUDED_DIRS = ("sageattention3_blackwell", "cutedsl_sage")

SOURCES = sorted(
    p
    for p in CSRC.rglob("*")
    if p.suffix in (".h", ".cu", ".cuh", ".cpp")
    and not any(d in p.parts for d in EXCLUDED_DIRS)
)

HEADERS = [p for p in SOURCES if p.suffix in (".h", ".cuh")]

# Canonical modifier order: storage class, then execution space (host before
# device), then the inline word last.
MODIFIER_RANK = {
    "static": 0,
    "__host__": 1,
    "__device__": 2,
    "__global__": 2,
    "__forceinline__": 3,
    "inline": 3,
    "__inline__": 3,
}
MODIFIER_RUN = re.compile(r"\b(?:" + "|".join(MODIFIER_RANK) + r")\b(?:\s+\b(?:" + "|".join(MODIFIER_RANK) + r")\b)+")

COMMENT_OR_STRING = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"', re.S)


def code_of(path):
    """File text with comments and string literals blanked out, line count kept."""
    text = path.read_text(encoding="utf-8")

    def blank(m):
        return re.sub(r"[^\n]", " ", m.group(0))

    return COMMENT_OR_STRING.sub(blank, text)


def violations(pattern, flags=0):
    """[(path, lineno, line)] for every match of `pattern` in code (not comments)."""
    rx = re.compile(pattern, flags)
    out = []
    for path in SOURCES:
        for n, line in enumerate(code_of(path).splitlines(), 1):
            if rx.search(line):
                out.append((path.relative_to(CSRC.parent), n, line.strip()))
    return out


def test_sources_found():
    assert len(SOURCES) > 20, f"csrc scan found only {len(SOURCES)} files"


def test_no_gnu_inline_spelling():
    """`__inline__` is only a hint; device helpers use `__forceinline__`."""
    bad = violations(r"\b__inline__\b")
    assert not bad, "use __forceinline__ instead of __inline__:\n" + "\n".join(map(str, bad))


def test_modifier_order():
    """`[static] __host__ __device__ __forceinline__`, inline word last."""
    bad = []
    for path in SOURCES:
        for n, line in enumerate(code_of(path).splitlines(), 1):
            for m in MODIFIER_RUN.finditer(line):
                run = m.group(0).split()
                want = sorted(run, key=MODIFIER_RANK.__getitem__)
                if run != want:
                    bad.append((str(path.relative_to(CSRC.parent)), n, " ".join(run), " ".join(want)))
    assert not bad, "modifier order (got -> want):\n" + "\n".join(map(str, bad))


def test_pragma_unroll_spelling():
    """Lowercase `#pragma unroll`. An explicit trip count (`#pragma unroll
    kUnroll` in quant_per_thread.cu) is a codegen decision, not a spelling, so
    only the casing is pinned here."""
    bad = violations(r"#\s*pragma\s+(?!unroll\b)(?i:unroll)\b")
    assert not bad, "write `#pragma unroll` in lowercase:\n" + "\n".join(map(str, bad))


@pytest.mark.parametrize("path", HEADERS, ids=lambda p: p.name)
def test_headers_use_pragma_once(path):
    assert re.search(r"^#pragma once$", path.read_text(encoding="utf-8"), re.M), (
        f"{path.relative_to(CSRC.parent)} has no `#pragma once`"
    )


def test_fixed_width_integer_spellings():
    """Fixed-width types only. The 64-bit stride contract and the deliberate
    uint32 wraparound in attn_utils.cuh both depend on the width being written
    down, so `unsigned`/`long long` must not creep back in."""
    bad = violations(r"\bunsigned\b|\blong\s+long\b|\bNULL\b")
    assert not bad, "use uint32_t/int64_t/nullptr:\n" + "\n".join(map(str, bad))
