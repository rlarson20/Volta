# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.15.0", "safetensors", "numpy"]
#
# # Same pins and index as gen_fixtures.py: keep the two blocks identical.
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Check that tests/fixtures/torch matches what gen_fixtures.py generates now.

Runs every group of gen_fixtures.py into a temporary directory and compares the result
with the committed files. Exit status 0 means they agree, 1 means they don't (each
mismatch is printed), so CI can run `uv run scripts/torch_ref/check_fixtures.py`.

Why a tolerance instead of `git diff --exit-code`: regeneration is not byte-stable. The
same torch==2.15.0 on different platforms and builds (macOS, Linux `+cpu`, Linux `+cu132`)
gives f32 results a few ulps apart in BatchNorm, convolution, softmax and Muon, and the
torch version string in the metadata carries the build suffix. Measured on 2026-11-11
against the committed files: worst f32 difference 3.3e-6 on a scale of max(|want|, 1),
no difference at all in f64, f16, bf16, integer or bool tensors. On the GitHub
ubuntu-latest runner the `autocast` group's f32 gradients also moved by up to 2.0 f16 eps
(`f16.*`) and 1.06 bf16 eps (`bf16.*`): they are f32 tensors, but they carry the precision
of the low dtype the forward ran in, so they get that dtype's tolerance (`LOW_PRECISION`).

What must match exactly: the set of files, each file's tensor names, every tensor's
dtype and shape, integer and bool values, NaN positions, infinities and their signs, and
the metadata once the torch build suffix ("+cpu") is removed. Finite floats must agree
to `TOL[dtype] * max(|want|, 1)`, the relative-with-floor-1 metric most Rust
`torch_ref_*` tests use. The f32 and f64 tolerances are tighter than those tests'. The
low-dtype ones are not: `torch_ref_autocast` bounds some cases at 1 to 2 eps, so torch's
own drift between machines is about as large as the error that test allows Volta.
"""

import re
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open

sys.dont_write_bytecode = True  # no __pycache__ next to the scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_fixtures as gen  # noqa: E402

COMMITTED = gen.OUT

# Relative tolerance with a floor of 1 on the scale. f32 allows ~80 eps for the
# cross-platform reduction-order differences above. f16 and bf16 allow 4 eps, twice the
# worst autocast drift measured; f64 differed by 0 when measured and gets a little slack.
TOL = {
    torch.float64: 1e-12,
    torch.float32: 1e-5,
    torch.float16: 4 * torch.finfo(torch.float16).eps,
    torch.bfloat16: 4 * torch.finfo(torch.bfloat16).eps,
}


# f32 tensors whose values passed through a low dtype: (file, key prefix) -> the dtype
# whose tolerance applies.
LOW_PRECISION = {
    ("autocast.safetensors", "bf16."): torch.bfloat16,
    ("autocast.safetensors", "f16."): torch.float16,
}


def tolerance(file, key, dtype):
    """The tolerance for `key` in `file`: its low-precision dtype's, else its own dtype's."""
    for (f, prefix), low in LOW_PRECISION.items():
        if f == file and key.startswith(prefix):
            return max(TOL[low], TOL.get(dtype, 0.0))
    return TOL.get(dtype)


def strip_build(meta):
    """Drop the local build suffix from `torch=<version>+<build>` in metadata values."""
    return {k: re.sub(r"(torch=[^;+]+)\+[^;]*", r"\1", v) for k, v in (meta or {}).items()}


def load(path):
    with safe_open(str(path), framework="pt") as f:
        return f.metadata(), {k: f.get_tensor(k) for k in f.keys()}


def compare_tensor(name, want, got, tol):
    """Return a list of problems with `got` against the committed `want`."""
    if want.dtype != got.dtype:
        return [f"{name}: dtype {got.dtype}, committed {want.dtype}"]
    if want.shape != got.shape:
        return [f"{name}: shape {tuple(got.shape)}, committed {tuple(want.shape)}"]
    if not want.is_floating_point():
        if not torch.equal(want, got):
            n = int((want != got).sum())
            return [f"{name}: {n} of {want.numel()} values differ ({want.dtype}, must be exact)"]
        return []
    if tol is None:
        return [f"{name}: no tolerance defined for {want.dtype}"]
    w, g = want.double(), got.double()
    problems = []
    if not torch.equal(w.isnan(), g.isnan()):
        problems.append(f"{name}: NaN positions differ")
    winf, ginf = w.isinf(), g.isinf()
    if not torch.equal(winf, ginf) or not torch.equal(w[winf], g[ginf]):
        problems.append(f"{name}: infinities differ")
    fin = w.isfinite() & g.isfinite()
    if fin.any():
        err = (w[fin] - g[fin]).abs() / w[fin].abs().clamp_min(1.0)
        worst = err.max().item()
        if worst > tol:
            n = int((err > tol).sum())
            problems.append(f"{name}: {n} values off, worst {worst:.3g} > {tol:.3g} ({want.dtype})")
    return problems


def main():
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        gen.OUT = fresh
        for build in gen.GROUPS.values():
            build()

        problems = []
        want_files = {p.name for p in COMMITTED.glob("*.safetensors")}
        got_files = {p.name for p in fresh.glob("*.safetensors")}
        for name in sorted(want_files - got_files):
            problems.append(f"{name}: committed but no longer generated")
        for name in sorted(got_files - want_files):
            problems.append(f"{name}: generated but not committed")

        for name in sorted(want_files & got_files):
            want_meta, want = load(COMMITTED / name)
            got_meta, got = load(fresh / name)
            if strip_build(want_meta) != strip_build(got_meta):
                problems.append(f"{name}: metadata {got_meta}, committed {want_meta}")
            for key in sorted(want.keys() - got.keys()):
                problems.append(f"{name}: {key} committed but no longer generated")
            for key in sorted(got.keys() - want.keys()):
                problems.append(f"{name}: {key} generated but not committed")
            for key in sorted(want.keys() & got.keys()):
                problems.extend(f"{name}: {p}" for p in compare_tensor(key, want[key], got[key], tolerance(name, key, want[key].dtype)))

    if problems:
        print(f"\n{len(problems)} mismatches between tests/fixtures/torch and gen_fixtures.py:")
        for p in problems:
            print(f"  {p}")
        print("\nIf the generator change is intended, run `just torch-fixtures` and commit the result.")
        return 1
    print(f"\nall {len(want_files)} fixture files match gen_fixtures.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
