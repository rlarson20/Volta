#!/usr/bin/env bash
# Fails if numeric code reintroduces a silent fallback (CLAUDE.md invariant 5, ADDENDUM §8):
#   1. `unwrap_or(0.0)` / `unwrap_or(&0.0)` / `unwrap_or(f32::NAN)` on a lookup;
#   2. `if let Some(x) = v.get(i)` / `.get_mut(i)` / `.get(a..b)` on a positional index, which skips
#      the read or write when `i` is out of bounds;
#   3. `unwrap_or(1)` / `unwrap_or(&1)`: the integer shape/stride default, which turns a rank bug
#      into a size-1 axis. Read a known-rank shape with `crate::shape::fixed_dims::<N>`, a position
#      with `fetch`, and spell out the rank check where a 1 really is meant (broadcast padding).
# Use `crate::index::{fetch, slot, span}` in library code and `src/` unit tests, or
# `Tensor::item()` / `v[i]` / zip in `tests/`, and `.expect("..")` in examples and benches. A line may opt out with `// index-ok: <reason>`.
#   4. a default for a missing element or slice: `unwrap_or(&[])`, `(&mut 0.0)`, `(&' ')`, `(&0)`, and
#      any `.get(i)` / `.get_mut(i)` / `.first()` / `.last()` followed by `.unwrap_or(..)`. A keyed
#      `.get(&k)` / `.get("k")` map lookup is exempt: absent is a real answer there.
# Not caught: the multi-line `if let Some(&v) = x.get(i) { v } else { 0.0 }` form, and an
# `.unwrap_or(..)` on the line after the `.get(..)`.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

dirs=(src tests examples benches)
# Lines that are comments, or carry the opt-out marker, are never violations.
skip='^[^:]+:[0-9]+:[[:space:]]*//|index-ok:'
# `StateDict` lookups by key are optional by design, so receivers named *state* are not positional.
keyed='state(_dict)?\.get(_mut)?\('

hits=$(
    {
        git grep -nE --untracked 'unwrap_or\((&)?0\.0\)|unwrap_or\(f32::NAN\)|unwrap_or\(&?1\)' -- "${dirs[@]}" || true
        # A default for a missing element or slice: `unwrap_or(&[])`, `(&mut 0.0)`, `(&' ')`,
        # `(&f32::NAN)`, `(&0)`. An out-of-range index becomes an empty slice or a skipped write.
        git grep -nE --untracked "unwrap_or\\(&(mut )?(\\[\\]|' '|f32::NAN|[0-9][0-9.]*)\\)" -- "${dirs[@]}" |
            grep -vE "get(_mut)?\\((&|\")" | grep -vE "$keyed" || true
        # `.get(i)`/`.first()`/`.last()` followed by any default, with a positional argument
        # (a keyed `.get(&k)` or `.get("k")` is a map lookup, filtered out below).
        git grep -nE --untracked '\.(get|get_mut|first|last)\(([^()]|\([^()]*\))*\)(\.copied\(\)|\.cloned\(\))?\.unwrap_or\(' -- "${dirs[@]}" |
            grep -vE "get(_mut)?\((&|\")" | grep -vE "$keyed" || true
        git grep -nE --untracked 'let Some\(&?[A-Za-z_]+\) = .*\.get(_mut)?\(' -- "${dirs[@]}" |
            grep -vE "get(_mut)?\((&|\")" | grep -vE "$keyed" || true
    } | grep -vE "$skip" | sort -u || true
)

if [ -n "$hits" ]; then
    echo "Silent fallbacks found (use crate::index::{fetch, slot, span}, or .expect(..) in tests/examples):" >&2
    echo "$hits" >&2
    exit 1
fi
echo "check-fallbacks: ok"
