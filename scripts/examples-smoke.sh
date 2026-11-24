#!/usr/bin/env bash
# Runs the examples that need no downloaded data and finish in seconds, each under a time
# limit, on the CPU lane (release, no default features). Fails when one exits non-zero (a
# panic, including its own final-loss assertion), runs past the limit, or prints a NaN or an
# infinity. The smoke set and the reason each other example is left out are in
# dev-docs/DEVELOPMENT.md ("Examples smoke run"); keep the two lists in step.
#
#   scripts/examples-smoke.sh [LIMIT_SECONDS]   (default 120)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

limit=${1:-120}
smoke=(
    readme1
    readme2
    showcase
    regression
    layernorm_demo
    super_resolution
    char_language_model
    char_transformer_model
    gpu_training
    load_external_mnist
    time_sequence
)

args=()
for e in "${smoke[@]}"; do args+=(--example "$e"); done
cargo build --release --no-default-features "${args[@]}"

bin=$PWD/target/release/examples
# Each example runs in an empty directory, so files it writes (readme1's model.safetensors) stay out
# of the tree and an example that reads optional files (load_external_mnist) takes the same
# path everywhere.
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# A printed NaN or infinity, as Rust's `{}` formats f32 ("NaN", "inf", "-inf").
nonfinite='(^|[^A-Za-z_])(NaN|-?inf)([^A-Za-z_]|$)'

failed=()
for e in "${smoke[@]}"; do
    log=$work/$e.log
    start=$SECONDS
    # A portable time limit (macOS has no `timeout`): a watchdog kills the run at $limit.
    (cd "$work" && exec "$bin/$e") >"$log" 2>&1 &
    pid=$!
    (sleep "$limit" && kill "$pid" 2>/dev/null) &
    watchdog=$!
    code=0
    wait "$pid" || code=$?
    kill "$watchdog" 2>/dev/null || true
    wait "$watchdog" 2>/dev/null || true
    took=$((SECONDS - start))

    reason=""
    if [ "$took" -ge "$limit" ] && [ "$code" -ne 0 ]; then
        reason="timed out after ${limit}s"
    elif [ "$code" -ne 0 ]; then
        reason="exit $code"
    elif grep -qE "$nonfinite" "$log"; then
        reason="printed a NaN or infinity: $(grep -nE "$nonfinite" "$log" | head -n1)"
    fi

    if [ -n "$reason" ]; then
        echo "FAIL $e (${took}s): $reason"
        tail -n 20 "$log" | sed 's/^/    /'
        failed+=("$e")
    else
        echo "ok   $e (${took}s)"
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "examples-smoke: ${#failed[@]} failed: ${failed[*]}" >&2
    exit 1
fi
echo "examples-smoke: ${#smoke[@]} examples ok"
