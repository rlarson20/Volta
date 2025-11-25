#!/usr/bin/env bash

runs=${1:-10}
total=0
count=0

for ((i = 1; i <= runs; i++)); do
	echo "=== Run $i ==="
	output=$(cargo test --release misc_tests::bench_matmul_speedup 2>&1)
	echo "$output"
	echo ""

	# Extract time from "256x256 matmul: 73.412791ms"
	if [[ $output =~ ([0-9]+\.[0-9]+)ms ]]; then
		time_ms="${BASH_REMATCH[1]}"
		total=$(awk "BEGIN {print $total + $time_ms}")
		((count++))
		echo "✓ Parsed: ${time_ms}ms"
	else
		echo "✗ Failed to parse timing"
	fi
	echo ""
done

if ((count > 0)); then
	avg=$(awk "BEGIN {print $total / $count}")
	echo "=== Summary ==="
	echo "Average: ${avg}ms over $count runs"
else
	echo "No successful runs"
	exit 1
fi
