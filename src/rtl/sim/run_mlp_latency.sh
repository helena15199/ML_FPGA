#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RTL_DIR="$REPO_ROOT/src/rtl/generated/src_ultra_sequential"
BUILD_DIR="$REPO_ROOT/vivado/sim/mlp_latency"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
xvlog --sv -i "$RTL_DIR" "$RTL_DIR"/*.sv "$REPO_ROOT/src/rtl/tb/TB_mlp_latency.sv" > compile.log 2>&1
xelab --timescale 1ns/1ps TB_mlp_latency -s mlp_latency -debug typical > elaborate.log 2>&1
xsim mlp_latency -wdb mlp_latency.wdb -tclbatch "$SCRIPT_DIR/run_mlp_latency.tcl" -log simulate.log
# Also fail if XSim returned success after a testbench fatal error.
grep -q 'PASS: 4 samples timed' simulate.log
