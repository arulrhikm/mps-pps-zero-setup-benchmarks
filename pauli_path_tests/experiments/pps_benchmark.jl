#!/usr/bin/env julia
#=
PPS Benchmark — PauliPropagation.jl  (LOCAL CPU)
=================================================
127-qubit TFI Trotter circuit on the IBM Eagle heavy-hex topology.
Sweeps the coefficient truncation threshold δ (min_abs_coeff) and
measures ⟨Z₆₂⟩ = Tr[ρ U† Z₆₂ U] with ρ = |0⟩⟨0|^⊗n.

Benchmarks LOCAL CPU performance of PauliPropagation.jl against the
BlueQubit REMOTE GPU pauli-path results (pps_gpu_benchmark.ipynb) and
the Qiskit pauli-prop LOCAL CPU benchmark (pps_benchmark_qiskit.py).

KEY DESIGN DECISIONS FOR FAIRNESS:
  • Topology:  We use the SAME IBM_127_HEAVY_HEX_MAP edge list (converted
              to 1-based indexing) that the GPU and Qiskit benchmarks use.
              Edge ordering matters because it determines gate application
              order, which affects which Pauli terms survive truncation.
  • Observable: Z on qubit 62 (0-based) = qubit 63 (1-based in Julia).
  • Circuit:    tfitrottercircuit with start_with_ZZ=true reproduces the
              GPU benchmark's "for step in 1:N: ZZ_layer; RX_layer" pattern.
  • Timing:     Julia's @elapsed macro around propagate() — high-resolution
              and excludes circuit build/JIT warm-up.  Note: the GPU benchmark
              runs on a remote GPU and uses internal per-gate-layer timing.
              Neither PauliPropagation.jl nor pauli-prop (Qiskit) expose
              built-in timing APIs, so @elapsed / time.perf_counter() are
              the best available options.
  • Deltas:     List `DELTAS` below — keep in sync with GPU/BQ sweeps you compare to.

Usage:
    julia pps_benchmark.jl              # full sweep (NUM_TRIALS per δ)
    julia pps_benchmark.jl --resume     # skip completed (delta_index, trial)

Same DELTAS, NUM_TRIALS, and circuit as pps_cpu_benchmark.py / pps_benchmark_qiskit.py.
=#

using Pkg

# ── Install dependencies if needed ────────────────────────────────────────────
for dep in ["PauliPropagation", "JSON"]
    try
        @eval using $(Symbol(dep))
    catch
        Pkg.add(dep)
        @eval using $(Symbol(dep))
    end
end

using PauliPropagation
using JSON
using Printf
using Dates
using Statistics: mean, std

# ── Output file ───────────────────────────────────────────────────────────────
const OUTPUT_DIR  = joinpath(dirname(@__DIR__), "data")
const OUTPUT_FILE = joinpath(OUTPUT_DIR, "pps_julia_benchmark.jsonl")
mkpath(OUTPUT_DIR)

# ── Parameters (matching the BlueQubit GPU benchmark exactly) ─────────────────
const NQUBITS          = 127
const NUM_TROTTER_STEPS = 20
const RZZ_ANGLE        = -π / 2      # ZZ coupling
const RX_ANGLE         = π / 4       # transverse field

# Observable: Z on qubit 62 (0-based) = qubit 63 (1-based in PauliPropagation.jl)
const OBS_QUBIT        = 63

# ── IBM Eagle 127-qubit heavy-hex topology ────────────────────────────────────
# Converted from the 0-based Python IBM_127_HEAVY_HEX_MAP to 1-based Julia
# indexing by adding 1 to every vertex.  The edge ORDER is preserved exactly
# to match the GPU and Qiskit benchmarks' gate application order.
const IBM_EAGLE_TOPOLOGY = Tuple{Int,Int}[
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (1, 15), (15, 19),
    (5, 16), (16, 23), (9, 17), (17, 27), (13, 18), (18, 31), (19, 20),
    (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
    (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (21, 34),
    (25, 35), (35, 44), (29, 36), (36, 48), (33, 37), (37, 52), (34, 40),
    (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45),
    (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52),
    (52, 53), (38, 53), (39, 54), (54, 61), (43, 55), (55, 65), (47, 56),
    (56, 69), (51, 57), (57, 73), (58, 59), (59, 60), (60, 61), (61, 62),
    (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (68, 69),
    (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (59, 75), (75, 79),
    (63, 76), (76, 83), (67, 77), (77, 87), (71, 78), (78, 91), (74, 86),
    (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86),
    (86, 87), (87, 88), (88, 89), (89, 90), (90, 91), (91, 92), (81, 93),
    (93, 103), (85, 94), (94, 101), (89, 95), (95, 105), (92, 96),
    (96, 110), (97, 98), (98, 99), (99, 100), (100, 101), (101, 102),
    (102, 103), (103, 104), (104, 105), (105, 106), (106, 107), (107, 108),
    (108, 109), (109, 110), (97, 111), (111, 119), (101, 112), (112, 123),
    (105, 113), (113, 117), (110, 114), (114, 115), (115, 116),
    (116, 117), (117, 118), (118, 119), (119, 120), (120, 121),
    (121, 122), (122, 123), (123, 124), (124, 125), (125, 126),
    (126, 127), (115, 110),
]

# Truncation thresholds to sweep — matches the GPU benchmark results exactly.
# `min_abs_coeff` in PauliPropagation.jl ≈ `truncation_threshold` / `atol` in BQ/Qiskit.
const DELTAS = [
    1.0e-2,
    5.0e-3,
    1.0e-3,
    5.0e-4,
    1.0e-4,
    5.0e-5,
    2.5e-5,
]

const NUM_TRIALS = 5

# ── Resume support ────────────────────────────────────────────────────────────
function load_completed(filepath::String)::Set{Tuple{Int,Int}}
    done = Set{Tuple{Int,Int}}()
    isfile(filepath) || return done
    for line in eachline(filepath)
        stripped = strip(line)
        (isempty(stripped) || startswith(stripped, "#")) && continue
        try
            r = JSON.parse(stripped)
            if haskey(r, "delta_index") && !haskey(r, "error")
                tr = Int(get(r, "trial", 0))
                push!(done, (Int(r["delta_index"]), tr))
            end
        catch
        end
    end
    return done
end

resume = "--resume" in ARGS
completed = resume ? load_completed(OUTPUT_FILE) : Set{Tuple{Int,Int}}()

if !resume
    open(OUTPUT_FILE, "w") do f
        println(f, "# PPS Julia Benchmark (PauliPropagation.jl) — ⟨Z_$(OBS_QUBIT-1)⟩ sweep  (qubit $(OBS_QUBIT-1) 0-based)")
        println(f, "# nqubits=$NQUBITS, trotter_steps=$NUM_TROTTER_STEPS")
        println(f, "# rx_angle=$(@sprintf("%.6f", RX_ANGLE)) (π/4), rzz_angle=$(@sprintf("%.6f", RZZ_ANGLE)) (-π/2)")
        println(f, "# deltas=$DELTAS")
        println(f, "# num_trials=$NUM_TRIALS")
        println(f, "# julia_version=$(VERSION)")
        println(f, "# started=$(now())")
    end
    println("Starting fresh.")
else
    println("Resuming: $(length(completed)) (delta_index, trial) pairs already complete.")
end

# ── Build circuit ─────────────────────────────────────────────────────────────
println("\n┌─────────────────────────────────────────────────────────────┐")
println("│  Building circuit: $(NQUBITS) qubits, $(NUM_TROTTER_STEPS) Trotter steps             │")
println("│  Topology: IBM Eagle heavy-hex (IBM_127_HEAVY_HEX_MAP)     │")
println("│  Observable: Z_$(OBS_QUBIT-1) (0-based) = Z_$(OBS_QUBIT) (1-based)                  │")
println("└─────────────────────────────────────────────────────────────┘")

# Use the IBM_127_HEAVY_HEX_MAP topology (1-based) — same edge ordering as
# the GPU and Qiskit benchmarks to ensure identical gate sequences.
topology = IBM_EAGLE_TOPOLOGY

# Build a TFI Trotter circuit on the heavy-hex topology.
# start_with_ZZ=true → each Trotter step is a full ZZ layer then full RX layer,
# matching pps_cpu_benchmark.py / pps_benchmark_qiskit.py.
circuit = tfitrottercircuit(NQUBITS, NUM_TROTTER_STEPS; topology=topology, start_with_ZZ=true)

# Set parameters: all ZZ angles = RZZ_ANGLE, all X angles = RX_ANGLE.
nparams = countparameters(circuit)
println("  Total parametrised gates: $nparams")

# Identify which parameters correspond to ZZ vs X gates and assign angles.
zz_indices = getparameterindices(circuit, PauliRotation, [:Z, :Z])
x_indices  = getparameterindices(circuit, PauliRotation, [:X])

parameters = zeros(Float64, nparams)
parameters[zz_indices] .= RZZ_ANGLE
parameters[x_indices]  .= RX_ANGLE

println("  ZZ gates: $(length(zz_indices)),  X gates: $(length(x_indices))")

# ── Observable ────────────────────────────────────────────────────────────────
# Z on qubit 63 (1-based) = qubit 62 (0-based), matching GPU benchmark's ⟨Z₆₂⟩
observable = PauliString(NQUBITS, :Z, OBS_QUBIT)
println("  Observable: Z_$(OBS_QUBIT) (1-based) = Z_$(OBS_QUBIT-1) (0-based)")

# ── Helper ────────────────────────────────────────────────────────────────────
function format_number(n::Integer)
    s = string(n)
    # Insert commas
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    pushfirst!(parts, s)
    return join(parts, ",")
end

# ── Warm-up (JIT compile with a tiny run) ─────────────────────────────────────
println("\n  Warming up JIT…")
warmup_obs = PauliString(NQUBITS, :Z, 1)
_ = propagate(circuit, warmup_obs, parameters; min_abs_coeff=0.5)
println("  JIT warm-up complete.\n")

# ── Run sweep ─────────────────────────────────────────────────────────────────
println("=" ^ 70)
println("  PPS-Julia: $NUM_TRIALS trials per δ (same settings as pps_cpu_benchmark.py)")
println("=" ^ 70)
@printf("  %-4s  %-3s  %-12s  %-14s  %-16s  %-10s  %s\n",
        "idx", "tr", "δ", "⟨Z₆₂⟩", "# Paulis", "Time (s)", "Status")
println("-" ^ 70)

for (j, delta) in enumerate(DELTAS)
    delta_index = j - 1   # 0-based to match Python

    for trial in 0:(NUM_TRIALS - 1)
        if (delta_index, trial) in completed
            @printf("  %-4d  %-3d  %-12.2e  %14s  %16s  %10s  %s\n",
                    delta_index, trial, delta, "—", "—", "—", "skip")
            continue
        end

        try
            local pauli_sum
            t_elapsed = @elapsed begin
                pauli_sum = propagate(
                    circuit, observable, parameters;
                    min_abs_coeff=delta,
                )
            end

            ev = real(overlapwithzero(pauli_sum))
            np = length(pauli_sum)

            record = Dict(
                "delta_index"       => delta_index,
                "delta"             => delta,
                "trial"             => trial,
                "num_trials"        => NUM_TRIALS,
                "expectation_value" => ev,
                "num_paulis"        => np,
                "run_time_s"        => t_elapsed,
                "num_qubits"        => NQUBITS,
                "num_trotter_steps" => NUM_TROTTER_STEPS,
                "rzz_angle"         => RZZ_ANGLE,
                "rx_angle"          => RX_ANGLE,
                "observable"        => "Z_$(OBS_QUBIT-1)",
                "package"           => "PauliPropagation.jl",
                "julia_version"     => string(VERSION),
                "timestamp"         => string(now()),
            )
            open(OUTPUT_FILE, "a") do f
                println(f, JSON.json(record))
                flush(f)
            end

            push!(completed, (delta_index, trial))

            @printf("  %-4d  %-3d  %-12.2e  %14.6f  %16s  %10.1f  %s\n",
                    delta_index, trial, delta, ev,
                    format_number(np), t_elapsed, "✓")

        catch e
            @printf("  %-4d  %-3d  %-12.2e  %14s  %16s  %10s  %s\n",
                    delta_index, trial, delta, "ERROR", "", "", "✗")
            println("        └─ ", sprint(showerror, e))

            open(OUTPUT_FILE, "a") do f
                println(f, JSON.json(Dict(
                    "delta_index" => delta_index,
                    "delta"       => delta,
                    "trial"       => trial,
                    "error"       => sprint(showerror, e),
                    "timestamp"   => string(now()),
                )))
                flush(f)
            end
        end

        GC.gc(false)
    end
end

println("=" ^ 70)
println("\nResults saved to: $OUTPUT_FILE")

# ── Summary: mean ± std per δ (from file) ─────────────────────────────────────
rows = Dict{Int,Vector{Any}}()
if isfile(OUTPUT_FILE)
    for line in eachline(OUTPUT_FILE)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        try
            r = JSON.parse(s)
            if !haskey(r, "error") && haskey(r, "expectation_value")
                di = Int(r["delta_index"])
                if !haskey(rows, di)
                    rows[di] = Any[]
                end
                push!(rows[di], r)
            end
        catch
        end
    end
end

println("\n", "=" ^ 75)
println(" " ^ 20 * "SUMMARY (mean ± std over trials)")
println("=" ^ 75)
@printf("%-5s %-12s %-14s %-12s %-12s %-12s\n",
        "idx", "δ", "mean ⟨Z⟩", "std ⟨Z⟩", "mean t(s)", "std t(s)")
println("-" ^ 75)
for (j, delta) in enumerate(DELTAS)
    di = j - 1
    sub = get(rows, di, Any[])
    if isempty(sub)
        @printf("%-5d %-12.2e  (no data)\n", di, delta)
        continue
    end
    evs = [Float64(x["expectation_value"]) for x in sub]
    ts = [Float64(x["run_time_s"]) for x in sub]
    @printf("%-5d %-12.2e %-14.6f %-12.6f %-12.4f %-12.4f\n",
            di, delta, mean(evs), std(evs), mean(ts), std(ts))
end
println("=" ^ 75)

# ── Plotting (optional, if Plots.jl is available) ─────────────────────────────
try
    @eval using Plots

    isempty(rows) && error("No valid results to plot")

    delta_log = Float64[]
    ev_m, ev_s = Float64[], Float64[]
    rt_m, rt_s = Float64[], Float64[]
    np_m, np_s = Float64[], Float64[]

    for (j, _) in enumerate(DELTAS)
        di = j - 1
        sub = get(rows, di, Any[])
        isempty(sub) && continue
        push!(delta_log, -log10(DELTAS[j]))
        evs = [Float64(x["expectation_value"]) for x in sub]
        ts = [Float64(x["run_time_s"]) for x in sub]
        nps = [Float64(x["num_paulis"]) for x in sub]
        push!(ev_m, mean(evs))
        push!(ev_s, std(evs))
        push!(rt_m, mean(ts))
        push!(rt_s, std(ts))
        push!(np_m, mean(nps))
        push!(np_s, std(nps))
    end

    p1 = plot(delta_log, ev_m; yerror=ev_s,
        xlabel="-log₁₀(δ)", ylabel="⟨Z₆₂⟩",
        title="Expectation (mean ± std)", marker=:circle, legend=false,
        ylims=(-0.05, 1.05), grid=true,
    )
    p2 = plot(delta_log, rt_m; yerror=rt_s,
        xlabel="-log₁₀(δ)", ylabel="Runtime (s)",
        title="Runtime (mean ± std)", marker=:circle, legend=false,
        yscale=:log10, grid=true,
    )
    p3 = plot(delta_log, np_m; yerror=np_s,
        xlabel="-log₁₀(δ)", ylabel="# Pauli strings",
        title="# Paulis (mean ± std)", marker=:circle, legend=false,
        yscale=:log10, grid=true,
    )
    p = plot(p1, p2, p3; layout=(1, 3), size=(1500, 450),
        plot_title="PauliPropagation.jl — $(NQUBITS)q, $(NUM_TROTTER_STEPS) steps, N=$(NUM_TRIALS) trials/δ",
    )
    out_png = joinpath(OUTPUT_DIR, "pps_julia_benchmark.png")
    savefig(p, out_png)
    println("\nPlot saved to: $out_png")

catch e
    println("\n(Plotting skipped — install Plots.jl: Pkg.add(\"Plots\"))")
end