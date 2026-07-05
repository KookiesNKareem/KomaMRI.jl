# Benchmark the direct BlochSimple CPU core fixture used in PR #2.
#
# Run from the repository root:
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=KomaMRICore/test --startup-file=no benchmarks/blochsimple_cpu_core_bench.jl
#
# Optional:
#   BLOCHSIMPLE_BENCH_SPINS=2,1024
#   BLOCHSIMPLE_BENCH_REACTANT=0
#
# The test environment supplies Reactant. This file pushes the benchmarks
# environment onto LOAD_PATH for BenchmarkTools.

push!(LOAD_PATH, @__DIR__)

using BenchmarkTools
using KomaMRIBase
using KomaMRICore
using Logging
using Reactant
using Statistics: median, quantile

const RUN_REACTANT = get(ENV, "BLOCHSIMPLE_BENCH_REACTANT", "1") != "0"
if RUN_REACTANT
    Reactant.set_default_backend("cpu")
    Reactant.allowscalar(false)
end

disable_logging(Logging.Info)

const TVEC = [
    -1.0e-14,
    0.0,
    1.0e-14,
    0.00020000000001,
    0.0002853658536585366,
    0.0004853658536585366,
    0.0005999999999899999,
    0.0006,
    0.0008666666666666666,
    0.0011333333333333332,
    0.0014,
    0.00140000000001,
]
const DTVEC = [
    1.0e-14,
    1.0e-14,
    0.0002,
    8.536585364853658e-5,
    0.00020000000000000004,
    0.00011463414633146329,
    1.0000030317702802e-14,
    0.0002666666666666667,
    0.00026666666666666657,
    0.0002666666666666668,
    1.0000030317702802e-14,
]
const PARTS = UnitRange{Int64}[1:2, 2:8, 8:11]
const EXCITATION_BOOL = Bool[0, 1, 0]
const ADC12 = falses(12)
const RF0 = [1.3, 1.7, 1.1]
const SPIN_COUNTS = parse.(Int, split(get(ENV, "BLOCHSIMPLE_BENCH_SPINS", "2,1024,8192,65536"), ","))

function make_inputs(n)
    x = collect(range(-1e-2, 1e-2; length=n))
    rho = ones(Float64, n)
    T1 = fill(1.0, n)
    T2 = fill(0.08, n)
    dw = 2π .* collect(range(-12.0, 10.0; length=n))
    target_z = fill(0.4, n)
    B1 = zeros(ComplexF64, 12)
    B1[3] = complex(RF0[1] * 1e-6)
    B1[4] = complex((0.5RF0[1] + 0.5RF0[2]) * 1e-6)
    B1[5] = complex(RF0[2] * 1e-6)
    B1[6] = complex((0.5RF0[2] + 0.5RF0[3]) * 1e-6)
    B1[7] = complex(RF0[3] * 1e-6)
    z12 = zeros(Float64, 12)
    xy0 = zeros(ComplexF64, n)
    z0 = copy(rho)
    return x, rho, T1, T2, dw, target_z, B1, z12, xy0, z0
end

function core_loss_inputs(x, rho, T1, T2, dw, target_z, B1, z12, xy0, z0)
    obj = Phantom(; x=x, ρ=rho, T1=T1, T2=T2, Δw=dw)
    seqd = DiscreteSequence(
        z12,
        copy(z12),
        copy(z12),
        B1,
        copy(z12),
        copy(z12),
        ADC12,
        TVEC,
        DTVEC,
    )
    Xt = Mag(copy(xy0), copy(z0))
    sig = similar(B1, ComplexF64, 0, 1, 1)
    KomaMRICore.run_sim_time_iter!(
        obj,
        seqd,
        sig,
        Xt,
        KomaMRICore.BlochSimple(),
        KomaMRICore.KA.CPU();
        Nblocks=3,
        Nthreads=1,
        precession_groupsize=256,
        excitation_groupsize=256,
        parts=PARTS,
        excitation_bool=EXCITATION_BOOL,
        callbacks=(),
    )
    return sum(abs2, Xt.xy) + sum(abs2, Xt.z .- target_z)
end

run_native(args) = core_loss_inputs(args...)
run_compiled(compiled, args) = Reactant.to_number(compiled(args...))
p95_us(t) = quantile(t.times, 0.95) / 1000
median_us(t) = median(t.times) / 1000

function benchmark_native(args, samples, seconds)
    loss = run_native(args)
    trial = @benchmark run_native($args) samples=samples evals=1 seconds=seconds
    return loss, trial
end

function print_trial(case, trial, loss, compile_s, samples)
    println(
        join(
            (
                case,
                round(median_us(trial); digits=2),
                round(p95_us(trial); digits=2),
                trial.memory,
                loss,
                compile_s,
                length(trial.times) == samples ? samples : length(trial.times),
            ),
            '\t',
        ),
    )
    flush(stdout)
end

println("case\tmedian_us\tp95_us\talloc_B\tloss\tcompile_s\tsamples")
flush(stdout)

for n in SPIN_COUNTS
    samples = n == 65536 ? 100 : 200
    seconds = n == 65536 ? 8 : 5
    args = make_inputs(n)

    native_loss, native_trial = benchmark_native(args, samples, seconds)
    print_trial("current native core loss, $(n) spins", native_trial, native_loss, "N/A", samples)

    if RUN_REACTANT
        react_args = map(Reactant.to_rarray, args)
        compile_seconds = @elapsed compiled = Reactant.@compile sync=true core_loss_inputs(react_args...)
        compiled_loss = run_compiled(compiled, react_args)
        compiled_trial = @benchmark run_compiled($compiled, $react_args) samples=samples evals=1 seconds=seconds
        print_trial(
            "current reactant compiled core loss, $(n) spins",
            compiled_trial,
            compiled_loss,
            round(compile_seconds; digits=2),
            samples,
        )
    end
end
