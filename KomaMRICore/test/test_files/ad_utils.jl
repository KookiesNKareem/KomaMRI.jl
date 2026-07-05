using FiniteDifferences: central_fdm, grad
using KomaMRIBase
using KomaMRICore

const BLOCHSIMPLE_AD_RF0 = [1.3, 1.7, 1.1]
const BLOCHSIMPLE_AD_DIRECTION = [0.2, -0.1, 0.15]

function blochsimple_ad_sequence(rf_scale)
    Trf = 0.6e-3
    Tadc = 0.8e-3
    gr = Grad(0.0, 0.0)
    GR = [gr gr; gr gr; gr gr]
    rf_off = RF(zeros(ComplexF64, length(rf_scale)), 0.0, 0.0, 0.0, 0.0, 0.0, Undefined(), Val(:preserve))
    RFs = [RF(complex.(rf_scale) .* 1e-6, Trf) rf_off]
    ADCs = [ADC(0, 0.0), ADC(4, Tadc)]
    return Sequence(GR, RFs, ADCs)
end

function blochsimple_ad_loss(rf_scale)
    obj = Phantom(
        x=[0.0, 1e-2],
        ρ=[1.0, 0.75],
        T1=[1.0, 0.8],
        T2=[0.08, 0.12],
        Δw=2π .* [10.0, -12.0],
    )
    sim_params = Dict{String, Any}(
        "sim_method" => KomaMRICore.BlochSimple(),
        "gpu" => false,
        "Nthreads" => 1,
        "return_type" => "state",
        "precision" => "f64",
        "Δt_rf" => 0.2e-3,
    )
    M = simulate(obj, blochsimple_ad_sequence(rf_scale), Scanner(); sim_params, verbose=false)
    target_z = [0.4, 0.2]
    return sum(abs2, M.xy) + sum(abs2, M.z .- target_z)
end

function blochsimple_ad_discrete_sequence(rf_scale)
    z = zeros(Float64, 12)
    B1 = zeros(ComplexF64, 12)
    B1[3] = complex(rf_scale[1] * 1e-6)
    B1[4] = complex((0.5rf_scale[1] + 0.5rf_scale[2]) * 1e-6)
    B1[5] = complex(rf_scale[2] * 1e-6)
    B1[6] = complex((0.5rf_scale[2] + 0.5rf_scale[3]) * 1e-6)
    B1[7] = complex(rf_scale[3] * 1e-6)
    ADC = Bool[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    t = [
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
    Δt = [
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
    return DiscreteSequence(z, copy(z), copy(z), B1, copy(z), copy(z), ADC, t, Δt)
end

function blochsimple_ad_lowlevel_sequence(rf_scale)
    Trf = 0.6e-3
    Tadc = 0.8e-3
    gr = Grad(0.0, 0.0)
    GR = [gr gr; gr gr; gr gr]
    rf = RF(
        complex.(rf_scale) .* 1e-6,
        Trf,
        0.0,
        0.0,
        Trf / 2,
        0.0,
        Undefined(),
        Val(:preserve),
    )
    rf_off = RF(
        zeros(ComplexF64, length(rf_scale)),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        Undefined(),
        Val(:preserve),
    )
    return Sequence(
        GR,
        [rf rf_off],
        [ADC(0, 0.0), ADC(4, Tadc)],
        [Trf, Tadc],
        [Extension[], Extension[]],
        Dict{String, Any}(),
    )
end

function blochsimple_ad_discretize_loss(rf_scale)
    obj = Phantom(
        x=[0.0, 1e-2],
        ρ=[1.0, 0.75],
        T1=[1.0, 0.8],
        T2=[0.08, 0.12],
        Δw=2π .* [10.0, -12.0],
    )
    seq = blochsimple_ad_lowlevel_sequence(rf_scale)
    seqd = discretize(seq; sampling_params=Dict("Δt" => 1e-3, "Δt_rf" => 0.2e-3))
    parts, excitation_bool = KomaMRICore.get_sim_ranges(
        seqd;
        max_block_length=512,
        max_rf_block_length=Inf,
    )
    Xt, obj = KomaMRICore.initialize_spins_state(obj, KomaMRICore.BlochSimple())
    sig = zeros(ComplexF64, 4, 1, 1)
    KomaMRICore.run_sim_time_iter!(
        obj,
        seqd,
        sig,
        Xt,
        KomaMRICore.BlochSimple(),
        KomaMRICore.KA.CPU();
        Nblocks=length(parts),
        Nthreads=1,
        precession_groupsize=256,
        excitation_groupsize=256,
        parts=parts,
        excitation_bool=excitation_bool,
        callbacks=(),
    )
    target_z = [0.4, 0.2]
    return sum(abs2, Xt.xy) + sum(abs2, Xt.z .- target_z)
end

function blochsimple_ad_reactant_vector(rf_scale, vals)
    seed = Reactant.allowscalar() do
        rf_scale[1]
    end
    out = similar(rf_scale, Float64, length(vals))
    Reactant.allowscalar() do
        for i in eachindex(vals)
            out[i] = vals[i] * one(seed)
        end
    end
    return out
end

function blochsimple_ad_reactant_discrete_sequence(rf_scale)
    z = blochsimple_ad_reactant_vector(rf_scale, zeros(Float64, 12))
    B1 = similar(rf_scale, ComplexF64, 12)
    a1, a2, a3 = Reactant.allowscalar() do
        (rf_scale[1], rf_scale[2], rf_scale[3])
    end
    Reactant.allowscalar() do
        for i in eachindex(B1)
            B1[i] = complex(zero(a1))
        end
        B1[3] = complex(a1 * 1e-6)
        B1[4] = complex((0.5a1 + 0.5a2) * 1e-6)
        B1[5] = complex(a2 * 1e-6)
        B1[6] = complex((0.5a2 + 0.5a3) * 1e-6)
        B1[7] = complex(a3 * 1e-6)
    end
    ADC = falses(12)
    t = [
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
    Δt = [
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
    return DiscreteSequence(z, copy(z), copy(z), B1, copy(z), copy(z), ADC, t, Δt)
end

function blochsimple_ad_core_loss(rf_scale)
    obj = Phantom(
        x=[0.0, 1e-2],
        ρ=[1.0, 0.75],
        T1=[1.0, 0.8],
        T2=[0.08, 0.12],
        Δw=2π .* [10.0, -12.0],
    )
    seqd = blochsimple_ad_discrete_sequence(rf_scale)
    Xt, obj = KomaMRICore.initialize_spins_state(obj, KomaMRICore.BlochSimple())
    sig = zeros(ComplexF64, 4, 1, 1)
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
        parts=UnitRange{Int64}[1:2, 2:8, 8:11],
        excitation_bool=Bool[0, 1, 0],
        callbacks=(),
    )
    target_z = [0.4, 0.2]
    return sum(abs2, Xt.xy) + sum(abs2, Xt.z .- target_z)
end

function blochsimple_ad_reactant_core_loss(rf_scale)
    obj = Phantom(
        x=blochsimple_ad_reactant_vector(rf_scale, [0.0, 1e-2]),
        ρ=blochsimple_ad_reactant_vector(rf_scale, [1.0, 0.75]),
        T1=blochsimple_ad_reactant_vector(rf_scale, [1.0, 0.8]),
        T2=blochsimple_ad_reactant_vector(rf_scale, [0.08, 0.12]),
        Δw=blochsimple_ad_reactant_vector(rf_scale, [2π * 10.0, -2π * 12.0]),
    )
    seqd = blochsimple_ad_reactant_discrete_sequence(rf_scale)
    xy = similar(rf_scale, ComplexF64, 2)
    z = similar(rf_scale, Float64, 2)
    sig = similar(rf_scale, ComplexF64, 4, 1, 1)
    Reactant.allowscalar() do
        for i in eachindex(xy)
            xy[i] = 0.0 + 0.0im
        end
        z[1] = 1.0
        z[2] = 0.75
        for i in eachindex(sig)
            sig[i] = 0.0 + 0.0im
        end
    end
    Xt = Mag(xy, z)
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
        parts=UnitRange{Int64}[1:2, 2:8, 8:11],
        excitation_bool=Bool[0, 1, 0],
        callbacks=(),
    )
    target_z = blochsimple_ad_reactant_vector(rf_scale, [0.4, 0.2])
    return sum(abs2, Xt.xy) + sum(abs2, Xt.z .- target_z)
end

function reactant_backend_available(backend)
    previous = Reactant.XLA.default_backend()
    try
        Reactant.set_default_backend(backend)
        return lowercase(String(Reactant.XLA.platform_name(Reactant.XLA.default_backend())))
    catch
        return nothing
    finally
        Reactant.set_default_backend(previous)
    end
end

function with_reactant_backend(f, backend)
    previous = Reactant.XLA.default_backend()
    Reactant.set_default_backend(backend)
    try
        return f(lowercase(String(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))))
    finally
        Reactant.set_default_backend(previous)
    end
end

blochsimple_ad_fd_gradient(rf_scale=BLOCHSIMPLE_AD_RF0) =
    grad(central_fdm(5, 1), blochsimple_ad_loss, rf_scale)[1]

blochsimple_ad_core_fd_gradient(rf_scale=BLOCHSIMPLE_AD_RF0) =
    grad(central_fdm(5, 1), blochsimple_ad_core_loss, rf_scale)[1]

blochsimple_ad_discretize_fd_gradient(rf_scale=BLOCHSIMPLE_AD_RF0) =
    grad(central_fdm(5, 1), blochsimple_ad_discretize_loss, rf_scale)[1]

blochsimple_ad_reactant_core_fd_gradient(rf_scale=BLOCHSIMPLE_AD_RF0) =
    grad(central_fdm(5, 1), blochsimple_ad_reactant_core_loss, rf_scale)[1]

function blochsimple_ad_gradient_matches_fd(ad_grad; rf_scale=BLOCHSIMPLE_AD_RF0)
    fd_grad = blochsimple_ad_fd_gradient(rf_scale)
    return all(isfinite, ad_grad) && isapprox(ad_grad, fd_grad; rtol=1e-3, atol=1e-7)
end

function blochsimple_ad_core_gradient_matches_fd(ad_grad; rf_scale=BLOCHSIMPLE_AD_RF0)
    fd_grad = blochsimple_ad_core_fd_gradient(rf_scale)
    return all(isfinite, ad_grad) && isapprox(ad_grad, fd_grad; rtol=1e-3, atol=1e-7)
end

function blochsimple_ad_discretize_gradient_matches_fd(ad_grad; rf_scale=BLOCHSIMPLE_AD_RF0)
    fd_grad = blochsimple_ad_discretize_fd_gradient(rf_scale)
    return all(isfinite, ad_grad) && isapprox(ad_grad, fd_grad; rtol=1e-3, atol=1e-7)
end

function blochsimple_ad_reactant_core_gradient_matches_fd(ad_grad; rf_scale=BLOCHSIMPLE_AD_RF0)
    fd_grad = blochsimple_ad_reactant_core_fd_gradient(rf_scale)
    return all(isfinite, ad_grad) && isapprox(ad_grad, fd_grad; rtol=1e-3, atol=1e-7)
end
