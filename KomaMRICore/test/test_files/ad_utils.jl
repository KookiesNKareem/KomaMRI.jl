using FiniteDifferences: central_fdm, grad

const BLOCHSIMPLE_AD_RF0 = [1.3, 1.7, 1.1]
const BLOCHSIMPLE_AD_DIRECTION = [0.2, -0.1, 0.15]

function blochsimple_ad_sequence(rf_scale)
    Trf = 0.6e-3
    Tadc = 0.8e-3
    seq = Sequence()
    @addblock seq += RF(complex.(rf_scale) .* 1e-6, Trf)
    @addblock seq += ADC(4, Tadc)
    return seq
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

blochsimple_ad_fd_gradient(rf_scale=BLOCHSIMPLE_AD_RF0) =
    grad(central_fdm(5, 1), blochsimple_ad_loss, rf_scale)[1]

function blochsimple_ad_gradient_matches_fd(ad_grad; rf_scale=BLOCHSIMPLE_AD_RF0)
    fd_grad = blochsimple_ad_fd_gradient(rf_scale)
    return all(isfinite, ad_grad) && isapprox(ad_grad, fd_grad; rtol=1e-3, atol=1e-7)
end

const BLOCHSIMPLE_DISCRETIZE_AD_RF0 = [1.3, 1.7, 1.1]

function blochsimple_discretize_ad_vector(template, values, ::Type{T}=Float64) where {T}
    out = similar(template, T, length(values))
    Reactant.allowscalar() do
        for i in eachindex(values)
            out[i] = values[i]
        end
    end
    return out
end

function blochsimple_discretize_ad_sequence(rf_scale; adc_delay=0.0)
    rf_duration = 0.6e-3
    total_duration = 1.4e-3
    adc_duration = total_duration - adc_delay
    rf = RF(
        complex.(rf_scale) .* 1e-6,
        rf_duration,
        0.0,
        0.0,
        rf_duration / 2,
        0.0,
        Excitation(),
        Val(:preserve),
    )
    gradient = Grad(0.0, 0.0)
    adc = ADC(3, adc_duration, adc_delay)
    return Sequence(
        reshape([gradient, gradient, gradient], 3, 1),
        reshape([rf], 1, 1),
        [adc],
        [total_duration],
        [Extension[]],
        Dict{String,Any}(),
    )
end

function blochsimple_simulate_ad_loss(rf_scale)
    zeros2 = zero.(rf_scale[1:2])
    density = blochsimple_discretize_ad_vector(rf_scale, (1.0, 0.75))
    obj = Phantom(
        x=copy(zeros2),
        ρ=density,
        T1=one.(density),
        T2=0.1 .* one.(density),
        Δw=copy(zeros2),
    )
    sim_params = Dict{String,Any}(
        "sim_method" => KomaMRICore.BlochSimple(),
        "gpu" => false,
        "Nthreads" => 1,
        "return_type" => "mat",
        "precision" => "f64",
        "sampling_rule" => MaxStepSizeRule(1e-3, 0.2e-3),
    )
    signal = simulate(
        obj,
        blochsimple_discretize_ad_sequence(rf_scale),
        Scanner();
        sim_params,
        verbose=false,
    )
    return sum(abs2, signal)
end

blochsimple_simulate_ad_fd_gradient(rf_scale=BLOCHSIMPLE_DISCRETIZE_AD_RF0) =
    grad(central_fdm(5, 1), blochsimple_simulate_ad_loss, rf_scale)[1]

const BLOCHSIMPLE_NODE_AD_RF0 = [0.0, 3.5, 0.0]

function blochsimple_node_ad_parameters()
    rf_duration = 0.6e-3
    z = collect(range(-4e-3, 4e-3; length=3))
    obj = Phantom(
        x=zeros(length(z)),
        y=zeros(length(z)),
        z=z,
        ρ=ones(length(z)),
        T1=ones(length(z)),
        T2=fill(0.1, length(z)),
        Δw=zeros(length(z)),
    )
    rf = RF(
        zeros(ComplexF64, 7),
        rf_duration,
        0.0,
        0.0,
        rf_duration / 2,
        0.0,
        Excitation(),
        Val(:preserve),
    )
    zero_gradient = Grad(0.0, 0.0)
    slice_gradient = Grad(8e-3, rf_duration)
    seq = Sequence(
        reshape([zero_gradient, zero_gradient, slice_gradient], 3, 1),
        reshape([rf], 1, 1),
        [ADC(0, 0.0)],
        [rf_duration],
        [Extension[]],
        Dict{String,Any}(),
    )
    sim_params = Dict{String,Any}(
        "sim_method" => KomaMRICore.BlochSimple(),
        "gpu" => false,
        "Nthreads" => 1,
        "return_type" => "state",
        "precision" => "f64",
        "sampling_rule" => MaxStepSizeRule(50e-6, 25e-6),
    )
    params = (;
        seq,
        obj,
        sys=Scanner(),
        sim_params,
        target_profile=zeros(ComplexF64, length(z)),
        node_times=range(0.0, rf_duration; length=3),
        rf_times=range(0.0, rf_duration; length=7),
        rf_scale=1e-6,
    )
    target_profile = copy(blochsimple_node_ad_forward([0.0, 5.0, 0.0], params).xy)
    return merge(params, (; target_profile))
end

function blochsimple_node_ad_forward(x, params)
    seq_aux = copy(params.seq)
    rf_samples = KomaMRIBase.linear_interpolate_samples(
        (t=params.node_times, A=x),
        params.rf_times,
    )
    seq_aux.RF[1].A .= complex.(rf_samples) .* params.rf_scale
    return simulate(
        params.obj,
        seq_aux,
        params.sys;
        sim_params=params.sim_params,
        verbose=false,
    )
end

function blochsimple_node_ad_loss(x, params)
    mag = blochsimple_node_ad_forward(x, params)
    return sum(abs2, mag.xy .- params.target_profile) / length(mag.xy)
end

blochsimple_node_ad_fd_gradient(params, x=BLOCHSIMPLE_NODE_AD_RF0) =
    grad(central_fdm(5, 1), x -> blochsimple_node_ad_loss(x, params), x)[1]

function blochsimple_node_ad_reactant_parameters(params)
    rf = params.seq.RF[1]
    rf_ra = RF(
        Reactant.to_rarray(rf.A),
        rf.T,
        rf.Δf,
        rf.delay,
        rf.center,
        rf.ϕ,
        rf.use,
        Val(:preserve),
    )
    seq_ra = Sequence(
        params.seq.GR,
        reshape([rf_ra], 1, 1),
        params.seq.ADC,
        params.seq.DUR,
        params.seq.EXT,
        params.seq.DEF,
    )
    return merge(params, (;
        seq=seq_ra,
        obj=Reactant.to_rarray(params.obj),
        target_profile=Reactant.to_rarray(params.target_profile),
    ))
end
