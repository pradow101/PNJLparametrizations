begin
    include("parameterslog.jl")
    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations, NonlinearSolve
end

begin
    Ep(p,M) = sqrt(p^2 + M^2)

    Z_1(mu, T, M, p) = 1 + exp(-(Ep(p,M) - mu)/T)

    Z_2(mu, T, M, p) = 1 + exp(-(Ep(p,M) + mu)/T)
end

function Imed(mu, T, M)
    quadgk(p -> p^2 * T*log(Z_1(mu, T, M, p) * Z_2(mu, T, M, p)), 0, Inf)[1]
end

function Ivac(M)
    quadgk(p -> p^2 * Ep(p,M), 0, L)[1]
end

function Tpot(mu, T, M)
    a = (M-m)^2/4G
    b = Nc*Nf/π^2
    return a - b*(Ivac(M) + Imed(mu, T, M))
end

function gap(mu, T, M)
    ForwardDiff.derivative(Mi -> Tpot(mu, T, Mi), M)
end

function systemgap(du, u, p)
    du[1] = gap(p[1], p[2], u[1])
end

function gapsolver(mu, T)
    x0alto = [0.34]
    x0baixo = [m]
    ad = AutoFiniteDiff()

    probalto = NonlinearProblem(systemgap, x0alto, [mu, T])
    solalto = solve(probalto, NewtonRaphson(; autodiff = ad), abstol = 1e-8, maxiters = 100)
    probbaixo = NonlinearProblem(systemgap, x0baixo, [mu, T])
    solbaixo = solve(probbaixo, NewtonRaphson(; autodiff = ad), abstol = 1e-8, maxiters = 100)

    if (solalto.u[1] - solbaixo.u[1]) < 1e-4
        return [solalto.u[1], solbaixo.u[1]]
    elseif Tpot(mu, T, solbaixo.u[1]) < Tpot(mu, T, solalto.u[1])
        return [solbaixo.u[1], solalto.u[1]]
    end
    [solalto.u[1], solbaixo.u[1]]
end




function solvermurange(muvals, T)
    sols = zeros(length(muvals),2)
    Threads.@threads for i in eachindex(muvals)
        sols[i, :] = gapsolver(muvals[i], T)
    end
    return sols
end


begin
    murange = range(0, 0.5, length = 100)
    T = 0.07
    Msols = solvermurange(murange, T)
end

begin
    scatter(murange, Msols[:,1])
end


function Trangesolver(muvals, Tvals)
    sols = zeros(length(muvals), 2, length(Tvals))
    Threads.@threads for i in eachindex(Tvals)
        sols[:, :, i] = solvermurange(muvals, Tvals[i])
    end
    return sols
end

begin
    rangeT = range(0.03, 0.2, length = 100)
    muv = murange
    allsols = Trangesolver(muv, rangeT)
end

begin
    scatter(muv, [allsols[:, 1, 15], allsols[:, 1, 50], allsols[:, 1, 75]])
end

