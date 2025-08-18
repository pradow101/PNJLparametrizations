begin
    include("parameterslog.jl")
    include("functionslog.jl")

    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations, NonlinearSolve
    plotly()
end


function gapsystem!(du, u, p)
    du[1] = dphilog(u[1],u[2],p[1],p[2],u[3])
    du[2] = dphiblog(u[1],u[2],p[1],p[2],u[3])
    du[3] = dMlog(u[1],u[2],p[1],p[2],u[3])
end

function gapsolver(mu, T)
    chutealto = [0.01,0.01,0.34]
    chutebaixo = [0.9,0.9,0.01]
    ad = AutoFiniteDiff()

    probalto = NonlinearProblem(gapsystem!,chutealto, [mu, T])
    solalto = solve(probalto, NewtonRaphson(; autodiff=ad), abstol=1e-8, maxiters = 100)
    probbaixo = NonlinearProblem(gapsystem!,chutebaixo, [mu, T])
    solbaixo = solve(probbaixo, NewtonRaphson(; autodiff=ad), abstol=1e-8, maxiters = 100)

    if abs(solalto.u[3] - solbaixo.u[3]) < 1e-4
        return [solalto.u[1], solalto.u[2], solalto.u[3], solbaixo.u[1], solbaixo.u[2], solbaixo.u[3]]
    elseif potentiallog(solbaixo.u[1], solbaixo.u[2], mu, T, solbaixo.u[3]) < potentiallog(solalto.u[1], solalto.u[2], mu, T, solalto.u[3])
        return [solbaixo.u[1], solbaixo.u[2], solbaixo.u[3], solalto.u[1], solalto.u[2], solalto.u[3]]
    end
    return [solalto.u[1], solalto.u[2], solalto.u[3], solbaixo.u[1], solbaixo.u[2], solbaixo.u[3]]
end
 

begin
    gapsolver(0.15, 0.1)
end

function solvermurange(mu, T)
    sols = zeros(length(mu), 6)
    for i in eachindex(mu)
        sols[i, :] = gapsolver(mu[i], T)
    end
    return sols
end

function solverTrange(mu, T)
    sols = zeros(length(T), 6)
    for i in eachindex(T)
        sols[i, :] = gapsolver(mu, T[i])
    end
    return sols
end

begin
    Trange = range(0.01,0.5,length=50)
    mu = 0.1
    sols = solverTrange(mu, Trange)
end


begin
    Mrange = range(-0.6,0.6, length=100)
    pot = [potentiallog(0.01, 0.01, 0.1, 0.6, M) for M in Mrange]
    plot(Mrange, pot)
end


function gapsolver2(mu,T,chute)
    sist = nlsolve(x -> [dphilog(x[1], x[2], mu, T, x[3]), dphiblog(x[1], x[2], mu, T, x[3]), dMlog(x[1], x[2], mu, T, x[3])], chute)
    return sist.zero
end


function gapsolver2range(mu, T, chute)
    sols = zeros(length(T), 3)
    for i in eachindex(T)
        sols[i, :] = gapsolver2(mu, T[i], chute)
        chute = sols[i, :]
    end
    return sols
end


function gapsolver2murange(mu, T, chute)
    sols = zeros(length(mu), 3)
    for i in eachindex(mu)
        sols[i, :] = gapsolver2(mu[i], T, chute)
        chute = sols[i, :]
    end
    return sols
end


begin
    rangemu = range(0.0,0.5,length=100)
    T = 0.05
    solucao = gapsolver2murange(rangemu, T, [0.01,0.01,0.4])
end

begin
    plot(rangemu, solucao[:,3])
end