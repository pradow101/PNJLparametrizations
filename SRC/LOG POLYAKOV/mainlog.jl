begin
    include("parameterslog.jl")
    include("functionslog.jl")

    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations, NonlinearSolve
end


function gapsystem!(du, u, p)
    du[1] = dphilog(u[1],u[2],p[1],p[2],u[3])
    du[2] = dphiblog(u[1],u[2],p[1],p[2],u[3])
    du[3] = dMlog(u[1],u[2],p[1],p[2],u[3])
end

function gapsolver(mu, T)
    chutealto = [0.01,0.01,1.0]
    chutebaixo = [1.0,1.0,0.01]
    ad = AutoFiniteDiff()

    probalto = NonlinearProblem(gapsystem!,chutealto, [mu, T])
    solalto = solve(probalto, NewtonRaphson(; autodiff=ad), abstol=1e-8, maxiters = 100)
    probbaixo = NonlinearProblem(gapsystem!,chutebaixo, [mu, T])
    solbaixo = solve(probbaixo, NewtonRaphson(; autodiff=ad), abstol=1e-8, maxiters = 100)

    if abs(solalto.u[1] - solbaixo.u[1]) < 1e-4
        return [solalto.u[1], solalto.u[2], solalto.u[3], solbaixo.u[1], solbaixo.u[2], solbaixo.u[3]]
    elseif potentiallog(solbaixo.u[1], solbaixo.u[2], mu, T, solbaixo.u[3]) < potentiallog(solalto.u[1], solalto.u[2], mu, T, solalto.u[3])
        return [solbaixo.u[1], solbaixo.u[2], solbaixo.u[3], solalto.u[1], solalto.u[2], solalto.u[3]]
    end
    solalto.u, solbaixo.u
end




begin
    sols1, sols2 = gapsolver(0.2, 0.1)
end