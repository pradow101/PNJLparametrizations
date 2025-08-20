begin
    include("parameterslog.jl")
    include("functionslog.jl")

    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations, NonlinearSolve
    plotly()
end

function gapsystem!(du, u, p)
    #=Esse "clamp" restringe as soluções. Definitivamente não é o ideal quando o modelo é desconhecido, mas
    sem isso o solver fica muito instável e não converge corretamente
    =#

    u[1] = clamp(u[1], 0.0, 1.0)
    u[2] = clamp(u[2], 0.0, 1.0)
    u[3] = max(u[3], 1e-5)

    println("Guess: phi = ", round(u[1], digits=3), ", phib = ", round(u[2], digits=3), ", M = ", round(u[3], digits=3), " for μ = ", round(p[1], digits=3), ", T = ", round(p[2], digits=3))

    du[1] = dphilog(u[1],u[2],p[1],p[2],u[3])
    du[2] = dphiblog(u[1],u[2],p[1],p[2],u[3])
    du[3] = dMlog(u[1],u[2],p[1],p[2],u[3])
end

function gapsolver(mu, T)
    chutealto = [0.01,0.01,0.34]
    chutebaixo = [0.9,0.9,0.01]
    ad = AutoFiniteDiff()

    probalto = NonlinearProblem(gapsystem!,chutealto, [mu, T])
    solalto = solve(probalto, TrustRegion(; autodiff=ad), abstol=1e-8, maxiters = 100)
    probbaixo = NonlinearProblem(gapsystem!,chutebaixo, [mu, T])
    solbaixo = solve(probbaixo, TrustRegion(; autodiff=ad), abstol=1e-8, maxiters = 100)

    if abs(solalto.u[3] - solbaixo.u[3]) < 1e-3
        return [solalto.u[1], solalto.u[2], solalto.u[3], solbaixo.u[1], solbaixo.u[2], solbaixo.u[3]]
    elseif potentiallog(solbaixo.u[1], solbaixo.u[2], mu, T, solbaixo.u[3]) < potentiallog(solalto.u[1], solalto.u[2], mu, T, solalto.u[3])
        return [solbaixo.u[1], solbaixo.u[2], solbaixo.u[3], solalto.u[1], solalto.u[2], solalto.u[3]]
    end
    return [solalto.u[1], solalto.u[2], solalto.u[3], solbaixo.u[1], solbaixo.u[2], solbaixo.u[3]]
end
 

begin
    gapsolver(0.2, 0.1)
end

function solvermurange(mu, T)
    sols = zeros(length(mu), 6)
    Threads.@threads for i in eachindex(mu)
        sols[i, :] = gapsolver(mu[i], T)
    end
    return sols
end

function solverTrange(mu, T)
    sols = zeros(length(T), 6)
    Threads.@threads for i in eachindex(T)
        sols[i, :] = gapsolver(mu, T[i])
    end
    return sols
end

function Tmusolver(mur, Tr)
    sols = zeros(length(mur), 6, length(Tr))
    Threads.@threads for i in eachindex(Tr)
        sols[:, :, i] = solvermurange(mur, Tr[i])
    end
    return sols
end


@time begin
    mur = range(0, 0.6, length = 100)
    T = 0.05
    solarr = solvermurange(mur, T)
    scatter(mur, solarr[:,3])
end



@time begin
    mur = range(0,0.6,length = 1000)
    Tr = range(0.09, 0.30, length = 30)
    allsols = Tmusolver(mur, Tr)
end


begin
    scatter(mur, [allsols[:, 3, 1], allsols[:, 3, 25], allsols[:, 3, 30]], xlabel = "μ [GeV]", ylabel = "M [GeV]")
end


#Agora para a fase quarkyonica#

function quarkyonic(mu, Msols)
    diffs = abs.(diff(Msols))

    jump_index = argmax(diffs)

    return mu[jump_index+1], Msols[jump_index+1]
end

function quarkyonicall(mur, Tr, sols)  
    mu_quark = zeros(length(Tr))
    T_quark = zeros(length(Tr))
    for i in eachindex(Tr)
        for j in eachindex(mur)
            if sols[j, 3, i] < mur[j]
                mu_quark[i] = mur[j]
                T_quark[i] = Tr[i]
                break
            end
        end
    end
    return mu_quark, T_quark
end

begin
    muquark, Tquark = quarkyonicall(mur, Tr, allsols)
    scatter(muquark, Tquark)
end