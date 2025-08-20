begin
    include("parameterslog.jl")
    include("functionslog.jl")

    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations, NonlinearSolve, NPZ
    plotly()
end

function gapsystem!(du, u, p)
    #=Esse "clamp" restringe as soluções. Definitivamente não é o ideal quando o modelo é desconhecido, mas
    sem isso o solver fica muito instável e não converge corretamente
    =#

    u[1] = clamp(u[1], 0.0, 1.0)
    u[2] = clamp(u[2], 0.0, 1.0)
    u[3] = clamp(u[3], 0.0, 1.2)

    println("phi = ", u[1], " phib = ", u[2], " M = ", u[3], " mu = ", p[1], " T = ", p[2])

    du[1] = dphilog(u[1],u[2],p[1],p[2],u[3])
    du[2] = dphiblog(u[1],u[2],p[1],p[2],u[3])
    du[3] = dMlog(u[1],u[2],p[1],p[2],u[3])
end

function gapsolver(mu, T)
    chutealto = [0.01,0.01,0.4]
    chutebaixo = [0.9,0.9,0.01]
    ad = AutoFiniteDiff()

    probalto = NonlinearProblem(gapsystem!,chutealto, [mu, T])
    solalto = solve(probalto, TrustRegion(; autodiff=ad), abstol=1e-8)
    probbaixo = NonlinearProblem(gapsystem!,chutebaixo, [mu, T])
    solbaixo = solve(probbaixo, TrustRegion(; autodiff=ad), abstol=1e-8)

    if abs(solalto.u[3] - solbaixo.u[3]) < 1e-4
        return [solalto.u[1], solalto.u[2], solalto.u[3]]
    elseif potentiallog(solbaixo.u[1], solbaixo.u[2], mu, T, solbaixo.u[3]) < potentiallog(solalto.u[1], solalto.u[2], mu, T, solalto.u[3])
        return [solbaixo.u[1], solbaixo.u[2], solbaixo.u[3]]
    end
    return [solalto.u[1], solalto.u[2], solalto.u[3]]
end

function solvermurange(mu, T)
    sols = zeros(length(mu), 3)
    Threads.@threads for i in eachindex(mu)
        sols[i, :] = gapsolver(mu[i], T)
    end
    return sols
end

function solverTrange(mu, T)
    sols = zeros(length(T), 3)
    Threads.@threads for i in eachindex(T)
        sols[i, :] = gapsolver(mu, T[i])
    end
    return sols
end

function Tmusolver(mur, Tr)
    sols = zeros(length(mur), length(Tr), 3)
    Threads.@threads for i in eachindex(Tr)
        sols[:, i, :] = solvermurange(mur, Tr[i])
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
    plot(mur, [allsols[:, 3, 13], allsols[:, 3, 14], allsols[:, 3, 15]], xlabel = "μ [GeV]", ylabel = "M [GeV]")
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

let 
    mur = range(0,0.6,length = 1000)
    muquark, Tquark = quarkyonicall(mur, Tr, allsols)
    scatter(muquark, Tquark, xlabel = "μ [GeV]", ylabel = "T [GeV]")
end

function maxfind(x,y)
    for i in 100:length(y)-1
        if y[i+1] < y[i] && y[i-1] < y[i]  
            return x[i], y[i]
        end
    end
    return NaN, NaN # Return NaN if no maximum is found
end

begin
    rightsols = zeros(length(mur), length(Tr), 3)
    for i in eachindex(Tr)
        for j in eachindex(mur)
            rightsols[j,i,1] = allsols[j,1,i]
            rightsols[j,i,2] = allsols[j,2,i]
            rightsols[j,i,3] = allsols[j,3,i]
        end
    end
end



npzwrite("SOLUTIONS.npz", rightsols)
sols = npzread("SOLUTIONS.npz")

let
    mur = range(0,0.6,length = 1000)
    plot(mur, sols[:,15,3], xlabel = "μ [GeV]", ylabel = "M [GeV]")  
end


#=
Esse right sols é uma matriz 1000x30x3.
rightsols[:, i, 1] são as soluções de ϕ para o i-ésimo T
rightsols[:, i, 2] são as soluções de ϕ̄* para o i-ésimo T
rightsols[:, i, 3] são as soluções de M para o i-ésimo T

Só para não me merder. μ não está guardado em nenhum lugar das soluções.
=#
function interp(phi, phib, M, mu)
    itpM = interpolate((mu,), M, Gridded(Linear()))
    itpphi = interpolate((mu,), phi, Gridded(Linear()))
    itpphib = interpolate((mu,), phib, Gridded(Linear()))
    interp = zeros(length(mu), 3)
    derinterp = zeros(length(mu), 3)
    for i in eachindex(mu)
        interp[i, 1] = itpphi(mu[i])
        interp[i, 2] = itpphib(mu[i])
        interp[i, 3] = itpM(mu[i])
        derinterp[i, 1] = only(Interpolations.gradient(itpphi, mu[i]))
        derinterp[i, 2] = only(Interpolations.gradient(itpphib, mu[i]))
        derinterp[i, 3] = -only(Interpolations.gradient(itpM, mu[i]))
    end
    return derinterp
end

function interploop(sols, Tr)
    phi_int = zeros(length(sols[:,1,1]), length(Tr))
    phib_int = zeros(length(sols[:,1,1]), length(Tr))
    M_int = zeros(length(sols[:,1,1]), length(Tr))
    for i in eachindex(Tr)
        phi, phib, M, mu = sols[:,1,i], sols[:,2,i], sols[:,3,i], mu[i]
        interploop = interp(phi, phib, M, T)
        phi_int[:,i] = interploop[:,1]
        phib_int[:,i] = interploop[:,2]
        M_int[:,i] = interploop[:,3]
    end
    return phi_int, phib_int, M_int
end

begin
    Trange = Tr
    phi_int, phib_int, M_int = interploop(allsols, Trange)
end