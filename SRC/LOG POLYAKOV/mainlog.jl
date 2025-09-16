begin
    include("parameterslog.jl")
    include("functionslog.jl")

    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations, NonlinearSolve, NPZ
end

function gapsystem!(du, u, p)
    #=Esse "clamp" restringe as soluções. Definitivamente não é o ideal quando o modelo é desconhecido, mas
    sem isso o solver fica muito instável e não converge corretamente
    =#

    u[1] = clamp(u[1], 0.0, 1.0)
    u[2] = clamp(u[2], 0.0, 1.0)
    u[3] = max(u[3], 0.0)

    println("phi = ", u[1], " phib = ", u[2], " M = ", u[3], " mu = ", p[1], " T = ", p[2])

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

begin
    gapsolver(0.1,0.2)
end

function cepsystem!(du, u, p=0)
    u[1] = clamp(u[1], 0.0, 1.0)
    u[2] = clamp(u[2], 0.0, 1.0)
    u[3] = clamp(u[3], 0.0, 1.0)
    u[4] = clamp(u[4], 0.0, 1.0)
    u[5] = clamp(u[5], 0.0, 1.0)


    du[1] = dphilog(u[1],u[2],u[3],u[4],u[5])
    du[2] = dphiblog(u[1],u[2],u[3],u[4],u[5])
    du[3] = dMlog(u[1],u[2],u[3],u[4],u[5])
    du[4] = eq1(u[1],u[2],u[3],u[4],u[5])
    du[5] = eq2(u[1],u[2],u[3],u[4],u[5])
end

begin
    CEP = nlsolve(cepsystem!, [0.3,0.1,0.4,0.08,0.25]).zero
    println("phi = ", CEP[1], " phib = ", CEP[2], " M = ", CEP[3], " mu = ", CEP[4], " T = ", CEP[5])
end

begin
    chute = [0.1,0.1,0.4,0.1,0.4]

    ad = AutoFiniteDiff()
    prob = NonlinearProblem(cepsystem!,chute)
    sol = solve(prob, NewtonRaphson(; autodiff=ad), abstol=1e-10, maxiters = 1000)
    println("phi = ", sol.u[1], " phib = ", sol.u[2], " M = ", sol.u[3], " mu = ", sol.u[4], " T = ", sol.u[5])
end


@time begin
    mur = range(0,0.8,length = 100)
    Tr = range(0.01, 0.30, length = 30)
    allsols = Tmusolver(mur, Tr)
end

begin
    scatter(mur, [allsols[:, 10, 1], allsols[:, 10, 2], allsols[:, 10, 3]], xlabel = "μ [GeV]", ylabel = "M [GeV]")
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
            if sols[j, i, 3] < mur[j]
                mu_quark[i] = mur[j]
                T_quark[i] = Tr[i]
                break
            end
        end
    end
    return mu_quark, T_quark
end

let 
    mur = range(0,0.6,length = 100)
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

#=
Esse right sols é uma matriz 100x30x3.
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

function densitysolver(T, nb, chute)
    sist = nlsolve(x -> [density(x[1], x[2], x[3], T, x[4], nb), dphilog(x[1], x[2], x[3], T, x[4]), dphiblog(x[1], x[2], x[3], T, x[4]), dMlog(x[1], x[2], x[3], T, x[4])], chute)
    return sist.zero
end

function densityTrange(T, nbrange)
    sols = zeros(length(nbrange), 4)  # phi, phib, mu, M
    potvals = zeros(length(nbrange))
    chute = [0.1, 0.1, 0.4, 4.0]
    
    for i in eachindex(nbrange)
        nb = nbrange[i]
        sol = densitysolver(T, nb, chute)
        sols[i, :] = sol
        potvals[i] = potentiallog(sol[1], sol[2], sol[3], T, sol[4])
        chute = sol
    end
    return sols, potvals
end


begin
    solsi, potvalsi = densityTrange(0.02, range(0.000001,0.02,150))
    plot(solsi[:,3], potvalsi)
end

#=
preciso: interpolar soluções allsols
ϕ = allsols[:, i, 1]
ϕ̄* = allsols[:, i, 2]
M = allsols[:, i, 3]
onde i é o i-ésimo valor de T

derivar as interpolações e  d_interp = 0  para cada T e obter o ponto de transição (!!!)

depois construir a transição de primeira ordem do mesmo jeito pq o sistema pra densidade tá funcionando

construir o diagrama com os pontos que consegui da quarkyonica.
=#

