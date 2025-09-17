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
    CEP = nlsolve(x -> [dphilog(x[1],x[2],x[3],x[4],x[5]), dphiblog(x[1],x[2],x[3],x[4],x[5]), dMlog(x[1],x[2],x[3],x[4],x[5]), eq1log(x[1],x[2],x[3],x[4],x[5]), eq2log(x[1],x[2],x[3],x[4],x[5])], [0.15,0.22,0.31,0.15,0.1]).zero
    println("CEP: ", CEP)
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


function densitysystem!(du,u,p)
    du[1] = density(u[1],u[2],u[3],p[1],u[4],p[2])
    du[2] = dphilog(u[1],u[2],u[3],p[1],u[4])
    du[3] = dphiblog(u[1],u[2],u[3],p[1],u[4])
    du[4] = dMlog(u[1],u[2],u[3],p[1],u[4])
end

function densitysolver1(T, nb)
    chute = [0.1,0.1,0.4,0.4]
    ad = AutoFiniteDiff()

    prob = NonlinearProblem(densitysystem!,chute, [T, nb])
    sol = solve(prob, TrustRegion(; autodiff=ad), abstol=1e-8, maxiters = 100)
    return sol.u
end



function densitysolver(T, nb, chute)
    sist = nlsolve(x -> [density(x[1], x[2], x[3], T, x[4], nb), dphilog(x[1], x[2], x[3], T, x[4]), dphiblog(x[1], x[2], x[3], T, x[4]), dMlog(x[1], x[2], x[3], T, x[4])], chute)
    return sist.zero
end

function densityTrange(T, nbrange)
    sols = zeros(length(nbrange), 4)  # phi, phib, mu, M
    potvals = zeros(length(nbrange))

    for i in eachindex(nbrange)
        nb = nbrange[i]
        sol = densitysolver1(T, nb)
        sols[i, :] = Float64.(sol)   # ensure numbers, not Duals/Any
        potvals[i] = potentiallog(sol[1], sol[2], sol[3], T, sol[4])
    end
    return sols, potvals
end



begin
    solsi, potvalsi = densityTrange(0.009, range(0.00001,0.01,150))
    println(solsi[:,3], potvalsi)
end

println(size(potvalsi))

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

function interpot(pvals, muvals)
    firstcurvex = []
    firstcurvey = []
    secondcurvex = []
    secondcurvey = []
    for i in 2:length(muvals)
        if muvals[i] < muvals[i-1]
            break
        end
        append!(firstcurvey, pvals[i])
        append!(firstcurvex, muvals[i])
    end
    for i in length(muvals)-1:-1:2
        if muvals[i] < muvals[i-1]
            break
        end
        append!(secondcurvey, pvals[i])
        append!(secondcurvex, muvals[i])
    end
    return firstcurvex, firstcurvey, secondcurvex, secondcurvey
end

function fofinder(T, chuteinit)
    sols, potvals = densityTrange(T, range(0.000001,0.02,150))
    firstcurvex, firstcurvey, secondcurvex, secondcurvey = interpot(potvals, sols[:,3])

    x1 = Vector{Float64}(firstcurvex)
    y1 = Vector{Float64}(firstcurvey)
    x2 = reverse(Vector{Float64}(secondcurvex))
    y2 = reverse(Vector{Float64}(secondcurvey))
    
    interp1 = DataInterpolations.LinearInterpolation(y1, x1; extrapolation=ExtrapolationType.Linear)
    interp2 = DataInterpolations.QuadraticInterpolation(y2, x2; extrapolation=ExtrapolationType.Linear)

    # return interp1, interp2
    diferenca(mu) = interp1(mu) - interp2(mu)
    
    mucritico = nlsolve(x -> [diferenca(x[1])], [chuteinit], method=:newton)
    return mucritico.zero[1], interp2(mucritico.zero[1]), interp1, interp2, x2, y2, x1, y1
end

# begin
#     spin1x = []
#     spin1y = []
#     spin2x = []
#     spin2y = []
#     Ti = range(0.01, 0.065, length=100)
#     for i in 1:length(Ti)
#         T = Ti[i]
#         _, muvalsspin, _, _, _, pvalsspin = Trange_density(T)
#         x1spin, y1spin, x2spin, y2spin = interpot(pvalsspin, muvalsspin)
#         append!(spin1x, x1spin[end])
#         append!(spin1y, Ti[i])
#         append!(spin2x, x2spin[end])
#         append!(spin2y, Ti[i])
#     end
    
#     plot(spin1x, spin1y)
#     plot!(spin2x, spin2y)
# end

begin
    Tcrits = range(0.02, 0.06, length=30)
    for T in eachindex(Tcrits)
        mucrit, potcrit, _, _, _, _, _, _ = fofinder(Tcrits[T], 0.02)
    end
    plot(Tcrits, mucrit)
    scatter!(muquark, Tquark, xlabel = "μ [GeV]", ylabel = "T [GeV]")
end