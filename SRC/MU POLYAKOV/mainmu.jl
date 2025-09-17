begin
    include("parametersmu.jl") 
    include("functionsmu.jl")
    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations, NonlinearSolve, NPZ

    plotly()
end

function gapsystem!(du, u, p)
    du[1] = dphiplkmu(u[1], u[2], p[1], p[2], u[3])
    du[2] = dphibplkmu(u[1], u[2], p[1], p[2], u[3])
    du[3] = dMplkmu(u[1], u[2], p[1], p[2], u[3])
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
    elseif potentialmu(solbaixo.u[1], solbaixo.u[2], mu, T, solbaixo.u[3]) < potentialmu(solalto.u[1], solalto.u[2], mu, T, solalto.u[3])
        return [solbaixo.u[1], solbaixo.u[2], solbaixo.u[3]]
    end
    return [solalto.u[1], solalto.u[2], solalto.u[3]]
end

function murangesolver(mu, T)
    sols = zeros(length(mu), 3)
    Threads.@threads for i in eachindex(mu)
        sols[i, :] = gapsolver(mu[i], T)
    end
    return sols
end

function Trangesolver(mu, T)
    sols = zeros(length(T), 3)
    Threads.@threads for i in eachindex(T)
        sols[i, :] = murangesolver(mu, T[i])
    end
    return sols
end

let
    mu = range(0.0,0.5, length = 100)
    T = 0.12
    sols = murangesolver(mu, T)
    plot(mu, sols[:,1])
end

function densitysystemu!(du, u, p)
    du[1] = densityeq(u[1], u[2], u[3], p[1], u[4], p[2])
    du[2] = dphiplkmu(u[1], u[2], u[3], p[1], u[4])
    du[3] = dphibplkmu(u[1], u[2], u[3], p[1], u[4])
    du[4] = dMplkmu(u[1], u[2], u[3], p[1], u[4])
end

function densitysolveru(T, nb)
    chute = [0.1,0.1,0.4,0.4]
    ad = AutoFiniteDiff()

    prob = NonlinearProblem(densitysystemu!,chute, [T, nb])
    sol = solve(prob, TrustRegion(; autodiff=ad), abstol=1e-8, maxiters = 100)

    return [sol.u[1], sol.u[2], sol.u[3], sol.u[4]]
end

function densityrangesolveru(T, nb)
    sols = zeros(length(nb), 4)
    potvals = zeros(length(nb))
    for i in eachindex(nb)
        sols[i, :] = densitysolveru(T, nb[i])
        potvals[i] = potentialmu(sols[i, 1], sols[i, 2], sols[i, 3], T, sols[i, 4])
    end
    return sols, potvals
end

let 
    nbr = range(1e-10,0.1,length=500)
    Tr = 0.06
    sols, potvals = densityrangesolveru(Tr, nbr)
    plot([sols[:,3]], potvals, markersize=1)
end

begin
    CEP = nlsolve(x -> [dMplkmu(x[1],x[2],x[3],x[4],x[5]), dphiplkmu(x[1],x[2],x[3],x[4],x[5]), dphibplkmu(x[1],x[2],x[3],x[4],x[5]), eq1(x[1],x[2],x[3],x[4],x[5]), eq2(x[1],x[2],x[3],x[4],x[5])], [0.15,0.22,0.31,0.15,0.1], autodiff=:forward)
    println("CEP: ", CEP.zero)
end

function Trange_density(T, nb)
    sols = zeros(length(T), length(nb), 4)
    potvals = zeros(length(T), length(nb))
    Threads.@threads for i in eachindex(T)
        sols[i, :, :], potvals[i, :] = densityrangesolveru(T[i], nb)
    end
    return sols, potvals
end

begin 
    nbr = range(1e-10,0.07,length=200)
    Tr = range(0.01,0.17,length=50)
    sols, potvals = Trange_density(Tr, nbr)
end

begin
    plot(sols[47,:,3], potvals[47,:])
end

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

function fofinder(T, nb)
    sols = zeros(length(T), 2)

    for i in eachindex(T)
        sols, potvals = densityrangesolveru(T[i], nb)
        firstcurvex, firstcurvey, secondcurvex, secondcurvey = interpot(potvals[i,:], sols[i, :, 3])

        x1 = Vector{Float64}(firstcurvex)
        y1 = Vector{Float64}(firstcurvency)
        x2 = reverse(Vector{Float64}(secondcurvex))
        y2 = reverse(Vector{Float64}(secondcurvey))

        interp1 = DataInterpolations.LinearInterpolation(y1, x1; extrapolation=ExtrapolationType.Linear)
        interp2 = DataInterpolations.QuadraticInterpolation(y2, x2; extrapolation=ExtrapolationType.Linear)

        diferenca(mu) = interp1(mu) - interp2(mu)
        mucritico = nlsolve(x -> [diferenca(x[1])], [0.2], method=:newton)

        sols[i, :] = [T[i], mucritico.zero[1]]
    end
    return sols
end


begin
    Tr = range(0.005,0.15,length=30)
    nbr = range(1e-10,0.07,length=300)
    sols = fofinder(Tr, nbr)
end