begin
    include("parametersmu.jl") 
    include("functionsmu.jl")
    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations, NonlinearSolve, NPZ, Optim
end

function maxfind(y,x)
    for i in 20:length(y)-1
        if y[i] > y[i-1] && y[i] > y[i+1]
            return x[i], y[i]
        end
    end
    return NaN, NaN
end

function gapsystem!(du, u, p)
    u[1] = clamp(u[1], 0.0, 1.0)
    u[2] = clamp(u[2], 0.0, 1.0)
    u[3] = max(u[3], 0.0)

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

function gapsystem2!(du, u, p)
    u[1] = clamp(u[1], 0.0, 1.0)
    u[2] = clamp(u[2], 0.0, 1.0)
    u[3] = max(u[3], 0.0)

    du[1] = DPHIPOT(u[1], u[2], p[1], p[2], u[3])
    du[2] = DPHIBPOT(u[1], u[2], p[1], p[2], u[3])
    du[3] = DMPOT(u[1], u[2], p[1], p[2], u[3])
end

function gapsolveranalytical(mu, T)
    chutealto = [0.01,0.01,0.34]
    chutebaixo = [0.9,0.9,0.01]
    ad = AutoFiniteDiff()

    probalto = NonlinearProblem(gapsystem2!,chutealto, [mu, T])
    solalto = solve(probalto, NewtonRaphson(; autodiff=ad), abstol=1e-8, maxiters = 100)
    probbaixo = NonlinearProblem(gapsystem2!,chutebaixo, [mu, T])
    solbaixo = solve(probbaixo, NewtonRaphson(; autodiff=ad), abstol=1e-8, maxiters = 100)

    if abs(solalto.u[3] - solbaixo.u[3]) < 1e-4
        return [solalto.u[1], solalto.u[2], solalto.u[3]]
    elseif potentialmu(solbaixo.u[1], solbaixo.u[2], mu, T, solbaixo.u[3]) < potentialmu(solalto.u[1], solalto.u[2], mu, T, solalto.u[3])
        return [solbaixo.u[1], solbaixo.u[2], solbaixo.u[3]]
    end
    return [solalto.u[1], solalto.u[2], solalto.u[3]]
end

let
    gapsolver(0.3, 0.1)
end

function murangesolver(mu, T)
    sols = zeros(length(mu), 3)
    Threads.@threads for i in eachindex(mu)
        sols[i, :] = gapsolver(mu[i], T)   # T is scalar here
    end
    return sols
end

function gap_solverforT(mu, T)
    sols = zeros(length(T), 3)
    Threads.@threads for i in eachindex(T)
        sols[i, :] = gapsolver(mu, T[i])   # mu is scalar here
    end
    return sols
end

function gap_analyticalforT(mu, T)
    sols = zeros(length(T), 3)
    for i in eachindex(T)
        sols[i, :] = gapsolveranalytical(mu, T[i])   # mu is scalar here
    end
    return sols
end

function Trangesolver(mu, T)
    sols = zeros(length(T), length(mu), 3)
    Threads.@threads for i in eachindex(T)
        sols[i, :, :] = murangesolver(mu, T[i])   # pass scalar T[i]
    end
    return sols
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

function Trange_density(T, nb)
    sols = zeros(length(T), length(nb), 4)
    potvals = zeros(length(T), length(nb))
    Threads.@threads for i in eachindex(T)
        sols[i, :, :], potvals[i, :] = densityrangesolveru(T[i], nb)
    end
    return sols, potvals
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

begin
    T = range(0.005, 0.1502, length=50)
    sols = zeros(50, 2)
    potsols = CSV.read("potvals.csv", DataFrame)
    musols = CSV.read("mudensity.csv", DataFrame)
    for i in 1:50
        firstcurvex, firstcurvey, secondcurvex, secondcurvey = interpot(collect(potsols[i, :]), collect(musols[i, :]))

        x1 = Vector{Float64}(firstcurvex)
        y1 = Vector{Float64}(firstcurvey)
        x2 = reverse(Vector{Float64}(secondcurvex))
        y2 = reverse(Vector{Float64}(secondcurvey))

        interp1 = DataInterpolations.LinearInterpolation(y1, x1; extrapolation=ExtrapolationType.Linear)
        interp2 = DataInterpolations.QuadraticInterpolation(y2, x2; extrapolation=ExtrapolationType.Linear)

        diferenca(mu) = interp1(mu) - interp2(mu)
        mucritico = nlsolve(x -> [diferenca(x[1])], [0.2], method=:newton)

        sols[i, :] = [T[i], mucritico.zero[1]]
    end
    plot(sols[:,2], sols[:,1])
    scatter!([CEP.zero[3]], [CEP.zero[4]])
end

# begin
#     murng = range(0.0, 0.5, length = 100)
#     Trng = range(0.15, 0.3, length = 70)
#     sols = Trangesolver(murng, Trng)
#     dfphi = DataFrame(sols[:,:,1], :auto)
#     dfphiB = DataFrame(sols[:,:,2], :auto)
#     dfM = DataFrame(sols[:,:,3], :auto)
#     CSV.write("phisolutions.csv", dfphi, writeheader=false)
#     CSV.write("phibsolutions.csv", dfphiB, writeheader=false)
#     CSV.write("Msolutions.csv", dfM, writeheader=false)
# end

begin
    M_solutions = Matrix(CSV.read("Msolutions.csv", DataFrame; header=false))
    phi_solutions = Matrix(CSV.read("phisolutions.csv", DataFrame; header=false))
    phib_solutions = Matrix(CSV.read("phibsolutions.csv", DataFrame; header=false))
end

begin
    CEP = nlsolve(x -> [dMplkmu(x[1],x[2],x[3],x[4],x[5]), dphiplkmu(x[1],x[2],x[3],x[4],x[5]), dphibplkmu(x[1],x[2],x[3],x[4],x[5]), eq1(x[1],x[2],x[3],x[4],x[5]), eq2(x[1],x[2],x[3],x[4],x[5])], [0.15,0.22,0.31,0.15,0.1], autodiff=:forward)
    println("CEP: ", CEP.zero)
end

# begin 
#     nbr = range(1e-10,0.07,length=200)
#     Tr = range(0.01,0.1502,length=50)
#     sols, potvals = Trange_density(Tr, nbr)
#     phidensity = DataFrame(sols[:,:,1], :auto)
#     phibdensity = DataFrame(sols[:,:,2], :auto)
#     mudensity = DataFrame(sols[:,:,3], :auto)
#     Mdensity = DataFrame(sols[:,:,4], :auto)
#     potvalsdf = DataFrame(potvals[:,:], :auto)
#     CSV.write("phidensity.csv", phidensity)
#     CSV.write("phibdensity.csv", phibdensity)
#     CSV.write("mudensity.csv", mudensity)
#     CSV.write("Mdensity.csv", Mdensity)
#     CSV.write("potvals.csv", potvalsdf)
# end

function Interpot(Mvals, phivals, phibvals, muvals)
    itpM = interpolate((muvals,), Mvals, Gridded(Linear()))
    itpPhi = interpolate((muvals,), phivals, Gridded(Linear()))
    itpPhiB = interpolate((muvals,), phibvals, Gridded(Linear()))
    dinterp = zeros(length(muvals), 3)
    for i in eachindex(muvals)
        dinterp[i,1] = only(Interpolations.gradient(itpM, muvals[i]))
        dinterp[i,2] = -only(Interpolations.gradient(itpPhi, muvals[i]))
        dinterp[i,3] = -only(Interpolations.gradient(itpPhiB, muvals[i])) 
    end
    return dinterp
end

begin
    copoints = zeros(length(Trng), 4)
    for i in eachindex(Trng)
        dinterp = Interpot(M_solutions[i, :], phi_solutions[i, :], phib_solutions[i, :], murng)
        
        x1 = Vector{Float64}(murng)
        y1 = Vector{Float64}(dinterp[:,1])
        y2 = Vector{Float64}(dinterp[:,2])
        y3 = Vector{Float64}(dinterp[:,3])

        interpM = DataInterpolations.CubicSpline(y1, x1; extrapolation=ExtrapolationType.Linear)
        interpPhi = DataInterpolations.CubicSpline(y2, x1; extrapolation=ExtrapolationType.Linear)
        interpPhiB = DataInterpolations.CubicSpline(y3, x1; extrapolation=ExtrapolationType.Linear)
    end
end



begin
    ditp = Interpot(M_solutions[30, :], phi_solutions[30, :], phib_solutions[30, :], murng)

    x1 = Vector{Float64}(murng)
    y1 = Vector{Float64}(ditp[:,1])
    y2 = Vector{Float64}(ditp[:,2])
    y3 = Vector{Float64}(ditp[:,3])

    interpM = DataInterpolations.QuadraticInterpolation(y1, x1; extrapolation=ExtrapolationType.Linear)
    interpPhi = DataInterpolations.QuadraticInterpolation(y2, x1; extrapolation=ExtrapolationType.Linear)
    interpPhiB = DataInterpolations.QuadraticInterpolation(y3, x1; extrapolation=ExtrapolationType.Linear)


end

begin
    plot(Trng, copoints[:,2], label="M", xlabel="T (GeV)", ylabel="Critical μ (GeV)", legend=:topright)
end

begin
    murng = range(0.0, 0.5, length = 100)
    Trng = range(0.15, 0.3, length = 70)
    copoints = zeros(length(Trng), 4)
    for i in eachindex(Trng)
        dinterp = Interpot(M_solutions[i, :], phi_solutions[i, :], phib_solutions[i, :], murng)
        maxdM = maxfind(dinterp[:,1], murng)[1]
        maxdPhi = maxfind(dinterp[:,2], murng)[1]
        maxdPhiB= maxfind(dinterp[:,3], murng)[1]
        copoints[i, :] = [Trng[i], maxdM[1], maxdPhi[1], maxdPhiB[1]] 
    end
    plot(Trng, copoints[:,2], label="M", xlabel="μ (GeV)", ylabel="T (GeV)", legend=:topright)
    plot!(Trng, copoints[:,3], label="Φ", xlabel="μ (GeV)", ylabel="T (GeV)", legend=:topright)
    plot!(Trng, copoints[:,4], label="ΦB", xlabel="μ (GeV)", ylabel="T (GeV)", legend=:topright)
end