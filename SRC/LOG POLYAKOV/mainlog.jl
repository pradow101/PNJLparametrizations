begin
    include("parameterslog.jl")
    include("functionslog.jl")

    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations
end


function gaplogsolver(mu, T, chute)
    sist = nlsolve(x -> (dphilog(x[1], x[2], mu, T, x[3]), dphiblog(x[1], x[2], mu, T, x[3]), dMlog(x[1], x[2], mu, T, x[3])), chute)
    return sist.zero
end

begin
    function Musolverlog(T, muvals)
        μphilogvals = zeros(length(muvals))
        μphiblogvals = zeros(length(muvals))
        μMlogvals = zeros(length(muvals))
        chute = [0.01, 0.01, 0.4]
        Threads.@threads for i in eachindex(muvals)
            mu = muvals[i]
            solution = gaplogsolver(mu, T, chute)
            μphilogvals[i] = solution[1]
            μphiblogvals[i] = solution[2]
            μMlogvals[i] = solution[3]
            chute = solution
        end
        return μphilogvals, μphiblogvals, μMlogvals, muvals
    end
end

begin
    T = 0.1
    murange = range(0, 0.7, length=50)
    phia, phiba, Ma, mua = Musolverlog(T, murange)
end

begin
    plot(mua, Ma)
end

#o potencial tem que ser diferente, não tá fazendo sentido