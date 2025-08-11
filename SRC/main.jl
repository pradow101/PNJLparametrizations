begin
    include("parameters.jl")
    include("functions.jl")

    using QuadGK, Plots, NLsolve, CSV, DataFrames, ForwardDiff, DataInterpolations, LocalFunctionApproximation, Interpolations
end


function gaplogsolver(mu, T, chute)
    sist = nlsolve(x -> (dphilog(x[1], x[2], mu, T, x[3]), dphiblog(x[1], x[2], mu, T, x[3]), dMlog(x[1], x[2], mu, T, x[3])), chute)
    return sist.zero
end

function gapplkvmusolver(mu, T, chute)
    sist = nlsolve(x -> (dphiplkmu(x[1], x[2], mu, T, x[3]), dphibplkmu(x[1], x[2], mu, T, x[3]), dMplkmu(x[1], x[2], mu, T, x[3])), chute)
    return sist.zero
end



begin
    function Trangelogsolver(mu, T_vals)
        philogvals = zeros(length(T_vals))
        phiblogvals = zeros(length(T_vals))
        Mlogvals = zeros(length(T_vals))
        chute = [0.01,0.01,1]
        for i in 1:length(T_vals)
            T = T_vals[i]
            solution = gaplogsolver(mu, T, chute)
            philogvals[i] = solution[1]
            phiblogvals[i] = solution[2]
            Mlogvals[i] = solution[3]
            chute = solution
        end
        return philogvals, phiblogvals, Mlogvals, T_vals
    end

    function Trangeplkvmusolver(mu, T_vals)
        philkmuvals = zeros(length(T_vals))
        phibplkmuvals = zeros(length(T_vals))
        Mplkmuvals = zeros(length(T_vals))
        chute = [0.01,0.01,0.4]
        for i in eachindex(T_vals)
            T = T_vals[i]
            solution = gapplkvmusolver(mu, T, chute)
            philkmuvals[i] = solution[1]
            phibplkmuvals[i] = solution[2]
            Mplkmuvals[i] = solution[3]
            chute = solution
        end
        return philkmuvals, phibplkmuvals, Mplkmuvals, T_vals
    end
end


let 
    Tvalores = range(0.01,0.3,20)
    mu = 0.1
    philog, phiblog, Mlog, Tvals = Trangelogsolver(mu, Tvalores)
    #phiplkmu, phibplkmu, Mplkmu, Tvals = Trangeplkvmusolver(mu, Tvalores)

    plot(Tvals, philog)
    plot!(Tvals, Mlog)
end
