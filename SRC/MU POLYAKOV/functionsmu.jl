begin
    include("parametersmu.jl") 
end

Ep(p,M) = sqrt(p^2 + M^2)

zminus(phi,phib,M,mu,T,p) = log(1 + 3*(phi + phib*exp(-(Ep(p,M) - mu)/T))*exp(-(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T))

zplus(phi,phib,M,mu,T,p) = log(1 + 3*(phib + phi*exp(-(Ep(p,M) + mu)/T))*exp(-(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T))

potplkvmu(phi, phib, mu, T) = (a0*T^4 + a1*mu^4 + a2*mu^2*T^2)*(phi*phib) + a3*T0^4*log(1 - 6*phi*phib + 4*(phi^3 + phib^3) - 3*(phi*phib)^2)

Gmod(phi, phib) = G0#*(1-phi*phib)

function Imed(phi,phib,M,mu,T)
    quadgk(p -> p^2 * (zminus(phi,phib,M,mu,T,p) + zplus(phi,phib,M,mu,T,p)), 0, Inf)[1]
end

function Ivac(M)
    quadgk(p -> p^2 * Ep(p,M), 0, lamb)[1]
end
    
function potentialmu(phi, phib, mu, T, M)
    ((M-m)^2/(4*Gmod(phi, phib))) - T*Nf*Imed(phi,phib,M,mu,T)/π^2 - 3*Nf*Ivac(M)/π^2 + potplkvmu(phi,phib,mu,T)
end

function dphiplkmu(phi,phib,mu,T,M)
    ForwardDiff.derivative(phix -> potentialmu(phix,phib,mu,T,M), phi)
end

function dphibplkmu(phi,phib,mu,T,M)
    ForwardDiff.derivative(phibx -> potentialmu(phi,phibx,mu,T,M), phib)
end

function dMplkmu(phi,phib,mu,T,M)
    ForwardDiff.derivative(Mi -> potentialmu(phi, phib, mu, T, Mi), M)
end

function densityeq(phi, phib, mu, T, M, nb)
    a = ForwardDiff.derivative(mux -> potentialmu(phi, phib, mux, T, M), mu)
    return a + nb
end

function dmuplkmu(phi,phib,mu,T,M)
    ForwardDiff.derivative(mux -> potentialmu(phi, phib, mux, T, M), mu)
end

function dM2plkmu(phi,phib,mu,T,M)
    ForwardDiff.derivative(Mi -> dMplkmu(phi, phib, mu, T, Mi), M)
end

function dM3plkmu(phi,phib,mu,T,M)
    ForwardDiff.derivative(Mi -> dM2plkmu(phi, phib, mu, T, Mi), M)
end

function eq1(phi, phib, mu, T, M)
    a = dM2plkmu(phi, phib, mu, T, M)
    b = dmuplkmu(phi, phib, mu, T, M)
    return a/b
end

function eq2(phi, phib, mu, T, M)
    a = dM3plkmu(phi, phib, mu, T, M)
    b = dmuplkmu(phi, phib, mu, T, M)
    return a/b
end

################## DEFINING ANALYTICAL FUNCTIONS
function Fminus(phi, phib, mu, T, M, p)
    a = phi*exp(2*(Ep(p,M)-mu)/T) + 2*phib*exp((Ep(p,M)-mu)/T) + 1
    b = 3*phi*exp(2*(Ep(p,M)-mu)/T) + 3*phib*exp((Ep(p,M)-mu)/T) + exp(3*(Ep(p,M)-mu)/T) + 1
    return a/b
end

function Fplus(phi, phib, mu, T, M, p)
    a = phib*exp(2*(Ep(p,M)+mu)/T) + 2*phi*exp((Ep(p,M)+mu)/T) + 1
    b = 3*phib*exp(2*(Ep(p,M)+mu)/T) + 3*phi*exp((Ep(p,M)+mu)/T) + exp(3*(Ep(p,M)+mu)/T) + 1
    return a/b
end

function Ivaccondensate(M)
    quadgk(p -> p^2 * (M/Ep(p,M)), 0, lamb)[1]
end

function Imedcondensate(phi, phib, M, mu, T)
    quadgk(p -> p^2 * (M/Ep(p,M)) * (Fminus(phi, phib, mu, T, M, p) + Fplus(phi, phib, mu, T, M, p)), 0, Inf)[1]
end

function condensate(phi, phib, mu, T, M)
    γ/(2π^2) * (Imedcondensate(phi, phib, M, mu, T) - Ivaccondensate(M))
end 

function dU_dphi(phi, phib, mu, T)
    a = 6*phib - 12*phi^2 + 6*phi*phib^2
    b = 1 - 6*phi*phib + 4*(phi^3 + phib^3) - 3*(phi*phib)^2
    return (a0*T^4 + a1*mu^4 + a2*mu^2*T^2)*phib + a3*T0^4*(a/b)
end

function dzplus_dphi(phi, phib, mu, T, M, p)
    a = 3*exp(-2(Ep(p,M) + mu)/T)
    b = 1 + 3*(phib + phi*exp(-(Ep(p,M) + mu)/T))*exp(-(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T)
    return a/b
end

function dzminus_dphi(phi, phib, mu, T, M, p)
    a = 3*exp(-(Ep(p,M) - mu)/T)
    b = 1 + 3*(phi + phib*exp(-(Ep(p,M) - mu)/T))*exp(-(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T)
    return a/b
end

function dU_dphib(phi, phib, mu, T)
    a = 6*phi - 12*phib^2 + 6*phib*phi^2
    b = 1 - 6*phi*phib + 4*(phi^3 + phib^3) - 3*(phi*phib)^2
    return (a0*T^4 + a1*mu^4 + a2*mu^2*T^2)*phi + a3*T0^4*(a/b)
end

function dzplus_dphib(phi, phib, mu, T, M, p)
    a = 3*exp(-(Ep(p,M) + mu)/T)
    b = 1 + 3*(phib + phi*exp(-(Ep(p,M) + mu)/T))*exp(-(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T)
    return a/b
end

function dzminus_dphib(phi, phib, mu, T, M, p)
    a = 3*exp(-2*(Ep(p,M) - mu)/T)
    b = 1 + 3*(phi + phib*exp(-(Ep(p,M) - mu)/T))*exp(-(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T)
    return a/b
end

function Imed_dphi(phi, phib, M, mu, T)
    quadgk(p -> p^2 * (dzminus_dphi(phi, phib, mu, T, M, p) + dzplus_dphi(phi, phib, mu, T, M, p)), 0, Inf)[1]
end