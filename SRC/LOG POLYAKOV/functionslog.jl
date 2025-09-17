begin
    include("parameterslog.jl")
end

aT(T) = a0 + a1*(T0/T) + a2*(T0/T)^2
bT(T) = b3 * (T0/T)^3

Ep(p,M) = sqrt(p^2 + M^2)

Gc(phi, phib) = G*(1 - alpha1*(phi*phib) - alpha2*(phi^3 + phib^3))

zminus(phi,phib,M,mu,T,p) = 1 + 3*phi*exp(-(Ep(p,M) - mu)/T) + 3*phib*exp(-2*(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T) + 1e-9

zplus(phi,phib,M,mu,T,p) = 1 + 3*phib*exp(-(Ep(p,M) + mu)/T) + 3*phi*exp(-2*(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T) + 1e-9

function Imed(phi,phib,mu,T,M)
    quadgk(p -> p^2 * (log(zminus(phi,phib,M,mu,T,p)) + log(zplus(phi,phib,M,mu,T,p))), 0, Inf)[1]
end

function Ivac(M)
    quadgk(p -> p^2 * Ep(p,M), 0, L)[1]
end

function potentiallog(phi,phib,mu,T,M)
    (M-m)^2/(4*G) - T*Nf*Imed(phi,phib,mu,T,M)/π^2 - 3*Nf*Ivac(M)/π^2 + U(phi, phib, T)
end

function dMlog(phi,phib,mu,T,M)
    ForwardDiff.derivative(Mi -> potentiallog(phi, phib, mu, T, Mi), M)
end

function dphilog(phi,phib,mu,T,M)
    ForwardDiff.derivative(phi_i -> potentiallog(phi_i, phib, mu, T, M), phi)
end

function dphiblog(phi,phib,mu,T,M)
    ForwardDiff.derivative(phib_i -> potentiallog(phi, phib_i, mu, T, M), phib)
end

function U(phi, phib, T)
    term1 = -0.5 * aT(T) * phi * phib
    term2 = bT(T) * log(1 - 6*phib*phi + 4*(phib^3 + phi^3) - 3*(phib^2 * phi^2))
    return T^4 * (term1 + term2)
end

function densityeqlog(phi, phib, mu, T, M, nb)
    a = ForwardDiff.derivative(mui -> potentiallog(phi, phib, mui, T, M), mu)
    return a + nb
end

function dpotmu(phi, phib, mu, T, M)
    ForwardDiff.derivative(mui -> potentiallog(phi, phib, mui, T, M), mu)
end

function dpot2M(phi, phib, mu, T, M)
    ForwardDiff.derivative(Mi -> dMlog(phi, phib, mu, T, Mi), M)
end

function dpot3M(phi, phib, mu, T, M)
    ForwardDiff.derivative(Mi -> dpot2M(phi, phib, mu, T, Mi), M)
end

function eq1log(phi, phib, mu, T, M)
    a = dpot2M(phi, phib, mu, T, M)
    b = dpotmu(phi, phib, mu, T, M)
    return a/b
end

function eq2log(phi, phib, mu, T, M)
    a = dpot3M(phi, phib, mu, T, M)
    b = dpotmu(phi, phib, mu, T, M)
    return a/b
end