begin
    include("parameterslog.jl")
end


aT(T) = a0 + a1*(T0/T) + a2*(T0/T)^2
bT(T) = b3 * (T0/T)^3

Ep(p,M) = sqrt(p^2 + M^2)

zminus(phi,phib,M,mu,T,p) = log(1 + 3*(phi + phib*exp(-(Ep(p,M) - mu)/T))*exp(-(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T))

zplus(phi,phib,M,mu,T,p) = log(1 + 3*(phib + phi*exp(-(Ep(p,M) + mu)/T))*exp(-(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T))

function U(phi, phib, T)
    term1 = -0.5 * aT(T) * phi * phib
    term2 = bT(T) * log(1 - 6*phib*phi + 4*(phib^3 + phi^3) - 3*(phib*phi)^2)
    return T^4 * (term1 + term2)
end


function Imed(phi,phib,M,mu,T)
    quadgk(p -> p^2 * (zminus(phi,phib,M,mu,T,p) + zplus(phi,phib,M,mu,T,p)), 0, Inf)[1]
end

function Ivac(M)
    quadgk(p -> p^2 * Ep(p,M), 0, L)[1]
end

function potentiallog(phi,phib,mu,T,M)
    (M-m)^2/4G - T*Nf*Imed(phi,phib,M,mu,T)/π^2 - 3*Nf*Ivac(M)/π^2 + U(phi, phib, T)
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
