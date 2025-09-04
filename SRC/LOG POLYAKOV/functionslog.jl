begin
    include("parameterslog.jl")
end

aT(T) = a0 + a1*(T0/T) + a2*(T0/T)^2
bT(T) = b3 * (T0/T)^3

Ep(p,M) = sqrt(p^2 + M^2)

Gc(phi, phib) = G*(1 - alpha1*(phi*phib) - alpha2*(phi^3 + phib^3))

zminus(phi,phib,M,mu,T,p) = 1 + 3*phi*exp(-(Ep(p,M) - mu)/T) + 3*phib*exp(-2*(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T)

zplus(phi,phib,M,mu,T,p) = 1 + 3*phib*exp(-(Ep(p,M) + mu)/T) + 3*phi*exp(-2*(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T)

function Imed(phi,phib,mu,T,M)
    quadgk(p -> p^2 * log(zminus(phi,phib,M,mu,T,p)*zplus(phi,phib,M,mu,T,p)), 0, Inf)[1]
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

function dUphi(phi, phib, T)
    a = -0.5*aT(T)*phib
    dg = -6*phib + 12*phi^2 - 6*phib^2*phi
    g  = 1 - 6*phib*phi + 4*(phib^3 + phi^3) - 3*(phib^2 * phi^2)
    return T^4 * (a + bT(T)*dg/g)
end

function dUphib(phi, phib, T)
    a = -0.5*aT(T)*phi
    dg = -6*phi + 12*phib^2 - 6*phi^2*phib
    g  = 1 - 6*phib*phi + 4*(phib^3 + phi^3) - 3*(phib^2 * phi^2)
    return T^4 * (a + bT(T)*dg/g)
end

function epsminus(mu, T, M, p)
    exp(-(Ep(p,M) - mu)/T)
end

function epsplus(mu, T, M, p)
    exp(-(Ep(p,M) + mu)/T)
end

function dzminusM(phi, phib, M, mu, T, p)
    eps = epsminus(mu, T, M, p)
    denom = 1 + 3*phi*eps + 3*phib*eps^2 + eps^3
    num = 3*phi*eps + 6*phib*eps^2 + 3*eps^3
    return -(M/(T*Ep(p,M))) * (num/denom)
end

function dzplusM(phi, phib, M, mu, T, p)
    eps = epsplus(mu, T, M, p)
    denom = 1 + 3*phib*eps + 3*phi*eps^2 + eps^3
    num = 3*phib*eps + 6*phi*eps^2 + 3*eps^3
    return -(M/(T*Ep(p,M))) * (num/denom)
end

function dzminusphi(phi, phib, M, mu, T, p)
    eps = epsminus(mu, T, M, p)
    denom = 1 + 3*phi*eps + 3*phib*eps^2 + eps^3
    return (3*eps)/denom
end

function dzplusphi(phi, phib, M, mu, T, p)
    eps = epsplus(mu, T, M, p)
    denom = 1 + 3*phib*eps + 3*phi*eps^2 + eps^3
    return (3*eps^2)/denom
end

function dzminusphib(phi, phib, M, mu, T, p)
    eps = epsminus(mu, T, M, p)
    denom = 1 + 3*phi*eps + 3*phib*eps^2 + eps^3
    return (3*eps^2)/denom
end

function dzplusphib(phi, phib, M, mu, T, p)
    eps = epsplus(mu, T, M, p)
    denom = 1 + 3*phib*eps + 3*phi*eps^2 + eps^3
    return (3*eps)/denom
end

function intvacM(M)
    return quadgk(p -> p^2 * M/Ep(p,M), 0, L)[1]
end

function intmedM(phi, phib, mu, T, M)
    return quadgk(p -> p^2 * (dzplusM(phi, phib, M, mu, T, p) + dzminusM(phi, phib, M, mu, T, p)), 0, Inf)[1]
end

function intmedphi(phi, phib, mu, T, M)
    return quadgk(p -> p^2 * (dzplusphi(phi, phib, M, mu, T, p) + dzminusphi(phi, phib, M, mu, T, p)), 0, Inf)[1]
end

function intmedphib(phi, phib, mu, T, M)
    return quadgk(p -> p^2 * (dzplusphib(phi, phib, M, mu, T, p) + dzminusphib(phi, phib, M, mu, T, p)), 0, Inf)[1]
end

function dpotM(phi, phib, mu, T, M)
    intvac = intvacM(M)
    intmed = intmedM(phi, phib, mu, T, M)
    a = (M-m)/2*G
    return a - T*Nf*intmed/π^2 - 3*Nf*intvac/π^2
end

function dpotphi(phi, phib, mu, T, M)
    intmed = intmedphi(phi, phib, mu, T, M)
    return -T*Nf*intmed/π^2 + dUphi(phi, phib, T)
end

function dpotphib(phi, phib, mu, T, M)
    intmed = intmedphib(phi, phib, mu, T, M)
    return -T*Nf*intmed/π^2 + dUphib(phi, phib, T)
end

function density(phi, phib, mu, T, M, nb)
    a = ForwardDiff.derivative(mui -> potentiallog(phi, phib, mui, T, M), mu)
    return a + nb
end
