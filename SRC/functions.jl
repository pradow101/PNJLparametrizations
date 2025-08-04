begin
    include("parameters.jl") 
end

Ep(p,M) = sqrt(p^2 + M^2)

zminus(phi,phib,M,mu,T,p) = log(1 + 3*(phi + phib*exp(-(Ep(p,M) - mu)/T))*exp(-(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T))

zplus(phi,phib,M,mu,T,p) = log(1 + 3*(phib + phi*exp(-(Ep(p,M) + mu)/T))*exp(-(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T))

a(T) = a0 + a1*(To/T) + a2*(To/T)^2

b(T) = b3*(To/T)^3

potplkvlog(phi, phib, T) = T^4 * (-0.5*a(T)*phib*phi + b(T)*log(complex(1 - 6*phib*phi + 4*(phib^3 + phi^3) - 3*(phib*phi)^2)))

potplkvmu(phi, phib, mu, T) = (A0*(T^4) + A1*(mu^4) + A2*(T^2)*(mu^2))*phib*phi + A3*(To^4)*log(complex(1 - 6*phib*phi + 4*(phib^3 + phi^3) - 3*(phib*phi)^2)))

function Imed(phi,phib,M,mu,T)
    quadgk(p -> p^2 * (zminus(phi,phib,M,mu,T,p) + zplus(phi,phib,M,mu,T,p)), 0, Inf)[1]
end

function Ivac(M)
    quadgk(p -> p^2 * Ep(p,M), 0, lamb)[1]
end

function potentiallog(phi,phib,mu,T,M)
    (M-m)^2/4G - T*Nf*Imed(phi,phib,M,mu,T)/π^2 - 3*Nf*Ivac(M)/π^2 + potplkvlog(phi,phib,T)
end
    
function potentialmu(phi, phib, mu, T, M)
    (M-m)^2/4G - T*Nf*Imed(phi,phib,M,mu,T)/π^2 - 3*Nf*Ivac(M)/π^2 + potplkvmu(phi,phib,mu,T)
end

function dphilog(phi,phib,mu,T,M)
    ForwardDiff.derivative(phix -> potentiallog(phix,phib,mu,T,M), phi)
end

function dphiblog(phi,phib,mu,T,M)
    ForwardDiff.derivative(phibx -> potentiallog(phi,phibx,mu,T,M), phib)
end

function dMlog(phi,phib,mu,T,M)
    ForwardDiff.derivative(Mi -> potentiallog(phi, phib, mu, T, Mi), M)
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



