begin
    include("parameters.jl") 
end

Ep(p,M) = sqrt(p^2 + M^2)

zminus(phi,phib,M,mu,T,p) = log(1 + 3*(phi + phib*exp(-(Ep(p,M) - mu)/T))*exp(-(Ep(p,M) - mu)/T) + exp(-3*(Ep(p,M) - mu)/T))

zplus(phi,phib,M,mu,T,p) = log(1 + 3*(phib + phi*exp(-(Ep(p,M) + mu)/T))*exp(-(Ep(p,M) + mu)/T) + exp(-3*(Ep(p,M) + mu)/T))

function Imed(phi,phib,M,mu,T)
    quadgk(p -> p^2 * (zminus(phi,phib,M,mu,T,p) + zplus(phi,phib,M,mu,T,p)), 0, Inf)[1]
end

function Ivac(M)
    quadgk(p -> p^2 * Ep(p,M), 0, lamb)[1]
end
    
function potentialmu(phi, phib, mu, T, M)
    (M-m)^2/4G - T*Nf*Imed(phi,phib,M,mu,T)/π^2 - 3*Nf*Ivac(M)/π^2 + potplkvmu(phi,phib,mu,T)
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



