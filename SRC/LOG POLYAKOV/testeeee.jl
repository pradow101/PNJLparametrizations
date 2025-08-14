# ===================================================================
# SEÇÃO 1: CONFIGURAÇÃO DO AMBIENTE E PARÂMETROS
# ===================================================================
println("Inicializando o ambiente...")

# Pacotes necessários para o script
using QuadGK: quadgk
using NLsolve: nlsolve
using Plots: plot, display

# Inclui o arquivo que define as constantes físicas do modelo
# Certifique-se de que o arquivo "parameterslog.jl" está na mesma pasta.
# Este arquivo deve definir: a0, a1, a2, b3, T0, m, G, Nf, L
include("parameterslog.jl")

# ===================================================================
# SEÇÃO 2: DEFINIÇÃO DAS EQUAÇÕES FÍSICAS
# (Baseado na nossa conversa anterior)
# ===================================================================

# Funções dependentes da temperatura para o potencial U(Φ, Φ̄, T)
a_T(T) = a0 + a1*(T0/T) + a2*(T0/T)^2
b_T(T) = b3 * (T0/T)^3

# Energia do quasipartícula
energia_p(p, M) = sqrt(p^2 + M^2)

# Derivada de U em relação a Φ
function deriv_U_phi(phi, phib, T)
    g = 1 - 6*phib*phi + 4*(phib^3 + phi^3) - 3*(phib^2 * phi^2)
    # Evita divisão por zero se g se tornar muito pequeno ou negativo
    if g <= 1e-9; return -Inf; end
    deriv_g = -6*phib + 12*phi^2 - 6*phib^2*phi
    termo_a = -0.5 * a_T(T) * phib
    termo_b = b_T(T) * deriv_g / g
    return T^4 * (termo_a + termo_b)
end

# Derivada de U em relação a Φ̄
function deriv_U_phib(phi, phib, T)
    g = 1 - 6*phib*phi + 4*(phib^3 + phi^3) - 3*(phib^2 * phi^2)
    if g <= 1e-9; return -Inf; end
    deriv_g = -6*phi + 12*phib^2 - 6*phi^2*phib
    termo_a = -0.5 * a_T(T) * phi
    termo_b = b_T(T) * deriv_g / g
    return T^4 * (termo_a + termo_b)
end

# ----- Integrandos para as integrais térmicas -----

function integrand_M(p, phi, phib, M, mu, T)
    Ep = energia_p(p, M)
    if Ep < 1e-9; return 0.0; end

    eps_p = exp(-(Ep + mu) / T)
    num_p = 3*phib*eps_p + 6*phi*eps_p^2 + 3*eps_p^3
    den_p = 1 + 3*phib*eps_p + 3*phi*eps_p^2 + eps_p^3
    dz_p_dM = -(M / (T * Ep)) * (num_p / den_p)

    eps_m = exp(-(Ep - mu) / T)
    num_m = 3*phi*eps_m + 6*phib*eps_m^2 + 3*eps_m^3
    den_m = 1 + 3*phi*eps_m + 3*phib*eps_m^2 + eps_m^3
    dz_m_dM = -(M / (T * Ep)) * (num_m / den_m)

    return p^2 * (dz_p_dM + dz_m_dM)
end

function integrand_phi(p, phi, phib, M, mu, T)
    Ep = energia_p(p, M)

    eps_p = exp(-(Ep + mu) / T)
    den_p = 1 + 3*phib*eps_p + 3*phi*eps_p^2 + eps_p^3
    dz_p_dphi = (3*eps_p^2) / den_p

    eps_m = exp(-(Ep - mu) / T)
    den_m = 1 + 3*phi*eps_m + 3*phib*eps_m^2 + eps_m^3
    dz_m_dphi = (3*eps_m) / den_m

    return p^2 * (dz_p_dphi + dz_m_dphi)
end

function integrand_phib(p, phi, phib, M, mu, T)
    Ep = energia_p(p, M)

    eps_p = exp(-(Ep + mu) / T)
    den_p = 1 + 3*phib*eps_p + 3*phi*eps_p^2 + eps_p^3
    dz_p_dphib = (3*eps_p) / den_p

    eps_m = exp(-(Ep - mu) / T)
    den_m = 1 + 3*phi*eps_m + 3*phib*eps_m^2 + eps_m^3
    dz_m_dphib = (3*eps_m^2) / den_m

    return p^2 * (dz_p_dphib + dz_m_dphib)
end

# ----- Funções Finais (∂Ω/∂x = 0) -----

function dpotM(phi, phib, M, mu, T)
    integral_vacuo, _ = quadgk(p -> p^2 * M / energia_p(p, M), 0, L, rtol=1e-6)
    integral_meio, _ = quadgk(p -> integrand_M(p, phi, phib, M, mu, T), 0, Inf, rtol=1e-6)
    termo_G = (M - m) / (2 * G)
    return termo_G - ((T * Nf * integral_meio) / (pi^2)) - ((3 * Nf * integral_vacuo) / (pi^2))
end

function dpotphi(phi, phib, M, mu, T)
    integral_meio, _ = quadgk(p -> integrand_phi(p, phi, phib, M, mu, T), 0, Inf, rtol=1e-6)
    termo_U = deriv_U_phi(phi, phib, T)
    return termo_U - ((T * Nf * integral_meio) / (pi^2))
end

function dpotphib(phi, phib, M, mu, T)
    integral_meio, _ = quadgk(p -> integrand_phib(p, phi, phib, M, mu, T), 0, Inf, rtol=1e-6)
    termo_U = deriv_U_phib(phi, phib, T)
    return termo_U - ((T * Nf * integral_meio) / (pi^2))
end

# ===================================================================
# SEÇÃO 3: SOLVER DO SISTEMA NÃO-LINEAR
# ===================================================================

"""
    gap_system!(F, x, mu, T)

Define o sistema de 3 equações de gap a ser resolvido.
`F` é o vetor de resíduos (o que queremos que seja zero).
`x` é o vetor de variáveis: [Φ, Φ̄, M].
`mu` e `T` são os parâmetros.
"""
function gap_system!(F, x, mu, T)
    phi, phib, M = x[1], x[2], x[3]
    F[1] = dpotphi(phi, phib, M, mu, T)
    F[2] = dpotphib(phi, phib, M, mu, T)
    F[3] = dpotM(phi, phib, M, mu, T)
end

"""
    solve_gap_system(mu, T, initial_guess)

Resolve o sistema de equações para um único par (μ, T)
dado um palpite inicial.
"""
function solve_gap_system(mu, T, initial_guess)
    # Cria uma função que "fecha" os parâmetros mu e T,
    # deixando-a no formato que `nlsolve` espera: f(F, x)
    system_to_solve! = (F, x) -> gap_system!(F, x, mu, T)

    # Resolve o sistema
    solution = nlsolve(system_to_solve!, initial_guess)
    return solution.zero
end

"""
    solve_for_mu_range(T, mu_range)

Itera sobre uma faixa de valores de μ, resolve o sistema para cada um
e armazena os resultados. Utiliza a solução anterior como o próximo
palpite inicial (método de continuação).
"""
function solve_for_mu_range(T, mu_range)
    # Arrays para armazenar os resultados
    phi_vals = similar(mu_range, Float64)
    phib_vals = similar(mu_range, Float64)
    M_vals = similar(mu_range, Float64)

    # Palpite inicial para o primeiro ponto (μ = 0)
    chute_inicial = [0.01, 0.01, 1]

    # Itera sobre cada valor de potencial químico
    for (i, mu) in enumerate(mu_range)
        println("Calculando para μ = $(round(mu, digits=3))...")
        
        # Resolve o sistema para o mu atual
        solucao = solve_gap_system(mu, T, chute_inicial)
        
        # Armazena os resultados
        phi_vals[i] = solucao[1]
        phib_vals[i] = solucao[2]
        M_vals[i] = solucao[3]
        
        # Usa a solução atual como palpite para o próximo ponto
        chute_inicial = solucao
    end
    
    return phi_vals, phib_vals, M_vals
end


# ===================================================================
# SEÇÃO 4: EXECUÇÃO PRINCIPAL E PLOTAGEM
# ===================================================================
function main()
    println("Iniciando a resolução do sistema...")
    
    # Define os parâmetros para a simulação
    T_fixo = 0.07  # GeV
    faixa_mu = range(0, 1, length=50) # GeV
    
    # Executa o solver para a faixa de μ
    phi_resultados, phib_resultados, M_resultados = solve_for_mu_range(T_fixo, faixa_mu)
    
    println("Cálculos finalizados. Gerando o gráfico...")
    
    # Gera o gráfico de M em função de μ
    p = plot(faixa_mu, [phi_resultados, M_resultados],
        xlabel="Potencial Químico μ [GeV]",
        ylabel="Massa Dinâmica M [GeV]",
        title="Massa vs. Potencial Químico (T = $(T_fixo) GeV)",
        legend=false,
        lw=2, # Largura da linha
        marker=:circle,
        markersize=3
    )
    
    # Exibe o gráfico
    display(p)
end

# Executa a função principal
main()

println("Script concluído.")