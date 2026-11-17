# Análise estatística dos resultados salvos de cobertura (N locutores × K frases)

Só CPU, sem treino novo. Todos os números vêm dos `results.csv` salvos em `reports/coverage/results/<treino>/step_2500/strength_<s>[_r<R>]/`. Condição principal: passo 2.500, força do estilo 1,0, R = 1. "Copiado" = `sim_ref > sim_default`; margem contínua `m = sim_ref − sim_default`; margem por locutor `ms = sim_spk − sim_default`; q = 1 − fração copiada (fração **não** copiada).

Scripts (todos em `scratchpad/teoria/`, rodados com `python3 -I script.py <results_dir> out [B]`):

| Script | Tarefa | Saídas em `out/` |
|---|---|---|
| `dados_comum.py`, `dados_modelos.py`, `dados_leis_bin.py` | carregador; motores de ajuste | — |
| `dados_00_check.py` | conferência (conjuntos de referência iguais entre treinos; forças disponíveis) | — |
| `dados_01_tabela.py` | 1 | `tabela_runs.csv` |
| `dados_02_margem.py` | 2 | `02.txt`, `varcomp_por_treino.csv`, `seed_sd.json`, `precisao_por_treino.csv`, `leis_margem_unseen.csv`, `boot_dev_margem_unseen.npy` |
| `dados_03_link.py` | 3 | `03.txt`, `link_probit.csv`, `tudo_ou_nada.csv` |
| `dados_04_leis_binario.py` | 4 | `04.txt`, `leis_binario.csv`, `previsoes_binario.csv` |
| `dados_05_colapso.py` | 5 | `05.txt`, `colapso.csv` |
| `dados_06_artefato.py` | 6 | `06.txt`, `artefato_expoentes.csv`, `artefato_por_locutor.csv` |
| `dados_07_R.py` | 7 | `07.txt`, `efeito_R.csv` |
| `dados_08_poder.py`, `dados_08b_poder_extra.py` | 8 | `08.txt`, `08b.txt`, `poder.csv`, `poder_extra.csv` |

**Verificação do trabalho anterior (interrompido).** Os arquivos `dados_comum.py`, `dados_01_tabela.py`, `dados_02_margem.py`, `dados_modelos.py` e as saídas de `out/` foram relidos e conferidos: (i) os 33 treinos usam exatamente as mesmas 4 referências "unseen"; 30 usam as mesmas 8 "seen" (os 3 treinos N = 2 usam 8 referências de 2 locutores); as 3 avaliações `-wide` têm os mesmos 38 locutores externos × 25 frases (950 amostras); os 4 treinos plain com vozes externas (N = 128, 200) usam as mesmas 4. Isso valida empilhar treinos × locutores × frases numa ANOVA cruzada. (ii) A tabela reproduz os números de `teoria_cobertura.md` (ex.: 128 × 32 s1: 74% novas; `-wide`: 57/70/79% externas). (iii) A força 0 só existe nos treinos `-spk`/`-spkdrop` e não inclui as 38 vozes externas (relevante para a tarefa 6). Os resultados anteriores foram reaproveitados; nada foi corrigido neles, exceto o que vai indicado abaixo.

---

## 1. Tabela dos treinos

Grade (plain) e avaliação larga (`-wide`, mesmo checkpoint do plain de mesmo nome, re-sintetizado). Copiado = fração; WER novas/treinadas/externas; margem m média (dp entre amostras). "Ext." nos plain = 4 vozes externas; nos `-wide` = 38.

| treino | N | K | semente | copiado novas | copiado treinadas | copiado ext. | WER nov / trein / ext | m novas | m treinadas | m ext. | ms ext. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n2_k128_s1 | 2 | 128 | 1 | 0.04 | 0.69 | — | 0.079 / 0.129 / — | -0.112 (0.065) | +0.098 (0.134) | — | — |
| n2_k128_s2 | 2 | 128 | 2 | 0.02 | 0.73 | — | 0.055 / 0.114 / — | -0.114 (0.048) | +0.091 (0.103) | — | — |
| n8_k8_s1 | 8 | 8 | 1 | 0.26 | 0.45 | — | 0.143 / 0.207 / — | -0.052 (0.097) | -0.005 (0.120) | — | — |
| n8_k8_s2 | 8 | 8 | 2 | 0.39 | 0.45 | — | 0.244 / 0.262 / — | -0.033 (0.096) | -0.009 (0.112) | — | — |
| n8_k32_s1 | 8 | 32 | 1 | 0.47 | 0.74 | — | 0.137 / 0.173 / — | -0.024 (0.103) | +0.105 (0.114) | — | — |
| n8_k32_s1-wide | 8 | 32 | 1 | 0.36 | 0.78 | 0.57 | 0.106 / 0.140 / 0.134 | -0.040 (0.088) | +0.105 (0.115) | -0.004 (0.106) | +0.026 (0.108) |
| n8_k128_s1 | 8 | 128 | 1 | 0.51 | 0.91 | — | 0.079 / 0.112 / — | -0.011 (0.106) | +0.172 (0.120) | — | — |
| n8_k128_s2 | 8 | 128 | 2 | 0.37 | 0.88 | — | 0.094 / 0.104 / — | -0.029 (0.107) | +0.157 (0.118) | — | — |
| n32_k8_s1 | 32 | 8 | 1 | 0.47 | 0.58 | — | 0.126 / 0.172 / — | +0.009 (0.114) | +0.047 (0.107) | — | — |
| n32_k8_s2 | 32 | 8 | 2 | 0.47 | 0.51 | — | 0.085 / 0.169 / — | -0.006 (0.109) | +0.019 (0.087) | — | — |
| n32_k32_s1 | 32 | 32 | 1 | 0.52 | 0.80 | — | 0.077 / 0.130 / — | +0.026 (0.104) | +0.088 (0.099) | — | — |
| n32_k32_s1-wide | 32 | 32 | 1 | 0.48 | 0.76 | 0.70 | 0.077 / 0.121 / 0.116 | +0.015 (0.118) | +0.095 (0.112) | +0.048 (0.085) | +0.071 (0.085) |
| n32_k128_s1 | 32 | 128 | 1 | 0.61 | 0.86 | — | 0.052 / 0.121 / — | +0.045 (0.105) | +0.147 (0.117) | — | — |
| n32_k128_s2 | 32 | 128 | 2 | 0.57 | 0.85 | — | 0.050 / 0.097 / — | +0.037 (0.097) | +0.149 (0.115) | — | — |
| n128_k8_s1 | 128 | 8 | 1 | 0.55 | 0.52 | 0.86 | 0.081 / 0.121 / 0.126 | +0.007 (0.086) | +0.034 (0.114) | +0.087 (0.065) | +0.063 (0.070) |
| n128_k32_s1 | 128 | 32 | 1 | 0.74 | 0.76 | 0.86 | 0.077 / 0.120 / 0.157 | +0.058 (0.075) | +0.090 (0.117) | +0.091 (0.070) | +0.096 (0.064) |
| n128_k32_s1-wide | 128 | 32 | 1 | 0.75 | 0.78 | 0.79 | 0.071 / 0.120 / 0.117 | +0.056 (0.077) | +0.095 (0.117) | +0.078 (0.094) | +0.101 (0.088) |
| n128_k32_s2 | 128 | 32 | 2 | 0.56 | 0.66 | 0.86 | 0.076 / 0.128 / 0.105 | +0.016 (0.082) | +0.063 (0.100) | +0.076 (0.058) | +0.055 (0.079) |
| n200_k32_s1 | 200 | 32 | 1 | 0.64 | 0.67 | 0.91 | 0.073 / 0.122 / 0.105 | +0.030 (0.073) | +0.073 (0.116) | +0.097 (0.071) | +0.077 (0.084) |

Variantes (uma semente salvo indicação; 4 novas × 25 e 8 treinadas × 25):

| treino | copiado novas | copiado treinadas | WER nov / trein | m novas | m treinadas |
|---|---|---|---|---|---|
| n2_k128_s1-spk | 0.04 | 0.69 | 0.079 / 0.139 | -0.101 | +0.097 |
| n8_k8_s1-aug | 0.28 | 0.50 | 0.139 / 0.175 | -0.042 | -0.004 |
| n8_k8_s1-contrast | 0.27 | 0.42 | 0.199 / 0.273 | -0.073 | -0.023 |
| n8_k8_s2-contrast | 0.31 | 0.47 | 0.208 / 0.229 | -0.072 | -0.026 |
| n8_k8_s1-spk | 0.30 | 0.43 | 0.140 / 0.183 | -0.053 | -0.003 |
| n8_k8_s1-swap | 0.27 | 0.52 | 0.206 / 0.230 | -0.075 | -0.026 |
| n8_k8_s2-swap | 0.32 | 0.47 | 0.245 / 0.254 | -0.062 | -0.007 |
| n8_k128_s1-aug | 0.42 | 0.83 | 0.077 / 0.113 | -0.009 | +0.141 |
| n8_k128_s1-contrast | 0.39 | 0.85 | 0.095 / 0.138 | -0.041 | +0.145 |
| n8_k128_s1-spk | 0.46 | 0.88 | 0.084 / 0.128 | -0.014 | +0.164 |
| n8_k128_s1-spkdrop | 0.37 | 0.88 | 0.076 / 0.141 | -0.019 | +0.165 |
| n8_k128_s1-swap | 0.38 | 0.87 | 0.079 / 0.122 | -0.047 | +0.145 |
| n32_k8_s1-spkdrop | 0.60 | 0.57 | 0.147 / 0.218 | +0.029 | +0.035 |
| n32_k128_s1-spkdrop | 0.62 | 0.83 | 0.056 / 0.119 | +0.038 | +0.142 |

Tabela completa (inclui sd de `ms`, `sim_ref`, `sim_default` médios e cópia por `sim_spk`): `out/tabela_runs.csv` e `out/tabela_runs.md`. Nas variantes, todas as margens das vozes novas caem dentro da faixa das duas sementes plain da mesma configuração ± 2 DP de semente (0,015), exceto `-spkdrop` 32 × 8 (+0,029 contra +0,009/−0,006), que fica ~1,5 DP acima: nenhuma variante tem efeito demonstrável com uma semente.

---

## 2. Margem contínua

### 2.1 Métodos

- **Por treino e grupo**: ANOVA de 2 vias balanceada locutor × frase (efeitos aleatórios, estimadores de momentos), separando DP entre locutores, entre frases e residual (que inclui a interação locutor × frase).
- **Conjunta**: ANOVA de 3 vias cruzada treino × locutor × frase (16 treinos plain × 4 novas × 25; 14 treinos × 8 treinadas sem N = 2; 3 treinos `-wide` × 38 externas × 25). O "efeito de treino" contém o efeito fixo de (N, K) e a semente; por isso a variância de semente é estimada à parte.
- **Sementes**: 6 pares de réplicas (2×128, 8×8, 8×128, 32×8, 32×128, 128×32). Para cada par, D = (Y₁ − Y₂)/√2 por locutor × frase; E[D̄²] estima a variância de semente da média do treino (com o conjunto de avaliação fixo). Decompõe-se em efeito principal de semente e interações semente × locutor e semente × frase (ANOVA de D). IC 95% qui-quadrado com 6 g.l.
- **Re-síntese**: os 3 pares plain/`-wide` são o mesmo checkpoint sintetizado de novo; dão a variância da síntese sozinha.

### 2.2 Componentes de variância (DP)

| Fonte | Margem m, novas (3 vias, 16 treinos) | Binário, novas | Margem m, externas (38, 3 treinos) | Binário, externas |
|---|---|---|---|---|
| locutor | 0,095 | 0,374 | 0,076 | 0,309 |
| treino × locutor | 0,033 | 0,196 | 0,046 | 0,209 |
| frase | 0,007 | 0,024 | 0,009 | 0,025 |
| treino × frase | 0,008 | 0,030 | 0,006 | 0,022 |
| locutor × frase | 0,016 | 0,077 | 0,014 | 0,055 |
| residual | 0,028 | 0,265 | 0,031 | 0,261 |
| treino (inclui o efeito de N, K) | 0,048 | 0,171 | 0,041 | 0,107 |

A frase quase não importa (DP 0,007–0,009 na margem); **a variabilidade é de locutor** (ICC da margem dentro do treino: mediana 0,91 nas novas, 0,92 nas treinadas, 0,81–0,85 nas externas).

DP **entre sementes** da média de um treino (avaliação fixa), 6 pares:

| | DP total (IC 95%) | efeito principal de semente | parte vinda de interações semente × locutor/frase |
|---|---|---|---|
| margem, novas | 0,0150 (0,0097–0,033) | 0,0127 | 0,0080 |
| binário, novas | 0,077 (0,050–0,169) | 0,054 | 0,054 |
| margem, treinadas | 0,0125 (0,0081–0,028) | 0,0098 | 0,0078 |
| binário, treinadas | 0,039 (0,025–0,085) | 0,000 | 0,041 |

Re-síntese do mesmo checkpoint (3 pares): DP da média 0,0080 (margem, novas) e 0,048 (binário, novas). Ou seja, **cerca de 40% da variância "de semente" do binário nas 4 vozes novas é só re-síntese** (amostragem do decodificador), não treino.

Implicação central: o efeito principal de semente (0,0127 na margem ≈ 0,05 na fração) **não diminui com mais vozes de avaliação**. Com 38 vozes, a parte locutor da incerteza cai para ~0,046/√38 ≈ 0,007, e a semente passa a dominar.

### 2.3 Ganho de precisão da margem sobre o binário

Comparação na mesma escala (fração copiada), por três caminhos:

| Caminho | Razão de variâncias binário / margem | Leitura |
|---|---|---|
| Dentro do treino: bootstrap locutor × frase de p̂ vs Φ(m̄/s) | mediana 1,00 (novas), 1,56 (treinadas), 1,31 (externas, 38) | quase nenhum ganho com 4 vozes, porque a incerteza é de locutor e afeta as duas medidas |
| Entre sementes (DP de semente × inclinação dp/dμ entre treinos = 3,83 novas, 2,68 treinadas) | 1,78 (novas), 1,32 (treinadas) | margem equivale a 1,3–1,8× mais treinos |
| Re-síntese | 2,4 (novas), 7,8 (treinadas) | a margem é bem mais estável à amostragem do decodificador |

Razão sinal/ruído (DP entre configurações / DP de semente): fração 2,38, margem 3,13 (novas). **O ganho é real, porém modesto (1,3–2×)**: a margem não resolve o problema principal, que é variância de locutor e de semente.

### 2.4 Leis para a margem média (novas, 16 treinos, 10 configurações)

Mínimos quadrados não ponderados por treino; CV deixando uma configuração de fora (LOCO: as duas sementes saem juntas); AICc/BIC com k+1 parâmetros (σ).

| Lei | k | RMSE | CV-RMSE (LOCO) | ΔAICc | ΔBIC | peso AICc |
|---|---|---|---|---|---|---|
| a + b/N | 2 | 0,0196 | 0,044 | 11,8 | 12,7 | 0,002 |
| a + b/N + c/(NK) [L1] | 3 | 0,0146 | 0,033 | 6,0 | 6,0 | 0,033 |
| a + b/√N | 2 | 0,0172 | 0,019 | 7,6 | 8,5 | 0,015 |
| a + b/√N + c/√K [L2] | 3 | 0,0121 | **0,016** | 0,0 | 0,0 | 0,65 |
| a + b ln N | 2 | 0,0252 | 0,034 | 19,9 | 20,8 | 0 |
| a + b ln N + c ln K | 3 | 0,0240 | 0,038 | 22,0 | 22,0 | 0 |
| a + b N^−α | 3 | 0,0170 | 0,141 | 10,8 | 10,9 | 0,003 |
| a + b N^−α + c ln K | 4 | 0,0111 | 0,069 | 1,6 | 0,0 | 0,30 |
| só K / constante | 2 / 1 | 0,048 / 0,049 | 0,065 / 0,055 | ≥38 | ≥39 | 0 |

- Piso de RMSE alcançável por treino = DP de semente = 0,015. **L2 já está no nível do ruído** (0,012 dentro, 0,016 fora da amostra); nada mais complexo se justifica.
- Sem N = 2 (14 treinos): L2 e a + b N^−α + c ln K continuam à frente (CV 0,016 e 0,015), mas 1/N, 1/√N e ln N ficam a ≤0,01 delas.
- Expoente α na forma a + b N^−α + c ln K: 0,64, IC bootstrap locutor × frase [0,08; 1,08]; sem N = 2 o ponto vai para 1,24. **O expoente da margem não é identificado.**
- Bootstrap (B = 1.000) das diferenças de CV-RMSE em relação a a + b ln N + c ln K: só "só K" e "constante" são piores com certeza (P = 1,00 e 0,99). L2 é melhor em 90% das reamostras, sem significância. (Ressalva: o bootstrap reamostra os 4 locutores, o que com 4 locutores cria conjuntos degenerados; nas tarefas seguintes os 4 locutores ficam fixos.)
- **Identificabilidade** (bootstrap paramétrico: dados = lei ajustada + N(0; 0,015²) por treino, B = 1.000): escolhendo pelo AICc, a lei certa é escolhida em 57–85% dos casos; a potência a + b N^−α é confundida com a + b/√N em 73% e a + b N^−α + c ln K com L2 em 46%. Na grade atual, formas com expoente livre são indistinguíveis de 1/√N.
- **Treinadas** (margem): só K importa. a + c ln K (sem N = 2: peso AICc 0,75); somar ln N dá coeficiente −0,0006. Isso confirma o fato 1 da teoria com a medida contínua.

---

## 3. Ligação probit e tudo-ou-nada por locutor

### 3.1 Φ(média/dp) prevê a fração binária?

Para cada treino × grupo (45 casos plain + wide):

| Preditor de p | viés médio | RMSE | correlação | máx. desvio |
|---|---|---|---|---|
| Φ(m̄/s) global (normal única) | +0,030 | 0,050 | 0,984 | 0,103 |
| mistura por locutor: média_s Φ(m̄_s/s_w), s_w = dp dentro do locutor | +0,003 | **0,018** | 0,996 | 0,062 |

O probit global superestima a cópia, sobretudo nas treinadas (viés +0,05): a distribuição da margem é uma mistura de locutores, não uma normal. Com a mistura por locutor o ajuste fica dentro do erro amostral (|z| mediano do probit global já é 0,28 EPs bootstrap). **A fração binária é, em boa aproximação, uma função determinística das margens médias por locutor e de um DP dentro do locutor (~0,034).** Isso justifica usar a margem como medida primária e converter para "% copiado" só no final.

### 3.2 Tudo-ou-nada

| Grupo | locutores com cópia ≤10% ou ≥90% (observado) | esperado sob binomial homogênea | ICC latente (probit, intercepto aleatório, MV com quadratura de Gauss-Hermite) | ICC da margem |
|---|---|---|---|---|
| novas (4) | mediana 50–75% | ~0–3% | mediana 0,87 (0,62–1,00) | 0,91 |
| treinadas (8) | 69–75% | ~0–30% | 0,93 | 0,92 |
| externas (38, wide) | 66% (25, 24, 28 de 38) | 0–8% | 0,83–0,91 | 0,82–0,89 |

Qui-quadrado de sobredispersão da fração por locutor: 584–687 com 37 g.l. nas externas (p ≈ 0). Histograma das 38 externas (≤10% | … | ≥90%): N = 8: 10 | 4 | 3 | 1 | 5 | 15; N = 32: 4 | 3 | 4 | 5 | 2 | 20; N = 128: 4 | 1 | 3 | 2 | 4 | 24. A dificuldade é da voz: correlação de postos da taxa por locutor entre N = 8–32, 32–128, 8–128: 0,75 / 0,74 / 0,59 (binário) e 0,78 / 0,84 / 0,72 (margem). **Consequência:** a unidade de amostragem efetiva é o locutor, não a amostra; 25 frases por voz são muito mais do que o necessário (frase tem DP 0,007), enquanto 4 vozes são muito poucas.

---

## 4. Leis L1–L4 para a fração binária e previsões

### 4.1 Métodos

- Resposta: q por treino (fração não copiada). Leis com 3 parâmetros (4 vozes, com K) ou 2 (38 vozes, só N):
  - L1: a + A/N + B/(NK) (coeficientes ≥ 0, NNLS); L1n: a + A/N.
  - L2: a + A/√N + C/√K (≥ 0); L2n: a + A/√N.
  - L3: c N^−α K^−β (grade α ∈ [0,01; 1,5], β ∈ [−0,5; 1,5], c por MQ); L3n: c N^−α.
  - L4: q = Φ(a + b ln N + c ln K), MQ não linear em q ("probit direto").
  - L4m: probit **da margem**: ajusta-se a margem média μ(N, K) = a + b ln N + c ln K (ou a + b/√N + c/√K, L4m′) e converte-se q = média_s Φ(−(μ + u_s)/σ), com u_s = efeito fixo de cada locutor de avaliação e σ² = variância dentro de treino×locutor + interação treino × locutor (seção 3.1). Os nuisances u_s, σ são estimados com todos os treinos (vazamento mínimo na CV).
- CV LOCO por configuração. AICc/BIC.
- Bootstrap B = 500: (A) "avaliação" — reamostra frases; os 4 locutores ficam **fixos** (com 4 locutores, reamostrar locutores gera conjuntos degenerados; a métrica de 4 vozes é, por definição, daqueles 4); nas 38 externas reamostra locutores e frases, pareados entre treinos. (B) "avaliação + semente" — soma N(0; s²) por treino, s = 0,077 (DP total de semente da fração, 4 vozes) ou 0,054 (efeito principal de semente, usado para as 38 vozes, onde a semente não foi medida). (B) é conservador (a semente observada já está nos dados).

### 4.2 Métrica de 4 vozes novas (16 treinos plain)

| Lei | RMSE | CV-RMSE | ΔAICc | Cópia prevista 247×32 | 1000×32 | 2456×32 | 128×128 |
|---|---|---|---|---|---|---|---|
| L1 | 0,060 | 0,103 | 2,2 | 0,60 | 0,61 | 0,61 | 0,60 |
| L2 | 0,056 | **0,075** | **0** | 0,67 | 0,70 | 0,71 | 0,68 |
| L3 (α = 0,25, β = 0,04) | 0,078 | 0,136 | 10,5 | 0,72 | 0,80 | 0,84 | 0,69 |
| L4 direto | 0,099 | 0,146 | 18,3 | 0,74 | 0,87 | 0,92 | 0,71 |
| L4m (margem ln) | 0,103 | 0,147 | 19,4 | 0,75 | 0,87 | 0,93 | 0,71 |
| L4m′ (margem √) | 0,070 | 0,079 | 7,1 | 0,68 | 0,71 | 0,72 | 0,69 |

Intervalos 95% (bootstrap avaliação + semente) das previsões: L1 1000×32: 0,61 [0,55; 0,66]; L2: 0,70 [0,63; 0,77]; L3: 0,80 [0,73; 0,85]; L4: 0,87 [0,78; 0,93]; 2456×32: L1 0,61 [0,55; 0,66], L2 0,71 [0,64; 0,78], L3 0,84 [0,78; 0,89], L4 0,92 [0,84; 0,96]; 128×128: 0,60–0,71 em todas as leis (L2 0,68 [0,61; 0,76]). Esses intervalos são **condicionais à lei**; a incerteza entre leis (0,61 a 0,93 em N = 2.456) é muito maior que a de cada uma.

Bootstrap das diferenças de CV-RMSE em relação a L2 (avaliação + semente): L1 +0,041 [−0,02; +0,18] (P pior 0,84); L3 +0,044 [0,000; +0,076] (0,97); L4 +0,052 [+0,013; +0,079] (0,996); L4m +0,056 [+0,009; +0,11] (0,99); L4m′ +0,0005 [−0,026; +0,029]. Mas **isso é inteiramente o ponto N = 2** (q ≈ 0,97), que as leis com ln N não conseguem acompanhar. Sem N = 2 (14 treinos): CV-RMSE 0,114 (L1), 0,058 (L2), 0,066 (L3), 0,064 (L4), 0,074 (L4m), 0,059 (L4m′); ΔAICc ≤ 1,3 entre L2, L3, L4, L4m, L4m′; diferenças de CV em relação a L2 com IC contendo 0 para todas menos L1 (+0,053 [+0,015; +0,098] só no bootstrap de avaliação; com semente, [−0,025; +0,17]). L3 sem N = 2 dá α = 0,17, β = 0,08.

**Identificabilidade** (dados gerados de cada lei ajustada + N(0; 0,077²) por treino, B = 200): escolhendo por AICc, a lei certa é recuperada em 62–80% para L1–L4, mas L4m (lei na margem) é tomada por L4 em 73% (são quase a mesma função) e L4m′ por L2 em 47%. Por CV LOCO a recuperação cai para 28–59%. **Os dados da grade não distinguem as leis fora de N = 2; distinguem só "com N = 2 dentro, formas com piso/raiz se ajustam melhor que log-probit".**

### 4.3 Métrica de 38 vozes externas (3 treinos `-wide`, K = 32, **uma semente**)

Atenção: são 3 pontos e 2 parâmetros por lei (1 g.l. de resíduo); não existe K para variar; a semente não foi replicada nessa avaliação. AICc com 3 pontos não é informativo (L3n passa exatamente pelos 3 pontos: q = 0,744 N^−0,26, porque as razões 0,302/0,434 = 0,696 e 0,211/0,302 = 0,698 são praticamente iguais).

| Lei | RMSE | CV (deixa 1 de 3) | 247 | 1000 | 2456 | IC 95% em 2456 (avaliação) | IC 95% em 2456 (avaliação + semente 0,054) |
|---|---|---|---|---|---|---|---|
| L1n a + A/N | 0,021 | 0,142 | 0,77 | 0,78 | 0,78 | [0,66; 0,88] | [0,62; 0,92] |
| L2n a + A/√N | 0,008 | 0,035 | 0,80 | 0,83 | 0,84 | [0,71; 0,95] | [0,66; 0,98] |
| L3n c N^−α (α = 0,26) | 0,000 | 0,000 | 0,82 | 0,88 | 0,90 | [0,79; 0,97] | [0,72; 0,98] |
| L4n Φ(a + b ln N) | 0,005 | 0,019 | 0,84 | 0,90 | 0,93 | [0,82; 0,99] | [0,71; 1,00] |
| L4mn (margem ln N) | 0,043 | 0,040 | 0,84 | 0,92 | 0,95 | [0,89; 0,99] | [0,85; 1,00] |
| L4mn′ (margem 1/√N) | 0,046 | 0,064 | 0,79 | 0,82 | 0,83 | [0,74; 0,91] | [0,72; 0,94] |

Bootstrap de locutores (38) das diferenças de CV em relação a L2n: todas as diferenças têm IC contendo 0 (ex.: L1n +0,10 [−0,06; +0,22], P pior 0,83; L3n −0,008 [−0,05; +0,04]). Probabilidade de cada lei ser a melhor: 0,10–0,25 para todas. **Com 3 pontos de uma semente não há como distinguir as leis.** Previsão honesta para N = 2.456, K = 32 nas 38 vozes: entre ~0,78 (piso em 1/N) e ~0,95 (probit em ln N), com cada lei tendo ainda ±0,1–0,15 de incerteza própria. A extrapolação "88% com 1.000 locutores" da proposta é a da lei L3n, uma de seis igualmente compatíveis.

Os níveis das duas métricas não se comparam: as 4 vozes novas são mais difíceis (36–75% contra 57–79% copiadas no mesmo checkpoint).

---

## 5. Colapso dos dados: x = N^a K^b

### 5.1 Métodos

Só a razão b/a é identificável num colapso não paramétrico (f(N^a K^b) = g(N^{ca} K^{cb})). Por isso estimei a e b de quatro formas:
- (i) potência na fração não copiada: q = c N^−a K^−b (grade a ∈ [−0,5; 1,5], b ∈ [−0,5; 2], passo 0,01);
- (ii) probit: Φ⁻¹(1 − q) = c₀ + a ln N + b ln K (MQ);
- (iii) margem média: m̄ = c₀ + a ln N + b ln K (MQ);
- (iv) colapso não paramétrico: direção θ no plano (ln N, ln K) que minimiza o SSE de uma regressão isotônica crescente de y em x = cos θ ln N + sin θ ln K; b/a = tan θ.

Intervalos 95%: bootstrap (B = 400) de frases com locutores fixos + ruído de semente N(0; s²) por treino (s = DP total de semente medido: 0,077/0,015 para fração/margem nas novas; 0,039/0,0125 nas treinadas); e a mesma coisa reamostrando também os locutores (4 novas, 8 treinadas). Treinadas: sem N = 2 (outras referências).

### 5.2 Resultados

| Grupo | Estimador | a (expoente de N) | b (expoente de K) | b/a |
|---|---|---|---|---|
| novas, com N = 2 | (i) potência | 0,25 [0,19; 0,31] (loc.: [0,07; 1,18]) | 0,04 [−0,03; 0,10] | 0,16 |
| | (ii) probit | 0,41 [0,29; 0,51] | 0,01 [−0,09; 0,12] | 0,03 |
| | (iii) margem | 0,032 [0,026; 0,037] | 0,007 [0,000; 0,013] | 0,22 |
| | (iv) isotônico | — | — | ≈0 (θ = 0,5°) |
| novas, sem N = 2 | (i) potência | 0,17 [0,08; 0,25] (loc.: [0,00; 0,68]) | 0,08 [0,01; 0,16] (loc.: [−0,02; 0,33]) | 0,47 |
| | (ii) probit | 0,21 [0,12; 0,33] | 0,12 [0,02; 0,22] | 0,54 |
| | (iii) margem | 0,022 [0,015; 0,028] | 0,012 [0,006; 0,019] | 0,55 |
| | (iv) isotônico | — | — | 0,51 [0,01; 1,33] (θ = 27° [0,5°; 53°]) |
| treinadas, sem N = 2 | (i) potência | 0,01 [−0,06; 0,07] (loc.: [−0,11; 0,15]) | 0,46 [0,38; 0,56] (loc.: [0,20; 0,99]) | ≈ 46 |
| | (ii) probit | −0,015 [−0,09; 0,05] | 0,42 [0,35; 0,53] | — |
| | (iii) margem | −0,001 [−0,006; 0,006] | 0,050 [0,044; 0,056] | — |
| | (iv) isotônico | — | — | 2,36 (θ = 67°; o bootstrap não varia: ver abaixo) |

Leitura:
- **Treinadas: x ≈ K^0,46 (N^0)**, robusto em todas as formas. O expoente de N tem IC centrado em 0 com largura ±0,07.
- **Novas: o colapso depende de N = 2.** Com N = 2, o ponto extremo (q ≈ 0,97) força x ≈ N^0,25 e K quase não entra (b/a ≈ 0–0,2). Sem N = 2, os três estimadores paramétricos concordam em **b/a ≈ 0,5** (x ≈ N K^0,5, ou N^0,17 K^0,08 na potência), com b > 0 significativo quando os locutores ficam fixos e não significativo quando também se reamostram os 4 locutores. Isso quer dizer que, entre N = 8 e 200, K ajuda as vozes novas com cerca de metade do peso de N (em log) — coerente com o termo C/√K de L2 e com o 128 × 32 > 128 × 8.
- Com reamostragem de locutores os ICs de a, b nas novas cobrem de 0 a ~0,7: com 4 vozes, o colapso é descritivo.
- O colapso isotônico é frágil com 14–16 pontos: o SSE é constante numa faixa larga de θ (várias direções produzem a mesma ordenação), e o argmin pega a borda da faixa; daí ICs degenerados (treinadas) ou enormes (novas). Não deve ser usado como estimativa.
- As 38 vozes não entram aqui (só K = 32).

---

## 6. Artefato da medida ou efeito real (38 vozes externas)

### 6.1 Proximidade da referência à voz padrão

A quantidade ideal seria c_s = semelhança entre a saída na voz padrão (força 0) e a referência. Ela só existe para as 4 vozes novas (treinos `-spk` a força 0, onde `sim_default` = 0,905): c_s = 0,600 / 0,610 / 0,603 / 0,690. Tentei estimá-la sem força 0, regredindo `sim_ref` em `sim_default` dentro do locutor e extrapolando a `sim_default` = 0,905; **a validação nas 4 vozes falhou** (estimativas 0,66–0,73 contra 0,60–0,69, inclinações instáveis), então essa estimativa foi descartada. Restam dois proxies, como pedido:

- f0 da referência (exógeno; a voz padrão é feminina);
- nível de `sim_default` do locutor (média dos 3 treinos): quanto a saída dessa voz fica parecida com a voz padrão. É **endógeno** (medido na saída e parte da própria margem), logo dá um limite superior do que "atração pela voz padrão" explica.

Correlação de postos entre os dois proxies: 0,69.

### 6.2 Quanto da falha por locutor cada coisa explica

Decomposição da soma de quadrados nas 38 vozes × 3 N (médias por locutor e N):

| Medida | N | locutor | resto (locutor × N + ruído) | do efeito de locutor: R² por f0 | por sim_default | ambos |
|---|---|---|---|---|---|---|
| q (binário) | 0,06 | 0,74 | 0,21 | 0,28 | 0,55 | 0,55 |
| margem m | 0,13 | 0,72 | 0,16 | 0,30 | 0,59 | 0,59 |
| sim_ref | 0,08 | 0,72 | 0,20 | 0,03 | 0,09 | 0,09 |
| sim_spk | 0,06 | 0,72 | 0,22 | 0,02 | 0,06 | 0,06 |

Regressão logística amostra a amostra (McFadden pseudo-R²): ln N sozinho 0,031; f0 0,117; sim_default do locutor 0,234; ln N + f0 0,154; ln N + sim_default 0,278; teto com o locutor como fator (+ ln N) 0,540.

Leitura:
- **A identidade da voz explica ~12 vezes mais que N** (74% contra 6% da variação de q nesta faixa de N).
- Na métrica que subtrai a voz padrão (q, m), cerca de 30% do efeito de locutor vem com o f0 e ~55–59% com o nível de `sim_default`. Nas métricas que **não** usam a voz padrão (`sim_ref`, `sim_spk`), f0 e `sim_default` explicam quase nada (2–9%). Portanto a parte da dificuldade ligada a vozes agudas/próximas da padrão é, em grande medida, **construída pelo critério** `sim_ref > sim_default`, enquanto a dificuldade medida só pela semelhança com a referência é outra (e não tem preditor simples aqui).
- Correlações por N: cópia × f0: −0,53, −0,52, −0,37; cópia × sim_default: −0,64, −0,67, −0,61; `sim_ref` × f0: +0,14, +0,30, −0,01. Vozes agudas não são reproduzidas pior pela semelhança à referência; elas perdem no critério binário porque a saída também fica parecida com a padrão.

### 6.3 A curva em N com medidas independentes da voz padrão

Observação de método: como os mesmos 38 locutores entram nos 3 treinos, **qualquer correção aditiva por locutor (por exemplo `sim_ref − c_s`) cancela nas diferenças entre N**; a forma da curva só muda com medidas não lineares (limiar) ou que dependam de outra coisa (`sim_default` muda com N). Usei o expoente pela razão de ganhos, livre de piso: α_g = log₄[(y₃₂ − y₈)/(y₁₂₈ − y₃₂)] (válido para y = y_∞ − A N^−α); IC por bootstrap dos 38 locutores (pareado entre N, B = 2.000). Para o binário, também α₀ = −log₁₆(q₁₂₈/q₈) (potência sem piso).

| Medida (média nas 38) | N=8 | 32 | 128 | ganho 8→32 | 32→128 | α_g (IC 95%) | P(α_g > 0,5) |
|---|---|---|---|---|---|---|---|
| cópia (sim_ref > sim_default) | 0,566 | 0,698 | 0,789 | 0,132 | 0,092 | 0,26 (−0,73; 1,57); α₀ = 0,26 | 0,31 |
| margem m | −0,004 | 0,048 | 0,078 | 0,052 | 0,030 | 0,39 (−0,09; 0,95) | 0,32 |
| margem por locutor ms | 0,026 | 0,071 | 0,101 | 0,045 | 0,029 | 0,31 (−0,22; 0,81) | 0,22 |
| cópia por sim_spk > sim_default | 0,661 | 0,805 | 0,878 | 0,144 | 0,073 | 0,50 (−0,30; 1,80); α₀ = 0,37 | 0,50 |
| **sim_ref** (sem voz padrão) | 0,648 | 0,677 | 0,690 | 0,029 | 0,014 | 0,55 (−0,17; 1,88) | 0,56 |
| **sim_spk** (sem voz padrão) | 0,678 | 0,700 | 0,713 | 0,022 | 0,013 | 0,41 (−0,59; 2,00) | 0,42 |
| −sim_default | −0,651 | −0,629 | −0,612 | 0,022 | 0,017 | 0,21 (−0,95; 1,54) | 0,27 |

Somando ruído de semente aos 3 pontos (efeito principal medido nas 4 vozes: 0,054 na fração, 0,0127 na margem), o IC de α_g da cópia vai para (−1,9; 2,3) e o da margem para (−0,9; 2,0); P(α_g < 0) = 0,42 e 0,28.

Subgrupos (19 + 19 locutores):

| Subgrupo | cópia 8 → 32 → 128 | sim_ref 8 → 32 → 128 |
|---|---|---|
| f0 abaixo da mediana (graves) | 0,76 → 0,83 → 0,88 | 0,635 → 0,649 → 0,680 (ganho acelera) |
| f0 acima da mediana (agudas) | 0,37 → 0,57 → 0,70 | 0,661 → 0,704 → 0,700 (para em 32) |
| saída longe da padrão | 0,81 → 0,92 → 0,91 (teto) | 0,637 → 0,654 → 0,689 (acelera) |
| saída perto da padrão | 0,32 → 0,48 → 0,67 (ganho acelera) | 0,659 → 0,699 → 0,692 (para) |

**Conclusão da tarefa 6.** (i) "Desacelera com N" sobrevive como sinal de ponto: em todas as medidas, inclusive nas que não usam a voz padrão (`sim_ref`, `sim_spk`), o ganho de 32→128 é menor que o de 8→32, e os pontos de α_g ficam entre 0,2 e 0,55. (ii) **O valor 0,26 não sobrevive como número**: a medida sem voz padrão dá 0,41–0,55; e mesmo sem semente nenhum IC exclui 0 ou 1. (iii) A curva agregada do binário é a soma de dois comportamentos opostos: vozes graves/longe da padrão batem no teto do binário já em N = 32 enquanto a semelhança contínua continua subindo; vozes agudas/perto da padrão ainda estão cruzando o limiar. A forma "potência lenta sem piso" do binário depende, portanto, da composição do conjunto de vozes relativo à voz padrão — é em parte artefato do critério. Ressalva: o subgrupo por `sim_default` é endógeno (regressão à média possível); o por f0 é exógeno e mostra o mesmo padrão.

---

## 7. Efeito de R (média dos códigos de 4/16 referências)

Pareado por amostra (mesmo treino, locutor e frase; R = 1 versus R). Diferença média sobre treinos e locutores; IC 95% por bootstrap dos locutores (os mesmos em todos os treinos do conjunto); Wilcoxon sobre as médias por locutor (só com ≥ 5 locutores não empatados). Treinos com R = 4: os 4 plain com N ≥ 128 e os 3 `-wide`; R = 16: 128×8 e 128×32 s1.

| Grupo (treinos) | n loc | Δ cópia (sim_ref) | Δ cópia por locutor (sim_spk) | Δ margem m | Δ ms | Δ sim_spk | Δ sim_default | Δ WER |
|---|---|---|---|---|---|---|---|---|
| externas, wide (3), R=4 | 38 | −0,097 [−0,169; −0,035], p=0,006 | −0,025 [−0,064; +0,008], 14↑ 18↓ | −0,020 [−0,033; −0,009] | −0,002 [−0,010; +0,006] | +0,005 [−0,002; +0,012] | +0,007 [−0,003; +0,017] | **−0,018 [−0,028; −0,009]**, p=0,009 |
| externas, plain (4), R=4 | 4 | −0,24 [−0,73; +0,02] | +0,105 [−0,02; +0,32] | −0,042 | +0,009 | +0,029 [+0,003; +0,071] | +0,020 | −0,033 [−0,071; +0,005] |
| novas, plain (4), R=4 | 4 | −0,28 [−0,58; +0,03] | **+0,13 [+0,01; +0,35]** (3↑ 0↓) | −0,041 | +0,024 [−0,003; +0,052] | +0,030 | +0,005 | +0,002 |
| novas, wide (3), R=4 | 4 | −0,24 [−0,49; −0,08] | +0,04 [−0,03; +0,14] | −0,034 | +0,008 | +0,030 [+0,005; +0,061] | +0,023 | −0,000 |
| treinadas, plain (4), R=4 | 8 | −0,14 [−0,31; −0,01] | −0,065 [−0,27; +0,08] | −0,033 [−0,062; −0,005] | −0,016 | +0,003 | +0,018 | −0,016 [−0,036; +0,003] |
| treinadas, wide (3), R=4 | 8 | −0,13 [−0,34; +0,03] | −0,07 [−0,20; +0,02] | −0,032 | −0,011 | +0,006 | +0,017 | −0,016 |
| novas, R=16 (2) | 4 | −0,38 [−0,70; −0,05] | +0,09 [−0,08; +0,33] | −0,068 | +0,017 | +0,028 | +0,011 | +0,016 |
| externas plain, R=16 (2) | 4 | −0,17 | +0,085 [−0,075; +0,30] | −0,047 | +0,008 | +0,040 [+0,007; +0,074] | +0,033 [+0,016; +0,049] | −0,049 [−0,112; −0,008] |

Leitura:
- Pela medida por frase (`sim_ref`), R > 1 sempre baixa a cópia — esperado, o código deixa de ser o daquela frase; essa medida não serve para avaliar R.
- Pela medida por locutor, o único ganho com IC acima de 0 é o das **4 vozes novas** nos 4 treinos plain (+13 pontos, IC [+1; +35], 3 locutores sobem, 0 descem). Nas **38 externas o efeito é nulo ou negativo** (−2,5 pontos, IC [−6,4; +0,8]; 14 sobem, 18 descem). Nas treinadas, negativo sem significância.
- `sim_spk` sobe ~+0,03 nas 4 novas/externas plain, mas `sim_default` sobe junto (+0,02): o código médio vai para o centro, e a margem por locutor `ms` quase não muda (+0,002 a +0,024).
- O efeito consistente é **WER menor** nas externas: −1,8 pontos com 38 vozes (IC [−2,8; −0,9]; p = 0,009 por locutor); nos outros grupos o IC contém 0.
- Conclusão: com 38 vozes, R = 4 é um estabilizador de inteligibilidade, não uma melhoria de cópia; o ganho nas 4 novas é compatível com acaso de 4 locutores (IC inferior +0,01).

---

## 8. Poder por simulação e desenho recomendado

### 8.1 Métodos

Modelo gerador por amostra na escala da margem (componentes medidos nas seções 2.2 e 2.2-sementes):

m = μ(N) + u_s + v_{treino,s} + w_{treino,frase} + z_frase + x_{s,frase} + ε + δ_semente; copiado = m > 0,

com DP: locutor 0,0764, locutor × frase 0,0143, frase 0,0092, treino × frase 0,0062, residual 0,0314 (ANOVA das 38 externas), efeito principal de semente 0,0127 (pares de sementes), treino × locutor 0,046 ("conservador": inclui a interação locutor × N real das 38 externas, tratada como ruído independente por treino) ou 0,0148 ("só semente": a interação semente × locutor medida). μ(N) é calibrado para que a fração não copiada da população siga cada hipótese, todas ajustadas aos 3 pontos medidos nas 38 vozes:

| Hipótese | q(8) | q(32) | q(128) | q(512) | q(2048) |
|---|---|---|---|---|---|
| H0,26: 0,744 N^−0,26 (sem piso) | 0,433 | 0,302 | 0,211 | 0,147 | 0,102 |
| H0,5: 0,145 + 0,828/√N | 0,437 | 0,291 | 0,218 | 0,181 | 0,163 |
| Hpiso: 0,219 + 1,762/N | 0,439 | 0,274 | 0,233 | 0,223 | 0,220 |

Em cada experimento simulado as vozes de avaliação são as mesmas em todos os N (pareamento, como na avaliação `-wide`). Análise: q̂ por N (média das sementes); ajuste das três formas de 2 parâmetros (c N^−α com α livre, a + A/√N, a + A/N); escolhe-se a de menor SSE. Também: dispersão de α̂ (forma c N^−α) e teste de "ainda há ganho de 512 para 2.048" (diferença pareada por locutor, IC 95% = bootstrap de locutores + variância de semente suposta conhecida). 400 simulações por célula (200 com 200 vozes); "P certa" = probabilidade de escolher a hipótese verdadeira (acaso = 0,33). A estimativa por mistura probit da margem (seção 3.1) deu os mesmos resultados que a fração binária (±0,03) e não está repetida.

### 8.2 Resultados (cenário conservador; "só semente" dá números iguais ±0,05)

| Vozes × frases | Sementes por N | Grade de N | Treinos | P certa: H0,26 / H0,5 / Hpiso | α̂ sob H0,26 (2,5–97,5%) | P(detectar ganho 512→2048) sob H0,26 / Hpiso |
|---|---|---|---|---|---|---|
| 38 × 25 (**atual**) | 1 | 8, 32, 128 | 3 | 0,44 / 0,16 / 0,62 | 0,06–0,53 | — |
| 4 × 25 | 1 | 8 … 2048 (5) | 5 | 0,18 / 0,40 / 0,58 | 0,04–1,18 | 0,03 / 0,04 |
| 4 × 25 | 3 | 5 pontos | 15 | 0,35 / 0,33 / 0,64 | 0,08–1,00 | 0,08 / 0,04 |
| 38 × 25 | 1 | 5 pontos | 5 | 0,57 / 0,37 / 0,62 | 0,17–0,47 | 0,12 / 0,03 |
| 38 × 25 | 2 | 5 pontos | 10 | 0,67 / 0,50 / 0,70 | 0,17–0,40 | 0,19 / 0,02 |
| 38 × 25 | 3 | 5 pontos | 15 | 0,74 / 0,57 / 0,72 | 0,18–0,37 | 0,29 / 0,04 |
| 38 × 25 | 5 | 5 pontos | 25 | 0,81 / 0,70 / 0,78 | 0,19–0,36 | 0,46 / 0,04 |
| 100 × 10 | 1 | 5 pontos | 5 | 0,63 / 0,41 / 0,68 | 0,18–0,41 | 0,16 / 0,04 |
| 100 × 25 | 1 | 5 pontos | 5 | 0,63 / 0,41 / 0,67 | 0,17–0,39 | 0,17 / 0,03 |
| 100 × 25 | 3 | 5 pontos | 15 | 0,78 / 0,63 / 0,80 | 0,20–0,34 | 0,38 / 0,03 |
| 100 × 25 | 5 | 5 pontos | 25 | 0,80 / 0,78 / 0,84 | 0,21–0,33 | 0,57 / 0,03 |
| 200 × 10 | 2 | 5 pontos | 10 | 0,73 / 0,55 / 0,78 | 0,20–0,32 | 0,31 / 0,03 |
| 200 × 25 | 3 | 5 pontos | 15 | 0,78 / 0,62 / 0,77 | 0,21–0,33 | 0,51 / 0,04 |
| 100 × 25 | 1 | 9 pontos (8, 16, …, 2048) | 9 | 0,67 / 0,53 / 0,70 | 0,18–0,36 | 0,19 / 0,04 |
| 100 × 25 | 2 | 9 pontos | 18 | 0,78 / 0,67 / 0,75 | 0,20–0,35 | 0,27 / 0,03 |
| 100 × 25 | 3 | 9 pontos | 27 | 0,78 / 0,74 / 0,84 | 0,21–0,34 | 0,39 / 0,02 |

Leitura:
- **O desenho atual (38 vozes, 1 semente, N ≤ 128) está perto do acaso** para escolher a lei (0,16–0,62), e α̂ varia de 0,06 a 0,53 mesmo quando a verdade é 0,26: o 0,26 observado é compatível com qualquer das três hipóteses.
- **4 vozes não servem** para isto, nem com 3 sementes e N até 2.048.
- **Depois de ~40–100 vozes, o gargalo é a semente.** De 38 para 200 vozes (3 sementes) a taxa certa sobe 0,74 → 0,78 em H0,26; de 1 para 5 sementes (38 vozes) sobe 0,57 → 0,81. 10 frases por voz bastam (100 × 10 ≈ 100 × 25), como previsto pelo DP de frase quase nulo.
- **Distinguir 0,26 de 0,5 (H0,26 contra H0,5) é o mais difícil**: as duas diferem em 3,4 pontos em N = 512 e 6 pontos em N = 2.048. Com 5 sementes por N e 38–100 vozes chega-se a 70–80% de acerto; abaixo de 3 sementes, não.
- **Testar piso** pela diferença 512 → 2.048 tem poder baixo (12–57%) mesmo com 5 sementes, com erro tipo I correto (2–5%); ajustar a curva inteira é mais informativo que esse teste.
- Uma grade densa (9 valores de N, 1 semente) rende o mesmo que 5 valores com 2 sementes, pelo mesmo número de treinos: **o que conta é o total de treinos**, desde que N vá até ~2.000.

### 8.3 Desenho recomendado para as próximas rodadas na GPU

1. **Conjunto de avaliação fixo e pareado**: 60–100 vozes nunca vistas (LibriTTS test-clean + dev-clean/other), **10 frases** cada, as mesmas em todos os treinos; estratificadas por f0 e por distância à voz padrão. Avaliar custa pouco perto de treinar.
2. **Sementes**: ≥ 3 por valor de N, idealmente 5 (é o que mais rende). Grade N ∈ {8, 32, 128, 512, 2048} com K fixo (32); 15–25 treinos. Se o orçamento for de ~10 treinos, preferir 5 N × 2 sementes ou 9 N × 1 semente a menos N com mais vozes.
3. **Re-sintetizar 2× cada checkpoint** (barato) para separar ruído de síntese de ruído de treino (hoje ~40% da variância "de semente" do binário é síntese).
4. **Salvar a força 0 para cada voz de avaliação** (uma vez por voz, com qualquer checkpoint): dá c_s = semelhança entre a voz padrão e a referência, e permite medidas que não dependem do limiar `sim_ref > sim_default` (tarefa 6).
5. **Medida primária contínua** (`sim_ref` / `sim_spk` e margem por locutor), com a fração binária derivada via mistura probit; relatar ambas. Para K, um segundo eixo pequeno (K ∈ {8, 128} em N = 128 e 512, 2 sementes) basta para checar b/a ≈ 0,5.
6. **Registrar antes** a regra de decisão: ajuste das três formas, critério por SSE/AICc e CV deixando um N de fora, com bootstrap de vozes + sementes; e declarar como "indistinguível" qualquer resultado em que a hipótese vencedora tenha < 80% de recuperação na simulação correspondente.

---

## O que os dados sustentam e o que não sustentam

**Sustentam (com a força indicada):**
1. **Vozes treinadas dependem de K, não de N** — na margem contínua, a + c ln K explica tudo; o coeficiente de ln N é ≈ 0. Forte.
2. **Vozes novas dependem de N** — margem média: só-K e constante são piores com certeza (bootstrap P ≥ 0,99); a margem sobe de −0,11 (N = 2) a +0,03–0,06 (N = 128–200). Forte, mas a maior parte da evidência vem de N = 2 versus o resto.
3. **Nas 38 vozes externas, mais locutores ajudam de 8 a 128** (cópia 57 → 70 → 79%; margem −0,004 → 0,048 → 0,078), e o ganho de 32 → 128 é menor que o de 8 → 32 em todas as medidas, inclusive nas que não usam a voz padrão. Moderado: uma semente; o DP de semente da fração (~0,05) é da ordem do ganho 32 → 128 (0,09).
4. **A cópia é quase tudo-ou-nada por voz** (ICC latente 0,83–0,94; 2/3 das vozes externas em ≤ 10% ou ≥ 90%), e a dificuldade é estável entre treinos (ρ ≈ 0,75). Forte.
5. **A fração binária é uma função das margens por locutor** (mistura probit, RMSE 0,018). Forte. A margem é ~1,3–2× mais precisa que o binário entre sementes, e muito mais estável à re-síntese; não é uma solução para o ruído de locutor e de semente.
6. **A frase importa pouco** (DP 0,007 na margem); locutor e semente dominam.
7. **R = 4 não melhora a cópia nas 38 vozes**, mas reduz o WER em ~1,8 pontos.
8. **Colapso**: para as treinadas, x ≈ K^0,46 com expoente de N ≈ 0 (±0,07); para as novas entre N = 8 e 200, K entra com cerca de metade do peso de N (b/a ≈ 0,5, ICs largos; depende de excluir N = 2).
9. **O gargalo de precisão é a semente**, não o número de vozes, a partir de ~40 vozes: a simulação mostra que separar 0,26 de 0,5 exige ≥ 3–5 sementes por N com N até ~2.000.

**Não sustentam:**
1. **O expoente 0,26.** Ele é exatamente o que dois parâmetros fazem passar por três pontos de uma semente. A razão de ganhos tem IC (−0,7; 1,6) só com reamostragem de locutores e (−1,9; 2,3) com semente; medidas sem a voz padrão dão 0,41–0,55. Afirmar "≈ 0,26, d ≈ 4" ou extrapolar "82% com 247 / 88% com 1.000 locutores" não tem apoio estatístico.
2. **Qualquer escolha entre L1 (piso em 1/N), L2 (raízes), L3 (potência) e L4 (probit em ln N).** Na grade de 4 vozes, a preferência por L2 vem do ponto N = 2; sem ele, todas (menos L1, marginalmente) empatam. Nas 38 vozes, 3 pontos e 1 g.l. Para N = 2.456 as leis preveem de 61% a 93% (4 vozes) e de 78% a 95% (38 vozes).
3. **A existência ou ausência de piso.** Não identificável com N ≤ 200.
4. **"Não há teto" ou "mais locutores rendem cada vez menos" como afirmação quantitativa.** Ambas as leituras cabem no ruído.
5. **Efeitos das variantes** (`-swap`, `-contrast`, `-spk`, `-spkdrop`, `-aug`): uma semente cada; todas dentro de ~2 DP de semente da linha de base.
6. **Que as vozes externas sejam "mais fáceis de imitar"** que as novas da grade: a diferença de nível vem em boa parte da distância à voz padrão, que o critério binário premia.
7. **A interpretação da falha de vozes agudas como dificuldade do decodificador**: pela semelhança com a referência (`sim_ref`, `sim_spk`), vozes agudas não são reproduzidas pior; elas perdem no critério binário porque a saída também se parece com a voz padrão feminina.
