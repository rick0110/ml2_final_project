# 03 — Análise teórico-informacional dos latentes salvos

Escopo: só CPU, só dados já salvos (`coverage2/<run>/step_2500/`: `latents.npz`, `metrics.jsonl`, `strength_1.0*/results.csv`). Nenhum decodificador foi guardado, então tudo aqui é sobre o **codificador** (o que o vetor de estilo de 32 dimensões carrega) e sobre a **associação** entre a geometria desse vetor e o resultado da cópia. Nada aqui mede o que o decodificador faz com o vetor.

## 0. Resumo em números

| Grandeza | Valor (runs de referência) | Leitura |
|---|---|---|
| Taxa KL por frase | 64–75 bits (base), 41–61 bits (variantes `-spkdrop`, `-swap`, `-contrast`); 32 de 32 dimensões ativas, 1,35–4,0 bits por dimensão | ~93% da taxa é o "termo de precisão" (σ posterior ≈ 0,16), não informação |
| I(x; z) gaussiano (agregado) | 17–29 bits; cresce ≈ 1,6 bit por duplicação de N e de K | a taxa excede a informação em ~45 bits |
| Informação sobre a voz, modelo gaussiano corrigido (μ) | 17,2–17,7 bits para N = 128–200, K = 32 (± 0,3) | estável de N = 32 a 200: capacidade do codificador, não do número de vozes |
| Mesma coisa no canal z = μ + σε | 9,8–10,2 bits (N = 128–200) | o que o decodificador vê no treino |
| Cota inferior discreta I(μ; identidade), classificador LOO | 5,1–5,3 de 6,9 bits (N = 128), 4,6 de 5,0 (N = 32), 2,9 de 3,0 (N = 8) | permutação: ≤ −0,4 bits |
| Direções com informação de voz | PR(Σ_b) ≈ 5,8; PR branqueada ≈ 9,7; 6–8 direções com SNR > 1 em z; 90% dos bits em 11–12 direções; espectro ~ k^−2 | |
| Dimensão intrínseca das vozes (médias sem ruído, N ≥ 124) | TwoNN 9,7–10,6; Levina–Bickel 7,0–7,5; expoente de cobertura γ = 0,12–0,14 ⇒ d ≈ 7–8 | abaixo do nulo gaussiano de mesmo espectro (TwoNN 13–14, γ ≈ 0,10) |
| Expoente 0,26 | cobertura linear exigiria d ≈ 3,8 (rejeitado pelo medido); erro quadrático na distância exige d ≈ 7,7 (compatível) | |
| Atrator da voz padrão (38 externas) | ρ(cópia, d_LJ) = 0,34 / 0,43 / 0,63 para N = 8 / 32 / 128; ρ(cópia, d_NN) = −0,48 / −0,22 / −0,12 | d_LJ vence d_NN em N = 128 (P = 0,99); d_NN vence em N = 8 |
| Transição da cópia | d50 ≈ 15,5 unidades branqueadas para N = 32 e 128, 19,5 para N = 8 (≈ distância centro LJ–centro LibriTTS), largura ≈ 2 (ruído de uma frase = 1) | |

## 1. Dados e convenções

- `latents.npz`: μ e log σ² (n × 32) de ~1.500 frases de **treino** por run (subamostra: LJ + LibriTTS), mais as referências (`source = refs`). Nas runs `-wide` as referências externas são **uma frase por locutor** (38 locutores do LibriTTS test-clean). `results.csv`: 25 frases sintetizadas por referência externa; "copiou" = `sim_ref > sim_default` (critério do projeto). Taxas de não-cópia nas externas: 0,43 / 0,30 / 0,21 (N = 8 / 32 / 128), confere com o relatório anterior.
- Bits = log₂. Nas tabelas geradas por script o separador decimal é o ponto. "Branqueado" = coordenadas W^−1/2 μ, com W a covariância dentro do locutor (LibriTTS, treino). Nessa métrica, uma frase isolada se afasta da média do seu locutor com desvio 1 por direção (‖e‖ ≈ √32 = 5,7).
- Σ_b = covariância entre locutores, estimada sem viés por componentes de variância: Σ_b = (MSB − MSW)/n₀ (ANOVA multivariada desbalanceada).
- Runs principais: grade base (16 runs) e `-wide` (3). `n8_k32_s1-wide12` ignorado. Variantes de treino aparecem só na seção 3.4.
- Incerteza: bootstrap sobre locutores (2.000 réplicas nas correlações, 1.000 no modelo agrupado), bootstrap paramétrico (100 simulações) para viés dos estimadores gaussianos, subamostragem de 80% dos locutores para estimadores de vizinhos (o bootstrap com repetição quebra TwoNN).

## 2. Tarefa 1 — Taxa (KL) por dimensão e total

Método: KL(q(z|x) ‖ N(0, I)) por dimensão, em bits, média sobre as frases de treino; decomposta em termo de média ½μ²/ln2 e termo de precisão ½(σ² − ln σ² − 1)/ln2. I(x; z) "gaussiano": ½ log₂ det(Cov μ + diag E σ²) − ½ log₂ det diag E σ² (aproximação do posterior agregado por uma gaussiana; não é cota rigorosa em nenhum sentido, porque o agregado não é gaussiano e usei E σ² no lugar de E log σ²). O estimador Monte Carlo sobre a mistura empírica satura em log₂ n = 10,55 e não serve (coluna omitida; está em `out/rate.json`).

| run | N | K | taxa total (bits/frase) | taxa LJ | taxa Libri | termo de precisão | termo de média | bits/dim mín–máx | I(x;z) gaussiano |
|---|---|---|---|---|---|---|---|---|---|
| n2_k128_s1 | 2 | 128 | 64.4 | 64.9 | 61.0 | 61.8 | 2.7 | 1.35–3.32 | 18.4 |
| n2_k128_s2 | 2 | 128 | 65.5 | 65.7 | 64.0 | 62.8 | 2.7 | 1.42–3.34 | 18.7 |
| n8_k8_s1 | 8 | 8 | 64.3 | 64.2 | 66.5 | 62.3 | 1.9 | 1.47–3.32 | 17.5 |
| n8_k8_s2 | 8 | 8 | 67.8 | 67.8 | 68.5 | 65.8 | 2.0 | 1.46–3.35 | 18.6 |
| n8_k32_s1 | 8 | 32 | 66.1 | 65.7 | 68.7 | 63.4 | 2.7 | 1.42–3.36 | 20.0 |
| n8_k128_s1 | 8 | 128 | 68.6 | 70.0 | 66.0 | 64.6 | 4.0 | 1.44–3.59 | 22.8 |
| n8_k128_s2 | 8 | 128 | 70.2 | 71.5 | 67.7 | 66.2 | 3.9 | 1.41–3.68 | 23.5 |
| n32_k8_s1 | 32 | 8 | 66.9 | 66.7 | 68.2 | 64.4 | 2.5 | 1.40–3.32 | 20.2 |
| n32_k8_s2 | 32 | 8 | 68.4 | 68.2 | 70.0 | 65.8 | 2.6 | 1.59–3.45 | 20.3 |
| n32_k32_s1 | 32 | 32 | 71.3 | 71.9 | 70.0 | 67.6 | 3.6 | 1.51–3.69 | 24.8 |
| n32_k128_s1 | 32 | 128 | 71.6 | 77.6 | 68.6 | 66.7 | 4.8 | 1.51–3.81 | 27.6 |
| n32_k128_s2 | 32 | 128 | 74.5 | 79.3 | 72.1 | 69.8 | 4.7 | 1.56–3.97 | 28.9 |
| n128_k8_s1 | 128 | 8 | 70.2 | 71.0 | 68.6 | 66.6 | 3.5 | 1.46–3.57 | 23.1 |
| n128_k32_s1 | 128 | 32 | 72.4 | 77.3 | 70.0 | 68.0 | 4.4 | 1.52–3.87 | 26.8 |
| n128_k32_s2 | 128 | 32 | 72.6 | 77.7 | 70.1 | 67.3 | 5.3 | 1.55–3.90 | 26.2 |
| n200_k32_s1 | 200 | 32 | 71.6 | 79.1 | 69.1 | 67.3 | 4.3 | 1.55–3.75 | 25.6 |
| n8_k8_s1-aug | 8 | 8 | 68.8 | 68.2 | 69.4 | 65.8 | 3.0 | 1.40–3.22 | 20.1 |
| n8_k128_s1-aug | 8 | 128 | 71.7 | 74.4 | 70.4 | 67.2 | 4.5 | 1.54–3.61 | 25.0 |
| n8_k8_s1-contrast | 8 | 8 | 54.0 | 53.9 | 57.8 | 47.8 | 6.2 | 1.35–2.16 | 3.7 |
| n8_k8_s2-contrast | 8 | 8 | 48.9 | 48.5 | 63.3 | 43.7 | 5.2 | 1.19–1.98 | 3.3 |
| n8_k128_s1-contrast | 8 | 128 | 54.9 | 54.6 | 55.7 | 51.5 | 3.4 | 1.31–2.24 | 14.5 |
| n2_k128_s1-spk | 2 | 128 | 66.6 | 66.8 | 64.9 | 64.1 | 2.4 | 1.53–3.52 | 18.4 |
| n8_k8_s1-spk | 8 | 8 | 63.5 | 63.5 | 63.0 | 61.8 | 1.8 | 1.43–3.35 | 16.5 |
| n8_k128_s1-spk | 8 | 128 | 68.4 | 69.8 | 65.5 | 64.9 | 3.5 | 1.45–3.57 | 21.2 |
| n8_k128_s1-spkdrop | 8 | 128 | 40.7 | 42.1 | 37.9 | 39.4 | 1.3 | 0.91–2.46 | 7.0 |
| n32_k8_s1-spkdrop | 32 | 8 | 45.6 | 45.5 | 46.2 | 44.4 | 1.1 | 1.09–2.48 | 8.2 |
| n32_k128_s1-spkdrop | 32 | 128 | 42.5 | 46.7 | 40.5 | 40.4 | 2.1 | 0.85–2.49 | 9.9 |
| n8_k8_s1-swap | 8 | 8 | 49.9 | 49.9 | 51.1 | 49.5 | 0.5 | 1.25–2.13 | 3.3 |
| n8_k8_s2-swap | 8 | 8 | 50.0 | 49.9 | 54.2 | 49.5 | 0.5 | 1.22–1.99 | 3.2 |
| n8_k128_s1-swap | 8 | 128 | 61.0 | 60.5 | 62.0 | 57.8 | 3.2 | 1.33–2.58 | 14.7 |
| n8_k32_s1-wide | 8 | 32 | 66.6 | 66.5 | 66.8 | 63.9 | 2.7 | 1.44–3.48 | 20.1 |
| n32_k32_s1-wide | 32 | 32 | 72.4 | 74.3 | 68.9 | 68.7 | 3.7 | 1.54–3.81 | 24.8 |
| n128_k32_s1-wide | 128 | 32 | 73.2 | 78.6 | 70.6 | 69.0 | 4.2 | 1.59–3.77 | 27.0 |

Regressão nas 16 runs base (erro-padrão entre parênteses):

```
taxa total  = 58,0 + 1,39 (0,20)·log₂N + 0,97 (0,24)·log₂K    RMSE 1,5 bit
taxa Libri  = 64,4 + 0,92 (0,26)·log₂N − 0,09 (0,31)·log₂K    RMSE 1,9
taxa LJ     = 50,0 + 2,50 (0,25)·log₂N + 1,99 (0,30)·log₂K    RMSE 1,9
I(x;z) gauss=  7,8 + 1,63 (0,17)·log₂N + 1,51 (0,20)·log₂K    RMSE 1,3
```

Leitura:

1. **A taxa quase não depende dos dados** (64 → 75 bits de N = 2 a 200) e quase toda ela (62–70 bits) é o termo de precisão: com σ ≈ 0,16 em cada dimensão, cada uma paga ≈ 2 bits por "ser precisa", carregue ou não algo que varie entre frases. O termo de média é só 2–5 bits. As 32 dimensões estão todas ativas (mínimo 1,35 bit/dim; razão de participação do perfil de taxa ≈ 31 de 32).
2. A informação efetiva I(x; z) é 17–29 bits: **~45 bits da taxa são desencontro entre o posterior agregado e o prior** (o agregado ocupa uma bola de desvio ≈ 0,4 dentro de um prior de desvio 1), não informação sobre a frase.
3. O peso do KL ainda estava subindo no passo 2.500 (0,0225 → 0,0238 nos últimos 25 passos do `metrics.jsonl`): os modelos não estão num ponto de equilíbrio taxa–distorção. O KL do log de treino (≈ 1,6–1,9 nat por dimensão ≈ 74–86 bits) é um pouco acima do medido nos latentes salvos (64–75 bits); a diferença é plausivelmente de modo treino/avaliação e não foi investigada.
4. A taxa das frases LJ cresce com N e K (2,5 e 2,0 bits por duplicação); a das LibriTTS cresce pouco com N e nada com K. O mecanismo do aumento da taxa LJ não foi decomposto; a informação por frase LibriTTS (taxa Libri) quase não muda.

## 3. Tarefa 2 — Informação sobre o locutor

### 3.1 Estimadores

- **Gaussiano entre/dentro** (o "primeiro passe"): I = ½ log₂ det(T)/det(W) com T e W não enviesados ("plug-in"), e a versão por componentes de variância I = ½ Σ log₂(1 + λᵢ), λᵢ autovalores generalizados de (Σ_b, W) com parte negativa truncada ("corr."). Versão canal: W + diag E σ² (o que chega ao decodificador no treino, z amostrado).
- **Correção de viés por bootstrap paramétrico**: simulo 100 conjuntos do modelo ajustado (mesmo número de locutores e de frases por locutor), reestimo e subtraio o viés médio (I_corr = 2·I − média das simulações). Também estimei Σ_b pela covariância cruzada entre metades disjuntas das frases de cada locutor (sem ruído por construção): dá o mesmo valor que a versão por componentes de variância com diferença ≤ 0,5 bit (`out/speaker2.json`, `I_split`).
- **Permutação**: rótulos de locutor embaralhados (200 vezes no gaussiano, 10 no classificador).
- **Cota inferior discreta**: I(μ; S) ≥ H(S) − CE, com CE a entropia cruzada (bits) de um classificador gaussiano de covariância compartilhada em validação **leave-one-out exata** (preditiva com W(1 + 1/n_s)), e de uma regressão logística em 5 dobras estratificadas. Fano: I ≥ H(S) − h(P_e) − P_e log₂(S − 1).

### 3.2 Resultados

| run | S | frases/loc. | log2 S | H(S) | Gauss plug-in | Gauss corr. (mu) | Gauss corr. (z) | perm. plug/vc | acc LDA-LOO | I_CE (LDA) | I_Fano | I_CE (logreg 5-fold) | I_CE perm. | LJ×Libri I/H | dist. LJ–Libri (dp) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n2_k128_s1 | 2 | 90.5 | 1.00 | 1.00 | 3.0 | 4.4 ± 1.2 | 3.4 | 0.01 / 0.25 | 1.00 | 1.00 | 1.00 | 0.99 | -0.21 | 0.53/0.53 | 7.3 |
| n2_k128_s2 | 2 | 90.5 | 1.00 | 1.00 | 3.1 | 4.4 ± 1.2 | 3.4 | 0.02 / 0.26 | 1.00 | 1.00 | 1.00 | 0.99 | -0.22 | 0.53/0.53 | 7.9 |
| n8_k32_s1 | 8 | 22.6 | 3.00 | 3.00 | 11.7 | 14.9 ± 0.8 | 10.0 | 0.09 / 0.91 | 0.98 | 2.69 | 2.83 | 2.89 | -1.45 | 0.53/0.53 | 6.1 |
| n8_k128_s1 | 8 | 64.1 | 3.00 | 2.99 | 12.5 | 16.0 ± 1.0 | 10.9 | 0.01 / 0.29 | 0.99 | 2.94 | 2.89 | 2.95 | -0.37 | 0.93/0.93 | 8.9 |
| n8_k128_s2 | 8 | 64.1 | 3.00 | 2.99 | 12.5 | 15.7 ± 1.0 | 10.9 | 0.01 / 0.30 | 0.99 | 2.92 | 2.90 | 2.96 | -0.40 | 0.93/0.93 | 9.1 |
| n32_k8_s1 | 32 | 5.7 | 5.00 | 4.97 | 16.7 | 17.9 ± 0.6 | 10.0 | 0.51 / 2.33 | 0.91 | 3.66 | 4.05 | 3.96 | -5.96 | 0.53/0.53 | 4.8 |
| n32_k8_s2 | 32 | 5.7 | 5.00 | 4.97 | 15.9 | 17.4 ± 0.6 | 9.7 | 0.51 / 2.33 | 0.89 | 3.70 | 3.92 | 3.85 | -5.97 | 0.53/0.53 | 4.7 |
| n32_k32_s1 | 32 | 16.0 | 5.00 | 4.98 | 17.0 | 19.2 ± 0.6 | 11.3 | 0.05 / 0.67 | 0.95 | 4.31 | 4.48 | 4.55 | -1.70 | 0.92/0.93 | 6.0 |
| n32_k128_s1 | 32 | 31.4 | 5.00 | 4.99 | 19.1 | 21.8 ± 0.6 | 12.3 | 0.01 / 0.33 | 0.97 | 4.59 | 4.63 | 4.74 | -0.84 | 0.91/0.91 | 5.5 |
| n32_k128_s2 | 32 | 31.4 | 5.00 | 4.99 | 19.4 | 22.0 ± 0.5 | 12.8 | 0.01 / 0.33 | 0.96 | 4.59 | 4.59 | 4.74 | -0.82 | 0.91/0.91 | 5.7 |
| n128_k8_s1 | 124 | 4.1 | 6.95 | 6.90 | 15.5 | 15.6 ± 0.4 | 9.0 | 0.25 / 1.58 | 0.69 | 3.73 | 3.83 | 4.15 | -6.73 | 0.92/0.93 | 5.7 |
| n128_k32_s1 | 127 | 7.9 | 6.99 | 6.92 | 17.2 | 17.7 ± 0.3 | 10.2 | 0.06 / 0.72 | 0.83 | 5.31 | 5.05 | 5.08 | -3.31 | 0.91/0.91 | 4.5 |
| n128_k32_s2 | 127 | 7.9 | 6.99 | 6.92 | 16.7 | 17.2 ± 0.3 | 9.8 | 0.05 / 0.71 | 0.80 | 5.11 | 4.81 | 4.98 | -3.32 | 0.91/0.91 | 4.4 |
| n200_k32_s1 | 197 | 5.7 | 7.62 | 7.52 | 17.1 | 17.4 ± 0.2 | 9.8 | 0.08 / 0.82 | 0.74 | 4.75 | 4.72 | 5.04 | -4.66 | 0.81/0.81 | 3.7 |
| n8_k32_s1-wide | 8 | 22.6 | 3.00 | 3.00 | 11.3 | 14.4 ± 0.9 | 9.4 | 0.10 / 0.91 | 0.98 | 2.64 | 2.78 | 2.88 | -1.36 | 0.53/0.53 | 6.0 |
| n32_k32_s1-wide | 32 | 16.0 | 5.00 | 4.98 | 17.0 | 19.1 ± 0.6 | 11.1 | 0.05 / 0.67 | 0.95 | 4.29 | 4.40 | 4.55 | -1.73 | 0.92/0.93 | 6.1 |
| n128_k32_s1-wide | 127 | 7.9 | 6.99 | 6.92 | 17.0 | 17.5 ± 0.3 | 10.2 | 0.05 / 0.71 | 0.83 | 5.26 | 5.08 | 5.07 | -3.32 | 0.91/0.91 | 4.6 |

(`dist. LJ–Libri` = √(λ₁ − 1), λ₁ maior autovalor generalizado total/dentro no problema binário LJ × LibriTTS; equivale a √(π(1−π)) vezes a distância de Mahalanobis.)

O que dá para afirmar:

1. **A estimativa gaussiana não estava enviesada para cima** como se temia. O nulo por permutação dá 0,01–0,5 bit (plug-in) e o viés paramétrico é **negativo** (−0,2 a −2,6 bits; maior com poucos locutores, porque Σ_b de S locutores tem posto ≤ S − 1). O problema do primeiro passe é de **interpretação**, não de viés: o número gaussiano (12–22 bits) passa de H(S) = log₂ S, logo não é I(z; identidade entre os S treinados). Ele é a informação sobre um **vetor de voz contínuo** (população infinita de vozes) sob o modelo gaussiano, μ = v + e. Com N = 8 ele é limitado pelo posto (7 direções), por isso cresce de N = 8 a 32 e para: **17,2–17,7 bits em μ e 9,8–10,2 bits em z para N = 128 e 200**, contra 17,4–22 bits com N = 32. O codificador não ganha capacidade de descrever voz com mais locutores; o que cresce com K é a precisão (menos ruído dentro do locutor).
2. **Cota discreta**: o classificador recupera quase toda a entropia com poucos locutores (2,9 de 3,0 bits com N = 8; 4,3–4,6 de 5,0 com N = 32 e K ≥ 32) e perde mais com muitos (5,1–5,3 de 6,9 com N = 128, K = 32; 4,7–5,0 de 7,5 com N = 200; 3,7 de 6,9 com N = 128, K = 8). O déficit H − I_CE cresce com S (≈ 0,1–0,4 bit com S = 8; 0,4–0,7 com S = 32 e K ≥ 32; 1,6–1,8 com S = 127; 2,8 com S = 197). Em termos de I_CE/log₂ S, para K ≥ 32: 0,90–0,98 (S = 8), 0,86–0,92 (S = 32), 0,73–0,76 (S = 127), 0,62 (S = 197). Uma frase isolada identifica o locutor entre 127 com acerto de 83% (LDA) / 72% (logística).
3. **Voz padrão × LibriTTS**: separação perfeita (acurácia 1,00 em todas as runs; I = H do rótulo binário, 0,53–0,93 bit conforme a proporção). A voz padrão é um aglomerado isolado, à distância ≈ 15–17,5 unidades branqueadas do centro das vozes LibriTTS — mais que a distância típica entre duas vozes LibriTTS (≈ 11 = √(2 tr Σ_b) branqueado, sem ruído) e três vezes a distância ao vizinho mais próximo entre 127 (≈ 5).

### 3.3 Quantas direções carregam a voz

| run | S | PR(Σb) | PR(W^-1/2 Σb W^-1/2) | PR(SNR) | dir. SNR>1 (mu) | dir. SNR>1 (z) | k 50% bits (z) | k 90% (z) | k 99% (z) | k 90% (mu) | α lei de potência (Σb, 16 dir.) | ganho R=4 previsto (mu/z, bits) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n8_k32_s1 | 8 | 2.8 | — | 4.0 | 7 | 6 | 3 | 5 | 7 | 6 | 2.37 [1.40, 2.73] | 6.5 / 2.6 |
| n8_k128_s1 | 8 | 3.8 | — | 4.6 | 7 | 6 | 3 | 6 | 7 | 6 | 1.76 [1.06, 2.12] | 6.6 / 2.8 |
| n8_k128_s2 | 8 | 3.7 | — | 4.3 | 7 | 7 | 3 | 6 | 7 | 6 | 1.82 [1.25, 2.07] | 6.6 / 2.8 |
| n32_k8_s1 | 32 | 5.0 | 7.7 | 7.6 | 12 | 7 | 4 | 9 | 16 | 14 | 2.60 [2.80, 3.43] | 14.7 / 4.1 |
| n32_k8_s2 | 32 | 5.0 | 7.5 | 7.4 | 13 | 6 | 3 | 10 | 16 | 14 | 2.61 [2.78, 3.60] | 14.5 / 4.0 |
| n32_k32_s1 | 32 | 5.6 | 8.7 | 8.7 | 13 | 7 | 4 | 11 | 18 | 15 | 2.37 [2.55, 3.06] | 15.5 / 4.7 |
| n32_k128_s1 | 32 | 6.1 | 9.7 | 9.7 | 15 | 8 | 4 | 11 | 19 | 16 | 2.15 [2.27, 2.78] | 16.9 / 4.9 |
| n32_k128_s2 | 32 | 6.1 | 9.7 | 9.7 | 14 | 8 | 4 | 12 | 19 | 16 | 2.14 [2.28, 2.78] | 16.9 / 5.2 |
| n128_k8_s1 | 124 | 5.2 | 8.4 | 8.4 | 13 | 6 | 4 | 11 | 19 | 18 | 2.24 [2.26, 2.43] | 16.3 / 4.4 |
| n128_k32_s1 | 127 | 5.8 | 9.8 | 9.8 | 14 | 7 | 4 | 12 | 21 | 19 | 2.03 [2.05, 2.20] | 17.8 / 4.9 |
| n128_k32_s2 | 127 | 5.7 | 9.8 | 9.8 | 13 | 6 | 4 | 11 | 21 | 19 | 2.10 [2.12, 2.29] | 17.4 / 4.8 |
| n200_k32_s1 | 197 | 5.8 | 8.9 | 8.9 | 14 | 7 | 4 | 11 | 21 | 20 | 2.06 [2.06, 2.20] | 17.8 / 4.6 |
| n8_k32_s1-wide | 8 | 3.0 | — | 4.0 | 7 | 6 | 3 | 5 | 7 | 6 | 2.46 [1.46, 2.77] | 6.5 / 2.5 |
| n32_k32_s1-wide | 32 | 5.6 | 8.7 | 8.7 | 13 | 7 | 4 | 11 | 18 | 15 | 2.34 [2.50, 3.09] | 15.4 / 4.6 |
| n128_k32_s1-wide | 127 | 5.9 | 9.6 | 9.6 | 13 | 8 | 4 | 12 | 22 | 19 | 1.99 [2.01, 2.16] | 17.5 / 5.0 |

- O espectro de Σ_b é concentrado: razão de participação ≈ 5,8 (bruta) e ≈ 9,7 (branqueada, isto é, relativa ao ruído dentro do locutor); decaimento em lei de potência λ_k ∝ k^−α com α ≈ 2,0–2,1 nas 16 primeiras direções para N ≥ 128 (≈ 2,1–2,6 para N = 32). O IC por bootstrap de α vem deslocado para cima em relação à estimativa pontual (o bootstrap com locutores repetidos alarga o espectro); trate α como ≈ 2 ± 0,3.
- Em z (com o ruído do posterior), **6–8 direções têm SNR > 1**; 50% dos bits de voz estão em 4 direções, 90% em 11–12, 99% em ~21. Em μ (sem ruído do posterior) são 13–15 direções com SNR > 1.
- Bits acumulados em z (N = 128, `-wide`): 1,6 (1 dir.), 3,0 (2), 5,3 (4), 6,9 (6), 7,9 (8), 9,2 (12), 9,7 (16), 10,0 (32).

### 3.4 Variantes de treino (separadas)

| run | I gauss corr. μ / z (bits) | I_CE / H | taxa (bits) |
|---|---|---|---|
| n8_k128_s1 (base) | 13,4 / 9,2 | 2,94 / 2,99 | 68,6 |
| -spk | 12,3 / 8,1 | 2,89 / 2,99 | 68,4 |
| -spkdrop | 8,9 / 3,0 | 2,76 / 2,99 | 40,7 |
| -contrast | 23,0 / 14,9 | 2,99 / 2,99 | 54,9 |
| -swap | 23,4 / 14,3 | 2,85 / 2,99 | 61,0 |
| -aug (58 pseudo-locutores) | 18,4 / 11,4 | 2,93 / 4,84 | 71,7 |
| n32_k128_s1 (base) / -spkdrop | 19,5 / 11,3 → 12,7 / 4,3 | 4,59 → 4,32 / 4,99 | 71,6 → 42,5 |

A informação de voz em z varia por um fator 3–5 entre variantes (3 a 15 bits), e segundo `teoria_cobertura.md` (fato 4) nenhuma delas mudou a dependência da cópia em N. Isso é coerente com a seção 5: **os bits de voz no latente não são o gargalo** (inferência fraca: não tenho aqui as taxas de cópia por variante com IC).

## 4. Tarefa 3 — Dimensão intrínseca da variedade de vozes

Método: médias de μ por locutor treinado, em coordenadas branqueadas. As médias brutas são ruidosas (com ~8 frases por locutor o ruído tem desvio 1/√8 por direção, e um nulo só de ruído já dá TwoNN ≈ 11), então removo o ruído pelo preditor BLUP v̂_s = Σ_b (Σ_b + W/n_s)^−1 (m_s − m̄). Estimadores: TwoNN (Facco et al. 2017, forma de máxima verossimilhança), Levina–Bickel k = 5 (versão de MacKay–Ghahramani), e um **expoente de cobertura** medido dentro do espaço latente: distância mediana de um locutor ao mais próximo de m outros sorteados, r_NN(m) ∝ m^−γ, ajustado em m = 4…64 (d_cov = 1/γ). Cada número vem com um **nulo gaussiano**: mesmo Σ_b, mesmo número de locutores e de frases, mesmo procedimento (simulado 100 vezes; 20 no expoente de cobertura). Só uso S ≥ 30.

| run | S | TwoNN médias brutas | TwoNN só ruído | TwoNN médias sem ruído [IC 80% subamostra] | TwoNN nulo gaussiano [IC95] | Levina–Bickel k=5 sem ruído | LB nulo | expoente de cobertura γ (r_NN ∝ m^-γ) | γ nulo gaussiano |
|---|---|---|---|---|---|---|---|---|---|
| n32_k8_s1 | 32 | 5.6 | 9.0 | 5.2 [4.5, 8.3] | 9.3 [6.7, 12.7] | 4.9 [3.9, 5.5] | 5.8 | — | — |
| n32_k8_s2 | 32 | 6.4 | 8.7 | 5.9 [4.9, 9.9] | 9.4 [6.3, 12.5] | 5.7 [4.7, 6.4] | 5.8 | — | — |
| n32_k32_s1 | 32 | 7.2 | 9.7 | 8.8 [6.5, 11.7] | 10.4 [7.1, 14.5] | 5.8 [4.8, 6.2] | 6.4 | — | — |
| n32_k128_s1 | 32 | 6.6 | 10.2 | 9.2 [6.8, 12.7] | 10.9 [7.5, 14.8] | 6.1 [5.0, 6.6] | 6.7 | — | — |
| n32_k128_s2 | 32 | 6.9 | 10.0 | 9.7 [7.3, 13.9] | 11.0 [7.9, 15.7] | 5.9 [5.1, 6.7] | 6.8 | — | — |
| n128_k8_s1 | 124 | 8.9 | 11.4 | 10.1 [8.5, 12.3] | 12.5 [10.5, 15.4] | 6.7 [5.9, 6.9] | 8.3 | 0.128 [0.119, 0.159] → d≈7.8 | 0.099 ± 0.009 |
| n128_k32_s1 | 127 | 7.7 | 11.3 | 10.6 [9.3, 12.3] | 13.6 [11.3, 16.1] | 7.3 [6.6, 7.5] | 8.8 | 0.122 [0.107, 0.153] → d≈8.2 | 0.096 ± 0.007 |
| n128_k32_s2 | 127 | 7.8 | 11.5 | 10.5 [9.1, 12.4] | 13.2 [10.7, 16.3] | 7.5 [6.5, 7.5] | 8.8 | 0.138 [0.111, 0.156] → d≈7.3 | 0.095 ± 0.008 |
| n200_k32_s1 | 197 | 8.9 | 11.6 | 9.7 [8.8, 11.0] | 13.8 [11.9, 15.8] | 7.0 [6.4, 7.0] | 9.2 | 0.134 [0.110, 0.155] → d≈7.4 | 0.097 ± 0.008 |
| n32_k32_s1-wide | 32 | 7.7 | 9.8 | 7.7 [6.0, 10.0] | 10.3 [7.1, 13.8] | 5.7 [4.8, 6.2] | 6.4 | — | — |
| n128_k32_s1-wide | 127 | 8.0 | 11.4 | 10.4 [9.0, 11.4] | 13.2 [11.2, 15.7] | 7.1 [6.4, 7.4] | 8.8 | 0.122 [0.103, 0.155] → d≈8.2 | 0.100 ± 0.009 |

Leitura:

- Os estimadores locais dão d ≈ 7–10 (Levina–Bickel ≈ 7, TwoNN ≈ 10, cobertura ≈ 7–8), consistentemente **abaixo** do que daria uma gaussiana com o mesmo espectro (≈ 9–14). Há sinal fraco de estrutura não linear (aglomerados, curvatura ou cauda pesada) que reduz a dimensão efetiva em 2–3 unidades em relação ao espectro linear, mas não há variedade de dimensão muito baixa (d = 2–4 está fora de todos os intervalos com S ≥ 124).
- Com S = 32 os intervalos são largos (TwoNN 5–14) e os números não são informativos.
- Os números são de uma amostra de 124–197 vozes do LibriTTS train-clean, num espaço latente por run. Estimadores de vizinhos com ~130 pontos em 32 dimensões são fortemente enviesados para baixo, por isso o nulo é a comparação que importa, não o valor absoluto.

### 4.1 O expoente 0,26 e a dimensão que cada teoria exige

Não-cópia nas 38 externas: 0,43 / 0,30 / 0,21 ⇒ p ∝ N^−0,26 (ajuste sem piso; três pontos, uma semente).

| Teoria (falha p em função de N) | Expoente previsto | d exigido para 0,26 | Compatível com o medido (d ≈ 7–10; γ ≈ 0,12–0,14)? |
|---|---|---|---|
| Cobertura linear: p ∝ r, r ∝ N^−1/d | 1/d | 3,8 | **Não**: exigiria γ ≈ 0,26; medido 0,12–0,14 (IC 0,10–0,16) |
| Cobertura com erro quadrático: p ∝ r², r ∝ N^−1/d | 2/d | 7,7 | **Sim**: 2γ = 0,24–0,28 (IC 0,21–0,31) |
| Regressão não paramétrica de suavidade β (Stone): N^−2β/(2β+d) | 2β/(2β+d) | 5,7·β | Sim para β ≈ 1,2–1,8 |
| Cobertura por ε fixo (massa fora das bolas) | decaimento exponencial | — | Não descreve uma lei de potência |
| Lei de dois níveis a + A/N (seção 3 do relatório de cobertura) | 1 (sem piso) | — | Só com piso a > 0; não é questão de d |

A medida que mais discrimina é γ dentro do espaço latente do modelo N = 128: **a distância ao vizinho treinado cai como m^−0,12…−0,14, metade do expoente da falha**. Isso descarta a forma "falha proporcional à distância" com a dimensão medida e é compatível com "falha proporcional ao quadrado da distância" (o que se espera de um erro suave, de segunda ordem, em torno de cada voz treinada). Ressalvas fortes: (i) a coincidência 2γ ≈ 0,26 vem de um ajuste com 3 pontos; (ii) entre runs, a distância (corrigida pelo ruído) da referência externa ao vizinho treinado mais próximo cai mais rápido, como N^−0,23 (tabela abaixo), mas essa correção é grosseira (o mínimo sobre vizinhos ruidosos é enviesado para baixo, mais quanto maior N) e cada run tem seu próprio codificador; com esse número, a cobertura linear voltaria a ser compatível; (iii) por locutor, d_NN **não** prevê a falha em N = 128 (seção 6), então cobertura não é o mecanismo inteiro em N alto.

| run | d_NN (whitened) | piso de ruído √E‖e‖² | d_NN corrigida [IC95] | d_LJ | dist. centro LJ–centro Libri | cópia média |
|---|---|---|---|---|---|---|
| n8_k32_s1-wide | 9.64 | 5.78 | 7.37 [6.18, 8.62] | 20.0 | 17.5 | 0.57 |
| n32_k32_s1-wide | 8.02 | 5.83 | 5.26 [4.58, 5.94] | 17.1 | 14.7 | 0.70 |
| n128_k32_s1-wide | 7.42 | 6.00 | 3.90 [3.12, 4.76] | 18.9 | 16.5 | 0.79 |

## 5. Tarefa 4 — Taxa-distorção e latente mínimo

### 5.1 Precisão necessária para a cópia virar

Nas runs `-wide`, ajustei uma logística binomial (25 frases por locutor) da cópia em d_LJ (distância branqueada da referência à média da voz padrão). A transição fica em **d50 ≈ 19,5 / 15,5 / 15,6** unidades (N = 8 / 32 / 128; IC 95% 17,6–21,3 / 13,2–16,4 / 11,5–17,7), com **largura 1/β ≈ 2** (IC ≈ 0,7–5,6). Para comparação: o ruído de uma frase é 1 por direção; a dispersão de d_LJ entre os 38 locutores é 2,1–2,8; a distância centro-LJ → centro-LibriTTS é 14,7–17,5. Ou seja: uma voz é copiada quando a referência cai mais longe da voz padrão do que o centro das vozes treinadas, e a decisão é sensível a ≈ 2 unidades num único eixo, o dobro do ruído de uma frase.

Em bits: o desfecho por locutor (copia/não copia) tem no máximo 1 bit, e o que dele se prevê pela geometria é pouco — a validação leave-one-speaker-out reduz a perda preditiva de 0,77 → 0,63 bit por amostra no melhor modelo (N = 128; tabela da seção 6). A identidade da voz no latente usa ≈ 10 bits (z) / 17 bits (μ) em 6–12 direções; a decisão de cópia depende, em primeira ordem, de ≈ 1–2 bits num eixo (distância à voz padrão), mais o F0.

### 5.2 Mais bits na entrada não mudam a cópia

O modelo gaussiano prevê que a média de R = 4 referências acrescenta **+2,5 / +4,6 / +5,0 bits** de informação de voz em z (N = 8 / 32 / 128; +6,5 / +15 / +18 bits em μ). O relatório de cobertura (8.8) mediu, nas mesmas 38 vozes, cópia por locutor 66 → 64%, 81 → 79%, 88 → 84% (R = 1 → 4). Cinco bits a mais sobre a voz não produziram cópia a mais. Junto com 3.4, isso sustenta que **o gargalo não é a quantidade de informação de voz no código**, e sim o que o decodificador consegue renderizar (a hipótese do relatório de cobertura). É uma inferência por associação: o critério `sim_spk` usado em R = 4 também mede outra coisa que `sim_ref`.

### 5.3 Previsões falsificáveis (latente mínimo e peso do KL)

Duas hipóteses competem e dão previsões diferentes:

- **H_bits** (a cópia precisa de toda a informação de voz que o código tem): a cópia nas 38 externas, treino 128 × 32, fica dentro de ±10 pontos dos 79% enquanto o código guardar ≥ 90% dos bits de voz em z, isto é, **latente com d ≥ 12**; cai mais de 10 pontos com **d ≤ 4** (≈ 53% dos bits).
- **H_eixo** (a cópia, pelo critério `sim_ref > sim_default`, depende sobretudo da distância à voz padrão): mesmo **d = 4** mantém a cópia em ±10 pontos.

Previsão registrada (minha aposta): **d_min ≈ 6–8** (o número de direções com SNR > 1 em z): cópia inalterada (±10 pontos) com d = 8, queda > 10 pontos com d = 2. Falseia-se se d = 12 já perde > 10 pontos (o decodificador usa mais que a voz: prosódia, canal) ou se d = 2 não perde (H_eixo).

Peso do KL: a taxa atual (≈ 72 bits) tem ≈ 45 bits de "precisão sem informação". Previsão: **subir o peso final do KL até a taxa medida cair para ≈ 25–30 bits não muda a cópia (±10 pontos)**; a cópia cai > 10 pontos só quando a taxa ficar abaixo de ≈ 12–15 bits (≈ informação de voz em z + 2–5 bits). Não consigo converter isso num peso β sem treinar (o peso ainda estava em rampa, 0,024 no passo 2.500); a varredura sugerida é β_final ∈ {0,024; 0,1; 0,3; 1}, lendo a taxa no próprio `metrics.jsonl`. Os runs `-spkdrop` (taxa 41–46 bits, 3–5 bits de voz em z) são o ponto de dado já existente mais próximo dessa varredura.

Previsão de "troca de referência": se a geometria for a variável operante, trocar a frase de referência de cada locutor externo por outra frase do mesmo locutor (ruído 1 por direção) muda o desfecho majoritário de **≈ 15% / 15% / 4% dos locutores** (≈ 5–6 / 5–6 / 1–2 de 38, N = 8 / 32 / 128; `out/flip.json`). Muito mais que isso (> 30%) significa que o "tudo ou nada por voz" não vem do latente da referência.

## 6. Tarefa 5 — Atrator da voz padrão

Preditores por locutor externo (uma referência cada): d_LJ (distância branqueada à média LJ), d_NN (à média do locutor treinado mais próximo), log F0 da referência. Desfechos: fração copiada (25 frases), margem média `sim_ref − sim_default`, `sim_ref`, `sim_default`. Spearman com IC por bootstrap de locutores; correlação parcial de postos; logística binomial com validação leave-one-speaker-out (perda em bits por amostra; menor é melhor); modelo agrupado com efeito fixo de run, preditores padronizados dentro de cada run, bootstrap por locutor (os mesmos 38 em todas as runs).

| run | cópia | ρ(cópia, d_LJ) | ρ(cópia, d_NN) | ρ(cópia, log F0) | P(abs ρ_LJ > abs ρ_NN) | ρ parcial d_LJ dado F0 | ρ parcial d_NN dado d_LJ | ρ(sim_ref, d_LJ) | ρ(sim_default, d_LJ) |
|---|---|---|---|---|---|---|---|---|---|
| n8_k32_s1-wide | 0.57 | 0.34 [-0.02, 0.62] | -0.48 [-0.68, -0.20] | -0.53 [-0.75, -0.26] | 0.24 | 0.12 | -0.41 | -0.14 [-0.44, 0.19] | -0.61 [-0.83, -0.29] |
| n32_k32_s1-wide | 0.70 | 0.43 [0.10, 0.67] | -0.22 [-0.55, 0.14] | -0.52 [-0.76, -0.23] | 0.77 | 0.35 | -0.33 | -0.36 [-0.62, -0.06] | -0.67 [-0.84, -0.42] |
| n128_k32_s1-wide | 0.79 | 0.63 [0.39, 0.78] | -0.12 [-0.41, 0.21] | -0.37 [-0.63, -0.08] | 0.99 | 0.60 | -0.14 | 0.09 [-0.23, 0.41] | -0.72 [-0.84, -0.53] |

| run | só intercepto | d_LJ | d_NN | log F0 | d_LJ+d_NN | d_LJ+F0 | d_LJ+d_NN+F0 | d50 (whitened) | largura 1/β | sobredispersão | fração tudo-ou-nada |
|---|---|---|---|---|---|---|---|---|---|---|---|
| n8_k32_s1-wide | 1.016 | 0.904 | 0.883 | 0.889 | 0.836 | 0.865 | 0.829 | 19.5 [17.6, 21.3] | 2.4 [1.4, 5.6] | 18 | 0.66 |
| n32_k32_s1-wide | 0.908 | 0.790 | 0.889 | 0.797 | 0.797 | 0.726 | 0.726 | 15.5 [13.2, 16.4] | 1.7 [1.0, 3.8] | 15 | 0.63 |
| n128_k32_s1-wide | 0.769 | 0.673 | 0.803 | 0.708 | 0.701 | 0.626 | 0.633 | 15.6 [11.5, 17.7] | 2.0 [0.7, 4.1] | 16 | 0.74 |
| agrupado (efeito fixo de run, LOSO) | 0.898 | 0.771 | 0.869 | 0.791 | 0.798 | — | 0.726 | — | — | — | — |

Agrupado (três runs, coeficientes padronizados, IC 95% por bootstrap de locutor): d_LJ **+0,91 [0,47; 1,80]**, d_NN −0,20 [−0,85; 0,27], log F0 **−0,85 [−1,43; −0,29]**. Spearman dentro da run, agrupado: d_LJ 0,47 [0,26; 0,62], d_NN −0,27 [−0,48; −0,05], F0 −0,48 [−0,67; −0,23].

Leitura:

1. **A pergunta "d_LJ prevê melhor que d_NN?" tem resposta que depende de N.** Com N = 8, d_NN prevê melhor (ρ = −0,48 contra 0,34; P(|ρ_LJ| > |ρ_NN|) = 0,24); com N = 32 empate (0,77); com N = 128 d_LJ claramente (0,63 contra −0,12; P = 0,99). No agregado, d_LJ ganha (perda LOSO 0,771 contra 0,869 bit/amostra; intercepto 0,898). É o padrão do relatório de cobertura (seção 8.9), agora com IC e com métrica branqueada.
2. **F0 e d_LJ são parcialmente redundantes** (ρ(d_LJ, F0) = −0,48 / −0,27 / −0,26). Com N = 8 o F0 explica quase tudo (parcial d_LJ | F0 = 0,12); com N = 128 é o contrário (parcial 0,60 contra −0,28 do F0). O melhor modelo em N = 32 e 128 é d_LJ + F0.
3. **Atrator ou artefato da medida? Não dá para separar com os dados salvos.** d_LJ não prevê `sim_ref` (ρ = −0,14 / −0,36 / +0,09) e prevê fortemente `sim_default` (−0,61 / −0,67 / −0,72). O efeito de d_LJ sobre a "cópia" passa inteiro pela semelhança da saída com a voz padrão. Isso é exatamente o que o critério `sim_ref > sim_default` produziria mesmo com cópia perfeita (uma referência parecida com LJ gera uma saída parecida com LJ), e também o que um atrator produziria. O teste que separa: medir `sim_default` **da própria referência**. Artefato prevê `sim_default(saída) ≈ sim_default(referência)`; atrator prevê `sim_default(saída) > sim_default(referência)`, com a diferença crescendo quando d_LJ diminui.
4. A cópia é quase tudo-ou-nada por voz: sobredispersão 15–18 vezes a binomial; 63–74% dos locutores têm cópia ≤ 10% ou ≥ 90%.

## 7. Tarefa 6 — O que não dá para medir e como medir

Sem checkpoints do decodificador não se mede:

- **I(z; saída)** e quanto de cada direção do latente o decodificador realmente usa (Jacobiano, sensibilidade por direção);
- **o latente mínimo diretamente**: projetar z nas k primeiras direções (entre/dentro) e sintetizar — com o checkpoint isso testa a previsão da seção 5.3 **sem treinar nada**;
- **bits por parâmetro**: quanto da capacidade do decodificador é gasto em cada voz treinada, que é a grandeza que a leitura "o decodificador não renderiza vozes novas" pede;
- **a separação atrator × artefato** (precisa de `sim_default` da referência, ou de teste de escuta);
- **variabilidade dentro do locutor externo**: só há uma frase de referência por locutor externo nos latentes; a média de R = 4 não foi salva.

Experimentos que medem (do mais barato ao mais caro):

1. Guardar o checkpoint final de cada treino (custo zero). Com ele: projeção em k direções (5.3), varredura de "força" ao longo do eixo LJ, Jacobiano do decodificador por direção.
2. Extrair latentes das 25 frases de cada locutor externo (só codificador) e repetir a síntese com outra referência: testa a previsão de troca de referência (≈ 15% / 15% / 4%).
3. `sim_default` das próprias referências (só o codificador de locutor): separa atrator de artefato.
4. Varredura de d ∈ {2, 4, 8, 12, 32} e de β_final, treino 128 × 32, avaliação nas 38 externas: testa H_bits × H_eixo e a previsão de taxa.
5. Bits por parâmetro: treinar o mesmo modelo com decodificador de larguras diferentes (½×, 1×, 2× parâmetros) e N ∈ {32, 128, 512}; medir o número de vozes treinadas copiadas acima de um limiar vs parâmetros, e o comprimento de descrição do decodificador (quantizar os pesos a b bits até a cópia das treinadas cair, ou código prequencial). A previsão de "capacidade" é que o expoente em N mude com a largura; a de "cobertura" é que não mude.

## 8. Previsões falsificáveis (lista)

1. **Latente mínimo**: com d = 8 a cópia nas 38 externas (128 × 32) fica em 79% ± 10 pontos; com d = 2 cai > 10 pontos. (Falseado se d = 12 perder > 10 ou d = 2 não perder.)
2. **Peso do KL**: taxa reduzida para 25–30 bits sem perda de cópia (±10 pontos); perda só abaixo de ~12–15 bits.
3. **Troca de referência**: muda o desfecho majoritário de ≈ 15% / 15% / 4% dos locutores (N = 8 / 32 / 128); > 30% falseia o papel do latente da referência.
4. **R = 16**: acrescenta ≈ +7,6–8,3 bits de voz em z (modelo gaussiano), e a cópia por locutor muda < 5 pontos (como R = 4).
5. **N maior (≥ 512)**: ρ(cópia, d_NN) ≈ 0 e ρ(cópia, d_LJ) ≥ 0,5; as falhas se concentram em d_LJ abaixo da distância centro LJ–centro LibriTTS.
6. **Expoente**: se a falha for ∝ r² com d ≈ 7–8, o expoente de não-cópia em N fica em 0,26 ± 0,05 também entre N = 128 e 512 (≈ 0,21 → 0,15); se for o comportamento de cobertura linear medido entre runs (γ ≈ 0,23), também ≈ 0,23–0,26 — **esse teste não separa as duas**; o que separa é a forma da falha por locutor em função da distância ao vizinho (linear vs quadrática), que pede dezenas de vozes por faixa de distância.

## 9. Limites

- Tudo é sobre o codificador e associações. Nenhuma afirmação causal sobre o decodificador.
- Os estimadores gaussianos supõem μ = v + e com v e e gaussianos e W comum a todos os locutores. Os números de "bits sobre um vetor de voz contínuo" dependem dessa suposição; a cota discreta (classificador) não, mas é só cota inferior.
- Com N = 8 (e N = 2), Σ_b tem posto ≤ 7 (≤ 1): as estimativas são da projeção nesse subespaço, não da população.
- Cada run tem seu codificador: distâncias entre runs não são estritamente comparáveis (branquear por W ajuda, não resolve).
- As 38 vozes externas têm uma referência cada; uma semente por N nas `-wide`; o critério de cópia é binário e vem de um único verificador de locutor.
- O expoente 0,26 é ajuste de 3 pontos sem piso; com piso o expoente muda (o relatório de cobertura ajusta leis com piso).
- A coincidência 2γ ≈ 0,26 é sugestiva, não demonstração.

## Apêndice — reprodução

Scripts (todos com `python3 -I`, caminhos como argumento), em `teoria/`:

```
python3 -I info_rate.py      ../coverage2 out        # tarefa 1 → out/rate.json, out/rate_table.md
python3 -I info_speaker.py   ../coverage2 out 200 200 # tarefa 2-3 (estimadores, permutação, LOO, TwoNN bruto) → out/speaker.json
python3 -I info_speaker2.py  ../coverage2 out 100     # viés paramétrico, metades, direções, ID sem ruído, cobertura → out/speaker2.json
python3 -I info_attractor.py ../coverage2 out 2000    # tarefas 4-5 → out/attractor.json
python3 -I info_flip.py      out                      # previsão de troca de referência → out/flip.json
python3 -I info_summary.py   out                      # tabelas → out/summary_tables.md
```

`info_speaker_lib.py` contém as funções de `info_speaker.py` sem efeitos colaterais; `info_common.py` as utilidades.
