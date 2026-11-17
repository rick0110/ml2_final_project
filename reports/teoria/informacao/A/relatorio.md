# Clonagem *zero-shot* como codificação de fonte: teoria taxa-distorção da cobertura de vozes

Documento de trabalho (teoria2/A), 17/11/2026. Só CPU, sem treino, nenhum arquivo de repositório alterado.
Scripts e saídas nesta pasta (`*.py`, `out/*.json|csv`), todos rodados com `python3 -I script.py <caminhos>`.

**Legenda.** **[PROVADO]** segue das hipóteses por conta completa; **[HEURÍSTICO]** conta com aproximação não
controlada; **[CONJECTURA]** hipótese de modelagem; **[SIMULADO]** conferido numericamente (confere a conta, não a
hipótese); **[DADOS]** medido nos latentes/resultados salvos.

---

## 0. Resumo

**Fórmula principal.** Seja $c$ o código (latente branqueado) da referência, $v_0$ a voz padrão (LJ), $P_v$ a
distribuição de vozes de treino e $V\sim P_v*\mathcal N(0,I/K)$ um *codeword* ruidoso. Então

$$
\boxed{\;q(N,K,M)=\mathbb E_c\Big[\big(1-F_c(t(c))\big)^N\Big],\qquad
F_c(t)=\Pr\big(\|c-V\|^2\le t\big),\qquad t(c)=\|c-v_0\|^2-\tau,\quad \tau=2T\ln\tfrac{M_{\rm ef}}{K}\;}
$$

"não copiada" $\iff$ a **distorção do vizinho mais próximo** num livro-código aleatório de $N$ vozes,
$D_N(c)=\min_i\|c-v_i\|^2$, passa da **tolerância** $t(c)$ que o atrator da voz padrão deixa àquela voz.
Equivalentemente (espectro de informação), $q(N)\approx\Pr_c\big[L(c)>\log_2N\big]$ com
$L(c)=-\log_2F_c(t(c))$ = **bits necessários para "acertar" a voz $c$ com vozes sorteadas**.

**O que foi medido, ajustado e previsto** (38 vozes externas, $K=32$):

| | $N=8$ | $N=32$ | $N=128$ | parâmetros tirados da curva |
|---|---|---|---|---|
| observado | 0,434 | 0,302 | 0,211 | — |
| decodificador de Bayes ($T=1$, $M=2000$ nominal) | 0,092 | 0,039 | 0,000 | 0 — **falha** |
| $\tau$ ajustado em $N=128$, livros-código reais, $T=1/2/3$ | 0,375/0,401/0,445 | 0,263/0,264/0,279 | (0,211) | 1 ($\tau$) |
| $\tau_0=\|\mu-v_0\|^2$ (identidade sem ajuste, achada *post hoc*), $T=2$ | 0,400 | 0,263 | 0,210 | 0 (*post hoc*) |
| idem, cada run no **próprio** espaço latente | 0,360 | 0,219 | 0,210 | 0 (*post hoc*) |
| $\tau$ de $N=128$, livros-código **sorteados** (média) | 0,32 ± 0,05 | 0,25 ± 0,02 | (0,211) | 1 |
| população gaussiana (espectro medido), refs. reais | 0,353 | 0,274 | 0,253 | 1 |

O modelo de um parâmetro, calibrado em um ponto, prevê os outros dois com erro de 0,03–0,06 (ou 0,00–0,04 com
$T=3$), no sentido de **subestimar a dependência em $N$** quando $T=1$. A versão totalmente paramétrica (vozes
gaussianas) erra mais (0,08) porque as vozes reais são mais aglomeradas que a gaussiana: $D_N$ real cai mais rápido
(mediana 39,2→19,5 contra 39,6→23,5). Por voz, o preditor ordena as 38 vozes com Spearman 0,50–0,58 em cada $N$.
**Falhas:** as 4 vozes novas da grade são mais difíceis do que o modelo diz (viés −0,19 a −0,24 em 14 treinos);
o efeito de $K$ nas vozes treinadas é maior do que o termo $\ln K$ prevê; o expoente não fica identificado.

**Resultados teóricos novos:**
1. Com espectro $\lambda_j\propto j^{-a}$, o livro-código ótimo dá $D(R)\propto R^{-(a-1)}$, isto é, **potência de
   $\log N$**, com expoente local $\beta_D(N)\approx(a-1)/\ln N$; o livro-código aleatório paga **2–4 bits** a mais
   ($N$ aleatórias valem $\approx N/10$ ótimas); alta resolução ($N^{-2/d}$) não é atingida até $N=4096$ [SIMULADO].
2. **Teorema tauberiano**: se a fração de vozes com probabilidade de bola $p_c\le x$ cresce como $x^{\beta}$,
   então $q(N)-q_\infty\sim C\,\Gamma(1+\beta)N^{-\beta}$; com vozes perto da fronteira do atrator e dimensão local
   $d$, $\beta=2/d$. Com $d\approx7$–8 (medido) dá 0,25–0,29, o "0,26" observado [PROVADO dado o modelo; SIMULADO].
3. **Média de referências não ajuda** sai da estrutura: o ruído da referência $\|\xi\|^2$ cancela exatamente entre
   $\|c-v_0\|^2$ e $D_N(c)$; só sobra a projeção do ruído no eixo $v_{\rm nn}-v_0$ [PROVADO].
4. **Piso**: com o atrator atual, $q_\infty\approx0{,}10$ (IC 90% por reamostragem 0,05–0,18): vozes cuja referência
   está a menos de $\sqrt\tau$ da voz padrão não são copiadas com nenhum $N$. Previsão: $q(2456)\approx0{,}15$
   (0,10–0,22), **90% de cópia inalcançável só com mais locutores**. Sem voz dominante (voz em duas etapas,
   $v_0\to\mu$), o mesmo cálculo dá $q=0{,}45/0{,}11/0{,}00$ em $N=8/32/128$.
5. **Bits**: uma frase carrega $I(v;c)\approx10{,}3$ bits de voz; copiar a voz mediana exige $L\approx2$ bits,
   75% das vozes 7,5 bits, 80% 10 bits ($\approx2^{10}$ locutores sorteados), 85% 16 bits — acima do que uma
   referência carrega. A escala de saturação é $N^\star\approx2^{I(v;c)}\approx10^3$ locutores.

---

## 1. Montagem

### 1.1 Espaço e quantidades medidas [DADOS, `a1_geometria.py` → `out/a1.json`]

Espaço comum: latente $\mu$ (32 dim.) do treino $N=128$ (`n128_k32_s1-wide`), branqueado pela covariância do canal
que o decodificador vê no treino, $W_c=W+\mathrm{diag}\,\mathbb E\sigma^2$ ($W$: dentro do locutor, LibriTTS). Nessas
unidades, um código amostrado tem ruído $I$ em torno da média do locutor. Os livros-código de $N=8$ e $N=32$ são
**subconjuntos** dos 128 (conferido nos `train_manifest.csv`), então os três treinos têm vozes treinadas
localizadas no mesmo espaço.

| Quantidade | Valor | Fonte |
|---|---|---|
| autovalores de $\Sigma_b$ (entre locutores, corrigido de ruído) | 8,3; 5,8; 5,2; 3,1; 2,2; 2,0; 1,1; 1,0; 0,85; 0,76; 0,44; … | componentes de variância |
| $\mathrm{tr}\,\Sigma_b$; razão de participação | 32,6; 7,0 | |
| expoente $a$ (8 / 16 primeiras) | 1,03 / 1,55 | |
| ruído de uma referência $\mathrm{tr}\,W_x$ ($\mu$ de uma frase) | 12,0 | |
| $\|v_0-\mu\|$ (LJ ao centro LibriTTS) | 10,8 (5,0 desvios de Mahalanobis nas 8 direções) | |
| distância típica entre duas vozes $\sqrt{2\,\mathrm{tr}\Sigma_b}$ | 8,1 | |
| $\|c-v_0\|$ das 38 referências | 12,7 ± 1,8 | |
| $I(v;c)=\tfrac12\sum_j\log_2(1+\lambda_j)$ | 10,3 bits (8 direções: 8,0 bits) | `a5`, `a6` |

O valor de 10,3 bits reproduz os 9,8–10,2 bits "em $z$" de `03_informacao.md` (mesma família de estimador; não é
confirmação independente). A voz padrão é uma voz atípica, a 5 desvios da população.

### 1.2 O decodificador como canal de teste de um livro-código

**Hipótese H (decodificador de Gibbs)** [CONJECTURA]. A voz renderizada para o código $c$ é a média das vozes de
treino ponderada por
$$
\pi_s(c)=\frac{n_s\,e^{-\|c-v_s\|^2/2T}}{\sum_r n_r\,e^{-\|c-v_r\|^2/2T}},\qquad n_s=K\ (\text{LibriTTS}),\ n_0=M\ (\text{LJ}).
$$
Esta é exatamente a forma do passo de Blahut–Arimoto (canal de teste ótimo de taxa-distorção restrito a um
livro-código com marginal $n_s/\sum n$, multiplicador $\beta=1/T$; *deterministic annealing*, Rose 1998): **as
massas de treino fazem o papel da marginal de saída, e o atrator da voz padrão é simplesmente o codeword de maior
massa**. A cópia é decidida pelo Lema 1 de `02_derivacao.md` ($m=(\tfrac12-\rho)\|\Delta\|^2$), aplicado no espaço
latente: $\rho=1-\langle\hat y-v_0,c-v_0\rangle/\|c-v_0\|^2$, $\Pr(\text{cópia}\mid c)=\Phi(\kappa(\tfrac12-\rho))$,
$\kappa=\|\Delta\|^2/\sigma_w=0{,}65/0{,}036$ ($\sigma_w$ medido; $\|\Delta\|^2$ é o valor suposto em 02; $\kappa$ só
suaviza um resultado quase 0/1).

**Proposição 1 (Bayes)** [PROVADO]. Se os códigos de treino do locutor $s$ são $c=v_s+\varepsilon$, $\varepsilon\sim\mathcal N(0,I)$
(branqueado), e o decodificador minimiza o erro quadrático da voz renderizada na mistura de treino, então
$\hat y(c)=\mathbb E[v_S\mid c]=\sum_s\pi_s(c)v_s$ com $T=1$ e $n_s$ = número de frases. *Prova:* regra de Bayes,
$\Pr(S=s\mid c)\propto n_s\,\mathcal N(c;v_s,I)$. $\square$

**Teste** [DADOS, `a2_gibbs_dados.py`]: com $T=1$, $M=2000$, $K=32$, o decodificador de Bayes prevê
$q=0{,}09/0{,}04/0{,}00$ contra $0{,}43/0{,}30/0{,}21$. A razão de verossimilhança LJ contra o vizinho treinado é de
~70 nats, contra $\ln(M/K)=4{,}1$: **o decodificador real está longe do ótimo de Bayes; ele puxa para a voz padrão
com uma força equivalente a $\tau/2\approx57$ nats**. A massa nominal das 2.000 frases não explica o atrator; o
candidato natural é o *checkpoint* de partida (um só locutor, muitas épocas), não modelado.

**Proposição 2 (limite de Poisson)** [PROVADO]. Com $\tau\equiv2T\ln(M/K)$ fixo e $T\to0$,
$\pi_0(c)\to\mathbf 1\big[\|c-v_0\|^2-\tau<D_N(c)\big]$, $D_N(c)=\min_i\|c-v_i\|^2$. *Prova:* o log da razão
$\pi_0/\pi_{\rm nn}$ é $\big(D_N(c)-\|c-v_0\|^2+\tau\big)/2T$; os demais termos são de ordem $e^{-\Delta/2T}$. $\square$
Para livros-código i.i.d. de $P_v$, $\Pr(D_N>t\mid c)=(1-F_c(t))^N$ exatamente, o que dá a fórmula do resumo.
Para $P_v$ gaussiana, $F_c$ é uma qui-quadrado generalizada não central (Imhof; ponto de sela de
Lugannani–Rice para a cauda; `rd_theory.py`).

**Verificação** [SIMULADO, `t2_verifica_formula.py` → `out/t2.json`]: geometria gaussiana calibrada, 600 vozes,
4 livros-código por $N$:

| caso | $N$ | fórmula | espectro de informação $\Pr(L>\log_2N)$ | Gibbs simulado |
|---|---|---|---|---|
| $T=1$, $K=32$, $\tau=116{,}5$ | 8 / 32 / 128 / 512 | 0,514 / 0,424 / 0,363 / 0,322 | 0,542 / 0,445 / 0,373 / 0,332 | 0,505 / 0,412 / 0,353 / 0,310 |
| $T=1$, $K=8$ | 8 / 128 / 512 | 0,562 / 0,413 / 0,366 | 0,593 / 0,437 / 0,377 | 0,545 / 0,406 / 0,347 |
| $T=2$, $\tau=116{,}5$ | 8 / 128 / 512 | 0,514 / 0,363 / 0,322 | — | 0,490 / 0,321 / 0,261 |
| Bayes, $M=2000$ ($\tau=8{,}3$) | 8 | 0,002 | 0,002 | 0,004 |

A fórmula de Poisson acerta o canal de Gibbs a ~0,01 com $T=1$; com $T=2$ ela superestima $q$ em $N$ grande
(0,06 em $N=512$), porque a massa de vários *codewords* próximos começa a somar (regime de lei dos grandes números).
A forma de espectro de informação erra por 0,01–0,03 (suavização de Gumbel de largura ~1,4 bit).

---

## 2. Parte 1 — $D(R)$, livro-código aleatório e vizinho mais próximo

### 2.1 Água reversa com espectro em lei de potência [HEURÍSTICO na aproximação contínua; a cota é PROVADA]

Para fonte gaussiana $\mathcal N(0,\mathrm{diag}\lambda_j)$, qualquer livro-código de $N$ pontos tem
$\mathbb E\,d\ge D(\log_2N)$ (cota *one-shot*: $I(X;\hat X)\le H(\hat X)\le\log_2N$) [PROVADO]. Com $\lambda_j=j^{-a}$,
o nível de água $\theta=J^{-a}$ ativa $J$ direções e
$$
R=\tfrac a2\sum_{j\le J}\log_2\tfrac Jj\approx\frac{aJ}{2\ln2},\qquad
D=J\theta+\sum_{j>J}j^{-a}\approx\frac{a}{a-1}J^{1-a}\ \Rightarrow\
D(\log_2N)\propto\Big(\frac{2\ln N}{a}\Big)^{-(a-1)},\quad \beta_D(N)\equiv-\frac{d\ln D}{d\ln N}\approx\frac{a-1}{\ln N}.
$$
A distorção ótima cai como **potência de $\log N$**; vista como função de $N$ numa janela curta, parece uma potência
com expoente pequeno que decai como $1/\ln N$. Para $a=1$ a queda é ainda mais lenta ($D\approx\ln(d/J)+1$).

### 2.2 Comparação numérica [SIMULADO, `t1_rd_teoria.py` → `out/t1.json`; $d=32$, traço 32,6]

$D$ total (soma em 32 dimensões) para o espectro **medido**:

| $N$ | 8 | 32 | 128 | 1024 | 4096 |
|---|---|---|---|---|---|
| ótimo (água reversa) $D(\log_2N)$ | 17,7 | 13,3 | 10,5 | 7,7 | 6,3 |
| aleatório, $N$ vozes i.i.d. de $P_v$ (Monte Carlo) | 32,4 | 22,7 | 16,5 | 11,6 | 9,5 |
| aleatório, ponto de sela (mediana) | 30,3 | 21,7 | 16,4 | 11,5 | 9,3 |
| alta resolução $\Gamma(1+\tfrac2d)(NV_d)^{-2/d}\mathbb E f^{-2/d}$ | 4,7 | 4,3 | 3,9 | 3,5 | 3,2 |
| bits a mais do aleatório, $\log_2N-R_{\rm wf}(D_{\rm rand})$ | 3,0 | 3,5 | 3,5 | 3,9 | 4,1 |

Expoentes locais $-d\ln D/d\ln N$ entre 8 e 128:

| espectro | ótimo | aleatório | $2/d_{\rm eff}$ |
|---|---|---|---|
| $a=1$ | 0,10 | 0,15 | — |
| $a=1{,}5$ | 0,16 | 0,20 | — |
| $a=2$ | 0,23 | 0,28 | — |
| medido | 0,19 | 0,24 | — |

Leituras: (i) a fórmula de alta resolução erra por um fator 4–7 em toda a faixa: **$N\le4096$ está longe do regime
$N^{-2/d}$**; o regime relevante é o de taxa baixa, $R=\log_2N\approx3$–12 bits, em que só as primeiras direções são
"codificadas"; (ii) um livro-código aleatório custa 3–4 bits (fator ~10 em $N$) a mais que o ótimo — o preço de
treinar com locutores sorteados em vez de escolhidos [SIMULADO]; (iii) o ponto de sela reproduz o Monte Carlo a
~5% [SIMULADO]. Verificação do cálculo de bola pequena (4 pontos × 3 quantis, 400.000 amostras): Imhof e ponto de sela
batem com o Monte Carlo a menos de 1–7% relativos até $p=10^{-3}$; a cota de Chernoff $e^{-I}$ fica 3–6 vezes acima,
como esperado (falta o prefator de Bahadur–Rao).

### 2.3 Distribuição de $D_N$ e de $q$

Dado $c$: $\Pr(D_N>t)=(1-F_c(t))^N\approx\exp(-N\,2^{-L_c(t)})$, uma Gumbel em $\log_2N$ centrada em $L_c(t)$ com
largura de ~1,4 bit [PROVADO]. Sobre vozes, $q(N)=\mathbb E_c\,e^{-Np_c}$, $p_c=F_c(t(c))$: **$q$ é a transformada de
Laplace da distribuição, entre vozes, da probabilidade de bola $p_c$**.

**Teorema (tauberiano/abeliano)** [PROVADO]. Seja $H(x)=\Pr_c(p_c\le x)$, $q_\infty=H(0)$. Se
$H(x)-q_\infty\sim Cx^{\beta}$ quando $x\to0$, então $q(N)-q_\infty=\int_0^1e^{-Nx}dH(x)\sim C\,\Gamma(1+\beta)\,N^{-\beta}$.
(Direção abeliana do teorema de Karamata.) Logo **$q$ é uma potência em $N$ se e só se a fração de vozes "difíceis"
($p_c$ pequeno) tem cauda regularmente variável em zero**.

**Corolário** [HEURÍSTICO]. Em alta resolução local, $p_c\approx f(c)V_d\,(t(c)-n_0)^{d/2}$ ($n_0$: distorção
irredutível). Se a densidade das tolerâncias $t(c)$ é positiva na fronteira $n_0$ (vozes "quase capturadas" pelo
atrator), $H(x)-q_\infty\propto x^{2/d}$ e
$$
q(N)-q_\infty\ \propto\ N^{-2/d}.
$$
O expoente vem das **vozes perto da fronteira do atrator** e da **dimensão local da variedade de vozes**, não do
espectro global. Com $d\approx7$–8 (Levina–Bickel e expoente de cobertura de `03`), $2/d=0{,}25$–0,29. Isto difere do
argumento de 03 ("falha $\propto r^2$"): aqui a potência vem da cauda da distribuição das vozes difíceis.

Verificação [SIMULADO, `t3_tauberiano.py`]: vozes uniformes em $[0,1]^d$ (toro), tolerâncias uniformes com 8% de
vozes capturadas:

| $d$ | inclinação de $q-q_\infty$ ($N=128$–8192) | previsto $-2/d$ | razão de decrementos $\frac{q_{32}-q_{128}}{q_8-q_{32}}$ | previsto $4^{-2/d}$ | expoente aparente 8–128 sem piso |
|---|---|---|---|---|---|
| 4 | −0,503 | −0,500 | 0,49 | 0,50 | 0,36 |
| 8 | −0,250 | −0,250 | 0,76 | 0,71 | 0,20 |
| 12 | −0,175 | −0,167 | 0,75 | 0,79 | 0,15 |

Nos dados, a razão de decrementos é $0{,}091/0{,}132=0{,}69$, o que corresponde a $d\approx7{,}4$; mas o IC desse número
é enorme (04: $\alpha_g$ de −0,7 a 1,6), então isto é **compatibilidade, não teste**. Note que o ajuste de três pontos
com $d=8$ implica $q_\infty\approx0$, enquanto o modelo calibrado (seção 5) diz $q_\infty\approx0{,}10$: os três pontos
não separam piso de expoente.

---

## 3. Parte 2 — o papel de $K$ e $M$

Do canal de Gibbs [PROVADO dado H]:

1. **$M$ e $K$ entram como massas** e, no limite de Poisson, só pela combinação $\tau=2T\ln(M/K)$: deslocam a
   tolerância de todas as vozes pelo mesmo valor. Dobrar $K$ equivale a reduzir $M$ à metade (como em 02).
2. **$K$ entra também como ruído do codeword**: a voz treinada é conhecida pela média de $K$ frases,
   $V=v_i+\mathcal N(0,I/K)$, o que soma $\approx d/K$ a $D_N$ (32/K em 32 dimensões).
3. **Regime de lei dos grandes números** ($T$ grande ou $N$ grande): $\sum_ik(c,v_i)\approx N\,\mathbb E\,k$, e a
   decisão depende de $\ln(NK/M)$; o encolhimento da região de falha é logarítmico em $NK/M$.

$$
q(N,K,M)=\mathbb E_c\Big[\Big(1-F^{(K)}_c\big(\|c-v_0\|^2-2T\ln\tfrac{M}{K}\big)\Big)^N\Big],\qquad
F^{(K)}_c(t)=\Pr\big(\|c-V\|^2\le t\big),\ V\sim P_v*\mathcal N(0,I/K).
$$

**O que os dados dizem sobre o regime** [DADOS, `a2_gibbs_dados.py`, perfil com $q_{128}$ fixado]: para cada $T$,
escolhendo a massa da LJ que reproduz $q_{128}$, a previsão em $N=8$ é

| $T$ | 1,2 | 1,8 | 2,8 | 4,4 | 6,8 | 10 | 16 | ≥25 |
|---|---|---|---|---|---|---|---|---|
| $q_8$ previsto | 0,38 | 0,40 | 0,42 | 0,53 | 0,68 | 0,79 | 0,97 | 1,0 |
| $q_{32}$ previsto | 0,26 | 0,26 | 0,27 | 0,30 | 0,38 | 0,52 | 0,70 | ≥0,89 |

O regime de massa ($T\gtrsim5$, em que $q$ depende de $\ln(NK/M)$) dá uma dependência em $N$ forte demais. Os
dados ficam no **regime de distorção** ($T\approx1$–3): o que decide é quão perto da voz pedida está a voz
treinada mais próxima, não quantas vozes há em volta. Nesse regime $T$ e $M_{\rm ef}$ não são identificáveis
separadamente; só $\tau$ é.

**Consequência**: no regime de distorção, $K$ e $M$ pesam pouco nas vozes novas (entram por $2T\ln$, alguns
unidades contra uma dispersão de folgas de ~60). Previsões em `a5` (livro-código empírico, seção 5.3), $N=128$:
$K=8$: 0,26 ($T=1$) a 0,30 ($T=3$); $K=128$: 0,18 a 0,16; referência ($K=32$) 0,25 no mesmo cálculo.

---

## 4. Parte 3 — o limiar $\tau$ e a contagem de parâmetros

### 4.1 O que fixa $\tau$

- Pela Proposição 1 (decodificador de Bayes), $\tau=2\ln(M/K)=8{,}3$: **refutado** (seção 1.2).
- Ajustado às 38 vozes em $N=128$: $\tau=114$ ($T=1$), 117 ($T=2$), 121 ($T=3$); regra dura: 113,4.
- **Identidade empírica** [DADOS, achada depois do ajuste]: $\|\mu-v_0\|^2=116{,}4$, onde $\mu$ é o centro das vozes
  treinadas. Conferência nos espaços próprios de cada treino `-wide` (`a4_tau_zero_param.py`), $\tau$ ajustado contra
  $\|\mu_N-v_0\|^2$: 147 contra 135 ($N=8$), 110 contra 107 ($N=32$), 117 contra 116 ($N=128$).

Com $\tau=\|\mu-v_0\|^2$ e $\|c-v_0\|^2=\|c-\mu\|^2+2\langle c-\mu,\mu-v_0\rangle+\|\mu-v_0\|^2$, a regra vira
$$
\text{copiada}\iff D_N(c)<\underbrace{\|c-\mu\|^2}_{D(0)\ \text{do código de taxa zero}}+\;2\langle c-\mu,\ \mu-v_0\rangle .
$$
**Leitura de taxa-distorção** [CONJECTURA]: a voz é copiada quando o livro-código treinado a representa melhor que o
código de taxa zero (a voz média), com uma correção linear que favorece vozes do lado oposto à voz padrão. Em
particular, uma voz exatamente no centro da população, perfeitamente coberta, fica na fronteira. Não tenho derivação
desta identidade a partir do treino (seria preciso modelar o *checkpoint* de partida); trato-a como hipótese de zero
parâmetros a testar, e não como resultado.

### 4.2 Ligação com a métrica e com o ruído

- Lema 1: $\rho<\tfrac12$ é o critério de cópia; no canal de Gibbs, $\rho\approx\pi_0(c)$ quando a mistura local está
  perto de $c$, então "copiada $\iff\pi_0<\tfrac12$" (decisão do Prop. 2) [HEURÍSTICO; no cálculo uso o $\rho$ exato].
- **Ruído de referência** [PROVADO]: com $c=v+\xi$, a folga
  $s(c)=\|c-v_0\|^2-\|c-v_{\rm nn}\|^2=\|v-v_0\|^2-\|v-v_{\rm nn}\|^2+2\langle\xi,v_{\rm nn}-v_0\rangle$:
  **$\|\xi\|^2$ cancela**. Média de $R$ referências só reduz o termo cruzado (desvio ~13 unidades contra dispersão de
  folgas ~60 entre vozes), e o efeito sobre $q$ é simétrico em torno do limiar: explica por que $R=4$ não mudou a
  cópia nas 38 vozes (66→64%, 81→79%, 88→84%; 03 §5.2).
- **Tudo-ou-nada**: o resultado por voz é determinístico dado $c$; o único ruído é o de síntese $\sigma_w$
  (via $\kappa$), o que reproduz a quase-dicotomia (ICC 0,83–0,94).

### 4.3 Contagem de parâmetros (honesta)

| Item | Status | Origem |
|---|---|---|
| espectro $\Sigma_b$, centro $\mu$, $v_0$, ruído $W_x$, vozes treinadas | medidos (não usam a cópia) | latentes `n128_k32_s1-wide` |
| $K$, $M$, identidade dos locutores de cada treino | conhecidos | manifestos |
| $\sigma_w=0{,}036$ | medido em `results.csv` (dentro da voz) | 02 |
| $\|\Delta\|^2=0{,}65$ (em $\kappa$) | suposto (02); só suaviza | — |
| $\tau$ | **1 parâmetro livre**, ajustado em 1 ponto ($q_{128}$); ou 0 se se aceitar $\tau=\|\mu-v_0\|^2$, achado *post hoc* | — |
| $T$ | escolhido em {1,2,3} (1 = valor de Bayes); muda $q_8$ em ±0,035 | — |
| escolha do espaço comum (latente do $N=128$) | decisão de modelagem [CONJECTURA: o espaço do codificador treinado com mais vozes aproxima o "espaço de vozes"] | — |

---

## 5. Parte 4 — calibração e previsão

### 5.1 Curva das 38 vozes [DADOS; `a3_modelo_codigo.py`, `a4`, `a5`, `a8_tau0_comum.py`]

| Variante | o que foi ajustado | $q_8$ | $q_{32}$ | $q_{128}$ | Spearman por voz (8/32/128) |
|---|---|---|---|---|---|
| observado | — | **0,434** | **0,302** | **0,211** | — |
| Gibbs $T=1$, livros-código reais | $\tau$ em $q_{128}$ | 0,375 | 0,263 | (0,211) | 0,54 / 0,54 / 0,50 |
| Gibbs $T=2$ | idem | 0,401 | 0,264 | (0,211) | 0,50 / 0,57 / 0,50 |
| Gibbs $T=3$ | idem | 0,445 | 0,279 | (0,211) | 0,50 / 0,56 / 0,50 |
| $\tau_0=\|\mu-v_0\|^2$, espaço comum, $T=1/2/3$ | nada (*post hoc*) | 0,39/0,40/0,39 | 0,26/0,26/0,24 | 0,23/0,21/0,19 | ~0,50 / 0,55–0,58 / 0,50 |
| $\tau_0$, espaço próprio de cada treino, $T=2$ | nada (*post hoc*) | 0,360 | 0,219 | 0,210 | 0,50 / 0,62 / 0,50 |
| regra dura, média sobre subconjuntos aleatórios dos 127 | $\tau$ em $q_{128}$ | 0,32 ± 0,045 | 0,25 ± 0,020 | (0,211) | — |
| população gaussiana, referências reais | $\tau$ em $q_{128}$ (livro real) | 0,353 | 0,274 | 0,253 | — |
| população gaussiana, referências gaussianas | idem | 0,486 | 0,391 | 0,333 | — |
| $\tau$ calibrado nas 8 vozes **treinadas** | $\tau$ em $q_{\rm seen}$ | 0,148 | 0,105 | 0,105 | — |

Comparação: as distâncias sozinhas dão Spearman $\rho(\text{cópia},d_{LJ})=0{,}38/0{,}49/0{,}58$ e
$\rho(\text{cópia},d_{NN})=-0{,}44/-0{,}19/-0{,}02$ no espaço comum (`a1`); a folga $d_{LJ}^2-d_{NN}^2$, 0,51/0,57/0,56.
O modelo junta as duas e ganha em $N=8$; em $N=128$ empata com $d_{LJ}$ sozinho, como deve (o vizinho deixa de
importar quando a cobertura melhora).

**Leitura.**
- O mecanismo "distorção do vizinho contra tolerância do atrator" **prevê a direção e ~70–100% da queda** de
  $q$ entre $N=8$ e 128 a partir de um único ponto. O livro-código específico de 8 locutores do treino foi pior que
  a média de subconjuntos de 8 (0,375 contra 0,32 ± 0,045 na regra dura): parte da queda observada é sorte do
  sorteio de locutores, que só uma semente não permite separar.
- **Hipótese gaussiana falha quantitativamente**: no mesmo conjunto de referências, $D_N$ mediano real é 39,2 / 26,6 /
  19,5 e o gaussiano 39,6 / 29,5 / 23,5 (±3,6/1,6/1,0). As vozes reais cobrem-se mais depressa — coerente com
  dimensão intrínseca 7–10 abaixo do nulo gaussiano (03). A versão gaussiana prevê uma curva achatada demais.
- Distâncias e expoente: $D_N$ real cai com expoente $\ln(39{,}2/19{,}5)/\ln16=0{,}25$ (com piso de ruído incluso).

### 5.2 Grade $N\times K$ e vozes treinadas [DADOS, `a7_grade.py`; sem nenhum ajuste à grade]

Espaço comum, $\tau$ das 38 vozes, dependência em $K$ pela teoria ($\tau(K)=\tau_{32}-2T\ln(K/32)$, ruído $1/K$):

| Grupo (14 treinos) | $T$ | RMSE | viés | correlação |
|---|---|---|---|---|
| 4 vozes novas | 1 | 0,261 | −0,235 | 0,29 |
| 4 vozes novas | 3 | 0,213 | −0,186 | 0,67 |
| 8 vozes treinadas | 1 | 0,152 | +0,100 | 0,86 |
| 8 vozes treinadas | 3 | 0,174 | +0,124 | 0,74 |

**Falhas reconhecidas.**
1. As 4 vozes novas da grade: o modelo diz que 3 das 4 têm folga grande em qualquer $N$ (prevê $q\approx0{,}25$);
   observa-se 0,26–0,74, com vozes meio copiadas. Já se sabia que essas 4 são mais difíceis que a média (02).
   Com $T=3$, $K=8$ ganha a direção certa (0,59–0,68 contra 0,61–0,74 em $N=8$).
2. Vozes treinadas: a ordem entre treinos é bem prevista (0,74–0,86), mas o benefício de $K$ grande é muito maior que
   o termo $2T\ln K$: $K=128$ observado 0,09–0,15 contra 0,375 previsto. Falta um termo de **qualidade do codeword**
   (o decodificador aprende a renderizar a voz $i$ com erro que cai com $K$), ausente do canal de Gibbs.
3. $\tau$ calibrado nas vozes treinadas (72–76) não serve para as novas (114–121). Com 8 vozes quase 0/1, o erro
   padrão de $q_{\rm seen}$ é ~0,15 e a diferença (0,375 previsto com $\tau_E$ contra 0,22–0,24) está a ~1 erro
   padrão; mas, junto com o item 2, sugere que vozes treinadas ganham algo além da geometria.
4. A previsão por voz é imperfeita: das 4 vozes que a regra dura põe no piso ($\|c-v_0\|^2<\tau$: locutores 4507,
   3729, 5639, 2961), duas já são copiadas em $N=128$ (5639: 0,96; 2961: 0,68). O piso é uma afirmação agregada,
   não uma lista confiável de vozes.

---

## 6. Parte 5 — complexidade amostral e bits

### 6.1 Extrapolação semi-paramétrica [`a5_extrapola.py` → `out/a5.json`; HEURÍSTICO]

$p_c$ empírico (fração das 127 vozes treinadas dentro de $t(c)$) quando há ≥3 vizinhos dentro da tolerância
(28 de 38 vozes); senão, cauda gaussiana (Imhof) multiplicada pelo fator de calibração empírico/gaussiano medido nas
28 (0,99). Regra dura, $\tau=113{,}4$. Bootstrap de 120 reamostras das 38 vozes, reajustando $\tau$ em cada uma.

| $N$ | 128 | 247 | 512 | 1000 | 2048 | 2456 | $10^4$ | $10^5$ | piso $q_\infty$ |
|---|---|---|---|---|---|---|---|---|---|
| $q$ (livro-código i.i.d.) | 0,247 | 0,236 | 0,218 | 0,201 | 0,185 | 0,182 | 0,163 | 0,156 | 0,105 |
| IC 90% bootstrap | | | 0,13–0,30 | | | 0,105–0,26 | 0,095–0,24 | | 0,05–0,18 |

O valor em 128 (0,247) está acima do observado (0,211) porque sortear com reposição de 127 vozes cobre só ~63%
delas; o que vale é a **queda prevista** de $N=128$ a 2456: −0,06 (gaussiano com referências reais: −0,06 também).
**Previsão: $q(2456)\approx0{,}15$ (0,10–0,22).**

### 6.2 Bits por voz [`a6_bits.py`]

$L(c)=-\log_2p_c$ (livro-código aleatório) e $R_{\rm wf}(c)$ = taxa de água reversa da fonte das referências na
tolerância $t(c)$ (melhor livro-código possível):

| fração de vozes copiadas | 50% | 75% | 80% | 85% | ≥90% |
|---|---|---|---|---|---|
| bits, livro-código aleatório $L$ | 2,0 | 7,5 | 10,1 | 15,9 | ∞ (piso) |
| locutores sorteados $\approx2^L$ | 4 | 180 | ~1.100 | ~60.000 | — |
| bits, livro-código ótimo $R_{\rm wf}$ | 0 | 5,4 | 7,8 | 12,8 | ∞ |
| locutores escolhidos $\approx2^{R}$ | 1 | 40 | ~220 | ~7.000 | — |

Diferença mediana aleatório − ótimo: 2,2 bits (fator ~5 em $N$). $N_{90\%}=N_{95\%}=\infty$ no ponto estimado (piso
0,105); no 5º percentil do bootstrap, $N_{90\%}\approx7{,}7\times10^3$.

**Relação com os ~17 bits / 7–10 dimensões** [HEURÍSTICO]:
- O que o decodificador precisa resolver não são os 17 bits de $\mu$ (descrição fina da voz, inclusive ruído de
  frase), mas a voz **até a tolerância** $t(c)$: 2 bits para a voz mediana, 8–10 bits para chegar a 80%. Os 8 bits
  ficam em ~8 direções (1,6; 3,0; 4,3; 5,3; 6,2; 7,0; 7,5; 8,0 bits acumulados nas 8 primeiras), de acordo com as
  6–8 direções úteis e a dimensão intrínseca 7–10.
- **Saturação**: uma referência carrega $I(v;c)=10{,}3$ bits. Com $N\approx2^{10}$, $D_{\rm rand}\approx11{,}6$ já é
  da ordem do ruído de referência ($\mathrm{tr}W_x+32/K\approx12{,}9$): acima de $N^\star\approx2^{I(v;c)}\approx10^3$,
  o livro-código fica mais fino que aquilo que a referência determina, e mais locutores rendem pouco. Vozes que exigem
  $L>I(v;c)$ (as 15–20% mais difíceis) não são resolvidas por cobertura com referência única [CONJECTURA].

---

## 7. Parte 6 — consequências para a arquitetura

### 7.1 Módulo de voz de dimensão $k$ [`a5`, DADOS + modelo]

O módulo renderiza só a projeção nas $k$ maiores direções de $\Sigma_b$; a parte descartada soma à distorção:
$D^{(k)}_N(c)=\min_i\|P_k(c-v_i)\|^2+\|P_\perp(c-\mu)\|^2$. Piso de taxa-distorção (livro-código perfeito no
subespaço): $q_{\rm floor}(k)=\Pr\big(\|P_\perp(c-\mu)\|^2>t(c)\big)$, com $\mathbb E\|P_\perp(v-\mu)\|^2=\sum_{j>k}\lambda_j$.

| $k$ | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| $\sum_{j>k}\lambda_j$ | 24,3 | 18,5 | 13,3 | 10,2 | 5,9 | 3,8 | 1,3 | 0,5 | 0 |
| piso $q_{\rm floor}(k)$ | 0,29 | 0,24 | 0,21 | 0,18 | 0,18 | 0,16 | 0,13 | 0,105 | 0,105 |
| $q$ com o livro real, $N=8$ | 0,29 | 0,24 | 0,29 | 0,29 | 0,32 | 0,32 | 0,34 | 0,34 | 0,37 |
| $N=32$ | 0,29 | 0,24 | 0,26 | 0,24 | 0,24 | 0,24 | 0,26 | 0,26 | 0,26 |
| $N=128$ | 0,29 | 0,24 | 0,21 | 0,21 | 0,21 | 0,21 | 0,21 | 0,21 | 0,21 |

(regra dura; o ponto $k=32$ reproduz 0,37/0,26/0,21.) Leitura [HEURÍSTICO]: reduzir a dimensão troca **cobertura
mais rápida** (em $k$ dimensões o vizinho fica mais perto: $k=4$ ganha 0,08 em $N=8$) por **piso mais alto**
(cauda descartada). Com $N=128$, $k\ge3$ não perde nada; com $N$ grande o piso manda e é preciso $k\gtrsim12$ para
o piso cair ao do atrator (0,105). A dimensão ótima cresce com $N$: $k^\star(N)\approx$ número de direções com
$\lambda_j$ acima da distorção de cobertura, a mesma lógica da água reversa ($\lambda_j>\theta(\log_2N)$).

### 7.2 Etapa de voz treinada com locutores sem transcrição [modelo; DADOS]

Na proposta em duas etapas o livro-código é o conjunto de $N_{\rm voz}$ locutores de áudio sem texto; a fórmula vale
com $N\to N_{\rm voz}$, $K\to K_{\rm voz}$. Dois casos:
- **Mantendo uma voz dominante** (o mesmo atrator): $q(N_{\rm voz})$ segue a tabela 6.1 e para no piso ≈0,10.
  O ganho é só de escala de dados: 2.456 → $10^4$ locutores dá 0,18→0,16 no cálculo i.i.d.
- **Sem voz dominante** ($M_B\approx K$, $v_0\to\mu$, $\tau\to0$): "copiada" passa a ser "o livro-código bate o código
  de taxa zero", $D_N(c)<\|c-\mu\|^2$. Com as mesmas 38 referências e os livros-código reais: $q=0{,}45/0{,}11/0{,}00$
  em $N=8/32/128$ (semi-paramétrico: 0,41/0,08/0,003). **O ganho principal viria de remover o atrator, não de mais
  locutores.** Ressalva: o critério `sim_ref > sim_default` deixa de existir sem voz padrão; aqui comparo com a voz
  média, e a tradução para a métrica real é [CONJECTURA].

---

## 8. Parte 7 — previsões falsificáveis

| # | Experimento | Previsão desta teoria | Contraste | Decide |
|---|---|---|---|---|
| A1 | $N=512$, $K=32$ (≥200 vozes de avaliação) | $q=0{,}18$–0,19 (0,13–0,30) | 02 cobertura: ~0,14; potência: 0,15 | fraco sozinho |
| A2 | $N=2048$ e 2456 | $q\approx0{,}15$ (0,10–0,22); curva achatando | 02: 0,09; potência 0,10 | $q(2456)<0{,}09$ refuta o piso |
| A3 | $N\to10^4$ (ou duas etapas com $N_{\rm voz}=10^4$ mantendo a LJ dominante) | $q\to$ piso 0,10–0,15 | sem piso: <0,07 | piso |
| A4 | $M$ à metade (1.000 frases LJ), $N=128$ | $\Delta q=-0{,}01$ ($T=1$) a $-0{,}06$ ($T=3$); se $\tau=\|\mu-v_0\|^2$ for geométrico, $\Delta q\approx0$ | 02: −0,025 | mede $T$; $\Delta q<-0{,}08$ refuta o regime de distorção |
| A5 | $K=8$ e $K=128$ com $N=128$ (vozes novas) | 0,26–0,30 e 0,16–0,18 (referência 0,25 no mesmo cálculo) | regime de massa: dependência maior | forma em $K$ |
| A6 | Latente de voz $k=4$ (projeção nas 4 primeiras direções) | $N=8$: melhora ~0,08 (0,29 contra 0,37); $N=128$: igual (0,21); piso sobe para ~0,18 | 03: $k=8$ sem perda | troca cobertura × piso |
| A7 | $k=2$ | $q\approx0{,}24$ em todo $N$ (piso) | | |
| A8 | Média de $R=16$ referências | sem efeito sistemático ($|\Delta q|<0{,}03$), porque $\|\xi\|^2$ cancela | modelo paramétrico: ganho | já consistente com $R=4$ |
| A9 | Duas etapas sem voz dominante | $q(N_{\rm voz}=128)<0{,}05$ na métrica "ref. contra voz média" | uma etapa: 0,21 | atrator × cobertura |
| A10 | Repetir os três treinos `-wide` com outra amostra de locutores | $q_8$ varia ±0,045, $q_{32}$ ±0,02 só pela escolha das vozes | | separa sorteio de semente |
| A11 | Embeddings Resemblyzer das 2.456 vozes | $D_N$ no espaço da métrica cai ~$N^{-0,25}$ até $N\sim10^3$ e depois satura no ruído de frase | | liga espaço latente e métrica |

---

## 9. Situação de cada afirmação

| Afirmação | Situação |
|---|---|
| Canal de Gibbs = passo de Blahut–Arimoto com massas de treino como marginal | PROVADO (identidade de forma); que o Tacotron se comporte assim é CONJECTURA |
| Decodificador MSE-ótimo = Gibbs com $T=1$ e massas nominais (Prop. 1) | PROVADO para nuvens gaussianas; **refutado como descrição dos dados** |
| Limite de Poisson → regra de distorção (Prop. 2) e fórmula $\mathbb E_c(1-F_c)^N$ | PROVADO; SIMULADO (erro ~0,01 com $T=1$) |
| Forma de espectro de informação $\Pr(L>\log_2N)$ | PROVADO como cota/aproximação; SIMULADO (0,01–0,03) |
| $D(R)\propto(\ln N)^{-(a-1)}$, expoente local $(a-1)/\ln N$ | HEURÍSTICO (Stirling); SIMULADO aproximado |
| Livro-código aleatório paga 2–4 bits | SIMULADO (espectro medido e $a=1$–2) |
| Alta resolução não atingida até $N=4096$ | SIMULADO |
| Teorema tauberiano e $q-q_\infty\sim N^{-2/d}$ | PROVADO (abeliano) dado $H(x)\sim x^\beta$; $\beta=2/d$ HEURÍSTICO; SIMULADO |
| Regime de distorção ($T\lesssim3$) em vez de massa | DADOS (perfil da seção 3) |
| $\tau\approx\|\mu-v_0\|^2$ | DADOS *post hoc* (3 espaços); interpretação CONJECTURA |
| Previsão de $q_8,q_{32}$ a partir de $q_{128}$ | DADOS: erro 0,03–0,06 ($T=1$), 0,01–0,02 ($T=3$) |
| Hipótese gaussiana para $P_v$ | **refutada quantitativamente** ($D_N$ real menor) |
| Cancelamento do ruído de referência (média de referências não ajuda) | PROVADO; consistente com DADOS |
| Grade de 4 vozes e efeito de $K$ nas treinadas | **falha** (viés −0,19/−0,24; benefício de $K$ subestimado) |
| Piso 0,10 e $q(2456)\approx0{,}15$ | HEURÍSTICO (extrapolação semi-paramétrica, IC largo) |
| Bits por fração copiada; $N^\star\approx2^{I(v;c)}$ | HEURÍSTICO / CONJECTURA |
| Piso do módulo de dimensão $k$ | PROVADO no modelo; números dependem do espaço latente |

**O que não está identificado.** Com três pontos e uma semente, expoente e piso não se separam (seção 2.3); $T$ e a
massa efetiva da voz padrão não se separam no regime de distorção (só $\tau$); a identidade $\tau=\|\mu-v_0\|^2$
foi notada depois do ajuste e confere em três espaços, que não são independentes (mesmos dados, codificadores
parecidos). A escolha do espaço latente do treino $N=128$ como "espaço de vozes" é a hipótese mais forte; os
números mudam em 0,02–0,04 se cada treino for avaliado no próprio espaço.

---

## Apêndice — arquivos

Todos em `/tmp/claude-0/-home-user-ml2-final-project/4c11fa8d-6b5f-5505-bd7b-deebf5dd9907/scratchpad/teoria2/A/`:

| Script | Faz | Saída |
|---|---|---|
| `rd_lib.py` | leitura de latentes/resultados; espaço branqueado; $\Sigma_b$ por componentes de variância | — |
| `rd_theory.py` | Imhof, ponto de sela (Lugannani–Rice), Chernoff, água reversa | — |
| `a1_geometria.py` | espectro, $v_0$, ruído, distâncias, livros-código aninhados | `out/a1.json`, `out/a1_pervoice_N*.csv` |
| `a2_gibbs_dados.py` | grade $(T,\ln M)$ do canal de Gibbs nos livros-código reais; perfil | `out/a2.json`, `out/a2_grade.csv` |
| `a3_modelo_codigo.py` | calibração de $\tau$ (externas e treinadas), população gaussiana, $D_N$ real × gaussiano | `out/a3.json` |
| `a4_tau_zero_param.py` | $\tau_0=\|\mu_N-v_0\|^2$ em cada espaço próprio (17 treinos) | `out/a4.csv` |
| `a5_extrapola.py` | semi-paramétrico, subconjuntos, bootstrap, $K$/$M$, módulo $k$, sem voz dominante | `out/a5.json` |
| `a6_bits.py` | $L(c)$, $R_{\rm wf}(c)$, bits por direção | `out/a6.json` |
| `a7_grade.py` | grade $N\times K$ sem ajuste | `out/a7.csv` |
| `a8_tau0_comum.py` | $\tau_0$ no espaço comum | `out/a8.json` |
| `t1_rd_teoria.py` | $D(R)$ ótimo × aleatório × ponto de sela × alta resolução; verificação Imhof/sela/MC | `out/t1.json` |
| `t2_verifica_formula.py` | fórmula de $q(N,K,M)$ contra Gibbs simulado | `out/t2.json` |
| `t3_tauberiano.py` | lei $q-q_\infty\sim N^{-2/d}$ | `out/t3.json` |

Dados lidos (só leitura): `scratchpad/coverage2/<run>/step_2500/{latents.npz,train_manifest.csv}` e
`reports/coverage/results/<run>/step_2500/strength_1.0/results.csv`.
