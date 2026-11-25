# Teoria de escala para clonagem de voz *zero-shot*: derivação

Documento de trabalho, 17/11/2026. Complementa `reports/teoria_cobertura.md` (fatos e previsões registradas),
`reports/proposta/proposta.tex` (proposta) e `01_literatura.md` (levantamento). Todas as simulações são de CPU
e estão nesta pasta (`sim_*.py`, saídas em `out/sim_*.txt`). Nenhum arquivo do repositório foi alterado.

**Legenda de status** (vale para cada afirmação marcada):

- **[PROVADO]**: segue das hipóteses enunciadas por uma conta completa (indicada).
- **[HEURÍSTICO]**: conta com aproximações não controladas (lei dos grandes números local, linearização, etc.).
- **[CONJECTURA]**: hipótese de modelagem, não derivada; é o que precisa ser testado.
- **[SIMULADO]**: conferido numericamente num modelo sintético (script indicado). Simulação confere a *conta*, não a *hipótese de modelagem*.
- **[DADOS]**: calculado dos `results.csv` por amostra (somente leitura).

---

## 0. Resumo

1. **A métrica binária é linear na saída** [PROVADO]. Com embeddings de locutor normalizados, a margem
   $m=\mathrm{sim}_{\rm ref}-\mathrm{sim}_{\rm default}=\langle \hat e,\ e_{\rm ref}-e_0\rangle=(\tfrac12-\rho)\,\|\Delta\|^2$,
   onde $\rho$ é o **encolhimento da voz gerada em direção à voz padrão**, medido só ao longo do eixo
   $\Delta=e_{\rm ref}-e_0$. Erros ortogonais a esse eixo não contam. "Copiada" $\iff \rho<\tfrac12$ (a menos de ruído).
2. **Elo binário** [PROVADO dado ruído gaussiano dentro da voz; DADOS]: $q=\mathbb E_s\,\Phi(-\mu_s/\sigma_w)$ com
   $\sigma_w\approx0{,}036$ (constante em $N$) e desvio entre vozes $\tau_\mu\approx0{,}08$–$0{,}10$. Com as margens medidas,
   prevê $q=0{,}442/0{,}304/0{,}217$ contra $0{,}434/0{,}302/0{,}211$ observados (38 vozes, $N=8/32/128$).
   O "tudo-ou-nada" por voz é $\tau_\mu/\sigma_w\approx2{,}5$.
3. **Três regimes** para $\rho$ (ou para o erro), com a mesma estrutura $E=\text{viés}(N)+\text{variância}(N)\,s^2(K)$,
   $s^2(K)=\sigma_b^2+\sigma^2/K$:
   - (a) paramétrico: $E_{\rm nova}=\sigma_b^2+s^2(K)\,\frac{d}{N-d-1}$ [PROVADO, SIMULADO] $\Rightarrow a+A/N+B/(NK)$ com $A/a=d$, $B/A=\sigma^2/\sigma_b^2$;
   - (b) cobertura: $E\sim N^{-2/d}$ (vizinho mais próximo), $(s^2/N)^{2s/(2s+d)}$ (núcleo) [conhecido; SIMULADO];
   - (c) espectral ($\lambda_j\propto j^{-\alpha}$): $E\sim (s^2/N)^{(\alpha-1)/\alpha}$ [HEURÍSTICO, padrão; SIMULADO aprox.].
4. **Atrator da voz padrão** [PROVADO para o suavizador de Nadaraya–Watson; CONJECTURA que o Tacotron se comporte assim]:
   $\rho(v)=\dfrac{M\,k(v,v_0)}{M\,k(v,v_0)+K\sum_i k(v,v_i)}$ ($M=2000$ frases LJ). Com poucos locutores:
   **copiada $\iff \|v-v_0\|^2-\min_i\|v-v_i\|^2>2\ell^2\ln(M/K)$**. Com muitos: depende de $NK/M$.
5. **Modelo único** (cobertura + atrator, vozes com espectro $\lambda_j=j^{-1}$, $d=16$) reproduz, com 5 parâmetros
   escolhidos numa grade para 3 pontos, a curva $0{,}43/0{,}29/0{,}21$, o tudo-ou-nada ($\tau_\mu/\sigma_w=2{,}5$–$2{,}8$), e os
   **sinais** das correlações por voz (distância à padrão +, F0 −, vizinho treinado −) [SIMULADO]. O expoente local
   fica em $0{,}23$–$0{,}31$ de $N=8$ a $10^4$, sem piso. Sem o espectro em lei de potência, nenhuma configuração
   isotrópica reproduz ao mesmo tempo o expoente lento e o tudo-ou-nada.
6. **Previsões mais afiadas**: (i) $q(2456)\approx0{,}09$ (cobertura) contra $\ge0{,}16$ (leis com piso), decidível com
   $\ge200$ vozes de avaliação; (ii) **reduzir as frases LJ à metade equivale a dobrar $K$** para vozes novas; (iii) em
   duas etapas, a cópia segue $N_{\rm voz}$, e um vazamento de voz nas unidades com peso $w$ cria um piso
   ($w=0{,}2\Rightarrow q\gtrsim0{,}15$; exige $I(\text{unidades};\text{locutor})\lesssim0{,}7$ bit por dimensão efetiva);
   (iv) um módulo de voz paramétrico de dimensão $k$ tem piso $q_{\rm floor}(k)=\Pr\big(\|P_\perp(v-\mu)\|>\|P_k(v-v_0)\|\big)$,
   calculável sem treino a partir do espectro dos embeddings.

---

## 1. O que a teoria precisa explicar

Fatos (de `teoria_cobertura.md` e da proposta), mais regularidades que medi agora nos `results.csv`
(`sim_0_dados.py`, saída na seção 1.2).

### 1.1 Fatos registrados

| # | Fato |
|---|---|
| F1 | Vozes novas: não copiadas caem com $N$ (4 vozes, ruidoso; 38 vozes: $0{,}434\to0{,}302\to0{,}211$ para $N=8,32,128$, $K=32$; potência aparente $N^{-0{,}26}$). |
| F2 | Vozes treinadas: $\approx0{,}14+2{,}87/K+0{,}22/N$ (dependem de $K$). |
| F3 | WER constante (12–13%) em $N$ nas vozes novas. |
| F4 | Dificuldade estável por voz (postos 0,6–0,75 entre $N$); cópia correlaciona com distância à voz padrão (+0,5) e com F0 (−0,4 a −0,5; a padrão é feminina); distância ao vizinho treinado prevê falha em $N=8,32$ ($-0{,}33$, $-0{,}40$), não em 128 ($+0{,}13$). Cópia quase tudo-ou-nada por voz. |
| F5 | Negativos: vetor ECAPA ignorado ou decorado; média de 4 referências não ajuda nas 38 vozes; pseudo-locutores por velocidade não contam; troca de referência e perda contrastiva não ajudam. |
| F6 | Latente: ~70 bits/frase (KL), 10–20 bits sobre o locutor, crescendo ~$\log N$. |

### 1.2 Regularidades novas medidas nos dados [DADOS]

Margem por amostra $m=\mathrm{sim}_{\rm ref}-\mathrm{sim}_{\rm default}$ (Resemblyzer), 38 vozes externas, $K=32$:

| $N$ | $q$ | margem média | desvio **entre** vozes $\tau_\mu$ | desvio **dentro** da voz $\sigma_w$ | assimetria |
|---|---|---|---|---|---|
| 8 | 0,434 | −0,004 | 0,101 | 0,036 | −0,68 |
| 32 | 0,302 | 0,048 | 0,078 | 0,036 | −0,12 |
| 128 | 0,211 | 0,078 | 0,088 | 0,037 | +0,07 |

- $\sigma_w$ é a mesma nos três treinos; o que muda com $N$ é a posição das margens por voz.
- Correlação das margens por voz entre treinos: 0,75 (8–32), 0,85 (32–128), 0,65 (8–128).
- O ganho de margem de 8 para 128 correlaciona $-0{,}55$ com a margem em $N=8$: as vozes piores ganham mais.
- O expoente é mal determinado por 3 pontos: pela margem média, $\beta=0{,}39$ (IC 95% de $-0{,}09$ a $0{,}94$, reamostrando vozes); pelo probit de $q$, $0{,}13$ ($-0{,}83$ a $1{,}39$). **O número 0,26 é uma descrição, não uma medida precisa.**
- **A variação entre sementes é uma voz que vira.** Nas 4 vozes fixas da grade, as sementes 1 e 2 usam os mesmos locutores; a cópia por voz é, por exemplo, $128\times32$: {696: 0,76/0,72; 4297: 1,0/1,0; **6415: 1,0/0,52**; 7800: 0,2/0,0}; $8\times128$: {0,28/0,12; 1,0/1,0; **0,76/0,36**; 0/0}. Duas vozes são sempre copiadas ou nunca; a diferença entre sementes vem quase toda de uma voz perto do limiar.

---

## 2. Modelo gerativo comum e a geometria da métrica

### 2.1 Hipóteses

- **H1 (espaço de vozes)** [CONJECTURA]. Cada locutor $s$ tem uma voz $v_s\in\mathbb R^D$, identificada com o seu embedding
  de locutor médio (o espaço onde a métrica mede). As vozes vêm de uma população $P$ com média $\bar v$, covariância
  $\Sigma_v$ de autovalores $\lambda_1\ge\lambda_2\ge\dots$, e dimensão intrínseca $d$ (ou espectro $\lambda_j\propto j^{-\alpha}$).
- **H2 (frases)**. A frase $j$ do locutor $s$ realiza a voz $v_s+b_s+\varepsilon_{sj}$: $b_s\sim(0,\sigma_b^2 I)$ é a parte
  idiossincrática que não cabe no código; $\varepsilon_{sj}\sim(0,\sigma^2 I)$ é o ruído de frase. A média de $K$ frases tem ruído
  $s^2(K)=\sigma_b^2+\sigma^2/K$ em torno da parte "explicável" $v_s$.
- **H3 (treino)**. $M=2000$ frases da voz padrão $v_0$ (LJSpeech, feminina) + $N$ locutores $\times K$ frases. O modelo
  parte de um checkpoint de um só locutor, que renderiza $v_0$ para qualquer código.
- **H4 (inferência)**. Uma referência dá um código $c=v^\*+\xi$ (ruído de codificador $\sigma_\xi^2$; $R$ referências: $\sigma_\xi^2/R$). O decodificador
  renderiza uma voz $\hat y(c)$; cada frase gerada sai com ruído $\eta\sim(0,\sigma_\eta^2 I)$.
- **H5 (erro)**. Erro de renderização numa voz: $E=\mathbb E\|\hat y-v^\*\|^2$. Vozes treinadas: $v^\*=v_i$; novas: $v^\*\sim P$.

### 2.2 Lema 1: a métrica mede só o encolhimento para a voz padrão [PROVADO]

Sejam $\hat e$, $e_{\rm ref}$, $e_0$ embeddings de norma 1 (o Resemblyzer normaliza) da fala gerada, da referência e da voz
padrão. Então
$$
m=\mathrm{sim}_{\rm ref}-\mathrm{sim}_{\rm default}=\langle\hat e,e_{\rm ref}\rangle-\langle\hat e,e_0\rangle=\langle \hat e,\Delta\rangle,\qquad \Delta=e_{\rm ref}-e_0 .
$$
Defina o encolhimento ao longo do eixo $\rho=1-\langle \hat e-e_0,\Delta\rangle/\|\Delta\|^2$. Como
$\langle e_0,\Delta\rangle=\langle e_0,e_{\rm ref}\rangle-1=-\|\Delta\|^2/2$,
$$
\boxed{\,m=\big(\tfrac12-\rho\big)\,\|\Delta\|^2\,}
$$
Equivalentemente, $m=\tfrac12\big(\|\hat e-e_0\|^2-\|\hat e-e_{\rm ref}\|^2\big)$. Consequências:

1. Qualquer erro ortogonal a $\Delta$ (timbre errado numa direção que não separa a referência da padrão) **não altera** $m$.
   A taxa de cópia mede o encolhimento para a voz padrão, não o erro total de renderização.
2. "Copiada" $\iff\rho<\tfrac12$: a voz gerada passou do ponto médio entre a padrão e a referência.

Com ruído de frase $\eta$ (H4) e $\hat e=\bar e+\eta$: $m=(\tfrac12-\rho)\|\Delta\|^2+\langle\eta,\Delta\rangle$. Se $\eta$ é gaussiano isotrópico,
$$
\Pr(\text{copiada}\mid\text{voz})=\Phi\!\Big(\frac{(1-2\rho)\,\|\Delta\|}{2\sigma_\eta}\Big). \tag{2.1}
$$
A dependência da distância à voz padrão vem daqui: com o mesmo $\rho<\tfrac12$, vozes longe da padrão são copiadas com
mais certeza. Vozes muito perto da padrão ($\|\Delta\|\lesssim2\sigma_\eta$) ficam perto de 50% mesmo com cópia perfeita
($\rho=0$): é o **piso da métrica**, separado do piso do modelo.

**Medida sem treino que isto habilita.** Para cada voz, $\rho_s=\tfrac12-\mu_s/\|\Delta_s\|^2$, com $\mu_s$ a margem média (já medida)
e $\|\Delta_s\|^2=2(1-\cos(e_{\rm ref},e_0))$ (basta embutir as referências e a voz padrão). Se $\cos(e_{\rm ref},e_0)\approx0{,}68$
($\|\Delta\|^2\approx0{,}65$; valor suposto, a medir), o encolhimento médio é $\bar\rho\approx0{,}51/0{,}43/0{,}38$ para $N=8/32/128$ [HEURÍSTICO].

### 2.3 Proposição 1: elo contínuo → binário [PROVADO dado H; DADOS]

Seja $\mu_s=\mathbb E[m\mid s]$ e $m=\mu_s+w$, $w\sim\mathcal N(0,\sigma_w^2)$ dentro da voz. Então a fração não copiada é
$$
q=\mathbb E_s\big[\Phi(-\mu_s/\sigma_w)\big]. \tag{2.2}
$$
Se além disso $\mu_s\sim\mathcal N(\bar\mu,\tau_\mu^2)$, $q=\Phi\big(-\bar\mu/\sqrt{\tau_\mu^2+\sigma_w^2}\big)$.

Verificação com as margens por voz medidas e $\sigma_w=0{,}036$ (`sim_6_previsoes.py`):

| $N$ | $q$ observado | (2.2) com $\mu_s$ medidas | probit gaussiano |
|---|---|---|---|
| 8 | 0,434 | 0,442 | 0,514 |
| 32 | 0,302 | 0,304 | 0,289 |
| 128 | 0,211 | 0,217 | 0,206 |

A forma (2.2) com as margens por voz acerta a menos de 1 ponto; a versão gaussiana erra em $N=8$ porque as margens
são assimétricas ($-0{,}68$). **Tudo-ou-nada**: a fração de vozes com cópia entre 10% e 90% é
$\Pr(|\mu_s|<1{,}28\,\sigma_w)\approx 2\cdot1{,}28\,\sigma_w\, g(0)$, com $g$ a densidade das margens por voz; com $\tau_\mu/\sigma_w=2{,}5$
isso dá cerca de 0,35–0,40, e medi 0,34/0,37/0,26.

---

## 3. Regime (a): paramétrico / hierárquico (efeitos aleatórios lineares)

### 3.1 Teorema 1 [PROVADO]

Hipóteses: código $u_s\sim\mathcal N(0,I_d)$, saída $y_{sj}=Wu_s+b_s+\varepsilon_{sj}\in\mathbb R^D$ (H2), intercepto
conhecido (fixado pelas $M$ frases da padrão), $W$ estimado por mínimos quadrados nas médias por locutor
$\bar y_s=Wu_s+e_s$, $e_s\sim\mathcal N(0,s^2(K)I_D)$, $N>d+1$. Para uma voz nova $u^\*\sim\mathcal N(0,I_d)$:
$$
\frac1D\,\mathbb E\|\hat Wu^\*-(Wu^\*+b^\*)\|^2=\sigma_b^2+s^2(K)\,\frac{d}{N-d-1}. \tag{3.1}
$$
*Prova.* $\hat W-W=E^\top U(U^\top U)^{-1}$, com $E$ independente de $U$; logo
$\mathbb E\|(\hat W-W)u^\*\|^2=Ds^2\,\mathbb E[u^{\*\top}(U^\top U)^{-1}u^\*]=Ds^2\,\mathrm{tr}\,\mathbb E(U^\top U)^{-1}=Ds^2\,d/(N-d-1)$
(média da Wishart inversa). O termo $b^\*$ é imprevisível e soma $D\sigma_b^2$. $\square$

Expandindo, para $N\gg d$:
$$
E_{\rm nova}\approx\underbrace{\sigma_b^2}_{a}+\underbrace{\frac{d\,\sigma_b^2}{N}}_{A/N}+\underbrace{\frac{d\,\sigma^2}{NK}}_{B/(NK)},\qquad \frac{A}{a}=d,\quad \frac BA=K_0\equiv\frac{\sigma^2}{\sigma_b^2}. \tag{3.2}
$$
O termo $A/N$ só existe porque há variação **entre locutores** que o código não explica ($\sigma_b^2>0$): é o ruído de
conglomerado da amostragem em dois estágios. Se $\sigma_b=0$, sobra só $d\sigma^2/(NK)$ (a forma de Du et al. 2021).

**Vozes treinadas** (BLUP: tabela por locutor encolhida para $\hat Wu_s$, com $\lambda_K=\sigma_b^2/s^2(K)$) [HEURÍSTICO: desprezo $O(d/N^2)$]:
$$
E_{\rm treinada}\approx\frac{\sigma_b^2\,\sigma^2/K}{\sigma_b^2+\sigma^2/K}+(1-\lambda_K)^2 s^2(K)\frac dN\ \xrightarrow{K\gg K_0}\ \frac{\sigma^2}{K}+O\!\Big(\frac{d}{NK^2}\Big). \tag{3.3}
$$

**Piso do codificador**: com código ruidoso $c=u^\*+\xi$, soma-se $\|W\|^2\sigma_\xi^2/R$: **neste regime, a média de $R$
referências tem de ajudar** [PROVADO].

**Simulação** (`sim_1_linear_gauss.py`, $d=4$, $D=24$, $\sigma_b^2=0{,}05$, $\sigma^2=0{,}5$, 400 réplicas) [SIMULADO]:
(3.1) bate com a simulação em todas as 24 células com $N\ge16$ (ex.: $N=32,K=8$: 0,0667 contra 0,0667); em $N=8$ a
diferença chega a 8% (cauda da Wishart inversa com $N-d-1=3$). (3.3) bate em todas as células. O ajuste de
$a+A/N+B/(NK)$ aos dados simulados recupera $a=0{,}049$ (verdadeiro 0,050), $K_0=8{,}9$ (10) e $\hat d=A/a=5{,}3$ (4):
o viés de $\hat d$ vem de aproximar $d/(N-d-1)$ por $d/N$. Inclinação do excesso: $-1{,}09$ e $-1{,}11$ (prevista $-1$). Média de
referências: $R=1,4,16$ dá 0,358/0,129/0,071 contra 0,352/0,127/0,071 previstos.

### 3.2 Leitura dos dados neste regime [HEURÍSTICO]

Se $q$ fosse proporcional ao erro, o ajuste da grade ($a=0{,}39$, $A=1{,}08$, $B=10{,}3$) daria $d\approx A/a\approx2{,}8$ e
$K_0\approx9{,}5$. Três coisas pesam contra este regime ser o gargalo: (i) a média de referências não ajuda nas 38 vozes
(o regime (a) exige que ajude); (ii) o expoente medido é bem menor que 1; (iii) o regime (a) dá encolhimento quase
uniforme entre vozes (atenuação por erro no código, $\rho_a\approx\sigma_\xi^2/(1+\sigma_\xi^2)$), e não explica a dependência
na distância ao vizinho treinado.

---

## 4. Regime (b): cobertura (o decodificador interpola entre vozes treinadas)

### 4.1 Taxas conhecidas [conhecido da literatura; SIMULADO]

Se a renderização $g$ é $L$-Lipschitz (ou $s$-Hölder) num espaço de dimensão intrínseca $d$ e o decodificador
renderiza a voz nova a partir das vozes treinadas vizinhas:

- vizinho mais próximo (sem ruído): $E\asymp L^2\,\mathbb E\,r_{\rm nn}^2\asymp N^{-2/d}$;
  com ruído, $E\to s^2(K)+O(N^{-2/d})$: **$K$ entra só no piso** (o vizinho não faz média);
- suavização por núcleo com banda ótima: $E-\sigma_b^2\asymp\big(s^2(K)/N\big)^{2s/(2s+d)}$, $s=2$ para núcleo simétrico e $g$ suave.

`sim_2_naoparam.py` ($v\sim U[0,1]^d$, $g=\sin(Av+c)$): vizinho mais próximo, inclinações $-1{,}01/-0{,}53/-0{,}23$ contra
$-1/-0{,}5/-0{,}25$ ($d=2,4,8$); Nadaraya–Watson com banda ótima, $-0{,}69/-0{,}51/-0{,}28$ contra $-0{,}67/-0{,}50/-0{,}33$.
No regime limitado por viés, **$N$ domina e $K$ quase não importa**: $N=64,K=128$ (erro 0,079) é pior que
$N=256,K=8$ (0,043), embora tenha $N/s^2$ menor. O expoente $0{,}26$ corresponderia a $d\approx8$ (vizinho, $2/d$) ou
$d\approx11$ (núcleo, $4/(4+d)$), se $q$ fosse proporcional ao erro, o que o Lema 1 desmente (seção 7).

### 4.2 Proposição 2: o atrator num suavizador por frase [PROVADO para Nadaraya–Watson]

Modelo do decodificador [CONJECTURA]: a voz renderizada para o código $c$ é uma média ponderada **por frase de treino**
das vozes que o decodificador ouviu, com núcleo $k(c,v)=e^{-\|c-v\|^2/2\ell^2}$:
$$
\hat y(c)=\frac{M\,k(c,v_0)\,v_0+K\sum_i k(c,v_i)\,\bar y_i}{M\,k(c,v_0)+K\sum_i k(c,v_i)}. \tag{4.1}
$$
O peso da voz padrão é $\omega_0(c)=Mk(c,v_0)/Z(c)$. No limite em que a média local das vozes treinadas é a própria
voz pedida, $\hat y-v_0=(1-\omega_0)(v^\*-v_0)$, e pelo Lema 1 $\rho=\omega_0$. Portanto
$$
\boxed{\ \text{copiada}\iff K\sum_i k(v^\*,v_i)>M\,k(v^\*,v_0)\ } \tag{4.2}
$$
"a massa local de frases de treino em torno da voz tem de superar a massa da voz padrão vista daquele ponto".
Fora desse limite, soma-se o viés da média local (puxa para o centro da população, não para $v_0$).

**Regime de Poisson (poucos locutores)**: a soma é dominada pelo vizinho mais próximo, $\sum_i k\approx e^{-r_{\rm nn}^2/2\ell^2}$, e (4.2) vira
$$
\boxed{\ \text{copiada}\iff \|v^\*-v_0\|^2-r_{\rm nn}(v^\*)^2>2\ell^2\ln\frac MK\ } \tag{4.3}
$$
A voz precisa estar mais perto de alguma voz treinada que da voz padrão, com uma desvantagem $2\ell^2\ln(M/K)$.
Isto dá, de uma vez, a correlação positiva com a distância à padrão e a negativa com a distância ao vizinho treinado
(F4), e explica por que $K$ pesa pouco nas vozes novas: entra só no logaritmo. Também explica F5 (ECAPA): um código
melhor não muda $r_{\rm nn}$; com 8 locutores o decodificador só renderiza vozes perto das 8 que ouviu ("decora").

**Regime de lei dos grandes números (muitos locutores)**: $\sum_i k(v,v_i)\approx N\,c_\ell\,\tilde p(v)$, com $\tilde p=p*k$ a
densidade suavizada. Então (4.2) depende de $(N,K,M)$ **só por $NK/M$**:
$$
\text{copiada}\iff \frac{NK}{M}\,c_\ell\,\tilde p(v^\*)\,e^{\|v^\*-v_0\|^2/2\ell^2}>1 . \tag{4.4}
$$
A região de falha tem duas partes: uma bola em torno de $v_0$ de raio$^2\approx2\ell^2\ln\!\big(M/(NKc_\ell\tilde p)\big)$ (vozes
capturadas pelo atrator) e a cauda da população onde $\tilde p$ é pequena (vozes atípicas, buracos de cobertura). Ambas
encolhem como **logaritmo** de $NK/M$ [HEURÍSTICO].

**Transição entre os dois** [HEURÍSTICO]: o número esperado de vozes treinadas dentro da escala $\ell$ de uma voz nova
gaussiana é $N\prod_j(1+2\lambda_j/\ell^2)^{-1/2}$. A dependência no vizinho mais próximo some quando ele passa de ~1:
$$
N^\*=\prod_j\Big(1+\frac{2\lambda_j}{\ell^2}\Big)^{1/2}. \tag{4.5}
$$
Os dados (correlação com o vizinho some entre $N=32$ e 128) sugerem $N^\*\sim50$–$100$; para vozes isotrópicas isso é
$d=6$, $\ell=0{,}75\tau$ ($N^\*=95$) ou $d=8$, $\ell=\tau$ (81) (`sim_9_piso_e_Nstar.py`). O modelo espectral que melhor
ajusta $q$ (seção 6) tem $N^\*\approx860$: tensão discutida na seção 12.

### 4.3 Versão processo gaussiano [HEURÍSTICO; SIMULADO]

Ajuste fino a partir do checkpoint de um locutor = regressão do resíduo $r(v)=g(v)-v_0$ com **média a priori $v_0$**
(processo gaussiano ou núcleo tangente) e uma observação quase sem ruído em $v_0$. Na região bem amostrada,
$\rho(v)\approx\tilde s^2/(\tilde s^2+n_{\rm loc}(v))$, $\tilde s^2=s^2(K)/a^2$: o efeito de $K$ satura em $\sigma_b^2$. Simulado (`sim_3_atrator.py gp`),
esta versão dá queda rápida demais ($q\approx0{,}54\to0{,}26\to0{,}10$) quando se exige o tudo-ou-nada; a versão por frase (4.1)
se sai melhor (seção 6).

---

## 5. Regime (c): espectral (autovalores em lei de potência)

### 5.1 Proposição 3 [HEURÍSTICO; resultados padrão de regressão por núcleo, Caponnetto–De Vito, Bordelon et al. 2020]

Vozes com covariância $\Sigma_v=\mathrm{diag}(\lambda_j)$, $\lambda_j=j^{-\alpha}$, $\alpha>1$; o decodificador aprende o mapa
(alvo: identidade) por ridge a partir de $N$ vozes com ruído $s^2(K)$. Uma direção $j$ é aprendida quando
$N\lambda_j\gtrsim\kappa$ ($\kappa$: regularização efetiva $\sim s^2$). O número de direções aprendidas é $J(N)\asymp(N/\kappa)^{1/\alpha}$ e
$$
E_{\rm nova}\asymp\sum_{j>J(N)}\lambda_j+\kappa\frac{J(N)}N\asymp\Big(\frac{s^2(K)}{N}\Big)^{(\alpha-1)/\alpha};\qquad
\text{sem ruído (interpolação)}:\ E\asymp N^{-(\alpha-1)}. \tag{5.1}
$$
`sim_7_espectral.py` ($D=1500$): com ruído e ridge ótimo, inclinações $-0{,}28/-0{,}42/-0{,}59$ contra $-0{,}33/-0{,}50/-0{,}67$
($\alpha=1{,}5/2/3$); sem ruído, $-0{,}85/-1{,}25/-2{,}12$ contra $-0{,}5/-1/-2$ (efeito de $D$ finito quando $N\to D$). Concordância
de 0,05–0,1 no expoente, com desvio na direção esperada de efeitos de tamanho finito [SIMULADO, aproximado].

### 5.2 O espectro medido no latente [DADOS da tentativa anterior, `out/speaker.json`]

Covariância entre locutores dos códigos (treinos com $N=128,200$): razão de participação 9–10 (ponderada por SNR) e 6
(bruta); 13–14 direções com SNR > 1; 6 direções explicam 90% da variância; expoente de decaimento
$\alpha_{\rm raw}\approx2{,}0$–$2{,}1$ e $\alpha_{\rm snr}\approx1{,}05$–$1{,}1$. Por (5.1), $\beta_E=(\alpha-1)/\alpha$ fica entre $0{,}05$ ($\alpha=1{,}05$) e
$0{,}5$ ($\alpha=2$): o 0,26 observado está dentro, mas o intervalo é largo demais para ser um teste. Ressalva: é o espectro
do latente aprendido, não o do espaço da métrica; o certo é medir o espectro dos embeddings Resemblyzer das 2.456
vozes do LibriTTS (sem treino; seção 10).

### 5.3 Unificação de (c) com o piso paramétrico [HEURÍSTICO]

Se o decodificador renderiza bem as $J$ direções aprendidas e nada nas outras, o Lema 1 dá
$\rho(v)=\sum_{j>J}\Delta_j^2/\|\Delta\|^2$ quando $v_0-\bar v$ está nas direções aprendidas, e
$$
q(N)\approx\Pr\Big(\textstyle\sum_{j>J(N)}(v-\bar v)_j^2>\sum_{j\le J(N)}(v-v_0)_j^2\Big)=q_{\rm floor}\big(J(N)\big). \tag{5.2}
$$
É a mesma função do piso do módulo paramétrico (seção 9.4), avaliada no número de direções que os dados resolvem.

---

## 6. O atrator da voz padrão e o modelo que junta tudo

### 6.1 Por que a voz padrão é um atrator [CONJECTURA de modelagem, consequências PROVADAS em (4.1)]

Duas vias, ambas presentes no nosso treino: (i) **média a priori**: o checkpoint de partida renderiza $v_0$ para qualquer
código, e o ajuste fino é curto (2.500 passos), então onde os dados não chegam a saída fica perto de $v_0$; (ii) **âncora
pesada**: as $M=2000$ frases de LJ são a maior massa individual do treino (de 98% do total em $8\times8$ a 33% em
$128\times32$). Em (4.1) as duas viram o mesmo termo $Mk(c,v_0)$.

O tom entra por $\Delta$: a padrão é feminina e aguda, então vozes agudas têm $\|\Delta\|$ menor e (i) caem mais facilmente na
bola de captura de (4.4), (ii) têm margem menor por (2.1) mesmo quando copiadas. Ambos os efeitos dão correlação
negativa entre cópia e F0 e positiva com a distância à padrão (F4).

### 6.2 Ajuste e teste do modelo cobertura + atrator [SIMULADO]

`sim_3_atrator.py`: vozes $v\sim\mathcal N(0,\mathrm{diag}(j^{-\alpha}))$ em $\mathbb R^d$, coordenada 0 = tom, $v_0=\delta e_0$; decodificador (4.1)
ou processo gaussiano; códigos com ruído $\sigma_\xi=0{,}2$, frases com ruído $\sigma_\eta$; 25 frases por voz; critério do Lema 1.

**Resultado negativo primeiro.** Com vozes isotrópicas ($\alpha=0$, $d=4$–16), nenhuma das 240 configurações ajusta os
três pontos mantendo o tudo-ou-nada: a melhor erra por 0,07 (RMSE) e decai com expoente ~0,45–0,5
(`out/sim_3_grade.txt`). Consegue-se o expoente lento só com ruído de frase grande ($\sigma_\eta\ge1$), que destrói o tudo-ou-nada
(fração de vozes entre 10% e 90% sobe para 0,9). **Um espaço de vozes isotrópico de baixa dimensão não explica F1+F4 juntos.**

Com espectro em lei de potência ($\alpha\in\{1,2\}$, $d\in\{16,32\}$), a melhor configuração do suavizador por frase é
$\alpha=1$, $d=16$, $\ell=0{,}5$, $\delta=2$, $\sigma_\eta=0{,}1$ (RMSE 0,017 nos três pontos; `out/sim_3_grade_espectro.txt`).
O $\alpha=1$ escolhido pela grade coincide com o $\alpha_{\rm snr}\approx1{,}05$–$1{,}1$ medido no latente, o que não foi imposto.
Rodando essa configuração com 6 sementes (`out/sim_3.txt`):

| Quantidade | Modelo ($d=16$) | Modelo ($d=32$, $\ell=0{,}3$) | Observado |
|---|---|---|---|
| $q$ em $N=8/32/128$ | 0,427 / 0,292 / 0,212 | 0,418 / 0,297 / 0,207 | 0,434 / 0,302 / 0,211 |
| $q$ em $N=247/1000/2456/10^4$ | 0,177 / 0,115 / 0,086 / 0,062 | 0,173 / 0,126 / 0,102 / 0,072 | — |
| expoente local | 0,23–0,31, sem piso | 0,23–0,27 | ~0,26 |
| $\tau_\mu/\sigma_w$ (tudo-ou-nada) | 2,5 / 2,7 / 2,8 | 2,1 | ~2,5 |
| vozes com cópia entre 10% e 90% | 0,27 / 0,22 / 0,20 | 0,34 / 0,29 / 0,27 | 0,34 / 0,37 / 0,26 |
| postos cópia × distância à padrão | +0,77 / +0,85 / +0,81 | +0,72 / +0,78 / +0,78 | +0,47 / +0,58 / +0,53 |
| postos cópia × tom | −0,83 / −0,89 / −0,85 | −0,79 / −0,83 / −0,83 | −0,53 / −0,52 / −0,37 |
| postos cópia × distância ao vizinho treinado | −0,25 / −0,18 / −0,15 | −0,35 / −0,30 / −0,21 | −0,33 / −0,40 / +0,13 |
| postos da cópia entre $N$ (8–32, 32–128, 8–128) | 0,86 / 0,91 / 0,80 | 0,82 / 0,84 / 0,76 | 0,75 / 0,74 / 0,59 |
| corr(ganho 8→128, margem em 8) | −0,40 | −0,55 | −0,55 |

O que o modelo acerta sem ter sido ajustado para isso: o tudo-ou-nada, os sinais das três correlações por voz, a
estabilidade da dificuldade entre $N$, o "as piores ganham mais". O que erra: as correlações com distância e tom são
fortes demais (no modelo o tom é o único eixo da padrão e não há ruído de medida nas covariáveis; nos dados, F0 e
distância no latente são medidas ruidosas da coordenada certa), e a correlação com o vizinho não troca de sinal em $N=128$
(nos dados $+0{,}13$, com erro padrão ~0,16 em 38 vozes: a diferença não é significativa).

Na grade $N\times K$ das 4 vozes fixas, o mesmo modelo (sem reajuste) prevê a ordem dos 10 treinos com correlação
0,97 e RMSE 0,041 depois de uma reta $q_{\rm obs}=0{,}11+1{,}19\,q_{\rm modelo}$ (as 4 vozes são mais difíceis que a média;
`out/sim_6.txt`). O RMSE fica abaixo do ruído de semente (~0,05–0,08).

### 6.3 O que o modelo diz sobre os negativos (F5) [SIMULADO; `sim_4_variantes.py`]

- **Média de referências**: $R=1,4,16$ dá $q=0{,}437/0{,}439/0{,}438$ ($N=8$) e $0{,}213/0{,}222/0{,}223$ ($N=128$). Sem efeito, como nas
  38 vozes: o gargalo é $\rho$ (o que o decodificador sabe renderizar), não o ruído do código. Os ganhos nas 4 vozes da grade
  são de vozes no limiar (seção 12).
- **Pseudo-locutores**: se as cópias com velocidade alterada fossem vozes da mesma população deslocadas no eixo do tom,
  o modelo prevê ganho grande ($N=8$: $0{,}437\to0{,}33$, perto de $5N$ vozes reais, $0{,}28$). Os dados dizem que não contam.
  **Tensão**: ou as pseudo-vozes caem fora da distribuição das vozes de teste (no espaço da métrica), ou o modelo de
  cobertura está errado. Teste sem treino na seção 10 (P9).
- **ECAPA, troca de referência, perda contrastiva**: mudam o código, não $n_{\rm loc}$ nem $M$; (4.2) não muda.

---

## 7. Do erro contínuo à fração binária: quando sai uma lei de potência

Por (2.2), $q$ é uma média de probits. Três casos:

**(i) Deslocamento uniforme** [PROVADO dado $\mu_s=\bar\mu(N)+\delta_s$, $\delta_s\sim\mathcal N(0,\tau^2)$]: $q=\Phi(-\bar\mu(N)/\sigma_T)$,
$\sigma_T^2=\tau^2+\sigma_w^2$. O expoente local é
$$
\beta_{\rm loc}(N)=-\frac{d\ln q}{d\ln N}=\frac{\varphi(z)}{\Phi(z)}\cdot\frac{1}{\sigma_T}\frac{d\bar\mu}{d\ln N},\qquad z=-\bar\mu/\sigma_T . \tag{7.1}
$$
Com os dados ($d\bar\mu/d\ln N=0{,}037$ entre 8 e 32 e 0,022 entre 32 e 128; $\sigma_T\approx0{,}097$): $\beta_{\rm loc}\approx0{,}39$ e $0{,}29$.
Bate com os 0,26 a menos do ruído. Se $\bar\mu(N)=\mu_\infty-AN^{-\gamma}$, $q\to\Phi(-\mu_\infty/\sigma_T)>0$: **um erro em lei de potência com
limite finito dá $q$ com piso, não lei de potência**. $q$ só é uma potência pura se $\bar\mu$ crescer sem limite como
$\sqrt{2\beta\ln N}\,\sigma_T$ (então $q\approx N^{-\beta}/\sqrt{4\pi\beta\ln N}$) [PROVADO, assintótica de Mills].

**(ii) Limiar por voz (visão de "quanta")** [PROVADO dada a hipótese]: se cada voz tem um limiar $N_v$ e é copiada para
$N>N_v$, então $q(N)=\Pr(N_V>N)$. Lei de potência $\iff$ $N_V$ tem cauda de Pareto. Em (4.4), $N_v=M/(Kc_\ell\tilde p(v)e^{\|v-v_0\|^2/2\ell^2})$:
o expoente é uma propriedade da **distribuição das densidades locais das vozes**, não do decodificador. Para vozes gaussianas
isotrópicas isso dá caudas do tipo $\chi^2_d$ (expoente subindo até ~1 com $N$); para espectro $j^{-1}$ a simulação dá
expoente ~0,25 constante em três décadas (seção 6.2). Não tenho prova de que $\alpha=1$ dê potência exata; é um fato de
simulação [SIMULADO; explicação CONJECTURA: à medida que a resolução melhora, aparecem novas direções do espectro,
e a dimensão efetiva $d_{\rm eff}(r)=r^{-2/\alpha}$ cresce, compensando o afinamento da cauda].

**(iii) Tudo-ou-nada** é o regime $\tau_\mu\gg\sigma_w$ de (2.2): cada voz está longe do limiar exceto uma fração
$\approx2{,}6\,\sigma_w g(0)$. Ele vem de $\rho_v$ ser uma função determinística da posição da voz (4.2) e de $\sigma_w$ ser pequeno.

---

## 8. Função de escala unificada, colapso, convergência, $N$ mínimo, divisão $N/K$

### 8.1 Função de escala [PROVADO como identidade dado o Lema 1 e (2.2); a forma de $\rho$ depende do regime]

$$
\boxed{\ q(N,K,M)=\mathbb E_{v\sim P}\ \Phi\!\Big(-\frac{\big(\tfrac12-\rho_{N,K,M}(v)\big)\|\Delta(v)\|^2}{\sigma_w}\Big)\ } \tag{8.1}
$$

| Regime | $\rho_{N,K,M}(v)$ | Variável de colapso |
|---|---|---|
| (a) paramétrico | $\rho_\infty+O\big(d\,s^2(K)/N\big)$, quase uniforme | $N/(1+K_0/K)$ |
| (b) Poisson ($N<N^\*$) | $\big[1+\tfrac KM\sum_i e^{(\|\Delta\|^2-\|v-v_i\|^2)/2\ell^2}\big]^{-1}$ | $N$ (e $K/M$ só no log) |
| (b) LGN ($N>N^\*$) | $\big[1+\tfrac{NK}{M}c_\ell\tilde p(v)e^{\|\Delta\|^2/2\ell^2}\big]^{-1}$ | $NK/M$ |
| (c) espectral | $\sum_{j>J(N)}\Delta_j^2/\|\Delta\|^2$, $J\propto(N/s^2(K))^{1/\alpha}$ | $N/(1+K_0/K)$ |

**Teste de colapso nos 16 treinos** (`sim_8_colapso.py`; $q\sim a+b\ln X+c\ln^2X$; ruído de semente ~0,06):
$X=N$: RMSE 0,074; $X=N/(1+K_0/K)$: 0,067 ($K_0=5{,}8$); $X=NK$: 0,137; mistura $N^{0{,}83}(NK)^{0{,}17}$: 0,068.
Só com $N\ge32$: $X=N$ dá 0,065 e $X=NK$ 0,049. Leitura [HEURÍSTICO]: abaixo de ~32 locutores só $N$ conta (Poisson);
acima, $K$ passa a contar e a variável $NK$ melhora, como (4.4) prevê. Com 4 vozes por treino é evidência fraca.

**Previsão de colapso para novos treinos**: no regime LGN, $q$ é função de $NK/M$. Em particular, no modelo,
$(K=64,M=2000)$ e $(K=32,M=1000)$ dão o mesmo $q$ (0,267 e 0,268 em $N=32$; 0,185 e 0,185 em $N=128$), e
$(K=128,M=2000)\approx(K=32,M=500)$ (0,238/0,240; 0,156/0,157) [SIMULADO].

### 8.2 Convergência

- $K\to\infty$, $N$ fixo: (a) $q\to$ função de $a+A/N$ (piso de conglomerado); (b) a desvantagem $2\ell^2\ln(M/K)$ em (4.3)
  cai e fica negativa para $K>M$, mas sobra $\Pr\big(r_{\rm nn}(v)>\|\Delta(v)\|\big)$: **mais frases não cobrem buracos**. Isso é F1/F2.
- $N\to\infty$, $K$ fixo: (a) $q\to\Phi$-piso de $\sigma_b$ e do erro de especificação; (b) $q\to0$ como $\Pr(N_V>N)$ (logarítmico a
  potência, conforme a distribuição das densidades); só sobra o piso da métrica $\Pr(\|\Delta\|\lesssim2\sigma_\eta)$ das vozes
  indistinguíveis da padrão.

### 8.3 $N$ mínimo para 90% e 95% de cópia ($K=32$, 38 vozes; `sim_6_previsoes.py`)

| Lei (ajustada aos 3 pontos) | $q(247)$ | $q(1000)$ | $q(2456)$ | $N_{90\%}$ | $N_{95\%}$ |
|---|---|---|---|---|---|
| potência $0{,}746\,N^{-0{,}26}$ | 0,178 | 0,123 | 0,098 | $2{,}2\times10^3$ | $3{,}2\times10^4$ |
| probit linear em $\ln N$: $\Phi(0{,}31-0{,}23\ln N)$ | 0,166 | 0,097 | 0,066 | $9\times10^2$ | $4{,}5\times10^3$ |
| $a+A/\sqrt N$ ($a=0{,}145$) | 0,198 | 0,171 | 0,162 | nunca | nunca |
| $a+A/N$ ($a=0{,}22$) | 0,226 | 0,221 | 0,220 | nunca | nunca |
| cobertura + atrator (seção 6, $d=16$) | 0,177 | 0,115 | 0,086 | $\approx1{,}7\times10^3$ | $\approx2\times10^4$ |
| margem média com piso (`sim_5_ajuste_previsoes.py`, $1/\sqrt N$) | 0,200 | 0,174 | 0,165 | nunca | nunca |

**Com o LibriTTS inteiro (2.456 locutores) o melhor caso é ~90% e as leis com piso dão ~84%.** 95% pede $10^4$ ou mais
locutores sob qualquer lei sem piso, ou uma mudança de regime (seção 9).

### 8.4 Divisão ótima de um orçamento $NK=B$ [HEURÍSTICO; SIMULADO]

- (a): $q\approx a+A/N+B'/B$; com $B$ fixo só $N$ conta $\Rightarrow$ $K$ mínimo.
- Lei $a+A/\sqrt N+C/\sqrt K$: $K^\*=(C/A)\sqrt B$; com $C/A=0{,}42$: $K^\*\approx27$ para $B=4096$, $54$ para $B=16384$.
- (b) LGN: $q$ depende de $NK/M$ $\Rightarrow$ plano em $K$; no modelo, $B=4096$: $q=0{,}205$ ($K=2$) a $0{,}246$ ($K=128$);
  $B=16384$: 0,144 a 0,169. (b) Poisson: prefere $N$. **Recomendação**: $K$ entre 8 e 32 e o maior $N$ possível; $K$ abaixo de 8
  só se o codificador ainda separar locutor de frase (não modelado aqui) e se as vozes treinadas não importarem (elas querem $K$, F2).

---

## 9. A proposta em duas etapas

Etapa A: texto → unidades de conteúdo $u$, treinada com $N_{\rm texto}$ locutores transcritos. Etapa B: $(u,z)\mapsto$ mel,
treinada por reconstrução com $N_{\rm voz}$ locutores sem transcrição ($K_{\rm voz}$ frases cada; $M_B$ frases de qualquer voz
dominante que entrar em B).

### 9.1 Proposição 4 (separação) [PROVADO no modelo das seções 2–6, dadas as hipóteses S1–S3]

- S1: as unidades não carregam voz: $u\perp v\mid$ conteúdo.
- S2: a distribuição das unidades previstas por A na inferência é a mesma das unidades extraídas no treino de B.
- S3: o código de voz de B só vê áudio (como hoje).

Então a renderização da voz em B é o mesmo problema da seção 4–6 com $(N,K,M)\to(N_{\rm voz},K_{\rm voz},M_B)$, e
$$
q_{2}(N_{\rm texto},N_{\rm voz})=q_1(N=N_{\rm voz},K=K_{\rm voz},M=M_B),\qquad
\mathrm{WER}_2\approx\mathrm{WER}_A(N_{\rm texto},\text{horas transcritas})+\epsilon_B(N_{\rm voz}K_{\rm voz}). \tag{9.1}
$$
A cópia não depende de $N_{\rm texto}$; o WER não depende de $N_{\rm voz}$ exceto pelo termo de interação $\epsilon_B$, que agora é
aprendido de todo o áudio sem texto. **Ganho extra**: se B não for dominado por uma voz ($M_B\approx K_{\rm voz}$), o atrator some.
No modelo, $q_2=0{,}25/0{,}16/0{,}10/0{,}06$ para $N_{\rm voz}=8/32/128/1000$, contra $0{,}41/0{,}30/0{,}22/0{,}12$ em uma etapa com
$M=2000$ [SIMULADO, `sim_4_variantes.py` (6)].

### 9.2 Vazamento de voz nas unidades [PROVADO para canais gaussianos lineares; SIMULADO]

Se as unidades vazam voz ($u$ contém $v+\xi_\ell$ com SNR por dimensão $\mathrm{SNR}_\ell$) e o código tem $\mathrm{SNR}_c$, o
decodificador que minimiza a reconstrução combina os dois canais com pesos de precisão. Na inferência o canal vazado
diz "a voz dos locutores de A" ($v_A$, perto da padrão), então
$$
\hat y=(1-w)\,\hat y_{\rm código}+w\,v_A,\qquad w=\frac{\mathrm{SNR}_\ell}{\mathrm{SNR}_\ell+\mathrm{SNR}_c}, \tag{9.2}
$$
e o encolhimento vira $\rho_2=w+(1-w)\rho_B$. Copiada exige $(1-w)(1-\rho_B)>\tfrac12$. Simulação (piso em $N_{\rm voz}\to\infty$):

| $w$ | 0 | 0,1 | 0,2 | 0,3 | 0,4 |
|---|---|---|---|---|---|
| $q_2(N_{\rm voz}=8/128/2456)$ | 0,25/0,10/0,05 | 0,33/0,14/0,09 | 0,47/0,21/0,15 | 0,67/0,37/0,31 | 0,86/0,67/0,67 |

Em bits, com $\mathrm{SNR}=2^{2I}-1$ por dimensão efetiva: $w\le0{,}1\iff I_\ell\le\tfrac12\log_2\!\big(1+(2^{2I_c}-1)/9\big)$. Com o código
atual (~16 bits sobre ~8 dimensões efetivas, $I_c\approx2$ bits/dim) isso dá $I_\ell\lesssim0{,}7$ bit/dim, ou ~6 bits no total:
**as unidades podem carregar no máximo cerca de um terço da informação de locutor do código** [HEURÍSTICO na conversão para bits].
Mensurável antes de treinar B: o mesmo estimador gaussiano de informação usado no latente, aplicado às unidades.
Remédio: treinar B com unidades extraídas de áudio com voz perturbada (o vazamento deixa de informar, $\mathrm{SNR}_\ell^{\rm ef}\to0$).

### 9.3 Transferência entre línguas [PROVADO no modelo de limiar; resto HEURÍSTICO]

A região de falha $F_N$ de (4.2) depende só do treino de B. Para vozes de outra população (português),
$q_{\rm PT}(N)=\int p_{\rm PT}\mathbf 1_{F_N}\le\big\|p_{\rm PT}/p_{\rm EN}\big\|_\infty\,q_{\rm EN}(N)$: a cobertura transfere na medida em que as vozes
portuguesas caem dentro da distribuição das inglesas. Condições adicionais: (i) inventário de unidades: a fração
$\pi_{\rm miss}$ de quadros portugueses com unidades raras ou ausentes no treino de B entra como erro de renderização
(e de WER) proporcional a $\pi_{\rm miss}$; unidades multilíngues reduzem $\pi_{\rm miss}$; (ii) sotaque e prosódia ligados à língua não
podem viver no código de voz, senão a voz copiada traz o sotaque inglês; (iii) S2 (descasamento entre unidades previstas e
extraídas) é independente da língua. Áudio português sem transcrição (abundante) em B resolve (i) e a razão de densidades.

### 9.4 Módulo de voz paramétrico de baixa dimensão: expoente e preço [PROVADO o piso no modelo; HEURÍSTICO o expoente]

Se B condiciona a voz por uma família de dimensão $k$ (afim, tipo "eigenvoices": $\hat y=\hat\mu+\hat P_k(c-\hat\mu)$), então:

- **Piso** (Lema 1 aplicado a $\hat y=\mu+P_k(v-\mu)$, com $v_0-\mu$ no subespaço):
  $$q_{\rm floor}(k)=\Pr\big(\|P_\perp(v-\mu)\|>\|P_k(v-v_0)\|\big). \tag{9.3}$$
  A voz é copiada se a família captura mais da sua distância à padrão do que a parte que fica de fora.
- **Excesso**: o erro de estimar $(\mu,P_k)$ é $O(k\,s^2/N)$ (Teorema 1), e o excesso de $q$ sobre o piso é linear nesse erro
  (expansão de (2.2)), logo **expoente 1** assintoticamente, contra ~0,26 hoje.

Números de (9.3) com espectros que balizam o medido (`sim_9_piso_e_Nstar.py`, $d=32$, padrão a $\delta$ desvios no 1º eixo):

| espectro | $\delta$ | $k=4$ | $k=6$ | $k=8$ | $k=12$ |
|---|---|---|---|---|---|
| $\alpha=2$ (latente bruto) | 1 | 0,045 | 0,010 | 0,002 | 0,000 |
| $\alpha=2$ | 2 | 0,011 | 0,002 | 0,000 | 0,000 |
| $\alpha=1{,}05$ (latente/SNR) | 1 | 0,374 | 0,218 | 0,114 | 0,022 |
| $\alpha=1{,}05$ | 2 | 0,133 | 0,069 | 0,033 | 0,006 |

Simulação (`sim_4_variantes.py` (5), `sim_5b_parametrico.py`; espectro $j^{-1}$, $d=16$): sem ruído, $q$ converge a 0,075–0,086
($k=4$; (9.3) dá 0,086) e 0,009–0,012 ($k=8$; 0,009). Com $k=8$: $q=0{,}069/0{,}036/0{,}025/0{,}018$ em $N=8/32/128/1000$, contra
$0{,}43/0{,}29/0{,}21/0{,}12$ do decodificador de cobertura no mesmo mundo. O expoente do excesso medido fica entre 0,5 e 1 e é
ruidoso (o excesso é de poucos pontos); "expoente 1" é a assintótica, não confirmada numericamente com precisão.
**Preço**: se a voz real precisa de mais dimensões que $k$, ou se a renderização é não linear no código, o piso (9.3)
mais o erro de especificação não linear ficam para sempre. **Ressalva grande**: no mundo simulado a renderização é linear
(identidade), o que favorece o módulo paramétrico por construção [CONJECTURA que valha no Tacotron/B].

---

## 10. Previsões falsificáveis

### 10.1 Sem treino (medidas)

| # | Medida | Previsão e o que decide |
|---|---|---|
| P1 | Espectro dos embeddings Resemblyzer médios por locutor das 2.456 vozes do LibriTTS (e dimensão intrínseca por TwoNN). | O modelo da seção 6 precisa de $\alpha\approx1$ no espaço da métrica. Se $\alpha\ge1{,}5$ e $d_{\rm int}\le8$, o modelo espectral está errado e o expoente lento tem de vir de outra coisa (por exemplo, do piso da métrica). $\alpha$ fixa também o preço (9.3) de um módulo de dimensão $k$. |
| P2 | $\rho_s=\tfrac12-\mu_s/\|\Delta_s\|^2$ por voz, com $\cos(e_{\rm ref},e_0)$ das 38 referências. | Distribuição bimodal, longe de 1/2, exceto ~1/3 das vozes; $\bar\rho$ cai com $N$. Vozes nunca copiadas devem ter $\rho_s>1/2$ com $\|\Delta_s\|$ grande; vozes com $\|\Delta_s\|<2\sigma_\eta$ devem ficar perto de 50%, não de 0% (piso da métrica). |
| P3 | Regra de Poisson (4.3) nos 38 locutores com os latentes guardados: preditor $\|\Delta\|^2-r_{\rm nn}^2$. | Em $N=8,32$ ele separa copiadas de não copiadas melhor (AUC) que cada distância sozinha; em $N=128$ a parte $r_{\rm nn}$ perde valor. |
| P9 | Distância (Resemblyzer) de cada voz de teste à pseudo-voz mais próxima (`-aug`) e à voz real treinada mais próxima. | Se as pseudo-vozes estão mais perto das vozes de teste que as reais e mesmo assim não ajudaram, o modelo de cobertura (4.1) está refutado. Se estão mais longe (fora da distribuição), o resultado negativo é explicado. |
| P8 | Variância entre sementes por voz (mesmos locutores). | Concentrada em vozes com $|\mu_s|\lesssim\sigma_w$ na outra semente; com 4 vozes, desvio de $q$ ~5–8 pontos; com 38, ~2 pontos. |

### 10.2 Novos treinos de uma etapa

| # | Treino | Previsão (38 vozes ou mais) | Decide |
|---|---|---|---|
| P4 | $N=247,1000,2456$, $K=32$ | Cobertura: $q=0{,}18/0{,}12/0{,}09$ ($\pm0{,}03$); potência: 0,18/0,12/0,10; leis com piso: $\ge0{,}16$ em 2456. | $q(2456)<0{,}12$ rejeita as leis com piso; $>0{,}14$ rejeita cobertura e potência pura. **Exige $\ge200$ vozes de avaliação** (erro padrão $\sqrt{q(1-q)/n_v}\approx0{,}02$); com 38, o IC é ±0,1 e não decide. |
| P5 | $N=128$, $K=32$, com $M=1000$ e $M=500$ frases LJ | $q$ cai como se $K$ dobrasse/quadruplicasse: 0,21 → 0,185 → 0,157 (modelo); quem ganha são as vozes perto da padrão (ganho correlaciona negativamente com $\|\Delta\|$). | Teste direto do atrator: se $M$ não importa, a seção 6 cai. |
| P6 | Orçamento $NK=4096$: $(512,8)$, $(128,32)$, $(32,128)$ | Cobertura LGN: diferenças $\le4$ pontos, $K$ pequeno levemente melhor; lei com raízes: ótimo em $K\approx27$; (a): $K$ pequeno bem melhor. | Distingue os regimes pela forma em $K$. |
| P10 | Latente de voz 32 → 16, 8, 4 (E4 da proposta) | Cópia inalterada até $k$ com $q_{\rm floor}(k)$ pequeno: $k\approx6$–8 se $\alpha\approx2$; $k\approx12$ se $\alpha\approx1$. | Liga P1 a um treino. |

### 10.3 Duas etapas ($N_{\rm texto}=8$, varrendo $N_{\rm voz}$)

| # | Previsão | Falha se |
|---|---|---|
| P7a | $q_2(N_{\rm voz})\le q_1(N=N_{\rm voz})$ em cada $N_{\rm voz}$ (sem LJ dominando B, até ~15 pontos melhor em $N_{\rm voz}\le128$). | $q_2$ seguir $N_{\rm texto}$ (ficar no nível de $N=8$ de uma etapa). |
| P7b | WER plano em $N_{\rm voz}$ (±2 pontos), no nível fixado por A. | WER subir com $N_{\rm voz}$ (interação conteúdo × voz mal aprendida). |
| P7c | Piso de $q_2$ dado por (9.2) a partir de $I(u;\text{locutor})$ medido antes: $w=0{,}2\Rightarrow q_2\gtrsim0{,}15$ para todo $N_{\rm voz}$. | Piso diferente do previsto pela informação medida. |
| P7d | Etapa A em português: $q_{\rm PT}\le\|p_{\rm PT}/p_{\rm EN}\|_\infty\,q_{\rm EN}$; WER em português perto do sistema atual (10%) se $\pi_{\rm miss}$ é pequeno. | Cópia em português muito pior com vozes portuguesas dentro da distribuição inglesa. |

---

## 11. Verificação por simulação: resumo

| Afirmação | Script | Previsto | Simulado | Situação |
|---|---|---|---|---|
| (3.1) erro de voz nova, regime (a) | `sim_1` | $\sigma_b^2+s^2d/(N-d-1)$ | igual a ~1% para $N\ge16$ | confere |
| excesso $\propto1/N$ em (a) | `sim_1` | −1 | −1,09; −1,11 | confere |
| (3.3) vozes treinadas | `sim_1` | fórmula | igual em todas as células | confere |
| piso do codificador $\sigma_\xi^2/R$ | `sim_1` | 0,352/0,127/0,071 | 0,358/0,129/0,071 | confere |
| vizinho mais próximo $N^{-2/d}$ | `sim_2` | −1/−0,5/−0,25 | −1,01/−0,53/−0,23 | confere |
| núcleo $N^{-4/(4+d)}$ | `sim_2` | −0,67/−0,5/−0,33 | −0,69/−0,51/−0,28 | confere ($d=8$ pré-assintótico) |
| espectral $(\alpha-1)/\alpha$ | `sim_7` | −0,33/−0,5/−0,67 | −0,28/−0,42/−0,59 | aproximado |
| elo binário (2.2) nos dados | `sim_6` | 0,442/0,304/0,217 | obs. 0,434/0,302/0,211 | confere (DADOS) |
| cobertura isotrópica explica F1+F4 | `sim_3` | — | não (RMSE ≥0,07 com tudo-ou-nada) | **refutado** |
| cobertura espectral ($\alpha=1$) + atrator | `sim_3` | curva + sinais | curva, tudo-ou-nada, sinais; magnitudes fortes demais | parcial |
| grade $N\times K$ (ordem) | `sim_6` | — | corr 0,97, RMSE 0,041 após reta | confere |
| $q$ depende de $K/M$ | `sim_4` | igualdade | 0,267=0,268; 0,185=0,185 | confere (no modelo) |
| média de referências não ajuda | `sim_4` | ~0 | ±0,01 | confere com dados |
| pseudo-locutores não contam | `sim_4` | — | modelo diz que contariam | **tensão** |
| piso paramétrico (9.3) | `sim_4/5b` | 0,086; 0,009 | 0,075–0,086; 0,009–0,012 | confere |
| excesso paramétrico expoente 1 | `sim_5b` | 1 | 0,5–1, ruidoso | não confirmado |
| vazamento (9.2) | `sim_4` | piso crescente em $w$ | 0,05/0,09/0,15/0,31/0,67 | ilustra |

---

## 12. Crítica: hipóteses fracas, o que fica sem explicação, situação de cada afirmação

### 12.1 Hipóteses mais fracas

1. **O Tacotron como suavizador por frase no espaço da métrica** (4.1) é a hipótese que carrega quase tudo e não foi
   derivada. Um decodificador neural pode extrapolar linearmente em algumas direções (regime (a)) e interpolar em outras
   (regime (b)); o modelo escolhe um só. A versão processo gaussiano (média a priori $v_0$) se sai pior, o que sugere que o
   peso **por frase** da voz padrão importa; P5 testa isso.
2. **Espaço da métrica = espaço do modelo.** O Lema 1 é exato no espaço do Resemblyzer; as hipóteses de cobertura são
   sobre o espaço onde o decodificador interpola. Se os dois não forem aproximadamente isométricos, os números das
   seções 4–6 mudam (os sinais, não).
3. **Vozes gaussianas com espectro $j^{-1}$.** Escolhido numa grade de 2 valores de $\alpha$; o acordo com $\alpha_{\rm snr}$ do latente é
   sugestivo, não um teste. P1 decide.
4. **Ajuste com poucos dados.** O modelo da seção 6 tem 5 parâmetros livres ($d,\ell,\delta,\sigma_\eta,\alpha$) escolhidos para 3 pontos.
   O ajuste da curva não testa nada; os testes são as quantidades que não entraram na escolha (tudo-ou-nada, correlações,
   grade, $K$). E as magnitudes das correlações com tom e distância saem fortes demais.
5. **O expoente 0,26** vem de 3 pontos com IC que inclui 0 e 0,9 (seção 1.2). Potência, logaritmo e piso não se distinguem
   até $N\sim10^3$ com 38 vozes.

### 12.2 O que fica sem explicação

- **Variância entre sementes de 18 pontos**: explicada como **uma voz no limiar que vira** (DADOS: voz 6415, 1,0 → 0,52), o que é
  consistente com (2.2) e com o tudo-ou-nada; mas por que a semente desloca $\rho$ daquela voz em ~0,1 não foi modelado (variância
  da função aprendida por inicialização e ordem dos dados). Com locutores fixos, o modelo só reproduz isso se se acrescentar
  um ruído de função por semente; não fiz.
- **Média de referências**: o modelo explica a ausência de efeito nas 38 vozes (o gargalo é $\rho$, não $\sigma_\xi$), mas não o ganho
  de até 20 pontos nas 4 vozes da grade; a única explicação disponível é que essas vozes estavam no limiar (mesmo
  mecanismo das sementes), o que é post hoc.
- **Pseudo-locutores**: o modelo prevê que contariam se fossem vozes da população; não contaram. A saída "estão fora da
  distribuição" (o latente dos treinos `-aug` concentrou a variância entre locutores: razão de participação 1,9 em
  $8\times8$-aug e 4,1 em $8\times128$-aug, contra 4,0–4,6 nos treinos normais com $N=8$) é plausível e testável (P9), mas não demonstrada.
- **Troca de sinal da correlação com o vizinho em $N=128$**: o modelo espectral ajustado tem $N^\*\approx860$ e não troca o sinal;
  um modelo isotrópico de $d\approx6$–8 troca em ~100 mas decai rápido demais. O valor medido (+0,13) está a ~1,7 erro padrão de −0,15.
- **WER constante em $N$**: compatível com a decomposição (conteúdo na linha longa de LJ, voz no código), mas não derivado.
- **Bits no latente crescendo com $\log N$** (F6): compatível com um código que só resolve as vozes que o decodificador sabe
  renderizar, mas não derivei a taxa.

### 12.3 Situação de cada afirmação

| Afirmação | Situação |
|---|---|
| Lema 1 ($m=(\tfrac12-\rho)\|\Delta\|^2$) | PROVADO (embeddings normalizados) |
| Elo (2.2) e tudo-ou-nada como $\tau_\mu\gg\sigma_w$ | PROVADO dado ruído gaussiano; DADOS conferem |
| Teorema 1 / forma $a+A/N+B/(NK)$ e significado de $A/a=d$, $B/A=K_0$ | PROVADO no modelo linear; SIMULADO |
| Regime (a) não é o gargalo dos dados | HEURÍSTICO (três argumentos da seção 3.2) |
| Taxas de cobertura $N^{-2/d}$, $N^{-2s/(2s+d)}$ | conhecidas; SIMULADO |
| Regra de cópia (4.2)–(4.4) | PROVADO para Nadaraya–Watson; CONJECTURA para o Tacotron |
| $N^\*$ (4.5) | HEURÍSTICO |
| Taxa espectral $(\alpha-1)/\alpha$ | HEURÍSTICO (padrão); SIMULADO aproximado |
| Cobertura isotrópica não explica F1+F4 | SIMULADO (dentro da família testada) |
| Cobertura espectral + atrator explica F1, F4, grade | SIMULADO, parcial; ajuste com 5 parâmetros |
| $q$ depende de $NK/M$ acima de $N^\*$ | PROVADO em (4.4); evidência fraca nos dados (seção 8.1); P5 decide |
| Potência em $q$ exige margem $\sim\sqrt{\ln N}$ ou limiares de Pareto | PROVADO |
| Duas etapas: cópia segue $N_{\rm voz}$ (9.1) | PROVADO dado S1–S3; S1–S3 são CONJECTURA |
| Piso por vazamento (9.2) e limite em bits | PROVADO para canais gaussianos; conversão em bits HEURÍSTICA |
| Transferência entre línguas por razão de densidades | PROVADO no modelo de limiar |
| Módulo paramétrico: piso (9.3) | PROVADO; SIMULADO |
| Módulo paramétrico: expoente → 1 | HEURÍSTICO; simulação não confirma com precisão |
| Previsões numéricas para $N=247$–2456 | dependem do modelo; com 38 vozes não decidem |

---

## Apêndice: arquivos

Scripts (nesta pasta): `sim_0_dados.py` (leitura dos resultados, margens; da tentativa anterior),
`sim_1_linear_gauss.py`, `sim_2_naoparam.py`, `sim_3_atrator.py` (`grade gp|nw [alphas]`, `run "dict(...)"`),
`sim_4_variantes.py`, `sim_5_ajuste_previsoes.py` (ajustes na margem; tentativa anterior), `sim_5b_parametrico.py`,
`sim_6_previsoes.py`, `sim_7_espectral.py`, `sim_8_colapso.py`, `sim_9_piso_e_Nstar.py`.
Saídas: `out/sim_1.txt` … `out/sim_9.txt`, `out/sim_3_grade*.txt`. Os scripts que leem dados abrem os `results.csv`
só para leitura e escrevem CSVs auxiliares (`dados_*.csv`) nesta pasta.
