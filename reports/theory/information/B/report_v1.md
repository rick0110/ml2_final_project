# Clonagem de voz *zero-shot* como teste de hipóteses: evidência bayesiana, gargalo de informação e generalização

Relatório de trabalho, 17/11/2026. Só CPU, sem treino, sem GPU. Nenhum arquivo de repositório foi alterado.
Tudo (scripts, saídas, este relatório) está em
`/tmp/claude-0/-home-user-ml2-final-project/4c11fa8d-6b5f-5505-bd7b-deebf5dd9907/scratchpad/teoria2/B/`.
Complementa `reports/teoria/02_derivacao.md` (Lema 1, regra cobertura + atrator), `03_informacao.md` (bits no latente)
e `04_dados.md` (componentes de variância).

**Legenda de status**: **[PROVADO]** segue das hipóteses enunciadas; **[HEURÍSTICO]** conta com aproximação não controlada;
**[CONJECTURA]** hipótese de modelagem; **[DADOS]** medido nos `results.csv`/`latents.npz`; **[TESTE ✓/✗]** previsão comparada
com dados que não entraram na calibração; **[AJUSTADO]** número escolhido para reproduzir o próprio dado mostrado.

---

## 0. Resumo

1. **Teorema (decodificador = estimador bayesiano)** [PROVADO para o decodificador ideal de erro quadrático]. Um decodificador
   treinado por erro quadrático sobre códigos ruidosos devolve a média posterior da voz sob a mistura empírica de treino
   (voz padrão com peso $M$, cada voz LibriTTS com peso $K$). Com o Lema 1 de `02`, **copiar é um teste de razão de
   verossimilhança**: copiada $\iff \Lambda(c)=\ln\frac{K\sum_i p(c\mid v_i)}{M\,p(c\mid v_0)}>0$. Em bits:
   $$\underbrace{E_v}_{\text{evidência contra a padrão}}-\underbrace{C_v}_{\text{déficit de cobertura}}>\underbrace{\log_2(M/K)}_{=5{,}97\text{ bits}}.$$
2. **A versão sem parâmetro livre é rejeitada** [TESTE ✗]. Com o ruído real do canal (dentro do locutor + ruído posterior do
   VAE) a referência carrega $\Lambda\approx 100$–$120$ nats (DP 45–50) de evidência: o decodificador ideal copiaria 95–100%
   das vozes. O decodificador real usa $\eta\approx1/20$–$1/25$ dessa evidência (**eficiência de evidência**).
3. **Nenhum suavizador de núcleo único (Nadaraya–Watson temperado) explica ao mesmo tempo** (a) vozes treinadas que só
   dependem de $K$, (b) a média de 4 referências não ajudar e (c) a evidência fraca contra a padrão [PROVADO como
   incompatibilidade de escalas; SIMULADO]: calibrado nas treinadas por $K$, o modelo populacional faz as treinadas
   dependerem de $N$ (não copiadas $0{,}48\to0{,}25\to0{,}10$ para $N=8\to128$), contra o plano observado.
4. **A estatística $\Lambda$ ordena as vozes externas melhor e de forma mais estável que qualquer distância isolada**
   [TESTE ✓]: Spearman(cópia, $\Lambda$) $=0{,}54/0{,}54/0{,}56$ em $N=8/32/128$, sem usar a cópia das externas; $d_{LJ}$ dá
   $0{,}34/0{,}43/0{,}63$ e $d_{NN}$ $-0{,}48/-0{,}22/-0{,}12$. A troca "vizinho em $N$ pequeno, voz padrão em $N$ grande" de `03`
   sai sozinha: o termo de cobertura satura ($\log_2 n_{\rm eff}=0{,}7\to2{,}4\to4{,}0$ bits).
5. **O nível absoluto não sai das treinadas** [TESTE ✗]: a lei calibrada nas treinadas dá às externas $q=0{,}16/0{,}12/0{,}08$
   contra $0{,}43/0{,}30/0{,}21$. Vozes novas recebem menos "crédito de cobertura" do que a soma bayesiana sobre vozes treinadas
   lhes daria. O ganho com $N$ medido é $\partial\Lambda/\partial\ln N\approx0{,}55$ vez o de $\partial\Lambda/\partial\ln K$ nas treinadas.
6. **Por que média de referências não ajuda**: o ruído da referência entra em $\Lambda$ só pela projeção no gradiente
   $(\bar u-v_0)/s^2$; a energia do ruído cancela entre as hipóteses [PROVADO]. Os +2,5 a +5 bits de voz que $R=4$ acrescenta ao
   código viram **+0,7 a +0,9 bit** sobre a estatística de decisão e **+0,03 a +0,13 bit** "usados" pelo decodificador [DADOS +
   modelo]. A mudança por voz prevista ($|\Delta p|=0{,}16/0{,}10/0{,}07$) tem a ordem certa ($0{,}07/0{,}10/0{,}06$ observados) [TESTE, parcial].
7. **Tudo-ou-nada = grandes desvios no comprimento da frase** [TESTE ✓]: o DP da margem dentro da voz cai como
   $T^{-0{,}38\pm0{,}03}$ (TCL com quadros correlacionados); prevê e confirma que a metade longa das frases de cada voz é
   mais tudo-ou-nada que a curta (vozes intermediárias −24 a −35% observado, −21 a −25% previsto). A largura do posterior
   fica **entre** média posterior pura (ICC 0,99, rejeitada, $\chi^2=33$–70) e amostragem da posterior (ICC 0,63, rejeitada,
   $\chi^2=8$–16): **posterior temperado**.
8. **Gargalo de informação**: o decodificador usa **1,1–2,4 bits** dos ~14 bits de voz da referência. O WER tem dois termos:
   deslocamento da voz para longe da padrão (inclinação −0,51 WER por unidade de `sim_default`, dentro do treino) e energia
   de "incômodo" da frase de referência (calibrada em $R=4$). Sem usar os dados de força 0,6, prevê a queda de WER na força
   0,6: $-5{,}7/-5{,}9/-5{,}8$ pontos contra $-7{,}5/-4{,}4/-6{,}4$ [TESTE ✓]. Mas a **queda de cópia** na força 0,6 prevista pela
   geometria é 2,5–3× grande demais [TESTE ✗].
9. **Generalização (Xu–Raginsky, meta-aprendizado)**: as cotas dão exatamente a forma $a+A/\sqrt N+C/\sqrt K$ (a lei L2 que
   venceu a validação cruzada em `04`) e preveem treinadas $\propto K^{-1/2}$ independente de $N$: medido $K^{-0{,}46}$ na cópia,
   $K^{-0{,}38\pm0{,}08}$ no WER [TESTE ✓ de forma]. A lacuna novas−treinadas cai como $N^{-0{,}68\pm0{,}18}$, compatível com
   informação mútua total decodificador–locutores **constante** em $N$ (orçamento fixo de 2.500 passos) [HEURÍSTICO].
10. **Previsões mais afiadas**: (i) **$M$ pela metade quase não muda nada** ($q$: 0,22 → 0,20–0,22 no estado por voz) se o peso da voz padrão
    vier da exposição total à LJ (as calibrações nas treinadas dão $M_{\rm eff}\approx8\times10^3$–$1{,}4\times10^5\gg2000$); sob o prior de
    contagem, 0,22 → 0,13. (ii) Latente de 2 dimensões **não piora** o critério binário ($q\approx0{,}09$–$0{,}13$) mas perde
    ~63% dos bits de voz usados; 8 dimensões mantêm 78–80%. (iii) $N=512/2048$: $q=0{,}12/0{,}054$ (lei por voz, $\gamma$ ajustado).
    (iv) Vazamento nas unidades da arquitetura em duas etapas: o limite é **por quadro** e acumula; entre 0,001 e 0,3 bit/quadro
    para $q$ subir de 0,21 a 0,25, conforme o decodificador integre a evidência das unidades como um teste ideal ou com a
    mesma eficiência $\eta\approx1/25$ medida para o código.

---

## 1. Dados, convenções e reprodução

- Latentes: `coverage2/<run>/step_2500/latents.npz` (μ, log σ², locutor; `lj` = voz padrão). Resultados:
  `reports/coverage/results/<run>/step_2500/strength_*/results.csv`. Treinos usados: grade base (N ≥ 8; N = 2 só onde dito)
  e `-wide` (38 vozes externas × 25 frases, uma semente).
- **Coordenadas branqueadas**: $W$ = covariância dentro do locutor (LibriTTS do treino), regularizada
  $W+0{,}05\,\mathrm{tr}(W)/32\cdot I$ (sem isso, treinos $N=8,K=8$ têm $W$ quase singular). Nessa métrica uma frase se afasta
  da média do seu locutor com variância 1 por direção ($\mathrm{tr}\,W\approx32$).
- Unidade de $\Lambda$: **nats de log-contagem** — 1 unidade = multiplicar $K$ por $e$ numa voz treinada (escala ancorada no
  probit de $q_{\rm seen}$ em $\ln K$, $b=0{,}42$, de `04`). $\tau$ = ruído de frase nessa unidade.
- Reprodução (todos com `python3 -I`, dados passados como argumento, nada escrito fora desta pasta):

```
python3 -I b0_extrai.py <coverage2> <results> out/b0.npz   # extrai geometria + resumos por referência
python3 -I b1_explora.py  out/b0.npz            # Bayes sem parâmetro (T=1) e temperado          -> out/b1.txt
python3 -I b2_ajuste.py   out/b0.npz out/       # núcleo por treino, calibrado nas treinadas      -> out/b2.txt, b2.json
python3 -I b2b_diag.py    out/b0.npz            # varredura diagnóstica s², M_eff                 -> out/b2b.txt
python3 -I b3_fatos.py    out/b0.npz            # WER×cópia, duração, R=4, força 0,6 (sem modelo) -> out/b3.txt
python3 -I b4_pop.py      out/b0.npz out/ n128_k32_s1-wide 20   # modelo populacional (no-go)     -> out/b4_n128.txt
python3 -I b5_geo.py      out/b0.npz            # eixo de decisão, ICC no eixo, dispersão de d²   -> out/b5.txt
python3 -I b6_previsoes.py out/b0.npz 28 0.88   # previsões por voz (força, R=4, M, N, dimensão)  -> out/b6_s*.txt
python3 -I b7_orcamento.py out/b0.npz           # orçamento de bits, duração, WER                 -> out/b7.txt
python3 -I b8_lei.py      out/b0.npz            # lei átomo+cobertura calibrada nas treinadas     -> out/b8.txt
python3 -I b9_extras.py   out/b0.npz            # bits usados, WER×K, ICC por ligação, vazamento  -> out/b9.txt
python3 -I b10_gap_hist.py out/b0.npz           # lacuna de generalização, histograma por voz     -> out/b10.txt
python3 -I b11_penalidade.py out/b0.npz         # penalidade de generalização por N               -> out/b11.txt
```
Bibliotecas: `bt_lib.py` (log-razão Λ, ligações), `bt_pop.py` (modelo populacional), `b2_core.py` (definições de `b2`).

---

## 2. O decodificador como estimador bayesiano da voz

### 2.1 Hipóteses

- **B1** [CONJECTURA]. A voz renderizada é (aproximadamente) linear no espaço de embeddings do verificador, e a voz de cada
  frase de treino é a voz do seu locutor mais um desvio próprio da frase.
- **B2**. Treino: $M$ frases da voz padrão $v_0$ e $K$ frases de cada uma de $N$ vozes $v_i$. O decodificador recebe o
  código ruidoso $z$ de cada frase; condicional à voz $i$, $z\sim p(\cdot\mid i)$ (gaussiana com covariância
  $\Sigma_c=W+\Sigma_{\rm post}$: ruído dentro do locutor + ruído posterior do VAE).
- **B3**. O decodificador minimiza o erro quadrático e é flexível o bastante para atingir o ótimo na distribuição de treino.

### 2.2 Teorema 1 (média posterior) [PROVADO sob B2–B3]

O minimizador de $\mathbb E\|y-f(z)\|^2$ é $f^\star(z)=\mathbb E[y\mid z]$. Pela parte de voz de $y$ (B1),
$$
f^\star_{\rm voz}(z)=\sum_{j\in\{0,1,\dots,N\}}\pi_j(z)\,v_j,\qquad
\pi_j(z)=\frac{n_j\,p(z\mid j)}{\sum_k n_k\,p(z\mid k)},\quad n_0=M,\ n_{i\ge1}=K .
$$
Logo o encolhimento para a padrão de `02` é **exatamente o peso posterior da voz padrão**, $\rho(c)=\pi_0(c)$, e
(no limite em que a média das vozes treinadas ponderada pela posterior cai sobre o eixo da referência) o Lema 1
($m=(\tfrac12-\rho)\|\Delta\|^2$) dá:

**Corolário 1 (cópia = teste de Neyman–Pearson)**:
$$
\text{copiada}\iff\rho<\tfrac12\iff
\Lambda(c)\equiv\ln\frac{\sum_{i}K\,p(c\mid v_i)}{M\,p(c\mid v_0)}>0 .
$$
É o teste de razão de verossimilhança "a referência veio da mistura de vozes treinadas" contra "veio da voz padrão", com
limiar igual ao log das chances a priori $\ln(M/K)$ ($\log_2(2000/32)=5{,}97$ bits).

### 2.3 Orçamento de informação [PROVADO para núcleos gaussianos de mesma largura $s^2$]

Com $p(c\mid v)=\mathcal N(c;v,s^2I)$ e $c=v^\star+\xi$:
$$
\Lambda=\ln\tfrac KM+\underbrace{\frac{\|c-v_0\|^2}{2s^2}}_{E_v\ \text{(evidência)}}\;-\;\underbrace{\Big(-\ln\sum_i e^{-\|c-v_i\|^2/2s^2}\Big)}_{C_v\ \text{(déficit de cobertura)}} ,
$$
com $E_v=D_{\rm KL}\big(\mathcal N(v^\star,s^2)\,\|\,\mathcal N(v_0,s^2)\big)$ e $C_v$ uma energia livre ("min suave" das divergências às
vozes treinadas, $C_v=D_{\rm nn}-\ln n_{\rm eff}$, $n_{\rm eff}\in[1,N]$ o número de vozes treinadas dentro da resolução $s$).
Duas propriedades decidem o resto do relatório:

- **(P1) O ruído da referência cancela em energia.** $\|c-v_0\|^2-\|c-v_i\|^2=\|v^\star-v_0\|^2-\|v^\star-v_i\|^2+2\langle\xi,v_i-v_0\rangle$:
  $\|\xi\|^2$ some; o ruído só entra pela projeção no gradiente $g=\nabla_c\Lambda=(\bar u_\pi-v_0)/s^2$, $\bar u_\pi$ a média posterior
  das vozes treinadas. Variância de decisão por referência: $\sigma_{\rm ref}^2=g^\top Wg/R$.
- **(P2) $N$ e $K$ entram por caminhos diferentes.** $K$ entra no prior ($\ln K$, inteiro); $N$ entra só pelo déficit de
  cobertura, no máximo $\ln N$ (regime de lei dos grandes números, `02` eq. 4.4) e tipicamente muito menos (regime de Poisson,
  dominado pelo vizinho).

### 2.4 Função de escala [PROVADO como identidade dado o Corolário e ruído de frase gaussiano]

$$
q(N,K,M)=\mathbb E_{v}\,\Phi\!\Big(-\frac{\Lambda_v(N,K,M)}{\tau}\Big),\qquad
\Lambda_v=\kappa\ln\frac{K}{M_{\rm eff}}+E_v-C_v(N),
$$
$\tau$ = ruído de decisão por frase; $\kappa$ = quanto o log-prior pesa (1 no teorema). Por voz, o desfecho é uma
probabilidade de erro de um teste binário com $T$ quadros por frase; daí o tudo-ou-nada (seção 5).

---

## 3. Testes do modelo bayesiano

### 3.1 Sem parâmetro livre: rejeitado; eficiência de evidência [DADOS; TESTE ✗]

`b1_explora.py`: posterior gaussiano exato com $\Sigma_c$ medido em cada treino (covariância dentro do locutor + ruído
posterior médio do VAE, por classe), pesos $M=2000$ e $K$.

| treino (externas) | $q$ observado | $q$ previsto (T = 1) | mediana de $\Lambda$ (nats) | DP entre vozes |
|---|---|---|---|---|
| n8_k32_s1-wide | 0,43 | 0,05 | 100 | 48 |
| n32_k32_s1-wide | 0,30 | 0,02 | 123 | 50 |
| n128_k32_s1-wide | 0,21 | 0,00 | 122 | 44 |

A referência carrega ~100–120 nats (~150–180 bits) de evidência contra a voz padrão, contra 4 nats (6 bits) de prior.
O decodificador real se comporta como se a evidência fosse ~20–25 vezes menor: a dispersão entre vozes que reproduz o
tudo-ou-nada e a curva de $q_{\rm seen}(K)$ é ~2,2 nats contra 44–50 nats do Bayes ideal, ou seja
**$\eta\approx0{,}045\approx1/22$**. Temperar a verossimilhança ($T=30$) faz o nível certo aparecer, mas não a forma (abaixo).

### 3.2 Núcleo único temperado: incompatibilidade (no-go) [PROVADO como argumento de escalas; SIMULADO]

**Proposição 2.** Num decodificador de média posterior com núcleo gaussiano de largura comum $s^2$ (qualquer $s^2$),
os três fatos abaixo exigem escalas incompatíveis:

1. *Treinadas só dependem de $K$* (expoente de $N$ = 0 ± 0,07, `04`): a massa das outras vozes treinadas vista de uma voz
   treinada tem de ser desprezível frente à própria, $\sum_{j\ne i}e^{-d_{ij}^2/2s^2}\ll1$, isto é $s^2\ll d_{\rm nn}^2/(2\ln N)$
   ($d_{\rm nn}\approx4$ branqueado em $N=128$ ⇒ $s^2\lesssim1{,}6$).
2. *Evidência fraca contra a padrão* (DP de $\Lambda$ entre vozes ≈ 2,2 nats, com DP de $\|v-v_0\|^2$ = 54, `b5`):
   $s^2\approx54/(2\cdot2{,}2)\approx12$ (ou ~28 na escala do Lema 1, seção 3.5).
3. *$R=4$ não ajuda* (P1 vale só com largura comum; com larguras diferentes $s_c^2\ne s_0^2$ entra
   $\tfrac12(\|\xi_1\|^2-\|\xi_4\|^2)(1/s_c^2-1/s_0^2)\approx12(1/s_c^2-1/s_0^2)$ nats, que com $s_c^2=1{,}6$ daria +6 nats, cópia quase total).

(1) pede $s^2\lesssim2$, (2) pede $s^2\approx12$–28, (3) proíbe usar larguras diferentes para contornar. $\square$

Conferência numérica (`b4_pop.py`, modelo populacional com $\Sigma_b$, $W$, ruído posterior e $v_0$ medidos em
n128_k32_s1-wide; $s^2$ ou $(s^2,M_{\rm eff})$ ajustados **só** em $q_{\rm seen}(K)$):

| variante | $s^2$ | $M_{\rm eff}$ | $q_{\rm seen}$ K=8/32/128 (obs 0,50/0,26/0,12) | $q_{\rm seen}$ em N=8/32/128 (obs plano ≈0,3) | $q$ novas N=8/32/128 (obs 0,43/0,30/0,21) |
|---|---|---|---|---|---|
| amostragem, M=2000 | 29 | 2000 | 0,53/0,25/0,09 | **0,48/0,25/0,10** | 0,55/0,27/0,10 |
| amostragem, $M$ livre | 14 | 16.500 | 0,51/0,28/0,13 | **0,42/0,28/0,16** | 0,56/0,33/0,16 |
| média posterior, $M$ livre | 9,4 | 143.000 | 0,52/0,30/0,13 | **0,39/0,30/0,18** | 0,61/0,38/0,19 |

Todas as variantes ajustam as treinadas em $K$ e **fazem as treinadas melhorarem com $N$** (emprestam cobertura das
vizinhas), o que os dados negam. O mesmo vale com a geometria de cada treino (`b2_ajuste.py`: previsões de $q_{\rm seen}$
0,31/0,34/0,11 em $N=8/32/128$, $K=32$). **Rejeitado: a voz padrão não compete com uma soma sobre vozes treinadas num
único núcleo.** O que sobrevive: (P1) e a identidade $M/2\equiv2K$ do modelo (0,165 vs 0,164 em $N=32$), e a previsão de que
$R=4$ não muda a média (0,556/0,267/0,094 vs 0,550/0,267/0,096).

### 3.3 A estatística $\Lambda$ como preditor por voz [TESTE ✓]

`b7_orcamento.py` (a). Unidades de log-contagem, $s^2=12{,}1$ (da dispersão das treinadas; nenhum dado das externas usado),
cobertura de Nadaraya–Watson com as vozes treinadas do próprio treino, distâncias corrigidas pelo ruído ($-\mathrm{tr}W$,
$-\mathrm{tr}W/n_i$).

| treino | $E_v$ mediana (bits) | $C_v$ mediana (bits) | $\log_2 n_{\rm eff}$ | prior | Spearman cópia ~ $\Lambda$ | ~ $E_v$ | ~ $-C_v$ |
|---|---|---|---|---|---|---|---|
| n8_k32-wide | 18,3 (DP 4,8) | +1,3 | 0,7 | −5,97 | **0,54** | 0,42 | 0,46 |
| n32_k32-wide | 12,5 (DP 3,3) | −1,2 | 2,4 | −5,97 | **0,54** | 0,50 | 0,13 |
| n128_k32-wide | 15,2 (DP 3,9) | −3,4 | 4,0 | −5,97 | **0,56** | 0,59 | −0,07 |

Leitura: a combinação dada pela teoria é o melhor preditor por voz em $N=8$ e 32 e empata em 128, e é **estável** em $N$
(0,54–0,56), enquanto cada componente troca de papel: com poucos locutores a cobertura varia muito entre vozes (vizinho
decide), com muitos ela satura ($n_{\rm eff}$ cresce de 1,6 para 16) e só a evidência contra a padrão discrimina. É a explicação
quantitativa do padrão de `03` (seção 6): $d_{NN}$ vence em $N=8$, $d_{LJ}$ em $N=128$. Teto de referência: a correlação de postos
da cópia por voz entre treinos é 0,59–0,75, então 0,55 está perto do que a geometria de um único treino permite.

### 3.4 Nível absoluto: não se transfere das treinadas [TESTE ✗]

`b8_lei.py`: lei "átomo + cobertura" (treinadas: átomo próprio, sem empréstimo — obedece à Proposição 2 por construção;
novas: cobertura NW ou só o vizinho), $s^2=12{,}1$, $\tau=0{,}83$, um único parâmetro $\ln M_{\rm eff}$ ajustado nas treinadas
($M_{\rm eff}=4\times10^4$).

| | N=8 | N=32 | N=128 |
|---|---|---|---|
| treinadas (média dos treinos) obs / prev | 0,30 / 0,10 | 0,27 / 0,27 | 0,32 / 0,34 |
| externas obs | 0,43 | 0,30 | 0,21 |
| externas prev (NW / só vizinho) | 0,16 / 0,18 | 0,12 / 0,34 | 0,08 / 0,13 |

A previsão dá cópia demais às externas e o erro não é monotônico em $N$ (`b11`: penalidade exigida 4,0/1,5/3,9 nats com
NW). **A geometria de cada treino (cada um com seu codificador) varia mais entre treinos do que o efeito procurado**; o nível
absoluto entre treinos não é identificável com latentes de um treino por ponto. Os efeitos *dentro* do treino (ordem das
vozes, mudanças de condição na mesma voz) são o que a teoria consegue prever; por isso as seções 4–6 trabalham com o
**estado por voz** medido e deslocamentos $\Delta\Lambda$.

### 3.5 Escala do posterior: três âncoras independentes concordam num posterior temperado [HEURÍSTICO; DADOS]

Seja $\kappa$ o peso efetivo do log-prior ($\Lambda\ni\kappa\ln K$). Três medidas que não usam a curva $q(N)$:

1. **Lema 1 + média posterior** dá o ruído de decisão por frase em nats verdadeiros: $\tau_{\rm Lema}=4\sigma_w/\|\Delta\|^2=0{,}19$
   ($\sigma_w=0{,}036$, $\|\Delta\|^2=0{,}75$ das 4 vozes com força 0) ou 0,48 (vozes perto do limiar, $\|\Delta\|^2\approx0{,}3$).
2. **ICC latente** 0,83–0,94 (`04`) com o probit 0,42 das treinadas: $\tau=0{,}83$ em unidades de log-contagem.
3. **Largura da transição logística** em $d_{LJ}$ (`03`: $d_{50}\approx15{,}5$, largura 2) ⇒ $s^2\approx31$ branqueado.

As três concordam se $\kappa\approx0{,}23$–$0{,}58$ (equivalente: $s_0^2\approx20$–36 em vez de 12): **o decodificador pesa as
frases de uma voz como $K^{\kappa}$, $\kappa\approx0{,}4$, não como $K$** — que é o expoente $K^{0{,}46}$ do colapso das treinadas.
Consequência para o tudo-ou-nada: a média posterior pura com $\kappa=1$ dá ICC 0,993 e a amostragem da posterior 0,63; o
observado (0,88) é um posterior **temperado** (seção 5).

Importante: **as previsões por voz das seções 4–6 são (quase) invariantes a $s^2$** — $\sigma_L$, $\tau$, $\sigma_{\rm ref}$ e $\Delta\Lambda$
escalam juntos com $1/s^2$ (`b6_s12.1.txt` a `b6_s36.txt` dão os mesmos números para força 0,6, R = 4 e M; a projeção em k
dimensões varia pouco, pela não linearidade do log-soma-exp). O único número que depende da escala é $\kappa$.

### 3.6 Lei por voz (o que fica de pé) [HEURÍSTICO; um parâmetro AJUSTADO]

Com $z_v=\Phi^{-1}(\hat p_v)$ o estado medido de cada voz numa condição de referência ($N_0,K_0,M_0$), em unidades de $\tau$:
$$
\boxed{\;q(N,K,M)=\frac1{|V|}\sum_v\Phi\Big(-\big[z_v+\underbrace{0{,}66\ln\tfrac N{N_0}}_{\gamma/\tau,\ \text{AJUSTADO (externas)}}+\underbrace{0{,}35\ln\tfrac K{K_0}}_{\text{grade, 4 vozes}}-\underbrace{1{,}21\ln\tfrac{M_{\rm eff}(M)}{M_{\rm eff}(M_0)}}_{\kappa/\tau,\ \text{treinadas}}\big]\Big)\;}
$$
(treinadas: $+1{,}21\ln(K/K_0)$ e nada em $N$). Os coeficientes vêm de três conjuntos diferentes: $1{,}21=0{,}42/\sqrt{1-{\rm ICC}}$
(treinadas), $0{,}35=0{,}12/\sqrt{1-{\rm ICC}}$ (4 novas da grade), $0{,}66=0{,}23/\sqrt{1-{\rm ICC}}$ (externas, ajustado — é o único termo
tirado da própria curva). Em termos de prior: **uma duplicação de $N$ vale, para uma voz nova, 0,55 de uma duplicação de $K$ para uma voz
treinada; uma duplicação de $K$ vale, para uma voz nova, 0,29 disso.** O teorema daria o mesmo peso às duas; a voz nova ganha com $N$
mais devagar do que um posterior sobre vozes treinadas permitiria (3.4) e quase nada com $K$ (o efeito de desenho em dois estágios
prevê K irrelevante: ICC ao longo do eixo de decisão = 0,76 ⇒ $K_0=0{,}3$, `b5`; o $K_0=5{,}8$ do colapso de `02` não é recuperado).

---

## 4. Por que mais referências não ajudam (evidência × cobertura)

**Proposição 3** [PROVADO dado (P1) e estado gaussiano]. Seja $\Lambda_v=\bar\Lambda_v+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma_{\rm ref}^2/R)$.
Média de $R$ referências muda a média de $\Lambda$ em zero (a energia do ruído cancela) e muda a decisão só nas vozes com
$|\bar\Lambda_v|\lesssim\sigma_{\rm ref}$. A informação que $R$ acrescenta **sobre a estatística de decisão** é
$\tfrac12\log_2\frac{1+V_b R/\sigma_{\rm ref}^2}{1+V_b/\sigma_{\rm ref}^2}\le\tfrac12\log_2R$ = **no máximo 1 bit para $R=4$**, qualquer que seja o ganho
sobre a identidade da voz no código.

Números (`b6`, `b9`):

| | N=8 | N=32 | N=128 |
|---|---|---|---|
| ganho em bits sobre a voz em $z$ (gaussiano, `03`) | +2,5 | +4,6 | +5,0 |
| $\sigma_{\rm ref}/\sigma_L$ (geometria: $\|\bar u-v_0\|/s^2$) | 0,65 | 0,49 | 0,55 |
| ganho em bits sobre a estatística de decisão | +0,72 | +0,86 | +0,82 |
| bits de voz **usados** pelo decodificador, R=1 → R=4 ($s^2=12$–28) | | 1,1–2,4 → +0,03 a +0,13 | |

Teste por voz (externas, critério `sim_spk` nos dois lados; simulação gerativa com o estado gaussiano, 25 frases, a média de 4
inclui a referência original):

| | prev N=8 | obs N=8 | prev N=32 | obs N=32 | prev N=128 | obs N=128 |
|---|---|---|---|---|---|---|
| $\mathbb E|\Delta p|$ por voz | 0,16 | 0,07 | 0,10 | 0,10 | 0,07 | 0,06 |
| fração que vira de maioria | 0,16 | 0,11 | 0,09 | 0,13 | 0,07 | 0,05 |
| corr($p_1,p_4$) | 0,77 | 0,94 | 0,81 | 0,81 | 0,77 | 0,90 |
| $\Delta$ cópia média | +0,02 | −0,02 | +0,02 | −0,02 | +0,03 | −0,04 |

[TESTE, parcial]: a ordem de grandeza da mudança por voz e o fato de ela se concentrar nas vozes intermediárias (observado:
$|\Delta p|=0{,}11$–$0{,}17$ nas intermediárias contra 0,04–0,07 nas extremas, `b3`) estão certos; em $N=8$ o ruído de referência
é superestimado; o sinal da média sai errado (a média de referências desloca a saída levemente para a padrão,
`sim_default` +0,007, efeito fora do modelo). **Conclusão: os bits extras vão para direções ortogonais ao eixo de decisão e
para um decodificador que usa ~1–2 bits; o gargalo é o prior/cobertura, não a evidência.** O mesmo argumento explica o
vetor ECAPA ignorado (6.4).

---

## 5. Tudo-ou-nada como grandes desvios

**Proposição 4** [PROVADO dado margem por frase = média de $T$ contribuições por quadro com cauda subgaussiana].
Para uma voz com margem média $\mu_v$ e quadros com variância $\sigma_f^2$ e tempo de correlação $t_c$,
$\Pr(\text{frase falha}\mid v)\le\exp\!\big(-T\,\mu_v^2/(2\sigma_f^2t_c)\big)$ (Chernoff; Stein para o teste de hipótese associado).
A fração de vozes intermediárias é $\approx2\,z_{0{,}9}\,\tau\,g(0)\propto T^{-1/2}$, e o ICC latente $=\sigma_L^2/(\sigma_L^2+\tau^2)\to1$.

Testes (`b3`, `b6`, `b7`, `b10`):

1. DP da margem dentro da voz por quintil de duração: 0,040 / 0,036 / 0,033 / 0,032 / 0,029 ⇒ **$\tau\propto T^{-0{,}38\pm0{,}03}$**
   (−0,5 com quadros independentes; o expoente menor indica correlação entre quadros ou um componente que não faz média) [DADOS].
2. **Previsão**: a metade longa das 25 frases de cada voz é mais tudo-ou-nada que a metade curta [TESTE ✓]:

| treino | vozes com 0<p<1, curtas: obs / prev | longas: obs / prev | redução obs / prev |
|---|---|---|---|
| n8-wide | 0,45 / 0,57 | 0,34 / 0,45 | −24% / −21% |
| n32-wide | 0,55 / 0,63 | 0,37 / 0,49 | −33% / −22% |
| n128-wide | 0,37 / 0,53 | 0,24 / 0,40 | −35% / −25% |

   O sentido acerta nos três treinos; o nível absoluto de intermediárias é superestimado (o estado gaussiano com $\tau$ do ICC
   alarga a distribuição) e a redução observada é um pouco maior que a prevista.
3. **Largura do posterior** (histograma das 38 externas em 6 faixas; estado gaussiano ajustado só na média):

| ligação | $\tau/\sigma_L$ | $\chi^2$ (N=8/32/128) |
|---|---|---|
| média posterior, $\kappa=1$ (Teorema 1 + Lema 1) | 0,085 | 35 / 70 / 33 ✗ |
| amostragem da posterior | 0,76 | 16 / 8 / 14 ✗ |
| temperado (do ICC 0,88) | 0,37 | 1,3 / 5,3 / 1,3 (circular: ICC vem dos mesmos dados) |

   Rejeita os dois extremos. O tudo-ou-nada vem de $\tau_\mu/\sigma_w\approx2{,}5$ (`02`), que aqui se lê como "a informação por
   voz ($\sigma_L$) é ~2,7× o ruído por frase ($\tau$)".
4. **Sementes**: a variação entre sementes deve se concentrar nas vozes com $|z_v|\lesssim1$ (≈ 25–35% das vozes); com 38
   vozes o DP de $q$ por semente previsto é ≈ 0,02 + o efeito principal de semente (`04`: 0,054), coerente com a voz 6415 que vira.

---

## 6. Gargalo de informação

### 6.1 Bits de voz usados [DADOS + modelo]

$I_{\rm usado}=\tfrac12\sum_j\log_2(1+\lambda_j/s^2)$ com $\lambda_j$ os autovalores branqueados entre locutores (n128-wide):

| largura | bits | fração em k = 1 / 2 / 4 / 8 direções |
|---|---|---|
| $s^2=1$ (o que o código carrega, μ) | 14,3 | 0,13 / 0,24 / 0,42 / 0,66 |
| $s^2=12$ (decodificador, escala de contagem) | 2,4 | 0,20 / 0,35 / 0,56 / 0,78 |
| $s^2=28$ (decodificador, escala do Lema 1) | 1,1 | 0,22 / 0,37 / 0,59 / 0,80 |

O canal carrega 64–75 bits/frase (KL), ~14–17 sobre a voz; o decodificador **usa 1–2,4 bits**. A taxa está ~30–60× acima do
ponto de operação do decodificador no plano informação–distorção.

### 6.2 WER em dois termos; força 0,6 e R = 4 [TESTE ✓ no WER, ✗ na cópia]

Fatos novos medidos (`b3`): entre vozes de um mesmo treino, **cópia e WER correlacionam +0,3 a +0,95** (vozes copiadas
são menos inteligíveis), mas dentro de uma voz as frases copiadas e não copiadas têm o mesmo WER (diferença −0,003 ± 0,007,
161 vozes mistas). O WER é função de onde a voz é renderizada, não do lado do limiar em que caiu a frase.

Modelo [HEURÍSTICO]: $\mathrm{WER}=w_0+a_v\,\mathrm{sim}_{\rm default}(\text{saída})+a_n\,\|\xi\|^2_{\rm eff}$ (deslocamento da voz para fora da região
bem treinada + energia de "incômodo" da frase de referência que não é voz).

- $a_v=-0{,}51$ (EP 0,03; $r=-0{,}64$) calibrado **transversalmente**, só força 1, R = 1, entre vozes dentro de cada treino.
- $a_n=6{,}0\times10^{-4}$ por unidade de $\|\xi\|^2$, calibrado em R = 4 (energia 32 → 8; parte de voz prevista −0,3 ponto, resto −1,4 ponto).
- Força 0,6 escala o código: incômodo $32\to0{,}36\cdot32$.

| grupo (treinos plain) | ΔWER obs (força 0,6) | prev. voz | prev. incômodo | prev. total |
|---|---|---|---|---|
| treinadas (128) | −7,5 pts | −4,5 | −1,2 | **−5,7** |
| novas (64) | −4,4 | −4,7 | −1,2 | **−5,9** |
| externas (16) | −6,4 | −4,6 | −1,2 | **−5,8** |

(Previsão condicional ao deslocamento de `sim_default` observado, +0,09.) **Força move ao longo da curva cópia–WER; R = 4 move
para fora dela** (só reduz o incômodo): é a separação "voz × resto" do gargalo. Explica por que força 0,6 baixa WER e cópia
juntas e R = 4 baixa só WER.

**Mas a cópia na força 0,6 não é prevista** [TESTE ✗]: escalar o código em 0,6 na geometria NW desloca $\Lambda$ em −4 a −7 $\tau$
(invariante a $s^2$), prevendo cópia 0,6–6%; o observado (25% / 19% / 47% em treinadas / novas / externas) exige −1,7 a −2,2 $\tau$.
O decodificador é ~2,5–3× menos sensível ao módulo do código do que a evidência quadrática prevê (talvez a projeção do estilo
tenha viés, ou o decodificador responda à direção mais que ao módulo).

**WER constante em N** fica só parcialmente explicado: o termo de voz prevê +2 pontos de N = 8 a 128 nas externas (`sim_default`
0,651 → 0,612), o observado é −1,7; precisa de um termo de fidelidade que melhora com dados (como nas treinadas, seção 7).

### 6.3 Dimensão mínima do latente [CONJECTURA; números do modelo]

Projetando código, vozes treinadas e voz padrão nas $k$ direções principais da covariância de treino (branqueada) e
recalculando $\Lambda$ (`b6`, estado por voz de n128-wide; faixas para $s^2=12$–36):

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| $q$ previsto (critério binário) | 0,06–0,11 | 0,09–0,13 | 0,11–0,13 | 0,16–0,17 | 0,20 | 0,21 (obs) |
| fração dos bits de voz usados | 0,20–0,22 | 0,35–0,38 | 0,56–0,59 | 0,78–0,80 | ~0,95 | 1 |

Duas previsões que se separam: **o critério binário não piora (até melhora) com k pequeno**, porque só precisa do eixo
voz padrão–população; **a identidade (`sim_ref`, `sim_spk`) perde** na proporção dos bits usados. Taxa mínima de cópia ≈ o eixo
(1 direção, <1 bit); taxa mínima de identidade ≈ 4–8 direções (60–80% de 1–2 bits).

### 6.4 Vetor de locutor pré-treinado ignorado [PROVADO para o decodificador ideal]

Se o treino é de reconstrução e $I(y;e\mid z)=0$ na distribuição de treino, então $\mathbb E[y\mid z,e]=\mathbb E[y\mid z]$: o ótimo
**ignora** $e$. A cota é $I(S;e\mid z)\le H(S\mid z)$: com a identificação de locutor por uma frase já em 2,9 de 3,0 bits
($N=8$), 4,3–4,6 de 5,0 ($N=32$) e 5,1–5,3 de 6,9 ($N=128$) (`03`), o ECAPA pode acrescentar no máximo 0,1–1,8 bit sobre
*quem* é o locutor treinado, e nada que o objetivo recompense para vozes novas. Previsão: o ECAPA só passa a ser usado
quando o código é degradado (`-spkdrop`) ou quando $N$ cresce até $H(S\mid z)$ ficar grande.

### 6.5 Duas etapas: vazamento acumula por quadro [PROVADO para canais independentes por quadro; números HEURÍSTICOS]

Na etapa B, as unidades $u_{1:T}$ vazando voz com divergência $D_u$ por quadro (entre a voz de A, perto da padrão, e a voz
pedida) somam ao teste uma evidência contrária $T_{\rm eff}\,\eta_u\,D_u$ (Stein: o expoente de erro cresce linearmente no número
de quadros independentes), enquanto o código entra uma vez. Copiada $\iff\Lambda_v>T_{\rm eff}\,\eta_u\,D_u$. Com o estado das 38 externas
em n128 (margem mediana das copiadas 1,7 nats; `b9`):

| $T_{\rm eff}$ (quadros independentes) | $\eta_u$ | $D_u$ para $q$: 0,21 → 0,25 | → 0,30 |
|---|---|---|---|
| 20 | 1 (teste ideal) | 0,012 bit/quadro | 0,031 |
| 20 | 1/25 (eficiência medida do código) | 0,30 | 0,77 |
| 200 | 1 | 0,001 | 0,003 |
| 200 | 1/25 | 0,030 | 0,076 |

`02` dava ~0,7 bit por dimensão para um vetor único; por quadro, o limite é 10–700 vezes menor. Mensurável antes de treinar B:
classificador de locutor por quadro nas unidades (dá $D_u$) e curva de acurácia × número de quadros (dá $T_{\rm eff}$).

---

## 7. Generalização teórico-informacional

Hipóteses: locutores i.i.d. de $P$, $K$ frases i.i.d. por locutor; perda $\ell\in[0,1]$ ($\sigma=\tfrac12$-subgaussiana); decodificador $W$.

- **Voz treinada** (Xu–Raginsky condicional à voz) [PROVADO]: $|\mathrm{gap}_i|\le\sqrt{2\sigma^2I(W;S_i\mid v_i)/K}$. Se os bits que $W$
  guarda sobre as frases de uma voz são limitados ($C_v$), o termo cai como $K^{-1/2}$ **e não depende de $N$**.
  Medido: cópia $K^{-0{,}46}$, expoente de $N$ 0 ± 0,07; WER das treinadas $-w_\infty\propto K^{-0{,}38\pm0{,}08}$ (0,186/0,132/0,109 para
  K = 8/32/128) [TESTE ✓ de forma; a cota é superior, então "expoente ≤ 1/2" é o que se testa].
- **Voz nova** (meta-aprendizado, Jose–Simeone / Chen et al. 2021) [PROVADO]:
  $\mathrm{gap}\le\sqrt{2\sigma^2I(W;v_{1:N})/N}+\mathbb E_i\sqrt{2\sigma^2I(W;S_i\mid v_{1:N})/K}$ — **a forma $a+A/\sqrt N+C/\sqrt K$ (L2), a melhor lei por
  validação cruzada em `04`**. Se $I(W;v_{1:N})\propto N^{\theta}$, o termo cai como $N^{-(1-\theta)/2}$.
- **Lacuna novas − treinadas** (4 novas da grade, todos os K): +0,31 / +0,21 / +0,03 / +0,03 em N = 8/32/128/200 ⇒
  $N^{-0{,}68\pm0{,}18}$, compatível com $\theta\le0$ (informação total sobre o conjunto de locutores constante — orçamento fixo de
  2.500 passos limita $I(W;S)$ pelo número de passos, Pensia et al.) [HEURÍSTICO].
- A queda lenta das externas (expoente aparente 0,26, ou 0,4–0,55 nas medidas contínuas) corresponde a $\theta\approx0{,}0$–0,5.
  Os dados não fixam $\theta$; o que a cota diz é que **nenhum expoente acima de 1/2 em $N$ ou $K$ é garantido**, e que $N$ e $K$ entram em
  termos separados (o colapso $NK/M$ do regime LGN não é implicado por ela).

---

## 8. Previsões falsificáveis

| # | Experimento | Previsão | Falseia |
|---|---|---|---|
| F1 | $N=128,K=32$, **$M=1000$** (metade da LJ) | prior de exposição ($M_{\rm eff}\approx8\times10^3$–$1{,}4\times10^5$, calibrações nas treinadas): $q$ cai ≤ 2 pts (0,22 → 0,20–0,22); prior de contagem: 0,13; `02`: 0,157 | queda ≥ 5 pts rejeita o prior de exposição; < 2 pts rejeita contagem e `02` |
| F2 | $M=4000$ | exposição: +0 a +4 pts (0,22–0,26); contagem: $q=0{,}36$ | > 8 pts rejeita exposição |
| F3 | latente **k = 2 vs 8 vs 32** (128×32) | critério binário: $q(2)\approx0{,}09$–0,13 ≤ $q(8)\approx0{,}16$ ≤ $q(32)$; `sim_spk` cai com k = 2 (37% dos bits usados), não com k = 8 (80%) | k = 2 perder > 10 pts no binário (H_bits de `03`) |
| F4 | $N=512/2048$, K = 32, ≥ 100 vozes | lei por voz: $q=0{,}12/0{,}054$; posterior NW de contagem ($\gamma=\kappa$): 0,002/0 (já rejeitado em 8→128) | $q(2048)>0{,}10$ rejeita $\gamma$ constante (há piso) |
| F5 | K = 64 nas externas | $q$: 0,21 → 0,19 (peso de K nas novas 0,35/ln) ; contagem: 0,13 | |
| F6 | frases 2× mais longas na avaliação | fração de vozes intermediárias cai ×$2^{-0{,}38}\approx0{,}77$; ICC sobe | inalterada |
| F7 | trocar a referência por outra frase do mesmo locutor | mudança por voz $|\Delta p|\approx0{,}1$, concentrada nas vozes com $|z_v|<1$; viradas 5–15% | viradas > 30% |
| F8 | peso do KL maior (taxa → 25–30 bits) | cópia ±5 pts (usa 1–2 bits); WER cai até ≈ $a_n\cdot32\approx2$ pts | cópia cai > 10 pts |
| F9 | duas etapas: medir $D_u$ e $T_{\rm eff}$ nas unidades | piso de $q$ dado pela tabela 6.5; com $\eta_u$ = 1/25 e $T_{\rm eff}$ = 20, $D_u<0{,}3$ bit/quadro mantém $q\le0{,}25$ | piso fora da faixa [ideal, η = 1/25] |
| F10 | sem treino: `sim_default` da própria referência | (atrator) $\mathrm{sim}_{\rm default}(\text{saída})-\mathrm{sim}_{\rm default}(\text{ref})$ cresce com $\rho_v$ estimado por $\Lambda$ | sem relação |

---

## 9. Contabilidade de parâmetros e status

| Parâmetro | Valor | Calibrado em | Usado para prever |
|---|---|---|---|
| $s^2$ (escala da evidência) | 12 (contagem) / 28 (Lema 1) | dispersão de $d^2$ das treinadas + probit $q_{\rm seen}(K)$ | ordem por voz (3.3); previsões por voz são invariantes a ele |
| $\tau/\sigma_L$ | 0,37 | ICC latente (`04`) | R = 4, duração, histograma (este último circular) |
| $\ln M_{\rm eff}$ | 9,0–11,9 | treinadas (seis variantes, `b2`, `b4`, `b8`) | F1, F2; nível absoluto (falhou) |
| $\kappa/\tau=1{,}21$ | — | probit das treinadas em ln K | M, K das treinadas |
| peso de K nas novas 0,35 | — | 4 novas da grade | F5 |
| $\gamma/\tau=0{,}66$ | — | **curva das externas (AJUSTADO)** | só F4 |
| $a_v$, $a_n$ (WER) | −0,51; 6×10⁻⁴ | transversal força 1; R = 4 | ΔWER força 0,6 |
| $T_{\rm eff},\eta_u$ (vazamento) | cenários | não calibrados | F9 |

| Afirmação | Status |
|---|---|
| Teorema 1 (média posterior), Corolário (TRV com limiar ln M/K), orçamento $E-C$ | PROVADO (B1–B3) |
| Bayes sem parâmetro explica os dados | **rejeitado** (prevê 0–5% de falha) |
| Eficiência de evidência $\eta\approx1/22$ | DADOS + modelo |
| Núcleo único temperado (NW) | **rejeitado** (Proposição 2; treinadas dependeriam de N) |
| $\Lambda$ ordena vozes (ρ ≈ 0,55 estável) e explica a troca $d_{NN}\to d_{LJ}$ | TESTE ✓ |
| Nível absoluto entre treinos a partir das treinadas | **falhou** (geometria de codificador por treino) |
| Posterior temperado ($\kappa\approx0{,}4$); extremos rejeitados | HEURÍSTICO; histograma ✓ |
| R = 4: ≤ 1 bit sobre a decisão; mudança por voz | PROVADO / TESTE parcial (sinal da média errado) |
| Tudo-ou-nada por grandes desvios; previsão curta × longa | PROVADO / TESTE ✓ |
| WER em dois termos; ΔWER força 0,6 | HEURÍSTICO / TESTE ✓ |
| Cópia na força 0,6 pela geometria | **rejeitado** (2,5–3× forte demais) |
| ECAPA ignorado | PROVADO (decodificador ideal) |
| Vazamento acumula por quadro | PROVADO (quadros independentes); números HEURÍSTICOS |
| Formas das cotas de generalização (L2; treinadas $K^{-1/2}$ só) | PROVADO (cotas); acordo de forma |
| Efeito de desenho explica o peso de K nas novas | **rejeitado** (ICC no eixo 0,76 ⇒ $K_0=0{,}3$, não 5,8) |
| Lei por voz e F1–F10 | CONJECTURA / previsões |

## 10. O que fica sem explicação

- **Por que a voz nova ganha com N só ~1/3 do que um posterior sobre as vozes treinadas lhe daria**, e por que as treinadas não
  emprestam cobertura. A leitura mais simples é que o decodificador tem dois mecanismos — memorização por voz (átomo, peso
  $K^{\kappa}$) e generalização (peso que cresce como $N^{0{,}3}$ em bits) — mas não derivei o segundo expoente.
- **A força 0,6** mexe menos na cópia do que a geometria do latente prevê (fator 2,5–3).
- **O nível absoluto** entre treinos: cada treino tem seu codificador; comparar níveis exige um espaço comum (embutir as
  referências num verificador fixo, P1 de `02`, ou guardar checkpoints).
- **$M_{\rm eff}$** é mal identificado (1,1×10⁴ a 1,4×10⁵ conforme a variante); o que é robusto é $M_{\rm eff}\gg2000$, e é isso que
  sustenta F1. A coincidência com as 13.100 frases do LJSpeech do pré-treino é sugestiva, não demonstrada.
