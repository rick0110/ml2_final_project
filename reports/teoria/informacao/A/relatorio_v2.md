# Clonagem *zero-shot* como codificação de fonte — v2 (revisão após a crítica)

Documento de trabalho (teoria2/A), 17/11/2026. A v1 (`relatorio.md`) fica como está; esta versão corrige a contagem
de parâmetros e responde aos 10 pontos da crítica. Derivações que não mudaram (Proposições 1–2, verificação da
fórmula, comparação água reversa × livro-código aleatório, teorema tauberiano) estão na v1 e só são resumidas aqui.
Novos scripts: `b1_revisao.py` (pontos 1, 3, 4, 5, 8, 10), `b2_aug.py` (ponto 6), `b3_regimes_extrap.py`
(pontos 5b, 7, 9); saídas em `out/b1.json`, `out/b2.json`, `out/b3.json`. Só CPU, `python3 -I`, nenhum repositório
tocado.

Legenda: **[PROVADO]**, **[HEURÍSTICO]**, **[CONJECTURA]**, **[SIMULADO]**, **[DADOS]** como na v1.
**[MORTO]** marca uma ideia que a revisão derruba.

---

## 0. Resumo da v2

**Modelo (inalterado).** O decodificador é tratado como um canal de teste de Gibbs sobre o livro-código de treino
(massas $K$ por voz LibriTTS e $M$ para a voz padrão $v_0$, temperatura $T$). No limite de Poisson:
$$
q(N,K,M)=\mathbb E_c\big[(1-F_c(\|c-v_0\|^2-\tau))^N\big],\qquad \tau=2T\ln(M_{\rm ef}/K),
$$
com $F_c(t)=\Pr(\|c-V\|^2\le t)$ e $V$ uma voz treinada sorteada.

**O que mudou de veredito.**

| Afirmação da v1 | v2 |
|---|---|
| "1 parâmetro livre" | **2 parâmetros** ($\tau$, $T$), mais a escolha do espaço comum. Com $T=1$ fixado *a priori*, o teste vira **1 parâmetro, 3 pontos, 2 g.l.**; esses 2 g.l. só rejeitam a hipótese "sem dependência em $N$"; não distinguem o modelo de alternativas razoáveis |
| $\tau_0=\|\mu-v_0\|^2$ como previsão sem parâmetro | **retirado**: vai para a seção de hipóteses pós-hoc (§2) |
| $M$ entra como contagem de frases | **[MORTO]**: com $T=1$, $M_{\rm ef}\approx2\times10^{26}$. Então $\tau$ não é contagem. Previsão para $M/2$: $\Delta q\approx0$ se $T\approx1$; $-0{,}12$ se $T\approx21$ (§4) |
| vozes treinadas como codewords ruidosos ($\sigma^2/K$) explicam o efeito de $K$ | **[MORTO]**: prevê $q_{\rm seen}=0{,}46/0{,}38/0{,}38$ contra $0{,}55/0{,}26/0{,}11$ (§5b) |
| média de referências "cancela exatamente" | vale para $\|\xi\|^2$; o termo cruzado não cancela e explica o ganho nas 4 vozes da grade (§8) |
| $q(2456)\approx0{,}15$, piso 0,10 | depende do modelo de extrapolação: 0,14–0,19, IC 90% de 0,07 a 0,29; o piso (0,105, IC 0,05–0,21) não depende do modelo de vozes, só de $\tau$ (§9) |
| bits por voz | a forma agregada fica; **a forma por voz não prevê melhor que a distância** (§10) |
| expoente | sem lei de potência em $8\le N\le2456$; o modelo prevê $\alpha_g\approx0{,}53$ para a medida contínua (observado 0,55 em `sim_ref`, 0,41 em `sim_spk`), com IC largo (§7) |

**Contagem honesta e tabela final** (38 vozes externas, $K=32$, espaço comum = latente do treino $N=128$):

| | ajustado em | $q_8$ | $q_{32}$ | $q_{128}$ |
|---|---|---|---|---|
| observado | — | 0,434 | 0,302 | 0,211 |
| **principal**: $T=1$ (*a priori*), $\tau$ ajustado | $q_{128}$ | **0,375** | **0,263** | (0,211) |
| resíduo obs − prev, IC 90% (bootstrap por voz, $\tau$ reajustado) | | [−0,15; +0,21] | [−0,11; +0,14] | |
| nulo "$q$ não depende de $N$", resíduo IC 90% | | [+0,14; +0,31] | [+0,02; +0,15] | |
| $T=3$ (escolhido entre {1,2,3} olhando a curva: **2 parâmetros**) | $q_{128}$ | 0,445 | 0,279 | (0,211) |
| espaço do treino $N=32$, $\tau$ ajustado em $q_{32}$ | $q_{32}$ | 0,557 | (0,302) | — |
| espaço do treino $N=128$, $\tau$ ajustado em $q_{32}$ | $q_{32}$ | 0,455 | (0,302) | — |

O modelo reproduz 74% (N=8) e 57% (N=32) da queda observada até $N=128$. Isso é compatível com os dados e
descarta "sem efeito de $N$", mas com 38 vozes e uma semente não separa este modelo de outra curva monótona
qualquer. A previsão de $q_8$ muda em ~0,1 conforme o espaço latente.

---

## 1. Contagem de parâmetros e $T$

**Tentativa de fixar $T$ por uma medida que não é a curva.** No canal de Gibbs, a probabilidade de cópia de uma voz
em função da folga $s=\|c-v_0\|^2-D_N(c)$ é logística, $\sigma((s-\tau)/2T)$. Ajustei essa logística às frações
por voz (38 vozes × 3 $N$, binomial com 25 frases), com $\tau$ e uma largura $w$ únicos. Isso usa as **posições**
das vozes, não a curva agregada:
$w=42$ (IC 90% 27–60) $\Rightarrow T_{\rm largura}=w/2\approx21$ (`b1.json: p1_width`).

**Esse valor contradiz a curva** [DADOS, `b3.json: regime_massa`]. Com $T=21$ e $\tau$ ajustado em $q_{128}$, o
modelo prevê $q_8=1{,}00$ e $q_{32}=0{,}85$. A largura medida inclui o erro da geometria: o latente não é o espaço
onde o decodificador decide, e por isso a largura é uma **cota superior** para $T$, não uma medida dele.
Nenhuma medida disponível fixa $T$ de forma independente.

**Decisão.** Fixo $T=1$ *a priori*: é a escala do ruído dos códigos de treino nas unidades branqueadas pelo canal,
a temperatura do decodificador MSE-ótimo (Proposição 1 da v1). Com isso o teste tem **1 parâmetro ajustado
($\tau$), 3 pontos, 2 graus de liberdade**.

**Quanto valem esses 2 g.l.** O bootstrap por voz, reajustando $\tau$ em cada reamostra, dá resíduos com IC 90% de
±0,15–0,2 em $N=8$. O nulo "sem dependência em $N$" fica fora do IC; qualquer curva monótona razoável fica dentro.
Valor como teste: **fraco**. Ele mostra que o mecanismo tem a direção e a ordem de grandeza certas, não que é o
mecanismo certo. Usar $T=3$ melhora o ajuste (0,445/0,279), mas aí são 2 parâmetros para 3 pontos (1 g.l.), o que
quase não testa nada.

## 2. Hipótese pós-hoc a testar: $\tau=\|\mu-v_0\|^2$

**Status: hipótese pós-hoc, não é previsão.** Notei a coincidência depois de ajustar $\tau$. Não tenho derivação: o
canal de Gibbs dá $\tau=2T\ln(M/K)$, que não envolve geometria. Uma justificativa exigiria modelar o *checkpoint* de
partida (decodificador de um só locutor, ajustado por 2.500 passos), o que não fiz.

Evidência a favor e contra, com $\tau$ ajustado contra $\|\mu-v_0\|^2$ no espaço de cada treino:

| treino | 8 | 32 | 128 |
|---|---|---|---|
| $\tau$ ajustado | 146 | 110 | 114 |
| $\|\mu_N-v_0\|^2$ | 135 | 107 | 116 |

O acordo é de 2–8%, mas os três espaços vêm dos mesmos dados e de codificadores parecidos.

**Teste prospectivo.** Num treino novo (por exemplo $N=512$), calcular $\|\mu_N-v_0\|^2$ nos latentes antes de
sintetizar e prever $q$ com $T=1$. Falha se o $\tau$ ajustado depois se afastar mais de 15%.

## 3. Circularidade da geometria

Na v1, a tabela principal usou **um espaço comum**: o latente do treino $N=128$, onde estão as 127 vozes treinadas
dos três livros-código aninhados, as 38 referências e a LJ. Só a `a4` usou o espaço de cada treino. Nenhum espaço
disponível é externo; o espaço do treino de 128 é parcialmente resultado do treino com essas vozes (circular para
$N=128$, menos para os subconjuntos 8 e 32).

Sensibilidade (`b1.json: p3_espacos`), mesmo procedimento (ajustar $\tau$ em $N=32$, prever $N=8$, $T=1$):

| espaço | $\tau$ | $q_8$ previsto (obs. 0,434) | Spearman por voz |
|---|---|---|---|
| latente do treino $N=32$ | 110 | 0,557 | 0,59 |
| latente do treino $N=128$ | 122 | 0,455 | 0,48 |
| próprio de cada treino | 146 / 110 / 114 | — ($\tau$ não é constante) | — |

Nos espaços próprios, um único $\tau$ **não** serve para os três treinos (146 contra 110–114). Nos espaços comuns,
$q_8$ muda em 0,1 conforme o espaço escolhido. **A incerteza da geometria é do mesmo tamanho do efeito a prever.**
Um espaço externo (embeddings de verificador de locutor das vozes de treino) resolveria; não há embeddings dos
locutores de treino salvos.

## 4. O fracasso de $M$ e a massa efetiva

$\tau=2T\ln(M_{\rm ef}/K)$ dá, com o $\tau$ ajustado:

| $T$ | $\ln(M_{\rm ef}/K)$ | $M_{\rm ef}$ |
|---|---|---|
| 1 (*a priori*) | 57 | $2\times10^{26}$ |
| 5 | 13 | $1{,}6\times10^{7}$ |
| 21 (largura) | 6,0 | $1{,}2\times10^{4}$ |

**[MORTO]** Com $T\approx1$, $M_{\rm ef}$ não é contagem de nada: o atrator não vem das 2.000 frases de ajuste fino.
Com $T\approx21$, $M_{\rm ef}\approx12.400$, curiosamente perto das 13.100 frases do LJSpeech completo (pré-treino).
Mas $T=21$ é incompatível com a curva em $N$ (§1). Registro como coincidência, sem peso.

**A teoria ainda prevê efeito de $M$/2? Depende de $T$, e é exatamente isso que o experimento mede**
(`b1.json: p4_M`, `b3.json: Mhalf_dq128`), $N=128$, $K=32$, reduzindo $M$ de 2.000 para 1.000:

| hipótese | $\Delta q_{128}$ previsto | IC 90% |
|---|---|---|
| regime de distorção, $T=1$ | −0,001 | [−0,07; 0,00] |
| $T=5$ | −0,055 | — |
| $T=10$ | −0,09 | — |
| regime de massa, $T=21$ | −0,115 | [−0,19; −0,04] |
| $\tau$ geométrico (§2) | 0 | — |

**Previsão registrada (2 treinos: $N=128,K=32$ com $M=2000$ e $M=1000$, 38 vozes):** $|\Delta q|\le0{,}03$ se a
teoria de distorção com $T\approx1$ estiver certa. $\Delta q\le-0{,}08$ derruba $T\approx1$ e favorece o regime de
massa. Ressalva de poder: o desvio padrão de $\Delta q$ só por semente é ~0,05–0,08 (04), então são necessárias
2 sementes por braço para separar 0 de −0,1.

## 5. Falhas fora da amostra

### 5a. As 4 vozes fixas da grade (`b1.json: p5a_vozes_grade`)

| voz | F0 | $\|c-v_0\|^2$ (percentil entre as 38) | folga em $N=8/32/128$ (percentil) | cópia obs. 8/32/128 (`-wide`) | prevista |
|---|---|---|---|---|---|
| 4297 | 171 | 201 (84%) | 160 / 187 / 187 (74–89%) | 1,0 / 1,0 / 1,0 | 1 / 1 / 1 |
| 696 | 168 | 161 (45%) | 125 / 139 / 142 (45–50%) | 0,0 / 0,04 / 0,64 | 1 / 1 / 1 |
| 6415 | 198 | 173 (50%) | 130 / 157 / 159 (53–71%) | 0,44 / 0,88 / 0,96 | 1 / 1 / 1 |
| 7800 | 183 | 60 (0%) | 15 / 15 / 27 (5–8%) | 0 / 0 / 0,40 | 0 / 0 / 0 |

Medianas das 38: F0 174 Hz, $\|c-v_0\|^2=168$, distância ao centro 6,3 (as 4: 4,4–6,6).

**Explicação.** As 4 vozes **não** são atípicas: nem longe do livro-código, nem perto da voz padrão, nem
deslocadas em F0 ou na distância ao centro. A diferença de origem (locutores de treino retirados: 4297, 696, 6415,
7800, contra test-clean nas 38) não aparece na geometria. O erro vem de duas vozes (696, 6415) com folga mediana,
11–45 unidades acima de $\tau$, que falham em $N$ pequeno (e 7800, no piso, copia 40% em $N=128$). O modelo de limiar duro chama essas vozes de "copiadas"; os dados dizem que vozes
nessa faixa são copiadas só às vezes.

O mesmo acontece nas 38: a largura por voz medida é $w\approx42$, contra $2T=2$ no modelo, e 2 das 4 vozes que o
modelo põe no piso já são copiadas em $N=128$. **O modelo é preciso demais por voz.** O acerto agregado nas 38
vem de erros que se compensam (vozes "copiáveis" que falham e vozes "no piso" que copiam). Com 4 vozes não há
compensação. A falha na grade não é anomalia das 4 vozes; é a falha da forma por voz (§10) aparecendo num conjunto
pequeno.

### 5b. Efeito de $K$ nas vozes treinadas, $N=8$ (`b1.json: p5b_seen`, `b3.json: regime_massa`)

**Derivação.** Uma voz treinada é o próprio codeword, conhecido pela média de $K$ frases. Para a referência
$c=v+\xi$, $D_{\rm self}=\|\xi-\bar\varepsilon_K\|^2$, com
$\mathbb E D_{\rm self}=\mathrm{tr}W_x+\mathrm{tr}(I)/K=11{,}9+32/K$ = 15,9 / 12,9 / 12,2 para $K=8/32/128$.
O limiar desloca $2T\ln(K/32)$. Com $T=1$, a folga muda ~10 unidades entre $K=8$ e 128, contra uma dispersão de
folgas entre vozes de ~60.

| $q_{\rm seen}$, $N=8$ | $K=8$ | 32 | 128 |
|---|---|---|---|
| observado (sementes) | 0,55 / 0,555 | 0,26 | 0,09 / 0,125 |
| codeword ruidoso, $T=1$ | 0,46 | 0,375 | 0,375 |
| + erro de renderização $A/K$ ajustado ($A=50$) | 0,50 | 0,375 | 0,375 |
| regime de massa $T=21$ | 1,00 | 0,97 | 0,70 |
| $T=21$, massa total $\propto N^{0,25}$ | 0,79 | 0,53 | 0,05 |
| $T=10$, $N^{0,25}$ (melhor compromisso) | 0,50 | 0,49 | 0,27 |

**[MORTO]** "Voz treinada = codeword ruidoso $\sigma^2/K$" não produz o efeito de $K$: o ruído de estimação da média
é pequeno demais. Nenhuma combinação de ($T$, massa sublinear em $N$) ajusta ao mesmo tempo a curva das treinadas
em $K$ e a curva das novas em $N$. O efeito de $K$ nas treinadas é outro mecanismo: quão bem o decodificador aprende
a **renderizar** cada voz (erro de aprendizado, não de geometria do código), que o canal de Gibbs não tem.
A teoria cobre vozes novas no regime de distorção e **não** cobre vozes treinadas.

## 6. Pseudo-locutores por velocidade (`b2_aug.py`, espaço de cada treino `-aug`)

Os latentes existem. O manifesto mostra dois tipos de pseudo-voz: perturbações das 8 vozes LibriTTS (16 a 47 nos
latentes) e **6 perturbações da própria LJ** (`lj_v±2..4`, 251–487 frases cada).

| medida (mediana) | 8×8-aug | 8×128-aug |
|---|---|---|
| pseudo-voz → sua voz de origem | 6,4 | 5,6 |
| pseudo-voz → outra voz real mais próxima | 7,8 | 8,0 |
| voz real → voz real mais próxima | 5,7 | 6,2 |
| pseudo-LJ → LJ | 5,8–12,4 | 4,9–9,2 |
| fração da variância dos deslocamentos na 1ª direção | 0,39 | 0,62 |

**Leitura.**
1. As pseudo-vozes **não** ficam coladas na origem: estão tão longe dela quanto duas vozes reais vizinhas entre si.
   Mas se deslocam sobretudo ao longo de uma direção comum (o eixo de tom/velocidade; 62% da variância em
   8×128). Elas formam uma família de dimensão ~1 em torno de cada voz e cobrem uma fatia fina do espaço, não uma
   região.
2. As pseudo-LJ põem codewords de **massa alta perto da voz padrão**, isto é, alargam o atrator. Nas referências,
   a folga cai até 18–58 unidades em vozes treinadas (ex.: 2952: 240→182).
3. Efeito previsto nas 4 vozes novas (folga só reais → com pseudo): 8×128: 4297 +15, 696 +24, 6415 −1, 7800 +7.
   8×8: −2 a 0. Com $\tau\approx110$–150, só 696 (122→146) pode cruzar o limiar. Previsão: no máximo 1 voz de 4
   muda, $|\Delta q|\le0{,}25$, direção ambígua (pseudo-LibriTTS ajudam, pseudo-LJ atrapalham).
   Observado: 8×128 0,51→0,42; 8×8 0,26→0,28 (frações copiadas). 696 foi de 0,28 para 0,32, sem cruzar.

**Veredito:** a teoria prevê um efeito pequeno, de sinal incerto, e o resultado nulo é compatível com isso. Não é
contradição. Também não é uma confirmação forte: com 4 vozes e uma semente, o teste não tem poder. A v1 citava a
previsão de 02 (ganho grande), feita num modelo em que as pseudo-vozes eram vozes da população; a geometria medida
mostra que não são (família quase unidimensional, mais uma extensão do atrator).

## 7. Regimes e expoente

**Cruzamento explícito** (espectro medido $a\approx1$–1,5, $d$ intrínseco 7–10):
- **Distorção ótima** $D(\log_2N)\propto(\ln N)^{-(a-1)}$: vale para $D$ enquanto o número de direções codificadas
  $J\approx2\ln N/a$ for menor que as ~8–14 direções com variância relevante, isto é, até $N\sim e^{4a\text{–}7a}$
  ≈ $10^2$–$10^4$. Na simulação, a fórmula de alta resolução erra por 4–7× até $N=4096$ (v1 §2.2). **Em 8–2456,
  $D_N$ está no regime de taxa baixa.**
- **$q-q_\infty\sim N^{-2/d}$** (tauberiano) é uma afirmação **assintótica**: só vale quando as vozes que restam
  são as da fronteira do atrator e cada uma já está em alta resolução local. No modelo gaussiano calibrado, o
  expoente local do excesso $q-q_\infty$ vale 0,27 / 0,11 / 0,17 / 0,25 / 0,22 / 0,09 / 0,03 / 0,15 entre
  $N=8,32,\dots,10^6$ (`b3.json: expoente_local_gauss`). **Não há lei de potência estável em nenhuma janela de
  8 a $10^6$.** A curva é a função de sobrevivência da distribuição de $L(c)$ ao longo de $\log_2N$
  (§10), cheia de degraus com 38 vozes. Em 8–2456 nenhum dos dois regimes assintóticos se aplica.
- A concordância "2/d ≈ 0,26" da v1 **não é evidência**: o expoente binário não está identificado (IC de −0,7 a
  1,6) e o modelo, com os mesmos números, não produz um expoente estável. Retiro esse argumento.

**Previsão para as medidas contínuas** (`b3.json: continua`). Se `sim_ref` (e, aproximadamente, `sim_spk`) for linear
em $-\|\hat y-c\|^2$, o modelo prevê o expoente pela razão de ganhos
$\alpha_g=\log_4\frac{y_8-y_{32}}{y_{32}-y_{128}}$:

| quantidade (modelo, livros-código reais) | $\alpha_g$ previsto (IC 90% por voz) | observado (04) |
|---|---|---|
| $\mathbb E\|\hat y-c\|^2$, $T=1$ | 0,53 [0,15; 0,92] | `sim_ref` 0,55 (−0,17; 1,88); `sim_spk` 0,41 (−0,59; 2,00) |
| $\mathbb E\|\hat y-c\|^2$, $T=21$ | 0,31 [0,25; 0,38] | |
| só $D_N$ (sem atrator) | 0,33 | |
| binário $q$, $T=1$ | 0,55 | 0,26 (−0,73; 1,57) |

Os pontos concordam com $T=1$, mas todos os ICs observados cobrem 0–1,5: **não é teste**. Previsão registrada: com
≥200 vozes de avaliação e $N\in\{128,512,2048\}$, $\alpha_g$ de `sim_ref` em 0,3–0,9.

## 8. Média de referências

A v1 dizia que o ruído cancela. Precisando: na folga,
$s=\|v-v_0\|^2-\|v-v_{\rm nn}\|^2+2\langle\xi,v_{\rm nn}-v_0\rangle$, **$\|\xi\|^2$ cancela exatamente; o termo
cruzado não** [PROVADO]. Seu desvio é $\sigma_\times=2\sqrt{(v_{\rm nn}-v_0)^\top W_x(v_{\rm nn}-v_0)}\approx20$
(mediana nas 38). Com $R$ referências, $\sigma_\times/\sqrt R$: só vozes com $|s-\tau|\lesssim\sigma_\times$ podem
virar.

Nas 4 vozes da grade (`b1.json: p8_R`), $z=(s-\tau)/\sigma_\times$: 696 tem $z=0{,}49$ ($N=8$) e 1,5 ($N=128$);
6415 tem 0,83 e 2,3. São exatamente as duas vozes falhas com folga **acima** de $\tau$, ou seja, cuja referência
única teve um termo cruzado desfavorável. Reduzir esse termo pela média as empurra para a cópia (regressão à
média). 4297 ($z=1{,}9$–3,2) já copia; 7800 ($z=-7$ a $-9$) não muda. Previsão: 1–2 vozes de 4 sobem, ou seja,
+10 a +25 pontos no critério por locutor. Observado: +13 (IC +1 a +35) nos 4 plain; 128×8: 47%→67%.

Nas 38, só 21% das vozes estão a menos de $1\sigma_\times$ do limiar, e as viradas vão nos dois sentidos.
Previsão ≈0; observado −2,5 pontos (IC −6,4 a +0,8).

**Conciliação:** o ganho nas 4 vozes vem da **seleção**. São vozes que falhavam com folga geométrica positiva; o
efeito de $R$ é real por voz e nulo na média de uma população.

## 9. Incerteza da extrapolação (`b3.json: extrapolacao`)

Três modelos para $P_v$. Todos usam o mesmo $\tau$ (regra dura ajustada em $q_{128}$) e são ancorados em
$q_{\rm obs}(128)=0{,}211$, somando a queda prevista por cada modelo. Bootstrap de 60 reamostras das vozes,
reajustando $\tau$:

| modelo de $P_v$ | $q(512)$ | $q(2456)$ | $q(10^4)$ | piso |
|---|---|---|---|---|
| empírico (127 vozes reamostradas; resolução 1/127, sem cauda) | 0,19 [0,12; 0,29] | 0,19 [0,12; 0,29] | 0,19 | 0,105 [0,05; 0,19] |
| gaussiano (espectro medido) | 0,18 [0,10; 0,29] | 0,145 [0,07; 0,28] | 0,13 [0,06; 0,28] | 0,105 [0,05; 0,21] |
| mistura de 2 gaussianas | 0,18 [0,09; 0,27] | 0,146 [0,07; 0,24] | 0,12 [0,06; 0,23] | 0,105 [0,05; 0,18] |

Notas:
1. O empírico satura porque não tem cauda: vozes sem vizinho entre as 127 nunca ficam cobertas. É uma cota
   pessimista.
2. O BIC prefere **1** componente (127 pontos em 32 dimensões não sustentam uma mistura); a mistura de 2 foi
   imposta. Ela não reproduz a cobertura real mais rápida: $D_N$ mediano 40,1/30,0/23,9 contra real 39,2/26,6/19,5,
   o mesmo erro do gaussiano. **Nenhum modelo paramétrico testado captura a aglomeração das vozes reais.**
   Os modelos paramétricos subestimam a cobertura e, portanto, superestimam $q$ em $N$ grande.
3. O **piso** é o único número que não depende do modelo de vozes: $\Pr(\|c-v_0\|^2<\tau)$, 4 das 38 vozes.
   Mas a forma por voz é ruim (§5a: 2 das 4 já copiam), então o piso real pode ser menor.

**Previsão revisada:** $q(2456)$ entre 0,14 e 0,19 (IC conjunto 0,07–0,29). $q(2456)<0{,}07$ refuta o piso; acima
de 0,29 refuta a teoria. A previsão das leis sem piso de 02 (~0,09) fica **dentro** do IC: a extrapolação não
discrimina com 38 vozes.

## 10. Bits como espectro de informação, por voz

**Derivação** [PROVADO]. Para um livro-código i.i.d., $\Pr(\text{falha}\mid c)=(1-p_c)^N$ com
$e^{-Np_c/(1-p_c)}\le(1-p_c)^N\le e^{-Np_c}$. Escrevendo $L(c)=-\log_2p_c$ (autoinformação da tolerância da voz
sob a distribuição de vozes treinadas):
$$
q(N)\approx\mathbb E_c\exp\!\big(-2^{\log_2N-L(c)}\big)=\Pr\!\big[L(c)>\log_2N\big]+\int h(\ell)\,g(\ell-\log_2N)\,d\ell,
$$
com $g(x)=e^{-2^{-x}}-\mathbf 1[x>0]$ e $\int g=-\gamma_E/\ln2$. Logo, para densidade $h$ suave,
$q(N)\approx\Pr\big[L(c)>\log_2N+\gamma_E/\ln2\big]$, isto é, o limiar efetivo fica 0,83 bit acima de $\log_2N$.
Uma voz passa a ser copiada quando $\log_2N\gtrsim L(c)-0{,}83$. A afirmação "80% de cópia ≈ 10 bits ≈ 1.000
locutores" é o quantil 80% de $L$ (v1 §6.2). Ela depende de $\tau$ e da cauda gaussiana de $P_v$ [HEURÍSTICO].

**Teste por voz** (`b1.json: p10_L`). $-L(c)$ (cauda gaussiana, ou contagem empírica) contra distância simples:

| | Spearman $-L$ (gauss / emp.) | Spearman $d_{LJ}$ | Spearman folga | AUC $-L$ | AUC $d_{LJ}$ | Brier de $1-e^{-N2^{-L}}$ | Brier constante |
|---|---|---|---|---|---|---|---|
| $N=8$ | 0,50 / 0,52 | 0,38 | 0,51 | 0,80 | 0,73 | 0,209 | 0,178 |
| $N=32$ | 0,54 / 0,56 | 0,49 | 0,57 | 0,85 | 0,80 | 0,136 | 0,130 |
| $N=128$ | 0,53 / 0,53 | 0,58 | 0,56 | 0,74 | 0,78 | 0,121 | 0,108 |

**Veredito.** $L(c)$ ordena as vozes melhor que $d_{LJ}$ em $N=8$ e 32, pior em 128, e nunca melhor que a folga
geométrica simples, da qual é quase uma transformação monótona. As probabilidades por voz que o espectro de
informação implica são **piores que uma constante** (Brier 0,209 contra 0,178 em $N=8$): são precisas demais
(§5a). **A forma por voz do espectro de informação não se sustenta.** A forma agregada (quantis de $L$ → $N$
necessário) continua sendo uma conta de modelo, sem teste por voz que a apoie.

A relação com os 10,3 bits por frase permanece **conjectura**: a escala $N^\star\approx2^{I(v;c)}\approx10^3$ é onde
$D_N$ do livro-código gaussiano atinge o ruído da referência, mas os modelos paramétricos subestimam a cobertura
real (§9), então o $N^\star$ real deve ser menor.

---

## 11. Situação final

| Afirmação | v2 |
|---|---|
| Proposição 1 (Bayes, $T=1$, massas nominais) | PROVADO; **refutado** como descrição (q≈0) |
| Proposição 2 e fórmula $\mathbb E_c(1-F_c)^N$ | PROVADO; SIMULADO |
| Mecanismo distorção × tolerância do atrator para vozes novas | compatível com DADOS (74%/57% da queda prevista com 1 parâmetro); teste fraco (2 g.l., IC ±0,15) |
| $T$ | não identificado; $T=1$ *a priori*; largura por voz dá 21, incompatível com a curva |
| $\tau=\|\mu-v_0\|^2$ | hipótese pós-hoc (§2) |
| $M$ como contagem de frases | MORTO; experimento $M/2$ mede $T$ (§4) |
| Vozes treinadas como codewords ruidosos | MORTO (§5b) |
| 4 vozes da grade | explicado como excesso de precisão por voz, não como anomalia (§5a) |
| Pseudo-locutores | nulo compatível (família ~1D + atrator alargado); teste sem poder (§6) |
| Leis assintóticas ($\ln N$, $N^{-2/d}$) em 8–2456 | não se aplicam; sem lei de potência estável (§7) |
| "0,26 ≈ 2/d" | retirado |
| Média de referências | termo cruzado explica o ganho seletivo nas 4 vozes e o nulo nas 38 (§8) |
| $q(2456)$, piso | 0,14–0,19 [0,07–0,29]; piso 0,105 [0,05–0,21]; o piso por voz é pouco confiável |
| Espectro de informação por voz | MORTO como preditor por voz; agregado é HEURÍSTICO |

**Parâmetros, contagem final.**
- Medidos, sem usar cópia: $\Sigma_b$, $\mu$, $v_0$, $W_x$, as vozes de cada livro-código, $\sigma_w$.
- Supostos: $\|\Delta\|^2=0{,}65$ (só suaviza); o espaço comum (o latente do treino $N=128$; muda $q_8$ em ~0,1).
- Fixado *a priori*: $T=1$.
- Ajustado à curva: $\tau$ (1 número, em $q_{128}$).
- Graus de liberdade restantes: 2 ($q_8$, $q_{32}$), ambos dentro do IC; só o nulo plano é rejeitado.
