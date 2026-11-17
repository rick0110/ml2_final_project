# Clonagem *zero-shot* como codificação de fonte — v3: temperatura, capacidade e cobertura

Documento de trabalho (teoria2/A), 17/11/2026. v1 (`relatorio.md`) e v2 (`relatorio_v2.md`) ficam como estão.
Pergunta desta rodada: a "largura por voz" que implica $T\approx21$ (v2) e o "decodificador usa ~1/22 da evidência"
(análise bayesiana independente) são o mesmo fato? E a temperatura cai com $N$?
Scripts novos: `c1_temperatura.py`, `c2_nominal.py`, `c3_extrap.py`, `c4_loro.py`; saídas `out/c1.json`–`out/c4.json`.
Só CPU, `python3 -I`, nenhum repositório tocado.

---

## 0. Resumo

**Resposta direta à hipótese: a temperatura NÃO cai com $N$.** Estimada pela forma da resposta por voz em cada
treino, $T(8)=18{,}7$, $T(32)=19{,}4$, $T(128)=27{,}1$, com ICs de 11 a 29–71 que se sobrepõem.
$\Pr(T_8>T_{128})=0{,}23$ no bootstrap. Um $T$ comum ($21{,}6$) não é rejeitado. A capacidade usada fica em
$C\approx0{,}8$–$1{,}1$ bit em todos os $N$. **Toda a queda de $q$ com $N$ vem do livro-código (cobertura/massa),
nada da capacidade.** A convergência com a outra análise é numérica ($1/T\approx1/19$–$1/27$ contra $1/22$), mas é
uma **propriedade fixa do decodificador** (ou um artefato comum de geometria, §6), não um mecanismo de escala.

| treino ($K=32$) | $T$ (IC 90%) | $1/T$ | $C(T)$ bits (IC) | $\lambda=\ln(M_{\rm ef}/K)$ |
|---|---|---|---|---|
| $N=8$ | 18,7 (11,8–29,1) | 1/19 | 1,13 (0,75–1,70) | 4,26 |
| $N=32$ | 19,4 (11,7–26,3) | 1/19 | 1,09 (0,83–1,71) | 4,98 |
| $N=128$ | 27,1 (11,2–71,1) | 1/27 | 0,80 (0,32–1,77) | 5,04 |
| comum | 21,6 | 1/22 | 1,06 | 3,91 / 4,72 / 5,45 |
| nominal $\ln(2000/32)$ | | | | 4,14 |

**Achados que mudam a v2.**
1. **Com $T$ alto, a massa nominal volta a funcionar.** O nível ajustado por treino é
   $\lambda=4{,}3$–$5{,}0$, contra o nominal $\ln(M/K)=4{,}14$, ou seja $M_{\rm ef}\approx2.300$–$5.000$ contra
   2.000 frases LJ. O "fracasso de $M$" da v2 ($M_{\rm ef}=10^{26}$) era artefato de $T=1$.
   A v2 rejeitou $T$ alto supondo massa fixa entre treinos; com o nível livre por treino, $T\approx21$ é compatível
   com a curva.
2. **Diluição de massa.** O nível sobe $0{,}56$ por unidade de $\ln N$: a massa efetiva das vozes LibriTTS cresce
   como $N^{0,44}$, não como $N$. Em bits: **cada duplicação de $N$ acrescenta 1 bit de livro-código, dos quais
   ~0,44 bit chega à decisão; a capacidade acrescenta 0.**
3. **Decomposição de $q_8\to q_{128}$** (Shapley, nível fixo): cobertura −0,40 a −0,48 (IC −0,50 a −0,39);
   capacidade +0,10 a +0,12 (IC −0,16 a +0,31, sinal errado e compatível com 0).

**Parâmetros e previsões** (38 vozes externas, espaço comum = latente do treino $N=128$):

| previsão para o treino $r$ | ajustado no nível de $r$? | $q_8$ | $q_{32}$ | $q_{128}$ |
|---|---|---|---|---|
| observado | — | 0,434 | 0,302 | 0,211 |
| $T$ próprio (forma) + $\lambda$ **nominal** | não (0 parâmetros de nível) | 0,409 | 0,186 | 0,112 |
| $T$ comum (forma) + $\lambda$ nominal | não | 0,481 | 0,214 | 0,090 |
| $T$ comum + $\lambda$ interpolado/extrapolado dos outros 2 treinos | não (deixando o treino de fora) | 0,453 | 0,294 | 0,223 |
| *nulo*: logit$(q)$ linear em $\ln N$, deixando o treino de fora | não | 0,412 | (interp.) | 0,196 |

A previsão deixando o treino de fora acerta a 0,01–0,02, **mas o nulo de dois parâmetros acerta igual**: com
3 pontos, esse teste não distingue a teoria de uma tendência suave. A versão com massa nominal (nenhum parâmetro de
nível) acerta $N=8$ e erra $N=32$ e 128 por ~0,1 no sentido de prever queda demais. A diferença é exatamente a
diluição de massa do item 2.

---

## 1. $T$ por treino

**Método** [DADOS, `c1_temperatura.py`]. O canal de Gibbs dá o logit de cópia de cada voz:
$$
\operatorname{logit}P(\text{cópia}\mid c)=\ln\!\sum_i e^{-\|c-v_i\|^2/2T}+\frac{\|c-v_0\|^2}{2T}-\lambda,\qquad \lambda=\ln\frac{M_{\rm ef}}K .
$$
O ajuste é binomial, com 25 frases por voz, livros-código reais e o espaço comum (latente do treino $N=128$,
branqueado). Em cada treino, $T$ fixa a **forma**: quão depressa a probabilidade muda com a geometria da voz.
$\lambda$ fixa o **nível** (a cópia média). As duas coisas são quase ortogonais. O bootstrap (150 reamostras das
38 vozes) refaz os dois ajustes.

**Resultado:** tabela do §0. $T$ não cai com $N$. A razão de verossimilhança entre "$T$ comum" e "$T$ por treino"
é 10,4 em escala binomial. A superdispersão por voz é ~15×; corrigida, fica em ~0,7 contra $\chi^2_2$, longe de
significativa.

**Capacidade usada** [HEURÍSTICO]. O canal de Gibbs a temperatura $T$ é o posterior de Bayes para códigos com
ruído $T$ por direção, em vez de 1. A informação de voz que o decodificador "enxerga" é
$$
C(T)=\tfrac12\sum_j\log_2\!\big(1+\lambda_j/T\big),
$$
com $\lambda_j$ o espectro entre locutores medido: $C(1)=10{,}3$, $C(5)=3{,}4$, $C(10)=2{,}0$, $C(20)=1{,}06$,
$C(27)=0{,}81$ bits. Com $T(N)$ medido, $C\approx1$ bit em todos os $N$.
Comparação com a outra análise (1,1–2,4 de ~14 bits; fração 1/22): **mesma ordem de grandeza**. Meu $C$ fica no
extremo inferior porque a escala total aqui é 10,3 bits (canal $z$), não 14.

---

## 2. Previsão sem ajuste no nível do treino previsto

As contas estão em `c2.json` e `c4.json`.

- **Zero parâmetros de nível** ($\lambda=\ln(2000/32)$; $T$ de forma, próprio ou comum): 0,41–0,48 / 0,19–0,21 /
  0,09–0,11. Acerta $N=8$ e prevê queda demais depois.
- **Nível dos outros dois treinos** (interpolação ou extrapolação de $\lambda$ em $\ln N$), com $T$ comum:
  0,453 / 0,294 / 0,223. Erros de 0,02 / 0,01 / 0,01. Com $T$ próprio de cada treino, mais ruidoso:
  0,565 / 0,252 / 0,316.

**Contagem de parâmetros honesta.**
- Medidos: espectro, $v_0$, ruído, livros-código.
- De forma: $T$, um número, estimado com a forma das respostas por voz dos três treinos (inclusive o previsto, mas
  não o nível dele).
- De nível: dois ($a$ e $b$ em $\lambda=a+b\ln N$), ajustados nos outros dois treinos.
- Total: 3 parâmetros para 3 pontos. Cada previsão deixando o treino de fora usa 0 parâmetros do nível do treino
  previsto.

**O que isso vale.** Um logit linear em $\ln N$ (2 parâmetros) faz a mesma previsão deixando o treino de fora com
erro semelhante (0,412 em $N=8$, 0,196 em $N=128$). **A teoria não ganha da tendência nula na curva.** O que ela
acrescenta é a *interpretação* dos dois números de nível: $\lambda$ em torno do valor nominal, com inclinação
0,56 (diluição).

---

## 3. Decomposição: cobertura × capacidade

Contrafactuais no modelo ajustado (`c1.json: decomposicao`): $q(\text{livro-código de }N_a,\ T\text{ de }N_b)$,
média das duas ordens (Shapley).

| passo | total (modelo) | cobertura | capacidade |
|---|---|---|---|
| 8→32 | −0,27 / −0,22 | −0,28 / −0,24 | +0,02 / +0,01 |
| 32→128 | −0,09 / −0,07 | −0,18 / −0,14 | +0,09 / +0,06 |
| 8→128 | −0,36 / −0,30 | −0,48 / −0,40 | +0,12 / +0,10 |
| IC 90% (8→128, bootstrap) | [−0,66; −0,11] | [−0,50; −0,39] | [−0,16; +0,31] |

(Primeiro número: $\lambda$ fixo na média dos ajustes; segundo: $\lambda$ nominal.)

**Leitura.** A cobertura explica mais que toda a queda; a capacidade não ajuda (a estimativa pontual piora, porque
$T_{128}$ saiu maior). A queda observada de 0,43→0,21 é **100% cobertura** dentro da incerteza.

Em unidades de informação, no regime de $T$ alto (lei dos grandes números), o logit de cópia de uma voz é
$$
\ln N+\ln\mathbb E_v\,k_T(c,v)+\frac{\|c-v_0\|^2}{2T}-\lambda(N),\qquad \frac{d\,\operatorname{logit}}{d\ln N}=1-0{,}56=0{,}44 .
$$
Cada duplicação de $N$ dá 1 bit de livro-código; a diluição devolve 0,56 bit; a capacidade dá 0.

**Por que "cobertura" aqui não é a distorção do vizinho da v1.** Com $T\approx21$, o núcleo cobre a população inteira
($\lambda_1=8{,}3<T$). O que importa é a **massa** de vozes treinadas que o canal "vê" em torno de $c$, não a
distância ao vizinho mais próximo. Por isso o livro-código gaussiano agora funciona (`c3.json`: 0,40/0,31/0,21 com o
gaussiano, contra 0,42/0,30 com subconjuntos das vozes reais), enquanto na v1 falhava.

---

## 4. $T$ e $K$

Grade com $N$ fixo, ajuste nas 12 referências de cada treino (4 novas + 8 treinadas; ICs muito largos):

| $N$ | $K=8$ | $K=32$ | $K=128$ |
|---|---|---|---|
| 8 | 5,3 / 8,4 | 9,5 | 15,1 / 18,2 |
| 32 | 13,2 / 8,8 | 15,1 | 15,7 / 15,6 |
| 128 | 12,0 | 15,3 / 11,1 | — |

Nível $\lambda$ em $N=8$: 14,5/9,2 ($K=8$) → 6,2 ($K=32$) → 3,2/3,3 ($K=128$).

**Leitura.** $T$ não cai com $K$; a estimativa pontual até sobe, e a capacidade não melhora com $K$. O efeito de $K$
está no **nível**: $\lambda$ cai ~8 unidades de $K=8$ para 128, contra $\ln16=2{,}8$ do nominal. É o efeito forte de
$K$ nas vozes treinadas (v2 §5b), que continua sem mecanismo: **$K$ age como massa ~3× mais forte que a contagem
de frases, não como capacidade**.

Ressalva: as temperaturas da grade (5–18) são menores que as das 38 externas (19–27). As vozes treinadas pesam
nessas 12 referências, então $T$ depende do conjunto de vozes. Isso não é o esperado de uma constante do
decodificador e reforça a ressalva do §6.

---

## 5. Consequências para a arquitetura em duas etapas

- **O gargalo atual é a capacidade**: ~1 bit usado de ~10 disponíveis. Ela não sobe com $N$ nem com $K$.
  No modelo, com massa nominal, baixar $T$ de 20 para 5 ($C$ de 1,1 para 3,4 bits) leva $q_8$ de 0,44 para 0,08
  e $q_{128}$ de 0,084 para 0,043 (`c2.json: varre_T`). **Ganhar capacidade vale mais que multiplicar $N$ por 16.**
- **A etapa de voz deve ser dimensionada pela capacidade, não só alimentada com mais locutores.** Mais locutores sem
  transcrição melhoram só o termo de cobertura, que rende 0,44 bit de logit por duplicação. Isso continua útil:
  $N_{\rm voz}=2048$ dá ~0,10 no modelo. O que pode baixar $T$ não aparece nos dados disponíveis; candidatos
  [CONJECTURA] são um condicionamento de voz mais largo ou com mais camadas, peso de KL e treino mais longo (os
  2.500 passos partem de um *checkpoint* de um só locutor).
- **Diagnóstico barato**: medir $T$ pela largura por voz (este método) em cada treino novo. É o número que diz se
  uma mudança de arquitetura aumentou a capacidade.

**Previsões falsificáveis** (38 vozes; melhor com ≥200):

| treino | termo de capacidade | termo de cobertura | $q$ previsto |
|---|---|---|---|
| $N=512$, $K=32$, arquitetura atual | $T=20$ (11–30), $C\approx1$ bit, igual a $N\le128$ | $\lambda(512)=6{,}2$ (diluição 0,56) | 0,14 (gaussiano) |
| $N=2048$ | $T$ igual | $\lambda=7{,}0$ | 0,096 |
| $N=2456$ | $T$ igual | $\lambda=7{,}1$ | 0,090 |
| $N=512$ sem diluição (massa ∝ $N$) | $T$ igual | $\lambda=\lambda(128)$ | 0,03 |
| duas etapas, sem voz dominante, $N_{\rm voz}=512$ | se $T$ é propriedade do decodificador: $T\approx20$ | — | $T$ medido < 10 refuta "capacidade fixa" |

Critérios de refutação: $T(512)<10$ (IC fora de 11–30) refuta "a capacidade não depende de $N$";
$q(512)<0{,}06$ refuta a diluição; $q(512)>0{,}22$ refuta o termo de cobertura.

---

## 6. Se $T$ não depende de $N$ (é o caso)

Digo com clareza: **a hipótese "treinar com mais locutores baixa $T$" é rejeitada pelos dados disponíveis**,
dentro de ICs largos. A convergência com a análise bayesiana então é uma de duas coisas:

1. **Propriedade fixa do decodificador**: ele usa ~1/20 da evidência de voz da referência em qualquer $N$, o que
   aponta para limitação de arquitetura ou de treino, não de dados. Implicação: escalar $N$ rende devagar (0,44 bit de
   logit por duplicação), e o que mudaria a curva é aumentar a capacidade.
2. **Artefato comum de geometria**: as duas análises medem o quanto a decisão do decodificador depende do latente
   da referência. Se o latente (espaço do treino $N=128$) for uma medida ruidosa do espaço onde o decodificador
   decide, os dois métodos veem uma resposta "amolecida" pelo mesmo ruído, e a concordância 1/22 ≈ 1/21 não é
   independente. Dois indícios a favor: $T$ muda com o conjunto de vozes (grade 5–18 contra externas 19–27), e na v2
   §3 a previsão mudava em ~0,1 conforme o espaço latente.

Não consigo separar 1 de 2 com os dados salvos. O teste é o diagnóstico da arquitetura: se aumentar a largura do
condicionamento de voz baixar o $T$ medido e o $q$ junto, é (1); se $q$ mudar sem $T$ mudar, ou o contrário, é (2).

---

## 7. Situação (atualiza a v2)

| Afirmação | v3 |
|---|---|
| $T$ cai com $N$ | **rejeitada** ($T\approx19$–27, sem tendência; $T$ comum aceito) |
| capacidade usada $\approx1$ bit, $1/T\approx1/22$ | DADOS; mesma ordem da análise independente; pode ser artefato comum (§6) |
| massa nominal $M$ explica o atrator | **reabilitada** com $T$ alto ($M_{\rm ef}$ 2.300–5.000 contra 2.000) |
| diluição: massa LibriTTS ∝ $N^{0,44}$ | DADOS (3 pontos); mecanismo não derivado |
| queda de $q$ = cobertura | DADOS no modelo; capacidade contribui 0 (IC −0,16 a +0,31) |
| $K$ age por nível (~3× o nominal), não por capacidade | DADOS na grade (ruidoso) |
| previsão deixando o treino de fora | 0,01–0,02 de erro, mas **igual ao nulo** logit-linear |
| regime de distorção ($T\approx1$) da v1/v2 | **não preferido**: compatível com a curva, incompatível com a largura por voz; com nível livre, $T$ alto explica curva, largura e $M$ nominal |
| piso 0,10 da v1/v2 | **não sobrevive** no regime de $T$ alto: a massa acumula e $q\to0$ lentamente ($q(10^4)\approx0{,}06$) |
