# Capacidade de voz do decodificador: uma lei de escala medida no áudio (v2)

Versão 2, 17/11/2026. Responde à crítica da rodada 1. A v1 (`relatorio.md`) foi mantida. Só CPU, sem treino; nenhum
repositório alterado. Pasta: `/tmp/claude-0/-home-user-ml2-final-project/4c11fa8d-6b5f-5505-bd7b-deebf5dd9907/scratchpad/teoria2/B/`.
Legenda: **[MEDIDO]** número tirado dos dados sem modelo; **[PROVADO]** segue das hipóteses; **[HEURÍSTICO]**; **[CONJECTURA]**;
**[PREVISTO ✓/✗]** comparado com dado que não entrou em nenhuma calibração; **[AJUSTADO]** escolhido para reproduzir o dado mostrado.

## 0. Resumo

1. **Capacidade de voz medida no áudio [MEDIDO].** $C(N)=I(\text{voz da saída};\text{identidade da referência})$, estimada por
   identificação entre 38 vozes externas (Resemblyzer nas saídas salvas; referências do LibriTTS test-clean):

   | | cota de Fano (acerto) | cota de entropia cruzada (LOVO) | teto do verificador (frase real) | bits de voz no latente |
   |---|---|---|---|---|
   | $N=8$ | **0,16–0,31** [IC até 0,78] | **0,61–0,75** [0,16; 1,20] | 4,2–4,7 (Fano) / 4,3–5,0 (EC) | μ: 17,1; z: 10,0 |
   | $N=32$ | **não medido** (não há áudio salvo de N = 32) | — | | |
   | $N=128$ | **0,89–1,13** [0,39; 1,93] | **1,99–2,03** [1,35; 2,68] | | |
   | $\log_2 38$ | 5,25 | | | |

   $C(128)-C(8)=+1{,}24$ bit (EC, IC 95% [+0,66; +1,82]). Por duplicação de $N$: **+0,31 bit** (EC) ou +0,21 (Fano). O decodificador
   passa ~1–2 dos 10 bits de voz que o código z leva, e ~1/5 do que o verificador consegue de uma frase real.
2. **O fator 1/22 explicado [MEDIDO + HEURÍSTICO].** O KL custa 7 bits (17,1 em μ → 10,0 em z). O resto (10 → 1–2 bits) é do
   decodificador: (a) vozes copiadas também têm pouca informação (2,3 bits contra 1,2 nas não copiadas), então o prior da voz
   padrão não é o gargalo principal; (b) na força 0,6 a capacidade cai 0,5–0,8 bit; ruído só do KL prevê queda de 0,
   ruído só no decodificador prevê 1,5–1,8 bit. Cerca de 1/3 a 1/2 do ruído efetivo fica depois da escala de estilo.
3. **Modelo mínimo: 3 parâmetros** ($\mu_{128}$, $t$, $\gamma$). A ordem das vozes vem de uma estatística geométrica sem
   parâmetro. Tudo em $N=32$ é previsto: $q=0{,}306$ contra 0,302; histograma por voz $\chi^2=4{,}4$ ($p=0{,}49$).
4. **Um único mecanismo, a capacidade, prevê a força 0,6** [PREVISTO ✓, com estimador de Fano]: cópia prevista
   0,34/0,18/0,45 (treinadas/novas/externas) contra 0,26/0,19/0,47. Para R = 4 o sinal fica certo (negativo), mas o tamanho
   sai grande demais.
5. **Morreram**: (i) um posterior bayesiano sobre a geometria do latente com largura dada pela capacidade (zero parâmetros):
   $q=0{,}76/0{,}08/0{,}08$ contra 0,43/0,30/0,21; (ii) a transferência de nível das treinadas para as novas; (iii) o efeito de
   desenho como explicação do peso de $K$.
6. **$M_{\rm eff}=2M+NK$** sai da exposição: o canal de estilo só existe a partir do estágio `en_vae_v1`, com 2.500 passos ×
   48 de LJ, igual ao ajuste fino. Com isso, $M$ pela metade dá $q$ 0,22 → **0,18**; a v1 dizia ≤ 2 pontos, o que estava errado.
   A teoria de `02` dá 0,157, e o prior de contagem ingênuo dá 0,13. **Dois treinos não separam essas previsões**.
7. **Duração [PREVISTO ✓, reforçado].** Frases longas são mais tudo-ou-nada: $T=+0{,}035$ [IC 95% +0,015; +0,053], permutação
   $p=0{,}0002$. As vozes intermediárias caem 14 pontos [−20; −8], com $p<2\times10^{-4}$. A razão de DP da margem é 0,850
   [0,808; 0,894], contra 0,829 previsto ($T^{-0{,}38}$).
8. **Vazamento (duas etapas), uma cota**: $I_{\rm frase}(\text{unidades};\text{locutor})\le C(N_{\rm voz})/9\approx0{,}2$ bit por frase,
   com a premissa enunciada na seção 7.

---

## 1. A capacidade de voz do decodificador, sem modelo (ponto 1)

### 1.1 Definição

$C\equiv I(Y;S)$, com $S$ uniforme sobre $|S|$ vozes de teste e $Y$ o embedding (Resemblyzer, o verificador do projeto) de uma
saída sintetizada a partir de uma referência de $S$. A frase de texto é a mesma em todas as saídas (`__s0`, "would not agree
with…"), então a variação é de voz. Duas cotas inferiores válidas:

- **Fano** [PROVADO]: $I(Y;S)\ge I(\hat S;S)\ge\log_2|S|-h(P_e)-P_e\log_2(|S|-1)$, com $\hat S$ o vizinho de maior cosseno.
- **Entropia cruzada variacional** (Barber–Agakov) [PROVADO]: $I(Y;S)\ge H(S)-\mathbb E[-\log_2 q(S\mid Y)]$ para qualquer
  $q$. Uso $q\propto e^{\beta\cos}$, com $\beta$ escolhido deixando a voz avaliada de fora (LOVO), para não ter otimismo.
- O **plug-in da matriz de confusão** (com correção de Miller–Madow) vale 2,6–4,0 bits, mas com uma amostra por classe ele é
  dominado pelo viés: a correção *aumenta* o número. Está reportado e **não é usado**.
- IC: bootstrap das 38 vozes (1.000 réplicas).

### 1.2 Execução

- `pip install --target pylib resemblyzer webrtcvad-wheels`: funcionou, com pesos incluídos.
- Áudio de referência: LibriTTS test-clean via o espelho indicado, extraído só o `.wav` em `libritts_tc/` (1,4 GB, 4.837 arquivos).
- Saídas: `wavs_wide.tgz`, `wavs_n128.tgz` e `wavs_n200.tgz` extraídos em `wavs_ext/`.
- **`wavs_wide.tgz` só tem N = 8 e N = 128** (50 referências × {R = 1, R = 4}): **C(32) não pode ser medido**. Não interpolei
  como se fosse medida.
- Validação do pipeline: o cosseno saída–referência daqui correlaciona 0,93 (N = 8) e 0,89 (N = 128) com o `sim_ref` do
  `results.csv` (diferença máx. 0,07). Deve ser diferença de pré-processamento ou versão; a mesma régua é usada em todas as
  comparações abaixo.
- Scripts: `b12_embed.py` (embeddings) e `b13_capacidade.py` (cotas e IC); saídas `out/b12_emb.npz`, `out/b13.txt`.

### 1.3 Resultados [MEDIDO]

| condição | acerto | top-5 | Fano (bits) | EC-LOVO (bits) |
|---|---|---|---|---|
| frase real retida × centroide (teto do verificador) | 0,947 | 0,97 | 4,68 [4,00; 5,25] | 4,96 [4,54; 5,23] |
| frase real × frase de referência | 0,895 | 0,95 | 4,21 | 4,31 |
| N = 8, R = 1, × centroide / × frase-ref | 0,184 / 0,132 | 0,45 / 0,42 | 0,31 / 0,16 | 0,75 / 0,61 |
| N = 8, R = 4 | 0,132 / 0,105 | 0,42 / 0,39 | 0,16 / 0,10 | 0,57 / 0,39 |
| N = 128, R = 1 | 0,342 / 0,395 | 0,74 / 0,71 | 0,89 / 1,13 | 1,99 / 2,03 |
| N = 128, R = 4 | 0,211 / 0,211 | 0,74 / 0,61 | 0,39 / 0,39 | 1,57 / 1,37 |

Diferenças pareadas (EC, centroide):

- $C(128)-C(8)=+1{,}24$ [+0,66; +1,82].
- $C(R4)-C(R1)=-0{,}18$ [−0,50; +0,11] em N = 8 e $-0{,}41$ [−0,86; +0,05] em N = 128.
- **R = 4 não aumenta a capacidade**: o ganho é ≤ +0,11 bit com 95% de confiança, contra +2,5 a +5 bits a mais no código.

Identificação saída–saída: mesma referência e mesma condição, frase s0 contra s1, nos treinos plain (16 vozes, log₂16 = 4):

| treino | 1,0 | 0,6 | R = 4 | R = 16 |
|---|---|---|---|---|
| n128_k8 (Fano / EC) | 2,97 / 3,68 | 1,30 / 2,44 | 3,42 / 3,82 | 2,57 / 3,35 |
| n128_k32 | 2,97 / 3,06 | 2,57 / 3,00 | 3,42 / 3,49 | 3,42 / 3,86 |
| n200_k32 | 1,88 / 2,64 | 1,58 / 2,45 | 2,21 / 2,72 | — |

Isto mede **consistência** ($I(Y_{s0};Y_{s1})$, inclui 8 vozes treinadas), não fidelidade à referência. Média de referências
aumenta a consistência (+0,1 a +0,45 bit) e não a fidelidade (tabela anterior): o código médio dá a mesma voz em frases
diferentes, mas não mais perto da referência.

### 1.4 A lei de escala

- **$C(N)\approx C(8)+\theta\ln(N/8)$, com $\theta=0{,}45$ bit por unidade de ln N (EC) ou 0,30 (Fano)**: +0,31 / +0,21 bit por
  duplicação de N [MEDIDO em 2 pontos; a forma log-linear é CONJECTURA, potência $C\propto N^{0{,}35}$ (EC) também passa].
- Em escala: em N = 128 o decodificador passa ≈ 1–2 bits, 20–40% de $\log_2 38$, 10–20% dos 10 bits de voz em z, 6–12% dos
  17 bits em μ.
- Mantido o ritmo log-linear (EC), chegar ao teto do verificador (~4,7 bits) pediria $\ln(N/8)\approx8{,}8$, isto é
  $N\sim5\times10^4$ locutores. Extrapolação de 2 pontos, só para dar escala.
- **Por voz**: em N = 128 as vozes copiadas têm 2,26 bits e as não copiadas 1,17 (Spearman cópia × $c_v$ = 0,23); em N = 8,
  0,45 contra 0,54 (Spearman 0,13). A capacidade não é a mesma coisa que o critério binário de cópia: ela mede resolução *entre*
  vozes, o critério mede o lado da voz padrão.

## 2. De onde vem o fator 1/22 (ponto 2)

A v1 tirou 1/22 de uma temperatura ajustada (razão de dispersões de log-verossimilhança). Agora, em bits medidos:

| estágio | bits de voz | perda | de onde vem o número |
|---|---|---|---|
| μ do codificador (gaussiano, componentes de variância) | 17,1 | — | `b17` |
| z = μ + ruído posterior (o que o decodificador viu no treino) | 10,0 | −7,1 bits: **KL / temperatura posterior** | `b17` |
| saída do decodificador, N = 128 | 1,1 (Fano) – 2,0 (EC) | −8 a −9 bits: **decodificador** | `b13` |
| saída, N = 8 | 0,3–0,75 | | `b13` |

Quatro candidatos:

1. **Prior da voz padrão.** Ele faria a perda vir das vozes atraídas. Só que as vozes copiadas também passam só ~2,3 bits.
   Mesmo se as 21% não copiadas virassem um ponto só, o limite superior seria ~4,4 bits. **Não é o principal.**
2. **Escala de estilo.** A força 1,0 é nominal; ela não reduz informação.
3. **KL / temperatura posterior.** Explica 7 dos ~15 bits perdidos.
4. **Capacidade do decodificador.** Teste discriminante, na força 0,6: ruído aplicado *antes* da escala (KL, codificador) é
   escalado junto com o sinal, e a informação não muda [PROVADO: transformação inversível]. Ruído fixo *depois* da escala
   (decodificador) dá $I(\alpha)=\tfrac12\sum_j\log_2(1+\alpha^2\lambda_j/s^2)$.

   | treino | queda observada (EC / Fano) | prevista, ruído só no decodificador | prevista, ruído só no KL |
   |---|---|---|---|
   | n128_k8 | −1,24 / −1,67 | −2,1 / −1,8 | 0 |
   | n128_k32 | −0,06 / −0,40 | −1,8 / −1,8 | 0 |
   | n200_k32 | −0,19 / −0,30 | −1,6 / −1,1 | 0 |
   | média | −0,50 / −0,79 | −1,8 / −1,5 | 0 |

   O observado fica entre os dois: com $s^2=s^2_{\rm pré}+s^2_{\rm pós}$, ~30–50% do ruído efetivo está depois da escala
   [HEURÍSTICO].

**Conclusão:** o "1/22" é (KL) × (capacidade do decodificador), sendo o decodificador ≈ 4–5× a perda do KL em razão de bits;
não é prior nem escala. E **não é constante**: a capacidade cresce com N (0,3–0,75 → 1,1–2,0 bits), ou seja, o fator efetivo
melhora de ~1/40 para ~1/10 entre N = 8 e 128.

## 3. Modelo mínimo (ponto 3)

**Morto primeiro: posterior com largura dada pela capacidade (zero parâmetros)** [PREVISTO ✗]. `b14_cap.py`:

- Largura de decodificação $s^2(N)$ escolhida para que $\tfrac12\sum\log_2(1+\lambda_j/s^2)=C(N)$ medido.
- $M_{\rm eff}=2M+NK$ derivado (seção 5); posterior sobre as vozes treinadas na geometria de cada treino; ligação do Lema 1.
- Resultado: $q=0{,}76/0{,}08/0{,}08$ (EC) ou 1,00/0,64/0,08 (Fano) contra 0,43/0,30/0,21.
- Com um deslocamento ajustado em N = 128, N = 8 e 32 vão a 1,00 e as treinadas também. A variação de $s^2$ com N
  (43 → 15) entra numa função quadrática da distância e dá um efeito muito mais forte que o observado.
- **A capacidade não pode entrar como largura de um posterior sobre a geometria do latente.**

**Modelo de 3 parâmetros** (`b16_min3.py`). Estado latente por voz $L_v\sim\mathcal N(\mu_N,1)$; cada frase é copiada com
probabilidade $\Phi(L_v/t)$; $\mu_N=\mu_{128}+\gamma\ln(N/128)$. A ordem das vozes é prevista pela estatística de razão de
verossimilhança geométrica $\Lambda_v$ (v1, seção 3.3: $s^2$ da dispersão das treinadas, nenhuma cópia externa usada; a ordem
não depende de $\mu,t,\gamma$).

| parâmetro | valor | calibrado em |
|---|---|---|
| $\mu_{128}$ | 0,908 | contagens por voz em N = 128 (verossimilhança marginal) — **curva de teste, 1 ponto** |
| $t$ (ICC latente $1/(1+t^2)=0{,}87$) | 0,393 | idem, N = 128 |
| $\gamma$ | 0,263 por ln N | $q(8)$ — **curva de teste, 1 ponto** |
| (ordem: $\Lambda_v$) | sem parâmetro | dispersão das treinadas, outra base |

**Tabela ajustado × previsto:**

| quantidade | N = 8 | N = 32 | N = 128 |
|---|---|---|---|
| $q$ obs / modelo | 0,434 / 0,434 *(ajustado)* | 0,302 / **0,306** *(previsto)* | 0,211 / 0,199 *(ajustado)* |
| vozes entre 10 e 90%, obs / modelo | 0,34 / **0,37** *(previsto)* | 0,37 / **0,33** *(previsto)* | 0,26 / 0,26 *(ajustado)* |
| histograma em 6 faixas, $\chi^2$ (p por simulação) | **1,6 (0,91)** *(previsto)* | **4,4 (0,49)** *(previsto)* | 1,5 (0,91) *(ajustado)* |
| ordem das vozes, Spearman($\Lambda_v$, cópia) | **0,54** | **0,54** | **0,56** *(todos previstos)* |

Ressalvas:

- Prever N = 32 a partir de 8 e 128 testa só a linearidade do estado médio em ln N; é um teste fraco, mas genuíno.
- Os histogramas em N = 8 e 32 são previsões de verdade: o ICC veio só de N = 128.
- Extrapolação: $q(512)=0{,}118$, $q(2048)=0{,}064$.

**Ligação com a capacidade.** Trocando $\gamma\ln N$ por $\beta\,C(N)$, com $C$ medido no áudio, $\beta=\gamma/(dC/d\ln N)=0{,}58$
(EC) ou 0,89 (Fano) desvio do estado por bit. Continua com 3 parâmetros, mas agora o termo de N é **capacidade medida**: com $C$
medido em outro N, $q$ é previsto sem tocar na curva. "Cada bit de capacidade de voz vale ~0,6–0,9 DP entre vozes no estado de cópia."

## 4. As quatro falhas da v1 (ponto 4) — teste do canal com capacidade limitada

| falha da v1 | com a capacidade medida | situação |
|---|---|---|
| **Nível absoluto das novas** (0,16/0,12/0,08 contra 0,43/0,30/0,21) | O canal limitado explica *qualitativamente* por que a geometria do latente superestima a evidência: o decodificador passa 1–2 bits, não 10–17. Mas o modelo quantitativo de capacidade como largura de posterior falha pior (seção 3). O nível fica como 1 parâmetro ($\mu_{128}$). | **não resolvido**; a ideia "nível = geometria do latente" está morta |
| **Cópia na força 0,6** (1–6% previsto) | O deslocamento do estado é $\beta\cdot\Delta C$, com $\Delta C$ medido saída–saída na força 0,6 (−0,79 bit, Fano) e $\beta=0{,}89$ (de N e C, sem dados de força): previsto 0,34/0,18/0,45, observado 0,26/0,19/0,47. Com EC ($\Delta C=-0{,}50$, $\beta=0{,}58$): 0,57/0,34/0,73, pouca queda. | **resolvido com Fano**, não com EC; a escolha do estimador é um grau de liberdade *post hoc* |
| **Sinal da média de R = 4** (v1: +0,02, observado −0,02/−0,04) | A capacidade de fidelidade *cai* com R = 4 (−0,18/−0,41 bit) ⇒ previsão −0,04/−0,12 (EC) ou −0,05/−0,25 (Fano) contra −0,02/−0,04. | **sinal resolvido**, tamanho 2–6× grande demais; os IC de $\Delta C$ tocam o zero |
| **Treinadas melhorando com N** no modelo calibrado | Identificação saída–saída R1→R4 entre as 8 treinadas: 0,875 (N = 8) → 1,000 (N = 128), já no teto; externas (8 sorteadas): 0,66 → 0,78. A capacidade das treinadas está saturada (memorização), a das novas cresce. Além disso $M_{\rm eff}=2M+NK$ cresce com N, o que puxa as treinadas para baixo (−0,64 nat de 8 a 128) e compensa parte do empréstimo. | **parcialmente**: 8 vozes no teto, sem poder; o modelo mínimo não tem termo de N para as treinadas, por construção |

## 5. Peso efetivo da voz padrão derivado (ponto 5)

Fatos do código (somente leitura):

- `scripts/cloud/stage_setup.sh` e `README.md`: o estágio `en_vae_v1` parte do Tacotron 2 da NVIDIA (sem entrada de estilo) e
  treina o VAE em LJSpeech por `--max-steps 2500 --batch-size 48`.
- `scripts/coverage/colab_job.sh`: o ajuste de cobertura parte de `ckpt/en_final` com `BATCH=48` e 2.500 passos.

**Derivação [HEURÍSTICO; premissa em negrito].** **O prior sobre códigos de estilo só pode ter sido aprendido onde existia
canal de estilo**: no `en_vae_v1` (120.000 apresentações, todas LJ) e no ajuste fino. A NVIDIA não conta, porque não tinha
entrada de estilo. No ajuste fino cada frase é apresentada $r=120.000/(M+NK)$ vezes. Medindo o peso em "frases do ajuste":
$$M_{\rm eff}=M+\frac{120.000}{r}=M+(M+NK)=2M+NK .$$

- Para 128×32: $M_{\rm eff}=8.096$. Fica na ponta de baixo das calibrações da v1 (8×10³ a 1,4×10⁵); acerta as variantes
  `b2` (8.232 e 11.089) e não as maiores.
- **$M$ = 1.000**: $M_{\rm eff}=6.096$, $\Delta\ln M_{\rm eff}=-0{,}28$ nat (contagem ingênua: −0,69). Com o coeficiente das
  treinadas (1,21 τ por nat): **q 0,222 → 0,179 (−4,3 pts)**. Contagem ingênua: → 0,129 (−9 pts). Teoria `02` ($M/2\equiv2K$):
  0,21 → 0,157 (−5,4 pts). Prior dominado pelo pré-treino NVIDIA (v1, $M_{\rm eff}\ge10^5$): ≤ 1 pt.
- **A v1 estava errada** ao afirmar "≤ 2 pts" como previsão da teoria; isso só vale se o pré-treino sem estilo contasse, o que a
  derivação acima nega.
- **Poder:**
  - A diferença entre esta derivação e `02` é **1,1 pt**. Entre esta e a contagem ingênua, ~5 pts.
  - O DP do efeito principal de semente é 0,054 na fração (`04`), logo 0,076 para a diferença de dois treinos.
  - Separar 4,3 de 5,4 pts está fora de alcance. Separar 4,3 de 9 pts com poder 0,8 pede ≈ 2(1,96 + 0,84)²·0,054²/0,047² ≈ 21
    treinos por braço. Separar "≤ 1 pt" de "−9 pts" pede ~6 por braço.
  - **Um experimento de 2 treinos não discrimina nenhuma das hipóteses.** Com R = 4 vozes não, porque a semente domina (`04`).

## 6. Duração: a previsão *a priori* confirmada, com IC e permutação (ponto 6)

`b15_min.py`. Para cada voz externa nos 3 treinos `-wide` (114 voz-treino), separo as 25 frases em 12 curtas e 12 longas
(ordenadas pela duração sintetizada; a do meio fica de fora).

- Estatística: $T=\overline{|p_{\rm longas}-\tfrac12|-|p_{\rm curtas}-\tfrac12|}$.
- IC: bootstrap de locutor (as 3 réplicas de cada locutor juntas, 2.000 réplicas).
- Nulo: permutação das frases dentro da voz (5.000 réplicas, unilateral).

| medida | efeito | IC 95% | p (permutação) | previsto (grandes desvios, $\tau\propto T^{-0{,}38}$) |
|---|---|---|---|---|
| $T$ (afastamento de ½) | **+0,035** | [+0,015; +0,053] | **0,0002** | > 0 |
| fração de vozes intermediárias (longas − curtas) | **−0,140** | [−0,202; −0,079] | **< 0,0002** | −0,12 a −0,14 (v1, `b7`) |
| razão DP da margem, longas/curtas | **0,850** | [0,808; 0,894] | — | 0,829 (expoente −0,38); 0,781 (−0,5) |

Razão média das durações longas/curtas: 1,64. O expoente −0,38 veio da dispersão por quintil de duração em todos os grupos
(`b3`); a previsão para as metades de cada voz só usa esse expoente, e acerta dentro do IC. Ressalva: as 25 frases de texto são
as mesmas para todas as vozes, então "longa" também quer dizer "estes textos". A permutação dentro da voz controla o acaso,
não o texto.

## 7. Vazamento na arquitetura de duas etapas: uma cota (ponto 7)

**Premissa (única)** [CONJECTURA, apoiada pela seção 1]: o decodificador da etapa B reúne a voz de todas as entradas num
orçamento por frase limitado pela própria capacidade $C$; ele não é um integrador ideal de evidência por quadro. A capacidade
medida (1–2 bits contra 10 disponíveis) é o que sustenta isso.

Com essa premissa, o código e as unidades são dois canais gaussianos sobre as mesmas direções de voz, combinados por
precisão. O peso da voz vazada é $w=\mathrm{SNR}_\ell/(\mathrm{SNR}_\ell+\mathrm{SNR}_c)$ (`02`, eq. 9.2), e no regime de baixo SNR
em que estamos, $I\approx\mathrm{SNR}\cdot k/(2\ln2)$, então $w\approx I_\ell/(I_\ell+I_c)$. A meta $w\le0{,}1$ (piso de $q$ ≈ 0,09–0,14
para $N_{\rm voz}$ = 128–2.456, `02`, tabela 9.2) dá:
$$
\boxed{\,I_{\rm frase}(\text{unidades};\text{locutor})\ \le\ C(N_{\rm voz})/9\ \approx\ 0{,}2\ \text{bit por frase (com }C(128)=2{,}0\text{, EC)}\,}
$$

Como medir, com o mesmo protocolo da seção 1: identificar o locutor entre 38 a partir das unidades agregadas por frase. 0,2 bit
de Fano corresponde a ~14% de acerto (acaso 2,6%). Se a premissa falhar (integração ideal por quadro), a cota vira ~0,2/$T_{\rm eff}$
bit por quadro (0,01 bit/quadro com $T_{\rm eff}=20$). Por isso o teste da premissa vem antes: medir a capacidade de B para um
canal de voz *conhecido* injetado só nas unidades, e ver se ela satura em ~C ou cresce com o número de quadros.

## 8. Previsões (atualizadas)

| # | experimento | previsão v2 | decide |
|---|---|---|---|
| P1 | $C(N)$ em N = 32, 512, 2.048 (só síntese + embeddings) | EC: 1,38 / 2,64 / 3,27 bits (log-linear) | $C$ saturar ≤ 2,5 bits em 2.048 rejeita a forma log |
| P2 | $q(N)$ com $C(N)$ medido, $\beta=0{,}58$–0,89 | $q(512)\approx0{,}12$, $q(2048)\approx0{,}06$ (via $\gamma$) | — |
| P3 | $M$ = 1.000 | 0,22 → 0,18 (derivado); contagem 0,13; `02` 0,157 | ≥ 6 treinos por braço para separar "≤ 1 pt" de "−9" |
| P4 | duas etapas | $I_{\rm frase}(u;\text{loc})\le C/9\approx0{,}2$ bit mantém $q$ a ≤ +5 pts de uma etapa | |
| P5 | frases 2× mais longas | vozes intermediárias ×0,77; DP da margem ×0,77 | |
| P6 | força 0,4 | $\Delta C$ (medido) × β prevê a cópia; ruído só do KL prevê $\Delta C=0$ | queda de $C$ nula rejeita a parte "depois da escala" |
| P7 | latente k = 2 / 8 | k = 8 mantém $C$; k = 2 baixa $C$ em ~60% (fração de $\tfrac12\sum\log_2$ fora do top-2) | |

## 9. Contas de parâmetros e situação

- **Modelo mínimo: 3 parâmetros**, dois deles ajustados em 2 dos 3 pontos da curva de teste ($\mu_{128}$, $\gamma$) e um na
  dispersão por voz em N = 128 ($t$). A estatística que ordena as vozes vem de outra base (treinadas).
- **Medido sem modelo**: $C(8)$, $C(128)$ e suas diferenças, o custo do KL, $\Delta C$ na força 0,6 e em R = 4, o efeito da duração.
- **Previsto**: N = 32 inteiro, histogramas em N = 8 e 32, a ordem das vozes nos 3 N, a cópia na força 0,6 (via $\Delta C$ medido;
  só com Fano), o sinal de R = 4.
- **Rejeitado nesta rodada**: posterior com largura dada pela capacidade (zero parâmetros); "≤ 2 pts para M/2" da v1;
  efeito de desenho; nível vindo da geometria do latente.

Reprodução (além dos scripts da v1):
```
python3 -I b12_embed.py wavs_ext libritts_tc ref_ids.txt out/b12_emb.npz 30
python3 -I b13_capacidade.py out/b12_emb.npz <results_root> 1000     # C(N), cotas, IC        -> out/b13.txt, b13.json
python3 -I b14_cap.py out/b0.npz ce|fano                             # capacidade como largura -> out/b14_*.txt (morto)
python3 -I b15_min.py out/b0.npz out/b12_emb.npz                     # info por voz, duração, M/2 -> out/b15.txt
python3 -I b16_min3.py out/b0.npz                                    # modelo de 3 parâmetros  -> out/b16.txt
python3 -I b17_forca.py out/b0.npz out/b13.json                      # mecanismo do fator      -> out/b17.txt
python3 -I b18_capshift.py out/b0.npz out/b13.json out/b16.json      # força 0,6 e R=4 via ΔC  -> out/b18.txt
python3 -I b19_seen.py out/b0.npz out/b12_emb.npz                    # treinadas × externas    -> out/b19.txt
```
