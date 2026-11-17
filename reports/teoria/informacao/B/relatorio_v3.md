# Capacidade de voz C(N, K) medida no áudio: curva, decomposição e cadeia de informação (v3)

Versão 3, 17/11/2026, rodada 3 da crítica. As versões v1 e v2 foram mantidas. Só CPU, sem treino; nenhum repositório alterado; dados
salvos lidos com `python3 -I`. Pasta: `/tmp/claude-0/-home-user-ml2-final-project/4c11fa8d-6b5f-5505-bd7b-deebf5dd9907/scratchpad/teoria2/B/`.

Legenda: **[MEDIDO]** sem modelo; **[MODELO]** usa covariâncias medidas num modelo gaussiano; **[CONJECTURA]**.
"EC" é a cota inferior de entropia cruzada com β escolhido deixando a voz de fora (LOVO). "Fano" é a cota pela taxa de acerto.
As duas são cotas **inferiores** de $I(\hat Y;S)$.

## 0. Resumo

| conjunto de vozes (teto) | N=2 | N=8 | N=32 | N=128 | N=200 | inclinação (bits por duplicação) |
|---|---|---|---|---|---|---|
| **38 externas, nunca ouvidas (5,25), EC** | — | **0,75** [0,28; 1,20] | não há áudio | **1,99** [1,35; 2,60] | — | **s_N = 0,31** [0,17; 0,47] |
| 38 externas, Fano | — | 0,31 | — | 0,89–1,13 | — | 0,15–0,21 |
| 12 comuns = 8 treinadas + 4 novas (3,58), EC | — | 1,38–2,29 | 1,63–2,58 | 2,13–2,30 | 2,05 | s_N = +0,10 [−0,09; +0,30]; s_K = +0,10 [−0,03; +0,24] |
| 8 treinadas (3,0), EC | — | 1,79–2,99 | 1,83–2,26 | 1,77–2,12 | 1,92 | s_N = −0,11; s_K = +0,17 |

- **Decomposição** (38 externas, gaussiana nas 6 direções principais):
  - $H(\hat Y)$ sobe **+1,44 bit** [+0,68; +2,32] de N=8 a 128;
  - $H(\hat Y\mid S)$ fica **+0,07** [−1,2; +1,5];
  - $I$ sobe **+1,37**.
  - **O ganho com N está no espalhamento das vozes alcançáveis (cobertura), e o ruído por voz não muda**. É compatível com a
    temperatura constante da análise A.
  - Ressalva: na variância total (escalar), o ruído por voz medido como R=1 × R=4 também cresce um pouco (+0,032).
- **Cadeia** (38 vozes, Fano):

  | elo | N=8 | N=128 |
  |---|---|---|
  | fala real [MEDIDO] | 4,2–4,7 | 4,2–4,7 |
  | μ [MODELO] | 5,1 | 5,2 |
  | z [MODELO] | 4,4 | 4,8 |
  | saída [MEDIDO] | 0,3 | 0,9–1,1 |

  Quase toda a perda está no **decodificador**: ~4 bits em N=8 e ~3,7 em N=128. O KL custa ~0,5 bit nesta escala.
- **Extrapolação** (log-linear, 38 externas, EC):
  - $C(512)=2{,}61$ [1,86; 3,54] e $C(2048)=3{,}23$ [2,25; 4,45].
  - A saída carregaria tanto quanto a fala real (4,96 bits EC) em $N^\*\approx10^5$ locutores (IC $10^{3{,}6}$ a $10^{8{,}2}$).
- **Variantes**: o vetor ECAPA **sozinho (força 0) carrega 0,0 bit** nos treinos `-spk`; o decodificador o ignora (medido).
  Quando o estilo é retirado em metade do treino (`-spkdrop`), o ECAPA passa a carregar 0,6–1,3 bit. Entre treinos, cópia e C
  correlacionam −0,49 (Spearman, 12 treinos).
- **Força 0,6** baixa C em 8 de 10 treinos (até −1,4 bit em K=8). **R = 4 e R = 16** mudam C entre −0,40 e +0,31 bit, sem
  direção consistente: confirma a v2.

---

## 1. Dados e método

### Conteúdo dos arquivos

Listado com `tar tzf`; extraído em `wavs_ext/` e `wavs_ext2/`.

| arquivo | treinos | condições | frases |
|---|---|---|---|
| `wavs_wide` | n8_k32-wide, n128_k32-wide (50 refs: 38 externas + 8 treinadas + 4 novas) | 1,0; 1,0_r4 | s0 |
| `wavs_n128` | n128_k8, n128_k32 (16 refs: 8 + 4 + 4 externas) | 1,0; 0,6; r4; r16 | s0, s1 |
| `wavs_n200` | n200_k32 (16) | 1,0; 0,6; r4 | s0, s1 |
| `wavs_spk_cov`, `_cov2` | n2_k128-spk, n8_k128-spk, n8_k8-spk (12 refs) | 0,0; 0,6; 1,0 | s0, s1 |
| `wavs_spkdrop_cov`, `_cov3` | n32_k128-spkdrop, n32_k8-spkdrop, n8_k128-spkdrop (12) | 0,0; 0,6; 1,0 | s0, s1 |
| `wavs_aug` | n8_k8-aug, n8_k128-aug (12) | 0,6; 1,0 | s0, s1 |

Não há áudio de nenhum treino base com N = 8 ou 32 além do `n8_k32-wide`. Também não há N = 32 nas 38 externas. Os pontos
N = 8 e 32 da grade vêm de **variantes** (`-spk`, `-aug`, `-spkdrop`), que em `04` não diferem do treino base por mais de ~2 DP
de semente.

### Conjunto comum de vozes

As **12 referências** (8 treinadas + 4 novas) são as mesmas em todos os treinos com N ≥ 8.
- Em N = 2 as treinadas são outras frases de 2 locutores; só as 4 novas são comuns.
- Áudio real dessas 12 vozes: LibriTTS train-clean-100, só os 12 locutores, extraído de `openslr.elda.org/resources/60` em
  `libritts_tc100/` (3.538 arquivos).
  - A primeira tentativa falhou porque os nomes no arquivo começam com `./`; foi refeita.
- As 38 externas usam o test-clean da v2.

### Estimadores

- Embedding Resemblyzer, saída identificada contra o centroide de até 30 frases reais de cada voz (excluindo a frase de
  referência); Fano e EC-LOVO.
- Bootstrap de vozes. Para inclinações, o bootstrap é pareado: as mesmas vozes são reamostradas em todos os treinos.
- O **teto** é $\log_2$ do número de vozes. Com 12 vozes o teto é 3,58 bits e o verificador em fala real já dá 3,52, então o
  conjunto pequeno **satura e comprime diferenças**. As 38 externas (teto 5,25; fala real 4,2–5,0) são a medida principal.

### Scripts

`b20_embed2.py` (embeddings), `b21_curva.py` (consistência), `b22_cadeia_latente.py` (μ e z nas 8 treinadas), `b23_latente38.py`
(μ e z para 38 vozes), `b24_lei.py` (curva, inclinações, q × C, força e R, decomposição em 12 vozes), `b25_decomp38.py`
(decomposição nas 38), `b26_extrap.py` (extrapolação), `b27_extra.py` (fala real, treinadas, força 0).
Saídas em `out/b2x.txt`.

## 2. A curva C(N, K) [MEDIDO]

Fidelidade, força 1,0, R = 1. Mais de uma frase por voz quando existe (s0 e s1).

| treino | N | K | variante | 12 vozes EC [IC 95%] | Fano | 8 treinadas EC | 4 novas EC | q (treinadas + novas) |
|---|---|---|---|---|---|---|---|---|
| n2_k128-spk | 2 | 128 | spk | — | — | — | 0,22 | 0,52 |
| n8_k8-aug | 8 | 8 | aug | 1,63 [0,72; 2,68] | 1,16 | 1,79 | 0,30 | 0,57 |
| n8_k8-spk | 8 | 8 | spk | 1,67 [0,80; 2,57] | 1,33 | 2,26 | 0,57 | 0,61 |
| n8_k32-wide | 8 | 32 | base | 1,38 [0,59; 2,02] | 1,16 | 1,92 | (1 frase) | 0,36 |
| n8_k128-aug | 8 | 128 | aug | 2,29 [1,44; 3,57] | 1,51 | 2,51 | 0,63 | 0,30 |
| n8_k128-spk | 8 | 128 | spk | 1,98 [0,99; 2,82] | 1,51 | 2,99 | 0,74 | 0,26 |
| n8_k128-spkdrop | 8 | 128 | spkdrop | 1,63 [0,43; 2,79] | 1,71 | 2,98 | 0,08 | 0,29 |
| n32_k8-spkdrop | 32 | 8 | spkdrop | 1,63 [1,00; 2,42] | 0,86 | 1,83 | 0,61 | 0,42 |
| n32_k128-spkdrop | 32 | 128 | spkdrop | 2,58 [2,12; 3,45] | 1,91 | 2,26 | 0,84 | 0,24 |
| n128_k8 | 128 | 8 | base | 2,30 [1,55; 3,34] | 2,13 | 1,77 | 1,88 | 0,47 |
| n128_k32 | 128 | 32 | base | 2,13 [1,50; 2,68] | 1,16 | 1,95 | 0,68 | 0,25 |
| n128_k32-wide | 128 | 32 | base (mesmo checkpoint) | 2,16 [1,45; 3,24] | 1,51 | 2,12 | 0,65 | 0,23 |
| n200_k32 | 200 | 32 | base | 2,05 [1,35; 2,93] | 1,33 | 1,92 | 0,56 | 0,34 |
| **38 externas** n8_k32-wide | 8 | 32 | base | **0,75** [0,28; 1,20] | 0,31 | | | q = 0,43 |
| **38 externas** n128_k32-wide | 128 | 32 | base | **1,99** [1,35; 2,60] | 0,89–1,13 | | | q = 0,21 |

Lei $I=c+s_N\log_2N+s_K\log_2K$ nas 12 vozes (10 treinos, N ≥ 8, sem `-wide`):
- $c=1{,}01$, $s_N=+0{,}10$ [−0,09; +0,30], $s_K=+0{,}10$ [−0,03; +0,24]; resíduo RMSE 0,23 bit (dentro dos IC por treino).
- Separando as vozes:
  - **treinadas**: $s_N=-0{,}11$, $s_K=+0{,}17$ — C delas depende de K, não de N, como a cópia (K^0,46);
  - **novas** (só 4, teto 2): $s_N=+0{,}12$, $s_K=-0{,}04$.

O que isso diz:
1. **Nas vozes nunca ouvidas a lei log-linear tem inclinação 0,31 bit por duplicação de N** [0,17; 0,47]. São só 2 pontos (8 e 128),
   então a forma "log" **não está testada**; potência ($C\propto N^{0{,}35}$) passa igual.
2. No conjunto de 12, a inclinação em N é pequena e não significativa. Isso é esperado: 2/3 dele são vozes treinadas, cuja
   capacidade não depende de N, e o teto de 3,58 bits está perto (fala real: 3,52).
   **O conjunto de 12 vozes é pequeno demais para medir a curva das vozes novas**: 4 novas, teto 2 bits, IC de ±1 bit.
3. Em K o sinal só aparece nas treinadas (+0,17 bit por duplicação): K melhora a memorização das vozes treinadas, não a
   generalização.

## 3. Decomposição $I(\hat Y;S)=H(\hat Y)-H(\hat Y\mid S)$ — o teste que reconcilia as duas teorias

**Pergunta.** A análise A não vê mudança na temperatura de resposta por voz entre N; aqui C cresce +1,24 bit. As duas coisas
só são compatíveis se o ganho estiver em $H(\hat Y)$ (mais vozes de saída distintas: cobertura), com $H(\hat Y\mid S)$ constante
(ruído por voz fixo).

**Método.**
- Embeddings unitários projetados numa base PCA comum, ajustada em todas as saídas mais os centroides reais.
- Entropias gaussianas $\tfrac12\log_2\det(2\pi e\Sigma)$:
  - $\Sigma_{\rm tot}$ das saídas;
  - $\Sigma_{\rm dentro}$ da diferença entre duas saídas da mesma voz. Nas 38 externas, R = 1 × R = 4, porque só há a frase s0;
    nas 12 vozes, frases s0 × s1.
- Versão escalar (sem PCA): variância entre vozes e dentro da voz.
- IC: bootstrap de vozes, pareado entre N.

**38 externas, N = 8 → 128** (`b25`):

| grandeza | N = 8 | N = 128 | diferença [IC 95%] |
|---|---|---|---|
| $H(\hat Y)$, k = 6 | −8,40 | −6,96 | **+1,44** [+0,68; +2,32] |
| $H(\hat Y\mid S)$, k = 6 | −13,53 | −13,46 | **+0,07** [−1,24; +1,54] |
| $I_G$, k = 6 | 5,13 | 6,50 | +1,37 [−0,10; +2,69] |
| $H(\hat Y)$, k = 3 | −2,45 | −1,96 | +0,49 [−0,03; +0,99] |
| $H(\hat Y\mid S)$, k = 3 | −5,75 | −6,48 | −0,72 [−1,54; +0,25] |
| espalhamento entre vozes (escalar) | 0,178 | 0,221 | +0,043 [+0,018; +0,066] |
| ruído por voz R1 × R4 (escalar) | 0,095 | 0,127 | +0,032 [+0,017; +0,048] |
| erro saída → centroide real | 0,667 | 0,546 | −0,121 [−0,156; −0,089] |

($I_G$ gaussiano não é cota; em k = 6 ele superestima porque $\Sigma_{\rm dentro}$ vem de só 38 pares.)

**Leitura.**
- O ganho de informação com N é **ganho de espalhamento**: a entropia das saídas sobe 1,4 bit nas direções principais.
- Nessas mesmas direções a entropia condicional não muda (+0,07, IC largo, centrado em zero), e o erro em relação à voz real cai.
- **Isso é a assinatura de cobertura com ruído por voz fixo**, o mesmo que a temperatura constante da análise A.
- Ressalva honesta: na variância total (escalar, todas as 256 dimensões) o "ruído" R1 × R4 cresce 34%. Esse ruído não é só do
  decodificador: mede quanto a saída se move quando o código muda. Num decodificador que segue mais o código (N maior), ele
  cresce, e o ruído de síntese puro não pode ser separado com uma frase por condição.
- A conclusão fica para o subespaço dominante: **ruído por voz constante, espalhamento crescente**.

**12 vozes (8 treinadas + 4 novas)**, k = 6, regressão nos 10 treinos:
- $H(\hat Y)$: +0,07 bit por duplicação de N e +0,25 por duplicação de K;
- $H(\hat Y\mid S)$: +0,16 por duplicação de N e −0,04 por duplicação de K.

Nada cresce com N, de novo coerente com a capacidade das treinadas não depender de N. O efeito de K está em $H(\hat Y)$:
memorizar melhor as treinadas espalha as saídas delas.

## 4. Cadeia de informação (desigualdade de processamento de dados)

O estimador é o mesmo em todos os elos: identificação pelo centroide mais próximo + Fano. Nos elos latentes, Euclides branqueado
em vez de cosseno.

**38 externas** (teto 5,25):

| elo | N = 8 | N = 128 | origem |
|---|---|---|---|
| fala real (frase retida × centroide) | 4,21–4,68 | 4,21–4,68 | [MEDIDO] `b13` |
| μ do codificador (uma frase) | 5,14 | 5,23 | [MODELO] `b23`: $\Sigma_b$, $W$ do treino; centroides verdadeiros |
| z = μ + ruído posterior | 4,36 | 4,76 | [MODELO] `b23` |
| saída do decodificador | 0,31 | 0,89–1,13 | [MEDIDO] `b13` |

**8 treinadas** (teto 3,0), tudo medido:

| elo | N = 8 | N = 128 |
|---|---|---|
| fala real | 3,00 | 3,00 |
| μ (frase retida) | 2,94 | 2,81 |
| z | 2,43 | 2,18 |
| saída | 1,49 (wide) | 2,11 (wide); 1,49 (plain); n200: 1,23 |

**Onde os bits se perdem.**
- Nas vozes novas, quase tudo no decodificador: 4,1 bits em N = 8, 3,7–3,9 em N = 128 (de z para a saída).
  - O codificador e o KL perdem ≤ 0,8 bit.
  - N mexe pouco no elo z (+0,4 bit, devido à covariância de cada treino) e recupera +0,6 a +0,8 bit no elo do decodificador.
- Nas vozes treinadas o decodificador perde ~0,5–1,3 bit e **não melhora com N**.

O elo μ ficar acima da fala real nas 38 vozes não viola a desigualdade: os estimadores não são idênticos (latente com
centroides verdadeiros de um modelo gaussiano contra áudio com centroides estimados).

## 5. Extrapolação (38 externas, EC; `b26`, 2.000 réplicas pareadas)

| | ponto | IC 95% |
|---|---|---|
| inclinação $s$ | 0,310 bit por duplicação | [0,166; 0,467] |
| $C(512)$ | 2,61 | [1,86; 3,54] |
| $C(2048)$ | 3,23 | [2,25; 4,45] |
| fala real (mesmo estimador) | 4,96 | |
| $N^\*$ (saída = fala real) | $\approx1{,}0\times10^5$ | $10^{3{,}6}$ a $10^{8{,}2}$ |

**Previsão registrada**: com N = 2.048 (K = 32), nas mesmas 38 vozes, $C\in[2{,}25;4{,}45]$ bits (EC), ponto 3,23.
- **Falseia a lei log-linear**: $C(2048)<2{,}25$, que indica saturação/piso, ou $C(512)>3{,}54$.
- **Falseia a leitura de cobertura**: $H(\hat Y\mid S)$ mudar mais de 1 bit entre N = 128 e 2.048 nas mesmas direções.

Mesmo no limite otimista do IC, igualar a fala real pede milhares de locutores, e no ponto central ~10⁵ (mais que o LibriTTS
inteiro, 2.456).

## 6. C e cópia entre treinos e variantes (negativos)

- Entre os 12 treinos com 12 vozes: **Spearman(q, C) = −0,49**. Mais capacidade, menos vozes não copiadas.
  - Exceção: n128_k8 tem C alto (2,30) e q alto (0,47).
  - Nas 4 novas, n128_k8 tem 1,88 bit e q = 0,45: com K = 8 as saídas são distintas mas puxadas para a voz padrão.
  - Capacidade e cópia não são a mesma coisa (v2, §1.4).
- **Vetor de locutor pré-treinado** (`b27`, força 0 = sem estilo VAE; nos `-spk` sobra só o ECAPA):

  | treino | força 0,0 (só ECAPA) | força 1,0 |
  |---|---|---|
  | n2_k128-spk | 0,00 | 0,42 |
  | n8_k8-spk | 0,00 | 1,33 |
  | n8_k128-spk | 0,00 | 1,51 |
  | n8_k128-spkdrop | 1,33 | 1,71 |
  | n32_k8-spkdrop | 0,01 | 0,86 |
  | n32_k128-spkdrop | 0,59 | 1,91 |

  (Fano, 12 vozes.) **O decodificador `-spk` passa 0 bit pelo ECAPA**, a prova direta de que o vetor é ignorado (v2, 6.4: ótimo
  de reconstrução com entrada redundante). Retirando o estilo em metade do treino, o ECAPA passa 0,6–1,3 bit, mas C total com
  estilo (1,7–1,9) não supera os outros treinos de mesmo N. Isso é coerente com a cópia não melhorar.
- `-aug` com N = 8: C = 1,63 (K = 8) e 2,29 (K = 128), dentro da faixa dos outros treinos N = 8. Os pseudo-locutores não somam
  capacidade.
  - Ressalva: o treino base n8_k8 e n8_k128 não tem áudio salvo, então a comparação é com outras variantes de mesmo N, não com
    o par exato.
- **Leitura unificada** [CONJECTURA apoiada]: as intervenções que não melhoraram a cópia não aumentaram a capacidade de voz do
  decodificador em vozes novas. A capacidade só sobe com o número de vozes reais vistas.

## 7. Força 0,6 e média de referências no conjunto maior (`b24`, 16 vozes onde há externas, senão 12)

| treino | 1,0 | 0,6 | R = 4 | R = 16 |
|---|---|---|---|---|
| n8_k8-aug | 1,63 | **0,25** | | |
| n8_k8-spk | 1,67 | **0,30** | | |
| n8_k128-aug | 2,29 | 1,55 | | |
| n8_k128-spk | 1,98 | 2,06 | | |
| n8_k128-spkdrop | 1,63 | 1,75 | | |
| n32_k8-spkdrop | 1,63 | 0,99 | | |
| n32_k128-spkdrop | 2,58 | 2,41 | | |
| n128_k8 | 2,13 | **0,68** | 2,04 | 2,36 |
| n128_k32 | 2,08 | 1,44 | 2,08 | 2,09 |
| n200_k32 | 1,85 | 1,50 | 2,16 | |
| n8_k32-wide (38 + 12) | 1,36 | | 1,58 | |
| n128_k32-wide | 2,47 | | 2,07 | |

(EC, bits; teto 4 com 16 vozes.)
- **Força 0,6**: cai em 8 de 10 treinos. A queda é grande com K = 8 (−1,4 a −1,5 bit) e pequena com K = 128 (−0,2 a +0,1).
  A escala de estilo derruba a capacidade justamente onde a memorização por voz é fraca.
- **R = 4 / 16**: −0,40 a +0,31, média ≈ +0,02 bit. Confirma: **média de referências não aumenta a capacidade**.

## 8. Situação

| afirmação | status |
|---|---|
| C cresce com N nas vozes novas: 0,31 bit por duplicação [0,17; 0,47] | MEDIDO (2 pontos, 38 vozes) |
| Forma log-linear | não testada (falta N = 32 ou ≥ 512 com áudio) |
| C das treinadas depende de K, não de N | MEDIDO (12 vozes; IC largos) |
| Ganho em $H(\hat Y)$, $H(\hat Y\mid S)$ constante (subespaço principal) | MEDIDO, IC de $H(\hat Y\mid S)$ largo; não vale para a variância total |
| Perda na cadeia está no decodificador | MEDIDO (saída, fala real), MODELO (μ, z para 38 vozes) |
| ECAPA ignorado = 0 bit | MEDIDO |
| Extrapolação $N^\*\sim10^5$ | CONJECTURA (log-linear) com IC de 4 ordens de grandeza |
