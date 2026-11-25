# Levantamento de literatura: lei de escala N/K para cópia de voz zero-shot

Data do levantamento: 16/11/2026. Método: buscas na web com consultas genéricas, mais leitura do texto
integral de cinco PDFs baixados do arXiv (indicados abaixo).

**Legenda de verificação** (vale para todas as entradas):

- **[LIDO]** — abri o PDF e li a tabela/trecho citado; os números foram copiados do texto.
- **[RESUMO]** — só vi o resumo (abstract) ou o trecho devolvido pela busca; fórmulas e números vêm
  desse resumo e **não** foram conferidos no texto integral.
- **[MEMÓRIA]** — afirmação que conheço de memória e que a busca **não** confirmou; conferir antes de citar.
- **[CÁLCULO MEU]** — conta feita por mim sobre números de terceiros; não é afirmação dos autores.

---

## (a) Trabalhos por tópico

### 1. TTS multi-locutor / clonagem de voz: similaridade vs. número de locutores de treino

Resultado geral da busca: **não encontrei nenhum trabalho que enuncie uma lei de escala explícita
(forma funcional + expoente) da similaridade de locutor zero-shot em função do número de locutores N**,
nem que varie N e K (frases por locutor) de forma fatorial. O que existe são ablações qualitativas,
quase sempre confundidas com mudança de corpus ou com horas totais.

| Trabalho | Link | O que mostra | Status |
|---|---|---|---|
| Transfer learning de verificação de locutor para TTS multi-locutor (Tacotron 2 + d-vector, NeurIPS 2018) | https://arxiv.org/abs/1806.04558 | Trabalho mais próximo do nosso em arquitetura. Afirma: "zero-shot transfer requires training on thousands of speakers". Tabela 5 varia o conjunto de treino do **speaker encoder** (sintetizador fixo em 1,2K locutores): 1,2K locutores → SV-EER 16,60% / 15,32%, similaridade MOS 2,23 / 2,27; 2,4K → 11,95%, 2,43; 8,4K → 10,14%, 2,54; 18K (interno) → 5,08%, 3,03. Sem ajuste de curva, sem expoente; os pontos misturam corpora e dimensões de embedding (64 vs 256). | [LIDO] |
| idem — expoente implícito | — | De 1,2K a 8,4K locutores (7x) o EER cai de 15,3–16,6% para 10,1%: expoente aparente ≈ 0,21–0,25. Incluindo o ponto de 18K (corpus interno diferente) o expoente sobe para ≈ 0,41. **É coincidência sugestiva com o nosso 0,26, mas totalmente confundida** (corpus, arquitetura, e é N do encoder, não do sintetizador). | [CÁLCULO MEU] |
| SC-GlowTTS | https://arxiv.org/abs/2104.05557 | Zero-shot com apenas 11 locutores de treino (VCTK), alegando similaridade estado-da-arte para locutores novos; aponta a lacuna visto/não-visto como problema aberto. Não varia N. | [RESUMO] |
| YourTTS | https://arxiv.org/abs/2112.02418 | Argumenta qualitativamente que limitar número/variedade de locutores prejudica a generalização para vozes não vistas. Sem curva. | [RESUMO] |
| Glow-WaveGAN 2 | https://arxiv.org/abs/2207.01832 | Lacuna visto/não-visto maior em VCTK que em LibriTTS, atribuída ao menor número de locutores; encoder pré-treinado reduz a lacuna. Comparação entre corpora, não varredura. | [RESUMO] |
| Common Voice para TTS multi-locutor (GlowTTS) | https://arxiv.org/abs/2210.06370 | Subconjuntos filtrados por qualidade com números de locutores diferentes (ex.: 633 → 537 → ...). Conclui que "more training speakers are key for modelling speaker variability", mas N está confundido com o limiar de qualidade; a similaridade para não vistos é não monotônica. | [LIDO parcialmente: tabela 1 e discussão] |
| Estratégias de pré-treino para TTS multi-locutor end-to-end | https://arxiv.org/abs/2011.04839 | Incluir mais locutores (LibriTTS) no treino inicial reduz overfitting aos locutores vistos. Qualitativo. | [RESUMO] |
| TTS zero-shot com embeddings neurais de locutor estado-da-arte | https://arxiv.org/abs/1910.10838 | Compara tipos de embedding (x-vector, LDE) para locutores não vistos; não varia N. | [RESUMO] |
| TTS multi-locutor cross-lingual para locutores não vistos | https://arxiv.org/abs/1911.11601 | Tem uma tabela do efeito de aumentar locutores de treino no EER de verificação; **não vi os números**. | [RESUMO] |
| ECAPA vs x-vector em YourTTS (2025) | https://arxiv.org/abs/2506.20190 | Afirma que o encoder e o número de locutores distintos são determinantes da similaridade; sem lei. | [RESUMO] |
| VALL-E | https://arxiv.org/abs/2301.02111 | Escala para 60K horas / ~7K locutores e mostra ganho grande em similaridade zero-shot; reconhece degradação para não vistos em sistemas anteriores. Não isola N. | [RESUMO] |
| VALL-E 2 | https://arxiv.org/abs/2406.05370 | Treino em Libriheavy (50K h, ~7000 locutores — conferido no texto). A busca resume que SIM melhora com mais dados; **não localizei a ablação de tamanho de dados no texto**. | [LIDO parcialmente] |
| MegaTTS 3 | https://arxiv.org/abs/2502.18924 | Curva de dados mais clara que achei: SIM-O 0,52 (2K h) → 0,65 (200K h) → 0,66 (600K h). Varia **horas**, não locutores; satura no topo. | [RESUMO] |
| CLaM-TTS, DiTTo-TTS, Mega-TTS, WavTTS | https://arxiv.org/abs/2404.02781 · https://arxiv.org/abs/2406.11427 · https://arxiv.org/abs/2306.03509 · https://arxiv.org/abs/2606.03455 | Todos reportam que mais dados melhoram similaridade; nenhum separa número de locutores de horas totais nem ajusta forma funcional. | [RESUMO] |
| MimicLM (2026) | https://arxiv.org/abs/2604.11552 | A busca devolveu a frase de que os ganhos de similaridade "seguem padrões típicos de lei de escala com retornos decrescentes" sem saturar. O abstract não diz isso; **não conferi no texto**. É sobre volume de dados, não N. | [RESUMO, não confirmado] |
| Modelagem generativa hierárquica para síntese controlável (GMVAE-Tacotron, ICLR 2019) | https://arxiv.org/abs/1810.07217 | Mesma família do nosso modelo (Tacotron 2 + latentes VAE). Mostra inferência one-shot de atributos de locutor para locutores não vistos. Não estuda dependência em N ou K. | [RESUMO] |
| Speaker Generation (TacoSpawn) | https://arxiv.org/abs/2111.05095 | Ajusta um prior paramétrico sobre embeddings de locutor e amostra locutores novos — evidência de que o "espaço de locutores" é modelável como distribuição de baixa complexidade. | [RESUMO] |

**Leis de escala em fala fora de TTS** (para contraste): leis de potência de WER vs. dados/parâmetros em
rescoring de ASR (https://arxiv.org/abs/2306.15815), em modelos fundacionais multilíngues OWLS
(https://arxiv.org/abs/2502.10373) e em ASR baseado em LLM vs. FLOPs (https://arxiv.org/abs/2508.04096)
— todos [RESUMO]. Ou seja, "lei de escala em fala" existe para ASR/horas, não para N locutores em TTS.

### 2. Verificação de locutor / embeddings: generalização vs. número de locutores

Também aqui **não achei lei de potência ajustada de EER vs. número de locutores**.

| Trabalho | Link | O que mostra | Status |
|---|---|---|---|
| Treino de reconhecimento de locutor com dados limitados (subconjuntos "tiny" do VoxCeleb2) | https://arxiv.org/abs/2203.14688 | Orçamento fixo de ~50K enunciados, três desenhos: poucos locutores (100 loc., ~51 sessões cada), muitos locutores/poucas sessões (5994 loc., 1 sessão), muitos locutores/muitas sessões (5994 loc., ~8 sessões, 1 enunciado por sessão). EER (dev, vox1-o) X-vector / ECAPA / wav2vec2: 12,9 / 12,2 / 7,5% · 21,7 / 16,0 / 15,7% · 9,8 / 6,0 / 3,6%; VoxCeleb2 completo: 6,3 / 2,9 / 2,4%. Conclusão dos autores: muitos locutores **sem** variabilidade de sessão é o pior caso; a hipótese inicial "maximizar locutores é primordial" não se sustenta sozinha. É o análogo mais direto do nosso trade-off N×K, mas com 3 pontos e sem forma funcional. | [LIDO] |
| VoxBlink2 (100K+ locutores) | https://arxiv.org/abs/2407.11510 | Subconjuntos aleatórios de 5K, 10K, 30K e 110K locutores. Em fine-tuning misto, EER 0,527 / 0,505 / 0,505 / 0,674% — "não melhora consistentemente"; no pré-treino, a Figura 3 mostra que mais locutores melhora (não li os valores da figura). Sem ajuste. | [LIDO parcialmente] |
| Estudo empírico do GE2E | https://arxiv.org/abs/2011.04896 | Efeito do número de enunciados de cadastro por locutor: decisivo para valores baixos, satura depois (compatível com um termo tipo 1/K). | [RESUMO] |
| Design de dataset base em few-shot de imagens (ECCV 2020) | https://arxiv.org/abs/2007.08872 | Análogo fora de fala: com orçamento fixo de anotação, o trade-off classes × imagens por classe tem grande efeito; o ótimo usa muitas classes e poucas imagens por classe. Empírico, sem lei. | [RESUMO] |
| Geração de classes novas para metric learning | https://arxiv.org/abs/2201.01008 | Com total de amostras fixo, Recall@1 em classes não vistas cresce com o número de classes de treino. Empírico. | [RESUMO] |

### 3. Teoria de leis de escala neurais e expoentes a partir da dimensão intrínseca

| Trabalho | Link | O que mostra | Status |
|---|---|---|---|
| A Neural Scaling Law from the Dimension of the Data Manifold | https://arxiv.org/abs/2004.10802 | Perda ∝ (nº de parâmetros)^(-α) com **α ≈ 4/d**, d = dimensão intrínseca da variedade de dados; testado em teacher/student, CNNs e GPT (para GPT: 4/α ≈ 53 vs. d ≥ 90, ou seja, só a desigualdade vale). Atenção: a lei é em **parâmetros**, não em número de amostras. | [RESUMO] |
| Explaining Neural Scaling Laws (PNAS 2024) | https://arxiv.org/abs/2102.06701 | Quatro regimes: limitado por variância (expoente universal, vem da existência de um limite bem-comportado) e limitado por resolução (expoente depende do espectro do kernel / da variedade de dados), nos eixos de dados e de largura; dualidade entre os dois expoentes. Em redes reais os expoentes são menos nítidos que em teacher-student. | [RESUMO] |
| Curvas de aprendizado assintóticas de métodos de kernel | https://arxiv.org/abs/1905.10843 | Teacher-student gaussiano: erro ∝ n^(-β), β fixado pela suavidade e pela dimensão (β = (α_T − d)/d quando o student não limita); via kernel-PCA estima β ≈ 0,36 (MNIST) e ≈ 0,07 (CIFAR10), em acordo com o observado. Menciona o baseline β = 1/d para funções Lipschitz. Refinamento exato: https://arxiv.org/abs/2006.09754. | [RESUMO] |
| Curvas de aprendizado dependentes do espectro em regressão de kernel | https://arxiv.org/abs/2002.02561 | Expressões analíticas do erro de generalização em função do autoespectro do kernel e do alinhamento do alvo. | [RESUMO] |
| Learning Curve Theory | https://arxiv.org/abs/2102.04074 | Modelo de brinquedo que produz n^(-β) para qualquer β > 0; o expoente depende da distribuição dos dados (cauda tipo Zipf), não é universal. | [RESUMO] |
| The Quantization Model of Neural Scaling (NeurIPS 2023) | https://arxiv.org/abs/2303.13506 | Lei de potência da perda emerge se "quanta" de habilidade têm frequências em lei de potência e são aprendidos em ordem de frequência. Alternativa à explicação por dimensão. | [RESUMO] |
| Deep Learning Scaling is Predictable, Empirically | https://arxiv.org/abs/1712.00409 | Curvas de aprendizado em lei de potência em tradução, LM, imagens e **fala**; expoentes medidos, não derivados; melhorias de modelo deslocam a curva sem mudar o expoente. | [RESUMO] |
| A Constructive Prediction of the Generalization Error Across Scales (ICLR 2020) | https://arxiv.org/abs/1909.12673 | Forma funcional conjunta em tamanho de modelo e de dados que extrapola de escalas pequenas para grandes. Padrão metodológico para "prever e depois testar". | [RESUMO] |
| DNNs adaptam-se à dimensão intrínseca | https://arxiv.org/abs/1907.02177 | Taxa O(n^(-2β/(2β+d))) com d = dimensão intrínseca (de Minkowski) do suporte, sem exigir estrutura de variedade. | [RESUMO] |
| Regressão não paramétrica em variedades com ReLU profundas | https://arxiv.org/abs/1908.01842 | Erro quadrático n^(-2(s+α)/(2(s+α)+d)) log³n para alvo Hölder em variedade riemanniana d-dimensional. Extensão sem hipótese de variedade: https://arxiv.org/abs/2306.14859. | [RESUMO] |
| k-NN adapta-se à dimensão intrínseca local (NeurIPS 2011) | https://arxiv.org/abs/1110.4300 | Taxa quase-minimax Õ(n^(-2/(2+d))) com d local e desconhecido. | [RESUMO] |
| Taxas de convergência de medidas empíricas em distância de Wasserstein (Bernoulli 2019) | https://arxiv.org/abs/1707.00087 | E[W_p(μ̂_n, μ)] ~ n^(-1/d), com d uma dimensão **intrínseca** da medida; comportamento multi-escala (taxas diferentes em faixas diferentes de n). | [RESUMO] |
| Generalização via dimensão do embedding e convergência distribucional (ICML 2026) | https://arxiv.org/abs/2601.22756 | Cota do gap de generalização = (taxa de Wasserstein fixada pela dimensão intrínseca dos embeddings) × (constante de Lipschitz do mapa embedding→predição). **É o argumento mais próximo do que queremos fazer**, mas em classificação i.i.d., não em generalização a grupos novos. | [RESUMO] |
| Dimensão intrínseca de imagens e impacto no aprendizado (ICLR 2021) | https://arxiv.org/abs/2104.08894 | Mede a dimensão intrínseca de datasets de imagens (baixa) e mostra, controlando-a com GANs, que ela — e não a dimensão ambiente — governa a complexidade amostral. Modelo de experimento para nós. | [RESUMO] |
| Taxa minimax n^(-2p/(2p+d)) (regressão não paramétrica, 1982) | — | Resultado clássico; **a busca não devolveu a fonte**. | [MEMÓRIA] |

### 4. Teoria de generalização para grupos/tarefas novos

| Trabalho | Link | O que mostra | Status |
|---|---|---|---|
| Modelo de aprendizado de viés indutivo (2000) | citado em https://arxiv.org/abs/1505.06279 | Formaliza "ambiente de tarefas": tarefas sorteadas i.i.d. de uma distribuição; análise de erro por números de cobertura em nº de tarefas e amostras por tarefa. É a moldura conceitual exata (locutor = tarefa). | [RESUMO, via citação] |
| The Benefit of Multitask Representation Learning (JMLR 2016) | https://arxiv.org/abs/1505.06279 | Cotas de convergência uniforme para learning-to-learn da ordem **O(1/√T) + O(1/√m)** (T tarefas, m amostras por tarefa); o termo 1/√T não pode ser melhorado nesse arcabouço. Corresponde à nossa forma a + A/√N + C/√K. A forma exata do teorema **não foi lida**; a ordem 1/√T + 1/√m veio da descrição feita no artigo do item seguinte. | [RESUMO] |
| On the Theory of Transfer Learning: The Importance of Task Diversity (NeurIPS 2020) | https://arxiv.org/abs/2006.11650 | Com tarefas "diversas": complexidade amostral para aprender a representação escala como C(H) + t·C(F); tarefa nova custa só C(F); a taxa decai com **todas** as amostras (n·t). Introduz um parâmetro de diversidade ν. | [RESUMO] |
| Few-Shot Learning via Learning the Representation, Provably (ICLR 2021) | https://arxiv.org/abs/2002.09434 | Representação linear de dimensão k em dimensão ambiente d: risco na tarefa-alvo **O(dk/(n₁T) + k/n₂)**; contorna a barreira Ω(1/T) do cenário de tarefas i.i.d. graças a uma hipótese de cobertura/diversidade. O primeiro termo é exatamente o nosso B/(NK). | [RESUMO] |
| Representações lineares compartilhadas, com cotas inferiores | https://arxiv.org/abs/2002.11684 | Taxas em T e amostras por tarefa com cotas inferiores quase ótimas. Detalhes não lidos. | [RESUMO] |
| A Distribution-Dependent Analysis of Meta-Learning (ICML 2021) | https://arxiv.org/abs/2011.00344 | Regressão linear gaussiana com distribuição gaussiana de tarefas (= modelo de efeitos aleatórios): cotas inferiores dependentes do problema para o risco de transferência e estimador que as atinge a menos de constante. **É o teorema mais próximo da forma a + A/N + B/(NK)**; a expressão exata não foi lida. | [RESUMO] |
| Cotas teóricas de erro de estimação em meta-aprendizado | https://arxiv.org/abs/2010.07140 | Modelo bayesiano de dois níveis para regressão linear; cotas minimax inferiores e superior (MAP) para tarefa nova. | [RESUMO] |
| Generalizing from Several Related Classification Tasks to a New Unlabeled Sample (NeurIPS 2011) | https://proceedings.neurips.cc/paper/2011/hash/b571ecea16a9824023ee1af16897a582-Abstract.html | Formaliza generalização de domínio: N domínios de treino sorteados de uma meta-distribuição; análise de erro e consistência universal via kernels sobre distribuições. A taxa em N não apareceu nos trechos. | [RESUMO] |
| Domain Generalization by Marginal Transfer Learning (JMLR 2021) | https://arxiv.org/abs/1711.07910 | Versão estendida; cota em N domínios e n amostras por domínio (Rademacher). Forma exata não vista. | [RESUMO] |
| How Many Domains Suffice for Domain Generalization? (NeurIPS 2025) | https://arxiv.org/abs/2506.16704 | Define "complexidade amostral em domínios" e uma dimensão de estilhaçamento de domínios; cotas superior e inferior casadas a menos de polilog. Responde formalmente a "quantos domínios (locutores) bastam", mas em PAC de classificação. | [RESUMO] |
| Análise teórica de aprendizado contrastivo não supervisionado (ICML 2019) | https://arxiv.org/abs/1902.09229 | Arcabouço de "classes latentes"; garantias para tarefas a jusante formadas pelas **mesmas** classes latentes — não cobre classes realmente novas. | [RESUMO] |
| Amostragem por conglomerados em dois estágios | (textos de amostragem; a busca devolveu só páginas didáticas) | A variância do estimador da média com N grupos e K unidades por grupo decompõe-se em componente entre-grupos (∝ 1/N) e intra-grupos (∝ 1/(NK)); efeito de desenho 1 + (K−1)ρ. É a origem estatística elementar da forma a + A/N + B/(NK). A fórmula explícita **não** apareceu em fonte primária na busca. | [MEMÓRIA + páginas didáticas] |

### 5. Medidas de teoria da informação

| Trabalho | Link | O que mostra | Status |
|---|---|---|---|
| Fixing a Broken ELBO (ICML 2018) | https://arxiv.org/abs/1711.00464 | Plano taxa-distorção para VAEs: cotas variacionais inferior e superior da informação mútua I(x;z); modelos com o mesmo ELBO podem ter taxas muito diferentes; decodificadores potentes ignoram o latente. A nossa medida "bits por enunciado = KL" é a **taxa R** deste arcabouço (cota superior de I). | [RESUMO] |
| Capacidade do embedding variacional em TTS expressivo (Capacitron) | https://arxiv.org/abs/1906.03402 | Em Tacotron com encoder de referência variacional: define capacidade = informação mútua embedding↔dados, limitada superiormente pelo KL; controla-se a capacidade restringindo o KL (em bits/nats); mais capacidade → mais precisão de transferência, menos generalidade. **Antecedente direto da nossa medida de 65–73 bits**; não decompõe a capacidade em bits "de locutor" nem relaciona com N. | [RESUMO] |
| Unicidade da representação i-vector da voz | https://arxiv.org/abs/2008.11985 | Estima a informação biométrica da voz: 43–52 (ou 43–70) bits em condição restrita com amostras de 5 s; cai para **13–20 bits** em condições menos restritas; dobrar a duração acrescenta ~20 bits. Estimativas anteriores citadas vão de 14 a 120+ bits. Os nossos 10–20 bits de identidade caem na faixa "irrestrita". | [RESUMO] |
| Divulgação de privacidade em fala medida em bits (2025) | https://arxiv.org/abs/2508.05250 | Mede em bits quanto cada tipo de característica revela sobre a identidade; divulgações de características independentes são aditivas; embeddings de reconhecimento de locutor são os mais informativos. Não vi os valores numéricos. | [RESUMO] |
| Interpretação das dimensões do espaço de embeddings de locutor (2025) | https://warwick.ac.uk/fac/soc/al/research/baap2026/ladefoged/interpreting-the-dimensions-of-speaker-embedding-space.pdf | Preservação de vizinhança (UMAP) satura após ~6 dimensões; 9 parâmetros acústicos interpretáveis explicam o mesmo que 7 componentes principais (>50% da variância). Indício — não estimativa formal — de dimensão efetiva baixa (~6–10). | [RESUMO] |
| M-Vec: embeddings de locutor "Matryoshka" | https://arxiv.org/abs/2409.15782 | Embeddings de locutor com dimensão flexível; pela proposta, dimensões bem menores que 192–512 retêm boa parte do desempenho (não vi os números). | [RESUMO] |
| Eigenvoices (adaptação rápida de locutor, ~2000) | https://eurecom.fr/publication/200/download/mm-nguypa-990301b.pdf | Locutores vivem num subespaço linear de baixa dimensão; adaptar = estimar poucos coeficientes. Precedente clássico de "espaço de locutores de baixa dimensão". | [RESUMO] |
| How much do language models memorize? (2025) | https://arxiv.org/abs/2505.24832 | Capacidade de memorização de modelos GPT ≈ 3,6 bits por parâmetro (3,5–4); quando a capacidade enche, a memorização não intencional cai e começa a generalização; deriva leis de escala para inferência de pertinência. Modelo de "capacidade prevista e depois testada". | [RESUMO] |

**Não encontrei**: estimativa formal (TwoNN, MLE etc.) da dimensão intrínseca de x-vectors/d-vectors;
nem medida de "bits de identidade de locutor" dentro de um latente de estilo de TTS em função de N.

### 6. "Modelo mínimo" / saturação de capacidade prevista e testada

| Trabalho | Link | O que mostra | Status |
|---|---|---|---|
| Don't Blame the ELBO! (NeurIPS 2019) | https://arxiv.org/abs/1911.02469 | VAE linear recupera pPCA; o número de dimensões latentes ativas é determinado pelo ruído de observação vs. autovalores; a análise linear prevê o comportamento de VAEs não lineares. Base para prever o "tamanho mínimo do latente". | [RESUMO] |
| Dependência do tamanho do dataset da curva taxa-distorção e do limiar de colapso em VAE linear | https://arxiv.org/abs/2309.07663 | Forma fechada relacionando β, tamanho do dataset, colapso posterior e curva taxa-distorção em VAE linear no limite de alta dimensão. **Único trabalho encontrado que liga R(D) do VAE ao tamanho do dataset** — mas i.i.d., sem estrutura de grupos. | [RESUMO] |
| Preprint 2026 sobre limiares de colapso em β-VAE | https://arxiv.org/abs/2605.22691 | No caso gaussiano linear, limiares de colapso, reduções de distorção e autovalores normalizados de PCA coincidem; alerta que "β dá N dimensões ativas" não é intrínseco sem normalização. Visto só por agregador. | [RESUMO, não revisado por pares] |
| Capacitron | https://arxiv.org/abs/1906.03402 | (ver item 5) Varia a capacidade-alvo e observa o trade-off precisão/generalidade — teste empírico de capacidade em TTS, sem previsão teórica do mínimo. | [RESUMO] |
| How much do language models memorize? | https://arxiv.org/abs/2505.24832 | (ver item 5) Previsão de saturação de capacidade confirmada em centenas de transformers. | [RESUMO] |

Não encontrei resultado de *information bottleneck* (transições de fase / representação mínima suficiente)
na busca; se for usado, precisa de busca dedicada.

---

## (b) O que já existe e o que seria novo

### O que já existe (com segurança razoável)

1. **Fato qualitativo**: mais locutores de treino melhora a cópia zero-shot; "milhares de locutores são
   necessários" (2018). Bem estabelecido e repetido em quase todo artigo de TTS zero-shot.
2. **Trade-off N×K com orçamento fixo**, em verificação de locutor (subconjuntos tiny do VoxCeleb2) e em
   few-shot de imagens — empírico, com 3–4 pontos, sem forma funcional.
3. **Formas a + A/N + B/(NK) e a + A/√N + C/√K** são, respectivamente, a forma de efeitos aleatórios /
   amostragem em dois estágios e a forma das cotas de convergência uniforme de learning-to-learn.
   Como *formas*, não são novas.
4. **Expoente a partir da dimensão intrínseca** (≈4/d em parâmetros; n^(-2s/(2s+d)) e n^(-1/d) em amostras)
   é ideia estabelecida para aprendizado i.i.d., e em 2026 já há cota "dimensão do embedding × Wasserstein".
5. **KL do latente de estilo como capacidade em bits** em Tacotron variacional (Capacitron), e **bits de
   identidade na voz** (13–70 bits, i-vectors).

### O que parece novo (até onde a busca alcançou)

- **Uma lei quantitativa conjunta em N e K para cópia de voz zero-shot**, com separação limpa
  "visto depende de K, não visto depende de N", em desenho fatorial. Não achei nada equivalente em TTS
  nem em verificação de locutor.
- **Amarrar o expoente em N à dimensão intrínseca do espaço de locutores** (tratando locutores, e não
  enunciados, como as amostras da variedade). A teoria existente trata amostras i.i.d.; a transposição
  para "amostras = grupos" com K observações ruidosas por grupo não apareceu.
- **Decomposição do KL do latente em bits de identidade vs. resto, crescendo como log N**, e um
  **tamanho mínimo de latente previsto e testado**. Não achei análogo.

### Trabalho anterior mais próximo

- Empírico: https://arxiv.org/abs/1806.04558 (Tabela 5) e https://arxiv.org/abs/2203.14688.
- Teórico: https://arxiv.org/abs/2011.00344 e https://arxiv.org/abs/2002.09434 (forma em N e NK);
  https://arxiv.org/abs/2004.10802, https://arxiv.org/abs/2102.06701 e https://arxiv.org/abs/2601.22756
  (expoente pela dimensão).
- Informacional: https://arxiv.org/abs/1906.03402 e https://arxiv.org/abs/2008.11985.

### Ressalvas honestas sobre a novidade

1. **Ausência de evidência não é evidência de ausência.** Foram ~35 buscas genéricas; Interspeech/ICASSP
   e teses são mal indexados. Um revisor de fala pode conhecer uma ablação de N que não achei. Antes de
   alegar "primeiro", fazer busca dedicada em ISCA Archive e IEEE Xplore.
2. **As duas formas candidatas contradizem o expoente observado.** A forma hierárquica prevê decaimento
   1/N; as cotas de convergência uniforme, 1/√N. O dado empírico é N^(-0,26) sem piso. Um expoente tão
   pequeno é assinatura de regime **não paramétrico / limitado por resolução**, não de efeitos aleatórios
   de dimensão finita. Isso é ao mesmo tempo a parte mais interessante e a mais frágil: com 16 execuções,
   as duas formas ajustam "igualmente bem" justamente porque o piso *a* absorve a diferença.
3. **Três valores de N (8, 32, 128) não distinguem lei de potência de decaimento logarítmico** nem de
   potência com piso. 43→30→21% dá razão 0,70 nos dois passos (limpo), mas são só dois intervalos.
4. **O expoente não identifica a dimensão sem hipótese adicional.** [CÁLCULO MEU] 0,26 corresponde a
   d ≈ 4 se a taxa for N^(-1/d); d ≈ 6 se for N^(-2/(2+d)); d ≈ 8 se N^(-2/d); d ≈ 15 se N^(-4/d).
   A previsão só é falseável se d for medido **independentemente** (ex.: TwoNN/MLE nos embeddings do
   speaker encoder) e a suavidade for fixada a priori.
5. **A métrica é um limiar** (fração de amostras mais parecidas com a referência do que com a voz padrão),
   não uma perda suave; taxas de erro de regressão não se transferem automaticamente a uma fração de
   cauda. Além disso a métrica depende de um speaker encoder específico.
6. **Escala pequena e modelo antigo.** Tacotron 2 + VAE com N ≤ 128 está muito longe do regime dos
   sistemas atuais (milhares de locutores, dezenas de milhares de horas), que saturam (SIM-O 0,65→0,66
   de 200K a 600K horas). "Sem piso visível" até N=128 não diz nada sobre N=10⁴.

**Veredito**: a *lei empírica fatorial N×K* e a *ligação expoente ↔ dimensão do espaço de locutores* são,
ao que tudo indica, novas como enunciado quantitativo; os *ingredientes teóricos* são todos conhecidos.
A contribuição defensável é a transposição e o teste, não um teorema novo — a menos que se prove uma
taxa para o cenário hierárquico não paramétrico (ver (c), lacuna).

---

## (c) Teoremas reutilizáveis e hipóteses necessárias

| Resultado | Termo que fornece | Hipóteses que teríamos de assumir | Risco |
|---|---|---|---|
| Decomposição de variância em dois estágios / efeitos aleatórios | A/N + B/(NK) | Locutores i.i.d. de uma população; enunciados i.i.d. dado o locutor; estimador linear/de dimensão finita | Prevê expoente 1 em N — contradiz 0,26. Serve para o regime "visto" (C/K) e como termo de variância. |
| Meta-aprendizado com distribuição gaussiana de tarefas (https://arxiv.org/abs/2011.00344) | Cotas inferior e superior do risco em tarefa nova | Regressão linear, design fixo, tarefas gaussianas | Linear; dá otimalidade da forma hierárquica, não o expoente pequeno. |
| Representação compartilhada (https://arxiv.org/abs/2002.09434, https://arxiv.org/abs/2006.11650) | dk/(NK) + k/n₂ | Representação de dimensão k comum a todas as tarefas; **diversidade de tarefas** (os N locutores cobrem as k direções) | No zero-shot n₂ é um único enunciado de referência; o termo k/n₂ vira o piso *a*. A hipótese de diversidade é exatamente o que falha com N pequeno. |
| Learning-to-learn por convergência uniforme (https://arxiv.org/abs/1505.06279) | 1/√N + 1/√K | Tarefas i.i.d.; classe de representação de complexidade finita | É cota superior de pior caso, não curva de aprendizado; não usar como "lei". |
| Taxa de Wasserstein com dimensão intrínseca (https://arxiv.org/abs/1707.00087) | N^(-1/d) | Distribuição de locutores com dimensão intrínseca d > 2p; suporte compacto | Dá d ≈ 4 para 0,26 — plausível frente a "~6 dimensões" do item 5, mas precisa de um mapa Lipschitz do embedding até a métrica. |
| Cota embedding × Lipschitz (https://arxiv.org/abs/2601.22756) | (taxa de Wasserstein) × L | Mapa embedding→saída Lipschitz; amostras i.i.d. | Molde pronto do argumento; teríamos de trocar "amostra" por "locutor" e lidar com o ruído de K enunciados. |
| Regressão não paramétrica adaptativa (https://arxiv.org/abs/1907.02177, https://arxiv.org/abs/1908.01842, https://arxiv.org/abs/1110.4300) | N^(-2s/(2s+d)) | Mapa locutor→voz Hölder-s sobre suporte de dimensão d; rede atinge a taxa minimax | Dois parâmetros livres (s, d) para um expoente: não identificável sem medir d. |
| Regimes variância/resolução (https://arxiv.org/abs/2102.06701) | Interpretação: N no regime de resolução, K no de variância | Limite bem-comportado em K; espectro em lei de potência em N | Encaixa qualitativamente no nosso achado (K → C/K; N → potência pequena). É a moldura narrativa mais natural. |
| Taxa-distorção de VAE (https://arxiv.org/abs/1711.00464, https://arxiv.org/abs/1906.03402) | KL ≥ I(x;z); capacidade em bits | Nenhuma forte: KL é cota **superior** da informação mútua | Os "10–20 bits de identidade" exigem um estimador próprio (cota inferior por classificador); declarar qual cota é usada. |
| Contagem informacional | Para distinguir N locutores são necessários ≥ log₂ N bits | Trivial (Fano) | Explica "cresce como log N" **dentro do treino**; não explica sozinho a generalização para não vistos. |
| VAE linear / pPCA (https://arxiv.org/abs/1911.02469, https://arxiv.org/abs/2309.07663) | Nº de dimensões ativas = nº de autovalores acima do ruído | Linear-gaussiano | Dá previsão testável do latente mínimo se medirmos o espectro da variância entre-locutores. |

**Lacuna teórica (onde poderia haver teorema novo)**: nenhum resultado encontrado cobre o caso
*hierárquico e não paramétrico* — N grupos amostrados de uma distribuição de dimensão intrínseca d, cada
um observado por K amostras ruidosas, risco medido num grupo novo. A conjectura natural, combinando os
ingredientes acima, é um erro da forma

    piso + (termo de resolução em N, expoente fixado por d e suavidade) + (termo de variância ∝ 1/(NK) ou 1/K)

Isso é **conjectura minha**, não resultado da literatura.

---

## (d) O que um artigo de venue de topo precisaria conter

1. **Mais décadas em N.** Pelo menos 5–6 valores de N cobrindo ≥ 2 décadas (ex.: 8 → 1024+; LibriTTS tem
   ~2,4K locutores), para distinguir potência, potência com piso e logaritmo. Mesma coisa em K.
2. **Sementes e incerteza.** Várias sementes por célula (N, K), intervalos de confiança, e comparação
   formal de modelos (AIC/BIC ou validação cruzada deixando células de fora). Com 16 execuções o empate
   entre as duas formas é esperado.
3. **Previsão fora da amostra.** Ajustar em N pequeno e **prever** N grande antes de rodar (padrão de
   "A Constructive Prediction..."). É o que transforma ajuste de curva em lei.
4. **Medida independente da dimensão do espaço de locutores**, com ≥ 2 estimadores (TwoNN, MLE,
   espectro de PCA) e ≥ 2 speaker encoders, e o expoente previsto **antes** de olhar a curva.
5. **Intervenção na dimensão.** Como no estudo de imagens com GANs: restringir a população de locutores
   (um só gênero, faixa de pitch estreita, ou locutores sintéticos amostrados de um prior de dimensão
   controlada) e mostrar que o expoente muda como previsto. Sem isso, "expoente ↔ dimensão" é correlação
   de um ponto só.
6. **Controles de confusão.** Separar N de horas totais (N·K fixo vs. K fixo); balancear gênero/sotaque;
   locutores de teste fixos e suficientes (38 é pouco para caudas); canal/gravação (identidade de locutor
   é correlacionada com canal).
7. **Robustez da métrica.** Repetir com outro speaker encoder, com similaridade contínua (cosseno, EER)
   além da fração-limiar, e com um teste subjetivo pequeno. Mostrar que o expoente não é artefato do limiar.
8. **Generalidade de arquitetura.** Ao menos um segundo modelo (ex.: um TTS zero-shot moderno em escala
   reduzida, ou um modelo de conversão de voz) — senão o revisor dirá que é propriedade de Tacotron 2 + VAE.
   Idealmente um segundo domínio (ex.: verificação de locutor, ou faces) para sustentar "lei".
9. **Parte informacional rigorosa.** Declarar cotas (KL é superior; bits de identidade por classificador
   é inferior), mostrar a curva taxa-distorção variando β/dimensão do latente, e testar o "latente
   mínimo": prever a dimensão/capacidade abaixo da qual a cópia degrada e verificar o joelho.
10. **Teoria com hipóteses explícitas e um caso solúvel.** Um modelo mínimo (ex.: linear-gaussiano
    hierárquico, ou teacher-student com locutores num toro/esfera d-dimensional) em que a lei é derivada
    exatamente e simulada, mais a ponte para a rede real. Enunciar a convergência (em que sentido, com
    que taxa) e o que falsearia a teoria.
11. **Posicionamento honesto.** Reconhecer que as formas 1/N, 1/(NK), 1/√N e os expoentes por dimensão
    são conhecidos; a novidade é o cenário hierárquico, a medida em fala e o teste. Discutir a saturação
    observada em sistemas de grande escala e por que (ou se) a lei deixa de valer lá.

---

## Limitações deste levantamento

- Só cinco artigos tiveram trechos do texto integral lidos (1806.04558, 2203.14688, 2407.11510,
  2406.05370, 2210.06370); todo o resto é [RESUMO].
- Fórmulas de teoremas (itens 3 e 4) vêm de resumos gerados pela busca e **devem ser conferidas** nos
  originais antes de entrar em qualquer texto.
- Clássicos não devolvidos pela busca (taxa minimax de 1982; "Scaling Laws for Neural Language Models";
  fórmulas de amostragem em dois estágios) estão marcados [MEMÓRIA] ou omitidos.
- Não foi feita busca em ISCA Archive / IEEE Xplore nem em teses.
