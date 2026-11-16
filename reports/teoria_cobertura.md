# Por que copiar uma voz nova exige tantos locutores: tratativa teórica e uma solução

Este texto organiza os resultados do experimento de cobertura (README, seção 9) num modelo simples, verifica o modelo contra os 12 treinos da grade, e tira dele uma solução e a testa (o teste deu negativo, seção 6). O argumento é heurístico: as fórmulas são uma lei de amostragem ajustada aos dados, não um teorema.

## 1. O que precisa ser explicado

O modelo lê um texto com a voz de um áudio de referência. Treina-se com 2.000 frases de uma voz (a "voz padrão", LJSpeech) mais N locutores com K frases cada. Mede-se, em frases novas, a parcela de amostras em que a voz da referência foi copiada.

1. A cópia de vozes **treinadas** depende de K (frases por locutor) e quase não depende de N.
2. A cópia de vozes **novas** depende de N (número de locutores) e pouco de K.
3. Quanto mais longe o código de estilo está da voz padrão, mais a voz é copiada e mais palavras saem erradas.
4. Quatro mudanças no modelo não reduziram a necessidade de dados: tirar o estilo de outra frase do mesmo locutor, uma perda contrastiva no latente, um vetor de locutor pré-treinado, e esse vetor com dropout de estilo.

## 2. Modelo: o treino cobre uma cruz, o teste pede o retângulo

O decodificador é uma função `f(c, s)`: `c` é o texto codificado e `s` o vetor de estilo (que carrega a voz). Gerar bem exige que `f` esteja certa no par `(c, s)` pedido. O conjunto de pares possíveis é o produto `C × S` (todos os textos × todas as vozes), e o treino só visita uma parte dele:

- uma linha longa: muitos textos (2.000) na voz padrão `s₀`;
- N linhas curtas: K textos em cada voz `sᵢ`.

O teste pede um texto novo `c*` numa voz `s`. O erro nesse ponto se decompõe em dois passos:

```
erro(c*, s) = erro(c*, s₀) + [erro(c*, s) − erro(c*, s₀)]
              \_________/    \________________________/
              ler texto novo      interação texto × voz:
              na voz padrão       o que muda ao trocar a voz
```

O primeiro termo é pequeno (WER de 4–6% com o estilo desligado), porque a linha longa cobre bem os textos. Todo o problema está no segundo termo, e sobre ele o treino só tem as N·K amostras das linhas curtas. Isso já explica o fato 3: longe de `s₀`, a leitura do texto só foi treinada em K frases por voz; o decodificador ou volta para a voz padrão (voz não copiada, texto certo) ou segue a voz e erra o texto. A força do estilo só escolhe o ponto entre esses dois erros.

## 3. Lei de amostragem em dois níveis

Os dados de treino fora da voz padrão vêm de uma amostragem em dois níveis: sorteiam-se N locutores e, dentro de cada um, K frases. Em qualquer estimador desse tipo (modelos mistos, amostragem por conglomerados) o erro tem três componentes, cada um caindo com a contagem que o alimenta:

- o que é comum às vozes (como uma voz qualquer altera a fala) é aprendido dos N locutores: erro ∝ 1/N;
- o que é próprio de um locutor treinado é aprendido das K frases dele: erro ∝ 1/K;
- a interação texto × voz é aprendida de todas as amostras: erro ∝ 1/(N·K).

Uma voz nova não tem o segundo componente (nenhuma frase dela foi vista), então a previsão para a parcela de amostras com a voz **não** copiada é:

```
vozes novas:      p = a + A/N + B/(N·K)
vozes treinadas:  p = a' + C/K + A'/N
```

`a` e `a'` são o que sobra com dados infinitos desse tipo.

**Ajuste aos 12 treinos** (passo 2.500, estilo inteiro; `scripts/coverage/fit_law.py`):

| Vozes | Lei | Parâmetros | Erro do ajuste (RMSE) |
|---|---|---|---|
| novas | a + A/N + B/(NK) | a = 0,42; A = 1,03; B = 8,9 | 0,050 |
| novas | a + A/N | a = 0,45; A = 1,05 | 0,068 |
| novas | a + B/(NK) (só o total de frases) | | 0,175 |
| novas | a + C/K | | 0,184 |
| treinadas | a' + C/K + A'/N | a' = 0,08; C = 3,2; A' = 0,36 | 0,034 |
| treinadas | a' + C/K | a' = 0,15; C = 2,8 | 0,065 |
| treinadas | a' + A'/N | | 0,162 |

O desvio entre duas sementes do mesmo treino é 0,048. As duas leis de dois níveis ajustam os dados até o nível do ruído; as alternativas de um fator só (só N, só K, só o total N·K) ficam de 2 a 4 vezes acima. Ressalva: são 3 parâmetros para 7 configurações, com N ≤ 32; o ajuste não distingue 1/N de uma queda mais lenta.

## 4. O que a teoria explica e o que ela prevê

**Os fatos 1 e 2** são a própria lei: K entra nas vozes treinadas pelo termo C/K, que não existe para vozes novas; N domina as vozes novas pelo termo A/N.

**Os quatro resultados negativos (fato 4).** As quatro mudanças atuam sobre como a voz é *representada* (o que o latente carrega, de onde vem o vetor). Nenhuma acrescenta pares (texto, voz) ao treino do decodificador, que é o que A/N e B/(NK) contam. O vetor de locutor pré-treinado dá boas coordenadas para o espaço de vozes, mas o decodificador continua vendo só N pontos nele: com 8 locutores ele decora (71% das treinadas, 5% das novas), com 32 começa a interpolar (47% das novas), como a lei prevê para qualquer caminho de entrada.

**Mais locutores reais rendem cada vez menos.** Pela lei ajustada, com K = 128:

| N | Vozes novas copiadas (previsão) |
|---|---|
| 8 | 44% (medido: 51% / 37%) |
| 32 | 55% (medido: 61% / 57%) |
| 128 | 57% |
| infinito | 58% |

Passar de 32 para 128 locutores (4 vezes mais dados) renderia cerca de 2 pontos. Se a queda real for mais lenta que 1/N, o ganho é maior, mas continua caro. Esta é a previsão mais arriscada da teoria, porque extrapola além de N = 32.

**O piso `a` = 0,42 é o problema que sobra.** Mesmo com infinitos locutores, 4 em cada 10 amostras de vozes novas não copiariam a voz, contra 1 em 10 nas treinadas. A diferença é o componente próprio do locutor, que para uma voz nova tem de vir inteiro de uma única referência curta passando por um vetor global de 32 dimensões.

## 5. Soluções que saem da teoria

A lei diz o que conta: pares (texto, voz) vistos pelo decodificador, com N (vozes distintas) pesando mais que K. Três caminhos:

**(a) Preencher o retângulo com pseudo-locutores (testado; não funcionou, seção 6).** Existe uma transformação barata que troca a voz e mantém o texto: tocar a gravação mais rápido ou mais devagar por um fator r sobe ou desce o tom e os formantes por r. Aplicada à linha longa (as 2.000 frases da voz padrão) e às linhas curtas, ela cria vozes novas com as mesmas frases, sem gravar nada. Em reconhecimento de locutor é prática comum tratar cópias com velocidade alterada como locutores novos. Na lei, isso aumenta N (e K) de graça. O limite previsível: as pseudo-vozes variam em uma direção só do espaço de vozes (tamanho do trato vocal/tom); elas contam como locutores apenas na medida em que as vozes reais diferem nessa direção.

Implementação: `scripts/coverage/augment_voices.py` acrescenta, para cada frase de treino, uma cópia com velocidade alterada entre 1,5 e 4 semitons para cima ou para baixo; variante `-aug` em `scripts/coverage/colab_job.sh`. O número de passos de treino é o mesmo.

Previsões registradas antes do resultado (vozes novas copiadas, passo 2.500, estilo inteiro):

| Treino | Normal (medido, 2 sementes) | Se as pseudo-vozes não contarem | Se contarem como locutores (N efetivo ≥ 32) |
|---|---|---|---|
| 8 × 8 | 26% / 39% | ~31% | ~50% |
| 8 × 128 | 51% / 37% | ~44% | ~55% |

Como uma semente varia ±5 pontos (até 14 entre sementes), só um ganho acima de ~10 pontos sobre a média normal conta como efeito.

**(b) Tirar o texto da conta: fala na voz padrão + conversão de voz.** Se `f(c, s) = V(T(c), s)`, com `T` lendo o texto na voz padrão e `V` trocando a voz sem olhar o texto, o termo de interação some por construção. `T` já funciona (primeiro termo da decomposição). `V` não precisa de transcrição, então pode ser treinado em milhares de locutores de fala não transcrita: N deixa de ser limitado pelos dados com texto. É o caminho estrutural; custa treinar um segundo modelo.

**(c) Baixar o piso `a`.** O piso vem de resumir uma voz nova em um vetor global tirado de uma referência. Usar várias referências do mesmo ator (na dublagem há o filme inteiro) e um código que varie ao longo da frase atacam esse termo, e não os termos de cobertura.

Para a dublagem, (b) e (c) combinam com o cenário: há muito áudio do ator original sem transcrição alinhada, e nenhum texto de treino na voz dele.

## 6. Resultado do teste de (a): as pseudo-vozes não contam como locutores

Dois treinos com pseudo-locutores (`n8_k8_s1-aug`, `n8_k128_s1-aug`; cada frase de treino ganha uma cópia com velocidade alterada, 41 e 54 pseudo-locutores no total), mesma quantidade de passos e mesma avaliação. Passo 2.500, estilo inteiro:

| Treino | Vozes treinadas copiadas | Vozes novas copiadas | WER vozes novas | Previsão "não contam" | Previsão "contam" |
|---|---|---|---|---|---|
| 8 × 8 normal (2 sementes) | 45% / 44% | 26% / 39% | 14% / 24% | | |
| 8 × 8 com pseudo-locutores | 50% | **28%** | 14% | ~31% | ~50% |
| 8 × 128 normal (2 sementes) | 91% / 88% | 51% / 37% | 8% / 9% | | |
| 8 × 128 com pseudo-locutores | 84% | **42%** | 8% | ~44% | ~55% |

Os dois resultados caem na previsão "as pseudo-vozes não contam" (28% contra ~31%; 42% contra ~44%) e dentro do intervalo das duas sementes normais. No passo 1.000 o quadro é o mesmo (22% contra 24%; 36% contra 30%). A solução (a) não funciona. É uma semente por configuração, mas a diferença para a previsão "contam" (22 e 13 pontos) é maior que a variação entre sementes.

O que isso diz sobre a teoria:

- A lei de dois níveis previu os dois números com N = 8, sem reajuste. O N que entra nela é o número de vozes **reais**; quinze a vinte vezes mais rótulos de locutor não mudaram nada.
- O limite antecipado na seção 5 é o que aconteceu: mudar a velocidade desloca a voz em uma direção só (tom e formantes juntos). As vozes novas do teste diferem das treinadas em muitas direções (timbre, fonte glotal, articulação, ritmo), e o decodificador precisa ter visto variação nessas direções.
- O termo A/N conta, portanto, a variedade real de vozes, e não pares (texto, voz) em geral. Preencher o retângulo com transformações simples do sinal não substitui locutores.

Consequência prática: sobram os caminhos (b) e (c). O (b) é o que ataca o termo A/N diretamente, porque a conversão de voz treina sem transcrição e pode usar milhares de locutores reais; o (c) ataca o piso `a`. Uma versão mais forte de (a) seria gerar as pseudo-vozes com um conversor de voz já treinado em muitos locutores (as 2.000 frases da voz padrão ditas em centenas de vozes reais), o que é o caminho (b) usado como aumento de dados.

## 7. Próximo teste: conversão de voz (caminho b), com previsões registradas

Dois experimentos com um conversor de voz pré-treinado (kNN-VC: troca as características WavLM do áudio pelas mais próximas entre as do locutor-alvo; não usa texto e não é treinado aqui).

**7.1 Cascata: ler na voz padrão, depois converter** (`scripts/coverage/vc_cascade.py`). O modelo de inglês, sem nenhum treino com outros locutores, lê as 25 frases na voz padrão; o conversor leva cada áudio para a voz da referência, descrita por R = 1, 4, 16 ou 64 frases do mesmo locutor. As referências agora incluem 4 locutores "externos" (LibriTTS test-clean), que nem os nossos modelos nem o conversor ouviram.

Previsões:

- O WER fica perto do da voz padrão (até ~8%) em todos os grupos, porque o termo de interação texto × voz some.
- A cópia da voz não depende de a voz ser treinada, nova ou externa.
- A cópia sobe com R (caminho c: mais referência baixa o piso). Com R ≥ 16, passa de 58% nas vozes novas, o teto que a lei dá ao modelo atual com infinitos locutores. Se com R = 64 ficar abaixo de 58%, o caminho (b) com este conversor falha.

**7.2 Vozes reais por conversão no treino** (`scripts/coverage/vc_voices.py`, variante `-vcaug`). As 2.000 frases da voz padrão ganham uma cópia convertida para 200 locutores reais do LibriTTS (10 frases por voz; locutores fora de todos os treinos e de todas as referências; só o áudio deles é usado). É o teste do que a seção 6 concluiu: se o termo A/N conta variedade real de vozes, estas cópias devem contar, ao contrário das de velocidade alterada.

| Treino | Normal (2 sementes) | Se as vozes convertidas não contarem | Se contarem (N ≈ 208) |
|---|---|---|---|
| 8 × 8 | 26% / 39% | ~31% | ~55% |
| 8 × 128 | 51% / 37% | ~44% | ~57% |

O critério é o mesmo: só mais de ~10 pontos acima da média normal conta como efeito.

**Situação de 7.1 e 7.2:** os scripts estão no repositório, mas não foram executados. Rodar o conversor de voz de terceiros foi barrado pelo controle de permissões do ambiente em que os experimentos são lançados; depende de autorização do autor.

## 8. Dois testes só com o nosso modelo, com previsões registradas

**8.1 A previsão arriscada: 128 locutores.** Treinos `n128_k8_s1` e `n128_k32_s1` (os 32 locutores da grade mais 96 do mesmo corpus; mesmas vozes novas, mesmos passos). A lei ajustada na seção 3 prevê, para as vozes novas copiadas, 56% (K = 8) e 57% (K = 32): quase nada acima dos 32 locutores (medido: 51% / 47% com K = 8; 61% / 57% com K = 128). Para as treinadas, 52% (K = 8) e 82% (K = 32). Se a queda real for mais lenta que 1/N, as vozes novas passam de 67% (previsão + 10 pontos) e a lei está errada nesse ponto, o que seria uma boa notícia: mais locutores ainda renderiam.

**8.2 Várias referências por voz (caminho c).** Na avaliação, o código de estilo passa a ser a média dos códigos de R = 4 e 16 frases do mesmo locutor (`--n-refs` em `scripts/eval/synth_check.py`), sem treinar nada. A medida de voz ganha uma segunda coluna, a semelhança com a média de 20 frases do locutor (`sim_spk`), comparável à da voz padrão. Duas hipóteses para o piso `a`:

- o piso é ruído de estimar a voz a partir de uma frase só: a cópia das vozes novas sobe 10 pontos ou mais de R = 1 para R = 16;
- o piso é do decodificador, que não sabe produzir a voz mesmo com o código certo: nada muda (ou piora, se a média puxar o código para perto da voz padrão).

As referências incluem agora 4 locutores externos (LibriTTS test-clean), além dos 8 treinados e 4 novos de sempre.

**8.3 Resultado** (passo 2.500, estilo inteiro; uma semente por treino; `sim_ref > sim_default` como nas seções anteriores):

| Treino | Treinadas copiadas (previsto → medido) | Novas copiadas (previsto → medido) | Externas copiadas | WER treinadas / novas / externas |
|---|---|---|---|---|
| 128 × 8 | 52% → **52%** | 56% → **55%** | 86% | 12% / 8% / 13% |
| 128 × 32 | 82% → **76%** | 57% → **74%** | 86% | 12% / 8% / 16% |

- Com K = 8 a lei acertou os dois números a menos de 1 ponto.
- Com K = 32 as vozes novas ficaram 17 pontos acima do previsto, além do limite de 67% que eu tinha fixado. Nesse ponto a lei errou: com 128 locutores, mais frases por locutor ainda rendem, e o teto de 58% não vale. É o melhor resultado do projeto para vozes novas (74% copiadas com 8% de WER; o melhor anterior era 61% / 57% com 32 × 128).
- As vozes externas (outro conjunto de gravações) são copiadas mais que as novas do mesmo corpus. Elas estão mais longe da voz padrão (`sim_default` menor), o que facilita o critério; não é sinal de que sejam mais fáceis de imitar.

Refazendo o ajuste com os 14 treinos, a lei a + A/N + B/(NK) dá a = 0,39 e RMSE 0,063 (antes 0,050), e continua prevendo 60% para 128 × 32. Leis com raiz quadrada, como a + A/√N + C/√K, ajustam igualmente bem os 12 treinos antigos (RMSE 0,052) e, ajustadas só neles, previam 58% e 65% para os dois treinos novos; nelas o piso `a` cai para 0,16. Conclusão honesta: **os dados não identificam o piso nem a forma da queda**. O que se mantém é a estrutura em dois níveis (N para vozes novas, K para treinadas, e K ajudando também as novas quando N é grande). A frase "mais locutores rendem cada vez menos" da seção 4 fica em suspenso até a réplica (seção 8.5).

**8.4 Resultado das várias referências.** Vozes copiadas pela medida por locutor (`sim_spk > sim_default`), que é a adequada quando o código vem de várias frases:

| Treino | Grupo | R = 1 | R = 4 | R = 16 |
|---|---|---|---|---|
| 128 × 8 | treinadas | 66% | 66% | 66% |
| 128 × 8 | novas | 47% | 67% | 67% |
| 128 × 8 | externas | 77% | 83% | 90% |
| 128 × 32 | treinadas | 84% | 71% | 75% |
| 128 × 32 | novas | 74% | 78% | 72% |
| 128 × 32 | externas | 95% | 94% | 99% |

Nenhuma das duas hipóteses se confirma por inteiro. Com K = 8 as vozes novas sobem 20 pontos já com 4 referências; com K = 32, onde a cópia já era alta, nada muda nas novas e as treinadas caem. A semelhança média com o locutor sobe em todos os grupos (por exemplo 0,713 → 0,743 nas novas com K = 32), mas a semelhança com a voz padrão sobe junto, porque a média dos códigos fica mais perto do centro. O WER das vozes externas cai (16% → 9% com K = 32; 13% → 10% com K = 8) e o das outras fica igual. Pela medida por frase (`sim_ref`), a cópia cai com R, como esperado: o código deixa de ser o daquela frase. Leitura: a média de códigos ajuda quando o modelo copia mal e deixa a fala mais estável, mas não é um ganho geral; um ganho consistente pediria treinar com o código médio do locutor, e não só usá-lo no teste.

**8.5 Réplica em curso, com previsões.** `n128_k32_s2` (mesma configuração, outra semente) e `n200_k32_s1` (200 locutores). Se os 74% forem reais, a réplica fica em 64% ou mais; se forem sorte da semente, perto de 60%. Para 200 × 32, a lei em 1/N reajustada prevê 60% e a lei com raízes prevê 68%.

## 9. Limites desta tratativa

- A lei foi ajustada depois de ver os dados; o teste honesto dela são as previsões das seções 4 e 5.
- 7 configurações, N ≤ 32, 4 vozes novas por avaliação, uma medida de voz (um codificador de locutor).
- A medida é binária por amostra (voz mais perto da referência que da voz padrão); não mede naturalidade.
- A decomposição da seção 2 é sobre o erro esperado; não foi medida termo a termo.
