# Pesquisa: quantas vozes são precisas para clonar uma?

## Pergunta central

Um modelo de síntese lê um texto com a voz de um áudio de referência que ele nunca ouviu (clonagem *zero-shot*). Queremos saber:

1. **Quanto da voz da referência chega ao áudio gerado**, medido em bits (identificação entre 38 vozes; teto de 5,25 bits).
2. **Como isso cresce com o número de locutores de treino (N)**: existe uma lei de escala? Ela satura?
3. **Onde a informação se perde**: no codificador, no latente ou no decodificador.

A resposta orienta a próxima arquitetura (e a dublagem): se o limite é o decodificador e a lei é regular, dá para prever quantas vozes e quanta computação são precisas.

## Estado atual (25/11/2026)

- **O gargalo é o decodificador.** A referência e o latente carregam 4–5 bits da voz; a saída carregava 0,3–1 bit na receita original ([R10](resultados/R10_bits_gargalo_do_decodificador.md)).
- **Os bits crescem de forma regular com N.** Na receita original, +0,27 bit por duplicação de 8 a 128 locutores, achatando em 241 ([R11](resultados/R11_curva_de_N_receita_original.md)).
- **A rampa do KL limitava tudo.** Treinar mais tempo piorava; sem a rampa, treinar mais tempo melhora muito ([R15](resultados/R15_rampa_do_kl.md)). Com 10.000 passos sem KL: 1,07 bit com 8 locutores, 2,84 com 128 e 2,90 com 241, cerca de +0,4 bit por duplicação ([R16](resultados/R16_curva_de_N_sem_kl.md)).
- **Condicionar o decodificador inteiro ajuda, e mais quanto maior N.** FiLM sem KL chega a 3,08 bits com 241 locutores, o melhor resultado até agora ([R12](resultados/R12_film.md), [R17](resultados/R17_film_sem_kl.md)).
- **Locutores valem mais que frases** com a mesma quantidade de dados ([R19](resultados/R19_locutores_vs_frases.md)).
- **Em aberto:** por que o ganho de 128 para 241 locutores é pequeno (falta de frases por locutor? de passos? limite real?) e se a curva continua acima de 241 locutores (exige preparar o LibriTTS train-clean-360).

## Como navegar

| Onde | O que tem |
|---|---|
| [`protocolo.md`](protocolo.md) | Como cada treino é montado e avaliado; definição de cada métrica; convenção de nomes dos runs |
| [`infra.md`](infra.md) | Cluster, filas de jobs, onde ficam os dados e como recuperá-los |
| [`resultados/`](resultados/README.md) | Um arquivo por resultado, com índice; dados brutos em `resultados/dados/` |
| [`caminhos/`](caminhos/README.md) | As linhas de investigação; cada uma com histórico (`README.md`) e o que está em curso (`em_andamento/`) |
| [`../reports/teoria_cobertura.md`](../reports/teoria_cobertura.md) | Nota teórica com as previsões registradas na fase do Colab (antes desta pasta) |

## Convenções

- **Resultados** se chamam `RNN_nome_curto.md`, numerados na ordem em que foram obtidos. Um resultado não é reescrito depois: se uma conclusão muda, cria-se um resultado novo que aponta para o antigo, e o antigo ganha uma nota no topo.
- **Caminhos** se chamam `NN_nome/`. O `README.md` de cada um tem a pergunta, a hipótese, o histórico (com links para os resultados) e o estado: *ativo*, *pausado*, *encerrado* ou *bloqueado*.
- **`em_andamento/`** guarda um arquivo por exploração em curso: o que está rodando, as previsões registradas antes e o critério de decisão. Quando a exploração termina, o arquivo vai para `concluido/` dentro do mesmo caminho e o resultado vira um arquivo em `resultados/`.
- Toda tabela de resultado diz quantas seeds há por número.
