# AGENTS.md

Guia do projeto para quem vai trabalhar nele, gente ou agente. Leia isto antes de mexer em qualquer coisa.

## O que é o projeto

Começou como o projeto final de ML2 (IMPA Tech): um **Tacotron 2 com VAE** que lê um texto com o estilo (voz, tom, ritmo) de um áudio de referência. O objetivo de longo prazo é **dublagem de filmes**.

A pesquisa atual pergunta: **quantas vozes de treino são precisas para clonar uma voz nova?** Medimos quantos bits da identidade da voz de referência chegam ao áudio gerado e como isso cresce com o número de locutores de treino (N). Até aqui, o gargalo está no decodificador, e a quantidade de voz copiada cresce de forma regular com N (cerca de 0,3–0,4 bit por duplicação). O estado atual está em [`research/README.md`](research/README.md).

## Mapa do repositório

```
AGENTS.md                 este guia
README.md                 apresentação do projeto (resultados do TTS e resumo do experimento de cobertura)
research/                 TODA a organização da pesquisa (comece por research/README.md)
  README.md               pergunta central, estado atual, como navegar
  protocolo.md            montagem dos treinos, avaliação e métricas (o que cada número significa)
  infra.md                onde os experimentos rodam, filas, onde ficam os dados
  resultados/             um arquivo .md por resultado (R01, R02, ...), com índice em resultados/README.md
    dados/                tabelas brutas (CSV) de onde saem os números
  caminhos/               linhas de investigação; cada uma tem README.md (histórico e estado)
    <caminho>/em_andamento/   o que está sendo explorado AGORA nesse caminho (planos, previsões registradas)
src/
  models/tacotron2_vae/   modelo (Tacotron 2 + VAE de estilo; FiLM opcional no decodificador)
  training/training-tacotron2-vae/   treino (train.py, train_utils.py)
scripts/
  coverage/               experimento de cobertura: job de treino+avaliação, manifestos, métricas, tabelas
    colab_job.sh          roda um treino e sua avaliação (nome do run -> configuração)
    coverage_manifest.py  monta o conjunto de treino (LJSpeech + N locutores x K frases)
    speaker_similarity.py "voz copiada" (Resemblyzer) e embeddings.npz
    voice_bits.py         bits de voz que chegam à saída (cotas de entropia cruzada e de Fano)
    results_table.py      tabela agrupada a partir de research/resultados/dados/execucoes_cluster.csv
  eval/                   síntese e conferência por reconhecimento de fala (Whisper)
  preprocess/, download_datasets/, cloud/   dados e execução na nuvem
reports/                  textos longos anteriores à pasta research/ (nota teórica, proposta, relatório)
  teoria_cobertura.md     nota teórica com as previsões registradas até a seção 8.12
  teoria/                 relatórios de teoria (literatura, derivação, informação, dados)
  proposta/               proposta de pesquisa (PDF)
docs/                     página do projeto e descrições antigas
```

## Regras de trabalho

**Commits e GitHub**
- O autor dos commits é só o dono do projeto (Richard Viana). Nada de `Co-Authored-By` ou outras linhas de atribuição.
- Trabalhe no branch `claude/film-dubbing-project-amuqaf`. Não abra pull request sem pedido.
- O repositório é **público**: não escreva IPs, nomes de usuário das máquinas, e-mails, senhas, tokens ou caminhos pessoais em nenhum arquivo versionado.

**Pesquisa**
- **Previsão antes do resultado.** Antes de lançar um experimento, escreva no arquivo `em_andamento/` do caminho o que cada teoria prevê e o critério de decisão. Previsão escrita depois não vale.
- **Cada resultado vira um arquivo** em `research/resultados/` (próximo número livre: veja o índice), com pergunta, montagem, tabela, conclusão, ressalvas e de onde vêm os dados. Depois atualize o índice, o README do caminho e, se mudou, o estado em `research/README.md`.
- **Números saem dos dados.** Atualize `research/resultados/dados/execucoes_cluster.csv` e gere as tabelas com `python scripts/coverage/results_table.py`. Não copie número à mão sem conferir.
- **Resultado negativo também é resultado**: registre do mesmo jeito.
- Diga quantas seeds há em cada número. Uma seed só é indício, não conclusão.
- Escreva em português nos documentos de pesquisa; código, comentários e mensagens de commit em inglês.

**Cluster** (detalhes em [`research/infra.md`](research/infra.md))
- Os dados de experimento ficam **nas máquinas do laboratório** e são enviados ao Google Drive por elas mesmas. Não copie dados para o PC de controle.
- Script que pode estar em uso por um job (por exemplo `colab_job.sh` nas máquinas) só se troca escrevendo um arquivo `.new` e fazendo `mv`; editar no lugar quebra o bash que está lendo o arquivo.
- Nunca deixe as GPUs paradas: confira as filas (`node_queue.txt` em cada máquina) e reabasteça antes que acabem.
- Não mate processos que você não criou.
