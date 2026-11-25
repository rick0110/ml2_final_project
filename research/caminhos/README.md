# Caminhos de investigação

Cada caminho é uma linha de pergunta. A pasta de cada um tem:

- `README.md`: pergunta, hipótese, histórico (com links para os resultados) e estado;
- `em_andamento/`: um arquivo por exploração em curso, com o que está rodando, as previsões registradas **antes** dos resultados e o critério de decisão;
- `concluido/` (quando houver): explorações terminadas, que já viraram resultados.

| Caminho | Pergunta | Estado | Em andamento |
|---|---|---|---|
| [01 — Lei de escala em N](01_lei_de_escala/README.md) | Como os bits de voz crescem com o número de locutores? Saturam? | **ativo** | curva longa sem KL; mais de 241 locutores (planejado) |
| [02 — Gargalo do decodificador](02_gargalo_do_decodificador/README.md) | O limite está no decodificador? Condicioná-lo melhor ajuda? | **ativo** | FiLM sem KL com mais seeds e 20.000 passos |
| [03 — Receita de treino](03_receita_de_treino/README.md) | KL, duração do treino e otimização | **ativo** | duração do treino sem KL |
| [04 — Locutores vs frases](04_locutores_vs_frases/README.md) | Mais vozes ou mais frases por voz? | **ativo** | K = 64 sem KL; grade de K na receita original |
| [05 — Representação da voz](05_representacao_da_voz/README.md) | Codificador, latente e referência limitam a cópia? | quase encerrado | controle de reinicialização em outros N |
| [06 — Atração pela voz padrão](06_atracao_pela_voz_padrao/README.md) | A voz padrão puxa as saídas? Quanto é artefato da medida? | pausado | nada |
| [07 — Aumento de vozes](07_aumento_de_vozes/README.md) | Dá para fabricar locutores? | bloqueado | nada |
| [08 — Flow matching (F5-TTS)](08_flow_matching/README.md) | A mesma lei vale numa arquitetura moderna e escalável? | pausado | preparação do F5-TTS |
| [09 — Dublagem](09_dublagem/README.md) | Aplicação final: dublar filmes | futuro | nada |
