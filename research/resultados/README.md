# Resultados

Um arquivo por resultado, na ordem em que foram obtidos. Cada arquivo tem pergunta, montagem, tabela, conclusão, ressalvas e de onde vêm os dados. Para acrescentar um resultado, use o próximo número livre e siga [`AGENTS.md`](../../AGENTS.md).

"Bits" são os bits de voz nas 38 vozes externas (teto 5,25), a métrica principal desde R10 ([protocolo](../protocolo.md)).

## Fase do Colab (avaliação em 4 vozes novas, depois 38 vozes)

| ID | Resultado | Caminho | Veredito |
|---|---|---|---|
| [R01](R01_grade_N_x_K.md) | Frases decidem as vozes treinadas, locutores decidem as novas | 01, 04 | confirmado |
| [R02](R02_troca_de_referencia_e_contrastiva.md) | Troca de referência e perda contrastiva não ajudam | 05 | negativo |
| [R03](R03_vetor_de_locutor_pre_treinado.md) | Vetor de locutor pré-treinado é ignorado pelo treino | 05 | negativo |
| [R04](R04_dropout_de_estilo.md) | Com dropout de estilo o vetor é usado, mas não reduz a necessidade de locutores | 05 | negativo |
| [R05](R05_pseudo_locutores.md) | Pseudo-locutores por velocidade não contam como locutores | 07 | negativo (previsto) |
| [R06](R06_128_e_200_locutores.md) | 128 e 200 locutores: a lei acerta K = 8; 4 vozes são ruidosas demais | 01 | parcial |
| [R07](R07_varias_referencias.md) | Média de várias referências não ajuda nas 38 vozes; só baixa o WER | 05 | negativo |
| [R08](R08_avaliacao_larga_38_vozes.md) | Em 38 vozes, o ganho de 32 para 128 locutores é real | 01 | confirmado |
| [R09](R09_vozes_dificeis.md) | Vozes perto da voz padrão falham mais; com N pequeno, as longe das treinadas | 06 | descritivo |
| [R10](R10_bits_gargalo_do_decodificador.md) | Teoria da informação: o gargalo é o decodificador; o ganho com N é cobertura | 02, 01 | confirmado |

## Fase do cluster (38 vozes externas, 2–3 seeds)

| ID | Resultado | Caminho | Número-chave | Veredito |
|---|---|---|---|---|
| [R11](R11_curva_de_N_receita_original.md) | Curva de N na receita original | 01 | +0,27 bit por duplicação; 1,76 bit em 241 | confirmado |
| [R12](R12_film.md) | FiLM no decodificador: ganho cresce com N | 02 | +0,44 bit em 241 | parcial |
| [R13](R13_latente_menor_e_controle.md) | Latente menor: a perda vem sobretudo da reinicialização | 05 | reinicializar −0,78; 32 → 8 dimensões −0,40 | previsão falhou (confundida) |
| [R14](R14_metade_da_voz_padrao.md) | Metade da voz padrão: +4 pontos de cópia, bits iguais | 06 | +0,08 bit | efeito da medida |
| [R15](R15_rampa_do_kl.md) | A rampa do KL fazia treinar mais piorar | 03 | sem KL: 1,81 → 2,44 → 2,84 | confirmado |
| [R16](R16_curva_de_N_sem_kl.md) | Curva de N sem KL | 01 | +0,41 bit por duplicação; 2,90 em 241 | confirmado |
| [R17](R17_film_sem_kl.md) | FiLM sem KL | 02 | 3,08 bits em 241 (melhor) | promissor |
| [R18](R18_duracao_do_treino.md) | Duração do treino: saturando perto de 10–20 mil passos | 03 | +0,63, +0,40, +0,12 por dobra | indício |
| [R19](R19_locutores_vs_frases.md) | Com os mesmos dados, locutores rendem mais que frases | 04 | 1.024 frases: 0,74 (8 loc.) vs 1,36 (128 loc.) | confirmado |

## Dados

[`dados/`](dados/README.md): tabela de todas as avaliações do cluster. As tabelas dos resultados R11 em diante saem dela com `python scripts/coverage/results_table.py`.
