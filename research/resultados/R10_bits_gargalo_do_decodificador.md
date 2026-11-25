# R10 — Teoria da informação: o gargalo é o decodificador, e o ganho com N é cobertura

- **Data:** novembro de 2026 (fase do Colab, análise sem treino novo)
- **Caminho:** [02 — gargalo do decodificador](../caminhos/02_gargalo_do_decodificador/README.md), [01 — lei de escala](../caminhos/01_lei_de_escala/README.md)
- **Veredito:** confirmado; definiu a métrica principal (bits de voz)

## Pergunta

Em que ponto da cadeia referência → codificador → latente → decodificador → áudio a informação sobre a voz se perde?

## Montagem

Medido nos áudios gerados das 38 vozes externas: C = I(voz da saída; referência), cotas pela matriz de confusão de identificação; teto log2 38 = 5,25 bits. Duas teorias propostas por agentes independentes e criticadas em três rodadas cada (`reports/teoria/informacao/A` e `B`).

## Resultado

| Elo da cadeia | N = 8 | N = 128 |
|---|---|---|
| Fala real (verificador) | 4,2–4,7 | 4,2–4,7 |
| μ (modelo gaussiano) | 5,1 | 5,2 |
| z (modelo gaussiano) | 4,4 | 4,8 |
| Saída, cota de Fano | 0,3 | 0,9–1,1 |
| Saída, cota de entropia cruzada | 0,75 [0,28; 1,20] | 1,99 [1,35; 2,60] |

## Conclusão

- **O decodificador perde cerca de 4 bits**; codificador e latente não são o problema.
- **O ganho com N é cobertura, não precisão:** H(Ŷ) sobe 1,44 bit [0,68; 2,32] de N = 8 para 128, e H(Ŷ|S) não muda. O decodificador alcança mais vozes diferentes, com o mesmo ruído por voz.
- **Lei em bits, com dois pontos:** 0,31 bit por duplicação de N [0,17; 0,47]. Igualar a fala real exigiria cerca de 10^5 locutores nessa taxa.
- **Previsão a priori confirmada:** referências mais longas tornam a cópia mais tudo-ou-nada (p = 0,0002).
- **Consequência:** o alvo de arquitetura é aumentar o conjunto de vozes que o decodificador alcança, com mais vozes no treino ou condicionamento em todo o decodificador. Isso levou ao FiLM ([R12](R12_film.md)).

## Ressalvas

Os valores da saída de N = 128 aqui são da fase do Colab (uma seed); no cluster, com 3 seeds, 128 × 32 deu 1,75 ± 0,04 bit ([R11](R11_curva_de_N_receita_original.md)).

## Dados

Seção 8.11 de [`reports/teoria_cobertura.md`](../../reports/teoria_cobertura.md); `reports/teoria/03_informacao.md`; `scripts/coverage/voice_bits.py`.
