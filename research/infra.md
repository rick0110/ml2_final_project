# Infraestrutura: onde os experimentos rodam e onde ficam os dados

Este arquivo descreve a estrutura, sem endereços. Os endereços das máquinas e os scripts operacionais ficam fora do repositório (que é público), no PC de controle em `~/ml2_backup/`.

## Máquinas

- **Cluster do laboratório:** 10 máquinas, cada uma com uma RTX A4500 (20 GB), 32 threads e cerca de 430 GB livres em disco. Acesso por SSH pela rede Tailscale, sem sudo. Nos dados elas são numeradas de 1 a 10, na ordem de `~/ml2_backup/hosts.txt` no PC de controle.
- **PC de controle:** só coordena (filas, verificação de saúde, página de inventário). **Não guarda dados de experimento.**

Tempos típicos por treino: cerca de 1,3 s por passo, uns 55 minutos para 2.500 passos e 30 a 60 minutos de avaliação; cerca de 9 GB de memória de GPU.

## Em cada máquina (`~/ml2/`)

| Caminho | Conteúdo |
|---|---|
| `repo/` | cópia do código, com os dados processados em `repo/data/processed/` e os checkpoints de partida em `repo/ckpt/` |
| `repo/experiments/tacotron2-vae/cov_<run>/` | treino: checkpoints a cada 500 passos, `hparams.json`, `metrics.jsonl` (perdas e KL por passo), TensorBoard, amostras |
| `out/<run>/step_<passos>/strength_1.0/` | avaliação: `results.csv`, `embeddings.npz`, `voice_bits.json`, áudios |
| `out/progress.txt` | uma linha por evento (treinando, avaliando, terminado) |
| `node_queue.txt` | fila de jobs desta máquina, uma linha por job: `<run> <passos> [<passos> ...]` |
| `worker.sh` | pega o próximo job da fila e roda (`run_node.sh` → `scripts/coverage/colab_job.sh`); funciona com o PC desligado |
| `drive_up.sh` | a cada 20 minutos envia os dados desta máquina ao Google Drive |

Os treinos de antes de 24/11/2026 (02:30) guardavam só os últimos checkpoints; os intermediários desses estão na cópia do GCS. Desde então, todos os checkpoints ficam.

## Filas

1. A fila geral fica no PC de controle (`~/ml2_backup/queue.txt`, e depois `backlog.txt`), em ordem de prioridade.
2. O despachante do PC (serviço `ml2-dispatch`) completa a fila de cada máquina até 6 jobs, um job por máquina por rodada, para distribuir as prioridades. Ele também garante que `worker.sh` e `drive_up.sh` estejam rodando.
3. Um job que **continua** um run existente (mais passos) vai direto para a fila da máquina que tem o checkpoint.
4. Se a fila geral e a reserva acabam, o PC escreve `QUEUE EMPTY` em `~/ml2_backup/alerts.log`. Máquina fora do ar gera `DOWN` no mesmo arquivo.

O serviço `ml2-sync` do PC verifica as máquinas a cada 2 minutos (`~/ml2_backup/status.txt`) e regenera a página de inventário (`~/ml2_backup/inventario.html`) a cada ~10 minutos. A página lista cada experimento com estado, máquina, resultados, checkpoints e caminhos, além das filas.

## Onde estão os dados

| Lugar | O que tem | Observação |
|---|---|---|
| Máquinas do laboratório | tudo (fonte principal) | caminhos acima |
| Google Drive, pasta `ml2_lab_cluster/<máquina>/` | `experiments/`, `out/`, `logs/` de cada máquina | enviado pelas próprias máquinas (rclone, só acrescenta). O login do Drive vence a cada 7 dias (app em modo de teste): renovar com `rclone config reconnect gdrive:` no PC e copiar o `rclone.conf` para as máquinas |
| Bucket do GCS do projeto, pasta `cluster/` | cópia de tudo até 24/11/2026 ~02:00 (113 GB) | envio desligado |
| Este repositório | tabelas de resultados (`research/resultados/dados/`) | sem checkpoints nem áudios |

## Recuperar tudo de uma máquina

```
rsync -a -e 'ssh -p 2222' <usuário>@<máquina>:ml2/repo/experiments/ ./<máquina>/experiments/
rsync -a -e 'ssh -p 2222' <usuário>@<máquina>:ml2/out/ ./<máquina>/out/
```

A página de inventário mostra esses comandos já preenchidos.
