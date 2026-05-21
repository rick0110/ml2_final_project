import torch


def length_regulator(encoder_outputs: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """Expand encoder outputs according to durations.

    Args:
        encoder_outputs: (batch, seq_len, dim)
        durations: (batch, seq_len) - integer durations (number of frames per token)

    Returns:
        expanded: (batch, total_frames, dim)
    """
    batch_size, seq_len, dim = encoder_outputs.size()
    outs = []
    for b in range(batch_size):
        rep = []
        for t in range(seq_len):
            d = int(durations[b, t].item())
            if d <= 0:
                continue
            rep.append(encoder_outputs[b, t].unsqueeze(0).expand(d, -1))
        if rep:
            rep = torch.cat(rep, dim=0)  # (N_frames, dim)
        else:
            rep = encoder_outputs.new_zeros((0, dim))
        outs.append(rep)

    # Pad to max length
    max_len = max([o.size(0) for o in outs]) if outs else 0
    expanded = encoder_outputs.new_zeros((batch_size, max_len, dim))
    for b in range(batch_size):
        L = outs[b].size(0)
        if L > 0:
            expanded[b, :L, :] = outs[b]
    return expanded
