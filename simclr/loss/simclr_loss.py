import torch
import torch.nn as nn


class SimCLRLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.temperature = self.cfg.LOSS.temperature

    def forward(self, pred_aug1, pred_aug2):
        assert pred_aug1.shape == pred_aug2.shape
        B = pred_aug1.shape[0]

        pred_aug1 = nn.functional.normalize(pred_aug1, p=2, dim=1)
        pred_aug2 = nn.functional.normalize(pred_aug2, p=2, dim=1)

        preds = torch.cat([pred_aug1, pred_aug2], dim=0)
        sim_mat = (preds @ preds.T) / self.temperature

        mask = torch.eye(2 * B, device=sim_mat.device, dtype=torch.bool)
        sim_mat = sim_mat.masked_fill(mask, float("-inf"))
        
        pos_idx = torch.arange(2 * B, device=sim_mat.device)
        pos_idx = (pos_idx + B) % (2 * B)
        positives = sim_mat[torch.arange(2 * B), pos_idx]
        
        denom = torch.logsumexp(sim_mat, dim=1)
        loss = (-positives + denom).mean()

        return loss
