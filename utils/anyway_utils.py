# ---------- Any-Way utilities ----------
import random
import torch
import torch.nn.functional as F


def compute_prototypes(embeddings, targets, n_classes, normalize):
    """
    embeddings: Tensor [B, d]  - support set의 backbone feature
    targets:    Tensor [B]     - support set label (numeric class 0..N-1)
    n_classes:  int            - 현재 task의 클래스 수 (N)
    """
    d = embeddings.size(-1)
    prototypes = torch.zeros(n_classes, d, device=embeddings.device)

    for c in range(n_classes):
        mask = targets == c
        if mask.sum() > 0:
            prototypes[c] = embeddings[mask].mean(0)

    if normalize:
        prototypes = torch.nn.functional.normalize(prototypes, p=2, dim=1)

    return prototypes  # shape: [N, d]

def build_non_overlapping_assignments(O: int, N: int, rng: random.Random):
    idx = list(range(O))
    rng.shuffle(idx)
    J = O // N
    return [idx[j*N:(j+1)*N] for j in range(J)]  # 각 원소는 길이 N의 인덱스 리스트

def anyway_loss(logits_O, y, S):
    # Σ_j CE(logits[:, s_j], y).  스케일 안정 위해 평균을 추천.
    losses = [F.cross_entropy(logits_O[:, torch.as_tensor(s, device=logits_O.device)], y) for s in S]
    return torch.stack(losses).mean()

def anyway_ensemble_logits(logits_O, S):
    # 논문 테스트 시 ensembling: Σ_j S_j(logit) (softmax 전 합)  :contentReference[oaicite:3]{index=3}
    chunks = [logits_O[:, torch.as_tensor(s, device=logits_O.device)] for s in S]
    return torch.stack(chunks, dim=0).sum(dim=0)  # (B, N)