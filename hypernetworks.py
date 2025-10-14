
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class HyperNetworkLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, device):
        super().__init__()
        self.device = device
        self.args = args

        # 공통 trunk
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.ReLU(inplace=True)

        self.linear_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, task_state):
        """
        task_state: [N, input_dim]  (예: class prototypes)
        return:
            W: [N, output_dim]  (classifier weights)
            b: [N]              (classifier biases)
        """

        out = self.linear1(task_state)
        out = self.activation1(out)

        W = self.linear_2(out)

        return W



class HyperNetworkAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim=10):
        super(HyperNetworkAutoencoder, self).__init__()

        # Encoder: Reduce input to latent space of size latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

        # Decoder: Map latent space back to image dimensions (3, 84, 84)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)

        # Decoder
        x = self.decoder(x)

        return x



def normalize_adj(adj: torch.Tensor):
    """
    adj: [N, N] (0/1 혹은 가중 인접행렬), self-loop 미포함 가능
    return: \hat{A} = D^{-1/2} (A + I) D^{-1/2}
    """
    N = adj.size(0)
    A = adj
    A = A + torch.eye(N, device=adj.device, dtype=adj.dtype)  # add self-loop
    deg = A.sum(-1)                                           # [N]
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


class SimpleGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        """
        x        : [N, d_in]  노드 feature
        adj_norm : [N, N] 정규화된 adjacency (D^{-1/2} (A+I) D^{-1/2})
        """
        agg = adj_norm @ x            # [N, d_in]
        out = self.linear(agg)        # [N, d_out]
        return F.relu(out, inplace=True)


class GNNWeightGenerator(nn.Module):
    """
    Linear classifier용 가중치/바이어스 생성기 (Any-Way 대응)
    prototypes -> GNN -> [W, b]
    """
    def __init__(self, d_proto=1600, hidden=256, out_dim=1600, use_bias=True, dropout_p=0.0):
        super().__init__()
        self.gnn1 = SimpleGNNLayer(d_proto, hidden)
        self.gnn2 = SimpleGNNLayer(hidden, hidden)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.proj_W = nn.Linear(hidden, out_dim)   # 클래스별 weight (d_out = feat_dim)
        self.use_bias = use_bias
        if use_bias:
            self.proj_b = nn.Linear(hidden, 1)     # 클래스별 bias

        # 초기화: linear head 안정화
        for m in [self.proj_W] + ([self.proj_b] if use_bias else []):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, prototypes, adj):
        """
        prototypes: [N, d_proto]  (support로부터 추출한 클래스 프로토타입)
        adj       : [N, N]        (클래스 간 그래프; 없으면 I 사용 가능)
        return    : W [N, out_dim], b [N] (linear classifier용)
        """
        if adj is None:
            N = prototypes.size(0)
            adj = torch.eye(N, device=prototypes.device, dtype=prototypes.dtype)

        adj_norm = normalize_adj(adj)
        h = self.gnn1(prototypes, adj_norm)     # [N, hidden]
        W = self.proj_W(h)                      # [N, out_dim]
        if self.use_bias:
            b = self.proj_b(h).squeeze(-1)      # [N]
        else:
            b = None
        return W, b