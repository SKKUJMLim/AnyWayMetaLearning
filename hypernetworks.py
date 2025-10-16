
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
    def __init__(self, input_dim, output_dim, latent_dim=64):
        super(HyperNetworkAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
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
    인접 행렬을 정규화합니다. GCN 공식: \hat{A} = D^{-1/2} (A + I) D^{-1/2}
    adj: [N, N] (0/1 혹은 가중 인접 행렬)
    return: 정규화된 인접 행렬
    """
    N = adj.size(0)
    A = adj

    # 1. Self-loop 추가 (A + I)
    A = A + torch.eye(N, device=adj.device, dtype=adj.dtype)

    # 2. Degree 계산
    deg = A.sum(-1)

    # 3. D^{-1/2} 계산
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)

    # D^{-1/2} A D^{-1/2}
    adj_norm = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)

    return adj_norm


class SimpleGNNLayer(nn.Module):
    """
    간단한 GNN 레이어 (GCN의 기본 연산 구조를 따름)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        """
        x        : [N, d_in]  (노드 feature)
        adj_norm : [N, N]     (정규화된 adjacency)
        """
        # 1. 메시지 집계 (Aggregation): \hat{A} x
        agg = torch.matmul(adj_norm, x)

        # 2. 선형 변환 및 활성화: W (\hat{A} x)
        out = self.linear(agg)
        return F.relu(out, inplace=True)


class GNNWeightGenerator(nn.Module):
    """
    GNN 기반 분류기 가중치/바이어스 생성기 (바이어스 제외).
    프로토타입과 클래스 관계 정보를 활용하여 Any-Way 분류기 W를 생성.
    """

    # use_bias는 False를 가정하거나, 파라미터에서 제외합니다.
    def __init__(self, d_proto=1600, hidden=256, out_dim=1600, use_bias=False, dropout_p=0.0):
        super().__init__()
        self.gnn1 = SimpleGNNLayer(d_proto, hidden)
        self.gnn2 = SimpleGNNLayer(hidden, hidden)
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.proj_W = nn.Linear(hidden, out_dim)  # 클래스별 weight (d_out = feat_dim)

        # self.use_bias = False를 가정하므로, self.proj_b 정의를 제외합니다.
        self.use_bias = False

        # 안정적인 학습을 위한 초기화 (proj_b 제외)
        for m in [self.proj_W]:  # use_bias 로직 제거
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # GNNLayer 내부의 Linear 계층 초기화는 SimpleGNNLayer에서 처리됩니다.

    def forward(self, prototypes, adj):
        """
        prototypes: [N, d_proto]  (z - 적응형 클래스 문맥 벡터)
        adj       : [N, N]        (클래스 간 관계를 나타내는 인접 행렬)
        return    : W [N, out_dim], b=None
        """
        # Adj가 제공되지 않았거나, GNNWeightGenerator를 일반 Hypernet처럼 사용할 경우 I를 Adj로 사용
        if adj is None:
            N = prototypes.size(0)
            adj = torch.eye(N, device=prototypes.device, dtype=prototypes.dtype)

        # 1. 정규화된 인접 행렬 계산
        adj_norm = normalize_adj(adj)

        # 2. GNN 순전파
        h = self.gnn1(prototypes, adj_norm)  # [N, hidden]
        h = self.dropout(h)
        h = self.gnn2(h, adj_norm)  # [N, hidden]
        h = self.dropout(h)

        # 3. 최종 W 생성
        W = self.proj_W(h)  # [N, out_dim]

        return W