
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
        self.linear1 = nn.Linear(input_dim, 256)
        self.activation1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(256, 512)
        self.activation2 = nn.ReLU(inplace=True)

        self.linear_3 = nn.Linear(512, output_dim)

    def forward(self, task_state):
        """
        task_state: [N, input_dim]  (예: class prototypes)
        return:
            W: [N, output_dim]  (classifier weights)
            b: [N]              (classifier biases)
        """

        out = self.linear1(task_state)
        out = self.activation1(out)

        out = self.linear2(out)
        out = self.activation2(out)

        W = self.linear_3(out)

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


# --- GNN 보조 함수 ---
def normalize_adj(adj: torch.Tensor):
    """
    인접 행렬 정규화: \hat{A} = D^{-1/2} (A + I) D^{-1/2}
    """
    N = adj.size(0)
    A = adj
    A = A + torch.eye(N, device=adj.device, dtype=adj.dtype)
    deg = A.sum(-1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    adj_norm = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return adj_norm


class SimpleGNNLayer(nn.Module):
    """
    간단한 GNN 레이어 (GCN의 기본 연산 구조)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        """
        x        : [N, d_in]  (노드 feature)
        adj_norm : [N, N]     (정규화된 adjacency)
        """
        agg = torch.matmul(adj_norm, x)
        out = self.linear(agg)
        return F.relu(out, inplace=True)


class GNNWeightGenerator(nn.Module):
    """
    GNN 기반 분류기 가중치 생성기 (Input Z: 100차원).
    """

    def __init__(self, d_proto=100, hidden=256, out_dim=1600, use_bias=False, dropout_p=0.1):
        super().__init__()

        # d_proto = 100으로 설정 (z의 입력 차원)
        self.gnn1 = SimpleGNNLayer(d_proto, hidden)
        self.gnn2 = SimpleGNNLayer(hidden, hidden)

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # out_dim = 1600으로 설정 (최종 W의 출력 차원)
        self.proj_W = nn.Linear(hidden, out_dim)

        self.use_bias = use_bias
        # 바이어스 생성 로직은 제외됨 (use_bias=False)

        # 안정적인 학습을 위한 초기화
        for m in [self.proj_W]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            # SimpleGNNLayer 내부의 Linear 계층 초기화는 SimpleGNNLayer에서 처리됩니다.

    def forward(self, prototypes, adj=None):
        """
        prototypes: [N, 100]  (z - 적응형 클래스 문맥 벡터)
        adj       : [N, N]    (클래스 간 관계를 나타내는 인접 행렬)
        return    : W [N, 1600], b=None
        """
        if adj is None:
            N = prototypes.size(0)
            adj = torch.eye(N, device=prototypes.device, dtype=prototypes.dtype)

        adj_norm = normalize_adj(adj)

        # GNN 순전파 (100 -> 256 -> 256)
        h = self.gnn1(prototypes, adj_norm)
        h = self.dropout(h)
        h = self.gnn2(h, adj_norm)
        h = self.dropout(h)

        # 최종 W 생성 (256 -> 1600)
        W = self.proj_W(h)
        b = None  # 바이어스 생성 제외

        return W , b



## GAT
import torch.nn as nn
import torch
import torch.nn.functional as F
import math


# --- GAT Layer 구현 ---

class GATLayer(nn.Module):
    """
    Graph Attention Network (GAT)의 단일 어텐션 헤드 구현
    """

    def __init__(self, in_dim, out_dim, dropout_p=0.1, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.dropout_p = dropout_p
        self.out_dim = out_dim
        self.alpha = alpha  # Negative slope parameter for Leaky ReLU
        self.concat = concat  # Concatenation for multi-head GAT (여기서는 단일 헤드)

        # 1. 선형 변환: h_i -> Wh_i
        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 2. 어텐션 벡터: a
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 3. 바이어스 및 드롭아웃
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout_p)

    def _prepare_attentional_mechanism_input(self, Wh):
        """ 어텐션 점수 e_ij 계산을 위한 입력 준비 """
        N = Wh.size()[0]
        # [N, 1, D_out]을 [N, N, D_out]으로 복제
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)
        # [1, N, D_out]을 [N, N, D_out]으로 복제
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)

        # [N, N, 2*D_out]로 두 노드의 특징을 결합
        return torch.cat([Wh_i, Wh_j], dim=2)

    def forward(self, h, adj=None):
        """
        h   : [N, d_in] (z)
        adj : [N, N] (0/1 인접 행렬, 마스킹용)
        """
        # 1. 특징 변환: Wh = h W
        Wh = torch.matmul(h, self.W)  # [N, D_out]

        # 2. 어텐션 입력 준비 및 점수 계산
        a_input = self._prepare_attentional_mechanism_input(Wh)  # [N, N, 2*D_out]

        # LeakyReLU(a_input * a)를 통해 어텐션 점수 e_ij 계산
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # [N, N]

        # 3. 마스킹 및 Softmax
        # Adjacency Matrix를 사용하여 연결되지 않은 노드의 어텐션 점수를 -inf로 마스킹
        zero_vec = -9e15 * torch.ones_like(e)

        # Adj가 None이면 모든 노드를 연결된 것으로 간주 (마스킹 없음)
        if adj is not None:
            # adj는 0 또는 1이므로, 0인 곳을 마스킹
            e = torch.where(adj > 0, e, zero_vec)

        # Softmax를 통해 정규화된 어텐션 계수 alpha_ij 획득
        attention = F.softmax(e, dim=1)  # [N, N]
        attention = self.dropout(attention)

        # 4. 정보 집계: h'_i = sum_j (alpha_ij * Wh_j)
        h_prime = torch.matmul(attention, Wh)  # [N, D_out]

        # 5. 활성화 함수
        if self.concat:
            return F.elu(h_prime + self.bias)
        else:
            return h_prime + self.bias


class GATWeightGenerator(nn.Module):
    """
    GAT 기반 분류기 가중치 생성기 (Any-Way 대응).
    """

    def __init__(self, d_proto=100, hidden=256, out_dim=1600, use_bias=False, dropout_p=0.1):
        super().__init__()

        # d_proto = 100, hidden = 256 가정
        self.gat1 = GATLayer(d_proto, hidden, dropout_p=dropout_p, concat=True)
        self.gat2 = GATLayer(hidden, hidden, dropout_p=dropout_p, concat=False)  # 마지막 계층은 Non-Concatenated

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # 최종 W 생성 (hidden -> out_dim)
        self.proj_W = nn.Linear(hidden, out_dim)
        self.use_bias = use_bias

        # 초기화는 nn.Linear에서 처리됨

    def forward(self, prototypes, adj=None):
        """
        prototypes: [N, 100]  (z - 적응형 클래스 문맥 벡터)
        adj       : [N, N]    (클래스 간 관계를 나타내는 인접 행렬 - 마스킹용)
        return    : W [N, 1600], b=None
        """
        # Adj가 유사도 행렬일 경우, 0/1 마스크로 변환 (GAT는 0/1 마스크를 선호)
        if adj is not None and torch.max(adj) > 1.0:
            # 0보다 큰 값은 연결(1), 나머지는 0으로 마스크 생성
            adj_mask = (adj > 1e-6).float()
        elif adj is not None:
            adj_mask = adj
        else:
            adj_mask = None  # 마스킹 없음 (Fully Connected)

        # GAT 순전파 (100 -> 256 -> 256)
        h = self.gat1(prototypes, adj_mask)
        h = self.dropout(h)
        h = self.gat2(h, adj_mask)
        h = self.dropout(h)

        # 최종 W 생성 (256 -> 1600)
        W = self.proj_W(h)
        b = None  # 바이어스 생성 제외

        return W, b