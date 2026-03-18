import os
import numpy as np

import torch
import torch.nn as nn
from mrinufft import get_operator


############ Input Positional Encoding ############
class Positional_Encoder():
    def __init__(self, params):
        if params['embedding'] == 'gauss':
            self.B = torch.randn((params['embedding_size'], params['coordinates_size'])) * params['scale']
            self.B = self.B.cuda()
        else:
            raise NotImplementedError

    def embedding(self, x):
        x_embedding = (2. * np.pi * x) @ self.B.t()
        x_embedding = torch.cat([torch.sin(x_embedding), torch.cos(x_embedding)], dim=-1)
        return x_embedding

############ Input Spactial and Temporal Encoding ############

class SpatioTemporal_Encoder(nn.Module):
    def __init__(self, params):
        """
        params:
        --------
        spatial_embedding: str, 'randn' or 'gauss'
        temporal_embedding: str, 'randn' or 'gauss'
        embedding_size_spatial: int
        embedding_size_temporal: int
        coordinates_size_spatial: int  # 2(2D) 或 3(3D) 等
        coordinates_size_temporal: int # 通常 1
        scale_spatial: float
        scale_temporal: float
        learnable_temporal: bool       # 是否让 B_temporal 可学习
        """
        super().__init__()

        Es = int(params['embedding_size_spatial'])
        Et = int(params['embedding_size_temporal'])
        Ds = int(params['coordinates_size_spatial'])
        Dt = int(params['coordinates_size_temporal'])

        # ===== Spatial B（固定）=====
        if params.get('spatial_embedding', 'randn') == 'randn':
            B_s = torch.randn((Es, Ds)) * float(params['scale_spatial'])
        elif params.get('spatial_embedding', 'randn') == 'gauss':
            B_s = torch.normal(mean=0.0, std=float(params['scale_spatial']),
                               size=(Es, Ds))
        else:
            raise NotImplementedError(f"Unknown spatial embedding: {params['spatial_embedding']}")
        # 用 buffer 保证随模型迁移/保存，但不参与优化
        self.register_buffer('B_spatial', B_s.float())

        # ===== Temporal B（可学习 or 固定）=====
        if params['temporal_embedding'] == 'randn':
            B_t = torch.randn((Et, Dt)) * float(params['scale_temporal'])
        elif params['temporal_embedding'] == 'gauss':
            B_t = torch.normal(mean=0.0, std=float(params['scale_temporal']),
                               size=(Et, Dt))
        else:
            raise NotImplementedError(f"Unknown temporal embedding: {params['temporal_embedding']}")

        if params.get('learnable_temporal', False):
            self.B_temporal = nn.Parameter(B_t.float())
        else:
            self.register_buffer('B_temporal', B_t.float())
        
    # ----------------- New: 独立空间编码 -----------------
    @torch.no_grad()  # 若你希望固定空间编码梯度，可保留；想要可训则去掉这一行
    def spatial_encode(self, coords_spatial: torch.Tensor) -> torch.Tensor:
        """
        coords_spatial: (..., Ds), 取值一般在 [0,1]
        return: (..., 2*Es)  先 sin 再 cos 拼接
        """
        # 拉平为 (N, Ds)
        orig_shape = coords_spatial.shape[:-1]
        x = coords_spatial.reshape(-1, coords_spatial.shape[-1]).to(self.B_spatial.dtype)

        # 投影 -> sin/cos
        # (N, Ds) @ (Ds, Es) = (N, Es)
        proj = (2.0 * torch.pi) * (x @ self.B_spatial.t())
        embed = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        # 还原形状
        return embed.reshape(*orig_shape, embed.shape[-1])

    # ----------------- New: 独立时间编码 -----------------
    def temporal_encode(self, coords_temporal: torch.Tensor) -> torch.Tensor:
        """
        coords_temporal: (..., Dt), 通常 Dt=1
        return: (..., 2*Et)
        """
        orig_shape = coords_temporal.shape[:-1]
        t = coords_temporal.reshape(-1, coords_temporal.shape[-1]).to(self.B_temporal.dtype)

        proj = (2.0 * torch.pi) * (t @ self.B_temporal.t())
        embed = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

        return embed.reshape(*orig_shape, embed.shape[-1])

    # ----------------- 保留旧接口：同时返回两者 -----------------
    def embedding(self, coords_spatial: torch.Tensor, coords_temporal: torch.Tensor):
        """
        coords_spatial:  (..., Ds)
        coords_temporal:(..., Dt)
        return: (spatial_embed, temporal_embed)
                其中维度分别为 (..., 2*Es) 与 (..., 2*Et)
        """
        return self.spatial_encode(coords_spatial), self.temporal_encode(coords_temporal)

    # ----------------- PyTorch 标准 forward（可选） -----------------
    def forward(self, coords_spatial: torch.Tensor, coords_temporal: torch.Tensor,
                concat: bool = False):
        """
        concat=False: 返回 (spatial_embed, temporal_embed)
        concat=True : 返回 concat(spatial, temporal) 于最后一维
        """
        s, t = self.embedding(coords_spatial, coords_temporal)
        if concat:
            return torch.cat([s, t], dim=-1)
        return s, t
    
############ Fourier Feature Network ############
class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class FFN(nn.Module):
    def __init__(self, params):
        super(FFN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out



############ SIREN Network ############
class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class SIREN(nn.Module):
    def __init__(self, params):
        super(SIREN, self).__init__()

        num_layers = params['network_depth']
        hidden_dim = params['network_width']
        input_dim = params['network_input_size']
        output_dim = params['network_output_size']

        layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
        for i in range(1, num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(SirenLayer(hidden_dim, output_dim, is_last=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)

        return out


def ema_update(ema_val, x: float, beta: float = 0.98):
    x = float(x)
    return x if ema_val is None else beta * ema_val + (1.0 - beta) * x

def ramp_factor(epoch: int, start: int, ramp: int) -> float:
    """
    Control the weight of certain loss term gradually increasing from 0 to 1.
    epoch < start -> 0
    epoch in [start, start+ramp] -> [0,1]
    epoch > start+ramp -> 1
    """
    if epoch < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return float(min(1.0, (epoch - start) / float(ramp)))

def complex_mse_loss(pred, target):
    return torch.mean(torch.abs(pred - target) ** 2)

def l2_grad3d(u: torch.Tensor) -> torch.Tensor:
    """
    u: (B,X,Y,Z) (float)
    return: scalar ~ ||∇u||_2^2
    """
    dx = u[:, 1:, :, :] - u[:, :-1, :, :]
    dy = u[:, :, 1:, :] - u[:, :, :-1, :]
    dz = u[:, :, :, 1:] - u[:, :, :, :-1]
    return (dx * dx).mean() + (dy * dy).mean() + (dz * dz).mean()

def tv3d(u: torch.Tensor, eps: float = 1e-12, isotropic: bool = True) -> torch.Tensor:
    """
    u: (B,X,Y,Z)  (float)
    return: scalar
    """
    # forward differences
    dx = u[:, 1:, :, :] - u[:, :-1, :, :]
    dy = u[:, :, 1:, :] - u[:, :, :-1, :]
    dz = u[:, :, :, 1:] - u[:, :, :, :-1]

    if isotropic:
        # align to common interior shape (B, X-1, Y-1, Z-1)
        dx = dx[:, :, :-1, :-1]
        dy = dy[:, :-1, :, :-1]
        dz = dz[:, :-1, :-1, :]
        return torch.sqrt(dx * dx + dy * dy + dz * dz + eps).mean()
    else:
        return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean()) / 3.0


class FourierFrameOP(torch.autograd.Function):
    """Forward NUFFT: Image -> K-space"""

    @staticmethod
    def forward(ctx, x, nufft_op, traj):
        # ctx.save_for_backward(x, trajs)
        ctx.save_for_backward(x, traj)
        ctx.nufft_op = nufft_op
        y = torch.zeros([traj.shape[0],nufft_op.n_coils, traj.shape[1]], dtype=torch.complex64, device=x.device)

        for i in range(len(traj)):
            nufft_op.samples = traj[i].to("cpu").numpy()  
            y[i] = nufft_op.op(x[i])[0]
        return y

    @staticmethod
    def backward(ctx, dy):
        x, traj = ctx.saved_tensors
        dx = torch.zeros((len(traj), *ctx.nufft_op.shape),
                         dtype=torch.complex64, device=dy.device)
        for i in range(len(traj)):
            ctx.nufft_op.samples = traj[i].to("cpu").numpy()
            dx[i] = ctx.nufft_op.adj_op(dy[i])

            # alpha = torch.mean(torch.linalg.norm(dy[i], axis=0)) / torch.mean(torch.linalg.norm(ctx.nufft_op.op(dx[i])))
            # dx[i] = dx[i] * alpha

        return dx, None, None

class FourierFrameADJOP(torch.autograd.Function):
    """Adjoint NUFFT: K-space -> Image"""

    @staticmethod
    def forward(ctx, y, nufft_op, traj):
        ctx.save_for_backward(y, traj)
        ctx.nufft_op = nufft_op

        x = torch.zeros((len(traj), *nufft_op.shape),
                        dtype=torch.complex64, device=y.device)
        for i in range(len(traj)):
            nufft_op.samples = traj[i].to("cpu").numpy()
            # nufft_op.samples = traj[i]
            x[i] = nufft_op.adj_op(y[i])

            # alpha = torch.mean(torch.linalg.norm(y[i], axis=0)) / torch.mean(torch.linalg.norm(nufft_op.op(x[i])))
            # x[i] = x[i] * alpha

        return x

    @staticmethod
    def backward(ctx, dx):
        y, traj = ctx.saved_tensors
        dy = torch.zeros(traj.shape[:2], dtype=torch.complex64, device=dx.device)
        for i in range(len(traj)):
            ctx.nufft_op.samples = traj[i].to("cpu").numpy()
            # ctx.nufft_op.samples = traj[i]
            dy[i] = ctx.nufft_op.op(dx[i])

        return dy, None, None

class NUFFTLayer(nn.Module):
    """High-level NUFFT module with forward & adjoint operators"""

    def __init__(self, shape, trajs, density, n_coils, smaps=None):
        super().__init__()
        # samples [N_frames, N_samples, 3]
        # shape [N_x, N_y, N_z]
        self.nufft_op = get_operator("cufinufft")(
            samples = trajs[0],  # use the first frame to initialize
            shape = shape, 
            n_coils = n_coils,
            density = density,
            smaps=smaps) # initialize with first frame trajectory.
        # self.trajs = trajs
        self.trajs = torch.from_numpy(trajs).float().to('cuda')
        self.n_coils = n_coils
    def op(self, x):
        """Image -> K-space (NUFFT)"""
        return FourierFrameOP.apply(x, self.nufft_op, self.trajs)

    def adj_op(self, y):
        """K-space -> Image (Adjoint NUFFT)"""
        return FourierFrameADJOP.apply(y, self.nufft_op, self.trajs)

class FourierFrameOP3Dstatic(torch.autograd.Function):
    """Forward NUFFT: Image -> K-space"""

    @staticmethod
    def forward(ctx, x, nufft_op):
        S = nufft_op.n_samples
        T = x.shape[0]
        ctx.nufft_op = nufft_op
        # ctx.save_for_backward(x, trajs)
        ctx.save_for_backward(x)
        y = torch.zeros((T, nufft_op.n_coils, S), dtype=torch.complex64, device=x.device)
        for i in range(T):
            y[i] = nufft_op.op(x[i]) # [0]
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        nufft_op = ctx.nufft_op
        T = x.shape[0]

        dx = torch.zeros((T, *nufft_op.shape), dtype=torch.complex64, device=dy.device)
        for i in range(T):
            dx[i] = nufft_op.adj_op(dy[i])

        return dx, None, None

class NUFFTLayer3Dstatic(nn.Module):
    """High-level NUFFT module with forward & adjoint operators"""

    def __init__(self, shape, trajs, density, n_coils, csm=None):
        super().__init__()
        # samples [N_frames, N_samples, 3]
        # shape [N_x, N_y, N_z]
        self.nufft_op = get_operator("cufinufft")(
            samples = trajs[0],  # use the first frame to initialize
            shape = shape, 
            n_coils = n_coils,
            density = density,
            smaps=csm) # initialize with first frame trajectory.
        # self.trajs = trajs
        # self.trajs = torch.from_numpy(trajs).float().to('cuda')
        self.n_coils = n_coils
    def op(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> K-space (NUFFT)"""
        return FourierFrameOP3Dstatic.apply(x, self.nufft_op)

class FourierFrameOP3D(torch.autograd.Function):
    """Forward NUFFT: Image -> K-space"""

    @staticmethod
    def forward(ctx, x, nufft_op, traj):
        T, S, _ = traj.shape
        ctx.nufft_op = nufft_op
        # ctx.save_for_backward(x, trajs)
        ctx.save_for_backward(x)
        ctx.traj = traj
        y = torch.zeros((T, nufft_op.n_coils, S), dtype=torch.complex64, device=x.device)
        for i in range(T):
            nufft_op.samples = traj[i]  
            y[i] = nufft_op.op(x[i]) # [0]
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        nufft_op = ctx.nufft_op
        traj  = ctx.traj
        T = traj.shape[0]

        dx = torch.zeros((T, *nufft_op.shape), dtype=torch.complex64, device=dy.device)
        for i in range(T):
            nufft_op.samples = traj[i]
            dx[i] = nufft_op.adj_op(dy[i])

        return dx, None, None

class NUFFTLayer3D(nn.Module):
    """High-level NUFFT module with forward & adjoint operators"""

    def __init__(self, shape, trajs, density, n_coils, csm=None):
        super().__init__()
        # samples [N_frames, N_samples, 3]
        # shape [N_x, N_y, N_z]
        self.nufft_op = get_operator("cufinufft")(
            samples = trajs[0],  # use the first frame to initialize
            shape = shape, 
            n_coils = n_coils,
            density = density,
            smaps=csm) # initialize with first frame trajectory.
    def op(self, x: torch.Tensor, traj) -> torch.Tensor:
        """Image -> K-space (NUFFT)"""
        return FourierFrameOP3D.apply(x, self.nufft_op, traj)

class NufftAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op):
        """
        x:  (Coils, H, W, D) 或 (H, W, D) 的 Tensor，requires_grad=True
        op: mrinufft 的算子对象
        """
        # 保存 op 到 context 以便 backward 使用
        ctx.op = op 
        kspace = op.op(x)
        return kspace

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: Loss 对 kspace 的梯度 (Coils, Samples)
        """
        op = ctx.op
        grad_input = op.adj_op(grad_output)
        return grad_input, None

def apply_nufft(x, op):
    # 辅助函数，简化调用
    return NufftAutograd.apply(x, op)

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def compute_grad_norm(model, norm_type=2.0):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total += float(param_norm.item() ** (norm_type))
    return total ** (1.0 / norm_type)

def add_param_grad_hists(writer, model, step, every=0):
    """每隔 every 步记录一次参数/梯度直方图；every=0 表示不记录。"""
    if every and (step % every == 0):
        for name, p in model.named_parameters():
            try:
                writer.add_histogram(f"param/{name}", p.detach().cpu().numpy(), step)
                if p.grad is not None:
                    writer.add_histogram(f"grad/{name}", p.grad.detach().cpu().numpy(), step)
            except Exception:
                pass

def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory

from torch.utils.data import Dataset, DataLoader, RandomSampler, BatchSampler

def create_grid_4d(t, x, y, z):
    grid_t, grid_x, grid_y, grid_z = torch.meshgrid([
        torch.linspace(0, 1, steps=t),
        torch.linspace(0, 1, steps=x),
        torch.linspace(0, 1, steps=y),
        torch.linspace(0, 1, steps=z),
    ])
    grid = torch.stack([grid_t, grid_x, grid_y, grid_z], dim=-1)
    return grid

def create_spatiotemporal_grid3DT(t, x, y, z):
    """
    Create a normalized 3D+T grid for (time, x, y, z), matching data order (T, H, W).
    Returns:
        coords_spatial: (X*Y*Z, 3)
        coords_temporal: (T, 1)
    """
    grid_t = torch.linspace(0, 1, steps=t).unsqueeze(-1)  # (T, 1)
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.linspace(0, 1, steps=x),
        torch.linspace(0, 1, steps=y),
        torch.linspace(0, 1, steps=z),
        indexing='ij'  # ensures (T, H, W) order
    )
    coords_spatial = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    coords_temporal = grid_t
    return coords_spatial, coords_temporal

# -----------------------------------------------------------------------------
# 1. Dataset 定义 (已修改为接收 operator list)
# -----------------------------------------------------------------------------
class FMRI3DT_DatasetwithNUFFT(Dataset):
    """
    Dataset for Dynamic MRI INR.
    一个样本 = 一个时间点 (frame)。
    返回: (kspace[t], operator[t], t_coord[t], t_index)
    """
    def __init__(self, kspace: torch.Tensor, nufft_operators: list, t_coords: torch.Tensor):
        """
        Args:
            kspace: (T, C, S) complex64/complex128
            nufft_operators: list, 长度为 T，存储每帧对应的 NUFFT 算子对象
            t_coords: (T,) float32
        """
        self.T = kspace.shape[0]
        
        # 维度检查
        assert len(nufft_operators) == self.T, \
            f"Operator 数量 ({len(nufft_operators)}) 与 kspace 帧数 ({self.T}) 不匹配"
        assert t_coords.shape[0] == self.T, \
            "t_coords 长度与 kspace 帧数不匹配"

        self.kspace = kspace
        self.nufft_operators = nufft_operators  # 直接存储对象列表
        self.t_coords = t_coords

    def __len__(self):
        return self.T

    def __getitem__(self, t_idx: int):
        # 注意：这里返回的 nufft_operators[t_idx] 是一个 Python 对象
        return (
            self.kspace[t_idx],          # (C, S)
            self.nufft_operators[t_idx], # Object (NUFFT Operator)
            self.t_coords[t_idx],        # ()
            t_idx                        # int
        )

# -----------------------------------------------------------------------------
# 2. 自定义 Collate Function (核心关键)
# -----------------------------------------------------------------------------
def nufft_collate_fn(batch):
    """
    自定义批处理函数。
    默认的 collate_fn 无法处理 nufft_operator 对象（因为它不能被 torch.stack）。
    我们需要手动将 operator 组织成一个 list。
    
    batch: List of tuples, 每个 tuple 是 __getitem__ 的返回值
           [(kspace, op, t_coord, idx), (kspace, op, t_coord, idx), ...]
    """
    # 解包 batch (zip(*batch) 会把同一位置的元素归类到一起)
    kspace_list, op_list, t_coord_list, idx_list = zip(*batch)

    # 1. 对 Tensor 类型的数据进行 Stack
    kspace_batch = torch.stack(kspace_list)     # (B, C, S)
    t_coord_batch = torch.stack(t_coord_list)   # (B,)
    
    # idx_list 是 int，先转 tensor
    idx_batch = torch.tensor(idx_list)          # (B,)

    # 2. 对 Operator 对象，保留为 Python List (Tuple 也可以，取决于你的后续使用习惯)
    # op_list 本身已经是 tuple of objects，直接返回即可
    op_batch = list(op_list) 

    return kspace_batch, op_batch, t_coord_batch, idx_batch

# -----------------------------------------------------------------------------
# 3. Loader Factory (基于你的原有代码修改)
# -----------------------------------------------------------------------------
def make_loader_random_fullcover_NUFFT(dataset, batch_size, num_workers=0, pin_memory=True):
    """
    随机但不放回、整轮全覆盖的 DataLoader。
    """
    # 强制检查：如果包含 CUDA 算子对象，num_workers 必须为 0
    if num_workers > 0:
        print("Warning: Dataset 包含 CUDA NUFFT 算子对象，强烈建议设置 num_workers=0，否则可能导致 Pickling Error 或 CUDA 初始化错误。")

    sampler = RandomSampler(dataset, replacement=False) # 打乱且不放回
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=nufft_collate_fn,  # <--- 替换为你自定义的 collate
        num_workers=num_workers,      # 建议保持 0
        pin_memory=pin_memory,
    )


