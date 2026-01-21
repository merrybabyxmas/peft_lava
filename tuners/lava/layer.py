import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from peft.tuners.tuners_utils import BaseTunerLayer

# ==========================================================
#   Unified Lava Adapter 
# ==========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LavaAdapter(nn.Module):
    def __init__(self, hidden_size, rank, alpha):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        self.scale = self.alpha / rank

        # 1. Mean Projection (Input-dependent)
        # 여기서 bias=True를 설정하면 사용자님이 말씀하신 bias_mu가 포함됩니다.
        self.W_mu = nn.Linear(hidden_size, rank, bias=True)
        self.W_o = nn.Linear(rank, hidden_size, bias=True)

        # 2. Log-variance Parameter (Input-invariant)
        # 입력 h와 곱해지는 가중치 없이, 오직 고정된 파라미터(Bias 역할)만 가집니다.
        # 이것이 곧 사용자님이 말씀하신 bias_logvar 역할을 합니다.
        self.logvar_bias = nn.Parameter(torch.ones(rank) * -6.0)
        
        # Gating parameter
        self.gate_scale = nn.Parameter(torch.ones(rank))

        # Initialization
        nn.init.kaiming_uniform_(self.W_mu.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_mu.bias) # mu의 초기 바이어스는 0
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, h):
        if not isinstance(h, torch.Tensor):
            return h

        # [Mean] 입력 h에 따라 결정됨: mu = W_mu * h + bias_mu
        mu = self.W_mu(h)
        
        # [Log-var] 입력 h와 무관하게 학습된 값 사용: logvar = bias_logvar
        logvar = torch.clamp(self.logvar_bias, -10, 2)
        std = torch.exp(0.5 * logvar)

        if self.training:
            # [Sampling] mu와 std가 준비되었으므로 샘플링 가능
            # Antithetic Sampling (z1, z2) 유지
            eps = torch.randn_like(mu)
            z1 = mu + eps * std
            z2 = mu - eps * std
            
            # 내부 Latent Stability 계산 (4 * sigma^2)
            self._latent_stability = F.mse_loss(z1, z2)
            
            z = z1 
        else:
            # 추론 시에는 노이즈 없이 평균값만 사용 (LoRA와 동일하게 작동)
            z = mu
            self._latent_stability = None

        # Gating & Output Projection
        gate = torch.sigmoid(mu * self.gate_scale)
        delta = self.W_o(z * gate) * self.scale
        
        # 트레이너 수집용 (VIB 계산에 사용)
        self._last_mu = mu
        self._last_logvar = logvar.expand_as(mu) 
        
        return h + delta

# ==========================================================
#   LavaLayer Wrapper (Cleaned Up)
# ==========================================================

class LavaLayer(BaseTunerLayer, nn.Module):
    is_adapter = True
    adapter_layer_names = ("lava",)

    def __init__(self, base_layer: nn.Linear, adapter_name: str, rank: int, alpha: int):
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LavaLayer can only wrap nn.Linear layers.")

        self.base_layer = base_layer

        # Freeze original weights
        for p in self.base_layer.parameters():
            p.requires_grad = False

        out_dim = base_layer.out_features

        # Adapter container
        self.lava = nn.ModuleDict({
            adapter_name: LavaAdapter(out_dim, rank, alpha)
        })

        self._active_adapters = [adapter_name]
        self._disable_adapters = False

    def set_adapter(self, adapter_names):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        self._active_adapters = adapter_names

    @property
    def active_adapters(self):
        return self._active_adapters

    @property
    def disable_adapters(self):
        return self._disable_adapters

    @disable_adapters.setter
    def disable_adapters(self, v: bool):
        self._disable_adapters = v

    def forward(self, x, *args, **kwargs):
        h = self.base_layer(x)
        
        if not isinstance(h, torch.Tensor) or self.disable_adapters:
            return h

        for name in self.active_adapters:
            if name in self.lava:
                h = self.lava[name](h)
        
        return h