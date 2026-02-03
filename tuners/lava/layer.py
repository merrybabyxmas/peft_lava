import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from peft.tuners.tuners_utils import BaseTunerLayer

class LavaAdapter(nn.Module):
    _global_seed = 42

    @classmethod
    def set_global_seed(cls, seed: int):
        cls._global_seed = seed

    def __init__(self, hidden_size: int, rank: int, alpha: int, lora_dropout: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        self.scale = self.alpha / rank

        # [1] 가중치 행렬 분리 (수식의 W_A와 B)
        # W_A 연산에서 bias를 False로 설정하여 입력 투영만 수행합니다.
        self.W_A = nn.Linear(hidden_size, rank, bias=False) 
        self.W_B = nn.Linear(rank, hidden_size, bias=False)

        # [2] 순수 Bias 파라미터 (이것이 VIB의 대상이 됩니다)
        self.b_mu = nn.Parameter(torch.zeros(rank))
        self.b_logvar = nn.Parameter(torch.ones(rank) * -6.0)
        self.gate_scale = nn.Parameter(torch.ones(rank))

        # [3] Dropout & Generator
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self._rng_generator = torch.Generator()
        self._rng_generator.manual_seed(LavaAdapter._global_seed)

        # 초기화
        nn.init.kaiming_uniform_(self.W_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_B.weight)
        nn.init.zeros_(self.b_mu)

        self._last_mu = None
        self._last_logvar = None
        self._latent_stability = None
    
    def reset_generator(self, seed: Optional[int] = None):
        target_seed = seed if seed is not None else LavaAdapter._global_seed
        self._rng_generator.manual_seed(target_seed)

    def _sample_noise(self, mu: torch.Tensor) -> torch.Tensor:
        """장치(Device) 불일치를 방지하며 노이즈 샘플링"""
        if self._rng_generator.device != mu.device:
            self._rng_generator = torch.Generator(device=mu.device)
            self._rng_generator.manual_seed(LavaAdapter._global_seed)
        return torch.randn(mu.shape, generator=self._rng_generator, dtype=mu.dtype, device=mu.device)

    def forward(self, h: torch.Tensor):
        if not isinstance(h, torch.Tensor):
            return h

        # Step 1: 입력 데이터 처리 (Deterministic Path)
        h_latent = self.lora_dropout(h)
        # 입력에 의존하는 투영값 (W_A * x)
        proj_h = self.W_A(h_latent)

        # Step 2: Bias 샘플링 (Stochastic Path)
        logvar = torch.clamp(self.b_logvar, -10, 2)
        std = torch.exp(0.5 * logvar)

        if self.training:
            # 입력 h와 상관없이 b_mu 크기에 맞춘 노이즈 생성
            eps = self._sample_noise(self.b_mu)
            # Antithetic Sampling (Bias만 흔듦)
            b_A_pos = self.b_mu + eps * std
            b_A_neg = self.b_mu - eps * std
            
            self._latent_stability = F.mse_loss(b_A_pos, b_A_neg)
            b_A = b_A_pos
        else:
            b_A = self.b_mu
            self._latent_stability = None

        # Step 3: 결합 및 게이팅
        # latent = (W_A * x) + b_A
        latent = proj_h + b_A
        
        # 게이팅은 잠재 공간의 활성화 정도를 조절
        # gate = torch.sigmoid(latent * self.gate_scale)
        gate = 1
        delta = self.W_B(latent * gate) * self.scale
        
        # [핵심] Step 4: VIB Loss용 데이터 저장
        # 이제 _last_mu는 입력 h를 포함하지 않는 순수 Bias 파라미터입니다.
        self._last_mu = self.b_mu
        self._last_logvar = logvar 
        
        return h + delta
# ==========================================================
#   LavaLayer Wrapper (PEFT 호환)
# ==========================================================

class LavaLayer(BaseTunerLayer, nn.Module):
    is_adapter = True
    adapter_layer_names = ("lava",)

    def __init__(self, base_layer: nn.Linear, adapter_name: str, rank: int, alpha: int, lora_dropout: float = 0.0):
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LavaLayer can only wrap nn.Linear layers.")

        self.base_layer = base_layer
        # 베이스 모델 가중치 고정
        for p in self.base_layer.parameters():
            p.requires_grad = False

        out_dim = base_layer.out_features
        # LavaAdapter 생성 시 lora_dropout 전달
        self.lava = nn.ModuleDict({
            adapter_name: LavaAdapter(out_dim, rank, alpha, lora_dropout)
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
        # 1. Base 모델 연산
        h = self.base_layer(x)
        
        if not isinstance(h, torch.Tensor) or self.disable_adapters:
            return h

        # 2. 어댑터 연산 (LavaAdapter 내부에서 h + delta 수행)
        for name in self.active_adapters:
            if name in self.lava:
                h = self.lava[name](h)
        
        return h