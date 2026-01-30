import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from peft.tuners.tuners_utils import BaseTunerLayer

# ==========================================================
#   LAVA FullWeight Adapter (A matrix sampling version)
# ==========================================================

class LavaFullWeightAdapter(nn.Module):
    """
    LAVA FullWeight: 기존 LAVA가 bias만 샘플링했다면,
    이 버전은 W_logvar를 통해 입력 의존적인 분산을 계산합니다.
    """
    _global_seed = 42

    @classmethod 
    def set_global_seed(cls, seed: int):
        cls._global_seed = seed

    def __init__(self, hidden_size: int, rank: int, alpha: int):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        self.scale = self.alpha / rank

        # 1. Low-rank Projection for Mean
        self.W_mu = nn.Linear(hidden_size, rank, bias=True)

        # 2. Low-rank Projection for Log-Variance (FullWeight: 입력 의존적 분산)
        self.W_logvar = nn.Linear(hidden_size, rank, bias=True)

        # 3. Output projection (no bias for fair comparison with LoRA)
        self.W_o = nn.Linear(rank, hidden_size, bias=False)

        # 4. 재현성을 위한 개별 Generator
        self._rng_generator = torch.Generator()
        self._rng_generator.manual_seed(LavaFullWeightAdapter._global_seed)

        # 초기화
        nn.init.kaiming_uniform_(self.W_mu.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_mu.bias)
        nn.init.kaiming_uniform_(self.W_logvar.weight, a=math.sqrt(5))
        nn.init.constant_(self.W_logvar.bias, -6.0)  # 초기에는 작은 분산
        nn.init.zeros_(self.W_o.weight)

        # 트레이너 수집용 변수
        self._last_mu = None
        self._last_logvar = None
        self._latent_stability = None

    def reset_generator(self, seed: Optional[int] = None):
        target_seed = seed if seed is not None else LavaFullWeightAdapter._global_seed
        self._rng_generator.manual_seed(target_seed)

    def _sample_noise(self, mu: torch.Tensor) -> torch.Tensor:
        if self._rng_generator.device != mu.device:
            self._rng_generator = torch.Generator(device=mu.device)
            self._rng_generator.manual_seed(LavaFullWeightAdapter._global_seed)

        return torch.randn(
            mu.shape,
            generator=self._rng_generator,
            dtype=mu.dtype,
            device=mu.device
        )

    def forward(self, h: torch.Tensor, external_noise: Optional[torch.Tensor] = None):
        if not isinstance(h, torch.Tensor):
            return h

        # [Mean] mu = W_mu * h
        mu = self.W_mu(h)

        # [Log-var] 입력 의존적 분산: logvar = W_logvar * h (FullWeight 방식)
        logvar = self.W_logvar(h)
        logvar = torch.clamp(logvar, -10, 2)  # 수치 안정성
        std = torch.exp(0.5 * logvar)

        if self.training:
            if external_noise is not None:
                eps = external_noise
            else:
                eps = self._sample_noise(mu)

            # Antithetic Sampling
            z1 = mu + eps * std
            z2 = mu - eps * std

            # Latent Stability
            self._latent_stability = F.mse_loss(z1, z2)
            z = z1
        else:
            z = mu
            self._latent_stability = None

        delta = self.W_o(z) * self.scale

        # 트레이너 수집용
        self._last_mu = mu
        self._last_logvar = logvar

        return h + delta


# ==========================================================
#   LavaFullWeightLayer Wrapper
# ==========================================================

class LavaFullWeightLayer(BaseTunerLayer, nn.Module):
    is_adapter = True
    adapter_layer_names = ("lava_fullweight",)

    def __init__(self, base_layer: nn.Linear, adapter_name: str, rank: int, alpha: int):
        nn.Module.__init__(self)
        BaseTunerLayer.__init__(self)

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LavaFullWeightLayer can only wrap nn.Linear layers.")

        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        out_dim = base_layer.out_features
        self.lava_fullweight = nn.ModuleDict({
            adapter_name: LavaFullWeightAdapter(out_dim, rank, alpha)
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

        external_noise = kwargs.get("external_noise", None)

        for name in self.active_adapters:
            if name in self.lava_fullweight:
                h = self.lava_fullweight[name](h, external_noise=external_noise)

        return h
