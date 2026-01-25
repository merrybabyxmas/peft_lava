import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from peft.tuners.tuners_utils import BaseTunerLayer

# ==========================================================
#   Optimized Lava Adapter (Gating Removed & Seeding Utils)
# ==========================================

class LavaAdapter(nn.Module):
    # 클래스 레벨에서 공유되는 기본 시드
    _global_seed = 42

    @classmethod
    def set_global_seed(cls, seed: int):
        """모든 LavaAdapter 인스턴스의 기본 시드를 설정"""
        cls._global_seed = seed

    def __init__(self, hidden_size: int, rank: int, alpha: int):
        super().__init__()
        self.alpha = alpha
        self.rank = rank
        self.scale = self.alpha / rank

        # 1. Low-rank Projection (W_mu: 입력의 평균점 매핑, W_o: 원래 차원 복구)
        self.W_mu = nn.Linear(hidden_size, rank, bias=True)
        self.W_o = nn.Linear(rank, hidden_size, bias=True)

        # 2. Input-invariant Variance (모든 입력에 대해 동일한 기저 노이즈 수준 학습)
        self.logvar_bias = nn.Parameter(torch.ones(rank) * -6.0)

        # 3. 재현성을 위한 개별 Generator (CPU에서 생성 후 GPU 이동하는 안전한 방식)
        self._rng_generator = torch.Generator()
        self._rng_generator.manual_seed(LavaAdapter._global_seed)

        # 초기화 (Kaiming Uniform 사용)
        nn.init.kaiming_uniform_(self.W_mu.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W_mu.bias)
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

        # 트레이너 수집용 변수 (LavaBaseTrainer 호환용)
        self._last_mu = None
        self._last_logvar = None
        self._latent_stability = None

    def reset_generator(self, seed: Optional[int] = None):
        """Generator의 상태를 초기화하여 노이즈 순서를 리셋"""
        target_seed = seed if seed is not None else LavaAdapter._global_seed
        self._rng_generator.manual_seed(target_seed)

    def _sample_noise(self, mu: torch.Tensor) -> torch.Tensor:
        """
        고정된 generator를 사용하여 재현 가능한 noise 샘플링.
        Generator와 mu의 장치가 일치하도록 보장합니다.
        """
        # 현재 제너레이터의 장치가 mu의 장치와 다르면 해당 장치용으로 새로 생성
        if self._rng_generator.device != mu.device:
            self._rng_generator = torch.Generator(device=mu.device)
            # 기존 시드를 유지하여 재현성 확보
            self._rng_generator.manual_seed(LavaAdapter._global_seed)

        return torch.randn(
            mu.shape, 
            generator=self._rng_generator, 
            dtype=mu.dtype, 
            device=mu.device
        )

    def forward(self, h: torch.Tensor, external_noise: Optional[torch.Tensor] = None):
        if not isinstance(h, torch.Tensor):
            return h

        # [Mean] mu = W_mu * h + bias_mu
        mu = self.W_mu(h)
        
        # [Log-var] 입력 독립적 분산 파라미터 (Clamp로 수치 안정성 확보)
        logvar = torch.clamp(self.logvar_bias, -10, 2)
        std = torch.exp(0.5 * logvar)

        if self.training:
            # [속도 최적화] 외부에서 주입된 노이즈가 있으면 사용(RNG 호출 횟수 감소)
            # 없을 경우 내부 Generator 또는 randn_like로 폴백
            if external_noise is not None:
                eps = external_noise
            else:
                eps = self._sample_noise(mu)
            
            # Antithetic Sampling (z1, z2)
            z1 = mu + eps * std
            z2 = mu - eps * std
            
            # Latent Stability 계산 (내부 MSE Loss)
            self._latent_stability = F.mse_loss(z1, z2)
            z = z1 
        else:
            # 추론 시에는 결정론적(Deterministic)으로 동작
            z = mu
            self._latent_stability = None

        # [Output] 시그모이드 게이트를 제거하고 샘플 z를 직접 투영
        delta = self.W_o(z) * self.scale
        
        # 트레이너 수집용 데이터 저장
        self._last_mu = mu
        self._last_logvar = logvar.expand_as(mu) 
        
        return h + delta

# ==========================================================
#   LavaLayer Wrapper (Adapter Management)
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
        for p in self.base_layer.parameters():
            p.requires_grad = False

        out_dim = base_layer.out_features
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
        # 1. Base 모델 연산
        h = self.base_layer(x)
        
        if not isinstance(h, torch.Tensor) or self.disable_adapters:
            return h

        # 2. Trainer에서 넘겨준 external_noise 추출
        external_noise = kwargs.get("external_noise", None)

        # 3. 활성화된 모든 어댑터 적용
        for name in self.active_adapters:
            if name in self.lava:
                h = self.lava[name](h, external_noise=external_noise)
        
        return h