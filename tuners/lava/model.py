import torch
import torch.nn as nn
import re

from peft.tuners.tuners_utils import BaseTuner
from peft.tuners.tuners_utils import check_target_module_exists
from peft.tuners.lava.layer import LavaLayer
from peft.tuners.lava.config import LavaConfig


class LavaModel(BaseTuner):
    """
    LoRA와 동일한 구조를 따르는 VAE-style Linear Adapter Model.
    BaseTuner를 그대로 확장하여
    - adapter injection
    - enabling/disabling adapters
    - trainable param marking
    모두 PEFT와 호환되도록 구현.
    """

    # ------------------------------------------------------------
    # 1. __init__
    # ------------------------------------------------------------
    def __init__(self, model: nn.Module, config: LavaConfig, adapter_name: str, low_cpu_mem_usage=False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage)
        # BaseTuner가 모든 injection 로직을 실행함.
        self.config = model.config          # <- Trainer가 요구함
        self.peft_config = config           # <- PEFT 내부 구조 유지
    # ------------------------------------------------------------
    # 2. adapter switching
    # ------------------------------------------------------------
    def set_adapter(self, adapter_names):
        """
        BaseTunerModel → BaseTunerLayer 구조와 동일하게,
        각 LavaLayer에 adapter 이름을 전달.
        """

        # adapter_names 를 리스트로 통일
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # LavaLayer 에 전달
        for module in self.model.modules():
            if isinstance(module, LavaLayer):
                module.set_adapter(adapter_names)

        # BaseTuner 내부에서 active adapter는 여기에 저장됨
        self._active_adapter = adapter_names
        
    def enable_input_require_grads(self, **kwargs):
        """
        Gradient Checkpointing을 위해 입력 임베딩에 grad를 요구하는 메서드를 
        내부의 실제 모델(Llama 등)로 전달합니다.
        """
        if hasattr(self.model, "enable_input_require_grads"):
            return self.model.enable_input_require_grads(**kwargs)
        else:
            # 직접 구현 (위 메서드가 없는 구형 모델 대비)
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            return self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # ------------------------------------------------------------
    # 3. adapter config 준비
    # ------------------------------------------------------------
    def _prepare_adapter_config(self, peft_config: LavaConfig, model_config):
        """
        LoRA 구조를 그대로 따름.
        target_modules 기본값 설정.
        """

        if peft_config.target_modules is None:
            # 기본적으로 transformer QKV projection linear 에 적용
            peft_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        if peft_config.r is None:
            peft_config.r = 8

        return peft_config

    # ------------------------------------------------------------
    # 4. 어떤 모듈 이름이 대상인지 체크
    # ------------------------------------------------------------
    def _check_target_module_exists(self, peft_config: LavaConfig, key: str):
        return check_target_module_exists(peft_config, key)

    # ------------------------------------------------------------
    # 5. injection — Linear 를 LavaLayer 로 교체
    # ------------------------------------------------------------
    def _create_and_replace(
        self,
        peft_config: LavaConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ):
        """
        LoRA 구조와 동일하게 Linear만 교체.
        """

        # Linear 가 아니면 skip
        if not isinstance(target, nn.Linear):
            return

        # ✅ 레이어 인덱스 추출
        layer_idx = 0
        match = re.search(r'\.(?:layers|block|h)\.(\d+)\.', current_key)
        if match:
            layer_idx = int(match.group(1))

        # r 가져오기
        rank = peft_config.r
        alpha = peft_config.alpha
        lora_dropout = getattr(peft_config, 'lora_dropout', 0.0)

        # 새로운 LavaLayer 생성
        new_module = LavaLayer(
            base_layer=target,
            adapter_name=adapter_name,
            rank=rank,
            alpha=alpha,
            lora_dropout=lora_dropout,
        )

        # parent 모듈에 교체 반영
        setattr(parent, target_name, new_module)

    # ------------------------------------------------------------
    # 6. trainable parameter 설정
    # ------------------------------------------------------------
    def _mark_only_adapters_as_trainable(self, model):
        """
        Lava adapter만 학습되도록 설정.
        """
        for name, param in model.named_parameters():
            if "lava" in name:   # LavaLayer 내부 adapter 키워드
                param.requires_grad = True
            else:
                param.requires_grad = False

    # ------------------------------------------------------------
    # 7. enable / disable adapters
    # ------------------------------------------------------------
    def enable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, LavaLayer):
                module.disable_adapters = False

    def disable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, LavaLayer):
                module.disable_adapters = True

    # ------------------------------------------------------------
    # 8. forward
    # ------------------------------------------------------------
    def forward(self, *args, **kwargs):
        """
        BaseTuner → BaseModel forward 호출
        ViT 같은 이미지 모델은 input_ids를 사용하지 않으므로 제거
        """
        kwargs["return_dict"] = True
        # ViT 등 이미지 모델 호환성: input_ids, attention_mask, inputs_embeds 제거
        if "pixel_values" in kwargs:
            kwargs.pop("input_ids", None)
            kwargs.pop("attention_mask", None)
            kwargs.pop("inputs_embeds", None)
        return self.model(*args, **kwargs)
    
        
        
    def gradient_checkpointing_enable(self, **kwargs):
        """
        Trainer compatibility:
        forward gradient checkpointing call to base model
        """
        if hasattr(self.model, "gradient_checkpointing_enable"):
            return self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            return self.model.gradient_checkpointing_disable()

    @property
    def is_gradient_checkpointing(self):
        if hasattr(self.model, "is_gradient_checkpointing"):
            return self.model.is_gradient_checkpointing
        return False
    # ------------------------------------------------------------
    # 9. generation interface (🔥 필수)
    # ------------------------------------------------------------
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    # (optional but recommended)
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()