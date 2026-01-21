# peft/tuners/moca/model.py

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner
from peft.tuners.tuners_utils import check_target_module_exists, _get_submodules

from .layer import MoCALayer
from .config import MoCAConfig


class MoCAModel(BaseTuner):
    """
    Complete PEFT-compatible Tuner for MoCA.
    LoRA와 동일한 구조를 따라 MoCAAdapter/MoCALayer만 교체해서 적용.
    """

    # ------------------------------------------------------------
    # Adapter Selection
    # ------------------------------------------------------------
    def set_adapter(self, adapter_names):
        """PEFT BaseTuner와 동일한 adapter 전환 방식"""
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        self.active_adapter = adapter_names

        for module in self.model.modules():
            if isinstance(module, MoCALayer):
                module.set_adapter(adapter_names)

    # ------------------------------------------------------------
    # 1. Prepare adapter config
    # ------------------------------------------------------------
    def _prepare_adapter_config(self, peft_config: MoCAConfig, model_config: dict):
        """
        Inject 전에 config 조정.
        """
        # target modules 기본값 미지정 시 Transformer Q/K/V/O
        if getattr(peft_config, "target_modules", None) is None:
            peft_config.target_modules = ["query_proj", "key_proj", "value_proj", "dense"]

        # rank
        if not hasattr(peft_config, "r"):
            peft_config.r = 8

        # adapter_size
        if not hasattr(peft_config, "adapter_size"):
            hidden_size = model_config.get("hidden_size", None)
            if hidden_size is None:
                raise ValueError("MoCAConfig.adapter_size not set and base model has no hidden_size in config")

            peft_config.adapter_size = hidden_size

        return peft_config

    # ------------------------------------------------------------
    # 2. Target module name check
    # ------------------------------------------------------------
    def _check_target_module_exists(self, peft_config: MoCAConfig, key: str):
        """기본적인 target module 존재 검사"""
        return check_target_module_exists(peft_config, key)

    # ------------------------------------------------------------
    # 3. Create & Replace layers
    # ------------------------------------------------------------
    def _create_and_replace(
        self,
        peft_config: MoCAConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ):
        """
        nn.Linear 레이어를 MoCALayer 로 교체
        """

        # Linear 에만 적용 (LoRA와 동일 정책)
        if not isinstance(target, nn.Linear):
            return

        in_dim = target.in_features
        out_dim = target.out_features

        # PEFT compatibility: hidden_size = out_features (Linear의 출력 채널)
        new_module = MoCALayer(
            base_layer=target,
            adapter_name=adapter_name,
            hidden_size=out_dim,
            rank=peft_config.r,
        )

        setattr(parent, target_name, new_module)

    # ------------------------------------------------------------
    # 4. Trainable parameters marking
    # ------------------------------------------------------------
    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        """
        "moca" 가 들어있는 파라미터만 학습.
        """
        for name, param in model.named_parameters():
            if "moca" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # ------------------------------------------------------------
    # 5. Enable / Disable MoCA layers
    # ------------------------------------------------------------
    def disable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, MoCALayer):
                module.disable_adapters = True

    def enable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, MoCALayer):
                module.disable_adapters = False

    # ------------------------------------------------------------
    # 6. Forward wrapper
    # ------------------------------------------------------------
    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        return self.model(*args, **kwargs)
