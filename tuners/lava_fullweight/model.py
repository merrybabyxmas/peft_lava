import torch
import torch.nn as nn
import re

from peft.tuners.tuners_utils import BaseTuner
from peft.tuners.tuners_utils import check_target_module_exists
from peft.tuners.lava_fullweight.layer import LavaFullWeightLayer
from peft.tuners.lava_fullweight.config import LavaFullWeightConfig


class LavaFullWeightModel(BaseTuner):
    """
    LAVA FullWeight Model: 입력 의존적인 분산을 사용하는 VAE-style Adapter
    """

    def __init__(self, model: nn.Module, config: LavaFullWeightConfig, adapter_name: str, low_cpu_mem_usage=False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage)
        self.config = model.config
        self.peft_config = config

    def set_adapter(self, adapter_names):
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for module in self.model.modules():
            if isinstance(module, LavaFullWeightLayer):
                module.set_adapter(adapter_names)

        self._active_adapter = adapter_names

    def enable_input_require_grads(self, **kwargs):
        if hasattr(self.model, "enable_input_require_grads"):
            return self.model.enable_input_require_grads(**kwargs)
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            return self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def _prepare_adapter_config(self, peft_config: LavaFullWeightConfig, model_config):
        if peft_config.target_modules is None:
            peft_config.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        if peft_config.r is None:
            peft_config.r = 8

        return peft_config

    def _check_target_module_exists(self, peft_config: LavaFullWeightConfig, key: str):
        return check_target_module_exists(peft_config, key)

    def _create_and_replace(
        self,
        peft_config: LavaFullWeightConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ):
        if not isinstance(target, nn.Linear):
            return

        rank = peft_config.r
        alpha = peft_config.alpha

        new_module = LavaFullWeightLayer(
            base_layer=target,
            adapter_name=adapter_name,
            rank=rank,
            alpha=alpha,
        )

        setattr(parent, target_name, new_module)

    def _mark_only_adapters_as_trainable(self, model):
        for name, param in model.named_parameters():
            if "lava_fullweight" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def enable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, LavaFullWeightLayer):
                module.disable_adapters = False

    def disable_adapter_layers(self):
        for module in self.model.modules():
            if isinstance(module, LavaFullWeightLayer):
                module.disable_adapters = True

    def forward(self, *args, **kwargs):
        kwargs["return_dict"] = True
        if "pixel_values" in kwargs:
            kwargs.pop("input_ids", None)
            kwargs.pop("attention_mask", None)
            kwargs.pop("inputs_embeds", None)
        return self.model(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs):
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

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()
