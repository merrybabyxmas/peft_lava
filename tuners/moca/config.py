from dataclasses import dataclass, field
from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class MoCAConfig(PeftConfig):
    r: int = 16
    adapter_size: int = None
    target_modules: list[str] = field(default_factory=list)

    # LoRA 호환 dummy 값들 (PEFT 내부가 요구)
    lora_alpha: int = 1
    lora_dropout: float = 0.0
    fan_in_fan_out: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MOCA
