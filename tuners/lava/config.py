from peft.config import PeftConfig

class LavaConfig(PeftConfig):
    def __init__(
        self,
        r=8,
        alpha = 4,
        target_modules=None,
        bias="none",
        inference_mode=False,
        lora_dropout=0.0,
        **kwargs
    ):
        # wrapper 비활성화
        self.modules_to_save = []

        # 먼저 PeftConfig 초기화
        super().__init__(**kwargs)

        # 반드시 여기에서 다시 지정해야 함
        self.peft_type = "LAVA"

        # 이제 나머지 설정
        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules
        self.bias = bias
        self.inference_mode = inference_mode
        self.lora_dropout = lora_dropout

        # task_type 기본값 설정
        if not hasattr(self, "task_type") or self.task_type is None:
            self.task_type = kwargs.get("task_type", "SEQ_CLS")  # Deberta는 이것이 맞음

        if not hasattr(self, "base_model_name_or_path"):
            self.base_model_name_or_path = None
