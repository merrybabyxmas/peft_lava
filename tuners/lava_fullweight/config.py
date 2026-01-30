from peft.config import PeftConfig


class LavaFullWeightConfig(PeftConfig):
    def __init__(
        self,
        r=8,
        alpha=4,
        target_modules=None,
        bias="none",
        inference_mode=False,
        **kwargs
    ):
        self.modules_to_save = []

        super().__init__(**kwargs)

        self.peft_type = "LAVA_FULLWEIGHT"

        self.r = r
        self.alpha = alpha
        self.target_modules = target_modules
        self.bias = bias
        self.inference_mode = inference_mode

        if not hasattr(self, "task_type") or self.task_type is None:
            self.task_type = kwargs.get("task_type", "SEQ_CLS")

        if not hasattr(self, "base_model_name_or_path"):
            self.base_model_name_or_path = None
