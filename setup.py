from setuptools import setup

# peft_lava 루트 디렉토리를 "peft" 패키지로 매핑
# 모든 하위 패키지를 명시적으로 정의
packages = [
    "peft",
    "peft.tuners",
    "peft.tuners.lokr",
    "peft.tuners.xlora",
    "peft.tuners.hra",
    "peft.tuners.boft",
    "peft.tuners.boft.fbd",
    "peft.tuners.multitask_prompt_tuning",
    "peft.tuners.bone",
    "peft.tuners.shira",
    "peft.tuners.ia3",
    "peft.tuners.prefix_tuning",
    "peft.tuners.mixed",
    "peft.tuners.vera",
    "peft.tuners.adaption_prompt",
    "peft.tuners.loha",
    "peft.tuners.ln_tuning",
    "peft.tuners.poly",
    "peft.tuners.lora",
    "peft.tuners.p_tuning",
    "peft.tuners.trainable_tokens",
    "peft.tuners.fourierft",
    "peft.tuners.c3a",
    "peft.tuners.randlora",
    "peft.tuners.cpt",
    "peft.tuners.prompt_tuning",
    "peft.tuners.moca",
    "peft.tuners.lava",
    "peft.tuners.miss",
    "peft.tuners.oft",
    "peft.tuners.adalora",
    "peft.tuners.vblora",
    "peft.utils",
    "peft.optimizers",
]

# 패키지 디렉토리 매핑: "peft" -> 현재 디렉토리 (peft_lava/)
package_dir = {"peft": "."}

# 하위 패키지도 올바른 경로로 매핑
for pkg in packages:
    if pkg != "peft":
        # "peft.tuners.lora" -> "tuners/lora"
        subpath = pkg.replace("peft.", "").replace(".", "/")
        package_dir[pkg] = subpath

setup(
    name="peft",
    version="0.1.0",
    packages=packages,
    package_dir=package_dir,
    install_requires=[
        "numpy",
        "packaging",
        "psutil",
        "pyyaml",
        "tqdm",
        "safetensors",
        "accelerate",
        "huggingface_hub",
    ],
    python_requires=">=3.10",
)
