#!/bin/bash
set -e

echo "🚀 Starting PEFT_LAVA installation..."

# 1. 환경 생성 및 활성화
conda create -n lava python=3.10.19 -y || echo "Environment already exists"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lava

# 2. RTX 5090 호환성 우선 설치 (가장 중요)
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 2>/dev/null || echo "No GPU")
if echo "$GPU_INFO" | grep -q "RTX 5090\|RTX 6000 Ada\|RTX 6090"; then
    echo "✨ Blackwell GPU detected - Installing PyTorch Nightly..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# 3. 패키지 설치 (중첩 로직 없이 현재 위치에서 바로 설치)
pip install -e .

# 4. 심볼릭 링크 (Conda 환경에서 peft를 바로 인식하도록 설정)
ENV_PATH=$(conda info --base)/envs/lava/lib/python3.10/site-packages
rm -rf "$ENV_PATH/peft"
ln -sf $(pwd)/peft "$ENV_PATH/peft"

echo "✅ Setup complete! Use 'conda activate lava'"