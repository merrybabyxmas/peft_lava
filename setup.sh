#!/bin/bash
set -e

echo "🚀 Starting PEFT_LAVA installation with check-and-skip logic..."

# 1. 환경 생성 확인
if conda env list | grep -q "lava"; then
    echo "✅ Conda environment 'lava' already exists. Skipping creation."
else
    echo "Creating conda environment 'lava'..."
    conda create -n lava python=3.10.19 -y
fi

# 환경 활성화
source $(conda info --base)/etc/profile.d/conda.sh
conda activate lava

# 2. PyTorch 설치 여부 확인 (특히 RTX 5090용 cu128 확인)
if python -c "import torch; print(torch.__version__)" &>/dev/null; then
    echo "✅ PyTorch is already installed. Skipping PyTorch installation."
else
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1 2>/dev/null || echo "No GPU")
    if echo "$GPU_INFO" | grep -q "RTX 5090\|RTX 6000 Ada\|RTX 6090"; then
        echo "✨ Blackwell GPU detected - Installing PyTorch Nightly (cu128)..."
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    else
        echo "Installing standard PyTorch (cu124)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    fi
fi

# 3. PEFT_LAVA 패키지 설치 여부 확인
if pip show peft &>/dev/null; then
    echo "✅ PEFT package is already installed. Skipping 'pip install -e .'"
else
    echo "Installing PEFT_LAVA in editable mode..."
    pip install -e .
fi

# 4. 심볼릭 링크 설정 (강제 갱신으로 경로 무결성 유지)
ENV_PATH=$(conda info --base)/envs/lava/lib/python3.10/site-packages
echo "Setting up symbolic links..."
rm -rf "$ENV_PATH/peft"
ln -sf $(pwd)/peft "$ENV_PATH/peft"

echo "✅ PEFT_LAVA Setup complete!"