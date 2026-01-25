#!/bin/bash
set -e

echo "========================================"
echo " PEFT_LAVA Installation"
echo "========================================"

# Conda 활성화 함수
init_conda() {
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    else
        echo "Error: Conda not found. Please install Miniconda or Anaconda first."
        exit 1
    fi
}

init_conda

# 1. 환경 생성 확인
if conda env list | grep -q "^lava "; then
    echo "[OK] Conda environment 'lava' already exists."
else
    echo "[..] Creating conda environment 'lava' with Python 3.10..."
    conda create -n lava python=3.10 -y
fi

# 환경 활성화
conda activate lava
echo "[OK] Activated 'lava' environment"

# 2. PyTorch 설치 여부 확인
if python -c "import torch; print(torch.__version__)" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    echo "[OK] PyTorch $TORCH_VER is already installed."
else
    # GPU 감지
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "No GPU")
    echo "[..] Detected GPU: $GPU_INFO"

    if echo "$GPU_INFO" | grep -qE "RTX 50|Blackwell"; then
        echo "[..] Blackwell/RTX 50 series detected - Installing PyTorch Nightly (cu128)..."
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    else
        echo "[..] Installing PyTorch (cu124)..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    fi
fi

# 3. 기존 peft 제거 (충돌 방지)
if pip show peft &>/dev/null; then
    PEFT_LOC=$(pip show peft | grep "Editable project location" || echo "")
    if echo "$PEFT_LOC" | grep -q "peft_lava"; then
        echo "[OK] Custom PEFT_LAVA is already installed."
    else
        echo "[..] Removing existing peft package to avoid conflicts..."
        pip uninstall peft -y
        echo "[..] Installing PEFT_LAVA in editable mode..."
        pip install -e .
    fi
else
    echo "[..] Installing PEFT_LAVA in editable mode..."
    pip install -e .
fi

# 4. 설치 확인
echo ""
echo "========================================"
echo " Verifying Installation"
echo "========================================"
python -c "
import peft
print(f'[OK] peft version: {peft.__version__}')
print(f'[OK] peft location: {peft.__file__}')
"

echo ""
echo "========================================"
echo " PEFT_LAVA Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  cd ../lava"
echo "  pip install -r requirements.txt"
echo ""
