#!/bin/bash
clear
export PYTHONPATH=$PYTHONPATH:/workspace

echo -e "\n------------------------------------------------------------------------------------\n"
figlet -c "GaussianCaR"
echo -e "\n------------------------------------ System info -----------------------------------\n"

# Update uv project installation
echo "🔄 Checking virtual environment..."
if [ ! -d ".venv" ]; then
  echo " .venv not found — updating virtual environment..."

  # Run uv sync and capture output
  if ! uv sync --link-mode=copy; then
    echo "❌ uv sync failed, cleaning up..."
    rm -rf .venv
    exit 1
  fi

  echo "✅ .venv updated"
else
  echo "✅ .venv found"
fi

# Get current username and store in a local variable.
CURRENT_USER=$(whoami)
PATH_TO_NUSCENES="/data/nuscenes"

# Check if nuScenes dataset is available.
echo -e "\n🔍 Checking datasets availability..."
if [ ! -d "$PATH_TO_NUSCENES/samples" ] || [ ! -d "$PATH_TO_NUSCENES/sweeps" ]; then
    echo -e "❌ \033[91m\033[1mnuScenes dataset not found or incomplete...\033[0m"
    echo -e "   Please download the dataset and share it in a volume at $PATH_TO_NUSCENES."
else
    echo -e "✅ \033[92m\033[1mnuScenes dataset at\033[0m $PATH_TO_NUSCENES\033[92m\033[1m found!\033[0m"
fi

# Check if CUDA is available
echo -e "\n🔍 Checking GPU and CUDA availability..."
if ! uv run --preview-features extra-build-dependencies python -c "import torch" 2>/dev/null; then
    echo -e "❌ \033[91m\033[1mFailed to import torch\033[0m"
    echo -e "   Please check your PyTorch installation!"
else

    CUDA_AVAILABLE=$(uv run --preview-features extra-build-dependencies python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" == "True" ]; then
        echo -e "✅ \033[92m\033[1mPyTorch is working properly with the GPU.\033[0m"
        echo -e "📍 GPU Information:"
        uv run --preview-features extra-build-dependencies python -c "import torch; print(f'   - CUDA version:     {torch.version.cuda}')"
        uv run --preview-features extra-build-dependencies python -c "import torch; print(f'   - Device name:      {torch.cuda.get_device_name(0)}')"
        uv run --preview-features extra-build-dependencies python -c "import torch; print(f'   - Number of GPUs:   {torch.cuda.device_count()}')"
    else
        echo -e "❌ \033[91m\033[1mCUDA is not available!\033[0m"
        echo -e "   Check your PyTorch installation"
    fi
fi

echo -e "\n------------------------------------------------------------------------------------\n"