#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'
RED='\033[0;31m'
YELLOW='\033[1;33m'

get_platform() {
    if command -v nvidia-smi > /dev/null; then
        echo "cuda"
    elif command -v rocminfo > /dev/null; then
        echo "hip"
    elif lspci | grep -i "Intel" | grep -i "VGA\|3D\|Display" > /dev/null; then
        echo "intel"
    else
        echo "cpu"
    fi
}

install_dependencies() {
    local PLATFORM=$(get_platform)

    echo -e "${BLUE}===========================================================================================${NC}"
    echo -e "${GREEN}Updating core build tools...${NC}"
    echo -e "${BLUE}===========================================================================================${NC}"
    pip install --upgrade pip setuptools wheel typing-extensions

    echo -e "${BLUE}===========================================================================================${NC}"
    echo -e -n "${GREEN}Do you want to install nightly build of pytorch? (y/[n]) ${NC}"
    read -r choice
    echo -e "${BLUE}===========================================================================================${NC}"

    if [[ $choice == "y" || $choice == "Y" ]]; then
        if [ "$PLATFORM" = "cuda" ]; then
            pip --no-cache-dir install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
        elif [ "$PLATFORM" = "hip" ]; then
            pip --no-cache-dir install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.1
        elif [ "$PLATFORM" = "intel" ]; then
            pip --no-cache-dir install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
        else
            pip --no-cache-dir install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
        fi
    else
        if [ "$PLATFORM" = "cuda" ]; then
            pip --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
        elif [ "$PLATFORM" = "hip" ]; then
            pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
        elif [ "$PLATFORM" = "intel" ]; then
            pip --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
        else
            pip --no-cache-dir install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        fi
    fi

    grep -E -v "^(torch|torchvision|torchaudio)([<>=~]|$)" requirements.txt > requirements_temp.txt

    pip install -r requirements_temp.txt

    rm requirements_temp.txt
}

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo -e "${BLUE}===========================================================================================${NC}"
    echo -e "${GREEN}Creating new virtual environment...${NC}"
    echo -e "${BLUE}===========================================================================================${NC}"
    if python -m venv venv; then
        source venv/bin/activate
        install_dependencies

        echo -e "${BLUE}===========================================================================================${NC}"
        echo -e "${GREEN}Setup completed.${NC}"
        echo -e "${GREEN}To run ComfyUI, you must activate the environment manually:${NC}"
        echo -e "${GREEN}source venv/bin/activate${NC}"
        echo -e "${BLUE}===========================================================================================${NC}"
    else
        echo -e "${BLUE}===========================================================================================${NC}"
        echo -e "${RED}Error creating venv.${NC}"
        echo -e "${BLUE}===========================================================================================${NC}"
        exit 1
    fi
else
    VENV_NAME=$(echo "$VIRTUAL_ENV" | sed -E 's#/.*/##g')
    echo -e "${BLUE}===========================================================================================${NC}"
    echo -e "${YELLOW}Virtual environment $VENV_NAME is active, a new venv will be created${NC}"
    echo -e "${BLUE}===========================================================================================${NC}"

    python -m venv venv
    source venv/bin/activate

    install_dependencies

    echo -e "${BLUE}===========================================================================================${NC}"
    echo -e "${GREEN}Setup completed.${NC}"
    echo -e "${GREEN}To run ComfyUI, you must activate the environment manually:${NC}"
    echo -e "${GREEN}source venv/bin/activate${NC}"
    echo -e "${BLUE}===========================================================================================${NC}"
fi