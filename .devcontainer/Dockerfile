FROM mcr.microsoft.com/devcontainers/base:resolute

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config git curl vim clangd glslang-tools \
    libvulkan-dev vulkan-validationlayers mesa-vulkan-drivers vulkan-tools \
    libwayland-dev libwayland-bin wayland-protocols \
    libxkbcommon-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
    libxxf86vm-dev libgl1-mesa-dev

RUN curl -fsSL https://opencode.ai/install | bash
