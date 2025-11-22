git submodule init
git submodule update

CUDA_ARCH="75;80;86;89;90"

if [ ! -z "$1" ]; then
    CUDA_ARCH="$1"
fi

echo "\nsetting up cuSZp...\n"
cmake -S cuSZp -B cuSZp/build \
    -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -D CMAKE_BUILD_TYPE=Release
cmake --build cuSZp/build -- -j

echo "\nsetting up cuZFP...\n"
cmake -S cuZFP -B cuZFP/build \
    -D ZFP_WITH_CUDA=on \
    -D CUDA_SDK_ROOT_DIR=$(dirname $(which nvcc))/.. \
    -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cuZFP/build -- -j

cmake -S cuSZ -B cuSZ/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cuSZ/build -- -j