CUDA_ARCH="75;80;86;89;90"

if [ ! -z "$1" ]; then
    CUDA_ARCH="$1"
fi

echo "installing PRISM..."
cd PRISM
mkdir build 
cd build
cmake .. -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
make -j
cd ../..

echo "installing cuSZHi..."
cp maxerr.thrust.inl cuSZ-Hi/src/stat/detail/maxerr.thrust.inl # for compatibility with newer CUDA versions
cp compare.thrust.inl cuSZ-Hi/src/stat/detail/compare.thrust.inl # for compatibility with newer CUDA versions
cd cuSZ-Hi
mkdir build 
cd build
cmake .. -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
make -j
cd ../..

echo "installing cuSZp2..."
cd cuSZp
git checkout 671d5f438f452f30192d333f206f3caa742f6350
cd ..
cmake -S cuSZp -B cuSZp/build \
    -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -D CMAKE_BUILD_TYPE=Release
cmake --build cuSZp/build -- -j

echo "installing cuZFP..."
cp shared_time.h cuZFP/src/cuda_zfp/shared.h # for printing time info
cmake -S cuZFP -B cuZFP/build \
    -D ZFP_WITH_CUDA=on \
    -D CUDA_SDK_ROOT_DIR=$(dirname $(which nvcc))/.. \
    -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cuZFP/build -- -j

echo "installing cuSZ..."
cmake -S cuSZ -B cuSZ/build \
    -D PSZ_BACKEND=cuda \
    -D PSZ_BUILD_EXAMPLES=off \
    -D CMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -D CMAKE_BUILD_TYPE=Release 
cmake --build cuSZ/build -- -j

echo "installing HP-MDR..."
bash install_mdr.sh $1
