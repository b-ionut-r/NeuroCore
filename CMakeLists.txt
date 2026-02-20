cmake_minimum_required(VERSION 4.0)
project(NeuroCore LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(NeuroCore
        include/neurocore/core/ndarray.cuh
        include/neurocore/core/tensor.cuh
        include/neurocore/core/slices.h
        include/neurocore/core/exceptions.h
        include/neurocore/core/type_traits.cuh
        include/neurocore/core/utils.h
        include/neurocore/functions/base.cuh
        include/neurocore/functions/arithmetic.cuh
        include/neurocore/functions/math.cuh
        include/neurocore/functions/grad_kernels.cuh
        include/neurocore/optim/optimizer.h
        include/neurocore/optim/optim_kernels.cuh
        include/neurocore/optim/sgd.cuh
        include/neurocore/optim/rmsprop.cuh
        include/neurocore/optim/adam.cuh
        src/core/utils.cu
        src/core/slices.cu
        src/optim/sgd.cu
        src/optim/rmsprop.cu
        src/optim/adam.cu
        examples/demo_main.cu
        include/neurocore/core/elementwise_kernels.cuh
        include/neurocore/core/random.cuh
        include/neurocore/functions/tensor_ops.cuh
)


set_target_properties(NeuroCore PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(NeuroCore PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/neurocore
)

target_link_libraries(NeuroCore PRIVATE
        CUDA::curand
        CUDA::cudart_static
)
