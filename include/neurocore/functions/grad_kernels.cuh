//
// Created by Bujor Ionut Raul on 17.02.2026.
//

#ifndef NEUROCORE_GRAD_KERNELS_CUH
#define NEUROCORE_GRAD_KERNELS_CUH

#include <cmath>
#include <cuda_runtime.h>

/// GENERIC KERNELS FOR GRADIENT OPERATIONS
/// Handles: 3 Inputs (a, b, gradOutput) -> 2 Outputs (gradA, gradB)

template<typename dtype, typename Op>
__global__ void gradKernelContiguous(
    Op op, const int size,
    const dtype *outGrad, const int outGradOffset,
    const dtype *a, const int aOffset,
    dtype *aGrad = nullptr, const int aGradOffset = 0, // optional by grad
    const dtype *b = nullptr, const int bOffset = 0, // optional by arity
    dtype *bGrad = nullptr, const int bGradOffset = 0  // optional by grad
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    op(
       outGrad[outGradOffset + idx],
       a[aOffset + idx],
       aGrad ? aGrad + aGradOffset + idx : nullptr,
       b ? b[bOffset + idx] : dtype(0),
       bGrad ? bGrad + bGradOffset + idx : nullptr
    );
}



template<typename dtype, typename Op>
__global__ void gradKernelStrided(
    Op op, const int size, const int ndim, const int *shape,
    const dtype *outGrad, const int outGradOffset, const int *outGradStrides,
    const dtype *a, const int aOffset, const int *aStrides,
    dtype *aGrad = nullptr, const int aGradOffset = 0, const int *aGradStrides = nullptr,
    const dtype *b = nullptr, const int bOffset = 0, const int *bStrides = nullptr,
    dtype *bGrad = nullptr, const int bGradOffset = 0, const int *bGradStrides = nullptr
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // decompose flat idx into per-dim coords, accumulate strided offsets in one pass
    int multiIdx, remaining = idx;
    int outGradIdx = outGradOffset, aIdx = aOffset, bIdx = bOffset;
    int aGradIdx = aGradOffset, bGradIdx = bGradOffset;
    for (int i = ndim - 1; i >= 0; i--) {
        multiIdx = remaining % shape[i];
        remaining /= shape[i];
        outGradIdx += multiIdx * outGradStrides[i];
        aIdx += multiIdx * aStrides[i];
        if (aGrad) aGradIdx += multiIdx * aGradStrides[i];
        if (b) bIdx += multiIdx * bStrides[i];
        if (bGrad) bGradIdx += multiIdx * bGradStrides[i];
    }
    op(
        outGrad[outGradIdx],
        a[aIdx],
        aGrad ? aGrad + aGradIdx : nullptr,
        b ? b[bIdx]: dtype(0),
        bGrad ? bGrad + bGradIdx : nullptr
    );
}


/// GRADIENT FUNCTORS
///
template <typename dtype>
struct DivGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype b, dtype *bGrad
    ) const {
        if (aGrad) *aGrad += outGrad / b;
        if (bGrad) *bGrad += -outGrad * a / (b * b);
    }
};

template <typename dtype>
struct PowGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype b, dtype *bGrad
    ) const {
        if (aGrad) *aGrad += outGrad * b * pow(a, b - 1);
        if (bGrad) *bGrad += outGrad * log(a) * pow(a, b - 1);
    }
};
///
template <typename dtype>
struct ExpGradOp {
    dtype base = static_cast<dtype>(std::exp(1.0));
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += outGrad * std::logf(base) * std::powf(base, a);
    }
};

template <typename dtype>
struct LogGradOp {
    dtype base = static_cast<dtype>(std::exp(1.0));
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += outGrad / a / std::logf(base);
    }
};

template <typename dtype>
struct RaiseGradOp {
    dtype power;
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += outGrad * power * powf(a, power - 1);
    }
};

template <typename dtype>
struct AbsGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += outGrad * (a > 0 ? 1 : (a < 0 ? -1 : 0));
    }
};

template <typename dtype>
struct ClipGradOp {
    dtype low, high;
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad && a >= low && a <= high) *aGrad += outGrad;
    }
};

template <typename dtype>
struct SinGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += outGrad * cosf(a);
    }
};

template <typename dtype>
struct CosGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += -outGrad * sinf(a);
    }
};

template <typename dtype>
struct TanGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += outGrad * 1 / (cosf(a) * cosf(a));
    }
};

template <typename dtype>
struct CotGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += -outGrad * 1 / (sinf(a) * sinf(a));
    }
};


template <typename dtype>
struct ASinGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += outGrad * 1 / sqrtf(1 - a * a);
    }
};

template <typename dtype>
struct ACosGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += -outGrad * 1 / sqrtf(1 - a * a);
    }
};

template <typename dtype>
struct ATanGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += outGrad * 1 / (1 + a * a);
    }
};

template <typename dtype>
struct ACotGradOp {
    __device__ void operator()(
        dtype outGrad,
        dtype a, dtype *aGrad,
        dtype, dtype *
    ) const {
        if (aGrad) *aGrad += -outGrad * 1 / (1 + a * a);
    }
};


#endif //NEUROCORE_GRAD_KERNELS_CUH
