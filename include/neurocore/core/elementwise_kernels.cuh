//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef NEUROCORE_ELEMENTWISE_KERNELS_CUH
#define NEUROCORE_ELEMENTWISE_KERNELS_CUH


/// Refactored templates for element-wise FORWARD operations
/// 2 variants: contiguous and strided (views) arrays
/// Supports different input/output types (e.g., for casting)
/// Utilizes grid-stride loops to decouple grid launch size
/// from tensor size when hardware-capped

template<typename OutDtype, typename InDtype, typename Op>
__global__ void elementWiseKernelContiguous(
    OutDtype *output, const int outOffset,
    const int size,
    Op op,
    const InDtype *a, const int aOffset,
    const InDtype *b = nullptr, const int bOffset = 0)
{
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += blockDim.x * gridDim.x)
    {
        output[outOffset + idx] = op(
            a[aOffset + idx],
            b ? b[bOffset + idx]: InDtype(0)
        );
    }
}


template<typename OutDtype, typename InDtype, typename Op>
__global__ void elementWiseKernelStrided(
    OutDtype *output, const int outOffset, const int *outStrides,
    const int size, const int ndim, const int *shape,
    Op op,
    const InDtype *a, const int aOffset, const int *aStrides,
    const InDtype *b = nullptr, const int bOffset = 0, const int *bStrides = nullptr
)
{
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < size;
         idx += blockDim.x * gridDim.x)
    {
        // decompose flat idx into per-dim coords, accumulate strided offsets in one pass
        size_t multiIdx, remaining = idx;
        size_t outIdx = outOffset, aIdx = aOffset, bIdx = bOffset;

        for (int i = ndim - 1; i >= 0; i--) {
            multiIdx = remaining % shape[i];  // coordinate along dim i
            remaining /= shape[i];
            outIdx += multiIdx * outStrides[i];
            aIdx   += multiIdx * aStrides[i];
            if (bStrides) bIdx += multiIdx * bStrides[i];  // skip if unary op
        }

        output[outIdx] = op(
            a[aIdx],
            b ? b[bIdx] : InDtype(0)
        );
    }
}


template <typename dtype>
__global__ void gatherKernel(
    dtype *dst,
    const dtype *src,
    const int *indices,    // separable index arrays, concatenated per dim
    const int nIndices,    // total length of indices (sum of dstShape)
    const int nDim,        // same for dst and src
    const int dstSize,
    const int *dstShape,
    const int *srcStrides,
    const int srcOffset
)
{
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < dstSize;
         idx += blockDim.x * gridDim.x)
    {
        size_t multiIdx, remaining = idx;
        size_t indicesIdx, srcIdx = srcOffset, runningSum = nIndices;

        // decompose flat idx into per-dim coordinates, accumulate src offset
        for (int i = nDim - 1; i >= 0; i--) {
            multiIdx = remaining % dstShape[i];  // coordinate along dim i
            remaining /= dstShape[i];
            runningSum -= dstShape[i];           // start of dim i's index block
            indicesIdx = runningSum + multiIdx;  // lookup into concatenated indices
            srcIdx += indices[indicesIdx] * srcStrides[i];
        }

        dst[idx] = src[srcIdx];
    }
}


/// FUNCTORS
template <typename newDtype, typename dtype>
struct CastOp {
    __device__ newDtype operator()(dtype a, dtype) const {
        return (newDtype)a;
    }
};
template <typename dtype>
struct SetConstantOp {
    dtype value;
    __device__ dtype operator()(dtype, dtype) const {
        return value;
    }
};
template <typename dtype>
struct AssignOp {
    __device__ dtype operator()(dtype a, dtype b) const {
        return b;
    }
};

template <typename dtype>
struct ScalarAddOp {
    dtype scalar;
    __device__ dtype operator()(dtype a, dtype) const {
        return a + scalar;
    }
};

template <typename dtype>
struct ScalarMulOp {
    dtype scalar;
    __device__ dtype operator()(dtype a, dtype) const {
        return a * scalar;
    }
};

template <typename dtype>
struct ScalarRSubOp {
    dtype scalar;
    __device__ dtype operator()(dtype a, dtype) const {
        return scalar - a;
    }
};

template <typename dtype>
struct ScalarRDivOp {
    dtype scalar;
    __device__ dtype operator()(dtype a, dtype) const {
        return scalar / a;
    }
};

template <typename dtype>
struct AffineAddOp {
    dtype alpha, beta;
    __device__ dtype operator()(dtype a, dtype b) const {
        return alpha * a + beta * b;
    }
};

template <typename dtype>
struct MulOp {
    dtype scalar = 1;
    __device__ dtype operator()(dtype a, dtype b) const {
        return scalar * a * b;
    }
};

template <typename dtype>
struct DivOp {
    dtype scalar = 1;
    __device__ dtype operator()(dtype a, dtype b) const {
        return scalar * a / b;
    }
};

template <typename dtype>
struct PowOp {
    __device__ dtype operator()(dtype a, dtype b) const {
        return std::powf(a, b);
    }
};

template <typename dtype>
struct RaiseOp {
    dtype exponent;
    __device__ dtype operator()(dtype a, dtype) const {
        return std::powf(a, exponent);
    }
};


template <typename dtype>
struct ExpOp {
    dtype base;
    __device__ dtype operator()(dtype a, dtype) const{
        return std::powf(base, a);
    }
};

template <typename dtype>
struct LogOp {
    dtype base;
    __device__ dtype operator()(dtype a, dtype) const {
        return std::logf(a) / std::logf(base);
    }
};


template <typename dtype>
struct SinOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return std::sinf(a);
    }
};

template <typename dtype>
struct CosOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return std::cosf(a);
    }
};

template <typename dtype>
struct TanOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return std::tanf(a);
    }
};

template <typename dtype>
struct CotOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return 1 / std::tanf(a);
    }
};

template <typename dtype>
struct ASinOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return std::asinf(a);
    }
};

template <typename dtype>
struct ACosOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return std::acosf(a);
    }
};

template <typename dtype>
struct ATanOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return std::atanf(a);
    }
};

template <typename dtype>
struct ACotOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return std::atan2f(1.0, a);
    }
};

template <typename dtype>
struct SigmoidOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return 1 / (1 + std::expf(-a));
    }
};

template <typename dtype>
struct AbsOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return std::abs(a);
    }
};

template <typename dtype>
struct SignOp {
    __device__ dtype operator()(dtype a, dtype) const {
        return (dtype)(a > 0) - (dtype)(a < 0);
    }
};

template <typename dtype>
struct ClipOp {
    dtype low, high;
    __device__ dtype operator()(dtype a, dtype) const {
        return max(low, min(high, a));
    }
};

#endif //NEUROCORE_ELEMENTWISE_KERNELS_CUH