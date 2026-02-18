#ifndef NEUROCORE_BASE_CUH
#define NEUROCORE_BASE_CUH

#include <memory>
#include <utility>
#include <vector>
#include "core/ndarray.cuh"
#include "functions/grad_kernels.cuh"

template <typename dtype>
struct TensorImpl;


template <typename dtype>
class Function {
protected:
    std::vector<std::shared_ptr<TensorImpl<dtype>>> parents;
public:
    explicit Function(std::vector<std::shared_ptr<TensorImpl<dtype>>> parents)
        : parents(std::move(parents)) {}
    virtual ~Function() = default;
    const std::vector<std::shared_ptr<TensorImpl<dtype>>> &getParents() const {
        return parents;
    }
    virtual NDArray<dtype> forward() = 0;
    virtual void backward(const NDArray<dtype> &outGrad) = 0;
};


template <typename dtype, typename Op>
void executeGrad(
    Op op,
    const NDArray<dtype> &outGrad,
    const NDArray<dtype> &a,
    NDArray<dtype> *aGrad = nullptr,
    const NDArray<dtype> *b = nullptr,
    NDArray<dtype> *bGrad = nullptr
) {
    int size = outGrad.getSize();
    int nBlocks = outGrad.getNBLOCKS();
    int nThreads = outGrad.getNTHREADS();
    bool allContig = outGrad.isContiguous() & a.isContiguous();
    if (aGrad) allContig &= aGrad->isContiguous();
    if (b) allContig &= b->isContiguous();
    if (bGrad) allContig &= bGrad->isContiguous();

    if (allContig) {
        gradKernelContiguous<dtype, Op><<<nBlocks, nThreads>>>(
            op, size,
            outGrad.getData(), outGrad.getOffset(),
            a.getData(), a.getOffset(),
            aGrad? aGrad->getData() : nullptr, aGrad? aGrad->getOffset() : 0,
            b? b->getData() : nullptr, b? b->getOffset() : 0,
            bGrad? bGrad->getData() : nullptr, bGrad? bGrad->getOffset() : 0
        );
    } else {
        int *dShape = nullptr;
        int *dAStrides = nullptr, *dBStrides = nullptr;
        int *dAGradStrides = nullptr, *dBGradStrides = nullptr, *dOutGradStrides = nullptr;
        try {
            outGrad.allocateDeviceMetadata(&dOutGradStrides, &dShape);
            a.allocateDeviceMetadata(&dAStrides, nullptr);
            if (b) b->allocateDeviceMetadata(&dBStrides, nullptr);
            if (aGrad) aGrad->allocateDeviceMetadata(&dAGradStrides, nullptr);
            if (bGrad) bGrad->allocateDeviceMetadata(&dBGradStrides, nullptr);
            gradKernelStrided<dtype, Op><<<nBlocks, nThreads>>>(
                op, size, outGrad.getNDim(), dShape,
                outGrad.getData(), outGrad.getOffset(), dOutGradStrides,
                a.getData(), a.getOffset(), dAStrides,
                b? b->getData() : nullptr, b? b->getOffset() : 0, dBStrides,
                aGrad? aGrad->getData() : nullptr, aGrad? aGrad->getOffset() : 0, dAGradStrides,
                bGrad? bGrad->getData() : nullptr, bGrad? bGrad->getOffset() : 0, dBGradStrides
            );
        } catch (...) {
            utils::cudaFreeMulti({dShape, dOutGradStrides,
                dAStrides, dAGradStrides, dBStrides, dBGradStrides
            });
            throw;
        }
        utils::cudaFreeMulti({dShape, dOutGradStrides,
               dAStrides, dAGradStrides, dBStrides, dBGradStrides
        });
    }
}


template <typename dtype>
NDArray<dtype> reduceGradToShape(const NDArray<dtype> &grad,
                                 const std::vector<int> &targetShape) {
    const auto gradShape = grad.getShape();
    if (gradShape == targetShape) return grad;

    if (gradShape.size() != targetShape.size()) {
        throw ShapeMismatchException("Gradient reduction target has different ndim.");
    }
    for (size_t i = 0; i < gradShape.size(); i++) {
        if (targetShape[i] != gradShape[i] && targetShape[i] != 1) {
            throw ShapeMismatchException("Gradient reduction target is not broadcast-compatible.");
        }
    }

    // TODO: Implement proper broadcasting gradient reduction
    // custom reduction CUDA Kernel may be needed here
    throw ShapeMismatchException("Gradient reduction to target shape not implemented.");
}

#endif //NEUROCORE_BASE_CUH
