//
// Created by Bujor Ionut Raul on 17.02.2026.
//

#ifndef NEUROCORE_TENSOR_OPS_CUH
#define NEUROCORE_TENSOR_OPS_CUH


// to be done
// - reshape/view, flatten, squeeze, unsqueeze
// - concatenate/cat, stack, split
// - expand, repeat
// - slice, gather, scatter
// - swapaxes, flip



#include <memory>
#include <utility>
#include <vector>
#include "base.cuh"
#include "core/tensor.cuh"


template <typename dtype>
class TransposeFunction : public Function<dtype> {
private:
    std::vector<int> perm;
    std::vector<int> invPerm;
public:
    explicit TransposeFunction(std::shared_ptr<TensorImpl<dtype>> a, std::vector<int> perm)
        : Function<dtype>({std::move(a)}), perm(std::move(perm)) {}

    NDArray<dtype> forward() override {
        return this->parents[0]->data.transpose(perm);
    }

    void backward(const NDArray<dtype> &gradOutput) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            // Calculate inverse permutation
            if (invPerm.empty()) {
                invPerm.resize(perm.size());
                for (int i = 0; i < perm.size(); ++i) {
                    invPerm[perm[i]] = i;
                }
            }
            NDArray<dtype> aGrad = gradOutput.transpose(invPerm);
            a->accumulateGrad(aGrad);
        }
    }
};


template <typename dtype>
class ReshapeFunction : public Function<dtype> {
private:
    const Shape origShape;
    const Shape newShape;
public:
    explicit ReshapeFunction(std::shared_ptr<TensorImpl<dtype>> a, const Shape &newShape)
        : Function<dtype>({std::move(a)}), origShape(origShape), newShape(newShape) {}

    NDArray<dtype> forward() override {
        return this->parents[0]->data.reshape(newShape);
    }

    void backward(const NDArray<dtype> &gradOutput) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            NDArray<dtype> aGrad = gradOutput.reshape(origShape);
            a->accumulateGrad(aGrad);
        }
    }
};



namespace tensor {
    template <typename dtype>
    Tensor<dtype> transpose(const Tensor<dtype> &tensor, const std::vector<int> &perm) {
        auto fn = std::make_shared<TransposeFunction<dtype>>(tensor.getImpl(), perm);
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), tensor.requiresGrad());
        if (tensor.requiresGrad()) {
            out.setGradFn(fn);
        }
        return out;
    }
    template <typename dtype>
    Tensor<dtype> reshape(const Tensor<dtype> &tensor, const Shape &newShape) {
        auto fn = std::make_shared<ReshapeFunction<dtype>>{tensor.getImpl(), tensor.shape(), newShape};
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), tensor.requiresGrad());
        if (tensor.requiresGrad()) {
            out.setGradFn(fn);
        }
        return out;
    }

    template <typename dtype>
    Tensor<dtype> flatten(const Tensor<dtype> &tensor, int start, int end=-1) {
        NDArray<dtype> outData = tensor.getImpl().data.flatten(start, end);
        Tensor<dtype> out(std::move(outData), tensor.requiresGrad());
        auto fn = std::make_shared<ReshapeFunction<dtype>>(tensor.getImpl(), tensor.shape(), out.shape());
        if (tensor.requiresGrad()) {
            out.setGradFn(fn);
        }
        return out;
    }

    template <typename dtype>
    Tensor<dtype> squeeze(const Tensor<dtype> &tensor, const std::vector<int> &axes) {
        NDArray<dtype> outData = tensor.getImpl().data.squeeze(axes);
        Tensor<dtype> out(std::move(outData), tensor.requiresGrad());
        auto fn = std::make_shared<ReshapeFunction<dtype>>(tensor.getImpl(), tensor.shape(), out.shape());
        if (tensor.requiresGrad()) {
            out.setGradFn(fn);
        }
        return out;
    }
    template <typename dtype>
    Tensor<dtype> squeeze(const Tensor<dtype> &tensor, int axis) {
        return squeeze(tensor, {axis});
    }

    template <typename dtype>
    Tensor<dtype> expandDims(const Tensor<dtype> &tensor, const std::vector<int> &axes) {
        NDArray<dtype> outData = tensor.getImpl().data.expandDims(axes);
        Tensor<dtype> out(std::move(outData), tensor.requiresGrad());
        auto fn = std::make_shared<ReshapeFunction<dtype>>(tensor.getImpl(), tensor.shape(), out.shape());
        if (tensor.requiresGrad()) {
            out.setGradFn(fn);
        }
        return out;
    }
    template <typename dtype>
    Tensor<dtype> expandDims(const Tensor<dtype> &tensor, int axis) {
        return expandDims(tensor, {axis});
    }
}



#endif //NEUROCORE_TENSOR_OPS_CUH