#ifndef NEUROCORE_TENSOR_CUH
#define NEUROCORE_TENSOR_CUH

#include <memory>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include "core/exceptions.h"
#include "functions/base.cuh"
#include "core/ndarray.cuh"

template <typename dtype>
class Tensor;

namespace tensor {
    template <typename dtype>
    Tensor<dtype> transpose(const Tensor<dtype> &tensor, const std::vector<int> &perm);
}

template <typename dtype>
struct TensorImpl;

template <typename dtype>
struct TensorImpl {
    NDArray<dtype> data;
    NDArray<dtype> grad;
    bool hasGrad = false;
    bool requiresGrad = false;
    bool isLeaf = true;
    std::shared_ptr<Function<dtype>> gradFn;
    std::shared_ptr<void> castParent;
    void (*castAccumulate)(const std::shared_ptr<void> &, const NDArray<dtype> &) = nullptr;
    void (*castBackward)(const std::shared_ptr<void> &) = nullptr;
    size_t id = 0;
    static size_t idGen;

    TensorImpl() : id(++idGen) {}
    TensorImpl(const NDArray<dtype> &data, bool requiresGrad)
        : data(data), requiresGrad(requiresGrad), id(++idGen) {}
    TensorImpl(NDArray<dtype> &&data, bool requiresGrad)
        : data(std::move(data)), requiresGrad(requiresGrad), id(++idGen) {}

    void ensureGrad() {
        if (!hasGrad) {
            grad = data.zeros_like();
            hasGrad = true;
        }
    }
    void zeroGrad() {
        if (!requiresGrad) return;
        ensureGrad();
        grad = static_cast<dtype>(0);
    }
    void accumulateGrad(const NDArray<dtype> &gradIn) {
        if (!requiresGrad) return;
        if (!hasGrad) {
            grad = NDArray<dtype>(gradIn);
            hasGrad = true;
        } else {
            grad = grad + gradIn;
        }
    }
};

template <typename dtype>
size_t TensorImpl<dtype>::idGen = 0;

template <typename dtype>
class Tensor {
private:
    std::shared_ptr<TensorImpl<dtype>> impl;
    static void backwardFromImpl(const std::shared_ptr<TensorImpl<dtype>> &root, bool seedOnes);
public:
    using value_type = dtype;
    Tensor()
        : impl(std::make_shared<TensorImpl<dtype>>()) {}
    Tensor(const Shape &shape, bool requiresGrad = false):
        impl(std::make_shared<TensorImpl<dtype>>(NDArray<dtype>(shape), requiresGrad)) {}
    Tensor(const NDArray<dtype> &data, bool requiresGrad = false)
        : impl(std::make_shared<TensorImpl<dtype>>(data, requiresGrad)) {}
    Tensor(const std::vector<dtype> &data, bool requiresGrad = false)
        : impl(std::make_shared<TensorImpl<dtype>>(NDArray<dtype>(data), requiresGrad)) {}
    Tensor(NDArray<dtype> &&data, bool requiresGrad = false)
        : impl(std::make_shared<TensorImpl<dtype>>(std::move(data), requiresGrad)) {}
    Tensor(const Tensor &rhs) = default;
    Tensor& operator=(const Tensor &rhs) = default;
    Tensor& operator=(const NDArray<dtype> &rhs) { impl->data = rhs; return *this; }
    Tensor& operator=(const std::vector<dtype> &rhs) { impl->data = NDArray<dtype>(rhs); return *this;}
    Tensor& operator=(dtype value){impl->data = value; return *this;}
    Tensor detach() const { return Tensor(impl->data, false);}
    ~Tensor() = default;
    NDArray<dtype> &data() { return impl->data; }
    const NDArray<dtype> &data() const { return impl->data; }
    NDArray<dtype> &grad() {
        if (!impl->requiresGrad) {
            throw BackPropException("Tensor does not require gradients.");
        }
        impl->ensureGrad();
        return impl->grad;
    }
    const NDArray<dtype> &grad() const {
        if (!impl->hasGrad) {
            throw BackPropException("Gradients not computed for this tensor.");
        }
        return impl->grad;
    }
    bool hasGrad() const { return impl->hasGrad; }
    bool requiresGrad() const { return impl->requiresGrad; }
    void setRequiresGrad(bool requiresGrad) { impl->requiresGrad = requiresGrad; }
    bool isLeaf() const { return impl->isLeaf; }
    std::vector<int> shape() const { return impl->data.getShape(); }
    int size() const { return impl->data.getSize(); }
    Tensor<dtype> transpose(const std::vector<int> &perm={}) const {
        return tensor::transpose(*this, perm);
    };
    void zeroGrad() { impl->zeroGrad(); }
    void backward() {
        if (!impl->requiresGrad) {
            throw BackPropException("Tensor does not require gradients.");
        }
        backwardFromImpl(impl, true);
    }

    Tensor<dtype> *get() { return this; }
    const Tensor<dtype> *get() const { return this; }

    std::shared_ptr<TensorImpl<dtype>> getImpl() const { return impl; }
    void setGradFn(const std::shared_ptr<Function<dtype>> &fn) {
        impl->gradFn = fn;
        impl->isLeaf = false;
    }

    template <typename newDtype>
    Tensor<newDtype> cast() const;
};



template <typename dtype>
template <typename newDtype>
Tensor<newDtype> Tensor<dtype>::cast() const {
    if constexpr (std::is_same_v<newDtype, dtype>) {
        return *this;
    }
    NDArray<newDtype> casted = impl->data.template cast<newDtype>();
    bool requiresGrad = impl->requiresGrad;
    Tensor<newDtype> out(std::move(casted), requiresGrad);
    if (requiresGrad) {
        out.getImpl()->castParent = impl;
        out.getImpl()->castAccumulate = [](const std::shared_ptr<void> &parentVoid,
                                           const NDArray<newDtype> &gradOut) {
            auto parent = std::static_pointer_cast<TensorImpl<dtype>>(parentVoid);
            parent->accumulateGrad(gradOut.template cast<dtype>());
        };
        out.getImpl()->castBackward = [](const std::shared_ptr<void> &parentVoid) {
            auto parent = std::static_pointer_cast<TensorImpl<dtype>>(parentVoid);
            Tensor<dtype>::backwardFromImpl(parent, false);
        };
    }
    return out;
}

template <typename dtype>
void Tensor<dtype>::backwardFromImpl(const std::shared_ptr<TensorImpl<dtype>> &root, bool seedOnes) {
    if (!root) return;
    if (seedOnes) root->accumulateGrad(root->data.ones_like());
    else if (!root->hasGrad) return;

    std::vector<std::shared_ptr<TensorImpl<dtype>>> topo;
    std::unordered_set<TensorImpl<dtype> *> visited;
    auto build = [&](auto &&self, const std::shared_ptr<TensorImpl<dtype>> &node) -> void {
        if (!node || visited.count(node.get())) return;
        visited.insert(node.get());
        if (node->gradFn) {
            for (const auto &parent : node->gradFn->getParents()) {
                self(self, parent);
            }
        }
        topo.push_back(node);
    };
    build(build, root);

    std::vector<std::pair<std::shared_ptr<void>, void (*)(const std::shared_ptr<void> &)>> castParents;
    std::unordered_set<void *> castParentSeen;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const auto &node = *it;
        if (node->gradFn && node->hasGrad) node->gradFn->backward(node->grad);
        if (node->castAccumulate && node->hasGrad) {
            node->castAccumulate(node->castParent, node->grad);
            if (node->castBackward && node->castParent) {
                void *key = node->castParent.get();
                if (castParentSeen.insert(key).second) {
                    castParents.emplace_back(node->castParent, node->castBackward);
                }
            }
        }
    }

    for (const auto &entry : castParents) entry.second(entry.first);
}

namespace tensor {
    using TensorVariant = std::variant<
        Tensor<int>,
        Tensor<float>,
        Tensor<double>,
        Tensor<__nv_bfloat16>
    >;
    using TensorPtrVariant = std::variant<
        Tensor<int>*,
        Tensor<float>*,
        Tensor<double>*,
        Tensor<__nv_bfloat16>*
    >;

    template <typename dtype>
    Tensor<dtype> zeros(const std::vector<int> &shape, bool requiresGrad = false) {
        NDArray<dtype> out(shape);
        out = static_cast<dtype>(0);
        return Tensor<dtype>(std::move(out), requiresGrad);
    }

    template <typename dtype>
    Tensor<dtype> ones(const std::vector<int> &shape, bool requiresGrad = false) {
        NDArray<dtype> out(shape);
        out = static_cast<dtype>(1);
        return Tensor<dtype>(std::move(out), requiresGrad);
    }
}

#endif //NEUROCORE_TENSOR_CUH
