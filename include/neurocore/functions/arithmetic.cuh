//
// Created by Bujor Ionut Raul on 17.02.2026.
//

#ifndef NEUROCORE_ARITHMETIC_CUH
#define NEUROCORE_ARITHMETIC_CUH


#include "functions/base.cuh"
#include "core/tensor.cuh"
#include "core/ndarray.cuh"
#include "functions/grad_kernels.cuh"
#include "core/type_traits.cuh"
#include <memory>
#include <optional>
#include <utility>


template <typename dtype>
class AddFunction : public Function<dtype> {
public:
    AddFunction(std::shared_ptr<TensorImpl<dtype>> a,
                std::shared_ptr<TensorImpl<dtype>> b)
        : Function<dtype>({std::move(a), std::move(b)}) {}

    NDArray<dtype> forward() override {
        return this->parents[0]->data + this->parents[1]->data;
    }

    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        auto &b = this->parents[1];
        if (a->requiresGrad)
            a->accumulateGrad(reduceGradToShape(outGrad, a->data.getShape()));
        if (b->requiresGrad)
            b->accumulateGrad(reduceGradToShape(outGrad, b->data.getShape()));
    }
};


template <typename dtype>
class SubFunction : public Function<dtype> {
public:
    SubFunction(std::shared_ptr<TensorImpl<dtype>> a,
                std::shared_ptr<TensorImpl<dtype>> b)
        : Function<dtype>({std::move(a), std::move(b)}) {}

    NDArray<dtype> forward() override {
        return this->parents[0]->data - this->parents[1]->data;
    }

    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        auto &b = this->parents[1];
        if (a->requiresGrad)
            a->accumulateGrad(reduceGradToShape(outGrad, a->data.getShape()));
        if (b->requiresGrad) {
            NDArray<dtype> neg = -outGrad;
            b->accumulateGrad(reduceGradToShape(neg, b->data.getShape()));
        }
    }
};


template <typename dtype>
class NegFunction : public Function<dtype> {
public:
    explicit NegFunction(std::shared_ptr<TensorImpl<dtype>> a)
        : Function<dtype>({std::move(a)}) {}

    NDArray<dtype> forward() override {
        return -this->parents[0]->data;
    }

    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            NDArray<dtype> aGrad = -outGrad;
            a->accumulateGrad(reduceGradToShape(aGrad, a->data.getShape()));
        }
    }
};


template <typename dtype>
class MulFunction : public Function<dtype> {
public:
    MulFunction(std::shared_ptr<TensorImpl<dtype>> a,
                std::shared_ptr<TensorImpl<dtype>> b)
        : Function<dtype>({std::move(a), std::move(b)}) {}

    NDArray<dtype> forward() override {
        return this->parents[0]->data * this->parents[1]->data;
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        auto &b = this->parents[1];
        if (a && a->requiresGrad) {
            NDArray<dtype> aGrad = outGrad * b->data;
            a->accumulateGrad(reduceGradToShape(aGrad, a->data.getShape()));
        }
        if (b && b->requiresGrad) {
            NDArray<dtype> bGrad = outGrad * a->data;
            b->accumulateGrad(reduceGradToShape(bGrad, b->data.getShape()));
        }
    }
};


template <typename dtype>
class DivFunction : public Function<dtype> {
public:
    DivFunction(std::shared_ptr<TensorImpl<dtype>> a,
                std::shared_ptr<TensorImpl<dtype>> b)
        : Function<dtype>({std::move(a), std::move(b)}) {}

    NDArray<dtype> forward() override {
        return this->parents[0]->data / this->parents[1]->data;
    }

    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        auto &b = this->parents[1];
        NDArray<dtype> aGrad, bGrad;
        if (a->requiresGrad) aGrad = NDArray<dtype>(outGrad.getShape());
        if (b->requiresGrad) bGrad = NDArray<dtype>(outGrad.getShape());
        executeGrad<dtype, DivGradOp<dtype>>(
            DivGradOp<dtype>{},
            outGrad,
            a->data,
            a->requiresGrad ? &aGrad : nullptr,
            &b->data,
            b->requiresGrad ? &bGrad : nullptr
        );
        if (a->requiresGrad)
            a->accumulateGrad(reduceGradToShape(aGrad, a->data.getShape()));
        if (b->requiresGrad)
            b->accumulateGrad(reduceGradToShape(bGrad, b->data.getShape()));
    }
};


template <typename dtype>
class PowFunction: public Function <dtype> {
public:
    PowFunction(std::shared_ptr<TensorImpl<dtype>> a,
                std::shared_ptr<TensorImpl<dtype>> b)
        : Function<dtype>({std::move(a), std::move(b)}) {}
    NDArray<dtype> forward() override {
        return arr::power(this->parents[0]->data, this->parents[1]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        auto &b = this->parents[1];
        NDArray<dtype> aGrad, bGrad;
        if (a->requiresGrad) aGrad = NDArray<dtype>(outGrad.getShape());
        if (b->requiresGrad) bGrad = NDArray<dtype>(outGrad.getShape());

        executeGrad<dtype, PowGradOp<dtype>>(
            PowGradOp<dtype>{},
            outGrad,
            a->data,
            a->requiresGrad ? &aGrad : nullptr,
            &b->data,
            b->requiresGrad ? &bGrad : nullptr
        );
        if (a->requiresGrad)
            a->accumulateGrad(reduceGradToShape(aGrad, a->data.getShape()));
        if (b->requiresGrad)
            b->accumulateGrad(reduceGradToShape(bGrad, b->data.getShape()));
    }
};

template <typename dtype>
Tensor<dtype> operator+(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    bool requiresGrad = a.requiresGrad() || b.requiresGrad();
    auto fn = std::make_shared<AddFunction<dtype>>(a.getImpl(), b.getImpl());
    NDArray<dtype> outData = fn->forward();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) out.setGradFn(fn);
    return out;
}

template <typename dtype>
Tensor<dtype> operator-(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    bool requiresGrad = a.requiresGrad() || b.requiresGrad();
    auto fn = std::make_shared<SubFunction<dtype>>(a.getImpl(), b.getImpl());
    NDArray<dtype> outData = fn->forward();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) out.setGradFn(fn);
    return out;
}

template <typename dtype>
Tensor<dtype> operator-(const Tensor<dtype> &a) {
    bool requiresGrad = a.requiresGrad();
    auto fn = std::make_shared<NegFunction<dtype>>(a.getImpl());
    NDArray<dtype> outData = fn->forward();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) out.setGradFn(fn);
    return out;
}

template <typename dtype>
Tensor<dtype> operator*(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    bool requiresGrad = a.requiresGrad() || b.requiresGrad();
    auto fn = std::make_shared<MulFunction<dtype>>(a.getImpl(), b.getImpl());
    NDArray<dtype> outData = fn->forward();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) out.setGradFn(fn);
    return out;
}

template <typename dtype>
Tensor<dtype> operator/(const Tensor<dtype> &a, const Tensor<dtype> &b) {
    bool requiresGrad = a.requiresGrad() || b.requiresGrad();
    auto fn = std::make_shared<DivFunction<dtype>>(a.getImpl(), b.getImpl());
    NDArray<dtype> outData = fn->forward();
    Tensor<dtype> out(std::move(outData), requiresGrad);
    if (requiresGrad) out.setGradFn(fn);
    return out;
}


TENSOR_BINARY_CROSS_OP(+);
TENSOR_BINARY_CROSS_OP(-);
TENSOR_BINARY_CROSS_OP(*);
TENSOR_BINARY_CROSS_OP(/);


namespace tensor {
    template <typename dtype>
    Tensor<dtype> add(const Tensor<dtype> &a, const Tensor<dtype> &b) {
        return a + b;
    }
    template <typename dtype>
    Tensor <dtype> subtract(const Tensor<dtype> &a, const Tensor<dtype> &b) {
        return a - b;
    }
    template <typename dtype>
    Tensor <dtype> subs(const Tensor<dtype> &a, const Tensor<dtype> &b) {
        return subtract(a, b);
    }
    template <typename dtype>
    Tensor <dtype> multiply(const Tensor<dtype> &a, const Tensor<dtype> &b) {
        return a * b;
    }
    template <typename dtype>
    Tensor <dtype> mul(const Tensor <dtype> &a, const Tensor <dtype> &b) {
        return multiply(a, b);
    }
    template <typename dtype>
    Tensor <dtype> divide(const Tensor<dtype> &a, const Tensor<dtype> &b) {
        return a / b;
    }
    template <typename dtype>
    Tensor <dtype> div(const Tensor<dtype> &a, const Tensor<dtype> &b) {
        return divide(a, b);
    }
    template <typename dtype>
    Tensor <dtype> power(const Tensor<dtype> &a, const Tensor<dtype> &b) {
        auto fn = std::make_shared<PowFunction<dtype>>(a.getImpl(), b.getImpl());
        bool requiresGrad = a.requiresGrad() || b.requiresGrad();
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }
    template <typename dtype>
    Tensor <dtype> pow(const Tensor<dtype> &a, const Tensor<dtype> &b) {
        return power(a, b);
    }
    TENSOR_BINARY_CROSS_FN(add);
    TENSOR_BINARY_CROSS_FN(subtract);
    TENSOR_BINARY_CROSS_FN(subs);
    TENSOR_BINARY_CROSS_FN(multiply);
    TENSOR_BINARY_CROSS_FN(mul);
    TENSOR_BINARY_CROSS_FN(divide);
    TENSOR_BINARY_CROSS_FN(div);
    TENSOR_BINARY_CROSS_FN(power);
    TENSOR_BINARY_CROSS_FN(pow);
}


#endif //NEUROCORE_ARITHMETIC_CUH
