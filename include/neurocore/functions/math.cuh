//
// Created by Bujor Ionut Raul on 17.02.2026.
//

#ifndef NEUROCORE_MATH_CUH
#define NEUROCORE_MATH_CUH

#include "functions/base.cuh"
#include "core/tensor.cuh"
#include "core/ndarray.cuh"
#include "functions/grad_kernels.cuh"
#include <cmath>
#include <memory>
#include <optional>
#include <utility>

template <typename dtype>
class ExpFunction: public Function<dtype> {
private:
    dtype base;
public:
    explicit ExpFunction(std::shared_ptr<TensorImpl<dtype>> a,
                         dtype base = static_cast<dtype>(std::exp(1.0))):
    Function<dtype>({std::move(a)}), base(base){}
    NDArray<dtype> forward() override {
        return arr::exp(this->parents[0]->data, base);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, ExpGradOp<dtype>>(
                ExpGradOp<dtype>{base},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};

template <typename dtype>
class LogFunction: public Function<dtype> {
private:
    dtype base;
public:
    explicit LogFunction(std::shared_ptr<TensorImpl<dtype>> a,
                         dtype base = static_cast<dtype>(std::exp(1.0))):
    Function<dtype>({std::move(a)}), base(base){}
    NDArray<dtype> forward() override {
        return arr::log(this->parents[0]->data, base);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, LogGradOp<dtype>>(
                LogGradOp<dtype>{base},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};

template <typename dtype>
class RaiseFunction: public Function<dtype> {
private:
    dtype power;
public:
    explicit RaiseFunction(std::shared_ptr<TensorImpl<dtype>> a,
                           dtype power):
    Function<dtype>({std::move(a)}), power(power){}
    NDArray<dtype> forward() override {
        return arr::raise(this->parents[0]->data, power);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, RaiseGradOp<dtype>>(
                RaiseGradOp<dtype>{power},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};


template <typename dtype>
class AbsFunction: public Function<dtype> {
public:
    explicit AbsFunction(std::shared_ptr<TensorImpl<dtype>> a): Function<dtype>({std::move(a)}) {}
    NDArray<dtype> forward() override {
        return arr::abs(this->parents[0]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, AbsGradOp<dtype>>(
                AbsGradOp<dtype>{},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};


template <typename dtype>
class ClipFunction: public Function<dtype> {
private:
    dtype low, high;
public:
    explicit ClipFunction(std::shared_ptr<TensorImpl<dtype>> a, dtype low, dtype high):
    Function<dtype>({std::move(a)}), low(low), high(high){}
    NDArray<dtype> forward() override {
        return arr::clip(this->parents[0]->data, low, high);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, ClipGradOp<dtype>>(
                ClipGradOp<dtype>{low, high},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};


template <typename dtype>
class SinFunction: public Function<dtype> {
public:
    explicit SinFunction(std::shared_ptr<TensorImpl<dtype>> a): Function<dtype>({std::move(a)}) {}
    NDArray<dtype> forward() override {
        return arr::sin(this->parents[0]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, SinGradOp<dtype>>(
                SinGradOp<dtype>{},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};

template <typename dtype>
class CosFunction: public Function<dtype> {
public:
    explicit CosFunction(std::shared_ptr<TensorImpl<dtype>> a): Function<dtype>({std::move(a)}) {}
    NDArray<dtype> forward() override {
        return arr::cos(this->parents[0]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, CosGradOp<dtype>>(
                CosGradOp<dtype>{},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};

template <typename dtype>
class TanFunction: public Function<dtype> {
public:
    explicit TanFunction(std::shared_ptr<TensorImpl<dtype>> a): Function<dtype>({std::move(a)}) {}
    NDArray<dtype> forward() override {
        return arr::tan(this->parents[0]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, TanGradOp<dtype>>(
                TanGradOp<dtype>{},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};


template <typename dtype>
class CotFunction: public Function<dtype> {
public:
    explicit CotFunction(std::shared_ptr<TensorImpl<dtype>> a): Function<dtype>({std::move(a)}) {}
    NDArray<dtype> forward() override {
        return arr::cot(this->parents[0]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, CotGradOp<dtype>>(
                CotGradOp<dtype>{},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};

template <typename dtype>
class ASinFunction: public Function<dtype> {
public:
    explicit ASinFunction(std::shared_ptr<TensorImpl<dtype>> a): Function<dtype>({std::move(a)}) {}
    NDArray<dtype> forward() override {
        return arr::asin(this->parents[0]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, ASinGradOp<dtype>>(
                ASinGradOp<dtype>{},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};


template <typename dtype>
class ACosFunction: public Function<dtype> {
public:
    explicit ACosFunction(std::shared_ptr<TensorImpl<dtype>> a): Function<dtype>({std::move(a)}) {}
    NDArray<dtype> forward() override {
        return arr::acos(this->parents[0]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, ACosGradOp<dtype>>(
                ACosGradOp<dtype>{},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};


template <typename dtype>
class ATanFunction: public Function<dtype> {
public:
    explicit ATanFunction(std::shared_ptr<TensorImpl<dtype>> a): Function<dtype>({std::move(a)}) {}
    NDArray<dtype> forward() override {
        return arr::atan(this->parents[0]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, ATanGradOp<dtype>>(
                ATanGradOp<dtype>{},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};

template <typename dtype>
class ACotFunction: public Function<dtype> {
public:
    explicit ACotFunction(std::shared_ptr<TensorImpl<dtype>> a): Function<dtype>({std::move(a)}) {}
    NDArray<dtype> forward() override {
        return arr::acot(this->parents[0]->data);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            auto aGrad = NDArray<dtype>(outGrad.getShape());
            executeGrad<dtype, ACotGradOp<dtype>>(
                ACotGradOp<dtype>{},
                outGrad,
                a->data,
                &aGrad
            );
            a->accumulateGrad(aGrad);
        }
    }
};



namespace tensor {
    template <typename dtype>
    Tensor<dtype> exp(const Tensor<dtype> &a, dtype base = static_cast<dtype>(std::exp(1.0))) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<ExpFunction<dtype>>(a.getImpl(), base);
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }

   template <typename dtype>
   Tensor<dtype> log(const Tensor<dtype> &a, dtype base = static_cast<dtype>(std::exp(1.0))) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<LogFunction<dtype>>(a.getImpl(), base);
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }
    template<typename dtype>
    Tensor<dtype> log2(const Tensor<dtype> &a) {
        return log(a, static_cast<dtype>(2.0));
    }
    template<typename dtype>
    Tensor<dtype> log10(const Tensor<dtype> &a) {
        return log(a, static_cast<dtype>(10.0));
    }


    template <typename dtype>
    Tensor<dtype> raise(const Tensor<dtype> &a, dtype power) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<RaiseFunction<dtype>>(a.getImpl(), power);
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }
    template <typename dtype>
    Tensor<dtype> square(const Tensor<dtype> &a) {
        return raise(a, static_cast<dtype>(2.0));
    }
    template <typename dtype>
    Tensor<dtype> cube(const Tensor<dtype> &a) {
        return raise(a, static_cast<dtype>(2.0));
    }
    template <typename dtype>
    Tensor<dtype> sqrt(const Tensor<dtype> &a) {
        return raise(a, static_cast<dtype>(0.5));
    }
    template <typename dtype>
    Tensor<dtype> cbrt(const Tensor<dtype> &a) {
        return raise(a, static_cast<dtype>(1.0/3.0));
    }

    template <typename dtype>
    Tensor<dtype> abs(const Tensor<dtype> &a) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<AbsFunction<dtype>>(a.getImpl());
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }

    template <typename dtype>
    Tensor<dtype> sin(const Tensor<dtype> &a) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<SinFunction<dtype>>(a.getImpl());
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }

    template <typename dtype>
    Tensor<dtype> cos(const Tensor<dtype> &a) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<CosFunction<dtype>>(a.getImpl());
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }

    template <typename dtype>
    Tensor<dtype> tan(const Tensor<dtype> &a) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<TanFunction<dtype>>(a.getImpl());
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }

    template <typename dtype>
    Tensor<dtype> cot(const Tensor<dtype> &a) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<CotFunction<dtype>>(a.getImpl());
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }

    template <typename dtype>
    Tensor<dtype> asin(const Tensor<dtype> &a) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<ASinFunction<dtype>>(a.getImpl());
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }

    template <typename dtype>
    Tensor<dtype> acos(const Tensor<dtype> &a) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<ACosFunction<dtype>>(a.getImpl());
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }

    template <typename dtype>
    Tensor<dtype> atan(const Tensor<dtype> &a) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<ATanFunction<dtype>>(a.getImpl());
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }

    template <typename dtype>
    Tensor<dtype> acot(const Tensor<dtype> &a) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<ACotFunction<dtype>>(a.getImpl());
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }


    template <typename dtype>
    Tensor<dtype> clip(const Tensor<dtype> &a, dtype low, dtype high) {
        bool requiresGrad = a.requiresGrad();
        auto fn = std::make_shared<ClipFunction<dtype>>(a.getImpl(), low, high);
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), requiresGrad);
        if (requiresGrad) out.setGradFn(fn);
        return out;
    }
}


#endif //NEUROCORE_MATH_CUH
