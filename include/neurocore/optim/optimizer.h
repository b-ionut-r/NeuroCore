#ifndef NEUROCORE_OPTIMIZER_H
#define NEUROCORE_OPTIMIZER_H


#include <vector>
#include <variant>
#include "core/tensor.cuh"

/*
 Abstract Base Class for Deep Learning Optimizers.
 Design Pattern 1: STRATEGY (step method).
*/

enum ComputeDType {
    HALF,
    FLOAT,
    DOUBLE
};

class Optimizer {
protected:
    std::vector<tensor::TensorPtrVariant> params;
    float lr;
    float weightDecay;
    size_t t = 0;
    ComputeDType dtype = FLOAT;
public:
    Optimizer(std::vector<tensor::TensorPtrVariant> params, const float &lr, const float &weightDecay,
              const ComputeDType &dtype = FLOAT):
              params(params), lr(lr), weightDecay(weightDecay), dtype(dtype) {};
    virtual ~Optimizer() {}
    virtual void step() = 0;
    void zeroGrad() {
        for (auto& param : params) {
            std::visit([](auto* p) {
                p->zeroGrad();
            }, param);
        }
    }
    float getLR() const {return lr;}
    float getWeightDecay() const {return weightDecay;}
    size_t getT() const {return t;}
    ComputeDType getDType() const {return dtype;}
    void setLR(const float &newLR) {lr = newLR;}
    void setWeightDecay(const float &newWeightDecay) {weightDecay = newWeightDecay;}
};

#endif //NEUROCORE_OPTIMIZER_H
