//
// Created by Bujor Ionut Raul on 22.12.2025.
//


#ifndef NEUROCORE_SGD_CUH
#define NEUROCORE_SGD_CUH


#include <iostream>
#include "optim/optimizer.h"
#include "core/ndarray.cuh"

class SGD: public Optimizer {
private:
    float beta;
    std::vector<NDArray<float>*> momentum;
public:
    SGD(std::vector<tensor::TensorPtrVariant> params, const float &lr,
        const float &weightDecay, const float &beta, const ComputeDType &dtype = FLOAT);
    ~SGD() override;
    void step() override;
    friend std::ostream & operator<<(std::ostream &os, const SGD &sgd);
    float getBeta() const {return beta;}
};

#endif //NEUROCORE_SGD_CUH
