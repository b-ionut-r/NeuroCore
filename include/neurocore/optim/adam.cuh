//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#ifndef NEUROCORE_ADAM_CUH
#define NEUROCORE_ADAM_CUH


#include <iostream>
#include "optim/optimizer.h"
#include "core/ndarray.cuh"

class Adam: public Optimizer {
private:
    float beta1, beta2;
    double eps;
    bool adamW;
    std::vector<NDArray<float>*> firstMomentum;
    std::vector<NDArray<float>*> secondMomentum;
public:
    Adam(std::vector<tensor::TensorVariant> params, const float &lr,
         const float &weightDecay, const float &beta1, const float &beta2,
         const double &eps = 1e-8, const ComputeDType &dtype = FLOAT,
         const bool &adamW = false);
    ~Adam() override;
    void step() override;
    friend std::ostream & operator<<(std::ostream &os, const Adam &adam);
    float getBeta1() const {return beta1;}
    float getBeta2() const {return beta2;}
    double getEps() const {return eps;}
    bool isAdamW() const {return adamW;}
};

#endif //NEUROCORE_ADAM_CUH
