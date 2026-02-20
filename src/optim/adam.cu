//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <utility>
#include "optim/adam.cuh"
#include "optim/optim_kernels.cuh"
#include "core/exceptions.h"
#include "core/utils.h"


Adam::Adam(std::vector<tensor::TensorVariant> params, const float &lr,
        const float &weightDecay, const float &beta1, const float &beta2,
        const double &eps, const ComputeDType &dtype,
        const bool &adamW):
       Optimizer(std::move(params), lr, weightDecay, dtype),
       beta1(beta1),beta2(beta2), eps(eps), adamW(adamW) {
    try {
        for (const auto &param : this->params) {
            std::visit([&](auto &p) {
                // Always create momentum in fp32 for numerical stability
                auto mom = new NDArray<float>(Shape(p.shape()));
                *mom = 0.0f;
                firstMomentum.push_back(mom);
            }, param);
        }
        for (const auto &param : this->params) {
            std::visit([&](auto &p) {
                // Always create momentum in fp32 for numerical stability
                auto mom = new NDArray<float>(Shape(p.shape()));
                *mom = 0.0f;
                secondMomentum.push_back(mom);
            }, param);
        }
    } catch (...) {
        Adam::~Adam();
        throw;
    }
}

Adam::~Adam(){
    for (auto *mom : firstMomentum)
        delete mom;
    for (auto *mom : secondMomentum)
        delete mom;
    firstMomentum.clear();
    secondMomentum.clear();
}


void Adam::step() {
    t++;
    double biasCorrection1 = 1.0 - pow(beta1, t);
    double biasCorrection2 = 1.0 - pow(beta2, t);
    for (size_t i = 0; i < params.size(); i++) {
        auto run = [&](auto dummy) {
            using compute_t = decltype(dummy);
            std::visit([&](auto &param) {
                using param_t = typename std::decay_t<decltype(param)>::value_type;
                if (param.requiresGrad() && param.hasGrad()) {
                    int NThreads = 256;
                    int NBlocks = utils::getNBlocks(param.size(), NThreads);
                    fusedAdamKernel<compute_t, param_t, param_t, float><<<NBlocks, NThreads>>>(
                        param.size(),
                        param.data().getData(),
                        param.grad().getData(),
                        firstMomentum[i]->getData(),
                        secondMomentum[i]->getData(),
                        lr,
                        weightDecay,
                        beta1,
                        beta2,
                        biasCorrection1,
                        biasCorrection2,
                        eps,
                        adamW
                    );
                }
            }, params[i]);
        };
        switch (dtype) {
            case HALF: run(__half(0)); break;
            case FLOAT: run(float{0}); break;
            case DOUBLE: run(double{0}); break;
        }
    }
    // Synchronize and check errors once per step
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelException(std::string("Adam Optimizer kernel error -> ") \
            + cudaGetErrorString(err));
    }
}

std::ostream & operator<<(std::ostream &os, const Adam &adam) {
    if (adam.adamW) {
        os << "AdamW optimizer: ";
    } else {
        os << "Adam optimizer: ";
    }
    os << "LR: " << adam.lr << ", ";
    os << "Weight Decay: " << adam.weightDecay << ", ";
    os << "Beta1: " << adam.beta1 << ", ";
    os << "Beta2: " << adam.beta2 << ", ";
    os << "Eps: " << adam.eps << ", ";
    os << "t: " << adam.t << std::endl;
    return os;
}

