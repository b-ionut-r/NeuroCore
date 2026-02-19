//
// Created by Bujor Ionut Raul on 22.12.2025.
//
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include "optim/rmsprop.cuh"
#include "optim/optim_kernels.cuh"
#include "core/exceptions.h"
#include "core/utils.h"

RMSProp::RMSProp(std::vector<tensor::TensorVariant> params, const float &lr,
                 const float &weightDecay, const float &beta,
                 const double &eps, const ComputeDType &dtype):
       Optimizer(std::move(params), lr, weightDecay, dtype), beta(beta), eps(eps) {
    try {
        for (const auto &param: this->params) {
            std::visit([&](auto &p) {
                // Always create momentum in fp32 for numerical stability
                auto mom = new NDArray<float>(Shape(p.shape()));
                *mom = 0.0f;
                momentum.push_back(mom);
            }, param);
        }
    } catch (...) {
        RMSProp::~RMSProp();
        throw;
    }
}

RMSProp::~RMSProp() {
    for (auto *mom: momentum)
        delete mom;
    momentum.clear();
}

void RMSProp::step() {
    t++;
    for (size_t i = 0; i < params.size(); i++) {
        auto run = [&](auto dummy) {
            using compute_t = decltype(dummy);
            std::visit([&](auto &param) {
                using param_t = typename std::decay_t<decltype(param)>::value_type;
                if (param.requiresGrad() && param.hasGrad()) {
                    int NThreads = 256;
                    int NBlocks = utils::getNBlocks(param.size(), NThreads);
                    fusedRMSPropKernel<compute_t, param_t, param_t, float><<<NBlocks, NThreads>>>(
                        param.size(),
                        param.data().getData(),
                        param.grad().getData(),
                        momentum[i]->getData(),
                        lr,
                        weightDecay,
                        beta,
                        eps
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
    // Syncronize and check errors once per step
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelException(std::string("RMSProp Optimizer kernel error -> ") \
            + cudaGetErrorString(err));
    }
}

std::ostream & operator<<(std::ostream &os, const RMSProp &rms) {
    os << "RMSProp optimizer: " << std::endl;
    os << "LR: " << rms.lr << ", ";
    os << "Weight Decay: " << rms.weightDecay << ", ";
    os << "Beta: " << rms.beta << ", ";
    os << "Eps: " << rms.eps << ", ";
    os << "t: " << rms.t << std::endl;
    return os;
}









