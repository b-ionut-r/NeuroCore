#ifndef NEUROCORE_OPTIM_KERNELS_CUH
#define NEUROCORE_OPTIM_KERNELS_CUH

#include <cstddef>
#include <cmath>
#include <cuda_runtime.h>

template <typename CompT, typename ParamT = CompT, typename GradT = CompT, typename MomT = CompT>
__global__ void fusedSGDKernel(
    const size_t size,
    ParamT *param,
    const GradT *grad,
    MomT *momentum,
    const float lr,
    const float weightDecay,
    const float beta
){
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x)
    {
        // Do computations in compute precision (CompT)
        CompT p = (CompT) param[idx];
        CompT g = (CompT) grad[idx];
        CompT m = (CompT) momentum[idx];

        g += (CompT)weightDecay * p;
        m = (CompT)beta * m + (CompT)(1 - beta) * g;
        p -= (CompT)lr * m;

        // Downcast precision if needed
        momentum[idx] = (MomT)m;
        param[idx] = (ParamT)p;
    }
}


template <typename CompT, typename ParamT = CompT, typename GradT = CompT, typename MomT = CompT>
__global__ void fusedRMSPropKernel(
    const size_t size,
    ParamT *param,
    const GradT *grad,
    MomT *momentum,
    const float lr,
    const float weightDecay,
    const float beta,
    const double eps
){
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < size;
        idx += blockDim.x * gridDim.x)
    {
        // Do computations in compute precision (CompT)
        CompT p = (CompT) param[idx];
        CompT g = (CompT) grad[idx];
        CompT m = (CompT) momentum[idx];
        CompT denom;

        g += (CompT) weightDecay * p;
        m = (CompT) beta * m + (CompT) (1 - beta) * g * g;
        denom = (CompT)::sqrt((double)m) + (CompT)eps;
        p -= (CompT)lr * g / denom;

        // Downcast precision if needed
        param[idx] = (ParamT)p;
        momentum[idx] = (MomT)m;
    }
}


template <typename CompT, typename ParamT = CompT, typename GradT = CompT, typename MomT = CompT>
__global__ void fusedAdamKernel(
    const size_t size,
    ParamT *param,
    const GradT *grad,
    MomT *first_momentum,
    MomT *second_momentum,
    const float lr,
    const float weightDecay,
    const float beta1,
    const float beta2,
    const double biasCorrection1,
    const double biasCorrection2,
    const double eps,
    bool adamW
) {
   for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < size;
       idx += blockDim.x * gridDim.x)
   {
       // Do computations in compute precision (CompT)
       CompT p = (CompT) param[idx];
       CompT g = (CompT) grad[idx];
       CompT m1 = (CompT) first_momentum[idx];
       CompT m2 = (CompT) second_momentum[idx];
       CompT m1Corr, m2Corr;

       // Adam update
       if(!adamW) {
            g += (CompT)weightDecay * p;
       } else {
            p -= (CompT)(lr * weightDecay) * p;
       }
       m1 = (CompT)beta1 * m1 + (CompT) (1 - beta1) * g;
       m2 = (CompT)beta2 * m2 + (CompT) (1 - beta2) * (g * g);
       m1Corr = m1 / (CompT)(biasCorrection1);
       m2Corr = m2 / (CompT)(biasCorrection2);
       p -= (CompT)lr * m1Corr / ((CompT)::sqrt((double)m2Corr) + (CompT)eps);

       // Downcast precision if needed
       param[idx] = (ParamT)p;
       first_momentum[idx] = (MomT)m1;
       second_momentum[idx] = (MomT)m2;
   }
}

#endif //NEUROCORE_OPTIM_KERNELS_CUH
