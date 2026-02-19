//
// Created by Bujor Ionut Raul on 19.02.2026.
//

#ifndef NEUROCORE_RANDOM_CUH
#define NEUROCORE_RANDOM_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "ndarray.cuh"
#include "tensor.cuh"
#include <numeric>
#include <random>


inline int SEED=42;
inline void setManualSeed(int seed) {SEED=seed;}
inline int getManualSeed() {return SEED;}
inline const int NOTSET = 2147483647;



template <typename dtype>
struct RandomUniformOp {
    int seed = 42;
    dtype lower = 0.0, upper = 1.0;
    __device__ dtype operator() (dtype, dtype) const {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        return lower + (upper - lower) * (dtype) curand_uniform(&state);
    }
};

struct RandomIntOp {
    int seed = 42;
    int lower = 0, upper = 1;
    __device__ int operator()(int, int) const {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        return lower + (curand(&state) % (upper - lower));
    }
};


template <typename dtype>
struct RandomNormalOp {
    int seed = 42;
    dtype mean = 0.0, stddev = 1.0;
    __device__ dtype operator()(dtype, dtype) const {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        return stddev * (dtype)curand_normal(&state) + mean;
    }
};


namespace arr{
    template <typename dtype>
    NDArray<dtype> rand(const Shape &shape, int seed=NOTSET,
        dtype lower=0.0, dtype upper=1.0) {
        if (seed == NOTSET) seed = getManualSeed();
        NDArray<dtype> result(shape);
        return result.executeElementWise(RandomUniformOp<dtype>{seed, lower, upper},
            nullptr, nullptr);
    }
    template <typename dtype>
    NDArray<dtype> randLike(const NDArray<dtype> &other, int seed=NOTSET,
     dtype lower=0.0, dtype upper=1.0) {
        return rand(Shape(other.getShape()), seed, lower, upper);
    }
    template <typename dtype>
    void rand(NDArray<dtype> &out, int seed=NOTSET,
        dtype lower=0.0, dtype upper=1.0) {
        if (seed == NOTSET) seed = getManualSeed();
        out.executeElementWise(RandomUniformOp<dtype>{seed, lower, upper},
            nullptr, &out);
    }

    NDArray<int> randInt(const Shape &shape, int seed=NOTSET,
        int lower=0, int upper=1) {
        if (seed == NOTSET) seed = getManualSeed();
        NDArray<int> result(shape);
        return result.executeElementWise(RandomIntOp{seed, lower, upper},
            nullptr, nullptr);
    }
    template <typename dtype>
    NDArray<int> randIntLike(const NDArray<dtype> &other, int seed=NOTSET,
     int lower=0, int upper=1) {
        return randInt(Shape(other.getShape()), seed, lower, upper);
    }
    void randInt(NDArray<int> &out, int seed=NOTSET,
        int lower=0, int upper=1) {
        if (seed == NOTSET) seed = getManualSeed();
        out.executeElementWise(RandomIntOp{seed, lower, upper},
            nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> randN(const Shape &shape, int seed=NOTSET,
        dtype mean=0.0, dtype stddev=1.0) {
        if (seed == NOTSET) seed = getManualSeed();
        NDArray<dtype> result(shape);
        return result.executeElementWise(RandomNormalOp<dtype>{seed, mean, stddev},
            nullptr, nullptr);
    }
    template <typename dtype>
    NDArray<dtype> randNLike(const NDArray<dtype> &other, int seed=NOTSET,
        dtype mean=0.0, dtype stddev=1.0) {
        return randN(Shape(other.getShape()), seed, mean, stddev);
    }
    template <typename dtype>
    void randN(NDArray<dtype> &out, int seed=NOTSET,
        dtype mean=0.0, dtype stddev=1.0) {
        if (seed == NOTSET) seed = getManualSeed();
        out.executeElementWise(RandomNormalOp<dtype>{seed, mean, stddev},
            nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> shuffle(const NDArray<dtype> &arr, int axis = 0, int seed=NOTSET,
        std::vector<Slice> *perm = nullptr) {
        int ndim = arr.getNDim();
        if (axis < 0) axis += ndim;
        if (axis < 0 || axis > ndim - 1)
            throw IndexingException("Wrong axis.");
        int n = arr.getShape()[axis];
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        if (seed == NOTSET) seed = getManualSeed();
        std::mt19937 gen(seed);
        std::shuffle(idx.begin(), idx.end(), gen);
        std::vector<Slice> slices(ndim, Slice(0, -1, 1));
        slices[axis] = Slice(idx);
        if (perm) *perm = slices;
        return arr[slices];
    }
}



template <typename dtype>
class ShuffleFunction : public Function<dtype> {
private:
    int axis;
    int seed;
    std::vector<Slice> perm;
    std::vector<Slice> invPerm;
public:
    explicit ShuffleFunction(std::shared_ptr<TensorImpl<dtype>> a,
        int axis=0, int seed=NOTSET):
          Function<dtype>({std::move(a)}), axis(axis),
          seed(seed == NOTSET? getManualSeed(): seed) {
        if (axis < 0) axis += this->parents[0]->data.getNDim();
        if (axis < 0 || axis > this->parents[0]->data.getNDim() - 1)
            throw IndexingException("Wrong axis.");
    }
    NDArray<dtype> forward() override {
        return arr::shuffle(this->parents[0]->data, axis, seed, &perm);
    }
    void backward(const NDArray<dtype> &outGrad) override {
        auto &a = this->parents[0];
        if (a->requiresGrad) {
            if (invPerm.empty()) {
                invPerm = perm;
                const std::vector<int> original = invPerm[axis].getIndices();
                std::vector<int> reversed(outGrad.getShape()[axis]);
                for (int i = 0; i < (int)reversed.size(); i++)
                    reversed[original[i]] = i;
                invPerm[axis] = reversed;
            }
            a->accumulateGrad(outGrad[invPerm]);
        }
    }
};


namespace tensor {
    template <typename dtype>
    Tensor<dtype> rand(const Shape &shape, bool requiresGrad=true, int seed=NOTSET,
        dtype lower=0.0, dtype upper=1.0) {
        return Tensor<dtype>(arr::rand(shape, seed, lower, upper), requiresGrad);
    }
    template <typename dtype>
    Tensor<dtype> randLike(const Tensor<dtype> &other, bool requiresGrad=true,
        int seed=NOTSET, dtype lower=0.0, dtype upper=1.0) {
        return rand(Shape(other.data().getShape()), requiresGrad, seed, lower, upper);
    }
    template <typename dtype>
    void rand(Tensor<dtype> &out, int seed=NOTSET,
        dtype lower=0.0, dtype upper=1.0) {
        arr::rand(out.data(), seed, lower, upper);
    }

    Tensor<int> randInt(const Shape &shape, bool requiresGrad=true, int seed=NOTSET,
        int lower=0, int upper=1) {
        return Tensor<int>(arr::randInt(shape, seed, lower, upper), requiresGrad);
    }
    template <typename dtype>
    Tensor<int> randIntLike(const Tensor<dtype> &other, bool requiresGrad=true,
        int seed=NOTSET, int lower=0, int upper=1) {
        return randInt(Shape(other.data().getShape()), requiresGrad, seed, lower, upper);
    }
    void randInt(Tensor<int> &out, int seed=NOTSET,
        int lower=0, int upper=1) {
        arr::randInt(out.data(), seed, lower, upper);
    }

    template <typename dtype>
    Tensor<dtype> randN(const Shape &shape, bool requiresGrad=true, int seed=NOTSET,
        dtype mean=0.0, dtype stddev=1.0) {
        return Tensor<dtype>(arr::randN(shape, seed, mean, stddev), requiresGrad);
    }
    template <typename dtype>
    Tensor<dtype> randNLike(const Tensor<dtype> &other, bool requiresGrad=true,
        int seed=NOTSET, dtype mean=0.0, dtype stddev=1.0) {
        return randN(Shape(other.data().getShape()), requiresGrad, seed, mean, stddev);
    }
    template <typename dtype>
    void randN(Tensor<dtype> &out, int seed=NOTSET,
        dtype mean=0.0, dtype stddev=1.0) {
        arr::randN(out.data(), seed, mean, stddev);
    }

    template <typename dtype>
    Tensor<dtype> shuffle(const Tensor<dtype> &tensor, int axis=0, int seed=NOTSET) {
        auto fn = std::make_shared<ShuffleFunction<dtype>>(tensor.getImpl(), axis, seed);
        NDArray<dtype> outData = fn->forward();
        Tensor<dtype> out(std::move(outData), tensor.requiresGrad());
        if (tensor.requiresGrad()) out.setGradFn(fn);
        return out;
    }

}


#endif //NEUROCORE_RANDOM_CUH