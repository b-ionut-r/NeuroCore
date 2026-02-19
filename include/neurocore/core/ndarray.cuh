//
// Created by Bujor Ionut Raul on 16.11.2025.
//

#ifndef NEUROCORE_NDARRAY_CUH
#define NEUROCORE_NDARRAY_CUH

#include <iostream>
#include <vector>
#include <string>
#include <list>
#include <memory>
#include <cmath>
#include "core/elementwise_kernels.cuh"
#include "core/slices.h"
#include "core/utils.h"
#include "core/exceptions.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "cuda/std/type_traits"
#include "core/type_traits.cuh"
#include <variant>


/// FORWARD DECLARATIONS FOR OSTREAM
template <typename dtype>
class NDArray;
template <typename dtype>
std::ostream& operator<<(std::ostream &os, const NDArray<dtype> &arr);
template <typename dtype>
std::istream& operator>>(std::istream &is, NDArray<dtype> &arr);


template <typename dtype>
class NDArray{
protected:
    dtype *data;
    std::vector<int> shape; int ndim; int size;
    std::vector<int> strides;
    int itemBytes;
    int offset; bool ownsData;
    int N_BLOCKS; int N_THREADS = 256;
    int id;
    // Static members
    static int idGenerator;
    static size_t totalAllocatedMemory; // GPU Memory

    // Helpers
    void allocateDeviceMetadata(int** dStrides=nullptr,
                                int** dShape=nullptr) const;
public:
    using value_type = dtype;
    /// CONSTRUCTORS and DESTRUCTORS
    NDArray(); // default constructor
    NDArray(const Shape &shape); // alocator constructor
    void _computeStrides();
    NDArray(dtype *data, const std::vector<int> &shape, const int &offset,
            const std::vector<int> &strides); // viewer constructor
    NDArray(const NDArray<dtype> &other); // copy constructor
    NDArray(const std::vector<dtype> &vec); // constructor from nested vector
    NDArray(NDArray<dtype> &&other) noexcept; /* move constructor
    for returned rvalues views of ndarray. */
    NDArray& operator=(NDArray<dtype> &&other) noexcept; // move assignment
    ~NDArray();
    /// GETTERS and SETTERS (inline)
    dtype* getData() {return data;}
    const dtype* getData() const {return data;}
    std::vector<int> getShape() const {return shape;}
    int getNDim() const {return ndim;}
    int getSize() const {return size;}
    std::vector<int> getStrides() const {return strides;}
    void setStrides(const std::vector <int> &new_strides) {
        if (strides.size() != new_strides.size()) {
            throw NDimMismatchException("New strides vector must have "
                                        "same size as old strides vector.");
        }
        strides = new_strides;
    }
    int getItemBytes() const {return itemBytes;}
    int getOffset() const {return offset;}
    bool getOwnsData() const {return ownsData;}
    int getNBLOCKS() const {return N_BLOCKS;}
    int getNTHREADS() const {return N_THREADS;}
    static size_t getTotalAllocatedMemory() { return totalAllocatedMemory; }
    void synchronize() const { cudaDeviceSynchronize(); }

    /// UTILITY FUNCTIONS
    bool isContiguous() const;

    /// OVERLOADED OPERATORS
    template <typename Op>
    NDArray<dtype> executeElementWise(Op op, const NDArray *other = nullptr,
                                      NDArray *final = nullptr) const;
    dtype& operator[](const std::vector<int>& idx);
    NDArray operator[](std::vector<Slice> slices);
    NDArray& operator=(const dtype &value);
    NDArray& operator=(const NDArray &other);
    NDArray& operator=(const std::vector<dtype> &vec);
    NDArray operator+(const NDArray &other) const;
    NDArray operator+(const dtype &value) const;
    NDArray operator-() const;
    NDArray operator-(const NDArray &other) const;
    NDArray operator-(const dtype &value) const;
    NDArray operator*(const NDArray &other) const;
    NDArray operator*(const dtype &value) const;
    NDArray operator/(const NDArray &other) const;
    NDArray operator/(const dtype &value) const;
    friend std::ostream& operator<< <>(std::ostream &os, const NDArray<dtype> &arr);
    friend std::istream& operator>> <>(std::istream &is, NDArray<dtype> &arr);

    /// OTHERS
    NDArray transpose(std::vector<int> perm={}) const;
    NDArray transposeInPlace(std::vector<int> perm={});
    std::vector<dtype> toVector() const;
    template <typename newDtype>
    NDArray<newDtype> cast() const;
    NDArray zeros_like() const;
    NDArray ones_like() const;
};

template<typename dtype>
int NDArray<dtype>::idGenerator = 0; // static variable (initialized outside class)
template<typename dtype>
size_t NDArray<dtype>::totalAllocatedMemory = 0;


/// DEFINITIONS FOR TEMPLATES ALSO NEED TO BE IN HEADER ///

template<typename dtype>
NDArray<dtype>::NDArray():
    data(nullptr),
    shape({}),
    ndim(0),
    strides({}),
    itemBytes(sizeof(dtype)),
    offset(0),
    ownsData(false),
    id(++idGenerator),
    size(0),
    N_BLOCKS(0){
};


template<typename dtype>
NDArray<dtype>::NDArray(const Shape &shape):
    shape(shape.dims),
    ndim(shape.dims.size()),
    strides(shape.dims.size()),
    itemBytes(sizeof(dtype)),
    offset(0),
    ownsData(true),
    id(++idGenerator)
{
    size = this->shape[0];
    for (int i = 1; i < ndim; i++) {
        size *= this->shape[i];
    }
    N_BLOCKS = (size + N_THREADS - 1) / N_THREADS;
    _computeStrides();
    cudaError_t err = cudaMallocManaged(&data, size * itemBytes);
    if (err != cudaSuccess) {
        throw CudaKernelException("OOM during NDArray allocation");
    }
    totalAllocatedMemory += size * itemBytes;
}

template<typename dtype>
void NDArray<dtype>::_computeStrides() {
    int prod = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strides[i] = prod;
        prod *= shape[i];
    }
}

template<typename dtype>
NDArray<dtype>::NDArray(dtype *data, const std::vector<int> &shape,
    const int &offset, const std::vector<int> &strides):
    data(data),
    shape(shape),
    ndim(shape.size()),
    strides(strides),
    itemBytes(sizeof(dtype)),
    offset(offset),
    ownsData(false),
    id(++idGenerator)
{
    size = shape[0];
    for (int i = 1; i < ndim; i++) {
        size *= shape[i];
    }
    N_BLOCKS = (size + N_THREADS - 1) / N_THREADS;
};


template<typename dtype>
NDArray<dtype>::NDArray(const NDArray<dtype> &other):
    shape(other.shape), ndim(other.ndim), size(other.size),
    strides(other.strides), itemBytes(sizeof(dtype)),
    offset(0), ownsData(true), id(++idGenerator)
{
    N_BLOCKS = (size + N_THREADS - 1) / N_THREADS;
    _computeStrides();
    cudaError_t err = cudaMallocManaged(&data, size * itemBytes);
    if (err != cudaSuccess) {
        throw CudaKernelException("OOM during NDArray allocation");
    }
    totalAllocatedMemory += size * itemBytes;
    if (other.isContiguous() && other.offset == 0) {
        cudaMemcpy(data, other.data, size * itemBytes, cudaMemcpyDeviceToDevice);
    } else {
        NDArray<dtype> temp(data, shape, 0, strides); // temp view
        temp.executeElementWise(AssignOp<dtype>{}, &other, &temp);
        temp.ownsData = false;
    }
}

template<typename dtype>
NDArray<dtype>::NDArray(const std::vector<dtype> &vec):
    shape(utils::nestedVectorShape(vec)),
    ndim(static_cast<int>(utils::nestedVectorRank(vec))),
    size(utils::nestedVectorElementCount(vec)),
    itemBytes(sizeof(dtype)), offset(0), ownsData(true), id(++idGenerator)
{
    if (!utils::isNestedVectorHomogeneous(vec))
        throw ShapeMismatchException("Cannot create NDArray from non-homogeneous vector.");
    N_BLOCKS = (size + N_THREADS - 1) / N_THREADS;
    _computeStrides();
    cudaError_t err = cudaMallocManaged(&data, size * itemBytes);
    if (err != cudaSuccess) {
        throw CudaKernelException("OOM during NDArray allocation");
    }
    totalAllocatedMemory += size * itemBytes;
    utils::flattenNestedVectorToStrided(vec, data, offset, strides, shape, ndim);
}


template<typename dtype>
NDArray<dtype>::NDArray(NDArray<dtype> &&other) noexcept:
    data(other.data), shape(move(other.shape)),
    ndim(other.ndim), size(other.size),
    strides(move(other.strides)), itemBytes(other.itemBytes),
    offset(other.offset), ownsData(other.ownsData),
    N_BLOCKS(other.N_BLOCKS), id(other.id) // "steals" id of rvalue
{
    other.data = nullptr;
    other.ownsData = false;
    other.ndim = 0;
    other.size = 0;
    other.offset = 0;
    other.N_BLOCKS = 0;
}

template<typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(NDArray<dtype> &&other) noexcept {
    if (this != &other) {
        // Free existing resources
        if (ownsData && data != nullptr) {
            cudaFree(data);
            totalAllocatedMemory -= size * itemBytes;
        }
        // Transfer ownership
        data = other.data;
        shape = move(other.shape);
        ndim = other.ndim;
        size = other.size;
        strides = move(other.strides);
        itemBytes = other.itemBytes;
        offset = other.offset;
        ownsData = other.ownsData;
        N_BLOCKS = other.N_BLOCKS;
        id = other.id;
        // Nullify source
        other.data = nullptr;
        other.ownsData = false;
        other.ndim = 0;
        other.size = 0;
        other.offset = 0;
        other.N_BLOCKS = 0;
    }
    return *this;
}

template<typename dtype>
NDArray<dtype>::~NDArray() {
    shape.clear(); strides.clear();
    if (ownsData && data != nullptr) {
        cudaFree(data);
        totalAllocatedMemory -= size * itemBytes;
    }
}

template <typename dtype>
bool NDArray<dtype>::isContiguous() const {
    if (ndim == 0) return true;
    int expected_stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        if (strides[i] != expected_stride) return false;
        expected_stride *= shape[i];
    }
    return true;
}

template<typename dtype>
dtype& NDArray<dtype>::operator[](const std::vector<int>& idx) {
    if (idx.size() != ndim) {
        throw IndexingException(std::to_string(ndim) + " indices are needed.");
    }
    synchronize();
    int flat_idx = 0;
    for (int i = 0; i < ndim; i++) {
        flat_idx += strides[i] * idx[i];
    }
    return *(data + offset + flat_idx);
}



template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator[](std::vector<Slice> slices) {
    int nSlices = slices.size();
    if (nSlices > ndim) {
        throw IndexingException("Too many slices. Only " + std::to_string(ndim)
            + " slices are needed.");
    }

    // pad trailing dims with full-range slices first
    while (nSlices < ndim) {
        slices.push_back(Slice(0, shape[nSlices], 1));
        nSlices++;
    }

    std::vector<int> indices, newShape, newStrides(ndim);
    newShape.reserve(ndim);
    int nIndices = 0;
    bool atLeastOneFromIndices = false;
    for (int i = 0; i < ndim; i++) {
        Slice &slice = slices[i];
        if (slice.isFromIndices())
            atLeastOneFromIndices = true;
        else slice.normalizeEnd(shape[i]);
        std::vector<int> idxs = slice.getIndices();
        indices.insert(indices.end(), idxs.begin(), idxs.end());
        nIndices += idxs.size();
        newShape.push_back(slice.size());
    }

    if (atLeastOneFromIndices) {
        NDArray<dtype> result((Shape(newShape)));
        int NBLOCKS = result.N_BLOCKS, NTHREADS = result.N_THREADS;
        int *indicesPtr = nullptr;
        try {
            cudaMallocManaged(&indicesPtr, nIndices * sizeof(int));
            cudaMemcpy(indicesPtr, indices.data(), nIndices * sizeof(int), cudaMemcpyHostToDevice);
            gatherKernel<<<NBLOCKS, NTHREADS>>>(
                result.data, this->data,
                indicesPtr, nIndices, ndim,
                result.size, result.shape,
                this->strides, this->offset
            );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) throw CudaKernelException(cudaGetErrorString(err));
            cudaDeviceSynchronize();
            cudaFree(indicesPtr);
        } catch (...) {
            if (indicesPtr) cudaFree(indicesPtr);
            throw;
        }
        return result;
    }

    // regular strided view
    int ptrOffset = offset;
    for (int i = 0; i < ndim; i++) {
        ptrOffset += slices[i].getStart() * strides[i];
        newStrides[i] = slices[i].getStep() * strides[i];
    }
    return NDArray<dtype>(data, newShape, ptrOffset, newStrides);
}



template <typename dtype>
template <typename Op>
NDArray<dtype> NDArray<dtype>::executeElementWise(
    Op op,
    const NDArray<dtype> *other,
    NDArray<dtype> *final) const
{
    /* first, second => result */
    /// HANDLE BROADCASTING
    NDArray<dtype> *result = nullptr;
    const NDArray<dtype> *first = nullptr;
    const NDArray<dtype> *second = nullptr;
    bool delFirst = false, delSecond = false, delResult = false;
    if (other == nullptr) {
        first = this;
        second = other;
        result = final? final: new NDArray<dtype>(first->shape);
        if(final == nullptr) delResult = true;
    } else {
        auto info = getBroadcastInfo(*this, *other);
        if (info.aBroadcastAxes.empty()) first = this;
        else {
            std::vector<int> newStrides = this -> strides;
            for (size_t i = 0; i < info.aBroadcastAxes.size(); i++) {
                newStrides[info.aBroadcastAxes[i]] = 0;
            }
            first = new NDArray<dtype>(this->data, info.finalShape, this->offset, newStrides);
            delFirst = true;
        }
        if (info.bBroadcastAxes.empty()) second = other;
        else{
            std::vector<int> newStrides = other -> strides;
            for (size_t i = 0; i < info.bBroadcastAxes.size(); i++) {
                newStrides[info.bBroadcastAxes[i]] = 0;
            }
            second = new NDArray<dtype>(other->data, info.finalShape, other->offset, newStrides);
            delSecond = true;
        }
        result = final ? final : new NDArray<dtype>(info.finalShape);
        delResult = (final == nullptr);
    }
    cudaError_t err = cudaSuccess;
    bool allContig = (first->isContiguous() && (second ? other->isContiguous() : true)
                      && (result? result->isContiguous(): true));
    try {
        if (allContig) {
            elementWiseKernelContiguous<dtype, dtype, Op><<<N_BLOCKS, N_THREADS>>>(
                result->data, result->offset, result->size,
                op,
                first->data, first->offset,
                second ? second->data : nullptr, second ? second->offset : 0
            );
        } else {
            int *dResultShape = nullptr;
            int *dResultStrides = nullptr, *dFirstStrides = nullptr, *dSecondStrides = nullptr;

            try {
                result->allocateDeviceMetadata(&dResultStrides, &dResultShape);
                first->allocateDeviceMetadata(&dFirstStrides, nullptr);
                if (second) second->allocateDeviceMetadata(&dSecondStrides, nullptr);

                elementWiseKernelStrided<dtype, dtype, Op><<<N_BLOCKS, N_THREADS>>>(
                    result->data, result->offset, dResultStrides,
                    result->size, result->ndim, dResultShape,
                    op,
                    first->data, first->offset, dFirstStrides,
                    second ? second->data : nullptr, second ? second->offset : 0, dSecondStrides
                );
            } catch (...) {
                // Clean up device memory on exception (to prevent leakage)
                if (dResultShape) cudaFree(dResultShape);
                if (dResultStrides) cudaFree(dResultStrides);
                if (dFirstStrides) cudaFree(dFirstStrides);
                if (dSecondStrides) cudaFree(dSecondStrides);
                throw;
            }

            utils::cudaFreeMulti({dResultShape, dResultStrides, dFirstStrides});
            if (second) cudaFree(dSecondStrides);
        }

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw CudaKernelException(cudaGetErrorString(err));
        }
    } catch (...) {
        // Clean up device memory on exception (to prevent leakage)
        if (delFirst) delete first;
        if (delSecond) delete second;
        if (delResult) delete result;
        throw;
    }

    // Normal cleanup
    if (delFirst) delete first;
    if (delSecond) delete second;
    if (delResult) {
        NDArray retVal = std::move(*result);
        delete result;
        return retVal;
    } else {
        return *result;
    }
}


template <typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(const dtype &value) {
    executeElementWise(SetConstantOp<dtype>{value}, nullptr, this); // inplace execution
    return *this;  // Return reference to this, not the temporary from executeElementWise
}

template <typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(const NDArray<dtype> &other) {
    if (shape != other.shape)
        throw ShapeMismatchException("Cannot assign arrays of different shapes.");
    if (isContiguous() && other.isContiguous() && offset == 0 && other.offset == 0) {
        cudaMemcpy(data, other.data, size * itemBytes, cudaMemcpyDeviceToDevice);
        return *this;
    }
    executeElementWise(AssignOp<dtype>{}, &other, this); // inplace execution
    return *this;  // Return reference to this, not the temporary from executeElementWise
}

template <typename dtype>
NDArray<dtype>& NDArray<dtype>::operator=(const std::vector<dtype> &vec) {
    if (!utils::isNestedVectorHomogeneous(vec))
        throw ShapeMismatchException("Cannot assign nested vector with non-homogenous data.");
    if (ndim != static_cast<int>(utils::nestedVectorRank(vec)))
        throw NDimMismatchException("Cannot assign nested vector of different ndim.");
    if (size != utils::nestedVectorElementCount(vec))
        throw SizeMismatchException("Cannot assign nested vector of different size.");
    if (shape != utils::nestedVectorShape(vec))
        throw ShapeMismatchException("Cannot assign nested vector of different shape.");
    utils::flattenNestedVectorToStrided(vec, data, offset, strides, shape, ndim);
    return *this;
}


template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator+(const NDArray<dtype> &other) const {
    return executeElementWise(AffineAddOp<dtype>{1, 1}, &other, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator+(const dtype &value) const {
    return executeElementWise(ScalarAddOp<dtype>{value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> operator+(dtype value, const NDArray<dtype> &arr) {
    return arr + value;
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-() const {
    return executeElementWise(ScalarMulOp<dtype>{static_cast<dtype>(-1)}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-(const NDArray<dtype> &other) const {
    return executeElementWise(AffineAddOp<dtype>{static_cast<dtype>(1), static_cast<dtype>(-1)},
                              &other, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator-(const dtype &value) const {
    return executeElementWise(ScalarAddOp<dtype>{-value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> operator-(const dtype &value, const NDArray<dtype> &arr) {
    return arr.executeElementWise(ScalarRSubOp<dtype>{value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator*(const NDArray<dtype> &other) const {
    return executeElementWise(MulOp<dtype>{}, &other, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator*(const dtype &value) const {
    return executeElementWise(ScalarMulOp<dtype>{value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> operator*(const dtype &value, const NDArray<dtype> &arr) {
    return arr * value;
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator/(const NDArray<dtype> &other) const {
    return executeElementWise(DivOp<dtype>{}, &other, nullptr);
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::operator/(const dtype &value) const {
    return executeElementWise(ScalarMulOp<dtype>{1/value}, nullptr, nullptr);
}

template <typename dtype>
NDArray<dtype> operator/(const dtype &value, const NDArray<dtype> &arr) {
    return arr.executeElementWise(ScalarRDivOp<dtype>{value}, nullptr, nullptr);
}

/// HANDLE CROSS-TYPE OPERATORS VIA MACRO
NDARRAY_BINARY_CROSS_OP(+)
NDARRAY_BINARY_CROSS_OP(-)
NDARRAY_BINARY_CROSS_OP(*)
NDARRAY_BINARY_CROSS_OP(/)
///

template <typename dtype>
std::ostream& operator<<(std::ostream &os, const NDArray<dtype> &arr) {
    if (arr.size == 0) {
        os << "[]";
        return os;
    }
    arr.synchronize();
    std::vector<int> multi_idx(arr.ndim);
    for (int i = 0; i < arr.size; i++) {
        int remaining = i;
        for (int d = arr.ndim - 1; d >= 0; d--) {
            multi_idx[d] = remaining % arr.shape[d];
            remaining /= arr.shape[d];
        }
        if (i == 0) {
            for(int d = 0; d < arr.ndim; ++d) os << "[";
        }
        else {
            for (int d = 0; d < arr.ndim - 1; d++) {
                if (multi_idx[d + 1] == 0) {
                    if (d == arr.ndim - 2) os << std::endl;
                    os << "[";
                } else {
                    break;
                }
            }
        }
        int data_idx = arr.offset;
        for (int d = 0; d < arr.ndim; d++) {
            data_idx += multi_idx[d] * arr.strides[d];
        }
        os << arr.data[data_idx];
        bool any_close = false;
        for (int d = arr.ndim - 1; d >= 0; d--) {
            if (multi_idx[d] == arr.shape[d] - 1) {
                os << "]";
                any_close = true;
            } else {
                break;
            }
        }
        if (any_close && i != arr.size - 1) {
        }
        if (!any_close) {
            os << ", ";
        }
    }
    return os;
}

template <typename dtype>
std::istream& operator>>(std::istream &is, NDArray<dtype> &arr) {
    arr.synchronize();
    if (arr.ownsData) {
        for (int i = 0; i < arr.size; i++) {
            is >> arr.data[i];
        }
    }
    else{
        for (int i = 0; i < arr.size; i++) {
            int real_idx = utils::flatToStridedIndex(i, arr.offset, arr.strides,
                arr.ndim, arr.shape);
            is >> arr.data[real_idx];
        }
    }
    return is;
}


// Helper to allocate device memory for strides/shape
template <typename dtype>
void NDArray<dtype>::allocateDeviceMetadata(int** dStrides, int** dShape) const {
    if (dStrides != nullptr) {
        cudaMalloc(dStrides, ndim * sizeof(int));
        cudaMemcpy(*dStrides, strides.data(),
                ndim * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (dShape != nullptr) {
        cudaMalloc(dShape, ndim * sizeof(int));
        cudaMemcpy(*dShape, shape.data(),
            ndim * sizeof(int), cudaMemcpyHostToDevice);
    }
}

template <typename dtype>
NDArray<dtype> NDArray<dtype>::transpose(std::vector<int> perm) const {
    if (perm.empty()) {
        perm.resize(ndim);
        for (int i = 0; i < ndim; i++) {
            perm[i] = i;
        }
        if (ndim >= 2) {
            std::swap(perm[ndim-2], perm[ndim-1]);
        }
    }
    if (perm.size() != ndim) {
        throw SizeMismatchException("Invalid permutation vector.");
    }
    std::vector<int> newShape(ndim);
    std::vector<int> newStrides(ndim);
    for (int i = 0; i < ndim; i++) {
        newShape[perm[i]] = shape[i];
        newStrides[perm[i]] = strides[i];
    }
    return NDArray<dtype>(data, newShape, offset, newStrides);
}

template<typename dtype>
NDArray<dtype> NDArray<dtype>::transposeInPlace(std::vector<int> perm) {
    if (perm.empty()) {
        perm.resize(ndim);
        for (int i = 0; i < ndim; i++) {
            perm[i] = i;
        }
        if (ndim >= 2) {
            std::swap(perm[ndim-2], perm[ndim-1]);
        }
    }
    if (perm.size() != ndim) {
        throw SizeMismatchException("Invalid permutation vector.");
    }
    std::vector<int> newShape(ndim);
    std::vector<int> newStrides(ndim);
    for (int i = 0; i < ndim; i++) {
        newShape[perm[i]] = shape[i];
        newStrides[perm[i]] = strides[i];
    }
    shape = newShape;
    strides = newStrides;
    return *this;
}

template <typename dtype>
std::vector<dtype> NDArray<dtype>::toVector() const {
    if (ndim != 1) throw NDimMismatchException("Cannot convert to vector of ndim != 1.");
    std::vector<dtype> vec(size);
    if (isContiguous()) {
        cudaMemcpy(vec.data(), data + offset, size * itemBytes, cudaMemcpyDeviceToHost);
        return vec;
    }
    return vec;
}


template <typename dtype>
template <typename newDtype>
NDArray<newDtype> NDArray<dtype>::cast() const {
    if constexpr (cuda::std::is_same_v<newDtype, dtype>) {
        return *this;  // if types are the same, return the object itself
    }
    // Else
    NDArray<newDtype> result(shape);
    CastOp<newDtype, dtype> op{};

    if (isContiguous()) {
        // Fast path: contiguous source
        elementWiseKernelContiguous<newDtype, dtype, CastOp<newDtype, dtype>>
            <<<N_BLOCKS, N_THREADS>>>(
            result.getData(), 0, size,
            op,
            data, offset
        );
    } else {
        // Slow path: strided source
        int *dShape = nullptr, *dSrcStrides = nullptr, *dDstStrides = nullptr;
        try {
            allocateDeviceMetadata(&dSrcStrides, &dShape);
            result.allocateDeviceMetadata(&dDstStrides, nullptr);

            elementWiseKernelStrided<newDtype, dtype, CastOp<newDtype, dtype>>
                <<<N_BLOCKS, N_THREADS>>>(
                result.getData(), 0, dDstStrides,
                size, ndim, dShape,
                op,
                data, offset, dSrcStrides
            );
        } catch (...) {
            // Clean up device memory on exception (to prevent leakage)
            utils::cudaFreeMulti({dShape, dSrcStrides, dDstStrides});
            throw CudaKernelException("Failed to cast array to new type.");
        }
        utils::cudaFreeMulti({dShape, dSrcStrides, dDstStrides});
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw CudaKernelException(cudaGetErrorString(err));
    }
    return result;
}


template<typename dtype>
NDArray<dtype> NDArray<dtype>::zeros_like() const {
    NDArray<dtype> zeros(shape);
    zeros = 0;
    return zeros;
}

template<typename dtype>
NDArray<dtype> NDArray<dtype>::ones_like() const {
    NDArray<dtype> ones(shape);
    ones = 1;
    return ones;
}


/// BROADCASTING HELPERS ///
template <typename dtype>
struct BroadcastInfo {
    std::vector<int> finalShape;
    std::vector<int> aBroadcastAxes;
    std::vector<int> bBroadcastAxes;
};

template <typename dtype>
BroadcastInfo<dtype> getBroadcastInfo(const NDArray<dtype> &a, const NDArray<dtype> &b) {
    BroadcastInfo<dtype> out;
    // Quick checks
    const int n = a.getNDim();
    if (n != b.getNDim()) {
        throw NDimMismatchException("Arrays of different ndims. Cannot broadcast.");
    }
    if (a.getShape() == b.getShape()) {
        out.finalShape = a.getShape();
        return out;
    }
    out.finalShape.reserve(n);
    for (int i = 0; i < n; i++) {
        const int da = a.getShape()[i];
        const int db = b.getShape()[i];
        if (da == db) {
            out.finalShape.push_back(da);
        } else if (da == 1) {
            out.aBroadcastAxes.push_back(i);
            out.finalShape.push_back(db);
        } else if (db == 1) {
            out.bBroadcastAxes.push_back(i);
            out.finalShape.push_back(da);
        } else {
            throw ShapeMismatchException("Arrays of different shapes. Cannot broadcast");
        }
    }
    return out;
}


/// VARIANTS
namespace arr {
    template <typename dtype>
    using NDArray = NDArray<dtype>;

    // Reduced variant types to avoid NVCC template recursion limits with MSVC's STL
    using NDArrayVariant = std::variant<
        NDArray<int>,
        NDArray<float>,
        NDArray<double>,
        NDArray<__half>
    >;
    using NDArrayPtrVariant = std::variant<
        NDArray<int>*,
        NDArray<float>*,
        NDArray<double>*,
        NDArray<__half>*
    >;

    /// Factories
    template <typename dtype>
    NDArray<dtype> make_constant(const std::vector<int> &shape, const dtype &value) {
        NDArray<dtype> array(shape);
        array = static_cast<dtype>(value);
        return array;
    }

    template <typename dtype>
    NDArray<dtype> make_zeros(const std::vector<int> &shape) {
        return make_constant(shape, static_cast<dtype>(0));
    }
    template <typename dtype>
    NDArray<dtype> make_ones(const std::vector<int> &shape) {
        return make_constant(shape, static_cast<dtype>(1));
    }

    /// Elementary binaries
    template <typename dtype>
    NDArray<dtype> add(const NDArray<dtype> &a, const NDArray<dtype> &b) {
        return a + b;
    }

    template <typename dtype>
    void add(const NDArray<dtype> &a, const NDArray<dtype> &b, NDArray<dtype> &out) {
        out = a + b;
    }

    template <typename dtype>
    NDArray<dtype> subtract(const NDArray<dtype> &a, const NDArray<dtype> &b) {
        return a - b;
    }
    template <typename dtype>
    void subtract(const NDArray<dtype> &a, const NDArray<dtype> &b, NDArray<dtype> &out) {
        out = a - b;
    }
    template <typename dtype>
    NDArray<dtype> subs(const NDArray<dtype> &a, const NDArray<dtype> &b) {
        return substract(a, b);
    }
    template <typename dtype>
    void subs(const NDArray<dtype> &a, const NDArray<dtype> &b, NDArray<dtype> &out) {
        substract(a, b, out);
    }

    template <typename dtype>
    NDArray<dtype> multiply(const NDArray<dtype> &a, const NDArray<dtype> &b) {
        return a * b;
    }
    template <typename dtype>
    void multiply(const NDArray<dtype> &a, const NDArray<dtype> &b, NDArray<dtype> &out) {
        out = a * b;
    }
    template <typename dtype>
    NDArray<dtype> mul(const NDArray<dtype> &a, const NDArray<dtype> &b) {
        return multiply(a, b);
    }
    template <typename dtype>
    void mul(const NDArray<dtype> &a, const NDArray<dtype> &b, NDArray<dtype> &out) {
        multiply(a, b, out);
    }

    template <typename dtype>
    NDArray<dtype> divide(const NDArray<dtype> &a, const NDArray<dtype> &b) {
        return a / b;
    }
    template <typename dtype>
    void divide(const NDArray<dtype> &a, const NDArray<dtype> &b, NDArray<dtype> &out) {
        out = a / b;
    }
    template <typename dtype>
    NDArray<dtype> div(const NDArray<dtype> &a, const NDArray<dtype> &b) {
        return divide(a, b);
    }
    template <typename dtype>
    void div(const NDArray<dtype> &a, const NDArray<dtype> &b, NDArray<dtype> &out) {
        divide(a, b, out);
    }

    template <typename dtype>
    NDArray<dtype> power(const NDArray<dtype> &a, const NDArray<dtype> &b) {
        return a.executeElementWise(PowOp<dtype>{}, &b, nullptr);
    }
    template <typename dtype>
    void power(const NDArray<dtype> &a, const NDArray<dtype> &b, NDArray<dtype> &out) {
        a.executeElementWise(PowOp<dtype>{}, &b, &out);
    }
    template <typename dtype>
    NDArray<dtype> pow(const NDArray<dtype> &a, const NDArray<dtype> &b) {
        return power(a, b);
    }
    template <typename dtype>
    void pow(const NDArray<dtype> &a, const NDArray<dtype> &b, NDArray<dtype> &out) {
        power(a, b, out);
    }

    // Cross-type handling for binary fns
    NDARRAY_BINARY_CROSS_FN(add);
    NDARRAY_BINARY_CROSS_FN(subtract);
    NDARRAY_BINARY_CROSS_FN(subs);
    NDARRAY_BINARY_CROSS_FN(multiply);
    NDARRAY_BINARY_CROSS_FN(mul);
    NDARRAY_BINARY_CROSS_FN(divide);
    NDARRAY_BINARY_CROSS_FN(div);
    NDARRAY_BINARY_CROSS_FN(power);
    NDARRAY_BINARY_CROSS_FN(pow);
    //


    /// Math unaries (exp, log, trigs, arctrigs, sigmoid, etc.)
    template <typename dtype>
    NDArray<dtype> raise(const NDArray<dtype> &a, const dtype exponent) {
        return a.executeElementWise(RaiseOp<dtype>{exponent}, nullptr, nullptr);
    }
    template <typename dtype>
    void raise(const NDArray<dtype> &a, const dtype exponent, NDArray<dtype> &out) {
        a.executeElementWise(RaiseOp<dtype>{exponent}, nullptr, &out);
    }
    template <typename dtype>
    NDArray<dtype> exp(const NDArray<dtype> &a,
        const dtype base=static_cast<dtype>(std::exp(1.0))) {
        return a.executeElementWise(ExpOp<dtype>{base}, nullptr, nullptr);
    }
    template <typename dtype>
    void exp(const NDArray<dtype> &a,
        NDArray<dtype> &out,
        const dtype base=static_cast<dtype>(std::exp(1.0))) {
        a.executeElementWise(ExpOp<dtype>{base}, nullptr, &out);
    }

    template <typename dtype>
    NDArray <dtype> log(const NDArray<dtype> &a,
        const dtype base=static_cast<dtype>(std::exp(1.0))) {
        return a.executeElementWise(LogOp<dtype>{base}, nullptr, nullptr);
    }
    template <typename dtype>
    void log(const NDArray<dtype> &a,
        NDArray<dtype> &out,
        const dtype base=static_cast<dtype>(std::exp(1.0))) {
        a.executeElementWise(LogOp<dtype>{base}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> sin(const NDArray<dtype> &a) {
        return a.executeElementWise(SinOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void sin(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(SinOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> cos(const NDArray<dtype> &a) {
        return a.executeElementWise(CosOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void cos(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(CosOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> tan(const NDArray<dtype> &a) {
        return a.executeElementWise(TanOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void tan(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(TanOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> cot(const NDArray<dtype> &a) {
        return a.executeElementWise(CotOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void cot(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(CotOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> asin(const NDArray<dtype> &a) {
        return a.executeElementWise(ASinOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void asin(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(ASinOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> acos(const NDArray<dtype> &a) {
        return a.executeElementWise(ACosOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void acos(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(ACosOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> atan(const NDArray<dtype> &a) {
        return a.executeElementWise(ATanOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void atan(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(ATanOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> acot(const NDArray<dtype> &a) {
        return a.executeElementWise(ACotOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void acot(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(ACotOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> sigmoid(const NDArray<dtype> &a) {
        return a.executeElementWise(SigmoidOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void sigmoid(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(SigmoidOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> abs(const NDArray<dtype> &a) {
        return a.executeElementWise(AbsOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void abs(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(AbsOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> sign(const NDArray<dtype> &a) {
        return a.executeElementWise(SignOp<dtype>{}, nullptr, nullptr);
    }
    template <typename dtype>
    void sign(const NDArray<dtype> &a, NDArray<dtype> &out) {
        a.executeElementWise(SignOp<dtype>{}, nullptr, &out);
    }

    template <typename dtype>
    NDArray<dtype> clip(const NDArray<dtype> &a, dtype low, dtype high) {
        return a.executeElementWise(ClipOp<dtype>{low, high}, nullptr, nullptr);
    }
    template <typename dtype>
    void clip(const NDArray<dtype> &a, dtype low, dtype high, NDArray<dtype> &out) {
        a.executeElementWise(ClipOp<dtype>{low, high}, nullptr, &out);
    }


}


#endif //NEUROCORE_NDARRAY_CUH
