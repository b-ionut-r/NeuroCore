//
// Created by Bujor Ionut Raul on 11.01.2026.
//

#ifndef NEUROCORE_TYPE_TRAITS_CUH
#define NEUROCORE_TYPE_TRAITS_CUH


#include <cuda/std/type_traits>

template <typename dtype>
class NDArray;
template <typename dtype>
class Tensor;

/////////////////////////////////// RECURSIVE VECTORS HELPERS //////////////////////////
namespace utils {
    template<typename T>
    struct NestedVec {
        std::vector<T> flat;
        std::vector<int> shape;
        NestedVec(T val) : flat{val}, shape{} {}
        NestedVec(std::initializer_list<NestedVec<T>> list) {
            if (list.size() == 0) return;
            shape = {(int)list.size()};
            auto it = list.begin();
            shape.insert(shape.end(), it->shape.begin(), it->shape.end());

            // Check all elements have the same shape as the first
            const std::vector<int> expectedShape = it->shape;
            for (auto &inner : list) {
                if (inner.shape != expectedShape)
                    throw std::invalid_argument(
                        "NestedVec: non-homogeneous initializer — all elements must have the same shape."
                    );
                flat.insert(flat.end(), inner.flat.begin(), inner.flat.end());
            }
        }
    };
}



/// CROSS-TYPE BINARY OPERATORS FOR NDARRAY AND TENSOR
#define NDARRAY_BINARY_CROSS_OP(OP) \
template<typename D1, typename D2> \
auto operator OP(const NDArray<D1> &a, const NDArray<D2> &b){ \
using DOut = cuda::std::common_type_t<D1, D2>; \
return a.template cast<DOut>() OP b.template cast<DOut>(); \
}

#define TENSOR_BINARY_CROSS_OP(OP) \
template<typename D1, typename D2> \
auto operator OP(const Tensor<D1> &a, const Tensor<D2> &b){ \
using DOut = cuda::std::common_type_t<D1, D2>; \
return a.template cast<DOut>() OP b.template cast<DOut>(); \
}


/// CROSS-TYPE BINARY FUNCTION OVERLOADS FOR NDARRAY AND TENSOR
#define NDARRAY_BINARY_CROSS_FN(FNAME) \
template <typename D1, typename D2, \
cuda::std::enable_if_t<!cuda::std::is_same_v<D1, D2>, int> = 0> \
auto FNAME(const NDArray<D1> &a, const NDArray<D2> &b) { \
using DOut = cuda::std::common_type_t<D1, D2>; \
return FNAME(a.template cast<DOut>(), b.template cast<DOut>()); \
} \
template <typename D1, typename D2, \
cuda::std::enable_if_t<!cuda::std::is_same_v<D1, D2>, int> = 0> \
void FNAME(const NDArray<D1> &a, const NDArray<D2> &b, \
NDArray<cuda::std::common_type_t<D1, D2>> &out) { \
using DOut = cuda::std::common_type_t<D1, D2>; \
FNAME(a.template cast<DOut>(), b.template cast<DOut>(), out); \
}


#define TENSOR_BINARY_CROSS_FN(FNAME) \
template <typename D1, typename D2, \
cuda::std::enable_if_t<!cuda::std::is_same_v<D1, D2>, int> = 0> \
auto FNAME(const Tensor<D1> &a, const Tensor<D2> &b) { \
using DOut = cuda::std::common_type_t<D1, D2>; \
return FNAME(a.template cast<DOut>(), b.template cast<DOut>()); \
}


#endif //NEUROCORE_TYPE_TRAITS_CUH