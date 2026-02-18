#ifndef NEUROCORE_UTILS_H
#define NEUROCORE_UTILS_H


#include <cstddef>
#include <type_traits>
#include <vector>

namespace utils {

std::vector<int> getSizeAgnosticKernelConfigParams();

int getNBlocks(int n, int threads);

int flatToStridedIndex(int idx, int offset,
                       const std::vector<int>& strides,
                       int ndim,
                       const std::vector<int>& shape);

void cudaFreeMulti(const std::vector<void*>& cuda_ptrs);




/////////////////////////////////// RECURSIVE VECTORS HELPERS //////////////////////////
namespace detail {
template<typename T>
struct remove_cvref {
    typedef typename std::remove_cv<
        typename std::remove_reference<T>::type
    >::type type;
};


template<typename T>
struct NDims {
    static const size_t value = 0;
};
template<typename T>
struct NDims<std::vector<T>> {
    static const size_t value = 1 + NDims<T>::value;
};
template<typename T>
struct NDimsDecay {
    static const size_t value =
        NDims<typename remove_cvref<T>::type>::value;
};
} // namespace detail

template<typename T>
constexpr size_t nestedVectorRank(T&&) {
    return detail::NDimsDecay<T>::value;
}


template<typename T>
std::vector<int> nestedVectorShape(const T&) {
    return {};
}
template<typename T>
std::vector<int> nestedVectorShape(const std::vector<T>& v) {
    std::vector<int> shape;
    shape.push_back(static_cast<int>(v.size()));
    if (!v.empty()) {
        std::vector<int> inner = nestedVectorShape(v[0]);
        shape.insert(shape.end(), inner.begin(), inner.end());
    }
    return shape;
}


template<typename T>
size_t nestedVectorElementCount(const T&) {
    return 1;
}
template<typename T>
size_t nestedVectorElementCount(const std::vector<T>& v) {
    size_t sum = 0;
    for (size_t i = 0; i < v.size(); ++i)
        sum += nestedVectorElementCount(v[i]);
    return sum;
}


template<typename T>
bool isNestedVectorHomogeneous(const T&) {
    return true;
}
template<typename T>
bool isNestedVectorHomogeneous(const std::vector<T>& v) {
    if (v.empty()) return true;
    const std::vector<int> expectedShape = nestedVectorShape(v[0]);
    for (size_t i = 0; i < v.size(); ++i) {
        // Check that this element's shape matches the first element's shape
        if (nestedVectorShape(v[i]) != expectedShape)
            return false;
        // Recursively verify the element itself is homogeneous
        if (!isNestedVectorHomogeneous(v[i]))
            return false;
    }
    return true;
}



template<typename dtype>
void flattenNestedVectorToStrided(const std::vector<dtype>& input, dtype* data, size_t& idx,
                                  const int offset, const std::vector<int>& strides,
                                  const std::vector<int>& shape, const int ndim) {
    for (size_t i = 0; i < input.size(); ++i) {
        data[flatToStridedIndex(static_cast<int>(idx), offset, strides, ndim, shape)] = input[i];
        ++idx;
    }
}
template<typename T, typename dtype>
void flattenNestedVectorToStrided(const std::vector<std::vector<T>>& input, dtype* data, size_t& idx,
                                  const int offset, const std::vector<int>& strides,
                                  const std::vector<int>& shape, const int ndim) {
    for (size_t i = 0; i < input.size(); ++i)
        flattenNestedVectorToStrided(input[i], data, idx, offset, strides, shape, ndim);
}
template<typename T, typename dtype>
void flattenNestedVectorToStrided(const std::vector<T>& input, dtype* data,
                                  const int offset, const std::vector<int>& strides,
                                  const std::vector<int>& shape, const int ndim) {
    size_t idx = 0;
    flattenNestedVectorToStrided(input, data, idx, offset, strides, shape, ndim);
}

} // namespace utils

#endif //NEUROCORE_UTILS_H