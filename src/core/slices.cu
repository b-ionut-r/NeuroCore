//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#include <stdexcept>
#include "core/ndarray.cuh"
#include "core/tensor.cuh"
#include "core/slices.h"

Slice::Slice(const NDArray<int> &indices) {
    if (indices.getNDim() != 1)
        throw NDimMismatchException("Indices must be array of 1 dimension.");
    this->indices = indices.toVector();
    this->fromIndices = true;
}

Slice::Slice(const Tensor<int> &indices) {
    if (indices.data().getNDim() != 1)
        throw NDimMismatchException("Indices must be array of 1 dimension.");
    this->indices = indices.data().toVector();
    this->fromIndices = true;
}


int Slice::size() const {
    if (fromIndices) return indices.size();
    if (step > 0) {
        if (stop <= start) return 0;
        return (stop - start + step - 1) / step; // ceil
    } else if (step < 0) {
        if (stop >= start) return 0; //for negative step
        return (start - stop - step - 1) / (-step); // ceil
    } else {
        return 0;
    }
}

void Slice::normalizeEnd(int shapeSize) {
    if (fromIndices)
        throw std::logic_error("Bad method. Can't work with indices slice.");
    if (stop < 0) {
        stop += shapeSize;
    }
}

int Slice::Iterator::operator*() const {
    if (indicesVec) {
        return (*indicesVec)[indexPos];
    }
    return current;
}

// Pre-increment operator
Slice::Iterator& Slice::Iterator::operator++() {
    if (indicesVec) {
        indexPos++;
        if (indexPos < indicesVec->size()) {
            current = (*indicesVec)[indexPos];
        }
    } else {
        current += step;
    }
    return *this;
}

// Post increment operator
Slice::Iterator Slice::Iterator::operator++(int) {
    Iterator tmp = *this;
    ++(*this);
    return tmp;
}

// Inequality operator
bool Slice::Iterator::operator!=(const Iterator &other) const {
    if (indicesVec) {
        return indexPos != other.indexPos;
    }
    // For range-based for loops: check if we've reached or passed the end
    // Also properly compare against other iterator for general use
    if (current == other.current) return false;
    if (step > 0) {
        return current < stop;
    } else {
        return current > stop;
    }
}

// Equality operator
bool Slice::Iterator::operator==(const Iterator &other) const {
    if (indicesVec) {
        return indexPos == other.indexPos;
    }
    return current == other.current;
}

Slice::Iterator Slice::begin() const {
    if (fromIndices) {
        return {&indices, 0};
    }
    return {start, stop, step};
}

Slice::Iterator Slice::end() const {
    if (fromIndices) {
        return {&indices, indices.size()};
    }
    return {stop, stop, step};
}
