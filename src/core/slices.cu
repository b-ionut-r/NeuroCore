//
// Created by Bujor Ionut Raul on 22.12.2025.
//

#include <stdexcept>
#include "core/ndarray.cuh"
#include "core/tensor.cuh"
#include "core/slices.h"
#include "core/exceptions.h"



int Shape::prod() const {
    int prod = 1;
    for (int dim : dims) {
        prod *= dim;
    }
    return prod;
}

std::vector<ReshapeMapping> Shape::inferReshape(const Shape &newShape) const {
    const std::vector<int> &original = this->getDims(), &final = newShape.getDims();
    std::vector<ReshapeMapping> res;
    if (this->prod() != newShape.prod())
        throw ShapeMismatchException("Bad new shape. Same no of elements required");
    int i = 0, j = 0;
    while (i < (int)original.size() && j < (int)final.size()) {
        if (original[i] == final[j]) {
            res.push_back({{i}, {j}});
            i++; j++; continue;
        }

        // Collapse path: multiple original dims -> one final dim
        {
            std::vector<int> axes;
            long long prod = 1;
            int iCopy = i;
            while (iCopy < (int)original.size() && prod < final[j]) {
                prod *= original[iCopy];
                axes.push_back(iCopy);
                iCopy++;
            }
            if (prod == final[j]) {
                res.push_back({axes, {j}});
                i = iCopy; j++;
                continue;
            }
        }

        // Expansion path: one original dim -> multiple final dims
        {
            std::vector<int> axes;
            long long prod = 1;
            int jCopy = j;
            while (jCopy < (int)final.size() && prod < original[i]) {
                prod *= final[jCopy];
                axes.push_back(jCopy);
                jCopy++;
            }
            if (prod == original[i]) {
                res.push_back({{i}, axes});
                i++; j = jCopy;
                continue;
            }
        }

        throw ShapeMismatchException("Cannot reshape to this shape.");
    }
    return res;
}




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

std::vector<int> Slice::getIndices() const {
    if (fromIndices)
        return indices;
    std::vector<int> indices(size());
    for (auto idx: *this)
        indices.push_back(idx);
    return indices;
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
