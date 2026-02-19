//
// Created by Bujor Ionut Raul on 22.12.2025.
//


#ifndef NEUROCORE_SLICES_H
#define NEUROCORE_SLICES_H

#include <vector>
#include <initializer_list>
#include "core/exceptions.h"

// Forward declarations to break circular imports
template <typename dtype>
class NDArray;
template<typename dtype>
class Tensor;

struct Shape {
    std::vector<int> dims;
    Shape() = default;
    explicit Shape(const std::initializer_list<int> &shape) : dims(shape) {}
    explicit Shape(const std::vector<int> &shapeVec) : dims(shapeVec) {}
};

class Slice {
    int start = 0, stop = 0, step = 0;
    std::vector<int> indices;
    bool fromIndices;
public:
    Slice(int start=0, int stop=-1, int step=1) : start(start), stop(stop), step(step), fromIndices(false) {};
    Slice(const std::vector<int> &indices): indices(indices), fromIndices(true) {};
    Slice(const NDArray<int> &indices);
    Slice(const Tensor<int> &indices);
    int size() const;
    void normalizeEnd(int shapeSize);
    int getStart() const {return start;}
    int getStop() const {return stop;}
    int getStep() const {return step;}
    std::vector<int> getIndices() const {return indices;}
    bool isFromIndices() const {return fromIndices;}
    // Iterator
    class Iterator {
        int current;
        int stop;
        int step;
        const std::vector<int>* indicesVec;
        size_t indexPos;
    public:
        Iterator(int current, int stop, int step) : current(current), stop(stop), step(step), indicesVec(nullptr), indexPos(0) {};
        Iterator(const std::vector<int>* indicesVec, size_t indexPos) : current(0), stop(0), step(0), indicesVec(indicesVec), indexPos(indexPos) {
            if (indicesVec && indexPos < indicesVec->size()) current = (*indicesVec)[indexPos];
        };
        // Dereference operator
        int operator*() const;
        // Pre-increment operator
        Iterator& operator++();
        // Post increment operator
        Iterator operator++(int);
        // Inequality operator
        bool operator!=(const Iterator &other) const;
        // Equality operator
        bool operator==(const Iterator &other) const;
    };
    Iterator begin() const;
    Iterator end() const;
};

#endif //NEUROCORE_SLICES_H
