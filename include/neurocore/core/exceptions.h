//
// Created by Bujor Ionut Raul on 16.12.2025.
//


#ifndef NEUROCORE_EXCEPTIONS_H
#define NEUROCORE_EXCEPTIONS_H

#include <exception>
#include <string>

class NeuroCoreException : public std::exception {
protected:
    std::string message;
public:
    explicit NeuroCoreException(const std::string &msg) : message("NeuroCore Error: " + msg) {}
    const char* what() const noexcept override {
        return message.c_str();
    }
};

class SizeMismatchException : public NeuroCoreException {
public:
    SizeMismatchException(const std::string &msg)
        : NeuroCoreException("Size Mismatch -> " + msg) {}
};

class NDimMismatchException : public NeuroCoreException {
public:
    NDimMismatchException(const std::string &msg)
        : NeuroCoreException("NDim Mismatch -> " + msg) {}
};


class ShapeMismatchException : public NeuroCoreException {
public:
    ShapeMismatchException(const std::string &msg)
        : NeuroCoreException("Shape Mismatch -> " + msg) {}
};


class IndexingException : public NeuroCoreException {
public:
    IndexingException(const std::string &msg)
        : NeuroCoreException("Indexing/Slicing Error -> " + msg) {}
};

class CudaKernelException : public NeuroCoreException {
public:
    explicit CudaKernelException(const std::string &cudaError)
        : NeuroCoreException("CUDA Kernel Failure -> " + cudaError) {}
};


class BackPropException : public NeuroCoreException {
public:
    explicit BackPropException(const std::string &msg)
        : NeuroCoreException("Backpropagation Error -> " + msg) {}
};

#endif //NEUROCORE_EXCEPTIONS_H
