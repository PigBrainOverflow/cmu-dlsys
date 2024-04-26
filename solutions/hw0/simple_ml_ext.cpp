#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

// additional
#include <vector>
#include <numeric>

namespace py = pybind11;

void copyToMatrix(const float *X, std::vector<std::vector<float>>& mX, int nrow, int ncol)
{
    for (int i = 0; i < nrow; ++i) {
        float* dest = mX[i].data();
        const float* src = X + i * ncol;
        std::memcpy(dest, src, ncol * sizeof(float));
    }
}

std::vector<std::vector<float>> matrixMultiply(const std::vector<std::vector<float>>& m1, const std::vector<std::vector<float>>& m2)
{
    int rows = m1.size();
    int cols = m2[0].size();
    int n = m2.size();

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    return result;
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix)
{
    if (matrix.empty() || matrix[0].empty()) return {};

    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<float>> transposedMatrix(cols, std::vector<float>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposedMatrix[j][i] = matrix[i][j];
        }
    }

    return transposedMatrix;
}

void softmax_regression_batch_cpp(
    const float *X, const unsigned char *y,
    float *theta, size_t m, size_t n, size_t k, float lr
)
{
    std::vector<std::vector<float>> mX(m, std::vector<float>(n));
    copyToMatrix(X, mX, m, n);
    std::vector<std::vector<float>> mTheta(n, std::vector<float>(k));
    copyToMatrix(theta, mTheta, n, k);

    // @
    auto mXTheta = matrixMultiply(mX, mTheta);

    // exp
    for (auto& r : mXTheta) {
        for (auto& e : r) {
            e = std::exp(e);
        }
    }

    // norm
    for (auto& r : mXTheta) {
        auto sum = std::accumulate(r.begin(), r.end(), 0.0f);
        for (auto& e : r) {
            e /= sum;
        }
    }

    auto& mZ = mXTheta;
    for (size_t i = 0; i < m; ++i) {
        mZ[i][y[i]] -= 1.0;
    }

    auto mG = matrixMultiply(transpose(mX), mZ);
    int i = 0;
    for (auto& r : mG) {
        for (auto& e : r) {
            theta[i++] -= e * lr / m;
        }
    }

}

void softmax_regression_epoch_cpp(
    const float *X, const unsigned char *y,
    float *theta, size_t m, size_t n, size_t k,
    float lr, size_t batch
)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    auto n_batch = m / batch;
    for (auto i = 0ull; i < n_batch; ++i) {
        auto offset = batch * i;
        softmax_regression_batch_cpp(X + offset * n, y + offset, theta, batch, n, k, lr);
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
        [](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
            static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
