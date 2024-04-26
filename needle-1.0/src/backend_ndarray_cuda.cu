#include <cuda_runtime.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
#define TILE 4
typedef float scalar_t; // float32


struct CudaArray {
    CudaArray(const size_t size) {
        cudaError_t err = cudaMalloc(&ptr, size * sizeof(scalar_t));
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
        this->size = size;
    }
    ~CudaArray() { cudaFree(ptr); }
    size_t py_ptr() { return (size_t)ptr; }

    scalar_t* ptr;
    size_t size;
};

struct CudaDims {
    dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

struct GPUInfo {
    std::string name;
    int memoryClockRate;
    int memoryBusWidth;
    double memoryBandwidth;
    unsigned long totalGlobalMemory;
    unsigned long sharedMemPerBlock;
    int maxThreadsPerBlock;
};

std::vector<GPUInfo> manage_device() {
    std::vector<GPUInfo> gpuInfos;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        GPUInfo info;
        info.name = deviceProp.name;
        info.memoryClockRate = deviceProp.memoryClockRate;
        info.memoryBusWidth = deviceProp.memoryBusWidth;
        info.memoryBandwidth = 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6;
        info.totalGlobalMemory = deviceProp.totalGlobalMem;
        info.sharedMemPerBlock = deviceProp.sharedMemPerBlock;
        info.maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

        gpuInfos.push_back(info);
    }

    return gpuInfos;
}

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

__device__ int32_t contiguousIndexToRealIndex(size_t i, const CudaVec& shape, const CudaVec& strides) {
    int32_t real_index = 0;
    for (size_t dim_index = shape.size - 1; dim_index > 0; --dim_index) {
        real_index += strides.data[dim_index] * (i % shape.data[dim_index]);
        i /= shape.data[dim_index];
    }
    return real_index + strides.data[0] * i;
}

__global__ void CompactKernel(
    const scalar_t* a,
    scalar_t* out,
    size_t size,
    CudaVec shape,
    CudaVec strides,
    size_t offset
) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    /// BEGIN SOLUTION

    if (gid < size) {
        auto real_index = contiguousIndexToRealIndex(gid, shape, strides);
        out[gid] = a[offset + real_index];
    }

    /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(
    const scalar_t* a,
    scalar_t* out,
    size_t size,
    CudaVec shape,
    CudaVec strides,
    size_t offset
) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        auto real_index = contiguousIndexToRealIndex(gid, shape, strides);
        out[offset + real_index] = a[gid];
    }
}

void EwiseSetitem(
    const CudaArray& a,
    CudaArray* out,
    std::vector<int32_t> shape,
    std::vector<int32_t> strides,
    size_t offset
) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
    /// BEGIN SOLUTION

    size_t size = 1;
    for (auto s : shape) {
        size *= s;
    }
    CudaDims dim = CudaOneDim(size);
    EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);

    /// END SOLUTION
}

__global__ void ScalarSetitemKernel(
    scalar_t val,
    scalar_t* out,
    size_t size,
    CudaVec shape,
    CudaVec strides,
    size_t offset
) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        auto real_index = contiguousIndexToRealIndex(gid, shape, strides);
        out[offset + real_index] = val;
    }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
    /// BEGIN SOLUTION

    CudaDims dim = CudaOneDim(size);
    ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);

    /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// BEGIN

////////////////////
// Macro
////////////////////

// expr should be in form: a[gid] op b[gid]
#define DECLARE_EWISE_KERNEL(name, expr)\
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {\
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
    if (gid < size) {\
        out[gid] = expr;\
    }\
}

// expr should be in form: a[gid] op val
#define DECLARE_SCALAR_KERNEL(name, expr)\
__global__ void Scalar##name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {\
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
    if (gid < size) {\
        out[gid] = expr;\
    }\
}

#define DECLARE_EWISE_UNARY_KERNEL(name, expr)\
__global__ void Ewise##name##Kernel(const scalar_t* a, scalar_t* out, size_t size) {\
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;\
    if (gid < size) {\
        out[gid] = expr;\
    }\
}

#define DECLARE_EWISE(name)\
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out) {\
    CudaDims dim = CudaOneDim(out->size);\
    Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);\
}

#define DECLARE_SCALAR(name)\
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out) {\
  CudaDims dim = CudaOneDim(out->size);\
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);\
}

#define DECLARE_EWISE_UNARY(name)\
void Ewise##name(const CudaArray& a, CudaArray* out) {\
    CudaDims dim = CudaOneDim(out->size);\
    Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);\
}

////////////////////
// Add
////////////////////
DECLARE_EWISE_KERNEL(Add, a[gid] + b[gid])
DECLARE_SCALAR_KERNEL(Add, a[gid] + val)
DECLARE_EWISE(Add)
DECLARE_SCALAR(Add)

////////////////////
// Mul
////////////////////
DECLARE_EWISE_KERNEL(Mul, a[gid] * b[gid])
DECLARE_SCALAR_KERNEL(Mul, a[gid] * val)
DECLARE_EWISE(Mul)
DECLARE_SCALAR(Mul)

////////////////////
// Div
////////////////////
DECLARE_EWISE_KERNEL(Div, a[gid] / b[gid])
DECLARE_SCALAR_KERNEL(Div, a[gid] / val)
DECLARE_EWISE(Div)
DECLARE_SCALAR(Div)

////////////////////
// Power
////////////////////
DECLARE_SCALAR_KERNEL(Power, pow(a[gid], val))
DECLARE_SCALAR(Power)

////////////////////
// Maximum
////////////////////
DECLARE_EWISE_KERNEL(Maximum, max(a[gid], b[gid]))
DECLARE_SCALAR_KERNEL(Maximum, max(a[gid], val))
DECLARE_EWISE(Maximum)
DECLARE_SCALAR(Maximum)

////////////////////
// Eq
////////////////////
DECLARE_EWISE_KERNEL(Eq, static_cast<scalar_t>(a[gid] == b[gid]))
DECLARE_SCALAR_KERNEL(Eq, static_cast<scalar_t>(a[gid] == val))
DECLARE_EWISE(Eq)
DECLARE_SCALAR(Eq)

////////////////////
// Ge
////////////////////
DECLARE_EWISE_KERNEL(Ge, static_cast<scalar_t>(a[gid] >= b[gid]))
DECLARE_SCALAR_KERNEL(Ge, static_cast<scalar_t>(a[gid] >= val))
DECLARE_EWISE(Ge)
DECLARE_SCALAR(Ge)

////////////////////
// Log & Exp & Tanh
////////////////////
DECLARE_EWISE_UNARY_KERNEL(Log, log(a[gid]))
DECLARE_EWISE_UNARY(Log)
DECLARE_EWISE_UNARY_KERNEL(Exp, exp(a[gid]))
DECLARE_EWISE_UNARY(Exp)
DECLARE_EWISE_UNARY_KERNEL(Tanh, tanh(a[gid]))
DECLARE_EWISE_UNARY(Tanh)

// END

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


///////////////////
// Matmul
///////////////////

  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

// BEGIN
__global__ void MatmulKernel(
    const scalar_t* a,
    const scalar_t* b,
    scalar_t* out,
    size_t M, size_t N, size_t P
) {
    __shared__ scalar_t a_block[TILE][TILE], b_block[TILE][TILE];
    scalar_t out_val = 0.0;
    // __shared__ scalar_t out_block[TILE][TILE];
    size_t i_start = blockIdx.x * TILE, j_start = blockIdx.y * TILE;
    size_t i_end = min(i_start + TILE, M), j_end = min(j_start + TILE, P);
    size_t ti = i_start + threadIdx.x, tj = j_start + threadIdx.y, fetch_by = j_start + threadIdx.x;

    // // initialize out_block
    // out_block[threadIdx.x][threadIdx.y] = 0.0;

    for (size_t k_start = 0; k_start < N; k_start += TILE) {
        // cooperative fetch
        __syncthreads();
        size_t k_end = min(k_start + TILE, N);
        size_t fetch_ay = k_start + threadIdx.y;    // fetch_ay = fetch_bx
        if (fetch_ay < k_end) {
            // fetch_ax = ti
            a_block[threadIdx.x][threadIdx.y] = (ti < i_end) ? a[ti * N + fetch_ay] : 0.0;
            b_block[threadIdx.y][threadIdx.x] = (fetch_by < j_end) ? b[fetch_ay * P + fetch_by] : 0.0;
        }
        else {
            a_block[threadIdx.x][threadIdx.y] = b_block[threadIdx.y][threadIdx.x] = 0.0;
        }

        // compute
        __syncthreads();
        for (size_t k = 0; k < TILE; ++k) {
            out_val += a_block[threadIdx.x][k] * b_block[k][threadIdx.y];
        }
    }

    // copy back
    if (ti < i_end && tj < j_end) {
        out[ti * P + tj] = out_val;
    }
}

void Matmul(
    const CudaArray& a,
    const CudaArray& b,
    CudaArray* out,
    uint32_t M, uint32_t N, uint32_t P
) {
    // assign each block of `out` (TILE * TILE) to a 2-dim block
    // blockIdx.x and blockIdx.y
    // for each block, launch TILE * TILE threads
    auto grid_dim = dim3((M + TILE - 1) / TILE, (P + TILE - 1) / TILE, 1);
    auto block_dim = dim3(TILE, TILE, 1);
    MatmulKernel<<<grid_dim, block_dim>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

// END

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(
    const scalar_t* a,
    scalar_t* out,
    size_t size,
    size_t reduce_size
) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t start = gid * reduce_size;
        scalar_t max_val = a[start];
        for (size_t i = 1; i < reduce_size; ++i) {
            max_val = max(max_val, a[start + i]);
        }
        out[gid] = max_val;
    }
}

void ReduceMax(
    const CudaArray& a,
    CudaArray* out,
    size_t reduce_size
) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
    /// BEGIN SOLUTION

    CudaDims dim = CudaOneDim(out->size);
    ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);

    /// END SOLUTION
}

__global__ void ReduceSumKernel(
    const scalar_t* a,
    scalar_t* out,
    size_t size,
    size_t reduce_size
) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        size_t start = gid * reduce_size;
        scalar_t sum = 0;
        for (size_t i = 0; i < reduce_size; ++i) {
            sum += a[start + i];
        }
        out[gid] = sum;
    }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
    /// BEGIN SOLUTION

    CudaDims dim = CudaOneDim(out->size);
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);

    /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(backend_ndarray_cuda, m) {
    namespace py = pybind11;
    using namespace needle;
    using namespace cuda;

    m.attr("__device_name__") = "cuda";
    m.attr("__tile_size__") = TILE;

    // define python class Array as CudaArray
    py::class_<CudaArray>(m, "Array")
        .def(py::init<size_t>(), py::return_value_policy::take_ownership)
        .def_readonly("size", &CudaArray::size)
        .def("ptr", &CudaArray::py_ptr);

    // define method: copy from cuda to numpy
    m.def(
        "to_numpy",
        [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides) {
            // transform strides
            std::vector<size_t> numpy_strides = strides;
            std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(), [](size_t& c) { return c * sizeof(scalar_t); });

            // from device to host
            scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * sizeof(scalar_t));
            if (host_ptr == nullptr) {
                throw std::bad_alloc();
            }
            cudaError_t error = cudaMemcpy(host_ptr, a.ptr, a.size * sizeof(scalar_t), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(err));
            }

            // from host to numpy
            py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
            return py::array_t<scalar_t>(shape, numpy_strides, host_ptr, deallocate_buffer);
        }
    );

    // define method: copy from numpy to cuda
    m.def(
        "from_numpy",
        [](py::array_t<scalar_t> a, CudaArray* out) {
            cudaError_t error = cudaMemcpy(out->ptr, a.request().ptr, out->size * sizeof(scalar_t), cudaMemcpyHostToDevice);
            if (error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }
    );

    m.def("fill", Fill);
    m.def("compact", Compact);
    m.def("ewise_setitem", EwiseSetitem);
    m.def("scalar_setitem", ScalarSetitem);
    m.def("ewise_add", EwiseAdd);
    m.def("scalar_add", ScalarAdd);

    m.def("ewise_mul", EwiseMul);
    m.def("scalar_mul", ScalarMul);
    m.def("ewise_div", EwiseDiv);
    m.def("scalar_div", ScalarDiv);
    m.def("scalar_power", ScalarPower);

    m.def("ewise_maximum", EwiseMaximum);
    m.def("scalar_maximum", ScalarMaximum);
    m.def("ewise_eq", EwiseEq);
    m.def("scalar_eq", ScalarEq);
    m.def("ewise_ge", EwiseGe);
    m.def("scalar_ge", ScalarGe);

    m.def("ewise_log", EwiseLog);
    m.def("ewise_exp", EwiseExp);
    m.def("ewise_tanh", EwiseTanh);

    m.def("matmul", Matmul);

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}
