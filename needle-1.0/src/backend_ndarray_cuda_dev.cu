#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

namespace py = pybind11;

namespace needle {
namespace cuda {


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

}   // end of namespace cuda
}   // end of namespace needle

PYBIND11_MODULE(cuda_info, m) {
    m.def("manage_device", &manage_device);
}
