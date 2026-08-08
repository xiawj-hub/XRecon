#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include "types.h"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__); \
        exit(EXIT_FAILURE); \
    }


struct CudaTexture3D {
    cudaArray_t cuArray;
    cudaTextureObject_t texObj;

    CudaTexture3D() : cuArray(nullptr), texObj(0) {}

    void create(const float* srcData, int width, int height, int depth) {

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();


        cudaExtent extent = make_cudaExtent(width, height, depth);
        CUDA_CHECK(cudaMalloc3DArray(&cuArray, &channelDesc, extent));


        cudaMemcpy3DParms copyParams = {0};
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToDevice;
        copyParams.srcPtr = make_cudaPitchedPtr((void*)srcData, width * sizeof(float), width, height);
        copyParams.dstArray = cuArray;
        CUDA_CHECK(cudaMemcpy3D(&copyParams));


        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;


        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeBorder;
        texDesc.addressMode[1] = cudaAddressModeBorder;
        texDesc.addressMode[2] = cudaAddressModeBorder;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;


        CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    }

    void destroy() {
        if (texObj) {
            CUDA_CHECK(cudaDestroyTextureObject(texObj));
            texObj = 0;
        }
        if (cuArray) {
            CUDA_CHECK(cudaFreeArray(cuArray));
            cuArray = nullptr;
        }
    }

    ~CudaTexture3D() {
        destroy();
    }
};