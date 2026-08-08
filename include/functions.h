#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include "types.h"


namespace xrecon {

using MatrixCoeffEmitter = void (*)(int row, int col, float value, void* user);

void forward2d_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void backward2d_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void weighted_backward2d_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void forward2d_t_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void backward2d_t_cuda(
    float* proj,
    float* image,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void weighted_backward2d_t_cuda(
    float* proj,
    float* image,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void forward2d_cpu(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void forward2d_t_cpu(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void system_matrix2d_cpu(
    std::vector<int64_t>& rows,
    std::vector<int64_t>& cols,
    std::vector<float>& values,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_view
);

void distance_driven2d_cpu(
    MatrixCoeffEmitter emit,
    void* user,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_view
);

void system_matrix3d_cpu(
    std::vector<int64_t>& rows,
    std::vector<int64_t>& cols,
    std::vector<float>& values,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_view
);

void distance_driven3d_cpu(
    MatrixCoeffEmitter emit,
    void* user,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_view
);

void backward2d_cpu(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void weighted_backward2d_cpu(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void backward2d_t_cpu(
    float* proj,
    float* image,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void weighted_backward2d_t_cpu(
    float* proj,
    float* image,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
);

void forward3d_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view
);

void forward3d_t_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view
);

void backward3d_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view
);

void weighted_backward3d_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view
);

void backward3d_t_cuda(
    float* proj,
    float* image,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view
);

void weighted_backward3d_t_cuda(
    float* proj,
    float* image,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view
);

void forward3d_flat_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void forward3d_arc_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void forward3d_t_flat_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void forward3d_t_arc_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void backward3d_flat_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void backward3d_arc_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void weighted_backward3d_flat_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void weighted_backward3d_arc_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void backward3d_t_flat_cuda(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void backward3d_t_arc_cuda(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void weighted_backward3d_t_flat_cuda(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void weighted_backward3d_t_arc_cuda(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void forward3d_parallel_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void forward3d_t_parallel_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void backward3d_parallel_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void weighted_backward3d_parallel_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void backward3d_t_parallel_cuda(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void weighted_backward3d_t_parallel_cuda(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);

void forward3d_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void forward3d_t_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void backward3d_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void weighted_backward3d_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void backward3d_t_cpu(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);
void weighted_backward3d_t_cpu(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view);

}
