#include <cuda_runtime.h>
#include "functions.h"

namespace xrecon {

void forward3d_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view)
{
    switch (geo.type) {
    case GeometryType::Parallel:
        forward3d_parallel_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanFlat:
        forward3d_flat_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanArc:
        forward3d_arc_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    default:
        break;
    }
}

void forward3d_t_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view)
{
    switch (geo.type) {
    case GeometryType::Parallel:
        forward3d_t_parallel_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanFlat:
        forward3d_t_flat_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanArc:
        forward3d_t_arc_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    default:
        break;
    }
}

}
