#include <cuda_runtime.h>
#include "functions.h"

namespace xrecon {

void backward3d_cuda(
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
        backward3d_parallel_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanFlat:
        backward3d_flat_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanArc:
        backward3d_arc_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    default:
        break;
    }
}

void weighted_backward3d_cuda(
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
        weighted_backward3d_parallel_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanFlat:
        weighted_backward3d_flat_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanArc:
        weighted_backward3d_arc_cuda(image, proj, ang, vol, geo, num_batch, num_view);
        break;
    default:
        break;
    }
}

void backward3d_t_cuda(
    float* proj,
    float* image,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view)
{
    switch (geo.type) {
    case GeometryType::Parallel:
        backward3d_t_parallel_cuda(proj, image, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanFlat:
        backward3d_t_flat_cuda(proj, image, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanArc:
        backward3d_t_arc_cuda(proj, image, ang, vol, geo, num_batch, num_view);
        break;
    default:
        break;
    }
}

void weighted_backward3d_t_cuda(
    float* proj,
    float* image,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view)
{
    switch (geo.type) {
    case GeometryType::Parallel:
        weighted_backward3d_t_parallel_cuda(proj, image, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanFlat:
        weighted_backward3d_t_flat_cuda(proj, image, ang, vol, geo, num_batch, num_view);
        break;
    case GeometryType::FanArc:
        weighted_backward3d_t_arc_cuda(proj, image, ang, vol, geo, num_batch, num_view);
        break;
    default:
        break;
    }
}

}
