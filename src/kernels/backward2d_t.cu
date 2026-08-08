#include <cuda_runtime.h>
#include "types.h"
#include "utils.h"
#include "xrmath.h"

#define BLOCK_DIM 16

namespace xrecon {

template<bool FBPWEIGHT>
__global__ void fanflat_backward_t_cuda_kernel(
    float* __restrict__ proj,
    const float* __restrict__ img,
    const float* __restrict__ ang,
    const Volume2d vol,
    const Geometry2d geo,
    const int num_view
) {
    const int idx_batch = blockIdx.x;
    const int idx_row   = blockIdx.y * blockDim.x + threadIdx.x;
    const int idx_col   = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_col >= vol.Nx || idx_row >= vol.Ny) return;

    float px = (idx_col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    float py = (idx_row - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    Point2d point(px, py);
    float image_val = img[(idx_batch * vol.Ny + (vol.Ny - 1 - idx_row)) * vol.Nx + idx_col];
    float invNu = 1.0f / geo.Nu;
    float halfNu = geo.Nu * 0.5f;

    for (int idx_view = 0; idx_view < num_view; idx_view++) {
        float sinv = sinf(ang[idx_view]);
        float cosv = cosf(ang[idx_view]);
        Point2d src(-sinv * geo.SOD, cosv * geo.SOD);

        bool useXAxis = (cosv * cosv > 0.5f);
        bool keepDim  = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

        float idxd0 = keepDim ? -halfNu : halfNu;
        float idxd1 = keepDim ? halfNu : -halfNu;

        Point2d det0((idxd0 * geo.du + geo.shift_u) * cosv,
                     (idxd0 * geo.du + geo.shift_u) * sinv);
        Point2d det1((idxd1 * geo.du + geo.shift_u) * cosv,
                     (idxd1 * geo.du + geo.shift_u) * sinv);

        float det_ix = (det1.x - det0.x) * invNu;
        float det_iy = (det1.y - det0.y) * invNu;

        float p0, p1, d0p, d1p, bound0, bound1, b2d;
        if (useXAxis) {
            p0 = map2x(src, Point2d(px - 0.5f * vol.dx, py));
            p1 = map2x(src, Point2d(px + 0.5f * vol.dx, py));
            d0p = map2x(src, det0);
            d1p = map2x(src, det1);
            bound0 = fmaxf(p0, d0p);
            bound1 = fminf(p1, d1p);
            b2d = src.y * bound0 / ((bound0 - src.x) * (sinv / cosv) + src.y);
        } else {
            p0 = map2y(src, Point2d(px, py - 0.5f * vol.dy));
            p1 = map2y(src, Point2d(px, py + 0.5f * vol.dy));
            d0p = map2y(src, det0);
            d1p = map2y(src, det1);
            bound0 = fmaxf(p0, d0p);
            bound1 = fminf(p1, d1p);
            b2d = src.x * bound0 / ((bound0 - src.y) * (cosv / sinv) + src.x);
        }

        int idxd = useXAxis ? floorf((b2d - det0.x) / det_ix)
                            : floorf((b2d - det0.y) / det_iy);
        idxd = max(0, idxd);
        idxd = min(geo.Nu - 1, idxd);

        Point2d det_cur(det0.x + idxd * det_ix, det0.y + idxd * det_iy);
        idxd = keepDim ? idxd : (geo.Nu - 1 - idxd);

        float base = image_val / (p1 - p0);
        if (FBPWEIGHT) base *= pix_weight(src, point, geo.SOD);

        while (bound0 < bound1) {
            Point2d det_next(det_cur.x + det_ix, det_cur.y + det_iy);
            float d1 = useXAxis ? map2x(src, det_next) : map2y(src, det_next);
            d1 = fminf(d1, bound1);
            atomicAdd(
                proj + (idx_batch * num_view + idx_view) * geo.Nu + idxd,
                base * (d1 - bound0) * geo.du
            );
            bound0 = d1;
            det_cur = det_next;
            idxd += keepDim ? 1 : -1;
        }
    }
}

template<bool FBPWEIGHT>
__global__ void fanarc_backward_t_cuda_kernel(
    float* __restrict__ proj,
    const float* __restrict__ img,
    const float* __restrict__ ang,
    const Volume2d vol,
    const Geometry2d geo,
    const int num_view
) {
    const int idx_batch = blockIdx.x;
    const int idx_row   = blockIdx.y * blockDim.x + threadIdx.x;
    const int idx_col   = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_col >= vol.Nx || idx_row >= vol.Ny) return;

    float px = (idx_col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    float py = (idx_row - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    Point2d point(px, py);
    float image_val = img[(idx_batch * vol.Ny + (vol.Ny - 1 - idx_row)) * vol.Nx + idx_col];
    float halfNu = geo.Nu * 0.5f;
    float det_rad = geo.SOD * tanf(geo.du / 2) * 2;

    for (int idx_view = 0; idx_view < num_view; idx_view++) {
        float view = ang[idx_view];
        float sinv = sinf(view);
        float cosv = cosf(view);
        Point2d src(-sinv * geo.SOD, cosv * geo.SOD);

        bool useXAxis = (cosv * cosv > 0.5f);
        bool keepDim  = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

        float det0_ang = keepDim ? (-halfNu * geo.du + geo.shift_u + view)
                                 : ( halfNu * geo.du + geo.shift_u + view);
        float det1_ang = keepDim ? ( halfNu * geo.du + geo.shift_u + view)
                                 : (-halfNu * geo.du + geo.shift_u + view);

        float p0, p1, d0p, d1p, bound0, bound1;
        if (useXAxis) {
            p0 = map2x(src, Point2d(px - 0.5f * vol.dx, py));
            p1 = map2x(src, Point2d(px + 0.5f * vol.dx, py));
            d0p = map2x(src, det0_ang);
            d1p = map2x(src, det1_ang);
            bound0 = fmaxf(p0, d0p);
            bound1 = fminf(p1, d1p);
        } else {
            p0 = map2y(src, Point2d(px, py - 0.5f * vol.dy));
            p1 = map2y(src, Point2d(px, py + 0.5f * vol.dy));
            d0p = map2y(src, det0_ang);
            d1p = map2y(src, det1_ang);
            bound0 = fmaxf(p0, d0p);
            bound1 = fminf(p1, d1p);
        }

        float det_start_ang = detector_axis_angle(src, useXAxis, bound0, geo.shift_u + view);
        float det_start = (det_start_ang - geo.shift_u - view) / geo.du;
        int idxd = floorf(halfNu + det_start);
        idxd = max(0, idxd);
        idxd = min(geo.Nu - 1, idxd);

        float det_cur_ang = (idxd - halfNu) * geo.du + geo.shift_u + view;
        det_cur_ang = keepDim ? det_cur_ang : (det_cur_ang + geo.du);

        float base = image_val / (p1 - p0);
        if (FBPWEIGHT) base /= pix_weight(src, point);

        while (bound0 < bound1) {
            float det_next_ang = keepDim ? (det_cur_ang + geo.du) : (det_cur_ang - geo.du);
            float d1 = useXAxis ? map2x(src, det_next_ang) : map2y(src, det_next_ang);
            d1 = fminf(d1, bound1);
            atomicAdd(
                proj + (idx_batch * num_view + idx_view) * geo.Nu + idxd,
                base * (d1 - bound0) * det_rad
            );
            bound0 = d1;
            det_cur_ang = det_next_ang;
            idxd += keepDim ? 1 : -1;
        }
    }
}

__global__ void parallel_backward_t_cuda_kernel(
    float* __restrict__ proj,
    const float* __restrict__ img,
    const float* __restrict__ ang,
    const Volume2d vol,
    const Geometry2d geo,
    const int num_view
) {
    const int idx_batch = blockIdx.x;
    const int idx_row   = blockIdx.y * blockDim.x + threadIdx.x;
    const int idx_col   = blockIdx.z * blockDim.y + threadIdx.y;
    if (idx_col >= vol.Nx || idx_row >= vol.Ny) return;

    float px = (idx_col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    float py = (idx_row - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    float image_val = img[(idx_batch * vol.Ny + (vol.Ny - 1 - idx_row)) * vol.Nx + idx_col];
    float halfNu = geo.Nu * 0.5f;

    for (int idx_view = 0; idx_view < num_view; idx_view++) {
        float sinv = sinf(ang[idx_view]);
        float cosv = cosf(ang[idx_view]);
        bool useXAxis = (cosv * cosv > 0.5f);
        bool keepDim  = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

        float idxd0 = keepDim ? -halfNu : halfNu;
        float idxd1 = keepDim ? halfNu : -halfNu;
        float det_interval = useXAxis ? (geo.du / fabsf(cosv)) : (geo.du / fabsf(sinv));
        float pix_interval = useXAxis ? vol.dx : vol.dy;
        float coef = geo.du / pix_interval;

        float p0, p1, d0p, d1p;
        if (useXAxis) {
            p0 = sinv / cosv * py + (px - 0.5f * vol.dx);
            p1 = p0 + pix_interval;
            d0p = (idxd0 * geo.du + geo.shift_u) / cosv;
            d1p = (idxd1 * geo.du + geo.shift_u) / cosv;
        } else {
            p0 = cosv / sinv * px + (py - 0.5f * vol.dy);
            p1 = p0 + pix_interval;
            d0p = (idxd0 * geo.du + geo.shift_u) / sinv;
            d1p = (idxd1 * geo.du + geo.shift_u) / sinv;
        }

        float bound0 = fmaxf(p0, d0p);
        float bound1 = fminf(p1, d1p);
        int idxd = floorf((bound0 - d0p) / det_interval);
        idxd = max(0, idxd);
        idxd = min(geo.Nu - 1, idxd);
        float det0p = d0p + idxd * det_interval;
        idxd = keepDim ? idxd : (geo.Nu - 1 - idxd);

        while (bound0 < bound1) {
            float d1 = fminf(det0p + det_interval, bound1);
            atomicAdd(
                proj + (idx_batch * num_view + idx_view) * geo.Nu + idxd,
                image_val * (d1 - bound0) * coef
            );
            bound0 = d1;
            det0p = d1;
            idxd += keepDim ? 1 : -1;
        }
    }
}

void backward2d_t_cuda(
    float* proj,
    float* image,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
) {
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim(num_batch, (vol.Ny - 1) / BLOCK_DIM + 1, (vol.Nx - 1) / BLOCK_DIM + 1);
    switch (geo.type) {
    case GeometryType::FanFlat:
        fanflat_backward_t_cuda_kernel<false><<<gridDim, blockDim>>>(proj, image, ang, vol, geo, num_view);
        break;
    case GeometryType::FanArc:
        fanarc_backward_t_cuda_kernel<false><<<gridDim, blockDim>>>(proj, image, ang, vol, geo, num_view);
        break;
    case GeometryType::Parallel:
        parallel_backward_t_cuda_kernel<<<gridDim, blockDim>>>(proj, image, ang, vol, geo, num_view);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void weighted_backward2d_t_cuda(
    float* proj,
    float* image,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
) {
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim(num_batch, (vol.Ny - 1) / BLOCK_DIM + 1, (vol.Nx - 1) / BLOCK_DIM + 1);
    switch (geo.type) {
    case GeometryType::FanFlat:
        fanflat_backward_t_cuda_kernel<true><<<gridDim, blockDim>>>(proj, image, ang, vol, geo, num_view);
        break;
    case GeometryType::FanArc:
        fanarc_backward_t_cuda_kernel<true><<<gridDim, blockDim>>>(proj, image, ang, vol, geo, num_view);
        break;
    case GeometryType::Parallel:
        parallel_backward_t_cuda_kernel<<<gridDim, blockDim>>>(proj, image, ang, vol, geo, num_view);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
