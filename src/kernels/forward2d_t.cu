#include <cuda_runtime.h>
#include "types.h"
#include "utils.h"
#include "xrmath.h"

#define BLOCK_DIM 256

namespace xrecon {

__device__ inline float fanflat_forward_t_scale(
    const Point2d& src,
    const float axis0,
    const float axis1,
    const bool useXAxis,
    const float pix_step
) {
    float coef1 = axis1 - axis0;
    float coef2 = useXAxis ? cos_weight_x(src, axis1, axis0) : cos_weight_y(src, axis1, axis0);
    return pix_step / (coef1 * coef2);
}

__device__ inline void add_forward_t_segment(
    float* __restrict__ img,
    const float* __restrict__ proj,
    const Point2d& src,
    const float* __restrict__ det_axis,
    const int local_det,
    const int det0,
    const bool keepDim,
    const bool useXAxis,
    const float segment,
    const float pix_step,
    const Volume2d vol,
    const Geometry2d geo,
    const int idx_batch,
    const int idx_view,
    const int num_view,
    const int row,
    const int col
) {
    int det = det0 + local_det;
    if (local_det < 0 || det < 0 || det >= geo.Nu) return;
    int det_out = keepDim ? det : (geo.Nu - 1 - det);
    float scale = fanflat_forward_t_scale(src, det_axis[local_det], det_axis[local_det + 1], useXAxis, pix_step);
    float p = proj[(idx_batch * num_view + idx_view) * geo.Nu + det_out];
    atomicAdd(img + (idx_batch * vol.Ny + row) * vol.Nx + col, segment * scale * p);
}

__global__ void fanflat_forward_t_cuda_kernel(
    float* __restrict__ img,
    const float* __restrict__ proj,
    const float* __restrict__ ang,
    const Volume2d vol,
    const Geometry2d geo,
    const int num_view
) {
    __shared__ float s_det_axis[BLOCK_DIM + 1];

    const int idx_batch = blockIdx.x;
    const int idx_view = blockIdx.y;
    const int idx_det0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idx_det = idx_det0 + tx;
    const int max_num_det = ((idx_det0 + blockDim.x) > geo.Nu) ? (geo.Nu - idx_det0) : blockDim.x;

    float sinv = sinf(ang[idx_view]);
    float cosv = cosf(ang[idx_view]);
    Point2d src(-sinv * geo.SOD, cosv * geo.SOD);
    bool useXAxis = (cosv * cosv > 0.5f);
    bool keepDim = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

    float idxd0 = keepDim ? idx_det0 - geo.Nu * 0.5f : geo.Nu * 0.5f - idx_det0;
    float idxd1 = keepDim ? idx_det + 1 - geo.Nu * 0.5f : geo.Nu * 0.5f - idx_det - 1.0f;
    Point2d det0((idxd0 * geo.du + geo.shift_u) * cosv,
                 (idxd0 * geo.du + geo.shift_u) * sinv);
    Point2d det1((idxd1 * geo.du + geo.shift_u) * cosv,
                 (idxd1 * geo.du + geo.shift_u) * sinv);

    if (tx == 0) s_det_axis[0] = useXAxis ? map2x(src, det0) : map2y(src, det0);
    s_det_axis[tx + 1] = useXAxis ? map2x(src, det1) : map2y(src, det1);
    __syncthreads();

    if (useXAxis) {
        float x0 = -vol.Nx * 0.5f * vol.dx + vol.shift_x;
        float x1 =  vol.Nx * 0.5f * vol.dx + vol.shift_x;
        float d_det = geo.du * fabsf(cosv);

        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int row = i * blockDim.x + tx;
            if (row < vol.Ny) {
                float y = (vol.Ny * 0.5f - row - 0.5f) * vol.dy + vol.shift_y;
                float p0 = map2x(src, Point2d(x0, y));
                float p1 = map2x(src, Point2d(x1, y));
                float d_p = (p1 - p0) / vol.Nx;
                float bound = fmaxf(p0, s_det_axis[0]);
                int col;
                int idxd;
                if (p0 == bound) {
                    float b2d = src.y * bound / ((bound - src.x) * sinv / cosv + src.y);
                    col = 0;
                    idxd = floorf((b2d - det0.x) / d_det);
                } else {
                    col = floorf((bound - p0) / d_p);
                    idxd = 0;
                }
                float p_next = (col + 1) * d_p + p0;
                float d_next = (idxd < max_num_det) ? s_det_axis[idxd + 1] : 0.f;
                while (col < vol.Nx && idxd < max_num_det) {
                    if (p_next < d_next) {
                        add_forward_t_segment(img, proj, src, s_det_axis, idxd, idx_det0, keepDim, true,
                                              p_next - bound, vol.dy, vol, geo, idx_batch, idx_view, num_view, row, col);
                        bound = p_next;
                        col++;
                        p_next += d_p;
                    } else {
                        add_forward_t_segment(img, proj, src, s_det_axis, idxd, idx_det0, keepDim, true,
                                              d_next - bound, vol.dy, vol, geo, idx_batch, idx_view, num_view, row, col);
                        bound = d_next;
                        idxd++;
                        if (idxd < max_num_det) d_next = s_det_axis[idxd + 1];
                    }
                }
            }
        }
    } else {
        float y0 = -vol.Ny * 0.5f * vol.dy + vol.shift_y;
        float y1 =  vol.Ny * 0.5f * vol.dy + vol.shift_y;
        float d_det = geo.du * fabsf(sinv);

        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int col = i * blockDim.x + tx;
            if (col < vol.Nx) {
                float x = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
                float p0 = map2y(src, Point2d(x, y0));
                float p1 = map2y(src, Point2d(x, y1));
                float d_p = (p1 - p0) / vol.Ny;
                float bound = fmaxf(p0, s_det_axis[0]);
                int row;
                int idxd;
                if (p0 == bound) {
                    float b2d = src.x * bound / ((bound - src.y) * cosv / sinv + src.x);
                    row = 0;
                    idxd = floorf((b2d - det0.y) / d_det);
                } else {
                    row = floorf((bound - p0) / d_p);
                    idxd = 0;
                }
                float p_next = (row + 1) * d_p + p0;
                float d_next = (idxd < max_num_det) ? s_det_axis[idxd + 1] : 0.f;
                while (row < vol.Ny && idxd < max_num_det) {
                    int img_row = vol.Ny - 1 - row;
                    if (p_next < d_next) {
                        add_forward_t_segment(img, proj, src, s_det_axis, idxd, idx_det0, keepDim, false,
                                              p_next - bound, vol.dx, vol, geo, idx_batch, idx_view, num_view, img_row, col);
                        bound = p_next;
                        row++;
                        p_next += d_p;
                    } else {
                        add_forward_t_segment(img, proj, src, s_det_axis, idxd, idx_det0, keepDim, false,
                                              d_next - bound, vol.dx, vol, geo, idx_batch, idx_view, num_view, img_row, col);
                        bound = d_next;
                        idxd++;
                        if (idxd < max_num_det) d_next = s_det_axis[idxd + 1];
                    }
                }
            }
        }
    }
}

__device__ inline void add_parallel_forward_t_segment(
    float* __restrict__ img,
    const float* __restrict__ proj,
    const int local_det,
    const int det0,
    const bool keepDim,
    const float segment,
    const float coef,
    const Volume2d vol,
    const Geometry2d geo,
    const int idx_batch,
    const int idx_view,
    const int num_view,
    const int row,
    const int col
) {
    int det = det0 + local_det;
    if (local_det < 0 || det < 0 || det >= geo.Nu) return;
    int det_out = keepDim ? det : (geo.Nu - 1 - det);
    float p = proj[(idx_batch * num_view + idx_view) * geo.Nu + det_out];
    atomicAdd(img + (idx_batch * vol.Ny + row) * vol.Nx + col, segment * coef * p);
}

__global__ void parallel_forward_t_cuda_kernel(
    float* __restrict__ img,
    const float* __restrict__ proj,
    const float* __restrict__ ang,
    const Volume2d vol,
    const Geometry2d geo,
    const int num_view
) {
    const int idx_batch = blockIdx.x;
    const int idx_view = blockIdx.y;
    const int idx_det0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int max_num_det = ((idx_det0 + blockDim.x) > geo.Nu) ? (geo.Nu - idx_det0) : blockDim.x;

    float sinv = sinf(ang[idx_view]);
    float cosv = cosf(ang[idx_view]);
    bool useXAxis = (cosv * cosv > 0.5f);
    bool keepDim = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

    float idxd0 = keepDim ? idx_det0 - geo.Nu * 0.5f : geo.Nu * 0.5f - idx_det0;
    float det02axis = useXAxis ? (idxd0 * geo.du + geo.shift_u) / cosv
                               : (idxd0 * geo.du + geo.shift_u) / sinv;
    float d_det = useXAxis ? geo.du / fabsf(cosv) : geo.du / fabsf(sinv);
    float d_p = useXAxis ? vol.dx : vol.dy;
    float coef = useXAxis ? (vol.dy / geo.du) : (vol.dx / geo.du);

    if (useXAxis) {
        float x0 = -vol.Nx * 0.5f * vol.dx + vol.shift_x;
        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int row = i * blockDim.x + tx;
            if (row < vol.Ny) {
                float y = (vol.Ny * 0.5f - row - 0.5f) * vol.dy + vol.shift_y;
                float p0 = (sinv / cosv) * y + x0;
                float bound = fmaxf(p0, det02axis);
                int col;
                int idxd;
                if (p0 == bound) {
                    col = 0;
                    idxd = floorf((bound - det02axis) / d_det);
                } else {
                    col = floorf((bound - p0) / d_p);
                    idxd = 0;
                }
                float p_next = (col + 1) * d_p + p0;
                float d_next = (idxd + 1) * d_det + det02axis;
                while (col < vol.Nx && idxd < max_num_det) {
                    if (p_next < d_next) {
                        add_parallel_forward_t_segment(img, proj, idxd, idx_det0, keepDim, p_next - bound,
                                                       coef, vol, geo, idx_batch, idx_view, num_view, row, col);
                        bound = p_next;
                        col++;
                        p_next += d_p;
                    } else {
                        add_parallel_forward_t_segment(img, proj, idxd, idx_det0, keepDim, d_next - bound,
                                                       coef, vol, geo, idx_batch, idx_view, num_view, row, col);
                        bound = d_next;
                        idxd++;
                        d_next += d_det;
                    }
                }
            }
        }
    } else {
        float y0 = -vol.Ny * 0.5f * vol.dy + vol.shift_y;
        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int col = i * blockDim.x + tx;
            if (col < vol.Nx) {
                float x = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
                float p0 = (cosv / sinv) * x + y0;
                float bound = fmaxf(p0, det02axis);
                int row;
                int idxd;
                if (p0 == bound) {
                    row = 0;
                    idxd = floorf((bound - det02axis) / d_det);
                } else {
                    row = floorf((bound - p0) / d_p);
                    idxd = 0;
                }
                float p_next = (row + 1) * d_p + p0;
                float d_next = (idxd + 1) * d_det + det02axis;
                while (row < vol.Ny && idxd < max_num_det) {
                    int img_row = vol.Ny - 1 - row;
                    if (p_next < d_next) {
                        add_parallel_forward_t_segment(img, proj, idxd, idx_det0, keepDim, p_next - bound,
                                                       coef, vol, geo, idx_batch, idx_view, num_view, img_row, col);
                        bound = p_next;
                        row++;
                        p_next += d_p;
                    } else {
                        add_parallel_forward_t_segment(img, proj, idxd, idx_det0, keepDim, d_next - bound,
                                                       coef, vol, geo, idx_batch, idx_view, num_view, img_row, col);
                        bound = d_next;
                        idxd++;
                        d_next += d_det;
                    }
                }
            }
        }
    }
}

__global__ void fanarc_forward_t_cuda_kernel(
    float* __restrict__ img,
    const float* __restrict__ proj,
    const float* __restrict__ ang,
    const Volume2d vol,
    const Geometry2d geo,
    const int num_view
) {
    __shared__ float s_det_axis[BLOCK_DIM + 1];

    const int idx_batch = blockIdx.x;
    const int idx_view = blockIdx.y;
    const int idx_det0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idx_det = idx_det0 + tx;
    const int max_num_det = ((idx_det0 + blockDim.x) > geo.Nu) ? (geo.Nu - idx_det0) : blockDim.x;

    float view = ang[idx_view];
    float sinv = sinf(view);
    float cosv = cosf(view);
    Point2d src(-sinv * geo.SOD, cosv * geo.SOD);
    bool useXAxis = (cosv * cosv > 0.5f);
    bool keepDim = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

    float det0_ang = keepDim
        ? (idx_det0 - geo.Nu * 0.5f) * geo.du + geo.shift_u + view
        : (geo.Nu * 0.5f - idx_det0) * geo.du + geo.shift_u + view;
    float det1_ang = keepDim
        ? (idx_det + 1 - geo.Nu * 0.5f) * geo.du + geo.shift_u + view
        : (geo.Nu * 0.5f - idx_det - 1.0f) * geo.du + geo.shift_u + view;

    if (tx == 0) s_det_axis[0] = useXAxis ? map2x(src, det0_ang) : map2y(src, det0_ang);
    s_det_axis[tx + 1] = useXAxis ? map2x(src, det1_ang) : map2y(src, det1_ang);
    __syncthreads();

    if (useXAxis) {
        float x0 = -vol.Nx * 0.5f * vol.dx + vol.shift_x;
        float x1 =  vol.Nx * 0.5f * vol.dx + vol.shift_x;
        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int row = i * blockDim.x + tx;
            if (row < vol.Ny) {
                float y = (vol.Ny * 0.5f - row - 0.5f) * vol.dy + vol.shift_y;
                float p0 = map2x(src, Point2d(x0, y));
                float p1 = map2x(src, Point2d(x1, y));
                float d_p = (p1 - p0) / vol.Nx;
                float tan0 = ((src.x - p0) == 0.f) ? 1e10f : (src.y / (src.x - p0));
                float tanc = -1.0f / (tanf(geo.shift_u + view) + 1e-10f);
                float delta = atanf((tan0 - tanc) / (1.0f + tan0 * tanc));
                int idxd = keepDim
                    ? floorf(geo.Nu * 0.5f - idx_det0 + delta / geo.du)
                    : floorf(geo.Nu * 0.5f - idx_det0 - delta / geo.du);
                int col;
                float bound;
                if (idxd < 0) {
                    bound = s_det_axis[0];
                    col = floorf((bound - p0) / d_p);
                    idxd = 0;
                } else {
                    bound = p0;
                    col = 0;
                }
                float p_next = (col + 1) * d_p + p0;
                float d_next = (idxd < max_num_det) ? s_det_axis[idxd + 1] : 0.f;
                while (col < vol.Nx && idxd < max_num_det) {
                    if (p_next < d_next) {
                        add_forward_t_segment(img, proj, src, s_det_axis, idxd, idx_det0, keepDim, true,
                                              p_next - bound, vol.dy, vol, geo, idx_batch, idx_view, num_view, row, col);
                        bound = p_next;
                        col++;
                        p_next += d_p;
                    } else {
                        add_forward_t_segment(img, proj, src, s_det_axis, idxd, idx_det0, keepDim, true,
                                              d_next - bound, vol.dy, vol, geo, idx_batch, idx_view, num_view, row, col);
                        bound = d_next;
                        idxd++;
                        if (idxd < max_num_det) d_next = s_det_axis[idxd + 1];
                    }
                }
            }
        }
    } else {
        float y0 = -vol.Ny * 0.5f * vol.dy + vol.shift_y;
        float y1 =  vol.Ny * 0.5f * vol.dy + vol.shift_y;
        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int col = i * blockDim.x + tx;
            if (col < vol.Nx) {
                float x = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
                float p0 = map2y(src, Point2d(x, y0));
                float p1 = map2y(src, Point2d(x, y1));
                float d_p = (p1 - p0) / vol.Ny;
                float tan0 = (src.y - p0) / src.x;
                float tanc = -1.0f / tanf(geo.shift_u + view);
                float delta = atanf((tan0 - tanc) / (1.0f + tan0 * tanc));
                int idxd = keepDim
                    ? floorf(geo.Nu * 0.5f - idx_det0 + delta / geo.du)
                    : floorf(geo.Nu * 0.5f - idx_det0 - delta / geo.du);
                int row;
                float bound;
                if (idxd < 0) {
                    bound = s_det_axis[0];
                    row = floorf((bound - p0) / d_p);
                    idxd = 0;
                } else {
                    bound = p0;
                    row = 0;
                }
                float p_next = (row + 1) * d_p + p0;
                float d_next = (idxd < max_num_det) ? s_det_axis[idxd + 1] : 0.f;
                while (row < vol.Ny && idxd < max_num_det) {
                    int img_row = vol.Ny - 1 - row;
                    if (p_next < d_next) {
                        add_forward_t_segment(img, proj, src, s_det_axis, idxd, idx_det0, keepDim, false,
                                              p_next - bound, vol.dx, vol, geo, idx_batch, idx_view, num_view, img_row, col);
                        bound = p_next;
                        row++;
                        p_next += d_p;
                    } else {
                        add_forward_t_segment(img, proj, src, s_det_axis, idxd, idx_det0, keepDim, false,
                                              d_next - bound, vol.dx, vol, geo, idx_batch, idx_view, num_view, img_row, col);
                        bound = d_next;
                        idxd++;
                        if (idxd < max_num_det) d_next = s_det_axis[idxd + 1];
                    }
                }
            }
        }
    }
}

void forward2d_t_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
) {
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim(num_batch, num_view, (geo.Nu + BLOCK_DIM - 1) / BLOCK_DIM);
    switch (geo.type) {
    case GeometryType::FanFlat:
        fanflat_forward_t_cuda_kernel<<<gridDim, blockDim>>>(image, proj, ang, vol, geo, num_view);
        break;
    case GeometryType::FanArc:
        fanarc_forward_t_cuda_kernel<<<gridDim, blockDim>>>(image, proj, ang, vol, geo, num_view);
        break;
    case GeometryType::Parallel:
        parallel_forward_t_cuda_kernel<<<gridDim, blockDim>>>(image, proj, ang, vol, geo, num_view);
        break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
