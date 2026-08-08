#include <cuda_runtime.h>
#include "types.h"
#include "utils.h"
#include "xrmath.h"

#define BLOCK_DIM 256

namespace xrecon {

__global__ void fanflat_forward_cuda_kernel(
    cudaTextureObject_t tex_obj,
    float* __restrict__ proj,
    const float* __restrict__ ang,
    const Volume2d vol,
    const Geometry2d geo,
    const int num_view) 
{
    __shared__ float s_proj[BLOCK_DIM];
    __shared__ float s_det_axis[BLOCK_DIM + 1];

    const int idx_batch = blockIdx.x;
    const int idx_view  = blockIdx.y;
    const int idx_det0  = blockIdx.z * blockDim.x;
    const int tx        = threadIdx.x;
    const int idx_det   = idx_det0 + tx;
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

    s_proj[tx] = 0.f;

    if (tx == 0) {
        s_det_axis[tx] = useXAxis ? map2x(src, det0) : map2y(src, det0);
    }
    s_det_axis[tx + 1] = useXAxis ? map2x(src, det1) : map2y(src, det1);

    __syncthreads();

    float coef1 = s_det_axis[tx + 1] - s_det_axis[tx];
    float coef2 = useXAxis ? cos_weight_x(src, s_det_axis[tx + 1], s_det_axis[tx])
                           : cos_weight_y(src, s_det_axis[tx + 1], s_det_axis[tx]);

    if (useXAxis) {
        float x0 = - vol.Nx * 0.5f * vol.dx + vol.shift_x;
        float x1 =   vol.Nx * 0.5f * vol.dx + vol.shift_x;
        float d_det = geo.du * abs(cosv);

        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int idx_row = i * blockDim.x + tx;
            if (idx_row < vol.Ny) {
                float y = (vol.Ny * 0.5f - idx_row - 0.5f) * vol.dy + vol.shift_y;
                float p0 = map2x(src, Point2d(x0, y));
                float p1 = map2x(src, Point2d(x1, y));
                float d_p = (p1 - p0) / vol.Nx;
                float bound = fmaxf(p0, s_det_axis[0]);
                int idx_col;
                int idxd;
                if (p0 == bound) {
                    float b2d = src.y * bound / ((bound - src.x) * sinv / cosv + src.y);
                    idx_col = 0;
                    idxd = floorf((b2d - det0.x) / d_det);
                } else {
                    idx_col = floorf((bound - p0) / d_p);
                    idxd = 0;
                }
                float p_next = (idx_col + 1) * d_p + p0;
                float d_next = (idxd < max_num_det) ? s_det_axis[idxd + 1] : 0.f;
                float acc = 0.f;
                while(idx_col < vol.Nx && idxd < max_num_det) {
                    if (p_next < d_next) {
                        acc += (p_next - bound) * tex3D<float>(tex_obj, idx_col, idx_row, idx_batch);
                        bound = p_next;
                        idx_col ++;
                        p_next += d_p;
                    } else {
                        acc += (d_next - bound) * tex3D<float>(tex_obj, idx_col, idx_row, idx_batch);
                        atomicAdd(s_proj + idxd, acc);
                        acc = 0.f;
                        bound = d_next;
                        idxd ++;
                        if (idxd < max_num_det) d_next = s_det_axis[idxd + 1];
                    }
                }
                if (acc != 0.f && idxd < max_num_det) atomicAdd(s_proj + idxd, acc);
            }
        }

        __syncthreads();

        if (idx_det < geo.Nu) {
            s_proj[tx] *= vol.dy / (coef1 * coef2);
            int idx_det_out = keepDim ? idx_det : (geo.Nu - 1 - idx_det);
            proj[(idx_batch * num_view + idx_view) * geo.Nu + idx_det_out] = s_proj[tx];
        }

    } else {
        float y0 = - vol.Ny * 0.5f * vol.dy + vol.shift_y;
        float y1 =   vol.Ny * 0.5f * vol.dy + vol.shift_y;
        float d_det = geo.du * abs(sinv);
        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int idx_col = i * blockDim.x + tx;
            if (idx_col < vol.Nx) {
                float x = (idx_col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
                float p0 = map2y(src, Point2d(x, y0));
                float p1 = map2y(src, Point2d(x, y1));
                float d_p = (p1 - p0) / vol.Ny;
                float bound = fmaxf(p0, s_det_axis[0]);
                int idx_row;
                int idxd;
                if (p0 == bound) {
                    float b2d = src.x * bound / ((bound - src.y) * cosv / sinv + src.x);
                    idx_row = 0;
                    idxd = floorf((b2d - det0.y) / d_det);
                } else {
                    idx_row = floorf((bound - p0) / d_p);
                    idxd = 0;
                }
                float p_next = (idx_row + 1) * d_p + p0;
                float d_next = (idxd < max_num_det) ? s_det_axis[idxd + 1] : 0.f;
                float acc = 0.f;
                while(idx_row < vol.Ny && idxd < max_num_det) {
                    if (p_next < d_next) {
                        acc += (p_next - bound) * tex3D<float>(tex_obj, idx_col, vol.Ny - 1 - idx_row, idx_batch);
                        bound = p_next;
                        idx_row ++;
                        p_next += d_p;
                    } else {
                        acc += (d_next - bound) * tex3D<float>(tex_obj, idx_col, vol.Ny - 1 - idx_row, idx_batch);
                        atomicAdd(s_proj + idxd, acc);
                        acc = 0.f;
                        bound = d_next;
                        idxd ++;
                        if (idxd < max_num_det) d_next = s_det_axis[idxd + 1];
                    }
                }
                if (acc != 0.f && idxd < max_num_det) atomicAdd(s_proj + idxd, acc);
            }
        }

        __syncthreads();

        if (idx_det < geo.Nu) {
            s_proj[tx] *= vol.dx / (coef1 * coef2);
            int idx_det_out = keepDim ? idx_det : (geo.Nu - 1 - idx_det);
            proj[(idx_batch * num_view + idx_view) * geo.Nu + idx_det_out] = s_proj[tx];
        }
    }
}

__global__ void fanarc_forward_cuda_kernel(
    cudaTextureObject_t tex_obj,
    float* __restrict__ proj,
    const float* __restrict__ ang,
    const Volume2d vol,
    const Geometry2d geo,
    const int num_view
) {
    __shared__ float s_proj[BLOCK_DIM];
    __shared__ float s_det_axis[BLOCK_DIM + 1];

    const int idx_batch = blockIdx.x;
    const int idx_view  = blockIdx.y;
    const int idx_det0  = blockIdx.z * blockDim.x;
    const int tx        = threadIdx.x;
    const int idx_det   = idx_det0 + tx;
    const int max_num_det = ((idx_det0 + blockDim.x) > geo.Nu) ? (geo.Nu - idx_det0) : blockDim.x;

    float sinv = sinf(ang[idx_view]);
    float cosv = cosf(ang[idx_view]);
    Point2d src(-sinv * geo.SOD, cosv * geo.SOD);

    bool useXAxis = (cosv * cosv > 0.5f);
    bool keepDim  = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

    float det0_ang = keepDim
        ? (idx_det0 -  geo.Nu * 0.5f) * geo.du + geo.shift_u + ang[idx_view]
        : ( geo.Nu * 0.5f - idx_det0) * geo.du + geo.shift_u + ang[idx_view];

    float det1_ang = keepDim
        ? (idx_det + 1 -  geo.Nu * 0.5f) * geo.du + geo.shift_u + ang[idx_view]
        : ( geo.Nu * 0.5f - idx_det - 1.0f) * geo.du + geo.shift_u + ang[idx_view];

    s_proj[tx] = 0.f;

    if (tx == 0) {
        s_det_axis[tx] = useXAxis ? map2x(src, det0_ang) : map2y(src, det0_ang);
    }
    s_det_axis[tx + 1] = useXAxis ? map2x(src, det1_ang) : map2y(src, det1_ang);

    __syncthreads();

    float coef1 = s_det_axis[tx + 1] - s_det_axis[tx];
    float coef2 = useXAxis
        ? cos_weight_x(src, s_det_axis[tx + 1], s_det_axis[tx])
        : cos_weight_y(src, s_det_axis[tx + 1], s_det_axis[tx]);

    if (useXAxis) {
        float x0 = - vol.Nx * 0.5f * vol.dx + vol.shift_x;
        float x1 =   vol.Nx * 0.5f * vol.dx + vol.shift_x;

        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int idx_row = i * blockDim.x + tx;
            if (idx_row < vol.Ny) {
                float y  = (vol.Ny * 0.5f - idx_row - 0.5f) * vol.dy + vol.shift_y;
                float p0 = map2x(src, Point2d(x0, y));
                float p1 = map2x(src, Point2d(x1, y));
                float d_p = (p1 - p0) / vol.Nx;

                float tan0   = ((src.x - p0) == 0.f) ? 1e10f : (src.y / (src.x - p0));
                float tanc   = -1.0f / (tanf(geo.shift_u + ang[idx_view]) + 1e-10f);
                float delta        = atanf((tan0 - tanc) / (1.0f + tan0 * tanc));

                int idxd = keepDim
                    ? floorf(geo.Nu * 0.5f - idx_det0 + delta / geo.du)
                    : floorf(geo.Nu * 0.5f - idx_det0 - delta / geo.du);

                int idx_col;
                float bound;
                if (idxd < 0) {
                    bound = s_det_axis[0];
                    idx_col = floorf((bound - p0) / d_p);
                    idxd = 0;
                } else {
                    bound = p0;
                    idx_col = 0;
                }

                float p_next = (idx_col + 1) * d_p + p0;
                float d_next = (idxd < max_num_det) ? s_det_axis[idxd + 1] : 0.f;
                float acc = 0.f;

                while (idx_col < vol.Nx && idxd < max_num_det) {
                    if (p_next < d_next) {
                        acc += (p_next - bound) * tex3D<float>(tex_obj, idx_col, idx_row, idx_batch);
                        bound = p_next;
                        idx_col++;
                        p_next += d_p;
                    } else {
                        acc += (d_next - bound) * tex3D<float>(tex_obj, idx_col, idx_row, idx_batch);
                        atomicAdd(s_proj + idxd, acc);
                        acc = 0.f;
                        bound = d_next;
                        idxd++;
                        if (idxd < max_num_det) d_next = s_det_axis[idxd + 1];
                    }
                }
                if (acc != 0.f && idxd < max_num_det) atomicAdd(s_proj + idxd, acc);
            }
        }

        __syncthreads();

        if (idx_det < geo.Nu) {
            s_proj[tx] *= vol.dy / (coef1 * coef2);
            int idx_det_out = keepDim ? idx_det : (geo.Nu - 1 - idx_det);
            proj[(idx_batch * num_view + idx_view) * geo.Nu + idx_det_out] = s_proj[tx];
        }

    } else {
        float y0 = - vol.Ny * 0.5f * vol.dy + vol.shift_y;
        float y1 =   vol.Ny * 0.5f * vol.dy + vol.shift_y;

        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int idx_col = i * blockDim.x + tx;
            if (idx_col < vol.Nx) {
                float x  = (idx_col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
                float p0 = map2y(src, Point2d(x, y0));
                float p1 = map2y(src, Point2d(x, y1));
                float d_p = (p1 - p0) / vol.Ny;

                float tan0 = (src.y - p0) / src.x;
                float tanc = -1.0f / tanf(geo.shift_u + ang[idx_view]);
                float delta      = atanf((tan0 - tanc) / (1.0f + tan0 * tanc));

                int idxd = keepDim
                    ? floorf(geo.Nu * 0.5f - idx_det0 + delta / geo.du)
                    : floorf(geo.Nu * 0.5f - idx_det0 - delta / geo.du);

                int idx_row;
                float bound;
                if (idxd < 0) {
                    bound = s_det_axis[0];
                    idx_row = floorf((bound - p0) / d_p);
                    idxd = 0;
                } else {
                    bound = p0;
                    idx_row = 0;
                }

                float p_next = (idx_row + 1) * d_p + p0;
                float d_next = (idxd < max_num_det) ? s_det_axis[idxd + 1] : 0.f;
                float acc = 0.f;

                while (idx_row < vol.Ny && idxd < max_num_det) {
                    if (p_next < d_next) {
                        acc += (p_next - bound) * tex3D<float>(tex_obj, idx_col, vol.Ny - 1 - idx_row, idx_batch);
                        bound = p_next;
                        idx_row++;
                        p_next += d_p;
                    } else {
                        acc += (d_next - bound) * tex3D<float>(tex_obj, idx_col, vol.Ny - 1 - idx_row, idx_batch);
                        atomicAdd(s_proj + idxd, acc);
                        acc = 0.f;
                        bound = d_next;
                        idxd++;
                        if (idxd < max_num_det) d_next = s_det_axis[idxd + 1];
                    }
                }
                if (acc != 0.f && idxd < max_num_det) atomicAdd(s_proj + idxd, acc);
            }
        }

        __syncthreads();

        if (idx_det < geo.Nu) {
            s_proj[tx] *= vol.dx / (coef1 * coef2);
            int idx_det_out = keepDim ? idx_det : (geo.Nu - 1 - idx_det);
            proj[(idx_batch * num_view + idx_view) * geo.Nu + idx_det_out] = s_proj[tx];
        }
    }
}

__global__ void parallel_forward_cuda_kernel(
    cudaTextureObject_t tex_obj,
    float* __restrict__ proj,
    const float* __restrict__ ang,
    const Volume2d vol,
    const Geometry2d geo,
    const int num_view
) {
    __shared__ float s_proj[BLOCK_DIM];

    const int idx_batch = blockIdx.x;
    const int idx_view  = blockIdx.y;
    const int idx_det0  = blockIdx.z * blockDim.x;
    const int tx        = threadIdx.x;
    const int idx_det   = idx_det0 + tx;
    const int max_num_det = ((idx_det0 + blockDim.x) > geo.Nu) ? (geo.Nu - idx_det0) : blockDim.x;

    float sinv = sinf(ang[idx_view]);
    float cosv = cosf(ang[idx_view]);

    bool useXAxis = (cosv * cosv > 0.5f);
    bool keepDim  = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

    float idxd0 = keepDim ? idx_det0 - geo.Nu * 0.5f : geo.Nu * 0.5f - idx_det0;
    float det02axis = useXAxis
        ? (idxd0 * geo.du + geo.shift_u) / cosv
        : (idxd0 * geo.du + geo.shift_u) / sinv;

    float d_det = useXAxis
        ? geo.du / fabsf(cosv)
        : geo.du / fabsf(sinv);

    float d_p = useXAxis ? vol.dx : vol.dy;

    float coef = useXAxis ? (vol.dy / geo.du) : (vol.dx / geo.du);

    s_proj[tx] = 0.f;
    __syncthreads();

    if (useXAxis) {
        float x0 = - vol.Nx * 0.5f * vol.dx + vol.shift_x;

        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int idx_row = i * blockDim.x + tx;
            if (idx_row < vol.Ny) {
                float y = (vol.Ny * 0.5f - idx_row - 0.5f) * vol.dy + vol.shift_y;
                float p0 = (sinv / cosv) * y + x0;

                float bound = fmaxf(p0, det02axis);
                int idx_col;
                int idxd;
                if (p0 == bound) {
                    idx_col = 0;
                    idxd    = floorf((bound - det02axis) / d_det);
                } else {
                    idx_col = floorf((bound - p0) / d_p);
                    idxd    = 0;
                }

                float p_next = (idx_col + 1) * d_p + p0;
                float d_next = (idxd + 1) * d_det + det02axis;

                float acc = 0.f;
                while (idx_col < vol.Nx && idxd < max_num_det) {
                    if (p_next < d_next) {
                        acc += (p_next - bound) * tex3D<float>(tex_obj, idx_col, idx_row, idx_batch);
                        bound  = p_next;
                        idx_col++;
                        p_next += d_p;
                    } else {
                        acc += (d_next - bound) * tex3D<float>(tex_obj, idx_col, idx_row, idx_batch);
                        atomicAdd(s_proj + idxd, acc);
                        acc   = 0.f;
                        bound = d_next;
                        idxd++;
                        d_next += d_det;
                    }
                }
                if (acc != 0.f && idxd < max_num_det) atomicAdd(s_proj + idxd, acc);
            }
        }

        __syncthreads();

        if (idx_det < geo.Nu) {
            s_proj[tx] *= coef;
            int idx_det_out = keepDim ? idx_det : (geo.Nu - 1 - idx_det);
            proj[(idx_batch * num_view + idx_view) * geo.Nu + idx_det_out] = s_proj[tx];
        }

    } else {
        float y0 = - vol.Ny * 0.5f * vol.dy + vol.shift_y;

        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int idx_col = i * blockDim.x + tx;
            if (idx_col < vol.Nx) {
                float x = (idx_col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
                float p0 = (cosv / sinv) * x + y0;

                float bound = fmaxf(p0, det02axis);
                int idx_row;
                int idxd;
                if (p0 == bound) {
                    idx_row = 0;
                    idxd    = floorf((bound - det02axis) / d_det);
                } else {
                    idx_row = floorf((bound - p0) / d_p);
                    idxd    = 0;
                }

                float p_next = (idx_row + 1) * d_p + p0;
                float d_next = (idxd + 1) * d_det + det02axis;

                float acc = 0.f;
                while (idx_row < vol.Ny && idxd < max_num_det) {
                    if (p_next < d_next) {
                        acc += (p_next - bound) * tex3D<float>(tex_obj, idx_col, vol.Ny - 1 - idx_row, idx_batch);
                        bound  = p_next;
                        idx_row++;
                        p_next += d_p;
                    } else {
                        acc += (d_next - bound) * tex3D<float>(tex_obj, idx_col, vol.Ny - 1 - idx_row, idx_batch);
                        atomicAdd(s_proj + idxd, acc);
                        acc   = 0.f;
                        bound = d_next;
                        idxd++;
                        d_next += d_det;
                    }
                }
                if (acc != 0.f && idxd < max_num_det) atomicAdd(s_proj + idxd, acc);
            }
        }

        __syncthreads();

        if (idx_det < geo.Nu) {
            s_proj[tx] *= coef;
            int idx_det_out = keepDim ? idx_det : (geo.Nu - 1 - idx_det);
            proj[(idx_batch * num_view + idx_view) * geo.Nu + idx_det_out] = s_proj[tx];
        }
    }
}

void forward2d_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
) {

    CudaTexture3D texture3D;
    texture3D.create(image, vol.Nx, vol.Ny, num_batch);
    
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim(num_batch, num_view, (geo.Nu + BLOCK_DIM - 1) / BLOCK_DIM);

    switch (geo.type) {
    case GeometryType::FanFlat:
        fanflat_forward_cuda_kernel<<<gridDim, blockDim>>>(
            texture3D.texObj, proj, ang, vol, geo, num_view);
        break;
    case GeometryType::FanArc:
        fanarc_forward_cuda_kernel<<<gridDim, blockDim>>>(
            texture3D.texObj, proj, ang, vol, geo, num_view);
        break;
    case GeometryType::Parallel:
        parallel_forward_cuda_kernel<<<gridDim, blockDim>>>(
            texture3D.texObj, proj, ang, vol, geo, num_view);
        break;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

}

}