#include <cuda.h>
#include "cone3d_common.cuh"

#define BLOCK_DIM 256


__global__ void coneflat_forward_cuda_kernel(
    float* __restrict__ proj,
    cudaTextureObject_t tex_obj,
    const float* __restrict__ ang,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    const int num_view) {

    const int idx_batch = blockIdx.x / num_view;
    const int idx_view = blockIdx.x % num_view;
    const int idx_det_row = blockIdx.y;
    const int idx_det_col0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idx_det_col = idx_det_col0 + tx;
    const int max_num_det_col = ((idx_det_col0 + blockDim.x) > geo.Nu) ? (geo.Nu - idx_det_col0) : blockDim.x;
    __shared__ float s_proj[BLOCK_DIM];
    __shared__ float s_row0_axis[BLOCK_DIM];
    __shared__ float s_row1_axis[BLOCK_DIM];
    __shared__ float s_col_axis[BLOCK_DIM + 1];

    float sinv = sin(ang[idx_view]);
    float cosv = cos(ang[idx_view]);
    float src_x = - sinv * geo.SOD;
    float src_y = cosv * geo.SOD;
    float virtual_du = geo.SOD / geo.SDD * geo.du;
    float virtual_shift_u = geo.SOD / geo.SDD * geo.shift_u;
    float det_row0_z = ((idx_det_row - geo.Nv / 2.0) * geo.dv + geo.shift_v) * geo.SOD / geo.SDD;
    float det_row1_z = det_row0_z + geo.SOD / geo.SDD * geo.dv;
    float det_col0_x;
    float det_col0_y;
    float det_col1_x;
    float det_col1_y;
    if (cosv * cosv > 0.5) {
        s_proj[tx] = 0;
        float det_col_y;
        if (cosv >= 0) {
            det_col0_x = ((idx_det_col0 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
            det_col0_y = ((idx_det_col0 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
            det_col1_x = ((idx_det_col + 1 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
            det_col1_y = ((idx_det_col + 1 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
            det_col_y = ((idx_det_col + 0.5 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
        } else {
            det_col0_x = ((geo.Nu / 2.0 - idx_det_col0) * virtual_du + virtual_shift_u) * cosv;
            det_col0_y = ((geo.Nu / 2.0 - idx_det_col0) * virtual_du + virtual_shift_u) * sinv;
            det_col1_x = ((geo.Nu / 2.0 - idx_det_col - 1) * virtual_du + virtual_shift_u) * cosv;
            det_col1_y = ((geo.Nu / 2.0 - idx_det_col - 1) * virtual_du + virtual_shift_u) * sinv;
            det_col_y = ((geo.Nu / 2.0 - idx_det_col - 0.5) * virtual_du + virtual_shift_u) * sinv;
        }
        float det_col0_axis = xrecon::map2x(src_x, src_y, det_col0_x, det_col0_y);
        float det_col1_axis = xrecon::map2x(src_x, src_y, det_col1_x, det_col1_y);
        float det_row0_axis = src_y / (src_y - det_col_y) * det_row0_z;
        float det_row1_axis = src_y / (src_y - det_col_y) * det_row1_z;
        if (tx == 0) s_col_axis[tx] = det_col0_axis;
        s_col_axis[tx + 1] = det_col1_axis;
        s_row0_axis[tx] = det_row0_axis;
        s_row1_axis[tx] = det_row1_axis;
        __syncthreads();
        float coef1 = (det_col1_axis - s_col_axis[tx]) * (det_row1_axis - det_row0_axis);
        float coef2 = xrecon::triangle_cosine((det_col1_axis + s_col_axis[tx]) / 2.0 - src_x, (det_row1_axis + det_row0_axis) / 2.0, src_y);
        float point0_x = - vol.Nx / 2.0 * vol.dx + vol.shift_x;
        float point1_x = vol.Nx / 2.0 * vol.dx + vol.shift_x;
        float point0_z = - vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float point1_z = vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float det_interval = virtual_du * abs(cosv);
        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < vol.Ny) {
                float point_y = (vol.Ny / 2.0 - idxrow - 0.5) * vol.dy + vol.shift_y;
                float point0_axis = xrecon::map2x(src_x, src_y, point0_x, point_y);
                float point1_axis = xrecon::map2x(src_x, src_y, point1_x, point_y);
                float point0_axis_z = src_y / (src_y - point_y) * point0_z;
                float point1_axis_z = src_y / (src_y - point_y) * point1_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Nx;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float bound0 = max(point0_axis, s_col_axis[0]);
                int idxcol;
                int idxd;
                if (point0_axis == bound0) {
                    float bound0_to_det = src_y * bound0 / ((bound0 - src_x) * sinv / cosv + src_y);
                    idxcol = 0;
                    idxd = floor((bound0_to_det - det_col0_x) / det_interval);
                } else {
                    idxcol = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                }
                point1_axis = (idxcol + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                float temp = 0;
                while(idxcol < vol.Nx && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idxslice;
                    if (bound0_z == point0_axis_z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idxslice + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = point1_axis - bound0;
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idxcol, idxrow, idx_batch * vol.Nz + idxslice);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idxcol++;
                        point1_axis += pix_interval;
                    } else {
                        float coef = det_col1_axis - bound0;
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idxcol, idxrow, idx_batch * vol.Nz + idxslice);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        atomicAdd(s_proj + idxd, temp);
                        temp = 0;
                        bound0 = det_col1_axis;
                        idxd ++;
                        if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(s_proj + idxd, temp);
            }
        }
        __syncthreads();
        if (idx_det_col < geo.Nu) {
            s_proj[tx] *= vol.dy / (coef1 * coef2);
            int idx_det_col_out = (cosv >= 0) ? idx_det_col : (geo.Nu - 1 - idx_det_col);
            proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out] = s_proj[tx];
        }
    } else {
        s_proj[tx] = 0;
        float det_col_x;
        if (sinv >= 0) {
            det_col0_x = ((idx_det_col0 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
            det_col0_y = ((idx_det_col0 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
            det_col1_x = ((idx_det_col + 1 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
            det_col1_y = ((idx_det_col + 1 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
            det_col_x = ((idx_det_col + 0.5 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
        } else {
            det_col0_x = ((geo.Nu / 2.0 - idx_det_col0) * virtual_du + virtual_shift_u) * cosv;
            det_col0_y = ((geo.Nu / 2.0 - idx_det_col0) * virtual_du + virtual_shift_u) * sinv;
            det_col1_x = ((geo.Nu / 2.0 - idx_det_col - 1) * virtual_du + virtual_shift_u) * cosv;
            det_col1_y = ((geo.Nu / 2.0 - idx_det_col - 1) * virtual_du + virtual_shift_u) * sinv;
            det_col_x = ((geo.Nu / 2.0 - idx_det_col - 0.5) * virtual_du + virtual_shift_u) * cosv;
        }
        float det_col0_axis = xrecon::map2y(src_x, src_y, det_col0_x, det_col0_y);
        float det_col1_axis = xrecon::map2y(src_x, src_y, det_col1_x, det_col1_y);
        float det_row0_axis = src_x / (src_x - det_col_x) * det_row0_z;
        float det_row1_axis = src_x / (src_x - det_col_x) * det_row1_z;
        if (tx == 0) s_col_axis[tx] = det_col0_axis;
        s_col_axis[tx + 1] = det_col1_axis;
        s_row0_axis[tx] = det_row0_axis;
        s_row1_axis[tx] = det_row1_axis;
        __syncthreads();
        float coef1 = (det_col1_axis - s_col_axis[tx]) * (det_row1_axis - det_row0_axis);
        float coef2 = xrecon::triangle_cosine((det_col1_axis + s_col_axis[tx]) / 2.0 - src_y, (det_row1_axis + det_row0_axis) / 2.0, src_x);
        float point0_y = - vol.Ny / 2.0 * vol.dy + vol.shift_y;
        float point1_y = vol.Ny / 2.0 * vol.dy + vol.shift_y;
        float point0_z = - vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float point1_z = vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float det_interval = virtual_du * abs(sinv);
        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < vol.Nx) {
                float point_x = (idxcol - vol.Nx / 2.0 + 0.5) * vol.dx + vol.shift_x;
                float point0_axis = xrecon::map2y(src_x, src_y, point_x, point0_y);
                float point1_axis = xrecon::map2y(src_x, src_y, point_x, point1_y);
                float point0_axis_z = src_x / (src_x - point_x) * point0_z;
                float point1_axis_z = src_x / (src_x - point_x) * point1_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Ny;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float bound0 = max(point0_axis, s_col_axis[0]);
                int idxrow;
                int idxd;
                if (point0_axis == bound0) {
                    float bound0_to_det = src_x * bound0 / ((bound0 - src_y) * cosv / sinv + src_x);
                    idxrow = 0;
                    idxd = floor((bound0_to_det - det_col0_y) / det_interval);
                } else {
                    idxrow = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                }
                point1_axis = (idxrow + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                float temp = 0;
                while(idxrow < vol.Ny && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idxslice;
                    if (bound0_z == point0_axis_z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idxslice + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = point1_axis - bound0;
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idxcol, vol.Ny - 1 - idxrow, idx_batch * vol.Nz + idxslice);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idxrow++;
                        point1_axis += pix_interval;
                    } else {
                        float coef = det_col1_axis - bound0;
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idxcol, vol.Ny - 1 - idxrow, idx_batch * vol.Nz + idxslice);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        atomicAdd(s_proj + idxd, temp);
                        temp = 0;
                        bound0 = det_col1_axis;
                        idxd ++;
                        if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                    }
                }
                if (temp != 0) atomicAdd(s_proj + idxd, temp);
            }
        }
        __syncthreads();
        if (idx_det_col < geo.Nu) {
            s_proj[tx] *= vol.dx / (coef1 * coef2);
            int idx_det_col_out = (sinv >= 0) ? idx_det_col : (geo.Nu - 1 - idx_det_col);
            proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out] = s_proj[tx];
        }
    }
}


__global__ void coneflat_forward_t_cuda_kernel(
    float* __restrict__ image,
    const float* __restrict__ proj,
    const float* __restrict__ ang,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    const int num_view) {

    const int idx_batch = blockIdx.x / num_view;
    const int idx_view = blockIdx.x % num_view;
    const int idx_det_row = blockIdx.y;
    const int idx_det_col0 = blockIdx.z * blockDim.x;
    const int tx = threadIdx.x;
    const int idx_det_col = idx_det_col0 + tx;
    const int max_num_det_col = ((idx_det_col0 + blockDim.x) > geo.Nu) ? (geo.Nu - idx_det_col0) : blockDim.x;
    __shared__ float s_proj[BLOCK_DIM];
    __shared__ float s_row0_axis[BLOCK_DIM];
    __shared__ float s_row1_axis[BLOCK_DIM];
    __shared__ float s_col_axis[BLOCK_DIM + 1];

    float sinv = sin(ang[idx_view]);
    float cosv = cos(ang[idx_view]);
    float src_x = - sinv * geo.SOD;
    float src_y = cosv * geo.SOD;
    float virtual_du = geo.SOD / geo.SDD * geo.du;
    float virtual_shift_u = geo.SOD / geo.SDD * geo.shift_u;
    float det_row0_z = ((idx_det_row - geo.Nv / 2.0) * geo.dv + geo.shift_v) * geo.SOD / geo.SDD;
    float det_row1_z = det_row0_z + geo.SOD / geo.SDD * geo.dv;
    float det_col0_x;
    float det_col0_y;
    float det_col1_x;
    float det_col1_y;
    if (cosv * cosv > 0.5) {
        if (idx_det_col < geo.Nu) {
            int idx_det_col_out = (cosv >= 0) ? idx_det_col : (geo.Nu - 1 - idx_det_col);
            s_proj[tx] = proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out];
        } else {
            s_proj[tx] = 0;
        }
        float det_col_y;
        if (cosv >= 0) {
            det_col0_x = ((idx_det_col0 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
            det_col0_y = ((idx_det_col0 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
            det_col1_x = ((idx_det_col + 1 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
            det_col1_y = ((idx_det_col + 1 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
            det_col_y = ((idx_det_col + 0.5 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
        } else {
            det_col0_x = ((geo.Nu / 2.0 - idx_det_col0) * virtual_du + virtual_shift_u) * cosv;
            det_col0_y = ((geo.Nu / 2.0 - idx_det_col0) * virtual_du + virtual_shift_u) * sinv;
            det_col1_x = ((geo.Nu / 2.0 - idx_det_col - 1) * virtual_du + virtual_shift_u) * cosv;
            det_col1_y = ((geo.Nu / 2.0 - idx_det_col - 1) * virtual_du + virtual_shift_u) * sinv;
            det_col_y = ((geo.Nu / 2.0 - idx_det_col - 0.5) * virtual_du + virtual_shift_u) * sinv;
        }
        float det_col0_axis = xrecon::map2x(src_x, src_y, det_col0_x, det_col0_y);
        float det_col1_axis = xrecon::map2x(src_x, src_y, det_col1_x, det_col1_y);
        float det_row0_axis = src_y / (src_y - det_col_y) * det_row0_z;
        float det_row1_axis = src_y / (src_y - det_col_y) * det_row1_z;
        if (tx == 0) s_col_axis[tx] = det_col0_axis;
        s_col_axis[tx + 1] = det_col1_axis;
        s_row0_axis[tx] = det_row0_axis;
        s_row1_axis[tx] = det_row1_axis;
        __syncthreads();
        float coef1 = (det_col1_axis - s_col_axis[tx]) * (det_row1_axis - det_row0_axis);
        float coef2 = xrecon::triangle_cosine((det_col1_axis + s_col_axis[tx]) / 2.0 - src_x, (det_row1_axis + det_row0_axis) / 2.0, src_y);
        s_proj[tx] *= vol.dy / (coef1 * coef2);
        __syncthreads();
        float point0_x = - vol.Nx / 2.0 * vol.dx + vol.shift_x;
        float point1_x = vol.Nx / 2.0 * vol.dx + vol.shift_x;
        float point0_z = - vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float point1_z = vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float det_interval = virtual_du * abs(cosv);
        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int idxrow = i * blockDim.x + tx;
            if (idxrow < vol.Ny) {
                float point_y = (vol.Ny / 2.0 - idxrow - 0.5) * vol.dy + vol.shift_y;
                float point0_axis = xrecon::map2x(src_x, src_y, point0_x, point_y);
                float point1_axis = xrecon::map2x(src_x, src_y, point1_x, point_y);
                float point0_axis_z = src_y / (src_y - point_y) * point0_z;
                float point1_axis_z = src_y / (src_y - point_y) * point1_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Nx;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float bound0 = max(point0_axis, s_col_axis[0]);
                int idxcol;
                int idxd;
                if (point0_axis == bound0) {
                    float bound0_to_det = src_y * bound0 / ((bound0 - src_x) * sinv / cosv + src_y);
                    idxcol = 0;
                    idxd = floor((bound0_to_det - det_col0_x) / det_interval);
                } else {
                    idxcol = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                }
                point1_axis = (idxcol + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                while(idxcol < vol.Nx && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idxslice;
                    if (bound0_z == point0_axis_z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idxslice + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = point1_axis - bound0;
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idxslice) * vol.Ny + idxrow) * vol.Nx + idxcol, temp);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idxcol++;
                        point1_axis += pix_interval;
                    } else {
                        float coef = det_col1_axis - bound0;
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idxslice) * vol.Ny + idxrow) * vol.Nx + idxcol, temp);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = det_col1_axis;
                        idxd ++;
                        if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                    }
                }
            }
        }
    } else {
        if (idx_det_col < geo.Nu) {
            int idx_det_col_out = (sinv >= 0) ? idx_det_col : (geo.Nu - 1 - idx_det_col);
            s_proj[tx] = proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out];
        } else {
            s_proj[tx] = 0;
        }
        float det_col_x;
        if (sinv >= 0) {
            det_col0_x = ((idx_det_col0 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
            det_col0_y = ((idx_det_col0 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
            det_col1_x = ((idx_det_col + 1 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
            det_col1_y = ((idx_det_col + 1 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * sinv;
            det_col_x = ((idx_det_col + 0.5 - geo.Nu / 2.0) * virtual_du + virtual_shift_u) * cosv;
        } else {
            det_col0_x = ((geo.Nu / 2.0 - idx_det_col0) * virtual_du + virtual_shift_u) * cosv;
            det_col0_y = ((geo.Nu / 2.0 - idx_det_col0) * virtual_du + virtual_shift_u) * sinv;
            det_col1_x = ((geo.Nu / 2.0 - idx_det_col - 1) * virtual_du + virtual_shift_u) * cosv;
            det_col1_y = ((geo.Nu / 2.0 - idx_det_col - 1) * virtual_du + virtual_shift_u) * sinv;
            det_col_x = ((geo.Nu / 2.0 - idx_det_col - 0.5) * virtual_du + virtual_shift_u) * cosv;
        }
        float det_col0_axis = xrecon::map2y(src_x, src_y, det_col0_x, det_col0_y);
        float det_col1_axis = xrecon::map2y(src_x, src_y, det_col1_x, det_col1_y);
        float det_row0_axis = src_x / (src_x - det_col_x) * det_row0_z;
        float det_row1_axis = src_x / (src_x - det_col_x) * det_row1_z;
        if (tx == 0) s_col_axis[tx] = det_col0_axis;
        s_col_axis[tx + 1] = det_col1_axis;
        s_row0_axis[tx] = det_row0_axis;
        s_row1_axis[tx] = det_row1_axis;
        __syncthreads();
        float coef1 = (det_col1_axis - s_col_axis[tx]) * (det_row1_axis - det_row0_axis);
        float coef2 = xrecon::triangle_cosine((det_col1_axis + s_col_axis[tx]) / 2.0 - src_y, (det_row1_axis + det_row0_axis) / 2.0, src_x);
        s_proj[tx] *= vol.dx / (coef1 * coef2);
        __syncthreads();
        float point0_y = - vol.Ny / 2.0 * vol.dy + vol.shift_y;
        float point1_y = vol.Ny / 2.0 * vol.dy + vol.shift_y;
        float point0_z = - vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float point1_z = vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float det_interval = virtual_du * abs(sinv);
        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int idxcol = i * blockDim.x + tx;
            if (idxcol < vol.Nx) {
                float point_x = (idxcol - vol.Nx / 2.0 + 0.5) * vol.dx + vol.shift_x;
                float point0_axis = xrecon::map2y(src_x, src_y, point_x, point0_y);
                float point1_axis = xrecon::map2y(src_x, src_y, point_x, point1_y);
                float point0_axis_z = src_x / (src_x - point_x) * point0_z;
                float point1_axis_z = src_x / (src_x - point_x) * point1_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Ny;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float bound0 = max(point0_axis, s_col_axis[0]);
                int idxrow;
                int idxd;
                if (point0_axis == bound0) {
                    float bound0_to_det = src_x * bound0 / ((bound0 - src_y) * cosv / sinv + src_x);
                    idxrow = 0;
                    idxd = floor((bound0_to_det - det_col0_y) / det_interval);
                } else {
                    idxrow = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                }
                point1_axis = (idxrow + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                while(idxrow < vol.Ny && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idxslice;
                    if (bound0_z == point0_axis_z) {
                        idxslice = 0;
                    } else {
                        idxslice = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idxslice + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = point1_axis - bound0;
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idxslice) * vol.Ny + vol.Ny - 1 - idxrow) * vol.Nx + idxcol, temp);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idxrow++;
                        point1_axis += pix_interval;
                    } else {
                        float coef = det_col1_axis - bound0;
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idxslice) * vol.Ny + vol.Ny - 1 - idxrow) * vol.Nx + idxcol, temp);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = det_col1_axis;
                        idxd ++;
                        if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                    }
                }
            }
        }
    }
}


namespace xrecon {

void forward3d_flat_cuda(
    float *image,
    float *proj,
    float *ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view) {

    CudaTexture3D texture3D;
    texture3D.create(image, vol.Nx, vol.Ny, num_batch * vol.Nz);

    int num_block_z = (geo.Nu - 1) / BLOCK_DIM + 1;
    const dim3 grid_dim(num_batch * num_view, geo.Nv, num_block_z);

    coneflat_forward_cuda_kernel<<<grid_dim, BLOCK_DIM>>>(
        proj, texture3D.texObj, ang, vol, geo, num_view);
}
void forward3d_t_flat_cuda(
    float *image,
    float *proj,
    float *ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view) {


    int num_block_z = (geo.Nu - 1) / BLOCK_DIM + 1;
    const dim3 grid_dim(num_batch * num_view, geo.Nv, num_block_z);

    coneflat_forward_t_cuda_kernel<<<grid_dim, BLOCK_DIM>>>(
        image, proj, ang, vol, geo, num_view);
}




}
