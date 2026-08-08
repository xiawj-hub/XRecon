#include <cuda.h>
#include "cone3d_common.cuh"

#define BLOCK_DIM 256


template<bool FBPWEIGHT>
__global__ void coneflat_backward_cuda_kernel(
    float* __restrict__ image,
    const float* __restrict__ proj,
    const float* __restrict__ angles,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    int num_batch,
    int num_view)
{
    const int num_depth_blocks = gridDim.x / num_batch;
    const int idx_batch = blockIdx.x / num_depth_blocks;
    const int idx_z = (blockIdx.x % num_depth_blocks) * blockDim.z + threadIdx.z;
    const int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx_x = blockIdx.z * blockDim.x + threadIdx.x;
    if (idx_batch >= num_batch || idx_x >= vol.Nx || idx_y >= vol.Ny || idx_z >= vol.Nz) return;

    const float px = (idx_x - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    const float py = (vol.Ny * 0.5f - idx_y - 0.5f) * vol.dy + vol.shift_y;
    const float pz = (idx_z - vol.Nz * 0.5f + 0.5f) * vol.dz + vol.shift_z;
    const xrecon::Point2d point(px, py);
    float voxel_value = 0.0f;

    for (int idx_view = 0; idx_view < num_view; ++idx_view) {
        const xrecon::View2d view = xrecon::make_view2d(angles, idx_view, geo);
        const bool use_x_axis = view.cosv * view.cosv > 0.5f;
        const bool keep_dim = (!use_x_axis && view.sinv >= 0.0f) || (use_x_axis && view.cosv >= 0.0f);
        const float half_nu = geo.Nu * 0.5f;
        const float inv_nu = 1.0f / geo.Nu;

        const float idxd0 = keep_dim ? -half_nu : half_nu;
        const float idxd1 = keep_dim ? half_nu : -half_nu;
        const xrecon::Point2d det0((idxd0 * geo.du + geo.shift_u) * geo.SOD / geo.SDD * view.cosv,
                                   (idxd0 * geo.du + geo.shift_u) * geo.SOD / geo.SDD * view.sinv);
        const xrecon::Point2d det1((idxd1 * geo.du + geo.shift_u) * geo.SOD / geo.SDD * view.cosv,
                                   (idxd1 * geo.du + geo.shift_u) * geo.SOD / geo.SDD * view.sinv);
        const float det_step_x = (det1.x - det0.x) * inv_nu;
        const float det_step_y = (det1.y - det0.y) * inv_nu;

        float voxel0, voxel1, z0, z1, det0_axis, det1_axis, bound0, bound1, bound_to_detector;
        if (use_x_axis) {
            voxel0 = xrecon::map2x(view.source, xrecon::Point2d(px - 0.5f * vol.dx, py));
            voxel1 = xrecon::map2x(view.source, xrecon::Point2d(px + 0.5f * vol.dx, py));
            z0 = view.source.y / (view.source.y - py) * (pz - 0.5f * vol.dz);
            z1 = view.source.y / (view.source.y - py) * (pz + 0.5f * vol.dz);
            det0_axis = xrecon::map2x(view.source, det0);
            det1_axis = xrecon::map2x(view.source, det1);
            bound0 = fmaxf(voxel0, det0_axis);
            bound1 = fminf(voxel1, det1_axis);
            bound_to_detector = view.source.y * bound0 / ((bound0 - view.source.x) * (view.sinv / view.cosv) + view.source.y);
        } else {
            voxel0 = xrecon::map2y(view.source, xrecon::Point2d(px, py - 0.5f * vol.dy));
            voxel1 = xrecon::map2y(view.source, xrecon::Point2d(px, py + 0.5f * vol.dy));
            z0 = view.source.x / (view.source.x - px) * (pz - 0.5f * vol.dz);
            z1 = view.source.x / (view.source.x - px) * (pz + 0.5f * vol.dz);
            det0_axis = xrecon::map2y(view.source, det0);
            det1_axis = xrecon::map2y(view.source, det1);
            bound0 = fmaxf(voxel0, det0_axis);
            bound1 = fminf(voxel1, det1_axis);
            bound_to_detector = view.source.x * bound0 / ((bound0 - view.source.y) * (view.cosv / view.sinv) + view.source.x);
        }
        if (!(bound0 < bound1)) continue;

        const float voxel_area = fabsf((voxel1 - voxel0) * (z1 - z0));
        if (voxel_area <= 1e-12f) continue;

        int idx_detector = use_x_axis ? static_cast<int>(floorf((bound_to_detector - det0.x) / det_step_x))
                                      : static_cast<int>(floorf((bound_to_detector - det0.y) / det_step_y));
        idx_detector = max(0, min(geo.Nu - 1, idx_detector));
        xrecon::Point2d det_cur(det0.x + idx_detector * det_step_x, det0.y + idx_detector * det_step_y);
        int idx_col = keep_dim ? idx_detector : (geo.Nu - 1 - idx_detector);

        float view_sum = 0.0f;
        while (bound0 < bound1 && idx_col >= 0 && idx_col < geo.Nu) {
            const xrecon::Point2d det_next(det_cur.x + det_step_x, det_cur.y + det_step_y);
            const float det_next_axis = use_x_axis ? xrecon::map2x(view.source, det_next)
                                                   : xrecon::map2y(view.source, det_next);
            const float next_bound = fminf(bound1, det_next_axis);
            const float col_overlap = next_bound - bound0;
            if (col_overlap <= 0.0f) {
                bound0 = det_next_axis;
                det_cur = det_next;
                idx_col += keep_dim ? 1 : -1;
                continue;
            }

            const xrecon::Point2d detc(0.5f * (det_cur.x + det_next.x), 0.5f * (det_cur.y + det_next.y));
            const float z_scale = use_x_axis
                ? view.source.y / (view.source.y - detc.y)
                : view.source.x / (view.source.x - detc.x);
            const float virtual_v_scale = geo.SOD / geo.SDD;
            const float det_z0_base = z_scale * virtual_v_scale * ((-geo.Nv * 0.5f) * geo.dv + geo.shift_v);
            const float det_dz = z_scale * virtual_v_scale * geo.dv;
            int row_begin;
            int row_end;
            xrecon::detector_row_overlap_range(z0, z1, det_z0_base, det_dz, geo.Nv, row_begin, row_end);

            int idx_row = row_begin;
            while (idx_row <= row_end) {
                const float det_z0 = det_z0_base + idx_row * det_dz;
                const float det_z1 = det_z0 + det_dz;

                const float row_overlap = xrecon::overlap_length(z0, z1, det_z0, det_z1);
                if (row_overlap > 0.0f) {
                    const float proj_value = xrecon::projection_sample(proj, idx_batch, idx_view, idx_row, idx_col, num_view, geo);
                    view_sum += proj_value * xrecon::flat_detector_col_width(geo) * col_overlap * row_overlap / voxel_area;
                }
                ++idx_row;
            }
            bound0 = next_bound;
            det_cur = det_next;
            idx_col += keep_dim ? 1 : -1;
        }

        if (FBPWEIGHT) {
            view_sum *= xrecon::cone_voxel_weight_flat(view, point, geo);
        }
        voxel_value += view_sum;
    }

    image[((idx_batch * vol.Nz + idx_z) * vol.Ny + idx_y) * vol.Nx + idx_x] = voxel_value;
}


template<bool FBPWEIGHT>
__global__ void coneflat_backward_t_cuda_kernel(
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
                float pix_area = pix_interval * pix_interval_z;
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
                float point_x = (idxcol + 0.5) * vol.dx + point0_x;
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
                        float coef = (point1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef *= xrecon::coordinate_weight(src_x, src_y, point_x, point_y, geo.SOD);
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idxcol, idxrow, idx_batch * vol.Nz + idxslice);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idxcol++;
                        point_x += vol.dx;
                        point1_axis += pix_interval;
                    } else {
                        float coef = (det_col1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef *= xrecon::coordinate_weight(src_x, src_y, point_x, point_y, geo.SOD);
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
            s_proj[tx] *= virtual_du;
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
                float pix_area = pix_interval * pix_interval_z;
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
                float point_y = (idxrow + 0.5) * vol.dy + point0_y;
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
                        float coef = (point1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef *= xrecon::coordinate_weight(src_x, src_y, point_x, point_y, geo.SOD);
                        while (bound0_z < bound1_z && idxslice < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idxcol, vol.Ny - 1 - idxrow, idx_batch * vol.Nz + idxslice);
                            bound0_z = point1_axis_z;
                            idxslice++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idxrow++;
                        point_y += vol.dy;
                        point1_axis += pix_interval;
                    } else {
                        float coef = (det_col1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef *= xrecon::coordinate_weight(src_x, src_y, point_x, point_y, geo.SOD);
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
            s_proj[tx] *= virtual_du;
            int idx_det_col_out = (sinv >= 0) ? idx_det_col : (geo.Nu - 1 - idx_det_col);
            proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out] = s_proj[tx];
        }
    }
}


namespace xrecon {

void backward3d_flat_cuda(
    float *image,
    float *proj,
    float *ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view) {
    const dim3 block_dim(8, 8, 4);
    const int num_depth_blocks = (vol.Nz + block_dim.z - 1) / block_dim.z;
    const dim3 grid_dim(num_batch * num_depth_blocks,
                        (vol.Ny + block_dim.y - 1) / block_dim.y,
                        (vol.Nx + block_dim.x - 1) / block_dim.x);

    coneflat_backward_cuda_kernel<false><<<grid_dim, block_dim>>>(
        image, proj, ang, vol, geo, num_batch, num_view);
}
void backward3d_t_flat_cuda(
    float *proj,
    float *image,
    float *ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view) {

    CudaTexture3D texture3D;
    texture3D.create(image, vol.Nx, vol.Ny, num_batch * vol.Nz);

    int num_block_z = (geo.Nu - 1) / BLOCK_DIM + 1;
    const dim3 grid_dim(num_batch * num_view, geo.Nv, num_block_z);

    coneflat_backward_t_cuda_kernel<false><<<grid_dim, BLOCK_DIM>>>(
        proj, texture3D.texObj, ang, vol, geo, num_view);
}
void weighted_backward3d_flat_cuda(
    float *image,
    float *proj,
    float *ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view) {
    const dim3 block_dim(8, 8, 4);
    const int num_depth_blocks = (vol.Nz + block_dim.z - 1) / block_dim.z;
    const dim3 grid_dim(num_batch * num_depth_blocks,
                        (vol.Ny + block_dim.y - 1) / block_dim.y,
                        (vol.Nx + block_dim.x - 1) / block_dim.x);

    coneflat_backward_cuda_kernel<true><<<grid_dim, block_dim>>>(
        image, proj, ang, vol, geo, num_batch, num_view);
}
void weighted_backward3d_t_flat_cuda(
    float *proj,
    float *image,
    float *ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view) {

    CudaTexture3D texture3D;
    texture3D.create(image, vol.Nx, vol.Ny, num_batch * vol.Nz);

    int num_block_z = (geo.Nu - 1) / BLOCK_DIM + 1;
    const dim3 grid_dim(num_batch * num_view, geo.Nv, num_block_z);

    coneflat_backward_t_cuda_kernel<true><<<grid_dim, BLOCK_DIM>>>(
        proj, texture3D.texObj, ang, vol, geo, num_view);
}


}
