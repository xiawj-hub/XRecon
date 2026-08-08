#include <cuda.h>
#include "utils.h"
#include "xrmath.h"

#define BLOCK_DIM 256
#define M_PI 3.14159265358979323846f

__device__ __forceinline__ float clamp_unit(float x) {
    return fminf(1.0f, fmaxf(0.0f, x));
}

__device__ __forceinline__ float smooth_step_weight(float x) {
    x = clamp_unit(x);
    return 0.5f * (1.0f - cosf(M_PI * x));
}

__device__ __forceinline__ float helical_window_quality(
    float beta,
    float beta_start,
    float beta_end,
    float gamma,
    float gamma_max) {

    float beta_low = fminf(beta_start, beta_end);
    float beta_high = fmaxf(beta_start, beta_end);
    if (beta < beta_low || beta > beta_high) return 0.0f;

    float gamma_clamp = fminf(fmaxf(gamma, -gamma_max), gamma_max);
    float left_width = fmaxf(1.0e-6f, 2.0f * (gamma_max - gamma_clamp));
    float right_width = fmaxf(1.0e-6f, 2.0f * (gamma_max + gamma_clamp));
    float left_end = beta_low + left_width;
    float right_start = beta_high - right_width;

    if (beta < left_end) return smooth_step_weight((beta - beta_low) / left_width);
    if (beta > right_start) return smooth_step_weight((beta_high - beta) / right_width);
    return 1.0f;
}

__device__ __forceinline__ float helical_longitudinal_window(
    float beta,
    float beta_start,
    float beta_end,
    float d_beta) {

    float beta_low = fminf(beta_start, beta_end);
    float beta_high = fmaxf(beta_start, beta_end);
    if (beta < beta_low || beta > beta_high) return 0.0f;

    float span = beta_high - beta_low;
    float width = fminf(0.25f * M_PI, 0.1f * span);
    width = fmaxf(width, 4.0f * fabsf(d_beta));
    if (width <= 1.0e-6f || 2.0f * width >= span) return 1.0f;

    float left = smooth_step_weight((beta - beta_low) / width);
    float right = smooth_step_weight((beta_high - beta) / width);
    return fminf(left, right);
}

__device__ __forceinline__ float helical_center_turn_window(
    float beta,
    float beta_center,
    float beta_start,
    float beta_end,
    float d_beta) {

    float beta_low = fminf(beta_start, beta_end);
    float beta_high = fmaxf(beta_start, beta_end);
    if (beta < beta_low || beta > beta_high) return 0.0f;

    float abs_d_beta = fmaxf(fabsf(d_beta), 1.0e-6f);
    float half_span = 0.5f * (beta_high - beta_low);
    float distance = fabsf(beta - beta_center);
    float center_radius = fminf(M_PI, half_span);
    if (distance <= center_radius) return 1.0f;

    float outer_width = fmaxf(half_span - center_radius, 4.0f * abs_d_beta);
    return smooth_step_weight((half_span - distance) / outer_width);
}

__device__ __forceinline__ float helical_source_z_at_beta(
    float beta,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    int num_view) {

    if (num_view <= 1) return source_z[0];

    float angle0 = angles[0];
    float angle1 = angles[num_view - 1];
    float beta_range = angle1 - angle0;
    if (fabsf(beta_range) <= 1.0e-6f) return source_z[0];

    float dz_dbeta = (source_z[num_view - 1] - source_z[0]) / beta_range;
    return source_z[0] + (beta - angle0) * dz_dbeta;
}

__device__ __forceinline__ float helical_view_edge_taper(
    float beta,
    float beta_center,
    float beta_start,
    float beta_end,
    float d_beta) {

    float beta_low = fminf(beta_start, beta_end);
    float beta_high = fmaxf(beta_start, beta_end);
    if (beta < beta_low || beta > beta_high) return 0.0f;

    float span = beta_high - beta_low;
    float half_span = 0.5f * span;
    if (span <= M_PI + 8.0f * fabsf(d_beta)) return 1.0f;

    float center_radius = fminf(M_PI, half_span);
    float distance = fabsf(beta - beta_center);
    if (distance <= center_radius) return 1.0f;

    float edge_width = fmaxf(half_span - center_radius, 4.0f * fabsf(d_beta));
    return smooth_step_weight((half_span - distance) / edge_width);
}

__device__ __forceinline__ float helical_center_beta(
    float point_z,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    int idxview_start,
    int idxview_end,
    int num_view) {

    float beta_s = angles[idxview_start];
    float beta_e = angles[idxview_end];
    if (num_view <= 1) return 0.5f * (beta_s + beta_e);

    float angle0 = angles[0];
    float angle1 = angles[num_view - 1];
    float z0 = source_z[0];
    float z1 = source_z[num_view - 1];
    float beta_range = angle1 - angle0;
    float dz_dbeta = (fabsf(beta_range) > 1.0e-6f) ? (z1 - z0) / beta_range : 0.0f;

    if (fabsf(dz_dbeta) <= 1.0e-8f) return 0.5f * (beta_s + beta_e);

    float beta = angle0 + (point_z - z0) / dz_dbeta;
    float beta_low = fminf(beta_s, beta_e);
    float beta_high = fmaxf(beta_s, beta_e);
    return fminf(fmaxf(beta, beta_low), beta_high);
}

__device__ __forceinline__ void helical_view_window(
    float point_z,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    xrecon::Geometry3d geo,
    int num_view,
    int& view_start,
    int& view_end) {

    if (num_view <= 1) {
        view_start = 0;
        view_end = 0;
        return;
    }

    float angle0 = angles[0];
    float angle1 = angles[num_view - 1];
    float z0 = source_z[0];
    float z1 = source_z[num_view - 1];
    float beta_range = angle1 - angle0;
    if (fabsf(beta_range) < 1.0e-6f) {
        view_start = 0;
        view_end = num_view - 1;
        return;
    }
    float d_beta = beta_range / fmaxf(1.0f, float(num_view - 1));
    float dz_dbeta = (z1 - z0) / beta_range;

    if (fabsf(d_beta) < 1.0e-8f || fabsf(dz_dbeta) < 1.0e-8f) {
        view_start = 0;
        view_end = num_view - 1;
        return;
    }

    float gamma_max = geo.du * (geo.Nu - 1) * 0.5f;
    float half_scan_span = (M_PI + 2.0f * gamma_max) * fmaxf(geo.helical_halfscan_coef, 1.0f);
    float window_span = 2.0f * M_PI;

    if (geo.helical_view_mode == 1) {
        window_span = half_scan_span;
    } else if (geo.helical_view_mode == 2) {
        window_span = 2.0f * M_PI;
    } else if (geo.helical_view_mode == 3) {
        window_span = fmaxf(0.25f, geo.helical_view_turns) * 2.0f * M_PI;
    } else {
        float pitch_per_turn = fabsf(dz_dbeta) * 2.0f * M_PI;
        float iso_detector_height = fmaxf(geo.dv * geo.Nv * geo.SOD / geo.SDD, geo.dv);
        float normalized_pitch = pitch_per_turn / fmaxf(iso_detector_height, 1.0e-6f);
        if (normalized_pitch >= 1.25f) {
            window_span = half_scan_span;
        } else if (normalized_pitch >= 0.45f) {
            window_span = 2.0f * M_PI;
        } else if (normalized_pitch >= 0.25f) {
            window_span = 3.0f * M_PI;
        } else {
            window_span = 5.0f * M_PI;
        }
    }
    int num_window_view = int(ceilf(window_span / fabsf(d_beta)));

    float beta_center = angle0 + (point_z - z0) / dz_dbeta;
    int view_center = int(roundf((beta_center - angle0) / d_beta));
    view_start = max(0, view_center - num_window_view / 2);
    view_end = min(num_view - 1, view_start + num_window_view - 1);
    view_start = max(0, view_end - num_window_view + 1);
}

__device__ __forceinline__ float conjugate_redundancy_weight(
    float beta,
    float gamma,
    float beta_center,
    float beta_start,
    float beta_end,
    float d_beta,
    float gamma_max) {

    float beta_span = fabsf(beta_end - beta_start);
    float abs_d_beta = fmaxf(fabsf(d_beta), 1.0e-6f);
    if (beta_span >= 2.0f * M_PI - 2.0f * abs_d_beta) {
        float num_turns = fmaxf(1.0f, floorf((beta_span + 2.0f * abs_d_beta) / (2.0f * M_PI)));
        return 0.5f * helical_center_turn_window(beta, beta_center, beta_start, beta_end, d_beta) / num_turns;
    }

    float quality = helical_window_quality(beta, beta_start, beta_end, gamma, gamma_max);
    if (quality <= 0.0f) return 0.0f;

    float direction = (d_beta >= 0.0f) ? 1.0f : -1.0f;
    float denom = 0.0f;
    for (int k = -4; k <= 4; ++k) {
        float turn = direction * 2.0f * M_PI * float(k);
        denom += helical_window_quality(beta + turn, beta_start, beta_end, gamma, gamma_max);
        denom += helical_window_quality(beta + direction * (M_PI + 2.0f * gamma) + turn, beta_start, beta_end, -gamma, gamma_max);
    }
    return quality / fmaxf(denom, 1.0e-6f);
}

__device__ __forceinline__ float tang3d_conjugate_weight(
    float point_x,
    float point_y,
    float point_z,
    float beta,
    float gamma,
    float src_z,
    float ray_xy,
    float beta_center,
    float beta_start,
    float beta_end,
    float d_beta,
    float gamma_max,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    int num_view,
    xrecon::Geometry3d geo) {

    if (geo.helical_weight_mode == 0) {
        return conjugate_redundancy_weight(beta, gamma, beta_center, beta_start, beta_end, d_beta, gamma_max);
    }

    float beta_span = fabsf(beta_end - beta_start);
    float direct_quality;
    float conjugate_quality;
    float direction = (beta < beta_center) ? 1.0f : -1.0f;
    float beta_c = beta + direction * (M_PI + 2.0f * gamma);

    if (beta_span < 2.0f * M_PI - 2.0f * fabsf(d_beta)) {
        direct_quality = helical_window_quality(beta, beta_start, beta_end, gamma, gamma_max);
        conjugate_quality = helical_window_quality(beta_c, beta_start, beta_end, -gamma, gamma_max);
    } else {
        direct_quality = helical_view_edge_taper(beta, beta_center, beta_start, beta_end, d_beta);
        conjugate_quality = helical_view_edge_taper(beta_c, beta_center, beta_start, beta_end, d_beta);
    }
    if (direct_quality <= 0.0f) return 0.0f;
    if (conjugate_quality <= 0.0f) return 1.0f;

    float beta_c_unshifted = beta_c;
    float src_z_c = helical_source_z_at_beta(beta_c_unshifted, angles, source_z, num_view);
    float src_x_c = -sinf(beta_c) * geo.SOD;
    float src_y_c =  cosf(beta_c) * geo.SOD;
    float ray_x_c = point_x - src_x_c;
    float ray_y_c = point_y - src_y_c;
    float ray_xy_c = sqrtf(ray_x_c * ray_x_c + ray_y_c * ray_y_c);

    float tan_alpha = fabsf(point_z - src_z) / fmaxf(ray_xy, 1.0e-6f);
    float tan_alpha_c = fabsf(point_z - src_z_c) / fmaxf(ray_xy_c, 1.0e-6f);
    float k = fminf(fmaxf(geo.helical_tang_k, 0.0f), 32.0f);
    float g = powf(fmaxf(tan_alpha, 1.0e-6f), k);
    float g_c = powf(fmaxf(tan_alpha_c, 1.0e-6f), k);

    float weight_2d = direct_quality / fmaxf(direct_quality + conjugate_quality, 1.0e-6f);
    float denom = direct_quality * g_c + conjugate_quality * g;
    float weight_3d = (direct_quality * g_c) / fmaxf(denom, 1.0e-6f);
    float blend = clamp_unit(geo.helical_tang_blend);
    return (1.0f - blend) * weight_2d + blend * weight_3d;
}

__device__ __forceinline__ bool helical_voxel_center_is_measured(
    float point_x,
    float point_y,
    float point_z,
    float beta,
    float source_z_view,
    float source_shift_radius,
    float source_shift_z,
    xrecon::Geometry3d geo,
    float du,
    float shift_u) {

    float sod = geo.SOD + source_shift_radius;
    float sdd = geo.SDD + source_shift_radius;
    float sinv = sinf(beta);
    float cosv = cosf(beta);
    float src_x = -sinv * sod;
    float src_y = cosv * sod;
    float src_z = source_z_view + source_shift_z;

    float x_beta = cosv * point_x - sinv * point_y;
    float y_beta = sinv * point_x + cosv * point_y;
    float gamma = atan2f(y_beta, sod - x_beta);
    float det_col = (gamma - shift_u) / du + geo.Nu * 0.5f - 0.5f;
    if (det_col < -0.5f || det_col > float(geo.Nu) - 0.5f) return false;

    float ray_x = point_x - src_x;
    float ray_y = point_y - src_y;
    float ray_xy = sqrtf(ray_x * ray_x + ray_y * ray_y);
    if (ray_xy <= 1.0e-6f) return false;

    float det_z = src_z + sdd * (point_z - src_z) / ray_xy;
    float det_row = (det_z - source_z_view - geo.shift_v) / geo.dv + geo.Nv * 0.5f - 0.5f;
    return det_row >= -0.5f && det_row <= float(geo.Nv) - 0.5f;
}

__device__ __forceinline__ float projection_bilinear_sample(
    const float* __restrict__ proj,
    int idx_batch,
    int idx_view,
    float idx_row,
    float idx_col,
    int num_view,
    xrecon::Geometry3d geo) {

    int col0 = int(floorf(idx_col));
    int row0 = int(floorf(idx_row));
    float dc = idx_col - float(col0);
    float dr = idx_row - float(row0);

    if (col0 < 0 || col0 + 1 >= geo.Nu || row0 < 0 || row0 + 1 >= geo.Nv) return 0.0f;

    int base = (idx_batch * num_view + idx_view) * geo.Nv * geo.Nu;
    float p00 = proj[base + row0 * geo.Nu + col0];
    float p01 = proj[base + row0 * geo.Nu + col0 + 1];
    float p10 = proj[base + (row0 + 1) * geo.Nu + col0];
    float p11 = proj[base + (row0 + 1) * geo.Nu + col0 + 1];
    float p0 = p00 + dc * (p01 - p00);
    float p1 = p10 + dc * (p11 - p10);
    return p0 + dr * (p1 - p0);
}

__global__ void helical_arc_forward_cuda_kernel(
    float* __restrict__ proj,
    cudaTextureObject_t tex_obj,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    const float* __restrict__ source_shift_radius,
    const float* __restrict__ source_shift_angle,
    const float* __restrict__ source_shift_z,
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

    float sod = geo.SOD + source_shift_radius[idx_view];
    float sdd = geo.SDD + source_shift_radius[idx_view];
    float du = geo.du * geo.SDD / sdd;
    float shift_u = geo.shift_u * geo.SDD / sdd
        - source_shift_angle[idx_view] * (sdd - sod) / sdd;
    float sinv = sin(angles[idx_view] + source_shift_angle[idx_view]);
    float cosv = cos(angles[idx_view] + source_shift_angle[idx_view]);
    float src_x = - sinv * sod;
    float src_y = cosv * sod;
    float src_z = source_z[idx_view] + source_shift_z[idx_view];
    float det_row0_z = (idx_det_row - geo.Nv / 2.0) * geo.dv + source_z[idx_view] + geo.shift_v;
    float det_row1_z = det_row0_z + geo.dv;
    float det_col0_ang;
    float det_col1_ang;
    int flag;
    if (cosv * cosv > 0.5) {
        s_proj[tx] = 0;
        float det_col_y;
        if (cosv >= 0) {
            det_col0_ang = (idx_det_col0 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col1_ang = (idx_det_col + 1 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col_y = - cos(det_col1_ang - 0.5 * du) * sdd + src_y;
            flag = 1;
        } else {
            det_col0_ang = (geo.Nu / 2.0 - idx_det_col0) * du + shift_u + angles[idx_view];
            det_col1_ang = (geo.Nu / 2.0 - idx_det_col - 1) * du + shift_u + angles[idx_view];
            det_col_y = - cos(det_col1_ang + 0.5 * du) * sdd + src_y;
            flag = - 1;
        }
        float det_col0_axis = xrecon::map2x(src_x, src_y, det_col0_ang);
        float det_col1_axis = xrecon::map2x(src_x, src_y, det_col1_ang);
        float det_row0_axis = src_y / (src_y - det_col_y) * (det_row0_z - src_z) + src_z;
        float det_row1_axis = src_y / (src_y - det_col_y) * (det_row1_z - src_z) + src_z;
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
        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int idx_y = i * blockDim.x + tx;
            if (idx_y < vol.Ny) {
                float point_y = (vol.Ny / 2.0 - idx_y - 0.5) * vol.dy + vol.shift_y;
                float point0_axis = xrecon::map2x(src_x, src_y, point0_x, point_y);
                float point1_axis = xrecon::map2x(src_x, src_y, point1_x, point_y);
                float point0_axis_z = src_y / (src_y - point_y) * (point0_z - src_z) + src_z;
                float point1_axis_z = src_y / (src_y - point_y) * (point1_z - src_z) + src_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Nx;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float tan0 = ((src_x - point0_axis) == 0)? 1e10 : src_y / (src_x - point0_axis);
                float tan1 = ((shift_u + angles[idx_view]) == 0)? 1e10 : - 1 / tan(shift_u + angles[idx_view]);
                float delta = atan((tan0 - tan1) / (1 + tan0 * tan1));
                int idxd = floor(geo.Nu / 2.0 - idx_det_col0 + delta * flag / du);
                int idx_x;
                float bound0;
                if (idxd < 0) {
                    bound0 = s_col_axis[0];
                    idx_x = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                } else {
                    bound0 = point0_axis;
                    idx_x = 0;
                }
                point1_axis = (idx_x + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                float temp = 0;
                while(idx_x < vol.Nx && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idx_z;
                    if (bound0_z == point0_axis_z) {
                        idx_z = 0;
                    } else {
                        idx_z = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idx_z + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = point1_axis - bound0;
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idx_x, idx_y, idx_batch * vol.Nz + idx_z);
                            bound0_z = point1_axis_z;
                            idx_z++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idx_x++;
                        point1_axis += pix_interval;
                    } else {
                        float coef = det_col1_axis - bound0;
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idx_x, idx_y, idx_batch * vol.Nz + idx_z);
                            bound0_z = point1_axis_z;
                            idx_z++;
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
            det_col0_ang = (idx_det_col0 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col1_ang = (idx_det_col + 1 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col_x = sin(det_col1_ang - 0.5 * du) * sdd + src_x;
            flag = 1;
        } else {
            det_col0_ang = (geo.Nu / 2.0 - idx_det_col0) * du + shift_u + angles[idx_view];
            det_col1_ang = (geo.Nu / 2.0 - idx_det_col - 1) * du + shift_u + angles[idx_view];
            det_col_x = sin(det_col1_ang + 0.5 * du) * sdd + src_x;
            flag = - 1;
        }
        float det_col0_axis = xrecon::map2y(src_x, src_y, det_col0_ang);
        float det_col1_axis = xrecon::map2y(src_x, src_y, det_col1_ang);
        float det_row0_axis = src_x / (src_x - det_col_x) * (det_row0_z - src_z) + src_z;
        float det_row1_axis = src_x / (src_x - det_col_x) * (det_row1_z - src_z) + src_z;
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
        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int idx_x = i * blockDim.x + tx;
            if (idx_x < vol.Nx) {
                float point_x = (idx_x - vol.Nx / 2.0 + 0.5) * vol.dx + vol.shift_x;
                float point0_axis = xrecon::map2y(src_x, src_y, point_x, point0_y);
                float point1_axis = xrecon::map2y(src_x, src_y, point_x, point1_y);
                float point0_axis_z = src_x / (src_x - point_x) * (point0_z - src_z) + src_z;
                float point1_axis_z = src_x / (src_x - point_x) * (point1_z - src_z) + src_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Ny;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float tan0 = (src_y - point0_axis) / src_x;
                float tan1 = - 1 / tan(shift_u + angles[idx_view]);
                float delta = atan((tan0 - tan1) / (1 + tan0 * tan1));
                int idxd = floor(geo.Nu / 2.0 - idx_det_col0 + delta * flag / du);
                int idx_y;
                float bound0;
                if (idxd < 0) {
                    bound0 = s_col_axis[0];
                    idx_y = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                } else {
                    bound0 = point0_axis;
                    idx_y = 0;
                }
                point1_axis = (idx_y + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                float temp = 0;
                while(idx_y < vol.Ny && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idx_z;
                    if (bound0_z == point0_axis_z) {
                        idx_z = 0;
                    } else {
                        idx_z = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idx_z + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = point1_axis - bound0;
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idx_x, vol.Ny - 1 - idx_y, idx_batch * vol.Nz + idx_z);
                            bound0_z = point1_axis_z;
                            idx_z++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idx_y++;
                        point1_axis += pix_interval;
                    } else {
                        float coef = det_col1_axis - bound0;
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idx_x, vol.Ny - 1 - idx_y, idx_batch * vol.Nz + idx_z);
                            bound0_z = point1_axis_z;
                            idx_z++;
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


__global__ void helical_arc_forward_t_cuda_kernel(
    float* __restrict__ image,
    const float* __restrict__ proj,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    const float* __restrict__ source_shift_radius,
    const float* __restrict__ source_shift_angle,
    const float* __restrict__ source_shift_z,
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

    float sod = geo.SOD + source_shift_radius[idx_view];
    float sdd = geo.SDD + source_shift_radius[idx_view];
    float du = geo.du * geo.SDD / sdd;
    float shift_u = geo.shift_u * geo.SDD / sdd
        - source_shift_angle[idx_view] * (sdd - sod) / sdd;
    float sinv = sin(angles[idx_view] + source_shift_angle[idx_view]);
    float cosv = cos(angles[idx_view] + source_shift_angle[idx_view]);
    float src_x = - sinv * sod;
    float src_y = cosv * sod;
    float src_z = source_z[idx_view] + source_shift_z[idx_view];
    float det_row0_z = (idx_det_row - geo.Nv / 2.0) * geo.dv + source_z[idx_view] + geo.shift_v;
    float det_row1_z = det_row0_z + geo.dv;
    float det_col0_ang;
    float det_col1_ang;
    int flag;
    if (cosv * cosv > 0.5) {
        if (idx_det_col < geo.Nu) {
            int idx_det_col_out = (cosv >= 0) ? idx_det_col : (geo.Nu - 1 - idx_det_col);
            s_proj[tx] = proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out];
        } else {
            s_proj[tx] = 0;
        }
        float det_col_y;
        if (cosv >= 0) {
            det_col0_ang = (idx_det_col0 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col1_ang = (idx_det_col + 1 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col_y = - cos(det_col1_ang - 0.5 * du) * sdd + src_y;
            flag = 1;
        } else {
            det_col0_ang = (geo.Nu / 2.0 - idx_det_col0) * du + shift_u + angles[idx_view];
            det_col1_ang = (geo.Nu / 2.0 - idx_det_col - 1) * du + shift_u + angles[idx_view];
            det_col_y = - cos(det_col1_ang + 0.5 * du) * sdd + src_y;
            flag = - 1;
        }
        float det_col0_axis = xrecon::map2x(src_x, src_y, det_col0_ang);
        float det_col1_axis = xrecon::map2x(src_x, src_y, det_col1_ang);
        float det_row0_axis = src_y / (src_y - det_col_y) * (det_row0_z - src_z) + src_z;
        float det_row1_axis = src_y / (src_y - det_col_y) * (det_row1_z - src_z) + src_z;
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
        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int idx_y = i * blockDim.x + tx;
            if (idx_y < vol.Ny) {
                float point_y = (vol.Ny / 2.0 - idx_y - 0.5) * vol.dy + vol.shift_y;
                float point0_axis = xrecon::map2x(src_x, src_y, point0_x, point_y);
                float point1_axis = xrecon::map2x(src_x, src_y, point1_x, point_y);
                float point0_axis_z = src_y / (src_y - point_y) * (point0_z - src_z) + src_z;
                float point1_axis_z = src_y / (src_y - point_y) * (point1_z - src_z) + src_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Nx;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float tan0 = ((src_x - point0_axis) == 0)? 1e10 : src_y / (src_x - point0_axis);
                float tan1 = ((shift_u + angles[idx_view]) == 0)? 1e10 : - 1 / tan(shift_u + angles[idx_view]);
                float delta = atan((tan0 - tan1) / (1 + tan0 * tan1));
                int idxd = floor(geo.Nu / 2.0 - idx_det_col0 + delta * flag / du);
                int idx_x;
                float bound0;
                if (idxd < 0) {
                    bound0 = s_col_axis[0];
                    idx_x = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                } else {
                    bound0 = point0_axis;
                    idx_x = 0;
                }
                point1_axis = (idx_x + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                while(idx_x < vol.Nx && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idx_z;
                    if (bound0_z == point0_axis_z) {
                        idx_z = 0;
                    } else {
                        idx_z = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idx_z + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = point1_axis - bound0;
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idx_z) * vol.Ny + idx_y) * vol.Nx + idx_x, temp);
                            bound0_z = point1_axis_z;
                            idx_z++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idx_x++;
                        point1_axis += pix_interval;
                    } else {
                        float coef = det_col1_axis - bound0;
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idx_z) * vol.Ny + idx_y) * vol.Nx + idx_x, temp);
                            bound0_z = point1_axis_z;
                            idx_z++;
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
            det_col0_ang = (idx_det_col0 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col1_ang = (idx_det_col + 1 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col_x = sin(det_col1_ang - 0.5 * du) * sdd + src_x;
            flag = 1;
        } else {
            det_col0_ang = (geo.Nu / 2.0 - idx_det_col0) * du + shift_u + angles[idx_view];
            det_col1_ang = (geo.Nu / 2.0 - idx_det_col - 1) * du + shift_u + angles[idx_view];
            det_col_x = sin(det_col1_ang + 0.5 * du) * sdd + src_x;
            flag = - 1;
        }
        float det_col0_axis = xrecon::map2y(src_x, src_y, det_col0_ang);
        float det_col1_axis = xrecon::map2y(src_x, src_y, det_col1_ang);
        float det_row0_axis = src_x / (src_x - det_col_x) * (det_row0_z - src_z) + src_z;
        float det_row1_axis = src_x / (src_x - det_col_x) * (det_row1_z - src_z) + src_z;
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
        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int idx_x = i * blockDim.x + tx;
            if (idx_x < vol.Nx) {
                float point_x = (idx_x - vol.Nx / 2.0 + 0.5) * vol.dx + vol.shift_x;
                float point0_axis = xrecon::map2y(src_x, src_y, point_x, point0_y);
                float point1_axis = xrecon::map2y(src_x, src_y, point_x, point1_y);
                float point0_axis_z = src_x / (src_x - point_x) * (point0_z - src_z) + src_z;
                float point1_axis_z = src_x / (src_x - point_x) * (point1_z - src_z) + src_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Ny;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float tan0 = (src_y - point0_axis) / src_x;
                float tan1 = - 1 / tan(shift_u + angles[idx_view]);
                float delta = atan((tan0 - tan1) / (1 + tan0 * tan1));
                int idxd = floor(geo.Nu / 2.0 - idx_det_col0 + delta * flag / du);
                int idx_y;
                float bound0;
                if (idxd < 0) {
                    bound0 = s_col_axis[0];
                    idx_y = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                } else {
                    bound0 = point0_axis;
                    idx_y = 0;
                }
                point1_axis = (idx_y + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                while(idx_y < vol.Ny && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idx_z;
                    if (bound0_z == point0_axis_z) {
                        idx_z = 0;
                    } else {
                        idx_z = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idx_z + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = point1_axis - bound0;
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idx_z) * vol.Ny + vol.Ny - 1 - idx_y) * vol.Nx + idx_x, temp);
                            bound0_z = point1_axis_z;
                            idx_z++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idx_y++;
                        point1_axis += pix_interval;
                    } else {
                        float coef = det_col1_axis - bound0;
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idx_z) * vol.Ny + vol.Ny - 1 - idx_y) * vol.Nx + idx_x, temp);
                            bound0_z = point1_axis_z;
                            idx_z++;
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



template<bool FBPWEIGHT>
__global__ void helical_arc_backward_cuda_kernel(
    float* __restrict__ image,
    const float* __restrict__ proj,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    const float* __restrict__ source_shift_radius,
    const float* __restrict__ source_shift_angle,
    const float* __restrict__ source_shift_z,
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

    float sod = geo.SOD + source_shift_radius[idx_view];
    float sdd = geo.SDD + source_shift_radius[idx_view];
    float du = geo.du * geo.SDD / sdd;
    float shift_u = geo.shift_u * geo.SDD / sdd
        - source_shift_angle[idx_view] * (sdd - sod) / sdd;
    float sinv = sin(angles[idx_view] + source_shift_angle[idx_view]);
    float cosv = cos(angles[idx_view] + source_shift_angle[idx_view]);
    float src_x = - sinv * sod;
    float src_y = cosv * sod;
    float src_z = source_z[idx_view] + source_shift_z[idx_view];
    float det_col_width = sod * du;
    float det_row0_z = (idx_det_row - geo.Nv / 2.0) * geo.dv + source_z[idx_view] + geo.shift_v;
    float det_row1_z = det_row0_z + geo.dv;
    float det_col0_ang;
    float det_col1_ang;
    int flag;
    if (cosv * cosv > 0.5) {
        if (idx_det_col < geo.Nu) {
            int idx_det_col_out = (cosv >= 0) ? idx_det_col : (geo.Nu - 1 - idx_det_col);
            s_proj[tx] = proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out] * det_col_width;
        } else {
            s_proj[tx] = 0;
        }
        float det_col_y;
        if (cosv >= 0) {
            det_col0_ang = (idx_det_col0 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col1_ang = (idx_det_col + 1 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col_y = - cos(det_col1_ang - 0.5 * du) * sdd + src_y;
            flag = 1;
        } else {
            det_col0_ang = (geo.Nu / 2.0 - idx_det_col0) * du + shift_u + angles[idx_view];
            det_col1_ang = (geo.Nu / 2.0 - idx_det_col - 1) * du + shift_u + angles[idx_view];
            det_col_y = - cos(det_col1_ang + 0.5 * du) * sdd + src_y;
            flag = - 1;
        }
        float det_col0_axis = xrecon::map2x(src_x, src_y, det_col0_ang);
        float det_col1_axis = xrecon::map2x(src_x, src_y, det_col1_ang);
        float det_row0_axis = src_y / (src_y - det_col_y) * (det_row0_z - src_z) + src_z;
        float det_row1_axis = src_y / (src_y - det_col_y) * (det_row1_z - src_z) + src_z;
        if (tx == 0) s_col_axis[tx] = det_col0_axis;
        s_col_axis[tx + 1] = det_col1_axis;
        s_row0_axis[tx] = det_row0_axis;
        s_row1_axis[tx] = det_row1_axis;
        __syncthreads();
        float point0_x = - vol.Nx / 2.0 * vol.dx + vol.shift_x;
        float point1_x = vol.Nx / 2.0 * vol.dx + vol.shift_x;
        float point0_z = - vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float point1_z = vol.Nz / 2.0 * vol.dz + vol.shift_z;
        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int idx_y = i * blockDim.x + tx;
            if (idx_y < vol.Ny) {
                float point_y = (vol.Ny / 2.0 - idx_y - 0.5) * vol.dy + vol.shift_y;
                float point0_axis = xrecon::map2x(src_x, src_y, point0_x, point_y);
                float point1_axis = xrecon::map2x(src_x, src_y, point1_x, point_y);
                float point0_axis_z = src_y / (src_y - point_y) * (point0_z - src_z) + src_z;
                float point1_axis_z = src_y / (src_y - point_y) * (point1_z - src_z) + src_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Nx;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float pix_area = pix_interval * pix_interval_z;
                float tan0 = ((src_x - point0_axis) == 0)? 1e10 : src_y / (src_x - point0_axis);
                float tan1 = ((shift_u + angles[idx_view]) == 0)? 1e10 : - 1 / tan(shift_u + angles[idx_view]);
                float delta = atan((tan0 - tan1) / (1 + tan0 * tan1));
                int idxd = floor(geo.Nu / 2.0 - idx_det_col0 + delta * flag / du);
                int idx_x;
                float bound0;
                if (idxd < 0) {
                    bound0 = s_col_axis[0];
                    idx_x = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                } else {
                    bound0 = point0_axis;
                    idx_x = 0;
                }
                float point_x = (idx_x + 0.5) * vol.dx + point0_x;
                point1_axis = (idx_x + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                while(idx_x < vol.Nx && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idx_z;
                    if (bound0_z == point0_axis_z) {
                        idx_z = 0;
                    } else {
                        idx_z = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idx_z + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = (point1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idx_z) * vol.Ny + idx_y) * vol.Nx + idx_x, temp);
                            bound0_z = point1_axis_z;
                            idx_z++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idx_x++;
                        point_x += vol.dx;
                        point1_axis += pix_interval;
                    } else {
                        float coef = (det_col1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idx_z) * vol.Ny + idx_y) * vol.Nx + idx_x, temp);
                            bound0_z = point1_axis_z;
                            idx_z++;
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
            s_proj[tx] = proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out] * det_col_width;
        } else {
            s_proj[tx] = 0;
        }
        float det_col_x;
        if (sinv >= 0) {
            det_col0_ang = (idx_det_col0 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col1_ang = (idx_det_col + 1 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col_x = sin(det_col1_ang - 0.5 * du) * sdd + src_x;
            flag = 1;
        } else {
            det_col0_ang = (geo.Nu / 2.0 - idx_det_col0) * du + shift_u + angles[idx_view];
            det_col1_ang = (geo.Nu / 2.0 - idx_det_col - 1) * du + shift_u + angles[idx_view];
            det_col_x = sin(det_col1_ang + 0.5 * du) * sdd + src_x;
            flag = - 1;
        }
        float det_col0_axis = xrecon::map2y(src_x, src_y, det_col0_ang);
        float det_col1_axis = xrecon::map2y(src_x, src_y, det_col1_ang);
        float det_row0_axis = src_x / (src_x - det_col_x) * (det_row0_z - src_z) + src_z;
        float det_row1_axis = src_x / (src_x - det_col_x) * (det_row1_z - src_z) + src_z;
        if (tx == 0) s_col_axis[tx] = det_col0_axis;
        s_col_axis[tx + 1] = det_col1_axis;
        s_row0_axis[tx] = det_row0_axis;
        s_row1_axis[tx] = det_row1_axis;
        __syncthreads();
        float point0_y = - vol.Ny / 2.0 * vol.dy + vol.shift_y;
        float point1_y = vol.Ny / 2.0 * vol.dy + vol.shift_y;
        float point0_z = - vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float point1_z = vol.Nz / 2.0 * vol.dz + vol.shift_z;
        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int idx_x = i * blockDim.x + tx;
            if (idx_x < vol.Nx) {
                float point_x = (idx_x - vol.Nx / 2.0 + 0.5) * vol.dx + vol.shift_x;
                float point0_axis = xrecon::map2y(src_x, src_y, point_x, point0_y);
                float point1_axis = xrecon::map2y(src_x, src_y, point_x, point1_y);
                float point0_axis_z = src_x / (src_x - point_x) * (point0_z - src_z) + src_z;
                float point1_axis_z = src_x / (src_x - point_x) * (point1_z - src_z) + src_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Ny;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float pix_area = pix_interval * pix_interval_z;
                float tan0 = (src_y - point0_axis) / src_x;
                float tan1 = - 1 / tan(shift_u + angles[idx_view]);
                float delta = atan((tan0 - tan1) / (1 + tan0 * tan1));
                int idxd = floor(geo.Nu / 2.0 - idx_det_col0 + delta * flag / du);
                int idx_y;
                float bound0;
                if (idxd < 0) {
                    bound0 = s_col_axis[0];
                    idx_y = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                } else {
                    bound0 = point0_axis;
                    idx_y = 0;
                }
                float point_y = (idx_y + 0.5) * vol.dy + point0_y;
                point1_axis = (idx_y + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                while(idx_y < vol.Ny && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idx_z;
                    if (bound0_z == point0_axis_z) {
                        idx_z = 0;
                    } else {
                        idx_z = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idx_z + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = (point1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idx_z) * vol.Ny + vol.Ny - 1 - idx_y) * vol.Nx + idx_x, temp);
                            bound0_z = point1_axis_z;
                            idx_z++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idx_y++;
                        point_y += vol.dy;
                        point1_axis += pix_interval;
                    } else {
                        float coef = (det_col1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            float temp = (point1_axis_z - bound0_z) * coef * s_proj[idxd];
                            atomicAdd(image + ((idx_batch * vol.Nz + idx_z) * vol.Ny + vol.Ny - 1 - idx_y) * vol.Nx + idx_x, temp);
                            bound0_z = point1_axis_z;
                            idx_z++;
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


__global__ void helical_arc_weighted_backward_cuda_kernel(
    float* __restrict__ image,
    const float* __restrict__ proj,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    const float* __restrict__ source_shift_radius,
    const float* __restrict__ source_shift_angle,
    const float* __restrict__ source_shift_z,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    const int num_view) {

    const int idx_batch = blockIdx.x / vol.Nz;
    const int idx_z = blockIdx.x % vol.Nz;
    const int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.z * blockDim.y + threadIdx.y;

    if (idx_y >= vol.Ny || idx_x >= vol.Nx || idx_z >= vol.Nz) return;
    float point_x = (idx_x - vol.Nx / 2.0f + 0.5f) * vol.dx + vol.shift_x;
    float point_y = (vol.Ny * 0.5f - idx_y - 0.5f) * vol.dy + vol.shift_y;
    float point_z = (idx_z - vol.Nz / 2.0f + 0.5f) * vol.dz + vol.shift_z;
    float voxel_value = 0.0f;

    float gamma_max = geo.du * (geo.Nu - 1) * 0.5f;
    float d_beta = (num_view > 1) ? (angles[num_view - 1] - angles[0]) / float(num_view - 1) : 0.0f;

    int idxview_start, idxview_end;
    helical_view_window(point_z, angles, source_z, geo, num_view, idxview_start, idxview_end);

    float beta_s = angles[idxview_start];
    float beta_e = angles[idxview_end];
    float beta_center = helical_center_beta(point_z, angles, source_z, idxview_start, idxview_end, num_view);
    float det_col_width = geo.SOD * geo.du;
    float valid_weight_sum = 0.0f;

    for (int idx_view = idxview_start; idx_view <= idxview_end; idx_view++) {
        float sod = geo.SOD + source_shift_radius[idx_view];
        float sdd = geo.SDD + source_shift_radius[idx_view];
        float du = geo.du * geo.SDD / sdd;
        float shift_u = geo.shift_u * geo.SDD / sdd
            - source_shift_angle[idx_view] * (sdd - sod) / sdd;
        float beta = angles[idx_view] + source_shift_angle[idx_view];
        float sinv = sinf(beta);
        float cosv = cosf(beta);
        float src_x = -sinv * sod;
        float src_y = cosv * sod;
        float src_z = source_z[idx_view] + source_shift_z[idx_view];
        float Xb = cosv * point_x - sinv * point_y;
        float Yb = sinv * point_x + cosv * point_y;
        float gamma = atan2f(Yb, sod - Xb);
        float ray_x = point_x - src_x;
        float ray_y = point_y - src_y;
        float ray_xy = sqrtf(ray_x * ray_x + ray_y * ray_y);
        if (ray_xy <= 1.0e-6f) continue;
        float weight = tang3d_conjugate_weight(
            point_x, point_y, point_z, beta, gamma, src_z, ray_xy,
            beta_center, beta_s, beta_e, d_beta, gamma_max,
            angles, source_z, num_view, geo);
        if (weight == 0.0f) continue;
        if (helical_voxel_center_is_measured(
                point_x, point_y, point_z, beta, source_z[idx_view],
                source_shift_radius[idx_view], source_shift_z[idx_view],
                geo, du, shift_u)) {
            valid_weight_sum += weight;
        }
    }

    float ideal_weight_sum = (fabsf(d_beta) > 1.0e-8f) ? (M_PI / fabsf(d_beta)) : valid_weight_sum;
    float weight_scale = ideal_weight_sum / fmaxf(valid_weight_sum, 1.0e-6f);
    weight_scale = fminf(32.0f, fmaxf(0.03125f, weight_scale));

    for (int idx_view = idxview_start; idx_view <= idxview_end; idx_view++) {
        float sod = geo.SOD + source_shift_radius[idx_view];
        float sdd = geo.SDD + source_shift_radius[idx_view];
        float du = geo.du * geo.SDD / sdd;
        float shift_u = geo.shift_u * geo.SDD / sdd
            - source_shift_angle[idx_view] * (sdd - sod) / sdd;
        float beta = angles[idx_view] + source_shift_angle[idx_view];
        
        float sinv = sin(beta);
        float cosv = cos(beta);
        float src_x = - sinv * sod;
        float src_y = cosv * sod;
        float src_z = source_z[idx_view] + source_shift_z[idx_view];
        float det_row0_z = (- geo.Nv / 2.0) * geo.dv + source_z[idx_view] + geo.shift_v;
        float det_row1_z = (geo.Nv / 2.0) * geo.dv + source_z[idx_view] + geo.shift_v;

        float Xb = cosv * point_x - sinv * point_y;
        float Yb = sinv * point_x + cosv * point_y;
        float gamma = atan2f(Yb, sod - Xb);

        bool useXAxis = (cosv * cosv > 0.5f);
        bool keepDim = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

        float ray_x = point_x - src_x;
        float ray_y = point_y - src_y;
        float ray_xy = sqrtf(ray_x * ray_x + ray_y * ray_y);
        if (ray_xy <= 1.0e-6f) continue;
        float weight = tang3d_conjugate_weight(
            point_x, point_y, point_z, beta, gamma, src_z, ray_xy,
            beta_center, beta_s, beta_e, d_beta, gamma_max,
            angles, source_z, num_view, geo);
        if (weight == 0.0f) continue;

        float det_col0_ang = keepDim?
            ( (- geo.Nu / 2.0) * du + shift_u + angles[idx_view] ) :
            ( (geo.Nu / 2.0) * du + shift_u + angles[idx_view] );
        float det_col1_ang = keepDim?
            ( (geo.Nu / 2.0) * du + shift_u + angles[idx_view] ) :
            ( (- geo.Nu / 2.0) * du + shift_u + angles[idx_view] );

        float point0_axis, point1_axis, point0_axis_z, point1_axis_z, det_col0_axis, det_col1_axis;

        if (useXAxis) {
            point0_axis = xrecon::map2x(src_x, src_y, point_x - 0.5f * vol.dx, point_y);
            point1_axis = xrecon::map2x(src_x, src_y, point_x + 0.5f * vol.dx, point_y);
            point0_axis_z = src_y / (src_y - point_y) * (point_z - 0.5f * vol.dz - src_z) + src_z;
            point1_axis_z = src_y / (src_y - point_y) * (point_z + 0.5f * vol.dz - src_z) + src_z;
            det_col0_axis = xrecon::map2x(src_x, src_y, det_col0_ang);
            det_col1_axis = xrecon::map2x(src_x, src_y, det_col1_ang);
        } else {
            point0_axis = xrecon::map2y(src_x, src_y, point_x, point_y - 0.5f * vol.dy);
            point1_axis = xrecon::map2y(src_x, src_y, point_x, point_y + 0.5f * vol.dy);
            point0_axis_z = src_x / (src_x - point_x) * (point_z - 0.5f * vol.dz - src_z) + src_z;
            point1_axis_z = src_x / (src_x - point_x) * (point_z + 0.5f * vol.dz - src_z) + src_z;
            det_col0_axis = xrecon::map2y(src_x, src_y, det_col0_ang);
            det_col1_axis = xrecon::map2y(src_x, src_y, det_col1_ang);
        }

        float bound0 = max(point0_axis, det_col0_axis);
        float bound1 = min(point1_axis, det_col1_axis);
        xrecon::Point2d src(src_x, src_y);
        float det_start_ang = xrecon::detector_axis_angle(src, useXAxis, bound0, shift_u + angles[idx_view]);
        float det_start = (det_start_ang - shift_u - angles[idx_view]) / du;
        int idxd = floor(geo.Nu / 2.0 + det_start);
        idxd = min(geo.Nu - 1, max(0, idxd));
        float pixel_sum = 0.0f;
        float det_cur_ang = (idxd - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
        det_cur_ang = keepDim? det_cur_ang : (det_cur_ang + du);
        float pix_area = (point1_axis_z - point0_axis_z) * (point1_axis - point0_axis);
        while (bound0 < bound1 && idxd >= 0 && idxd < geo.Nu) {
            float det_next_ang = keepDim ? (det_cur_ang + du) : (det_cur_ang - du);
            float det_col1_axis = useXAxis ?
                xrecon::map2x(src_x, src_y, det_next_ang) :
                xrecon::map2y(src_x, src_y, det_next_ang);

            float det_col_y = - cos((det_cur_ang + det_next_ang) * 0.5f) * sdd + src_y;
            float det_col_x = sin((det_cur_ang + det_next_ang) * 0.5f) * sdd + src_x;

            float det_row0_axis, det_row1_axis;
            if (useXAxis) {
                det_row0_axis = src_y / (src_y - det_col_y) * (det_row0_z - src_z) + src_z;
                det_row1_axis = src_y / (src_y - det_col_y) * (det_row1_z - src_z) + src_z;
            } else {
                det_row0_axis = src_x / (src_x - det_col_x) * (det_row0_z - src_z) + src_z;
                det_row1_axis = src_x / (src_x - det_col_x) * (det_row1_z - src_z) + src_z;
            }
            float det_interval_z = (det_row1_axis - det_row0_axis) / geo.Nv;
            float bound0_z = max(point0_axis_z, det_row0_axis);
            float bound1_z = min(point1_axis_z, det_row1_axis);
            int idx_det_row_z = floor((bound0_z - det_row0_axis) / det_interval_z);
            det_row1_axis = (idx_det_row_z + 1) * det_interval_z + det_row0_axis;
            float next_bound = fminf(bound1, det_col1_axis);
            float col_overlap = next_bound - bound0;
            if (col_overlap <= 0.0f) {
                bound0 = det_col1_axis;
                det_cur_ang = det_next_ang;
                idxd += keepDim ? 1 : -1;
                continue;
            }
            float coef = col_overlap / pix_area;
            coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);
            float temp = 0.f;
            
            while (bound0_z < bound1_z) {
                det_row1_axis = min(bound1_z, det_row1_axis);
                if (idx_det_row_z >= 0 && idx_det_row_z < geo.Nv) {
                    float val = proj[(idx_batch * num_view + idx_view) * geo.Nv * geo.Nu + idx_det_row_z * geo.Nu + idxd] * det_col_width;
                    temp += (det_row1_axis - bound0_z) * val;
                }
                bound0_z = det_row1_axis;
                idx_det_row_z++;
                det_row1_axis += det_interval_z;
            }
            pixel_sum += temp * coef;
            bound0 = next_bound;
            det_cur_ang = det_next_ang;
            idxd += keepDim? 1 : -1;            
        }
        voxel_value += pixel_sum * weight * weight_scale;
    }
    image[((idx_batch * vol.Nz + idx_z) * vol.Ny + idx_y) * vol.Nx + idx_x] = voxel_value;
}

__global__ void helical_arc_weighted_backward_t_cuda_kernel(
    const float* __restrict__ image,
    float* __restrict__ proj,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    const float* __restrict__ source_shift_radius,
    const float* __restrict__ source_shift_angle,
    const float* __restrict__ source_shift_z,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    const int num_view) {

    const int idx_batch = blockIdx.x / vol.Nz;
    const int idx_z = blockIdx.x % vol.Nz;
    const int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.z * blockDim.y + threadIdx.y;

    if (idx_y >= vol.Ny || idx_x >= vol.Nx || idx_z >= vol.Nz) return;

    const float image_value = image[(idx_batch * vol.Nz + idx_z) * vol.Ny * vol.Nx + idx_y * vol.Nx + idx_x];
    if (image_value == 0.0f) return;

    float point_x = (idx_x - vol.Nx / 2.0f + 0.5f) * vol.dx + vol.shift_x;
    float point_y = (vol.Ny / 2.0f - idx_y - 0.5f) * vol.dy + vol.shift_y;
    float point_z = (idx_z - vol.Nz / 2.0f + 0.5f) * vol.dz + vol.shift_z;

    float gamma_max = geo.du * (geo.Nu - 1) * 0.5f;
    float d_beta = (num_view > 1) ? (angles[num_view - 1] - angles[0]) / float(num_view - 1) : 0.0f;

    int idxview_start, idxview_end;
    helical_view_window(point_z, angles, source_z, geo, num_view, idxview_start, idxview_end);

    float beta_s = angles[idxview_start];
    float beta_e = angles[idxview_end];
    float beta_center = helical_center_beta(point_z, angles, source_z, idxview_start, idxview_end, num_view);
    float valid_weight_sum = 0.0f;

    for (int idx_view = idxview_start; idx_view <= idxview_end; idx_view++) {
        float sod = geo.SOD + source_shift_radius[idx_view];
        float sdd = geo.SDD + source_shift_radius[idx_view];
        float du = geo.du * geo.SDD / sdd;
        float shift_u = geo.shift_u * geo.SDD / sdd
            - source_shift_angle[idx_view] * (sdd - sod) / sdd;
        float beta = angles[idx_view] + source_shift_angle[idx_view];

        float sinv = sinf(beta);
        float cosv = cosf(beta);
        float src_x = - sinv * sod;
        float src_y = cosv * sod;
        float src_z = source_z[idx_view] + source_shift_z[idx_view];

        float ray_x = point_x - src_x;
        float ray_y = point_y - src_y;
        float ray_xy = sqrtf(ray_x * ray_x + ray_y * ray_y);
        if (ray_xy <= 1.0e-6f) continue;

        float Xb = cosv * point_x - sinv * point_y;
        float Yb = sinv * point_x + cosv * point_y;
        float gamma = atan2f(Yb, sod - Xb);
        float weight = tang3d_conjugate_weight(
            point_x, point_y, point_z, beta, gamma, src_z, ray_xy,
            beta_center, beta_s, beta_e, d_beta, gamma_max,
            angles, source_z, num_view, geo);
        if (weight == 0.0f) continue;
        if (helical_voxel_center_is_measured(
                point_x, point_y, point_z, beta, source_z[idx_view],
                source_shift_radius[idx_view], source_shift_z[idx_view],
                geo, du, shift_u)) {
            valid_weight_sum += weight;
        }
    }

    float ideal_weight_sum = (fabsf(d_beta) > 1.0e-8f) ? (M_PI / fabsf(d_beta)) : valid_weight_sum;
    float weight_scale = ideal_weight_sum / fmaxf(valid_weight_sum, 1.0e-6f);
    weight_scale = fminf(32.0f, fmaxf(0.03125f, weight_scale));
    float det_col_width = geo.SOD * geo.du;

    for (int idx_view = idxview_start; idx_view <= idxview_end; idx_view++) {
        float sod = geo.SOD + source_shift_radius[idx_view];
        float sdd = geo.SDD + source_shift_radius[idx_view];
        float du = geo.du * geo.SDD / sdd;
        float shift_u = geo.shift_u * geo.SDD / sdd
            - source_shift_angle[idx_view] * (sdd - sod) / sdd;
        float beta = angles[idx_view] + source_shift_angle[idx_view];
        float sinv = sinf(beta);
        float cosv = cosf(beta);
        float src_x = -sinv * sod;
        float src_y = cosv * sod;
        float src_z = source_z[idx_view] + source_shift_z[idx_view];

        float ray_x = point_x - src_x;
        float ray_y = point_y - src_y;
        float ray_xy = sqrtf(ray_x * ray_x + ray_y * ray_y);
        if (ray_xy <= 1.0e-6f) continue;

        float Xb = cosv * point_x - sinv * point_y;
        float Yb = sinv * point_x + cosv * point_y;
        float gamma = atan2f(Yb, sod - Xb);
        float weight = tang3d_conjugate_weight(
            point_x, point_y, point_z, beta, gamma, src_z, ray_xy,
            beta_center, beta_s, beta_e, d_beta, gamma_max,
            angles, source_z, num_view, geo);
        if (weight == 0.0f) continue;

        bool useXAxis = (cosv * cosv > 0.5f);
        bool keepDim = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));

        float det_col0_ang = keepDim ?
            ((-geo.Nu * 0.5f) * du + shift_u + angles[idx_view]) :
            (( geo.Nu * 0.5f) * du + shift_u + angles[idx_view]);
        float det_col1_ang = keepDim ?
            (( geo.Nu * 0.5f) * du + shift_u + angles[idx_view]) :
            ((-geo.Nu * 0.5f) * du + shift_u + angles[idx_view]);

        float point0_axis, point1_axis, point0_axis_z, point1_axis_z;
        float det_col0_axis, det_col1_axis;
        if (useXAxis) {
            point0_axis = xrecon::map2x(src_x, src_y, point_x - 0.5f * vol.dx, point_y);
            point1_axis = xrecon::map2x(src_x, src_y, point_x + 0.5f * vol.dx, point_y);
            point0_axis_z = src_y / (src_y - point_y) * (point_z - 0.5f * vol.dz - src_z) + src_z;
            point1_axis_z = src_y / (src_y - point_y) * (point_z + 0.5f * vol.dz - src_z) + src_z;
            det_col0_axis = xrecon::map2x(src_x, src_y, det_col0_ang);
            det_col1_axis = xrecon::map2x(src_x, src_y, det_col1_ang);
        } else {
            point0_axis = xrecon::map2y(src_x, src_y, point_x, point_y - 0.5f * vol.dy);
            point1_axis = xrecon::map2y(src_x, src_y, point_x, point_y + 0.5f * vol.dy);
            point0_axis_z = src_x / (src_x - point_x) * (point_z - 0.5f * vol.dz - src_z) + src_z;
            point1_axis_z = src_x / (src_x - point_x) * (point_z + 0.5f * vol.dz - src_z) + src_z;
            det_col0_axis = xrecon::map2y(src_x, src_y, det_col0_ang);
            det_col1_axis = xrecon::map2y(src_x, src_y, det_col1_ang);
        }

        float bound0 = max(point0_axis, det_col0_axis);
        float bound1 = min(point1_axis, det_col1_axis);
        if (!(bound0 < bound1)) continue;

        xrecon::Point2d src(src_x, src_y);
        float det_start_ang = xrecon::detector_axis_angle(src, useXAxis, bound0, shift_u + angles[idx_view]);
        float det_start = (det_start_ang - shift_u - angles[idx_view]) / du;
        int idxd = floor(geo.Nu * 0.5f + det_start);
        idxd = min(geo.Nu - 1, max(0, idxd));
        float det_cur_ang = (idxd - geo.Nu * 0.5f) * du + shift_u + angles[idx_view];
        det_cur_ang = keepDim ? det_cur_ang : (det_cur_ang + du);

        float pix_area = (point1_axis_z - point0_axis_z) * (point1_axis - point0_axis);
        if (fabsf(pix_area) <= 1.0e-12f) continue;

        int proj_base = (idx_batch * num_view + idx_view) * geo.Nv * geo.Nu;
        while (bound0 < bound1 && idxd >= 0 && idxd < geo.Nu) {
            float det_next_ang = keepDim ? (det_cur_ang + du) : (det_cur_ang - du);
            float det_next_axis = useXAxis ?
                xrecon::map2x(src_x, src_y, det_next_ang) :
                xrecon::map2y(src_x, src_y, det_next_ang);
            float next_bound = fminf(bound1, det_next_axis);
            float col_overlap = next_bound - bound0;
            if (col_overlap <= 0.0f) {
                bound0 = det_next_axis;
                det_cur_ang = det_next_ang;
                idxd += keepDim ? 1 : -1;
                continue;
            }

            float det_col_y = -cosf((det_cur_ang + det_next_ang) * 0.5f) * sdd + src_y;
            float det_col_x = sinf((det_cur_ang + det_next_ang) * 0.5f) * sdd + src_x;
            float det_row0_z = (-geo.Nv * 0.5f) * geo.dv + source_z[idx_view] + geo.shift_v;
            float det_row1_z = ( geo.Nv * 0.5f) * geo.dv + source_z[idx_view] + geo.shift_v;
            float det_row0_axis = useXAxis
                ? src_y / (src_y - det_col_y) * (det_row0_z - src_z) + src_z
                : src_x / (src_x - det_col_x) * (det_row0_z - src_z) + src_z;
            float det_row1_axis = useXAxis
                ? src_y / (src_y - det_col_y) * (det_row1_z - src_z) + src_z
                : src_x / (src_x - det_col_x) * (det_row1_z - src_z) + src_z;
            float det_interval_z = (det_row1_axis - det_row0_axis) / geo.Nv;
            float bound0_z = max(point0_axis_z, det_row0_axis);
            float bound1_z = min(point1_axis_z, det_row1_axis);
            int idx_det_row_z = floor((bound0_z - det_row0_axis) / det_interval_z);
            float det_next_z = (idx_det_row_z + 1) * det_interval_z + det_row0_axis;
            float coef = image_value * weight * weight_scale * col_overlap / pix_area;
            coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);

            while (bound0_z < bound1_z) {
                det_next_z = min(bound1_z, det_next_z);
                if (idx_det_row_z >= 0 && idx_det_row_z < geo.Nv) {
                    float row_overlap = det_next_z - bound0_z;
                    atomicAdd(
                        proj + proj_base + idx_det_row_z * geo.Nu + idxd,
                        coef * row_overlap * det_col_width);
                }
                bound0_z = det_next_z;
                idx_det_row_z++;
                det_next_z += det_interval_z;
            }

            bound0 = next_bound;
            det_cur_ang = det_next_ang;
            idxd += keepDim ? 1 : -1;
        }
    }
}

template<bool FBPWEIGHT>
__global__ void helical_arc_backward_t_cuda_kernel(
    float* __restrict__ proj,
    cudaTextureObject_t tex_obj,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    const float* __restrict__ source_shift_radius,
    const float* __restrict__ source_shift_angle,
    const float* __restrict__ source_shift_z,
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

    float sod = geo.SOD + source_shift_radius[idx_view];
    float sdd = geo.SDD + source_shift_radius[idx_view];
    float du = geo.du * geo.SDD / sdd;
    float shift_u = geo.shift_u * geo.SDD / sdd
        - source_shift_angle[idx_view] * (sdd - sod) / sdd;
    float sinv = sin(angles[idx_view] + source_shift_angle[idx_view]);
    float cosv = cos(angles[idx_view] + source_shift_angle[idx_view]);
    float src_x = - sinv * sod;
    float src_y = cosv * sod;
    float src_z = source_z[idx_view] + source_shift_z[idx_view];
    float det_col_width = sod * du;
    float det_row0_z = (idx_det_row - geo.Nv / 2.0) * geo.dv + source_z[idx_view] + geo.shift_v;
    float det_row1_z = det_row0_z + geo.dv;
    float det_col0_ang;
    float det_col1_ang;
    int flag;
    if (cosv * cosv > 0.5) {
        s_proj[tx] = 0;
        float det_col_y;
        if (cosv >= 0) {
            det_col0_ang = (idx_det_col0 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col1_ang = (idx_det_col + 1 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col_y = - cos(det_col1_ang - 0.5 * du) * sdd + src_y;
            flag = 1;
        } else {
            det_col0_ang = (geo.Nu / 2.0 - idx_det_col0) * du + shift_u + angles[idx_view];
            det_col1_ang = (geo.Nu / 2.0 - idx_det_col - 1) * du + shift_u + angles[idx_view];
            det_col_y = - cos(det_col1_ang + 0.5 * du) * sdd + src_y;
            flag = - 1;
        }
        float det_col0_axis = xrecon::map2x(src_x, src_y, det_col0_ang);
        float det_col1_axis = xrecon::map2x(src_x, src_y, det_col1_ang);
        float det_row0_axis = src_y / (src_y - det_col_y) * (det_row0_z - src_z) + src_z;
        float det_row1_axis = src_y / (src_y - det_col_y) * (det_row1_z - src_z) + src_z;
        if (tx == 0) s_col_axis[tx] = det_col0_axis;
        s_col_axis[tx + 1] = det_col1_axis;
        s_row0_axis[tx] = det_row0_axis;
        s_row1_axis[tx] = det_row1_axis;
        __syncthreads();
        float point0_x = - vol.Nx / 2.0 * vol.dx + vol.shift_x;
        float point1_x = vol.Nx / 2.0 * vol.dx + vol.shift_x;
        float point0_z = - vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float point1_z = vol.Nz / 2.0 * vol.dz + vol.shift_z;
        for (int i = 0; (i * blockDim.x) < vol.Ny; i++) {
            int idx_y = i * blockDim.x + tx;
            if (idx_y < vol.Ny) {
                float point_y = (vol.Ny / 2.0 - idx_y - 0.5) * vol.dy + vol.shift_y;
                float point0_axis = xrecon::map2x(src_x, src_y, point0_x, point_y);
                float point1_axis = xrecon::map2x(src_x, src_y, point1_x, point_y);
                float point0_axis_z = src_y / (src_y - point_y) * (point0_z - src_z) + src_z;
                float point1_axis_z = src_y / (src_y - point_y) * (point1_z - src_z) + src_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Nx;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float pix_area = pix_interval * pix_interval_z;
                float tan0 = ((src_x - point0_axis) == 0)? 1e10 : src_y / (src_x - point0_axis);
                float tan1 = ((shift_u + angles[idx_view]) == 0)? 1e10 : - 1 / tan(shift_u + angles[idx_view]);
                float delta = atan((tan0 - tan1) / (1 + tan0 * tan1));
                int idxd = floor(geo.Nu / 2.0 - idx_det_col0 + delta * flag / du);
                int idx_x;
                float bound0;
                if (idxd < 0) {
                    bound0 = s_col_axis[0];
                    idx_x = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                } else {
                    bound0 = point0_axis;
                    idx_x = 0;
                }
                float point_x = (idx_x + 0.5) * vol.dx + point0_x;
                point1_axis = (idx_x + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                float temp = 0;
                while(idx_x < vol.Nx && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idx_z;
                    if (bound0_z == point0_axis_z) {
                        idx_z = 0;
                    } else {
                        idx_z = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idx_z + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = (point1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idx_x, idx_y, idx_batch * vol.Nz + idx_z);
                            bound0_z = point1_axis_z;
                            idx_z++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idx_x++;
                        point_x += vol.dx;
                        point1_axis += pix_interval;
                    } else {
                        float coef = (det_col1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idx_x, idx_y, idx_batch * vol.Nz + idx_z);
                            bound0_z = point1_axis_z;
                            idx_z++;
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
            s_proj[tx] *= det_col_width;
            int idx_det_col_out = (cosv >= 0) ? idx_det_col : (geo.Nu - 1 - idx_det_col);
            proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out] = s_proj[tx];
        }
    } else {
        s_proj[tx] = 0;
        float det_col_x;
        if (sinv >= 0) {
            det_col0_ang = (idx_det_col0 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col1_ang = (idx_det_col + 1 - geo.Nu / 2.0) * du + shift_u + angles[idx_view];
            det_col_x = sin(det_col1_ang - 0.5 * du) * sdd + src_x;
            flag = 1;
        } else {
            det_col0_ang = (geo.Nu / 2.0 - idx_det_col0) * du + shift_u + angles[idx_view];
            det_col1_ang = (geo.Nu / 2.0 - idx_det_col - 1) * du + shift_u + angles[idx_view];
            det_col_x = sin(det_col1_ang + 0.5 * du) * sdd + src_x;
            flag = - 1;
        }
        float det_col0_axis = xrecon::map2y(src_x, src_y, det_col0_ang);
        float det_col1_axis = xrecon::map2y(src_x, src_y, det_col1_ang);
        float det_row0_axis = src_x / (src_x - det_col_x) * (det_row0_z - src_z) + src_z;
        float det_row1_axis = src_x / (src_x - det_col_x) * (det_row1_z - src_z) + src_z;
        if (tx == 0) s_col_axis[tx] = det_col0_axis;
        s_col_axis[tx + 1] = det_col1_axis;
        s_row0_axis[tx] = det_row0_axis;
        s_row1_axis[tx] = det_row1_axis;
        __syncthreads();
        float point0_y = - vol.Ny / 2.0 * vol.dy + vol.shift_y;
        float point1_y = vol.Ny / 2.0 * vol.dy + vol.shift_y;
        float point0_z = - vol.Nz / 2.0 * vol.dz + vol.shift_z;
        float point1_z = vol.Nz / 2.0 * vol.dz + vol.shift_z;
        for (int i = 0; (i * blockDim.x) < vol.Nx; i++) {
            int idx_x = i * blockDim.x + tx;
            if (idx_x < vol.Nx) {
                float point_x = (idx_x - vol.Nx / 2.0 + 0.5) * vol.dx + vol.shift_x;
                float point0_axis = xrecon::map2y(src_x, src_y, point_x, point0_y);
                float point1_axis = xrecon::map2y(src_x, src_y, point_x, point1_y);
                float point0_axis_z = src_x / (src_x - point_x) * (point0_z - src_z) + src_z;
                float point1_axis_z = src_x / (src_x - point_x) * (point1_z - src_z) + src_z;
                float pix_interval = (point1_axis - point0_axis) / vol.Ny;
                float pix_interval_z = (point1_axis_z - point0_axis_z) / vol.Nz;
                float pix_area = pix_interval * pix_interval_z;
                float tan0 = (src_y - point0_axis) / src_x;
                float tan1 = - 1 / tan(shift_u + angles[idx_view]);
                float delta = atan((tan0 - tan1) / (1 + tan0 * tan1));
                int idxd = floor(geo.Nu / 2.0 - idx_det_col0 + delta * flag / du);
                int idx_y;
                float bound0;
                if (idxd < 0) {
                    bound0 = s_col_axis[0];
                    idx_y = floor((bound0 - point0_axis) / pix_interval);
                    idxd = 0;
                } else {
                    bound0 = point0_axis;
                    idx_y = 0;
                }
                float point_y = (idx_y + 0.5) * vol.dy + point0_y;
                point1_axis = (idx_y + 1) * pix_interval + point0_axis;
                if (idxd < max_num_det_col) det_col1_axis = s_col_axis[idxd + 1];
                float temp = 0;
                while(idx_y < vol.Ny && idxd < max_num_det_col) {
                    float bound0_z = max(point0_axis_z, s_row0_axis[idxd]);
                    float bound1_z = min(point1_axis_z, s_row1_axis[idxd]);
                    int idx_z;
                    if (bound0_z == point0_axis_z) {
                        idx_z = 0;
                    } else {
                        idx_z = floor((bound0_z - point0_axis_z) / pix_interval_z);
                    }
                    float point1_axis_z = (idx_z + 1) * pix_interval_z + point0_axis_z;
                    if (point1_axis < det_col1_axis) {
                        float coef = (point1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idx_x, vol.Ny - 1 - idx_y, idx_batch * vol.Nz + idx_z);
                            bound0_z = point1_axis_z;
                            idx_z++;
                            point1_axis_z += pix_interval_z;
                        }
                        bound0 = point1_axis;
                        idx_y++;
                        point_y += vol.dy;
                        point1_axis += pix_interval;
                    } else {
                        float coef = (det_col1_axis - bound0) / pix_area;
                        if (FBPWEIGHT) coef /= xrecon::coordinate_weight(src_x, src_y, point_x, point_y);
                        while (bound0_z < bound1_z && idx_z < vol.Nz) {
                            point1_axis_z = (point1_axis_z > bound1_z) ? bound1_z : point1_axis_z;
                            temp += (point1_axis_z - bound0_z) * coef * tex3D<float>(tex_obj, idx_x, vol.Ny - 1 - idx_y, idx_batch * vol.Nz + idx_z);
                            bound0_z = point1_axis_z;
                            idx_z++;
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
            s_proj[tx] *= det_col_width;
            int idx_det_col_out = (sinv >= 0) ? idx_det_col : (geo.Nu - 1 - idx_det_col);
            proj[(blockIdx.x * geo.Nv + idx_det_row) * geo.Nu + idx_det_col_out] = s_proj[tx];
        }
    }
}


void helical_arc_forward_cuda(
    float *image,
    float *proj,
    float *angles,
    float *source_z,
    float *source_shift_radius,
    float *source_shift_angle,
    float *source_shift_z,
    int num_batch,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    int num_view) {
    CudaTexture3D texture3D;
    texture3D.create(image, vol.Nx, vol.Ny, num_batch * vol.Nz);

    int num_col_blocks = (geo.Nu - 1) / BLOCK_DIM + 1;
    const dim3 grid_size(num_batch * num_view, geo.Nv, num_col_blocks);

    helical_arc_forward_cuda_kernel<<<grid_size, BLOCK_DIM>>>(
        proj, texture3D.texObj, angles, source_z, source_shift_radius, source_shift_angle, source_shift_z,
        vol, geo, num_view
    );

    
}


void helical_arc_forward_t_cuda(
    float *image,
    float *proj,
    float *angles,
    float *source_z,
    float *source_shift_radius,
    float *source_shift_angle,
    float *source_shift_z,
    int num_batch,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    int num_view) {

    int num_col_blocks = (geo.Nu - 1) / BLOCK_DIM + 1;
    const dim3 grid_size(num_batch * num_view, geo.Nv, num_col_blocks);

    helical_arc_forward_t_cuda_kernel<<<grid_size, BLOCK_DIM>>>(
        image, proj, angles, source_z, source_shift_radius, source_shift_angle, source_shift_z,
        vol, geo, num_view
    );
}


void helical_arc_backward_cuda(
    float *image,
    float *proj,
    float *angles,
    float *source_z,
    float *source_shift_radius,
    float *source_shift_angle,
    float *source_shift_z,
    int num_batch,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    int num_view) {

    int num_col_blocks = (geo.Nu - 1) / BLOCK_DIM + 1;
    const dim3 grid_size(num_batch * num_view, geo.Nv, num_col_blocks);

    helical_arc_backward_cuda_kernel<false><<<grid_size, BLOCK_DIM>>>(
        image, proj, angles, source_z, source_shift_radius, source_shift_angle, source_shift_z,
        vol, geo, num_view
    );
}


void helical_arc_backward_t_cuda(
    float *image,
    float *proj,
    float *angles,
    float *source_z,
    float *source_shift_radius,
    float *source_shift_angle,
    float *source_shift_z,
    int num_batch,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    int num_view) {
    CudaTexture3D texture3D;
    texture3D.create(image, vol.Nx, vol.Ny, num_batch * vol.Nz);

    int num_col_blocks = (geo.Nu - 1) / BLOCK_DIM + 1;
    const dim3 grid_size(num_batch * num_view, geo.Nv, num_col_blocks);

    helical_arc_backward_t_cuda_kernel<false><<<grid_size, BLOCK_DIM>>>(
        proj, texture3D.texObj, angles, source_z, source_shift_radius, source_shift_angle, source_shift_z,
        vol, geo, num_view
    );

    
}


__global__ void helical_arc_direct_weighted_backward_cuda_kernel(
    float* __restrict__ image,
    const float* __restrict__ proj,
    const float* __restrict__ angles,
    const float* __restrict__ source_z,
    const float* __restrict__ source_shift_radius,
    const float* __restrict__ source_shift_angle,
    const float* __restrict__ source_shift_z,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    const int num_view) {

    const int idx_batch = blockIdx.x / vol.Nz;
    const int idx_z = blockIdx.x % vol.Nz;
    const int idx_x = blockIdx.y * blockDim.x + threadIdx.x;
    const int idx_y = blockIdx.z * blockDim.y + threadIdx.y;

    if (idx_y >= vol.Ny || idx_x >= vol.Nx || idx_z >= vol.Nz) return;

    const float point_x = (idx_x - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    const float point_y = (vol.Ny * 0.5f - idx_y - 0.5f) * vol.dy + vol.shift_y;
    const float point_z = (idx_z - vol.Nz * 0.5f + 0.5f) * vol.dz + vol.shift_z;

    const float gamma_max = geo.du * (geo.Nu - 1) * 0.5f;
    const float d_beta = (num_view > 1) ? (angles[num_view - 1] - angles[0]) / float(num_view - 1) : 0.0f;

    int idxview_start;
    int idxview_end;
    helical_view_window(point_z, angles, source_z, geo, num_view, idxview_start, idxview_end);
    const float beta_s = angles[idxview_start];
    const float beta_e = angles[idxview_end];
    const float beta_center = helical_center_beta(point_z, angles, source_z, idxview_start, idxview_end, num_view);

    float voxel_value = 0.0f;
    for (int idx_view = idxview_start; idx_view <= idxview_end; ++idx_view) {
        const float sod = geo.SOD + source_shift_radius[idx_view];
        const float sdd = geo.SDD + source_shift_radius[idx_view];
        const float du = geo.du * geo.SDD / sdd;
        const float shift_u = geo.shift_u * geo.SDD / sdd
            - source_shift_angle[idx_view] * (sdd - sod) / sdd;
        const float beta = angles[idx_view] + source_shift_angle[idx_view];
        const float sinv = sinf(beta);
        const float cosv = cosf(beta);
        const float src_x = -sinv * sod;
        const float src_y = cosv * sod;
        const float src_z = source_z[idx_view] + source_shift_z[idx_view];

        const float x_beta = cosv * point_x - sinv * point_y;
        const float y_beta = sinv * point_x + cosv * point_y;
        const float gamma = atan2f(y_beta, sod - x_beta);
        const float det_col = (gamma - shift_u) / du + geo.Nu * 0.5f - 0.5f;
        if (det_col < 0.0f || det_col > float(geo.Nu - 1)) continue;

        const float ray_x = point_x - src_x;
        const float ray_y = point_y - src_y;
        const float ray_xy = sqrtf(ray_x * ray_x + ray_y * ray_y);
        if (ray_xy <= 1.0e-6f) continue;

        const float det_z = src_z + sdd * (point_z - src_z) / ray_xy;
        const float det_row = (det_z - source_z[idx_view] - geo.shift_v) / geo.dv + geo.Nv * 0.5f - 0.5f;
        if (det_row < 0.0f || det_row > float(geo.Nv - 1)) continue;

        const float redundancy_weight = conjugate_redundancy_weight(beta, gamma, beta_center, beta_s, beta_e, d_beta, gamma_max);
        if (redundancy_weight == 0.0f) continue;

        const float sample = projection_bilinear_sample(proj, idx_batch, idx_view, det_row, det_col, num_view, geo);
        const float distance_weight = (sod * sod) / fmaxf(ray_xy * ray_xy, 1.0e-6f);
        voxel_value += sample * distance_weight * redundancy_weight;
    }

    image[((idx_batch * vol.Nz + idx_z) * vol.Ny + idx_y) * vol.Nx + idx_x] = voxel_value;
}


void helical_arc_weighted_backward_cuda(
    float *image,
    float *proj,
    float *angles,
    float *source_z,
    float *source_shift_radius,
    float *source_shift_angle,
    float *source_shift_z,
    int num_batch,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    int num_view) {

    const dim3 block_size(16, 16);
    int block_y = (vol.Ny - 1) / 16 + 1;
    int block_z = (vol.Nx - 1) / 16 + 1;
    const dim3 grid_size(num_batch * vol.Nz, block_y, block_z);

    helical_arc_weighted_backward_cuda_kernel<<<grid_size, block_size>>>(
        image, proj, angles, source_z, source_shift_radius, source_shift_angle, source_shift_z,
        vol, geo, num_view
    );
}


void helical_arc_weighted_backward_t_cuda(
    float *image,
    float *proj,
    float *angles,
    float *source_z,
    float *source_shift_radius,
    float *source_shift_angle,
    float *source_shift_z,
    int num_batch,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    int num_view) {
    const dim3 block_size(16, 16);
    int block_y = (vol.Nx - 1) / 16 + 1;
    int block_z = (vol.Ny - 1) / 16 + 1;
    const dim3 grid_size(num_batch * vol.Nz, block_y, block_z);

    helical_arc_weighted_backward_t_cuda_kernel<<<grid_size, block_size>>>(
        image, proj, angles, source_z, source_shift_radius, source_shift_angle, source_shift_z,
        vol, geo, num_view
    );
}
