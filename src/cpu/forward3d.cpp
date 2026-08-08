#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>
#include "functions.h"
#include "cpu3d_common.h"

namespace xrecon {

namespace {

constexpr int kConeBlock = 256;

inline void zero_buffer(float* data, int64_t size)
{
    std::fill(data, data + size, 0.0f);
}

inline int thread_count(int work_items)
{
    unsigned int hw = std::thread::hardware_concurrency();
    int n = hw == 0 ? 1 : static_cast<int>(hw);
    return std::max(1, std::min(n, work_items));
}

inline int64_t image_index(int batch, int z, int y, int x, Volume3d vol)
{
    return ((static_cast<int64_t>(batch) * vol.Nz + z) * vol.Ny + y) * vol.Nx + x;
}

inline int64_t proj_index(int batch, int view, int row, int col, int num_view, Geometry3d geo)
{
    return ((static_cast<int64_t>(batch) * num_view + view) * geo.Nv + row) * geo.Nu + col;
}

template <typename Fn>
void visit_parallel3d_coefficients(
    int ix,
    int iy,
    int iz,
    int view,
    const float* angles,
    Volume3d vol,
    Geometry3d geo,
    Fn&& fn)
{
    const float sinv = std::sin(angles[view]);
    const float cosv = std::cos(angles[view]);
    const bool use_x_axis = cosv * cosv > 0.5f;
    const bool keep_order = ((!use_x_axis && sinv >= 0.0f) || (use_x_axis && cosv >= 0.0f));

    const float idx_det0 = keep_order ? -0.5f * geo.Nu : 0.5f * geo.Nu;
    const float det_axis0 = use_x_axis ? (idx_det0 * geo.du + geo.shift_u) / cosv
                                       : (idx_det0 * geo.du + geo.shift_u) / sinv;
    const float det_interval = use_x_axis ? geo.du / std::abs(cosv) : geo.du / std::abs(sinv);
    const float pixel_interval = use_x_axis ? vol.dx : vol.dy;
    const float xy_scale = use_x_axis ? vol.dy / geo.du : vol.dx / geo.du;

    float voxel_axis0;
    if (use_x_axis) {
        const float px0 = (ix - 0.5f * vol.Nx) * vol.dx + vol.shift_x;
        const float py = (0.5f * vol.Ny - iy - 0.5f) * vol.dy + vol.shift_y;
        voxel_axis0 = (sinv / cosv) * py + px0;
    } else {
        const float px = (ix - 0.5f * vol.Nx + 0.5f) * vol.dx + vol.shift_x;
        const float py0 = (0.5f * vol.Ny - iy - 1.0f) * vol.dy + vol.shift_y;
        voxel_axis0 = (cosv / sinv) * px + py0;
    }
    const float voxel_axis1 = voxel_axis0 + pixel_interval;

    float axis_bound = std::max(voxel_axis0, det_axis0);
    const float axis_end = std::min(voxel_axis1, det_axis0 + geo.Nu * det_interval);
    if (!(axis_bound < axis_end)) return;

    int det = static_cast<int>(std::floor((axis_bound - det_axis0) / det_interval));
    det = std::max(0, std::min(geo.Nu - 1, det));
    float det_next = det_axis0 + (det + 1) * det_interval;

    const float voxel_z0 = (iz - 0.5f * vol.Nz) * vol.dz + vol.shift_z;
    const float voxel_z1 = voxel_z0 + vol.dz;
    const float det_z0 = -0.5f * geo.Nv * geo.dv + geo.shift_v;
    float z_bound = std::max(voxel_z0, det_z0);
    const float z_end = std::min(voxel_z1, det_z0 + geo.Nv * geo.dv);
    if (!(z_bound < z_end)) return;

    int row0 = static_cast<int>(std::floor((z_bound - det_z0) / geo.dv));
    row0 = std::max(0, std::min(geo.Nv - 1, row0));

    while (axis_bound < axis_end && det >= 0 && det < geo.Nu) {
        const float axis_next = std::min(det_next, axis_end);
        const float axis_coeff = (axis_next - axis_bound) * xy_scale;
        float row_bound = z_bound;
        int row = row0;
        float row_next = det_z0 + (row + 1) * geo.dv;
        while (row_bound < z_end && row >= 0 && row < geo.Nv) {
            const float z_next = std::min(row_next, z_end);
            const float coeff = axis_coeff * (z_next - row_bound) / geo.dv;
            const int det_out = keep_order ? det : (geo.Nu - 1 - det);
            fn(row, det_out, coeff);
            row_bound = z_next;
            ++row;
            row_next += geo.dv;
        }
        axis_bound = axis_next;
        ++det;
        det_next += det_interval;
    }
}

void parallel3d_forward_impl(float* proj, const float* image, const float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    zero_buffer(proj, static_cast<int64_t>(num_batch) * num_view * geo.Nv * geo.Nu);
    const int total = num_batch * num_view * vol.Nz;
    const int nt = thread_count(total);
    std::vector<std::vector<float>> local(nt, std::vector<float>(static_cast<int64_t>(num_batch) * num_view * geo.Nv * geo.Nu, 0.0f));
    std::vector<std::thread> workers;
    workers.reserve(nt);
    for (int t = 0; t < nt; ++t) {
        int begin = total * t / nt;
        int end = total * (t + 1) / nt;
        workers.emplace_back([=, &local]() {
            float* local_proj = local[t].data();
            for (int item = begin; item < end; ++item) {
                int tmp = item;
                const int iz = tmp % vol.Nz;
                tmp /= vol.Nz;
                const int view = tmp % num_view;
                const int batch = tmp / num_view;
                for (int iy = 0; iy < vol.Ny; ++iy) {
                    for (int ix = 0; ix < vol.Nx; ++ix) {
                        const float value = image[image_index(batch, iz, iy, ix, vol)];
                        if (value == 0.0f) continue;
                        visit_parallel3d_coefficients(ix, iy, iz, view, ang, vol, geo,
                            [&](int row, int col, float coeff) {
                                local_proj[proj_index(batch, view, row, col, num_view, geo)] += value * coeff;
                            });
                    }
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    const int64_t proj_size = static_cast<int64_t>(num_batch) * num_view * geo.Nv * geo.Nu;
    for (int t = 0; t < nt; ++t) {
        const float* src = local[t].data();
        for (int64_t i = 0; i < proj_size; ++i) proj[i] += src[i];
    }
}

void parallel3d_projection_t_impl(float* image, const float* proj, const float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    const int total = num_batch * vol.Nz * vol.Ny;
    const int nt = thread_count(total);
    std::vector<std::thread> workers;
    workers.reserve(nt);
    for (int t = 0; t < nt; ++t) {
        int begin = total * t / nt;
        int end = total * (t + 1) / nt;
        workers.emplace_back([=]() {
            for (int item = begin; item < end; ++item) {
                int tmp = item;
                const int iy = tmp % vol.Ny;
                tmp /= vol.Ny;
                const int iz = tmp % vol.Nz;
                const int batch = tmp / vol.Nz;
                for (int ix = 0; ix < vol.Nx; ++ix) {
                    float value = 0.0f;
                    for (int view = 0; view < num_view; ++view) {
                        visit_parallel3d_coefficients(ix, iy, iz, view, ang, vol, geo,
                            [&](int row, int col, float coeff) {
                                value += proj[proj_index(batch, view, row, col, num_view, geo)] * coeff;
                            });
                    }
                    image[image_index(batch, iz, iy, ix, vol)] = value;
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

struct ConeForwardBlock {
    int batch;
    int view;
    int det_row;
    int det_col0;
    int max_det_col;
};

inline float image_value(const float* image, int batch, int z, int y, int x, Volume3d vol)
{
    return image[cpu3d::image_index(batch, z, y, x, vol)];
}

inline float safe_tan(float value)
{
    const float t = std::tan(value);
    if (std::abs(t) < 1e-12f) return (t < 0.0f ? -1e-12f : 1e-12f);
    return t;
}

template <typename Emit>
void visit_depth_overlap(
    float z_begin,
    float z_end,
    float voxel_z0,
    float voxel_dz,
    int num_z,
    float axis_coeff,
    Emit&& emit)
{
    float bound_z = std::max(voxel_z0, z_begin);
    const float end_z = std::min(voxel_z0 + voxel_dz * num_z, z_end);
    if (!(bound_z < end_z)) return;

    int iz = static_cast<int>(std::floor((bound_z - voxel_z0) / voxel_dz));
    iz = std::max(0, std::min(num_z - 1, iz));
    float next_z = voxel_z0 + (iz + 1) * voxel_dz;
    while (bound_z < end_z && iz < num_z) {
        const float clipped_next_z = std::min(next_z, end_z);
        emit(iz, (clipped_next_z - bound_z) * axis_coeff);
        bound_z = clipped_next_z;
        ++iz;
        next_z += voxel_dz;
    }
}

inline float flat_detector_axis(
    const View2d& view,
    const Geometry3d& geo,
    int det_col,
    bool positive,
    bool use_x_axis)
{
    const float u = positive
        ? ((det_col - geo.Nu * 0.5f) * geo.du + geo.shift_u) * geo.SOD / geo.SDD
        : ((geo.Nu * 0.5f - det_col) * geo.du + geo.shift_u) * geo.SOD / geo.SDD;
    const float x = u * view.cosv;
    const float y = u * view.sinv;
    return use_x_axis ? map2x(view.source.x, view.source.y, x, y)
                      : map2y(view.source.x, view.source.y, x, y);
}

inline float flat_detector_boundary_coord(
    const View2d& view,
    const Geometry3d& geo,
    int det_col,
    bool positive,
    bool want_x)
{
    const float u = positive
        ? ((det_col - geo.Nu * 0.5f) * geo.du + geo.shift_u) * geo.SOD / geo.SDD
        : ((geo.Nu * 0.5f - det_col) * geo.du + geo.shift_u) * geo.SOD / geo.SDD;
    return want_x ? u * view.cosv : u * view.sinv;
}

inline float flat_detector_center_coord(
    const View2d& view,
    const Geometry3d& geo,
    int det_col,
    bool positive,
    bool want_x)
{
    const float u = positive
        ? ((det_col + 0.5f - geo.Nu * 0.5f) * geo.du + geo.shift_u) * geo.SOD / geo.SDD
        : ((geo.Nu * 0.5f - det_col - 0.5f) * geo.du + geo.shift_u) * geo.SOD / geo.SDD;
    return want_x ? u * view.cosv : u * view.sinv;
}

inline float arc_detector_axis(
    const View2d& view,
    const Geometry3d& geo,
    int det_col,
    bool positive,
    bool use_x_axis)
{
    const float det_angle = positive
        ? (det_col - geo.Nu * 0.5f) * geo.du + geo.shift_u + view.angle
        : (geo.Nu * 0.5f - det_col) * geo.du + geo.shift_u + view.angle;
    return use_x_axis ? map2x(view.source.x, view.source.y, det_angle)
                      : map2y(view.source.x, view.source.y, det_angle);
}

inline float arc_detector_center_coord(
    const View2d& view,
    const Geometry3d& geo,
    int det_col,
    bool positive,
    bool want_x)
{
    const float det_angle = positive
        ? (det_col + 0.5f - geo.Nu * 0.5f) * geo.du + geo.shift_u + view.angle
        : (geo.Nu * 0.5f - det_col - 0.5f) * geo.du + geo.shift_u + view.angle;
    if (want_x) return std::sin(det_angle) * geo.SDD + view.source.x;
    return -std::cos(det_angle) * geo.SDD + view.source.y;
}

template <typename Emit>
void cone3d_forward_block(
    const float* image,
    const float* angles,
    Volume3d vol,
    Geometry3d geo,
    const ConeForwardBlock& block,
    Emit&& emit)
{
    const View2d view = make_view2d(angles, block.view, geo);
    const bool use_x_axis = view.cosv * view.cosv > 0.5f;
    const bool positive = use_x_axis ? (view.cosv >= 0.0f) : (view.sinv >= 0.0f);

    std::vector<float> col_axis(block.max_det_col + 1);
    std::vector<float> row0_axis(block.max_det_col);
    std::vector<float> row1_axis(block.max_det_col);
    std::vector<float> scale(block.max_det_col);

    const float flat_scale = geo.SOD / geo.SDD;
    const float row0_v = ((block.det_row - geo.Nv * 0.5f) * geo.dv + geo.shift_v)
        * (geo.type == GeometryType::FanFlat ? flat_scale : 1.0f);
    const float row1_v = row0_v + geo.dv * (geo.type == GeometryType::FanFlat ? flat_scale : 1.0f);

    for (int local_col = 0; local_col < block.max_det_col; ++local_col) {
        const int det_col = block.det_col0 + local_col;
        if (geo.type == GeometryType::FanFlat) {
            if (local_col == 0) {
                col_axis[local_col] = flat_detector_axis(view, geo, block.det_col0, positive, use_x_axis);
            }
            col_axis[local_col + 1] = flat_detector_axis(view, geo, det_col + 1, positive, use_x_axis);
            const float center = flat_detector_center_coord(view, geo, det_col, positive, !use_x_axis);
            if (use_x_axis) {
                row0_axis[local_col] = view.source.y / (view.source.y - center) * row0_v;
                row1_axis[local_col] = view.source.y / (view.source.y - center) * row1_v;
                const float area = (col_axis[local_col + 1] - col_axis[local_col]) * (row1_axis[local_col] - row0_axis[local_col]);
                const float cosw = triangle_cosine(0.5f * (col_axis[local_col + 1] + col_axis[local_col]) - view.source.x,
                    0.5f * (row1_axis[local_col] + row0_axis[local_col]), view.source.y);
                scale[local_col] = vol.dy / (area * cosw);
            } else {
                row0_axis[local_col] = view.source.x / (view.source.x - center) * row0_v;
                row1_axis[local_col] = view.source.x / (view.source.x - center) * row1_v;
                const float area = (col_axis[local_col + 1] - col_axis[local_col]) * (row1_axis[local_col] - row0_axis[local_col]);
                const float cosw = triangle_cosine(0.5f * (col_axis[local_col + 1] + col_axis[local_col]) - view.source.y,
                    0.5f * (row1_axis[local_col] + row0_axis[local_col]), view.source.x);
                scale[local_col] = vol.dx / (area * cosw);
            }
        } else {
            if (local_col == 0) {
                col_axis[local_col] = arc_detector_axis(view, geo, block.det_col0, positive, use_x_axis);
            }
            col_axis[local_col + 1] = arc_detector_axis(view, geo, det_col + 1, positive, use_x_axis);
            const float center = arc_detector_center_coord(view, geo, det_col, positive, !use_x_axis);
            if (use_x_axis) {
                row0_axis[local_col] = view.source.y / (view.source.y - center) * row0_v;
                row1_axis[local_col] = view.source.y / (view.source.y - center) * row1_v;
                const float area = (col_axis[local_col + 1] - col_axis[local_col]) * (row1_axis[local_col] - row0_axis[local_col]);
                const float cosw = triangle_cosine(0.5f * (col_axis[local_col + 1] + col_axis[local_col]) - view.source.x,
                    0.5f * (row1_axis[local_col] + row0_axis[local_col]), view.source.y);
                scale[local_col] = vol.dy / (area * cosw);
            } else {
                row0_axis[local_col] = view.source.x / (view.source.x - center) * row0_v;
                row1_axis[local_col] = view.source.x / (view.source.x - center) * row1_v;
                const float area = (col_axis[local_col + 1] - col_axis[local_col]) * (row1_axis[local_col] - row0_axis[local_col]);
                const float cosw = triangle_cosine(0.5f * (col_axis[local_col + 1] + col_axis[local_col]) - view.source.y,
                    0.5f * (row1_axis[local_col] + row0_axis[local_col]), view.source.x);
                scale[local_col] = vol.dx / (area * cosw);
            }
        }
    }

    const float z0 = -vol.Nz * 0.5f * vol.dz + vol.shift_z;
    const float z1 = vol.Nz * 0.5f * vol.dz + vol.shift_z;

    if (use_x_axis) {
        const float x0 = -vol.Nx * 0.5f * vol.dx + vol.shift_x;
        const float x1 = vol.Nx * 0.5f * vol.dx + vol.shift_x;
        const float det_interval = (geo.type == GeometryType::FanFlat) ? flat_detector_col_width(geo) * std::abs(view.cosv) : 0.0f;

        for (int iy = 0; iy < vol.Ny; ++iy) {
            const float py = (vol.Ny * 0.5f - iy - 0.5f) * vol.dy + vol.shift_y;
            const float p0 = map2x(view.source.x, view.source.y, x0, py);
            const float p1 = map2x(view.source.x, view.source.y, x1, py);
            const float z_axis0 = view.source.y / (view.source.y - py) * z0;
            const float z_axis1 = view.source.y / (view.source.y - py) * z1;
            const float pix_interval = (p1 - p0) / vol.Nx;
            const float pix_interval_z = (z_axis1 - z_axis0) / vol.Nz;

            int ix = 0;
            int local_det = 0;
            float bound = std::max(p0, col_axis[0]);
            if (geo.type == GeometryType::FanFlat) {
                if (p0 == bound) {
                    const float det0x = flat_detector_boundary_coord(view, geo, block.det_col0, positive, true);
                    const float det_coord = view.source.y * bound / ((bound - view.source.x) * view.sinv / view.cosv + view.source.y);
                    local_det = static_cast<int>(std::floor((det_coord - det0x) / det_interval));
                } else {
                    ix = static_cast<int>(std::floor((bound - p0) / pix_interval));
                }
            } else {
                const float tan0 = (view.source.x - p0 == 0.0f) ? 1e10f : view.source.y / (view.source.x - p0);
                const float tan1 = -1.0f / safe_tan(geo.shift_u + view.angle);
                const float delta = std::atan((tan0 - tan1) / (1.0f + tan0 * tan1));
                const float flag = positive ? 1.0f : -1.0f;
                local_det = static_cast<int>(std::floor(geo.Nu * 0.5f - block.det_col0 + delta * flag / geo.du));
                if (local_det < 0) {
                    bound = col_axis[0];
                    ix = static_cast<int>(std::floor((bound - p0) / pix_interval));
                    local_det = 0;
                }
            }

            ix = std::max(0, ix);
            float next_pixel = p0 + (ix + 1) * pix_interval;
            float next_det = (local_det < block.max_det_col) ? col_axis[local_det + 1] : 0.0f;
            while (ix < vol.Nx && local_det < block.max_det_col) {
                const float z_begin = std::max(z_axis0, row0_axis[local_det]);
                const float z_end = std::min(z_axis1, row1_axis[local_det]);
                const float next = std::min(next_pixel, next_det);
                const float axis_coeff = next - bound;
                visit_depth_overlap(z_begin, z_end, z_axis0, pix_interval_z, vol.Nz, axis_coeff,
                    [&](int iz, float coeff) {
                        emit(local_det, ix, iy, iz, coeff, scale[local_det]);
                    });
                bound = next;
                if (next_pixel < next_det) {
                    ++ix;
                    next_pixel += pix_interval;
                } else {
                    ++local_det;
                    if (local_det < block.max_det_col) next_det = col_axis[local_det + 1];
                }
            }
        }
    } else {
        const float y0 = -vol.Ny * 0.5f * vol.dy + vol.shift_y;
        const float y1 = vol.Ny * 0.5f * vol.dy + vol.shift_y;
        const float det_interval = (geo.type == GeometryType::FanFlat) ? flat_detector_col_width(geo) * std::abs(view.sinv) : 0.0f;

        for (int ix = 0; ix < vol.Nx; ++ix) {
            const float px = (ix - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
            const float p0 = map2y(view.source.x, view.source.y, px, y0);
            const float p1 = map2y(view.source.x, view.source.y, px, y1);
            const float z_axis0 = view.source.x / (view.source.x - px) * z0;
            const float z_axis1 = view.source.x / (view.source.x - px) * z1;
            const float pix_interval = (p1 - p0) / vol.Ny;
            const float pix_interval_z = (z_axis1 - z_axis0) / vol.Nz;

            int iy_axis = 0;
            int local_det = 0;
            float bound = std::max(p0, col_axis[0]);
            if (geo.type == GeometryType::FanFlat) {
                if (p0 == bound) {
                    const float det0y = flat_detector_boundary_coord(view, geo, block.det_col0, positive, false);
                    const float det_coord = view.source.x * bound / ((bound - view.source.y) * view.cosv / view.sinv + view.source.x);
                    local_det = static_cast<int>(std::floor((det_coord - det0y) / det_interval));
                } else {
                    iy_axis = static_cast<int>(std::floor((bound - p0) / pix_interval));
                }
            } else {
                const float tan0 = (view.source.y - p0) / view.source.x;
                const float tan1 = -1.0f / safe_tan(geo.shift_u + view.angle);
                const float delta = std::atan((tan0 - tan1) / (1.0f + tan0 * tan1));
                const float flag = positive ? 1.0f : -1.0f;
                local_det = static_cast<int>(std::floor(geo.Nu * 0.5f - block.det_col0 + delta * flag / geo.du));
                if (local_det < 0) {
                    bound = col_axis[0];
                    iy_axis = static_cast<int>(std::floor((bound - p0) / pix_interval));
                    local_det = 0;
                }
            }

            iy_axis = std::max(0, iy_axis);
            float next_pixel = p0 + (iy_axis + 1) * pix_interval;
            float next_det = (local_det < block.max_det_col) ? col_axis[local_det + 1] : 0.0f;
            while (iy_axis < vol.Ny && local_det < block.max_det_col) {
                const int iy = vol.Ny - 1 - iy_axis;
                const float z_begin = std::max(z_axis0, row0_axis[local_det]);
                const float z_end = std::min(z_axis1, row1_axis[local_det]);
                const float next = std::min(next_pixel, next_det);
                const float axis_coeff = next - bound;
                visit_depth_overlap(z_begin, z_end, z_axis0, pix_interval_z, vol.Nz, axis_coeff,
                    [&](int iz, float coeff) {
                        emit(local_det, ix, iy, iz, coeff, scale[local_det]);
                    });
                bound = next;
                if (next_pixel < next_det) {
                    ++iy_axis;
                    next_pixel += pix_interval;
                } else {
                    ++local_det;
                    if (local_det < block.max_det_col) next_det = col_axis[local_det + 1];
                }
            }
        }
    }
}

void cone3d_forward_impl(float* proj, const float* image, const float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    cpu3d::zero_buffer(proj, static_cast<int64_t>(num_batch) * num_view * geo.Nv * geo.Nu);
    const int num_blocks_u = (geo.Nu - 1) / kConeBlock + 1;
    const int total = num_batch * num_view * geo.Nv * num_blocks_u;
    const int nt = cpu3d::thread_count(total);
    std::vector<std::thread> workers;
    workers.reserve(nt);
    for (int t = 0; t < nt; ++t) {
        int begin = total * t / nt;
        int end = total * (t + 1) / nt;
        workers.emplace_back([=]() {
            for (int item = begin; item < end; ++item) {
                int tmp = item;
                const int block_u = tmp % num_blocks_u;
                tmp /= num_blocks_u;
                const int det_row = tmp % geo.Nv;
                tmp /= geo.Nv;
                const int view = tmp % num_view;
                const int batch = tmp / num_view;
                const int det_col0 = block_u * kConeBlock;
                const int max_det_col = std::min(kConeBlock, geo.Nu - det_col0);
                std::vector<float> values(max_det_col, 0.0f);
                const ConeForwardBlock block{batch, view, det_row, det_col0, max_det_col};
                cone3d_forward_block(image, ang, vol, geo, block,
                    [&](int local_det, int ix, int iy, int iz, float coeff, float scale) {
                        values[local_det] += image_value(image, batch, iz, iy, ix, vol) * coeff * scale;
                    });
                const View2d view_info = make_view2d(ang, view, geo);
                const bool use_x_axis = view_info.cosv * view_info.cosv > 0.5f;
                const bool keep_order = use_x_axis ? (view_info.cosv >= 0.0f) : (view_info.sinv >= 0.0f);
                for (int local_det = 0; local_det < max_det_col; ++local_det) {
                    const int det_col = det_col0 + local_det;
                    const int out_col = keep_order ? det_col : (geo.Nu - 1 - det_col);
                    proj[cpu3d::proj_index(batch, view, det_row, out_col, num_view, geo)] = values[local_det];
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

void cone3d_projection_t_impl(float* image, const float* proj, const float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    const int64_t image_size = static_cast<int64_t>(num_batch) * vol.Nz * vol.Ny * vol.Nx;
    cpu3d::zero_buffer(image, image_size);
    const int num_blocks_u = (geo.Nu - 1) / kConeBlock + 1;
    const int total = num_batch * num_view * geo.Nv * num_blocks_u;
    const int nt = cpu3d::thread_count(total);
    std::vector<std::vector<float>> local(nt, std::vector<float>(image_size, 0.0f));
    std::vector<std::thread> workers;
    workers.reserve(nt);
    for (int t = 0; t < nt; ++t) {
        int begin = total * t / nt;
        int end = total * (t + 1) / nt;
        workers.emplace_back([=, &local]() {
            float* local_image = local[t].data();
            for (int item = begin; item < end; ++item) {
                int tmp = item;
                const int block_u = tmp % num_blocks_u;
                tmp /= num_blocks_u;
                const int det_row = tmp % geo.Nv;
                tmp /= geo.Nv;
                const int view = tmp % num_view;
                const int batch = tmp / num_view;
                const int det_col0 = block_u * kConeBlock;
                const int max_det_col = std::min(kConeBlock, geo.Nu - det_col0);
                const View2d view_info = make_view2d(ang, view, geo);
                const bool use_x_axis = view_info.cosv * view_info.cosv > 0.5f;
                const bool keep_order = use_x_axis ? (view_info.cosv >= 0.0f) : (view_info.sinv >= 0.0f);
                std::vector<float> values(max_det_col, 0.0f);
                for (int local_det = 0; local_det < max_det_col; ++local_det) {
                    const int det_col = det_col0 + local_det;
                    const int in_col = keep_order ? det_col : (geo.Nu - 1 - det_col);
                    values[local_det] = proj[cpu3d::proj_index(batch, view, det_row, in_col, num_view, geo)];
                }
                const ConeForwardBlock block{batch, view, det_row, det_col0, max_det_col};
                cone3d_forward_block(nullptr, ang, vol, geo, block,
                    [&](int local_det, int ix, int iy, int iz, float coeff, float scale) {
                        local_image[cpu3d::image_index(batch, iz, iy, ix, vol)] += values[local_det] * coeff * scale;
                    });
            }
        });
    }
    for (auto& worker : workers) worker.join();
    for (int t = 0; t < nt; ++t) {
        const float* src = local[t].data();
        for (int64_t i = 0; i < image_size; ++i) image[i] += src[i];
    }
}

} // namespace

void forward3d_parallel_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    parallel3d_forward_impl(proj, image, ang, vol, geo, num_batch, num_view);
}

void forward3d_t_parallel_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    parallel3d_projection_t_impl(image, proj, ang, vol, geo, num_batch, num_view);
}

void forward3d_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    if (geo.type == GeometryType::Parallel) forward3d_parallel_cpu(image, proj, ang, vol, geo, num_batch, num_view);
    else cone3d_forward_impl(proj, image, ang, vol, geo, num_batch, num_view);
}

void forward3d_t_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    if (geo.type == GeometryType::Parallel) forward3d_t_parallel_cpu(image, proj, ang, vol, geo, num_batch, num_view);
    else cone3d_projection_t_impl(image, proj, ang, vol, geo, num_batch, num_view);
}

void distance_driven3d_cpu(
    MatrixCoeffEmitter emit,
    void* user,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_view
) {
    if (geo.type == GeometryType::Parallel) {
        for (int view = 0; view < num_view; ++view) {
            for (int iz = 0; iz < vol.Nz; ++iz) {
                for (int iy = 0; iy < vol.Ny; ++iy) {
                    for (int ix = 0; ix < vol.Nx; ++ix) {
                        const int image_col = static_cast<int>(cpu3d::image_index(0, iz, iy, ix, vol));
                        visit_parallel3d_coefficients(ix, iy, iz, view, ang, vol, geo,
                            [&](int row, int det_col, float coeff) {
                                const int sino_row = static_cast<int>((static_cast<int64_t>(view) * geo.Nv + row) * geo.Nu + det_col);
                                emit(sino_row, image_col, coeff, user);
                            });
                    }
                }
            }
        }
        return;
    }

    const int num_blocks_u = (geo.Nu - 1) / kConeBlock + 1;
    for (int view = 0; view < num_view; ++view) {
        const View2d view_info = make_view2d(ang, view, geo);
        const bool use_x_axis = view_info.cosv * view_info.cosv > 0.5f;
        const bool keep_order = use_x_axis ? (view_info.cosv >= 0.0f) : (view_info.sinv >= 0.0f);
        for (int det_row = 0; det_row < geo.Nv; ++det_row) {
            for (int block_u = 0; block_u < num_blocks_u; ++block_u) {
                const int det_col0 = block_u * kConeBlock;
                const int max_det_col = std::min(kConeBlock, geo.Nu - det_col0);
                const ConeForwardBlock block{0, view, det_row, det_col0, max_det_col};
                cone3d_forward_block(nullptr, ang, vol, geo, block,
                    [&](int local_det, int ix, int iy, int iz, float coeff, float scale) {
                        const int det_col = det_col0 + local_det;
                        const int out_col = keep_order ? det_col : (geo.Nu - 1 - det_col);
                        const int sino_row = static_cast<int>((static_cast<int64_t>(view) * geo.Nv + det_row) * geo.Nu + out_col);
                        const int image_col = static_cast<int>(cpu3d::image_index(0, iz, iy, ix, vol));
                        emit(sino_row, image_col, coeff * scale, user);
                    });
            }
        }
    }
}

} // namespace xrecon
