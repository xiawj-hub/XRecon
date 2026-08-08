#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include "types.h"
#include "xrmath.h"

namespace xrecon::cpu3d {

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

inline float cone_detector_col_width(Geometry3d geo)
{
    if (geo.type == GeometryType::FanFlat) return flat_detector_col_width(geo);
    if (geo.type == GeometryType::FanArc) return arc_detector_col_width(geo);
    return geo.du;
}

inline float cone_voxel_weight(Geometry3d geo, const View2d& view, const Point2d& point)
{
    if (geo.type == GeometryType::FanFlat) return cone_voxel_weight_flat(view, point, geo);
    if (geo.type == GeometryType::FanArc) return cone_voxel_weight_arc(view, point);
    return 1.0f;
}

template <typename Fn>
void visit_cone3d_backprojection_coefficients(
    int ix,
    int iy,
    int iz,
    int view_id,
    const float* angles,
    Volume3d vol,
    Geometry3d geo,
    bool weighted,
    Fn&& fn)
{
    const float px = (ix - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    const float py = (vol.Ny * 0.5f - iy - 0.5f) * vol.dy + vol.shift_y;
    const float pz = (iz - vol.Nz * 0.5f + 0.5f) * vol.dz + vol.shift_z;
    const Point2d point(px, py);
    const View2d view = make_view2d(angles, view_id, geo);
    const bool use_x_axis = view.cosv * view.cosv > 0.5f;

    float voxel0, voxel1, z0, z1;
    if (use_x_axis) {
        voxel0 = map2x(view.source, Point2d(px - 0.5f * vol.dx, py));
        voxel1 = map2x(view.source, Point2d(px + 0.5f * vol.dx, py));
        z0 = view.source.y / (view.source.y - py) * (pz - 0.5f * vol.dz);
        z1 = view.source.y / (view.source.y - py) * (pz + 0.5f * vol.dz);
    } else {
        voxel0 = map2y(view.source, Point2d(px, py - 0.5f * vol.dy));
        voxel1 = map2y(view.source, Point2d(px, py + 0.5f * vol.dy));
        z0 = view.source.x / (view.source.x - px) * (pz - 0.5f * vol.dz);
        z1 = view.source.x / (view.source.x - px) * (pz + 0.5f * vol.dz);
    }

    const float voxel_area = std::abs((voxel1 - voxel0) * (z1 - z0));
    if (voxel_area <= 1e-12f) return;

    const float col_width = cone_detector_col_width(geo);
    float voxel_weight = 1.0f;
    if (weighted) voxel_weight = cone_voxel_weight(geo, view, point);

    const bool keep_dim = (!use_x_axis && view.sinv >= 0.0f) || (use_x_axis && view.cosv >= 0.0f);
    const float half_nu = geo.Nu * 0.5f;
    float bound0, bound1;
    int col;
    float det_center_x = 0.0f;
    float det_center_y = 0.0f;

    if (geo.type == GeometryType::FanFlat) {
        const float inv_nu = 1.0f / geo.Nu;
        const float idxd0 = keep_dim ? -half_nu : half_nu;
        const float idxd1 = keep_dim ? half_nu : -half_nu;
        const Point2d det0((idxd0 * geo.du + geo.shift_u) * geo.SOD / geo.SDD * view.cosv,
                           (idxd0 * geo.du + geo.shift_u) * geo.SOD / geo.SDD * view.sinv);
        const Point2d det1((idxd1 * geo.du + geo.shift_u) * geo.SOD / geo.SDD * view.cosv,
                           (idxd1 * geo.du + geo.shift_u) * geo.SOD / geo.SDD * view.sinv);
        const float det_step_x = (det1.x - det0.x) * inv_nu;
        const float det_step_y = (det1.y - det0.y) * inv_nu;
        const float det0_axis = use_x_axis ? map2x(view.source, det0) : map2y(view.source, det0);
        const float det1_axis = use_x_axis ? map2x(view.source, det1) : map2y(view.source, det1);
        bound0 = std::max(voxel0, det0_axis);
        bound1 = std::min(voxel1, det1_axis);
        if (!(bound0 < bound1)) return;

        const float bound_to_detector = use_x_axis
            ? view.source.y * bound0 / ((bound0 - view.source.x) * (view.sinv / view.cosv) + view.source.y)
            : view.source.x * bound0 / ((bound0 - view.source.y) * (view.cosv / view.sinv) + view.source.x);
        int det_id = use_x_axis ? static_cast<int>(std::floor((bound_to_detector - det0.x) / det_step_x))
                                : static_cast<int>(std::floor((bound_to_detector - det0.y) / det_step_y));
        det_id = std::max(0, std::min(geo.Nu - 1, det_id));
        Point2d det_cur(det0.x + det_id * det_step_x, det0.y + det_id * det_step_y);
        col = keep_dim ? det_id : (geo.Nu - 1 - det_id);

        while (bound0 < bound1 && col >= 0 && col < geo.Nu) {
            const Point2d det_next(det_cur.x + det_step_x, det_cur.y + det_step_y);
            const float det_next_axis = use_x_axis ? map2x(view.source, det_next) : map2y(view.source, det_next);
            const float next_bound = std::min(bound1, det_next_axis);
            const float col_overlap = next_bound - bound0;
            if (col_overlap > 0.0f) {
                det_center_x = 0.5f * (det_cur.x + det_next.x);
                det_center_y = 0.5f * (det_cur.y + det_next.y);
                const float z_scale = use_x_axis
                    ? view.source.y / (view.source.y - det_center_y)
                    : view.source.x / (view.source.x - det_center_x);
                const float det_z0_base = z_scale * (geo.SOD / geo.SDD) * ((-geo.Nv * 0.5f) * geo.dv + geo.shift_v);
                const float det_dz = z_scale * (geo.SOD / geo.SDD) * geo.dv;
                int row_begin;
                int row_end;
                detector_row_overlap_range(z0, z1, det_z0_base, det_dz, geo.Nv, row_begin, row_end);
                int row = row_begin;
                while (row <= row_end) {
                    const float det_z0 = det_z0_base + row * det_dz;
                    const float det_z1 = det_z0 + det_dz;
                    const float row_overlap = overlap_length(z0, z1, det_z0, det_z1);
                    if (row_overlap > 0.0f) {
                        fn(row, col, voxel_weight * col_width * col_overlap * row_overlap / voxel_area);
                    }
                    ++row;
                }
            }
            bound0 = col_overlap > 0.0f ? next_bound : det_next_axis;
            det_cur = det_next;
            col += keep_dim ? 1 : -1;
        }
    } else {
        const float det0_ang = keep_dim ? (-half_nu * geo.du + geo.shift_u + view.angle)
                                        : ( half_nu * geo.du + geo.shift_u + view.angle);
        const float det1_ang = keep_dim ? ( half_nu * geo.du + geo.shift_u + view.angle)
                                        : (-half_nu * geo.du + geo.shift_u + view.angle);
        const float det0_axis = use_x_axis ? map2x(view.source, det0_ang) : map2y(view.source, det0_ang);
        const float det1_axis = use_x_axis ? map2x(view.source, det1_ang) : map2y(view.source, det1_ang);
        bound0 = std::max(voxel0, det0_axis);
        bound1 = std::min(voxel1, det1_axis);
        if (!(bound0 < bound1)) return;

        const float det_start_ang = detector_axis_angle(
            view.source, use_x_axis, bound0, geo.shift_u + view.angle);
        const float det_start = (det_start_ang - geo.shift_u - view.angle) / geo.du;
        col = static_cast<int>(std::floor(half_nu + det_start));
        col = std::max(0, std::min(geo.Nu - 1, col));
        float det_cur_ang = (col - half_nu) * geo.du + geo.shift_u + view.angle;
        det_cur_ang = keep_dim ? det_cur_ang : det_cur_ang + geo.du;

        while (bound0 < bound1 && col >= 0 && col < geo.Nu) {
            const float det_next_ang = keep_dim ? det_cur_ang + geo.du : det_cur_ang - geo.du;
            const float det_next_axis = use_x_axis ? map2x(view.source, det_next_ang) : map2y(view.source, det_next_ang);
            const float next_bound = std::min(bound1, det_next_axis);
            const float col_overlap = next_bound - bound0;
            if (col_overlap > 0.0f) {
                const float det_center_ang = 0.5f * (det_cur_ang + det_next_ang);
                det_center_x = arc_detector_x(view, geo, det_center_ang);
                det_center_y = arc_detector_y(view, geo, det_center_ang);
                const float z_scale = use_x_axis
                    ? view.source.y / (view.source.y - det_center_y)
                    : view.source.x / (view.source.x - det_center_x);
                const float det_z0_base = z_scale * ((-geo.Nv * 0.5f) * geo.dv + geo.shift_v);
                const float det_dz = z_scale * geo.dv;
                int row_begin;
                int row_end;
                detector_row_overlap_range(z0, z1, det_z0_base, det_dz, geo.Nv, row_begin, row_end);
                int row = row_begin;
                while (row <= row_end) {
                    const float det_z0 = det_z0_base + row * det_dz;
                    const float det_z1 = det_z0 + det_dz;
                    const float row_overlap = overlap_length(z0, z1, det_z0, det_z1);
                    if (row_overlap > 0.0f) {
                        fn(row, col, voxel_weight * col_width * col_overlap * row_overlap / voxel_area);
                    }
                    ++row;
                }
            }
            bound0 = col_overlap > 0.0f ? next_bound : det_next_axis;
            det_cur_ang = det_next_ang;
            col += keep_dim ? 1 : -1;
        }
    }
}

} // namespace xrecon::cpu3d
