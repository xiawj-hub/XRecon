#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include "types.h"

namespace xrecon {

    __host__ __device__ inline float map2x(const Point2d& src, const Point2d& pnt) {
        return (src.x * pnt.y - src.y * pnt.x) / (pnt.y - src.y);
    }

    __host__ __device__ inline float map2y(const Point2d& src, const Point2d& pnt) {
        return (src.y * pnt.x - src.x * pnt.y) / (pnt.x - src.x);
    }

    __host__ __device__ inline float map2x(float source_x, float source_y, float point_x, float point_y) {
        return (source_x * point_y - source_y * point_x) / (point_y - source_y);
    }

    __host__ __device__ inline float map2y(float source_x, float source_y, float point_x, float point_y) {
        return (source_y * point_x - source_x * point_y) / (point_x - source_x);
    }

    __host__ __device__ inline float map2x(const Point2d& src, const float ang) {
        return src.x + src.y * tanf(ang);
    }

    __host__ __device__ inline float map2y(const Point2d& src, const float ang) {
        return src.y + src.x / tanf(ang);
    }

    __host__ __device__ inline float map2x(float source_x, float source_y, float ang) {
        return source_x + source_y * tanf(ang);
    }

    __host__ __device__ inline float map2y(float source_x, float source_y, float ang) {
        return source_y + source_x / tanf(ang);
    }

    __host__ __device__ inline float normalize_angle_pi(float angle, float reference) {
        const float pi = 3.14159265358979323846f;
        const float half_pi = 0.5f * pi;
        while (angle - reference > half_pi) angle -= pi;
        while (angle - reference < -half_pi) angle += pi;
        return angle;
    }

    __host__ __device__ inline float detector_axis_angle(
        const Point2d& src,
        bool use_x_axis,
        float axis_value,
        float reference_angle)
    {
        float angle = use_x_axis
            ? atan2f(axis_value - src.x, src.y)
            : atan2f(src.x, axis_value - src.y);
        return normalize_angle_pi(angle, reference_angle);
    }

    __host__ __device__ inline float triangle_cosine(float a, float b, float c) {
        return fabsf(c) / sqrtf(a * a + b * b + c * c);
    }

    __host__ __device__ inline float cos_weight_x(const Point2d& src, const float x, const float y) {
        float a = (x + y) * 0.5f - src.x;
        float b = src.y;
        return fabsf(b) / sqrtf(a * a + b * b);
    }

    __host__ __device__ inline float cos_weight_y(const Point2d& src, const float x, const float y) {
        float a = (x + y) * 0.5f - src.y;
        float b = src.x;
        return fabsf(b) / sqrtf(a * a + b * b);
    }

    __host__ __device__ inline float pix_weight(const Point2d& src, const Point2d& pnt, const float r) {
        float d = (src.x * pnt.x + src.y * pnt.y) / r;
        float denom = (r - d);
        return (r * r) / (denom * denom);
    }

    __host__ __device__ inline float pix_weight(const Point2d& src, const Point2d& pnt) {
        float dx = src.x - pnt.x;
        float dy = src.y - pnt.y;
        return dx * dx + dy * dy;
    }

    __host__ __device__ inline float coordinate_weight(
        float source_x,
        float source_y,
        float point_x,
        float point_y,
        float radius) {
        float d = (source_x * point_x + source_y * point_y) / radius;
        float denom = radius - d;
        return radius * radius / (denom * denom);
    }

    __host__ __device__ inline float coordinate_weight(
        float source_x,
        float source_y,
        float point_x,
        float point_y) {
        float dx = source_x - point_x;
        float dy = source_y - point_y;
        return dx * dx + dy * dy;
    }

    struct View2d {
        float angle;
        float sinv;
        float cosv;
        Point2d source;
    };

    __host__ __device__ inline View2d make_view2d(const float* angles, int idx_view, const Geometry3d& geo) {
        View2d view;
        view.angle = angles[idx_view];
        view.sinv = sinf(view.angle);
        view.cosv = cosf(view.angle);
        view.source = Point2d(-view.sinv * geo.SOD, view.cosv * geo.SOD);
        return view;
    }

    __host__ __device__ inline float overlap_length(float a0, float a1, float b0, float b1) {
        float lo = fmaxf(fminf(a0, a1), fminf(b0, b1));
        float hi = fminf(fmaxf(a0, a1), fmaxf(b0, b1));
        return fmaxf(hi - lo, 0.0f);
    }

    __host__ __device__ inline void detector_row_overlap_range(
        float voxel_z0,
        float voxel_z1,
        float detector_z0,
        float detector_dz,
        int num_rows,
        int& row_begin,
        int& row_end)
    {
        if (num_rows <= 0 || fabsf(detector_dz) <= 1.0e-12f) {
            row_begin = 0;
            row_end = -1;
            return;
        }

        const float z_min = fminf(voxel_z0, voxel_z1);
        const float z_max = fmaxf(voxel_z0, voxel_z1);
        float t0 = (z_min - detector_z0) / detector_dz;
        float t1 = (z_max - detector_z0) / detector_dz;
        if (t0 > t1) {
            const float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        row_begin = static_cast<int>(floorf(t0)) - 1;
        row_end = static_cast<int>(floorf(t1)) + 1;
        row_begin = row_begin < 0 ? 0 : row_begin;
        row_end = row_end > num_rows - 1 ? num_rows - 1 : row_end;
    }

    __host__ __device__ inline float voxel_x(const Volume3d& vol, int ix) {
        return (ix - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    }

    __host__ __device__ inline float voxel_y(const Volume3d& vol, int iy) {
        return (iy - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    }

    __host__ __device__ inline float voxel_z(const Volume3d& vol, int iz) {
        return (iz - vol.Nz * 0.5f + 0.5f) * vol.dz + vol.shift_z;
    }

    __host__ __device__ inline float flat_virtual_u(const Geometry3d& geo, int iu_boundary) {
        float scale = geo.SOD / geo.SDD;
        return ((iu_boundary - geo.Nu * 0.5f) * geo.du + geo.shift_u) * scale;
    }

    __host__ __device__ inline float flat_virtual_v(const Geometry3d& geo, int iv_boundary) {
        float scale = geo.SOD / geo.SDD;
        return ((iv_boundary - geo.Nv * 0.5f) * geo.dv + geo.shift_v) * scale;
    }

    __host__ __device__ inline Point2d flat_virtual_detector_point(const View2d& view, float u) {
        return Point2d(u * view.cosv, u * view.sinv);
    }

    __host__ __device__ inline float flat_detector_col_width(const Geometry3d& geo) {
        return geo.du * geo.SOD / geo.SDD;
    }

    __host__ __device__ inline float arc_detector_angle(const View2d& view, const Geometry3d& geo, int iu_boundary) {
        return (iu_boundary - geo.Nu * 0.5f) * geo.du + geo.shift_u + view.angle;
    }

    __host__ __device__ inline float arc_detector_col_width(const Geometry3d& geo) {
        return 2.0f * geo.SOD * tanf(0.5f * geo.du);
    }

    __host__ __device__ inline float arc_detector_y(const View2d& view, const Geometry3d& geo, float col_angle) {
        return -cosf(col_angle) * geo.SDD + view.source.y;
    }

    __host__ __device__ inline float arc_detector_x(const View2d& view, const Geometry3d& geo, float col_angle) {
        return sinf(col_angle) * geo.SDD + view.source.x;
    }

    __host__ __device__ inline float projection_sample(
        const float* __restrict__ proj,
        int batch, int view, int row, int col,
        int num_view, const Geometry3d& geo) {
        return proj[((batch * num_view + view) * geo.Nv + row) * geo.Nu + col];
    }

    __host__ __device__ inline float cone_voxel_weight_flat(
        const View2d& view,
        const Point2d& point,
        const Geometry3d& geo) {
        return pix_weight(view.source, point, geo.SOD);
    }

    __host__ __device__ inline float cone_voxel_weight_arc(
        const View2d& view,
        const Point2d& point) {
        return 1.0f / pix_weight(view.source, point);
    }

}
