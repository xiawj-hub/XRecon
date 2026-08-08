#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>
#include "functions.h"
#include "xrmath.h"

namespace xrecon {

namespace {

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

inline float proj_at(const float* proj, int batch, int view, int det, int num_view, const Geometry2d& geo)
{
    if (det < 0 || det >= geo.Nu) return 0.0f;
    return proj[(batch * num_view + view) * geo.Nu + det];
}

template<bool FBPWEIGHT>
float fanflat_pixel(const float* proj, const float* ang, int batch, int row, int col, Volume2d vol, Geometry2d geo, int num_view)
{
    float px = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    float py = (row - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    Point2d point(px, py);

    float pix_val = 0.0f;
    float inv_nu = 1.0f / geo.Nu;
    float half_nu = geo.Nu * 0.5f;

    for (int view = 0; view < num_view; ++view) {
        float sinv = std::sin(ang[view]);
        float cosv = std::cos(ang[view]);
        Point2d src(-sinv * geo.SOD, cosv * geo.SOD);
        bool use_x = (cosv * cosv > 0.5f);
        bool keep = ((cosv * cosv <= 0.5f && sinv >= 0.0f) || (cosv * cosv > 0.5f && cosv >= 0.0f));

        float idxd0 = keep ? -half_nu : half_nu;
        float idxd1 = keep ? half_nu : -half_nu;
        Point2d det0((idxd0 * geo.du + geo.shift_u) * cosv, (idxd0 * geo.du + geo.shift_u) * sinv);
        Point2d det1((idxd1 * geo.du + geo.shift_u) * cosv, (idxd1 * geo.du + geo.shift_u) * sinv);
        float det_ix = (det1.x - det0.x) * inv_nu;
        float det_iy = (det1.y - det0.y) * inv_nu;

        float p0, p1, d0p, d1p, bound0, bound1, b2d;
        if (use_x) {
            p0 = map2x(src, Point2d(px - 0.5f * vol.dx, py));
            p1 = map2x(src, Point2d(px + 0.5f * vol.dx, py));
            d0p = map2x(src, det0);
            d1p = map2x(src, det1);
            bound0 = std::max(p0, d0p);
            bound1 = std::min(p1, d1p);
            b2d = src.y * bound0 / ((bound0 - src.x) * (sinv / cosv) + src.y);
        } else {
            p0 = map2y(src, Point2d(px, py - 0.5f * vol.dy));
            p1 = map2y(src, Point2d(px, py + 0.5f * vol.dy));
            d0p = map2y(src, det0);
            d1p = map2y(src, det1);
            bound0 = std::max(p0, d0p);
            bound1 = std::min(p1, d1p);
            b2d = src.x * bound0 / ((bound0 - src.y) * (cosv / sinv) + src.x);
        }
        if (!(bound0 < bound1)) continue;

        int det = use_x ? static_cast<int>(std::floor((b2d - det0.x) / det_ix))
                        : static_cast<int>(std::floor((b2d - det0.y) / det_iy));
        det = std::max(0, std::min(geo.Nu - 1, det));

        Point2d det_cur(det0.x + det * det_ix, det0.y + det * det_iy);
        det = keep ? det : (geo.Nu - 1 - det);

        float acc = 0.0f;
        while (bound0 < bound1 && det >= 0 && det < geo.Nu) {
            Point2d det_next(det_cur.x + det_ix, det_cur.y + det_iy);
            float next = use_x ? map2x(src, det_next) : map2y(src, det_next);
            next = std::min(next, bound1);
            acc += (next - bound0) * proj_at(proj, batch, view, det, num_view, geo) * geo.du;
            bound0 = next;
            det_cur = det_next;
            det += keep ? 1 : -1;
        }

        if (FBPWEIGHT) acc *= pix_weight(src, point, geo.SOD);
        pix_val += acc / (p1 - p0);
    }

    return pix_val;
}

template<bool FBPWEIGHT>
float fanarc_pixel(const float* proj, const float* ang, int batch, int row, int col, Volume2d vol, Geometry2d geo, int num_view)
{
    float px = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    float py = (row - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    Point2d point(px, py);

    float pix_val = 0.0f;
    float half_nu = geo.Nu * 0.5f;
    float det_rad = geo.SOD * std::tan(geo.du / 2.0f) * 2.0f;

    for (int view = 0; view < num_view; ++view) {
        float beta = ang[view];
        float sinv = std::sin(beta);
        float cosv = std::cos(beta);
        Point2d src(-sinv * geo.SOD, cosv * geo.SOD);
        bool use_x = (cosv * cosv > 0.5f);
        bool keep = ((cosv * cosv <= 0.5f && sinv >= 0.0f) || (cosv * cosv > 0.5f && cosv >= 0.0f));

        float det0_ang = keep ? (-half_nu * geo.du + geo.shift_u + beta)
                              : ( half_nu * geo.du + geo.shift_u + beta);
        float det1_ang = keep ? ( half_nu * geo.du + geo.shift_u + beta)
                              : (-half_nu * geo.du + geo.shift_u + beta);

        float p0, p1, d0p, d1p, bound0, bound1;
        if (use_x) {
            p0 = map2x(src, Point2d(px - 0.5f * vol.dx, py));
            p1 = map2x(src, Point2d(px + 0.5f * vol.dx, py));
            d0p = map2x(src, det0_ang);
            d1p = map2x(src, det1_ang);
            bound0 = std::max(p0, d0p);
            bound1 = std::min(p1, d1p);
        } else {
            p0 = map2y(src, Point2d(px, py - 0.5f * vol.dy));
            p1 = map2y(src, Point2d(px, py + 0.5f * vol.dy));
            d0p = map2y(src, det0_ang);
            d1p = map2y(src, det1_ang);
            bound0 = std::max(p0, d0p);
            bound1 = std::min(p1, d1p);
        }
        if (!(bound0 < bound1)) continue;

        float det_start_ang = detector_axis_angle(src, use_x, bound0, geo.shift_u + beta);
        float det_start = (det_start_ang - geo.shift_u - beta) / geo.du;
        int det = static_cast<int>(std::floor(half_nu + det_start));
        det = std::max(0, std::min(geo.Nu - 1, det));

        float det_cur_ang = (det - half_nu) * geo.du + geo.shift_u + beta;
        det_cur_ang = keep ? det_cur_ang : (det_cur_ang + geo.du);

        float acc = 0.0f;
        while (bound0 < bound1 && det >= 0 && det < geo.Nu) {
            float det_next_ang = keep ? (det_cur_ang + geo.du) : (det_cur_ang - geo.du);
            float next = use_x ? map2x(src, det_next_ang) : map2y(src, det_next_ang);
            next = std::min(next, bound1);
            acc += (next - bound0) * proj_at(proj, batch, view, det, num_view, geo) * det_rad;
            bound0 = next;
            det_cur_ang = det_next_ang;
            det += keep ? 1 : -1;
        }

        if (FBPWEIGHT) acc /= pix_weight(src, point);
        pix_val += acc / (p1 - p0);
    }

    return pix_val;
}

float parallel_pixel(const float* proj, const float* ang, int batch, int row, int col, Volume2d vol, Geometry2d geo, int num_view)
{
    float px = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    float py = (row - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    float pix_val = 0.0f;
    float half_nu = geo.Nu * 0.5f;

    for (int view = 0; view < num_view; ++view) {
        float sinv = std::sin(ang[view]);
        float cosv = std::cos(ang[view]);
        bool use_x = (cosv * cosv > 0.5f);
        bool keep = ((cosv * cosv <= 0.5f && sinv >= 0.0f) || (cosv * cosv > 0.5f && cosv >= 0.0f));

        float idxd0 = keep ? -half_nu : half_nu;
        float idxd1 = keep ? half_nu : -half_nu;
        float det_interval = use_x ? (geo.du / std::abs(cosv)) : (geo.du / std::abs(sinv));
        float pix_interval = use_x ? vol.dx : vol.dy;
        float coef = geo.du / pix_interval;

        float p0, p1, d0p, d1p;
        if (use_x) {
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

        float bound0 = std::max(p0, d0p);
        float bound1 = std::min(p1, d1p);
        if (!(bound0 < bound1)) continue;

        int det = static_cast<int>(std::floor((bound0 - d0p) / det_interval));
        det = std::max(0, std::min(geo.Nu - 1, det));
        float det0p = d0p + det * det_interval;
        det = keep ? det : (geo.Nu - 1 - det);

        while (bound0 < bound1 && det >= 0 && det < geo.Nu) {
            float next = std::min(det0p + det_interval, bound1);
            pix_val += (next - bound0) * proj_at(proj, batch, view, det, num_view, geo) * coef;
            bound0 = next;
            det0p = next;
            det += keep ? 1 : -1;
        }
    }

    return pix_val;
}

template<bool FBPWEIGHT>
void backward_impl(float* image, const float* proj, const float* ang, Volume2d vol, Geometry2d geo, int num_batch, int num_view)
{
    int total_rows = num_batch * vol.Ny;
    int nt = thread_count(total_rows);
    std::vector<std::thread> workers;
    workers.reserve(nt);
    for (int t = 0; t < nt; ++t) {
        int begin = total_rows * t / nt;
        int end = total_rows * (t + 1) / nt;
        workers.emplace_back([=]() {
            for (int global_row = begin; global_row < end; ++global_row) {
                int batch = global_row / vol.Ny;
                int row = global_row % vol.Ny;
                for (int col = 0; col < vol.Nx; ++col) {
                    float value;
                    if (geo.type == GeometryType::FanFlat) {
                        value = fanflat_pixel<FBPWEIGHT>(proj, ang, batch, row, col, vol, geo, num_view);
                    } else if (geo.type == GeometryType::FanArc) {
                        value = fanarc_pixel<FBPWEIGHT>(proj, ang, batch, row, col, vol, geo, num_view);
                    } else {
                        value = parallel_pixel(proj, ang, batch, row, col, vol, geo, num_view);
                    }
                    image[(batch * vol.Ny + (vol.Ny - 1 - row)) * vol.Nx + col] = value;
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

inline void add_proj(float* proj, int batch, int view, int det, int num_view, const Geometry2d& geo, float value)
{
    if (det < 0 || det >= geo.Nu || value == 0.0f || !std::isfinite(value)) return;
    proj[(batch * num_view + view) * geo.Nu + det] += value;
}

template<bool FBPWEIGHT>
void fanflat_pixel_t(float* proj, float image_val, const float* ang, int batch, int row, int col, Volume2d vol, Geometry2d geo, int num_view)
{
    float px = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    float py = (row - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    Point2d point(px, py);
    float inv_nu = 1.0f / geo.Nu;
    float half_nu = geo.Nu * 0.5f;

    for (int view = 0; view < num_view; ++view) {
        float sinv = std::sin(ang[view]);
        float cosv = std::cos(ang[view]);
        Point2d src(-sinv * geo.SOD, cosv * geo.SOD);
        bool use_x = (cosv * cosv > 0.5f);
        bool keep = ((cosv * cosv <= 0.5f && sinv >= 0.0f) || (cosv * cosv > 0.5f && cosv >= 0.0f));

        float idxd0 = keep ? -half_nu : half_nu;
        float idxd1 = keep ? half_nu : -half_nu;
        Point2d det0((idxd0 * geo.du + geo.shift_u) * cosv, (idxd0 * geo.du + geo.shift_u) * sinv);
        Point2d det1((idxd1 * geo.du + geo.shift_u) * cosv, (idxd1 * geo.du + geo.shift_u) * sinv);
        float det_ix = (det1.x - det0.x) * inv_nu;
        float det_iy = (det1.y - det0.y) * inv_nu;

        float p0, p1, d0p, d1p, bound0, bound1, b2d;
        if (use_x) {
            p0 = map2x(src, Point2d(px - 0.5f * vol.dx, py));
            p1 = map2x(src, Point2d(px + 0.5f * vol.dx, py));
            d0p = map2x(src, det0);
            d1p = map2x(src, det1);
            bound0 = std::max(p0, d0p);
            bound1 = std::min(p1, d1p);
            b2d = src.y * bound0 / ((bound0 - src.x) * (sinv / cosv) + src.y);
        } else {
            p0 = map2y(src, Point2d(px, py - 0.5f * vol.dy));
            p1 = map2y(src, Point2d(px, py + 0.5f * vol.dy));
            d0p = map2y(src, det0);
            d1p = map2y(src, det1);
            bound0 = std::max(p0, d0p);
            bound1 = std::min(p1, d1p);
            b2d = src.x * bound0 / ((bound0 - src.y) * (cosv / sinv) + src.x);
        }
        if (!(bound0 < bound1)) continue;

        int det = use_x ? static_cast<int>(std::floor((b2d - det0.x) / det_ix))
                        : static_cast<int>(std::floor((b2d - det0.y) / det_iy));
        det = std::max(0, std::min(geo.Nu - 1, det));
        Point2d det_cur(det0.x + det * det_ix, det0.y + det * det_iy);
        det = keep ? det : (geo.Nu - 1 - det);

        float base = image_val / (p1 - p0);
        if (FBPWEIGHT) base *= pix_weight(src, point, geo.SOD);

        while (bound0 < bound1 && det >= 0 && det < geo.Nu) {
            Point2d det_next(det_cur.x + det_ix, det_cur.y + det_iy);
            float next = use_x ? map2x(src, det_next) : map2y(src, det_next);
            next = std::min(next, bound1);
            add_proj(proj, batch, view, det, num_view, geo, base * (next - bound0) * geo.du);
            bound0 = next;
            det_cur = det_next;
            det += keep ? 1 : -1;
        }
    }
}

template<bool FBPWEIGHT>
void fanarc_pixel_t(float* proj, float image_val, const float* ang, int batch, int row, int col, Volume2d vol, Geometry2d geo, int num_view)
{
    float px = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    float py = (row - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    Point2d point(px, py);
    float half_nu = geo.Nu * 0.5f;
    float det_rad = geo.SOD * std::tan(geo.du / 2.0f) * 2.0f;

    for (int view = 0; view < num_view; ++view) {
        float beta = ang[view];
        float sinv = std::sin(beta);
        float cosv = std::cos(beta);
        Point2d src(-sinv * geo.SOD, cosv * geo.SOD);
        bool use_x = (cosv * cosv > 0.5f);
        bool keep = ((cosv * cosv <= 0.5f && sinv >= 0.0f) || (cosv * cosv > 0.5f && cosv >= 0.0f));

        float det0_ang = keep ? (-half_nu * geo.du + geo.shift_u + beta)
                              : ( half_nu * geo.du + geo.shift_u + beta);
        float det1_ang = keep ? ( half_nu * geo.du + geo.shift_u + beta)
                              : (-half_nu * geo.du + geo.shift_u + beta);

        float p0, p1, d0p, d1p, bound0, bound1;
        if (use_x) {
            p0 = map2x(src, Point2d(px - 0.5f * vol.dx, py));
            p1 = map2x(src, Point2d(px + 0.5f * vol.dx, py));
            d0p = map2x(src, det0_ang);
            d1p = map2x(src, det1_ang);
            bound0 = std::max(p0, d0p);
            bound1 = std::min(p1, d1p);
        } else {
            p0 = map2y(src, Point2d(px, py - 0.5f * vol.dy));
            p1 = map2y(src, Point2d(px, py + 0.5f * vol.dy));
            d0p = map2y(src, det0_ang);
            d1p = map2y(src, det1_ang);
            bound0 = std::max(p0, d0p);
            bound1 = std::min(p1, d1p);
        }
        if (!(bound0 < bound1)) continue;

        float det_start_ang = detector_axis_angle(src, use_x, bound0, geo.shift_u + beta);
        float det_start = (det_start_ang - geo.shift_u - beta) / geo.du;
        int det = static_cast<int>(std::floor(half_nu + det_start));
        det = std::max(0, std::min(geo.Nu - 1, det));
        float det_cur_ang = (det - half_nu) * geo.du + geo.shift_u + beta;
        det_cur_ang = keep ? det_cur_ang : (det_cur_ang + geo.du);

        float base = image_val / (p1 - p0);
        if (FBPWEIGHT) base /= pix_weight(src, point);

        while (bound0 < bound1 && det >= 0 && det < geo.Nu) {
            float det_next_ang = keep ? (det_cur_ang + geo.du) : (det_cur_ang - geo.du);
            float next = use_x ? map2x(src, det_next_ang) : map2y(src, det_next_ang);
            next = std::min(next, bound1);
            add_proj(proj, batch, view, det, num_view, geo, base * (next - bound0) * det_rad);
            bound0 = next;
            det_cur_ang = det_next_ang;
            det += keep ? 1 : -1;
        }
    }
}

void parallel_pixel_t(float* proj, float image_val, const float* ang, int batch, int row, int col, Volume2d vol, Geometry2d geo, int num_view)
{
    float px = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
    float py = (row - vol.Ny * 0.5f + 0.5f) * vol.dy + vol.shift_y;
    float half_nu = geo.Nu * 0.5f;

    for (int view = 0; view < num_view; ++view) {
        float sinv = std::sin(ang[view]);
        float cosv = std::cos(ang[view]);
        bool use_x = (cosv * cosv > 0.5f);
        bool keep = ((cosv * cosv <= 0.5f && sinv >= 0.0f) || (cosv * cosv > 0.5f && cosv >= 0.0f));

        float idxd0 = keep ? -half_nu : half_nu;
        float idxd1 = keep ? half_nu : -half_nu;
        float det_interval = use_x ? (geo.du / std::abs(cosv)) : (geo.du / std::abs(sinv));
        float pix_interval = use_x ? vol.dx : vol.dy;
        float coef = geo.du / pix_interval;

        float p0, p1, d0p, d1p;
        if (use_x) {
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

        float bound0 = std::max(p0, d0p);
        float bound1 = std::min(p1, d1p);
        if (!(bound0 < bound1)) continue;

        int det = static_cast<int>(std::floor((bound0 - d0p) / det_interval));
        det = std::max(0, std::min(geo.Nu - 1, det));
        float det0p = d0p + det * det_interval;
        det = keep ? det : (geo.Nu - 1 - det);

        while (bound0 < bound1 && det >= 0 && det < geo.Nu) {
            float next = std::min(det0p + det_interval, bound1);
            add_proj(proj, batch, view, det, num_view, geo, image_val * (next - bound0) * coef);
            bound0 = next;
            det0p = next;
            det += keep ? 1 : -1;
        }
    }
}

template<bool FBPWEIGHT>
void backward_t_impl(float* proj, const float* image, const float* ang, Volume2d vol, Geometry2d geo, int num_batch, int num_view)
{
    int64_t proj_size = static_cast<int64_t>(num_batch) * num_view * geo.Nu;
    zero_buffer(proj, proj_size);

    int total_rows = num_batch * vol.Ny;
    int nt = thread_count(total_rows);
    std::vector<std::vector<float>> local(nt, std::vector<float>(proj_size, 0.0f));
    std::vector<std::thread> workers;
    workers.reserve(nt);

    for (int t = 0; t < nt; ++t) {
        int begin = total_rows * t / nt;
        int end = total_rows * (t + 1) / nt;
        workers.emplace_back([=, &local]() {
            float* out = local[t].data();
            for (int global_row = begin; global_row < end; ++global_row) {
                int batch = global_row / vol.Ny;
                int row = global_row % vol.Ny;
                for (int col = 0; col < vol.Nx; ++col) {
                    float image_val = image[(batch * vol.Ny + (vol.Ny - 1 - row)) * vol.Nx + col];
                    if (geo.type == GeometryType::FanFlat) {
                        fanflat_pixel_t<FBPWEIGHT>(out, image_val, ang, batch, row, col, vol, geo, num_view);
                    } else if (geo.type == GeometryType::FanArc) {
                        fanarc_pixel_t<FBPWEIGHT>(out, image_val, ang, batch, row, col, vol, geo, num_view);
                    } else {
                        parallel_pixel_t(out, image_val, ang, batch, row, col, vol, geo, num_view);
                    }
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();

    for (int t = 0; t < nt; ++t) {
        const float* src = local[t].data();
        for (int64_t i = 0; i < proj_size; ++i) proj[i] += src[i];
    }
}

} // namespace

void backward2d_cpu(float* image, float* proj, float* ang, Volume2d vol, Geometry2d geo, int num_batch, int num_view)
{
    backward_impl<false>(image, proj, ang, vol, geo, num_batch, num_view);
}

void weighted_backward2d_cpu(float* image, float* proj, float* ang, Volume2d vol, Geometry2d geo, int num_batch, int num_view)
{
    backward_impl<true>(image, proj, ang, vol, geo, num_batch, num_view);
}

void backward2d_t_cpu(float* proj, float* image, float* ang, Volume2d vol, Geometry2d geo, int num_batch, int num_view)
{
    backward_t_impl<false>(proj, image, ang, vol, geo, num_batch, num_view);
}

void weighted_backward2d_t_cpu(float* proj, float* image, float* ang, Volume2d vol, Geometry2d geo, int num_batch, int num_view)
{
    backward_t_impl<true>(proj, image, ang, vol, geo, num_batch, num_view);
}

} // namespace xrecon
