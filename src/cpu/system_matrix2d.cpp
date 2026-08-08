#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "functions.h"

namespace xrecon {

namespace {

struct MatrixTriplets {
    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    std::vector<float> values;

    void reserve(int64_t n) {
        rows.reserve(n);
        cols.reserve(n);
        values.reserve(n);
    }

    void add(int row, int col, float value) {
        if (value == 0.0f || !std::isfinite(value)) return;
        rows.push_back(static_cast<int64_t>(row));
        cols.push_back(static_cast<int64_t>(col));
        values.push_back(value);
    }
};

struct CallbackTriplets {
    MatrixCoeffEmitter emit;
    void* user;

    void add(int row, int col, float value) {
        if (value == 0.0f || !std::isfinite(value)) return;
        emit(row, col, value, user);
    }
};

inline float map2x(const Point2d& src, const Point2d& pnt) {
    return (src.x * pnt.y - src.y * pnt.x) / (pnt.y - src.y);
}

inline float map2y(const Point2d& src, const Point2d& pnt) {
    return (src.y * pnt.x - src.x * pnt.y) / (pnt.x - src.x);
}

inline float map2x(const Point2d& src, float ang) {
    return src.x + src.y * std::tan(ang);
}

inline float map2y(const Point2d& src, float ang) {
    return src.y + src.x / std::tan(ang);
}

inline float cos_weight_x(const Point2d& src, float x1, float x0) {
    float a = (x1 + x0) * 0.5f - src.x;
    float b = src.y;
    return std::abs(b) / std::sqrt(a * a + b * b);
}

inline float cos_weight_y(const Point2d& src, float y1, float y0) {
    float a = (y1 + y0) * 0.5f - src.y;
    float b = src.x;
    return std::abs(b) / std::sqrt(a * a + b * b);
}

inline int image_col(const xrecon::Volume2d& vol, int row, int col) {
    return row * vol.Nx + col;
}

inline int sino_row(const xrecon::Geometry2d& geo, int view, int det) {
    return view * geo.Nu + det;
}

template <typename Sink>
void add_forward_coeff(
    Sink& triplets,
    int row,
    int col,
    int view,
    int det_local,
    int det0,
    bool keep_dim,
    float segment,
    float scale,
    const xrecon::Volume2d& vol,
    const xrecon::Geometry2d& geo
) {
    int det = det0 + det_local;
    if (det_local < 0 || det < 0 || det >= geo.Nu || row < 0 || row >= vol.Ny || col < 0 || col >= vol.Nx) return;
    int det_out = keep_dim ? det : (geo.Nu - 1 - det);
    triplets.add(sino_row(geo, view, det_out), image_col(vol, row, col), segment * scale);
}

template <typename Sink>
void matrix_parallel(Sink& triplets, const float* angles, int num_view, const xrecon::Volume2d& vol, const xrecon::Geometry2d& geo) {
    float half_nu = geo.Nu * 0.5f;
    for (int view = 0; view < num_view; ++view) {
        float sinv = std::sin(angles[view]);
        float cosv = std::cos(angles[view]);
        bool use_x = (cosv * cosv > 0.5f);
        bool keep = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));
        float idxd0 = keep ? -half_nu : half_nu;
        float det0 = use_x ? (idxd0 * geo.du + geo.shift_u) / cosv
                           : (idxd0 * geo.du + geo.shift_u) / sinv;
        float det_interval = use_x ? geo.du / std::abs(cosv) : geo.du / std::abs(sinv);
        float pix_interval = use_x ? vol.dx : vol.dy;
        float scale = use_x ? vol.dy / geo.du : vol.dx / geo.du;

        if (use_x) {
            float x0 = -vol.Nx * 0.5f * vol.dx + vol.shift_x;
            for (int row = 0; row < vol.Ny; ++row) {
                float y = (vol.Ny * 0.5f - row - 0.5f) * vol.dy + vol.shift_y;
                float p0 = (sinv / cosv) * y + x0;
                float bound = std::max(p0, det0);
                int col, det;
                if (p0 == bound) {
                    col = 0;
                    det = static_cast<int>(std::floor((bound - det0) / det_interval));
                } else {
                    col = static_cast<int>(std::floor((bound - p0) / pix_interval));
                    det = 0;
                }
                float p_next = (col + 1) * pix_interval + p0;
                float d_next = (det + 1) * det_interval + det0;
                while (col < vol.Nx && det < geo.Nu) {
                    if (p_next < d_next) {
                        add_forward_coeff(triplets, row, col, view, det, 0, keep, p_next - bound, scale, vol, geo);
                        bound = p_next;
                        col++;
                        p_next += pix_interval;
                    } else {
                        add_forward_coeff(triplets, row, col, view, det, 0, keep, d_next - bound, scale, vol, geo);
                        bound = d_next;
                        det++;
                        d_next += det_interval;
                    }
                }
            }
        } else {
            float y0 = -vol.Ny * 0.5f * vol.dy + vol.shift_y;
            for (int col = 0; col < vol.Nx; ++col) {
                float x = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
                float p0 = (cosv / sinv) * x + y0;
                float bound = std::max(p0, det0);
                int row, det;
                if (p0 == bound) {
                    row = 0;
                    det = static_cast<int>(std::floor((bound - det0) / det_interval));
                } else {
                    row = static_cast<int>(std::floor((bound - p0) / pix_interval));
                    det = 0;
                }
                float p_next = (row + 1) * pix_interval + p0;
                float d_next = (det + 1) * det_interval + det0;
                while (row < vol.Ny && det < geo.Nu) {
                    int img_row = vol.Ny - 1 - row;
                    if (p_next < d_next) {
                        add_forward_coeff(triplets, img_row, col, view, det, 0, keep, p_next - bound, scale, vol, geo);
                        bound = p_next;
                        row++;
                        p_next += pix_interval;
                    } else {
                        add_forward_coeff(triplets, img_row, col, view, det, 0, keep, d_next - bound, scale, vol, geo);
                        bound = d_next;
                        det++;
                        d_next += det_interval;
                    }
                }
            }
        }
    }
}

template <typename Sink>
void matrix_fanflat(Sink& triplets, const float* angles, int num_view, const xrecon::Volume2d& vol, const xrecon::Geometry2d& geo) {
    std::vector<float> det_axis(geo.Nu + 1);
    for (int view = 0; view < num_view; ++view) {
        float sinv = std::sin(angles[view]);
        float cosv = std::cos(angles[view]);
        xrecon::Point2d src(-sinv * geo.SOD, cosv * geo.SOD);
        bool use_x = (cosv * cosv > 0.5f);
        bool keep = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));
        for (int det = 0; det <= geo.Nu; ++det) {
            float idxd = keep ? (det - geo.Nu * 0.5f) : (geo.Nu * 0.5f - det);
            xrecon::Point2d p((idxd * geo.du + geo.shift_u) * cosv, (idxd * geo.du + geo.shift_u) * sinv);
            det_axis[det] = use_x ? map2x(src, p) : map2y(src, p);
        }
        if (use_x) {
            float x0 = -vol.Nx * 0.5f * vol.dx + vol.shift_x;
            float x1 =  vol.Nx * 0.5f * vol.dx + vol.shift_x;
            float det_step = geo.du * std::abs(cosv);
            float det0_coord = ((keep ? -geo.Nu * 0.5f : geo.Nu * 0.5f) * geo.du + geo.shift_u) * cosv;
            for (int row = 0; row < vol.Ny; ++row) {
                float y = (vol.Ny * 0.5f - row - 0.5f) * vol.dy + vol.shift_y;
                float p0 = map2x(src, xrecon::Point2d(x0, y));
                float p1 = map2x(src, xrecon::Point2d(x1, y));
                float pix_step = (p1 - p0) / vol.Nx;
                float bound = std::max(p0, det_axis[0]);
                int col, det;
                if (p0 == bound) {
                    float b2d = src.y * bound / ((bound - src.x) * sinv / cosv + src.y);
                    col = 0;
                    det = static_cast<int>(std::floor((b2d - det0_coord) / det_step));
                } else {
                    col = static_cast<int>(std::floor((bound - p0) / pix_step));
                    det = 0;
                }
                float p_next = (col + 1) * pix_step + p0;
                float d_next = (det < geo.Nu) ? det_axis[det + 1] : 0.f;
                while (col < vol.Nx && det < geo.Nu) {
                    float scale = vol.dy / ((det_axis[det + 1] - det_axis[det]) * cos_weight_x(src, det_axis[det + 1], det_axis[det]));
                    if (p_next < d_next) {
                        add_forward_coeff(triplets, row, col, view, det, 0, keep, p_next - bound, scale, vol, geo);
                        bound = p_next;
                        col++;
                        p_next += pix_step;
                    } else {
                        add_forward_coeff(triplets, row, col, view, det, 0, keep, d_next - bound, scale, vol, geo);
                        bound = d_next;
                        det++;
                        if (det < geo.Nu) d_next = det_axis[det + 1];
                    }
                }
            }
        } else {
            float y0 = -vol.Ny * 0.5f * vol.dy + vol.shift_y;
            float y1 =  vol.Ny * 0.5f * vol.dy + vol.shift_y;
            float det_step = geo.du * std::abs(sinv);
            float det0_coord = ((keep ? -geo.Nu * 0.5f : geo.Nu * 0.5f) * geo.du + geo.shift_u) * sinv;
            for (int col = 0; col < vol.Nx; ++col) {
                float x = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
                float p0 = map2y(src, xrecon::Point2d(x, y0));
                float p1 = map2y(src, xrecon::Point2d(x, y1));
                float pix_step = (p1 - p0) / vol.Ny;
                float bound = std::max(p0, det_axis[0]);
                int row, det;
                if (p0 == bound) {
                    float b2d = src.x * bound / ((bound - src.y) * cosv / sinv + src.x);
                    row = 0;
                    det = static_cast<int>(std::floor((b2d - det0_coord) / det_step));
                } else {
                    row = static_cast<int>(std::floor((bound - p0) / pix_step));
                    det = 0;
                }
                float p_next = (row + 1) * pix_step + p0;
                float d_next = (det < geo.Nu) ? det_axis[det + 1] : 0.f;
                while (row < vol.Ny && det < geo.Nu) {
                    int img_row = vol.Ny - 1 - row;
                    float scale = vol.dx / ((det_axis[det + 1] - det_axis[det]) * cos_weight_y(src, det_axis[det + 1], det_axis[det]));
                    if (p_next < d_next) {
                        add_forward_coeff(triplets, img_row, col, view, det, 0, keep, p_next - bound, scale, vol, geo);
                        bound = p_next;
                        row++;
                        p_next += pix_step;
                    } else {
                        add_forward_coeff(triplets, img_row, col, view, det, 0, keep, d_next - bound, scale, vol, geo);
                        bound = d_next;
                        det++;
                        if (det < geo.Nu) d_next = det_axis[det + 1];
                    }
                }
            }
        }
    }
}

template <typename Sink>
void matrix_fanarc(Sink& triplets, const float* angles, int num_view, const xrecon::Volume2d& vol, const xrecon::Geometry2d& geo) {
    std::vector<float> det_axis(geo.Nu + 1);
    for (int view = 0; view < num_view; ++view) {
        float beta = angles[view];
        float sinv = std::sin(beta);
        float cosv = std::cos(beta);
        xrecon::Point2d src(-sinv * geo.SOD, cosv * geo.SOD);
        bool use_x = (cosv * cosv > 0.5f);
        bool keep = ((cosv * cosv <= 0.5f && sinv >= 0.f) || (cosv * cosv > 0.5f && cosv >= 0.f));
        int flag = keep ? 1 : -1;
        for (int det = 0; det <= geo.Nu; ++det) {
            float det_ang = (keep ? (det - geo.Nu * 0.5f) : (geo.Nu * 0.5f - det)) * geo.du + geo.shift_u + beta;
            det_axis[det] = use_x ? map2x(src, det_ang) : map2y(src, det_ang);
        }
        if (use_x) {
            float x0 = -vol.Nx * 0.5f * vol.dx + vol.shift_x;
            float x1 =  vol.Nx * 0.5f * vol.dx + vol.shift_x;
            for (int row = 0; row < vol.Ny; ++row) {
                float y = (vol.Ny * 0.5f - row - 0.5f) * vol.dy + vol.shift_y;
                float p0 = map2x(src, xrecon::Point2d(x0, y));
                float p1 = map2x(src, xrecon::Point2d(x1, y));
                float pix_step = (p1 - p0) / vol.Nx;
                float tan0 = ((src.x - p0) == 0.f) ? 1e10f : (src.y / (src.x - p0));
                float tanc = (geo.shift_u + beta == 0.f) ? 1e10f : (-1.0f / std::tan(geo.shift_u + beta));
                float delta = std::atan((tan0 - tanc) / (1.0f + tan0 * tanc));
                int det = static_cast<int>(std::floor(geo.Nu * 0.5f + delta * flag / geo.du));
                float bound;
                int col;
                if (det < 0) {
                    bound = det_axis[0];
                    col = static_cast<int>(std::floor((bound - p0) / pix_step));
                    det = 0;
                } else {
                    bound = p0;
                    col = 0;
                }
                float p_next = (col + 1) * pix_step + p0;
                float d_next = (det < geo.Nu) ? det_axis[det + 1] : 0.f;
                while (col < vol.Nx && det < geo.Nu) {
                    float scale = vol.dy / ((det_axis[det + 1] - det_axis[det]) * cos_weight_x(src, det_axis[det + 1], det_axis[det]));
                    if (p_next < d_next) {
                        add_forward_coeff(triplets, row, col, view, det, 0, keep, p_next - bound, scale, vol, geo);
                        bound = p_next;
                        col++;
                        p_next += pix_step;
                    } else {
                        add_forward_coeff(triplets, row, col, view, det, 0, keep, d_next - bound, scale, vol, geo);
                        bound = d_next;
                        det++;
                        if (det < geo.Nu) d_next = det_axis[det + 1];
                    }
                }
            }
        } else {
            float y0 = -vol.Ny * 0.5f * vol.dy + vol.shift_y;
            float y1 =  vol.Ny * 0.5f * vol.dy + vol.shift_y;
            for (int col = 0; col < vol.Nx; ++col) {
                float x = (col - vol.Nx * 0.5f + 0.5f) * vol.dx + vol.shift_x;
                float p0 = map2y(src, xrecon::Point2d(x, y0));
                float p1 = map2y(src, xrecon::Point2d(x, y1));
                float pix_step = (p1 - p0) / vol.Ny;
                float tan0 = (src.y - p0) / src.x;
                float tanc = -1.0f / std::tan(geo.shift_u + beta);
                float delta = std::atan((tan0 - tanc) / (1.0f + tan0 * tanc));
                int det = static_cast<int>(std::floor(geo.Nu * 0.5f + delta * flag / geo.du));
                float bound;
                int row;
                if (det < 0) {
                    bound = det_axis[0];
                    row = static_cast<int>(std::floor((bound - p0) / pix_step));
                    det = 0;
                } else {
                    bound = p0;
                    row = 0;
                }
                float p_next = (row + 1) * pix_step + p0;
                float d_next = (det < geo.Nu) ? det_axis[det + 1] : 0.f;
                while (row < vol.Ny && det < geo.Nu) {
                    int img_row = vol.Ny - 1 - row;
                    float scale = vol.dx / ((det_axis[det + 1] - det_axis[det]) * cos_weight_y(src, det_axis[det + 1], det_axis[det]));
                    if (p_next < d_next) {
                        add_forward_coeff(triplets, img_row, col, view, det, 0, keep, p_next - bound, scale, vol, geo);
                        bound = p_next;
                        row++;
                        p_next += pix_step;
                    } else {
                        add_forward_coeff(triplets, img_row, col, view, det, 0, keep, d_next - bound, scale, vol, geo);
                        bound = d_next;
                        det++;
                        if (det < geo.Nu) d_next = det_axis[det + 1];
                    }
                }
            }
        }
    }
}

} // namespace

void system_matrix2d_cpu(
    std::vector<int64_t>& rows,
    std::vector<int64_t>& cols,
    std::vector<float>& values,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_view
) {
    MatrixTriplets triplets;
    triplets.reserve(static_cast<int64_t>(num_view) * geo.Nu * std::max(vol.Nx, vol.Ny) * 2);

    switch (geo.type) {
    case GeometryType::Parallel:
        matrix_parallel(triplets, ang, num_view, vol, geo);
        break;
    case GeometryType::FanFlat:
        matrix_fanflat(triplets, ang, num_view, vol, geo);
        break;
    case GeometryType::FanArc:
        matrix_fanarc(triplets, ang, num_view, vol, geo);
        break;
    }

    rows = std::move(triplets.rows);
    cols = std::move(triplets.cols);
    values = std::move(triplets.values);
}

void distance_driven2d_cpu(
    MatrixCoeffEmitter emit,
    void* user,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_view
) {
    CallbackTriplets triplets{emit, user};

    switch (geo.type) {
    case GeometryType::Parallel:
        matrix_parallel(triplets, ang, num_view, vol, geo);
        break;
    case GeometryType::FanFlat:
        matrix_fanflat(triplets, ang, num_view, vol, geo);
        break;
    case GeometryType::FanArc:
        matrix_fanarc(triplets, ang, num_view, vol, geo);
        break;
    }
}

} // namespace xrecon
