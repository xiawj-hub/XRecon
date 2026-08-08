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

    void reserve(int64_t n)
    {
        rows.reserve(n);
        cols.reserve(n);
        values.reserve(n);
    }

    void add(int row, int col, float value)
    {
        if (value == 0.0f || !std::isfinite(value)) return;
        rows.push_back(static_cast<int64_t>(row));
        cols.push_back(static_cast<int64_t>(col));
        values.push_back(value);
    }
};

void emit_triplet(int row, int col, float value, void* user)
{
    static_cast<MatrixTriplets*>(user)->add(row, col, value);
}

} // namespace

void system_matrix3d_cpu(
    std::vector<int64_t>& rows,
    std::vector<int64_t>& cols,
    std::vector<float>& values,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_view
) {
    MatrixTriplets triplets;
    const int64_t image_size = static_cast<int64_t>(vol.Nx) * vol.Ny * vol.Nz;
    const int64_t detector_size = static_cast<int64_t>(geo.Nu) * geo.Nv;
    const int64_t reserve_hint = std::min<int64_t>(
        image_size * num_view * 16,
        static_cast<int64_t>(num_view) * detector_size * std::max({vol.Nx, vol.Ny, vol.Nz}) * 8
    );
    triplets.reserve(std::max<int64_t>(reserve_hint, 1));

    distance_driven3d_cpu(emit_triplet, &triplets, ang, vol, geo, num_view);

    rows = std::move(triplets.rows);
    cols = std::move(triplets.cols);
    values = std::move(triplets.values);
}

} // namespace xrecon
