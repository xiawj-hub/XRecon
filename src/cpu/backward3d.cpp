#include <vector>
#include <thread>
#include "functions.h"
#include "cpu3d_common.h"

namespace xrecon {

namespace {

void cone3d_backprojection_impl(
    float* image,
    const float* proj,
    const float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view,
    bool weighted)
{
    const int total = num_batch * vol.Nz * vol.Ny;
    const int nt = cpu3d::thread_count(total);
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
                        cpu3d::visit_cone3d_backprojection_coefficients(ix, iy, iz, view, ang, vol, geo, weighted,
                            [&](int row, int col, float coeff) {
                                value += proj[cpu3d::proj_index(batch, view, row, col, num_view, geo)] * coeff;
                            });
                    }
                    image[cpu3d::image_index(batch, iz, iy, ix, vol)] = value;
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

void cone3d_backprojection_t_impl(
    float* proj,
    const float* image,
    const float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view,
    bool weighted)
{
    cpu3d::zero_buffer(proj, static_cast<int64_t>(num_batch) * num_view * geo.Nv * geo.Nu);
    const int total = num_batch * num_view * vol.Nz;
    const int nt = cpu3d::thread_count(total);
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
                        const float image_val = image[cpu3d::image_index(batch, iz, iy, ix, vol)];
                        if (image_val == 0.0f) continue;
                        cpu3d::visit_cone3d_backprojection_coefficients(ix, iy, iz, view, ang, vol, geo, weighted,
                            [&](int row, int col, float coeff) {
                                local_proj[cpu3d::proj_index(batch, view, row, col, num_view, geo)] += image_val * coeff;
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

} // namespace

void backward3d_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    if (geo.type == GeometryType::Parallel) {
        forward3d_t_cpu(image, proj, ang, vol, geo, num_batch, num_view);
    } else {
        cone3d_backprojection_impl(image, proj, ang, vol, geo, num_batch, num_view, false);
    }
}

void weighted_backward3d_cpu(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    if (geo.type == GeometryType::Parallel) {
        forward3d_t_cpu(image, proj, ang, vol, geo, num_batch, num_view);
    } else {
        cone3d_backprojection_impl(image, proj, ang, vol, geo, num_batch, num_view, true);
    }
}

void backward3d_t_cpu(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    if (geo.type == GeometryType::Parallel) {
        forward3d_cpu(image, proj, ang, vol, geo, num_batch, num_view);
    } else {
        cone3d_backprojection_t_impl(proj, image, ang, vol, geo, num_batch, num_view, false);
    }
}

void weighted_backward3d_t_cpu(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    if (geo.type == GeometryType::Parallel) {
        forward3d_cpu(image, proj, ang, vol, geo, num_batch, num_view);
    } else {
        cone3d_backprojection_t_impl(proj, image, ang, vol, geo, num_batch, num_view, true);
    }
}

} // namespace xrecon
