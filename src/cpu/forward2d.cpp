#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>
#include "functions.h"

namespace xrecon {

namespace {

inline void zero_buffer(float* ptr, int64_t n) {
    std::fill(ptr, ptr + n, 0.0f);
}

inline int thread_count(int work_items)
{
    unsigned int hw = std::thread::hardware_concurrency();
    int n = hw == 0 ? 1 : static_cast<int>(hw);
    return std::max(1, std::min(n, work_items));
}

struct ForwardUser {
    float* image;
    float* proj;
    int num_batch;
    int64_t image_size;
    int64_t proj_size;
    int row_offset;
};

void emit_forward(int row, int col, float value, void* user)
{
    auto* ctx = static_cast<ForwardUser*>(user);
    row += ctx->row_offset;
    for (int batch = 0; batch < ctx->num_batch; ++batch) {
        ctx->proj[static_cast<int64_t>(batch) * ctx->proj_size + row] +=
            value * ctx->image[static_cast<int64_t>(batch) * ctx->image_size + col];
    }
}

void emit_forward_t(int row, int col, float value, void* user)
{
    auto* ctx = static_cast<ForwardUser*>(user);
    row += ctx->row_offset;
    for (int batch = 0; batch < ctx->num_batch; ++batch) {
        ctx->image[static_cast<int64_t>(batch) * ctx->image_size + col] +=
            value * ctx->proj[static_cast<int64_t>(batch) * ctx->proj_size + row];
    }
}

} // namespace

void forward2d_cpu(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
) {
    zero_buffer(proj, static_cast<int64_t>(num_batch) * num_view * geo.Nu);

    const int64_t image_size = static_cast<int64_t>(vol.Nx) * vol.Ny;
    const int64_t proj_size = static_cast<int64_t>(num_view) * geo.Nu;
    int nt = thread_count(num_view);
    std::vector<std::thread> workers;
    workers.reserve(nt);
    for (int t = 0; t < nt; ++t) {
        int begin = num_view * t / nt;
        int end = num_view * (t + 1) / nt;
        workers.emplace_back([=]() {
            ForwardUser user{image, proj, num_batch, image_size, proj_size, begin * geo.Nu};
            distance_driven2d_cpu(emit_forward, &user, ang + begin, vol, geo, end - begin);
        });
    }
    for (auto& worker : workers) worker.join();
}

void forward2d_t_cpu(
    float* image,
    float* proj,
    float* ang,
    Volume2d vol,
    Geometry2d geo,
    int num_batch,
    int num_view
) {
    const int64_t image_size = static_cast<int64_t>(vol.Nx) * vol.Ny;
    const int64_t proj_size = static_cast<int64_t>(num_view) * geo.Nu;
    const int64_t image_size_all = static_cast<int64_t>(num_batch) * image_size;
    zero_buffer(image, image_size_all);

    int nt = thread_count(num_view);
    std::vector<std::vector<float>> local(nt, std::vector<float>(image_size_all, 0.0f));
    std::vector<std::thread> workers;
    workers.reserve(nt);
    for (int t = 0; t < nt; ++t) {
        int begin = num_view * t / nt;
        int end = num_view * (t + 1) / nt;
        workers.emplace_back([=, &local]() {
            ForwardUser user{local[t].data(), proj, num_batch, image_size, proj_size, begin * geo.Nu};
            distance_driven2d_cpu(emit_forward_t, &user, ang + begin, vol, geo, end - begin);
        });
    }
    for (auto& worker : workers) worker.join();

    for (int t = 0; t < nt; ++t) {
        const float* src = local[t].data();
        for (int64_t i = 0; i < image_size_all; ++i) image[i] += src[i];
    }
}

} // namespace xrecon
