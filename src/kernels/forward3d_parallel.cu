#include <cuda.h>
#include "cone3d_common.cuh"

#define BLOCK_X 8
#define BLOCK_Y 8
#define BLOCK_Z 4

namespace {

__device__ inline float image_value(
    const float* __restrict__ image,
    int batch,
    int iz,
    int iy,
    int ix,
    xrecon::Volume3d vol)
{
    return image[((batch * vol.Nz + iz) * vol.Ny + iy) * vol.Nx + ix];
}

__device__ inline float projection_value(
    const float* __restrict__ proj,
    int batch,
    int view,
    int row,
    int col,
    int num_view,
    xrecon::Geometry3d geo)
{
    return proj[((batch * num_view + view) * geo.Nv + row) * geo.Nu + col];
}

template <typename Fn>
__device__ void parallel3d_visit_coefficients(
    int ix,
    int iy,
    int iz,
    int view,
    const float* __restrict__ angles,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    Fn&& fn)
{
    const float sinv = sinf(angles[view]);
    const float cosv = cosf(angles[view]);
    const bool use_x_axis = cosv * cosv > 0.5f;
    const bool keep_order = ((!use_x_axis && sinv >= 0.0f) || (use_x_axis && cosv >= 0.0f));

    const float idx_det0 = keep_order ? -0.5f * geo.Nu : 0.5f * geo.Nu;
    const float det_axis0 = use_x_axis ? (idx_det0 * geo.du + geo.shift_u) / cosv
                                       : (idx_det0 * geo.du + geo.shift_u) / sinv;
    const float det_interval = use_x_axis ? geo.du / fabsf(cosv) : geo.du / fabsf(sinv);
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

    float axis_bound = fmaxf(voxel_axis0, det_axis0);
    float axis_end = fminf(voxel_axis1, det_axis0 + geo.Nu * det_interval);
    if (!(axis_bound < axis_end)) return;

    int det = static_cast<int>(floorf((axis_bound - det_axis0) / det_interval));
    det = max(0, min(geo.Nu - 1, det));
    float det_next = det_axis0 + (det + 1) * det_interval;

    const float voxel_z0 = (iz - 0.5f * vol.Nz) * vol.dz + vol.shift_z;
    const float voxel_z1 = voxel_z0 + vol.dz;
    const float det_z0 = -0.5f * geo.Nv * geo.dv + geo.shift_v;
    float z_bound = fmaxf(voxel_z0, det_z0);
    const float z_end = fminf(voxel_z1, det_z0 + geo.Nv * geo.dv);
    if (!(z_bound < z_end)) return;

    int row0 = static_cast<int>(floorf((z_bound - det_z0) / geo.dv));
    row0 = max(0, min(geo.Nv - 1, row0));

    while (axis_bound < axis_end && det >= 0 && det < geo.Nu) {
        const float axis_next = fminf(det_next, axis_end);
        const float axis_coeff = (axis_next - axis_bound) * xy_scale;
        float row_bound = z_bound;
        int row = row0;
        float row_next = det_z0 + (row + 1) * geo.dv;
        while (row_bound < z_end && row >= 0 && row < geo.Nv) {
            const float z_next = fminf(row_next, z_end);
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

__global__ void parallel3d_forward_cuda_kernel(
    float* __restrict__ proj,
    const float* __restrict__ image,
    const float* __restrict__ angles,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    int num_batch,
    int num_view)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z % ((vol.Nz + blockDim.z - 1) / blockDim.z) * blockDim.z + threadIdx.z;
    const int batch_view = blockIdx.z / ((vol.Nz + blockDim.z - 1) / blockDim.z);
    const int batch = batch_view / num_view;
    const int view = batch_view % num_view;
    if (batch >= num_batch || ix >= vol.Nx || iy >= vol.Ny || iz >= vol.Nz) return;

    const float value = image_value(image, batch, iz, iy, ix, vol);
    if (value == 0.0f) return;

    parallel3d_visit_coefficients(ix, iy, iz, view, angles, vol, geo,
        [&](int row, int col, float coeff) {
            atomicAdd(proj + ((batch * num_view + view) * geo.Nv + row) * geo.Nu + col, value * coeff);
        });
}

__global__ void parallel3d_projection_t_cuda_kernel(
    float* __restrict__ image,
    const float* __restrict__ proj,
    const float* __restrict__ angles,
    xrecon::Volume3d vol,
    xrecon::Geometry3d geo,
    int num_batch,
    int num_view)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int iz = blockIdx.z % ((vol.Nz + blockDim.z - 1) / blockDim.z) * blockDim.z + threadIdx.z;
    const int batch = blockIdx.z / ((vol.Nz + blockDim.z - 1) / blockDim.z);
    if (batch >= num_batch || ix >= vol.Nx || iy >= vol.Ny || iz >= vol.Nz) return;

    float value = 0.0f;
    for (int view = 0; view < num_view; ++view) {
        parallel3d_visit_coefficients(ix, iy, iz, view, angles, vol, geo,
            [&](int row, int col, float coeff) {
                value += projection_value(proj, batch, view, row, col, num_view, geo) * coeff;
            });
    }
    image[((batch * vol.Nz + iz) * vol.Ny + iy) * vol.Nx + ix] = value;
}

} // namespace

namespace xrecon {

void forward3d_parallel_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view)
{
    CUDA_CHECK(cudaMemset(proj, 0, static_cast<size_t>(num_batch) * num_view * geo.Nv * geo.Nu * sizeof(float)));
    const dim3 block_dim(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const int num_depth_blocks = (vol.Nz + block_dim.z - 1) / block_dim.z;
    const dim3 grid_dim((vol.Nx + block_dim.x - 1) / block_dim.x,
                        (vol.Ny + block_dim.y - 1) / block_dim.y,
                        num_batch * num_view * num_depth_blocks);
    parallel3d_forward_cuda_kernel<<<grid_dim, block_dim>>>(proj, image, ang, vol, geo, num_batch, num_view);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void forward3d_t_parallel_cuda(
    float* image,
    float* proj,
    float* ang,
    Volume3d vol,
    Geometry3d geo,
    int num_batch,
    int num_view)
{
    const dim3 block_dim(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const int num_depth_blocks = (vol.Nz + block_dim.z - 1) / block_dim.z;
    const dim3 grid_dim((vol.Nx + block_dim.x - 1) / block_dim.x,
                        (vol.Ny + block_dim.y - 1) / block_dim.y,
                        num_batch * num_depth_blocks);
    parallel3d_projection_t_cuda_kernel<<<grid_dim, block_dim>>>(image, proj, ang, vol, geo, num_batch, num_view);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void backward3d_parallel_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    forward3d_t_parallel_cuda(image, proj, ang, vol, geo, num_batch, num_view);
}

void weighted_backward3d_parallel_cuda(float* image, float* proj, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    forward3d_t_parallel_cuda(image, proj, ang, vol, geo, num_batch, num_view);
}

void backward3d_t_parallel_cuda(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    forward3d_parallel_cuda(image, proj, ang, vol, geo, num_batch, num_view);
}

void weighted_backward3d_t_parallel_cuda(float* proj, float* image, float* ang, Volume3d vol, Geometry3d geo, int num_batch, int num_view)
{
    forward3d_parallel_cuda(image, proj, ang, vol, geo, num_batch, num_view);
}

} // namespace xrecon
