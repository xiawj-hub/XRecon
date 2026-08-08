#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "types.h"
#include "functions.h"

void helical_arc_forward_cuda(float*, float*, float*, float*, float*, float*, float*, int, xrecon::Volume3d, xrecon::Geometry3d, int);
void helical_arc_forward_t_cuda(float*, float*, float*, float*, float*, float*, float*, int, xrecon::Volume3d, xrecon::Geometry3d, int);
void helical_arc_backward_cuda(float*, float*, float*, float*, float*, float*, float*, int, xrecon::Volume3d, xrecon::Geometry3d, int);
void helical_arc_backward_t_cuda(float*, float*, float*, float*, float*, float*, float*, int, xrecon::Volume3d, xrecon::Geometry3d, int);
void helical_arc_weighted_backward_cuda(float*, float*, float*, float*, float*, float*, float*, int, xrecon::Volume3d, xrecon::Geometry3d, int);
void helical_arc_weighted_backward_t_cuda(float*, float*, float*, float*, float*, float*, float*, int, xrecon::Volume3d, xrecon::Geometry3d, int);

#define CHECK_DEVICE(a, b, c) \
  TORCH_CHECK( \
      (a.device() == b.device()) && (a.device() == c.device()), \
      "Expected all tensors on the same device, but got devices: ", \
      a.device(), ", ", b.device(), ", ", c.device())

#define CHECK_CONTIGUOUS(a, b, c) \
  TORCH_CHECK( \
      (a.is_contiguous() && b.is_contiguous() && c.is_contiguous()), \
      "Expected all tensors to be contiguous")

#define CHECK_INPUT(a, b, c) \
  CHECK_DEVICE(a, b, c); \
  CHECK_CONTIGUOUS(a, b, c)

#define CHECK_FLOAT_CUDA(t) \
  TORCH_CHECK(t.is_cuda(), #t " must be a CUDA tensor"); \
  TORCH_CHECK(t.scalar_type() == torch::kFloat32, #t " must be float32")

namespace xrecon_torch {

struct Packed2d {
    xrecon::Volume2d vol;
    xrecon::Geometry2d geo;
    int W;
    int H;
    int Nu;
};

struct Packed3d {
    xrecon::Volume3d vol;
    xrecon::Geometry3d geo;
    int W, H, D;
    int Nu, Nv;
    float dx, dy, dz;
    float du, dv;
    float SOD, SDD;
    float shx, shy, shz;
    float shu, shv;
    int scan_id;
    int helical_view_mode;
    int helical_weight_mode;
    float helical_view_turns;
    float helical_halfscan_coef;
    float helical_tang_k;
    float helical_tang_blend;
};

Packed2d unpack2d(torch::Tensor options)
{
    TORCH_CHECK(options.numel() >= 12, "Option tensor must have at least 12 elements.");
    int W = options[0].item<int>();
    int H = options[1].item<int>();
    int Nu = options[2].item<int>();
    float dx = options[3].item<float>();
    float dy = options[4].item<float>();
    float du = options[5].item<float>();
    float SOD = options[6].item<float>();
    float SDD = options[7].item<float>();
    float shx = options[8].item<float>();
    float shy = options[9].item<float>();
    float binShift = options[10].item<float>();
    auto geom_type = static_cast<xrecon::GeometryType>(options[11].item<int>());
    return {xrecon::Volume2d(W, H, dx, dy, shx, shy),
            xrecon::Geometry2d(geom_type, Nu, du, binShift, SOD, SDD),
            W, H, Nu};
}

Packed3d unpack3d(torch::Tensor options)
{
    TORCH_CHECK(options.numel() >= 18, "3D option tensor must have at least 18 elements.");
    int W = options[0].item<int>();
    int H = options[1].item<int>();
    int D = options[2].item<int>();
    int Nu = options[3].item<int>();
    int Nv = options[4].item<int>();
    float dx = options[5].item<float>();
    float dy = options[6].item<float>();
    float dz = options[7].item<float>();
    float du = options[8].item<float>();
    float dv = options[9].item<float>();
    float SOD = options[10].item<float>();
    float SDD = options[11].item<float>();
    float shx = options[12].item<float>();
    float shy = options[13].item<float>();
    float shz = options[14].item<float>();
    float shu = options[15].item<float>();
    float shv = options[16].item<float>();
    auto geom_type = static_cast<xrecon::GeometryType>(options[17].item<int>());
    int helical_view_mode = options.numel() > 18 ? options[18].item<int>() : 2;
    int helical_weight_mode = options.numel() > 19 ? options[19].item<int>() : 0;
    float helical_view_turns = options.numel() > 20 ? options[20].item<float>() : 0.0f;
    float helical_halfscan_coef = options.numel() > 21 ? options[21].item<float>() : 1.0f;
    float helical_tang_k = options.numel() > 22 ? options[22].item<float>() : 5.0f;
    float helical_tang_blend = options.numel() > 23 ? options[23].item<float>() : 0.0f;
    TORCH_CHECK(geom_type == xrecon::GeometryType::Parallel ||
                geom_type == xrecon::GeometryType::FanFlat ||
                geom_type == xrecon::GeometryType::FanArc,
                "3D scan_type must be parallel, flat, or arc");
    return {
        xrecon::Volume3d(W, H, D, dx, dy, dz, shx, shy, shz),
        xrecon::Geometry3d(
            geom_type, Nu, Nv, du, dv, shu, shv, SOD, SDD,
            helical_view_mode, helical_weight_mode,
            helical_view_turns, helical_halfscan_coef, helical_tang_k, helical_tang_blend),
        W, H, D, Nu, Nv, dx, dy, dz, du, dv, SOD, SDD, shx, shy, shz, shu, shv,
        static_cast<int>(geom_type),
        helical_view_mode, helical_weight_mode,
        helical_view_turns, helical_halfscan_coef, helical_tang_k, helical_tang_blend,
    };
}

using HelicalArc3dFn = void (*)(float*, float*, float*, float*, float*, float*, float*, int, xrecon::Volume3d, xrecon::Geometry3d, int);

void launch_helical_arc3d(HelicalArc3dFn fn, float* image, float* proj, float* ang, torch::Tensor source_pos, int batch_channel, int num_view, Packed3d p)
{
    TORCH_CHECK(source_pos.dim() == 2 && source_pos.size(0) == 4 && source_pos.size(1) == num_view,
                "source_pos must have shape [4, num_view]");
    float* source_z = source_pos.data_ptr<float>();
    float* shift_radius = source_z + num_view;
    float* shift_angle = source_z + 2 * num_view;
    float* shift_z = source_z + 3 * num_view;
    fn(image, proj, ang, source_z, shift_radius, shift_angle, shift_z,
       batch_channel, p.vol, p.geo, num_view);
}

torch::Tensor forward_projection2d(torch::Tensor image, torch::Tensor options, torch::Tensor ang)
{
    
    CHECK_INPUT(image, options, ang);
    TORCH_CHECK(image.scalar_type() == torch::kFloat32, "image must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack2d(options);

    int B = static_cast<int>(image.size(0));
    int C = static_cast<int>(image.size(1));
    int num_view = static_cast<int>(ang.size(0));
    
    auto proj = torch::empty({B, C, num_view, packed.Nu}, image.options());

    if (image.is_cuda()) {
        CHECK_FLOAT_CUDA(image);
        CHECK_FLOAT_CUDA(options);
        CHECK_FLOAT_CUDA(ang);
        xrecon::forward2d_cuda(
            image.data_ptr<float>(),
            proj.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    } else {
        xrecon::forward2d_cpu(
            image.data_ptr<float>(),
            proj.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    }

    return proj;  
}

torch::Tensor backward_projection2d(torch::Tensor proj, torch::Tensor options, torch::Tensor ang)
{
    
    CHECK_INPUT(proj, options, ang);
    TORCH_CHECK(proj.scalar_type() == torch::kFloat32, "proj must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack2d(options);

    int B = static_cast<int>(proj.size(0));
    int C = static_cast<int>(proj.size(1));
    int num_view = static_cast<int>(ang.size(0));
    
    auto image = torch::empty({B, C, packed.H, packed.W}, proj.options());

    if (proj.is_cuda()) {
        CHECK_FLOAT_CUDA(proj);
        CHECK_FLOAT_CUDA(options);
        CHECK_FLOAT_CUDA(ang);
        xrecon::backward2d_cuda(
            image.data_ptr<float>(),
            proj.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    } else {
        xrecon::backward2d_cpu(
            image.data_ptr<float>(),
            proj.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    }

    return image;  
}

torch::Tensor weighted_backward_projection2d(torch::Tensor proj, torch::Tensor options, torch::Tensor ang)
{
    
    CHECK_INPUT(proj, options, ang);
    TORCH_CHECK(proj.scalar_type() == torch::kFloat32, "proj must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack2d(options);

    int B = static_cast<int>(proj.size(0));
    int C = static_cast<int>(proj.size(1));
    int num_view = static_cast<int>(ang.size(0));
    
    auto image = torch::empty({B, C, packed.H, packed.W}, proj.options());

    if (proj.is_cuda()) {
        CHECK_FLOAT_CUDA(proj);
        CHECK_FLOAT_CUDA(options);
        CHECK_FLOAT_CUDA(ang);
        xrecon::weighted_backward2d_cuda(
            image.data_ptr<float>(),
            proj.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    } else {
        xrecon::weighted_backward2d_cpu(
            image.data_ptr<float>(),
            proj.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    }

    return image;  
}

torch::Tensor projection_transpose2d(torch::Tensor proj, torch::Tensor options, torch::Tensor ang)
{
    CHECK_INPUT(proj, options, ang);
    TORCH_CHECK(proj.scalar_type() == torch::kFloat32, "proj must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack2d(options);

    int B = static_cast<int>(proj.size(0));
    int C = static_cast<int>(proj.size(1));
    int num_view = static_cast<int>(ang.size(0));
    auto image = torch::zeros({B, C, packed.H, packed.W}, proj.options());

    if (proj.is_cuda()) {
        CHECK_FLOAT_CUDA(proj);
        CHECK_FLOAT_CUDA(options);
        CHECK_FLOAT_CUDA(ang);
        xrecon::forward2d_t_cuda(
            image.data_ptr<float>(),
            proj.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    } else {
        xrecon::forward2d_t_cpu(
            image.data_ptr<float>(),
            proj.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    }
    return image;
}

torch::Tensor backward_projection_transpose2d(torch::Tensor image, torch::Tensor options, torch::Tensor ang)
{
    CHECK_INPUT(image, options, ang);
    TORCH_CHECK(image.scalar_type() == torch::kFloat32, "image must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack2d(options);

    int B = static_cast<int>(image.size(0));
    int C = static_cast<int>(image.size(1));
    int num_view = static_cast<int>(ang.size(0));
    auto proj = torch::zeros({B, C, num_view, packed.Nu}, image.options());

    if (image.is_cuda()) {
        CHECK_FLOAT_CUDA(image);
        CHECK_FLOAT_CUDA(options);
        CHECK_FLOAT_CUDA(ang);
        xrecon::backward2d_t_cuda(
            proj.data_ptr<float>(),
            image.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    } else {
        xrecon::backward2d_t_cpu(
            proj.data_ptr<float>(),
            image.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    }
    return proj;
}

torch::Tensor weighted_backward_projection_transpose2d(torch::Tensor image, torch::Tensor options, torch::Tensor ang)
{
    CHECK_INPUT(image, options, ang);
    TORCH_CHECK(image.scalar_type() == torch::kFloat32, "image must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack2d(options);

    int B = static_cast<int>(image.size(0));
    int C = static_cast<int>(image.size(1));
    int num_view = static_cast<int>(ang.size(0));
    auto proj = torch::zeros({B, C, num_view, packed.Nu}, image.options());

    if (image.is_cuda()) {
        CHECK_FLOAT_CUDA(image);
        CHECK_FLOAT_CUDA(options);
        CHECK_FLOAT_CUDA(ang);
        xrecon::weighted_backward2d_t_cuda(
            proj.data_ptr<float>(),
            image.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    } else {
        xrecon::weighted_backward2d_t_cpu(
            proj.data_ptr<float>(),
            image.data_ptr<float>(),
            ang.data_ptr<float>(),
            packed.vol, packed.geo, B*C, num_view
        );
    }
    return proj;
}

torch::Tensor system_matrix2d(torch::Tensor options, torch::Tensor ang)
{
    TORCH_CHECK(!options.is_cuda() && !ang.is_cuda(), "system_matrix2d builds the matrix on CPU");
    CHECK_INPUT(options, ang, ang);
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");

    auto packed = unpack2d(options);
    int num_view = static_cast<int>(ang.size(0));
    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    std::vector<float> values;
    xrecon::system_matrix2d_cpu(
        rows, cols, values,
        ang.data_ptr<float>(),
        packed.vol, packed.geo, num_view
    );

    auto idx_i = torch::from_blob(rows.data(), {static_cast<int64_t>(rows.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
    auto idx_j = torch::from_blob(cols.data(), {static_cast<int64_t>(cols.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
    auto vals = torch::from_blob(values.data(), {static_cast<int64_t>(values.size())}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto indices = torch::stack({idx_i, idx_j}).view({2, -1});
    int64_t matrix_rows = static_cast<int64_t>(num_view) * packed.Nu;
    int64_t matrix_cols = static_cast<int64_t>(packed.W) * packed.H;
    return torch::sparse_coo_tensor(indices, vals, {matrix_rows, matrix_cols}).coalesce();
}

torch::Tensor system_matrix3d(torch::Tensor options, torch::Tensor ang)
{
    TORCH_CHECK(!options.is_cuda() && !ang.is_cuda(), "system_matrix3d builds the matrix on CPU");
    CHECK_INPUT(options, ang, ang);
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");

    auto packed = unpack3d(options);
    int num_view = static_cast<int>(ang.size(0));
    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    std::vector<float> values;
    xrecon::system_matrix3d_cpu(
        rows, cols, values,
        ang.data_ptr<float>(),
        packed.vol, packed.geo, num_view
    );

    auto idx_i = torch::from_blob(rows.data(), {static_cast<int64_t>(rows.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
    auto idx_j = torch::from_blob(cols.data(), {static_cast<int64_t>(cols.size())}, torch::TensorOptions().dtype(torch::kInt64)).clone();
    auto vals = torch::from_blob(values.data(), {static_cast<int64_t>(values.size())}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
    auto indices = torch::stack({idx_i, idx_j}).view({2, -1});
    int64_t matrix_rows = static_cast<int64_t>(num_view) * packed.Nv * packed.Nu;
    int64_t matrix_cols = static_cast<int64_t>(packed.D) * packed.H * packed.W;
    return torch::sparse_coo_tensor(indices, vals, {matrix_rows, matrix_cols}).coalesce();
}

torch::Tensor forward_projection3d(torch::Tensor image, torch::Tensor options, torch::Tensor ang)
{
    CHECK_INPUT(image, options, ang);
    TORCH_CHECK(image.scalar_type() == torch::kFloat32, "image must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack3d(options);
    TORCH_CHECK(image.dim() == 5, "Expected image shape [B, C, D, H, W]");
    TORCH_CHECK(image.size(2) == packed.D && image.size(3) == packed.H && image.size(4) == packed.W,
                "Image shape does not match 3D geometry");
    int B = static_cast<int>(image.size(0));
    int C = static_cast<int>(image.size(1));
    int V = static_cast<int>(ang.size(0));
    auto proj = torch::empty({B, C, V, packed.Nv, packed.Nu}, image.options());
    if (image.is_cuda()) {
        xrecon::forward3d_cuda(image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    } else {
        xrecon::forward3d_cpu(image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    }
    return proj;
}

torch::Tensor projection_transpose3d(torch::Tensor proj, torch::Tensor options, torch::Tensor ang)
{
    CHECK_INPUT(proj, options, ang);
    TORCH_CHECK(proj.scalar_type() == torch::kFloat32, "proj must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack3d(options);
    TORCH_CHECK(proj.dim() == 5, "Expected projection shape [B, C, V, Nv, Nu]");
    int B = static_cast<int>(proj.size(0));
    int C = static_cast<int>(proj.size(1));
    int V = static_cast<int>(ang.size(0));
    auto image = torch::zeros({B, C, packed.D, packed.H, packed.W}, proj.options());
    if (proj.is_cuda()) {
        xrecon::forward3d_t_cuda(image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    } else {
        xrecon::forward3d_t_cpu(image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    }
    return image;
}

torch::Tensor backward_projection3d(torch::Tensor proj, torch::Tensor options, torch::Tensor ang)
{
    CHECK_INPUT(proj, options, ang);
    TORCH_CHECK(proj.scalar_type() == torch::kFloat32, "proj must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack3d(options);
    int B = static_cast<int>(proj.size(0));
    int C = static_cast<int>(proj.size(1));
    int V = static_cast<int>(ang.size(0));
    auto image = torch::zeros({B, C, packed.D, packed.H, packed.W}, proj.options());
    if (proj.is_cuda()) {
        xrecon::backward3d_cuda(image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    } else {
        xrecon::backward3d_cpu(image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    }
    return image;
}

torch::Tensor weighted_backward_projection3d(torch::Tensor proj, torch::Tensor options, torch::Tensor ang)
{
    CHECK_INPUT(proj, options, ang);
    TORCH_CHECK(proj.scalar_type() == torch::kFloat32, "proj must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack3d(options);
    int B = static_cast<int>(proj.size(0));
    int C = static_cast<int>(proj.size(1));
    int V = static_cast<int>(ang.size(0));
    auto image = torch::zeros({B, C, packed.D, packed.H, packed.W}, proj.options());
    if (proj.is_cuda()) {
        xrecon::weighted_backward3d_cuda(image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    } else {
        xrecon::weighted_backward3d_cpu(image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    }
    return image;
}

torch::Tensor backward_projection_transpose3d(torch::Tensor image, torch::Tensor options, torch::Tensor ang)
{
    CHECK_INPUT(image, options, ang);
    TORCH_CHECK(image.scalar_type() == torch::kFloat32, "image must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack3d(options);
    int B = static_cast<int>(image.size(0));
    int C = static_cast<int>(image.size(1));
    int V = static_cast<int>(ang.size(0));
    auto proj = torch::empty({B, C, V, packed.Nv, packed.Nu}, image.options());
    if (image.is_cuda()) {
        xrecon::backward3d_t_cuda(proj.data_ptr<float>(), image.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    } else {
        xrecon::backward3d_t_cpu(proj.data_ptr<float>(), image.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    }
    return proj;
}

torch::Tensor weighted_backward_projection_transpose3d(torch::Tensor image, torch::Tensor options, torch::Tensor ang)
{
    CHECK_INPUT(image, options, ang);
    TORCH_CHECK(image.scalar_type() == torch::kFloat32, "image must be float32");
    TORCH_CHECK(options.scalar_type() == torch::kFloat32, "options must be float32");
    TORCH_CHECK(ang.scalar_type() == torch::kFloat32, "ang must be float32");
    auto packed = unpack3d(options);
    int B = static_cast<int>(image.size(0));
    int C = static_cast<int>(image.size(1));
    int V = static_cast<int>(ang.size(0));
    auto proj = torch::empty({B, C, V, packed.Nv, packed.Nu}, image.options());
    if (image.is_cuda()) {
        xrecon::weighted_backward3d_t_cuda(proj.data_ptr<float>(), image.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    } else {
        xrecon::weighted_backward3d_t_cpu(proj.data_ptr<float>(), image.data_ptr<float>(), ang.data_ptr<float>(), packed.vol, packed.geo, B * C, V);
    }
    return proj;
}

torch::Tensor forward_projection_helical3d(torch::Tensor image, torch::Tensor options, torch::Tensor ang, torch::Tensor source_pos)
{
    CHECK_INPUT(image, options, ang);
    CHECK_INPUT(source_pos, options, ang);
    TORCH_CHECK(image.is_cuda(), "Helical 3D projection currently supports CUDA tensors only");
    TORCH_CHECK(image.scalar_type() == torch::kFloat32 && options.scalar_type() == torch::kFloat32 &&
                ang.scalar_type() == torch::kFloat32 && source_pos.scalar_type() == torch::kFloat32,
                "Helical 3D tensors must be float32");
    auto packed = unpack3d(options);
    int B = static_cast<int>(image.size(0));
    int C = static_cast<int>(image.size(1));
    int V = static_cast<int>(ang.size(0));
    auto proj = torch::empty({B, C, V, packed.Nv, packed.Nu}, image.options());
    launch_helical_arc3d(helical_arc_forward_cuda, image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), source_pos, B * C, V, packed);
    return proj;
}

torch::Tensor projection_transpose_helical3d(torch::Tensor proj, torch::Tensor options, torch::Tensor ang, torch::Tensor source_pos)
{
    CHECK_INPUT(proj, options, ang);
    CHECK_INPUT(source_pos, options, ang);
    TORCH_CHECK(proj.is_cuda(), "Helical 3D projection transpose currently supports CUDA tensors only");
    auto packed = unpack3d(options);
    int B = static_cast<int>(proj.size(0));
    int C = static_cast<int>(proj.size(1));
    int V = static_cast<int>(ang.size(0));
    auto image = torch::zeros({B, C, packed.D, packed.H, packed.W}, proj.options());
    launch_helical_arc3d(helical_arc_forward_t_cuda, image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), source_pos, B * C, V, packed);
    return image;
}

torch::Tensor backward_projection_helical3d(torch::Tensor proj, torch::Tensor options, torch::Tensor ang, torch::Tensor source_pos)
{
    CHECK_INPUT(proj, options, ang);
    CHECK_INPUT(source_pos, options, ang);
    TORCH_CHECK(proj.is_cuda(), "Helical 3D backprojection currently supports CUDA tensors only");
    auto packed = unpack3d(options);
    int B = static_cast<int>(proj.size(0));
    int C = static_cast<int>(proj.size(1));
    int V = static_cast<int>(ang.size(0));
    auto image = torch::zeros({B, C, packed.D, packed.H, packed.W}, proj.options());
    launch_helical_arc3d(helical_arc_backward_cuda, image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), source_pos, B * C, V, packed);
    return image;
}

torch::Tensor weighted_backward_projection_helical3d(torch::Tensor proj, torch::Tensor options, torch::Tensor ang, torch::Tensor source_pos)
{
    CHECK_INPUT(proj, options, ang);
    CHECK_INPUT(source_pos, options, ang);
    TORCH_CHECK(proj.is_cuda(), "Helical 3D weighted backprojection currently supports CUDA tensors only");
    auto packed = unpack3d(options);
    int B = static_cast<int>(proj.size(0));
    int C = static_cast<int>(proj.size(1));
    int V = static_cast<int>(ang.size(0));
    auto image = torch::zeros({B, C, packed.D, packed.H, packed.W}, proj.options());
    launch_helical_arc3d(helical_arc_weighted_backward_cuda, image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), source_pos, B * C, V, packed);
    return image;
}

torch::Tensor backward_projection_transpose_helical3d(torch::Tensor image, torch::Tensor options, torch::Tensor ang, torch::Tensor source_pos)
{
    CHECK_INPUT(image, options, ang);
    CHECK_INPUT(source_pos, options, ang);
    TORCH_CHECK(image.is_cuda(), "Helical 3D backprojection transpose currently supports CUDA tensors only");
    auto packed = unpack3d(options);
    int B = static_cast<int>(image.size(0));
    int C = static_cast<int>(image.size(1));
    int V = static_cast<int>(ang.size(0));
    auto proj = torch::empty({B, C, V, packed.Nv, packed.Nu}, image.options());
    launch_helical_arc3d(helical_arc_backward_t_cuda, image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), source_pos, B * C, V, packed);
    return proj;
}

torch::Tensor weighted_backward_projection_transpose_helical3d(torch::Tensor image, torch::Tensor options, torch::Tensor ang, torch::Tensor source_pos)
{
    CHECK_INPUT(image, options, ang);
    CHECK_INPUT(source_pos, options, ang);
    TORCH_CHECK(image.is_cuda(), "Helical 3D weighted backprojection transpose currently supports CUDA tensors only");
    auto packed = unpack3d(options);
    int B = static_cast<int>(image.size(0));
    int C = static_cast<int>(image.size(1));
    int V = static_cast<int>(ang.size(0));
    auto proj = torch::zeros({B, C, V, packed.Nv, packed.Nu}, image.options());
    launch_helical_arc3d(helical_arc_weighted_backward_t_cuda, image.data_ptr<float>(), proj.data_ptr<float>(), ang.data_ptr<float>(), source_pos, B * C, V, packed);
    return proj;
}

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_projection2d", &xrecon_torch::forward_projection2d, "2D forward projection");
    m.def("backward_projection2d", &xrecon_torch::backward_projection2d, "2D backward projection");
    m.def("weighted_backward_projection2d", &xrecon_torch::weighted_backward_projection2d, "2D weighted backward projection");
    m.def("projection_transpose2d", &xrecon_torch::projection_transpose2d, "2D projection transpose");
    m.def("backward_projection_transpose2d", &xrecon_torch::backward_projection_transpose2d, "2D backward projection transpose");
    m.def("weighted_backward_projection_transpose2d", &xrecon_torch::weighted_backward_projection_transpose2d, "2D weighted backward projection transpose");
    m.def("system_matrix2d", &xrecon_torch::system_matrix2d, "2D distance-driven projection matrix");
    m.def("back_projection2d", &xrecon_torch::backward_projection2d, "Alias for backward_projection2d");
    m.def("weighted_back_projection2d", &xrecon_torch::weighted_backward_projection2d, "Alias for weighted_backward_projection2d");
    m.def("forward_projection3d", &xrecon_torch::forward_projection3d, "3D cone-beam forward projection");
    m.def("system_matrix3d", &xrecon_torch::system_matrix3d, "3D distance-driven projection matrix");
    m.def("projection_transpose3d", &xrecon_torch::projection_transpose3d, "3D cone-beam projection transpose");
    m.def("backward_projection3d", &xrecon_torch::backward_projection3d, "3D cone-beam backprojection");
    m.def("weighted_backward_projection3d", &xrecon_torch::weighted_backward_projection3d, "3D cone-beam weighted backprojection");
    m.def("backward_projection_transpose3d", &xrecon_torch::backward_projection_transpose3d, "3D cone-beam backprojection transpose");
    m.def("weighted_backward_projection_transpose3d", &xrecon_torch::weighted_backward_projection_transpose3d, "3D cone-beam weighted backprojection transpose");
    m.def("forward_projection_helical3d", &xrecon_torch::forward_projection_helical3d, "3D helical forward projection");
    m.def("projection_transpose_helical3d", &xrecon_torch::projection_transpose_helical3d, "3D helical projection transpose");
    m.def("backward_projection_helical3d", &xrecon_torch::backward_projection_helical3d, "3D helical backprojection");
    m.def("weighted_backward_projection_helical3d", &xrecon_torch::weighted_backward_projection_helical3d, "3D helical weighted backprojection");
    m.def("backward_projection_transpose_helical3d", &xrecon_torch::backward_projection_transpose_helical3d, "3D helical backprojection transpose");
    m.def("weighted_backward_projection_transpose_helical3d", &xrecon_torch::weighted_backward_projection_transpose_helical3d, "3D helical weighted backprojection transpose");
}
