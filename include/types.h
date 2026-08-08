#pragma once

#include <cuda_runtime.h>

namespace xrecon {

    enum class GeometryType : int {
        Parallel = 0,
        FanFlat  = 1,
        FanArc   = 2
    };

    struct Point2d {
        float x, y;

        __host__ __device__ Point2d(float _x = 0.0f, float _y = 0.0f) : x(_x), y(_y) {}

        __host__ __device__ Point2d operator+(const Point2d& other) const {
            return Point2d(x + other.x, y + other.y);
        }
        __host__ __device__ Point2d operator-(const Point2d& other) const {
            return Point2d(x - other.x, y - other.y);
        }
        __host__ __device__ Point2d operator*(float scalar) const {
            return Point2d(x * scalar, y * scalar);
        }
        __host__ __device__ Point2d operator/(float scalar) const {
            return Point2d(x / scalar, y / scalar);
        }
        __host__ __device__ float dot(const Point2d& other) const {
            return x * other.x + y * other.y;
        }
    };

    struct __align__(16) Volume2d {
        int Nx;
        int Ny;
        float dx;
        float dy;
        float shift_x;
        float shift_y;

        __host__ __device__ Volume2d(int _Nx = 0, int _Ny = 0, float _dx = 1.0f, float _dy = 1.0f,
                                     float _shift_x = 0.0f, float _shift_y = 0.0f)
            : Nx(_Nx), Ny(_Ny), dx(_dx), dy(_dy), shift_x(_shift_x), shift_y(_shift_y) {}

    };

    struct Geometry2d {
        GeometryType type;
        int Nu;
        float du;
        float shift_u;
        float SOD; 
        float SDD; 

        __host__ __device__ Geometry2d(GeometryType _type = GeometryType::Parallel, int _Nu = 0, float _du = 1.0f,
                                       float _shift_u = 0.0f, float _SOD = 0.0f, float _SDD = 0.0f)
            : type(_type), Nu(_Nu), du(_du), shift_u(_shift_u), SOD(_SOD), SDD(_SDD) {}
    };

    struct __align__(16) Volume3d {
        int Nx;
        int Ny;
        int Nz;
        float dx;
        float dy;
        float dz;
        float shift_x;
        float shift_y;
        float shift_z;

        __host__ __device__ Volume3d(
            int _Nx = 0, int _Ny = 0, int _Nz = 0,
            float _dx = 1.0f, float _dy = 1.0f, float _dz = 1.0f,
            float _shift_x = 0.0f, float _shift_y = 0.0f, float _shift_z = 0.0f)
            : Nx(_Nx), Ny(_Ny), Nz(_Nz),
              dx(_dx), dy(_dy), dz(_dz),
              shift_x(_shift_x), shift_y(_shift_y), shift_z(_shift_z) {}
    };

    struct Geometry3d {
        GeometryType type;
        int Nu;
        int Nv;
        float du;
        float dv;
        float shift_u;
        float shift_v;
        float SOD;
        float SDD;
        int helical_view_mode;
        int helical_weight_mode;
        float helical_view_turns;
        float helical_halfscan_coef;
        float helical_tang_k;
        float helical_tang_blend;

        __host__ __device__ Geometry3d(
            GeometryType _type = GeometryType::FanFlat,
            int _Nu = 0, int _Nv = 0,
            float _du = 1.0f, float _dv = 1.0f,
            float _shift_u = 0.0f, float _shift_v = 0.0f,
            float _SOD = 0.0f, float _SDD = 0.0f,
            int _helical_view_mode = 2, int _helical_weight_mode = 0,
            float _helical_view_turns = 0.0f,
            float _helical_halfscan_coef = 1.0f,
            float _helical_tang_k = 5.0f,
            float _helical_tang_blend = 0.0f)
            : type(_type), Nu(_Nu), Nv(_Nv),
              du(_du), dv(_dv), shift_u(_shift_u), shift_v(_shift_v),
              SOD(_SOD), SDD(_SDD),
              helical_view_mode(_helical_view_mode),
              helical_weight_mode(_helical_weight_mode),
              helical_view_turns(_helical_view_turns),
              helical_halfscan_coef(_helical_halfscan_coef),
              helical_tang_k(_helical_tang_k),
              helical_tang_blend(_helical_tang_blend) {}
    };

}
