#include "lbvh.cuh"
#include <random>
#include <vector>
#include <thrust/random.h>
#include <torch/extension.h>

struct bounding_box {
    float x_min;
    float x_max;
    float y_min;
    float y_max;
    float z_min;
    float z_max;
};

struct aabb_getter
{
    __device__
    lbvh::aabb<float> operator()(const struct bounding_box f) const noexcept
    {
        lbvh::aabb<float> retval;
        retval.upper.x = f.x_max;
        retval.upper.y = f.y_max;
        retval.upper.z = f.z_max;
        retval.lower.x = f.x_min;
        retval.lower.y = f.y_min;
        retval.lower.z = f.z_min;
        return retval;
    }
};
// struct distance_calculator
// {
//     __device__
//     float operator()(const float4 point, const float4 object) const noexcept
//     {
//         return (point.x - object.x) * (point.x - object.x) +
//                (point.y - object.y) * (point.y - object.y) +
//                (point.z - object.z) * (point.z - object.z);
//     }
// };

std::tuple<torch::Tensor, torch::Tensor>
BuildBVH (
    const torch::Tensor& objects // Shape (N, 3, 2)
) {
    size_t N = objects.size(0);
    auto int_opts = objects.options().dtype(torch::kInt32);
    auto float_opts = objects.options().dtype(torch::kFloat32);

    bounding_box *object_bbs = (bounding_box *)objects.contiguous().data<float>();
    std::vector<bounding_box> objects_vec(object_bbs, object_bbs + N);

    lbvh::bvh<float, bounding_box, aabb_getter> bvh(objects_vec.begin(), objects_vec.end(), true);

    auto nodes = bvh.nodes_host();
    auto aabbs = bvh.aabbs_host();

    torch::Tensor bvh_nodes = torch::from_blob(nodes.data(), {(long)nodes.size(), 4}, torch::kInt32).clone();
    torch::Tensor bvh_aabbs = torch::from_blob(aabbs.data(), {(long)aabbs.size(), 2, 4}, torch::kFloat32).clone();
    bvh_aabbs = bvh_aabbs.narrow(2, 0, 3);
    bvh_aabbs = bvh_aabbs.permute({0, 2, 1});

    return std::make_tuple(bvh_nodes, bvh_aabbs);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("BuildBVH", &BuildBVH, "Build BVH (CUDA)");
}
