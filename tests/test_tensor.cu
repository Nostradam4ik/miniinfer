#include "test_framework.h"
#include "tensor.cuh"

TEST(tensor_roundtrip_host_device_host) {
    std::vector<float> h = {1.f, -2.f, 3.5f, 0.f, 7.f, 8.f};
    Tensor t = Tensor::from_host(h, {2, 3});
    CHECK(t.ndim() == 2);
    CHECK(t.dim(0) == 2 && t.dim(1) == 3);
    CHECK(t.numel() == 6);
    CHECK(t.to_host() == h);
}

TEST(tensor_new_is_zeroed) {
    Tensor t({4, 5});
    for (float v : t.to_host()) CHECK(v == 0.f);
}

TEST(tensor_empty_is_valid) {
    Tensor t({0, 3});
    CHECK(t.numel() == 0);
    CHECK(t.data() == nullptr);
    CHECK(t.to_host().empty());
}

TEST(tensor_from_host_wrong_size_throws) {
    std::vector<float> h(5, 1.f);
    CHECK_THROWS(Tensor::from_host(h, {2, 3}));
}

TEST(tensor_negative_dim_throws) { CHECK_THROWS(Tensor({-1, 3})); }

TEST(tensor_move_transfers_ownership) {
    Tensor a = Tensor::from_host({1.f, 2.f, 3.f, 4.f}, {2, 2});
    const float* ptr = a.data();
    Tensor b = std::move(a);
    CHECK(b.data() == ptr);
    CHECK(a.data() == nullptr);  // NOLINT: після переміщення a порожній
    CHECK(b.to_host()[3] == 4.f);
}
