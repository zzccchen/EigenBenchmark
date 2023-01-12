#include <time.h>

#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#define TensorType double
#define LX         2002
#define LY         402
#define LZ         42
#define LX2        2002
#define LY2        402
#define LZ2        42

int main() {
  Eigen::Tensor<TensorType, 3> tensor_full(LX, LY, LZ);
  Eigen::Tensor<TensorType, 3> tensor_no_full_e(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_no_full_w(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_no_full_s(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_no_full_n(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_no_full_t(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_no_full_b(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_e(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_w(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_s(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_n(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_t(LX2, LY2, LZ2);
  Eigen::Tensor<TensorType, 3> tensor_b(LX2, LY2, LZ2);
  tensor_full.setRandom().eval();
  tensor_no_full_e.setRandom().eval();
  tensor_no_full_w.setRandom().eval();
  tensor_no_full_s.setRandom().eval();
  tensor_no_full_n.setRandom().eval();
  tensor_no_full_t.setRandom().eval();
  tensor_no_full_b.setRandom().eval();

  // tensor_e.setRandom().eval();
  // tensor_w.setRandom().eval();
  // tensor_s.setRandom().eval();
  // tensor_n.setRandom().eval();
  // tensor_t.setRandom().eval();
  // tensor_b.setRandom().eval();
  tensor_e.constant(0.1).eval();
  tensor_w.constant(0.12).eval();
  tensor_s.constant(0.23).eval();
  tensor_n.constant(0.21).eval();
  tensor_t.constant(0.19).eval();
  tensor_b.constant(0.15).eval();

  clock_t start, end;
  start = clock();
  std::cout << tensor_full(555, 95, 36) << "\t";

  // Eigen::array<Eigen::Index, 3> offsets_e = {2, 1, 1};
  // Eigen::array<Eigen::Index, 3> offsets_w = {0, 1, 1};
  // Eigen::array<Eigen::Index, 3> offsets_s = {1, 2, 1};
  // Eigen::array<Eigen::Index, 3> offsets_n = {1, 0, 1};
  // Eigen::array<Eigen::Index, 3> offsets_t = {1, 1, 0};
  // Eigen::array<Eigen::Index, 3> offsets_b = {1, 1, 2};
  // Eigen::array<Eigen::Index, 3> extents = {LX2, LY2, LZ2};
  // tensor_no_full_e = tensor_full.slice(offsets_e, extents);
  // tensor_no_full_w = tensor_full.slice(offsets_w, extents);
  // tensor_no_full_s = tensor_full.slice(offsets_s, extents);
  // tensor_no_full_n = tensor_full.slice(offsets_n, extents);
  // tensor_no_full_t = tensor_full.slice(offsets_t, extents);
  // tensor_no_full_b = tensor_full.slice(offsets_b, extents);

  for (int i = 0; i < 1000; i++) {
    auto tensor_temp = tensor_no_full_e * tensor_e.constant(0.1) +
                       tensor_no_full_w * tensor_w.constant(0.12) +
                       tensor_no_full_s * tensor_s.constant(0.23) +
                       tensor_no_full_n * tensor_n.constant(0.21) +
                       tensor_no_full_t * tensor_t.constant(0.19) +
                       tensor_no_full_b * tensor_b.constant(0.15);
    tensor_full = tensor_temp;
    tensor_no_full_e = tensor_full * tensor_e.constant(0.1) +
                       tensor_no_full_e * tensor_e.constant(0.9);
    tensor_no_full_w = tensor_full * tensor_w.constant(0.12) +
                       tensor_no_full_w * tensor_w.constant(0.88);
    tensor_no_full_s = tensor_full * tensor_s.constant(0.23) +
                       tensor_no_full_s * tensor_s.constant(0.77);
    tensor_no_full_n = tensor_full * tensor_n.constant(0.21) +
                       tensor_no_full_n * tensor_n.constant(0.79);
    tensor_no_full_t = tensor_full * tensor_t.constant(0.19) +
                       tensor_no_full_t * tensor_t.constant(0.81);
    tensor_no_full_b = tensor_full * tensor_b.constant(0.15) +
                       tensor_no_full_b * tensor_b.constant(0.85);
    // std::cout << i << "\n";
  }

  std::cout << tensor_full(555, 95, 36) << "\t";

  end = clock();  // 结束时间
  std::cout << "耗时 = " << double(end - start) / CLOCKS_PER_SEC << "s"
            << std::endl;  // 输出时间（单位：ｓ）
  return 0;
}