#define EIGEN_USE_THREADS
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

int main() {  // 256s 多线程Matrix
  Eigen::setNbThreads(8);
  std::cout << Eigen::nbThreads() << "\n";
  Eigen::ThreadPool pool(8);
  Eigen::ThreadPoolDevice dev(&pool, 8);

  int lx = LX;
  int ly = LY;
  int lz = LZ;
  TensorType* _tensor_full = new TensorType[LX * LY * LZ];
  Eigen::TensorMap<Eigen::Tensor<TensorType, 3, 1>> tensor_full(_tensor_full,
                                                                LX, LY, LZ);
  TensorType* _tensor_no_full_e = new TensorType[LX * LY * LZ];
  Eigen::TensorMap<Eigen::Tensor<TensorType, 3, 1>> tensor_no_full_e(
      _tensor_no_full_e, LX, LY, LZ);
  TensorType* _tensor_no_full_w = new TensorType[LX * LY * LZ];
  Eigen::TensorMap<Eigen::Tensor<TensorType, 3, 1>> tensor_no_full_w(
      _tensor_no_full_w, LX, LY, LZ);
  TensorType* _tensor_no_full_s = new TensorType[LX * LY * LZ];
  Eigen::TensorMap<Eigen::Tensor<TensorType, 3, 1>> tensor_no_full_s(
      _tensor_no_full_s, LX, LY, LZ);
  TensorType* _tensor_no_full_n = new TensorType[LX * LY * LZ];
  Eigen::TensorMap<Eigen::Tensor<TensorType, 3, 1>> tensor_no_full_n(
      _tensor_no_full_n, LX, LY, LZ);
  TensorType* _tensor_no_full_t = new TensorType[LX * LY * LZ];
  Eigen::TensorMap<Eigen::Tensor<TensorType, 3, 1>> tensor_no_full_t(
      _tensor_no_full_t, LX, LY, LZ);
  TensorType* _tensor_no_full_b = new TensorType[LX * LY * LZ];
  Eigen::TensorMap<Eigen::Tensor<TensorType, 3, 1>> tensor_no_full_b(
      _tensor_no_full_b, LX, LY, LZ);
  tensor_full.setRandom().eval();
  tensor_no_full_e.setRandom().eval();
  tensor_no_full_w.setRandom().eval();
  tensor_no_full_s.setRandom().eval();
  tensor_no_full_n.setRandom().eval();
  tensor_no_full_t.setRandom().eval();
  tensor_no_full_b.setRandom().eval();

  clock_t start, end;
  start = clock();
  std::cout << _tensor_full[(555 * ly + 95) * lz + 36] << "\t";
  // #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < 1000; i++) {
    for (int x = 0; x < LX; x++) {
      for (int y = 0; y < LY; y++) {
        for (int z = 0; z < LZ; z++) {
          int index = (x * ly + y) * lz + z;
          int k = _tensor_full[index] =
              _tensor_no_full_e[index] * 0.1 + _tensor_no_full_w[index] * 0.12 +
              _tensor_no_full_s[index] * 0.23 +
              _tensor_no_full_n[index] * 0.21 +
              _tensor_no_full_t[index] * 0.19 + _tensor_no_full_b[index] * 0.15;
          _tensor_no_full_e[index] = k * 0.1 + _tensor_no_full_e[index] * 0.9;
          _tensor_no_full_w[index] = k * 0.12 + _tensor_no_full_w[index] * 0.88;
          _tensor_no_full_s[index] = k * 0.23 + _tensor_no_full_s[index] * 0.77;
          _tensor_no_full_n[index] = k * 0.21 + _tensor_no_full_n[index] * 0.79;
          _tensor_no_full_t[index] = k * 0.19 + _tensor_no_full_t[index] * 0.81;
          _tensor_no_full_b[index] = k * 0.15 + _tensor_no_full_b[index] * 0.85;
        }
      }
    }
    // std::cout << i << "\n";
  }

  std::cout << _tensor_full[(555 * ly + 95) * lz + 36] << "\t";

  end = clock();  // 结束时间
  std::cout << "耗时 = " << double(end - start) / CLOCKS_PER_SEC << "s"
            << std::endl;  // 输出时间（单位：ｓ）

  return 0;
}

// int main() {  // 256s 多线程Matrix
//   Eigen::setNbThreads(8);
//   std::cout << Eigen::nbThreads() << "\n";
//   Eigen::ThreadPool pool(8);
//   Eigen::ThreadPoolDevice dev(&pool, 8);
//
//   int lx = LX;
//   int ly = LY;
//   int lz = LZ;
//   typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Layer;
//   typedef Eigen::Matrix<Layer, LZ, 1> MutiLayer;
//   MutiLayer tensor_full;
//   MutiLayer tensor_no_full_e;
//   MutiLayer tensor_no_full_w;
//   MutiLayer tensor_no_full_s;
//   MutiLayer tensor_no_full_n;
//   MutiLayer tensor_no_full_t;
//   MutiLayer tensor_no_full_b;
//   MutiLayer tensor_e;
//   MutiLayer tensor_w;
//   MutiLayer tensor_s;
//   MutiLayer tensor_n;
//   MutiLayer tensor_t;
//   MutiLayer tensor_b;
//   MutiLayer temp1;
//   MutiLayer temp2;
//
//   for (int i = 0; i < LZ; i++) {
//     tensor_full(i) = Layer::Random(lx, ly);
//     tensor_no_full_e(i) = Layer::Random(lx, ly);
//     tensor_no_full_w(i) = Layer::Random(lx, ly);
//     tensor_no_full_s(i) = Layer::Random(lx, ly);
//     tensor_no_full_n(i) = Layer::Random(lx, ly);
//     tensor_no_full_t(i) = Layer::Random(lx, ly);
//     tensor_no_full_b(i) = Layer::Random(lx, ly);
//     tensor_e(i) = Layer::Random(lx, ly);
//     tensor_w(i) = Layer::Random(lx, ly);
//     tensor_s(i) = Layer::Random(lx, ly);
//     tensor_n(i) = Layer::Random(lx, ly);
//     tensor_t(i) = Layer::Random(lx, ly);
//     tensor_b(i) = Layer::Random(lx, ly);
//   }
//
//   clock_t start, end;
//   start = clock();
//   std::cout << tensor_full(36)(555, 95) << "\t";
//
//   for (int i = 0; i < 100; i++) {
// #pragma omp parallel for schedule(dynamic)
//     for (int j = 0; j < LZ; j++) {
//       tensor_full(j) = tensor_no_full_e(j) * 0.1 + tensor_no_full_w(j) * 0.12
//       +
//                        tensor_no_full_s(j) * 0.23 + tensor_no_full_n(j) *
//                        0.21 + tensor_no_full_t(j) * 0.19 +
//                        tensor_no_full_b(j) * 0.15;
//     }
//
// #pragma omp parallel for schedule(dynamic)
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_e(j) = tensor_full(j) * 0.1 + tensor_no_full_e(j) * 0.9;
//     }
//
// #pragma omp parallel for schedule(dynamic)
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_w(j) = tensor_full(j) * 0.12 + tensor_no_full_w(j) *
//       0.88;
//     }
//
// #pragma omp parallel for schedule(dynamic)
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_s(j) = tensor_full(j) * 0.23 + tensor_no_full_s(j) *
//       0.77;
//     }
//
// #pragma omp parallel for schedule(dynamic)
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_n(j) = tensor_full(j) * 0.21 + tensor_no_full_n(j) *
//       0.79;
//     }
//
// #pragma omp parallel for schedule(dynamic)
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_t(j) = tensor_full(j) * 0.19 + tensor_no_full_t(j) *
//       0.81;
//     }
//
// #pragma omp parallel for schedule(dynamic)
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_b(j) = tensor_full(j) * 0.15 + tensor_no_full_b(j) *
//       0.85;
//     }
//     // std::cout << i << "\n";
//   }
//
//   std::cout << tensor_full(36)(555, 95) << "\t";
//
//   end = clock();  // 结束时间
//   std::cout << "耗时 = " << double(end - start) / CLOCKS_PER_SEC << "s"
//             << std::endl;  // 输出时间（单位：ｓ）
//   return 0;
// }

// int main() {  // 256s 单线程Matrix
//   Eigen::setNbThreads(8);
//   std::cout << Eigen::nbThreads() << "\n";
//   Eigen::ThreadPool pool(8);
//   Eigen::ThreadPoolDevice dev(&pool, 8);
//
//   int lx = LX;
//   int ly = LY;
//   int lz = LZ;
//   typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Layer;
//   typedef Eigen::Matrix<Layer, LZ, 1> MutiLayer;
//   MutiLayer tensor_full;
//   MutiLayer tensor_no_full_e;
//   MutiLayer tensor_no_full_w;
//   MutiLayer tensor_no_full_s;
//   MutiLayer tensor_no_full_n;
//   MutiLayer tensor_no_full_t;
//   MutiLayer tensor_no_full_b;
//   MutiLayer tensor_e;
//   MutiLayer tensor_w;
//   MutiLayer tensor_s;
//   MutiLayer tensor_n;
//   MutiLayer tensor_t;
//   MutiLayer tensor_b;
//
//   for (int i = 0; i < LZ; i++) {
//     tensor_full(i) = Layer::Random(lx, ly);
//     tensor_no_full_e(i) = Layer::Random(lx, ly);
//     tensor_no_full_w(i) = Layer::Random(lx, ly);
//     tensor_no_full_s(i) = Layer::Random(lx, ly);
//     tensor_no_full_n(i) = Layer::Random(lx, ly);
//     tensor_no_full_t(i) = Layer::Random(lx, ly);
//     tensor_no_full_b(i) = Layer::Random(lx, ly);
//     tensor_e(i) = Layer::Random(lx, ly);
//     tensor_w(i) = Layer::Random(lx, ly);
//     tensor_s(i) = Layer::Random(lx, ly);
//     tensor_n(i) = Layer::Random(lx, ly);
//     tensor_t(i) = Layer::Random(lx, ly);
//     tensor_b(i) = Layer::Random(lx, ly);
//   }
//
//   clock_t start, end;
//   start = clock();
//   std::cout << tensor_full(36)(555, 95) << "\t";
//
//   for (int i = 0; i < 1000; i++) {
//     for (int j = 0; j < LZ; j++) {
//       tensor_full(j) = tensor_no_full_e(j) * 0.1 + tensor_no_full_w(j) * 0.12
//       +
//                        tensor_no_full_s(j) * 0.23 + tensor_no_full_n(j) *
//                        0.21 + tensor_no_full_t(j) * 0.19 +
//                        tensor_no_full_b(j) * 0.15;
//     }
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_e(j) = tensor_full(j) * 0.1 + tensor_no_full_e(j) * 0.9;
//     }
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_w(j) = tensor_full(j) * 0.12 + tensor_no_full_w(j) *
//       0.88;
//     }
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_s(j) = tensor_full(j) * 0.23 + tensor_no_full_s(j) *
//       0.77;
//     }
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_n(j) = tensor_full(j) * 0.21 + tensor_no_full_n(j) *
//       0.79;
//     }
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_t(j) = tensor_full(j) * 0.19 + tensor_no_full_t(j) *
//       0.81;
//     }
//     for (int j = 0; j < LZ; j++) {
//       tensor_no_full_b(j) = tensor_full(j) * 0.15 + tensor_no_full_b(j) *
//       0.85;
//     }
//     // std::cout << i << "\n";
//   }
//
//   std::cout << tensor_full(36)(555, 95) << "\t";
//
//   end = clock();  // 结束时间
//   std::cout << "耗时 = " << double(end - start) / CLOCKS_PER_SEC << "s"
//             << std::endl;  // 输出时间（单位：ｓ）
//   return 0;
// }

// int main() {  //230.7s 单线程tensor
//   Eigen::Tensor<TensorType, 3> tensor_full(LX, LY, LZ);
//   Eigen::Tensor<TensorType, 3> tensor_no_full_e(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_no_full_w(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_no_full_s(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_no_full_n(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_no_full_t(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_no_full_b(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_e(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_w(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_s(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_n(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_t(LX2, LY2, LZ2);
//   Eigen::Tensor<TensorType, 3> tensor_b(LX2, LY2, LZ2);
//   tensor_full.setRandom().eval();
//   tensor_no_full_e.setRandom().eval();
//   tensor_no_full_w.setRandom().eval();
//   tensor_no_full_s.setRandom().eval();
//   tensor_no_full_n.setRandom().eval();
//   tensor_no_full_t.setRandom().eval();
//   tensor_no_full_b.setRandom().eval();
//
//   // tensor_e.setRandom().eval();
//   // tensor_w.setRandom().eval();
//   // tensor_s.setRandom().eval();
//   // tensor_n.setRandom().eval();
//   // tensor_t.setRandom().eval();
//   // tensor_b.setRandom().eval();
//   tensor_e.constant(0.1).eval();
//   tensor_w.constant(0.12).eval();
//   tensor_s.constant(0.23).eval();
//   tensor_n.constant(0.21).eval();
//   tensor_t.constant(0.19).eval();
//   tensor_b.constant(0.15).eval();
//
//   clock_t start, end;
//   start = clock();
//   std::cout << tensor_full(555, 95, 36) << "\t";
//
//   // Eigen::array<Eigen::Index, 3> offsets_e = {2, 1, 1};
//   // Eigen::array<Eigen::Index, 3> offsets_w = {0, 1, 1};
//   // Eigen::array<Eigen::Index, 3> offsets_s = {1, 2, 1};
//   // Eigen::array<Eigen::Index, 3> offsets_n = {1, 0, 1};
//   // Eigen::array<Eigen::Index, 3> offsets_t = {1, 1, 0};
//   // Eigen::array<Eigen::Index, 3> offsets_b = {1, 1, 2};
//   // Eigen::array<Eigen::Index, 3> extents = {LX2, LY2, LZ2};
//   // tensor_no_full_e = tensor_full.slice(offsets_e, extents);
//   // tensor_no_full_w = tensor_full.slice(offsets_w, extents);
//   // tensor_no_full_s = tensor_full.slice(offsets_s, extents);
//   // tensor_no_full_n = tensor_full.slice(offsets_n, extents);
//   // tensor_no_full_t = tensor_full.slice(offsets_t, extents);
//   // tensor_no_full_b = tensor_full.slice(offsets_b, extents);
//
//   for (int i = 0; i < 1000; i++) {
//     auto tensor_temp = tensor_no_full_e * tensor_e.constant(0.1) +
//                        tensor_no_full_w * tensor_w.constant(0.12) +
//                        tensor_no_full_s * tensor_s.constant(0.23) +
//                        tensor_no_full_n * tensor_n.constant(0.21) +
//                        tensor_no_full_t * tensor_t.constant(0.19) +
//                        tensor_no_full_b * tensor_b.constant(0.15);
//     tensor_full = tensor_temp;
//     tensor_no_full_e = tensor_full * tensor_e.constant(0.1) +
//                        tensor_no_full_e * tensor_e.constant(0.9);
//     tensor_no_full_w = tensor_full * tensor_w.constant(0.12) +
//                        tensor_no_full_w * tensor_w.constant(0.88);
//     tensor_no_full_s = tensor_full * tensor_s.constant(0.23) +
//                        tensor_no_full_s * tensor_s.constant(0.77);
//     tensor_no_full_n = tensor_full * tensor_n.constant(0.21) +
//                        tensor_no_full_n * tensor_n.constant(0.79);
//     tensor_no_full_t = tensor_full * tensor_t.constant(0.19) +
//                        tensor_no_full_t * tensor_t.constant(0.81);
//     tensor_no_full_b = tensor_full * tensor_b.constant(0.15) +
//                        tensor_no_full_b * tensor_b.constant(0.85);
//     // std::cout << i << "\n";
//   }
//
//   std::cout << tensor_full(555, 95, 36) << "\t";
//
//   end = clock();  // 结束时间
//   std::cout << "耗时 = " << double(end - start) / CLOCKS_PER_SEC << "s"
//             << std::endl;  // 输出时间（单位：ｓ）
//   return 0;
// }