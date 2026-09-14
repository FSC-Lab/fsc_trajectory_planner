// Parity of the flat B-spline backend against the Python
// flat_bspline_planner.py (fixture test/data/flat_plan_t650.txt, from
// fsc_PegasusSimulator's dump_flat_reference.py), through the registry.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/trajectory.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"

using namespace fsc_trajectory_planner;

namespace
{
std::vector<double> readRow(std::istream & in, int n)
{
  std::vector<double> v(n);
  for (int i = 0; i < n; ++i) {in >> v[i];}
  return v;
}
}  // namespace

TEST(FlatBSpline, ParityWithPython)
{
  std::ifstream in("data/flat_plan_t650.txt");
  ASSERT_TRUE(in.good()) << "run from the test/ directory";
  const auto vehicle = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PlanRequest req;
  req.rest0.x_b << 0.0, 0.0, 1.2;
  req.rest0.phi = 0.0;
  req.rest0.q << 0.0, 40.0 * M_PI / 180.0, 40.0 * M_PI / 180.0, 0.0;
  RestSpec r1;
  r1.x_b << 0.6, -0.35, 1.55;
  r1.phi = 25.0 * M_PI / 180.0;
  r1.q << 15.0 * M_PI / 180.0, 25.0 * M_PI / 180.0, 20.0 * M_PI / 180.0,
    35.0 * M_PI / 180.0;
  req.rest1 = r1;

  std::string tok;
  int ncases = 0;
  in >> tok >> ncases;
  for (int ci = 0; ci < ncases; ++ci) {
    std::string name;
    double T_py = 0.0;
    in >> tok >> name >> tok >> T_py;
    PlanOptions o;
    o.n_check = 101;
    if (name == "dilated") {
      o.v_max = o.a_max = o.w_max = o.dw_max = 9.0;
      o.tau_joint_max = 0.7625;
    }
    const auto traj = planner->plan(*vehicle, req, o);
    SCOPED_TRACE(name);
    EXPECT_NEAR(traj->duration(), T_py, 1e-9);
    std::cout << "  " << name << ": " << traj->diag().summary << "\n";
    for (int k = 0; k < 3; ++k) {
      std::string key;
      int rows = 0, cols = 0;
      in >> tok >> key >> rows >> cols;
      readRow(in, rows * cols);
    }
    int nsamp = 0;
    in >> tok >> nsamp;
    double worst = 0.0;
    for (int s = 0; s < nsamp; ++s) {
      const std::vector<double> row = readRow(in, 1 + 14 * 3 + 2 * 4);
      const WbReference r = traj->eval(row[0]);
      const double * got[16] = {
        r.x_cd.data(), r.x_cd_dot.data(), r.x_cd_ddot.data(),
        r.x_cd_d3.data(), r.x_cd_d4.data(), r.b1_d.data(),
        r.b1_d_dot.data(), r.b1_d_ddot.data(), r.r_ed.data(),
        r.r_ed_dot.data(), r.r_ed_ddot.data(), r.b1_de.data(),
        r.b1_de_dot.data(), r.b1_de_ddot.data(), r.q_d.data(),
        r.qdot_d.data()};
      int off = 1;
      for (int f = 0; f < 16; ++f) {
        const int n = (f >= 14) ? kNumJoints : 3;
        for (int j = 0; j < n; ++j) {
          worst = std::max(worst, std::abs(got[f][j] - row[off + j]));
        }
        off += n;
      }
    }
    EXPECT_LT(worst, 1e-9);
    EXPECT_LT(traj->diag().max_dyn_defect, 1e-9);
  }
  const auto t0 = std::chrono::steady_clock::now();
  PlanOptions o;
  for (int i = 0; i < 10; ++i) {planner->plan(*vehicle, req, o);}
  std::cout << "  bspline plan: "
            << std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count() / 10
            << " ms (python: ~45-190 ms)\n";
}
