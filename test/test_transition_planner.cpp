// The straight-line planner against the Python transition_planner (fixture
// test/data/python_transition_t650.txt), plus the Python _selftest checks
// (endpoints on the holds, FD-consistent derivative chains, refusals) and a
// timing line.
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
RestSpec readRest(std::istream & in)
{
  RestSpec r;
  in >> r.x_b(0) >> r.x_b(1) >> r.x_b(2) >> r.phi >> r.q(0) >> r.q(1) >> r.q(2) >> r.q(3);
  return r;
}
}  // namespace

TEST(StraightLine, ParityWithPython)
{
  std::ifstream in("data/python_transition_t650.txt");
  ASSERT_TRUE(in.good()) << "run from the test/ directory";
  const auto vehicle = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("straight_line");
  std::string tok;
  int ncases = 0;
  in >> tok >> ncases;
  for (int c = 0; c < ncases; ++c) {
    std::string name;
    in >> tok >> name;
    PlanRequest req;
    in >> tok; req.rest0 = readRest(in);
    in >> tok; req.rest1 = readRest(in);
    double T_py = 0.0;
    in >> tok >> T_py;
    int it_py = 0;
    double defect_py, sig_py, vpk_py, end_py;
    in >> tok >> it_py >> defect_py >> sig_py >> vpk_py >> end_py;
    int nsamp = 0;
    in >> tok >> nsamp;

    const auto traj = planner->plan(*vehicle, req, PlanOptions{});
    SCOPED_TRACE(name);
    EXPECT_NEAR(traj->duration(), T_py, 1e-9);
    const PlanDiag & d = traj->diag();
    EXPECT_NEAR(d.min_sigma_nd, sig_py, 1e-6);
    EXPECT_NEAR(d.peak_com_speed, vpk_py, 1e-6);
    EXPECT_LT(d.max_dyn_defect, 2e-7);
    std::cout << "  " << name << ": " << d.summary << "  (python: T=" << T_py
              << ", " << it_py << " iters, defect " << defect_py << ")\n";

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
        // the prescribed channels are algebraic (parity to roundoff); the
        // solved CoM chain and the recovered q inherit the polynomial
        // least-squares noise of two different solvers
        const double tol = (f <= 4 || f >= 14) ? 5e-6 : 1e-9;
        for (int j = 0; j < n; ++j) {
          const double e = std::abs(got[f][j] - row[off + j]);
          worst = std::max(worst, e);
          EXPECT_LT(e, tol) << "field " << f << " comp " << j << " t=" << row[0];
        }
        off += n;
      }
    }
    std::cout << "  worst sample deviation vs python " << worst << "\n";
  }
}

TEST(StraightLine, EndpointsAndDerivativeChains)
{
  const auto vehicle = makeVehicleModel("t650_aerial_manipulator");
  const WholeBodyParams & P = vehicle->params;
  const auto planner = makePlanner("straight_line");
  PlanRequest req;
  req.rest0.x_b << 0.0, 0.0, 1.2;
  req.rest0.q = vehicle->home;
  RestSpec r1;
  r1.x_b << 0.4, 0.3, 1.4;
  r1.phi = 20.0 * M_PI / 180.0;
  r1.q << 15.0 * M_PI / 180.0, 30.0 * M_PI / 180.0, 35.0 * M_PI / 180.0,
    -20.0 * M_PI / 180.0;
  req.rest1 = r1;
  const auto traj = planner->plan(*vehicle, req, PlanOptions{});
  const double T = traj->duration();

  const RestSpec * rests[2] = {&req.rest0, &r1};
  const double ts[2] = {0.0, T};
  for (int e = 0; e < 2; ++e) {
    const WbReference want = restReference(P, *rests[e]);
    const WbReference got = traj->eval(ts[e]);
    EXPECT_LT((got.q_d - want.q_d).cwiseAbs().maxCoeff(), 1e-3);
    EXPECT_LT((got.r_ed - want.r_ed).norm(), 1e-9);
    EXPECT_LT((got.x_cd - want.x_cd).norm(), 2e-3);
    EXPECT_LT(got.x_cd_dot.norm(), 2e-3);
  }
  // FD consistency of every prescribed derivative chain (interior)
  const double h = 1e-5;
  double worst = 0.0;
  for (int k = 0; k < 9; ++k) {
    const double tq = 0.15 * T + (0.70 * T) * k / 8.0;
    const WbReference ra = traj->eval(tq - h), r0 = traj->eval(tq),
      rb = traj->eval(tq + h);
    auto chk = [&](const Vec3 & a, const Vec3 & b, const Vec3 & d) {
        worst = std::max(worst, (((b - a) / (2 * h)) - d).cwiseAbs().maxCoeff());
      };
    chk(ra.r_ed, rb.r_ed, r0.r_ed_dot);
    chk(ra.b1_de, rb.b1_de, r0.b1_de_dot);
    chk(ra.b1_d, rb.b1_d, r0.b1_d_dot);
    chk(ra.x_cd, rb.x_cd, r0.x_cd_dot);
    chk(ra.x_cd_dot, rb.x_cd_dot, r0.x_cd_ddot);
  }
  EXPECT_LT(worst, 1e-4);
  // the goal rest is what the next hold uses
  EXPECT_LT((traj->goalRest().q - r1.q).norm(), 1e-12);
}

TEST(StraightLine, RefusesWristSingularGoal)
{
  const auto vehicle = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("straight_line");
  PlanRequest req;
  req.rest0.x_b << 0.0, 0.0, 1.2;
  req.rest0.q = vehicle->home;
  RestSpec r1 = req.rest0;
  r1.q << 0.0, 0.02, 0.02, 0.0;
  req.rest1 = r1;
  EXPECT_THROW(planner->plan(*vehicle, req, PlanOptions{}), std::runtime_error);
}

TEST(StraightLine, Timing)
{
  const auto vehicle = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("straight_line");
  PlanRequest req;
  req.rest0.x_b << 0.0, 0.0, 1.2;
  req.rest0.q = vehicle->home;
  RestSpec r1;
  r1.x_b << 0.4, 0.3, 1.4;
  r1.phi = 20.0 * M_PI / 180.0;
  r1.q << 15.0 * M_PI / 180.0, 30.0 * M_PI / 180.0, 35.0 * M_PI / 180.0,
    -20.0 * M_PI / 180.0;
  req.rest1 = r1;
  planner->plan(*vehicle, req, PlanOptions{});
  const int n = 10;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) {planner->plan(*vehicle, req, PlanOptions{});}
  const double ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count() / n;
  std::cout << "  straight_line plan: " << ms << " ms (python: ~260 ms)\n";
  EXPECT_LT(ms, 200.0);
}

TEST(PlannerRegistry, NamesAndUnknown)
{
  const auto names = plannerNames();
  EXPECT_EQ(names.size(), 2u);
  EXPECT_THROW(makePlanner("figure8"), std::runtime_error);
}
