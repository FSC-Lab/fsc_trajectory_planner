// Parity of the C++ kinematics against the Python transition_planner
// (fixture test/data/python_kinematics_t650.txt, regenerate with
// scripts/dump_python_fixtures.py) plus the IK round-trip self-check.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"

using namespace fsc_trajectory_planner;

namespace
{
template<int R, int C>
Eigen::Matrix<double, R, C> readMat(std::istream & in)
{
  // Python ravel() is row-major
  Eigen::Matrix<double, R, C> m;
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {in >> m(i, j);}
  }
  return m;
}
}  // namespace

TEST(Kinematics, ParityWithPython)
{
  std::ifstream in("data/python_kinematics_t650.txt");
  ASSERT_TRUE(in.good()) << "run from the test/ directory";
  const auto vehicle = makeVehicleModel("t650_aerial_manipulator");
  const WholeBodyParams & P = vehicle->params;
  std::string tok;
  int ncases = 0;
  in >> tok >> ncases;
  ASSERT_GT(ncases, 0);
  for (int c = 0; c < ncases; ++c) {
    RestSpec rest;
    in >> tok; rest.x_b = readMat<3, 1>(in);
    in >> tok >> rest.phi;
    in >> tok; rest.q = readMat<4, 1>(in);
    in >> tok; const Vec3 r0c_py = readMat<3, 1>(in);
    in >> tok; const Vec3 r0e_py = readMat<3, 1>(in);
    in >> tok; const Mat3 re_py = readMat<3, 3>(in);
    in >> tok; const Vec3 x_cd_py = readMat<3, 1>(in);
    in >> tok; const Vec3 r_ed_py = readMat<3, 1>(in);
    in >> tok; const Vec3 b1_d_py = readMat<3, 1>(in);
    in >> tok; const Vec3 b1_de_py = readMat<3, 1>(in);
    double sig_py = 0.0;
    in >> tok >> sig_py;
    in >> tok; const VecN q_ik_py = readMat<4, 1>(in);
    int ok_py = 0;
    in >> ok_py;

    Vec3 r0c, r0e;
    Mat3 re;
    armKinematics(rest.q, P, &r0c, &r0e, &re);
    EXPECT_LT((r0c - r0c_py).norm(), 1e-12) << "case " << c;
    EXPECT_LT((r0e - r0e_py).norm(), 1e-12) << "case " << c;
    EXPECT_LT((re - re_py).norm(), 1e-12) << "case " << c;
    const WbReference rr = restReference(P, rest);
    EXPECT_LT((rr.x_cd - x_cd_py).norm(), 1e-12);
    EXPECT_LT((rr.r_ed - r_ed_py).norm(), 1e-12);
    EXPECT_LT((rr.b1_d - b1_d_py).norm(), 1e-12);
    EXPECT_LT((rr.b1_de - b1_de_py).norm(), 1e-12);
    EXPECT_NEAR(sigmaNd(rest.q, P), sig_py, 1e-10);

    const double az = std::atan2(rr.b1_de(1), rr.b1_de(0));
    const VecN seed = (VecN() << 0.0, 0.7, 0.7, 0.0).finished();
    const IkResult ik = ikWorld(P, rest.x_b, rest.phi, rr.r_ed, az, seed);
    EXPECT_EQ(ik.ok, ok_py == 1) << "case " << c << ": " << ik.reason;
    if (ik.ok) {
      EXPECT_LT((ik.q - q_ik_py).cwiseAbs().maxCoeff(), 1e-7) << "case " << c;
      EXPECT_LT((ik.q - rest.q).cwiseAbs().maxCoeff(), 1e-7) << "case " << c;
    }
  }
  double m_total = 0.0;
  in >> tok >> m_total;
  EXPECT_NEAR(P.totalMass(), m_total, 1e-12);
}

TEST(Kinematics, IkRoundTripSelfCheck)
{
  const auto vehicle = makeVehicleModel("t650_aerial_manipulator");
  const WholeBodyParams & P = vehicle->params;
  VecN q_star;
  q_star << 15.0 * M_PI / 180.0, 30.0 * M_PI / 180.0, 35.0 * M_PI / 180.0,
    -20.0 * M_PI / 180.0;
  RestSpec r;
  r.x_b << 0.2, -0.1, 1.2;
  r.phi = 25.0 * M_PI / 180.0;
  r.q = q_star;
  const WbReference rr = restReference(P, r);
  const double az = std::atan2(rr.b1_de(1), rr.b1_de(0));
  const IkResult ik = ikWorld(P, r.x_b, r.phi, rr.r_ed, az, vehicle->home);
  EXPECT_TRUE(ik.ok) << ik.reason;
  EXPECT_LT((ik.q - q_star).cwiseAbs().maxCoeff(), 1e-7);
  EXPECT_GT(ik.sigma_nd, kSigmaNdMargin);
}

TEST(Kinematics, IkRefusesUnreachable)
{
  const auto vehicle = makeVehicleModel("t650_aerial_manipulator");
  const IkResult ik = ikWorld(
    vehicle->params, Vec3{0, 0, 1.2}, 0.0, Vec3{2.0, 0.0, 1.2}, 0.0,
    vehicle->home);
  EXPECT_FALSE(ik.ok);
  EXPECT_FALSE(ik.reason.empty());
}

TEST(VehicleRegistry, UnknownNameThrows)
{
  EXPECT_THROW(makeVehicleModel("no_such_vehicle"), std::runtime_error);
  EXPECT_EQ(vehicleNames().size(), 1u);
}
