// MIT License
// Copyright (c) 2026 FSC Lab
//
// The usable (r, z) workspace grid (see workspace.hpp). Moved out of the node
// (2026-10-01) so it can be tested; the rasterisation is unchanged.
#include "fsc_trajectory_planner/workspace.hpp"

#include <algorithm>
#include <cmath>

namespace fsc_trajectory_planner
{

bool WorkspaceGrid::contains(double r, double z) const
{
  if (nr <= 0 || nz <= 0) {return false;}
  if (r < r_min || r > r_max || z < z_min || z > z_max) {return false;}
  const int ri = static_cast<int>((r - r_min) / (r_max - r_min) * (nr - 1) + 0.5);
  const int zi = static_cast<int>((z_max - z) / (z_max - z_min) * (nz - 1) + 0.5);
  if (ri < 0 || ri >= nr || zi < 0 || zi >= nz) {return false;}
  return cells[static_cast<size_t>(zi) * nr + ri] != 0;
}

double WorkspaceGrid::filledFraction() const
{
  if (cells.empty()) {return 0.0;}
  double filled = 0.0;
  for (uint8_t c : cells) {filled += c;}
  return filled / cells.size();
}

WorkspaceGrid usableWorkspace(const VehicleModel & v, double fold_min)
{
  const WholeBodyParams & P = v.params;
  const double beta_guard = v.beta_min_deg * M_PI / 180.0;
  std::vector<double> rs, zs;
  const int n2 = 241, n3 = 181;
  for (int i = 0; i < n2; ++i) {
    const double q2 = v.q_min(1) + (v.q_max(1) - v.q_min(1)) * i / (n2 - 1);
    for (int k = 0; k < n3; ++k) {
      const double q3 = v.q_min(2) + (v.q_max(2) - v.q_min(2)) * k / (n3 - 1);
      if (q2 + q3 < fold_min) {continue;}
      VecN q;
      q << 0.0, q2, q3, 0.0;
      // below the EE-trajectory guard only well-conditioned poses count
      if (q2 + q3 < beta_guard && sigmaNd(q, P) < v.sigma_nd_margin) {continue;}
      Vec3 r0e;
      armKinematics(q, P, nullptr, &r0e, nullptr);
      const Vec3 b = v.r_model * r0e;
      rs.push_back(std::hypot(b(0), b(1)));
      zs.push_back(b(2));
    }
  }
  WorkspaceGrid g;
  if (rs.empty()) {return g;}
  g.rs_min = *std::min_element(rs.begin(), rs.end());
  g.rs_max = *std::max_element(rs.begin(), rs.end());
  g.zs_min = *std::min_element(zs.begin(), zs.end());
  g.zs_max = *std::max_element(zs.begin(), zs.end());
  const double pad = 0.012;
  g.r_min = 0.0;
  g.r_max = g.rs_max + pad;
  g.z_min = g.zs_min - pad;
  g.z_max = std::max(0.0, g.zs_max) + pad;
  g.nr = 192;
  g.nz = 192;
  g.cells.assign(static_cast<size_t>(g.nr) * g.nz, 0);
  for (size_t i = 0; i < rs.size(); ++i) {
    const int ci = static_cast<int>((rs[i] - g.r_min) / (g.r_max - g.r_min) * (g.nr - 1) + 0.5);
    const int ri = static_cast<int>((g.z_max - zs[i]) / (g.z_max - g.z_min) * (g.nz - 1) + 0.5);
    if (ci < 0 || ci >= g.nr || ri < 0 || ri >= g.nz) {continue;}
    // 3x3 stamp seals the sampling holes so the region draws as an area
    for (int dr = -1; dr <= 1; ++dr) {
      for (int dc = -1; dc <= 1; ++dc) {
        const int rr = std::min(g.nz - 1, std::max(0, ri + dr));
        const int cc = std::min(g.nr - 1, std::max(0, ci + dc));
        g.cells[static_cast<size_t>(rr) * g.nr + cc] = 1;
      }
    }
  }
  return g;
}

}  // namespace fsc_trajectory_planner
