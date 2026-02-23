  // Copyright 2022 DeepMind Technologies Limited
  //
  // Licensed under the Apache License, Version 2.0 (the "License");
  // you may not use this file except in compliance with the License.
  // You may obtain a copy of the License at
  //
  //     http://www.apache.org/licenses/LICENSE-2.0
  //
  // Unless required by applicable law or agreed to in writing, software
  // distributed under the License is distributed on an "AS IS" BASIS,
  // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  // See the License for the specific language governing permissions and
  // limitations under the License.

  #include <algorithm>
  #include <cstddef>
  #include <sstream>
  #include <optional>

  #include <mujoco/mjplugin.h>
  #include <mujoco/mjtnum.h>
  #include <mujoco/mujoco.h>
  #include "cable.h"


  namespace mujoco::plugin::elasticity {
  namespace {

  // Jet color palette
  void scalar2rgba(float rgba[4], mjtNum stress[3], mjtNum vmin, mjtNum vmax) {
    // L2 norm of the stress
    mjtNum v = mju_norm3(stress);
    v = v < vmin ? vmin : v;
    v = v > vmax ? vmax : v;
    mjtNum dv = vmax - vmin;

    if (v < (vmin + 0.25 * dv)) {
      rgba[0] = 0;
      rgba[1] = 4 * (v - vmin) / dv;
      rgba[2] = 1;
    } else if (v < (vmin + 0.5 * dv)) {
      rgba[0] = 0;
      rgba[1] = 1;
      rgba[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
    } else if (v < (vmin + 0.75 * dv)) {
      rgba[0] = 4 * (v - vmin - 0.5 * dv) / dv;
      rgba[1] = 1;
      rgba[2] = 0;
    } else {
      rgba[0] = 1;
      rgba[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
      rgba[2] = 0;
    }
  }

  // Compute the discrete Darboux vector between two adjacent segments.
  //   Omega = (2/l) * Im[conj(qi) * qi1]    (Kugelstadt & Schoemer Eq. 7)
  //
  //   inputs:
  //     quati      - orientation of body i (world quaternion)
  //     quati1     - orientation of body i+1 (world quaternion)
  //     li         - length of the segment
  //   outputs:
  //     omega      - Darboux vector (curvature in body i's frame)
  void DarbouxVector(mjtNum omega[3], const mjtNum quati[4],
                    const mjtNum quati1[4], mjtNum li){
    mjtNum quati_conj[4];
    quati_conj[0] =  quati[0];
    quati_conj[1] = -quati[1];
    quati_conj[2] = -quati[2];
    quati_conj[3] = -quati[3];

    mjtNum quat_prod[4];
    mju_mulQuat(quat_prod, quati_conj, quati1);

    mjtNum scl = 2.0 / li;

    omega[0] = scl * quat_prod[1];
    omega[1] = scl * quat_prod[2];
    omega[2] = scl * quat_prod[3];
  }

  // Jacobian of the Darboux vector w.r.t. q_{i+1}.
  //   dOmega/dq_{i+1} = (2/l) * d(Im[conj(qi)*q])/dq
  //
  //   inputs:
  //     qi         - orientation of body i (world quaternion)
  //     li         - length of the segment
  //   outputs:
  //     J          - 3x4 Jacobian matrix (row-major)
  void DarbouxJacobian_qi1(mjtNum J[12], const mjtNum qi[4], mjtNum li) {
    mjtNum scl = 2.0 / li;

    J[0]  = scl * (-qi[1]);
    J[1]  = scl * ( qi[0]);
    J[2]  = scl * ( qi[3]);
    J[3]  = scl * (-qi[2]);

    J[4]  = scl * (-qi[2]);
    J[5]  = scl * (-qi[3]);
    J[6]  = scl * ( qi[0]);
    J[7]  = scl * ( qi[1]);

    J[8]  = scl * (-qi[3]);
    J[9]  = scl * ( qi[2]);
    J[10] = scl * (-qi[1]);
    J[11] = scl * ( qi[0]);
  }

  // Jacobian of the Darboux vector w.r.t. q_i.
  //   dOmega/dq_i = -(2/l) * d(Im[conj(q)*qi1])/dq     (Eq. 11)
  //
  //   inputs:
  //     qi1        - orientation of body i+1 (world quaternion)
  //     li         - length of the segment
  //   outputs:
  //     J          - 3x4 Jacobian matrix (row-major)
  void DarbouxJacobian_qi(mjtNum J[12], const mjtNum qi1[4], mjtNum li) {
    mjtNum scl = -2.0 / li;

    J[0]  = scl * (-qi1[1]);
    J[1]  = scl * ( qi1[0]);
    J[2]  = scl * ( qi1[3]);
    J[3]  = scl * (-qi1[2]);

    J[4]  = scl * (-qi1[2]);
    J[5]  = scl * (-qi1[3]);
    J[6]  = scl * ( qi1[0]);
    J[7]  = scl * ( qi1[1]);

    J[8]  = scl * (-qi1[3]);
    J[9]  = scl * ( qi1[2]);
    J[10] = scl * (-qi1[1]);
    J[11] = scl * ( qi1[0]);
  }

  // Body-frame quaternion-angular velocity map.
  //   q_dot = G(q) * omega_body                          (Eq. 27)
  //   G(q)  = (1/2) * (-Im[q]^T ; Re[q]*I + [Im[q]]x)
  //
  //   inputs:
  //     q          - orientation quaternion (world frame)
  //   outputs:
  //     G          - 4x3 matrix (row-major)
  void Gmat(mjtNum G[12], const mjtNum q[4]) {
    mjtNum scl = 0.5;

    G[0]  = -q[1] * scl;
    G[1]  = -q[2] * scl;
    G[2]  = -q[3] * scl;

    G[3]  =  q[0] * scl;
    G[4]  = -q[3] * scl;
    G[5]  =  q[2] * scl;

    G[6]  =  q[3] * scl;
    G[7]  =  q[0] * scl;
    G[8]  = -q[1] * scl;

    G[9]  = -q[2] * scl;
    G[10] =  q[1] * scl;
    G[11] =  q[0] * scl;
  }

  // Compute local stress from curvature deviation.
  //   stress = K * (omega - omega0)
  //   K = diag(GJ, EIy, EIz)  (rod axis = local x)
  //
  //   inputs:
  //     stiffness  - material parameters [GJ, EIy, EIz, length]
  //     omegai     - current Darboux vector
  //     omega0     - rest Darboux vector
  //   outputs:
  //     stress     - local stress contribution
  void LocalStress(mjtNum stress[3],
                  const mjtNum stiffness[4],
                  const mjtNum omegai[3], const mjtNum omega0[3]) {
    stress[0] = stiffness[0] * (omegai[0] - omega0[0]);
    stress[1] = stiffness[1] * (omegai[1] - omega0[1]);
    stress[2] = stiffness[2] * (omegai[2] - omega0[2]);
  }

  // reads numeric attributes
  bool CheckAttr(const char* name, const mjModel* m, int instance) {
    char *end;
    std::string value = mj_getPluginConfig(m, instance, name);
    value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
    strtod(value.c_str(), &end);
    return end == value.data() + value.size();
  }

  }  // namespace


  // factory function
  std::optional<Cable> Cable::Create(
    const mjModel* m, mjData* d, int instance) {
    if (CheckAttr("twist", m, instance) && CheckAttr("bend", m, instance)) {
      return Cable(m, d, instance);
    } else {
      mju_warning("Invalid parameter specification in cable plugin");
      return std::nullopt;
    }
  }

  // plugin constructor
  Cable::Cable(const mjModel* m, mjData* d, int instance) {
    // parameters were validated by the factor function
    std::string flat = mj_getPluginConfig(m, instance, "flat");
    mjtNum G = strtod(mj_getPluginConfig(m, instance, "twist"), nullptr);
    mjtNum E = strtod(mj_getPluginConfig(m, instance, "bend"), nullptr);
    vmax = strtod(mj_getPluginConfig(m, instance, "vmax"), nullptr);
    // count plugin bodies
    n = 0;
    for (int i = 1; i < m->nbody; i++) {
      if (m->body_plugin[i] == instance) {
        if (!n++) {
          i0 = i;
        }
      }
    }

    // allocate arrays
    prev.assign(n, 0);         // index of previous body
    next.assign(n, 0);         // index of next body
    omega0.assign(3*n, 0);     // reference curvature
    stress.assign(3*n, 0);     // mechanical stress
    stiffness.assign(4*n, 0);  // material parameters

    // run forward kinematics to populate xquat (mjData not yet initialized)
    mju_zero(d->mocap_quat, 4*m->nmocap);
    mju_copy(d->qpos, m->qpos0, m->nq);
    mj_kinematics(m, d);

    // compute initial curvature
    for (int b = 0; b < n; b++) {
      int i = i0 + b;
      if (m->body_plugin[i] != instance) {
        mju_error("This body does not have the requested plugin instance");
      }
      bool first = (b == 0), last = (b == n-1);
      prev[b] = first ? 0 : -1;
      next[b] =  last ? 0 : +1;

      // compute physical parameters
      int geom_i = m->body_geomadr[i];
      mjtNum J = 0, Iy = 0, Iz = 0;
      if (m->geom_type[geom_i] == mjGEOM_CYLINDER ||
          m->geom_type[geom_i] == mjGEOM_CAPSULE) {
        // https://en.wikipedia.org/wiki/Torsion_constant#Circle
        // https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        J = mjPI * pow(m->geom_size[3*geom_i+0], 4) / 2;
        Iy = Iz = mjPI * pow(m->geom_size[3*geom_i+0], 4) / 4.;
      } else if (m->geom_type[geom_i] == mjGEOM_BOX) {
        // https://en.wikipedia.org/wiki/Torsion_constant#Rectangle
        // https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        mjtNum h = m->geom_size[3*geom_i+1];
        mjtNum w = m->geom_size[3*geom_i+2];
        mjtNum a = std::max(h, w);
        mjtNum b = std::min(h, w);
        J = a*pow(b, 3)*(16./3.-3.36*b/a*(1-pow(b, 4)/pow(a, 4)/12));
        Iy = pow(2 * w, 3) * 2 * h / 12.;
        Iz = pow(2 * h, 3) * 2 * w / 12.;
      }
      stiffness[4*b+0] = J * G;
      stiffness[4*b+1] = Iy * E;
      stiffness[4*b+2] = Iz * E;
      stiffness[4*b+3] =
        prev[b] ? mju_dist3(d->xpos+3*i, d->xpos+3*(i+prev[b])) : 0;

      // compute omega0: curvature at equilibrium
      if (prev[b] && flat != "true") {
        int ip = i + prev[b];
        mjtNum li = stiffness[4*b+3];
        DarbouxVector(omega0.data()+3*b, d->xquat+4*ip, d->xquat+4*i, li);
      } else {
        mju_zero3(omega0.data()+3*b);
      }
    }
  }

  // Cosserat rod force computation.
  //
  // For each edge connecting bodies i and i+1:
  //   Energy:  E = (l/2) * dOmega^T * K * dOmega
  //   Torque:  tau = -dE/dtheta = -l * G_half^T * J_full^T * K * dOmega
  //
  // where G_half = (1/2)*G(q) maps angular velocity to quaternion derivative,
  // J_full = (2/l)*dIm/dq is the Darboux Jacobian, and dOmega = Omega - Omega0.
  // The accumulation factor per edge is -l (the segment length).
  void Cable::Compute(const mjModel* m, mjData* d, int instance) {
    for (int b = 0; b < n; b++)  {
      // index into body array
      int i = i0 + b;
      if (m->body_plugin[i] != instance) {
        mju_error(
          "This body is not associated with the requested plugin instance");
      }

      // if no stiffness, skip body
      if (!stiffness[b*4+0] && !stiffness[b*4+1] && !stiffness[b*4+2]) {
        continue;
      }

      // elastic forces
      mjtNum omega[3] = {0};
      mjtNum lfrc[3] = {0};

      // contribution from left edge (body b is the right segment, i.e. q_{i+1})
      if (prev[b]) {
        int ip = i + prev[b];
        mjtNum li = stiffness[4*b+3];

        DarbouxVector(omega, d->xquat+4*ip, d->xquat+4*i, li);
        LocalStress(stress.data() + 3*b, stiffness.data() + 4*b,
                    omega, omega0.data() + 3*b);

        mjtNum jacobian_qi1[12];
        DarbouxJacobian_qi1(jacobian_qi1, d->xquat+4*ip, li);

        mjtNum f_quat[4];
        mju_mulMatTVec(f_quat, jacobian_qi1, stress.data() + 3*b, 3, 4);

        mjtNum G[12];
        Gmat(G, d->xquat+4*i);

        mjtNum torque[3];
        mju_mulMatTVec(torque, G, f_quat, 4, 3);
        mju_addToScl3(lfrc, torque, -li);
      }

      // contribution from right edge (body b is the left segment, i.e. q_i)
      if (next[b]) {
        int bn = b + next[b];
        int in = i + next[b];
        mjtNum li = stiffness[4*bn+3];

        DarbouxVector(omega, d->xquat+4*i, d->xquat+4*in, li);
        LocalStress(stress.data() + 3*bn, stiffness.data() + 4*bn,
                    omega, omega0.data() + 3*bn);

        mjtNum jacobian_qi[12];
        DarbouxJacobian_qi(jacobian_qi, d->xquat+4*in, li);

        mjtNum f_quat[4];
        mju_mulMatTVec(f_quat, jacobian_qi, stress.data() + 3*bn, 3, 4);

        mjtNum G[12];
        Gmat(G, d->xquat+4*i);

        mjtNum torque[3];
        mju_mulMatTVec(torque, G, f_quat, 4, 3);
        mju_addToScl3(lfrc, torque, -li);
      }

      // convert to world frame and apply
      mjtNum xfrc[3] = {0};
      mju_rotVecQuat(xfrc, lfrc, d->xquat+4*i);
      mj_applyFT(m, d, 0, xfrc, d->xpos+3*i, i, d->qfrc_passive);
    }
  }

  void Cable::Visualize(const mjModel* m, mjData* d, mjvScene* scn,
                        int instance) {
    if (!vmax) {
      return;
    }

    for (int b = 0; b < n; b++)  {
      int i = i0 + b;
      int bn = b + next[b];

      // set geometry color based on stress norm
      mjtNum stress_m[3] = {0};
      mjtNum *stress_l = prev[b] ? stress.data()+3*b : stress.data()+3*bn;
      mjtNum *stress_r = next[b] ? stress.data()+3*bn : stress.data()+3*b;
      mju_add3(stress_m, stress_l, stress_r);
      mju_scl3(stress_m, stress_m, 0.5);
      scalar2rgba(m->geom_rgba + 4*m->body_geomadr[i], stress_m, 0, vmax);
    }
  }

  void Cable::RegisterPlugin() {
    mjpPlugin plugin;
    mjp_defaultPlugin(&plugin);

    plugin.name = "mujoco.elasticity.cable";
    plugin.capabilityflags |= mjPLUGIN_PASSIVE;

    const char* attributes[] = {"twist", "bend", "flat", "vmax"};
    plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
    plugin.attributes = attributes;
    plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

    plugin.init = +[](const mjModel* m, mjData* d, int instance) {
      auto elasticity_or_null = Cable::Create(m, d, instance);
      if (!elasticity_or_null.has_value()) {
        return -1;
      }
      d->plugin_data[instance] = reinterpret_cast<uintptr_t>(
          new Cable(std::move(*elasticity_or_null)));
      return 0;
    };
    plugin.destroy = +[](mjData* d, int instance) {
      delete reinterpret_cast<Cable*>(d->plugin_data[instance]);
      d->plugin_data[instance] = 0;
    };
    plugin.compute =
        +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
          auto* elasticity = reinterpret_cast<Cable*>(d->plugin_data[instance]);
          elasticity->Compute(m, d, instance);
        };
    plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn,
                          int instance) {
      auto* elasticity = reinterpret_cast<Cable*>(d->plugin_data[instance]);
      elasticity->Visualize(m, d, scn, instance);
    };

    mjp_registerPlugin(&plugin);
  }

  }  // namespace mujoco::plugin::elasticity
