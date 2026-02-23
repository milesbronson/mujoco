// Standalone diagnostic for the Cosserat cable plugin.
// Build:  cd build && cmake --build . --target cable_debug
// Run:    MUJOCO_PLUGIN_DIR=lib ./bin/cable_debug

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mujoco/mujoco.h>

static const char* make_cable_xml(int count, double twist, double bend,
                                   double damping, double force_range) {
  static char buf[2048];
  snprintf(buf, sizeof(buf), R"(
<mujoco>
  <option gravity="0 0 0" timestep="0.001"/>
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>
  <worldbody>
    <composite type="cable" curve="s" count="%d 1 1" size="1" offset="0 0 0" initial="none">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="%.0f"/>
        <config key="bend" value="%.0f"/>
      </plugin>
      <joint kind="main" damping="%.4f"/>
      <geom type="capsule" size=".005" density="1"/>
    </composite>
  </worldbody>
  <sensor>
    <framepos objtype="site" objname="S_last"/>
  </sensor>
  <actuator>
    <motor site="S_last" gear="0 0 1 0 0 0" ctrllimited="true" ctrlrange="0 %.6f"/>
  </actuator>
</mujoco>
)", count, twist, bend, damping, force_range);
  return buf;
}

static mjModel* load_xml_string(const char* xml) {
  char error[1024] = {0};
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", xml, strlen(xml));
  mjModel* m = mj_loadXML("model.xml", &vfs, error, sizeof(error));
  mj_deleteVFS(&vfs);
  if (!m) {
    printf("Failed: %s\n", error);
  }
  return m;
}

// Run cantilever beam deflection test for given segment count
static void test_deflection(int count, int steps, int ramp_steps) {
  const char* xml = make_cable_xml(count, 1e6, 1e6, 0.1, 1e-4);
  mjModel* m = load_xml_string(xml);
  if (!m) return;
  mjData* d = mj_makeData(m);

  double F = 1e-5;
  for (int i = 0; i < steps; i++) {
    if (i < ramp_steps) {
      d->ctrl[0] += F / ramp_steps;
    }
    mj_step(m, d);
  }

  double L = 1.0;
  double E = 1e6;
  double I = M_PI * pow(0.005, 4) / 4.0;
  double analytical = F * pow(L, 3) / (3.0 * E * I);
  double simulated = d->sensordata[2];
  double error_pct = 100.0 * fabs(simulated - analytical) / analytical;

  printf("  count=%3d (%3d segs): simulated=%.6e  analytical=%.6e  error=%.1f%%\n",
         count, count-1, simulated, analytical, error_pct);

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Run cantilever into circle test for given segment count
static void test_circle(int count) {
  const char* xml_fmt;
  static char buf[2048];
  snprintf(buf, sizeof(buf), R"(
<mujoco>
  <option gravity="0 0 0"/>
  <extension>
    <plugin plugin="mujoco.elasticity.cable"/>
  </extension>
  <worldbody>
    <geom type="plane" size="0 0 1" quat="1 0 0 0"/>
    <site name="reference" pos="0 0 0"/>
    <composite type="cable" curve="s" count="%d 1 1" size="1" offset="0 0 1" initial="none">
      <plugin plugin="mujoco.elasticity.cable">
        <config key="twist" value="1e6"/>
        <config key="bend" value="1e9"/>
      </plugin>
      <joint kind="main" damping="2"/>
      <geom type="capsule" size=".005" density="1"/>
    </composite>
  </worldbody>
  <contact>
    <exclude body1="B_first" body2="B_last"/>
  </contact>
  <sensor>
    <framepos objtype="site" objname="S_last"/>
  </sensor>
  <actuator>
    <motor site="S_last" gear="0 0 0 0 1 0" ctrllimited="true" ctrlrange="0 4"/>
  </actuator>
</mujoco>
)", count);

  mjModel* m = load_xml_string(buf);
  if (!m) return;
  mjData* d = mj_makeData(m);

  double Iy = M_PI * pow(0.005, 4) / 4.0;
  double torque = 2 * M_PI * 1e9 * Iy;
  for (int i = 0; i < 1300; i++) {
    if (i < 300) {
      d->ctrl[0] += torque / 300;
    }
    mj_step(m, d);
  }

  printf("  count=%3d: tip=[%.6f, %.6f, %.6f]  err_x=%.6f  err_z=%.6f\n",
         count,
         d->sensordata[0], d->sensordata[1], d->sensordata[2],
         fabs(d->sensordata[0]), fabs(d->sensordata[2] - 1.0));

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Diagnostic: print body quaternions and positions for a small chain.
// This test revealed that the cable composite keeps all body frames
// identity-aligned, meaning the rod axis = local X (not local Z).
static void test_body_frames(int count) {
  const char* xml = make_cable_xml(count, 1e6, 1e6, 0.1, 1e-4);
  mjModel* m = load_xml_string(xml);
  if (!m) return;
  mjData* d = mj_makeData(m);

  mj_forward(m, d);

  printf("  nbody=%d, nq=%d, nv=%d\n", m->nbody, m->nq, m->nv);
  for (int i = 1; i < m->nbody && i <= count; i++) {
    printf("  body %2d: pos=[%8.4f %8.4f %8.4f]  quat=[%6.3f %6.3f %6.3f %6.3f]\n",
           i,
           d->xpos[3*i+0], d->xpos[3*i+1], d->xpos[3*i+2],
           d->xquat[4*i+0], d->xquat[4*i+1], d->xquat[4*i+2], d->xquat[4*i+3]);
  }

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Diagnostic: rotate a single joint about x, y, z and measure the resulting
// forces. This verifies the stiffness-to-axis mapping:
//   x-rotation -> GJ  (torsion)  = stiffness[0]
//   y-rotation -> EIy (bending)  = stiffness[1]
//   z-rotation -> EIz (bending)  = stiffness[2]
// For circular cross-section with E=G, GJ = 2*EIy, so x-rotation force
// should be 2x the y or z-rotation force.
static void test_single_joint_rotation() {
  // 4-body cable (3 segments)
  const char* xml = make_cable_xml(4, 1e6, 1e6, 0.0, 1e-4);
  mjModel* m = load_xml_string(xml);
  if (!m) return;
  mjData* d = mj_makeData(m);

  double angle = 0.01;  // small rotation in radians
  const char* axis_names[] = {"x", "y", "z"};

  for (int axis = 0; axis < 3; axis++) {
    // reset state
    mju_copy(d->qpos, m->qpos0, m->nq);
    mju_zero(d->qvel, m->nv);
    mju_zero(d->qfrc_passive, m->nv);

    // find the ball joint DOFs for body 2 (second segment)
    // ball joints have 3 DOFs: rotations about x, y, z
    int jnt_adr = m->body_jntadr[2];
    int qpos_adr = m->jnt_qposadr[jnt_adr];

    // set quaternion for rotation about the chosen axis
    double half = angle / 2.0;
    d->qpos[qpos_adr + 0] = cos(half);  // w
    d->qpos[qpos_adr + 1] = (axis == 0) ? sin(half) : 0;  // x
    d->qpos[qpos_adr + 2] = (axis == 1) ? sin(half) : 0;  // y
    d->qpos[qpos_adr + 3] = (axis == 2) ? sin(half) : 0;  // z

    mj_forward(m, d);

    // print passive forces on all DOFs
    int nv = m->nv;
    printf("  %s-rotation (%.3f rad): qfrc_passive = [", axis_names[axis], angle);
    for (int v = 0; v < nv; v++) {
      if (v > 0) printf(", ");
      printf("%.4e", d->qfrc_passive[v]);
    }
    printf("]\n");
  }

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Diagnostic: measure forces at different damping values to show
// convergence behavior (overdamped system).
static void test_damping_sweep() {
  double dampings[] = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0};
  double L = 1.0, E = 1e6, F = 1e-5;
  double I = M_PI * pow(0.005, 4) / 4.0;
  double analytical = F * pow(L, 3) / (3.0 * E * I);

  for (double damp : dampings) {
    const char* xml = make_cable_xml(51, 1e6, 1e6, damp, 1e-4);
    mjModel* m = load_xml_string(xml);
    if (!m) continue;
    mjData* d = mj_makeData(m);

    for (int i = 0; i < 10000; i++) {
      if (i < 1000) d->ctrl[0] += F / 1000;
      mj_step(m, d);
    }

    double simulated = d->sensordata[2];
    double error_pct = 100.0 * fabs(simulated - analytical) / analytical;
    double tau = damp / (E * I / (L / 50.0));  // approx time constant

    printf("  damping=%.3f: simulated=%.6e  error=%5.1f%%  tau~%.1fs\n",
           damp, simulated, error_pct, tau);

    mj_deleteData(d);
    mj_deleteModel(m);
  }
}

int main() {
  const char* plugin_dir = std::getenv("MUJOCO_PLUGIN_DIR");
  if (!plugin_dir) { printf("Set MUJOCO_PLUGIN_DIR=lib\n"); return 1; }
  char pp[1024];
#if defined(__APPLE__)
  snprintf(pp, sizeof(pp), "%s/libelasticity.dylib", plugin_dir);
#else
  snprintf(pp, sizeof(pp), "%s/libelasticity.so", plugin_dir);
#endif
  mj_loadPluginLibrary(pp);

  printf("=== 1. Body frame diagnostic (4-body chain) ===\n");
  test_body_frames(4);

  printf("\n=== 2. Single-joint rotation: stiffness-axis mapping ===\n");
  printf("  Expected: x-rotation force ~2x y/z (GJ = 2*EI for circle)\n");
  test_single_joint_rotation();

  printf("\n=== 3. Damping sweep: convergence behavior ===\n");
  printf("  Analytical: F*L^3/(3EI) = %.6e\n", 1e-5 / (3.0 * 1e6 * M_PI * pow(0.005,4) / 4.0));
  test_damping_sweep();

  printf("\n=== 4. Cantilever beam deflection vs segment count ===\n");
  printf("  Analytical: F*L^3/(3EI) = %.6e\n",
         1e-5 * 1.0 / (3.0 * 1e6 * M_PI * pow(0.005,4) / 4.0));
  int counts[] = {11, 21, 41, 51, 81, 101, 201};
  for (int c : counts) {
    test_deflection(c, 10000, 1000);
  }

  printf("\n=== 5. Cantilever into circle vs segment count ===\n");
  int ccounts[] = {21, 41, 81, 101};
  for (int c : ccounts) {
    test_circle(c);
  }

  printf("\nDone.\n");
  return 0;
}
