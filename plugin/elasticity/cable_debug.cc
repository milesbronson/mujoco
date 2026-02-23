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

  printf("=== Cantilever beam deflection vs segment count ===\n");
  printf("  Analytical: F*L^3/(3EI) = %.6e\n",
         1e-5 * 1.0 / (3.0 * 1e6 * M_PI * pow(0.005,4) / 4.0));
  int counts[] = {11, 21, 41, 51, 81, 101, 201};
  for (int c : counts) {
    test_deflection(c, 10000, 1000);
  }

  printf("\n=== Cantilever into circle vs segment count ===\n");
  int ccounts[] = {21, 41, 81, 101};
  for (int c : ccounts) {
    test_circle(c);
  }

  printf("\nDone.\n");
  return 0;
}
