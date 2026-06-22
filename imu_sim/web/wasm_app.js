// Browser client for the MuJoCo-WASM render demo.
//
// The Python server runs the physics + IMU + control loop and streams generalized
// positions (qpos) over Server-Sent Events. Here we compile the *same* model in
// MuJoCo-WASM (official google-deepmind `mujoco-js`), then every frame write the
// streamed qpos into mjData, run mj_forward to get world geom transforms, and draw
// them with three.js. No physics runs in the browser — the WASM is used purely as a
// forward-kinematics + geometry source so the render matches the Python sim exactly.
//
// MuJoCo is Z-up; we keep the three.js world Z-up (camera.up = +Z) and drive each
// mesh directly from data.geom_xpos / data.geom_xmat, so no coordinate conversion is
// needed. Collision geom groups 0 and 3 are hidden to match the Python renderer.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from 'mujoco-js';

const $ = (s) => document.querySelector(s);
const HIDDEN_GROUPS = new Set([0, 3]);   // collision geoms, as in server.py

let mujoco, model, data;
let scene, camera, renderer, controls;
let geomMeshes = [];                      // [{ mesh, g }]
let robotName = null, switching = false;

// telemetry charts
const COL_TRUTH = '#3fb950', COL_MEAS = '#f0883e';
let tiltChart, gyroChart, axisIdx = 1;    // 0 = roll, 1 = pitch

// Rolling two-line strip chart on a <canvas>. Series `a` = ground truth, `b` = sensor/
// estimate. Auto-scales y to the visible window (always including 0), breaks the line on
// null (e.g. controller off), and resets if the sim clock jumps backward (reset).
class Strip {
  constructor(canvas, windowSec = 12) {
    this.cv = canvas; this.ctx = canvas.getContext('2d');
    this.win = windowSec; this.data = [];
  }
  reset() { this.data = []; }
  push(t, a, b) {
    if (this.data.length && t < this.data[this.data.length - 1].t) this.data = [];
    this.data.push({ t, a, b });
    const tmin = t - this.win;
    let i = 0; while (i < this.data.length && this.data[i].t < tmin) i++;
    if (i) this.data.splice(0, i);
  }
  draw() {
    const cv = this.cv, c = this.ctx;
    const W = cv.clientWidth, H = cv.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, W, H);
    const x0 = 44, x1 = W - 8, y0 = 8, y1 = H - 16;
    c.strokeStyle = '#2b3647'; c.lineWidth = 1;
    c.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0, y1 - y0);
    const d = this.data;
    if (d.length < 2) return;

    let lo = Infinity, hi = -Infinity;
    for (const p of d) {
      for (const v of [p.a, p.b]) {
        if (v == null || Number.isNaN(v)) continue;
        if (v < lo) lo = v; if (v > hi) hi = v;
      }
    }
    if (!isFinite(lo)) return;
    lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    const span = (hi - lo) || 1, pad = span * 0.12;
    lo -= pad; hi += pad;
    const tEnd = d[d.length - 1].t, tStart = tEnd - this.win;
    const X = (t) => x0 + (t - tStart) / this.win * (x1 - x0);
    const Y = (v) => y1 - (v - lo) / (hi - lo) * (y1 - y0);

    c.fillStyle = '#8b98a9'; c.font = '10px system-ui'; c.textAlign = 'right'; c.textBaseline = 'middle';
    c.fillText(hi.toFixed(2), x0 - 5, Y(hi) + 4);
    c.fillText(lo.toFixed(2), x0 - 5, Y(lo) - 4);

    c.save();
    c.beginPath(); c.rect(x0, y0, x1 - x0, y1 - y0); c.clip();
    if (lo < 0 && hi > 0) {
      const yz = Y(0);
      c.strokeStyle = '#3a475a'; c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(x0, yz); c.lineTo(x1, yz); c.stroke(); c.setLineDash([]);
    }
    const line = (key, color) => {
      c.strokeStyle = color; c.lineWidth = 1.5; c.beginPath();
      let started = false;
      for (const p of d) {
        const v = p[key];
        if (v == null || Number.isNaN(v)) { started = false; continue; }
        const xx = X(p.t), yy = Y(v);
        if (!started) { c.moveTo(xx, yy); started = true; } else c.lineTo(xx, yy);
      }
      c.stroke();
    };
    line('a', COL_TRUTH);
    line('b', COL_MEAS);
    c.restore();
  }
}

function updateAxisLabels() {
  const tl = axisIdx === 0 ? 'roll' : 'pitch';
  const gl = axisIdx === 0 ? 'roll rate' : 'pitch rate';
  $('#tilt_ttl').textContent = 'Attitude — ' + tl + ' (°)';
  $('#gyro_ttl').textContent = 'Gyro — ' + gl + ' (rad/s)';
}

// ---------------------------------------------------------------- server I/O
async function cmd(obj) {
  try {
    await fetch('/command', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });
  } catch (e) { /* ignore transient errors */ }
}
function overlay(msg) {
  const el = $('#overlay');
  if (!el) return;
  if (msg) { el.textContent = msg; el.style.display = 'flex'; }
  else el.style.display = 'none';
}

// ---------------------------------------------------------------- three.js
function initThree() {
  const canvas = $('#canvas');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1419);

  camera = new THREE.PerspectiveCamera(45, 4 / 3, 0.01, 200);
  camera.up.set(0, 0, 1);                 // MuJoCo world is Z-up
  camera.position.set(2.6, -2.6, 1.7);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.8);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(3, -3, 5);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.3);
  fill.position.set(-3, 2, 2);
  scene.add(fill);

  const grid = new THREE.GridHelper(20, 40, 0x2b3647, 0x1a2230);
  grid.rotateX(Math.PI / 2);              // GridHelper is XZ by default -> make it XY
  scene.add(grid);

  resize();
  window.addEventListener('resize', resize);
}

function resize() {
  const canvas = $('#canvas');
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  if (tiltChart) tiltChart.draw();
  if (gyroChart) gyroChart.draw();
}

// ---------------------------------------------------------------- geometry
function makeGeometry(g) {
  const t = model.geom_type[g], s = model.geom_size;
  const sx = s[g * 3], sy = s[g * 3 + 1], sz = s[g * 3 + 2];
  switch (t) {
    case 0: return null;                                   // plane (hidden anyway)
    case 2: return new THREE.SphereGeometry(sx, 24, 16);   // sphere
    case 3: {                                              // capsule (axis Z)
      const geo = new THREE.CapsuleGeometry(sx, 2 * sy, 8, 16);
      geo.rotateX(Math.PI / 2); return geo;
    }
    case 4: {                                              // ellipsoid
      const geo = new THREE.SphereGeometry(1, 24, 16);
      geo.scale(sx, sy, sz); return geo;
    }
    case 5: {                                              // cylinder (axis Z)
      const geo = new THREE.CylinderGeometry(sx, sx, 2 * sy, 24);
      geo.rotateX(Math.PI / 2); return geo;
    }
    case 6: return new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz); // box
    case 7: return makeMeshGeometry(g);                    // mesh
    default: return null;
  }
}

function makeMeshGeometry(g) {
  const mid = model.geom_dataid[g];
  if (mid < 0) return null;
  const v0 = model.mesh_vertadr[mid], vn = model.mesh_vertnum[mid];
  const f0 = model.mesh_faceadr[mid], fn = model.mesh_facenum[mid];
  // mesh_face indices are local (0-based within this mesh's vertex slice).
  const verts = Float32Array.from(model.mesh_vert.subarray(v0 * 3, (v0 + vn) * 3));
  const faces = Uint32Array.from(model.mesh_face.subarray(f0 * 3, (f0 + fn) * 3));
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setIndex(new THREE.BufferAttribute(faces, 1));
  geo.computeVertexNormals();
  return geo;
}

function geomColor(g) {
  let r = model.geom_rgba, i = g * 4;
  let col = [r[i], r[i + 1], r[i + 2]], op = r[i + 3];
  const matid = model.geom_matid[g];
  if (matid >= 0 && model.mat_rgba) {
    const m = model.mat_rgba, j = matid * 4;
    col = [m[j], m[j + 1], m[j + 2]]; op = m[j + 3];
  }
  return { col, op: op > 0 ? op : 1 };
}

function buildScene() {
  for (const o of geomMeshes) {
    scene.remove(o.mesh);
    o.mesh.geometry.dispose();
    o.mesh.material.dispose();
  }
  geomMeshes = [];

  for (let g = 0; g < model.ngeom; g++) {
    if (HIDDEN_GROUPS.has(model.geom_group[g])) continue;
    const geo = makeGeometry(g);
    if (!geo) continue;
    const { col, op } = geomColor(g);
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(col[0], col[1], col[2]),
      metalness: 0.15, roughness: 0.75,
      transparent: op < 1, opacity: op,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.matrixAutoUpdate = false;
    scene.add(mesh);
    geomMeshes.push({ mesh, g });
  }
  syncPoses();
}

function syncPoses() {
  const xpos = data.geom_xpos, xmat = data.geom_xmat;
  for (const o of geomMeshes) {
    const p = o.g * 3, r = o.g * 9;
    o.mesh.matrix.set(
      xmat[r],     xmat[r + 1], xmat[r + 2], xpos[p],
      xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
      xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
      0, 0, 0, 1,
    );
  }
}

// ---------------------------------------------------------------- model load
function writeFileMkdir(path, bytes) {
  const parts = path.split('/').filter(Boolean);
  let cur = '';
  for (let i = 0; i < parts.length - 1; i++) {
    cur += '/' + parts[i];
    try { mujoco.FS.mkdir(cur); } catch (e) { /* already exists */ }
  }
  mujoco.FS.writeFile(path, bytes);
}

async function loadRobot(robot) {
  overlay('Loading model: ' + robot + ' …');
  const man = await (await fetch('/model/manifest?robot=' + encodeURIComponent(robot))).json();
  await Promise.all(man.files.map(async (rel) => {
    const buf = await (await fetch(
      '/model/file?robot=' + encodeURIComponent(robot) + '&path=' + encodeURIComponent(rel))).arrayBuffer();
    writeFileMkdir('/work/' + rel, new Uint8Array(buf));
  }));

  if (data) { data.delete(); data = null; }
  if (model) { model.delete(); model = null; }
  model = mujoco.MjModel.loadFromXML('/work/' + man.scene);
  data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);

  buildScene();
  robotName = robot;
  overlay('');
}

// ---------------------------------------------------------------- pose stream
function startStream() {
  const es = new EventSource('/poses');
  es.onmessage = (e) => {
    let msg;
    try { msg = JSON.parse(e.data); } catch (_) { return; }
    if (!switching && model && Array.isArray(msg.qpos) && msg.qpos.length === model.nq) {
      const q = data.qpos;
      for (let i = 0; i < q.length; i++) q[i] = msg.qpos[i];
      mujoco.mj_forward(model, data);
      syncPoses();
    }
    if (msg.tel) updateTelemetry(msg.tel);
  };
  es.onerror = () => { /* EventSource auto-reconnects */ };
}

function updateTelemetry(t) {
  if (!t || !t.status) return;
  const s = $('#t_status'); s.textContent = t.status; s.className = 'v status ' + t.status;
  $('#t_time').textContent = (t.t ?? '—') + ' s';
  $('#t_temp').textContent = (t.temperature == null ? 'n/a' : t.temperature + ' °C');
  $('#t_err').textContent  = (t.tilt_err ?? '—') + '°';
  $('#g_err').textContent = (t.gyro_err == null ? 'n/a' : t.gyro_err + ' rad/s');
  $('#a_err').textContent = (t.accel_err == null ? 'n/a' : t.accel_err + ' m/s²');

  const a = axisIdx;
  const trp = t.true_rp, erp = t.est_rp, tg = t.true_gyro, ig = t.imu_gyro;
  const at = trp ? trp[a] : null, ae = erp ? erp[a] : null;
  const gt = tg ? tg[a] : null, gi = ig ? ig[a] : null;
  $('#lg_t_t').textContent = at == null ? '—' : at.toFixed(2) + '°';
  $('#lg_t_e').textContent = ae == null ? '—' : ae.toFixed(2) + '°';
  $('#lg_g_t').textContent = gt == null ? '—' : gt.toFixed(3);
  $('#lg_g_i').textContent = gi == null ? '—' : gi.toFixed(3);
  if (tiltChart && t.t != null) tiltChart.push(t.t, at, ae);
  if (gyroChart && t.t != null) gyroChart.push(t.t, gt, gi);
}

// ---------------------------------------------------------------- UI wiring
const magnitude = () => parseFloat($('#mag').value);

function pushScreen(e) {                   // shift-click -> shove toward the click point
  const r = $('#canvas').getBoundingClientRect();
  const nx = (e.clientX - r.left) / r.width - 0.5;   // right +
  const ny = (e.clientY - r.top) / r.height - 0.5;   // down +
  const fwd = new THREE.Vector3();
  camera.getWorldDirection(fwd); fwd.z = 0;
  if (fwd.lengthSq() < 1e-6) return;
  fwd.normalize();
  const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 0, 1)).normalize();
  const dir = new THREE.Vector3().addScaledVector(right, nx).addScaledVector(fwd, -ny);
  if (dir.lengthSq() < 1e-6) return;
  dir.normalize();
  const m = magnitude();
  cmd({ action: 'push', fx: dir.x * m, fy: dir.y * m });
}

function initUI(meta) {
  const robot = $('#robot'), imu = $('#imu');
  meta.robots.forEach((r) => robot.add(new Option(r, r)));
  meta.imus.forEach((i) => imu.add(new Option(i, i)));
  robot.value = meta.robot; imu.value = meta.imu;

  tiltChart = new Strip($('#chart_tilt'));
  gyroChart = new Strip($('#chart_gyro'));
  updateAxisLabels();
  $('#axis').onchange = () => {
    axisIdx = parseInt($('#axis').value, 10);
    tiltChart.reset(); gyroChart.reset();
    updateAxisLabels();
  };

  robot.onchange = async () => {
    switching = true;
    overlay('Switching robot…');
    await cmd({ action: 'set_robot', robot: robot.value });
    await loadRobot(robot.value);
    switching = false;
  };
  imu.onchange = () => cmd({ action: 'set_imu', imu: imu.value });

  $('#reset').onclick = () => cmd({ action: 'reset' });
  $('#mag').oninput = () => { $('#magval').textContent = $('#mag').value; };

  document.querySelectorAll('.pad button[data-fx]').forEach((b) => {
    b.onclick = () => {
      const m = magnitude();
      cmd({ action: 'push', fx: parseFloat(b.dataset.fx) * m, fy: parseFloat(b.dataset.fy) * m });
    };
  });
  $('#rand').onclick = () => {
    const a = Math.random() * 2 * Math.PI, m = magnitude();
    cmd({ action: 'push', fx: Math.cos(a) * m, fy: Math.sin(a) * m });
  };

  $('#canvas').addEventListener('pointerdown', (e) => { if (e.shiftKey) pushScreen(e); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'r' || e.key === 'R') cmd({ action: 'reset' });
  });
}

// ---------------------------------------------------------------- boot
async function main() {
  initThree();
  animate();
  overlay('Loading MuJoCo WASM…');
  mujoco = await loadMujoco();
  mujoco.FS.mkdir('/work');
  mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/work');

  const meta = await (await fetch('/meta')).json();
  initUI(meta);
  await loadRobot(meta.robot);
  startStream();
}

main().catch((err) => {
  console.error(err);
  overlay('Failed to start: ' + (err && err.message ? err.message : err));
});
