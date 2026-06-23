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

// input + interaction
const keys = {};                          // pressed WASD/QE state
let command = [0, 0, 0];                  // vx, vy, yaw last sent
const raycaster = new THREE.Raycaster();
let arrow = null;                         // force arrow shown during a shove drag
let drag = null;                          // active shove drag {body, x0, y0, point}

// telemetry charts
const COL_TRUTH = '#3fb950', COL_MEAS = '#f0883e';
let tiltChart, gyroChart, axisIdx = 1;    // 0 = roll, 1 = pitch

// Smallest "nice" number (1/2/5 × 10^k) >= x. Used for grid increments.
function niceStep(x) {
  if (!(x > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  const f = x / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

// Rolling two-line strip chart on a <canvas>. Series `a` = ground truth, `b` = sensor/
// estimate. The y-axis snaps to whole "nice" increments and is sticky: it grows
// immediately when data exceeds it but shrinks only with hysteresis, so a slowly growing
// offset visibly climbs a stable grid instead of the grid rescaling under it. Breaks the
// line on null (controller off) and resets if the sim clock jumps backward (reset).
class Strip {
  constructor(canvas, minSpan = 1, windowSec = 12) {
    this.cv = canvas; this.ctx = canvas.getContext('2d');
    this.win = windowSec; this.minSpan = minSpan; this.data = [];
    this.lo = this.hi = this.step = null;
  }
  reset() { this.data = []; this.lo = this.hi = this.step = null; }
  push(t, a, b) {
    if (this.data.length && t < this.data[this.data.length - 1].t) this.reset();
    this.data.push({ t, a, b });
    const tmin = t - this.win;
    let i = 0; while (i < this.data.length && this.data[i].t < tmin) i++;
    if (i) this.data.splice(0, i);
  }
  _resnap(dlo, dhi) {
    this.lo = Math.floor(Math.min(dlo, 0) / this.step) * this.step;
    this.hi = Math.ceil(Math.max(dhi, 0) / this.step) * this.step;
    if (this.hi <= this.lo) this.hi = this.lo + this.step;
  }
  _range(dlo, dhi) {
    dlo = Math.min(dlo, 0); dhi = Math.max(dhi, 0);
    const minStep = niceStep(this.minSpan / 4);
    if (this.step == null) {
      this.step = Math.max(minStep, niceStep((dhi - dlo) / 4));
      this._resnap(dlo, dhi);
    }
    const s = this.step;
    while (dhi > this.hi - 1e-9) this.hi += s;                 // grow up in whole steps
    while (dlo < this.lo + 1e-9) this.lo -= s;                 // grow down
    while (this.hi - s > dhi + 1e-9 && this.hi - s > 1e-9) this.hi -= s;   // hysteresis shrink
    while (this.lo + s < dlo - 1e-9 && this.lo + s < -1e-9) this.lo += s;
    // Re-bucket the step if the grid gets too dense or too sparse (discrete jump).
    const divs = (this.hi - this.lo) / s;
    if (divs > 8) { this.step = niceStep(s * 1.5); this._resnap(dlo, dhi); }
    else if (s > minStep && divs < 3) { this.step = Math.max(minStep, niceStep(s * 0.6)); this._resnap(dlo, dhi); }
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

    let dlo = Infinity, dhi = -Infinity;
    for (const p of d) for (const v of [p.a, p.b]) {
      if (v == null || Number.isNaN(v)) continue;
      if (v < dlo) dlo = v; if (v > dhi) dhi = v;
    }
    if (!isFinite(dlo)) return;
    this._range(dlo, dhi);
    const lo = this.lo, hi = this.hi, step = this.step;
    const tEnd = d[d.length - 1].t, tStart = tEnd - this.win;
    const X = (t) => x0 + (t - tStart) / this.win * (x1 - x0);
    const Y = (v) => y1 - (v - lo) / (hi - lo) * (y1 - y0);
    const dec = Math.max(0, -Math.floor(Math.log10(step)) + 0);
    const fmt = (v) => (Math.abs(v) < 1e-9 ? '0' : v.toFixed(dec));

    // Gridlines at each increment (zero emphasised), with labels.
    c.font = '10px system-ui'; c.textAlign = 'right'; c.textBaseline = 'middle';
    for (let v = lo; v <= hi + 1e-9; v += step) {
      const yy = Y(v), zero = Math.abs(v) < 1e-9;
      c.strokeStyle = zero ? '#3a475a' : '#212c3b'; c.lineWidth = 1;
      c.beginPath(); c.moveTo(x0, yy + 0.5); c.lineTo(x1, yy + 0.5); c.stroke();
      c.fillStyle = '#8b98a9'; c.fillText(fmt(v), x0 - 5, yy);
    }

    c.save();
    c.beginPath(); c.rect(x0, y0, x1 - x0, y1 - y0); c.clip();
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

  arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.5, 0xf85149, 0.12, 0.08);
  arrow.visible = false;
  scene.add(arrow);

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
    mesh.userData.body = model.geom_bodyid[g];   // for raycast-pick shoving
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
    o.mesh.matrixWorld.copy(o.mesh.matrix);   // meshes are scene children -> world == local
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

// ---------------------------------------------------------------- WASD walking
const CMD = [0.8, 0.4, 0.8];               // vx, vy(strafe), yaw scales
function sendCommand() {
  const vx  = (keys['w'] ? 1 : 0) - (keys['s'] ? 1 : 0);
  const vy  = (keys['q'] ? 1 : 0) - (keys['e'] ? 1 : 0);   // strafe: body +y = left
  const yaw = (keys['a'] ? 1 : 0) - (keys['d'] ? 1 : 0);   // turn left = +yaw
  command = [vx * CMD[0], vy * CMD[1], yaw * CMD[2]];
  $('#t_cmd').textContent = command.map((x) => x.toFixed(1)).join(', ');
  cmd({ action: 'move', vx: command[0], vy: command[1], yaw: command[2] });
}

// ---------------------------------------------------------------- mouse shove
function pointerNDC(e) {
  const r = renderer.domElement.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1,
                           -((e.clientY - r.top) / r.height) * 2 + 1);
}
function screenToWorld(dx, dy) {            // screen px drag -> world horizontal direction
  const fwd = new THREE.Vector3();
  camera.getWorldDirection(fwd); fwd.z = 0;
  if (fwd.lengthSq() < 1e-6) return new THREE.Vector3();
  fwd.normalize();
  const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 0, 1), fwd).normalize();
  return new THREE.Vector3().addScaledVector(right, dx).addScaledVector(fwd, -dy);
}
function dragForce(e) {                     // -> {dir, force}
  const dx = e.clientX - drag.x0, dy = e.clientY - drag.y0;
  const dir = screenToWorld(dx, dy);
  const force = Math.min(600, Math.hypot(dx, dy) * 2.0);
  return { dir, force };
}
function onPointerDown(e) {
  if (e.button !== 0 || !model || switching) return;
  raycaster.setFromCamera(pointerNDC(e), camera);
  const hits = raycaster.intersectObjects(geomMeshes.map((o) => o.mesh), false);
  if (!hits.length) return;                 // empty space -> let OrbitControls orbit
  drag = { body: hits[0].object.userData.body, x0: e.clientX, y0: e.clientY, point: hits[0].point.clone() };
  controls.enabled = false;
  renderer.domElement.style.cursor = 'grabbing';
}
function onPointerMove(e) {
  if (!drag) return;
  const { dir, force } = dragForce(e);
  if (dir.lengthSq() < 1e-9 || force < 1) { arrow.visible = false; return; }
  dir.normalize();
  arrow.position.copy(drag.point);
  arrow.setDirection(dir);
  arrow.setLength(0.2 + force / 300, 0.1, 0.07);
  arrow.visible = true;
}
function onPointerUp() {
  if (!drag) return;
  const { dir, force } = dragForce({ clientX: lastPointer.x, clientY: lastPointer.y });
  if (dir.lengthSq() > 1e-9 && force >= 5) {
    dir.normalize();
    cmd({ action: 'push', body: drag.body, fx: dir.x * force, fy: dir.y * force });
  }
  drag = null; arrow.visible = false; controls.enabled = true;
  renderer.domElement.style.cursor = 'grab';
}
const lastPointer = { x: 0, y: 0 };

// ---------------------------------------------------------------- UI wiring
function fillSelect(sel, items, current) {
  sel.innerHTML = '';
  items.forEach((it) => sel.add(new Option(it, it)));
  if (current != null) sel.value = current;
}

function initUI(meta) {
  const robot = $('#robot'), imu = $('#imu'), policy = $('#policy');
  fillSelect(robot, meta.robots, meta.robot);
  fillSelect(imu, meta.imus, meta.imu);
  fillSelect(policy, meta.policies || [], meta.policy);

  tiltChart = new Strip($('#chart_tilt'), 4);     // min span ~4° so ideal IMU isn't over-zoomed
  gyroChart = new Strip($('#chart_gyro'), 0.2);   // min span ~0.2 rad/s
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
    const m = await (await fetch('/meta')).json();
    fillSelect(policy, m.policies || [], m.policy);
    tiltChart.reset(); gyroChart.reset();
    switching = false;
  };
  imu.onchange = () => cmd({ action: 'set_imu', imu: imu.value });
  policy.onchange = () => cmd({ action: 'set_policy', policy: policy.value });
  $('#reset').onclick = () => cmd({ action: 'reset' });

  // Keyboard: WASD/QE walk, R reset.
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === 'r') { cmd({ action: 'reset' }); return; }
    if ('wasdqe'.includes(k) && !keys[k]) { keys[k] = true; sendCommand(); }
  });
  window.addEventListener('keyup', (e) => {
    const k = e.key.toLowerCase();
    if ('wasdqe'.includes(k)) { keys[k] = false; sendCommand(); }
  });

  // Mouse: drag the robot to shove it; drag empty space orbits (OrbitControls).
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  window.addEventListener('pointermove', (e) => { lastPointer.x = e.clientX; lastPointer.y = e.clientY; onPointerMove(e); });
  window.addEventListener('pointerup', onPointerUp);
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
