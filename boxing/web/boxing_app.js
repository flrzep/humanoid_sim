// Browser client for the G1 boxing game.
//
// Same render approach as the IMU demo: Python runs the two-fighter physics and
// streams qpos over SSE; here we compile the identical combined model in MuJoCo-WASM
// (mujoco-js) and render it with three.js (mj_forward on each streamed qpos). Input
// is two-player: keyboard for both, plus a PS4 gamepad per player (Gamepad API).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from 'mujoco-js';

const $ = (s) => document.querySelector(s);
const HIDDEN_GROUPS = new Set([0, 3]);     // collision geoms

let mujoco, model, data;
let scene, camera, renderer, controls;
let geomMeshes = [];
let wins = [0, 0], lastStatus = 'fighting', round = 1;

const MOVE = [0.8, 0.4, 0.8];              // vx, vy(strafe), yaw scales
const lastSent = [null, null];             // last move command per player

async function cmd(obj) {
  try {
    await fetch('/command', { method: 'POST',
      headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(obj) });
  } catch (e) { /* ignore */ }
}
function overlay(msg) {
  const el = $('#overlay');
  if (msg) { $('#ovmsg').textContent = msg; el.classList.remove('hidden'); }
  else el.classList.add('hidden');
}

// ---------------------------------------------------------------- three.js
function initThree() {
  const canvas = $('#canvas');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1419);
  camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.01, 200);
  camera.up.set(0, 0, 1);
  camera.position.set(0.2, -4.2, 1.6);     // side-on view of the two fighters
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.85);
  controls.enableDamping = true;

  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const key = new THREE.DirectionalLight(0xffffff, 0.95); key.position.set(2, -3, 5); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.3); fill.position.set(-3, 2, 2); scene.add(fill);
  const grid = new THREE.GridHelper(12, 24, 0x2b3647, 0x1a2230);
  grid.rotateX(Math.PI / 2); scene.add(grid);

  resize(); window.addEventListener('resize', resize);
}
function resize() {
  const c = $('#canvas'), w = c.clientWidth, h = c.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
function animate() {
  requestAnimationFrame(animate);
  pollGamepads();
  controls.update();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- geometry
function makeGeometry(g) {
  const t = model.geom_type[g], s = model.geom_size;
  const sx = s[g * 3], sy = s[g * 3 + 1], sz = s[g * 3 + 2];
  switch (t) {
    case 0: return null;
    case 2: return new THREE.SphereGeometry(sx, 24, 16);
    case 3: { const geo = new THREE.CapsuleGeometry(sx, 2 * sy, 8, 16); geo.rotateX(Math.PI / 2); return geo; }
    case 4: { const geo = new THREE.SphereGeometry(1, 24, 16); geo.scale(sx, sy, sz); return geo; }
    case 5: { const geo = new THREE.CylinderGeometry(sx, sx, 2 * sy, 24); geo.rotateX(Math.PI / 2); return geo; }
    case 6: return new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz);
    case 7: return makeMeshGeometry(g);
    default: return null;
  }
}
function makeMeshGeometry(g) {
  const mid = model.geom_dataid[g];
  if (mid < 0) return null;
  const v0 = model.mesh_vertadr[mid], vn = model.mesh_vertnum[mid];
  const f0 = model.mesh_faceadr[mid], fn = model.mesh_facenum[mid];
  const verts = Float32Array.from(model.mesh_vert.subarray(v0 * 3, (v0 + vn) * 3));
  const faces = Uint32Array.from(model.mesh_face.subarray(f0 * 3, (f0 + fn) * 3));
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setIndex(new THREE.BufferAttribute(faces, 1));
  geo.computeVertexNormals();
  return geo;
}
function geomTint(g) {
  // Tint each fighter so they're distinguishable: r1 -> blue, r2 -> orange.
  // World body is 0; the two identical robots split the remaining bodies evenly.
  const body = model.geom_bodyid[g];
  const perRobot = (model.nbody - 1) / 2;
  if (body >= 1 && body <= perRobot) return new THREE.Color(0.30, 0.55, 1.0);
  if (body > perRobot) return new THREE.Color(0.96, 0.53, 0.22);
  return new THREE.Color(0.7, 0.7, 0.74);
}
function buildScene() {
  for (const o of geomMeshes) { scene.remove(o.mesh); o.mesh.geometry.dispose(); o.mesh.material.dispose(); }
  geomMeshes = [];
  for (let g = 0; g < model.ngeom; g++) {
    if (HIDDEN_GROUPS.has(model.geom_group[g])) continue;
    const geo = makeGeometry(g);
    if (!geo) continue;
    const mat = new THREE.MeshStandardMaterial({ color: geomTint(g), metalness: 0.15, roughness: 0.7 });
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
      xmat[r], xmat[r + 1], xmat[r + 2], xpos[p],
      xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
      xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
      0, 0, 0, 1);
  }
}

// ---------------------------------------------------------------- model load
function writeFileMkdir(path, bytes) {
  const parts = path.split('/').filter(Boolean);
  let cur = '';
  for (let i = 0; i < parts.length - 1; i++) { cur += '/' + parts[i]; try { mujoco.FS.mkdir(cur); } catch (e) {} }
  mujoco.FS.writeFile(path, bytes);
}
async function loadModel() {
  overlay('Loading arena…');
  const man = await (await fetch('/model/manifest')).json();
  await Promise.all(man.files.map(async (rel) => {
    const buf = await (await fetch('/model/file?path=' + encodeURIComponent(rel))).arrayBuffer();
    writeFileMkdir('/work/' + rel, new Uint8Array(buf));
  }));
  model = mujoco.MjModel.loadFromXML('/work/' + man.scene);
  data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);
  buildScene();
  overlay('');
}

// ---------------------------------------------------------------- pose stream
function startStream() {
  const es = new EventSource('/poses');
  es.onmessage = (e) => {
    let msg; try { msg = JSON.parse(e.data); } catch (_) { return; }
    if (model && Array.isArray(msg.qpos) && msg.qpos.length === model.nq) {
      const q = data.qpos;
      for (let i = 0; i < q.length; i++) q[i] = msg.qpos[i];
      mujoco.mj_forward(model, data);
      syncPoses();
    }
    if (msg.tel) updateGame(msg.tel);
  };
}
function updateGame(tel) {
  if (!tel || !tel.status) return;
  if (tel.status === 'ko' && lastStatus !== 'ko') {
    if (tel.winner === 0) wins[0]++; else if (tel.winner === 1) wins[1]++;
    $('#w1').textContent = wins[0];
    $('#w2').textContent = wins[1];
    const txt = tel.winner == null ? 'DOUBLE KO — DRAW'
      : (tel.winner === 0 ? 'PLAYER 1 WINS' : 'PLAYER 2 WINS');
    $('#kotext').textContent = txt;
    $('#ko').classList.remove('hidden');
  } else if (tel.status === 'fighting' && lastStatus === 'ko') {
    $('#ko').classList.add('hidden');
  }
  lastStatus = tel.status;
}
function rematch() {
  round++; $('#round').textContent = 'Round ' + round;
  $('#ko').classList.add('hidden');
  cmd({ action: 'reset' });
}

// ---------------------------------------------------------------- input
function sendMove(player, vx, vy, yaw) {
  const key = `${vx.toFixed(2)},${vy.toFixed(2)},${yaw.toFixed(2)}`;
  if (lastSent[player] === key) return;
  lastSent[player] = key;
  cmd({ action: 'move', player, vx, vy, yaw });
}

// keyboard state
const keys = {};
const KEYMAP = [
  { fwd: 'w', back: 's', left: 'a', right: 'd', sl: 'q', sr: 'e', pl: 'f', pr: 'g' },
  { fwd: 'arrowup', back: 'arrowdown', left: 'arrowleft', right: 'arrowright', sl: ',', sr: '.', pl: 'k', pr: 'l' },
];
function keyboardMove(player) {
  const m = KEYMAP[player];
  const vx = (keys[m.fwd] ? 1 : 0) - (keys[m.back] ? 1 : 0);
  const vy = (keys[m.sl] ? 1 : 0) - (keys[m.sr] ? 1 : 0);
  const yaw = (keys[m.left] ? 1 : 0) - (keys[m.right] ? 1 : 0);
  sendMove(player, vx * MOVE[0], vy * MOVE[1], yaw * MOVE[2]);
}
function initKeyboard() {
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === 'enter' || k === 'r') { rematch(); return; }
    for (let p = 0; p < 2; p++) {
      const m = KEYMAP[p];
      if (k === m.pl && !keys[k]) cmd({ action: 'punch', player: p, side: 1 });
      if (k === m.pr && !keys[k]) cmd({ action: 'punch', player: p, side: -1 });
      if ([m.fwd, m.back, m.left, m.right, m.sl, m.sr].includes(k) && !keys[k]) { keys[k] = true; keyboardMove(p); }
    }
  });
  window.addEventListener('keyup', (e) => {
    const k = e.key.toLowerCase();
    for (let p = 0; p < 2; p++) {
      const m = KEYMAP[p];
      if ([m.fwd, m.back, m.left, m.right, m.sl, m.sr].includes(k)) { keys[k] = false; keyboardMove(p); }
    }
  });
}

// PS4 gamepads: pad index i -> player i. Standard mapping.
const prevButtons = [[], []];
const DZ = 0.18;
function pollGamepads() {
  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  for (let p = 0; p < 2; p++) {
    const gp = pads[p];
    $('#c' + (p + 1)).textContent = gp ? '— gamepad ✓' : (p === 0 ? '— keyboard' : '— keyboard / gamepad');
    if (!gp) continue;
    const dz = (v) => (Math.abs(v) < DZ ? 0 : v);
    const lx = dz(gp.axes[0] || 0), ly = dz(gp.axes[1] || 0), rx = dz(gp.axes[2] || 0);
    sendMove(p, -ly * MOVE[0], -lx * MOVE[1], -rx * MOVE[2]);   // stick: up=fwd, left=strafe, right-stick=turn
    const b = gp.buttons.map((x) => x.pressed);
    const pb = prevButtons[p];
    const edge = (i) => b[i] && !pb[i];
    if (edge(4) || edge(2)) cmd({ action: 'punch', player: p, side: 1 });   // L1 / Square = left
    if (edge(5) || edge(1)) cmd({ action: 'punch', player: p, side: -1 });  // R1 / Circle = right
    if (edge(9)) rematch();                                                 // Options = rematch
    prevButtons[p] = b;
  }
}

// ---------------------------------------------------------------- boot
async function main() {
  initThree();
  animate();
  initKeyboard();
  $('#restart').onclick = rematch;
  overlay('Loading MuJoCo WASM…');
  mujoco = await loadMujoco();
  mujoco.FS.mkdir('/work');
  mujoco.FS.mount(mujoco.MEMFS, { root: '.' }, '/work');
  await loadModel();
  startStream();
}
main().catch((err) => { console.error(err); overlay('Failed to start: ' + (err.message || err)); });
