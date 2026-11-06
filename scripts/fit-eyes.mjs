import { Matrix3, Matrix4, Vector3 } from 'three';
import { loadCharacter } from '../src/character.js';
import { eyeDepth } from '../src/eyes.js';
import { clamp, Performance } from '../src/performance.js';
import { ELEVATION, projectReferencePoint, STUDIO_LIGHTS } from '../src/studio.js';

const referenceView = new Vector3(0, Math.sin(ELEVATION), Math.cos(ELEVATION));
const directLights = [STUDIO_LIGHTS.key, STUDIO_LIGHTS.bounce].map(({direction, intensity}) => {
  const unit = new Vector3(...direction).normalize();
  return { direction: unit, half: unit.clone().add(referenceView).normalize(), intensity };
});

function fittedReflectance(shade, normal, material) {
  const srgb = shade / 255;
  const target = srgb < 0.04045 ? srgb / 12.92 : ((srgb + 0.055) / 1.055)**2.4;
  const dotView = Math.max(0, normal.dot(referenceView));
  const a2 = material.roughness**4;
  const f0 = ((material.ior - 1)/(material.ior + 1))**2 * material.specularIntensity;
  let diffuse = STUDIO_LIGHTS.ambient / Math.PI;
  let highlight = 0;
  for (const { direction, half, intensity } of directLights) {
    const dotLight = Math.max(0, normal.dot(direction));
    if (dotLight === 0) continue;
    const dotHalf = Math.max(0, normal.dot(half));
    const viewHalf = referenceView.dot(half);
    const schlick = 2**((-5.55473*viewHalf - 6.98316)*viewHalf);
    const fresnel = f0 + (material.specularIntensity - f0)*schlick;
    const visibility = 0.5 / Math.max(1e-6,
      dotLight*Math.sqrt(a2 + (1-a2)*dotView**2) + dotView*Math.sqrt(a2 + (1-a2)*dotLight**2));
    const distribution = a2 / (Math.PI * (1 + (a2-1)*dotHalf**2)**2);
    diffuse += intensity*dotLight / Math.PI;
    highlight += intensity*dotLight*fresnel*visibility*distribution;
  }
  return Number(clamp((target - highlight) / diffuse, 0.025, 0.4).toFixed(6));
}

function project(x, y, radius, transform, point) {
  point.set(x, y, eyeDepth(x, y, radius)).applyMatrix4(transform);
  return projectReferencePoint(point);
}

// Recover the face-local point on the sculpt whose projection matches the clip.
function fitPoint(px, py, radius, start, transform, point) {
  let [x, y] = start;
  const epsilon = 0.0001;
  for (let i = 0; i < 11; i++) {
    const [sx, sy] = project(x, y, radius, transform, point);
    const dx = px - sx;
    const dy = py - sy;
    if (Math.hypot(dx, dy) < 0.002) break;
    const [ux, uy] = project(x + epsilon, y, radius, transform, point);
    const [vx, vy] = project(x, y + epsilon, radius, transform, point);
    const ax = (ux - sx) / epsilon;
    const ay = (uy - sy) / epsilon;
    const bx = (vx - sx) / epsilon;
    const by = (vy - sy) / epsilon;
    const det = ax*by - ay*bx;
    if (Math.abs(det) < 5) break;
    x = clamp(x + clamp((dx*by - dy*bx)/det, -0.075, 0.075), -0.41, 0.41);
    y = clamp(y + clamp((ax*dy - ay*dx)/det, -0.075, 0.075), 2.06, 2.41);
  }
  return [Number(x.toFixed(6)), Number(y.toFixed(6))];
}

export async function fitReferenceEyes(model, cues, outlines) {
  const binary = model.buffer.slice(model.byteOffset, model.byteOffset + model.byteLength);
  const character = await loadCharacter(binary);
  const timeline = new Performance(cues, 0);
  const point = new Vector3();
  const transform = new Matrix4();
  const normalTransform = new Matrix3();
  const normal = new Vector3();
  const reflectance = [];
  let maxError = 0;
  const frames = outlines.frames.map((eyes, frame) => {
    character.update(frame * outlines.step, timeline);
    character.scene.updateMatrixWorld(true);
    const shades = [];
    reflectance.push(shades);
    return eyes.map((outline, side) => {
      const mesh = character.eyes[side];
      const index = mesh.skeleton.bones.indexOf(character.head);
      transform.multiplyMatrices(mesh.matrixWorld, mesh.bindMatrixInverse)
        .multiply(character.head.matrixWorld).multiply(mesh.skeleton.boneInverses[index])
        .multiply(mesh.bindMatrix);
      const local = [];
      let center = [(side === 0 ? -1 : 1)*0.242, 2.241];
      for (let i = 0; i < outline.length; i += 2) {
        const radius = i === 0 ? 0 : 1;
        // The decoded pixel lattice is centered half a pixel into the viewport.
        const [px, py] = [outline[i] + 0.5, outline[i + 1] + 0.5];
        const xy = fitPoint(px, py, radius, center, transform, point);
        const [sx, sy] = project(...xy, radius, transform, point);
        maxError = Math.max(maxError, Math.hypot(px - sx, py - sy));
        local.push(...xy);
        if (i === 0) center = xy;
      }
      const [cx, cy] = center;
      const depth = eyeDepth(cx, cy, 0);
      const delta = 0.0001;
      normal.set((depth - eyeDepth(cx+delta, cy, 0))/delta, (depth - eyeDepth(cx, cy+delta, 0))/delta, 1);
      normal.applyMatrix3(normalTransform.getNormalMatrix(transform)).normalize();
      shades.push(fittedReflectance(outlines.luminance[frame][side], normal, mesh.material));
      return local;
    });
  });
  console.log(`Fit ${frames.length} frames of eye contours to the model (max reprojection error ${maxError.toFixed(2)} px).`);
  return { step: outlines.step, segments: outlines.segments, frames, reflectance };
}
