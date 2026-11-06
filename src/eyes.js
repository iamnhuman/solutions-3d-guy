import { DynamicDrawUsage } from 'three';
import { clamp } from './performance.js';

// The front surface of the sculpt from scripts/build-character.py, in glTF axes.
export function eyeDepth(x, y, radius) {
  const h = (y - 2.239) / 0.513;
  const ring = Math.sqrt(Math.max(0, 1 - h*h)) * (1 + 0.016*h - 0.014*h*h);
  const depth = 0.463 * Math.sqrt(Math.max(0, ring*ring - (x/0.503)**2));
  const lift = 0.0015 + 0.0055 * (1 - radius**7);
  return depth - 0.005 + 0.024 * Math.max(-h, 0) + lift;
}

export class EyeMotion {
  constructor(meshes, curves) {
    this.curves = curves;
    this.eyes = meshes.map(mesh => {
      const { geometry } = mesh;
      const position = geometry.attributes.position;
      geometry.computeBoundingBox();
      const { min, max } = geometry.boundingBox;
      const radiusX = (max.x - min.x) / 2;
      const radiusY = (max.y - min.y) / 2;
      const radii = new Float32Array(position.count);
      const offsets = new Uint16Array(position.count);
      for (let i = 0; i < position.count; i++) {
        const x = (position.getX(i) - min.x) / radiusX - 1;
        const y = (position.getY(i) - min.y) / radiusY - 1;
        radii[i] = Math.min(1, Math.hypot(x, y));
        const segment = Math.round(Math.atan2(y, x) * curves.segments / (2*Math.PI));
        offsets[i] = 2 * (1 + (segment + curves.segments) % curves.segments);
      }
      position.setUsage(DynamicDrawUsage);
      geometry.attributes.normal.setUsage(DynamicDrawUsage);
      mesh.material = mesh.material.clone();
      return { geometry, position, radii, offsets, color: mesh.material.color };
    });
  }

  update(time) {
    const { step, frames, reflectance } = this.curves;
    const sample = Math.max(0, time / step);
    const index = Math.min(frames.length - 1, Math.floor(sample));
    const fraction = clamp(sample - index, 0, 1);
    const next = (index + 1) % frames.length;
    for (const [side, eye] of this.eyes.entries()) {
      const a = frames[index][side];
      const b = frames[next][side];
      const cx = a[0] + (b[0] - a[0]) * fraction;
      const cy = a[1] + (b[1] - a[1]) * fraction;
      const shade = reflectance[index][side];
      eye.color.setScalar(shade + (reflectance[next][side] - shade) * fraction);
      for (let i = 0; i < eye.position.count; i++) {
        const offset = eye.offsets[i];
        const radius = eye.radii[i];
        const borderX = a[offset] + (b[offset] - a[offset]) * fraction;
        const borderY = a[offset + 1] + (b[offset + 1] - a[offset + 1]) * fraction;
        const x = cx + (borderX - cx) * radius;
        const y = cy + (borderY - cy) * radius;
        eye.position.setXYZ(i, x, y, eyeDepth(x, y, radius));
      }
      eye.position.needsUpdate = true;
      eye.geometry.computeVertexNormals();
    }
  }
}
