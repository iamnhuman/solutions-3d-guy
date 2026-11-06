import { Euler, Quaternion, Vector3 } from 'three';
import { clamp } from './performance.js';

export const DAB_DURATION = 1.68;

function ease(value) {
  const t = clamp(value, 0, 1);
  return t ** 3 * (10 + t * (6 * t - 15));
}

export function dabWeight(seconds) {
  return ease(seconds / 0.46) * ease((DAB_DURATION - seconds) / 0.49);
}

function rotation(pitch, yaw, roll) {
  return new Quaternion().setFromEuler(new Euler(pitch, yaw, roll, 'YXZ'));
}

export function makeDabPose(headRest, arms) {
  const armTargets = [
    // Left elbow comes out and forward, with the forearm folded across the face.
    { upper: [-0.84, 0.18, 0.51], forearm: [0.82, 0.31, 0.48], offset: [-0.023, 0.05, 0.146], stretch: 1.43 },
    { upper: [0.73, 0.68, 0.07], forearm: [0.70, 0.71, 0.04], offset: [-0.016, 0.035, 0], stretch: 1 },
  ];

  return {
    pelvis: rotation(0.016, 0.045, 0.030),
    chest: rotation(0.038, 0.050, 0.032),
    head: rotation(0.57, -0.48, 0.10),
    headPosition: headRest.clone().add(new Vector3(0.036, -0.031, -0.061)),
    arms: arms.map(([shoulder, wrist], index) => {
      const { upper, forearm, offset, stretch } = armTargets[index];
      const side = index === 0 ? -1 : 1;
      const upperRotation = new Quaternion().setFromUnitVectors(
        wrist.position.clone().normalize(), new Vector3(...upper).normalize(),
      );
      // Both rest axes are parallel. Convert the forearm aim to shoulder space.
      const forearmRotation = new Quaternion().setFromUnitVectors(
        new Vector3(side * 0.041, -0.398, 0.038).normalize(), new Vector3(...forearm).normalize(),
      );
      return {
        shoulder: upperRotation,
        wrist: upperRotation.clone().invert().multiply(forearmRotation),
        restPosition: shoulder.position.clone(),
        position: shoulder.position.clone().add(new Vector3(...offset)),
        restWristScale: wrist.scale.y,
        stretch,
      };
    }),
  };
}
