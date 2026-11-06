import { Euler, Quaternion } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { makeDabPose } from './dab.js';
import { EyeMotion } from './eyes.js';
import { clamp } from './performance.js';

export async function loadCharacter(buffer, eyeCurves) {
  const { scene } = await new GLTFLoader().parseAsync(buffer, '');
  return new ClayCharacter(scene, eyeCurves);
}

class ClayCharacter {
  constructor(scene, eyeCurves) {
    this.scene = scene;
    const parts = new Map();
    scene.traverse(object => {
      const name = object.userData.name ?? object.name;
      if (name) parts.set(name, object);
      if (object.isMesh) {
        object.castShadow = !/^(Eye|Mouth)/.test(name);
        object.receiveShadow = true;
        if (object.isSkinnedMesh) object.frustumCulled = false;
      }
    });
    const part = name => {
      const object = parts.get(name);
      if (!object) throw new Error(`Missing character part ${name}`);
      return object;
    };
    this.pelvis = part('Pelvis');
    this.chest = part('Chest');
    this.head = part('Head');
    this.headRest = this.head.position.clone();
    this.arms = ['L', 'R'].map(side => [part(`Shoulder.${side}`), part(`Wrist.${side}`)]);
    this.dab = makeDabPose(this.headRest, this.arms);
    this.body = part('Body');
    this.mouth = part('Mouth');
    this.eyes = [part('Eye.L'), part('Eye.R')];
    this.eyeMotion = eyeCurves ? new EyeMotion(this.eyes, eyeCurves) : null;
    this.pose = new Float64Array(6);
    this.delayed = new Float64Array(6);
    this.angles = new Euler(0, 0, 0, 'YXZ');
    this.worldRotation = new Quaternion();
  }

  update(time, performance, dab = 0) {
    const [yaw, pitch, roll, x, y, mouth] = performance.poseAt(time, this.pose);
    const [previousYaw, previousPitch] = performance.poseAt(Math.max(0, time - 0.18), this.delayed);
    const breath = Math.sin(time * 1.53);
    const drift = Math.sin(time * 0.81) * Math.sin(time * 0.37);
    this.pelvis.rotation.set(0, previousYaw * 0.014, drift * 0.004, 'YXZ');
    this.chest.rotation.set(breath * 0.005 + previousPitch * 0.075,
      previousYaw * 0.043, roll * 0.074 + drift * 0.004, 'YXZ');
    this.head.position.copy(this.headRest);
    this.head.position.x += x - drift * 0.006;
    this.head.position.y += y + breath * 0.0016;
    this.angles.set(pitch + breath * 0.002, yaw, roll, 'YXZ');
    this.worldRotation.setFromEuler(this.angles);
    this.head.quaternion.copy(this.pelvis.quaternion).multiply(this.chest.quaternion)
      .invert().multiply(this.worldRotation);

    this.pelvis.quaternion.slerp(this.dab.pelvis, dab);
    this.chest.quaternion.slerp(this.dab.chest, dab);
    this.head.position.lerp(this.dab.headPosition, dab);
    this.head.quaternion.slerp(this.dab.head, dab);

    for (const [i, [shoulder, wrist]] of this.arms.entries()) {
      const side = i === 0 ? -1 : 1;
      const lag = yaw - previousYaw;
      const response = Math.sin(time * 1.53 - 0.43 + i * 0.35);
      shoulder.rotation.set(response * 0.008 - previousPitch * 0.048,
        -0.011 * previousYaw, side * (response * 0.005 + lag * 0.043), 'YXZ');
      wrist.rotation.set(-response * 0.007 + lag * 0.022,
        side * breath * 0.005, -side * lag * 0.034, 'YXZ');
      const target = this.dab.arms[i];
      shoulder.position.copy(target.restPosition).lerp(target.position, dab);
      shoulder.quaternion.slerp(target.shoulder, dab);
      wrist.quaternion.slerp(target.wrist, dab);
      wrist.scale.y = target.restWristScale * (1 + (target.stretch - 1) * dab);
    }

    this.body.morphTargetInfluences[this.body.morphTargetDictionary.Breath] = 0.31 + 0.29 * breath;
    this.mouth.morphTargetInfluences[this.mouth.morphTargetDictionary.MouthNarrow] = clamp(-mouth, 0, 1);
    this.mouth.morphTargetInfluences[this.mouth.morphTargetDictionary.MouthRound] = clamp(mouth, 0, 1);
    this.eyeMotion?.update(performance.sourceTime(time));
  }
}
