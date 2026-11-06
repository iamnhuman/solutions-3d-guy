import { seededRandom } from './performance.js';

export function createBlinkRoom(context) {
  const rate = context.sampleRate;
  const impulse = context.createBuffer(2, Math.ceil(rate * 0.43), rate);
  for (let channel = 0; channel < 2; channel++) {
    const random = seededRandom(0x65F3A1 + 7919*channel);
    const samples = impulse.getChannelData(channel);
    const tailStart = Math.round(rate * (0.024 + channel * 0.004));
    let damped = 0;
    let energy = 0;
    for (let i = tailStart; i < samples.length; i++) {
      const t = (i - tailStart) / rate;
      damped = 0.66*damped + 0.34*(2*random() - 1);
      samples[i] = damped * Math.min(t / 0.011, 1) * Math.exp(-t / 0.047);
      energy += samples[i] ** 2;
    }
    const tailLevel = 0.28 / Math.sqrt(energy);
    for (let i = tailStart; i < samples.length; i++) samples[i] *= tailLevel;
    for (const [delay, gain] of [[0.014, 0.23], [0.031, -0.16], [0.057, 0.12], [0.086, 0.083], [0.124, -0.044]]) {
      const offset = Math.round(rate * (delay + (channel ? 0.0027 : 0)));
      samples[offset] += gain;
    }
  }
  const room = context.createConvolver();
  room.normalize = false;
  room.buffer = impulse;
  return room;
}
