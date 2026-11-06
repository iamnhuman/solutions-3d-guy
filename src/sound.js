import { clamp, cueTime, seededRandom, timeForCue } from './performance.js';

const TAU = 2 * Math.PI;
const LEAD_TIME = 0.032;

function sampleEnvelope(levels, position) {
  const index = Math.min(levels.length - 1, Math.floor(position));
  const next = Math.min(levels.length - 1, index + 1);
  return levels[index] + (levels[next] - levels[index]) * (position - index);
}

export class StudioSound {
  constructor(cues, performance) {
    this.cues = cues;
    this.performance = performance;
    this.context = null;
    this.bus = null;
    this.sources = new Set();
  }

  get enabled() {
    return this.bus !== null;
  }

  time() {
    return Math.max(0, this.context.currentTime - this.zero);
  }

  async start(readVisualTime) {
    if (!this.context) {
      const Audio = window.AudioContext ?? window.webkitAudioContext;
      if (!Audio) throw new Error('Web Audio is unavailable');
      this.context = new Audio();
      this.limiter = this.context.createDynamicsCompressor();
      this.limiter.threshold.value = -12;
      this.limiter.knee.value = 14;
      this.limiter.ratio.value = 4;
      this.limiter.attack.value = 0.008;
      this.limiter.release.value = 0.18;
      this.limiter.connect(this.context.destination);
    }
    await this.context.resume();
    if (this.context.state !== 'running') throw new Error('Audio context could not start');
    this.zero = this.context.currentTime - readVisualTime();
    this.bus = this.context.createGain();
    this.bus.gain.value = 0.66;
    this.bus.connect(this.limiter);
    this.nextTake = this.performance.takeAt(this.time()).index;
    this.schedule();
    this.timer = window.setInterval(() => this.schedule(), 125);
  }

  stop() {
    const time = this.time();
    if (!this.enabled) return time;
    window.clearInterval(this.timer);
    const now = this.context.currentTime;
    const bus = this.bus;
    bus.gain.setTargetAtTime(0.0001, now, 0.022);
    for (const source of this.sources) source.stop(now + 0.12);
    this.sources.clear();
    this.bus = null;
    window.setTimeout(() => bus.disconnect(), 180);
    return time;
  }

  register(source, ...nodes) {
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      source.disconnect();
      for (const node of nodes) node.disconnect();
    };
  }

  schedule() {
    if (!this.enabled || this.context.state !== 'running') return;
    const elapsed = this.time();
    const current = this.performance.takeAt(elapsed);
    this.nextTake = Math.max(this.nextTake, current.index);
    let take = this.performance.take(this.nextTake);
    while (take.start < elapsed + 0.7) {
      this.scheduleTake(take);
      take = this.performance.take(++this.nextTake);
    }
  }

  scheduleTake(take) {
    const context = this.context;
    const start = Math.max(this.zero + take.start, context.currentTime + LEAD_TIME);
    const end = this.zero + take.end;
    const length = end - start;
    if (length < 0.03) return;
    const { duration, audio } = this.cues;
    const random = seededRandom(take.seed ^ 0x53C96AFB);
    const points = Math.max(2, Math.ceil(length / 0.031));
    const envelopeTimes = new Float32Array(points);
    const fades = new Float32Array(points);
    for (let i = 0; i < points; i++) {
      const offset = length * i / (points - 1);
      envelopeTimes[i] = cueTime(take, start - this.zero + offset, duration) / audio.step;
      fades[i] = Math.min(1, offset / 0.061, (length - offset) / 0.095);
    }

    for (const [frequency, levels] of audio.tones) {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const pan = context.createStereoPanner();
      oscillator.frequency.value = frequency;
      oscillator.detune.value = take.cents + (random() - 0.5) * 0.45;
      const phase = random() * TAU;
      oscillator.setPeriodicWave(context.createPeriodicWave(
        new Float32Array([0, Math.cos(phase)]), new Float32Array([0, Math.sin(phase)]),
        { disableNormalization: true },
      ));
      pan.pan.value = frequency < 150 ? 0 : (random() - 0.5) * 0.66;
      const gainScale = take.audioLevel * (0.97 + random()*0.06);
      const curve = Float32Array.from(envelopeTimes,
        (time, i) => Math.max(0, sampleEnvelope(levels, time) * fades[i] * gainScale));
      gain.gain.setValueCurveAtTime(curve, start, length);
      oscillator.connect(gain).connect(pan).connect(this.bus);
      this.register(oscillator, gain, pan);
      oscillator.start(start);
      oscillator.stop(end + 0.002);
    }

    for (const [cue, level] of audio.accents) {
      const time = this.zero + timeForCue(take, cue, duration);
      if (time >= start - 0.01) this.scheduleAccent(Math.max(start, time), level, random);
    }
  }

  scheduleAccent(time, level, random) {
    const context = this.context;
    const length = 0.135;
    const burst = context.createBuffer(1, Math.ceil(context.sampleRate * length), context.sampleRate);
    const samples = burst.getChannelData(0);
    for (let i = 0; i < samples.length; i++) {
      const t = i / context.sampleRate;
      samples[i] = (2*random() - 1) * Math.min(t/0.003, 1) * Math.exp(-t/0.023);
    }
    const source = context.createBufferSource();
    const filter = context.createBiquadFilter();
    const gain = context.createGain();
    const pan = context.createStereoPanner();
    source.buffer = burst;
    filter.type = 'bandpass';
    filter.frequency.value = 2600 + 1000 * random();
    filter.Q.value = 0.68;
    gain.gain.value = 0.105 * level;
    pan.pan.value = (random() - 0.5) * 0.68;
    source.connect(filter).connect(gain).connect(pan).connect(this.bus);
    this.register(source, filter, gain, pan);
    source.start(time);

    for (const [frequency, decay, amplitude] of [[326, 0.12, 0.055], [641, 0.083, 0.032], [2327, 0.035, 0.015]]) {
      const ring = context.createOscillator();
      const volume = context.createGain();
      ring.frequency.value = frequency * (0.984 + 0.032*random());
      volume.gain.setValueAtTime(0, time);
      volume.gain.linearRampToValueAtTime(amplitude*level, time + 0.003);
      volume.gain.exponentialRampToValueAtTime(0.0001, time + decay);
      ring.connect(volume).connect(this.bus);
      this.register(ring, volume);
      ring.start(time);
      ring.stop(time + decay + 0.002);
    }
  }
}
