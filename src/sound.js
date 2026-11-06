import { createBlinkRoom } from './blink-room.js';
import { clamp, cueTime, seededRandom, timeForCue } from './performance.js';

const TAU = 2 * Math.PI;
const LEAD_TIME = 0.032;
const JOIN_DURATION = 0.16;

function smoothMix(value) {
  const t = clamp(value, 0, 1);
  return t*t*(3 - 2*t);
}

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
      this.limiter.threshold.value = -5;
      this.limiter.knee.value = 6;
      this.limiter.ratio.value = 3;
      this.limiter.attack.value = 0.004;
      this.limiter.release.value = 0.075;
      this.limiter.connect(this.context.destination);
    }
    await this.context.resume();
    if (this.context.state !== 'running') throw new Error('Audio context could not start');
    this.zero = this.context.currentTime - readVisualTime();
    this.audioStart = this.context.currentTime + LEAD_TIME;
    this.bus = this.context.createGain();
    this.bus.gain.value = 0.66;
    this.bus.connect(this.limiter);
    this.blinkRoom = createBlinkRoom(this.context);
    this.blinkRoom.connect(this.bus);
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
    const room = this.blinkRoom;
    bus.gain.setTargetAtTime(0.0001, now, 0.022);
    for (const source of this.sources) source.stop(now + 0.12);
    this.sources.clear();
    this.bus = null;
    window.setTimeout(() => {
      room?.disconnect();
      bus.disconnect();
    }, 180);
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
    const halfJoin = JOIN_DURATION / 2;
    const fadeInFrom = take.start - halfJoin;
    const fadeOutFrom = take.end - halfJoin;
    const start = Math.max(this.zero + (take.index === 0 ? take.start : fadeInFrom), context.currentTime + LEAD_TIME);
    const end = this.zero + take.end + halfJoin;
    const length = end - start;
    if (length < 0.03) return;
    const { duration, audio } = this.cues;
    const random = seededRandom(take.seed ^ 0x53C96AFB);
    const timbre = seededRandom(this.performance.seed ^ 0x293C982D);
    const points = Math.max(2, Math.ceil(length / 0.031));
    const envelopeTimes = new Float32Array(points);
    const fades = new Float32Array(points);
    for (let i = 0; i < points; i++) {
      const offset = length * i / (points - 1);
      const elapsed = start + offset - this.zero;
      envelopeTimes[i] = clamp(cueTime(take, elapsed, duration), audio.step, duration - audio.step) / audio.step;
      const fadeIn = take.index === 0 ? 1 : smoothMix((elapsed - fadeInFrom) / JOIN_DURATION);
      const fadeOut = 1 - smoothMix((elapsed - fadeOutFrom) / JOIN_DURATION);
      fades[i] = fadeIn * fadeOut * smoothMix((start + offset - this.audioStart) / 0.052);
    }

    for (const [frequency, levels] of audio.tones) {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const pan = context.createStereoPanner();
      oscillator.frequency.value = frequency;
      // Matching phase and pan across the overlap keeps identical carriers from cancelling.
      const phase = TAU * (timbre() + ((start - this.zero)*frequency % 1));
      oscillator.setPeriodicWave(context.createPeriodicWave(
        new Float32Array([0, Math.cos(phase)]), new Float32Array([0, -Math.sin(phase)]),
        { disableNormalization: true },
      ));
      pan.pan.value = frequency < 150 ? 0 : (timbre() - 0.5) * 0.66;
      const gainScale = take.audioLevel * (0.97 + random()*0.06);
      const curve = Float32Array.from(envelopeTimes,
        (time, i) => Math.max(0, sampleEnvelope(levels, time) * fades[i] * gainScale));
      gain.gain.setValueCurveAtTime(curve, start, length);
      oscillator.connect(gain).connect(pan).connect(this.bus);
      this.register(oscillator, gain, pan);
      oscillator.start(start);
      oscillator.stop(end + 0.002);
    }

    const eventTime = cue => {
      const time = this.zero + timeForCue(take, cue, duration);
      return time >= start - 0.01 ? Math.max(time, start) : null;
    };
    const detune = take.index === 0 ? 0 : take.cents;
    for (const { close, brush, open, after } of audio.blinks) {
      const brushAt = eventTime(brush);
      const snapAt = eventTime(open.time);
      if (brushAt !== null) this.blinkNoise(brushAt, 0.083, 0.030, 0.013, 7900, 0.047, random);
      if (snapAt !== null) this.blinkNoise(snapAt, 0.057, 0.011, 0.008, open.noiseCenter, 0.24, random);
      for (const burst of [close, open, ...after]) {
        const time = eventTime(burst.time);
        if (time !== null) this.blinkRings(time, burst, detune);
      }
    }
  }

  blinkNoise(time, length, attack, decay, frequency, amplitude, random) {
    const context = this.context;
    const burst = context.createBuffer(1, Math.ceil(context.sampleRate * length), context.sampleRate);
    const samples = burst.getChannelData(0);
    for (let i = 0; i < samples.length; i++) {
      const t = i / context.sampleRate;
      const envelope = Math.min(t / attack, 1) * Math.exp(-Math.max(0, t - attack) / decay);
      samples[i] = (2*random() - 1) * envelope;
    }
    const source = context.createBufferSource();
    const filter = context.createBiquadFilter();
    const gain = context.createGain();
    source.buffer = burst;
    filter.type = 'bandpass';
    filter.frequency.value = frequency;
    filter.Q.value = 0.61;
    gain.gain.value = amplitude;
    source.connect(filter).connect(gain).connect(this.bus);
    gain.connect(this.blinkRoom);
    this.register(source, filter, gain);
    source.start(time);
  }

  blinkRings(time, burst, detune) {
    const context = this.context;
    for (const [frequency, phase, levels] of burst.modes) {
      const ring = context.createOscillator();
      const volume = context.createGain();
      ring.frequency.value = frequency;
      ring.detune.value = detune;
      ring.setPeriodicWave(context.createPeriodicWave(
        new Float32Array([0, Math.cos(phase)]), new Float32Array([0, -Math.sin(phase)]),
        { disableNormalization: true },
      ));
      volume.gain.setValueCurveAtTime(Float32Array.from(levels), time, burst.duration);
      ring.connect(volume).connect(this.bus);
      volume.connect(this.blinkRoom);
      this.register(ring, volume);
      ring.start(time);
      ring.stop(time + burst.duration + 0.002);
    }
  }
}
