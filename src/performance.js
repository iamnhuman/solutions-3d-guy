export const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let n = Math.imul(state ^ (state >>> 15), state | 1);
    n ^= n + Math.imul(n ^ (n >>> 7), n | 61);
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
  };
}

const range = (random, min, max) => min + (max - min) * random();

// A gentle time warp changes the length of the holds without a kink at the join.
export function cueTime(take, elapsed, duration) {
  const u = clamp((elapsed - take.start) / take.duration, 0, 1);
  const window = Math.sin(Math.PI * u) ** 2;
  return (u + take.warp * window * Math.sin(2 * Math.PI * u)) * duration;
}

export function timeForCue(take, cue, duration) {
  let low = take.start;
  let high = take.end;
  for (let i = 0; i < 18; i++) {
    const center = (low + high) * 0.5;
    if (cueTime(take, center, duration) < cue) low = center;
    else high = center;
  }
  return (low + high) * 0.5;
}

function cubic(p0, p1, p2, p3, t) {
  return p1 + 0.5 * t * (p2 - p0 + t * (2*p0 - 5*p1 + 4*p2 - p3 + t * (3*(p1 - p2) + p3 - p0)));
}

export class Performance {
  constructor(cues, seed) {
    this.cues = cues;
    this.seed = seed;
    this.takes = [];
    this.take(0);
  }

  take(index) {
    while (this.takes.length <= index) {
      const i = this.takes.length;
      const seed = (this.seed ^ Math.imul(i + 1, 0x9E3779B1)) >>> 0;
      const random = seededRandom(seed);
      const start = i === 0 ? 0 : this.takes[i - 1].end;
      const duration = this.cues.duration * (i === 0 ? 1 : range(random, 0.948, 1.067));
      this.takes.push({
        index: i, seed, start, duration, end: start + duration,
        warp: i === 0 ? 0 : range(random, -0.023, 0.023),
        yaw: i === 0 ? 1 : range(random, 0.9, 1.075),
        pitch: i === 0 ? 1 : range(random, 0.93, 1.07),
        bias: i === 0 ? 0 : range(random, -0.027, 0.027),
        audioLevel: range(random, 0.94, 1.045),
        cents: range(random, -0.75, 0.75),
      });
    }
    return this.takes[index];
  }

  takeAt(elapsed) {
    let last = this.takes.length - 1;
    while (elapsed >= this.takes[last].end) this.take(++last);
    let low = 0;
    let high = last;
    while (low < high) {
      const mid = (low + high) >>> 1;
      if (elapsed < this.takes[mid].end) high = mid;
      else low = mid + 1;
    }
    return this.takes[low];
  }

  poseAt(elapsed, out) {
    const take = this.takeAt(elapsed);
    const { step, frames } = this.cues.motion;
    const maxIndex = frames.length - 1;
    const time = cueTime(take, elapsed, this.cues.duration);
    const frame = Math.min(maxIndex, time / step);
    const i = Math.floor(frame);
    const a = frames[Math.max(0, i - 1)];
    const b = frames[i];
    const c = frames[Math.min(maxIndex, i + 1)];
    const d = frames[Math.min(maxIndex, i + 2)];
    for (let axis = 0; axis < out.length; axis++) {
      out[axis] = cubic(a[axis], b[axis], c[axis], d[axis], frame - i);
    }
    const window = Math.sin(Math.PI * (elapsed - take.start) / take.duration) ** 2;
    out[0] = out[0] * take.yaw + take.bias * window;
    out[1] *= take.pitch;
    return out;
  }
}
