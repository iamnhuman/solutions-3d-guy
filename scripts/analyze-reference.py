"""Extract pose cues and oscillator envelopes from the local reference.

Requires FFmpeg and NumPy (available in Blender's Python). The output is
control data. No images, audio samples, or video are embedded in the page.
"""

from collections import deque
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FPS = 30
EYE_SEGMENTS = 56
CROP_X, CROP_Y, WIDTH, HEIGHT = 378, 392, 294, 268
DURATION = 9.033333333
ELEVATION = 0.185
# Closing tap, air, opening snap, and the pitched response after each blink.
BLINK_EVENTS = [[0.931, 0.977, 1.037, 1.163, 1.233], [7.031, 7.177, 7.237, 7.363, 7.398]]
PIVOT = np.array([0, 1.823, -0.006])
LANDMARKS = np.array([
    [0, 2.239, -0.005], [-0.242, 2.241, 0.405],
    [0.242, 2.241, 0.405], [0, 2.027, 0.423],
])


def raw_video(path):
    pixels = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(path), '-an',
        '-vf', f'crop={WIDTH}:{HEIGHT}:{CROP_X}:{CROP_Y},format=gray',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-',
    ])
    return np.frombuffer(pixels, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH)


def component_near_center(mask, distances):
    if not np.any(mask):
        return None
    nearest = int(np.argmin(np.where(mask, distances, 1e9)))
    start_y, start_x = divmod(nearest, mask.shape[1])
    if distances[start_y, start_x] > 8**2:
        return None
    todo = deque([(start_y, start_x)])
    mask[start_y, start_x] = False
    component = []
    rows, cols = mask.shape
    while todo:
        row, col = todo.popleft()
        component.append((row, col))
        for ny, nx in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
            if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx]:
                mask[ny, nx] = False
                todo.append((ny, nx))
    return np.asarray(component)


def dark_blob(frame, point, half_width, half_height, eye):
    """Follow a local dark facial mark and measure its centroid and span."""
    x, y = np.rint(point - [CROP_X, CROP_Y]).astype(int)
    x0, x1 = max(0, x - half_width), min(WIDTH, x + half_width + 1)
    y0, y1 = max(0, y - half_height), min(HEIGHT, y + half_height + 1)
    region = frame[y0:y1, x0:x1]
    if not region.size:
        return None
    yy, xx = np.indices(region.shape)
    distances = ((xx + x0 - x)**2 + (yy + y0 - y)**2)
    core = region[distances <= (5 if eye else 3)**2]
    if not core.size:
        return None
    dark = float(np.percentile(core, 30))
    bright = float(np.percentile(region, 72))
    if bright - dark < (18 if eye else 13):
        return None
    coords = component_near_center(region <= dark + min(28, 0.46*(bright - dark)), distances)
    if coords is None:
        return None
    rows, cols = region.shape
    h, w = np.ptp(coords, axis=0) + 1
    min_area, max_area = (250, 1950) if eye else (13, 400)
    if not min_area <= len(coords) <= max_area or h >= rows - 2 or w >= cols - 2:
        return None
    cy, cx = coords.mean(axis=0)
    return np.array([cx + x0 + CROP_X, cy + y0 + CROP_Y]), w, h


def eye_outline(frame, point):
    """Measure the half-contrast edge, including a closed lid a few pixels tall."""
    x, y = np.rint(point - [CROP_X, CROP_Y]).astype(int)
    left, top = x - 30, y - 30
    patch = frame[top:top+61, left:left+61].astype(float)
    yy, xx = np.indices(patch.shape)
    top_light = np.median(patch[:4], axis=0)
    bottom_light = np.median(patch[-4:], axis=0)
    background = top_light + (bottom_light - top_light) * yy / (patch.shape[0] - 1)
    contrast = background - patch
    peak = np.percentile(contrast[20:41, 17:44], 96)
    threshold = max(13, peak * 0.52)
    distance = (xx-30)**2 + (yy-30)**2
    region = ((xx-30)/29)**2 + ((yy-30)/29)**2 <= 1
    component = component_near_center((contrast >= threshold) & region, distance)
    if component is None or not 38 <= len(component) <= 2200:
        raise ValueError(f'No eye contour near {point.tolist()}')

    ey, ex = component.T
    cx, cy = ex.mean(), ey.mean()
    # Follow the slant of a narrow closed lid when sampling the outline.
    dx, dy = ex - cx, ey - cy
    slope = float(np.dot(dx, dy) / np.dot(dx, dx))
    rx = max(2, (ex.max()-ex.min()+1)/2)
    ry = max(1, (np.ptp(dy - slope*dx)+1)/2)
    angles = np.arange(EYE_SEGMENTS)*2*math.pi/EYE_SEGMENTS
    horizontal = rx * np.cos(angles)
    vertical = -ry * np.sin(angles) + slope*horizontal

    def sample(xs, ys):
        valid = (xs >= 0) & (xs < 60) & (ys >= 0) & (ys < 60)
        xs, ys = np.clip(xs, 0, 59.99), np.clip(ys, 0, 59.99)
        ix, iy = xs.astype(int), ys.astype(int)
        fx, fy = xs - ix, ys - iy
        above = contrast[iy, ix]*(1-fx) + contrast[iy, ix+1]*fx
        below = contrast[iy+1, ix]*(1-fx) + contrast[iy+1, ix+1]*fx
        return valid & (above*(1-fy) + below*fy >= threshold)

    low, high = np.zeros(EYE_SEGMENTS), np.full(EYE_SEGMENTS, 1.65)
    for _ in range(13):
        middle = (low + high) / 2
        inside = sample(cx + middle*horizontal, cy + middle*vertical)
        low, high = np.where(inside, middle, low), np.where(inside, high, middle)
    radius = (low + high) / 2
    center = np.array([cx + left + CROP_X, cy + top + CROP_Y])
    border = center + np.column_stack((horizontal, vertical))*radius[:, None]
    core = (dx/rx)**2 + ((dy - slope*dx)/ry)**2 <= 0.56**2
    shade = float(np.median(patch[ey[core], ex[core]])) if core.any() else float(np.median(patch[ey, ex]))
    return center, np.round(np.vstack((center, border)).ravel(), 3).tolist(), shade


def eye_cues(frames):
    centers = [np.array([462., 552.]), np.array([579., 553.])]
    outlines = []
    luminance = []
    for number, frame in enumerate(frames):
        row = []
        shades = []
        for i, center in enumerate(centers):
            try:
                centers[i], outline, shade = eye_outline(frame, center)
            except ValueError as error:
                raise ValueError(f'Frame {number}, eye {i + 1}: {error}') from error
            row.append(outline)
            shades.append(shade)
        outlines.append(row)
        luminance.append(shades)
    print(f'Measured both eye outlines at all {len(outlines)} frames ({FPS} fps).')
    return {'step': 1/FPS, 'segments': EYE_SEGMENTS, 'frames': outlines, 'luminance': luminance}


def head_center(frame):
    centers = []
    for row in frame[94:174:3]:
        xs = np.flatnonzero(row < 198)
        if len(xs) > 150:
            centers.append((xs[0] + xs[-1]) * 0.5)
    x = float(np.median(centers)) if centers else 521 - CROP_X
    mid = round(x)
    tops = np.flatnonzero((frame[:82, mid-32:mid+33] < 199).sum(axis=1) >= 9)
    top = tops[0] if len(tops) else 410 - CROP_Y
    return np.array([x + CROP_X, top + CROP_Y + 121.5])


def project_pose(params):
    yaw, pitch, roll, x, y = params
    sy, cy = math.sin(yaw), math.cos(yaw)
    sx, cx = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(roll), math.cos(roll)
    rotate = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([
        [1, 0, 0], [0, cx, -sx], [0, sx, cx],
    ]) @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    xx, yy, zz = ((LANDMARKS - PIVOT) @ rotate.T + PIVOT + [x, y, 0]).T
    return np.column_stack([240 * xx, -240 * (math.cos(ELEVATION)*yy - math.sin(ELEVATION)*zz)])


def fit_pose(delta, start):
    neutral = project_pose(np.zeros(5))
    weights = np.array([[0.90, 0.70], [1, 1], [1, 1], [0.50, 0.43]])

    def residual(params):
        error = ((project_pose(params) - neutral - delta) * weights).ravel()
        return np.append(error, params * [1.5, 9.0, 6.0, 37, 35])

    params = start.copy()
    for _ in range(5):
        value = residual(params)
        epsilon = 0.0001
        gradients = [(residual(params + basis*epsilon) - value)/epsilon for basis in np.eye(5)]
        step, *_ = np.linalg.lstsq(np.stack(gradients, axis=-1), -value, rcond=None)
        params += np.clip(step, -0.16, 0.16)
        params = np.clip(params, [-0.68, -0.23, -0.19, -0.115, -0.075],
                                 [0.68, 0.23, 0.19, 0.115, 0.075])
    return params


def smooth_signal(values, width):
    radius = int(width * 3)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / width)**2)
    kernel /= kernel.sum()
    return np.apply_along_axis(lambda x: np.convolve(np.pad(x, radius, mode='edge'), kernel, mode='valid'),
                               0, values)


def motion_cues(frames):
    eyes = [np.array([462.0, 552.0]), np.array([579.0, 552.0])]
    mouth = np.array([521.0, 603.0])
    last_mouth_height = 14
    poses = []
    params = np.zeros(5)
    failures = [0, 0, 0]
    for frame in frames:
        for i, previous in enumerate(eyes):
            mark = dark_blob(frame, previous, 28, 29, True)
            if mark is not None:
                eyes[i] = mark[0]
            else:
                failures[i] += 1
        midpoint = (eyes[0] + eyes[1]) * 0.5
        mouth_guess = mouth*0.65 + (midpoint + [0, 49])*0.35
        mark = dark_blob(frame, mouth_guess, 19, 17, False)
        if mark is not None:
            mouth, _, last_mouth_height = mark
        else:
            mouth = mouth_guess
            last_mouth_height = max(5, last_mouth_height - 1)
            failures[2] += 1
        observed = np.array([head_center(frame), *eyes, mouth])
        if not poses:
            neutral = observed.copy()
            neutral_height = last_mouth_height
        params = fit_pose(observed - neutral, params)
        jaw = np.clip((last_mouth_height - neutral_height) / max(6, neutral_height * 0.65), -1, 1)
        poses.append([*params, jaw])

    values = smooth_signal(np.array(poses), 2.2)
    values -= values[0]
    # Bridge the two neutral ends, preserving all of the visible turns in between.
    edge = np.clip((np.arange(len(values))[::-1] / FPS) / 0.52, 0, 1)
    values *= (edge*edge*(3 - 2*edge))[:, None]
    print(f'Tracked {len(values)} frames; local marks missed on {failures} frames (left eye, right eye, mouth).')
    print('Head cue range in degrees:', np.round(np.rad2deg(values[:, :3].min(axis=0)), 1),
          'to', np.round(np.rad2deg(values[:, :3].max(axis=0)), 1))
    return {'step': 0.1, 'frames': np.round(values[::3], 5).tolist()}


def blink_burst(signal, rate, onset, bands, duration, opening=False):
    """Fit ringing partials and their envelopes to one transient in the reference."""
    size = 16384
    frequencies = np.fft.rfftfreq(size, 1/rate)

    def spectrum(span):
        first, last = (round((onset + t)*rate) for t in span)
        samples = signal[first:last]
        window = np.hanning(len(samples))
        amplitude = 2*np.abs(np.fft.rfft(samples*window, size))/window.sum()
        peaks = np.flatnonzero((amplitude[1:-1] > amplitude[:-2]) & (amplitude[1:-1] >= amplitude[2:])) + 1
        return amplitude, peaks, samples*window, np.arange(first, last)/rate - onset

    primary = spectrum((0.002, 0.036) if opening else (0.001, 0.041))
    levels_at = np.linspace(0, duration, round(duration/0.0025)+1)
    half = round(rate * (0.008 if opening else 0.006))
    samples_at = np.arange(-half, half+1)
    local_window = np.hanning(len(samples_at))
    modes = []
    for lo, hi, *span in bands:
        amplitude, peaks, samples, times = spectrum(span) if span else primary
        candidates = peaks[(frequencies[peaks] >= lo) & (frequencies[peaks] <= hi)]
        if not len(candidates):
            continue
        peak = candidates[np.argmax(amplitude[candidates])]
        if amplitude[peak] < 0.015:
            continue
        a, b, c = np.log(amplitude[peak-1:peak+2] + 1e-14)
        fraction = np.clip(0.5*(a-c)/(a-2*b+c), -0.5, 0.5)
        frequency = frequencies[peak] + fraction*frequencies[1]
        phase = np.angle(np.sum(samples*np.exp(-2j*math.pi*frequency*times)))
        demod = local_window * np.exp(-2j*math.pi*frequency*samples_at/rate)

        def level(t):
            around = round((onset+t)*rate) + samples_at
            return 2*abs(np.dot(signal[around], demod))/local_window.sum()

        background = np.median([level(t) for t in (-0.061, -0.046, -0.031, -0.017)])
        envelope = np.array([level(t) for t in levels_at])
        envelope = np.sqrt(np.maximum(0, envelope**2 - background**2))
        edge = np.minimum(1, np.minimum(levels_at/0.004, (duration-levels_at)/0.017))
        modes.append([round(float(frequency), 2), round(float(phase), 4),
                      np.round(envelope*edge, 5).tolist()])
    result = {'time': onset, 'duration': duration, 'modes': modes}
    if opening:
        band = (frequencies > 1000) & (frequencies < 6500)
        energy = primary[0][band]**2
        result['noiseCenter'] = round(float(np.dot(frequencies[band], energy)/energy.sum()))
    return result


def blink_cues(signal, rate):
    events = []
    # The first response rises through three notes. The second follows a lower chord.
    phrases = [
        [(388,435,0.049,0.082), (530,590), (705,769,0.030,0.058),
         (790,850), (1075,1140), (1200,1290,0.050,0.080)],
        [(350,395), (445,485,0.045,0.079), (524,580), (668,712,0.044,0.075),
         (716,765), (900,962,0.044,0.072)],
    ]
    for i, (tap, brush, snap, response, phrase) in enumerate(BLINK_EVENTS):
        close = blink_burst(signal, rate, tap, [(300,360), (610,690), (920,1015), (1240,1360)], 0.087)
        opening = blink_burst(signal, rate, snap, [(480,630), (650,1050), (1320,1540), (1800,2120), (2300,3600)],
                              0.060, opening=True)
        chime = blink_burst(signal, rate, response, [(455,580), (1040,1260), (1360,1630)], 0.073)
        notes = blink_burst(signal, rate, phrase, phrases[i], 0.118)
        events.append({'close': close, 'brush': brush, 'open': opening, 'after': [chime, notes]})
        print(f'Blink {i+1} modes (Hz): close {[m[0] for m in close["modes"]]}, '
              f'open {[m[0] for m in opening["modes"]]}, response {[m[0] for m in notes["modes"]]}')
    return events


def audio_cues(video):
    sample_rate, fft_size, window_length, step = 22050, 16384, 4096, 0.067
    raw = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(video), '-vn', '-ac', '1', '-ar', str(sample_rate),
        '-f', 'f32le', '-',
    ])
    signal = np.frombuffer(raw, dtype='<f4').astype(float)
    blinks = blink_cues(signal, sample_rate)
    padded = np.pad(signal, window_length // 2, mode='edge')
    window = np.hanning(window_length)
    starts = np.minimum(len(signal) - 1, np.rint(np.arange(0, DURATION + step, step)*sample_rate)).astype(int)
    spectra = np.array([2 * np.abs(np.fft.rfft(padded[i:i+window_length]*window, fft_size)) / window.sum()
                        for i in starts])
    bins = np.fft.rfftfreq(fft_size, 1/sample_rate)
    # Choose the sustained voices from portions of the track without blink sounds.
    times = starts / sample_rate
    stable = np.ones(len(times), dtype=bool)
    for event in blinks:
        tail = event['after'][-1]
        stable &= (times < event['close']['time'] - 0.10) | (times > tail['time'] + tail['duration'] + 0.10)
    bursts = [burst for event in blinks for burst in [event['close'], event['open'], *event['after']]]
    average = np.mean(spectra[stable]**2, axis=0)
    peaks = np.flatnonzero((average[1:-1] > average[:-2]) & (average[1:-1] >= average[2:])) + 1
    peaks = [i for i in peaks if 39 < bins[i] < 4500]
    carriers = []
    for index in sorted(peaks, key=lambda i: average[i], reverse=True):
        f = bins[index]
        if all(abs(f - bins[i]) > max(10.2, f*0.008) for i in carriers):
            carriers.append(index)
        if len(carriers) >= 32:
            break

    tones = []
    for index in sorted(carriers):
        left, middle, right = np.log(average[index-1:index+2] + 1e-16)
        fraction = np.clip(0.5 * (left - right) / (left - 2*middle + right), -0.5, 0.5)
        frequency = bins[index] + fraction * bins[1]
        clean = np.ones(len(times), dtype=bool)
        for burst in bursts:
            if any(abs(mode[0] - frequency) < (13 if frequency < 400 else 20) for mode in burst['modes']):
                clean &= (times < burst['time'] - 0.09) | (times > burst['time'] + burst['duration'] + 0.09)
        envelope = smooth_signal(np.interp(times, times[clean], spectra[clean, index]), 0.83)
        envelope = np.maximum(0, envelope - 0.00085)
        tones.append([round(float(frequency), 3), np.round(envelope, 5).tolist()])
    print('Resynthesis carriers (Hz):', ', '.join(str(frequency) for frequency, _ in tones))
    return {'step': step, 'tones': tones, 'blinks': blinks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', nargs='?', type=Path)
    args = parser.parse_args()
    video = args.video or next(ROOT.glob('*.mp4'))
    frames = raw_video(video)
    eyes = eye_cues(frames)
    data = {'duration': DURATION, 'motion': motion_cues(frames), 'audio': audio_cues(video)}
    output = ROOT / 'assets' / 'reference-cues.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(data, separators=(',', ':')) + '\n', encoding='utf-8')
    output.with_name('eye-contours.json').write_text(json.dumps(eyes, separators=(',', ':')) + '\n',
                                                    encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
