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
CROP_X, CROP_Y, WIDTH, HEIGHT = 378, 392, 294, 268
DURATION = 9.033333333
ELEVATION = 0.185
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
    mask = region <= dark + min(28, 0.46 * (bright - dark))
    if not np.any(mask):
        return None
    nearest = int(np.argmin(np.where(mask, distances, 1e9)))
    start_y, start_x = divmod(nearest, region.shape[1])
    if distances[start_y, start_x] > 8**2:
        return None
    todo = deque([(start_y, start_x)])
    mask[start_y, start_x] = False
    component = []
    rows, cols = region.shape
    while todo:
        row, col = todo.popleft()
        component.append((row, col))
        for ny, nx in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
            if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx]:
                mask[ny, nx] = False
                todo.append((ny, nx))
    coords = np.asarray(component)
    h, w = np.ptp(coords, axis=0) + 1
    min_area, max_area = (250, 1950) if eye else (13, 400)
    if not min_area <= len(coords) <= max_area or h >= rows - 2 or w >= cols - 2:
        return None
    cy, cx = coords.mean(axis=0)
    return np.array([cx + x0 + CROP_X, cy + y0 + CROP_Y]), w, h


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


def motion_cues(video):
    eyes = [np.array([462.0, 552.0]), np.array([579.0, 552.0])]
    mouth = np.array([521.0, 603.0])
    last_mouth_height = 14
    poses = []
    params = np.zeros(5)
    failures = [0, 0, 0]
    for frame in raw_video(video):
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


def audio_cues(video):
    sample_rate, fft_size, window_length, step = 22050, 16384, 4096, 0.067
    raw = subprocess.check_output([
        'ffmpeg', '-v', 'error', '-i', str(video), '-vn', '-ac', '1', '-ar', str(sample_rate),
        '-f', 'f32le', '-',
    ])
    signal = np.frombuffer(raw, dtype='<f4').astype(float)
    padded = np.pad(signal, window_length // 2, mode='edge')
    window = np.hanning(window_length)
    starts = np.minimum(len(signal) - 1, np.rint(np.arange(0, DURATION + step, step)*sample_rate)).astype(int)
    spectra = np.array([2 * np.abs(np.fft.rfft(padded[i:i+window_length]*window, fft_size)) / window.sum()
                        for i in starts])
    bins = np.fft.rfftfreq(fft_size, 1/sample_rate)
    average = np.mean(spectra**2, axis=0)
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
        envelope = smooth_signal(spectra[:, index], 0.83)
        envelope = np.maximum(0, envelope - 0.00085)
        tones.append([round(float(frequency), 3), np.round(envelope, 5).tolist()])
    print('Resynthesis carriers (Hz):', ', '.join(str(frequency) for frequency, _ in tones))
    return {'step': step, 'tones': tones,
            'accents': [[0.985, 1.0], [1.12, 0.45], [7.14, 0.48], [7.30, 1.05], [7.43, 0.39]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', nargs='?', type=Path)
    args = parser.parse_args()
    video = args.video or next(ROOT.glob('*.mp4'))
    data = {'duration': DURATION, 'motion': motion_cues(video), 'audio': audio_cues(video)}
    output = ROOT / 'assets' / 'reference-cues.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(data, separators=(',', ':')) + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
