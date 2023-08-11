import argparse

import librosa
import numpy as np
import zunda
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
from scipy.interpolate import RegularGridInterpolator


def get_audio_image(path: str):
    audio, sampling_rate = librosa.load(path)
    duration = len(audio) / sampling_rate
    freq = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    db_array = librosa.amplitude_to_db(freq, ref=np.max)

    y_linear = np.linspace(0, 1, 128)
    y = np.linspace(0, 1, db_array.shape[0])
    x = np.linspace(0, 1, db_array.shape[1])
    interpolator = RegularGridInterpolator((y, x), db_array)
    db_resampled = interpolator(np.array([[(yy, xx) for xx in x] for yy in y_linear]))
    m, M = db_resampled.min(), db_resampled.max()
    audio_img = (db_resampled - m) / (M - m)
    return audio_img, duration


class FrequencyLayer:

    def __init__(self, audio_img: np.ndarray, duration: float, size: tuple[int, int], mode: str = 'line'):
        self.audio_img = audio_img
        self.duration = duration
        self.size = size
        self.mode = mode
        self.margin = 10
        self.length = 200

    def __call__(self, time: float) -> np.ndarray:
        if time < 0 or self.duration < time:
            return np.zeros((self.size[1], self.size[0], 4), dtype=np.uint8)
        w = self.audio_img.shape[1]
        i = int(time * w / self.duration)
        array = self.audio_img[:, i]
        frame = PILImage.new('RGBA', (self.size[0], self.size[1]))
        draw = PILImageDraw.Draw(frame)
        if self.mode == 'line':
            points = np.linspace(
                self.margin, self.size[0] - self.margin, len(array), dtype=np.int32)
            for px, v in zip(points, array):
                h = v * (self.size[1] - self.margin * 2)
                draw.line(
                    (px, (self.size[1] - h) // 2, px, (self.size[1] + h) // 2),
                    fill=(255, 255, 255, 255), joint='curve', width=7)
        elif self.mode == 'circle':
            n_point = len(array)
            theta = np.linspace(0., 2 * np.pi, n_point, endpoint=False)
            center = np.array([self.size[0] / 2, self.size[1] / 2], dtype=float)
            radius = min(self.size[0], self.size[1]) / 2 - self.length / 2 - self.margin
            points = np.concatenate([np.cos(theta)[:, None], np.sin(theta)[:, None]], axis=1)
            points_start = np.round(center + (radius - array[:, None] * self.length / 2) * points)
            points_end = np.round(center + (radius + array[:, None] * self.length / 2) * points)
            for p0, p1 in zip(points_start, points_end):
                draw.line(
                    (p0[0], p0[1], p1[0], p1[1]),
                    fill=(255, 255, 255, 255), joint='curve', width=7)
        else:
            raise ValueError
        return np.asarray(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', choices=['line', 'circle'], default='line')
    parser.add_argument('-i', '--input', default='nyancat.mp3')
    parser.add_argument('-o', '--output', default='output.mp4')
    parser.add_argument('--background', default='bg.jpg')
    parser.add_argument('--no-logo', action='store_true')
    args = parser.parse_args()

    size = (1920, 1080)
    eps = 0.1
    audio_img, duration = get_audio_image(args.input)
    scene = zunda.layer.Composition(size, duration=duration + eps)
    scene.add_layer(zunda.layer.Image(args.background, duration=duration + eps))

    if not args.no_logo:
        logo_position = (size[0] // 2, size[1] // 2 - 200) if args.type == 'line' \
            else (size[0] // 2, size[1] // 2)
        scene.add_layer(
            zunda.layer.Image('logo.png', duration=duration + eps),
            transform=zunda.Transform(position=logo_position))

    freq_size = (1920, 256) if args.type == 'line' else (1080, 1080)
    freq_position = (size[0] // 2, size[1] // 2 + 200) if args.type == 'line' \
        else (size[0] // 2, size[1] // 2)
    scene.add_layer(
        FrequencyLayer(audio_img, duration, freq_size, mode=args.type),
        transform=zunda.Transform(position=freq_position, opacity=0.8))
    scene.write_video('no_audio.mp4')
    zunda.add_materials_to_video('no_audio.mp4', args.input, dst_file=args.output)


if __name__ == '__main__':
    main()
