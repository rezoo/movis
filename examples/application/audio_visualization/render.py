import argparse

import librosa
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from scipy.interpolate import RegularGridInterpolator

import movis as mv
from movis.imgproc import qimage_to_numpy


def get_audio_image(path: str):
    audio, sampling_rate = librosa.load(path)
    duration = len(audio) / sampling_rate
    freq = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    db_array = librosa.amplitude_to_db(freq, ref=np.max)
    m, M = db_array.min(), db_array.max()
    db_array = (db_array - m) / (M - m)
    p = np.percentile(db_array.mean(axis=1), 5)
    db_array = db_array[db_array.mean(axis=1) > p, :]

    y_linear = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, db_array.shape[0])
    x = np.linspace(0, 1, db_array.shape[1])
    interpolator = RegularGridInterpolator((y, x), db_array)
    db_resampled = interpolator(np.array([[(yy, xx) for xx in x] for yy in y_linear]))
    return db_resampled, duration


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
        image = QImage(self.size[0], self.size[1], QImage.Format.Format_ARGB32)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(255, 255, 255, 255))
        pen.setWidthF(5.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        if self.mode == 'line':
            points = np.linspace(
                self.margin, self.size[0] - self.margin, len(array), dtype=np.float64)
            for px, v in zip(points, array):
                h = v * (self.size[1] - self.margin * 2)
                painter.drawLine(QPointF(px, (self.size[1] - h) / 2), QPointF(px, (self.size[1] + h) / 2))
        elif self.mode == 'circle':
            n_point = len(array)
            theta = np.linspace(0., 2 * np.pi, n_point, endpoint=False)
            center = np.array([self.size[0] / 2, self.size[1] / 2], dtype=float)
            radius = min(self.size[0], self.size[1]) / 2 - self.length / 2 - self.margin
            points = np.concatenate([np.cos(theta)[:, None], np.sin(theta)[:, None]], axis=1)
            points_start = center + radius * points
            points_end = center + (radius + array[:, None] * self.length / 2) * points
            for p0, p1 in zip(points_start, points_end):
                painter.drawLine(QPointF(p0[0], p0[1]), QPointF(p1[0], p1[1]))
        else:
            raise ValueError
        painter.end()
        return qimage_to_numpy(image)


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
    scene = mv.layer.Composition(size, duration=duration + eps)
    scene.add_layer(mv.layer.Image(args.background, duration=duration + eps))

    if not args.no_logo:
        logo_position = (size[0] // 2, size[1] // 2 - 200) if args.type == 'line' \
            else (size[0] // 2, size[1] // 2)
        scene.add_layer(
            mv.layer.Image('logo.png', duration=duration + eps),
            position=logo_position)

    freq_size = (1920, 256) if args.type == 'line' else (1080, 1080)
    freq_position = (size[0] // 2, size[1] // 2 + 200) if args.type == 'line' \
        else (size[0] // 2, size[1] // 2)
    scene.add_layer(
        FrequencyLayer(audio_img, duration, freq_size, mode=args.type),
        position=freq_position, opacity=0.9)
    scene.add_layer(mv.layer.Audio(args.input))
    scene.write_video(args.output, audio=True)


if __name__ == '__main__':
    main()
