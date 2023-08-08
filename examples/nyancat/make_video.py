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

    def __init__(self, audio_img: np.ndarray, duration: float, size: tuple[int, int]):
        self.audio_img = audio_img
        self.duration = duration
        self.size = size
        self.distance = 10

    def __call__(self, time: float) -> np.ndarray:
        if time < 0 or self.duration < time:
            return np.zeros((self.size[1], self.size[0], 4), dtype=np.uint8)
        w = self.audio_img.shape[1]
        x = int(time * w / self.duration)
        array = self.audio_img[:, x]
        frame = PILImage.new('RGBA', (self.size[0], self.size[1]))
        draw = PILImageDraw.Draw(frame)
        points = np.linspace(self.distance, self.size[0] - self.distance, len(array), dtype=np.int32)
        for px, v in zip(points, array):
            h = v * (self.size[1] - self.distance * 2)
            draw.line(
                (px, (self.size[1] - h) // 2, px, (self.size[1] + h) // 2),
                fill=(255, 255, 255, 255), joint='curve', width=5)
        return np.asarray(frame)


def main():
    size = (1920, 1080)
    audio_img, duration = get_audio_image('nyancat.mp3')
    scene = zunda.layer.Composition(size, duration=duration)
    scene.add_layer(zunda.layer.Image('bg.png', duration=duration))
    scene.add_layer(
        FrequencyLayer(audio_img, duration, (1920, 256)),
        transform=zunda.Transform(position=(size[0] // 2, size[1] // 2 + 200)))
    scene.make_video('output.mp4')
    zunda.add_materials_to_video('output.mp4', 'nyancat.mp3', dst_file='output2.mp4')


if __name__ == '__main__':
    main()
