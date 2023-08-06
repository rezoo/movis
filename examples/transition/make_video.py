import numpy as np
from PIL import Image
import zunda


class OriginalTransitionLayer:

    def __init__(self, img_file: str, duration=4.0, effect_time=1.0, rect_time=0.5):
        self.image = np.asarray(Image.open(img_file).convert("RGBA"))
        self.size = (self.image.shape[1], self.image.shape[0])
        self.duration = duration
        self.effect_time = effect_time
        self.rect_time = rect_time
        self.N = 8

    def get_key(self, time: float) -> float:
        if time < self.effect_time:
            return time / self.effect_time
        elif self.effect_time <= time <= self.duration - self.effect_time:
            return 1.0
        else:
            return (self.duration - time) / self.effect_time

    def __call__(self, time: float) -> np.ndarray:
        t = self.get_key(time)
        return self._draw(t)

    def _draw(self, t: float) -> np.ndarray:
        img = self.image.copy()
        img[:, :, 3] = 0
        ri = self.size[0] / self.N
        for i in range(self.N):
            start_time = i * (1 - self.rect_time) / (self.N - 1)
            ti = np.clip((t - start_time) / self.rect_time, 0, 1)
            weight = 1.0 - (1.0 - ti) ** 3
            w = weight * self.size[0]
            r0 = round(i * ri)
            r1 = round((i + 1) * ri)
            img[r0:r1, :int(np.round(w)), 3] = 255
        return img


def main():
    transition_duration = 3.0
    scene = zunda.Composition((1920, 1080), duration=6.0)
    scene.add_layer(zunda.ImageLayer('scene_a.png', duration=3.0))
    scene.add_layer(zunda.ImageLayer('scene_b.png', duration=3.0), offset=3.0)
    scene.add_layer(OriginalTransitionLayer(img_file='logo.png', duration=transition_duration), offset=3.0 - transition_duration / 2)

    scene.make_video('transition.mp4')


if __name__ == '__main__':
    main()
