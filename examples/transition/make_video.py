import numpy as np
import zunda


class TransitionEffect:

    def __init__(self, duration=4.0, effect_time=1.0, rect_time=0.5):
        assert duration >= 2 * effect_time
        assert rect_time < effect_time
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
            return 1.0 + (time - (self.duration - self.effect_time)) / self.effect_time

    def __call__(self, time: float, prev_image: np.ndarray) -> np.ndarray:
        if time < 0 or self.duration < time:
            return prev_image
        t = self.get_key(time)
        img = prev_image.copy()
        if 0 <= t < 1:
            img[:, :, 3] = self._alpha(t, img.shape)
        else:
            alpha = self._alpha(t - 1.0, img.shape)
            img[:, :, 3] = 255 - alpha
        return img

    def _alpha(self, t: float, shape: tuple[int, ...]) -> np.ndarray:
        alpha = np.zeros(shape[:2], dtype=np.uint8)
        ri = shape[0] / self.N
        for i in range(self.N):
            start_time = i * (1 - self.rect_time) / (self.N - 1)
            ti = np.clip((t - start_time) / self.rect_time, 0, 1)
            weight = 1.0 - (1.0 - ti) ** 3
            w = weight * shape[1]
            r0 = round(i * ri)
            r1 = round((i + 1) * ri)
            alpha[r0:r1, :int(np.round(w))] = 255
        return alpha


def main():
    transition_time = 2.5
    scene = zunda.Composition((1920, 1080), duration=6.0)
    scene.add_layer(zunda.ImageLayer('scene_a.png', duration=3.0))
    scene.add_layer(zunda.ImageLayer('scene_b.png', duration=3.0), offset=3.0)
    scene.add_layer(zunda.ImageLayer('logo.png', duration=transition_time), name='logo', offset=3.0 - transition_time / 2)
    scene['logo'].add_effect(TransitionEffect(duration=transition_time))

    scene.make_video('transition.mp4')


if __name__ == '__main__':
    main()
