import numpy as np

import movis as mv


class TransitionEffect:

    def __init__(self, duration=4.0, effect_time=1.0, rect_time=0.5, n_rectangle=8):
        assert duration >= 2 * effect_time
        assert rect_time < effect_time
        self.duration = duration
        self.effect_time = effect_time
        self.rect_time = rect_time
        self.N = n_rectangle

    def get_key(self, time: float) -> float:
        if time < self.effect_time:
            return time / self.effect_time
        elif self.effect_time <= time <= self.duration - self.effect_time:
            return 1.0
        else:
            return 1.0 + (time - (self.duration - self.effect_time)) / self.effect_time

    def __call__(self, prev_image: np.ndarray, time: float) -> np.ndarray:
        def get_alpha(t: float) -> np.ndarray:
            alpha = np.zeros(prev_image.shape[:2], dtype=np.uint8)
            ri = prev_image.shape[0] / self.N
            for i in range(self.N):
                start_time = i * (1 - self.rect_time) / (self.N - 1)
                ti = np.clip((t - start_time) / self.rect_time, 0, 1)
                weight = 1.0 - (1.0 - ti) ** 3
                w = weight * prev_image.shape[1]
                r0 = round(i * ri)
                r1 = round((i + 1) * ri)
                alpha[r0:r1, :round(w)] = 255
            return alpha

        if time < 0 or self.duration < time:
            return prev_image
        t = self.get_key(time)
        img = prev_image.copy()
        if 0 <= t < 1:
            img[:, :, 3] = get_alpha(t)
        else:
            alpha = get_alpha(t - 1.0)
            img[:, :, 3] = 255 - alpha
        return img


def main():
    size = (1920, 1080)
    transition_time = 2.5

    scene = mv.layer.Composition(size, duration=6.0)
    scene.add_layer(mv.layer.Image('scene_a.png', duration=3.0))
    scene.add_layer(mv.layer.Image('scene_b.png', duration=3.0), offset=3.0)

    logo = mv.layer.Composition(size, duration=transition_time)
    logo.add_layer(mv.layer.Rectangle(size, color=(178, 217, 186), duration=transition_time))
    logo.add_layer(mv.layer.Image('logo.png', duration=transition_time), name='image')
    logo['image'].scale.enable_motion().extend(
        keyframes=[0., transition_time], values=[0.9, 1.0], easings=['ease_out3'])
    scene.add_layer(logo, name='logo', offset=3.0 - transition_time / 2)
    scene['logo'].add_effect(TransitionEffect(duration=transition_time))

    scene.write_video('transition.mp4')


if __name__ == '__main__':
    main()
