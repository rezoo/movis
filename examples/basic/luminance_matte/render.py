import numpy as np

import movis as mv


def main():
    size = (1920, 256)
    duration = 8.0
    height = 64

    matte = mv.layer.Composition(size=size, duration=duration)
    matte.add_layer(mv.layer.Rectangle(size, color='#ffffff', duration=duration))
    matte.add_layer(
        mv.layer.Gradient(
            (size[0], height), start_point=(0, 0), end_point=(0, height),
            start_color='#000000', end_color='#ffffff', duration=duration),
        position=(size[0] // 2, height), origin_point='bottom_center')
    matte.add_layer(
        mv.layer.Gradient(
            (size[0], height), start_point=(0, 0), end_point=(0, height),
            start_color='#000000', end_color='#ffffff', duration=duration),
        position=(size[0] // 2, size[1] - height), origin_point='bottom_center', rotation=180)

    text = mv.layer.Composition(size=size, duration=duration)
    text.add_layer(
        mv.layer.Text(
            'Infrastructure\nPlatform\nSoftware\nVideo Editing', font_size=128, font_family='Helvetica Neue',
            font_style='Thin', color='#ffffff', line_spacing=180,
            text_alignment=mv.TextAlignment.RIGHT, duration=duration),
        position=(size[0] // 2 - 500, size[1] // 2 + 280), name='text')
    text.add_layer(
        mv.layer.Text(
            'as a Code', font_size=128, font_family='Helvetica Neue',
            font_style='Thin', color='#ffffff', duration=duration),
        position=(size[0] // 2 + 160, size[1] // 2), name='suffix')

    p = text['text'].transform.position.init_value
    dp = np.array([0, 180])
    text['text'].position.enable_motion().extend(
        [0, 1, 2, 3, 4],
        [p - np.array([100, 0]), p - 0 * dp, p - 1 * dp, p - 2 * dp, p - 3 * dp],
        ['ease_out12'] + ['ease_in_out12'] * 4)
    text['text'].opacity.enable_motion().extend(
        [0, 1], [0, 1], ['ease_out12'])

    ps = text['suffix'].position.init_value
    text['suffix'].position.enable_motion().extend(
        [0, 1], [ps + np.array([100, 0]), ps], ['ease_out12'])
    text['suffix'].opacity.enable_motion().extend(
        [0, 1], [0, 1], ['ease_out12'])

    scene_size = (1920, 1080)
    scene = mv.layer.Composition(size=scene_size, duration=duration)
    scene.add_layer(mv.layer.Rectangle(scene_size, color='#373737', duration=duration))
    scene.add_layer(
        mv.layer.LuminanceMatte(matte, text),
        position=(scene_size[0] // 2 + 170, scene_size[1] // 2),
        name='subtitle'
    )

    p_subtitle = scene['subtitle'].transform.position.init_value
    scene.add_layer(
        mv.layer.Text(
            'Movis', font_size=256, font_family='Helvetica Neue',
            font_style='Thin', color='#ffffff', duration=duration),
        offset=5.0,
        name='title')
    p_title = scene['title'].position.init_value
    scene['title'].position.enable_motion().extend(
        [0, 2.0], [p_title + np.array([0, 50]), p_title], ['ease_out12'])
    scene['title'].opacity.enable_motion().extend([0, 2.0], [0, 1], ['ease_out12'])
    scene['subtitle'].position.enable_motion().extend(
        [4.0, 6.0], [p_subtitle, p_subtitle + np.array([-70, 150])], ['ease_in_out12'])
    scene['subtitle'].scale.enable_motion().extend(
        [4.0, 6.0], [1.0, 0.48], ['ease_in_out12'])

    scene.write_video('output.mp4')


if __name__ == '__main__':
    main()
