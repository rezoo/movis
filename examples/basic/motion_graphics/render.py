import movis as mv


def circle_anim(
        scene: mv.layer.Composition, color: str, offset: float,
        duration: float, sizes: tuple[int, int]):
    layer = mv.layer.Ellipse(
        (sizes[0], sizes[0]),
        contents=[mv.layer.StrokeProperty(mv.to_rgb(color), 5.0)],
        duration=duration)
    item = scene.add_layer(layer, offset=offset)
    layer.size.enable_motion().extend([0, duration], [sizes[0], sizes[1]], ['ease_out5'])
    item.opacity.enable_motion().extend([0, duration], [1, 0], ['ease_out5'])


def main():
    size = (1920, 1080)
    duration = 6.0
    circle_color = "#4e4e4e"
    bg_color = "#d2d2d2"

    scene = mv.layer.Composition(size, duration)
    scene.add_layer(mv.layer.Rectangle(size, color=bg_color))
    scene.add_layer(
        mv.layer.Ellipse((10.0, 10.0), color=circle_color, duration=duration),
        offset=0.25, name='c1')
    scene['c1'].layer.size.enable_motion().extend(
        [0.0, 1.0, 2.0 - 0.25], [0.0, 400.0, 0.0], ['ease_out3', 'ease_in3'])
    circle_anim(scene, color=circle_color, offset=0.0, duration=1.0, sizes=(0, 1200))
    circle_anim(scene, color=circle_color, offset=0.2, duration=1.0, sizes=(0, 2200))
    circle_anim(scene, color=circle_color, offset=1.6, duration=1.0, sizes=(2200, 0))

    c2 = mv.layer.Ellipse((10.0, 10.0), color=circle_color, duration=duration)
    scene.add_layer(c2, offset=2.0, name='c2')
    c2.size.enable_motion().extend([0.0, 1.0], [0.0, 2200.0], ['ease_out5'])
    circle_anim(scene, color=bg_color, offset=2.25, duration=1.0, sizes=(0, 2000))

    p = size[0] / 2, size[1] / 2
    scene.add_layer(
        mv.layer.Text(
            'Movis', font_size=200, font_family='Helvetica Neue',
            font_style='Thin', color="#ffffff"),
        position=(p[0], p[1] - 50), offset=2.25, name='logo')
    scene['logo'].opacity.enable_motion().extend([0.0, 0.5], [0.0, 1.0])
    scene['logo'].position.enable_motion().extend(
        [0.0, 0.5], [(p[0], p[1] - 30), (p[0], p[1] - 50)], ['ease_out5'])
    scene.add_layer(
        mv.layer.Text(
            'Video Editing as a Code', font_size=48, font_family='Helvetica Neue',
            font_style='Light', color="#ffffff"),
        position=(p[0], p[1] - 50), offset=2.5, name='subtitle')
    scene['subtitle'].opacity.enable_motion().extend([0.0, 0.5], [0.0, 1.0])
    scene['subtitle'].position.enable_motion().extend(
        [0.0, 0.5], [(p[0], p[1] + 100), (p[0], p[1] + 80)], ['ease_out3'])

    scene.write_video('output.mp4')


if __name__ == '__main__':
    main()
