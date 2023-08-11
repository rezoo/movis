import zunda


def main():
    size = (640, 480)
    duration = 5.0

    scene = zunda.layer.Composition(size, duration=duration)
    scene.add_layer(
        zunda.layer.Rectangle(
            size, color=(127, 127, 127), line_width=0, duration=duration),
        name='bg')
    rectangle = zunda.layer.Rectangle(
        (10, 10), color=(255, 83, 49), line_width=5,
        line_color=(255, 255, 255), duration=duration)
    scene.add_layer(rectangle, name='rect')

    rectangle.size.enable_animation().extend(
        keyframes=[0, 1, 2, 3, 4],
        values=[(10, 10), (400, 400), (10, 10), (100, 400), (400, 100)],
        motion_types=['ease_out_expo'] * 5)

    scene.write_video('rectangles.mp4')


if __name__ == '__main__':
    main()
