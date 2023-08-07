import zunda


def main():
    size = (640, 480)
    duration = 5.0

    scene = zunda.Composition((640, 480), duration=duration)
    scene.add_layer(
        zunda.RectangleLayer(
            size, color=(255, 186, 49), line_width=0, duration=duration),
        name='bg')
    rectangle = zunda.RectangleLayer(
        (10, 10), color=(255, 83, 49), line_width=5,
        line_color=(255, 255, 255), duration=duration)
    scene.add_layer(rectangle, name='rect')

    rectangle.size.enable_animation().extend(
        keyframes=[0, 1, 2, 3, 4],
        values=[(10, 10), (400, 400), (10, 10), (100, 400), (400, 100)],
        motion_types=['ease_out_expo'] * 5)

    scene.make_video('rectangles.mp4')


if __name__ == '__main__':
    main()
