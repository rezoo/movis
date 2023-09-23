import movis as mv


def main():
    size = (640, 480)
    duration = 5.0

    scene = mv.layer.Composition(size, duration=duration)
    scene.add_layer(
        mv.layer.Rectangle(size, color=(127, 127, 127), duration=duration),
        name='bg')
    rectangle = mv.layer.Rectangle(
        size=(10, 10),
        contents=[
            mv.layer.FillProperty(color=(255, 83, 49)),
            mv.layer.StrokeProperty(color=(255, 255, 255), width=5),
        ],
        duration=duration)
    scene.add_layer(rectangle, name='rect')

    rectangle.size.enable_motion().extend(
        keyframes=[0, 1, 2, 3, 4],
        values=[(0, 0), (400, 400), (0, 0), (100, 400), (400, 100)],
        easings=['ease_out5'] * 5)
    scene['rect'].rotation.enable_motion().extend(
        keyframes=[0, 1, 2, 3, 4],
        values=[0, 90, 180, 0, 0],
        easings=['ease_out5'] * 5)

    scene.write_video('output.mp4')


if __name__ == '__main__':
    main()
