import movis as mv


def main():
    size = (1920, 1080)
    duration = 1.0

    square = mv.layer.Composition(size, duration=duration)
    rect = mv.layer.Rectangle((600, 600), radius=10.0, color=(255, 255, 255), duration=duration)
    square.add_layer(rect, transform=mv.Transform(position=(30 + 300, size[1] / 2)))
    square.add_layer(rect)  # If transform is not specified, the layer is centered by default
    square.add_layer(rect, transform=mv.Transform(position=(size[0] - 30 - 300, size[1] / 2)))

    for component in square.layers:
        component.transform.scale.enable_animation().extend(
            keyframes=[0, duration], values=[2.0, 0.5], motion_types=['ease_out', 'ease_in'])

    scene = mv.layer.Composition(size, duration=duration)
    scene.add_layer(mv.layer.Rectangle(size, color=(55, 55, 55), duration=duration), name='bg')
    scene.add_layer(square, name='square')
    # Note that the image can be downloaded from: https://unsplash.com/photos/J6LMHbdW1k8
    scene.add_layer(mv.layer.Image('image.jpg', duration=duration), name='image')
    # Specify image as the target of the alpha matte of square.
    # It overwrites the color of square with image.
    scene.enable_alpha_matte('square', 'image')
    # Now let's make a video.
    scene.write_video('alpha_matte.mp4')


if __name__ == '__main__':
    main()
