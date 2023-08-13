import movis as mv


def main():
    size = (1920, 1080)
    duration = 2.0

    # An example of a simple alphamat is shown below.
    # First, create a composition with three rectangles placed in the center as the base.
    square = mv.layer.Composition(size, duration=duration / 2)
    # Similar layers can be placed multiple times in this manner.
    # This is useful when placing many layers like particles.
    rect = mv.layer.Rectangle((600, 600), radius=10.0, color=(255, 255, 255), duration=duration / 2)
    square.add_layer(rect, transform=mv.Transform(position=(30 + 300, size[1] / 2)))
    square.add_layer(rect)  # If transform is not specified, the layer is centered by default
    square.add_layer(rect, transform=mv.Transform(position=(size[0] - 30 - 300, size[1] / 2)))

    # Next, create the main composition.
    scene = mv.layer.Composition(size, duration=duration)
    # There is no need to specify a name if no animation or filter is specified.
    # but for the sake of clarity, the name "bg" is used here.
    scene.add_layer(mv.layer.Rectangle(size, color=(55, 55, 55), duration=duration), name='bg')
    scene.add_layer(square, name='square', offset=duration / 2)
    # Note that the image can be downloaded from: https://unsplash.com/photos/J6LMHbdW1k8
    scene.add_layer(mv.layer.Image('image.jpg', duration=duration / 2))
    scene.add_layer(mv.layer.Image('image.jpg', duration=duration / 2), name='image', offset=duration / 2)
    # Specify image as the target of the alpha matte of square.
    # It overwrites the color of square with image.
    scene.enable_alpha_matte('square', 'image')
    # Now let's make a video.
    scene.write_video('alpha_matte.mp4')


if __name__ == '__main__':
    main()
