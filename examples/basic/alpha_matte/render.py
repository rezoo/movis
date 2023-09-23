import numpy as np

import movis as mv


def main():
    size = (1920, 1080)
    duration = 3.0
    box_duration = 1.0

    square = mv.layer.Composition(size, duration=duration)
    rect = mv.layer.Rectangle((600, 600), radius=10.0, color=(255, 255, 255), duration=duration)
    square.add_layer(rect, position=(30 + 300, size[1] / 2))
    square.add_layer(rect)  # If position is not specified, the layer is centered by default
    square.add_layer(rect, position=(size[0] - 30 - 300, size[1] / 2))

    for layer_item in square.layers:
        layer_item.scale.enable_motion().extend(
            keyframes=[0, box_duration], values=[1.8, 0.4], easings=['ease_in_out3', 'linear'])

    text_kwargs = dict(font_family='Helvetica', font_size=60, color="#ffffff", duration=duration)
    text_ypos = size[1] / 2 - 150
    square.add_layer(
        mv.layer.Text('Hello', **text_kwargs), position=(30 + 300, text_ypos), name='text1')
    square.add_layer(
        mv.layer.Text('Alpha', **text_kwargs), position=(size[0] / 2, text_ypos), name='text2')
    square.add_layer(
        mv.layer.Text('Matte!', **text_kwargs), position=(size[0] - 30 - 300, text_ypos), name='text3')

    def move_text(layer_item: mv.layer.LayerItem):
        after = layer_item.position.init_value
        before = after + np.array([0, 100])
        layer_item.position.enable_motion().extend(
            keyframes=[box_duration - 0.5, duration], values=[before, after],
            easings=['ease_out3', 'linear'])
    move_text(square['text1'])
    move_text(square['text2'])
    move_text(square['text3'])

    image = mv.layer.Composition(size, duration=duration)
    image.add_layer(mv.layer.Image('image.jpg', duration=duration))

    scene = mv.layer.Composition(size, duration=duration)
    scene.add_layer(mv.layer.Rectangle(size, color="#373737", duration=duration))
    scene.add_layer(mv.layer.AlphaMatte(square, image))
    scene.write_video('output.mp4')


if __name__ == '__main__':
    main()
