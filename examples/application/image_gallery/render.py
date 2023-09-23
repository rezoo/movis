import pandas as pd

import movis as mv


def make_logo(
    text: str, duration: float, font_size: int,
    margin_x: int = 20, margin_y: int = -20
) -> mv.layer.Composition:
    text_layer = mv.layer.Text(
        text, font_family='Helvetica Neue', font_size=font_size,
        font_style='Thin', color='#ffffff', duration=duration)
    w, h = text_layer.get_size(time=0.0)
    W, H = (w + 2 * margin_x, h + 2 * margin_y)
    title = mv.layer.Composition(size=(W, H), duration=duration)
    rect_item = title.add_layer(
        mv.layer.Rectangle(size=(W, H), color='#202020', duration=duration),
        position=(W, H / 2), opacity=0.75, origin_point=mv.Direction.CENTER_RIGHT)
    text_item = title.add_layer(text_layer, name='text')

    rect_item.scale.enable_motion().extend(
        keyframes=[0.0, 1.0, duration - 1.0, duration],
        values=[(0.0, 1.0), (1.0, 1.0), (1.0, 1.0), (0.0, 1.0)],
        easings=['ease_in_out5', 'linear', 'ease_in_out5'])
    text_item.opacity.enable_motion().extend(
        keyframes=[0.5, 0.75, duration - 0.75, duration - 0.5],
        values=[0.0, 1.0, 1.0, 0.0])
    return title


def main():
    size = (1920, 1080)
    timeline = pd.DataFrame([
        {
            'duration': 5.0, 'image': 'images/erik-karits.jpg',
            'title': 'Image Gallery Example', 'title_position': 'center'},
        {
            'duration': 5.0, 'image': 'images/doncoombez.jpg',
            'title': 'Doncoombez.jpg', 'title_position': 'bottom_right'},
        {
            'duration': 5.0, 'image': 'images/wolfgang-hasselmann.jpg',
            'title': 'Wolfgang-Hasselmann.jpg', 'title_position': 'bottom_right'},
    ])
    transitions = [2.0, 2.0]

    total_time = timeline['duration'].sum() + sum(transitions) + 1.0
    scene = mv.layer.Composition(size=size, duration=total_time)
    scene.add_layer(mv.layer.Rectangle(size=size, color='#202020', duration=scene.duration), name='bg')

    time = 0.
    prev_transitions = [0.] + transitions
    next_transitions = transitions + [0.]
    for (i, row), t_prev, t_next in zip(timeline.iterrows(), prev_transitions, next_transitions):
        T = row['duration']
        image = scene.add_layer(
            mv.layer.Image(row['image'], duration=T + t_prev + t_next), offset=time - t_prev)
        if i == 0:
            # Add fadein effect
            image.opacity.enable_motion().extend(keyframes=[0.0, 1.0], values=[0.0, 1.0])
        elif i == len(timeline) - 1:
            # Add fadeout effect
            t = image.duration
            image.opacity.enable_motion().extend(keyframes=[t - 1.0, t], values=[1.0, 0.0])

        kwargs_dict = {
            'center': {'position': (size[0] / 2, size[1] / 2), 'origin_point': mv.Direction.CENTER},
            'bottom_right': {'position': (size[0] - 50, size[1] - 50), 'origin_point': mv.Direction.BOTTOM_RIGHT}}
        position = kwargs_dict[row['title_position']]['position']
        origin_point = kwargs_dict[row['title_position']]['origin_point']
        scene.add_layer(
            make_logo(row['title'], duration=T, font_size=64),
            offset=time, position=position, origin_point=origin_point)

        if 0 < i:
            # Add fade effects
            image.opacity.enable_motion().extend(keyframes=[0.0, t_prev], values=[0.0, 1.0])

        # Add scale effect
        values = [1.15, 1.25] if i % 2 == 0 else [1.25, 1.15]
        image.scale.enable_motion().extend(
            keyframes=[0.0, T + t_prev + t_next], values=values)
        time += (T + t_next)

    scene.write_video('output.mp4')


if __name__ == '__main__':
    main()
