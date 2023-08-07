import zunda
from zunda import Transform
import numpy as np
import pandas as pd


def main():
    timeline = pd.read_csv('outputs/timeline.tsv', sep='\t')
    audio_timeline = zunda.make_voicevox_dataframe('audio')
    tl = pd.merge(timeline, audio_timeline, left_index=True, right_index=True)

    scene = zunda.layer.Composition(size=(1920, 1080), duration=tl['end_time'].max())
    scene.add_layer(
        zunda.layer.Image(img_file='../../assets/bg2.png', duration=tl['end_time'].max()),
        transform=Transform(position=(960, 540)))
    scene.add_layer(
        zunda.layer.Slide(
            tl['start_time'], tl['end_time'],
            slide_file='slide.pdf', slide_counter=tl['slide']),
        transform=Transform(position=(960, 421), scale=0.71))
    scene.add_layer(
        zunda.layer.Character(
            tl['start_time'], tl['end_time'],
            characters=tl['character'], character_status=tl['status'],
            character_name='zunda', character_dir='../../assets/character/zunda'),
        name='zunda',
        transform=Transform(position=(1779, 950), scale=0.7))
    scene.add_layer(
        zunda.layer.Character(
            tl['start_time'], tl['end_time'],
            characters=tl['character'], character_status=tl['status'],
            character_name='metan', character_dir='../../assets/character/metan'),
        name='metan',
        transform=Transform(position=(79, 1037), scale=0.7))
    actions = zunda.make_action_functions(tl['start_time'], tl['end_time'], tl['action'])
    for layer_name, action_func in actions:
        action_func(scene, layer_name)

    scene.add_layer(
        zunda.layer.Image(img_file='images/logo_zunda.png', duration=7.0),
        name='zunda_logo', offset=0.5,
        transform=Transform(position=(1755, 340)))
    scene.add_layer(
        zunda.layer.Image(img_file='images/logo_metan.png', duration=7.0),
        name='metan_logo', offset=0.5,
        transform=Transform(position=(170, 340)))

    def slide_in_out(item: zunda.layer.Component, offset: np.ndarray):
        p = item.transform.position.init_value
        item.transform.position.enable_animation() \
            .append(0.0, p) \
            .append(0.0, p + offset, 'ease_out_expo') \
            .append(1.0, p).append(5.0, p, 'ease_in_expo') \
            .append(6.0, p + offset)
        item.transform.opacity.enable_animation().extend(
            keyframes=[0, 1, 5, 6], values=[0, 1, 1, 0],
            motion_types=['ease_out', 'linear', 'ease_in', 'linear'])

    slide_in_out(scene['zunda_logo'], np.array([500, 0]))
    slide_in_out(scene['metan_logo'], np.array([-500, 0]))

    bgm = zunda.make_loop_music('../../assets/bgm2.wav', tl['end_time'].max()) - 25
    bgm = bgm.fade_out(5 * 1000)
    voice = zunda.concat_audio_files(tl['start_time'], tl['audio_file'])
    bgm.overlay(voice).export('outputs/dialogue.wav', format='wav')

    styles = (
        zunda.ASSStyleType(
            'zunda', font_name='Hiragino Maru Gothic Pro', font_size=96,
            outline_color='&H38a65e', back_color='&HA0000000', outline=5, shadow=3),
        zunda.ASSStyleType(
            'metan', font_name='Hiragino Maru Gothic Pro', font_size=96,
            outline_color='&H734aab', back_color='&HA0000000', outline=5, shadow=3),
    )
    zunda.make_ass_file(
        tl['start_time'], tl['end_time'], tl['text'], 'outputs/subtitle.ass',
        size=scene.size, characters=tl['character'], styles=styles)
    scene.make_video('outputs/video.mp4')
    zunda.add_materials_to_video(
        'outputs/video.mp4', 'outputs/dialogue.wav',
        subtitle_file='outputs/subtitle.ass', dst_file='outputs/video2.mp4')


if __name__ == '__main__':
    main()
