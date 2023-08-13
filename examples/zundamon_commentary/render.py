import numpy as np
import pandas as pd
import zunda
from zunda import Transform


def main():
    timeline = pd.read_csv('outputs/timeline.tsv', sep='\t')
    audio_timeline = zunda.make_voicevox_dataframe('audio')
    tl = pd.merge(timeline, audio_timeline, left_index=True, right_index=True)
    font_path = '/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc'

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

    def slide_in_out(item: zunda.layer.Component, offset: np.ndarray):
        p = item.transform.position.init_value
        item.transform.position.enable_animation() \
            .append(0.0, p + offset, 'ease_out_expo') \
            .append(1.0, p).append(item.duration - 1.0, p, 'ease_in_expo') \
            .append(item.duration, p + offset)
        item.transform.opacity.enable_animation().extend(
            keyframes=[0, 1, item.duration - 1.0, item.duration], values=[0, 1, 1, 0],
            motion_types=['ease_out', 'linear', 'ease_in', 'linear'])
        return item

    def make_table_of_contents(
            text: str, duration: float, margin: int = 20, line_margin: int = 4,
            font_size: int = 46, bg_color=(72, 172, 154), line_width=4):

        layer = zunda.layer.Text(
            text, font=font_path, font_size=font_size, color=(255, 255, 255), duration=duration)
        W, H = layer.get_size()
        cp = zunda.layer.Composition(
            size=(W + 3 * (margin + line_margin), H + 2 * (margin + line_margin)), duration=duration)
        cp.add_layer(
            zunda.layer.Rectangle(
                (W + 3 * margin, H + 2 * margin), radius=8,
                contents=[
                    zunda.layer.FillProperty(color=bg_color),
                    zunda.layer.StrokeProperty(color=(255, 255, 255), width=line_width)],
                duration=duration))
        cp.add_layer(layer)
        return cp

    slide_in_out(scene.add_layer(
        zunda.layer.Image(img_file='images/logo_zunda.png', duration=6.0),
        name='zunda_logo', offset=0.5,
        transform=Transform(position=(1755, 340))), np.array([500, 0]))
    slide_in_out(scene.add_layer(
        zunda.layer.Image(img_file='images/logo_metan.png', duration=6.0),
        name='metan_logo', offset=0.5,
        transform=Transform(position=(170, 340))), np.array([-500, 0]))

    slide_in_out(
        scene.add_layer(
            make_table_of_contents(
                'イントロダクション', font_size=46, duration=100.0),
            transform=Transform(position=(1920 + 20, 35)),
            origin_point='center_right',
            offset=tl[tl['slide'] == 1].iloc[0]['start_time']),
        np.array([500, 0]))

    for character in tl['character'].unique():
        character_tl = tl[tl['character'] == character]
        texts = [c.replace('\\n', '\n') for c in character_tl['text'].tolist()]
        color_dict = {'zunda': (94, 166, 56), 'metan': (171, 74, 115)}
        scene.add_layer(
            zunda.layer.Text.from_timeline(
                character_tl['start_time'], character_tl['end_time'], texts,
                font_size=72, font=font_path, contents=[
                    zunda.layer.StrokeProperty(color=color_dict[character], width=12),
                    zunda.layer.FillProperty(color=(255, 255, 255))],
                duration=character_tl['end_time'].max()),
            transform=Transform(position=(960, 1000)),
            origin_point=zunda.Direction.BOTTOM_CENTER)

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
    #zunda.write_ass_file(
    #    tl['start_time'], tl['end_time'], tl['text'], 'outputs/subtitle.ass',
    #    size=scene.size, characters=tl['character'], styles=styles)
    scene.write_video('outputs/video.mp4', end_time=10.0)
    #zunda.add_materials_to_video(
    #    'outputs/video.mp4', 'outputs/dialogue.wav',
    #    subtitle_file='outputs/subtitle.ass', dst_file='outputs/video2.mp4')


if __name__ == '__main__':
    main()
