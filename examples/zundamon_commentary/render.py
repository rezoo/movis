import movis as mv
import numpy as np
import pandas as pd
from movis import Transform
from movis.contrib.commentary import Character, Slide
from movis.contrib.voicevox import make_voicevox_dataframe


def main():
    timeline = pd.read_csv('outputs/timeline.tsv', sep='\t')
    audio_timeline = make_voicevox_dataframe('audio')
    tl = pd.merge(timeline, audio_timeline, left_index=True, right_index=True)
    font_name = 'Hiragino Maru Gothic ProN'

    scene = mv.layer.Composition(size=(1920, 1080), duration=tl['end_time'].max())
    scene.add_layer(
        mv.layer.Image(img_file='assets/bg2.png', duration=tl['end_time'].max()),
        transform=Transform(position=(960, 540)))
    scene.add_layer(
        Slide(
            tl['start_time'], tl['end_time'],
            slide_file='slide.pdf', slide_counter=tl['slide']),
        transform=Transform(position=(960, 421), scale=0.71))
    scene.add_layer(
        Character(
            tl['start_time'], tl['end_time'],
            characters=tl['character'], character_status=tl['status'],
            character_name='zunda', character_dir='assets/character/zunda'),
        name='zunda',
        transform=Transform(position=(1779, 950), scale=0.7))
    scene.add_layer(
        Character(
            tl['start_time'], tl['end_time'],
            characters=tl['character'], character_status=tl['status'],
            character_name='metan', character_dir='assets/character/metan'),
        name='metan',
        transform=Transform(position=(79, 1037), scale=0.7))

    def slide_in_out(item: mv.layer.LayerItem, offset: np.ndarray):
        p = item.transform.position.init_value
        item.transform.position.enable_motion() \
            .append(0.0, p + offset, 'ease_out5') \
            .append(1.0, p).append(item.duration - 1.0, p, 'ease_in5') \
            .append(item.duration, p + offset)
        item.transform.opacity.enable_motion().extend(
            keyframes=[0, 1, item.duration - 1.0, item.duration], values=[0, 1, 1, 0],
            motion_types=['ease_out', 'linear', 'ease_in', 'linear'])
        return item

    def make_table_of_contents(
            text: str, duration: float, margin: int = 60,
            font_size: int = 46, bg_color="#48AC9A", line_width=4):

        layer = mv.layer.Text(
            text, font_family=font_name, font_size=font_size, color="#ffffff", duration=duration)
        W, H = layer.get_size()
        cp = mv.layer.Composition(
            size=(W + margin, H), duration=duration)
        cp.add_layer(
            mv.layer.Rectangle(
                (W + margin - line_width // 2, H - line_width // 2), radius=8,
                contents=[
                    mv.layer.FillProperty(color=bg_color),
                    mv.layer.StrokeProperty(color='#ffffff', width=line_width)],
                duration=duration))
        cp.add_layer(layer)
        return cp

    slide_in_out(scene.add_layer(
        mv.layer.Image(img_file='assets/logo_zunda.png', duration=6.0),
        name='zunda_logo', offset=0.5,
        transform=Transform(position=(1755, 340))), np.array([500, 0]))
    slide_in_out(scene.add_layer(
        mv.layer.Image(img_file='assets/logo_metan.png', duration=6.0),
        name='metan_logo', offset=0.5,
        transform=Transform(position=(170, 340))), np.array([-500, 0]))

    sections = tl[~tl['section'].isnull()]
    section_times = np.concatenate([sections['start_time'].to_numpy(), [3600.0]])
    for i in range(len(section_times) - 1):
        start_time = section_times[i]
        end_time = section_times[i + 1]
        duration = end_time - start_time
        section = sections['section'].iloc[i]
        slide_in_out(
            scene.add_layer(
                make_table_of_contents(
                    section, font_size=46, duration=duration),
                transform=Transform(position=(1920 + 10, 35), origin_point='center_right'),
                offset=start_time),
            np.array([500, 0]))

    for character in tl['character'].unique():
        character_tl = tl[tl['character'] == character]
        texts = [c.replace('\\n', '\n') for c in character_tl['text'].tolist()]
        color_dict = {'zunda': "#5EA638", 'metan': "#AB4A73"}
        item = scene.add_layer(
            mv.layer.Text.from_timeline(
                character_tl['start_time'], character_tl['end_time'], texts,
                font_size=72, font_family=font_name, line_spacing=100, contents=[
                    mv.layer.StrokeProperty(color=color_dict[character], width=12),
                    mv.layer.FillProperty(color='#ffffff')],
                duration=character_tl['end_time'].max(),
                text_alignment='center'),
            transform=Transform.from_positions(scene.size, bottom=40.0))
        item.add_effect(mv.effect.DropShadow(offset=5.0))

    bgm = mv.make_loop_music('assets/bgm2.wav', tl['end_time'].max()) - 25
    bgm = bgm.fade_out(5 * 1000)
    voice = mv.concat_audio_files(tl['start_time'], tl['audio_file'])
    bgm.overlay(voice).export('outputs/dialogue.wav', format='wav')

    mv.write_srt_file(
        tl['start_time'], tl['end_time'], tl['text'], 'outputs/dialogue.srt')
    scene.write_video('outputs/video.mp4', audio_path='outputs/dialogue.wav')


if __name__ == '__main__':
    main()
