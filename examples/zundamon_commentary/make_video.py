import zunda
from zunda import Transform
import pandas as pd


def main():
    timeline = pd.read_csv('outputs/timeline.tsv', sep='\t')
    audio_timeline = zunda.make_voicevox_dataframe('audio')
    tl = pd.merge(timeline, audio_timeline, left_index=True, right_index=True)

    scene = zunda.Composition(size=(1920, 1080), duration=tl['end_time'].max())
    scene.add_layer(
        zunda.ImageLayer(img_path='../../assets/bg2.png', duration=tl['end_time'].max()),
        transform=Transform.create(position=(960, 540)))
    scene.add_layer(
        zunda.SlideLayer(
            tl['start_time'], tl['end_time'],
            slide_path='slide.pdf', slide_counter=tl['slide']),
        transform=Transform.create(position=(960, 421), scale=0.71))
    scene.add_layer(
        zunda.CharacterLayer(
            tl['start_time'], tl['end_time'],
            characters=tl['character'], character_status=tl['status'],
            character_name='zunda', character_dir='../../assets/character/zunda'),
        name='zunda',
        transform=Transform.create(position=(1779, 878), scale=0.7))
    scene.add_layer(
        zunda.CharacterLayer(
            tl['start_time'], tl['end_time'],
            characters=tl['character'], character_status=tl['status'],
            character_name='metan', character_dir='../../assets/character/metan'),
        name='metan',
        transform=Transform.create(position=(79, 1037), scale=0.7))
    actions = zunda.make_action_functions(tl['start_time'], tl['end_time'], tl['action'])
    for layer_name, action_func in actions:
        action_func(scene, layer_name)

    bgm = zunda.make_loop_music('../../assets/bgm2.wav', tl['end_time'].max()) - 20
    bgm = bgm.fade_out(5 * 1000)
    voice = zunda.concat_audio_files(tl['start_time'], tl['audio_file'])
    bgm.overlay(voice).export('outputs/dialogue.wav', format='wav')
    zunda.make_ass_file(
        tl['start_time'], tl['end_time'], tl['character'], tl['text'],
        'outputs/subtitle.ass', font_name='Hiragino Maru Gothic Pro')
    scene.make_video('outputs/video.mp4')
    zunda.add_materials_to_video(
        'outputs/video.mp4', 'outputs/dialogue.wav',
        subtitle_file='outputs/subtitle.ass', dst_file='outputs/video2.mp4')


if __name__ == '__main__':
    main()
