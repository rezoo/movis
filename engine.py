import os
import subprocess
import tempfile
from typing import List, Optional

import pandas as pd
from pydub import AudioSegment
from PIL import Image


def _get_audio_length(filename: str) -> float:
    audio = AudioSegment.from_file(filename, format="wav")
    return audio.duration_seconds


def _concat_audio_files(wav_files: List[str], bgm_file: str, wav_path: str, bgm_volume: int = -15) -> None:
    concatenated_audio = AudioSegment.empty()

    for wav_file in wav_files:
        audio = AudioSegment.from_wav(wav_file)
        concatenated_audio += audio

    # Load BGM
    bgm = AudioSegment.from_wav(bgm_file)
    bgm = bgm + bgm_volume  # Decrease the volume
    # Repeat the BGM to be at least as long as the main audio
    bgm_repeat_times = int(round(
       concatenated_audio.duration_seconds / _get_audio_length(bgm_file)))
    bgm = bgm * bgm_repeat_times
    # Trim the BGM to the same length as the main audio
    bgm = bgm[:len(concatenated_audio)]
    # Overlay the main audio with the BGM
    final_output = concatenated_audio.overlay(bgm)

    final_output.export(wav_path, format="wav")


def _insert_newlines(text: str, max_length: int) -> str:
    import MeCab
    tagger = MeCab.Tagger()
    parsed = tagger.parse(text).split("\n")
    words = [p.split("\t")[0] for p in parsed[:-2]]
    lines = []
    for w in words:
        if len(lines) == 0 or max_length < len(lines[-1]) + len(w):
            lines.append('')
        lines[-1] = lines[-1] + w
    return '\\n'.join(lines)


def _get_paths(src_dir: str, ext: str) -> List[str]:
    return sorted([
        os.path.join(src_dir, f)
        for f in os.listdir(src_dir) if f.endswith(ext)])


def make_wav_file(audio_dir: str, bgm_path: str, dst_wav_path: str, bgm_volume: int = -20) -> None:
    wav_files = _get_paths(audio_dir, '.wav')
    _concat_audio_files(wav_files, bgm_path, dst_wav_path, bgm_volume)


def make_srt_file(audio_dir: str, dst_srt_path: str, dst_timeline_path: Optional[str] = None) -> None:

    wav_files = _get_paths(audio_dir, '.wav')
    txt_files = _get_paths(audio_dir, '.txt')

    frame = []
    with open(dst_srt_path, 'w') as srt:
        start_time = 0
        for i, (wav_file, txt_file) in enumerate(zip(wav_files, txt_files)):
            duration = _get_audio_length(wav_file)
            end_time = start_time + duration
            srt.write('{}\n'.format(i + 1))
            srt.write('{:02d}:{:02d}:{:02d},{:03d} --> {:02d}:{:02d}:{:02d},{:03d}\n'.format(
                int(start_time / 3600), int((start_time / 60) % 60),
                int(start_time % 60), int((start_time % 1) * 1000),
                int(end_time / 3600), int((end_time / 60) % 60),
                int(end_time % 60), int((end_time % 1) * 1000),
            ))

            text = open(txt_file, 'r', encoding='utf-8-sig').read()
            modified_text = _insert_newlines(text, max_length=25)
            srt.write(modified_text + '\n\n')
            if dst_timeline_path:
                character_dict = {
                    '四国めたん（ノーマル）': 'metan',
                    'ずんだもん（ノーマル）': 'zunda',
                }
                filename = os.path.splitext(os.path.basename(txt_file))[0]
                character = filename.split('_')[1]
                frame.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'character': character_dict[character],
                    'text': text,
                    'slide': 0,
                    'status': 'n',
                })
            start_time = end_time
    if dst_timeline_path:
        frame = pd.DataFrame(frame)
        frame.to_csv(dst_timeline_path, index=False)


def make_still_video(bg_path: str, audio_path: str, dst_video_path: str) -> None:
    import ffmpeg

    ffmpeg.input(bg_path, loop=1, r=5).output(
        dst_video_path, vcodec='libx264', tune='stillimage',
        acodec='aac', audio_bitrate='192k', pix_fmt='yuv420p', shortest=None) \
        .global_args('-i', audio_path).run(overwrite_output=True)


def make_still_images(
        bg_path: str, character_dir: str, slide_path: str,
        timeline_path: str, video_config: str, dst_dir: str) -> None:
    from pdf2image import convert_from_path

    bg_image = Image.open(bg_path).convert('RGBA')
    timeline = pd.read_csv(timeline_path)
    character_imgs = {}
    for c in os.listdir(character_dir):
        c_dir = os.path.join(character_dir, c)
        if not os.path.isdir(c_dir):
            continue
        status_filenames = [x for x in os.listdir(c_dir) if os.path.splitext(x)[1] == '.png']
        ratio = video_config['character'][c]['ratio']
        images = {}
        for fn in status_filenames:
            key = os.path.splitext(fn)[0]
            img = Image.open(os.path.join(c_dir, fn)).convert('RGBA')
            w, h = img.size
            img = img.resize(
                (int(w * ratio), int(h * ratio)), Image.Resampling.BICUBIC)
            images[key] = img
        character_imgs[c] = images
    status = {k: v['initial_status'] for k, v in video_config['character'].items()}

    slide_images = []
    for img in convert_from_path(slide_path):
        img = img.convert('RGBA')
        w, h = img.size
        ratio = video_config['slide']['ratio']
        img = img.resize(
            (int(w * ratio), int(h * ratio)), Image.Resampling.BICUBIC)
        slide_images.append(img)
    slide_number = 0
    os.makedirs(dst_dir, exist_ok=True)
    for t, row in timeline.iterrows():
        status[row['character']] = row['status']
        slide_number += row['slide']

        img_t = bg_image.copy()
        img_t.alpha_composite(slide_images[slide_number], video_config['slide']['offset'])
        for c, imgs in character_imgs.items():
            img_t.alpha_composite(imgs[status[c]], video_config['character'][c]['offset'])
        dst_path = os.path.join(dst_dir, f'{t:03d}.png')
        img_t.save(dst_path)


def make_video_from_images(
        images_dir: str, audio_path: str, timeline_path: str, dst_video_path: str) -> None:
    timeline = pd.read_csv(timeline_path)
    temp_file = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', dir=script_dir, delete=False)
        duration_path = temp_file.name
        duration = timeline['end_time'] - timeline['start_time']
        with open(duration_path, 'w') as fp:
            for i, d in enumerate(duration):
                file_path = os.path.join(images_dir, f'{i:03d}.png')
                fp.write(f"file '{file_path}'\n")
                fp.write(f"duration {d}\n")
        
            # Handle the known bug where the last duration of ffmpeg is ignored
            fp.write(f"file '{file_path}'\n")
            fp.write(f"duration 0.1\n")

        # Use subprocess since concat is not available in ffmpeg-python
        command = f'ffmpeg -y -f concat -i {duration_path} -i {audio_path} -c:v libx264 -c:a aac -b:a 192k -pix_fmt yuv420p -r 30 {dst_video_path}'
        subprocess.call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print("subprocess execution failed:", e)
    finally:
        if temp_file:
            temp_file.close()
            os.remove(duration_path)


def make_subtitle_video(
        video_path: str, srt_path: str, timeline_path: str, dst_video_path: str) -> None:
    import ffmpeg
    timeline = pd.read_csv(timeline_path)
    length = timeline['end_time'].iloc[-1]

    video_option_str = f"subtitles={srt_path}:force_style='Fontsize=26,FontName=Hiragino Maru Gothic Pro"
    ffmpeg.input(video_path).output(
        dst_video_path, vf=video_option_str, t=length).run(overwrite_output=True)


if __name__ == '__main__':
    audio_dir = 'audio'
    slide_path = 'slide.pdf'

    bgm_path = 'assets/bgm2.wav'
    character_dir = 'assets/character'
    audio_path = 'outputs/dialogue.wav'
    subtitle_path = 'outputs/subtitile.srt'
    timeline_path = 'outputs/timeline.csv'
    images_dir = 'outputs/images'
    bg_path = 'assets/bg.png'
    video_wo_subtitle_path = 'outputs/zunda_bg.mp4'
    video_path = 'outputs/zunda.mp4'

    make_wav_file(audio_dir, bgm_path, audio_path)
    make_srt_file(audio_dir, subtitle_path, dst_timeline_path=timeline_path)
    config = {
        'slide': {'offset': (250, 22), 'ratio': 0.71},
        'character': {
            'zunda': {'initial_status': 'n', 'offset': (1400, 300), 'ratio': 0.7},
            'metan': {'initial_status': 'n', 'offset': (-300, 400), 'ratio': 0.7},
        }
    }
    make_still_images(bg_path, character_dir, slide_path, timeline_path, config, images_dir)
    make_video_from_images(images_dir, audio_path, timeline_path, video_wo_subtitle_path)
    make_subtitle_video(video_wo_subtitle_path, subtitle_path, timeline_path, video_path)
