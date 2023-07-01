import os

import ffmpeg
import numpy as np
import pandas as pd
import imageio
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from zunda.utils import _get_audio_dataframe


def _get_character_imgs(character_dir: str, video_config: dict):
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
    return character_imgs


def _get_slide_imgs(slide_path: str, slide_config: dict):
    slide_images = []
    for img in convert_from_path(slide_path):
        img = img.convert('RGBA')
        w, h = img.size
        ratio = slide_config['ratio']
        img = img.resize(
            (int(w * ratio), int(h * ratio)), Image.Resampling.BICUBIC)
        slide_images.append(img)
    return slide_images


def render_video(
        bg_path: str, character_dir: str, slide_path: str,
        timeline_path: str, video_config: dict, audio_dir: str,
        dst_video_path: str, fps: float = 30.0) -> None:
    delta = 1 / fps

    bg_image = Image.open(bg_path).convert('RGBA')
    audio_df = _get_audio_dataframe(audio_dir)
    timeline = pd.read_csv(timeline_path)
    timeline = pd.merge(timeline, audio_df, left_index=True, right_index=True)
    character_imgs = _get_character_imgs(character_dir, video_config)
    slide_images = _get_slide_imgs(slide_path, video_config['slide'])

    time = 0.0
    slide_number = 0
    status = {k: v['initial_status'] for k, v in video_config['character'].items()}
    writer = imageio.get_writer(
        dst_video_path, fps=fps, codec='libx264',
        macro_block_size=None)
    for _, row in tqdm(timeline.iterrows(), total=len(timeline)):
        status[row['character']] = row['status']
        slide_number += row['slide']

        img_t = bg_image.copy()
        img_t.alpha_composite(slide_images[slide_number], tuple(video_config['slide']['offset']))
        for c, imgs in character_imgs.items():
            img_t.alpha_composite(imgs[status[c]], tuple(video_config['character'][c]['offset']))
        img_t = np.asarray(img_t)

        frames = np.arange(time, row['end_time'], delta)
        for _ in frames:
            writer.append_data(img_t)
        time = frames[-1] + delta
    writer.close()


def render_subtitle_video(
        video_path: str, subtitle_path: str, audio_path: str, dst_video_path: str) -> None:
    video_option_str = f"ass={subtitle_path}"
    video_input = ffmpeg.input(video_path)
    audio_input = ffmpeg.input(audio_path)
    output = ffmpeg.output(
        video_input.video, audio_input.audio, dst_video_path,
        vf=video_option_str, acodec='aac', ab='128k')
    output.run(overwrite_output=True)
