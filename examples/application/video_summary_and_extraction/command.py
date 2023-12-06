import argparse
import tempfile
from pathlib import Path

import ffmpeg
import librosa
import movis as mv
import numpy as np
import openai
import pandas as pd
import soundfile as sf


def extract(args: argparse.Namespace):
    audio, sr = librosa.load(str(args.input))

    non_silent_slices = librosa.effects.split(
        audio, top_db=args.threshold, frame_length=args.frame_length)
    non_silent_regions = non_silent_slices / sr
    dst_audio = np.concatenate([audio[start:end] for start, end in non_silent_slices])
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        audio_file = temp_dir / 'audio.wav'
        sf.write(str(audio_file), dst_audio, sr)
        ffmpeg.input(str(audio_file)).output(str(args.output), b='128k').run(overwrite_output=True)

    non_silent_frame = pd.DataFrame(non_silent_regions, columns=['start_time', 'end_time'])
    non_silent_frame.to_csv(args.region, sep='\t', index=False)


def transcribe(args: argparse.Namespace):
    client = openai.OpenAI()
    transcription = client.audio.transcriptions.create(
        model="whisper-1", file=open(args.input, 'rb'), response_format='verbose_json')
    rows = []
    for segment in transcription.segments:
        rows.append({
            'start_time': segment['start'],
            'end_time': segment['end'],
            'text': segment['text'],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, sep='\t', index=False)


def render(args: argparse.Namespace):
    subtitle = pd.read_csv(args.subtitle, sep='\t')
    region = pd.read_csv(args.region, sep='\t')
    font_name = 'M PLUS 1p'
    font_style = 'Medium'

    video = mv.trim(mv.layer.Video(args.input, audio=True), region['start_time'], region['end_time'])
    scene = mv.layer.Composition(video.size, video.duration)
    scene.add_layer(video)

    bg_layer = mv.layer.LuminanceMatte(
        mask=mv.layer.Gradient(
            size=(video.size[0], 200),
            start_point=(0.0, 0.0), end_point=(0.0, 200.0),
            start_color='#000000', end_color='#ffffff'),
        target=mv.layer.Image.from_color((video.size[0], 200), '#1b4290'))
    bg = scene.add_layer(
        bg_layer,
        position=(video.size[0] / 2, video.size[1]),
        origin_point=mv.Direction.BOTTOM_CENTER,
        opacity=0.75)
    bg.opacity.enable_motion().extend([0.0, 2.0], [0.0, bg.opacity.init_value], ['ease_out'])
    item = scene.add_layer(
        mv.layer.Text.from_timeline(
            subtitle['start_time'], subtitle['end_time'], subtitle['text'],
            font_size=64, font_family=font_name, font_style=font_style, line_spacing=100, contents=[
                mv.layer.StrokeProperty(color='#13557b', width=12),
                mv.layer.FillProperty(color='#ffffff')],
            duration=scene.duration, text_alignment='center'),
        position=(video.size[0] / 2, video.size[1] - 20.0),
        origin_point=mv.Direction.BOTTOM_CENTER)
    item.add_effect(mv.effect.DropShadow(offset=5.0, opacity=0.4, radius=3.0))

    with scene.preview(level=1):
        scene.write_video(args.output, audio=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()
    parser_extract = subparsers.add_parser(
        'extract', help='Extract non-silent regions from audio and generate subtitle')
    parser_extract.add_argument('-i', '--input', default='video.mp4', type=Path)
    parser_extract.add_argument('-o', '--output', default='truncated_audio.m4a', type=Path)
    parser_extract.add_argument('--region', default='non_silent_regions.tsv', type=Path)
    parser_extract.add_argument('--threshold', default=30, type=int)
    parser_extract.add_argument('--frame_length', default=512, type=int)
    parser_extract.set_defaults(func=extract)

    parser_transcribe = subparsers.add_parser('transcribe', help='Transcribe audio with OpenAI API')
    parser_transcribe.add_argument('-i', '--input', default='truncated.m4a', type=Path)
    parser_transcribe.add_argument('-o', '--output', default='subtitle.tsv', type=Path)
    parser_transcribe.set_defaults(func=transcribe)

    parser_render = subparsers.add_parser('render', help='Render video with subtitle and non-silent regions')
    parser_render.add_argument('-i', '--input', default='video.mp4', type=Path)
    parser_render.add_argument('--region', default='non_silent_regions.tsv', type=Path)
    parser_render.add_argument('--subtitle', default='subtitle.tsv', type=Path)
    parser_render.add_argument('-o', '--output', default='video_truncated.mp4', type=Path)
    parser_render.set_defaults(func=render)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
