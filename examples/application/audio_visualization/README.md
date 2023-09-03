# Example of an Audio visualizer

This sample is aimed at enabling users to create simple audio visualizer videos using this library and to create custom layers as desired. By replacing the images in this sample, users can easily create videos of their own musical performances.

Sample videos can be produced with the following commands:

```bash
$ python3 render.py
```

The video should be generated in this directory with the name `output.mp4`.

If you want to create a customized video, you may use the following commands to produce.

```bash
$ python3 render.py --type circle -i music.mp3 -o output.mp4
```

The library supports both circular (`circle`) and linear (`line`) waveforms. If you want to change the background image to something else, either replace `bg.jpg` or specify it explicitly using the `--background` option.