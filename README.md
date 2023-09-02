<div align="center">
<img src="https://github.com/rezoo/movis/blob/main/images/movis_logo.png?raw=true" width="800" alt="logo"></img>
</div>

# Movis: Video Editing as a Code

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/rezoo/movis)
![Continuous integration](https://github.com/rezoo/movis/actions/workflows/python-package.yml/badge.svg)

## What is Movis?

Movisは動画制作用のPythonエンジンである。このライブラリを用いて、ユーザはプレゼン動画、解説動画、トレーニング動画、ゲーム実況を含む、多くの動画教材をPythonを通して制作できる。

他の多くの動画制作ソフトウェアとは異なり、MovisはGUIを搭載していない。これは初学者においては不適であるが、動画制作の自動化においては都合が良い。具体的には、エンジニアが自身のAIモデルを利用して、顔画像の匿名化などの処理を自動化したい場合、あるいは、動画の変化点を検知することで要約動画を自動的に生成したい場合などに、Movisは有効である。また、LLMなどのプログラミングと親和性の高い対話型のインターフェイスを利用することで、動画編集を自動化することもできるだろう。

Movisは多くの動画制作用のソフトウェアと同じように、"コンポジション"を一つの編集単位とする。コンポジションに多くのレイヤを追加し、そのレイヤの各属性を時間軸にそって動かすことで動画を作成する。状況に応じて、対象のレイヤにエフェクトを適用することもできる。具体的なコードを以下に示す。

```python
import movis as mv

scene = mv.layer.Composition(size=(1920, 1080), duration=5.0)
scene.add_layer(mv.layer.Rectangle(scene.size, color='#fb4562'))
scene.add_layer(
    mv.text.Text('Hello World!', font_size=128, font_family='Helvetica', color='#ffffff'),
    name='text')
scene['text'].add_effect(mv.layer.DropShadow(offset=5.0))
scene['text'].scale.enable_motion().extend(
    keyframes=[0.0, 1.0], values=[0.0, 1.0], motion_types=['ease_in_out'])
scene['text'].opacity.enable_motion().extend([0.0, 1.0], [0.0, 1.0])

scene.write_video('output.mp4')
```

コンポジションはレイヤとしても利用できる。複数のコンポジションとレイヤを組み合わせることで、複雑な動画を最終的に作成する。

## Simple implementation of custom layers and effects

### custom layers

MovisはPythonで書かれた独自のレイヤとエフェクトを追加できる。これらのモジュールの実装に要求される仕様は単純で、具体的には、独自のレイヤの実装には次の、時間から`(H, W, 4)`のシェイプをもつ、RGBAオーダーで`np.uint8`のdtypeをもつ`np.ndarray`もしくは`None`を返す関数を実装するだけで良い。

```python
import numpy as np
import movis as mv

size = (640, 480)

def get_radial_gradient_image(time: float) -> None | np.ndarray:
    if time < 0.:
        return None
    center = np.array([size[0] // 2, size[1] // 2])
    radius = min(size)
    inds = np.mgrid[:size[1], :size[0]] - center[:, None, None]
    r = np.sqrt((inds ** 2).sum(axis=0))
    p = (np.clip(r / radius, 0, 1) * 255).astype(np.uint8)
    img = np.zeros(size[1], size[0], 4, dype=np.uint8)
    img[:, :, :3] = p[:, :, None]
    img[:, :, 3] = 255
    return img

scene = mv.layer.Composition(size, duration=5.0)
scene.add_layer(get_radial_gradient_image)
scene.write_video('output.mp4')
```

レイヤの長さを明示したい場合には`duration`プロパティが要求される。キャッシュ機構を利用して静止画が続くレイヤのレンダリングを高速化したい場合には`get_key(time: float)`メソッドを実装する。

```python
class RadialGradientLayer:
    def __init__(self, size: tuple[int, int], duration: float):
        self.size = size
        self.duration = duration
        self.center = np.array([size[0] // 2, size[1] // 2])
    
    def get_key(self, time: float) -> Hashable:
        # Returns 0 since the same image is always returned
        return 0
    
    def __call__(self, time: float) -> None | np.ndarray:
        # ditto.
```

### custom effects

レイヤのエフェクトも、同様の手順でシンプルに実装できる。

```python
def apply_gaussian_blur(prev_image: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(prev_image, ksize=(7, 7))

scene = mv.layer.Composition(size=(1920, 1080), duration=5.0)
scene.add_layer(mv.layer.Rectangle(scene.size, color='#fb4562'))
scene.add_layer(
    mv.text.Text('Hello World!', font_size=128, font_family='Helvetica', color='#ffffff'),
    name='text')
scene['text'].add_effect(apply_gaussian_blur)
```



## Installation

Movisは純粋なPythonライブラリであり、Python Package Index経由でイントールできる:

```bash
# PyPI
$ pip install movis
```

われわれはMovisがPython 3.9から3.11まで動作することを確認している。

## License

MIT License (`LICENSE` ファイルを参照)。
