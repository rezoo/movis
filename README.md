<div align="center">
<img src="https://github.com/rezoo/movis/blob/main/images/movis_logo.jpg?raw=true" width="500" alt="logo"></img>
</div>

# Movis: Video Editing as a Code

Movisは動画制作用のPythonエンジンである。このライブラリは以下を目的とする:

* **動画教材の作成**: プレゼン動画、解説動画、トレーニング動画、ゲーム実況を含む、多くの動画教材をこのライブラリを通じて直感的に制作する。
* **動画制作の自動化**: 多くの動画編集ソフトウェアはGUIを通して操作するが、これは動画教材制作の自動化には不適である。Pythonを経由することで、特にLLMなどのAIモデルを利用した動画制作の自動化を容易にする。
* **教育用途**: 動画を作るという成果物が現れやすい教材を通して、初学者にプログラミングの楽しさを伝える。
* **リアルタイム映像配信**: その他にGPUを使用した映像のリアルタイム編集・配信も考えているが、優先事項ではない。

## Installation

### Pip

```bash
pip install -e .
```

## How to use

本ライブラリは多くのテンプレートとなるスクリプトを `examples` で提供している。使用したい用途に合致したプロジェクトディレクトリに移動し、スクリプトを編集・実行されたい。本ライブラリの仕様について知りたい場合は、 `tutorials` ディレクトリを参照されたい。

## License

MIT License (`LICENSE` ファイルを参照)。
