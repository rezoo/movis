# zunda

ずんだもんの解説動画を作るのだ！

# install

```bash
pip install -e .
```

# How to create zunda videos

プロジェクトディレクトリ(``your_project/``)で以下のコマンドを実行するのだ！

```bash
$ zunda init
```

動画に必要なconfigファイルが``your_project/config.yaml``に生成されるのだ！

## Make a timeline

次はVOICE VOXから生成された音声を ``your_project/audio`` ディレクトリに入れた後、次のコマンドを叩くのだ！

```bash
$ zunda make timeline
```

``outputs/timeline.csv``に、スライドの内容や表情を記述するタイムラインファイルが生成されたのだ！
スライドを送る場合は対象のslideカラムに1を入れればいいし、表情はstatusカラムで変えることができるのだ！

終わったらあとは次のコマンドを実行すれば完成なのだ！``your_project/outputs/dst.mp4``に動画が生成されるのだ！

```bash
$ zunda make video
```

「ちょっと時間がかかるわね・・・」

そこは適当に作ったのでしょうがないのだ！
