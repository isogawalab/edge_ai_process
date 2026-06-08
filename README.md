# 実行方法
- 入力ディレクトリにはスライドのデータセット収集（4/5）の手順に沿って作成したディレクトリを指定してください(スライド内ではmydata)
- 出力ディレクトリには訓練と評価データセットが分割されたものが出力されます(スライドではmydataset)


```bash
uv run prepare_dataset.py --input <入力ディレクトリ> --output <出力ディレクトリ>
```
もしくは
```bash
python prepare_dataset.py --input <入力ディレクトリ> --output <出力ディレクトリ>
```
