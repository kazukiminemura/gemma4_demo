# Gemma 4 OpenVINO Demo

`google/gemma-4-E4B-it` を OpenVINO IR に変換してから推論する Python デモです。
実装の流れは OpenVINO Notebook の Gemma 4 サンプルを参考にしています。

参照:
- https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/gemma4/gemma4.ipynb
- https://huggingface.co/google/gemma-4-E4B-it

## セットアップ

事前に Hugging Face 側で Gemma の利用許諾に同意し、必要に応じてログインしてください。

```bash
pip install -r requirements.txt
```

## モデル変換

推論前に OpenVINO IR へ変換します。

```bash
optimum-cli export openvino --model google/gemma-4-E4B-it --task image-text-to-text --trust-remote-code --weight-format int8 gemma-4-E4B-it_ov_int8
```

## 実行

```bash
# テキスト推論
python gemma4_demo.py

# システムプロンプト指定
python gemma4_demo.py --system-prompt "You are a concise technical assistant." --prompt "OpenVINOの特徴を説明して"

# 画像付き推論
python gemma4_demo.py --image https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/GoldenGate.png --prompt "この画像を説明して"
```

デフォルトでは `gemma-4-E4B-it_ov_int8` から変換済みモデルを読み込みます。
別の場所に変換した場合は `--model-dir` で指定してください。

```bash
python gemma4_demo.py --model-dir path/to/exported_model
```

## オプション

```bash
python gemma4_demo.py --help
```
