# Gemma 4 OpenVINO Demo

`google/gemma-4-E4B-it` を OpenVINO IR に変換して推論する Python デモです。
実装の流れは OpenVINO Notebook の Gemma 4 サンプルを参考にしています。

参照:
- https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/gemma4/gemma4.ipynb
- https://huggingface.co/google/gemma-4-E4B-it

## セットアップ

事前に Hugging Face 側で Gemma の利用許諾に同意し、必要に応じてログインしてください。

```bash
pip install git+https://github.com/rkazants/optimum-intel.git@support_gemma_4
pip install --pre -U openvino openvino-tokenizers nncf --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
pip install transformers==5.5.0
pip install -r requirements.txt
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

初回実行時は `optimum-cli export openvino` でモデルを `models/gemma-4-E4B-it/INT8` に変換します。
2回目以降は変換済みモデルを再利用します。

## オプション

```bash
python gemma4_demo.py --help
```

`requirements.txt` は補助パッケージだけを入れます。
`optimum-intel` / `openvino` / `transformers` は依存解決順の都合で README の順番どおりに個別インストールしてください。
