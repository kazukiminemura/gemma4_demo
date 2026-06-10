# Gemma 4 OpenVINO Demo

Gemma 4 系モデルを OpenVINO IR に変換してから推論する Python デモです。
実装の流れは OpenVINO Notebook の Gemma 4 サンプルを参考にしています。

参照:
- https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/gemma4/gemma4.ipynb
- https://huggingface.co/google/gemma-4-E4B-it
- https://huggingface.co/google/gemma-4-12B-it
- https://huggingface.co/google/gemma-4-12B-it-assistant

## セットアップ

事前に Hugging Face 側で Gemma の利用許諾に同意し、必要に応じてログインしてください。

```bash
uv sync
```

## モデル変換

推論前に OpenVINO IR へ変換します。デフォルトでは `google/gemma-4-E4B-it` を使います。

```bash
uv run python export_gemma4.py
```

現時点の `optimum-intel` OpenVINO exporter は `google/gemma-4-12B-it` の `gemma4_unified` と
`google/gemma-4-12B-it-assistant` の `gemma4_unified_assistant` には未対応です。
12B assistant は 12B 本体の speculative decoding 用 drafter なので、OpenVINO IR へ変換する対象にはできません。

## 実行

```bash
# テキスト推論
uv run python gemma4_demo.py

# システムプロンプト指定
uv run python gemma4_demo.py --system-prompt "You are a concise technical assistant." --prompt "OpenVINOの特徴を説明して"

# 画像付き推論
uv run python gemma4_demo.py --image https://raw.githubusercontent.com/google-gemma/cookbook/refs/heads/main/Demos/sample-data/GoldenGate.png --prompt "この画像を説明して"
```

デフォルトの実行デバイスは `CPU` です。`--device AUTO` を指定した場合、このデモは `AUTO:GPU,CPU` に変換します。

Intel NPU で実行する場合は、NPU 向けに INT4 symmetric のモデルを別途変換してください。

```bash
uv run python export_gemma4.py --npu
uv run python gemma4_demo.py --device NPU --prompt "OpenVINOの特徴を短く説明して"
```

`--device NPU` では Optimum の `generate()` ではなく OpenVINO GenAI の `VLMPipeline` を使います。
NPU では greedy decoding のみを使うため、サンプリングは無効です。

CLI でチャットする場合:

```bash
uv run python gemma4_demo.py --chat
```

最初の入力をコマンドラインで渡してからチャットを続けることもできます。

```bash
uv run python gemma4_demo.py --chat --prompt "OpenVINOの特徴を短く説明して"
```

チャットでは各応答の末尾に以下のようなメトリクスを表示します。

```text
[metrics] FTTP: 0.532s | output tokens: 128 | total: 8.421s | tokens/sec: 16.22
```

`FTTP` は生成開始から最初の出力 token までの時間です。
`tokens/sec` は最初の出力 token 以降の生成速度です。

デフォルトでは `gemma-4-E4B-it_ov_int8` から変換済みモデルを読み込みます。
別の場所に変換した場合は `--model-dir` で指定してください。

```bash
uv run python gemma4_demo.py --model-dir path/to/exported_model
```

## オプション

```bash
uv run python gemma4_demo.py --help
```
