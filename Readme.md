# Gemma 4 OpenVINO Demo

Gemma 4 系モデルを OpenVINO IR に変換してから推論する Python デモです。
CPU 側の Python 生成ループを避けるため、実行時は OpenVINO GenAI の `VLMPipeline` を使います。
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

# プロンプト指定
uv run python gemma4_demo.py --prompt "OpenVINOの特徴を説明して"

# チャット
uv run python gemma4_demo.py --chat

# 最初の入力を渡してからチャット
uv run python gemma4_demo.py --chat --prompt "OpenVINOの特徴を短く説明して"
```

デフォルトの実行デバイスは `GPU` です。CPU で実行したい場合だけ `--device CPU` を指定してください。
`--device AUTO` を指定した場合、このデモは `AUTO:GPU,CPU` に変換します。

Intel NPU で実行する場合は、NPU 向けに INT4 symmetric のモデルを別途変換してください。

```bash
uv run python export_gemma4.py --npu
uv run python gemma4_demo.py --device NPU --prompt "OpenVINOの特徴を短く説明して"
```

`gemma4_demo.py` は OpenVINO GenAI の `VLMPipeline` を使います。
CPU 負荷を減らすため、チャット履歴は `pipe.start_chat()` に持たせ、Transformers/Optimum の `generate()` は使いません。
ストリーミングと画像入力は含めていません。
サンプリングは無効で、greedy decoding のみを使います。

推論時は以下のメトリクスを表示します。

```text
[metrics] model_load: 12.326s | time_to_first_token: 0.592s | output_tokens: 4 | tokens/sec: 79.62
```

`model_load` は `VLMPipeline` のロード時間、`time_to_first_token` は `generate()` 開始から最初の出力までの時間です。
`tokens/sec` は最初の出力以降の生成速度です。

デフォルトでは `gemma-4-E4B-it_ov_int8` から変換済みモデルを読み込みます。
別の場所に変換した場合は `--model-dir` で指定してください。

```bash
uv run python gemma4_demo.py --model-dir path/to/exported_model
```

## オプション

```bash
uv run python gemma4_demo.py --help
```
