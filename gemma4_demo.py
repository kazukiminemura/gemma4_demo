"""
Gemma 4 E4B OpenVINO demo.

Reference:
https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/gemma4/gemma4.ipynb
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from io import BytesIO
from pathlib import Path

DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_WEIGHT_FORMAT = "int8"
DEFAULT_EXPORT_ROOT = Path("models")
DEFAULT_DEVICE = "AUTO"


def has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run google/gemma-4-E4B-it inference with OpenVINO."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model id. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--prompt",
        default="OpenVINO上でGemma 4を動かす利点を3つ説明してください。",
        help="User prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt.",
    )
    parser.add_argument(
        "--image",
        help="Optional local image path or URL for multimodal inference.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"OpenVINO device name. Default: {DEFAULT_DEVICE}",
    )
    parser.add_argument(
        "--weight-format",
        choices=("fp16", "int8", "int4"),
        default=DEFAULT_WEIGHT_FORMAT,
        help=f"Weight format used during export. Default: {DEFAULT_WEIGHT_FORMAT}",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Directory containing the exported OpenVINO model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Gemma 4 thinking mode.",
    )
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="Re-export the model even if the OpenVINO files already exist.",
    )
    return parser.parse_args()


def resolve_export_dir(args: argparse.Namespace) -> Path:
    if args.export_dir is not None:
        return args.export_dir

    model_name = args.model_id.split("/")[-1]
    return DEFAULT_EXPORT_ROOT / model_name / args.weight_format.upper()


def has_exported_model(export_dir: Path) -> bool:
    return export_dir.exists() and any(export_dir.glob("*.xml"))


def run_export(model_id: str, export_dir: Path, weight_format: str) -> None:
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    export_args = [
        "export",
        "openvino",
        "--model",
        model_id,
        "--task",
        "image-text-to-text",
        "--weight-format",
        weight_format,
        str(export_dir),
    ]

    command = None
    if has_module("optimum.commands.optimum_cli"):
        command = [sys.executable, "-m", "optimum.commands.optimum_cli", *export_args]
    elif has_module("optimum"):
        command = [sys.executable, "-m", "optimum.commands.optimum_cli", *export_args]
    else:
        command = ["optimum-cli", *export_args]

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "optimum export CLI が見つかりません。gemma4_demo.py を実行している Python 環境に "
            "optimum-intel をインストールしてください。"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"OpenVINO export failed with exit code {exc.returncode}.") from exc


def load_image(path_or_url: str):
    import requests
    from PIL import Image

    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=60)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    image_path = Path(path_or_url)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    return Image.open(image_path).convert("RGB")


def build_messages(args: argparse.Namespace) -> list[dict]:
    messages: list[dict] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    if args.image:
        image = load_image(args.image)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        )
        return messages

    messages.append({"role": "user", "content": args.prompt})
    return messages


def generate_text(args: argparse.Namespace, export_dir: Path) -> str:
    from optimum.intel.openvino import OVModelForVisualCausalLM
    from transformers import AutoProcessor

    messages = build_messages(args)
    processor = AutoProcessor.from_pretrained(export_dir, trust_remote_code=True)
    model = OVModelForVisualCausalLM.from_pretrained(
        export_dir,
        device=args.device,
        trust_remote_code=True,
    )

    if args.image:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
    else:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        inputs = processor(text=prompt_text, return_tensors="pt")

    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(response)
        if isinstance(parsed, dict):
            return parsed.get("text") or str(parsed)
        return str(parsed)

    return response


def main() -> None:
    args = parse_args()
    export_dir = resolve_export_dir(args)

    if args.force_export or not has_exported_model(export_dir):
        print(f"[1/3] Exporting {args.model_id} to OpenVINO: {export_dir}")
        run_export(args.model_id, export_dir, args.weight_format)
    else:
        print(f"[1/3] Reusing exported model: {export_dir}")

    print(f"[2/3] Running inference on {args.device}")
    result = generate_text(args, export_dir)

    print("[3/3] Response")
    print(result.strip())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.") from None
