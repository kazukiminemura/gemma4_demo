"""
Gemma 4 E4B OpenVINO inference demo.

Before running this script, export the model with:
optimum-cli export openvino --model google/gemma-4-E4B-it --task image-text-to-text --trust-remote-code --weight-format int8 gemma-4-E4B-it_ov_int8
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

DEFAULT_MODEL_DIR = Path("gemma-4-E4B-it_ov_int8")
DEFAULT_DEVICE = "AUTO"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a pre-exported google/gemma-4-E4B-it OpenVINO model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing the exported OpenVINO model. Default: {DEFAULT_MODEL_DIR}",
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
    return parser.parse_args()


def ensure_model_dir(model_dir: Path) -> None:
    if model_dir.exists() and any(model_dir.glob("*.xml")):
        return

    raise SystemExit(
        "OpenVINO model files were not found. Export the model first:\n"
        "optimum-cli export openvino --model google/gemma-4-E4B-it "
        "--task image-text-to-text --trust-remote-code --weight-format int8 "
        "gemma-4-E4B-it_ov_int8"
    )


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


def generate_text(args: argparse.Namespace) -> str:
    from optimum.intel.openvino import OVModelForVisualCausalLM
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = OVModelForVisualCausalLM.from_pretrained(
        args.model_dir,
        device=args.device,
        trust_remote_code=True,
    )
    messages = build_messages(args)

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
    ensure_model_dir(args.model_dir)

    print(f"[1/2] Loading OpenVINO model: {args.model_dir}")
    result = generate_text(args)

    print(f"[2/2] Response from {args.device}")
    print(result.strip())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.") from None
