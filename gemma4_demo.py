"""
Gemma 4 OpenVINO inference demo.

Before running this script, export the model with:
uv run python export_gemma4.py

"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from io import BytesIO
from pathlib import Path

MODEL_CONFIGS = {
    "google/gemma-4-E4B-it": Path("gemma-4-E4B-it_ov_int8"),
}
NPU_MODEL_CONFIGS = {
    "google/gemma-4-E4B-it": Path("gemma-4-E4B-it_ov_int4_npu"),
}
DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_DEVICE = "CPU"
DEFAULT_PROMPT = "OpenVINO上でGemma 4を動かす利点を3つ説明してください。"


def export_command(model_id: str, model_dir: Path, npu: bool = False) -> str:
    default_dir = NPU_MODEL_CONFIGS[model_id] if npu else MODEL_CONFIGS[model_id]
    command = f"uv run python export_gemma4.py --model-id {model_id}"
    if npu:
        command += " --npu"
    if model_dir != default_dir:
        command += f" --output-dir {model_dir}"
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a pre-exported Gemma 4 OpenVINO model."
    )
    parser.add_argument(
        "--model-id",
        choices=sorted(MODEL_CONFIGS),
        default=DEFAULT_MODEL_ID,
        help=f"Gemma model to use. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory containing the exported OpenVINO model. Default depends on --model-id.",
    )
    parser.add_argument(
        "--prompt",
        help=f"User prompt. Default in one-shot mode: {DEFAULT_PROMPT}",
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
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start an interactive CLI chat. Type /exit or /quit to stop.",
    )
    args = parser.parse_args()
    if args.model_dir is None:
        args.model_dir = (
            NPU_MODEL_CONFIGS[args.model_id]
            if device_contains_npu(args.device)
            else MODEL_CONFIGS[args.model_id]
        )
    if not args.chat and args.prompt is None:
        args.prompt = DEFAULT_PROMPT
    return args


def ensure_model_dir(model_id: str, model_dir: Path, npu: bool = False) -> None:
    if model_dir.exists() and any(model_dir.glob("*.xml")):
        return

    raise SystemExit(
        "OpenVINO model files were not found. Export the model first:\n"
        f"{export_command(model_id, model_dir, npu=npu)}"
    )


def device_contains_npu(device: str) -> bool:
    normalized = device.upper().replace(":", ",").replace(";", ",")
    return any(part.strip() == "NPU" or part.strip().startswith("NPU.") for part in normalized.split(","))


def resolve_openvino_device(device: str) -> str:
    requested_device = device.strip()
    upper_device = requested_device.upper()

    if upper_device != "AUTO":
        return requested_device

    import openvino as ov

    available_devices = set(ov.Core().available_devices)
    auto_devices = [candidate for candidate in ("GPU", "CPU") if candidate in available_devices]
    if not auto_devices:
        return "CPU"
    return "AUTO:" + ",".join(auto_devices)


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


def load_openvino_model(args: argparse.Namespace):
    from optimum.intel.openvino import OVModelForVisualCausalLM
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = OVModelForVisualCausalLM.from_pretrained(
        args.model_dir,
        device=args.device,
        trust_remote_code=True,
    )
    return processor, model


def build_inputs(processor, messages: list[dict], args: argparse.Namespace):
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    return processor(text=prompt_text, return_tensors="pt")


def build_multimodal_inputs(processor, messages: list[dict], args: argparse.Namespace):
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )


def parse_response(processor, response: str) -> str:
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(response)
        if isinstance(parsed, dict):
            return parsed.get("text") or str(parsed)
        return str(parsed)

    return response


def is_npu_device(device: str) -> bool:
    return device_contains_npu(device)


def build_prompt_text(processor, messages: list[dict], args: argparse.Namespace) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )


def generate_text(args: argparse.Namespace) -> str:
    processor, model = load_openvino_model(args)
    messages = build_messages(args)

    if args.image:
        inputs = build_multimodal_inputs(processor, messages, args)
    else:
        inputs = build_inputs(processor, messages, args)

    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    return parse_response(processor, response)


def generate_text_genai(args: argparse.Namespace) -> str:
    import openvino_genai as ov_genai
    from transformers import AutoProcessor

    if args.image:
        raise SystemExit("--device NPU currently supports text-only prompts in this demo.")

    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    messages = build_messages(args)
    prompt_text = build_prompt_text(processor, messages, args)
    pipe = ov_genai.VLMPipeline(
        args.model_dir,
        "NPU",
        MAX_PROMPT_LEN=1024,
        MIN_RESPONSE_LEN=min(args.max_new_tokens, 32),
        GENERATE_HINT="FAST_COMPILE",
    )
    results = pipe.generate(
        prompt_text,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        apply_chat_template=False,
    )
    response = str(results)
    return parse_response(processor, response)


def print_generation_metrics(metrics: dict[str, float | int | None]) -> None:
    first_token_at = metrics["first_token_at"]
    token_count = int(metrics["token_count"] or 0)
    started_at = float(metrics["started_at"] or 0)
    finished_at = float(metrics["finished_at"] or time.perf_counter())

    fttp = None if first_token_at is None else float(first_token_at) - started_at
    decode_seconds = None if first_token_at is None else max(finished_at - float(first_token_at), 1e-9)
    tokens_per_second = 0.0 if decode_seconds is None else token_count / decode_seconds

    fttp_text = "n/a" if fttp is None else f"{fttp:.3f}s"
    total_seconds = max(finished_at - started_at, 0.0)
    print(
        f"\n[metrics] FTTP: {fttp_text} | output tokens: {token_count} | "
        f"total: {total_seconds:.3f}s | tokens/sec: {tokens_per_second:.2f}",
        file=sys.stderr,
    )


def stream_chat_response(processor, model, messages: list[dict], args: argparse.Namespace) -> str:
    from transformers.generation.streamers import TextIteratorStreamer

    class MetricsTextIteratorStreamer(TextIteratorStreamer):
        def __init__(self, *streamer_args, metrics: dict[str, float | int | None], **streamer_kwargs):
            super().__init__(*streamer_args, **streamer_kwargs)
            self.metrics = metrics

        def put(self, value):
            if self.skip_prompt and self.next_tokens_are_prompt:
                return super().put(value)

            if len(value.shape) > 1:
                token_count = int(value.shape[-1])
            else:
                token_count = int(value.numel())

            if self.metrics["first_token_at"] is None:
                self.metrics["first_token_at"] = time.perf_counter()
            self.metrics["token_count"] = int(self.metrics["token_count"] or 0) + token_count
            return super().put(value)

    inputs = build_inputs(processor, messages, args)
    metrics: dict[str, float | int | None] = {
        "started_at": time.perf_counter(),
        "first_token_at": None,
        "finished_at": None,
        "token_count": 0,
    }
    streamer = MetricsTextIteratorStreamer(
        processor,
        skip_prompt=True,
        skip_special_tokens=False,
        metrics=metrics,
    )
    generation_error: list[BaseException] = []

    def generate() -> None:
        try:
            model.generate(**inputs, max_new_tokens=args.max_new_tokens, streamer=streamer)
        except BaseException as exc:
            generation_error.append(exc)
        finally:
            metrics["finished_at"] = time.perf_counter()

    thread = threading.Thread(target=generate, daemon=True)
    thread.start()

    chunks: list[str] = []
    print("assistant> ", end="", flush=True)
    for chunk in streamer:
        print(chunk, end="", flush=True)
        chunks.append(chunk)

    thread.join()
    if generation_error:
        raise generation_error[0]

    if metrics["finished_at"] is None:
        metrics["finished_at"] = time.perf_counter()
    print_generation_metrics(metrics)
    return parse_response(processor, "".join(chunks))


def run_chat(args: argparse.Namespace) -> None:
    if args.image:
        raise SystemExit("--chat currently supports text-only prompts. Omit --image for CLI chat.")

    processor, model = load_openvino_model(args)
    messages: list[dict] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    print("Interactive chat. Type /exit or /quit to stop.", file=sys.stderr)
    if args.prompt:
        pending_prompt = args.prompt
    else:
        pending_prompt = None

    while True:
        if pending_prompt is None:
            try:
                user_text = input("user> ").strip()
            except EOFError:
                print(file=sys.stderr)
                break
        else:
            user_text = pending_prompt
            pending_prompt = None
            print(f"user> {user_text}")

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break

        messages.append({"role": "user", "content": user_text})
        assistant_text = stream_chat_response(processor, model, messages, args).strip()
        messages.append({"role": "assistant", "content": assistant_text})


def main() -> None:
    args = parse_args()
    args.device = resolve_openvino_device(args.device)
    npu = is_npu_device(args.device)
    ensure_model_dir(args.model_id, args.model_dir, npu=npu)

    print(f"[1/2] Loading OpenVINO model: {args.model_id} from {args.model_dir}")
    if npu and args.chat:
        raise SystemExit("--chat is not supported on NPU yet. Run one-shot prompts with --device NPU.")

    if args.chat:
        run_chat(args)
        return

    result = generate_text_genai(args) if npu else generate_text(args)

    print(f"[2/2] Response from {args.device}")
    print(result.strip())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.") from None
