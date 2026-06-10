"""
Gemma 4 OpenVINO GenAI demo.

Before running this script, export the model with:
uv run python export_gemma4.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ONEDNN_VERBOSE", "0")
os.environ.setdefault("DNNL_VERBOSE", "0")
os.environ.setdefault("OV_LOG_LEVEL", "ERROR")

MODEL_CONFIGS = {
    "google/gemma-4-E4B-it": Path("gemma-4-E4B-it_ov_int8"),
}
NPU_MODEL_CONFIGS = {
    "google/gemma-4-E4B-it": Path("gemma-4-E4B-it_ov_int4_npu"),
}
DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_DEVICE = "GPU"
DEFAULT_PROMPT = "OpenVINO上でGemma 4を動かす利点を3つ説明してください。"


def configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def export_command(model_id: str, model_dir: Path, npu: bool = False) -> str:
    default_dir = NPU_MODEL_CONFIGS[model_id] if npu else MODEL_CONFIGS[model_id]
    command = f"uv run python export_gemma4.py --model-id {model_id}"
    if npu:
        command += " --npu"
    if model_dir != default_dir:
        command += f" --output-dir {model_dir}"
    return command


def device_contains_npu(device: str) -> bool:
    normalized = device.upper().replace(":", ",").replace(";", ",")
    return any(part.strip() == "NPU" or part.strip().startswith("NPU.") for part in normalized.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a prompt with a pre-exported Gemma 4 OpenVINO GenAI model."
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
        help="Directory containing the exported OpenVINO model. Default depends on --model-id and --device.",
    )
    parser.add_argument(
        "--prompt",
        help=f"User prompt. Default in one-shot mode: {DEFAULT_PROMPT}",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help="System prompt for chat mode.",
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


def validate_openvino_device(device: str) -> None:
    import openvino as ov

    available_devices = set(ov.Core().available_devices)
    requested_roots = [
        part.strip().split(".", 1)[0]
        for part in device.upper().replace(":", ",").replace(";", ",").split(",")
        if part.strip()
    ]
    missing_devices = [
        requested_root
        for requested_root in requested_roots
        if requested_root not in {"AUTO", "HETERO", "MULTI"} and requested_root not in available_devices
    ]
    if not missing_devices:
        return

    raise SystemExit(
        "Requested OpenVINO device is not available: "
        f"{', '.join(missing_devices)}. Available devices: {', '.join(sorted(available_devices))}"
    )


def create_pipeline(model_dir: Path, device: str):
    import openvino_genai as ov_genai

    started_at = time.perf_counter()
    if device_contains_npu(device):
        pipe = ov_genai.VLMPipeline(
            model_dir,
            "NPU",
            MAX_PROMPT_LEN=1024,
            MIN_RESPONSE_LEN=1,
            GENERATE_HINT="FAST_COMPILE",
        )
        return pipe, time.perf_counter() - started_at

    pipe = ov_genai.VLMPipeline(model_dir, device)
    return pipe, time.perf_counter() - started_at


def count_output_tokens(pipe, text: str) -> int:
    if not text:
        return 0

    encoded = pipe.get_tokenizer().encode(text)
    return int(encoded.input_ids.shape[-1])


def print_generation_metrics(
    model_load_seconds: float,
    generation_started_at: float,
    first_token_at: float | None,
    finished_at: float,
    token_count: int,
) -> None:
    first_token_seconds = None if first_token_at is None else first_token_at - generation_started_at
    decode_seconds = None if first_token_at is None else max(finished_at - first_token_at, 1e-9)
    tokens_per_second = 0.0 if decode_seconds is None else token_count / decode_seconds
    first_token_text = "n/a" if first_token_seconds is None else f"{first_token_seconds:.3f}s"

    print(
        "[metrics] "
        f"model_load: {model_load_seconds:.3f}s | "
        f"time_to_first_token: {first_token_text} | "
        f"output_tokens: {token_count} | "
        f"tokens/sec: {tokens_per_second:.2f}"
    )


def generate_with_metrics(
    pipe,
    prompt: str,
    args: argparse.Namespace,
    model_load_seconds: float,
    apply_chat_template: bool,
) -> str:
    first_token_at: float | None = None

    def mark_first_token(_: str) -> bool:
        nonlocal first_token_at
        if first_token_at is None:
            first_token_at = time.perf_counter()
        return False

    generation_started_at = time.perf_counter()
    result = pipe.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        apply_chat_template=apply_chat_template,
        streamer=mark_first_token,
    )
    finished_at = time.perf_counter()
    result_text = str(result)
    token_count = count_output_tokens(pipe, result_text)
    print_generation_metrics(
        model_load_seconds,
        generation_started_at,
        first_token_at,
        finished_at,
        token_count,
    )
    return result_text


def generate_text(args: argparse.Namespace) -> str:
    pipe, model_load_seconds = create_pipeline(args.model_dir, args.device)
    return generate_with_metrics(pipe, args.prompt, args, model_load_seconds, apply_chat_template=True)


def run_chat(args: argparse.Namespace) -> None:
    pipe, model_load_seconds = create_pipeline(args.model_dir, args.device)
    pipe.start_chat(args.system_prompt)

    try:
        print("Interactive chat. Type /exit or /quit to stop.")
        pending_prompt = args.prompt

        while True:
            if pending_prompt is None:
                try:
                    user_text = input("user> ").strip()
                except EOFError:
                    print()
                    break
            else:
                user_text = pending_prompt
                pending_prompt = None
                print(f"user> {user_text}")

            if not user_text:
                continue
            if user_text.lower() in {"/exit", "/quit"}:
                break

            result = generate_with_metrics(
                pipe,
                user_text,
                args,
                model_load_seconds,
                apply_chat_template=False,
            )
            print(f"assistant> {str(result).strip()}")
    finally:
        pipe.finish_chat()


def main() -> None:
    configure_stdio()
    args = parse_args()
    args.device = resolve_openvino_device(args.device)
    validate_openvino_device(args.device)

    npu = device_contains_npu(args.device)
    ensure_model_dir(args.model_id, args.model_dir, npu=npu)

    print(f"[1/2] Loading OpenVINO GenAI model: {args.model_id} from {args.model_dir}")
    if args.chat:
        run_chat(args)
        return

    result = generate_text(args)

    print(f"[2/2] Response from {args.device}")
    print(result.strip())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.") from None
