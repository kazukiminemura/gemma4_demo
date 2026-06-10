"""
Export a Gemma 4 model to OpenVINO IR.

This wrapper always passes --library transformers to optimum-cli so the exporter
does not try to infer the library by listing Hugging Face repository files first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODEL_CONFIGS = {
    "google/gemma-4-E4B-it": Path("gemma-4-E4B-it_ov_int8"),
}
NPU_MODEL_CONFIGS = {
    "google/gemma-4-E4B-it": Path("gemma-4-E4B-it_ov_int4_npu"),
}
UNSUPPORTED_MODEL_MESSAGES = {
    "google/gemma-4-12B-it": (
        "google/gemma-4-12B-it uses model_type `gemma4_unified`, which is not "
        "supported by the current optimum-intel OpenVINO exporter."
    ),
    "google/gemma-4-12B-it-assistant": (
        "google/gemma-4-12B-it-assistant is a speculative decoding drafter for "
        "google/gemma-4-12B-it and uses model_type `gemma4_unified_assistant`, "
        "which is not supported by the current optimum-intel OpenVINO exporter."
    ),
}
DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Gemma 4 model to OpenVINO IR.")
    parser.add_argument(
        "--model-id",
        choices=sorted([*MODEL_CONFIGS, *UNSUPPORTED_MODEL_MESSAGES]),
        default=DEFAULT_MODEL_ID,
        help=f"Gemma model to export. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Default depends on --model-id.",
    )
    parser.add_argument(
        "--weight-format",
        default="int8",
        choices=["fp32", "fp16", "int8", "int4", "mxfp4", "nf4", "cb4"],
        help="OpenVINO weight format. Default: int8",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--npu",
        action="store_true",
        help="Export an INT4 symmetric model into the NPU default directory.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    if args.model_id in UNSUPPORTED_MODEL_MESSAGES:
        raise SystemExit(
            f"{UNSUPPORTED_MODEL_MESSAGES[args.model_id]}\n"
            "Use google/gemma-4-E4B-it for this OpenVINO demo, or run the 12B "
            "models with Transformers until optimum-intel adds gemma4_unified export support."
        )

    output_dir = args.output_dir or (NPU_MODEL_CONFIGS[args.model_id] if args.npu else MODEL_CONFIGS[args.model_id])
    weight_format = "int4" if args.npu and args.weight_format == "int8" else args.weight_format
    command = [
        sys.executable,
        "-m",
        "optimum.commands.optimum_cli",
        "export",
        "openvino",
        "--model",
        args.model_id,
        "--task",
        "image-text-to-text",
        "--library",
        "transformers",
        "--trust-remote-code",
        "--weight-format",
        weight_format,
    ]
    if args.npu and weight_format == "int4":
        command.extend(["--sym", "--ratio", "1.0", "--group-size", "128"])
    if args.cache_dir:
        command.extend(["--cache_dir", str(args.cache_dir)])
    command.append(str(output_dir))
    return command


def patch_gemma4_exporter() -> None:
    from optimum.exporters.openvino import model_patcher

    original_enter = model_patcher.Gemma4LMModelPatcher.__enter__
    if getattr(original_enter, "_gemma4_demo_patched", False):
        return

    def patched_enter(self):
        result = original_enter(self)
        source_layer_by_type = {}
        for decoder_layer in self._model.model.language_model.layers:
            attention = decoder_layer.self_attn
            if getattr(attention, "store_full_length_kv", False):
                source_layer_by_type[attention.layer_type] = attention.layer_idx

        for decoder_layer in self._model.model.language_model.layers:
            attention = decoder_layer.self_attn
            if getattr(attention, "is_kv_shared_layer", False):
                attention.kv_shared_layer_index = source_layer_by_type[attention.layer_type]
        return result

    patched_enter._gemma4_demo_patched = True
    model_patcher.Gemma4LMModelPatcher.__enter__ = patched_enter


def export_with_openvino_api(args: argparse.Namespace) -> None:
    from optimum.exporters.openvino.__main__ import main_export
    from optimum.intel.openvino.configuration import OVConfig, OVWeightQuantizationConfig

    if args.model_id in UNSUPPORTED_MODEL_MESSAGES:
        raise SystemExit(
            f"{UNSUPPORTED_MODEL_MESSAGES[args.model_id]}\n"
            "Use google/gemma-4-E4B-it for this OpenVINO demo, or run the 12B "
            "models with Transformers until optimum-intel adds gemma4_unified export support."
        )

    output_dir = args.output_dir or (NPU_MODEL_CONFIGS[args.model_id] if args.npu else MODEL_CONFIGS[args.model_id])
    weight_format = "int4" if args.npu and args.weight_format == "int8" else args.weight_format

    quantization_config = None
    if weight_format in {"int8", "int4"}:
        quantization_config = OVWeightQuantizationConfig(
            bits=8 if weight_format == "int8" else 4,
            dtype=weight_format,
            sym=args.npu if weight_format == "int4" else False,
            ratio=1.0,
            group_size=128 if weight_format == "int4" else -1,
            processor=args.model_id,
            tokenizer=args.model_id,
        )

    patch_gemma4_exporter()
    main_export(
        model_name_or_path=args.model_id,
        output=output_dir,
        task="image-text-to-text",
        framework="pt",
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        trust_remote_code=True,
        ov_config=OVConfig(quantization_config=quantization_config) if quantization_config else None,
        stateful=True,
        convert_tokenizer=True,
        library_name="transformers",
    )


def main() -> None:
    args = parse_args()
    if args.npu:
        export_with_openvino_api(args)
        return

    command = build_command(args)
    print("Running:")
    print(" ".join(command))
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
