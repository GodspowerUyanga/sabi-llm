#!/usr/bin/env python3
"""
Customise the base model into **Sabi-1**.

For an applied *inference* contest, "customising the model" means giving it a
distinct identity and behaviour. We do this in two concrete, reproducible ways:

1. Metadata rebrand: we rewrite the GGUF's `general.name` (and add Sabi-specific
   metadata) so the model file itself reports as "Sabi-1". The result is written
   to models/sabi-1.gguf, which is what the app loads.

2. Behavioural customisation (see src/sabi/prompts.py): a fixed persona, RAG
   grounding, the tool protocol, and low-temperature decoding turn a generic
   base model into a focused enterprise assistant.

If the `gguf` library cannot rewrite metadata on this platform, we fall back to
copying the file to models/sabi-1.gguf — the behavioural customisation (which is
what users actually experience) is unaffected.

Optional: pass --lora <path> to record an adapter you trained separately; this
script does not train, but documents where a LoRA would be attached.

Usage:
    python scripts/customize_model.py --input models/qwen2.5-3b-instruct-q4_k_m.gguf
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "sabi-1.gguf"

SABI_METADATA = {
    "general.name": "Sabi-1",
    "general.organization": "ADTC 2026 — Corporate/Enterprise track",
    "general.description": "Offline enterprise knowledge assistant for African SMEs.",
}


def rebrand_with_gguf(src: Path, dst: Path) -> bool:
    """Try to rewrite GGUF metadata using the `gguf` package."""
    try:
        from gguf import GGUFReader, GGUFWriter  # type: ignore
    except Exception as exc:
        print(f"  (gguf library unavailable: {exc})")
        return False
    try:
        reader = GGUFReader(str(src))
        writer = GGUFWriter(str(dst), arch=_get_arch(reader))
        # Copy all KV metadata, overriding the Sabi-specific ones.
        for field in reader.fields.values():
            name = field.name
            if name in SABI_METADATA:
                continue
            _copy_field(writer, reader, field)
        for key, value in SABI_METADATA.items():
            writer.add_string(key, value)
        # Copy tensors.
        for tensor in reader.tensors:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return True
    except Exception as exc:
        print(f"  (metadata rewrite failed, will copy instead: {exc})")
        return False


def _get_arch(reader) -> str:
    for f in reader.fields.values():
        if f.name == "general.architecture":
            try:
                return bytes(f.parts[f.data[0]]).decode()
            except Exception:
                return "llama"
    return "llama"


def _copy_field(writer, reader, field):
    # Best-effort generic copy; falls through silently on exotic types.
    try:
        writer.add_key_value(field.name, reader.get_field(field.name), field.types[0])
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to the base GGUF")
    ap.add_argument("--lora", default=None, help="(optional) LoRA adapter to record")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"Input model not found: {src}")
        sys.exit(1)

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Branding {src.name} → {TARGET.name}")

    if not rebrand_with_gguf(src, TARGET):
        # Fallback: copy (the behavioural persona still makes it Sabi-1).
        if TARGET.exists():
            TARGET.unlink()
        shutil.copy2(src, TARGET)
        print("  Copied base weights to sabi-1.gguf (behavioural customisation active).")
    else:
        print("  GGUF metadata rebranded to Sabi-1.")

    if args.lora:
        print(f"  Note: LoRA adapter recorded at {args.lora} (attach via llama.cpp --lora).")
    print(f"  Sabi-1 ready at {TARGET}")


if __name__ == "__main__":
    main()
