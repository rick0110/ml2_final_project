#!/usr/bin/env python3
"""Extract and structure TensorBoard logs for a training run.

The script reads TensorBoard event files from a log directory, exports a
structured JSON report, and optionally writes a combined CSV for scalars and
text/summary data that can be passed to another AI system.

By default it keeps media lightweight: text summaries, scalar series,
histogram statistics, and tensor metadata are exported in JSON. Optionally, it
can also materialize image/audio media files to disk.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tensorboard.backend.event_processing import event_accumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured data from TensorBoard event logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log-dir", type=str, required=True, help="TensorBoard log directory or experiment directory")
    parser.add_argument("--output-json", type=str, required=True, help="Path to the JSON report to generate")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional path to a combined CSV of scalar series")
    parser.add_argument("--media-dir", type=str, default=None, help="Optional directory where image/audio events are exported")
    parser.add_argument("--include-binary", action="store_true", help="Embed image/audio bytes as base64 in JSON instead of only metadata")
    parser.add_argument("--max-events-per-tag", type=int, default=None, help="Optional cap for the number of events exported per tag")
    return parser.parse_args()


def resolve_log_dir(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        tensorboard_dir = candidate / "tensorboard"
        if tensorboard_dir.is_dir():
            return tensorboard_dir
        return candidate
    return candidate


def safe_tag_name(tag: str) -> str:
    return tag.replace("/", "__").replace(" ", "_").replace(".", "_")


def limited(items: Iterable[Any], max_items: Optional[int]) -> Iterable[Any]:
    if max_items is None:
        yield from items
        return

    for index, item in enumerate(items):
        if index >= max_items:
            break
        yield item


def tensor_proto_to_python(tensor_proto: Any) -> Any:
    if tensor_proto is None:
        return None

    if getattr(tensor_proto, "string_val", None):
        values = [value.decode("utf-8", errors="replace") for value in tensor_proto.string_val]
        if len(values) == 1:
            return values[0]
        return values

    if getattr(tensor_proto, "float_val", None):
        values = [float(value) for value in tensor_proto.float_val]
        if len(values) == 1:
            return values[0]
        return values

    if getattr(tensor_proto, "double_val", None):
        values = [float(value) for value in tensor_proto.double_val]
        if len(values) == 1:
            return values[0]
        return values

    if getattr(tensor_proto, "int_val", None):
        values = [int(value) for value in tensor_proto.int_val]
        if len(values) == 1:
            return values[0]
        return values

    if getattr(tensor_proto, "bool_val", None):
        values = [bool(value) for value in tensor_proto.bool_val]
        if len(values) == 1:
            return values[0]
        return values

    try:
        tensor_shape = [dimension.size for dimension in tensor_proto.tensor_shape.dim]
    except Exception:
        tensor_shape = None

    dtype_name = str(getattr(tensor_proto, "dtype", "unknown"))
    return {"dtype": dtype_name, "shape": tensor_shape}


def encode_bytes(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def event_to_dict(event: Any, include_binary: bool = False, media_path: Optional[Path] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"wall_time": float(getattr(event, "wall_time", 0.0)), "step": int(getattr(event, "step", 0))}

    if hasattr(event, "value"):
        result["value"] = float(event.value)

    if hasattr(event, "histogram_value") and event.histogram_value is not None:
        histo = event.histogram_value
        result["histogram"] = {
            "min": float(histo.min),
            "max": float(histo.max),
            "num": float(histo.num),
            "sum": float(histo.sum),
            "sum_squares": float(histo.sum_squares),
            "bucket_limit": [float(value) for value in histo.bucket_limit],
            "bucket": [float(value) for value in histo.bucket],
        }

    if hasattr(event, "tensor_proto") and event.tensor_proto is not None:
        result["tensor"] = tensor_proto_to_python(event.tensor_proto)

    if hasattr(event, "image") and event.image is not None:
        image = event.image
        result["image"] = {
            "height": int(image.height),
            "width": int(image.width),
            "colorspace": int(getattr(image, "colorspace", 0)),
            "encoded_bytes": len(image.encoded_image_string),
        }
        if media_path is not None:
            media_path.parent.mkdir(parents=True, exist_ok=True)
            media_path.write_bytes(image.encoded_image_string)
            result["image"]["file"] = str(media_path)
        elif include_binary:
            result["image"]["encoded_b64"] = encode_bytes(image.encoded_image_string)

    if hasattr(event, "audio") and event.audio is not None:
        audio = event.audio
        result["audio"] = {
            "sample_rate": float(audio.sample_rate),
            "length_frames": int(getattr(audio, "length_frames", 0)),
            "num_channels": int(getattr(audio, "num_channels", 0)),
            "content_type": str(getattr(audio, "content_type", "")),
            "encoded_bytes": len(audio.encoded_audio_string),
        }
        if media_path is not None:
            media_path.parent.mkdir(parents=True, exist_ok=True)
            media_path.write_bytes(audio.encoded_audio_string)
            result["audio"]["file"] = str(media_path)
        elif include_binary:
            result["audio"]["encoded_b64"] = encode_bytes(audio.encoded_audio_string)

    return result


def build_accumulator(log_dir: Path) -> event_accumulator.EventAccumulator:
    accumulator = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE,
    )
    accumulator.Reload()
    return accumulator


def extract_scalars(accumulator: event_accumulator.EventAccumulator, max_events: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    scalars: Dict[str, List[Dict[str, Any]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = []
        for event in limited(accumulator.Scalars(tag), max_events):
            events.append(
                {
                    "step": int(event.step),
                    "wall_time": float(event.wall_time),
                    "value": float(event.value),
                }
            )
        scalars[tag] = events
    return scalars


def extract_histograms(accumulator: event_accumulator.EventAccumulator, max_events: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    histograms: Dict[str, List[Dict[str, Any]]] = {}
    for tag in accumulator.Tags().get("histograms", []):
        events = []
        for event in limited(accumulator.Histograms(tag), max_events):
            histo = getattr(event, "histogram_value", None)
            if histo is None:
                histo = getattr(event, "histo", None)
            events.append(
                {
                    "step": int(event.step),
                    "wall_time": float(event.wall_time),
                    "min": float(getattr(histo, "min", 0.0)),
                    "max": float(getattr(histo, "max", 0.0)),
                    "num": float(getattr(histo, "num", 0.0)),
                    "sum": float(getattr(histo, "sum", 0.0)),
                    "sum_squares": float(getattr(histo, "sum_squares", 0.0)),
                }
            )
        histograms[tag] = events
    return histograms


def extract_tensors(accumulator: event_accumulator.EventAccumulator, max_events: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    tensors: Dict[str, List[Dict[str, Any]]] = {}
    for tag in accumulator.Tags().get("tensors", []):
        events = []
        for event in limited(accumulator.Tensors(tag), max_events):
            events.append(
                {
                    "step": int(event.step),
                    "wall_time": float(event.wall_time),
                    "tensor": tensor_proto_to_python(getattr(event, "tensor_proto", None)),
                }
            )
        tensors[tag] = events
    return tensors


def extract_images(
    accumulator: event_accumulator.EventAccumulator,
    max_events: Optional[int],
    media_dir: Optional[Path],
    include_binary: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    images: Dict[str, List[Dict[str, Any]]] = {}
    for tag in accumulator.Tags().get("images", []):
        events = []
        for event in limited(accumulator.Images(tag), max_events):
            media_path = None
            if media_dir is not None:
                media_path = media_dir / "images" / f"{safe_tag_name(tag)}_step{event.step}.png"
            events.append(event_to_dict(event, include_binary=include_binary, media_path=media_path))
        images[tag] = events
    return images


def extract_audio(
    accumulator: event_accumulator.EventAccumulator,
    max_events: Optional[int],
    media_dir: Optional[Path],
    include_binary: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    audio: Dict[str, List[Dict[str, Any]]] = {}
    for tag in accumulator.Tags().get("audio", []):
        events = []
        for event in limited(accumulator.Audio(tag), max_events):
            media_path = None
            if media_dir is not None:
                media_path = media_dir / "audio" / f"{safe_tag_name(tag)}_step{event.step}.wav"
            events.append(event_to_dict(event, include_binary=include_binary, media_path=media_path))
        audio[tag] = events
    return audio


def export_scalars_csv(scalars: Dict[str, List[Dict[str, Any]]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tag", "step", "wall_time", "value"])
        writer.writeheader()
        for tag, events in scalars.items():
            for event in events:
                writer.writerow(
                    {
                        "tag": tag,
                        "step": event["step"],
                        "wall_time": event["wall_time"],
                        "value": event["value"],
                    }
                )


def build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    def count_events(section: Dict[str, List[Dict[str, Any]]]) -> int:
        return sum(len(events) for events in section.values())

    return {
        "scalar_tags": len(report["scalars"]),
        "scalar_events": count_events(report["scalars"]),
        "histogram_tags": len(report["histograms"]),
        "histogram_events": count_events(report["histograms"]),
        "tensor_tags": len(report["tensors"]),
        "tensor_events": count_events(report["tensors"]),
        "image_tags": len(report["images"]),
        "image_events": count_events(report["images"]),
        "audio_tags": len(report["audio"]),
        "audio_events": count_events(report["audio"]),
        "text_tags": sorted(tag for tag in report["tensors"] if any(isinstance(item.get("tensor"), str) for item in report["tensors"][tag])),
    }


def main() -> None:
    args = parse_args()
    log_dir = resolve_log_dir(args.log_dir)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv) if args.output_csv else None
    media_dir = Path(args.media_dir) if args.media_dir else None

    accumulator = build_accumulator(log_dir)

    report: Dict[str, Any] = {
        "log_dir": str(log_dir),
        "tags": accumulator.Tags(),
        "scalars": extract_scalars(accumulator, args.max_events_per_tag),
        "histograms": extract_histograms(accumulator, args.max_events_per_tag),
        "tensors": extract_tensors(accumulator, args.max_events_per_tag),
        "images": extract_images(accumulator, args.max_events_per_tag, media_dir, args.include_binary),
        "audio": extract_audio(accumulator, args.max_events_per_tag, media_dir, args.include_binary),
    }
    report["summary"] = build_summary(report)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if output_csv is not None:
        export_scalars_csv(report["scalars"], output_csv)

    print(f"Saved structured log report to: {output_json}")
    if output_csv is not None:
        print(f"Saved scalar CSV to: {output_csv}")
    if media_dir is not None:
        print(f"Saved media exports to: {media_dir}")


if __name__ == "__main__":
    main()