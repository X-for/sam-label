#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests>=2.32,<3",
# ]
# ///

from __future__ import annotations

import argparse
import copy
import glob
import json
import mimetypes
import os
import re
import sys
import time
import tomllib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import requests


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TERMINAL_STATES = {"succeeded", "failed"}
CLIENT_CONFIG_KEYS = {
    "server",
    "inputs",
    "api_key",
    "api_key_env",
    "recursive",
    "batch_size",
    "poll_interval",
    "wait_timeout",
    "request_timeout",
    "output_dir",
    "no_wait",
    "no_download",
    "dry_run",
}


class ClientError(RuntimeError):
    pass


def parse_assignment(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"{option} must use LABEL=VALUE syntax: {value!r}")
    key, item = (part.strip() for part in value.split("=", 1))
    if not key or not item:
        raise argparse.ArgumentTypeError(f"{option} cannot contain a blank label or value")
    return key, item


def discover_images(inputs: Iterable[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = path.rglob("*") if recursive else path.glob("*")
        elif glob.has_magic(raw):
            candidates = (Path(match) for match in glob.iglob(raw, recursive=recursive))
        else:
            raise ClientError(f"input does not exist and is not a valid glob: {raw}")
        found.extend(
            candidate.resolve()
            for candidate in candidates
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        )

    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        identity = str(path).casefold()
        if identity not in seen:
            seen.add(identity)
            unique.append(path)
    unique.sort(key=lambda item: str(item).casefold())
    if not unique:
        raise ClientError("no supported images were found")
    return unique


def _relative_to_config(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else base_dir / path)


def normalize_task_config(task_config: dict[str, Any]) -> dict[str, Any]:
    """Expand the compact TOML label table into the API prompt_groups shape."""
    task = copy.deepcopy(task_config)
    labels = task.pop("labels", None)
    if labels is None:
        return task
    if "prompt_groups" in task:
        raise ClientError("[task].labels and [[task.prompt_groups]] cannot be used together")
    if not isinstance(labels, dict) or not labels:
        raise ClientError("[task.labels] must be a non-empty table")

    aggregation = task.pop("aggregation", "deduplicate")
    merge_iou = task.pop("merge_iou", 0.7)
    groups: list[dict[str, Any]] = []
    for class_id, (label, raw_prompts) in enumerate(labels.items()):
        prompts = [raw_prompts] if isinstance(raw_prompts, str) else raw_prompts
        if (
            not isinstance(prompts, list)
            or not prompts
            or not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts)
        ):
            raise ClientError(
                f"[task.labels].{label!r} must be a prompt string or a non-empty string array"
            )
        groups.append(
            {
                "label": label,
                "class_id": class_id,
                "prompts": prompts,
                "aggregation": aggregation,
                "merge_iou": merge_iou,
            }
        )
    task["prompt_groups"] = groups
    return task


def apply_run_config(args: argparse.Namespace) -> None:
    if not args.config:
        args.config_task = None
        return
    if args.task_json:
        raise ClientError("--config and --task-json cannot be used together")
    try:
        payload = tomllib.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ClientError(f"cannot read run config: {exc}") from exc
    unknown_sections = set(payload) - {"client", "task"}
    if unknown_sections:
        raise ClientError(f"unknown top-level config sections: {sorted(unknown_sections)}")
    client_config = payload.get("client")
    task_config = payload.get("task")
    if not isinstance(client_config, dict):
        raise ClientError("run config requires a [client] table")
    if not isinstance(task_config, dict):
        raise ClientError("run config requires a [task] table")
    unknown_client = set(client_config) - CLIENT_CONFIG_KEYS
    if unknown_client:
        raise ClientError(f"unknown [client] options: {sorted(unknown_client)}")
    if not client_config.get("server"):
        raise ClientError("[client].server is required")
    inputs = client_config.get("inputs")
    if not isinstance(inputs, list) or not inputs or not all(isinstance(item, str) for item in inputs):
        raise ClientError("[client].inputs must be a non-empty string array")
    if "api_key" in client_config and "api_key_env" in client_config:
        raise ClientError("configure only one of [client].api_key or api_key_env")

    base_dir = args.config.resolve().parent
    for key, value in client_config.items():
        if key in {"inputs", "output_dir", "api_key_env"}:
            continue
        setattr(args, key, value)
    args.inputs = [_relative_to_config(value, base_dir) for value in inputs]
    if "output_dir" in client_config:
        args.output_dir = Path(_relative_to_config(str(client_config["output_dir"]), base_dir))
    if "api_key_env" in client_config:
        environment_name = str(client_config["api_key_env"])
        args.api_key = os.getenv(environment_name)
        if not args.api_key:
            raise ClientError(f"environment variable {environment_name!r} is not set")
    args.config_task = normalize_task_config(task_config)


def build_task(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "config_task", None) is not None:
        return copy.deepcopy(args.config_task)
    if args.task_json:
        try:
            payload = json.loads(args.task_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ClientError(f"cannot read task JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ClientError("task JSON root must be an object")
        if args.client_reference:
            payload["client_reference"] = args.client_reference
        return payload

    if not args.prompt:
        raise ClientError("provide at least one --prompt or use --task-json")

    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for raw in args.prompt:
        label, prompt = parse_assignment(raw, "--prompt")
        prompts = grouped.setdefault(label, [])
        if prompt not in prompts:
            prompts.append(prompt)

    explicit_ids: dict[str, int] = {}
    for raw in args.class_id or []:
        label, raw_id = parse_assignment(raw, "--class-id")
        try:
            explicit_ids[label] = int(raw_id)
        except ValueError as exc:
            raise ClientError(f"class ID must be an integer: {raw}") from exc
    unknown = set(explicit_ids) - set(grouped)
    if unknown:
        raise ClientError(f"class IDs reference unknown labels: {sorted(unknown)}")

    groups = []
    for index, (label, prompts) in enumerate(grouped.items()):
        groups.append(
            {
                "label": label,
                "class_id": explicit_ids.get(label, index),
                "prompts": prompts,
                "aggregation": args.aggregation,
                "merge_iou": args.merge_iou,
            }
        )
    return {
        "task_type": args.task_type,
        "output_format": args.output_format,
        "client_reference": args.client_reference,
        "prediction": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "max_det": args.max_det,
            "retina_masks": args.retina_masks,
        },
        "prompt_groups": groups,
    }


def batched(items: list[Path], size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class Sam3Client:
    def __init__(self, server: str, api_key: str | None, request_timeout: float):
        self.server = server.rstrip("/")
        self.timeout = (10.0, request_timeout)
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-API-Key"] = api_key

    def close(self) -> None:
        self.session.close()

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        try:
            response = self.session.request(
                method, f"{self.server}{path}", timeout=self.timeout, **kwargs
            )
        except requests.RequestException as exc:
            raise ClientError(f"request failed: {exc}") from exc
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text[:1000]
            raise ClientError(f"HTTP {response.status_code} {path}: {detail}")
        return response

    def create_job(self, task: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/v1/jobs", json=task).json()

    def upload(self, job_id: str, images: list[Path]) -> dict[str, Any]:
        handles = []
        files = []
        try:
            for path in images:
                handle = path.open("rb")
                handles.append(handle)
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                files.append(("files", (path.name, handle, content_type)))
            return self.request("POST", f"/v1/jobs/{job_id}/images", files=files).json()
        finally:
            for handle in handles:
                handle.close()

    def commit(self, job_id: str) -> dict[str, Any]:
        return self.request("POST", f"/v1/jobs/{job_id}/commit").json()

    def status(self, job_id: str) -> dict[str, Any]:
        return self.request("GET", f"/v1/jobs/{job_id}").json()

    def wait(self, job_id: str, interval: float, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last_status = None
        while True:
            job = self.status(job_id)
            if job["status"] != last_status:
                print(f"任务状态: {job['status']}", flush=True)
                last_status = job["status"]
            if job["status"] in TERMINAL_STATES:
                if job["status"] == "failed":
                    raise ClientError(f"job failed: {job.get('error') or 'unknown error'}")
                return job
            if time.monotonic() >= deadline:
                raise ClientError(f"timed out waiting for job {job_id}; the server job is still running")
            time.sleep(interval)

    def download(self, job_id: str, destination: Path) -> Path:
        response = self.request("GET", f"/v1/jobs/{job_id}/result", stream=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        return destination


def default_result_path(output_dir: Path, job_id: str, output_format: str) -> Path:
    suffix = "zip" if output_format == "yolo" else "json"
    return output_dir / f"{job_id}-{output_format}.{suffix}"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch upload images to the remote SAM3 pre-annotation API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("inputs", nargs="*", help="Image files, directories, or glob patterns")
    parser.add_argument("--config", type=Path, help="Complete TOML run configuration")
    parser.add_argument("--server", help="API base URL, e.g. http://192.168.110.101:8000")
    parser.add_argument("--api-key", help="X-API-Key value")
    parser.add_argument("--task-json", type=Path, help="Use a complete JobCreate JSON file")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Repeatable LABEL=PROMPT mapping, e.g. --prompt person='sleeping person'",
    )
    parser.add_argument("--class-id", action="append", help="Optional repeatable LABEL=INTEGER override")
    parser.add_argument("--task-type", choices=("detect", "segment", "semantic"), default="segment")
    parser.add_argument("--output-format", choices=("yolo", "coco"), default="yolo")
    parser.add_argument(
        "--aggregation",
        choices=("deduplicate", "keep_all", "best", "union"),
        default="deduplicate",
    )
    parser.add_argument("--merge-iou", type=float, default=0.7)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=644, help="Must be a multiple of SAM3 stride 14")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--retina-masks", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--client-reference")
    parser.add_argument("--recursive", action="store_true", help="Scan input directories recursively")
    parser.add_argument("--batch-size", type=int, default=16, help="Images per upload request")
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--wait-timeout", type=float, default=7200.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-wait", action="store_true", help="Return after commit")
    parser.add_argument("--no-download", action="store_true", help="Wait but do not download result")
    parser.add_argument("--dry-run", action="store_true", help="Print task and images without network calls")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.server:
        raise ClientError("provide --server or set [client].server in --config")
    if not args.inputs:
        raise ClientError("provide image inputs or set [client].inputs in --config")
    if args.batch_size < 1:
        raise ClientError("--batch-size must be at least 1")
    if args.imgsz % 14 and not args.task_json:
        raise ClientError(f"--imgsz must be divisible by 14; try {(args.imgsz // 14 + 1) * 14}")
    if args.aggregation == "union" and args.task_type == "detect" and not args.task_json:
        raise ClientError("union aggregation cannot be used with detect")
    if args.poll_interval <= 0 or args.wait_timeout <= 0 or args.request_timeout <= 0:
        raise ClientError("timeout and polling values must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    try:
        apply_run_config(args)
        if args.config:
            # Parse again with file values as defaults. Explicit command-line
            # options then naturally take precedence over the TOML config.
            parser.set_defaults(**vars(args))
            args = parser.parse_args(raw_argv)
        validate_args(args)
        images = discover_images(args.inputs, args.recursive)
        task = build_task(args)
        print(f"发现图片: {len(images)} 张")
        print("任务参数:")
        print(json.dumps(task, ensure_ascii=False, indent=2))
        if args.dry_run:
            for image in images:
                print(image)
            return 0

        client = Sam3Client(args.server, args.api_key, args.request_timeout)
        try:
            job = client.create_job(task)
            job_id = job["id"]
            print(f"任务已创建: {job_id}", flush=True)
            uploaded = 0
            for batch in batched(images, args.batch_size):
                job = client.upload(job_id, batch)
                uploaded += len(batch)
                print(f"上传进度: {uploaded}/{len(images)}", flush=True)
            job = client.commit(job_id)
            print(f"任务已提交: {job['status']}", flush=True)
            if args.no_wait:
                print(f"查询地址: {args.server.rstrip('/')}/v1/jobs/{job_id}")
                return 0
            job = client.wait(job_id, args.poll_interval, args.wait_timeout)
            if args.no_download:
                print(f"结果地址: {args.server.rstrip('/')}{job['result_url']}")
                return 0
            output_format = task.get("output_format", "coco")
            destination = default_result_path(args.output_dir, job_id, output_format)
            client.download(job_id, destination)
            print(f"结果已保存: {destination.resolve()}")
            return 0
        finally:
            client.close()
    except (ClientError, OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
