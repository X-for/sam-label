from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import gc
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .postprocess import Candidate, aggregate
from .schemas import ImageResult, JobRecord, JobResult, ModelStatus


class Sam3Runtime:
    """Owns the only model instance and serializes all GPU access."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._predictor: Any = None
        self._lock = asyncio.Lock()
        self._loading = False
        self._last_used = time.monotonic()
        self._queue_idle = True
        self._last_error: str | None = None
        # CUDA and the stateful Ultralytics predictor always run on the same
        # dedicated OS thread. The asyncio lock additionally prevents overlap.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sam3-gpu"
        )

    def status(self) -> ModelStatus:
        return ModelStatus(
            loaded=self._predictor is not None,
            loading=self._loading,
            lifecycle=self.settings.lifecycle,
            device=self.settings.device,
            model_path=str(self.settings.model_path),
            last_error=self._last_error,
        )

    async def prewarm(self) -> None:
        try:
            async with self._lock:
                await self._ensure_loaded_locked()
        except Exception as exc:
            self._last_error = str(exc)

    async def predict(self, job: JobRecord) -> JobResult:
        async with self._lock:
            await self._ensure_loaded_locked()
            try:
                return await self._run_sync(self._predict_sync, job)
            finally:
                self._last_used = time.monotonic()
                if self.settings.lifecycle == "per_job":
                    await self._unload_locked()

    def queue_became_active(self) -> None:
        self._queue_idle = False

    def queue_became_idle(self) -> None:
        self._queue_idle = True
        # The idle timeout starts only after the last queued job, including its
        # export work, has completed.
        self._last_used = time.monotonic()

    async def maybe_unload_idle(self) -> None:
        if (
            self.settings.lifecycle == "resident"
            or self._predictor is None
            or not self._queue_idle
        ):
            return
        if time.monotonic() - self._last_used < self.settings.idle_unload_seconds:
            return
        async with self._lock:
            if (
                self._queue_idle
                and time.monotonic() - self._last_used >= self.settings.idle_unload_seconds
            ):
                await self._unload_locked()

    async def close(self) -> None:
        async with self._lock:
            await self._unload_locked()
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def _ensure_loaded_locked(self) -> None:
        if self._predictor is not None:
            return
        model_path = self.settings.model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"SAM3 model not found: {model_path}")
        self._loading = True
        self._last_error = None
        try:
            self._predictor = await self._run_sync(self._load_sync, model_path)
            self._last_used = time.monotonic()
        except Exception as exc:
            self._last_error = str(exc)
            raise
        finally:
            self._loading = False

    def _load_sync(self, model_path: Path) -> Any:
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = {
            "model": str(model_path),
            "task": "segment",
            "mode": "predict",
            "device": self.settings.device,
            "quantize": self.settings.quantize,
            "verbose": False,
            "save": False,
        }
        predictor = SAM3SemanticPredictor(overrides=overrides)
        # Predictor construction alone is lazy in Ultralytics. setup_model makes
        # task creation genuinely overlap checkpoint loading with image upload.
        predictor.setup_model()
        return predictor

    async def _unload_locked(self) -> None:
        if self._predictor is None:
            return
        predictor, self._predictor = self._predictor, None
        await self._run_sync(self._dispose_sync, predictor)

    async def _run_sync(self, function: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(function, *args))

    @staticmethod
    def _dispose_sync(predictor: Any) -> None:
        del predictor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _predict_sync(self, job: JobRecord) -> JobResult:
        predictor = self._predictor
        params = job.config.prediction
        for name in ("conf", "iou", "imgsz", "max_det", "retina_masks"):
            if hasattr(predictor.args, name):
                setattr(predictor.args, name, getattr(params, name))

        image_results: list[ImageResult] = []
        for image in job.images:
            predictor.set_image(image.stored_path)
            group_candidates: list[tuple[Any, list[Candidate]]] = []
            width = height = 0
            try:
                for group in job.config.prompt_groups:
                    candidates: list[Candidate] = []
                    for prompt in group.prompts:
                        results = predictor(text=[prompt])
                        extracted, width, height = self._extract(results, prompt, width, height)
                        candidates.extend(extracted)
                    group_candidates.append((group, candidates))
            finally:
                reset = getattr(predictor, "reset_image", None)
                if callable(reset):
                    reset()

            annotations = []
            next_id = 1
            for group, candidates in group_candidates:
                processed = aggregate(candidates, group, job.config.task_type, next_id)
                annotations.extend(processed)
                next_id += len(processed)
            image_results.append(
                ImageResult(
                    image_id=image.image_id,
                    file_name=image.original_name,
                    width=width,
                    height=height,
                    annotations=annotations,
                )
            )

        return JobResult(
            job_id=job.id,
            task_type=job.config.task_type,
            images=image_results,
            meta={
                "model": str(self.settings.model_path),
                "device": self.settings.device,
                "bbox_format": "xywh",
                "segmentation_format": "polygon",
            },
        )

    @staticmethod
    def _extract(results: Any, prompt: str, fallback_width: int, fallback_height: int):
        candidates: list[Candidate] = []
        width, height = fallback_width, fallback_height
        if results is None:
            return candidates, width, height
        if not isinstance(results, (list, tuple)):
            results = [results]
        for result in results:
            shape = getattr(result, "orig_shape", None)
            if shape:
                height, width = int(shape[0]), int(shape[1])
            boxes_obj = getattr(result, "boxes", None)
            if boxes_obj is None:
                continue
            xyxy = boxes_obj.xyxy.detach().cpu().tolist()
            scores = boxes_obj.conf.detach().cpu().tolist()
            polygons_by_mask: list[list[list[float]]] = [[] for _ in xyxy]
            masks_obj = getattr(result, "masks", None)
            if masks_obj is not None:
                for index, polygon_set in enumerate(masks_obj.xy):
                    if index >= len(polygons_by_mask):
                        break
                    if getattr(polygon_set, "ndim", 0) == 2 and len(polygon_set) >= 3:
                        polygons_by_mask[index].append(
                            [round(float(value), 3) for point in polygon_set.tolist() for value in point]
                        )
            for index, (box, score) in enumerate(zip(xyxy, scores)):
                candidates.append(
                    Candidate(
                        bbox_xyxy=tuple(float(value) for value in box),
                        score=float(score),
                        prompt=prompt,
                        polygons=polygons_by_mask[index],
                    )
                )
        return candidates, width, height
