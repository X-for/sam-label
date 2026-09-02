import asyncio
import json
import time

from fastapi.testclient import TestClient

from sam_api.app import app
from sam_api.model_runtime import Sam3Runtime
from sam_api.schemas import Annotation, ImageResult, JobResult


async def fake_prewarm(self):
    raise AssertionError("creating or uploading a job must not prewarm the model")


async def fake_close(self):
    return None


async def fake_predict(self, job, progress_callback=None):
    image = job.images[0]
    if progress_callback is not None:
        await asyncio.to_thread(progress_callback, 1, len(job.images))
    return JobResult(
        job_id=job.id,
        task_type=job.config.task_type,
        images=[
            ImageResult(
                image_id=image.image_id,
                file_name=image.original_name,
                width=100,
                height=50,
                annotations=[
                    Annotation(
                        id=1,
                        label=job.config.prompt_groups[0].label,
                        class_id=0,
                        score=0.9,
                        bbox=[10, 5, 20, 10],
                        segmentation=[[10, 5, 30, 5, 30, 15, 10, 15]],
                        source_prompts=[job.config.prompt_groups[0].prompts[0]],
                    )
                ],
            )
        ],
    )


def wait_for_success(client, job_id):
    for _ in range(100):
        response = client.get(f"/v1/jobs/{job_id}")
        if response.json()["status"] in {"succeeded", "failed"}:
            return response
        time.sleep(0.01)
    raise AssertionError("job did not complete")


def test_create_upload_commit_and_download_coco(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "prewarm", fake_prewarm)
    monkeypatch.setattr(Sam3Runtime, "predict", fake_predict)
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    with TestClient(app) as client:
        created = client.post(
            "/v1/jobs",
            json={
                "task_type": "seg",
                "output_format": "coco",
                "prompt_groups": [{"label": "car", "prompts": ["car", "automobile"]}],
            },
        )
        assert created.status_code == 201
        job_id = created.json()["id"]

        uploaded = client.post(
            f"/v1/jobs/{job_id}/images",
            files=[("files", ("sample.jpg", b"fake-jpeg", "image/jpeg"))],
        )
        assert uploaded.status_code == 200
        assert uploaded.json()["image_count"] == 1

        progress = client.get(f"/v1/jobs/{job_id}/progress")
        assert progress.status_code == 200
        assert progress.json()["status"] == "uploading"
        assert progress.json()["processed_images"] == 0
        assert progress.json()["total_images"] == 1
        assert progress.json()["progress_percent"] == 0.0

        committed = client.post(f"/v1/jobs/{job_id}/commit")
        assert committed.status_code == 200
        assert committed.json()["status"] == "queued"
        finished = wait_for_success(client, job_id)
        assert finished.json()["status"] == "succeeded"

        progress = client.get(f"/v1/jobs/{job_id}/progress")
        assert progress.status_code == 200
        assert progress.json()["status"] == "succeeded"
        assert progress.json()["processed_images"] == 1
        assert progress.json()["total_images"] == 1
        assert progress.json()["progress_percent"] == 100.0

        result = client.get(f"/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        payload = json.loads(result.content)
        assert payload["categories"][0]["name"] == "car"
        assert payload["annotations"][0]["score"] == 0.9

        result_path = tmp_path / "results" / job_id / "annotations.json"
        assert result_path.is_file()
        deleted = client.delete(f"/v1/jobs/{job_id}/result")
        assert deleted.status_code == 204
        assert not result_path.parent.exists()
        assert client.get(f"/v1/jobs/{job_id}/result").status_code == 404
        assert client.get(f"/v1/jobs/{job_id}").json()["result_url"] is None
        assert client.delete(f"/v1/jobs/{job_id}/result").status_code == 404


def test_one_shot_yolo(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "prewarm", fake_prewarm)
    monkeypatch.setattr(Sam3Runtime, "predict", fake_predict)
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    task = {
        "task_type": "detect",
        "output_format": "yolo",
        "prompt_groups": [{"label": "car", "prompts": ["car"]}],
    }
    with TestClient(app) as client:
        response = client.post(
            "/v1/predict",
            data={"task": json.dumps(task)},
            files=[("files", ("sample.jpg", b"fake-jpeg", "image/jpeg"))],
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert wait_for_success(client, job_id).json()["status"] == "succeeded"
        result = client.get(f"/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        assert result.headers["content-type"] == "application/zip"
        assert result.content.startswith(b"PK")


def test_job_progress_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    with TestClient(app) as client:
        response = client.get("/v1/jobs/missing/progress")

    assert response.status_code == 404


def test_list_jobs_supports_status_filter_and_pagination(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)
    payload = {
        "output_format": "yolo",
        "prompt_groups": [{"label": "car", "prompts": ["car"]}],
    }

    with TestClient(app) as client:
        first = client.post("/v1/jobs", json={**payload, "client_reference": "first"})
        time.sleep(0.001)
        second = client.post("/v1/jobs", json={**payload, "client_reference": "second"})
        assert first.status_code == 201
        assert second.status_code == 201

        response = client.get("/v1/jobs", params={"status": "uploading", "limit": 1})
        assert response.status_code == 200
        assert response.json()["total"] == 2
        assert response.json()["limit"] == 1
        assert response.json()["offset"] == 0
        assert [item["client_reference"] for item in response.json()["items"]] == ["second"]

        response = client.get(
            "/v1/jobs", params={"status": "uploading", "limit": 1, "offset": 1}
        )
        assert [item["client_reference"] for item in response.json()["items"]] == ["first"]

        assert client.get("/v1/jobs", params={"status": "unknown"}).status_code == 422
