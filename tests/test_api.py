import json
import time

from fastapi.testclient import TestClient

from sam_api.app import app
from sam_api.model_runtime import Sam3Runtime
from sam_api.schemas import Annotation, ImageResult, JobResult


async def fake_prewarm(self):
    return None


async def fake_close(self):
    return None


async def fake_predict(self, job):
    image = job.images[0]
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

        committed = client.post(f"/v1/jobs/{job_id}/commit")
        assert committed.status_code == 200
        assert committed.json()["status"] == "queued"
        finished = wait_for_success(client, job_id)
        assert finished.json()["status"] == "succeeded"

        result = client.get(f"/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        payload = json.loads(result.content)
        assert payload["categories"][0]["name"] == "car"
        assert payload["annotations"][0]["score"] == 0.9


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
