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


def test_web_ui_is_served(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    with TestClient(app) as client:
        response = client.get("/ui")
        script = client.get("/ui.js")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "SAM3 预标注控制台" in response.text
    assert "/v1/jobs" in response.text
    assert "/progress" in response.text

    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert "jobsNeedingProgress" in script.text


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


def test_upload_accepts_multiple_images_without_loading_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "prewarm", fake_prewarm)
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    with TestClient(app) as client:
        created = client.post(
            "/v1/jobs",
            json={"prompt_groups": [{"label": "car", "prompts": ["car"]}]},
        )
        job_id = created.json()["id"]
        uploaded = client.post(
            f"/v1/jobs/{job_id}/images",
            files=[
                ("files", ("first.jpg", b"first-image", "image/jpeg")),
                ("files", ("second.png", b"second-image", "image/png")),
            ],
        )

    assert uploaded.status_code == 200
    assert uploaded.json()["image_count"] == 2
    assert len(list((tmp_path / "uploads" / job_id).iterdir())) == 2


def test_manifest_blocks_commit_until_missing_images_are_uploaded(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "prewarm", fake_prewarm)
    monkeypatch.setattr(Sam3Runtime, "predict", fake_predict)
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    with TestClient(app) as client:
        created = client.post(
            "/v1/jobs",
            json={"prompt_groups": [{"label": "car", "prompts": ["car"]}]},
        )
        job_id = created.json()["id"]
        manifest = client.put(
            f"/v1/jobs/{job_id}/manifest",
            json={
                "files": [
                    {"relative_path": "dataset/first.jpg", "size_bytes": 5},
                    {"relative_path": "dataset/second.jpg", "size_bytes": 6},
                ]
            },
        )
        assert manifest.status_code == 200
        assert manifest.json()["missing_files"] == [
            "dataset/first.jpg",
            "dataset/second.jpg",
        ]

        first = client.post(
            f"/v1/jobs/{job_id}/images",
            data={"paths": "dataset/first.jpg"},
            files=[("files", ("first.jpg", b"first", "image/jpeg"))],
        )
        assert first.status_code == 200
        assert first.json()["image_count"] == 1

        rejected = client.post(f"/v1/jobs/{job_id}/commit")
        assert rejected.status_code == 409
        assert rejected.json()["detail"] == {
            "message": "upload is incomplete",
            "missing_files": ["dataset/second.jpg"],
        }

        second = client.post(
            f"/v1/jobs/{job_id}/images",
            data={"paths": "dataset/second.jpg"},
            files=[("files", ("second.jpg", b"second", "image/jpeg"))],
        )
        assert second.status_code == 200
        committed = client.post(f"/v1/jobs/{job_id}/commit")
        assert committed.status_code == 200
        assert committed.json()["status"] == "queued"


def test_manifest_commit_reports_every_missing_file_before_any_upload(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    with TestClient(app) as client:
        created = client.post(
            "/v1/jobs",
            json={"prompt_groups": [{"label": "car", "prompts": ["car"]}]},
        )
        job_id = created.json()["id"]
        client.put(
            f"/v1/jobs/{job_id}/manifest",
            json={"files": [{"relative_path": "missing.jpg", "size_bytes": 4}]},
        )

        response = client.post(f"/v1/jobs/{job_id}/commit")

        assert response.status_code == 409
        assert response.json()["detail"] == {
            "message": "upload is incomplete",
            "missing_files": ["missing.jpg"],
        }


def test_manifest_rejects_wrong_size_without_registering_image(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    with TestClient(app) as client:
        created = client.post(
            "/v1/jobs",
            json={"prompt_groups": [{"label": "car", "prompts": ["car"]}]},
        )
        job_id = created.json()["id"]
        client.put(
            f"/v1/jobs/{job_id}/manifest",
            json={"files": [{"relative_path": "bad.jpg", "size_bytes": 10}]},
        )

        response = client.post(
            f"/v1/jobs/{job_id}/images",
            data={"paths": "bad.jpg"},
            files=[("files", ("bad.jpg", b"short", "image/jpeg"))],
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "uploaded size does not match manifest: bad.jpg"
        assert client.get(f"/v1/jobs/{job_id}").json()["image_count"] == 0
        assert list((tmp_path / "uploads" / job_id).glob("*")) == []


def test_manifest_upload_is_idempotent_for_retried_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    with TestClient(app) as client:
        created = client.post(
            "/v1/jobs",
            json={"prompt_groups": [{"label": "car", "prompts": ["car"]}]},
        )
        job_id = created.json()["id"]
        client.put(
            f"/v1/jobs/{job_id}/manifest",
            json={"files": [{"relative_path": "same.jpg", "size_bytes": 4}]},
        )
        for _ in range(2):
            response = client.post(
                f"/v1/jobs/{job_id}/images",
                data={"paths": "same.jpg"},
                files=[("files", ("same.jpg", b"same", "image/jpeg"))],
            )
            assert response.status_code == 200

        assert response.json()["image_count"] == 1
        status = client.get(f"/v1/jobs/{job_id}/manifest")
        assert status.status_code == 200
        assert status.json()["missing_files"] == []
        assert len(list((tmp_path / "uploads" / job_id).glob("*"))) == 1


def test_manifest_commit_detects_uploaded_file_removed_from_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("SAM3_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Sam3Runtime, "prewarm", fake_prewarm)
    monkeypatch.setattr(Sam3Runtime, "predict", fake_predict)
    monkeypatch.setattr(Sam3Runtime, "close", fake_close)

    with TestClient(app) as client:
        created = client.post(
            "/v1/jobs",
            json={"prompt_groups": [{"label": "car", "prompts": ["car"]}]},
        )
        job_id = created.json()["id"]
        client.put(
            f"/v1/jobs/{job_id}/manifest",
            json={"files": [{"relative_path": "lost.jpg", "size_bytes": 4}]},
        )
        uploaded = client.post(
            f"/v1/jobs/{job_id}/images",
            data={"paths": "lost.jpg"},
            files=[("files", ("lost.jpg", b"data", "image/jpeg"))],
        )
        assert uploaded.status_code == 200
        next((tmp_path / "uploads" / job_id).iterdir()).unlink()

        response = client.post(f"/v1/jobs/{job_id}/commit")

        assert response.status_code == 409
        assert response.json()["detail"]["missing_files"] == ["lost.jpg"]


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
