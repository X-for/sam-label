const assert = require("node:assert/strict");
const test = require("node:test");

const {
  filesForMissingPaths,
  jobsNeedingProgress,
  terminalProgress,
  uploadManifest,
  uniqueImageFiles,
} = require("../code/sam_api/static/ui.js");

test("progress requests exclude terminal jobs and run newest first", () => {
  const jobs = [
    { id: "old-running", status: "running", created_at: "2026-09-01T08:00:00Z" },
    { id: "new-succeeded", status: "succeeded", created_at: "2026-09-03T08:00:00Z" },
    { id: "new-queued", status: "queued", created_at: "2026-09-03T07:00:00Z" },
    { id: "failed", status: "failed", created_at: "2026-09-02T08:00:00Z" },
    { id: "uploading", status: "uploading", created_at: "2026-09-02T09:00:00Z" },
  ];

  assert.deepEqual(
    jobsNeedingProgress(jobs).map((job) => job.id),
    ["new-queued", "uploading", "old-running"],
  );
});

test("succeeded jobs render complete without a progress request", () => {
  assert.deepEqual(
    terminalProgress({ id: "done", status: "succeeded", image_count: 12 }),
    { processed_images: 12, total_images: 12, progress_percent: 100 },
  );
  assert.equal(terminalProgress({ id: "active", status: "running", image_count: 12 }), null);
});

test("file and recursive directory selections keep supported images", () => {
  const fileImage = { name: "single.JPG", size: 10, lastModified: 1, webkitRelativePath: "" };
  const directoryImage = {
    name: "nested.png",
    size: 20,
    lastModified: 2,
    webkitRelativePath: "dataset/sub/nested.png",
  };
  const duplicate = { ...directoryImage };
  const ignored = {
    name: "notes.txt",
    size: 30,
    lastModified: 3,
    webkitRelativePath: "dataset/notes.txt",
  };

  assert.deepEqual(
    uniqueImageFiles([fileImage], [directoryImage, duplicate, ignored]),
    [fileImage, duplicate],
  );
});

test("upload manifest preserves directory paths without reading file contents", () => {
  const files = [
    { name: "one.jpg", webkitRelativePath: "dataset/a/one.jpg", size: 12 },
    { name: "two.png", webkitRelativePath: "", size: 34 },
  ];

  assert.deepEqual(uploadManifest(files), {
    files: [
      { relative_path: "dataset/a/one.jpg", size_bytes: 12 },
      { relative_path: "two.png", size_bytes: 34 },
    ],
  });
});

test("missing upload paths select only files requested by the server", () => {
  const files = [
    { name: "one.jpg", webkitRelativePath: "dataset/a/one.jpg", size: 12 },
    { name: "two.png", webkitRelativePath: "dataset/two.png", size: 34 },
    { name: "three.jpg", webkitRelativePath: "dataset/three.jpg", size: 56 },
  ];

  assert.deepEqual(
    filesForMissingPaths(files, ["dataset/two.png", "dataset/a/one.jpg"]),
    [files[1], files[0]],
  );
});
