(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SamUi = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  const activeStatuses = new Set(["uploading", "queued", "running"]);
  const imagePattern = /\.(jpe?g|png|webp|bmp|tiff?)$/i;

  function newestFirst(jobs) {
    return [...jobs].sort((left, right) => {
      const timeDifference = Date.parse(right.created_at) - Date.parse(left.created_at);
      return timeDifference || String(right.id).localeCompare(String(left.id));
    });
  }

  function jobsNeedingProgress(jobs) {
    return newestFirst(jobs).filter((job) => activeStatuses.has(job.status));
  }

  function terminalProgress(job) {
    if (job.status !== "succeeded") return null;
    return {
      processed_images: job.image_count,
      total_images: job.image_count,
      progress_percent: 100,
    };
  }

  function uniqueImageFiles(files, directoryFiles) {
    const unique = new Map();
    for (const file of [...files, ...directoryFiles]) {
      if (!imagePattern.test(file.name)) continue;
      const path = file.webkitRelativePath || file.name;
      unique.set(`${path}:${file.size}:${file.lastModified}`, file);
    }
    return [...unique.values()];
  }

  return { jobsNeedingProgress, newestFirst, terminalProgress, uniqueImageFiles };
});
