# SAM3 远程预标注服务

这是一个基于 **FastAPI + SAM3 + Ultralytics** 的远程预标注服务。它将显存占用较大的 SAM3 模型部署在 GPU 服务器上，本地只负责上传图片、提交任务和下载标注结果。

项目支持目标检测、实例分割和语义分割式任务，可将多个文本 prompt 映射到同一个标签，并导出 YOLO 或 COCO 格式。模型既可以常驻显存，也可以按需加载或在每个任务完成后卸载。

## 主要功能

- 批量上传单张图片、目录或多个数据集目录
- 支持 `detect`、`segment`、`semantic` 三类任务
- 支持多个 prompt 对应同一个类别，例如 `爆字圆牌`、`圆形爆字标志` → `explosive_sign`
- 支持置信度、NMS IoU、推理尺寸、最大检测数等 SAM3 参数
- 支持 `deduplicate`、`keep_all`、`best`、`union` 四种 prompt 聚合策略
- 导出 YOLO 检测、YOLO 分割或 COCO 标注
- 支持异步任务、状态查询和结果压缩包下载
- 支持按需加载、模型常驻和按任务卸载三种显存管理方式
- 附带几何过滤、OCR 复核、结果预览和 YOLO 数据集整理工具

## 工作流程

```text
本地图片目录
    │
    ├─ 创建任务并上传图片
    ▼
FastAPI 任务队列
    │
    ├─ 加载/复用 SAM3
    ├─ 按 prompt 推理
    ├─ 同类结果聚合
    └─ 转换标签格式
    ▼
YOLO ZIP / COCO JSON
```

每个服务进程只有一个任务执行队列，同一块 GPU 上的任务会顺序执行，避免多个 SAM3 实例同时争抢显存。

## 项目结构

```text
sam-label/
├─ code/
│  ├─ main.py                    # Uvicorn 入口
│  └─ sam_api/                   # FastAPI、任务队列、SAM3 和导出逻辑
├─ client/
│  ├─ batch_upload.py            # 批量任务客户端
│  ├─ batch-config.example.toml  # 批量任务配置示例
│  ├─ tasks/                     # 常用任务配置
│  └─ *.py                       # OCR、过滤、预览和数据集整理工具
├─ tests/                        # 自动化测试
├─ models/sam3.pt                # 默认模型位置
├─ .env.example                  # 服务端环境变量示例
├─ .rsyncinclude                 # rsync 白名单
├─ pyproject.toml                # uv 项目配置和依赖
└─ Dockerfile                    # 可选容器部署
```

`data/` 和 `results/` 是运行时目录，默认不会通过 rsync 同步。

## 环境要求

推荐环境：

- Python 3.11 或 3.12，项目默认使用 Python 3.12
- [uv](https://docs.astral.sh/uv/)
- NVIDIA GPU 和可用的 NVIDIA 驱动
- 默认依赖为 PyTorch 2.6.0 + CUDA 12.4
- SAM3 权重文件 `models/sam3.pt`

生产环境建议只启动一个 Uvicorn worker。多个 worker 会分别加载模型，显存占用也会成倍增加。

## 快速开始

### 1. 安装服务端依赖

在项目根目录执行：

```bash
uv sync --no-dev
```

需要运行测试时安装开发依赖：

```bash
uv sync
```

检查 PyTorch 和 CUDA：

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

最后一个值应当为 `True`。

### 2. 配置服务

复制环境变量示例：

```bash
cp .env.example .env
```

常用配置如下：

```dotenv
SAM3_MODEL_PATH=models/sam3.pt
SAM3_DATA_DIR=data
SAM3_DEVICE=cuda:0
SAM3_QUANTIZE=16
SAM3_LIFECYCLE=on_demand
SAM3_IDLE_UNLOAD_SECONDS=600
SAM3_MAX_FILE_BYTES=52428800
SAM3_MAX_JOB_BYTES=21474836480
SAM3_API_KEY=
```

| 配置 | 说明 |
|---|---|
| `SAM3_MODEL_PATH` | SAM3 权重路径 |
| `SAM3_DATA_DIR` | 上传文件、任务状态和结果目录 |
| `SAM3_DEVICE` | 推理设备，例如 `cuda:0`、`cuda:2` 或 `cpu` |
| `SAM3_QUANTIZE` | 模型精度，仅支持 `16` 或 `32` |
| `SAM3_LIFECYCLE` | `on_demand`、`resident` 或 `per_job` |
| `SAM3_IDLE_UNLOAD_SECONDS` | 按需模式下空闲多久后卸载模型，`0` 表示不自动卸载 |
| `SAM3_MAX_FILE_BYTES` | 单个上传文件最大字节数 |
| `SAM3_MAX_JOB_BYTES` | 单个任务累计最大字节数 |
| `SAM3_UPLOAD_CHUNK_BYTES` | 上传流的读取块大小，默认 1 MiB |
| `SAM3_API_KEY` | 可选 API 密钥；设置后客户端需发送 `X-API-Key` |

模型生命周期：

- `on_demand`：服务启动、创建任务和上传图片时均不加载模型；任务进入执行队列后才加载。队列完全清空后开始计算空闲时间，达到 `SAM3_IDLE_UNLOAD_SECONDS` 后卸载。
- `resident`：服务启动时预热模型，并持续驻留显存。
- `per_job`：任务进入执行队列后加载，并在每个任务推理结束后立即卸载。

### 3. 启动服务

```bash
uv run --no-dev uvicorn --app-dir code main:app \
  --host 0.0.0.0 --port 8000 --workers 1
```

也可以在启动命令中临时选择 GPU 和数据目录：

```bash
SAM3_DEVICE=cuda:2 SAM3_DATA_DIR=data-pilot \
uv run --no-dev uvicorn --app-dir code main:app \
  --host 0.0.0.0 --port 8002 --workers 1
```

### 4. 检查服务

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/model
```

正常响应示例：

```json
{"status":"ok"}
```

模型状态中的 `loaded: false` 不一定代表故障。在默认 `on_demand` 模式下，没有任务时模型可以处于未加载状态。

浏览器访问以下地址可以查看和调试全部接口：

- Swagger UI：`http://服务器地址:端口/docs`
- OpenAPI JSON：`http://服务器地址:端口/openapi.json`

## 推荐用法：批量任务客户端

批量任务建议使用 TOML 配置文件，避免在命令行中反复填写大量标签映射和推理参数。

复制示例配置：

```bash
cp client/batch-config.example.toml client/my-task.toml
```

一个完整配置示例：

```toml
[client]
server = "http://192.168.110.101:8002"
inputs = ["../datasets/images"]
output_dir = "../results"
recursive = true
batch_size = 16
poll_interval = 3.0
wait_timeout = 7200.0
request_timeout = 600.0
# 如果服务端配置了 API key，再取消下一行注释：
# api_key_env = "SAM3_API_KEY"

[task]
client_reference = "explosive-circle-sign"
task_type = "detect"
output_format = "yolo"
aggregation = "deduplicate"
merge_iou = 0.70

[task.prediction]
conf = 0.30
iou = 0.70
imgsz = 644
max_det = 300
retina_masks = true

[task.labels]
explosive_sign = [
  "a circular vehicle sign with the Chinese character 爆",
  "圆形爆字标识",
  "车身上的爆字圆牌",
]
```

说明：

- 相对路径以 TOML 配置文件所在目录为基准。
- `[task.labels]` 中配置键是最终标签名，数组内容是发给 SAM3 的 prompts。
- 类别 ID 按 `[task.labels]` 的书写顺序从 `0` 开始生成。
- 推理尺寸应为 14 的倍数，推荐使用 `644`，不要使用 `640`。

提交任务并等待结果下载：

```bash
uv run --script client/batch_upload.py --config client/my-task.toml
```

常用选项：

```bash
# 只检查配置和图片数量，不上传
uv run --script client/batch_upload.py --config client/my-task.toml --dry-run

# 临时覆盖配置中的输入目录
uv run --script client/batch_upload.py --config client/my-task.toml D:/datasets/new-images

# 提交后立即返回，不等待任务完成
uv run --script client/batch_upload.py --config client/my-task.toml --no-wait
```

配置中的所有输入会合并到同一个任务中。任务完成后，结果会下载到配置中的 `output_dir`；如果希望不同目录分别生成结果，请为它们分别运行一次客户端。

## 任务参数

### 任务类型

| 值 | 用途 | 可用输出 |
|---|---|---|
| `detect` | 目标框预标注 | YOLO detect、COCO bbox |
| `segment` | 实例分割预标注 | YOLO segment、COCO polygon |
| `semantic` | 同类区域合并式分割 | YOLO segment、COCO polygon |

兼容别名：`det`、`seg`、`instance_seg`。

### Prompt 与标签映射

多个 prompt 可以映射到同一个标签，用不同语言和外观描述提升召回率：

```json
{
  "id": 0,
  "name": "explosive_sign",
  "prompts": [
    "圆形爆字标志",
    "a round sign with the Chinese character 爆",
    "车身上的爆字圆牌"
  ]
}
```

SAM3 会分别对每个 prompt 推理，然后按照 `aggregation` 合并同一标签下的结果。

| 聚合策略 | 行为 | 适用场景 |
|---|---|---|
| `deduplicate` | 对高重叠结果做 NMS，保留更高分结果 | 默认推荐，兼顾召回和去重 |
| `keep_all` | 保留所有 prompt 的结果 | 需要最大召回，后续人工清洗 |
| `best` | 每个标签只保留全图最高分的一个实例 | 每张图确定只有一个目标 |
| `union` | 将同标签 mask 合并为一个区域 | 分割/语义任务，不支持 `detect` |

`merge_iou` 控制聚合去重时的重叠阈值，与 SAM3 推理参数中的 `prediction.iou` 不是同一个参数。

### 推理参数

| 参数 | 范围/默认值 | 说明 |
|---|---|---|
| `conf` | `0~1`，默认 `0.25` | 置信度阈值 |
| `iou` | `0~1`，默认 `0.70` | SAM3/NMS IoU 阈值 |
| `imgsz` | 默认 `644` | 推理尺寸，必须是 14 的倍数 |
| `max_det` | 默认 `300` | 单张图片最大实例数 |
| `retina_masks` | 默认 `true` | 是否输出高分辨率 mask |

## 直接调用 HTTP API

### 1. 创建任务

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H "Content-Type: application/json" \
  -d @client/example-task.json
```

返回结果中包含任务 `id`。

### 2. 上传图片

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs/JOB_ID/images \
  -F "files=@example-a.jpg" \
  -F "files=@example-b.jpg"
```

### 3. 提交任务

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs/JOB_ID/commit
```

### 4. 查询状态和下载结果

```bash
curl http://127.0.0.1:8000/v1/jobs/JOB_ID
curl -OJ http://127.0.0.1:8000/v1/jobs/JOB_ID/result
```

任务状态变化：

```text
uploading → queued → running → succeeded
                              └→ failed
```

也可以使用 `POST /v1/predict` 一次性提交少量图片和任务 JSON。大数据集建议使用分步任务接口，上传失败时更容易定位和重试。

如果配置了 API key，需要为 curl 增加：

```bash
-H "X-API-Key: YOUR_API_KEY"
```

## 输出格式

### YOLO

YOLO 结果以 ZIP 返回，典型结构为：

```text
result.zip
├─ labels/
│  ├─ image001.txt
│  └─ image002.txt
├─ data.yaml
└─ manifest.json
```

- `detect`：每行是 `class_id x_center y_center width height`
- `segment`/`semantic`：每行是 `class_id x1 y1 x2 y2 ...`
- 坐标均归一化到 `0~1`
- 没有检测结果的图片也会生成空标签文件

### COCO

COCO 结果返回一个 JSON 文件，包含：

- `images`
- `annotations`
- `categories`
- `info`，保存任务类型、模型、设备和坐标格式等信息

检测任务主要使用 `bbox`，分割任务还会包含 polygon `segmentation`。

## 模型生命周期与显存

| 模式 | 行为 | 建议 |
|---|---|---|
| `on_demand` | 首个任务到来时加载，空闲一段时间后卸载 | 默认推荐，平衡响应时间和显存 |
| `resident` | 服务启动时加载并保持常驻 | 任务连续、追求低延迟 |
| `per_job` | 每个任务前加载，完成后立即卸载 | 显存必须及时让给其他程序 |

`on_demand` 模式下，创建任务后服务会在上传数据的同时异步预热模型，因此上传大数据集时可以减少等待时间。模型加载失败会反映在 `/v1/model` 的 `last_error` 中。

多 GPU 服务器可以启动多个独立服务实例：

```bash
SAM3_DEVICE=cuda:0 SAM3_DATA_DIR=data-gpu0 \
uv run --no-dev uvicorn --app-dir code main:app --host 0.0.0.0 --port 8000 --workers 1

SAM3_DEVICE=cuda:1 SAM3_DATA_DIR=data-gpu1 \
uv run --no-dev uvicorn --app-dir code main:app --host 0.0.0.0 --port 8001 --workers 1
```

不同实例必须使用不同端口和数据目录。

## 查看任务是否正在运行

查询任务状态：

```bash
curl http://服务器地址:端口/v1/jobs/JOB_ID
```

重点观察：

- `status`：是否为 `queued`、`running`、`succeeded` 或 `failed`
- `image_count`：服务端已接收的图片数量
- `uploaded_bytes`：已上传字节数
- `error`：失败原因
- `result_url`：结果下载地址

当前接口提供的是任务级状态，不是逐图片百分比。任务处于 `running` 时，还可以在服务器上查看：

```bash
nvidia-smi
ps -fp SERVER_PID
```

同时观察 Uvicorn 控制台日志。首次任务可能需要先完成模型加载，短时间没有新的 HTTP 日志属于正常现象。

## 可选后处理工具

文本 prompt 模型可能会把外形相似的目标混在一起。例如检测“中间有汉字爆的圆形标识”时，SAM3 可能同时找出菱形危险品告示牌。项目提供了一套精度优先的后处理流程。

### 几何和多 prompt 一致性过滤

```bash
python client/coco_to_filtered_yolo.py \
  results/result.json datasets/images
```

该工具可以按置信度、mask 填充率、相对面积、长宽比和 prompt 投票数过滤 COCO 结果，并生成 YOLO 检测标签。

### OCR 复核“爆”字

```bash
uv run --script client/ocr_verify_yolo.py \
  datasets results/ocr-report.json
```

先查看 JSON 报告并备份现有标签，再执行：

```bash
python client/apply_ocr_report.py \
  results/ocr-report.json datasets
```

`apply_ocr_report.py` 会直接改写标签文件，只保留 OCR 命中的候选框，因此正式执行前请保留原结果。

### 生成标注预览

```bash
uv run --no-project --with pillow python client/render_yolo_preview.py \
  datasets results/preview.jpg
```

### 整理 YOLO 数据集

```bash
python client/build_yolo_dataset.py datasets --val-ratio 0.2
```

工具会生成 `train.txt`、`val.txt`、`data.yaml` 和预标注统计文件，方便后续训练或人工复核。

## 同步到服务器

项目提供 `.rsyncinclude` 白名单，只同步运行所需代码、配置和模型，排除本地数据、结果、缓存和虚拟环境。

在 WSL、Linux 或 Git Bash 中执行：

```bash
rsync -avm --progress \
  --filter="merge .rsyncinclude" \
  ./ user@server:/home/user/prj/sam-label/
```

预览将要同步的内容：

```bash
rsync -avmn --itemize-changes \
  --filter="merge .rsyncinclude" \
  ./ user@server:/home/user/prj/sam-label/
```

如果模型文件已经单独放在服务器上，可以在同步时增加：

```bash
--exclude=/models/sam3.pt
```

## Docker（可选）

```bash
docker build -t sam3-api .

docker run --rm --gpus all \
  -p 8000:8000 \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/data:/app/data" \
  -e SAM3_DEVICE=cuda:0 \
  -e SAM3_LIFECYCLE=on_demand \
  sam3-api
```

服务器直接部署时仍推荐使用 uv，依赖版本更容易与 `pyproject.toml` 保持一致。使用 Docker 前需确认宿主机 NVIDIA Container Toolkit、驱动和镜像 CUDA 版本兼容。

## 常见问题

### `imgsz=[640] must be multiple of max stride 14`

将任务配置中的 `imgsz` 改为 `644`。

### `quantize Input should be 16 or 32`

确保使用最新代码，并在 `.env` 中写：

```dotenv
SAM3_QUANTIZE=16
```

不要填写 `fp16`、`"16"` 之外的自定义值。

### NumPy 2.x 兼容警告

项目已经约束 `numpy<2`。执行：

```bash
uv sync --no-dev
```

然后重新启动服务。

### Ultralytics 自动安装 `timm` 或 CLIP 后要求重启

`timm` 和 Ultralytics CLIP 已声明为项目依赖。先执行 `uv sync --no-dev`，再重启 Uvicorn，避免在任务运行期间自动更新环境。

### uv 尝试安装不兼容的 `torch==2.13.0`

当前项目已固定为 PyTorch 2.6.0 + CUDA 12.4。确认服务器拿到最新的 `pyproject.toml`，然后重新执行：

```bash
uv sync --no-dev
```

### 端口被占用

换一个端口，例如 `8002`，并同步修改客户端配置中的 `server`。

### 显存不足

- 保持 `--workers 1`
- 使用 `SAM3_QUANTIZE=16`
- 降低 `imgsz`，但仍需保持为 14 的倍数
- 减小 `max_det`
- 使用 `on_demand` 或 `per_job`
- 将服务切换到空闲 GPU，例如 `SAM3_DEVICE=cuda:2`

### `healthz` 正常但感觉任务没有运行

`/healthz` 只表示 API 服务存活。请保存创建任务时返回的 `job_id`，查询 `/v1/jobs/{job_id}`，并检查 `/v1/model`、服务器日志和 `nvidia-smi`。

## 测试

```bash
uv run pytest
```

## 使用建议

- 预标注结果应经过抽样或人工复核，不建议直接作为最终训练标签。
- 对形状相似但语义不同的目标，优先使用“多 prompt 召回 + 几何过滤/OCR 复核”。
- 正式批量任务前，先选一个小目录验证 prompt、阈值、聚合策略和输出格式。
- 对公网提供服务时务必配置 `SAM3_API_KEY`，并在前面增加 HTTPS、访问控制和上传大小限制。

## 许可证

本项目原创代码采用 [Apache License 2.0](LICENSE) 开源。

该许可证仅适用于本仓库中的原创代码，不涵盖 SAM3 模型权重、数据集、Ultralytics 及其他第三方组件；这些内容仍遵循各自的许可证和使用条款。
