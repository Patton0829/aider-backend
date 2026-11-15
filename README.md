# Aider Agent Backend

中心云控制台的 FastAPI 智能体后端，包含故障问答、代码智能体、作业日志和
算法代码语言转换能力：

- `/ws/agent`：千问聊天与 aider 代码修改。
- `/ws/jobs/{job_id}/logs`：Slurm 算法日志增量推送。
- `/ws/knowledge`：检索列车故障诊断知识库并调用 GLM 或千问 API 流式回答。
- `/api/code-conversion/jobs/{id}`：识别算法入口语言并把非 Python 项目转换为 Python。

## 目录

```text
app/
  code_conversion/  安全解压、语言识别、多文件转写和异步任务实现
  api/routes/       HTTP 和 WebSocket 路由
  core/             配置和鉴权
  schemas/          消息协议
  services/         千问、aider、日志流实现
tests/
```

## 启动

controller 服务器使用下文“服务器 Git 部署”的步骤。项目路径配置支持相对路径，
统一相对于本仓库根目录解析，代码不依赖 Mac 上的 `/Users/...` 路径。

故障诊断页面依赖本服务（默认端口 `8000`）；仅启动前端 Vite（端口 `8088`）
不会自动启动本服务。页面显示“服务已断开，正在重连”时，先检查本服务是否运行。
默认 CORS 配置允许本地 `5173` 和 `8088` 端口；若在 `.env` 中覆盖了
`CORS_ORIGINS`，也需要包含实际使用的前端地址。

已有 `aider` Conda 环境时：

```bash
cd aider-backend
conda run -n aider python -m pip install -e '.[milvus]'
cp .env.example .env
# 在 .env 中配置 DASHSCOPE_API_KEY
conda run -n aider python -m uvicorn app.main:app \
  --host 127.0.0.1 --port 8000
```

也可以创建独立虚拟环境：

```bash
cd aider-backend
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,milvus]'
cp .env.example .env
# 在 .env 中配置 DASHSCOPE_API_KEY
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/models
curl http://127.0.0.1:8000/api/knowledge/info
```

## 服务器 Git 部署（controller）

目标环境：Debian 13、x86_64、GLIBC 2.41、Conda Python 3.13。
`requirements-server.lock` 锁定已验证的运行依赖；安装脚本使用项目 `.venv`，
不修改服务器的 Conda base / aider 环境。Milvus Lite 和 PyMilvus 固定为
`milvus-lite==3.2.2`、`pymilvus==3.0.2`。

初次部署时克隆仓库；已有仓库则跳过 `git clone`：

```bash
cd /root/system
git clone https://github.com/Patton0829/aider-backend.git
cd /root/system/aider-backend
git pull --ff-only
PYTHON_BIN=/opt/miniconda3/bin/python3 bash scripts/setup.sh
nano .env
```

在服务器自己的 `.env` 填入 `DASHSCOPE_API_KEY`。模板已配置千问生成、混合召回、
千问重排与 Milvus Lite；若服务器原本有 `.env`，安装脚本会保留它，需要确认：

```dotenv
KNOWLEDGE_PROVIDER=qwen
KNOWLEDGE_QWEN_MODEL=qwen3.8-flash
KNOWLEDGE_RETRIEVAL_MODE=hybrid
KNOWLEDGE_VECTOR_STORE=milvus_lite
KNOWLEDGE_MILVUS_URI=storage/rag/milvus.db
KNOWLEDGE_INDEX_CATALOG_PATH=storage/rag/index_catalog.sqlite3
KNOWLEDGE_EMBEDDING_MODEL=qwen3.7-text-embedding
KNOWLEDGE_EMBEDDING_DIMENSIONS=1024
KNOWLEDGE_RERANK_ENABLED=true
KNOWLEDGE_RERANK_MODEL=qwen3.7-text-rerank
```

检查配置并启动：

```bash
cd /root/system/aider-backend
.venv/bin/python -m app.manage check
bash scripts/start.sh
```

服务启动后，在另一终端运行 `.venv/bin/python -m app.manage smoke`，
验证真实问答、分段输出、证据版本，以及检索、首字、完整回答的耗时。

首次启动会从仓库中的 `列车故障诊断经验知识库.md` 初始化 `knowledge-bases/`，
调用千问生成文档向量，在服务器创建 Milvus 索引；已有兼容向量缓存时直接复用。
确认 `/api/knowledge/info` 中 `vectorStore=milvus_lite`、`retrievalReady=true`、
`indexedChunkCount=290`（知识库未修改时）且 `indexVersion` 非空。
`knowledge-bases/` 已存在时不会覆盖已管理的文档；之后可通过知识库管理接口更新。

### systemd 托管

先停止同端口上的旧后端或前台测试进程，再安装服务：

```bash
cp /root/system/aider-backend/deploy/aider-backend.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now aider-backend
systemctl status aider-backend --no-pager
journalctl -u aider-backend -n 80 --no-pager
```

后续代码更新：

```bash
cd /root/system/aider-backend
git pull --ff-only
PYTHON_BIN=/opt/miniconda3/bin/python3 bash scripts/setup.sh
.venv/bin/python -m app.manage check
systemctl restart aider-backend
```

启动脚本固定使用 **一个 Uvicorn worker**。Milvus Lite 的同一数据目录只能由
一个进程持有；写入与查询通过同一客户端执行，写入串行化。需要多 worker 或
多副本时应改为远端 Milvus 部署，不能让多个进程直接打开同一个 Lite 目录。

前端生产环境通过同域 `/aider/` 访问后端。如现有 Nginx 尚未配置此代理，
把 `deploy/nginx-aider.conf` 中的 location 放入现有中心云 `server {}`，
执行 `nginx -t` 检查后再 reload。该配置同时支持 HTTP 和 WebSocket。

### 数据与 Git 的边界

- Git 包含代码、依赖锁、安装/启动脚本、服务配置和原始故障诊断知识库。
- `.env` 密钥、`.venv/`、`knowledge-bases/` 和 `storage/rag/` 均不提交 Git。
- `storage/rag/embeddings.sqlite3`：可复用的内容哈希与向量缓存。
- `storage/rag/index_catalog.sqlite3`：索引构建状态、片段快照、发布指针及历史版本。
- `storage/rag/milvus.db`：Milvus Lite 3.x 的**存储目录**，不是单个数据库文件。

服务器独立建库，不依赖开发机生成的数据库。备份或搬迁时应停止服务，并把
知识库、索引目录和 catalog 作为一组保存；切换模型/维度会生成新的索引版本。

### 索引重建、查看与回滚

以下命令通过运行中的 HTTP API 操作，不会另开一个进程持有 Milvus 数据目录。
若配置了 `AGENT_API_TOKEN`，命令会从服务器 `.env` 读取并通过 Header 鉴权：

```bash
cd /root/system/aider-backend
.venv/bin/python -m app.manage indexes
.venv/bin/python -m app.manage rebuild
.venv/bin/python -m app.manage activate 完整的64位历史版本号
```

对应接口：

```text
GET  /api/knowledge/indexes
POST /api/knowledge/indexes/rebuild
POST /api/knowledge/indexes/{version}/activate
GET  /api/knowledge/indexes/{version}/chunks/{chunk_id}
```

文档新增、删除或修改后先构建独立 collection，验证维度、行数及可查询性后再
原子发布。BM25、Dense 和证据元数据固定使用同一发布快照；在途请求继续使用
原版本。新构建失败会保留旧版本并报告 `indexUpdateError`。
回滚不会改写源文档；在源文档未再次变化时，回滚选择会跨请求和重启保留。
显式执行 `rebuild` 可切回当前源文档对应版本。旧版本暂不自动清理，便于回滚
和按原版本追溯引用。物理存储丢失时，重建会利用缓存建立新 collection 后切换。

片段 ID 使用文档、章节、条目/正文内容及相同内容出现序号生成的稳定哈希，
不依赖行号或全局顺序；前面插入段落不会改变后续未修改片段的 ID。
来源事件包含 `chunkId`、`documentId`、`kbVersion` 和 `contentHash`。

## 算法代码语言转换

中心云后端以 `application/zip` 原始请求体提交算法包，Aider 在后台安全解压并
读取 `algorithm.json` 的 `entry`。入口为合法 `.py` 文件时直接完成；C/C++、
Java、Go、Rust、TypeScript 或 JavaScript 源码会在保留多文件上下文的情况下
调用 GLM 转成 Python，并生成新的算法 ZIP。

```text
POST   /api/code-conversion/jobs/{id}?filename=algorithm.zip
GET    /api/code-conversion/jobs/{id}
GET    /api/code-conversion/jobs/{id}/artifact
DELETE /api/code-conversion/jobs/{id}
```

任务状态依次为 `queued`、`extracting`、`detecting`、`converting`、
`validating`、`packaging`、`completed`；失败时为 `failed`。生产环境配置了
`AGENT_API_TOKEN` 时，中心云后端需要用 Bearer Token 调用这些内部接口。

## Agent WebSocket 协议

连接：

```text
ws://127.0.0.1:8000/ws/agent
```

普通聊天，默认使用快速模型：

```json
{
  "type": "chat",
  "content": "解释当前 transfer 算法的训练流程",
  "model": "flash"
}
```

复杂问题可以使用 `plus`。代码修改由 aider 执行：

```json
{
  "type": "edit",
  "content": "为训练入口增加参数校验",
  "workspace": "transfer",
  "files": ["target_finetune.py"]
}
```

服务端事件包括：

- `connected`
- `status`
- `delta`：千问流式文本
- `output`：aider 输出
- `ready`
- `error`
- `closed`

## 故障诊断知识问答

首次启动时，服务会把仓库根目录的 `列车故障诊断经验知识库.md` 复制到
`KNOWLEDGE_BASE_DIR`，之后从该目录加载所有 `.md`、`.markdown` 和 `.txt`
文档。上传或删除文档后会立即重建检索索引。生成服务通过
`KNOWLEDGE_PROVIDER=glm|qwen` 选择，默认保留 GLM。

### 使用阿里云百炼千问 API

在本地 `.env` 中配置以下字段，然后重启后端：

```dotenv
KNOWLEDGE_PROVIDER=qwen
DASHSCOPE_API_KEY=你的百炼API密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
KNOWLEDGE_QWEN_MODEL=qwen3.8-flash
KNOWLEDGE_QWEN_ENABLE_THINKING=false
KNOWLEDGE_QWEN_MAX_TOKENS=2048
KNOWLEDGE_QWEN_TIMEOUT_SECONDS=60
```

上面的 Base URL 是北京地域的兼容地址；百炼也提供业务空间专属地址
`https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`。
使用与 API Key 同地域的控制台地址，详见
[百炼 OpenAI 兼容接口](https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions)。
密钥只保存在后端环境中，`.env` 已被 Git 忽略；不要把密钥写入前端 `VITE_*` 配置。

千问请求使用 `extra_body={"enable_thinking": false}` 并保持 `stream=true`。
知识库继续在本地检索，问题及选中的相关片段会发送到百炼生成回答；
前端和 `/api/knowledge/info` 显示实际模型名。
已有 `/ws/agent` 的千问代码助手、代码转换任务仍使用各自的模型配置。
`KNOWLEDGE_PROVIDER=qwen` 且未提供密钥时，启动会明确报错，不会自动回落到 GLM。

### 千问向量召回与重排

目前支持三个独立阶段：本地文档解析、候选召回、重排；生成服务继续使用
`KNOWLEDGE_PROVIDER` 选择。向量与重排都是百炼 API，不需要下载模型权重。
例如启用混合召回并增加重排：

```dotenv
KNOWLEDGE_RETRIEVAL_MODE=hybrid
KNOWLEDGE_EMBEDDING_MODEL=qwen3.7-text-embedding
KNOWLEDGE_EMBEDDING_DIMENSIONS=1024
KNOWLEDGE_CANDIDATE_K=20
KNOWLEDGE_BM25_RECALL_K=20
KNOWLEDGE_DENSE_RECALL_K=20
KNOWLEDGE_RRF_K=60
KNOWLEDGE_RERANK_ENABLED=true
KNOWLEDGE_RERANK_MODEL=qwen3.7-text-rerank
KNOWLEDGE_TOP_K=5
```

链路为：解析文档 → 缓存文档向量 → 查询向量 → 向量召回＋关键词召回 →
倒数排名融合（RRF）保留候选 → 千问重排 → 最终片段 → 流式生成回答。
关键词路沿用本地 BM25 评分（带条目编号及短语加分），Dense 路按归一化向量
余弦相似度排序。RRF 使用 `sum(1 / (k + rank))`，按片段 ID 去重，
不直接相加量纲不同的 BM25 与余弦分数。
`keyword` 保留原有关键词检索；`vector` 只做向量召回；`hybrid` 融合两路召回。
`KNOWLEDGE_BM25_RECALL_K` 和 `KNOWLEDGE_DENSE_RECALL_K` 分别控制两路召回量；
`KNOWLEDGE_RRF_K` 控制 RRF 平滑常数；`KNOWLEDGE_CANDIDATE_K` 控制重排候选数，
`KNOWLEDGE_TOP_K` 控制最终片段数。

`storage/rag/embeddings.sqlite3` 保存文档内容哈希及向量，已被 Git 忽略。
重启会复用缓存，上传、删除或修改文档后使用新的片段快照；只有新增或改变的
片段需要重新调用向量 API。模型名、维度、API 地址或向量文本格式变化后，
缓存键自动变化，旧模型的向量不会与新模型混用。查询向量和重排按每次请求调用。
索引失败会显示检索不可用，不会静默回落到关键词检索。

模块边界：

- `services/knowledge.py`：文档管理、切片、来源信息及关键词评分。
- `services/rag_models.py`：`Embedder`、`Reranker` 接口和百炼 API 适配。
- `services/embedding_cache.py`：向量缓存。
- `services/rag_pipeline.py`：召回、RRF 融合、重排编排。
- `services/knowledge_llm.py`：生成模型及流式回答。

HTTP 地址默认从 `DASHSCOPE_BASE_URL` 推导为同地域的 `/api/v1`；
自定义代理可单独配置 `DASHSCOPE_HTTP_BASE_URL`。
接口详情：[向量模型](https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api)、
[重排模型](https://help.aliyun.com/zh/model-studio/text-rerank-api)。

### GLM 等待时间排查

知识问答默认通过 `chat_template_kwargs.enable_thinking=false` 关闭模型的深度
推理，以减少正文首字等待；需要深度分析时可设置 `GLM_ENABLE_THINKING=true`
并重启本服务。该参数适用于当前 llama.cpp GLM 服务，不改变代码转换任务的配置。
后端日志记录本地并发等待、上游响应头、首个流式事件、正文首字和总耗时，
不记录问题、知识库正文或模型推理内容。

若远端模型只有一个推理槽位，其他长任务会使问答排队；关闭推理不能消除
这种等待。需要在模型服务端为问答配置独立实例，或根据硬件容量调整并行槽位，
并限制长任务的上下文和输出长度。

知识库管理接口：

```text
GET    /api/knowledge/info
POST   /api/knowledge/documents?filename=补充知识库.md
DELETE /api/knowledge/documents/补充知识库.md
```

上传接口的请求体是 UTF-8 文档原始字节，`Content-Type` 使用
`application/octet-stream`，默认单文件上限为 10MB。

连接：

```text
ws://127.0.0.1:8000/ws/knowledge
```

提问：

```json
{
  "type": "ask",
  "question": "转辙机锁闭不到位可能是什么原因？"
}
```

服务先判断问题是否需要知识库，并发送 `route` 事件（`knowledge` 或
`direct`）。知识库模式继续发送 `sources`；直接回答模式则跳过检索，由所选模型
直接流式作答。两种模式都会发送 `status`、`delta`、`done` 和 `ready` 事件。
发送 `{"type":"clear"}` 可清空当前连接的对话上下文。

## 算法日志 WebSocket

```text
ws://127.0.0.1:8000/ws/jobs/109534/logs
```

服务默认读取：

```text
/home/mec-share-dc/{job_id}/output/live.log
```

若不存在，会依次检查 `stdout.log`、`stderr.log`、`job.log`，然后选择最新的
`.log`、`.out` 或 `.err` 文件。Slurm 脚本需要把计算节点日志同步到任务 output
目录，WebSocket 才能实时推送。

运行测试：

```bash
conda run -n aider python -m unittest discover -s tests -v
```

## 安全边界

- aider 只能操作 `ALLOWED_WORKSPACES` 中的目录。
- 前端不能传入任意绝对路径。
- 生产环境应配置 `AGENT_API_TOKEN`，WebSocket 使用 `?token=...` 或
  `Authorization: Bearer ...`。
- API Key 只从环境变量读取。

部署到 controller 时，相对路径自动按仓库位置解析；如果代码工作区或作业目录
位于其他磁盘，在 `.env` 中覆盖 `AIDER_SOURCE_ROOT`、`WORKSPACE_ROOT` 和
`JOB_WORKSPACE_ROOT` 即可。
