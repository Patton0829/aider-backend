# Aider Agent Backend

中心云控制台的 FastAPI 智能体后端，包含三条独立实时通道：

- `/ws/agent`：千问聊天与 aider 代码修改。
- `/ws/jobs/{job_id}/logs`：Slurm 算法日志增量推送。
- `/ws/knowledge`：检索列车故障诊断知识库并调用 GLM 流式回答。

## 目录

```text
app/
  api/routes/       HTTP 和 WebSocket 路由
  core/             配置和鉴权
  schemas/          消息协议
  services/         千问、aider、日志流实现
tests/
```

## 启动

已有 `aider` Conda 环境时：

```bash
cd /Users/patton/system/aider-backend
cp .env.example .env
# 在 .env 中配置 DASHSCOPE_API_KEY
conda run -n aider python -m uvicorn app.main:app \
  --host 127.0.0.1 --port 8000
```

也可以创建独立虚拟环境：

```bash
cd /Users/patton/system/aider-backend
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
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
文档。上传或删除文档后会立即重建检索索引。GLM 使用 OpenAI 兼容接口，
配置见 `.env.example`。

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
`direct`）。知识库模式继续发送 `sources`；直接回答模式则跳过检索，由 GLM
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

部署到 controller 时，需要在 `.env` 中把 `AIDER_SOURCE_ROOT`、
`WORKSPACE_ROOT` 和 `JOB_WORKSPACE_ROOT` 改为服务器上的真实绝对路径。
