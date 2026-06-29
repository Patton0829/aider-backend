# Aider Agent Backend

中心云控制台的 FastAPI 智能体后端，包含两条独立实时通道：

- `/ws/agent`：千问聊天与 aider 代码修改。
- `/ws/jobs/{job_id}/logs`：Slurm 算法日志增量推送。

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
