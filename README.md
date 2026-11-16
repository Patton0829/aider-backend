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
KNOWLEDGE_EMBEDDING_MODEL=qwen3.7-text-embedding-flash
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

## 第二阶段：LangGraph 问答编排与评测

后端使用 LangGraph 的单次有界工作流，继续兼容现有 `/ws/knowledge` 前端：

```text
conversation（路由、追问补全）
  ├─ 通用问题 → generate
  └─ 知识问题 → retrieve → assess
                           ├─ 有全部或部分依据 → generate
                           └─ 无依据或检查失败 → clarify
                 → validate → finish
```

独立问题不调用改写模型。连续追问使用独立保存的故障主题与最近对话补全检索问题，
即使历史被截断也保留原始主题；改写失败时退回“原始主题＋最近问题＋本轮追问”。
显式更换设备主题会清除旧主题。基础路由与追问识别仍使用规则，并非模型语义路由。
未启用第四阶段 MySQL 历史功能时，会话只在当前 WebSocket 连接内保存，清空或断开即释放。

检索后进行一次关闭思考的短 JSON 模型请求，判断资料是否支持问题，并选择可用证据：

- 完整支持：生成带引用的答案。
- 部分支持：只使用可用证据，回答中追加缺失资料的说明。
- 无事实支持：说明知识不足并请求补充信息，不调用答案生成。
- 证据检查超时、格式错误或连接失败：明确说明检查未完成，不把错误当成“知识库无答案”。
- 问题明确指定的条目未被召回：不使用邻近条目替代。

新回答优先直接使用条目编号引用，例如 `[SW-016]`，与右侧相关条目对应。
没有条目编号时使用 `[文档名:L起始行-结束行]`；同编号对应多个片段时补充文档和行号，
仍重名时附加本轮片段序号，避免将不同来源合并成一个引用。
内部资料序号仍对应本轮检索结果顺序，用于证据审核；模型输出 `[资料1]` 时也会转换为上述可读标识。
历史已保存的回答保持原文，不重新改写旧引用。此变更只需部署后端，不涉及前端构建或索引重建。
流式引用校验会暂存未闭合的方括号，
检查引用是否属于本轮获准证据，再输出；无效引用替换为标记。支持资料编号、唯一条目编号、
完整来源标签和简写章节标签。最终检查还会发现正文中未被证据支持的条目编号；缺少有效引用
或存在无效引用时，在回答末尾追加警示。引用元数据绑定 `chunk_id`、文档和 `kb_version`。
**该校验验证引用身份，不证明每个自然语言结论均被引用内容蕴含，也不能替代现场诊断核验。**

新配置有默认值，已有服务器 `.env` 不必覆盖，按需补充：

```dotenv
KNOWLEDGE_REWRITE_FOLLOWUPS=true
KNOWLEDGE_EVIDENCE_CHECK=model
KNOWLEDGE_CONTROL_TIMEOUT_SECONDS=12
```

证据检查每轮知识问题增加一次模型请求；追问改写还增加一次。
`KNOWLEDGE_CONTROL_TIMEOUT_SECONDS` 覆盖控制请求的排队和模型等待。
可设 `KNOWLEDGE_EVIDENCE_CHECK=off` 对比旧路径延迟，关闭后仅保留无结果/指定条目缺失检查，
失去模型语义充分性判断。引用校验继续启用。生成、向量、重排模型及 Milvus 目录配置不变。

服务器更新（保持一个 worker；无需删除数据或手动重建索引）：

```bash
cd /root/system/aider-backend
git pull --ff-only
PYTHON_BIN=/opt/miniconda3/bin/python3 bash scripts/setup.sh
.venv/bin/python -m app.manage check
systemctl restart aider-backend
.venv/bin/python -m app.manage smoke
```

`/api/knowledge/info` 增加 `workflow=langgraph_single`、`evidenceCheck` 和
`conversationPersistence=connection`。原有事件不变，新增带 `data` 的 `assessment`、
`validation`、`metrics` 事件，均携带原 `requestId`。旧前端可以忽略新事件，仍能看到
状态、证据、流式回答及正文中的资料不足/引用警示。

`metrics` 分别记录 `conversationMs`、`retrievalMs`（含召回与重排）、`assessmentMs`、
`generationMs`、`firstContentMs` 和 `totalMs`，以及改写方式、证据结果和索引版本。
第一字与完整耗时覆盖额外控制请求；不能只比较检索耗时来判断用户等待时间。

### 评测方法

仓库提供 `evaluation/fault_diagnosis.json`：32 组、38 轮问题，覆盖精确条目、语义表达、
连续追问、资料不足和通用问题，使用原始故障诊断知识库作为标签依据。
默认顺序请求运行中的服务，不另开 Milvus 客户端；每组新建连接，组内追问复用连接。

```bash
# 快速验证前 3 组
.venv/bin/python -m app.manage evaluate --limit 3
# 完整评测；会调用实际千问 API，消耗模型额度
.venv/bin/python -m app.manage evaluate
```

报告默认保存在被 Git 忽略的 `storage/rag/evaluations/latest.json`，可通过 `--output` 改位置。
报告包含每轮回答、来源、证据结果、引用校验及计时；汇总路由/答案支持检查、标签片段 Recall@5、
引用身份通过率，以及独立知识问题和追问各自的均值、P50/P95（最近秩法）。
检查失败不记为正确拒答；部分支持记为可回答，但报告标明缺失内容。
报告逐轮保存进度；中途停止时可读取 `completed=false` 的部分报告。
这些标签和引用检查不等于“诊断准确率”，需人工阅读报告核查事实支撑。
服务器上传了不同知识库时，应同步维护评测标签；不要为通过测试强行放宽证据检查。

开发机使用 290 个片段的热索引验证了整套 32 组/38 轮，标签检查全部通过。
记录见 [开发基线](evaluation/stage2-baseline.json)，独立知识问题首字 P50 约 1.77 秒、
完整回答 P50 约 3.62 秒，证据检查 P50 约 1.05 秒。
这不是服务器 SLA，也不是诊断准确率；controller 的 331 个片段需在部署后另行评测。

编排依据：[LangGraph 官方流式文档](https://docs.langchain.com/oss/python/langgraph/streaming)，
通过 custom stream 保持现有 OpenAI 兼容模型客户端和 WebSocket 协议。

## 第三阶段：Redis 召回与重排缓存

可选启用 Redis 两层缓存，默认 TTL 为 300 秒：

1. 召回缓存：缓存 BM25/Dense/RRF 的候选片段 ID、内容哈希和分数；命中后跳过查询向量 API、
   BM25、Milvus 检索与 RRF。
2. 重排缓存：按问题、知识库版本、重排模型、TopK 和**有序候选片段**缓存重排结果；命中后跳过重排 API。

缓存不保存问答原文、文档正文、模型密钥、证据检查结论或生成答案。
查询只做 Unicode NFC 和首尾空白归一化，问题正文以哈希进入缓存键，不做语义相似缓存。
追问使用补全后的检索问题，因此相同追问在不同上下文中可能产生不同缓存键。
命中不会延长 TTL；模型配置、召回参数或 TTL 改动后使用对应的新键。

每轮查询先固定发布版本，再访问缓存。键绑定知识库版本、语料目录、API 账号哈希、
向量模型/维度/地址、召回模式与参数。缓存内容必须通过结构、数量、唯一性、有限分数、
片段归属及内容哈希校验后，才从该版本快照恢复证据。上传、修改、删除文档后，新版本不会
读到旧版本的结果；显式回滚后可复用尚未过期的对应版本缓存。旧键自行过期，不清空其他业务缓存。
修改检索算法或缓存结构时，需要同步提升代码中的策略/缓存格式版本。

Redis 连接或命令失败时继续执行原有检索链路，不改变知识库就绪状态。
每次缓存命令最多等待 0.2 秒，失败后 30 秒内跳过缓存访问，再自动尝试恢复。
日志和状态只报告错误类型，不输出 Redis URL 或密码。
相同问题、版本和配置的并发检索在当前单 worker 内合并，某个用户断开不会取消其他用户共享的检索。
这不是跨进程分布式锁；后续切到远端 Milvus 多副本部署时需要另行设计请求合并。

### 在 controller 安装专用缓存实例

先拉取代码和依赖：

```bash
cd /root/system/aider-backend
git pull --ff-only
PYTHON_BIN=/opt/miniconda3/bin/python3 bash scripts/setup.sh
```

已有可用 Redis 时，可直接配置它的连接地址，跳过以下专用实例安装。
以下配置使用独立的 `127.0.0.1:6380`，不改动已有 6379 实例的配置；若 6380 已被其他服务占用，
应先选择其他端口并同步修改配置与 `REDIS_URL`，不要停止不明进程。

```bash
apt-get update
apt-get install -y redis-server
install -m 644 deploy/redis-rag.conf /etc/redis/aider-rag.conf
install -m 644 deploy/aider-rag-redis.service /etc/systemd/system/aider-rag-redis.service
systemctl daemon-reload
systemctl enable --now aider-rag-redis
redis-cli -h 127.0.0.1 -p 6380 ping
```

该实例仅用于临时缓存，限制内存 128 MB、使用 allkeys-lru，并关闭磁盘持久化。
知识库及索引仍存放于原有 Milvus/SQLite，Redis 重启后的首次查询会重新填充缓存。

在服务器原有 `.env` 中补充或修改以下字段（不要覆盖密钥、模型、路径等现有配置）：

```dotenv
KNOWLEDGE_CACHE_ENABLED=true
REDIS_URL=redis://127.0.0.1:6380/0
KNOWLEDGE_CACHE_TTL_SECONDS=300
```

默认关闭缓存，保证没有 Redis 的既有部署可以继续运行。
可选配置 `KNOWLEDGE_CACHE_NAMESPACE`、`KNOWLEDGE_CACHE_TIMEOUT_SECONDS` 和
`KNOWLEDGE_CACHE_RETRY_SECONDS`，见 `.env.example`。远端有认证/TLS 的 Redis 应在本地 `.env`
配置实际 `redis://` / `rediss://` 地址，密钥不要提交 Git。

```bash
.venv/bin/python -m app.manage check
systemctl restart aider-backend
curl -fsS --retry 15 --retry-connrefused --retry-delay 1 http://127.0.0.1:8000/health
.venv/bin/python -m app.manage cache-check
.venv/bin/python -m app.manage cache-smoke
.venv/bin/python -m app.manage smoke
```

`cache-check` 通过运行中的后端检查缓存连接；看到 `state=ready` 才代表后端真实连接成功。
`manage check` 只检查配置与 Python 依赖，不探测 Redis。
`cache-smoke` 连续执行两次真实检索，验证第二次缓存命中以及片段 ID、内容哈希、分数和版本一致；
不会清理已有缓存，也不调用证据检查或答案生成。第一次可能已经命中，不能一概称为冷查询。
返回 `passed=true` 且第二次 `recallCacheHit=true`、`rerankCacheHit=true` 表示两层均生效。
`smoke` 则继续验证完整问答与引用。

管理端点均沿用 `AGENT_API_TOKEN` Bearer 鉴权：

```text
GET  /api/knowledge/cache
POST /api/knowledge/cache/probe   {"question":"转辙机锁闭不到位的原因是什么？"}
```

`/api/knowledge/info` 增加 `queryCache`；WebSocket 的 `metrics` 增加 `queryCacheEnabled`、
`queryCacheState`、`recallCacheHit`、`rerankCacheHit`、`cacheCoalesced`、`recallMs`、`rerankMs`
和 `searchMs`。缓存命中标志是本次检索的局部数据，不使用共享的“最后一次命中”变量，避免并发串扰。
缓存状态的 counters 为当前后端进程累计值，重启后归零。
完整评测报告新增缓存命中率、合并请求数和降级轮数；Redis 状态、URL 与凭据不会进入生成提示。

开发机真实 Redis＋千问＋Milvus 联调中，同一问题首次检索约 359 ms，缓存命中约 13 ms，
证据与版本一致，见 [缓存探测记录](evaluation/stage3-cache-baseline.json)。这是检索阶段数据；
证据检查和回答生成仍需模型请求，不代表整轮回答也缩短相同比例。

运行包含真实 Redis 的回归测试：

```bash
RAG_TEST_REDIS_URL=redis://127.0.0.1:6380/0 .venv/bin/python -m unittest discover -s tests -q
```

测试使用独立随机键前缀，不清空 Redis；不设置测试地址时跳过真实 Redis 连接测试。

异步连接、超时与客户端关闭遵循 [Redis 官方 Python 异步客户端文档](https://redis.io/docs/latest/develop/clients/redis-py/async/)。

## 第四阶段：MySQL 历史会话与结果持久化

可选将诊断会话与每轮问答保存到专用 MySQL 数据库。默认关闭，旧前端、旧 WebSocket 客户端
继续使用原有临时会话。新前端支持新建、历史列表、恢复、分页加载、清空、删除，以及查看每轮依据。
刷新页面或服务重启后，可恢复**已经完成并提交**的对话、故障主题和最近上下文。
回答过程中断会记录为 `failed` / `interrupted`，保留已收到的部分文本，但不加入后续模型上下文。
进程崩溃遗留的运行记录在租约到期后标记为中断，不会当作完整回答使用。

数据库包含以下带 `rag_` 前缀的表，初始化不会删除或改写中心云业务表：

- `rag_schema_version`：本功能表结构版本。
- `rag_conversations`：会话、归属凭据哈希、当前上下文、修订号及写入租约。
- `rag_turns`：问题、答案、完成状态、模型、证据快照、知识库版本、引用校验及阶段耗时。

同一会话使用 MySQL 行锁和可续期的租约串行写入；不同会话可以并发。
每个持久问答必须携带 `requestId`。相同编号、相同问题的已完成请求回放原记录，
不再次调用模型；相同编号换问题、重复的未完成请求或当前会话正在生成时返回明确冲突。
提交回答和更新上下文位于同一事务；数据库提交失败时不发送成功保存状态。
已有历史数据不可读取时明确报错，不会静默创建空会话覆盖原历史。临时问答仍可独立使用。

目前访问隔离采用浏览器保存的随机会话凭据，服务端只保存 SHA-256 哈希。
HTTP 通过 `X-Conversation-Token`，WebSocket 通过消息体 `ownerToken` 提供，凭据不放入 URL，
并继续沿用原 `AGENT_API_TOKEN` 服务鉴权。知道会话 ID 并不足以读取、删除或恢复会话。
前端按当前登录账号分开保存浏览器凭据，但后端尚未对接中心云 JWT/统一用户身份，
**不提供跨设备账号历史同步**。清除浏览器存储，或现有退出登录流程清空本地存储后，
原凭据丢失将无法从该浏览器访问对应旧记录；后续接入统一用户身份时再提供账号级恢复。

新增配置：

```dotenv
KNOWLEDGE_HISTORY_STORE=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=aider_rag
DB_USERNAME=aider_rag
DB_PASSWORD="本机配置的原始密码"
KNOWLEDGE_DATABASE_URL=
KNOWLEDGE_HISTORY_DB_TIMEOUT_SECONDS=5
KNOWLEDGE_HISTORY_LEASE_SECONDS=300
```

配置只放在 `aider-backend/.env`，不读取中心云后端的 `.env`。连接 RDS 时将 `DB_HOST`、
`DB_PORT` 改为客户端使用的数据库连接地址和端口，`DB_DATABASE` 填专用库 `aider_rag`。
`DB_PASSWORD` 直接填写原始密码，程序自动完成 URL 编码；密码含 `#` 或空格时用双引号包住。
兼容旧的 `KNOWLEDGE_DATABASE_URL`：非空时优先使用连接串，使用分项配置时必须删除或留空该项。
同名进程环境变量优先于 `.env` 文件中的值。
首次配置本机 MySQL 可使用下面的辅助脚本（RDS 已建库、已有账号时无需运行），
它生成独立随机密码并直接写入本地文件，保留现有模型、密钥、路径及 Redis 配置。
MySQL/MariaDB 业务实例如已存在，不要为本功能重新安装或替换数据库服务。
初始化命令显式创建新表，不在每次后端启动时自动建表或升级表结构。

```bash
# 首次创建本机专用数据库和用户；会提示输入现有 MySQL 管理员密码
cd /root/system/aider-backend
.venv/bin/python scripts/configure_mysql_history.py
# 若管理员用户名、MySQL 命令路径或端口不同，使用脚本 --help 查看参数。
```

该脚本仅授予专用数据库权限，不修改已有账号密码；数据库用户已存在时会停止，
不会自动旋转凭据。已有专用数据库连接时跳过脚本，直接配置 `.env` 后执行 `db-init`。
脚本依赖服务器已有的 `mysql` 命令；管理员密码通过客户端终端提示输入，生成的业务密码经 stdin
传入 MySQL，不进入进程命令参数或输出日志。SQL 执行失败时保留原 `.env`。

```bash
.venv/bin/python -m app.manage db-init
.venv/bin/python -m app.manage check
systemctl restart aider-backend
curl -fsS --retry 15 --retry-connrefused --retry-delay 1 http://127.0.0.1:8000/health
.venv/bin/python -m app.manage history-check
.venv/bin/python -m app.manage history-smoke
```

`history-check` 验证运行中后端的数据库连接及表结构。
`history-smoke` 创建仅归本次测试所有的临时持久会话，验证保存、断线重连、原请求回放、
追问上下文以及其他凭据访问被拒绝；结束后只删除本次测试会话，不清空业务记录。
该测试会调用真实问答 API。

管理 API：

```text
GET    /api/knowledge/history/status
POST   /api/knowledge/conversations
GET    /api/knowledge/conversations?limit=20&offset=0
GET    /api/knowledge/conversations/{id}?limit=50&before=序号
DELETE /api/knowledge/conversations/{id}
```

会话 API 除状态检查外，均要求有效的 `X-Conversation-Token`。持久 WebSocket 示例：

```json
{"type":"resume","conversationId":"REST创建的UUID","ownerToken":"浏览器随机凭据","requestId":"resume-1"}
```

```json
{"type":"ask","question":"转辙机锁闭不到位的原因？","conversationId":"REST创建的UUID","ownerToken":"浏览器随机凭据","requestId":"唯一问答编号"}
```

新增 `history`、`session` 事件及 `conversationId` 字段；只有 MySQL 提交成功后，
才发送带 `persisted=true` 的完成状态。清空持久会话需携带相同凭据，且会删除本会话的问答和上下文；
新建会话不删除旧记录。前端完成恢复前会暂停发送，避免在旧上下文尚未加载时提交追问。
第三阶段 Redis 仍只缓存检索和重排，不缓存历史问答或最终答案。

MySQL 数据应与原始知识库、索引 catalog 一同纳入现有备份流程。
表中保留证据的片段、内容哈希及知识库版本，历史原文追溯还依赖对应版本的 catalog/知识库文件。

### 第三、四阶段一起部署

1. 更新后端并安装锁定依赖：

   ```bash
   cd /root/system/aider-backend
   git pull --ff-only
   PYTHON_BIN=/opt/miniconda3/bin/python3 bash scripts/setup.sh
   ```

2. 按第三阶段说明安装专用 Redis（或复用已有实例），启用 `KNOWLEDGE_CACHE_ENABLED=true`。
   按第四阶段配置专用 MySQL 连接，并运行 `.venv/bin/python -m app.manage db-init`。

3. 更新前端诊断页和 `useHistory.ts`。从前端仓库拉取相应提交；若使用补丁交付，先检查再应用：

   ```bash
   cd /root/system/control-center-fe
   git apply --check /上传目录/frontend-history.patch
   git apply /上传目录/frontend-history.patch
   pnpm build
   ```

   仓库拉取和补丁是两种交付方式，已经包含该提交时不要重复应用补丁。
   补丁冲突时保留现场改动并人工合并，不强制覆盖。生成的 `dist/` 按现有 Nginx 静态目录发布流程发布，
   不修改中心云的其他页面或后端业务配置。前端未更新时仍只能使用临时会话。

   也可使用随前端发布的 Git bundle 在服务器离线获取提交，再进行三方合并：

   ```bash
   git fetch /上传目录/frontend-history.bundle HEAD
   git cherry-pick FETCH_HEAD
   pnpm build
   ```

   bundle 需要服务器前端仓库包含其基础提交；可先运行 `git bundle verify /上传目录/frontend-history.bundle`。
   应用前先检查工作区改动，不将未提交的现场修改强行覆盖。

4. 配置完成后统一重启并验证：

   ```bash
   cd /root/system/aider-backend
   .venv/bin/python -m app.manage check
   systemctl restart aider-backend
   curl -fsS --retry 15 --retry-connrefused --retry-delay 1 http://127.0.0.1:8000/health
   .venv/bin/python -m app.manage cache-check
   .venv/bin/python -m app.manage cache-smoke
   .venv/bin/python -m app.manage history-check
   .venv/bin/python -m app.manage history-smoke
   ```

5. 打开前端发送问题，刷新后确认记录和依据恢复，再发追问。
   后端服务重启后再恢复同一会话，确认已完成的历史仍在。

回退时可设 `KNOWLEDGE_HISTORY_STORE=off`、`KNOWLEDGE_CACHE_ENABLED=false` 后重启，
原有临时问答继续可用，MySQL 历史表和 Milvus 数据保留。

实现依据：[SQLAlchemy 异步连接文档](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)。

## 第五阶段：上下文预算与会话记忆

完整问答仍保存在 `rag_turns`；近期原文和旧内容摘要保存在现有
`rag_conversations.state` JSON 中。无需新增表、无需重新建库或重建知识库向量。
本阶段不修改前端，已有“已保存”、恢复、清空和删除功能继续适用。

每次问答链路中的模型调用（包括改写、证据审核和摘要选择）都会检查输入预算。
输入上限为 `min(MAX_INPUT, CONTEXT_WINDOW - 预留输出 - 安全余量)`。
当前使用 **UTF-8 字节数 + 消息开销的保守 token 估算**，不是千问 tokenizer 的精确计数；
无需额外下载模型或 tokenizer。下列 64K/48K 是应用侧预算，不代表实际模型的最大上下文窗口，
接入较小窗口的模型时必须相应降低配置。

```dotenv
KNOWLEDGE_CONTEXT_WINDOW_TOKENS=65536
KNOWLEDGE_MAX_INPUT_TOKENS=49152
KNOWLEDGE_CONTEXT_SAFETY_TOKENS=2048
KNOWLEDGE_EVIDENCE_BUDGET_TOKENS=24000
KNOWLEDGE_HISTORY_TURNS=6
KNOWLEDGE_MEMORY_ENABLED=true
KNOWLEDGE_MEMORY_HISTORY_TOKENS=12000
KNOWLEDGE_MEMORY_SUMMARY_TOKENS=2048
KNOWLEDGE_MEMORY_SUMMARY_TIMEOUT_SECONDS=4
KNOWLEDGE_MEMORY_RECALL_TOKENS=8000
KNOWLEDGE_MEMORY_RECALL_LIMIT=6
KNOWLEDGE_MEMORY_SCAN_LIMIT=200
```

近期原文同时受轮数和估算 token 预算约束，原文预算还限制在有效输入预算的三分之一以内。
达到阈值后才调用一次摘要选择，普通短对话不增加摘要请求；压缩时将近期轮数降至上限约一半，
留出后续轮次空间，减少达到上限后每轮都做摘要的情况。摘要选择默认最多等待 4 秒。
摘要是**有来源的结构化摘录**：
保留 user/assistant 身份和原消息 SHA-256，模型只选择候选下标，不能生成或改写事实。
输入给模型时明确区分用户曾说过的话和模型先前建议，历史内容不是本轮故障诊断证据。
摘要超时、输出非法或候选编号不存在时，退回有界的原文摘录，不影响本轮回答保存。
摘要与答案、上下文通过原有事务一起提交；失败或中断的回答不更新完成态记忆。
诊断话题切换仍会重置工作上下文；原始旧记录可在回顾请求中按需读取。

“我第一条问题是什么”“第十二条问题是什么”等回顾问题，会在**当前归属凭据和当前会话**内
按原始 sequence 精确读取已完成记录，不受近期窗口和主题扫描条数限制。
失败轮次不作为记忆，序号存在空缺时不会把其他轮次冒充成指定轮次。
“我问过哪些问题”返回有限条原始记录并标记还有更多；
“回顾之前关于轴承温度的对话”对最近最多 200 条已完成记录进行词汇相关性匹配。
这是有界的 MySQL 历史检索，尚未建立历史问答的向量索引，不提供跨用户、跨会话的记忆查询。
返回给模型的旧回答只是带截断标记的摘录，原文仍在 MySQL 中。

证据预算按完整片段选择，保留可核验的原文与引用编号，不截断半条证据来凑长度。
最后一次消息装配还会移除较早的可选历史，并保留系统规则、完整当前问题及回顾请求的原始记录。
必须保留的输入仍放不下时，返回 `ContextBudgetExceeded`，不会静默截断问题。
上游明确报告上下文超限时，仅在尚未开始流式输出、能够进一步缩减历史时重试一次；
已经输出部分回答后不会重发生成请求。模型实际窗口仍应与本地配置一致。

`metrics` 新增估算输入量、输入预算、裁剪消息数、上下文重试、摘要方法/条数、
旧历史读取方式/条数和记忆处理耗时。`manage check` 与 `/api/knowledge/info` 同时提供记忆配置。
只有调用模型选择摘录的轮次标记 `model_selection`；退回原文摘录时标记 `extractive_fallback`。

代码拉取后仍需按现有部署流程重启后端才生效。部署后的验收可运行：

```bash
cd /root/system/aider-backend
.venv/bin/python -m app.manage check
.venv/bin/python -m app.manage memory-smoke
```

`memory-smoke` 会创建独立临时会话，发送超过近期轮数的短问题，检查摘要生成、重新连接恢复、
第一条原始问题读取、输入预算与完整记录保留，最后删除自己创建的临时会话。
默认近期轮数为 6 时，共生成 8 轮回答并可能产生摘要选择调用，会消耗少量 API 额度。
可手动验证：连续追问超过 6 轮后询问“我第一条问题是什么”，再刷新恢复并继续诊断追问。

设计参考：[LangGraph 记忆管理](https://docs.langchain.com/oss/python/langgraph/add-memory)、
[百炼输入长度错误说明](https://help.aliyun.com/en/model-studio/error-code)。

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
KNOWLEDGE_EMBEDDING_MODEL=qwen3.7-text-embedding-flash
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
默认向量模型为 `qwen3.7-text-embedding-flash`，维度为 1024。
已有部署需要单独更新服务器 `.env` 中的 `KNOWLEDGE_EMBEDDING_MODEL`；
拉取代码不会覆盖现有 `.env`。切换模型后重启服务，会为现有知识库重新生成
文档向量并发布新的 Milvus Collection，旧版本保留，无需删除数据库或重新上传文档。
新模型建库失败时不能使用旧模型的索引，需要恢复原模型配置或解决 API 调用问题。
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
