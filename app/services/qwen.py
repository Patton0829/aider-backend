from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from app.core.config import Settings


SYSTEM_PROMPT = """你是中心云控制台中的 aider 智能体。
你负责解释算法、Slurm 作业和代码，并给出准确、简洁的工程建议。
普通聊天不得声称已经修改文件；只有 edit 模式由 aider 实际执行代码修改。
"""


class QwenService:
    def __init__(self, settings: Settings):
        if not settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured")
        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model_alias: str,
    ) -> AsyncIterator[str]:
        model = self.settings.model_aliases.get(model_alias)
        if not model:
            raise ValueError(f"Unsupported model: {model_alias}")

        stream = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

