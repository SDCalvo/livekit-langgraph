# plugin_langchain/llm.py

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableSerializable

from livekit.agents import llm


class LLM(llm.LLM):
    def __init__(self, *, runnable: RunnableSerializable) -> None:
        """
        Create a new instance of Langchain Runnable.
        """
        super().__init__()
        self._runnable = runnable

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        **kwargs: Any,
    ) -> "LLMStream":
        return LLMStream(runnable=self._runnable, chat_ctx=chat_ctx)


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        *,
        runnable: RunnableSerializable[dict, BaseMessage],
        chat_ctx: llm.ChatContext,
    ) -> None:
        super().__init__(chat_ctx=chat_ctx, fnc_ctx=None)
        # All messages except the system message fill the placeholder
        messages = [(m.role, m.content) for m in chat_ctx.messages[1:]]
        self._stream = runnable.astream({"messages": messages})

    async def __anext__(self) -> llm.ChatChunk | None:
        index = 0
        async for chunk in self._stream:
            index += 1
            if chunk.content:
                return llm.ChatChunk(
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                content=chunk.content, role="assistant"
                            ),
                            index=index,
                        )
                    ],
                )
        raise StopAsyncIteration