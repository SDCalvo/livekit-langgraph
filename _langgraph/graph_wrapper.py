# graph_wrapper.py
from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator
from livekit.agents import llm
from langgraph.graph.state import CompiledGraph
from langchain_core.messages import BaseMessage
from langgraph.constants import CONF
from livekit.agents.llm.llm import APIConnectOptions
import logging

logger = logging.getLogger(__name__)

# LiveKit wrapper for a LangGraph-compiled graph.
class LivekitGraphRunner(llm.LLM):
    def __init__(self, graph: CompiledGraph) -> None:
        super().__init__()  # Initializes base attributes (e.g., _events)
        self.graph = graph

    def chat(
        self, *, chat_ctx: llm.ChatContext, **kwargs: Any
    ) -> GraphStream:
        # Pass self as the LLM so that _llm is not None.
        return GraphStream(llm=self, graph=self.graph, chat_ctx=chat_ctx)

# GraphStream implementation, fulfilling the _run abstract method.
class GraphStream(llm.LLMStream):
    def __init__(self, *, llm: llm.LLM, graph: CompiledGraph, chat_ctx: llm.ChatContext) -> None:
        # Create dummy connection options. In production, replace with real values.
        dummy_conn_options = APIConnectOptions(
            max_retry=1,
            retry_interval=0.2,
            timeout=10,
        )
        # Pass the real LLM instance (from LivekitGraphRunner) so _label is available.
        super().__init__(llm=llm, chat_ctx=chat_ctx, fnc_ctx=None, conn_options=dummy_conn_options)
        # Convert LiveKit ChatContext messages (skip the system message) into a list of (role, content) pairs.
        messages = [(m.role, m.content) for m in chat_ctx.messages[1:]]
        config = {"configurable": {"thread_id": "1"}}
        self._stream = graph.astream({
            "messages": messages
        }, config=config)

    async def _run(self) -> None:
        """
        We need this here so that the abstract method is fulfilled.
        """

    async def __anext__(self) -> llm.ChatChunk | None:
        index = 0
        async for update in self._stream:
            logger.info(f"Chunk update: {update}")
            index += 1
            if update["llm_node"]["messages"]:
                # Retrieve the last message.
                last_msg = update["llm_node"]["messages"][-1]
                # If the message object has a 'content' attribute and it's non-empty, wrap it.
                if hasattr(last_msg, "content") and last_msg.content:
                    return llm.ChatChunk(
                        request_id=index,
                        choices=[
                            llm.Choice(
                                delta=llm.ChoiceDelta(content=last_msg.content, role="assistant"),
                                index=index,
                            )
                        ]
                    )
        raise StopAsyncIteration
