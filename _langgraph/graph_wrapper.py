# graph_wrapper.py
from __future__ import annotations
from typing import Any
from livekit.agents import llm
from langgraph.graph.state import CompiledGraph
from livekit.agents.llm.llm import APIConnectOptions
import logging

logger = logging.getLogger(__name__)

# LiveKit wrapper for a LangGraph-compiled graph.
class LivekitGraphRunner(llm.LLM):
    """
    A wrapper for a compiled graph to be used with LiveKit

    Args:
        graph (CompiledGraph): The compiled graph to be used.
    """
    def __init__(self, graph: CompiledGraph) -> None:
        """
        Initializes the LiveKit wrapper.
        """
        super().__init__()  # Initializes base attributes (e.g., _events)
        self.graph = graph

    def chat(
        self, *, chat_ctx: llm.ChatContext, **kwargs: Any
    ) -> GraphStream:
        """
        Creates a new GraphStream instance for the given ChatContext.

        Args:
            chat_ctx (llm.ChatContext): The chat context to be used.

        Returns:
            GraphStream: The new GraphStream instance.
        """
        # Pass self as the LLM so that _llm is not None.
        return GraphStream(llm=self, graph=self.graph, chat_ctx=chat_ctx)

# GraphStream implementation, fulfilling the _run abstract method.
class GraphStream(llm.LLMStream):
    """
    A stream that processes a chat context using a compiled graph.

    Args:
        llm (llm.LLM): The LLM instance to be used.
        graph (CompiledGraph): The compiled graph to be used.
        chat_ctx (llm.ChatContext): The chat context to be processed.

    Attributes:
        _stream (AsyncIterator): The stream that processes the chat context.
    """
    def __init__(self, *, llm: llm.LLM, graph: CompiledGraph, chat_ctx: llm.ChatContext) -> None:
        """
        Initializes the GraphStream.
        """
        # Create dummy connection options. In production, replace with real values.
        default_conn_options = APIConnectOptions(
            max_retry=1,
            retry_interval=0.2,
            timeout=10,
        )
        # Pass the LLM instance (from LivekitGraphRunner) so _label is available.
        super().__init__(llm=llm, chat_ctx=chat_ctx, fnc_ctx=None, conn_options=default_conn_options)
        # Convert LiveKit ChatContext messages (skip the system message) into a list of (role, content) pairs.
        messages = [(m.role, m.content) for m in chat_ctx.messages[1:]]
        config = {"configurable": {"thread_id": "1"}}
        self._stream = graph.astream({
            "messages": messages
        }, config=config, stream_mode="messages") # Stream mode is "messages" for now, if changed to "updates" the interface of __anext__ should change. 
        # Instead of update[0].content, it should be update["node_name"]["messages"][-1]["content"] or something like that, I can't remember exactly, but just print a chunk to see the structure.

    async def _run(self) -> None:
        """
        We need this here so that the abstract method is fulfilled. This method is not used. The inference is done in __anext__ instead but we need this method to fulfill the abstract method.
        """

    async def __anext__(self) -> llm.ChatChunk | None:
        """
        Processes the chat context and returns the next ChatChunk.

        Returns:
            llm.ChatChunk | None: The next ChatChunk, or None if the stream is done

        Raises:
            StopAsyncIteration: If the stream is done

        This method is an async generator that processes the chat context and yields the next ChatChunk.
        It does so by iterating over the stream and returning the last message as a Choice in a ChatChunk.
        Implementation should change if the stream method changes. Right now we are using "messages".

        This is where the magic hapens, the __anext__ method is the one that LiveKit expects to stream the messages to the client.
        And in order to do so it expects an output of type llm.ChatChunk, which is a class that contains a list of llm.Choice objects.
        But we are getting the inference done by langgraph, which is a compiled graph, and we are getting the output in the form of a stream of chunks
        that we then need to convert to the llm.ChatChunk format. This way we can use langgraph to do the inference as if it was a LiveKit LLM.
        """
        index = 0
        async for chunk in self._stream:
            index += 1
            if chunk[0].content:
                # Retrieve the last message.
                return llm.ChatChunk(
                    request_id=index,
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(content=chunk[0].content, role="assistant"),
                            index=index,
                        )
                    ]
                )
        raise StopAsyncIteration
