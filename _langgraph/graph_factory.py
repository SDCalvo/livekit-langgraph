# graph_factory.py
import asyncio
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Callable, Any

class LangGraphFactory:
    def __init__(self, state_schema: Any, checkpointer: Any = None) -> None:
        self.state_schema = state_schema
        self.checkpointer = checkpointer or MemorySaver()

    async def create_graph(self, build_fn: Callable[[StateGraph], Any]) -> Any:
        graph_builder = StateGraph(self.state_schema)
        if asyncio.iscoroutinefunction(build_fn):
            await build_fn(graph_builder)
        else:
            build_fn(graph_builder)
        return graph_builder.compile(checkpointer=self.checkpointer)
