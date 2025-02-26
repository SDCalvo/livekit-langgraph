# graph_factory.py
import asyncio
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import Callable, Any

class LangGraphFactory:
    """
    A factory for creating compiled graphs from a state schema.
    """
    def __init__(self, state_schema: Any, checkpointer: Any = None) -> None:
        """
        Initialize the factory with a state schema and an optional checkpointer.
        """
        self.state_schema = state_schema
        self.checkpointer = checkpointer or MemorySaver()

    async def create_graph(self, build_fn: Callable[[StateGraph], Any]) -> CompiledStateGraph:
        """
        Create a compiled graph using the state schema and a build function.

        Args:
            build_fn: A function that builds the graph using a StateGraph instance.

        Returns:
            CompiledStateGraph: The compiled graph.
        """
        graph_builder = StateGraph(self.state_schema)
        if asyncio.iscoroutinefunction(build_fn):
            await build_fn(graph_builder)
        else:
            build_fn(graph_builder)
        return graph_builder.compile(checkpointer=self.checkpointer)
