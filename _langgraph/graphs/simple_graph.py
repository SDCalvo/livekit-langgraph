# simple_graph.py
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from _langgraph.graph_factory import LangGraphFactory

# Define the state schema. Here we use a simple message list.
class State(TypedDict):
    """
    A simple state schema with a list of messages.
    """
    messages: list[BaseMessage]

def build_simple_graph(graph: StateGraph) -> None:
    """
    Build a simple graph that chains a prompt with an LLM model.

    Args:
        graph (StateGraph): The graph to build.

    Returns:
        None
    """

    # Configure a real LLM instance.
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini", streaming=True)

    async def llm_node(state: State) -> State:
        """
        The LLM node that processes the messages.

        Args:
            state (State): The current state.

        Returns:
            State: The updated state.
        """
        messages = state["messages"]
        result = await llm.ainvoke(messages)
        return {"messages": [result]}

    # Add the node to the graph.
    graph.add_node("llm_node", llm_node)
    graph.add_edge(START, "llm_node")
    graph.add_edge("llm_node", END)

# Create a factory for our state schema.
factory = LangGraphFactory(state_schema=State)
# Compile the graph using our builder function.
async def get_compiled_graph() -> CompiledStateGraph:
    """
    Create a compiled graph using the simple graph builder.

    Returns:
        CompiledStateGraph: The compiled graph.
    """
    compiled_graph = await factory.create_graph(build_simple_graph)
    initial_state = {
        "messages": []
    }
    return compiled_graph, initial_state