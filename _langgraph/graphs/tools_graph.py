# tool_workflow_graph.py
import asyncio
from langgraph.graph import StateGraph, START, END
from _langgraph.graph_factory import LangGraphFactory
from _langgraph.base_state import BaseState, NodeMetadata
from _langgraph.nodes.llm_node import LLMNode  # Our custom LLM node
from _langgraph.tools.mtg_tool import mtg_search     # Our MTG search tool (decorated with @tool)
from _langgraph.base_state import BaseState
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from typing import Tuple

# Define a routing function that checks if the last message contains tool calls.
def route_tools(state: BaseState) -> str:
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

async def build_tool_graph(graph: StateGraph) -> None:
    # Add nodes to the graph.
    
    # Instantiate the LLM node, passing in the model and the list of tools.
    llm_instance = ChatOpenAI(temperature=0.7, model="gpt-4o-mini", streaming=True)
    llm_node = LLMNode(
        name="llm_node",
        description="Generates responses using an LLM with bound tools based on the conversation history.",
        func=LLMNode.run,
        model=llm_instance,
        tools=[mtg_search]  # Bind our MTG search tool.
    )
    graph.add_node("llm_node", llm_node.run)
    
    # Instantiate the built-in ToolNode with our mtg_search tool.
    tool_node_instance = ToolNode(tools=[mtg_search])
    graph.add_node("tool_node", tool_node_instance.invoke)
    
    # Build the graph edges:
    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges("llm_node", route_tools, {"tool_node": "tool_node", END: END})
    graph.add_edge("tool_node", "llm_node")

# Create a factory for our BaseState.
factory = LangGraphFactory(BaseState)

async def get_compiled_graph() -> Tuple[BaseState, StateGraph]:
    """
    Compiles the graph and defines an initial state.
    
    Returns:
        A tuple of (compiled_graph, initial_state)
    """
    compiled_graph = await factory.create_graph(build_tool_graph)
    initial_state = {
        "node_registry": {
            "llm_node": {"name": "llm_node", "description": "Generates LLM responses with bound tools."},
            "tool_node": {"name": "tool_node", "description": "Executes MTG search tool calls."}
        },
        "context": {}
    }
    return compiled_graph, initial_state