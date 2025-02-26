# simple_graph.py
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from _langgraph.graph_factory import LangGraphFactory

# Define the state schema. Here we use a simple message list.
class State(TypedDict):
    messages: list  # Optionally, you can annotate with add_messages

def build_simple_graph(graph: StateGraph) -> None:

    # Set up a prompt template for the conversation.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that provides interesting facts."),
            ("placeholder", "{messages}"),
        ]
    )
    # Configure a real LLM instance.
    llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini", streaming=False)
    # Chain the prompt with the LLM.
    chain = prompt | llm

    async def llm_node(state: State):
        messages = state["messages"]
        # Ensure you’re invoking the chain (prompt | model) asynchronously.
        result = await chain.ainvoke({"messages": messages})
        return {"messages": [result]}

    # Add the node to the graph.
    graph.add_node("llm_node", llm_node)
    graph.add_edge(START, "llm_node")
    graph.add_edge("llm_node", END)

# Create a factory for our state schema with the MemorySaver as the checkpointer.
factory = LangGraphFactory(State)
# Compile the graph using our builder function.
async def get_compiled_graph():
    return await factory.create_graph(build_simple_graph)
