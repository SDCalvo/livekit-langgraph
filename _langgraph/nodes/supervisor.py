from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing import Any
from _langgraph.base_state import BaseState

async def supervisor_node(state: BaseState) -> BaseState:
    """
    A supervisor node that examines the conversation history,
    node metadata, and state context (e.g. last node executed and its output)
    to decide which node to execute next.
    """
    # Extract conversation history (only text content from messages)
    conversation = "\n".join(
        m.content for m in state.messages if hasattr(m, "content")
    )
    
    # Extract last executed node and its output from the context (if available)
    last_node = state.context.get("last_node", "None")
    last_output = state.context.get("last_output", "No output available")
    
    # Format the node registry as a bullet list (assuming node_registry values are NodeMetadata)
    node_list_str = "\n".join(
        f"- {meta.name}: {meta.description}"
        for meta in state.node_registry.values()
    )
    
    # Build the prompt that includes:
    # 1. The full conversation history.
    # 2. Information about the last node executed and its output.
    # 3. The list of available nodes.
    prompt_text = f"""
        You are a workflow supervisor. Based on the following details, decide which node should run next.
            
        Conversation so far:
        {conversation}

        Last node executed: {last_node}
        Output from last node: {last_output}

        Available nodes:
        {node_list_str}

        Please output ONLY the exact name of the node that should execute next.
    """.strip()
    
    # Create a ChatOpenAI instance (adjust model parameters as needed)
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=False)
    
    # Call the model asynchronously to get a decision.
    result = await model.ainvoke({"messages": [SystemMessage(content=prompt_text)]})
    chosen_node = result.content.strip()
    
    # Update the state context with the supervisor's decision.
    state.context["supervisor_decision"] = chosen_node
    # Optionally, record the decision as a new message.
    state.messages.append(HumanMessage(content=f"[Supervisor] Next node: {chosen_node}"))
    
    return state