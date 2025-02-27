from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages

class NodeMetadata(BaseModel):
    """
    Contains metadata for a node.
    """
    name: str = Field(..., description="Unique identifier for the node.")
    description: str = Field(..., description="A brief description of what the node does.")
    

class BaseState(BaseModel):
    """
    Base state for the workflow. It holds conversation messages and a registry of node metadata.
    """
    messages: Annotated[List[Union[BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage]], add_messages] = Field(default_factory=list, description="List of conversation messages.")
    node_registry: Dict[str, NodeMetadata] = Field(default_factory=dict, description="Mapping of node names to their metadata.")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional state context.")

    def register_nodes(self, nodes: List[NodeMetadata]) -> None:
        """
        Registers a list of node metadata entries into the node registry.
        """
        for node in nodes:
            self.node_registry[node.name] = node

    def update_state(self, initial_state: Dict[str, Any]) -> None:
        """
        Sets the initial state of the workflow.
        """
        for key, value in initial_state.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.context[key] = value