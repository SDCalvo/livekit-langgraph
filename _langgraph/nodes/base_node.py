"""
Base Node class for langgraph nodes
"""


from pydantic import BaseModel, Field
from typing import Callable, Any, Optional

class BaseNode(BaseModel):
    name: str = Field(..., description="Unique identifier for the node.")
    description: str = Field(..., description="A brief description of what the node does.")
    # The callable implementing the node's behavior.
    # It's excluded from serialization since it doesn't belong to the data model.
    func: Optional[Callable[[Any], Any]] = Field(
        None, exclude=True, description="The async function that implements the node's functionality."
    )