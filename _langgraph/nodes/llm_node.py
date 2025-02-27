from typing import Callable, Dict, Any, List, Optional, Union
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from _langgraph.nodes.base_node import BaseNode
from _langgraph.base_state import BaseState
import logging

logger = logging.getLogger(__name__)

class LLMNode(BaseNode):
    """
    An LLM node that processes conversation messages and generates a response using an LLM.
    The node now receives a model and a list of tools so that it can bind the tools to the model.
    """
    model: ChatOpenAI
    tools: Optional[List[Callable[[Union[Callable, Runnable]], BaseTool]]] = None

    async def run(self, state: BaseState) -> Dict[str, Any]:
        messages = state.messages
        
        # Build the prompt template.
        system_prompt = SystemMessage(content="You are a helpful assistant.")
        messages = [system_prompt] + messages
        model = self.model
        # Bind tools to the model.
        if self.tools:
            model = self.model.bind_tools(self.tools)
        
        # Asynchronously invoke the chain.
        result = await model.ainvoke(messages)
        
        return {"messages": [result]}

# When instantiating the node, pass in the model and the list of tools.
llm_node = LLMNode(
    name="llm_node",
    description="Generates responses using an LLM based on conversation history.",
    func=LLMNode.run,  # assign the run method as the node's functionality.
    model=ChatOpenAI(temperature=0.7, model="gpt-4o-mini", streaming=True),
    tools=[]  # populate this list with your tools, e.g., [mtg_search]
)