from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from _langgraph.nodes.base_node import BaseNode
from _langgraph.base_state import BaseState

class LLMNode(BaseNode):
    """
    An LLM node that processes conversation messages and generates a response using an LLM.
    The node now receives a model and a list of tools so that it can bind the tools to the model.
    """
    model: ChatOpenAI
    tools: List[Any] = []  # list of tool functions decorated with @tool

    async def run(self, state: BaseState) -> Dict[str, Any]:
        messages = state.messages
        
        # Build the prompt template.
        system_prompt = SystemMessage(content="You are a helpful assistant.")
        messages = [system_prompt] + messages
        
        # Bind tools to the model.
        model_with_tools = self.model.bind_tools(self.tools)
        
        # Asynchronously invoke the chain.
        result = await model_with_tools.ainvoke(messages)
        
        # Append the LLM response as an AIMessage.
        new_message = AIMessage(content=result.content)
        
        return {"messages": [new_message]}

# When instantiating the node, pass in the model and the list of tools.
llm_node = LLMNode(
    name="llm_node",
    description="Generates responses using an LLM based on conversation history.",
    func=LLMNode.run,  # assign the run method as the node's functionality.
    model=ChatOpenAI(temperature=0.7, model="gpt-4o-mini", streaming=True),
    tools=[]  # populate this list with your tools, e.g., [mtg_search]
)