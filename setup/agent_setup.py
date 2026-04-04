import json
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from langchain_classic.prompts import PromptTemplate
from langchain_core.globals import set_debug
from langchain_community.tools import BaseTool

import tools.map_tools
import tools.personalization_tools
import static.constants as constants
import tools.conversation_tools as conversation_tools
import tools.rag_tools as rag_tools

# Wrapping the tools with LangChain's @tool decorator for better integration

tools : list[BaseTool] = [*tools.personalization_tools.tools_list, *tools.map_tools.tool_list, *rag_tools.tools]

for tool in tools:
    tool.handle_tool_error = True

llm = ChatOllama(model=constants.PRIMARY_MODEL, request_timeout=1000, num_gpu=1, temperature=0, streaming=True)
# prompt = PromptTemplate.from_template(constants.AGENT_REACT_TEMPLATE)
prompt = hub.pull(constants.AGENT_PROPMT_PULL)

# The agent can now choose between looking up facts or calculating routes
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, handle_parsing_errors=True)

set_debug(True)

async def chunked_response(query, summary_of_context=""):
    thoughts = ""

    yield json.dumps({"type": "start", "content": f"Shuka is thinking with the assumed premise of {summary_of_context}", "sequence": 0})
    output = ""
    current_mood="Thought"
    sequence_number = 1

    async for event in agent_executor.astream_events({"input": query}, version="v2"):
        kind = event["event"]
    
        # --- A. CAPTURE TOKENS (Thinking & Answer) ---
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                output += content
                # Detect transition from Thought to Final Answer
                if output.strip().endswith("Final Answer:"):
                    yield json.dumps(process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Final Answer:').strip(), event, sequence_number))
                    current_mood = "Final Answer"
                    output = ""
                elif output.strip().endswith("Thought:"):
                    yield json.dumps(process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Thought:').strip(), event, sequence_number))
                    current_mood = "Thought"
                    output = ""
                elif output.strip().endswith("Action:"):
                    yield json.dumps(process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Action:').strip(), event, sequence_number))
                    current_mood = "Action"
                    output = ""
                elif output.strip().endswith("Action Input:"):
                    yield json.dumps(process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Action Input:').strip(), event, sequence_number))
                    current_mood = "Action Input"
                    output = ""
                elif output.strip().endswith("Observation:"):
                    yield json.dumps(process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Observation:').strip(), event, sequence_number))
                    current_mood = "Observation"
                    output = ""
                else:
                    # Continue accumulating content for the current mood
                    pass

        # --- B. CAPTURE TOOL USAGE (Status Updates) ---
        elif kind == "on_tool_start":
            if not event["name"].endswith("Exception"):
                yield json.dumps(process_thought(thoughts, "status", f"Started using tool: {event['name']} with input: {event['data']['input']}", event, sequence_number))

        # --- C. CAPTURE SOURCE DATA ---
        elif kind == "on_tool_end":
            # Access the text retrieved from your ChromaDB
            if not event["name"].endswith("Exception"):
                yield json.dumps(process_thought(thoughts, "source_found", f"Data retrieved from {event['name']}.", event, sequence_number))
    
    if output.strip():
        yield json.dumps(process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip(), event, sequence_number))

    summary_of_thoughts = conversation_tools.summarize_agent_thoughts(question=query, summary_of_context=summary_of_context, thought_chain="\n".join(thoughts))

    yield json.dumps({"type": "summary", "content": summary_of_thoughts, "sequence": sequence_number + 1})

def process_thought(thoughts_tracker: str, type, content, event, sequence_number):
    """Processes the thought content and yields it in a structured format."""
    sequence_number += 1
    content = content.strip()
    thought_json = {
        "type": type,
        "content": content,
        "details": "event",
        "sequence": sequence_number
    }
    thought_str = write_thought_json_to_string(thought_json)
    if thought_str:
        thoughts_tracker += thought_str + "\n"
    return thought_json

def write_thought_json_to_string(thought_json):
    """Converts the thought JSON into a readable string format."""
    type = thought_json.get("type", "unknown")
    content = thought_json.get("content", "")
    if(content):
        return f"{type.capitalize()}: {content}"