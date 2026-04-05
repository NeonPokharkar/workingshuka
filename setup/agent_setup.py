import json
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_react_agent, create_self_ask_with_search_agent
from langchain_classic.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_classic import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.globals import set_debug
from langchain_community.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from langchain_core.output_parsers import StrOutputParser

import tools.map_tools
import tools.personalization_tools
import static.constants as constants
import tools.conversation_tools as conversation_tools
import tools.rag_tools as rag_tools

class ThoughtDetail(BaseModel):
    thought_json: dict
    thought_str: str

# Wrapping the tools with LangChain's @tool decorator for better integration

tools : list[BaseTool] = [*tools.personalization_tools.tools_list, *tools.map_tools.tool_list, *rag_tools.tools]

for tool in tools:
    tool.handle_tool_error = True

llm = ChatOllama(model=constants.PRIMARY_MODEL, request_timeout=1000, num_gpu=1, temperature=0, streaming=True)
# prompt = PromptTemplate.from_template(constants.AGENT_REACT_TEMPLATE)
prompt = hub.pull(constants.AGENT_PROPMT_PULL)

react_prompt = [
    ("system", prompt.template),
    ("system", "Summmary of Context: {summary_of_context}"),
    ("system", "Last Relevant Conversation: {last_relevant_conversation}"),
    ("system", "User's Possible Thoughts: {user_thoughts}"),
    ("human","{input}"),
    ("human", "{agent_scratchpad}")
]

summary_prompt = "You are Shuka, an expert assistant. You have access to a local database of past conversations, and many other tools. Use this information to think step by step and provide personalized answers. Extra Information: Summary of Conversation Context: {summary_of_context}, Last Relevant Conversation: {last_relevant_conversation}. User Query: {input}"

thought_prompt = "You are Shuka, an expert assistant. You have thought over person's query and considered the context. Her's your thought process : {agent_thoughts}. Please summarize what you've thought about and why you answered in the way you did. Extra Information: Summary of Conversation Context: {summary_of_context}.. User Query: {input}"

# The agent can now choose between looking up facts or calculating routes
react_agent = create_react_agent(llm, tools, ChatPromptTemplate.from_messages(react_prompt))
react_agent_executor = AgentExecutor(agent=react_agent, tools=tools, return_intermediate_steps=True, handle_parsing_errors=True)

set_debug(True)

def ask_llm_for_summary_of_thoughts(query: str, summary_of_context: str, last_relevant_conversation: str) -> str:
    """Asks the LLM to summarize the user's possible thoughts based on the conversation context and past conversations."""
    prompt_formatted = summary_prompt.format(input=query, summary_of_context=summary_of_context, last_relevant_conversation=last_relevant_conversation)
    return llm.invoke(prompt_formatted).content.strip()

def summarize_agent_thoughts(question: str, summary_of_context: str, thought_chain: str) -> str:
    """Asks the LLM to summarize the agent's thought process and reasoning based on the question, conversation context, and past conversations."""
    summary_prompt_formatted = thought_prompt.format(input=question, summary_of_context=summary_of_context, agent_thoughts=thought_chain)
    return llm.invoke(summary_prompt_formatted).content.strip()

async def chunked_response(input, summary_of_context=""):
    thoughts = ""

    yield json.dumps({"type": "start", "content": f"Shuka is thinking with the assumed premise of {summary_of_context}", "sequence": 0})
    output = ""
    current_mood="Thought"
    sequence_number = 1

    async for event in react_agent_executor.astream_events(input, version="v2"):
        kind = event["event"]
    
        # --- A. CAPTURE TOKENS (Thinking & Answer) ---
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                output += content
                # Detect transition from Thought to Final Answer
                if output.strip().endswith("Final Answer:"):
                    detail = process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Final Answer:').strip(), event, sequence_number)
                    thoughts += detail.thought_str + "\n"
                    yield json.dumps(detail.thought_json)
                    current_mood = "Final Answer"
                    output = ""
                elif output.strip().endswith("Thought:"):
                    detail = process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Thought:').strip(), event, sequence_number)
                    thoughts += detail.thought_str + "\n"
                    yield json.dumps(detail.thought_json)
                    current_mood = "Thought"
                    output = ""
                elif output.strip().endswith("Action:"):
                    detail = process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Action:').strip(), event, sequence_number)
                    thoughts += detail.thought_str + "\n"
                    yield json.dumps(detail.thought_json)
                    current_mood = "Action"
                    output = ""
                elif output.strip().endswith("Action Input:"):
                    detail = process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Action Input:').strip(), event, sequence_number)
                    thoughts += detail.thought_str + "\n"
                    yield json.dumps(detail.thought_json)
                    current_mood = "Action Input"
                    output = ""
                elif output.strip().endswith("Observation:"):
                    detail = process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip('Observation:').strip(), event, sequence_number)
                    thoughts += detail.thought_str + "\n"
                    yield json.dumps(detail.thought_json)
                    current_mood = "Observation"
                    output = ""
                else:
                    # Continue accumulating content for the current mood
                    pass

        # --- B. CAPTURE TOOL USAGE (Status Updates) ---
        elif kind == "on_tool_start":
            if not event["name"].endswith("Exception"):
                detail = process_thought(thoughts, "status", f"Started using tool: {event['name']} with input: {event['data']['input']}", event, sequence_number)
                thoughts += detail.thought_str + "\n"
                yield json.dumps(detail.thought_json)

        # --- C. CAPTURE SOURCE DATA ---
        elif kind == "on_tool_end":
            # Access the text retrieved from your ChromaDB
            if not event["name"].endswith("Exception"):
                detail = process_thought(thoughts, "source_found", f"Data retrieved from {event['name']}.", event, sequence_number)
                thoughts += detail.thought_str + "\n"
                yield json.dumps(detail.thought_json)

    if output.strip():
        detail = process_thought(thoughts, current_mood.lower().replace(" ", "_"), output.strip(), event, sequence_number)
        thoughts += detail.thought_str + "\n"
        yield json.dumps(detail.thought_json)
    
    print("Final Thought Chain:\n", thoughts)

    summary_of_thoughts = summarize_agent_thoughts(input["input"], summary_of_context, thoughts.join("\n"))

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
    thought_str = f"{type.capitalize()}: {content}"
    return ThoughtDetail(thought_json=thought_json, thought_str=thought_str)