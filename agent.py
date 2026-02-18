import os
import logging
import google.cloud.logging

from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.models import Gemini
from google.genai import types
from google.adk.tools import exit_loop
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

model_name = os.getenv("MODEL")

RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

# Tools

def append_to_state(
    tool_context: ToolContext, field: str, response: str
) -> dict[str, str]:
    """Append new output to an existing state key."""
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    logging.info(f"[Added to {field}] {response}")
    return {"status": "success"}


def write_file(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    """Write content to a file."""
    target_path = os.path.join(directory, filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w") as f:
        f.write(content)
    logging.info(f"File written to {target_path}")
    return {"status": "success"}


# Agents

admirer = Agent(
    name="admirer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Researches the positive aspects and achievements of a historical figure or event.",
    instruction="""
    INSTRUCTIONS:
    - You are The Admirer. Your role is to find positive information.
    - Use your Wikipedia tool to research the achievements and positive impacts of the subject in the PROMPT.
    - Append your findings to the 'pos_data' state field using the 'append_to_state' tool.
    - Focus on terms like "achievements", "success", "positive impact", "legacy".
    - Summarize your findings.

    PROMPT: {PROMPT?}
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)

critic_researcher = Agent(
    name="critic_researcher",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Researches the negative aspects, controversies, and failures of a historical figure or event.",
    instruction="""
    INSTRUCTIONS:
    - You are The Critic. Your role is to find negative or controversial information.
    - Use your Wikipedia tool to research the controversies, failures, and negative impacts of the subject in the PROMPT.
    - Append your findings to the 'neg_data' state field using the 'append_to_state' tool.
    - Focus on terms like "controversy", "criticism", "failures", "negative impact".
    - Summarize your findings.

    PROMPT: {PROMPT?}
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        append_to_state,
    ],
)

investigation_team = ParallelAgent(
    name="investigation_team",
    description="Gathers both positive and negative information in parallel.",
    sub_agents=[admirer, critic_researcher],
)

judge = Agent(
    name="judge",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Reviews the gathered information for balance and completeness.",
    instruction="""
    INSTRUCTIONS:
    - Review the data in 'pos_data' and 'neg_data'.
    - If either field is empty or the information seems unbalanced, instruct the appropriate researcher to find more specific information. You can do this by adding feedback to the 'CRITICAL_FEEDBACK' field.
    - If the information from both sides is sufficient and balanced, use the 'exit_loop' tool to conclude the trial.
    - Explain your decision.

    POSITIVE DATA:
    {pos_data?}

    NEGATIVE DATA:
    {neg_data?}
    """,
    tools=[append_to_state, exit_loop],
)

trial_loop = LoopAgent(
    name="trial_loop",
    description="Iteratively gathers and reviews information until it is balanced.",
    sub_agents=[investigation_team, judge],
    max_iterations=3,
)

verdict_writer = Agent(
    name="verdict_writer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Summarizes the findings and writes the final verdict to a file.",
    instruction="""
    INSTRUCTIONS:
    - Create a neutral and balanced report comparing the facts from 'pos_data' and 'neg_data'.
    - Use the 'write_file' tool to save the report.
    - The directory should be 'verdicts'.
    - The filename should be based on the PROMPT, ending with '.txt'.
    - The content should be a well-structured report with sections for positive and negative aspects, followed by a concluding summary.
    - The entire report must be in Thai.

    PROMPT: {PROMPT?}
    POSITIVE DATA: {pos_data?}
    NEGATIVE DATA: {neg_data?}
    """,
    tools=[write_file],
)

historical_court_system = SequentialAgent(
    name="historical_court_system",
    description="Runs the full historical court process.",
    sub_agents=[trial_loop, verdict_writer],
)

root_agent = Agent(
    name="judge_greeter",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="Starts the historical court process by getting a topic from the user.",
    instruction="""
    - Greet the user and explain that you will help them conduct a mock trial for a historical figure or event.
    - Ask them for the topic they want to investigate.
    - When they respond, use the 'append_to_state' tool to store the user's response in the 'PROMPT' state key.
    - Then, transfer control to the 'historical_court_system' agent to begin the investigation.
    - Answer in Thai.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0,
    ),
    tools=[append_to_state],
    sub_agents=[historical_court_system],
)
