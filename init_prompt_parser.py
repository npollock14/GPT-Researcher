from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
import os
from dotenv import load_dotenv
from enum import Enum
from utils import generate_search_results
from schemas import ResearchActionPlanSchema

INITIAL_PROMPT_PARSE_SYSTEM_PROMPT = """
Based on the given user prompt for research:

Parse it into the following:

1. Topic of Research
2. Purpose or Context
3. Primary Audience
4. Type of Paper to be Written
5. The Structure of the Paper (ex: Abstract, Introduction etc.)

Then, create a concise initial research plan. Also generate a list of Google search queries that can be used to execute the research plan.

Provide the result in the following JSON format:

{json_schema}
"""

async def parse_user_prompt_for_research(user_prompt: str, api_key: str) -> ResearchActionPlanSchema:
    gpt_json = GPTJSON[ResearchActionPlanSchema](api_key)
    
    payload = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=INITIAL_PROMPT_PARSE_SYSTEM_PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=f"Prompt: {user_prompt}",
            )
        ]
    )
    
    return payload.response