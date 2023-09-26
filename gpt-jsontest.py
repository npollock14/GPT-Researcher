import asyncio
from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
import os
from dotenv import load_dotenv
import json
from enum import Enum
from utils import generate_search_results
from schemas import ResearchActionPlanSchema






SYSTEM_PROMPT = """
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

async def parse_user_prompt_for_research(user_prompt: str, api_key: str):
    gpt_json = GPTJSON[ResearchActionPlanSchema](api_key)
    
    payload = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=f"Prompt: {user_prompt}",
            )
        ]
    )
    
    return payload.response

async def main():
    load_dotenv()

    API_KEY = os.getenv("OPENAI_API_KEY")
    user_prompt = "I have an upcoming business meeting with Taylor Morrison. Create a business focused briefing for me."
    # check if response has been cached
    if os.path.exists('./outputs/parsed_user_prompt_for_research.json'):
        with open('./outputs/parsed_user_prompt_for_research.json') as json_file:
            result_data = json_file.read()
            result = ResearchActionPlanSchema.model_validate_json(result_data) # This parses the raw JSON data back into the model
    else:
        result = await parse_user_prompt_for_research(user_prompt, API_KEY)
        # save result to a file parsed_user_prompt_for_research.json
        with open('./outputs/parsed_user_prompt_for_research.json', 'w') as outfile:
            outfile.write(result.model_dump_json()) # Serialize the instance to JSON

    search_results = generate_search_results(result)
    
    

if __name__ == "__main__":
    asyncio.run(main())
