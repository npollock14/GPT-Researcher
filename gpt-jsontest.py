import asyncio
from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
import json
from enum import Enum


class ResearchOutputSchema(BaseModel):
    topic_of_research: str
    purpose_of_context: str
    primary_audience: str
    type_of_paper_to_be_written: str
    paper_structure: list[str] 
    research_plan: list[str]
    search_queries: list[str]




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
    gpt_json = GPTJSON[ResearchOutputSchema](api_key)
    
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
            result = ResearchOutputSchema.model_validate_json(result_data) # This parses the raw JSON data back into the model
    else:
        result = await parse_user_prompt_for_research(user_prompt, API_KEY)
        # save result to a file parsed_user_prompt_for_research.json
        with open('./outputs/parsed_user_prompt_for_research.json', 'w') as outfile:
            outfile.write(result.model_dump_json()) # Serialize the instance to JSON

    # now, for each query in the search_queries list:
        # make a google search for that query
        # ask gpt to pick what results it would like to explore
        # for those picked results, summarize them in context to the search and current action plan
    

if __name__ == "__main__":
    asyncio.run(main())
