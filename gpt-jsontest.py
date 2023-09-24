import asyncio
from gpt_json import GPTJSON, GPTMessage, GPTMessageRole
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv


class ResearchOutputSchema(BaseModel):
    topic_of_research: str
    purpose_of_context: str
    primary_audience: str
    research_plan: list[str]
    search_queries: list[str]

SYSTEM_PROMPT = """
Based on the given user prompt for research:

Parse it into the following:

1. Topic of Research
2. Purpose or Context
3. Primary Audience

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
    user_prompt = "I have an upcoming business meeting with Deloitte. Create a business focused briefing for me."
    
    result = await parse_user_prompt_for_research(user_prompt, API_KEY)
    print(f"Topic: {result.topic_of_research}")
    print(f"Context: {result.purpose_of_context}")
    print(f"Audience: {result.primary_audience}")
    print("Research Plan:")
    for point in result.research_plan:
        print(f"- {point}")
    print("Search Queries:")
    for query in result.search_queries:
        print(f"- {query}")

if __name__ == "__main__":
    asyncio.run(main())
