from dotenv import load_dotenv
from init_prompt_parser import parse_user_prompt_for_research
from schemas import ResearchActionPlanSchema, SearchResultSchema
import os
from utils import generate_search_results, summarize_results
import asyncio
import openai

if __name__ == "__main__":
    load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = "I have an upcoming meeting with Taylor Morrison. Please give me a briefing on the company before my meeting."
    action_plan:ResearchActionPlanSchema = parse_user_prompt_for_research(prompt, openai.api_key)
    search_results: list[SearchResultSchema] = generate_search_results(action_plan)
    summarized_research = asyncio.run(summarize_results(search_results,action_plan))
    

