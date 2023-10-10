import logging
from dotenv import load_dotenv
from init_prompt_parser import parse_user_prompt_for_research
from schemas import ResearchActionPlanSchema, PaperSchema, SearchResultSchema, str_to_model_enum, SearchResultSummary, SectionSchema, ModelEnum
import os
from utils import generate_search_results, summarize_results, convert_paper_to_pdf
import asyncio
import openai
from writer import write_paper

# 1. Set up the logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler('./logs/detail.log')])  # Save logs to 'logs/detail.log'
logger = logging.getLogger()

def main(prompt, **kwargs):
    load_dotenv()

    if not prompt:
        logger.error("Prompt is required")  # Logging error
        raise Exception("Prompt is required")
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        logger.error("OPENAI_API_KEY is required")  # Logging error
        raise Exception("OPENAI_API_KEY is required")

    logger.info("Parsing the user prompt for research...")  # User feedback
    action_plan:ResearchActionPlanSchema = asyncio.run(parse_user_prompt_for_research(prompt, openai.api_key))

    logger.info("Generating search results...")
    search_results: list[SearchResultSchema] = generate_search_results(action_plan)

    logger.info("Summarizing research results...")
    summarized_research = asyncio.run(summarize_results(search_results, action_plan))
    
    logger.info("Writing the research paper...")
    paper: PaperSchema = asyncio.run(write_paper(action_plan, summarized_research, context_limit=kwargs.get("context_limit", ModelEnum.GPT4_8K.value.max_context)))

    # save the paper to a file in outputs
    logger.info("Saving raw paper to './outputs/paper.json'...")
    with open('./outputs/paper.json', 'w') as outfile:
        import json
        json.dump(paper.to_json(), outfile, indent=4)

    # convert the json paper to a pdf
    logger.info("Converting to pdf...")
    convert_paper_to_pdf(paper)

    logger.info("Operation completed successfully!")

if __name__ == "__main__":
    prompt = "Please write a paper introducing me as a Software Engineer what HuggingFace is and its applications."
    main(prompt, context_limit=4000)
