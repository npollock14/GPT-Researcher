from schemas import ResearchActionPlanSchema, PaperSchema, SearchResultSchema, str_to_model_enum, SearchResultSummary, SectionSchema, ModelEnum
from utils import load_summary, load_research_action_plan, fits_in_model, get_num_tokens
from prompt_renderer import render_writing_prompt
import asyncio
from writer import write_paper

def test_write_paper():
    # load test data:
    # research_action_plan: ResearchActionPlanSchema, curr_paper: PaperSchema, curr_section: int, research: list[SearchResultSummary]
    parsed_user_prompt_output_file = "./tests/test_data/parsed_user_prompt_for_research.json"
    summary_output_file = "./tests/test_data/summary.json"
    action_plan: ResearchActionPlanSchema = load_research_action_plan(parsed_user_prompt_output_file)
    summaries: list[SearchResultSummary] = load_summary(summary_output_file)

    paper: PaperSchema = asyncio.run(write_paper(action_plan, summaries))
    # save the paper to a file in outputs
    # save the paper to file
    with open('./outputs/paper.json', 'w') as outfile:
        import json
        json.dump(paper.to_json(), outfile, indent=4)