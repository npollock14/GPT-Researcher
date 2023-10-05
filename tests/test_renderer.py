from schemas import ResearchActionPlanSchema, PaperSchema, SearchResultSchema, str_to_model_enum, SearchResultSummary, SectionSchema, ModelEnum
from utils import load_summary, load_research_action_plan, fits_in_model, get_num_tokens
from prompt_renderer import render_writing_prompt
import asyncio


def test_render_writing_prompt():
    # load test data:
    # research_action_plan: ResearchActionPlanSchema, curr_paper: PaperSchema, curr_section: int, research: list[SearchResultSummary]
    parsed_user_prompt_output_file = "./tests/test_data/parsed_user_prompt_for_research.json"
    summary_output_file = "./tests/test_data/summary.json"
    action_plan: ResearchActionPlanSchema = load_research_action_plan(parsed_user_prompt_output_file)
    summaries: list[SearchResultSummary] = load_summary(summary_output_file)
    curr_paper: PaperSchema = PaperSchema(sections=[])

    # sections will be "Introduction, Company Overview, Recent Developments, Financial Performance, Strategic Initiatives, Conclusion"

    # render the writing prompt:
    # writing_prompt = asyncio.run(render_writing_prompt(action_plan, curr_paper, 0, summaries))
    # print("=== Writing Prompt ===")
    # print(writing_prompt)
    # print("=== End Writing Prompt ===")

    # check that the writing prompt is correct:

    
    # make sure writing prompt is less than tokens for gpt-4-8k
    # assert fits_in_model(writing_prompt, ModelEnum.GPT4_8K)
    # now test the second section:
    long_intro_text = ""
    for i in range(0, 200):
        long_intro_text += "This is verbose context. "
    summary_text = "This is key detail context. "
    for i in range(0, 50):
        summary_text += "This is key detail context. "
    critical_text = "This is critical context. "
    for i in range(0, 5):
        critical_text += "This is key detail context. "

    curr_paper = PaperSchema(sections=[SectionSchema(name="Introduction", lods=[long_intro_text, summary_text, critical_text]), SectionSchema(name="Company Overview", lods=[long_intro_text, summary_text, critical_text]), SectionSchema(name="Recent Developments", lods=[long_intro_text, summary_text, critical_text]), SectionSchema(name="Financial Performance", lods=[long_intro_text, summary_text, critical_text]), SectionSchema(name="Strategic Initiatives", lods=[long_intro_text, summary_text, critical_text])])

    # writing_prompt = asyncio.run(render_writing_prompt(action_plan, curr_paper, 5, summaries, total_tokens= 4000))
    # print("=== Writing Prompt ===")
    # print(writing_prompt)
    # print("=== End Writing Prompt ===")
    # print(f"Number of tokens: {get_num_tokens(writing_prompt, ModelEnum.GPT4_8K)}")

    # load example text from file
    example_intro_full: str = ""
    with open("./tests/test_data/example_intro_full.txt", "r") as f:
        example_intro_full = f.read()

    curr_paper = PaperSchema(sections=[SectionSchema(name="Introduction", lods=[example_intro_full])])
    writing_prompt = asyncio.run(render_writing_prompt(action_plan, curr_paper, 1, summaries, total_tokens= 700))

    print("=== Writing Prompt ===")
    print(writing_prompt)
    print("=== End Writing Prompt ===")
    print(f"Number of tokens: {get_num_tokens(writing_prompt, ModelEnum.GPT4_8K)}")

    # save the paper to file
    with open('./outputs/paper.json', 'w') as outfile:
        import json
        json.dump(curr_paper.to_json(), outfile, indent=4)



    