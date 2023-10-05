from prompt_renderer import render_writing_prompt
from schemas import ResearchActionPlanSchema, ModelEnum, PaperSchema, SearchResultSummary
import asyncio
import openai
from dotenv import load_dotenv
import os

async def generate_section_content(writing_prompt: str, model: ModelEnum=ModelEnum.GPT4_8K):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    completion = await openai.ChatCompletion.acreate(
        model=model.value.official_name,
        messages=[
            {
                "role": "user",
                "content": writing_prompt
            },
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    new_section: str = completion.choices[0].message.content
    return new_section

def write_paper(research_action_plan: ResearchActionPlanSchema, research: list[SearchResultSummary]):
    curr_paper = PaperSchema()
    curr_section = 0
    while curr_section < len(research_action_plan.paper_structure):
        prompt, curr_paper = render_writing_prompt(research_action_plan, curr_paper, curr_section, research)
        new_section = asyncio.run(generate_section_content(prompt))
        curr_paper.sections.append(new_section)
        print(prompt)
        curr_section += 1
    return curr_paper