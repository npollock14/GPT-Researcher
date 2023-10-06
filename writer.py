from prompt_renderer import render_writing_prompt
from schemas import ResearchActionPlanSchema, ModelEnum, PaperSchema, SearchResultSummary, SectionSchema
import asyncio
import openai
from dotenv import load_dotenv
import os

async def generate_section_content(writing_prompt: str, model: ModelEnum=ModelEnum.GPT4_8K):
    print(f"!!!Generating section content with model: {model.value.official_name}!!!")
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

async def write_paper(research_action_plan: ResearchActionPlanSchema, research: list[SearchResultSummary]):
    paper = PaperSchema()
    curr_section = 0
    ttl_sections = len(research_action_plan.paper_structure)
    while curr_section < ttl_sections:
        prompt = await render_writing_prompt(research_action_plan, paper, curr_section, research)
        print(prompt)
        raw_new_section_text = await generate_section_content(prompt)
        new_section:SectionSchema = SectionSchema(name=research_action_plan.paper_structure[curr_section], lods=[raw_new_section_text])
        print(new_section)
        paper.sections.append(new_section)
        curr_section += 1
    return paper