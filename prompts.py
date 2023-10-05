import math
import sys
from typing import Tuple
from schemas import ResearchActionPlanSchema, SearchResultSchema, PaperSchema, ModelEnum, SearchResultSummary, SectionSchema
from utils import get_num_tokens, generate_lods
import asyncio

def get_research_summary_prompt(research_action_plan: ResearchActionPlanSchema):
    csv_sections = (", ".join(research_action_plan.paper_structure)).strip()
    return f"""
You are a researcher reading source material to create a {research_action_plan.type_of_paper_to_be_written} on {research_action_plan.topic_of_research}.

Please extract information dense key details from the following raw source material that may be useful to write the {research_action_plan.type_of_paper_to_be_written} later. 

Also try to extract an author and date from the article. If you cannot find this information, put "not found". 

Finally, estimate a relevancy score for each section of the paper, representing how relevant the source is to that section of the paper. Use a scale from 0 to 10, where 10 is extremely relevant. The sections of this paper are: {csv_sections}.

Format your answer as a list of details in the following style:
Author(s): author 1, author 2, ...
Date: date
- detail 1
- detail 2
...
- detail n
Relevancy:
- {research_action_plan.paper_structure[0]}: 5
- {research_action_plan.paper_structure[1]}: 9
...
- {research_action_plan.paper_structure[-1]}: 1

If there is nothing of value in the source material or some sort of page error, simply respond in the style:
Error: error message
"""

def prepare_source_material_for_summary_prompt(search_result: SearchResultSchema):
    return f"""
Title: 
{search_result.title}

Content: 
{search_result.content}
"""

def get_l1_write_prompt(research_action_plan: ResearchActionPlanSchema, curr_section: str):
    return f"""You are a world renowned researcher, known for your concise and information dense work. Use your knowledge as well as the provided notes to write the {curr_section} section to a {research_action_plan.type_of_paper_to_be_written} paper on: {research_action_plan.topic_of_research}. 
The purpose of the paper is: {research_action_plan.purpose_of_context}
The primary audience is: {research_action_plan.primary_audience}
The paper structure will be: {", ".join(research_action_plan.paper_structure)}
"""

def get_l2_write_prompt_text(research_action_plan: ResearchActionPlanSchema, curr_section: int, curr_paper: PaperSchema, lods: list[int]):
    prev_sections = research_action_plan.paper_structure[:curr_section]
    prev_sections_prompt = ""
    curr_idx = 0
    mentioned_text = ""
    for section in prev_sections:
        curr_lod = lods[curr_idx]
        if curr_lod is 0:
            mentioned_text = f"In the {section}, you wrote:"
        elif curr_lod is 1:
            mentioned_text = f"The key details of what you wrote in the {section} section are:"
        else:
            mentioned_text = f"The critical information from what you wrote the {section} section is:"
        prev_sections_prompt += f"""
{mentioned_text}
{curr_paper.sections[curr_idx].lods[curr_lod]}
"""
        curr_idx += 1


    full_l2 = f"""{prev_sections_prompt}
You are now in charge of writing the {research_action_plan.paper_structure[curr_section]} section. Make sure that your writing is information dense and fact based. Use markdown formatting.
""".strip()
    
    full_l2 = "\n" + full_l2 + "\n\n"
    return full_l2

LOD_FULL = 0
LOD_SUMMARY = 1
LOD_CRITICAL = 2

async def get_l2_write_prompt(research_action_plan: ResearchActionPlanSchema, curr_section: int, curr_paper: PaperSchema, max_tokens: int) -> str:
    lods = [LOD_FULL] * curr_section

    def update_l2_text():
        return get_l2_write_prompt_text(research_action_plan, curr_section, curr_paper, lods)
    
    def compute_score(i: int) -> float:
        distance = abs(i - curr_section)
        detail = lods[i]
        # if detail is already at max, return 999
        # should probably change to be a magic number
        if detail == LOD_CRITICAL:
            return 999
        # choose highest detail at the farthest distance
        return (detail + 1) / (distance ** 2 + 1)

    curr_l2_text = update_l2_text()
    max_iterations = 3 * len(curr_paper.sections)
    iteration_count = 0

    
    while get_num_tokens(curr_l2_text, ModelEnum.GPT4_8K) > max_tokens and iteration_count < max_iterations:
        iteration_count += 1
        
        # Compute scores for all sections
        scores = [compute_score(i) for i in range(len(curr_paper.sections))]
        print(f"Scores: {scores}")
        
        # Get the index of the section with the lowest score to reduce its LOD
        section_to_reduce = scores.index(min(scores))
        
        # If the section with the lowest score is already at maximum LOD, raise exception
        if lods[section_to_reduce] == LOD_CRITICAL:
            raise Exception("Cannot reduce content further to meet token limit.")
            
        lods[section_to_reduce] += 1

        # if there is only one LOD, generate the rest
        if len(curr_paper.sections[section_to_reduce].lods) == 1: 
            updated_section = await generate_lods(curr_paper.sections[section_to_reduce], research_action_plan)
            curr_paper.sections[section_to_reduce] = updated_section

        curr_l2_text = update_l2_text()

    return curr_l2_text

def get_l3_write_prompt_text(research: list[SearchResultSummary], rel_limit: int, section: str, detail_limit: int = -1):
    research_list_text = ""

    # filter out all research below the relevancy limit and those causing KeyError
    def filter_func(summary: SearchResultSummary):
        try:
            return summary.relevancy[section] >= rel_limit
        except KeyError:
            return False

    research = list(filter(filter_func, research))

    # sort by relevancy
    research.sort(key=lambda summary: summary.relevancy[section], reverse=True)

    # Handle empty research list to prevent IndexError
    if not research:
        print(f"rel_limit: {rel_limit}, section: {section}, detail_limit: {detail_limit}")
        return "No relevant notes found.", 0

    # if detail limit is not -1, limit the number of details per summary
    if detail_limit != -1:
        for summary in research:
            summary.details = summary.details[:detail_limit]

    for summary in research:
        # add a bullet point list of all details in that summary
        for detail in summary.details:
            research_list_text += f"- {detail}\n"

    

    return f"""
Notes:
{research_list_text}
""", len(research)

def get_l3_write_prompt(curr_section: str, curr_research: list[SearchResultSummary], max_tokens: int):
    # start by including all the summaries, go down from there, filtering out the least relevant ones first
    rel_limit = 0
    curr_l3_text, detail_limit = get_l3_write_prompt_text(curr_research, rel_limit, curr_section)
    # while the text is too long, increase the relevancy limit
    while get_num_tokens(curr_l3_text, ModelEnum.GPT4_8K) > max_tokens:
        rel_limit += 1
        if rel_limit > 10:
            rel_limit = 10
            # decrease the detail limit if the relevancy limit is too high
            detail_limit -= 1
            if detail_limit == 0:
                # throw an exception if the detail limit is 0
                raise Exception("Cannot reduce content further to meet token limit.")
        curr_l3_text, curr_len_details = get_l3_write_prompt_text(curr_research, rel_limit, curr_section, detail_limit)

    return curr_l3_text.strip()

def get_lod_generation_prompt(section: SectionSchema, action_plan: ResearchActionPlanSchema):
    return f"""The following is the {section.name.lower()} section of an in progress {action_plan.type_of_paper_to_be_written.lower()} on {action_plan.topic_of_research}. Create a concise bullet point summary of key details from this section. Finally, create one sentence that is a highlights the most critical information in the entire section. Format this as:
Key Details:
- detail 1
- detail 2
...
- detail n
Critical Info:
Critical info sentence here.

{section.name} Section:
{section.lods[0]}
""".strip()