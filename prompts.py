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
            mentioned_text = "wrote"
        else:
            mentioned_text = "have talked about"
        prev_sections_prompt += f"""
So far in the {section}, you {mentioned_text}:
{curr_paper.sections[curr_idx].lods[curr_lod]}
"""
        curr_idx += 1


    full_l2 = f"""{prev_sections_prompt}
You are now in charge of writing the {research_action_plan.paper_structure[curr_section]} section. Make sure that your writing is information dense and fact based. Use markdown formatting.
""".strip()
    
    full_l2 = "\n" + full_l2 + "\n"
    return full_l2

def get_l2_write_prompt(research_action_plan: ResearchActionPlanSchema, curr_section: int, curr_paper: PaperSchema, max_tokens: int):
    # initialize an array of lods at maximum detail (0)
    lods = [0] * curr_section
    curr_l2_text = get_l2_write_prompt_text(research_action_plan, curr_section, curr_paper, lods)
    lowest_lod = 0
    # while the text is too long, increase the lod of the lowest lod section, starting from index 0
    while get_num_tokens(curr_l2_text, ModelEnum.GPT4_8K) > max_tokens:
        min_lod = min(lods)
        lowest_lod = lods.index(min_lod)
        lods[lowest_lod] += 1
        # check if this is a valid configuration
        if len(curr_paper.sections[lowest_lod].lods) == 1: # only gen lods when only 1 lod exists
            # generate lods for this section
            updated_section = asyncio.run(generate_lods(curr_paper.sections[lowest_lod], research_action_plan))
            curr_paper.sections[lowest_lod] = updated_section
            pass
        elif len(curr_paper.sections[lowest_lod].lods) <= lods[lowest_lod]:
            raise Exception("Out of bounds LOD")
        curr_l2_text = get_l2_write_prompt_text(research_action_plan, curr_section, curr_paper, lods)

    return curr_l2_text.strip() + "\n\n", curr_paper

def get_l3_write_prompt_text(research: list[SearchResultSummary], rel_limit: int, section: str):
    research_list_text = ""

    # filter out all research below the relevancy limit and those causing KeyError
    def filter_func(summary: SearchResultSummary):
        try:
            return summary.relevancy[section] >= rel_limit
        except KeyError:
            return False

    research = list(filter(filter_func, research))

    for summary in research:
        # add a bullet point list of all details in that summary
        for detail in summary.details:
            research_list_text += f"- {detail}\n"

    # Handle empty research list to prevent IndexError
    if not research:
        return "No relevant notes found."

    return f"""
Notes:
{research_list_text}
"""

def get_l3_write_prompt(curr_section: str, curr_research: list[SearchResultSummary], max_tokens: int):
    # start by including all the summaries, go down from there, filtering out the least relevant ones first
    rel_limit = 0
    curr_l3_text = get_l3_write_prompt_text(curr_research, rel_limit, curr_section)
    # while the text is too long, increase the relevancy limit
    while get_num_tokens(curr_l3_text, ModelEnum.GPT4_8K) > max_tokens:
        rel_limit += 1
        if rel_limit > 10:
            raise Exception("Relevancy limit exceeded")
        curr_l3_text = get_l3_write_prompt_text(curr_research, rel_limit, curr_section)

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