from schemas import ResearchActionPlanSchema, SearchResultSchema, PaperSchema, ModelEnum
from utils import fits_in_model

def get_summary_prompt(research_action_plan: ResearchActionPlanSchema):
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

def get_l1_write_prompt(research_action_plan: ResearchActionPlanSchema):
    return f"""
You are a world renowned researcher, known for your concise and information dense work. Use your knowledge as well as the provided notes to write the company overview section to a business briefing paper on: {research_action_plan.topic_of_research}. 
The purpose of the paper is: {research_action_plan.purpose_of_context}
The primary audience is: {research_action_plan.primary_audience}
The paper structure will be:
{", ".join(research_action_plan.paper_structure)}

"""

def get_l2_write_prompt_text(research_action_plan: ResearchActionPlanSchema, curr_section: int, curr_paper: PaperSchema, lods: list[int]):
    prev_sections = research_action_plan.paper_structure[:curr_section]
    prev_sections_prompt = ""
    curr_idx = 0
    mentioned_text = ""
    for section in prev_sections:
        if curr_idx is 0:
            mentioned_text = "wrote"
        else:
            mentioned_text = "have talked about"
        prev_sections_prompt += f"""
So far in the {section}, you {mentioned_text}:
{curr_paper.sections[curr_idx]}
"""
        curr_idx += 1


    return f"""
    {prev_sections_prompt}

You are now in charge of writing the company overview section. Make sure that your writing is information dense and fact based. Use markdown formatting.

"""

def get_l2_write_prompt(research_action_plan: ResearchActionPlanSchema, curr_section: int, curr_paper: PaperSchema, max_tokens: int):
    # initialize an array of lods at maximum detail (0)
    if(curr_section == 0):
        return get_l2_write_prompt_text(research_action_plan, curr_section, curr_paper, [0] * len(research_action_plan.paper_structure))
    lods = [0] * curr_section
    lowest_lod = 0
    while not fits_in_model(get_l2_write_prompt_text(research_action_plan, curr_section, curr_paper, lods), ModelEnum.GPT4_8K):
        lods[curr_section - 1] += 1
