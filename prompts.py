from schemas import ResearchActionPlanSchema, SearchResultSchema

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
"""

def prepare_source_material_for_summary_prompt(search_result: SearchResultSchema):
    return f"""
Title: 
{search_result.title}

Content: 
{search_result.content}
"""