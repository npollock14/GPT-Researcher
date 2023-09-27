from schemas import ResearchActionPlanSchema

SEARCH_CHOICE_PROMPT = """
You are a researcher helping a user choose sources that are worth reading for a {paper_type} on {topic}.

Your current research plan is to:
{research_plan}.

You will be presented with several sources that the user has found so far but you need to choose which ones are likely to assist with the {paper_type}. Choose as many or as few sources as you like. Each source listing will contain a source number. Respond with a JSON list of source numbers that you have selected.

Provide your response in the following JSON format:
{json_schema}
"""

def get_summary_prompt(research_action_plan: ResearchActionPlanSchema):
    return f"""
    TODO: Implement get_summary_prompt
    """
