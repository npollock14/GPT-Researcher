from schemas import ResearchActionPlanSchema, ModelEnum, PaperSchema, SearchResultSummary
from prompts import get_l1_write_prompt, get_l2_write_prompt, get_l3_write_prompt
from utils import get_num_tokens

def render_writing_prompt(research_action_plan: ResearchActionPlanSchema, curr_paper: PaperSchema, curr_section: int, research: list[SearchResultSummary]):
    # get the max context limit of the writing model
    total_tokens = ModelEnum.GPT4_8K.value.max_context

    # l1 is non negotiable and must be written in full
    l1_text = get_l1_write_prompt(research_action_plan, research_action_plan.paper_structure[curr_section])
    total_tokens -= get_num_tokens(l1_text, ModelEnum.GPT4_8K)

    # Assign half of the remaining tokens to both l2 and l3
    l2_max_tokens = total_tokens // 2
    l3_max_tokens = total_tokens - l2_max_tokens  # This ensures that rounding doesn't cause us to assign more tokens than available

    # l2 gives context on previous sections already written
    l2_text = get_l2_write_prompt(research_action_plan, curr_section, curr_paper, max_tokens=l2_max_tokens)

    # Calculate how many tokens were used by l2 and adjust the max for l3
    l2_actual_tokens = get_num_tokens(l2_text, ModelEnum.GPT4_8K)
    l3_max_tokens += l2_max_tokens - l2_actual_tokens  # Add the unused tokens from l2 to l3's max

    # l3 gives source material summaries to inform the writing of the current section
    l3_text = get_l3_write_prompt(curr_section, research, max_tokens=l3_max_tokens)

    return l1_text + l2_text + l3_text
