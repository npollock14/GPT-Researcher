from utils import parse_raw_summary, load_research_action_plan, load_summary
from schemas import SearchResultSummary, SearchResultSchema, ModelEnum, ResearchActionPlanSchema

def test_parse_valid_summary():
    raw_summary = """
    Author(s): John Doe, Jane Smith
    Date: 2023-09-29
    - Detail 1
    - Detail 2
    Relevancy:
    - Section1: 5
    - Section2: 3
    """
    search_result: SearchResultSchema = SearchResultSchema(
        title="Test Title",
        link="https://www.google.com",
        content="Test Content",
        cost=0.1,
        model=ModelEnum.GPT3_5_TURBO_16K
    )
    result:SearchResultSummary = parse_raw_summary(raw_summary, search_result)
    assert result.source_material.title == "Test Title"
    assert result.details == ["Detail 1", "Detail 2"]
    assert result.relevancy == {"Section1": 5, "Section2": 3}
    assert result.error is None

def test_parse_missing_sections():
    raw_summary = """
    Title: Test Title
    Author(s): John Doe, Jane Smith
    Date: 2023-09-29
    """
    search_result: SearchResultSchema = SearchResultSchema(
        title="Test Title",
        link="https://www.google.com",
        content="Test Content",
        cost=0.1,
        model=ModelEnum.GPT3_5_TURBO_16K
    )
    result = parse_raw_summary(raw_summary, search_result)
    assert result.error == "Details section is missing or empty. Relevancy section is missing or empty."


def test_load_summary():
    # summary file is ./tests/test_data/summary.json
    # looks like:
    """
    [
    {
        "source_material": {
            "title": "Taylor Morrison Home Corp. - About - Company Profile",
            "link": "https://investors.taylormorrison.com/about/Company-Profile/default.aspx",
            "content": "Who We Are\nBased in Scottsdale, Arizona, Taylor Morrison is the one of the nation\u2019s leading homebuilders and developers with operations across 19 markets in 11 states.\nWe serve a wide array of consumers from coast to coast, including first-time, move-up, luxury and resort lifestyle homebuyers and renters under our family of brands\u2014including Taylor Morrison, Esplanade, Darling Homes Collection by Taylor Morrison and Yardly.\nWith a company legacy dating back over 100 years, we are committed to developing sustainable communities in which our homebuyers aspire to live while generating long-term value for our shareholders and other stakeholders. This commitment has helped us earn the reputation as America\u2019s Most Trusted\u00ae Builder by Lifestory Research from 2016-2023.",
            "cost": 0.001255,
            "model": "gpt-3.5-turbo"
        },
        "details": [
            "Taylor Morrison is based in Scottsdale, Arizona.",
            "Taylor Morrison is one of the nation's leading homebuilders and developers.",
            "Taylor Morrison operates in 19 markets across 11 states.",
            "Taylor Morrison serves a wide range of consumers, including first-time, move-up, luxury, and resort lifestyle homebuyers and renters.",
            "Taylor Morrison has a family of brands, including Taylor Morrison, Esplanade, Darling Homes Collection by Taylor Morrison, and Yardly.",
            "Taylor Morrison has a company legacy dating back over 100 years.",
            "Taylor Morrison is committed to developing sustainable communities and generating long-term value for shareholders and stakeholders.",
            "Taylor Morrison has been recognized as America's Most Trusted\u00ae Builder by Lifestory Research from 2016-2023."
        ],
        "authors": [],
        "date": "",
        "relevancy": {
            "Introduction": 7,
            "Company Overview": 10,
            "Recent Developments": 3,
            "Financial Performance": 2,
            "Strategic Initiatives": 8,
            "Conclusion": 1
        },
        "error": null
    },
    ...
    ]
    """
    summary = load_summary("./tests/test_data/summary.json")
    # len = 12
    assert len(summary) == 12
    # first item:
    first_item = summary[0]
    assert first_item.source_material.title == "Taylor Morrison Home Corp. - About - Company Profile"
    assert first_item.source_material.link == "https://investors.taylormorrison.com/about/Company-Profile/default.aspx"
    assert first_item.source_material.content == "Who We Are\nBased in Scottsdale, Arizona, Taylor Morrison is the one of the nation\u2019s leading homebuilders and developers with operations across 19 markets in 11 states.\nWe serve a wide array of consumers from coast to coast, including first-time, move-up, luxury and resort lifestyle homebuyers and renters under our family of brands\u2014including Taylor Morrison, Esplanade, Darling Homes Collection by Taylor Morrison and Yardly.\nWith a company legacy dating back over 100 years, we are committed to developing sustainable communities in which our homebuyers aspire to live while generating long-term value for our shareholders and other stakeholders. This commitment has helped us earn the reputation as America\u2019s Most Trusted\u00ae Builder by Lifestory Research from 2016-2023."
    assert first_item.source_material.cost == 0.001255
    assert first_item.source_material.model == ModelEnum.GPT3_5_TURBO_4K
    assert first_item.details == [
        "Taylor Morrison is based in Scottsdale, Arizona.",
        "Taylor Morrison is one of the nation's leading homebuilders and developers.",
        "Taylor Morrison operates in 19 markets across 11 states.",
        "Taylor Morrison serves a wide range of consumers, including first-time, move-up, luxury, and resort lifestyle homebuyers and renters.",
        "Taylor Morrison has a family of brands, including Taylor Morrison, Esplanade, Darling Homes Collection by Taylor Morrison, and Yardly.",
        "Taylor Morrison has a company legacy dating back over 100 years.",
        "Taylor Morrison is committed to developing sustainable communities and generating long-term value for shareholders and stakeholders.",
        "Taylor Morrison has been recognized as America's Most Trusted\u00ae Builder by Lifestory Research from 2016-2023."
    ]
    assert first_item.authors == []
    assert first_item.date == ""
    assert first_item.relevancy == {
        "Introduction": 7,
        "Company Overview": 10,
        "Recent Developments": 3,
        "Financial Performance": 2,
        "Strategic Initiatives": 8,
        "Conclusion": 1
    }
    assert first_item.error is None
    
def test_load_research_action_plan():
    research_plan: ResearchActionPlanSchema = load_research_action_plan("./tests/test_data/parsed_user_prompt_for_research.json")
    # looks like:
    """
    {
    "topic_of_research": "Taylor Morrison",
    "purpose_of_context": "To prepare for a business meeting with Taylor Morrison",
    "primary_audience": "Business professionals",
    "type_of_paper_to_be_written": "Business Briefing",
    "paper_structure": [
        "Introduction",
        "Company Overview",
        "Recent Developments",
        "Financial Performance",
        "Strategic Initiatives",
        "Conclusion"
    ],
    "research_plan": [
        "Research the history and background of Taylor Morrison",
        "Investigate recent developments and news related to Taylor Morrison",
        "Analyze the financial performance of Taylor Morrison",
        "Understand the strategic initiatives and future plans of Taylor Morrison"
    ],
    "search_queries": [
        "Taylor Morrison company profile",
        "Taylor Morrison recent news",
        "Taylor Morrison financial performance",
        "Taylor Morrison strategic initiatives",
        "Taylor Morrison future plans"
    ]
}
    """
    assert research_plan.topic_of_research == "Taylor Morrison"
    assert research_plan.purpose_of_context == "To prepare for a business meeting with Taylor Morrison"
    assert research_plan.primary_audience == "Business professionals"
    assert research_plan.type_of_paper_to_be_written == "Business Briefing"
    assert research_plan.paper_structure == [
        "Introduction",
        "Company Overview",
        "Recent Developments",
        "Financial Performance",
        "Strategic Initiatives",
        "Conclusion"
    ]
    assert research_plan.research_plan == [
        "Research the history and background of Taylor Morrison",
        "Investigate recent developments and news related to Taylor Morrison",
        "Analyze the financial performance of Taylor Morrison",
        "Understand the strategic initiatives and future plans of Taylor Morrison"
    ]
    assert research_plan.search_queries == [
        "Taylor Morrison company profile",
        "Taylor Morrison recent news",
        "Taylor Morrison financial performance",
        "Taylor Morrison strategic initiatives",
        "Taylor Morrison future plans"
    ]
    