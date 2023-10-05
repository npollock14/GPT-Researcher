from utils import parse_raw_summary, load_research_action_plan, load_summary, parse_raw_lod, generate_lods
from schemas import SearchResultSummary, SearchResultSchema, ModelEnum, ResearchActionPlanSchema, SectionSchema
import asyncio

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
    

def test_parse_raw_lod():
    section = SectionSchema(
        name="Test Title",
        lods=["This is full content"]
    )
    ai_response = """Key Details:
- Taylor Morrison reported a Q2 2023 net income of $235 million.
- Home closings revenue reached $2.0 billion with a 3% increase in closings.
- Net sales orders increased by 18%.
- Moody's upgraded the company's credit rating to Ba2.
- Taylor Morrison released its fifth annual ESG report, focusing on sustainability and diversity.
- The company received corporate recognitions, including Newsweek's Most Responsible Companies list.
- Taylor Morrison appointed a Corporate Director of Sustainability and advanced its DEIB strategy.
- The company collaborated with the National Wildlife Federation for environmental stewardship.

Critical Info:
Taylor Morrison reported strong Q2 2023 financial performance with increased net income and home closings revenue. Additionally, the company's commitment to sustainability and diversity, as highlighted in its ESG report and corporate recognitions, underscores its position as a forward-thinking and responsible homebuilder."""

    section = parse_raw_lod(ai_response, section)

    assert section.lods == [
        "This is full content",
        "- Taylor Morrison reported a Q2 2023 net income of $235 million.\n- Home closings revenue reached $2.0 billion with a 3% increase in closings.\n- Net sales orders increased by 18%.\n- Moody's upgraded the company's credit rating to Ba2.\n- Taylor Morrison released its fifth annual ESG report, focusing on sustainability and diversity.\n- The company received corporate recognitions, including Newsweek's Most Responsible Companies list.\n- Taylor Morrison appointed a Corporate Director of Sustainability and advanced its DEIB strategy.\n- The company collaborated with the National Wildlife Federation for environmental stewardship.",
        "Taylor Morrison reported strong Q2 2023 financial performance with increased net income and home closings revenue. Additionally, the company's commitment to sustainability and diversity, as highlighted in its ESG report and corporate recognitions, underscores its position as a forward-thinking and responsible homebuilder."
    ]

def test_generate_lods():
    section = SectionSchema(
        name="Recent Developments",
        lods=["""## Recent Developments

1. **Financial Performance Highlights**:
   - Taylor Morrison reported a net income of $235 million, equivalent to $2.12 per diluted share, in Q2 2023.
   - The company witnessed an increase of 3% in closings, amounting to 3,125 homes, at an average price of $639,000, resulting in home closings revenue of $2.0 billion.
   - Despite a YoY decline of 240 basis points, home closings gross margin rose 30 basis points sequentially, settling at 24.2%.
   - A notable 18% increase was observed in net sales orders, totaling 3,023, propelled by a monthly absorption rate of 3.1 per community.
   - The quarter ended with around 72,000 homebuilding lots owned and controlled, equivalent to 5.8 years of total supply.
   - Total liquidity achieved an unprecedented high, touching $2.3 billion, and homebuilding debt-to-capitalization dropped to 29.7% gross, and 15.4% net of $1.2 billion unrestricted cash.
   - Moody's upgraded the company's credit rating to Ba2 from Ba3, maintaining a Stable outlook.
   - The book value per share surged 30% to reach $45.96.

2. **Sustainability and ESG Commitment**:
   - Taylor Morrison released its fifth annual Environmental, Social, and Governance (ESG) Report. The report encapsulates three central pillars: Building for the Future, People First, and Transparency and Accountability.
   - Significant milestones encompassed an initial review of the company's greenhouse gas inventory, alignment with the Task Force on Climate-Related Financial Disclosures, an innovative Board Fellowship Program to bolster under-represented business leaders, and renewed focus on diverse homebuyers. 
   - The company fortified its partnership with the National Wildlife Federation and met disclosure standards as per the Sustainability Accounting Standards Board (SASB), Global Reporting Initiative (GRI), and Task Force on Climate-Related Financial Disclosures.

3. **Corporate Recognitions**:
   - Taylor Morrison's dedication to corporate responsibility secured its place on Newsweek's 2023 America's Most Responsible Companies list, excelling in corporate governance and social categories.
   - The company marked its eighth consecutive year as America's Most Trusted® Home Builder by Lifestory Research.
   - Other accolades included features on Wall Street Journal's 2022 Management Top 250, Bloomberg Gender-Equality Index (GEI), and the Fortune 500 list.

4. **Strategic Appointments and Initiatives**:
   - Taylor Morrison announced the appointment of its inaugural Corporate Director of Sustainability.
   - Progressing its diversity, equity, inclusion, and belonging (DEIB) strategy, the company established a majority-diverse board of directors along with other pivotal initiatives.

5. **Community and Partnership Developments**:
   - Taylor Morrison has underlined its dedication to environmental and community stewardship by collaborating with the National Wildlife Federation to rejuvenate and safeguard wildlife habitats.

These recent developments reinforce Taylor Morrison's position as a leading homebuilder with a commitment to sustainable growth, corporate responsibility, and financial robustness. The company continues to be forward-thinking, embedding ESG initiatives into its core business strategies and bolstering its market position."""]
    )
    research_action_plan: ResearchActionPlanSchema = load_research_action_plan("./tests/test_data/parsed_user_prompt_for_research.json")
    section = asyncio.run(generate_lods(section, research_action_plan))
    for lod in section.lods:
        print(lod)
        print()