from schemas import ResearchActionPlanSchema, PaperSchema, SearchResultSchema, str_to_model_enum, SearchResultSummary, SectionSchema, ModelEnum
from utils import load_summary, load_research_action_plan, fits_in_model, get_num_tokens
from prompt_renderer import render_writing_prompt


def test_render_writing_prompt():
    # load test data:
    # research_action_plan: ResearchActionPlanSchema, curr_paper: PaperSchema, curr_section: int, research: list[SearchResultSummary]
    parsed_user_prompt_output_file = "./tests/test_data/parsed_user_prompt_for_research.json"
    summary_output_file = "./tests/test_data/summary.json"
    action_plan: ResearchActionPlanSchema = load_research_action_plan(parsed_user_prompt_output_file)
    summaries: list[SearchResultSummary] = load_summary(summary_output_file)
    curr_paper: PaperSchema = PaperSchema(sections=[])

    # render the writing prompt:
    writing_prompt = render_writing_prompt(action_plan, curr_paper, 0, summaries)
    # print("=== Writing Prompt ===")
    # print(writing_prompt)
    # print("=== End Writing Prompt ===")

    # check that the writing prompt is correct:
    assert writing_prompt == """You are a world renowned researcher, known for your concise and information dense work. Use your knowledge as well as the provided notes to write the Introduction section to a Business Briefing paper on: Taylor Morrison. 
The purpose of the paper is: To prepare for a business meeting with Taylor Morrison
The primary audience is: Business professionals
The paper structure will be: Introduction, Company Overview, Recent Developments, Financial Performance, Strategic Initiatives, Conclusion
You are now in charge of writing the Introduction section. Make sure that your writing is information dense and fact based. Use markdown formatting.

Notes:
- Taylor Morrison is based in Scottsdale, Arizona.
- Taylor Morrison is one of the nation's leading homebuilders and developers.
- Taylor Morrison operates in 19 markets across 11 states.
- Taylor Morrison serves a wide range of consumers, including first-time, move-up, luxury, and resort lifestyle homebuyers and renters.
- Taylor Morrison has a family of brands, including Taylor Morrison, Esplanade, Darling Homes Collection by Taylor Morrison, and Yardly.
- Taylor Morrison has a company legacy dating back over 100 years.
- Taylor Morrison is committed to developing sustainable communities and generating long-term value for shareholders and stakeholders.
- Taylor Morrison has been recognized as America's Most Trusted® Builder by Lifestory Research from 2016-2023.
- Taylor Morrison is a homebuilder and developer.
- It designs, builds, and sells single-family and multi-family detached and attached homes.
- It provides residential homebuilding and the development of lifestyle communities across 11 states.
- The company provides financial services to customers through its mortgage subsidiary.
- It also offers title insurance and closing settlement services through its title company, Inspired Title Services, LLC.
- Taylor Morrison operates under the Taylor Morrison and Darling Homes brand names.
- It operates a Build to Rent homebuilding business under the brand, Yardly.
- Taylor Morrison operates in Arizona, Nevada, California, Oregon, Colorado, Florida, Georgia, North and South Carolina, Washington, and Texas.
- Taylor Morrison is headquartered in Scottsdale, Arizona, the US.
- Taylor Morrison is one of the largest home building companies in the United States
- Corporate headquarters are in Scottsdale, Arizona
- Taylor Morrison formed when Taylor Woodrow and Morrison Homes merged in July 2007
- Taylor Morrison operates in Arizona, California, Colorado, Georgia, Florida, Illinois, North Carolina, South Carolina, Nevada, and Texas
- Taylor Morrison builds mid-to-upscale housing, as well as first-time and mid-market homes
- Taylor Woodrow was founded in 1921 by Frank Taylor and Jack Woodrow in England
- Taylor Woodrow initially focused on providing low-cost, high-quality housing in the Lancashire area
- In the 1930s, Taylor Woodrow diversified into general construction and entered the Canadian construction market in 1936
- Morrison Homes was founded in Seattle in 1905 by C.G. Morrison and later moved to northern California
- Morrison Homes built primarily first-time and midmarket home communities in multiple cities
- George Wimpey Plc acquired Morrison Homes in 1984 and later integrated Richardson Homes under the Morrison brand
- In July 2007, Taylor Woodrow Plc and George Wimpey Plc formed a new company, Taylor Wimpey Plc, which included Morrison Homes
- Taylor Morrison began operating under its new brand in 2008
- In 2011, Taylor Morrison became a wholly owned subsidiary of TMM Holdings Limited Partnership, which is owned by investment funds managed by TPG Capital, Oaktree Capital Management, and JH Investments
- Taylor Morrison went public in 2013
- Taylor Morrison is based in Scottsdale, Arizona.
- Taylor Morrison is one of the country's largest homebuilders and developers.
- Sheryl Palmer is the CEO of Taylor Morrison, and she is the only female CEO of a publicly traded homebuilder.
- Taylor Morrison operates in 19 markets across 11 states.
- Taylor Morrison serves various types of consumers, including first-time, move-up, luxury, and resort lifestyle homebuyers and renters.
- Taylor Morrison is based in Scottsdale, Arizona.
- Taylor Morrison is one of the country's largest homebuilders and developers.
- Sheryl Palmer is the CEO of Taylor Morrison.
- Taylor Morrison serves consumers across 19 markets in 11 states.
- Taylor Morrison reported net income of $235 million, or $2.12 per diluted share, for the second quarter of 2023.
- Closings increased 3% to 3,125 homes at an average price of $639,000, generating home closings revenue of $2.0 billion.
- Home closings gross margin declined 240 basis points year over year but increased 30 basis points sequentially to 24.2%.
- Net sales orders increased 18% to 3,023, driven by a monthly absorption pace of 3.1 per community.
- Ended the quarter with approximately 72,000 homebuilding lots owned and controlled, representing 5.8 years of total supply.
- Total liquidity reached an all-time high of $2.3 billion.
- Homebuilding debt-to-capitalization declined to 29.7% on a gross basis and 15.4% net of $1.2 billion of unrestricted cash.
- The Company's credit rating was upgraded by Moody's to Ba2 from Ba3 with a Stable outlook.
- Book value per share increased 30% to $45.96.
- Taylor Morrison is based in Scottsdale, Arizona.
- Taylor Morrison is one of the nation's leading land developers and homebuilders.
- Taylor Morrison has a reputation as America's Most Trusted Builder® from 2016-2023.
- Taylor Morrison serves consumers from coast to coast, including first-time, move-up, and resort lifestyle homebuyers.
- Taylor Morrison published its fifth annual Environmental, Social and Governance (ESG) Report
- The report is organized around three pillars: Building for the Future, People First, and Transparency and Accountability
- Key milestones in the report include:
- Initial assessment of the Company's greenhouse gas inventory
- Alignment with the Task Force on Climate-Related Financial Disclosures
- Board Fellowship Program to give under-represented senior business leaders real-world public board experience
- Focus on ethnically diverse homebuyers and commitment to serving them with online solutions
- Expanded disclosure on the racial and ethnic makeup of its workforce
- Launch of programs to introduce career changers and address the industry's labor shortage
- Strengthening of partnership with the National Wildlife Federation
- Taylor Morrison's disclosures align with guidelines published by the Sustainability Accounting Standards Board (SASB), the Global Reporting Initiative (GRI), and the Task Force on Climate-Related Financial Disclosures
- Taylor Morrison has earned a spot on Newsweek's 2023 America's Most Responsible Companies list.
- Taylor Morrison scored highly in corporate governance and social categories.
- The company will release its fifth annual Environmental, Social and Governance (ESG) report later this month.
- Taylor Morrison is committed to integrating ESG priorities across all aspects of its business.
- The company has a partnership with the National Wildlife Federation to restore and protect wildlife habitat.
- Taylor Morrison recently appointed its first Corporate Director of Sustainability.
- The company was named America's Most Trusted® Home Builder by Lifestory Research for the eighth consecutive year.
- Taylor Morrison advanced its diversity, equity, inclusion, and belonging (DEIB) strategy with a majority-diverse board of directors and other initiatives.
- Taylor Morrison is included on Newsweek's list of America's Most Responsible Companies, Wall Street Journal's 2022 Management Top 250, Bloomberg Gender-Equality Index (GEI), and Fortune 500 list.
- Taylor Morrison is headquartered in Scottsdale, Arizona and is one of the nation's leading homebuilders and developers.
- The company serves a wide array of consumers and has various brands, including Taylor Morrison, Esplanade, Darling Homes Collection by Taylor Morrison, and Yardly.
- Taylor Morrison has a strong commitment to sustainability, communities, and its team.
- Taylor Morrison released its fourth annual Environmental, Social and Governance (ESG) Report.
- The report focuses on the company's commitment to sustainable principles.
- Taylor Morrison emphasizes its long-term strategy for managing its carbon footprint and increasing diversity in leadership positions.
- The report highlights achievements in areas such as environmental stewardship, diversity and inclusion, and governance.
- Taylor Morrison utilized guidelines by the Global Reporting Initiative (GRI), Sustainability Accounting Standards Board (SASB), and the U.N. Sustainable Development Goals (U.N. SDGs) in preparing the report."""

    # make sure writing prompt is less than tokens for gpt-4-8k
    assert fits_in_model(writing_prompt, ModelEnum.GPT4_8K)
    # now test the second section:
    long_intro_text = ""
    for i in range(0, 500):
        long_intro_text += "This is a long intro sentence. "
    curr_paper = PaperSchema(sections=[SectionSchema(name="Introduction", lods=[long_intro_text, "<this is a quick summary of the intro 123454321>"])])
    writing_prompt = render_writing_prompt(action_plan, curr_paper, 1, summaries, total_tokens= 1800)
    # assert that contains the lower level details
    assert "This is a long intro sentence." not in writing_prompt
    assert "<this is a quick summary of the intro 123454321>" in writing_prompt
    print("=== Writing Prompt ===")
    print(writing_prompt)
    print("=== End Writing Prompt ===")
    assert fits_in_model(writing_prompt, ModelEnum.GPT4_8K)
    print(f"Number of tokens: {get_num_tokens(writing_prompt, ModelEnum.GPT4_8K)}")

    med_intro_text = ""
    for i in range(0, 100):
        med_intro_text += "This is a long intro sentence. "
    curr_paper = PaperSchema(sections=[SectionSchema(name="Introduction", lods=[med_intro_text, "<this is a quick summary of the intro 123454321>"])])
    writing_prompt = render_writing_prompt(action_plan, curr_paper, 1, summaries, total_tokens= 1800)

    # assert that contains the full intro
    assert "This is a long intro sentence." in writing_prompt
    assert "<this is a quick summary of the intro 123454321>" not in writing_prompt

    