from bs4 import BeautifulSoup
from newspaper import Article
import tiktoken
from schemas import ResearchActionPlanSchema, SearchResultSchema, ModelEnum, str_to_model_enum, SearchResultSummary
from googlesearch import search
from prompts import get_summary_prompt, prepare_source_material_for_summary_prompt
import re
import os
import openai
from aiohttp import ClientSession
import asyncio


def fetch_site_content(url:str):
    article = Article(url)
    article.download()
    article.parse()
    if article.text == "":
        raise Exception("Article text is empty")
    return article

def str_cost_analysis(input: str="", num_output_tokens: int=0, model: ModelEnum=ModelEnum.GPT4_8K):
    enc = tiktoken.encoding_for_model(model.value.official_name)
    input_pricing_per_thousand = model.value.pricing['input']
    output_pricing_per_thousand = model.value.pricing['output']
    num_input_tokens = len(enc.encode(input))
    input_cost = num_input_tokens * input_pricing_per_thousand / 1000
    output_cost = num_output_tokens * output_pricing_per_thousand / 1000
    total_cost = input_cost + output_cost
    return total_cost

def get_num_tokens(input:str="", model=ModelEnum.GPT4_8K):
    enc = tiktoken.encoding_for_model(model.value.official_name)
    num_input_tokens = len(enc.encode(input))
    return num_input_tokens

def fits_in_model(context:str, model:ModelEnum, padding_tokens: int = 0):
    enc = tiktoken.encoding_for_model(model.value.official_name)
    context_tokens = len(enc.encode(context))
    return context_tokens + padding_tokens <= model.value.max_context


def est_cost_search_result(search_result: SearchResultSchema):
    # input tokens will be content of search result
    # output tokens need to be estimated, for now assume 512 tokens
    return str_cost_analysis(input=search_result.content, num_output_tokens=512, model=search_result.model)


def generate_search_results(research_action_plan: ResearchActionPlanSchema) -> list[SearchResultSchema]:
    search_results = []
    for query in research_action_plan.search_queries:
        res = list(search(query, num_results=3, advanced=True))
        for search_result in res:
            link = search_result.url
            site_content = None
            try:
                site_content = fetch_site_content(link)
                site_content = clean_content(site_content.text)
            except:
                continue
            if len(site_content) < 10:
                # site content is too short, skip this result
                continue
            new_result = SearchResultSchema(
                title=search_result.title,
                link=search_result.url,
                content=site_content,
                cost=0,
                model=ModelEnum.GPT3_5_TURBO_16K
            )
            # check what model to use based on content length
            if fits_in_model(get_summary_prompt(research_action_plan) + prepare_source_material_for_summary_prompt(new_result), ModelEnum.GPT3_5_TURBO_4K, padding_tokens=100):
                new_result.model = ModelEnum.GPT3_5_TURBO_4K
            elif fits_in_model(get_summary_prompt(research_action_plan) + prepare_source_material_for_summary_prompt(new_result), ModelEnum.GPT3_5_TURBO_16K, padding_tokens=100):
                new_result.model = ModelEnum.GPT3_5_TURBO_16K
            else:
                # skip this result
                continue
            new_result.cost = est_cost_search_result(new_result)
            search_results.append(new_result)
    return search_results

def parse_raw_summary(raw_summary: str, search_result: SearchResultSchema) -> SearchResultSummary:
    lines = [line.strip() for line in raw_summary.strip().split("\n")]
    
    source_material = {}
    current_section = None
    details = []
    relevancy = {}
    error_messages = []
    
    for line in lines:
        if line.startswith("Title:"):
            source_material["Title"] = line.split("Title:")[1].strip()
        elif line.startswith("Author(s):"):
            authors = line.split("Author(s):")[1].strip()
            if authors.lower() == "not found":
                source_material["Author(s)"] = []
            else:
                source_material["Author(s)"] = [author.strip() for author in authors.split(",")]
        elif line.startswith("Date:"):
            date = line.split("Date:")[1].strip()
            source_material["Date"] = "" if date.lower() == "not found" else date
        elif line.startswith("- ") and current_section is None:
            details.append(line[2:].strip())
        elif line.startswith("Relevancy:"):
            current_section = "Relevancy"
        elif current_section == "Relevancy" and ": " in line:
            section, score = line.split(":")
            relevancy[section.lstrip("- ").strip()] = int(score.strip())
    
    # Check for missing or empty sections
    if not details:
        error_messages.append("Details section is missing or empty.")
    if not relevancy:
        error_messages.append("Relevancy section is missing or empty.")
    
    error = None if not error_messages else " ".join(error_messages)
    
    return SearchResultSummary(
        source_material=search_result,
        details=details,
        authors=source_material.get("Author(s)", []),
        date=source_material.get("Date", ""),
        relevancy=relevancy,
        error=error
    )

async def summarize_result_async(search_result: SearchResultSchema, research_action_plan: ResearchActionPlanSchema) -> SearchResultSummary:
    completion = await openai.ChatCompletion.acreate(
        model=search_result.model.value.official_name,
        messages=[
            {
                "role": "system",
                "content": get_summary_prompt(research_action_plan)
            },
            {
                "role": "user",
                "content": prepare_source_material_for_summary_prompt(search_result)
            }
        ],
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    raw_result: str = completion.choices[0].message.content
    return parse_raw_summary(raw_result, search_result)

async def summarize_results(search_results: list[SearchResultSchema], research_action_plan: ResearchActionPlanSchema) -> list[dict]:
    # Create a new aiohttp client session
    session = ClientSession()
    openai.aiosession.set(session)

    # Parallelize the summarization of search results
    summaries: list[SearchResultSummary] = await asyncio.gather(
        *[summarize_result_async(result, research_action_plan) for result in search_results]
    )

    # Close the session at the end
    await session.close()
    
    return summaries

def clean_content(content: str) -> str:
    # Replace multiple newline characters with a single newline
    cleaned_content = re.sub(r'\n+', '\n', content)
    
    # Trim leading and trailing whitespace
    cleaned_content = cleaned_content.strip()
    
    # Replace multiple spaces with a single space
    cleaned_content = re.sub(r' +', ' ', cleaned_content)
    
    return cleaned_content

if __name__ == "__main__":
    # load a test schema from ./outputs/parsed_user_prompt_for_research.json
    import json
    import os
    from dotenv import load_dotenv
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parsed_user_prompt_output_file = "./outputs/parsed_user_prompt_for_research.json"

    if not os.path.exists(parsed_user_prompt_output_file):
        print(f"Error: {parsed_user_prompt_output_file} not found")
        exit(1)
    with open(parsed_user_prompt_output_file) as json_file:
        result_data = json_file.read()
        action_plan = ResearchActionPlanSchema.model_validate_json(result_data)

    search_result_output_file = "./outputs/search_results.json"
    search_results: list[SearchResultSchema] = []
    if os.path.exists(search_result_output_file):
        # just use the cached search results
        with open(search_result_output_file) as json_file:
            temp_results = json.load(json_file)
            for search_result in temp_results:
                res = SearchResultSchema(
                    title=search_result["title"],
                    link=search_result["link"],
                    content=search_result["content"],
                    cost=search_result["cost"],
                    model=str_to_model_enum(search_result["model"])
                )
                search_results.append(res)
    else:
        # generate search results
        search_results = generate_search_results(action_plan)
        # save search results to a file outputs/search_results.json
        with open(search_result_output_file, 'w') as outfile:
            json.dump([search_result.to_json() for search_result in search_results], outfile, indent=4)

    summary_output_file = "./outputs/summary.json"
    summaries: list[SearchResultSummary] = []
    if os.path.exists(summary_output_file):
        # just use the cached summary
        with open(summary_output_file) as json_file:
            summaries = json.load(json_file)
    else:
        # summarize search result for index 0
        # sum the cost of all search results and ask user if they want to continue
        total_cost = 0
        for search_result in search_results:
            total_cost += search_result.cost

        print(f"Total cost of processing {len(search_results)} search results: ${total_cost}")
        ipt = input("Do you want to continue? (y/n): ")
        if ipt.lower() != "y":
            exit(0)

        # parallelize the summarization of search results
        summaries: list[SearchResultSummary] = asyncio.run(summarize_results(search_results, action_plan))

        summary_output_file = "./outputs/summary.json"
        # save the summaries to a file outputs/summary.json
        with open(summary_output_file, 'w') as outfile:
            json.dump([summary.to_json() for summary in summaries], outfile, indent=4)

    num_details = 0
    num_details_with_error = 0
    relevance_scores = {}
    for summary in summaries:
        # filter out summaries with error or empty details
        if summary["error"] is not None or len(summary["details"]) == 0:
            continue
        print("- "+"\n- ".join(summary["details"]))
        num_details += len(summary["details"])
        if summary["error"] is not None:
            num_details_with_error += len(summary["details"])
        for section, score in summary["relevancy"].items():
            if section not in relevance_scores:
                relevance_scores[section] = 0
            relevance_scores[section] += score * len(summary["details"])

    print(f"Total number of details: {num_details}")
    print(f"Total number of details with error: {num_details_with_error}")
    print(f"Relevance scores: {relevance_scores}")