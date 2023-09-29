from bs4 import BeautifulSoup
from newspaper import Article
import tiktoken
from schemas import ResearchActionPlanSchema, SearchResultSchema, ModelEnum
from googlesearch import search
from prompts import get_summary_prompt, prepare_source_material_for_summary_prompt
import re
import os
import openai


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
        res = list(search(query, num_results=1, advanced=True))
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

def parse_raw_summary(raw_summary: str) -> dict:
    lines = [line.strip() for line in raw_summary.strip().split("\n")]
    
    parsed_data = {}
    current_section = None
    details = []
    relevancy = {}
    
    for line in lines:
        if line.startswith("Title:"):
            parsed_data["Title"] = line.split("Title:")[1].strip()
        elif line.startswith("Author(s):"):
            authors = line.split("Author(s):")[1].strip()
            if authors.lower() == "not found":
                parsed_data["Author(s)"] = []
            else:
                parsed_data["Author(s)"] = [author.strip() for author in authors.split(",")]
        elif line.startswith("Date:"):
            date = line.split("Date:")[1].strip()
            parsed_data["Date"] = "" if date.lower() == "not found" else date
        elif line.startswith("- ") and current_section is None:
            details.append(line[2:].strip())
        elif line.startswith("Relevancy:"):
            current_section = "Relevancy"
        elif current_section == "Relevancy" and ": " in line:
            section, score = line.split(":")
            relevancy[section.lstrip("- ").strip()] = int(score.strip())
    
    parsed_data["Details"] = details
    parsed_data["Relevancy"] = relevancy
    
    return parsed_data

def summarize_result(search_result: SearchResultSchema, research_action_plan: ResearchActionPlanSchema) -> dict:
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
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
    temperature=0.9,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    
    return parse_raw_summary(completion.choices[0].message.content)

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

    if not os.path.exists('./outputs/parsed_user_prompt_for_research.json'):
        print("Error: ./outputs/parsed_user_prompt_for_research.json not found")
        exit(1)
    with open('./outputs/parsed_user_prompt_for_research.json') as json_file:
        result_data = json_file.read()
        result = ResearchActionPlanSchema.model_validate_json(result_data)

    # generate search results
    search_results = generate_search_results(result)

    # save search results to a file outputs/search_results.json
    # with open('./outputs/search_results.json', 'w') as outfile:
    #     json.dump([search_result.to_json() for search_result in search_results], outfile, indent=4)

    # summarize search result for index 0
    summary = summarize_result(search_results[3], result)
    print(summary)
    # save the summary to a file outputs/summary.json
    with open('./outputs/summary.json', 'w') as outfile:
        json.dump(summary, outfile, indent=4)

    
