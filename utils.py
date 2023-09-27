import requests
from bs4 import BeautifulSoup
from newspaper import Article
import tiktoken
from summa.summarizer import summarize
from summa import keywords
from schemas import ResearchActionPlanSchema, SearchResultSchema, ModelEnum
from googlesearch import search
from prompts import SEARCH_CHOICE_PROMPT, get_summary_prompt
import asyncio
from gpt_json import GPTJSON, GPTMessage, GPTMessageRole


def fetch_site_content(url):
    article = Article(url)
    article.download()
    article.parse()
    if article.text == "":
        raise Exception("Article text is empty")
    article.summary, article.keywords = summarize_content(article.text)
    return article

def summarize_content(content):
    wordsToUse = 1000
    contentLen = len(content.split(' '))
    if contentLen < wordsToUse and contentLen > 100:
        wordsToUse = contentLen * 0.5
    else:
        wordsToUse = contentLen
    summary = summarize(content, words=wordsToUse)
    maxKeywords = 10
    extracted_keywords = keywords.keywords(content)
    extracted_keywords = extracted_keywords.split('\n')
    if len(extracted_keywords) > maxKeywords:
        extracted_keywords = extracted_keywords[:maxKeywords]
    return summary, extracted_keywords

def str_cost_analysis(input="", output="", model=ModelEnum.GPT4_8K):
    enc = tiktoken.encoding_for_model(model.value.official_name)
    input_pricing_per_thousand = model.value.pricing['input']
    output_pricing_per_thousand = model.value.pricing['output']
    num_input_tokens = len(enc.encode(input))
    num_output_tokens = len(enc.encode(output))
    input_cost = num_input_tokens * input_pricing_per_thousand / 1000
    output_cost = num_output_tokens * output_pricing_per_thousand / 1000
    total_cost = input_cost + output_cost
    return total_cost

def fits_in_model(context:str, model=ModelEnum.GPT3_5_TURBO_16K, padding_tokens=0):
    enc = tiktoken.encoding_for_model(model.value.official_name)
    context_tokens = len(enc.encode(context))
    return context_tokens + padding_tokens <= model.value.max_context


def est_cost_search_result(search_result: SearchResultSchema):
    # input tokens will be content of search result
    # output tokens need to be estimated, for now assume entire content length
    return str_cost_analysis(input=search_result.content, output=search_result.content)


def generate_search_results(research_action_plan: ResearchActionPlanSchema):
    search_results = []
    for query in research_action_plan.search_queries:
        res = list(search(query, num_results=1, advanced=True))
        for search_result in res:
            link = search_result.url
            site_content = None
            try:
                site_content = fetch_site_content(link)
            except:
                continue
            new_result = SearchResultSchema(
                title=search_result.title,
                link=search_result.url,
                description=search_result.description,
                keywords=site_content.keywords,
                summary=site_content.summary,
                content=site_content.text,
                cost=0
            )
            # if the content + the summary prompt is greater than the context limit, skip for now
            if not fits_in_model(new_result.content + get_summary_prompt(research_action_plan), ModelEnum.GPT3_5_TURBO_16K, padding_tokens=200):
                continue
            new_result.cost = est_cost_search_result(new_result)
            search_results.append(new_result)
    return search_results

async def summarize_result(search_result: SearchResultSchema, api_key: str):
    gpt_json = GPTJSON[TODO](api_key)
    
    payload = await gpt_json.run(
        messages=[
            GPTMessage(
                role=GPTMessageRole.SYSTEM,
                content=get_summary_prompt(search_result),
            ),
            GPTMessage(
                role=GPTMessageRole.USER,
                content=search_result.content,
            )
        ]
    )
    
    return payload.response

# async def choose_results(search_results: list[SearchResultSchema], api_key: str):
#     gpt_json = GPTJSON[ResultChoiceSchema](api_key)
    
#     payload = await gpt_json.run(
#         messages=[
#             GPTMessage(
#                 role=GPTMessageRole.SYSTEM,
#                 content=SYSTEM_PROMPT,
#             ),
#             GPTMessage(
#                 role=GPTMessageRole.USER,
#                 content=f"Sources:\n{user_prompt}",
#             )
#         ]
#     )
    
#     return payload.response

if __name__ == "__main__":
    # load a test schema from ./outputs/parsed_user_prompt_for_research.json
    import json
    import os
    if not os.path.exists('./outputs/parsed_user_prompt_for_research.json'):
        print("Error: ./outputs/parsed_user_prompt_for_research.json not found")
        exit(1)
    with open('./outputs/parsed_user_prompt_for_research.json') as json_file:
        result_data = json_file.read()
        result = ResearchActionPlanSchema.model_validate_json(result_data)

    search_results = generate_search_results(result)
    # save search results to ./outputs/search_results.json
    with open('./outputs/search_results.json', 'w') as outfile:
        # Convert the list of SearchResultSchema instances to a list of dictionaries
        serialized_search_results = [result.model_dump() for result in search_results]
        
        # Serialize the list of dictionaries to JSON and write it to the file
        outfile.write(json.dumps(serialized_search_results))

    # summarize the search results

    
