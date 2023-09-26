import requests
from bs4 import BeautifulSoup
from newspaper import Article
import tiktoken
from summa.summarizer import summarize
from summa import keywords
from schemas import ResearchActionPlanSchema, SearchResultSchema, ModelEnum
from googlesearch import search


def fetch_site_content(url):
    article = Article(url)
    article.download()
    article.parse()
    article.summary, article.keywords = summarize_content(article.text)
    return article

def summarize_content(content):
    wordsToUse = 1000
    contentLen = len(content.split(' '))
    if contentLen < wordsToUse:
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
            new_result.cost = est_cost_search_result(new_result)
            search_results.append(new_result)
    return search_results

        

if __name__ == "__main__":
    # load a test schema from ./outputs/parsed_user_prompt_for_research.json
    import json
    import os
    if os.path.exists('./outputs/parsed_user_prompt_for_research.json'):
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


    else:
        # error
        print("Error: ./outputs/parsed_user_prompt_for_research.json not found")
# def fetch_site_content(url):
#     html = requests.get(url).text
#     soup = BeautifulSoup(html, features="html.parser")

#     # kill all script and style elements
#     for script in soup(["script", "style"]):
#         script.extract()    # rip it out

#     # get text
#     text = soup.get_text()

#     # break into lines and remove leading and trailing space on each
#     lines = (line.strip() for line in text.splitlines())
#     # break multi-headlines into a line each
#     chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#     # drop blank lines
#     text = '\n'.join(chunk for chunk in chunks if chunk)
#     return text


