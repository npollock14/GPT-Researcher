import requests
from bs4 import BeautifulSoup
from newspaper import Article
import tiktoken

def fetch_site_content(url):
    article = Article(url)
    article.download()
    article.parse()
    return article

def cost_analysis(input="", output=""):
    enc = tiktoken.encoding_for_model("gpt-4")
    input_pricing_per_thousand = 0.03
    output_pricing_per_thousand = 0.06
    num_input_tokens = len(enc.encode(input))
    num_output_tokens = len(enc.encode(output))
    input_cost = num_input_tokens * input_pricing_per_thousand / 1000
    output_cost = num_output_tokens * output_pricing_per_thousand / 1000
    total_cost = input_cost + output_cost
    return total_cost

if __name__ == "__main__":
    print(cost_analysis("""Topic of Research: Deloitte
Purpose of Context: Preparation for a business meeting with Deloitte
Primary Audience: Business professionals
Research Plan:
- Understand the history and background of Deloitte
- Identify the key services and industries Deloitte operates in
- Research recent news and updates about Deloitte
- Analyze Deloitte's business strategies and performance
- Identify key personnel and decision-makers in Deloitte
- Understand Deloitte's corporate culture and values
- Review any recent case studies or projects undertaken by Deloitte
Search Queries:
- Deloitte company history
- Deloitte services and industries
- Deloitte recent news
- Deloitte business strategies
- Deloitte key personnel
- Deloitte corporate culture
- Deloitte case studies"""))


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


