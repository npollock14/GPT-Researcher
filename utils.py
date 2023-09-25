import requests
from bs4 import BeautifulSoup
from newspaper import Article
import tiktoken
from summa.summarizer import summarize
from summa import keywords

def fetch_site_content(url):
    article = Article(url)
    article.download()
    article.parse()
    article.summary, article.keywords = summarize_content(article.text)
    return article

def summarize_content(content):
    summary = summarize(content, words=2000)
    extracted_keywords = keywords.keywords(content)
    return summary, extracted_keywords

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
    article = fetch_site_content("https://www.sec.gov/Archives/edgar/data/1562476/000119312515070090/d839495d10k.htm")
    print(article.summary)
    # print(article.keywords)


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


