# open ./outputs/paper.json
# looks like
"""
{
    "sections": [
        {
            "name": "Introduction",
            "lods": [
            ...
                """

# take the lod[0] and combine to form the final paper

# use the md2pdf library to convert the markdown to pdf

from md2pdf.core import md2pdf

content = ""

with open('./outputs/paper.json', 'r') as outfile:
    import json
    paper = json.load(outfile)
    for section in paper["sections"]:
        content += section["lods"][0]

md2pdf("./outputs/paper.pdf",
       md_content=content)