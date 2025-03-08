from bs4 import BeautifulSoup as bs
from urllib.request import urlopen, urlretrieve, Request
import requests
import pandas as pd
from tqdm import tqdm

hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

from IPython.display import clear_output
import os 

def htmlize(URL):
    global hdr
    req = Request(url=URL, headers=hdr)
    html_opened = urlopen(req)
    html = bs(html_opened, features='html.parser')
    return html



def retrieve_drug_data(rxlist_url):
    rxlist_html = htmlize(rxlist_url)
    target_div = rxlist_html.find("div", {"class": "monograph_cont"})  # Adjust class if needed
    list_of_terms = ["What is", "What Are Side Effects", "Children", "Dosage", "Interact", "Pregnancy"]

    # Initialize variables
    sections = []
    current_section = None

    # Iterate through all elements inside the div
    for element in target_div.children:
        if element.name == "h4":
            # Check if the h4 text contains at least one term from list_of_terms
            h4_text = element.text.strip()
            
            if any(term.lower() in h4_text.lower() for term in list_of_terms):
                
                if current_section:  
                    sections.append(current_section)  # Save the previous section
                current_section = {"header": h4_text, "content": ""}
            else:
                if current_section:  
                    sections.append(current_section)  # Save the previous section
                current_section = None  # Ignore this section since it doesn't match the terms

        elif current_section is not None and element.name in ["p", "ul", "li"]:
            # Add content to the valid section
            current_section["content"] += "\n" + (element.text)

    # Append the last valid section if it exists

    if current_section:
        sections.append(current_section)

    return sections

# sections = retrieve_drug_data('https://www.rxlist.com/tadliq-drug.htm')
# print(sections)