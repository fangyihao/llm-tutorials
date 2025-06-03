import requests
from bs4 import BeautifulSoup
import logging
import pandas as pd
import arxiv
import os
from tavily import TavilyClient
import requests
import fitz  # PyMuPDF
from rouge_score import rouge_scorer
# logger = logging.getLogger(__name__)
# logging.basicConfig(filename="cvpr_crawler.log", level=logging.INFO)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
tavily_client = TavilyClient(api_key="")
# URL of the CVPR accepted papers page (update with the correct year or URL)
url = "https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers"
download_dir = "paper/cvpr_2025"

def crawl_cvpr_accepted_papers():
    """
    Crawls the CVPR accepted papers page and extracts paper titles, authors, and sections.
    Downloads the corresponding PDFs from arXiv if available.
    """
    
    # Send a GET request to fetch the webpage content
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        # logger.info(html)
        soup = BeautifulSoup(html, "html.parser")

        results = []

        for row in soup.find_all("tr"):
            if row.find("div", class_="indented"):
                # Title: could be in <a> or <strong>
                title_tag = row.find("a") or row.find("strong")
                title = title_tag.get_text(strip=True) if title_tag else "N/A"

                # Authors: in <div class="indented"><i>...</i></div>
                authors_tag = row.find("div", class_="indented")
                authors = authors_tag.get_text(strip=True).replace("·", ",") if authors_tag else "N/A"

                # Section: includes poster session, hall, and poster number
                section_lines = row.get_text(separator="\n").splitlines()
                section_info = [line.strip() for line in section_lines if "Poster Session" in line or "ExHall" in line]
                section = " | ".join(section_info)

                results.append({
                    "Title": title,
                    "Authors": authors,
                    "Section": section
                })

        
        os.makedirs(download_dir, exist_ok=True)
        # Display results
        for entry in results:
            print(f"Title: {entry['Title']}")
            print(f"Authors: {entry['Authors']}")
            print(f"Section: {entry['Section']}")
            
            arxiv_results = []
            try:
                search = arxiv.Search(
                    query=f'ti:"{entry['Title']}"',
                    max_results=1,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                arxiv_results = list(search.results())
            except:
                print(f"Error searching for {entry['Title']}")
            if len(arxiv_results) > 0:
                for result in arxiv_results:
                    entry['URL'] = result.entry_id.replace('/abs/', '/pdf/')
                    entry['Arxiv Title'] = result.title
                    # Download the PDF
                    try:
                        result.download_pdf(dirpath=download_dir)
                    except:
                        print(f"Error downloading PDF for {result.title}")
                    print(f"Downloaded PDF for {result.title}")
            else:
                try:
                    tavily_response = tavily_client.search(entry['Title'])
                    for result in tavily_response['results']:
                        tavily_title = result['title'].replace('Title:','').rstrip('.').strip()
                        
                        if 'arxiv' in result['url'] and '/abs/' in result['url']:
                            scores = scorer.score(tavily_title, entry['Title'])
                            if scores['rougeL'].fmeasure >= 0.6:
                                entry['URL'] = result['url'].replace('/abs/', '/pdf/')
                                entry['Arxiv Title'] = tavily_title
                                # Download the PDF
                                try:
                                    pdf_response = requests.get(result['url'].replace('/abs/', '/pdf/'))
                                    if pdf_response.status_code == 200:
                                        with open(f"{download_dir}/{result['url'].split('/')[-1]}.{tavily_title.replace(':', '_').replace(' ', '_')}.pdf", "wb") as f:
                                            f.write(pdf_response.content)
                                        print(f"Downloaded PDF for {tavily_title}")
                                    else:
                                        print(f"Failed to download PDF. Status code: {pdf_response.status_code}")
                                except:
                                    print(f"Error downloading PDF for {tavily_title}")
                                break
                        else:
                            print(f"No PDF available for {tavily_title}")
                except:
                    print(f"Error searching for {entry['Title']} with Tavily")
            print("-" * 60)
        pd.DataFrame(results).to_csv("cvpr_accepted_papers_2025.csv", index=False)
    else:
        print(f"Failed to fetch the webpage. Status code: {response.status_code}")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def is_pdf_corrupted(file_path):
    try:
        with fitz.open(file_path) as pdf:
            # Accessing the first page to ensure the PDF is valid
            _ = pdf[0]
        return False  # PDF is not corrupted
    except Exception as e:
        print(f"Error: {e}")
        return True  # PDF is corrupted

def verify_paper_downloads():
    # Verify the downloaded papers against the titles in the CSV file
    df = pd.read_csv("cvpr_accepted_papers_2025.csv", keep_default_na=False)
    for index, row in df.iterrows():
        if len(row['Arxiv Title'].strip()) > 0:
            scores = scorer.score(row['Arxiv Title'], row['Title'])
            if scores['rougeL'].fmeasure < 0.6:
                print(f"Title mismatch ({scores['rougeL'].fmeasure}) for {row['Arxiv Title']} vs {row['Title']}")
                filename = f"{download_dir}/{row['URL'].split('/')[-1]}.{row['Arxiv Title'].replace(':', '_').replace(' ', '_')}.pdf"
                if os.path.exists(filename):
                    os.remove(filename)
                    print(f"Removed mismatched file: {filename}")
                row['Arxiv Title'] = ''
                row['URL'] = ''
    df.to_csv("cvpr_accepted_papers_2025.csv", index=False)

    # Check for corrupted PDFs
    for index, row in df.iterrows():
        if len(row['Arxiv Title'].strip()) > 0:
            filename = f"{download_dir}/{row['URL'].split('/')[-1]}.{row['Arxiv Title'].replace(':', '_').replace(' ', '_')}.pdf"
            if os.path.exists(filename):
                if is_pdf_corrupted(filename):
                    print(f"Corrupted PDF: {filename}")
                
if __name__ == "__main__":
    crawl_cvpr_accepted_papers()
    verify_paper_downloads()
