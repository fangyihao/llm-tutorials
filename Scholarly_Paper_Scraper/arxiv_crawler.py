import arxiv
import requests
import time
import json
import threading
import os
import logging

# Function to fetch citation count from Semantic Scholar
def get_citation_count(paper_title):
    api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": paper_title, "fields": "citationCount", "limit": 1}
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("data"):
            return data["data"][0].get("citationCount", 0)
    return 0

# Search arXiv for papers
def search_arxiv(client, query):
    papers = []
    search = arxiv.Search(
        query=query,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    for result in client.results(search):
        citation_count = get_citation_count(result.title)
        papers.append({
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published": result.published,
            "citation_count": citation_count,
            "url": result.entry_id,
            "content": result
        })
    papers = sorted(papers, key=lambda x: (x["citation_count"], x["published"]), reverse=True)[:5]

    return papers

def get_logger(log_filename):
    logger = logging.getLogger(f"{__name__}_{log_filename}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_filename)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_crawler(query, log_filename, download_dir):
    # Construct the default API client.
    client = arxiv.Client()
    logger = get_logger(log_filename)
    while True:
        try:
            results = search_arxiv(client, query)
            for paper in results:
                logger.info(f"Title: {paper['title']}")
                logger.info(f"Authors: {', '.join(paper['authors'])}")
                logger.info(f"Published: {paper['published']}")
                logger.info(f"Citations: {paper['citation_count']}")
                logger.info(f"URL: {paper['url']}\n")
                try:
                    paper["content"].download_pdf(dirpath=download_dir)
                except:
                    print(f"Failed to download PDF for {paper['title']}")
        except:
            print("An error occurred while fetching papers.")
        time.sleep(3600*24)

def main():
    with open("arxiv_crawler.cfg","r") as f:
        config = json.load(f)
    threads = []
    for entry in config["topics"]:
        query = entry["query"]
        print(f"Starting crawler for query: {query}", flush=True)
        log_filename = entry["log_filename"]
        download_dir = entry["download_dir"]
        os.makedirs(download_dir, exist_ok=True)
        thread = threading.Thread(target=setup_crawler, args=(query, log_filename, download_dir))
        thread.start()
        threads.append(thread)
        time.sleep(3600)
    for thread in threads:
        thread.join()
# Example usage
if __name__ == "__main__":
    main()

''' config.json
{
    "topics": [
        {
            "query": "all:\"diffusion models\"",
            "log_filename": "log/diffusion_models.log",
            "download_dir": "paper/diffusion_models"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:reasoning",
            "log_filename": "log/reasoning.log",
            "download_dir": "paper/reasoning"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:\"video generation\"",
            "log_filename": "log/video_generation.log",
            "download_dir": "paper/video_generation"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND (all:multi-modalality OR all:\"multi-modal\" OR all:\"multimodal\")",
            "log_filename": "log/multi_modalality.log",
            "download_dir": "paper/multi_modalality"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:\"deep research\"",
            "log_filename": "log/deep_research.log",
            "download_dir": "paper/deep_research"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND (all:\"peer review\" OR all:\"paper review\")",
            "log_filename": "log/peer_review.log",
            "download_dir": "paper/peer_review"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:\"code review\"",
            "log_filename": "log/peer_review.log",
            "download_dir": "paper/peer_review"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND (all:\"fact checking\" OR all:\"fact-checking\")",
            "log_filename": "log/fact_checking.log",
            "download_dir": "paper/fact_checking"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:healthcare",
            "log_filename": "log/healthcare.log",
            "download_dir": "paper/healthcare"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:\"embodied intelligence\"",
            "log_filename": "log/embodied_intelligence.log",
            "download_dir": "paper/embodied_intelligence"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:\"autonomous driving\"",
            "log_filename": "log/autonomous_driving.log",
            "download_dir": "paper/autonomous_driving"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND (all:\"drug discovery\" OR all:\"drug design\")",
            "log_filename": "log/drug_discovery.log",
            "download_dir": "paper/drug_discovery"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:gaming",
            "log_filename": "log/gaming.log",
            "download_dir": "paper/gaming"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:quantization",
            "log_filename": "log/quantization.log",
            "download_dir": "paper/quantization"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:\"kv cache\"",
            "log_filename": "log/kv_cache.log",
            "download_dir": "paper/kv_cache"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND (all:\"mixture of experts\" OR all:\"mixture-of-experts\")",
            "log_filename": "log/mixture_of_experts.log",
            "download_dir": "paper/mixture_of_experts"
        },
        {
            "query": "(all:\"large language models\" OR all:\"diffusion models\") AND all:\"sparce attention\"",
            "log_filename": "log/sparce_attention.log",
            "download_dir": "paper/sparce_attention"
        }
    ]
}
'''
