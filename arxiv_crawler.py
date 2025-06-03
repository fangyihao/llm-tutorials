import arxiv
import requests
import time
import json
import multiprocessing
import os
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
def search_arxiv(query):
    papers = []

    search = arxiv.Search(
        query=query,
        max_results=100,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    for result in search.results():
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

def setup_crawler(query, log_filename, download_dir):
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    while True:
        try:
            results = search_arxiv(query)
            for paper in results:
                logger.info(f"Title: {paper['title']}")
                logger.info(f"Authors: {', '.join(paper['authors'])}")
                logger.info(f"Published: {paper['published']}")
                logger.info(f"Citations: {paper['citation_count']}")
                logger.info(f"URL: {paper['url']}\n")
                try:
                    paper["content"].download_pdf(dirpath=download_dir)
                except:
                    logger.exception(f"Failed to download PDF for {paper['title']}")
        except:
            logger.exception("An error occurred while fetching papers.")
        time.sleep(3600*24)
''' config.json
{
    "topics": [
        {
            "query": "large language models AND diffusion models",
            "log_filename": "log/diffusion_models.log",
            "download_dir": "paper/diffusion_models"
        },
        {
            "query": "large language models AND reasoning",
            "log_filename": "log/reasoning.log",
            "download_dir": "paper/reasoning"
        },
        {
            "query": "large language models AND video generation",
            "log_filename": "log/video_generation.log",
            "download_dir": "paper/video_generation"
        },
        {
            "query": "large language models AND deep research",
            "log_filename": "log/deep_research.log",
            "download_dir": "paper/deep_research"
        },
        {
            "query": "large language models AND peer review",
            "log_filename": "log/peer_review.log",
            "download_dir": "paper/peer_review"
        },
        {
            "query": "large language models AND fact checking",
            "log_filename": "log/fact_checking.log",
            "download_dir": "paper/fact_checking"
        },
        {
            "query": "large language models AND healthcare",
            "log_filename": "log/healthcare.log",
            "download_dir": "paper/healthcare"
        },
        {
            "query": "large language models AND embodied intelligence",
            "log_filename": "log/embodied_intelligence.log",
            "download_dir": "paper/embodied_intelligence"
        },
        {
            "query": "large language models AND autonomous driving",
            "log_filename": "log/autonomous_driving.log",
            "download_dir": "paper/autonomous_driving"
        },
        {
            "query": "large language models AND drug discovery",
            "log_filename": "log/drug_discovery.log",
            "download_dir": "paper/drug_discovery"
        },
        {
            "query": "large language models AND gaming",
            "log_filename": "log/gaming.log",
            "download_dir": "paper/gaming"
        },
        {
            "query": "large language models AND quantization",
            "log_filename": "log/quantization.log",
            "download_dir": "paper/quantization"
        },
        {
            "query": "large language models AND kv cache",
            "log_filename": "log/kv_cache.log",
            "download_dir": "paper/kv_cache"
        },
        {
            "query": "large language models AND mixture of experts",
            "log_filename": "log/mixture_of_experts.log",
            "download_dir": "paper/mixture_of_experts"
        },
        {
            "query": "large language models AND sparce attention",
            "log_filename": "log/sparce_attention.log",
            "download_dir": "paper/sparce_attention"
        },
        {
            "query": "large language models AND multi-modalality",
            "log_filename": "log/multi_modalality.log",
            "download_dir": "paper/multi_modalality"
        }
    ]
}
'''
def main():
    with open("config/config.json","r") as f:
        config = json.load(f)
    processes = []
    for entry in config["topics"]:
        query = entry["query"]
        log_filename = entry["log_filename"]
        download_dir = entry["download_dir"]
        os.makedirs(download_dir, exist_ok=True)
        process = multiprocessing.Process(target=setup_crawler, args=(query, log_filename, download_dir))
        process.daemon = True
        process.start()
        processes.append(process)
        time.sleep(3600)
    for process in processes:
        process.join()
# Example usage
if __name__ == "__main__":
    main()

