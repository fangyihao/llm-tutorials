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

