# web_to_corpus.py
import os
import requests
from bs4 import BeautifulSoup
import pdfkit
from urllib.parse import urljoin, urlparse, urldefrag
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import streamlit as st
import tempfile
import shutil
from PyPDF2 import PdfMerger
import json

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class WebsiteCrawler:
    def __init__(self, root_url, output_format):
        self.root_url = root_url
        self.output_format = output_format
        self.visited_links = set()
        self.to_visit_links = set([root_url])
        self.domain_name = urlparse(root_url).netloc
        self.temp_dir = tempfile.mkdtemp(dir=SCRIPT_DIR)
        self.crawled_data = []

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920x1080")
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        if self.output_format == "PDF":
            self.pdf_options = {
                'quiet': '',
                'enable-local-file-access': ''
            }

    def __del__(self):
        self.driver.quit()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def fetch_page(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            st.error(f"Failed to fetch {url}: {e}")
            return None

    def fetch_page_selenium(self, url):
        try:
            self.driver.get(url)
            time.sleep(3)
            return self.driver.page_source
        except Exception as e:
            st.error(f"Failed to fetch {url} with Selenium: {e}")
            return None

    def save_page_as_pdf(self, url, content):
        filename = self.get_filename(url, "pdf")
        st.info(f"Saving {url} as PDF")
        try:
            pdfkit.from_string(content, filename, options=self.pdf_options)
            return filename
        except IOError as e:
            st.error(f"Failed to convert {url} to PDF: {e}")
            return None

    def get_filename(self, url, extension):
        parsed_url = urlparse(url)
        path = parsed_url.path.strip("/").replace("/", "_")
        if not path:
            path = "index"
        # Create the 'files' folder if it doesn't exist
        files_folder = os.path.join(SCRIPT_DIR, "files")
        if not os.path.exists(files_folder):
            os.makedirs(files_folder)
        return os.path.join(files_folder, f"{path}.{extension}")

    def crawl(self):
        progress_bar = st.progress(0)
        status_text = st.empty()

        while self.to_visit_links:
            current_url = self.to_visit_links.pop()
            if current_url in self.visited_links:
                continue

            status_text.text(f"Visiting: {current_url}")
            page_content = self.fetch_page(current_url)
            if page_content is None:
                page_content = self.fetch_page_selenium(current_url)

            if page_content is None:
                continue

            self.visited_links.add(current_url)
            
            if self.output_format == "PDF":
                pdf_file = self.save_page_as_pdf(current_url, page_content)
                if pdf_file:
                    self.crawled_data.append({"url": current_url, "file": pdf_file})
            else:
                self.crawled_data.append({"url": current_url, "content": page_content})

            self.find_links_on_page(current_url, page_content)

            progress = len(self.visited_links) / (len(self.to_visit_links) + len(self.visited_links))
            progress_bar.progress(progress)

        status_text.text("Crawling completed!")

    def find_links_on_page(self, base_url, page_content):
        soup = BeautifulSoup(page_content, 'html.parser')
        for link in soup.find_all('a', href=True):
            url = urljoin(base_url, link['href'])
            url, _ = urldefrag(url)
            if self.is_valid_url(url):
                if url not in self.visited_links and url not in self.to_visit_links:
                    self.to_visit_links.add(url)

    def is_valid_url(self, url):
        parsed_url = urlparse(url)
        return (
            parsed_url.scheme in {"http", "https"} and
            parsed_url.netloc == self.domain_name and
            not parsed_url.path.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.js', '.css')) and
            'entry_point=' not in parsed_url.query and
            'share=' not in parsed_url.query and
            'utm_' not in parsed_url.query
        )

    def generate_output(self, output_filename):
        output_path = os.path.join(SCRIPT_DIR, "files", output_filename)
        if self.output_format == "PDF":
            self.merge_pdfs(output_path)
        elif self.output_format == "JSON":
            self.save_as_json(output_path)
        else:  # TXT
            self.save_as_txt(output_path)

    def merge_pdfs(self, output_filename):
        merger = PdfMerger()
        for item in self.crawled_data:
            merger.append(item['file'])
        merger.write(output_filename)
        merger.close()

    def save_as_json(self, output_filename):
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(self.crawled_data, f, ensure_ascii=False, indent=4)

    def save_as_txt(self, output_filename):
        with open(output_filename, 'w', encoding='utf-8') as f:
            for item in self.crawled_data:
                f.write(f"URL: {item['url']}\n\n")
                f.write(f"Content:\n{item['content']}\n\n")
                f.write("-" * 80 + "\n\n")

def main():
    st.title("Website Crawler Corpus File Generator")
    st.write("Enter the website URL you want to crawl in the box below. Choose your preferred output format (PDF, JSON, or TXT) from the dropdown menu. Click 'Start Crawling' to begin. Once complete, the generated file will be saved to the 'files' folder. You can access and manage this file in the 'Files' tab under the 'Chat' section or through the 'Document' section.")
    root_url = st.text_input("Enter the root URL to crawl:")
    
    output_format = st.selectbox(
        "Choose output format",
        ("PDF", "JSON", "TXT")
    )
    
    if st.button("Start Crawling"):
        if root_url:
            crawler = WebsiteCrawler(root_url, output_format)
            crawler.crawl()
           
            st.success("Crawling completed! Generating output file...")
            
            output_filename = f"{urlparse(root_url).netloc}.{output_format.lower()}"
            
            crawler.generate_output(output_filename)
            
            st.success(f"{output_format} generation completed! File saved as {output_filename}")
            
            output_path = os.path.join(SCRIPT_DIR, "files", output_filename)
            with open(output_path, "rb") as file:
                st.download_button(
                    label=f"Download {output_format} File",
                    data=file,
                    file_name=output_filename,
                    mime=f"application/{output_format.lower()}"
                )
        else:
            st.error("Please enter a valid URL.")

if __name__ == "__main__":
    main()