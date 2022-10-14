import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

def get_blog_article_content(url, header = {'User-Agent': 'Codeup Data Science'}):
    url = 'https://codeup.com/blog/'
    soup = BeautifulSoup(requests.get(url, headers=header).content, 'html.parser')
    return [link['href'] for link in soup.select('a.more-link')]

def get_blog_articles(base_url, header = {'User-Agent': 'Codeup Data Science'}):
    base_url = 'https://codeup.com/blog/'
    urls = get_blog_article_content(base_url)
    output = []
    for blog in urls:
        article_soup = BeautifulSoup(requests.get(blog, headers=header).content)
        blog_output = {'title': article_soup.select_one('h1.entry-title').text,
        'content': article_soup.select_one('div.entry-content').text.strip()}
        output.append(blog_output)
    filename = "blog_articles.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from URL
    # and write it as csv locally for future use
    else:
        df = pd.DataFrame(output)
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

    # Return the dataframe to the calling code
        return df 

def get_cats(url):
    url = 'https://inshorts.com/en/read'
    soup = BeautifulSoup(requests.get(url).content)
    return [cat.text.lower() for cat in soup.find_all('li')[1:]]

def get_news_articles(url):
    url = 'https://inshorts.com/en/read'
    cats = get_cats(url)
    output = []
    for cat in cats:
        cat_url = url + '/' + cat
        cat_soup = BeautifulSoup(requests.get(cat_url).content)
        cat_titles = [title.text for title in cat_soup.find_all('span', itemprop='headline')]
        cat_bodies = [body.text for body in cat_soup.find_all('div', itemprop='articleBody')]
        caticles = [{'title': title,
                     'content': body,
                     'category': cat} for title, body in zip(cat_titles, cat_bodies)]
        output.extend(caticles)
    filename = "news_articles.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    
    # if file not available locally, acquire data from URL
    # and write it as csv locally for future use
    else:
        df = pd.DataFrame(output)
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename, index=False)

    # Return the dataframe to the calling code
        return df 