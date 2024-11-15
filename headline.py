import requests
from bs4 import BeautifulSoup
from datetime import datetime
from classify import classify

# URL of the website to scrape
main_url = 'https://www.philstar.com/'


def get_soup(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

def get_headlines(url):
    soup = get_soup(url)

    # Find all occurrences of div with class="tiles late ribbon-cont"
    ribbon_containers = soup.find_all('div', class_='tiles late ribbon-cont')

    headlines = []

    # Iterate through each ribbon container
    for ribbon_cont in ribbon_containers:
        # Find all <a> tags within the specified CSS selector
        links = ribbon_cont.select('.ribbon .ribbon_content .ribbon_title a[href]')
        
        # Extract and print the href attribute of each <a> tag
        for link in links:
            href = link['href']
            headlines.append(href)
        
    return headlines

def headline_data(urls):
    headline_data = []

    for url in urls:
        soup = get_soup(url)
    
        # Find and extract headline title if present
        title_div = soup.find('div', class_='article__title')

        if not title_div:
            continue
        
        title = title_div.find('h1').text.strip()

        # Find and extract date and time
        date_time_str = soup.find('div', class_='article__date-published').text.strip()
        
        # Format date and time
        formatted_date_time = datetime.strptime(date_time_str, '%B %d, %Y | %I:%M%p')

        # Classify disaster type using trained model
        disaster_type = classify(title)['prediction']

        headline_data.append({
            'title': title,
            'link': url,
            'datetime': formatted_date_time,
            'disasterType': disaster_type
        })
    
    return headline_data

def data():
    links = get_headlines(main_url)    
    article_data = headline_data(links)

    return article_data
