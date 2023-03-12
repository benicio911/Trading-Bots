import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData

# Set the connection parameters
host = "LIZVAI"
port = 9969
database = "postgres"
user = "LIZVAI"
password = "Selaromor00"

# Connect to the PostgreSQL server
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{database}')
metadata = MetaData()
mytable = Table('mytable', metadata, Column('id', Integer, primary_key=True), Column('title', String), Column('link', String))
metadata.create_all(engine)

# Scrape data from the website
url = 'https://stackoverflow.com/questions/tagged/python'
res = requests.get(url)
soup = BeautifulSoup(res.text, 'html.parser')
for question in soup.select('div.question-summary'):
    title = question.select_one('h3 a').get_text()
    link = 'https://stackoverflow.com' + question.select_one('h3 a')['href']
    
    # Insert the data into the database
    conn = engine.connect()
    conn.execute(mytable.insert().values(title=title, link=link))
    conn.close()