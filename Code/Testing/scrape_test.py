import requests
from bs4 import BeautifulSoup
from datetime import date

url = 'https://www.bbc.com/sport/football/champions-league/scores-fixtures'

today = date.today()
today = today.strftime("%d %B")

def scrape_ucl(url:str=url, today:str=today):
    response = requests.get(url)
    html_content = response.content

    soup = BeautifulSoup(html_content, 'lxml')

    days = soup.select(".ssrcss-7k0bq5-HeaderWrapper.e4zdov50")
    if days:
        for i in days:
            game_date = i.select(".ssrcss-m22vfq-GroupHeader.ejnn8gi5")[0].text
            game_date = game_date[(game_date.find(" ")+1):]
            game_date = game_date[:(game_date.find(" ")-2)] + game_date[(game_date.find(" ")):]
            if today != game_date:
                continue
            if today == game_date:
                phase = i.select(".ssrcss-v43q73-SecondaryHeading.ejnn8gi4")[0].text
                matches = i.select(".ssrcss-1bjtunb-GridContainer.e1efi6g55")
                match_info = []
                for j in matches:
                    teams = j.select(".ssrcss-1p14tic-DesktopValue.emlpoi30")
                    team_home = teams[0].text
                    team_away = teams[1].text
                    score_home = j.select(".ssrcss-qsbptj-HomeScore.e56kr2l2")[0].text
                    score_away = j.select(".ssrcss-fri5a2-AwayScore.e56kr2l1")[0].text
                    agg_score = j.select(".ssrcss-xxixzn-AggregateScore.ep2tsh20")[0].text
                    match_info.append([team_home, score_home, team_away, score_away, agg_score])
                return today, phase, match_info
            return None
                

if __name__ == "__main__":
    print(scrape_ucl())