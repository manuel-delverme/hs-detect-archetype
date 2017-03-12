import requests
import bs4
import sys
import time
import json

def url_to_deck(deck_url):
	resp = requests.get(deck_url)
	soup = bs4.BeautifulSoup(resp.text, "html.parser")
	archetype_tags = soup.select("div.archetype-box > div.pull-left-responsive > a")
	if archetype_tags:
		archetype_name = archetype_tags[0].text
	else:
		archetype_name = "UNKNOWN"
	cards = soup.select(".card-name")
	deck = []
	for card in cards:
		count = card.parent.parent.select(".card-count")[0].text
		for _ in range(int(count)):
			card_name = card.text
			deck.append(card_name)
	return archetype_name, deck


test_set = []
with open("test_set.json", "w") as fout:
	for page in range(1, 1000):
		search_url = "http://www.hearthstonetopdecks.com/deck-category/constructed-seasons/season-29/page/{}/?st&class_id&style_id&t_id&f_id=715&pt_id=1&sort=new".format(page)
		print("page", page)
		resp = requests.get(search_url)
		soup = bs4.BeautifulSoup(resp.text, "html.parser")
		for link in soup.select("#deck-list > tbody > tr > td > h4 > a"):
			label, deck = url_to_deck(link['href'])
			print(link['href'], label)
			json.dump((label, deck), fout)
			fout.write("\n")
			fout.flush()
			time.sleep(0.5)
