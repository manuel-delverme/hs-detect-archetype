import requests
import json
from fireplace.utils import random_draft
from fireplace import cards
from hearthstone.enums import CardClass
from hearthstone import cardxml

def classify(y, x):
	for klass_name in ["ROGUE", "MAGE", "WARRIOR", "HUNTER", "PRIEST", "PALADIN", "WARLOCK", "SHAMAN", "DRUID"]:
		if klass_name.lower() in y.lower():
			klass = klass_name
	resp = requests.post('http://88.99.185.31:31337', data={'deck': json.dumps(x), 'klass': klass})
	try:
		deck, probs = json.loads(resp.text)
	except:
		print(resp.text)
	return deck, probs

def names_to_ids(name_to_id, names):
	ids = []
	for name in names:
		if name not in name_to_id:
			print("skipping", card.name)
		else:
			ids.append(name_to_id[name])
	return ids

def main():
	name_to_id = {}
	for card in card_db.values():
		name_to_id[card.name] = card.id

	test_set = []
	with open("scrape-decks/test_set.json", 'r') as f:
		for row in f:
			test_set.append(json.loads(row))

	for test_point in test_set:
		y, x = test_point
		if y != "UNKNOWN":
			print("sending", y, sorted(set(x)))
			report, probs = classify(y, names_to_ids(name_to_id, x))
			print("confidences", probs)
			print("canonical deck:")
			print(report[-1])
			# for row in report: print(row)
			input()

"""
# deck = random_draft(CardClass.WARRIOR)
# deck = sorted(deck)

# matches = []
# only_canon = []
# only_deck = []
# for card in canon_deck:
#     if card in deck:
#         deck.remove(card)
#         matches.append(card)
#     else:
#         only_canon.append(card)
# only_deck = deck
# 
# print("\tmatches:")
# for card in matches:
#     print("\t", card_db[card])
# 
# print("only in canon:")
# for card in only_canon:
#     print(card_db[card])
# 
# print("\t\tonly in my deck:")
# for card in only_deck:
#     print("\t\t", card_db[card])
"""

try:
	firstRun
except:
	cards.db.initialize()
	card_db, _ = cardxml.load()
	firstRun = False
main()
