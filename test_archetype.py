import requests
import json
from fireplace.utils import random_draft
from fireplace import cards
from hearthstone.enums import CardClass
from hearthstone import cardxml

cards.db.initialize()
card_db, _ = cardxml.load()

deck = random_draft(CardClass.WARRIOR)
deck = sorted(deck)
resp = requests.post('http://localhost:31337', data={'deck': json.dumps(deck), 'klass': "Warrior"})
name, canon_deck = json.loads(resp.text)
print("name:", name)

matches = []
only_canon = []
only_deck = []
for card in canon_deck:
    if card in deck:
        deck.remove(card)
        matches.append(card)
    else:
        only_canon.append(card)
only_deck = deck

print("\tmatches:")
for card in matches:
    print("\t", card_db[card])

print("only in canon:")
for card in only_canon:
    print(card_db[card])

print("\t\tonly in my deck:")
for card in only_deck:
    print("\t\t", card_db[card])
