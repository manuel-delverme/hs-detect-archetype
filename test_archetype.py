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
            print("skipping", name)
        else:
            ids.append(name_to_id[name])
    return ids

def names_to_cards(name_to_card, names):
    cards = []
    for name in names:
        if name not in name_to_card:
            print("skipping", name)
        else:
            cards.append(name_to_card[name])
    return cards

def load():
    data = []
    name_to_card = {}
    for card in card_db.values():
        name_to_card[card.name] = card

    test_set = []
    with open("scrape-decks/test_set.json", 'r') as f:
        for row in f:
            test_set.append(json.loads(row))

    for test_point in test_set:
        y, x = test_point
        if y == "UNKNOWN": continue

        x = names_to_cards(name_to_card, x)
        data.append((y, x))
    return data

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
            print("sending", y)
            # print("sending", y, sorted(set(x)))
            report, probs = classify(y, names_to_ids(name_to_id, x))
            print("confidences", probs)
            print("canonical deck:")
            for k,v in report.items():
                print(k, ":",  v)
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

# x = load()
# aggro = []
# control = []
# tempo = []
# skipped = set()
# for name, deck in x:
#     if "tempo" in name.lower():
#         tempo.append(deck)
#     elif "control" in name.lower() or "zoth" in name.lower():
#         control.append(deck)
#     elif "aggro" in name.lower():
#         aggro.append(deck)
#     else:
#         skipped.add(name)
# 
# from collections import defaultdict
# 
# def get_dist(decks):
#     mana_curve = defaultdict(int)
#     for deck in decks:
#         for card in deck:
#             mana_curve[card.cost] += 1
#     dist = []
#     for turn in range(10):
#         dist.append(mana_curve[turn] / sum(list(dict(mana_curve).values())))
#     return dist
# 
# aggro_dist = get_dist(aggro)
# tempo_dist = get_dist(tempo)
# control_dist = get_dist(control)
# del aggro
# del tempo
# del control

main()
