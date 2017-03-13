import json
import numpy as np

from flask import Flask
from flask_restful import Resource, Api
from flask import request

from detect_archetype import DeckClassifier, vectorizer_1hot


class DeckClassifierAPI(Resource):
    @classmethod
    def make_api(cls, classifier):
        cls.classifier = classifier
        return cls

    def post(self):
        deck = json.loads(request.form['deck'])
        klass = request.form['klass']

        archetype_deck, prob, ignored_cards = self.classifier.classify(deck, klass)
        return (archetype_deck, (prob*100).transpose().tolist()[0], ignored_cards), 201


class DeckClassifierWrapper(object):
    CLASSIFIER_CACHE = "klass_classifiers.pkl"

    def __init__(self):
        self.app = Flask(__name__)
        app_api = Api(self.app)
        classifier_api = DeckClassifierAPI.make_api(self)
        app_api.add_resource(classifier_api, "/")  # "/api/v0.1/detect_archetype")

        model_path = "models/kara_classifier_state"
        print("loading model")
        self.classifier = DeckClassifier()
        self.classifier.load_state_from_file(model_path)
        print("calc canonical decks [REMOVEME]")
        self.classifier.calculate_canonical_decks()
        print("done")

    def classify(self, deck, klass):
        # klass = int(''.join(filter(str.isdigit, klass)))
        # hero_to_class = ['UNKNOWN', 'WARRIOR', 'SHAMAN', 'ROGUE', 'PALADIN', 'HUNTER', 'DRUID', 'WARLOCK', 'MAGE', 'PRIEST']
        # klass = hero_to_class[klass]
        predicted_deck_report, prob, ignored_cards = self.classifier.predict(deck, klass.upper())
        return predicted_deck_report, prob / np.sum(prob), ignored_cards

    def run(self):
        self.app.run(host="0.0.0.0", port=31337)

if __name__ == '__main__':
    DeckClassifierWrapper().run()
