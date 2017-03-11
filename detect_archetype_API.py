import json

from flask import Flask
from flask_restful import Resource, Api
from flask import request

from detect_archetype import DeckClassifier


class DeckClassifierAPI(Resource):
    @classmethod
    def make_api(cls, classifier):
        cls.classifier = classifier
        return cls

    def post(self):
        deck = json.loads(request.form['deck'])
        klass = request.form['klass']

        canonical_deck = self.classifier.classify(deck, klass)
        return (canonical_deck), 201


class DeckClassifierWrapper(object):
    CLASSIFIER_CACHE = "klass_classifiers.pkl"

    def __init__(self):
        self.app = Flask(__name__)
        app_api = Api(self.app)
        classifier_api = DeckClassifierAPI.make_api(self)
        app_api.add_resource(classifier_api, "/")  # "/api/v0.1/detect_archetype")

        # train_data_path = "datasets/Deck_List_Training_Data.csv"
        kara_data = "datasets/kara_data.json"

        print("classifier dataset")
        classifier = DeckClassifier()
        loaded_data, popular_decks, semi_popular_decks = classifier.load_decks_from_json_file(kara_data)
        self.classifier.fit_transform(loaded_data, popular_decks, semi_popular_decks)
        self.classifier.calculate_canonical_decks()
        print("done")

        # decks = self.classifier.load_decks_from_file(dataset_path)
        # klass = int(''.join(filter(str.isdigit, klass)))
        # hero_to_class = ['UNKNOWN', 'WARRIOR', 'SHAMAN', 'ROGUE', 'PALADIN', 'HUNTER', 'DRUID', 'WARLOCK', 'MAGE', 'PRIEST']
        # klass = hero_to_class[klass]
        # results = []
        # for deck in klass_decks:
        #     predicted_deck, prob = classifier.predict_update(deck, klass)
        #     archetype_number = prob.argmax()
        #     results_writer.writerow([archetype_number] + deck)

    def classify(self, deck, klass):
        print("eval ", klass, deck)
        """
        klass = int(''.join(filter(str.isdigit, klass)))
        hero_to_class = ['UNKNOWN', 'WARRIOR', 'SHAMAN', 'ROGUE', 'PALADIN', 'HUNTER', 'DRUID', 'WARLOCK', 'MAGE', 'PRIEST']
        klass = hero_to_class[klass]
        predicted_deck, prob = self.classifier.predict(deck, klass)
        archetype_number = prob.argmax()
        canonical_deck = self.classifier.canonical_decks[archetype_number]
        return canonical_deck
        """

    def run(self):
        self.app.run(host="0.0.0.0", port=31337)

if __name__ == '__main__':
    DeckClassifierWrapper().run()
