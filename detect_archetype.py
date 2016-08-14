import random

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import sys

from sklearn.cluster import DBSCAN

import pickle
import collections
import xmltodict

from flask import Flask, request
from flask_restful import Resource, Api, reqparse


class DeckClassifierAPI(Resource):

    @classmethod
    def make_api(cls, classifier):
        cls.classifier = classifier
        return cls

    def get(self):
        return None

    def post(self):
        # deck = request.form['deck']
        # x = self.deck_to_vector([deck])

        x = np.array([0] * 669)
        for i in range(30):
            x[random.randint(0, len(x))] += 1
        x = x.reshape(1, -1)
        y = self.classifier.dbscan_predict(x)
        if y == -1:
            return "unknown", 201
        else:
            return self.lookup_table[y], 201


class DeckClassifier(object):
    CLASSIFIER_CACHE = "classifier.pkl"

    def __init__(self):
        self.app = Flask(__name__)
        app_api = Api(self.app)
        classifier_api = DeckClassifierAPI.make_api(self)
        app_api.add_resource(classifier_api, "/")  # "/api/v0.1/detect_archetype")

        DATA_FILE = "10kdecks.pkl"
        eps = 6  # int(sys.argv[1])
        min_samples = 10  # int(sys.argv[2])
        self.maybe_train_classifier(DATA_FILE, eps, min_samples)

    def run(self):
        self.app.run()

    def deck_to_vector(self, decks):
        data = []
        for deck in decks:
            datapoint = [0] * len(self.lookup_table)
            for card in deck:
                card_dimension = self.lookup_table.index(card)
                datapoint[card_dimension] = deck[card]
            data.append(datapoint)
        return np.array(data)

    def load_data_from_file(self, file_name):
        decks, deck_classes, deck_names = self.load_decks_from_file(file_name)
        self.lookup_table = list({card for deck in decks for card in deck})

        data = self.deck_to_vector(decks)
        return data, deck_classes, deck_names

    def train_classifier(self, data, eps, min_samples):
        self.pca = PCA(n_components=50)
        data = self.pca.fit_transform(data)

        db_model = DBSCAN(eps=eps, min_samples=min_samples, metric="manhattan")
        db_model.fit(data)
        return db_model

    def maybe_train_classifier(self, data_file, eps, min_samples):
        try:
            with open(self.CLASSIFIER_CACHE, 'rb') as d:
                classifier, self.lookup_table, self.deck_classes, self.deck_names, self.pca = pickle.load(d)
        except IOError:
            data, self.deck_classes, self.deck_names = self.load_data_from_file(data_file)
            classifier = self.train_classifier(data, eps, min_samples)
            with open(self.CLASSIFIER_CACHE, 'wb') as d:
                pickle.dump((classifier, self.lookup_table, self.deck_classes, self.deck_names, self.pca), d)
        self.classifier = classifier

    # consider the newest decks more important
    def dbscan_predict(self, x_new, distance=sklearn.metrics.pairwise.manhattan_distances):
        x_new = self.pca.transform(x_new)
        # Find a core sample closer than EPS
        for index, x_core in enumerate(self.classifier.components_):
            if distance(x_new, x_core.reshape(1, -1)) < self.classifier.eps:
                return self.classifier.labels_[self.classifier.core_sample_indices_[index]]
        return -1

    def load_decks_from_file(self, file_name):
        decks = []
        deck_names = []
        deck_classes = []
        with open(file_name, 'rb') as f:
            try:
                while True:
                    d, c, n = pickle.load(f)
                    if "arena" in n.lower():
                        continue
                    # print(n)
                    deck_names.append(n)
                    deck_classes.append(c)
                    decks.append(d)
            except EOFError:
                pass
        return decks, deck_classes, deck_names

    def import_decks_from_hdt(self, file_name):
        with open(file_name) as f:
            tree = xmltodict.parse(f.read())
        # cards.db.initialize()
        decks = []
        deck_names = []
        deck_classes = []

        for d in tree['Decks']['Deck']:
            # print(d['Name'],d['Class'])
            deck = collections.Counter()
            for c in d['Cards']['Card']:
                deck[c['Id']] = c['Count']
            decks.append(deck)
            deck_names.append(d['Name'])
            deck_classes.append(d['Class'])
        return decks, deck_classes, deck_names

    def print_data(self, data, deck_names, clusters):
        sets = collections.defaultdict(list)
        for (i, name) in enumerate(deck_names):
            sets[clusters[i]].append(name)
        groups = []
        for cluster_number in sets:
            groups.append(sets[cluster_number])

        for group in sorted(groups, key=len):
            print(len(group), group, "\n")
        print("found {} clusters".format(len(set(clusters))))


if __name__ == '__main__':
    DeckClassifier().run()
