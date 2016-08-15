import os
import random

import nltk
from nltk.util import ngrams
from nltk import FreqDist
import pprint

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
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

    def get_placeholder_deck(self):
        x = np.array([0] * 669)
        for i in range(30):
            x[random.randint(0, len(x))] += 1
        return x.reshape(1, -1)

    def post(self):
        # deck = request.form['deck']
        # x = self.deck_to_vector([deck])
        x = self.get_placeholder_deck()
        label = self.classifier.dbscan_predict(x, 'Warrior')
        return label, 201


class DeckClassifier(object):
    CLASSIFIER_CACHE = "klass_classifiers.pkl"

    def __init__(self):
        self.app = Flask(__name__)
        app_api = Api(self.app)
        classifier_api = DeckClassifierAPI.make_api(self)
        app_api.add_resource(classifier_api, "/")  # "/api/v0.1/detect_archetype")
        self.dimension_to_card_name = None
        self.cluster_names = {}

        DATA_FILE = "100kdecks.pkl"
        eps = 6  # int(sys.argv[1])
        min_samples = 10  # int(sys.argv[2])
        self.maybe_train_classifier(DATA_FILE, eps, min_samples)

    def run(self):
        self.app.run()

    def load_decks_from_file(self, file_name):
        decks = collections.defaultdict(list)
        deck_names = collections.defaultdict(list)
        with open(file_name, 'rb') as f:
            try:
                while True:
                    d, c, n = pickle.load(f)
                    if "arena" in n.lower() or c != "Warrior":
                        continue
                    deck_names[c].append(n)
                    decks[c].append(d)
            except EOFError:
                pass
        return decks, deck_names

    def deck_to_vector(self, decks):
        data = {klass: None for klass in decks.keys()}
        for klass in decks:
            klass_data = []
            for deck in decks[klass]:
                datapoint = [0] * len(self.dimension_to_card_name)
                for card in deck:
                    card_dimension = self.dimension_to_card_name.index(card)
                    datapoint[card_dimension] = deck[card]
                klass_data.append(datapoint)
            data[klass] = np.array(klass_data)
        return data

    def load_data_from_file(self, file_name):
        decks, deck_names = self.load_decks_from_file(file_name)
        # TODO: use vectorizer
        self.dimension_to_card_name = list({card for klass in decks for deck in decks[klass] for card in deck})

        data = self.deck_to_vector(decks)
        return data, deck_names

    def train_classifier(self, data, eps, min_samples):
        self.pca = PCA(n_components=50)
        data = self.pca.fit_transform(data)

        db_model = DBSCAN(eps=eps, min_samples=min_samples, metric="manhattan")
        db_model.fit(data)
        return db_model

    def get_decks_in_cluster(self, classifier, klass, cluster_index):
        decks = []
        clusters = classifier[klass].labels_
        for i in range(len(clusters)):
            if clusters[i] == cluster_index:
                decks.append(self.deck_names[klass][i])
        return decks


    def maybe_train_classifier(self, data_file, eps, min_samples):
        try:
            raise IOError()
            with open(self.CLASSIFIER_CACHE, 'rb') as d:
                state_tuple = pickle.load(d)
                klass_classifier, self.dimension_to_card_name,\
                self.deck_names, self.pca, self.cluster_names = state_tuple
        except IOError:

            klass_classifier = {}
            data, self.deck_names = self.load_data_from_file(data_file)

            for klass in data:
                klass_classifier[klass] = self.train_classifier(data[klass], eps, min_samples)

            for klass in klass_classifier:
                self.cluster_names[klass] = self.name_clusters(klass_classifier[klass], self.deck_names[klass], klass)

            print("train results:")
            for klass, cluster_names in self.cluster_names.items():
                print(klass, ":")
                for cluster_index, cluster_name in cluster_names.items():
                    decks = self.get_decks_in_cluster(klass_classifier, klass, cluster_index)
                    print(len(decks), "\t", cluster_name, "\t", decks)

            with open(self.CLASSIFIER_CACHE, 'wb') as d:
                state_tuple = (klass_classifier, self.dimension_to_card_name,
                               self.deck_names, self.pca, self.cluster_names)
                pickle.dump(state_tuple, d)

        self.klass_classifiers = klass_classifier

    # consider the newest decks more important
    def dbscan_predict(self, x_new, klass, distance=sklearn.metrics.pairwise.manhattan_distances):
        x_new = self.pca.transform(x_new)
        # Find a core sample closer than EPS
        core_components = self.klass_classifiers[klass].components_
        eps = self.klass_classifiers[klass].eps
        labels = self.klass_classifiers[klass].labels_
        core_samples_indexes = self.klass_classifiers[klass].core_sample_indices_

        prediction = -1
        for index, x_core in enumerate(core_components):
            if distance(x_new, x_core.reshape(1, -1)) < eps:
                prediction = labels[core_samples_indexes[index]]
                break
        return self.cluster_names[prediction]

    def print_data(self, deck_names, clusters):
        sets = collections.defaultdict(list)
        for (i, name) in enumerate(deck_names):
            sets[clusters[i]].append(name)
        groups = []
        for cluster_number in sets:
            groups.append(sets[cluster_number])

        for group in sorted(groups, key=len):
            print(len(group), group, "\n")
        print("found {} clusters".format(len(set(clusters))))

    def name_clusters(self, classifier, deck_names, klass):
        labels = classifier.labels_
        cluster_decknames = collections.defaultdict(list)
        cluster_names = {}

        for (i, name) in enumerate(deck_names):
            cluster_decknames[labels[i]].append(name)

        for cluster_index, decknames in cluster_decknames.items():
            if cluster_index == -1:
                cluster_name = "UNKNOWN"
            else:
                klass_ = klass.lower()
                decknames = [n.lower().replace(klass_, "") for n in decknames if n.lower()]
                stopwords = set(nltk.corpus.stopwords.words('english'))

                # Freq
                tokenizer = nltk.RegexpTokenizer(r'\w+')
                words = [word for name in decknames for word in tokenizer.tokenize(name) if word not in stopwords]
                fdist = FreqDist(words)

                """
                # ngrams
                twograms = []
                for deckname in decknames:
                    deck_tokens = tokenizer.tokenize(deckname)
                    ngs = list(ngrams(deck_tokens, 2))
                    twograms.extend([" ".join(ng) for ng in ngs])
                fdist2 = FreqDist(twograms)
                cluster_name = "|".join([dn[0] for dn in fdist2.most_common(2)])
                cluster_name += " || " + "|".join([dn[0] for dn in fdist.most_common(3)])
                """

                keywords = fdist.most_common(5)
                cluster_name = keywords[0][0]
                cutoff = 0.5 * keywords[0][1]
                for dn in keywords[1:]:
                    if dn[1] > cutoff:
                        cluster_name += " " + dn[0]

            cluster_names[cluster_index] = cluster_name
        return cluster_names


if __name__ == '__main__':
    DeckClassifier().run()
