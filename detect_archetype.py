import os
import random
import redis
import seaborn as sns

import nltk
from nltk.util import ngrams
from nltk import FreqDist
import pprint
import matplotlib.pyplot as plt

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import sys

from sklearn.cluster import DBSCAN
import hdbscan

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

    @staticmethod
    def get_placeholder_deck():
        x = np.array([0] * 669)
        for i in range(30):
            x[random.randint(0, len(x))] += 1
        return x.reshape(1, -1)

    def post(self):
        # deck = request.form['deck']
        # x = self.deck_to_vector([deck])
        x = self.get_placeholder_deck()
        klass = 'Warrior'
        index = self.classifier.dbscan_predict(x, klass)
        name = self.classifier.cluster_names[index]
        races = self.classifier.cluster_races[index]
        categories = self.classifier.cluster_categories[index]

        return name, races, categories, 201


def print_data(deck_names, clusters):
    sets = collections.defaultdict(list)
    for (i, name) in enumerate(deck_names):
        sets[clusters[i]].append(name)
    groups = []
    for cluster_number in sets:
        groups.append(sets[cluster_number])

    for group in sorted(groups, key=len):
        print(len(group), group, "\n")
    print("found {} clusters".format(len(set(clusters))))


class DeckClassifier(object):
    CLASSIFIER_CACHE = "klass_classifiers.pkl"

    def __init__(self):
        self.app = Flask(__name__)
        app_api = Api(self.app)
        classifier_api = DeckClassifierAPI.make_api(self)
        app_api.add_resource(classifier_api, "/")  # "/api/v0.1/detect_archetype")

        self.test_labels = []
        self.cluster_names = {}
        self.deck_names = {}
        self.pca = None
        self.klass_classifiers = {}
        self.dimension_to_card_name = {}

        DATA_FILE = "100kdecks.pkl"
        REDIS_ADDR = "localhost"
        REDIS_PORT = 6379
        REDIS_DB = 0
        eps = 2 * 2  # int(sys.argv[1])
        min_samples = 15 # int(sys.argv[1])
        self.redis_db = None  # redis.StrictRedis(host=REDIS_ADDR, port=REDIS_PORT, db=REDIS_DB)
        self.maybe_train_classifier(DATA_FILE, eps, min_samples)

    def run(self):
        # self.app.run()
        pass

    @staticmethod
    def load_decks_from_file(file_name):
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

    def plot_data(self, data, db, cluster_names):
        # notes = []
        plt.axis("equal")
        model = TSNE(n_components=2, random_state=0)

        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # n_clusters_ = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

        data = self.pca.transform(data[:1000])
        embed = model.fit_transform(data)

        uniq_labels = set(db.labels_)
        uniq_labels.remove(-1)
        cluster_labels = []

        for cluster_index in db.labels_:
            cluster_labels.append(cluster_names[cluster_index] + str(cluster_index))

        already_annotated = []
        for label, x, y in zip(cluster_labels, *np.transpose(embed)):
            if label not in already_annotated:
                already_annotated.append(label)
                plt.annotate(label, xy=(x, y), fontsize=20)

        for deck_name, x, y in zip(self.deck_names['Warrior'], *np.transpose(embed)):
            plt.annotate(deck_name, xy=(x, y), fontsize=5)
        # adjust_text(notes, arrowprops=dict(arrowstyle="->", color='r'), force_text=0.25, lim=10)
        # plt.title("thres %f; clusters: %d" % (thresh, len(set(clusters))))

        color_palette = sns.color_palette('deep', db.labels_.max() + 1)
        plt.title("min_cluster_size %f; clusters: %d" % (0, db.labels_.max() + 1))
        for i, prop in enumerate(db.probabilities_):
            if not 0 <= prop <= 1:
                db.probabilities_[i] = 0

        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in db.labels_]
        cluster_member_colors = [sns.desaturate(x, p) for x, p in
                                 zip(cluster_colors, db.probabilities_)]

        tsne_model = TSNE(n_components=2, random_state=0)
        embed = tsne_model.fit_transform(data[:3000])
        # plt.scatter(*embed.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.5)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i, xy in enumerate(embed):
            x, y = xy
            c = cluster_member_colors[i]
            ax1.scatter(x, y, c=c, marker='o', s=50, alpha=0.25)

        print_data(self.deck_names['Warrior'], db.labels_)
        plt.show()

    def deck_to_vector(self, decks):
        data = {klass: None for klass in decks.keys()}
        for klass in decks:
            klass_data = []
            for deck in decks[klass]:
                datapoint = [0] * len(self.dimension_to_card_name[klass])
                for card in deck:
                    card_dimension = self.dimension_to_card_name[klass].index(card)
                    datapoint[card_dimension] = deck[card]
                klass_data.append(datapoint)
            data[klass] = np.array(klass_data)
        return data

    def load_data_from_file(self, file_name):
        decks, deck_names = self.load_decks_from_file(file_name)

        # TODO: use vectorizer
        for klass in decks.keys():
            self.dimension_to_card_name[klass] = list({card for deck in decks[klass] for card in deck})
            data = self.deck_to_vector(decks)

        return data, deck_names

    def train_classifier(self, data, eps, min_samples):
        self.pca = PCA(n_components=50)
        data = self.pca.fit_transform(data)

        # db_model = DBSCAN(eps=eps, min_samples=min_samples, metric="manhattan")
        model = hdbscan.HDBSCAN(metric="manhattan", min_cluster_size=40, min_samples=min_samples)
        model.fit(data)
        return model

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
            if self.redis_db:
                self.klass_classifiers = self.redis_db.get('klass_classifier')
                self.dimension_to_card_name = self.redis_db.get('dimension_to_card_name')
                self.deck_names = self.redis_db.get('deck_names')
                self.pca = self.redis_db.get('pca')
                self.cluster_names = self.redis_db.get('cluster_names')
            else:
                with open(self.CLASSIFIER_CACHE, 'rb') as d:
                    state_tuple = pickle.load(d)
                    self.klass_classifiers, self.dimension_to_card_name, \
                    self.deck_names, self.pca, self.cluster_names = state_tuple
        except IOError:
            loaded_data, self.deck_names = self.load_data_from_file(data_file)
            data, test_data, self.test_labels = self.split_dataset(loaded_data)

            for klass in data:
                self.klass_classifiers[klass] = self.train_classifier(data[klass], eps, min_samples)

            for klass in self.klass_classifiers:
                self.cluster_names[klass], _, _ = self.name_clusters(self.klass_classifiers[klass],
                                                                     self.deck_names[klass], klass)
                # self.plot_data(data[klass], self.klass_classifiers[klass], self.cluster_names[klass])

            print("train results:")
            accuracy = self.test_accuracy(classifier, test_data, test_labels)
            print(accuracy)
            # for klass, cluster_names in self.cluster_names.items():
            #    print(klass, ":")
            #    for cluster_index, cluster_name in cluster_names.items():
            #        decks = self.get_decks_in_cluster(self.klass_classifiers, klass, cluster_index)
            #        print(len(decks), "\t", cluster_name, "\t", decks)

            with open(self.CLASSIFIER_CACHE, 'wb') as d:
                state_tuple = (self.klass_classifiers, self.dimension_to_card_name,
                               self.deck_names, self.pca, self.cluster_names)
                pickle.dump(state_tuple, d)

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

        return prediction

    def test_accuracy(self, classifier, test_data, test_labels):
        hits = 0
        for (deck, klass, target_label) in zip(test_data, test_labels):
            label = self.dbscan_predict(deck, klass)
            if label == target_label:
                hits += 1
        return float(hits) / len(test_data)

    @staticmethod
    def name_clusters(classifier, deck_names, klass):
        labels = classifier.labels_
        cluster_decknames = collections.defaultdict(list)
        cluster_names = {}
        cluster_races = {}
        cluster_categories = {}
        pRaces = None
        pCategories = None

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

                keywords = fdist.most_common(10)
                cluster_name = ""
                naming_cutoff = 0.5 * keywords[0][1]

                categories = ['aggro', 'combo', 'control', 'fatigue', 'midrange', 'ramp', 'tempo', 'token']
                pCategories = {}
                for cat in categories:
                    pCategories[cat] = fdist[cat] / len(deck_names)

                pRaces = {}
                races = ['murloc', 'dragon', 'pirate', 'mech', 'beast']
                for race in races:
                    pRaces[race] = fdist[race] / len(deck_names)

                for dn in keywords:
                    if dn[1] > naming_cutoff:
                        cluster_name += " " + dn[0]

            cluster_names[cluster_index] = cluster_name.lstrip()
            cluster_races[cluster_index] = pRaces
            cluster_categories[cluster_index] = pCategories
        return cluster_names, cluster_races, cluster_categories

    def split_dataset(self, loaded_data):
        known_archetypes = {'Warrior': ["patron", "control", "pirate"]}
        test_dataset = {}
        test_labels = {}
        for klass in loaded_data.keys():
            test_dataset[klass] = []
            test_labels[klass] = []
            test_data_size = int(len(loaded_data[klass])*0.02)
            while len(test_dataset[klass]) < test_data_size:
                index = random.randint(0, len(loaded_data[klass]))
                name = self.deck_names[klass][index].lower().replace(klass.lower(), "")
                if name in known_archetypes[klass]:
                    test_dataset[klass].append(loaded_data[klass][index])
                    test_labels[klass].append(name)
                    del self.deck_names[klass][index]
                    # THIS IS NOT INPLACE.... TODO: speed up
                    loaded_data[klass] = np.delete(loaded_data[klass], index, axis=0)
        return loaded_data, test_dataset, test_labels


if __name__ == '__main__':
    DeckClassifier().run()
