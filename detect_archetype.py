import collections
import json
import pickle
import random
import sys

import nltk
import numpy as np
import multiprocessing
import sklearn
from flask import Flask
from flask_restful import Resource, Api
from flask import request

from hearthstone import cardxml
from nltk import FreqDist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


class DeckClassifierAPI(Resource):
    @classmethod
    def make_api(cls, classifier):
        cls.classifier = classifier
        return cls

    @staticmethod
    def get_placeholder_deck():
        x = np.array([0] * 669)
        for i in range(30):
            x[random.randint(0, len(x))] += 1
        return x.reshape(1, -1)

    def post(self):
        deck = json.loads(request.form['deck'])
        klass = request.form['klass']

        lookup = self.classifier.dimension_to_card_name
        x = self.classifier.deck_to_vector([deck], lookup[klass])

        index = int(self.classifier.dbscan_predict(x, klass))
        name = self.classifier.cluster_names[klass][index]
        # races = self.classifier.cluster_races[klass][index]
        # categories = self.classifier.cluster_categories[klass][index]
        canonical_deck = self.classifier.canonical_decks[klass][index]

        return (name, canonical_deck), 201


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
        self.card_db, _ = cardxml.load()
        self.cluster_names = {}
        self.pca = {}
        self.klass_classifiers = {}
        self.dimension_to_card_name = {}
        self.canonical_decks = {}

        DATA_FILE = "100kdecks.pkl"
        REDIS_ADDR = "localhost"
        REDIS_PORT = 6379
        REDIS_DB = 0
        eps = float(sys.argv[1])
        min_samples = int(sys.argv[2])
        self.redis_db = None  # redis.StrictRedis(host=REDIS_ADDR, port=REDIS_PORT, db=REDIS_DB)
        self.maybe_train_classifier(DATA_FILE, eps, min_samples)

    def run(self):
        self.app.run()

    @staticmethod
    def load_decks_from_file(file_name):
        decks = collections.defaultdict(list)
        deck_names = collections.defaultdict(list)
        with open(file_name, 'rb') as f:
            try:
                while True:
                    d, c, n = pickle.load(f)
                    if "arena" in n.lower() or c == "":
                        continue
                    deck_names[c].append(n)
                    decks[c].append(d)
            except EOFError:
                pass
        return decks, dict(deck_names)

    def test_accuracy(self, test_data, test_labels, klass):
        hits = 0
        unknowns = 0
        for (deck, target_label) in zip(test_data, test_labels):
            label = self.dbscan_predict(deck, klass)
            label = self.cluster_names[klass][label[0]]
            if target_label in label:
                hits += 1
            else:
                if label == "UNKNOWN":
                    unknowns += 1
                else:
                    # print("predicted [", label, "] was [", target_label, end=' ]\t')
                    for cluster_index, prob in zip(self.klass_classifiers[klass].classes_,
                                                   self.dbscan_explain(deck, klass)[0]):
                        prob = int(prob * 100)
                        if prob != 0:
                            pass
                            # print(self.cluster_names[klass][cluster_index], prob, "%", end="; ")
                            # print("")
        # print("predicted UNKNOWN", unknowns, "times")
        return float(hits) / len(test_data)

    def deck_to_vector(self, decks, dimension_to_card_name):
        klass_data = []
        for deck in decks:
            datapoint = np.zeros(len(dimension_to_card_name))
            for card in deck:
                try:
                    card_dimension = dimension_to_card_name.index(card)
                except ValueError:
                    continue
                if isinstance(deck, list):
                    datapoint[card_dimension] += 1
                else:
                    datapoint[card_dimension] = deck[card]
            klass_data.append(datapoint)
        data = np.array(klass_data)
        return data

    def load_data_from_file(self, file_name):
        decks, deck_names = self.load_decks_from_file(file_name)

        data = {}
        # TODO: use vectorizer
        for klass in decks.keys():
            self.dimension_to_card_name[klass] = list({card for deck in decks[klass] for card in deck})
            data[klass] = self.deck_to_vector(decks[klass], self.dimension_to_card_name[klass])

        return data, deck_names

    @staticmethod
    def train_classifier(data, eps, min_samples):
        transform = PCA(n_components=15)
        data = transform.fit_transform(data)
        labels = DeckClassifier.label_data(data, min_samples, eps)

        classifier = sklearn.svm.SVC(probability=True)

        noise_mask = labels == -1
        dims = data.shape[1]

        X = data[~noise_mask].reshape(-1, dims)
        Y = labels[~noise_mask]
        classifier.fit(X, Y)

        return classifier, transform, labels

    @staticmethod
    def get_decks_in_cluster(labels, cluster_index, deck_names):
        decks = []
        for i in range(len(labels)):
            if labels[i] == cluster_index:
                decks.append(deck_names[i])
        return decks

    def maybe_train_classifier(self, data_file, eps, min_samples):
        try:
            if self.redis_db:
                self.klass_classifiers = self.redis_db.get('klass_classifier')
                self.dimension_to_card_name = self.redis_db.get('dimension_to_card_name')
                # self.deck_names = self.redis_db.get('deck_names')
                self.pca = self.redis_db.get('pca')
                self.cluster_names = self.redis_db.get('cluster_names')
            else:
                with open(self.CLASSIFIER_CACHE, 'rb') as d:
                    state_tuple = pickle.load(d)
                self.klass_classifiers, self.dimension_to_card_name, self.pca, self.cluster_names, self.canonical_decks = state_tuple

        except IOError:
            loaded_data, loaded_deck_names = self.load_data_from_file(data_file)
            data, deck_names, test_data, test_labels = self.split_dataset(loaded_data, loaded_deck_names)
            del loaded_data
            del loaded_deck_names
            labels = {}

            pool = multiprocessing.Pool(5)
            parameters = []
            for klass in data.keys():
                parameters.append((data[klass], eps, min_samples, deck_names[klass], klass))
            results = pool.map(DeckClassifier.train_class, parameters)

            for result in results:
                classifier, pca, cluster_names, klass, klass_labels = result
                self.klass_classifiers[klass], self.pca[klass], self.cluster_names[klass], labels[klass] \
                    = classifier, pca, cluster_names, klass_labels

            print("train results:")
            mean_unknown_ratio = 0
            for klass, cluster_names in self.cluster_names.items():
                print(klass, "clusters", len(cluster_names), end="{")
                for cluster_index, cluster_name in cluster_names.items():
                    decks = self.get_decks_in_cluster(labels[klass], cluster_index, deck_names[klass])
                    if cluster_name == "UNKNOWN":
                        # print(int((float(len(decks)) / len(data[klass])) * 100), end=" ")
                        print("}")
                        unknown_ratio = (float(len(decks)) / len(data[klass])) * 100
                        mean_unknown_ratio += unknown_ratio / len(self.cluster_names)
                        print("\t{}[{}, {:.0f}%]".format(cluster_name, len(decks), unknown_ratio))
                    else:
                        print(cluster_name, len(decks), end=", ")
                        self.canonical_decks[klass] = self.get_canonical_decks(data[klass], self.pca[klass],
                                                                               labels[klass],
                                                                               self.dimension_to_card_name[klass])

            print("test results:")
            mean_accuracy = 0
            for klass in self.klass_classifiers:
                accuracy = self.test_accuracy(test_data[klass], test_labels[klass], klass)
                mean_accuracy += accuracy / len(self.klass_classifiers)
                # print(int(accuracy * 100))
                print(klass, "accuracy {:.2f}%".format(accuracy * 100))

            print("mean accuracy {:.2f}%".format(mean_accuracy * 100))
            print("mean unknown ratio {:.2f}%".format(mean_unknown_ratio))

            with open(self.CLASSIFIER_CACHE, 'wb') as d:
                state_tuple = (self.klass_classifiers, self.dimension_to_card_name, self.pca,
                               self.cluster_names, self.canonical_decks)
                pickle.dump(state_tuple, d)

    @staticmethod
    def train_class(params):
        data, eps, min_samples, deck_names, klass = params
        klass_classifier, pca, labels = DeckClassifier.train_classifier(data, eps, min_samples)
        cluster_names, _, _ = DeckClassifier.name_clusters(deck_names, klass, labels)
        return klass_classifier, pca, cluster_names, klass, labels

    # consider the newest decks more important
    def dbscan_predict(self, x_new, klass):
        x_new = x_new.reshape(1, -1)
        x_new = self.pca[klass].transform(x_new)
        prediction = self.klass_classifiers[klass].predict(x_new)
        return prediction

    def dbscan_explain(self, x_new, klass):
        x_new = x_new.reshape(1, -1)
        x_new = self.pca[klass].transform(x_new)
        probs = self.klass_classifiers[klass].predict_proba(x_new)
        return probs

    @staticmethod
    def name_clusters(deck_names, klass, labels):
        cluster_decknames = collections.defaultdict(list)
        cluster_names = {}
        cluster_races = {}
        cluster_categories = {}
        pRaces = None
        pCategories = None

        for (i, name) in enumerate(deck_names):
            deck_label = labels[i]
            cluster_decknames[deck_label].append(name)

        for cluster_index, decknames in cluster_decknames.items():
            if cluster_index == -1:
                cluster_name = "UNKNOWN"
            else:
                klass_ = klass.lower()
                decknames = [n.lower().replace(klass_, "") for n in decknames if n.lower()]
                # stopwords = set(nltk.corpus.stopwords.words('english'))

                # Freq
                tokenizer = nltk.RegexpTokenizer(r'\w+')
                words = [word for name in decknames for word in tokenizer.tokenize(name)]  # if word not in stopwords]
                fdist = FreqDist(words)

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

    @staticmethod
    def split_dataset(loaded_data, loaded_deck_names):
        known_archetypes = {
            'Warrior': {"patron", "control", "pirate", "dragon", "warrior"},
            'Paladin': {"aggro murloc", "aggro", "control", "dragon"},
            'Shaman': {"aggro", "midrange"},
            'Druid': {"beast", "control", "aggro", "ramp"},
            'Priest': {"control", "dragon"},
            'Mage': {"freeze", "reno", "tempo"},
            'Hunter': {"midrange"},
            'Rogue': {"miracle", "old", "mill"},
            'Warlock': {"reno", "zoo"},
        }
        test_dataset = {}
        test_labels = {}
        train_data = {}
        deck_names = {}

        for klass in loaded_data.keys():
            test_dataset[klass] = []
            test_labels[klass] = []
            klass_data = loaded_data[klass]

            test_data_size = int(len(klass_data) * 0.02)
            normalized_names = [name.lower().replace(klass.lower(), "").strip() for name in loaded_deck_names[klass]]

            possibilities = []
            for index, name in enumerate(normalized_names):
                if name in known_archetypes[klass]:
                    possibilities.append(index)
            random.shuffle(possibilities)
            test_data_size = min(len(possibilities), test_data_size)
            # reversed so to delete from the bottom of the list
            test_indexes = list(reversed(sorted(possibilities[:test_data_size])))

            mask = np.ones_like(klass_data, dtype=bool)

            for index in test_indexes:
                name = normalized_names[index]
                test_dataset[klass].append(klass_data[index])
                test_labels[klass].append(name)

            for index in test_indexes:
                mask[index] = False
                del loaded_deck_names[klass][index]
                del normalized_names[index]

            deck_names[klass] = loaded_deck_names[klass]
            train_data[klass] = klass_data[mask].reshape(-1, klass_data.shape[1])
        return train_data, deck_names, test_dataset, test_labels

    @staticmethod
    def label_data(data, min_samples, eps):
        model = DBSCAN(eps=eps, min_samples=min_samples, metric="manhattan")
        # model = hdbscan.HDBSCAN(metric="manhattan", min_cluster_size=cluster_size, min_samples=min_samples)
        model.fit(data)
        model.labels_.reshape(-1, 1)
        return model.labels_

    def get_canonical_decks(self, data, transform, labels, lookup):
        data = transform.transform(data)
        canonical_decks = {}
        mask = np.ones_like(labels, dtype=bool)
        for label in set(labels):
            mask = labels == label
            centroid = np.average(data[mask], axis=0)
            avg_deck = transform.inverse_transform(centroid)
            card_indexes = reversed(avg_deck.argsort()[-30:])
            canonical_deck = []
            for index in card_indexes:
                if len(canonical_decks) <= 30:
                    n_cards = avg_deck[index]
                    n_cards = min(2, int(np.round(n_cards)))
                    for i in range(n_cards):
                        canonical_deck.append(lookup[index])
            canonical_decks[label] = canonical_deck
        return canonical_decks


if __name__ == '__main__':
    DeckClassifier().run()
