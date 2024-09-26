# models.py
import nltk
from sentiment_data import *
from utils import *
import numpy as np
from nltk.corpus import stopwords
import string, random
nltk.download('stopwords')
print(stopwords.words('english'))
seed_value = 0
np.random.seed(seed_value)
random.seed(seed_value)
from collections import Counter, defaultdict
#import matplotlib.pyplot as plt
#from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


# class UnigramFeatureExtractor(FeatureExtractor):
#     """
#     Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
#     and any additional preprocessing you want to do.
#     """
#     def __init__(self, indexer: Indexer):
#         raise Exception("Must be implemented")
class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(stopwords.words('english'))


    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        for word in sentence:
            word = word.lower().strip(string.punctuation)
            if word and word not in self.stopwords:
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"Unigram={word}", add=True)
                else:
                    idx = self.indexer.index_of(f"Unigram={word}")
                if idx != -1:
                    features[idx] += 1
        return features



class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(stopwords.words('english'))

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()

        for i in range(len(sentence)-1):
            bigram = (sentence[i].lower().strip(string.punctuation), sentence[i+1].lower().strip(string.punctuation))
            if bigram[0] and bigram[0] not in self.stopwords and bigram[1] and bigram[1] not in self.stopwords:
                bigram_feature = f"Bigram={bigram[0]}|{bigram[1]}"
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(bigram_feature, add=True)
                else:
                    idx = self.indexer.index_of(bigram_feature)
                if idx != -1:
                    features[idx] += 1
        return features


class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.document_count = defaultdict(int)
        self.total_docs = 0

    def get_indexer(self):
        return self.indexer

    def add_document(self, sentence):

        self.total_docs += 1
        unique_words = set(word.lower().strip(string.punctuation) for word in sentence if word)
        for word in unique_words:
            self.document_count[word] += 1

    def calculate_idf(self, word):

        if word in self.document_count:
            return np.log(self.total_docs / (1 + self.document_count[word]))
        else:
            return 0

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        tf_counter = Counter()
        for word in sentence:
            word = word.lower().strip(string.punctuation)
            if word:
                tf_counter[word] += 1

        total_terms = sum(tf_counter.values())
        tf = {word: count / total_terms for word, count in tf_counter.items()}

        features = Counter()
        for word, tf_val in tf.items():
            idf_val = self.calculate_idf(word)
            tfidf = tf_val * idf_val
            if add_to_indexer:
                feature_index = self.indexer.add_and_get_index(f"TFIDF={word}", add=True)
            else:
                feature_index = self.indexer.index_of(f"TFIDF={word}")
            if feature_index != -1:
                features[feature_index] = tfidf

        return features


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")

class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


# class PerceptronClassifier(SentimentClassifier):
#     """
#     Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
#     superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
#     modify the constructor to pass these in.
#     """
#     def __init__(self):
#         raise Exception("Must be implemented")


class PerceptronClassifier(SentimentClassifier):
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = 0.0
        for feat, value in features.items():
            score += self.weights[feat] * value
        return 1 if score >= 0 else 0



# class LogisticRegressionClassifier(SentimentClassifier):
#     """
#     Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
#     superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
#     modify the constructor to pass these in.
#     """
#     def __init__(self):
#         raise Exception("Must be implemented")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor



    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = sum(self.weights[feat] * value for feat, value in features.items())
        probability = sigmoid(score)
        return 1 if probability >= 0.5 else 0


# def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
#     """
#     Train a classifier with the perceptron.
#     :param train_exs: training set, List of SentimentExample objects
#     :param feat_extractor: feature extractor to use
#     :return: trained PerceptronClassifier model
#     """
#     raise Exception("Must be implemented")

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs: int =10,
                     initial_learning_rate: float = 0.1) -> PerceptronClassifier:
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)
    #first tried zero Initialisation. Instances of Accuracy <74% have been found.
    #weights = np.zeros(len(feat_extractor.get_indexer()))
    #implemented Xavier Initialisation.
    #fan_in = len(feat_extractor.get_indexer())
    #weights = np.random.uniform(low=-np.sqrt(6 / fan_in), high=np.sqrt(6 / fan_in), size=fan_in)
    #Random Uniform intialisation
    weights = np.random.uniform(low=-0.01, high=0.01, size=len(feat_extractor.get_indexer()))
    for epoch in range(1,num_epochs + 1):
        np.random.shuffle(train_exs)
        # constant learning rate
        #learning_rate = initial_learning_rate
        # Inverse learning rate
        learning_rate = initial_learning_rate/ epoch
        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            if sum(weights[feat] * value for feat, value in features.items()) >= 0:
                pred = 1
            else:
                pred = 0
            if pred != ex.label:
                for feat, value in features.items():
                    weights[feat] += learning_rate * value * (ex.label - pred)

        #learning_rate *= 0.9

    return PerceptronClassifier(weights, feat_extractor)

def get_top_words_and_weights(perceptron_model: PerceptronClassifier, top_n: int = 10):
    """
    Extract the top N words with the highest and lowest weights.

    :param perceptron_model: The trained PerceptronClassifier model
    :param top_n: Number of top positive/negative words to display
    """
    weights = perceptron_model.weights
    indexer = perceptron_model.feat_extractor.get_indexer()

    sorted_indices = np.argsort(weights)

    top_pos_indices = sorted_indices[-top_n:]  # indiced corresponding to the highest weights
    top_neg_indices = sorted_indices[:top_n]  # Lowest (most negative) weights

    top_pos_words = [(indexer.get_object(i), weights[i]) for i in top_pos_indices]
    top_neg_words = [(indexer.get_object(i), weights[i]) for i in top_neg_indices]

    print(f"Top {top_n} positive words (highest weights):")
    for word, weight in top_pos_words:
        print(f"{word}: {weight:.4f}")

    print(f"\nTop {top_n} negative words (lowest weights):")
    for word, weight in top_neg_words:
        print(f"{word}: {weight:.4f}")

# def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
#     """
#     Train a logistic regression model.
#     :param train_exs: training set, List of SentimentExample objects
#     :param feat_extractor: feature extractor to use
#     :return: trained LogisticRegressionClassifier model
#     """
#     raise Exception("Must be implemented")

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, num_epochs: int = 50,
                              learning_rate: float = 0.01) -> LogisticRegressionClassifier:
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    weights = np.random.uniform(low=-0.01, high=0.01, size=len(feat_extractor.get_indexer()))

    for epoch in range(num_epochs):
        np.random.shuffle(train_exs)  # Shuffle training examples

        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[feat] * value for feat, value in features.items())

            probab_scores = sigmoid(score)
            error_out = ex.label - probab_scores
            for feat, value in features.items():
                weights[feat] += learning_rate * error_out * value

    return LogisticRegressionClassifier(weights, feat_extractor)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Log likelihood calculation
def calculate_log_like(train_exs, feat_extractor, weights):
    log_like = 0.0
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        score = sum(weights[feat] * value for feat, value in features.items())
        probab_scores = sigmoid(score)
        log_like += ex.label * np.log(probab_scores + 1e-10) + (1 - ex.label) * np.log(1 - probab_scores + 1e-10)
    return log_like


def train_logistic_regression_with_tracking(train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                            feat_extractor: FeatureExtractor, num_epochs: int = 10,
                                            learning_rate: float = 0.1):
    # Initialize weights using random uniform initialization
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    weights = np.random.uniform(low=-0.01, high=0.01, size=len(feat_extractor.get_indexer()))


    log_likes = []
    dev_accuracies = []

    for epoch in range(num_epochs):
        np.random.shuffle(train_exs)  # Shuffle training examples


        for ex in train_exs:
            features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            score = sum(weights[feat] * value for feat, value in features.items())
            probability = sigmoid(score)
            error_out_out = ex.label - probability
            for feat, value in features.items():
                weights[feat] += learning_rate * error_out_out * value


        log_like = calculate_log_like(train_exs, feat_extractor, weights)
        log_likes.append(log_like)


        dev_accuracy = evaluate_accuracy(LogisticRegressionClassifier(weights, feat_extractor), dev_exs)
        dev_accuracies.append(dev_accuracy)

    return weights, log_likes, dev_accuracies



def evaluate_accuracy(classifier: SentimentClassifier, examples: List[SentimentExample]) -> float:
    correct = 0
    for ex in examples:
        pred = classifier.predict(ex.words)
        if pred == ex.label:
            correct += 1
    return correct / len(examples)


def plot_log_like_and_accuracy(log_likes_list, dev_accuracies_list, learning_rates, num_epochs):
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i, log_likes in enumerate(log_likes_list):
        plt.plot(epochs, log_likes, label=f'LR={learning_rates[i]}')
    plt.title('Log Likelihood vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Log Likelihood')
    plt.legend()

    # Development Accuracy subplot
    plt.subplot(1, 2, 2)
    for i, dev_accuracies in enumerate(dev_accuracies_list):
        plt.plot(epochs, dev_accuracies, label=f'LR={learning_rates[i]}')
    plt.title('Dev Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dev Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


learning_rates = [0.01, 0.1]
num_epochs = 20

log_likes_list = []
dev_accuracies_list = []




def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """


    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
        for ex in train_exs:
            feat_extractor.add_document(ex.words)
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")
    # Train the model


    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
        print(get_top_words_and_weights(model, top_n=10)) #also prints the top_positive and top_negative words and their weights.
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)

        # for lr in learning_rates:
        #     _, log_likes, dev_accuracies = train_logistic_regression_with_tracking(train_exs, dev_exs,
        #                                                                                  feat_extractor,
        #                                                                                  num_epochs=num_epochs,
        #                                                                                  learning_rate=lr)
        #     log_likes_list.append(log_likes)
        #     dev_accuracies_list.append(dev_accuracies)
        #
        #
        # plot_log_like_and_accuracy(log_likes_list, dev_accuracies_list, learning_rates, num_epochs)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model





