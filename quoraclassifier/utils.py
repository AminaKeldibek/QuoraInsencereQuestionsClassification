import os
import errno
import numpy as np
import re
import pickle
import string

from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.special import expit


def plot_pred_hist(y_pred_prob, y_act, labels=["zero", "one"]):
    if y_act.ndim > 1:
        y_act = np.reshape(y_act, (y_act.shape[0]))

    class1_idxs = np.where(y_act == 1)
    class2_idxs = np.where(y_act == 0)

    fig, ax = plt.subplots()
    ax.hist(y_pred_prob[class1_idxs], 50, color="red", alpha=0.5, density=True)
    ax.hist(y_pred_prob[class2_idxs], 50, color="blue", alpha=0.5, density=True)
    ax.set_ylim(0, 5)
    plt.show()


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def binarize(x, threshold):
    x_out = np.zeros_like(x)
    x_out[x > threshold] = 1
    x_out[x <= threshold] = 0

    return (x_out)


def optimize_f1(pred_labels, labels):
    min_threshold = 0.4
    max_threshold = 0.7
    step = 0.02
    thresholds = np.arange(min_threshold, max_threshold, step)
    f1_at_tresholds = np.zeros_like(thresholds)
    for i in range(len(thresholds)):
        f1_at_tresholds[i] = f1_score(
            y_true=labels,
            y_pred=binarize(pred_labels, thresholds[i])
        )

    return thresholds[np.argmax(f1_at_tresholds)]


def sigmoid(x):
    return expit(x)


def remove_nonascii(text):
    printable = set(string.printable)
    text = ''.join(list(filter(lambda x: x in printable, text)))

    return text


def clean_text(text):
    remove_patterns = r"\\{2,}|\.{2,}|_"
    text = re.sub(remove_patterns, "", text)

    return text


def separate_punctuation(text):
    # copied from kaggle
    puncts =  ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-',
              '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^',
              '_', '`', '{', '|', '}', '~']
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    return text


def hide_numbers(x):
    # copied from kaggle
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)

    return x


def remove_units(text):
    return re.sub("(#+)[a-zA-Z]+", "", text)


def replace_text(text):
    token_from = [
        "cryptocurrencies", "Cryptocurrency", "Brexit", "brexit", "Redmi", "redmi", "Coinbase", "coinbase", "\.net", "OnePlus",
        "Bhakts", "bhakts", "Qur'an", "qur'an", "anti-trump", "anti-Trump", "www.youtube.com/watch", "r-aping", "raaping"
        "f\*\*k", "F'king", "sh\*t", "´", "i'am", "f.r.i.e.n.d.s", "ai/ml", "₹",
        "²", "°", "zuckerburg", "demonetisation", "demonitisation", "Demonetization", "\^2", "c/c\+\+", "he's", "she's", "it's", "how's", "'s", "'ve", "'ll",
        "won't", "Won't", "Can't", "n't", "'re", "'d", "Quorans", "UCEED", "Blockchain",
        "GDPR", "BNBR", "Boruto", "ethereum", "DCEU", "IIEST", "SJWs", "Qoura", "LNMIIT",
        "Upwork", "upwork", "Zerodha", "Doklam", "HackerRank", "altcoins", "altcoin", "Litecoin", "litecoin",
        "Amazon.in", "NICMAR", "Vajiram", "\u200b", " adhaar", "Adhaar", "fortnite", "Trumpcare", "Quoras", "Tensorflow", "blockchains",
        "Unacademy", "chsl"
    ]

    token_to = [
        "cryptocurrency", "cryptocurrency", "Britain exit", "Britain exit", "Xiaomi smartphone", "Xiaomi smartphone", "Bitcoin",
        "bitcoin", "dotnet", "BBK", "bhakt", "bhakt", "Quran", "Quran", "anti trump", "anti trump", "youtube",
        "raping", "raping", "fuck", "fuck", "shit", "'", "I am", 'friends', "AI", " Rupee",
        " squared", " degrees", "zuckerberg", "demonetization", "demonetization", "demonetization", " squared", "c", "he is", "she is", "it is", "how is"," 's", " have", " will",
        "will not", "Will not", "can not", " not", " are", " would", "of Quora", "exam",
        "blockchain", "data protection", "be nice be respectful", "naruto", "Ethereum",
        "comics", "Indian Institutes", "SJW", "Quora", "LNM", "freelance", "freelance", "stock",
        "Tibet", "algorithms", "bitcoin", "bitcoin", "bitcoin", "bitcoin", "Amazon", "institute",
        "exam", " ", " aadhaar", "aadhaar", "Fortnite", "AHCA", "Quora", "DL", "blockchain",
        "Indian Coursera", "CHSL"
    ]
    for i in range(len(token_from)):
        text = re.sub(token_from[i], token_to[i], text)

    return text


def preprocess_text(text):
    text = remove_nonascii(text)
    text = clean_text(text)
    text = replace_text(text)
    text = separate_punctuation(text)
    text = hide_numbers(text)
    text = remove_units(text)

    tokens = TweetTokenizer().tokenize(text)

    return tokens


def xavier_weight_init(shape):
    """Defines an initializer for the Xavier distribution.
    Specifically, the output should be sampled uniformly from [-epsilon, epsilon] where
        epsilon = sqrt(6) / <sum of the sizes of shape's dimensions>
    e.g., if shape = (2, 3), epsilon = sqrt(6 / (2 + 3))
    This function will be used as a variable initializer.
    Args:
        shape: Tuple or 1-d array that species the dimensions of the requested tensor.
    Returns:
        out: tf.Tensor of specified shape sampled from the Xavier distribution.
    """
    lim = np.sqrt(6. / sum(shape))
    out = np.random.uniform(-lim, lim, shape)

    return out
