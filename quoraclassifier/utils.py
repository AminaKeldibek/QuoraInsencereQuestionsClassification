import os
import errno
import numpy as np
import re
import nltk
from nltk.tokenize import TweetTokenizer


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


def preprocess_text(text):
    token_from = [
        "cryptocurrencies", "Cryptocurrency", "Brexit", "brexit", "Redmi", "redmi", "Coinbase", "\.net", "OnePlus",
        "Bhakts", "bhakts", "Qur'an", "qur'an", "anti-trump", "anti-Trump", "www.youtube.com/watch", "r-aping",
        "f\*\*k", "F'king", "sh\*t", "´", "i'am", "f.r.i.e.n.d.s", "ai/ml", "₹",
        "²", "°", "zuckerburg", "demonetisation", "\^2", "c/c\+\+", "'s", "'ve", "'ll",
        "won't", "Won't", "Can't", "n't", "'re", "'d", "Quorans", "UCEED", "Blockchain",
        "GDPR", "BNBR", "Boruto", "ethereum", "DCEU", "IIEST", "SJWs", "Qoura", "LNMIIT",
        "Upwork", "Zerodha", "Doklam", "HackerRank", "altcoins", "altcoin", "Litecoin",
        "Amazon.in", "NICMAR", "Vajiram", "-", "\u200b"
    ]

    token_to = [
        "cryptocurrency", "cryptocurrency", "Britain exit", "Britain exit", "Xiaomi smartphone", "Xiaomi smartphone", "Bitcoin",
        "dotnet", "BBK", "bhakt", "bhakt", "Quran", "Quran", "anti trump", "anti trump", "www.youtube.com",
        "raping", "fuck", "fuck", "shit", "'", "I'm", 'friends', "AI", " Rupee",
        " squared", " degrees", "zuckerberg", "demonetization", " squared", "c", " 's", " have", " will",
        "will not", "Will not", "can not", " not", " are", " would", "of Quora", "exam",
        "blockchain", "data protection", "be nice be respectful", "naruto", "Ethereum",
        "comics", "Indian Institutes", "SJW", "Quora", "LNM", "freelance", "stock",
        "Tibet", "algorithms", "bitcoin", "bitcoin", "bitcoin", "Amazon", "institute",
        "exam", " - ", " "
    ]

    remove_patterns = r"\\{2,}|\.{2,}|…|_|मिल|गई|कलेजे|को|ठंडक|ऋ|ॠ|ऌ|ॡ|ɾ̃|nɖ|nɾ|谢谢六佬|ह"
    tokenizer = TweetTokenizer()
    text = re.sub(remove_patterns, "", text)
    for i in range(len(token_from)):
        text = re.sub(token_from[i], token_to[i], text)

    tokens = tokenizer.tokenize(text)

    return tokens
