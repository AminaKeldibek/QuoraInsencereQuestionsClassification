import sys
sys.path.insert(0, "/Users/keldibek/Projects/QuoraCompetition")

import unittest
from quoraclassifier.utils import preprocess_text


class DataPreprocessTestCase(unittest.TestCase):
    def test_remove_symbols(self):
        text = [
            "Sunny day and …dog",
            "गई this ia not from my कलेजे and  को",
            "Everytime I see it ..want to remove",
            "Everytime I see it . want to remove",
            r"Amina and \\\\ Birzhan",
            r"Amina and \\\ Birzhan",
            "Amina and Birzhan_",
            r"x \y \\",
            r"Amina and \\Birzhan"
        ]
        out = [
            ["Sunny", "day", "and", "dog"],
            ["this", "ia", "not", "from", "my", "and"],
            ["Everytime", "I", "see", "it", "want", "to", "remove"],
            ["Everytime", "I", "see", "it", ".", "want", "to", "remove"],
            ["Amina", "and", "Birzhan"],
            ["Amina", "and", "Birzhan"],
            ["Amina", "and", "Birzhan"],
            ["x", "\\", "y"],
            ["Amina", "and", "Birzhan"]
        ]
        for i in range(len(text)):
            self.assertEqual(preprocess_text(text[i]), out[i])

    def test_replace_patterns(self):
        text = [
            "My …cryptocurrencies",
            "There are lots of brexit news",
            "Did you hear about redmi?",
            "I used .net",
            "What does hashtag bhakts mean",
            "qur'an is misspelling",
            "anti-trump campaign is started",
            "to see this go to www.youtube.com/watch",
            "They hide r-aping by adding hyphen",
            "f**k and F'king and sh*t are found in questions",
            "Wrong apostroph sign ´ is used in I´m and don´t",
            "Why people write f.r.i.e.n.d.s",
            "Simple ai/ml application costs 100 ₹",
            "write c/c++ function Two² plus 4^2 equals 18",
            "300 ° and zuckerburg or demonetisation",
            "Won't you come or you're scared?",
            "Can't help and don't want cause we're weak and I'd like to ask Quorans"

        ]
        out = [
            ["My", "cryptocurrency"],
            ["There", "are", "lots", "of", "Britain", "exit", "news"],
            ["Did", "you", "hear", "about", "Xiaomi", "smartphone", "?"],
            ["I", "used", "dotnet"],
            ["What", "does", "hashtag", "bhakt", "mean"],
            ["Quran", "is", "misspelling"],
            ["anti", "trump", "campaign", "is", "started"],
            ["to", "see", "this", "go", "to", "www.youtube.com"],
            ["They", "hide", "raping", "by", "adding", "hyphen"],
            ["fuck", "and", "fuck", "and", "shit", "are", "found", "in", "questions"],
            ["Wrong", "apostroph", "sign", "'", "is", "used", "in", "I'm", "and", "do", "not"],
            ["Why", "people", "write", "friends"],
            ["Simple", "AI", "application", "costs", "100", "Rupee"],
            ["write", "c", "function", "Two", "squared", "plus", "4", "squared", "equals", "18"],
            ["300", "degrees", "and", "zuckerberg", "or", "demonetization"],
            ["Will", "not", "you", "come", "or", "you", "are", "scared", "?"],
            ["can", "not", "help", "and", "do", "not", "want", "cause", "we",
             "are", "weak", "and", "I","would", "like", "to", "ask", "of", "Quora"],
        ]
        for i in range(len(text)):
            self.assertEqual(preprocess_text(text[i]), out[i])


if __name__ == '__main__':
    unittest.main()
