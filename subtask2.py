import nltk, string
from nltk.stem.porter import PorterStemmer
from copy import deepcopy
from nltk.util import ngrams
from time import time
from itertools import chain, combinations
import copy
from joblib import Parallel, delayed


stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def pad_sequence(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    if pad_left:
        sequence = chain((pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n-1))
    return sequence

def skipngrams(sequence, n, k, pad_left=False, pad_right=False, pad_symbol=None):
    sequence_length = len(sequence)
    sequence = iter(sequence)
    sequence = pad_sequence(sequence, n, pad_left, pad_right, pad_symbol)

    if sequence_length + pad_left + pad_right < k:
        # raise Exception("The length of sentence + padding(s) < skip")
        yield []

    if n < k:
        raise Exception("Degree of Ngrams (n) needs to be bigger than skip (k)")

    history = []
    nk = n+k

    # Return point for recursion.
    if nk < 1:
        return
    # If n+k longer than sequence, reduce k by 1 and recur
    elif nk > sequence_length:
        for ng in skipngrams(list(sequence), n, k-1):
            yield ng

    while nk > 1: # Collects the first instance of n+k length history
        history.append(next(sequence))
        nk -= 1

    # Iterative drop first item in history and picks up the next
    # while yielding skipgrams for each iteration.
    for item in sequence:
        history.append(item)
        current_token = history.pop(0)
        # Iterates through the rest of the history and
        # pick out all combinations the n-1grams
        for idx in list(combinations(range(len(history)), n-1)):
            ng = [current_token]
            for _id in idx:
                ng.append(history[_id])
            yield tuple(ng)

    # Recursively yield the skigrams for the rest of seqeunce where
    # len(sequence) < n+k
    for ng in list(skipngrams(history, n, k-1)):
        yield ng

def GetAllFeatures (Sent, SentIndex, Term):
    T0= time()
    try:
        Tokens = tokenize(Sent.lower())
        Chars = list(''.join(Tokens))

        # Get the POS tag features
        WordPOSTags = [Item[1] for Item in nltk.pos_tag(Tokens)]

        # Word ngram features
        WordUnigrams = deepcopy(Tokens)
        WordBigrams = ['_'.join(list(Item)) for Item in list(ngrams(Tokens, n=2))]
        WordTrigrams = ['_'.join(list(Item)) for Item in list(ngrams(Tokens, n=3))]

        # Character ngram features
        CharTrigrams = ['_'.join(list(Item)) for Item in list(ngrams(Chars, n=3))]
        CharFourgrams = ['_'.join(list(Item)) for Item in list(ngrams(Chars, n=4))]
        CharFivegrams = ['_'.join(list(Item)) for Item in list(ngrams(Chars, n=5))]
        CharSixgrams = ['_'.join(list(Item)) for Item in list(ngrams(Chars, n=6))]

        # Term specific word ngram features
        WordUnigramsTerm = [Item + '+' + str(Term[SentIndex]) for Item in WordUnigrams]
        WordBigramsTerm = [Item + '+' + str(Term[SentIndex]) for Item in WordBigrams]
        WordTrigramsTerm = [Item + '+' + str(Term[SentIndex]) for Item in WordTrigrams]

        # Term specific char ngram features
        CharTrigramsCat = [Item + '+' + str(Term[SentIndex]) for Item in CharTrigrams]
        CharFourgramsCat = [Item + '+' + str(Term[SentIndex]) for Item in CharFourgrams]
        CharFivegramsCat = [Item + '+' + str(Term[SentIndex]) for Item in CharFivegrams]
        CharSixgramsCat = [Item + '+' + str(Term[SentIndex]) for Item in CharSixgrams]

        # Term specific POS tag features
        WordPOSTagsTerm = [Item + '+' + str(Term[SentIndex]) for Item in WordPOSTags]

        # Non continous ngram features
        TwoSkipBiGrams = list(skipngrams(WordUnigrams, n=2, k=2))
        TwoSkipTriGrams = list(skipngrams(WordUnigrams, n=3, k=2))
        ThreeSkipTriGrams = list(skipngrams(WordUnigrams, n=3, k=3))
        ThreeSkipFourGrams = list(skipngrams(WordUnigrams, n=4, k=3))
        TwoSkipBiGrams = ['_'.join(list(Item)) for Item in TwoSkipBiGrams]
        TwoSkipTriGrams = ['_'.join(list(Item)) for Item in TwoSkipTriGrams]
        ThreeSkipTriGrams = ['_'.join(list(Item)) for Item in ThreeSkipTriGrams]
        ThreeSkipFourGrams = ['_'.join(list(Item)) for Item in ThreeSkipFourGrams]

        # Term specific non continous ngram features
        TwoSkipBiGramsTerm = [Item + '+' + str(Term[SentIndex]) for Item in TwoSkipBiGrams]
        TwoSkipTriGramsTerm = [Item + '+' + str(Term[SentIndex]) for Item in TwoSkipTriGrams]
        ThreeSkipTriGramsTerm = [Item + '+' + str(Term[SentIndex]) for Item in ThreeSkipTriGrams]
        ThreeSkipFourGramsTerm = [Item + '+' + str(Term[SentIndex]) for Item in ThreeSkipFourGrams]
        AllFeats = WordUnigrams + WordUnigramsTerm + WordBigrams + WordBigramsTerm + WordTrigrams + WordTrigramsTerm \
                   + CharTrigrams + CharTrigramsCat + CharFourgrams + CharFourgramsCat + CharFivegrams + CharFivegramsCat \
                   + CharSixgrams + CharSixgramsCat + TwoSkipBiGrams + TwoSkipTriGrams + ThreeSkipTriGrams + \
                   ThreeSkipFourGrams + TwoSkipBiGramsTerm + TwoSkipTriGramsTerm + ThreeSkipTriGramsTerm + \
                   ThreeSkipFourGramsTerm + WordPOSTags + WordPOSTagsTerm
        AllFeats = ' '.join(AllFeats)

    except:
        AllFeats = ''
    print 'processed sentence :{} in {} sec'.format(SentIndex, time()-T0)
    return AllFeats
def Main ():
    T0 = time()
    Sentences = [';'.join(l.strip().split(';')[:-2]) for l in open ('../../Task 4/RestAspTermABSA.data').xreadlines()]#[:20]
    Term = [l.strip().split(';')[-2] for l in open ('../../Task 4/RestAspTermABSA.data').xreadlines()]
    Label = [l.strip().split(';')[-1] for l in open ('../../Task 4/RestAspTermABSA.data').xreadlines()]
    # AllFeatsExpSentences = []
    # for SentIndex, Sent in enumerate(Sentences):
    #     AllFeatsExpSentences.append (GetAllNRCFeats(Sent, SentIndex, Cat))
    AllFeatsExpSentences = Parallel(n_jobs=8)(delayed(GetAllFeatures)(Sent, SentIndex, Term) for SentIndex, Sent in enumerate(Sentences))
    with open ('AllNRCFeats.txt','w') as FH:
        for Index, Item in enumerate(AllFeatsExpSentences):
            print >>FH, str(Item)+';'+Label[Index]

    print 'processed {} sentences in a total of {} sec. with 8 cpu'.format(len(AllFeatsExpSentences), round (time()-T0,2))
if __name__ == '__main__':
    Main()


