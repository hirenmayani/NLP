#!/bin/python
import pickle,collections
from operator import itemgetter
clusters = {}
trueLbls = {}
postfixs = {}
freq = {}

cnt  = collections.defaultdict(int)
nbits = 4 
nbitsClusters= 5
ncntparts =5

def stem(word):
    for suffix in ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return None

def storeCluster(data):
    from clustering import ClassLMClusters
    c = ClassLMClusters(data)
#                         max_vocab_size=args.max_vocab_size,
#                         batch_size=args.batch_size, lower=args.lower)
    c.save_clusters("clusterOP.txt")

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """
    
    global cnt
    global freq
    import nltk
    from nltk.stem.lancaster import LancasterStemmer
    lancaster_stemmer = LancasterStemmer()
    sfxs = ['ing', 'ly', 'ed', 'ious', 'ies', 'ive', 'es', 's', 'ment']
    #storeCluster(train_sents)
    file = open("clusterOP.txt", "r")
 
    for line in file:
        w, code, wcnt = line.split()
        if len(code) < nbitsClusters:
            code = "0"*(nbitsClusters-len(code))+code 
        clusters[w] = code[:nbitsClusters]
        
    for s in train_sents:
        for w in s:
            cnt[w]+=1
            #stemmed = lancaster_stemmer.stem(w)
#             if len(stemmed) != w:
#                 postfixs[w] = w[len(stemmed):]
            stemmed = stem(w)
            if stemmed:
                postfixs[w] = stemmed
    
    cnt = sorted(cnt.items(), key= itemgetter(0))
    totalby5 = int(len(cnt)/ncntparts)
    
    for i,(w,c) in enumerate(cnt):
        freq[w] = int(i/totalby5)
    

    nltk.download('brown')
    for w, tag in nltk.corpus.brown.tagged_words():
        trueLbls[w] = tag
          
    def save_obj(obj, name ):
        with open( name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)
  
    save_obj(trueLbls, 'brownLblsDct1')
#     
#     def load_obj(name ):
#         with open( name + '.pkl', 'rb') as f:
#             return pickle.load(f)
#     
#     trueLbls = load_obj('brownLblsDct1')
    
     
def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    ftrs = []
    # bias
    ftrs.append("BIAS")
    # position features
    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")

    # the word itself
    word = unicode(sent[i])
    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")

#     if i == 0:
#         ftrs.append("IS_FIRST_WORD")
#       
#     if i == len(sent)-1:
#         ftrs.append("IS_LAST_WORD")
#   
#     if word in freq:
#          ftrs.append("IS_CNT_"+str(freq[word]))
#     else:
#          ftrs.append("IS_CNT_"+'RARE')
#           
#            
#     if word in postfixs:
#          ftrs.append("IS_POST_"+postfixs[word])
#
    if sent[0].isupper():
        ftrs.append("IS_FIRST_CHAR_UPPER")

    for j in range(0,nbits):
        ftrs.append("IS_PRE_"+word[:j])
        ftrs.append("IS_POST_"+word[-j:])

    if sent[i] in trueLbls:        
        ftrs.append("IS_T_LBL"+trueLbls[sent[i]])

#     if word in clusters and clusters[word]:
#         for ci in range(nbitsClusters):
#             if clusters[word][ci]:
#                 ftrs.append("IS_CLUSTERBIT_"+str(ci))
         
    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
    if word in stopwords:        
        ftrs.append("IS_STOPWORD")
    
    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
                
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    par = "I used to roll the dice. \n Feel the fear in my enemy's eyes \n Listen as the crowd would sing \n Now the old king is dead! Long live the king!"
    sents = [ s.split() for s in par.split('\n')]
    
    sents = [
    [ "I", "love", "food" ,"food2xs"],
    [ "I", "love", "xyz" ,"food2xs"],
    [ "we", "love", "pizza" ,"we"]
    ]
    
    preprocess_corpus(sents)
    
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
