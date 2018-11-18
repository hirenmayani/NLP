import os
import pickle
import numpy as np
import math
from scipy.fftpack.realtransforms import dst

model_path = './models_4/'
loss_model = 'cross_entropy'
loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'r'))

print(len(dictionary.items()))
print(steps)
print(embeddings.shape)
"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
def findAngle(v1,v2):
    import math
    
    def dotproduct(v1, v2):
      return sum((a*b) for a, b in zip(v1, v2))
    
    def length(v):
      return math.sqrt(dotproduct(v, v))
    
    
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
  

def findDist(w1,w2):
    v1 = embeddings[dictionary[w1]]
    v2 = embeddings[dictionary[w2]]
    #return np.linalg.norm(v1,v2)
    from scipy.spatial import distance
    return distance.euclidean(v2,v1), findAngle(v1, v2)

  
first = []
american = []
would = []

for w,cnt in dictionary:
    first.append(findAngle(w, 'first'))
    american.append(findAngle(w, 'american'))
    would.append(findAngle(w, 'would'))
     
first.sort(reverse=True)
american.sort(reverse=True)
would.sort(reverse=True)

print(first[:21]) 
print(american[:21]) 
print(would[:21]) 
print('-------sample output---------')
print(findDist('pilgrim','shrine'))
print(findDist('hunter','quarry'))
print(findDist('assassin','victim'))
print(findDist('climber','peak'))

print(findDist('pig','mud'))
print(findDist('politician','votes'))
print(findDist('dog','bark'))
print(findDist('bird','worm'))


print('----------------')

def readFile():
    
    file = open('/Users/hiren/eclipse-workspace/NLP/Assignment1_for_students/word_analogy_dev.txt')
    #filePred = open('/Users/hiren/eclipse-workspace/NLP/Assignment1_for_students/word_analogy_dev_mturk_answers.txt')
    # open a (new) file to write
    outF = open("/Users/hiren/eclipse-workspace/NLP/Assignment1_for_students/word_analogy_dev_sample_predictions.txt", "w")

    for line in file.readlines():
        example, choices = line.split('||')[0],line.split('||')[1]
        
        if choices[-1] == '\n':
            choices = choices[:-1]
            
        score = 0.0
        cnt = 0.0
        lst = []
        #print(example,choices)
        for e in example.split(','):
            ab = e[1:-1].split(':')
            tscore = findDist(ab[0], ab[1])[1]
            cnt += 1
            score+=tscore
            lst.append(tscore)
        lst.sort()
        
        mx,mxw = None, None
        mn, mnw = None, None
        targetS = (score/cnt + lst[1])/2.0
        
        import statistics
        target = statistics.median(lst)
        
        for e in choices.split(','):
            cd = e[1:-1].split(':')
            dst =0
#             for l in lst:
#                 dst+=abs(findDist(cd[0], cd[1])[0] - l) 
            dst =abs(findDist(cd[0], cd[1])[1] - targetS) 
            
            if (not mx) or dst > mx:
                mx = dst
                mxw = e
                 
            if (not mn) or dst < mn:
                mn = dst
                mnw = e
             
        line = choices.replace(',',' ')
        line+= ' '+mxw
        line+= ' '+mnw +'\n'
        #print(line)
        outF.write(line)
    
    outF.close()

readFile()   
     