from collections import Counter
import math

train_file = open('berp-POS-train.txt','r')
test_file = open('berp-POS-test.txt','r')

train_file_contents = ""                            
test_file_contents = ""
for word in train_file:
    train_file_contents += str(word)
for word in test_file:
    test_file_contents += str(word)

lines = []
words = []
tags = [] 
wordandtags = []
sentences = train_file_contents.split('\n\n')
for sentence in sentences:
    words.append("<s>")
    tags.append("S")
    lines = sentence.split('\n')
    for line in lines:
        if line!="":
            word = line.split('\t')[1]
            tag = line.split('\t')[2]
            words.append(word)
            tags.append(tag)

word_count = Counter(words)                                            #get unduplicated count of words 
tag_count = Counter(tags)                                              #get unduplicated count of tags

singlecountwords = []                                       #get words that occur only once in training data
for key,value in word_count.items():
    if value == 1:
        singlecountwords.append(key)

for index,word in enumerate(words):                                 #replace the words that occur only once in words with "UNK"
    if word in singlecountwords:
        words[index] = "UNK"

word_count = Counter(words) 

for iteration in range(0,len(words)):
    wordandtags.append((words[iteration],tags[iteration]))    

unduplicated_tags = list(set(tags))
unduplicated_words = list(set(words))

emission_count = []
wordandtag_count = Counter(wordandtags)                                #unduplicated counts of (word,tag)
emission_count = [list(wordandtag_count.keys()),list(wordandtag_count.values())]     #combined tuple(word,tag) and their respective unduplicated counts

emission_word = []
emission_tag = []
for i in emission_count[0]:                                             #get emission word and the corresponding tag from the tuple
    emission_word.append(i[0])
    emission_tag.append(i[1])

emission_probability = {}
for i in range(0,len(emission_count[0])):
    emission_probability[emission_count[0][i]] = (emission_count[1][i]) / (tag_count.get(emission_tag[i]))

tagandtag = []                                                          #extract (tag,previous tag) tuples
for i in range(1,len(tags)):
    tagandtag.append((tags[i],tags[i-1]))

flag1 = {}                                                              #add 1 count to (tag, previous tag) tuples
for tag in tagandtag:
    if tag not in flag1:
        tagandtag.append((tag))
        flag1[tag] = 1

for tag1 in unduplicated_tags:                                            #add (tag,previous tag) for unoccured combinations (smoothing)
    for tag2 in unduplicated_tags:
        if (tag1,tag2) not in tagandtag:
            tagandtag.append((tag1,tag2))

tagandtag_list = []
tagandtagcount_list = []
tagandtag_count = Counter(tagandtag)
tagandtag_list = list(tagandtag_count.keys())
tagandtagcount_list = list(tagandtag_count.values())

transition_count = []
transition_count = [tagandtag_list,tagandtagcount_list]                 #combined tuple(tag,tag) and their respective unduplicated counts

transition_tag1 = []
transition_tag2 = []
for i in transition_count[0]:                                             #get transition tag1 and the corresponding tag2 from the tuple
    transition_tag1.append(i[0])
    transition_tag2.append(i[1])

transition_probability = {} 
for i in range(0,len(transition_count[0])):
    transition_probability[transition_count[0][i]] = (transition_count[1][i]) / (tag_count.get(transition_tag2[i]) + ((len(unduplicated_tags))-1))

test_sentences = []                                                      #extract observations(words) from test data
test_sentence = test_file_contents.split('\n\n')
for sentence in test_sentence:
    if sentence!="":
        test_sentences.append(sentence.split('\n'))

test_sentence_observations = []
for sentence in test_sentences:
    temp = []
    for observation in sentence:
        if observation!="":
            temp.append(observation.split('\t')[1])
    test_sentence_observations.append(temp)
unduplicated_tags.remove("S")
unduplicated_words.remove("<s>")

def Viterbi(obs,states):
    global argmax_final
    global argmax
    for index,word in enumerate(obs):
        if word not in unduplicated_words:
            obs[index] = "UNK"
    vit_dict = {}
    backpointer = {}
    for state in states:                                                 #initialization
        vit_dict[(state,0)] = (transition_probability.get((state,"S"))) * (emission_probability.get((obs[0], state),0.00))
        backpointer[(state,0)] = 0
    
    for t in range(1,len(obs)):                                      #recursion
        for state in states:
            maxval = 0
            for new_state in states:
                temp = (vit_dict.get((new_state,t-1))) * (transition_probability.get((state,new_state)))
                if maxval <= temp: 
                    maxval = temp
                    argmax = new_state
            vit_dict[(state,t)] = (maxval) * (emission_probability.get((obs[t],state),0.00))
            backpointer[(state,t)] = argmax
    
    maxval_final = 0
    T = (len(obs)-1)
    for state in states:                                                 #termination
        if maxval_final <= vit_dict[(state,T)]:
            maxval_final = vit_dict[(state,T)]
            argmax_final = state
    vit_dict[("final",T)] = maxval_final
    backpointer[("final",T)] = argmax_final
    end_index = (len(obs))-1
    
    backpointer_return = [0] * (end_index+1)                        #return backpointer by tracing back the tags
    backpointer_return[end_index] = backpointer.get(("final",end_index))
    for i in range(end_index,0,-1):
        backpointer_return[(i-1)] = backpointer.get((backpointer_return[i],i))
    return backpointer_return


viterbi_tags = [0] * len(test_sentence_observations)              #call Viterbi for every sentence in the test data
for i in range(0,len(test_sentence_observations)):
    viterbi_tags[i] = list((Viterbi(list(test_sentence_observations[i]), list(unduplicated_tags))))

test_outputlist = []
for i in range(0,len(test_sentence_observations)):    
    for j in range(0,len(test_sentence_observations[i])):
        test_outputlist.append(str(j+1)+'\t'+ test_sentence_observations[i][j]+'\t'+ viterbi_tags[i][j]+'\n')
    test_outputlist.append('\n')

test_outputfile = open ("test-output.txt","w+")
for i in test_outputlist:
    test_outputfile.write(i)
test_outputfile.close()
