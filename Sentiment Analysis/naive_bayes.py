from collections import Counter
import math

positive_file = open('hotelPosT-train.txt','r')
negative_file = open('hotelNegT-train.txt','r')
test_file = open('hotel-test.txt','r')
positive_file_contents = ""
negative_file_contents = ""
test_file_contents = ""

for word in positive_file:
    positive_file_contents +=str(word)
for word in negative_file:
    negative_file_contents +=str(word)
for word in test_file:
    test_file_contents +=str(word)

positive_lines = []
temp_positive_lines = []
temp_positive_lines = positive_file_contents.split('\n')
for i in temp_positive_lines:
    if i!='':
        positive_lines.append(i.split('\t')[1])                         #Ignore ID and extract only the review
negative_lines = []
temp_negative_lines = []
temp_negative_lines = negative_file_contents.split('\n')
for i in temp_negative_lines:
    if i!='':
        negative_lines.append(i.split('\t')[1])

test_lines = []
temp_test_lines = []
temp_test_lines = test_file_contents.split('\n')                        #Extract test reviews

N_positive = len(positive_lines)                                        #Number of positive docs
N_negative = len(negative_lines)                                        #Number of negative docs
N_doc = N_positive + N_negative                                         #Total number of docs

prior_positive = math.log ((N_positive/N_doc))                          #Logarithmic Prior
prior_negative = math.log ((N_negative/N_doc))

positive_vocabulary_list = []
for line in positive_lines:
    words = line.split(' ')
    for word in words:
        if word!='':
            positive_vocabulary_list.append(word)                       #all words in positive doc (including duplicates)

negative_vocabulary_list = []
for line in negative_lines:
    words = line.split(' ')
    for word in words:
        if word!='':
            negative_vocabulary_list.append(word)                       #words in negative doc (including duplicates)

vocabulary_list = []
vocabulary_list = positive_vocabulary_list + negative_vocabulary_list   #all words from all docs (including duplicates)

vocabulary = set(vocabulary_list)                                       #unique words from pos and neg doc
positive_word_count = Counter(positive_vocabulary_list)                 #no of unique words in positive doc
negative_word_count = Counter(negative_vocabulary_list)                 #no of unique words in negative doc
vocabulary_count = Counter(vocabulary)                                  #no of unique words from pos and neg doc

positive_likelihood = {}
negative_likelihood = {}
for i in vocabulary:
    positive_likelihood[i] = math.log( (positive_word_count.get((i),0.00) + 1) / (len(positive_vocabulary_list) + len(vocabulary)) )
    negative_likelihood[i] = math.log( (negative_word_count.get((i),0.00) + 1) / (len(negative_vocabulary_list) + len(vocabulary)) )

def bayes(input_line):
    test_line = input_line.split('\t')[1]                               #Ignore ID and extract only the review
    words = test_line.split(' ')
    test_words = []
    for i in words:
        if i!='':
            if i in vocabulary:
                test_words.append(i)
    positive_value = prior_positive
    negative_value = prior_negative
    for i in test_words:
        positive_value = ( positive_value) + ( positive_likelihood.get(i))
        negative_value = ( negative_value) + ( negative_likelihood.get(i))
    if positive_value > negative_value:
        return "POS"
    else:
        return "NEG"

prediction = {}
for i in temp_test_lines:
    if i!= "":
        prediction[i.split('\t')[0]] = bayes(i)

output_file = open("naive_bayes_output.txt","w+")
for i in prediction:
    output_file.write(i + "\t" + prediction.get(i) + "\n")
output_file.close()