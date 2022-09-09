import sys
import string
import nltk
import re
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

sw = stopwords.words('english')
sw.extend([p for p in string.punctuation])
sw.extend(['sound=','"um"','seconds='])
sw.extend(['``',"''"])

# Process file into dictionary of meanings ('sense' of word)
def read_dict(filename):
    f = open(filename, 'r')
    sense_dict = defaultdict(lambda:[])
    
    for line in f:
        if line == '\n':
            continue
        line = line.lstrip('<').rstrip('>')
        match = re.search(r'(uid=)\w+', line)
        if match:
            line = line.split()
            sense_dict[line[1][4:]] = {'tag':line[3][4:-1]}
    return sense_dict

# Process test data into workable list
def read_test(filename):
    f = open(filename, 'r')
    test_data = []
    flag = False
    
    bag = []
    for line in f:
        if re.match(r'(700)\w+', line):
            flag = True
            continue
        if line == '\n':
            flag = False
            bag = []
        if flag:
            bag.extend([w for w in word_tokenize(line.lower()) if not w in sw])
            match = re.search(r'(<tag>)\w', line)
            if match:
                bag.remove('tag')
                test_data.append(bag)
    return test_data

# Process gold standard data for model evaluation
def read_gold(filename):
    f = open(filename, 'r')
    gold_data = []
    for line in f:
        if line == '\n':
            continue
        gold_data.append(line[7:-1])
    return gold_data

# Process training data
def read_train(filename):
    f = open(filename, 'r')
    train_data = defaultdict(lambda:[])
    flag = False
    vocab = []
    bag = []
    for line in f:
        if re.match(r'(800)\w+', line):
            flag = True
            continue
        if line == '\n':
                flag = False
                bag = []
        if flag:
            bag.extend([w for w in word_tokenize(line.lower()) if not w in sw])
            match = re.search(r'(<tag \")\w+', line)
            if match:
                bag.remove('tag')
                bag.remove(match[0][6:])
                vocab.extend(bag)
                train_data[match[0][6:]].append(bag)
    vocab = set(vocab)
    train_data['vocab'] = vocab
    return train_data

# Calculate probability of seeing a sense class based on training data frequency
def prob_c(training, smooth=1):
    classprobs = {}
    for k in training:
        if k != 'vocab':
            classprobs[k] = len(training[k])
    n = sum(classprobs.values())
    for c in classprobs:
        classprobs[c] = float(classprobs[c] + smooth)/(n + smooth * len(classprobs))
    return classprobs

# Calculate probability of seeing certain linguistic features based on previously seen sense class
def prob_feature(training, feat_vocab, smooth=1):
    featureprobs = defaultdict(dict)
    featurecounts = defaultdict(dict)
    for c in training:
        if c != 'vocab':
            flat_train = [i for sub in training[c] for i in sub]
            counts = Counter(flat_train)
            for f in feat_vocab:
                featurecounts[c][f] = counts[f] + 1
    for c in featurecounts:
        n = sum(featurecounts[c].values())
        for f in feat_vocab:
            featureprobs[c][f] = float(featurecounts[c].get(f,0) + smooth)/(n + smooth * len(feat_vocab))
        featureprobs[c]['LAMBDA'] = float(smooth)/(n + smooth * len(feat_vocab))
    return featureprobs

# Predict what sense an observed word is being used in based on the probability distributions of sense class/features
def predict_sense(c_probdist, feat_probdist, obs):
    sense_probs = defaultdict(dict)
    for c in c_probdist:
        if c != 'vocab':
            sense_probs[c] = c_probdist[c]
            for w in obs:
                if w in feat_probdist[c]:
                    sense_probs[c] *= feat_probdist[c][w]
                else:
                    sense_probs[c] *= feat_probdist[c]['LAMBDA']
    sense_choice = max(sense_probs, key=sense_probs.get)
    return sense_choice

# Determine how accurate the model is, judged against gold standard data
def eval_accuracy(c_probdist, feat_probdist, test_data, gold_data, dict_data):
    correct = 0
    wrong = 0
    for i in range(len(test_data)):
        pred = predict_sense(c_probdist, feat_probdist, test_data[i])
        tag = dict_data[pred]['tag']
        g = gold_data[i]
        if tag in gold_data[i]:
            correct+=1
        else:
            wrong+=1
    acc = float(correct) / (correct + wrong)
    return acc

def main():
    if len(sys.argv) != 5:
        print("Usage: python naivebayes.py training_data testing_data gold_standard dict_data")
        return
    training = read_train(sys.argv[1])
    testing = read_test(sys.argv[2])
    gold = read_gold(sys.argv[3])
    dictionary = read_dict(sys.argv[4])
    c_probdist = prob_c(training)
    f_probdist = prob_feature(training, training['vocab'])
    print("Accuracy of model:\n")
    print(eval_accuracy(c_probdist, f_probdist, testing, gold, dictionary))
    return

if __name__ == '__main__':
    main()