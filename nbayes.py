
import math
import pandas as pd

import argparse
from pathlib import Path
import os
from mldata import *
import random

def dataDiscretize(dataSet):
    m,n = pd.shape(dataSet)    #获取数据集行列（样本数和特征数)
    disMat = pd.tile([0],pd.shape(dataSet))  #初始化离散化数据集
    for i in range(n-1):    #由于最后一列为类别，因此遍历前n-1列，即遍历特征列
        x = [l[i] for l in dataSet] #获取第i+1特征向量
        y = pd.cut(x,10,labels=[0,1,2,3,4,5,6,7,8,9])   #调用cut函数，将特征离散化为10类，可根据自己需求更改离散化种类
        for k in range(n):  #将离散化值传入离散化数据集
            disMat[k][i] = y[k]
    return disMat


def cross_validation_5folds(dataset):
    random.seed(12345)
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / 5)
    for i in range(5):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    return dataset_split

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
 
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
 
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries
 
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries
 
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions
 
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
 
def nbayes():
	splitRatio = 0.67

	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)



# Command line parameters


parser = argparse.ArgumentParser(prog='nbayes.py', description='Run Naive Bayes Algorithm')

# Option 1 : path to the data
parser.add_argument('dataLocation', help="Input your data location after Python file",
                    type=str)
parser.add_argument('validationType', help="0 for cross validation, 1 for run algorithm on the full sample",
                    type=int, choices=[0, 1])
parser.add_argument('-n','--number_bins', help="Number of bins for any continuous feature, at least 2.",
                    type=int, choices=range(2, 1000))

parser.add_argument('m_value', help="m-estimate, Laplace smoothing if negative, maximum likelihood estimate if 0",
                    type=float)

args = parser.parse_args()

print("Dateset Location:", args.dataLocation, "Validation Type:", args.validationType,
      "Max Depth:", args.max_depth_of_tree, "Info Gain Type:", args.informationGainType)

# load dataset
dataPath = Path(args.dataLocation)
(dirname, dataname) = os.path.split(dataPath)

dataset = parse_c45(os.path.basename(dataPath), rootdir=dirname)

max_depth = args.max_depth_of_tree
validation_type = args.validationType
split_criterion = args.informationGainType


dataset = parse_c45('voting')
print(dataset[10])

nbayes()
