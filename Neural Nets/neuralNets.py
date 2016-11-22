import numpy as np 		# numpy for nan values
import random			# random to create crossfold validation

#from sklearn.neural_network import MLPClassifier	# neaural nets

filename = "./train_set.csv"

def isNan(item):
	# Check if data entry is nan
	return (item == "nan")

# row is a list, dataMap is a list of string labels for row indices
# sets nps_score to -1 for detractor, 1 for promotoer, and 0 for indifferent
def preprocessNPSScore(row, dataMap):
	npsIndex = dataMap.index("last_nps_score")
	mapDict = {"det": -1, "pas": 0, "prom": 1}

	if isNan(row[npsIndex]):
		row[npsIndex] = mapDict["pas"]
		return row
	else:
		try:
			if (int(row[npsIndex]) <= 6):
				row[npsIndex] = mapDict["det"]
			elif (int(row[npsIndex]) >= 9):
				row[npsIndex] = mapDict["prom"]
			else:
				row[npsIndex] = mapDict["pas"]
		except():
			print("Invalid NPS score value entered: " + str(row[npsIndex]))
			exit()
		return row


# returns (dataSet, dataMap) where data set is a list of lists of values, 
# and dataMap is string labels for each value
def preprocessData(data):
	##### process features:
	dataSet = list()
	dataMap = data.readline().split(",")

	rowNumber = 0
	for row in data.readlines():
		rowList = row.split(",")
		### NPS Score
		## change to score of promoter, detractor, indifferent. Null values are set to indifferent
		rowList = preprocessNPSScore(rowList, dataMap)
		### time_since_action values
		## For each in {client_created, invoice_created, quote_created, basic_task_completed, payment_recieved, epayments_used, } :
		## 		Add boolean value for "hasDoneInLastWeek"
		## 		Add boolean value for "hasDoneInLastMonth"
		## 		Remove time_since_action


		### Account value
		## Change to avg value or possible just remove feature
		dataSet.append(rowList)
		rowNumber += 1
	return (dataSet, dataMap)

def removeUnwantedEntries(data):
	# Remove entries that requested card termination
	return


def splitOffRandomValidation(data, percentValidation):
	# Now move random percentValidation% to a validation set
	# And remove from test set
	return

def verifyData(trainData):
	# Check that no features are invalid
	return


def main():
	trainData = open(filename, 'r')
	(dataSet, dataMap) = preprocessData(trainData)
	for row in dataSet:
		#print("===")
		#print(row)
		#print("nps_score: " + row[dataMap.index("last_nps_score")])
		return

	# trainData = removeUnwantedEntries(trainData)

	# verifyData(trainData)

	# (trainSet, validationSet) = splitOffRandomValidation(trainSet, percentValidation)


	# Learn trainSet

	# Test against validationSet

main()