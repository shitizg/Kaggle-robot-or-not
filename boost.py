import pandas as pd
import numpy as np


"""

train = pd.read_csv('train.csv')
feature_set = pd.read_csv('features.csv')
test = pd.read_csv('test.csv')

#print (list(feature_set.columns.values))

features_train = pd.DataFrame(columns=('dt_others_median', 'dt_self_median', 'ips_per_bidder', 'dt_change_ip_median', 'bids_per_auction_per_ip_entropy_mean', 'n_bids', 'bids_per_auction_mean', 'ip_entropy', 'vasstdc27m7nks3', 'n_bids_url', 'payment_account_0875307e1731af94b3b64725ad0deb7d', 'countries_per_bidder_per_auction_max', 'on_ip_that_has_a_bot_mean', 'f_urls', 'url_entropy'))
features_test = pd.DataFrame(columns=('dt_others_median', 'dt_self_median', 'ips_per_bidder', 'dt_change_ip_median', 'bids_per_auction_per_ip_entropy_mean', 'n_bids', 'bids_per_auction_mean', 'ip_entropy', 'vasstdc27m7nks3', 'n_bids_url', 'payment_account_0875307e1731af94b3b64725ad0deb7d', 'countries_per_bidder_per_auction_max', 'on_ip_that_has_a_bot_mean', 'f_urls', 'url_entropy'))
#feattrain = []
#feattest = []

for i in range(train.shape[0]):
	row = feature_set[feature_set.bidder_id == train.loc[i]['bidder_id']]
	others = row['dt_others_median'][i]
	#feattrain.append(row.fillna(0).values)
	#print (feattrain[i])
	self = row['dt_self_median'][i]
	ips = row['ips_per_bidder_per_auction_mean'][i]
	change = row['dt_change_ip_median'][i]
	bids_per_auction_per_ip_entropy_mean = row['bids_per_auction_per_ip_entropy_mean'][i]
	n_bids = row['n_bids'][i]
	bids_per_auction_mean = row['bids_per_auction_mean'][i]
	ip_entropy = row['ip_entropy'][i]
	vasstdc27m7nks3 = row['vasstdc27m7nks3'][i]
	n_bids_url = row['n_bids_url'][i]
	payment_account_0875307e1731af94b3b64725ad0deb7d = row['payment_account_0875307e1731af94b3b64725ad0deb7d'][i]
	countries_per_bidder_per_auction_max = row['countries_per_bidder_per_auction_max'][i]
	on_ip_that_has_a_bot_mean = row['on_ip_that_has_a_bot_mean'][i]
	f_urls = row['f_urls'][i]
	url_entropy = row['url_entropy'][i]
	features_train.loc[i] = [others, self, ips, change, bids_per_auction_per_ip_entropy_mean, n_bids, bids_per_auction_mean, ip_entropy, vasstdc27m7nks3, n_bids_url, payment_account_0875307e1731af94b3b64725ad0deb7d, countries_per_bidder_per_auction_max, on_ip_that_has_a_bot_mean, f_urls, url_entropy]

#feattrain = feattrain[1:]
features_train = features_train.fillna(0)
trainingdata = features_train[['dt_others_median', 'dt_self_median', 'ips_per_bidder', 'dt_change_ip_median', 'bids_per_auction_per_ip_entropy_mean', 'n_bids', 'bids_per_auction_mean', 'ip_entropy', 'vasstdc27m7nks3', 'n_bids_url', 'payment_account_0875307e1731af94b3b64725ad0deb7d', 'countries_per_bidder_per_auction_max', 'on_ip_that_has_a_bot_mean', 'f_urls', 'url_entropy']].values

for i in range(test.shape[0]):
	row = feature_set[feature_set.bidder_id == test.loc[i]['bidder_id']]
	level = (np.where(feature_set['bidder_id'] == test.loc[i]['bidder_id'])[0])[0]
	#feattest.append(row.fillna(0).values)
	others = row['dt_others_median'][level]
	self = row['dt_self_median'][level]
	ips = row['ips_per_bidder_per_auction_mean'][level]
	change = row['dt_change_ip_median'][level]
	bids_per_auction_per_ip_entropy_mean = row['bids_per_auction_per_ip_entropy_mean'][level]
	n_bids = row['n_bids'][level]
	bids_per_auction_mean = row['bids_per_auction_mean'][level]
	ip_entropy = row['ip_entropy'][level]
	vasstdc27m7nks3 = row['vasstdc27m7nks3'][level]
	n_bids_url = row['n_bids_url'][level]
	payment_account_0875307e1731af94b3b64725ad0deb7d = row['payment_account_0875307e1731af94b3b64725ad0deb7d'][level]
	countries_per_bidder_per_auction_max = row['countries_per_bidder_per_auction_max'][level]
	on_ip_that_has_a_bot_mean = row['on_ip_that_has_a_bot_mean'][level]
	f_urls = row['f_urls'][level]
	url_entropy = row['url_entropy'][level]
	features_test.loc[i] = [others, self, ips, change, bids_per_auction_per_ip_entropy_mean, n_bids, bids_per_auction_mean, ip_entropy, vasstdc27m7nks3, n_bids_url, payment_account_0875307e1731af94b3b64725ad0deb7d, countries_per_bidder_per_auction_max, on_ip_that_has_a_bot_mean, f_urls, url_entropy]

#feattest = feattest[1:]
features_test = features_test.fillna(0)
test_values = features_test[['dt_others_median', 'dt_self_median', 'ips_per_bidder' , 'dt_change_ip_median', 'bids_per_auction_per_ip_entropy_mean', 'n_bids', 'bids_per_auction_mean', 'ip_entropy', 'vasstdc27m7nks3', 'n_bids_url', 'payment_account_0875307e1731af94b3b64725ad0deb7d', 'countries_per_bidder_per_auction_max', 'on_ip_that_has_a_bot_mean', 'f_urls', 'url_entropy']].values
#print (test_values)

"""



class decStump:
    def __init__(self):
        self.gtlabel = None
        self.ltlabel = None
        self.splitThreshold = None
        self.splitFeature = None

    def classify(self, point):
        if point[self.splitFeature] > self.splitThreshold :
            return self.gtlabel
        else:
            return self.ltlabel
"""

def classify(dataM, dim, thresh, lt_gt):
    ret = np.ones((np.shape(dataM)[0],1))
    if lt_gt == 'lt':
        ret[dataM[:,dim] <= thresh] = -1.0
    else:
        ret[dataM[:,dim] > thresh] = -1.0
    return ret

def create_Stump(data, classLabels, data_weight):
	dataM = np.matrix(data)
	#print ("Calling create_stump")
	labelM = np.matrix(classLabels).T
	m,n = np.shape(dataM)
	steps = 1000.0		#steps for cross validation splitting
	bestStump = {}      #dictionary for stump
	bestClassEst = np.zeros((m,1))
	minErr = np.inf
	for i in range(n):					#iterate through features
		minimum = dataM[:,i].min()
		maximum = dataM[:,i].max()
		step_size = (maximum - minimum)/steps
		for j in range(-1,int(steps)+1):
			for lt_gt in ['lt', 'gt']:
				thresh = (minimum + float(j) * step_size)	#threshold
				pVals = classify(dataM, i, thresh, lt_gt)	#predicted values per feature
				errVals = np.ones((m,1))
				errVals[pVals == labelM] = 0
				w_error = np.transpose(data_weight).dot(errVals)
				#print ("w_error", w_error)
				#print ("minErr", minErr)
				if w_error < minErr:
					minErr = w_error
					bestClassEst = pVals.copy()
					bestStump['dim'] = i
					bestStump['threshold'] = thresh
					bestStump['lt_gt'] = lt_gt
	return bestStump, minErr, bestClassEst


def boostingTrain(data, classLabels, rounds=40):
	weakClassArr = []
	m = np.shape(data)[0]             #length
	data_weight = (np.ones((m,1)))/m    #weight vector
	aggregate_class_estimate = np.zeros((m,1))
	for i in range(rounds):
		bestStump, err, classEst = create_Stump(data, classLabels, data_weight)
		alpha = float(0.5*np.log((1.0-err)/max(err, 1e-16)))
		#alpha = 0.001
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		ex = np.multiply(-1*alpha*np.matrix(classLabels).T ,classEst)
		data_weight = np.multiply(data_weight, np.exp(ex))
		data_weight = data_weight/data_weight.sum()
		aggregate_class_estimate += alpha*classEst
		aggregateErrors = np.multiply(np.sign(aggregate_class_estimate) != np.matrix(classLabels).T, np.ones((m,1)))
		errRate = aggregateErrors.sum()/m
		if errRate == 0.0 : break
	return weakClassArr

labels = train['outcome'].values
for i in range(len(labels)):
	if labels[i] == 1:
		labels[i] = 1.0
	else:
		labels[i] = -1.0

classifiers = boostingTrain(trainingdata, labels, 250)
#classifiers = boostingTrain(feattrain, labels, 100)
#print (classifiers)

def adaClassify(datToClass, classifiers):       #classifies test data with weak classifiers
    #dataM = np.matrix(datToClass)
    m = np.shape(datToClass)[0]
    aggregate_class_estimate = np.zeros((m,1))
    for i in range(len(classifiers)):
        classEst = classify(datToClass, classifiers[i]['dim'], classifiers[i]['threshold'], classifiers[i]['lt_gt'])
        aggregate_class_estimate += classifiers[i]['alpha']*classEst
    return np.sign(aggregate_class_estimate)


def find_best_predictor(test_values, training_data, labels):
	iterations = 1
	maxCount = 0
	for i in range(50):
		classifierzz = boostingTrain(training_data, labels, i)
		prediction = adaClassify(test_values, classifierzz)
		count = 0
		for j in range(len(prediction)):
			if prediction[j] == 1:
				count += 1
		if count > maxCount:
			iterations = i
			maxCount = count
	return iterations

prediction = adaClassify(test_values, classifiers)
#prediction = adaClassify(feattest, classifiers)

count = 0
for i in range(len(prediction)):
	if prediction[i] == 1:
		prediction[i] = 1
		count += 1
	else:
		prediction[i] = 0

print ("Bots in Data: ", count)

output = pd.DataFrame()
output['bidder_id'] = test['bidder_id']
output['prediction'] = prediction
output.to_csv('prediction_leggo.csv', sep=',', index = False, header=True, columns=['bidder_id', 'prediction'])

"""
