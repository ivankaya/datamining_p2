import numpy as np
import time
from sklearn.kernel_approximation import RBFSampler

np.random.seed(13)

def transform(x_original):
	#x_original = x_original.reshape([1, -1])
	#print x_original.shape
	rbf_features = RBFSampler(gamma=30, random_state=1, n_components=5300)
	x_trans = rbf_features.fit_transform(x_original)
	#x_trans = x_trans.reshape([-1])
	return x_trans


def mapper(key, value):
    start = time.clock()
    # key: None
    # value: one line of input file
    alpha = 0.003
    beta_1 = 0.9
    beta_2 = 0.99
    weights = np.zeros([5300], dtype='float') #maybe adjust this
    m = np.zeros([5300], dtype='float')
    v = np.zeros([5300], dtype='float')
    m_hat = np.zeros([5300], dtype='float')
    v_hat = np.zeros([5300], dtype='float')
    counter = 0.0
    print 'Mapping...'

    values = []
    for i in value:
     	tokens = i.strip()
     	features = np.fromstring(tokens[0:], dtype=float, sep=' ')
	if len(values) == 0:
	    values = [features]
	else:
	    values = np.vstack((values, features))
    #print values.shape
    print 'Permuting...'
    values = np.random.permutation(values)
    
    y = values[:, 0]
    kaki = values[:, 1:]
   	
    #print labels[0:10]
    #print features[0, 0:10]
    print 'Transforming...'
    x = transform(kaki)
    #print x[0, 0:10]

    
    print 'Fitting...'
    for j in xrange(0, 75):
		for i in xrange(x.shape[0]):
		    counter = counter + 1.0
		    if (y[i]*np.dot(weights,x[i])) < 1:
		        m = beta_1 * m - (1.0 - beta_1)*y[i]*x[i]
		        v = beta_2 * v + (1.0 - beta_2)*(np.multiply(x[i],x[i]))
		    else:
		    	m = beta_1 * m
		    	v = beta_2 * v
		    m_hat = m / (1.0 - beta_1**counter)
		    v_hat = v / (1.0 - beta_2**counter)
		    weights = weights - np.divide((alpha*m_hat), (np.sqrt(v_hat) + 0.000001))

    print 'Time elapsed for mapper: ', time.clock()-start
    yield 1,weights


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    
    weights_final = 0.0
    num_weights = 0.0
    for i in values:
        weights_final = weights_final + i
        num_weights = num_weights + 1.0
    #print num_weights
    yield weights_final/num_weights
