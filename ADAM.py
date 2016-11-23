import numpy as np

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return X


def mapper(key, value):
    # key: None
    # value: one line of input file
    alpha = 0.001
    beta_1 = 0.9
    beta_2 = 0.99
    weights = np.zeros([400], dtype='float') #maybe adjust this
    m = np.zeros([400], dtype='float')
    v = np.zeros([400], dtype='float')
    m_hat = np.zeros([400], dtype='float')
    v_hat = np.zeros([400], dtype='float')
    counter = 0.0
    for j in xrange(0,10):
		for i in value:
		    tokens = i.split()
		    y = float(tokens[0])
		    x = np.asarray(tokens[1:]).astype(np.float)
		    counter = counter + 1.0
		    if (y*np.inner(weights,x)) < 1:
		        m = beta_1 * m - (1.0 - beta_1)*y*x
		        v = beta_2 * v + (1.0 - beta_2)*np.multiply((0.0-y*x), (0.0-y*x))
		    else:
		    	m = beta_1 * m
		    	v = beta_2 * v
		    m_hat = m / (1.0 - beta_1**counter)
		    v_hat = v / (1.0 - beta_2**counter)
		    weights = weights - np.divide((alpha*m_hat), (np.sqrt(v_hat) + 0.000001))

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
