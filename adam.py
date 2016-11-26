import numpy as np
#import time

#Parameters that can be tuned:
mDim=400#number of dimensions in data
gamma = 30 #the sqrt of the regular gamma of regular rbf
epsilon = 0.000001 #small tweak number
lam = 1.0 #parameter to tune the power within each iteration in mapper
kk = 2 #number of iterations for convergence
alpha = 0.003 #alpha parameter for adam
beta_1 = 0.9 # beta1 param for adam
beta_2 = 0.99 # beta2 param for adam
np.random.seed(13)
w1=np.zeros([mDim*400],dtype='float')
b=np.zeros([mDim],dtype='float')
w1=np.random.normal(loc=0.0, scale=1.0, size=400*mDim)
b=np.random.uniform(0,2*np.pi,size=mDim)
def transform(x_original):
    global w1
    global b
    global gamma
    global mDim
    Xtrans=gamma*np.dot(x_original,np.reshape(w1, (400, mDim)))
    Xtrans = (np.sqrt(2.0) / np.sqrt(mDim))*np.concatenate((np.cos(Xtrans+np.reshape(b, (1, mDim))),np.cos(-Xtrans+np.reshape(b, (1, mDim)))),axis = 1)
    return Xtrans

def mapper(key, value):
    global lam
    global epsilon
    global kk
    global alpha
    global beta_1
    global beta_2
    #start = time.clock()
    # key: None
    # value: one line of input file
    weights = np.zeros([2*mDim], dtype='float') #maybe adjust this
    m = np.zeros([2*mDim], dtype='float')
    v = np.zeros([2*mDim], dtype='float')
    m_hat = np.zeros([2*mDim], dtype='float')
    v_hat = np.zeros([2*mDim], dtype='float')
    counter = 0.0
    #print('Mapping...')
    #start = time.clock()
    dat = np.zeros([len(value),401], dtype='f')
    count = 0
    for i in value:
        tokens = i.split()
        x = np.asarray(tokens[0:]).astype(np.float)
        dat[count,:] = x
        count = count + 1
    #dat = np.array(value).astype(np.float)
    #print values.shape
    #print 'Permuting...'
    #values = np.random.permutation(values)
    
    y = dat[:, 0]
    #kaki = dat[:, 1:]
    #print('Time elapsed for mapper: ', time.clock()-start)   	
    #print labels[0:10]
    #print features[0, 0:10]
    #print('Transforming...')
    x = transform(dat[:, 1:])
    #print x[0, 0:10]
    for j in range(kk):
        randindex = np.random.permutation(np.shape(x)[0])
        for i in range(x.shape[0]):
            counter = counter + 1.0
            rand = randindex[i]
            #print(np.shape(x[rand]))
            L = np.dot(weights,x[rand,:])
            if (y[rand]*L <=0):
                m = beta_1 * m - (1.0 - beta_1)*y[rand]*x[rand,:]
                v = beta_2 * v + (1.0 - beta_2)*(np.multiply(x[rand,:],x[rand,:]))
            elif (y[rand]*L > 0) and (y[rand]*L < 1):
                m = beta_1 * m - (1.0 - beta_1)*y[rand]*(1-L)*x[rand,:]
                v = beta_2 * v + (1.0 - beta_2)*(1-L)*(1-L)*(np.multiply(x[rand,:],x[rand,:]))
            else:
                m = beta_1 * m
                v = beta_2 * v
            m_hat = m / (1.0 - beta_1**(lam*counter))
            v_hat = v / (1.0 - beta_2**(lam*counter))
            weights = weights - np.divide((alpha*m_hat), (np.sqrt(v_hat) + epsilon))
    print('fin')
    yield 1,weights


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    #start = time.clock()
    weights_final = 0.0
    num_weights = 0.0
    for i in values:
        weights_final = weights_final + i
        num_weights = num_weights + 1.0
    #print num_weights
    #print 'Time elapsed for reducer: ', time.clock()-start                              
    yield weights_final/num_weights
