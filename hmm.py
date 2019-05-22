import numpy as np
import argparse
import datetime
import extractor

TOLERANCE = 1e-6
MAX_ITERATIONS = 1e+5

# FOR ALL IMPLEMENTATIONS, THE FOLLOWING PARAMETERS ARE USED: 
# there are N hidden states, t observations with each having M elements. 
# In this t observations, each of them may have different distributionsm see the extractor.py file for ours.
# a is the state transition matrix, size NxN
# b is the emission matrix, size NxM, every ith row has the same distribution as ith observation
# o is the observation matrix, size Nxt 
# pi is the initial state probability, for our case it is fixed with the first state having 1 and the rest 0

def scalar_dict_mult(scalar, dct):
    # DICTIONARY MULTIPLICATION WITH SCALAR CODE
    # either uses lst for list of dictionaries or uses dct for a single dictionary
    newdict = {}
    for key in dct:
        newdict[key] = scalar*dct[key]

    return newdict

def scalar_listofdicts_mult(scalar, lst):
    newlist = []
    
    for val in lst:
        newdict = {}
        for key in val:
            newdict[key] = 2*val[key]
        newlist.append(newdict)

    return newlist

def list_listofdicts_mult(lst, dictlst):
    newlist = []
    
    for i in range(0, len(lst)):


    for val in lst:
        newdict = {}
        for key in val:
            newdict[key] = 2*val[key]
        newlist.append(newdict)

    return newlist

def getprob(b, o, state):
    # gets the probability of observing o by getting all the probabilities of elements in o and multiplying them
    prob = 1
    for i in range(0, len(b[state])):
        prob *= b[state][i][o[i]]
    return prob

def forward(a, b, o, pi):
    # HMM forward algorithm implementation 

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]
    alpha = [[]]

    # initialization step
    for i in range(0, numberOfStates):
        alpha[0].append(pi[i], getprob(b, o[0], i))

    # inductive step
    for t in range(1, timeStep):
        tmp = []
        
        for i in range(0, numberOfStates):
            probsum = 0
            for j in range(0, numberOfStates):
                probsum += alpha[t-1][j] * a[j][i]
            tmp.append(probsum * getprob(b, o[t], i))
        
        alpha.append(tmp)

    return alpha

def backward(a, b, o):
    # HMM backward algorithm implementation

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]

    beta = [[]]

    # initialization step
    for i in range(0, numberOfStates):
        beta[0].append(1)

    # inductive step
    for t in range(timeStep-2, -1, -1):
        for i in range(0, numberOfStates):
            probsum = 0
            for j in range(0, numberOfStates):
                # beta[0] is used here because since we fill it by using the procedure, the last element inserted is the result of the last iteration
                probsum += beta[0][j] * a[j][i] * getprob(b, o[t+1], j)
            tmp.append(probsum)
        beta.insert(0, tmp)

    return beta

def viterbi(a, b, o, pi):
    # HMM viterbi algorithm implementation

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]

    delta = [[]]
    phi = [[]]

    path = []

    # initialization step
    for i in range(0, numberOfStates):
        phi[0][i] = 0
        delta[0][i] = pi[i] * getprob(b, o[0], i)

    # inductive step
    for t in range(1, timeStep):.
        path.append([])
        for i in range(0, numberOfStates):
            maxarr = []
            # note that for phi and delta the same argument is used, delta has an extra factor and phi uses argmax instead of max, so we use the same list
            for j in range(0, numberOfStates):
                maxarr.append(delta[t-1][j] * a[j][i])
            
            phi[t][i] = np.argmax(maxarr)
            delta[t][i] = max(maxarr) * getprob(b, o[t], i)

    path[-1] = np.argmax(delta[timeStep-1])

    for t in range(timeStep-2, -1, -1):
        path[t] = phi[t+1][path[t+1]]

    return path, delta, phi
 
    def HMMBaumWelch(self, o, N, dirichlet=False, verbose=False, rand_seed=1):
        # Implements HMM Baum-Welch algorithm        
        
        T = np.shape(o)[0]

        M = int(max(o))+1 # now all hist time-series will contain all observation vals, but we have to provide for all

        digamma = np.zeros((N,N,T))

    
        # Initialise A, B and pi randomly, but so that they sum to one
        np.random.seed(rand_seed)
        
        # Initialisation can be done either using dirichlet distribution (all randoms sum to one) 
        # or using approximates uniforms from matrix sizes
        if dirichlet:
            pi = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))
            
            a = np.random.dirichlet(np.ones(N),size=N)
            
            b=np.random.dirichlet(np.ones(M),size=N)
        else:
            
            pi_randomizer = np.ndarray.flatten(np.random.dirichlet(np.ones(N),size=1))/100
            pi=1.0/N*np.ones(N)-pi_randomizer

            a_randomizer = np.random.dirichlet(np.ones(N),size=N)/100
            a=1.0/N*np.ones([N,N])-a_randomizer

            b_randomizer=np.random.dirichlet(np.ones(M),size=N)/100
            b = 1.0/M*np.ones([N,M])-b_randomizer

        
        error = self.tolerance+10
        itter = 0
        while ((error > self.tolerance) & (itter < self.max_iter)):   

            prev_a = a.copy()
            prev_b = b.copy()
    
            # Estimate model parameters
            alpha, c = self.HMMfwd(a, b, o, pi)
            beta = self.HMMbwd(a, b, o, c) 
    
            for t in xrange(T-1):
                for i in xrange(N):
                    for j in xrange(N):
                        digamma[i,j,t] = alpha[i,t]*a[i,j]*b[j,o[t+1]]*beta[j,t+1]
                digamma[:,:,t] /= np.sum(digamma[:,:,t])
    

            for i in xrange(N):
                for j in xrange(N):
                    digamma[i,j,T-1] = alpha[i,T-1]*a[i,j]
            digamma[:,:,T-1] /= np.sum(digamma[:,:,T-1])
    
            # Maximize parameter expectation
            for i in xrange(N):
                pi[i] = np.sum(digamma[i,:,0])
                for j in xrange(N):
                    a[i,j] = np.sum(digamma[i,j,:T-1])/np.sum(digamma[i,:,:T-1])
    	

                for k in xrange(M):
                    filter_vals = (o==k).nonzero()
                    b[i,k] = np.sum(digamma[i,:,filter_vals])/np.sum(digamma[i,:,:])
    
            error = (np.abs(a-prev_a)).max() + (np.abs(b-prev_b)).max() 
            itter += 1            
            
            if verbose:            
                print ("Iteration: ", itter, " error: ", error, "P(O|lambda): ", np.sum(alpha[:,T-1]))
    
        return a, b, pi, alpha  

    (a, b, pi_est, alpha_est) = hmm.HMMBaumWelch(hist_O, 2, False, True)
    (path, delta, phi)=hmm.HMMViterbi(a, b, hist_O, pi_est)

def initializeMatrices(statecount):
    # initializes the a and b matrices that will be used for HMM
    a = []
    b = []
    pi = [1]
    prob = 1 / statecount
    # TODO: noise can be added to prob in order to differentiate the probabilities
    # initialize a, currently uniform probabilities
    for i in range(0, statecount):
        tmp = []
        for j in range(0, statecount):
            tmp.append(prob)
        a.append(tmp)

    # a = np.random.dirichlet(np.ones(statecount),size=statecount)

    # initialize b
    # b array structure is as follows:
    # in one row:
    # x1-x3 -> exists params (0 or 1)
    # x4 -> firstSTUNMessageTime (0 or 1)
    # x5-x6 -> DNSAtStart-End params (0 or 1)
    # x7-x9 -> STUN-ICMP-UDP rates (0,1,2)
    # x10-x13 -> UDP-TLS-STUN-TCP host count (0,1,2)
    # x14-x16 -> TLS-STUN-TCP percentages (0,1,2)
    for i in range(0, statecount):
        tmp = []
        # TODO: the probabilities in dct can be random using dirichlet distr.
        for j in range(1, 7):
            dct = {'0':0.5, '1':0.5}
            tmp.append(dct)
        for j in range(7, 17):
            dct = {'0':0.33, '1':0.33, '2':0.34}
            tmp.append(dct)
        b.append(tmp)

    # initialize pi, our starting state is always the first state
    for i in range(0, statecount-1):
        pi.append(0)

    return (a, b, pi)

def main():
    parser = argparse.ArgumentParser(description = 'Extract features from given pcap file.')
    parser.add_argument("filename", type = str)
    parser.add_argument("stateCount", type = int)
    parser.add_argument("interval", type = float)
    args = parser.parse_args()

    print("Extracting from file {} with an interval of {} seconds\n".format(args.filename, args.interval))
    extractor.extract_features(args.filename, args.interval)

if __name__ == "__main__":
    main()