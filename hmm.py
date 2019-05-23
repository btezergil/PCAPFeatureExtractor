import numpy as np
import argparse
import datetime
import extractor

TOLERANCE = 1e-6
MAX_ITERATIONS = 1e+5
OBSERVATION_LENGTH = 17
OBSERVATION_BINARY_COUNT = 6

# FOR ALL IMPLEMENTATIONS, THE FOLLOWING PARAMETERS ARE USED: 
# there are N hidden states, t observations with each having M elements. 
# In this t observations, each of them may have different distributionsm see the extractor.py file for ours.
# a is the state transition matrix, size NxN
# b is the emission matrix, size NxM, every ith row has the same distribution as ith observation
# o is the observation matrix, size Nxt 
# pi is the initial state probability, for our case it is fixed with the first state having 1 and the rest 0

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
    for t in range(1, timeStep):
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

def baum_welch(a, b, pi, o):

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]

    xi = np.zeros((numberOfStates, numberOfStates, timeStep))

    iters = 0
    error = TOLERANCE + 10
    while iters < MAX_ITERATIONS and error > TOLERANCE:
        prev_a = a.copy()
        prev_b = b.copy()

        # Estimate model parameters
        alpha = forward(a, b, o, pi)
        beta = backward(a, b, o)

        # Expectation step
        for t in range(0, timeStep-1):
            for i in range(0, numberOfStates):
                for j in range(0, numberOfStates):
                    xi[i,j,t] = alpha[t][i] * a[i][j] * getprob(b, o[t], j) * beta[t+1][j]
                xi[:,:,t] /= np.sum(xi[:,:,t])

        for i in range(0, numberOfStates):
            for j in range(0, numberOfStates):
                xi[i,j,timeStep-1] = alpha[timeStep-1][i] * a[i][j]
            xi[:,:,timeStep-1] /= np.sum(xi[:,:,timeStep-1])

        # Maximization step
        # note that delta[t][i] = sum(0, numberOfStates-1, xi[i,:,t])
        for i in range(0, numberOfStates):
            pi[i] = np.sum(xi[i,:,0])
            for j in range(0, numberOfStates):
                a[i][j] = np.sum(xi[i,j,:timeStep-1]) / np.sum(xi[i,:,:timeStep-1])    

            gamma_tarr = []
            # gamma_tarr holds all gamma_t(i) values for a fixed i and variable t
            for t in range(0,timeStep):
                gamma_tarr.append(np.sum(xi[i,:,t]))

            for k in range(1, OBSERVATION_BINARY_COUNT+1):
                zerosum = []
                onesum = []
                for t in range(0, timeStep):
                    obs = o[t][k]
                    if obs:
                        zerosum.append(0)
                        onesum.append(1)
                    else:
                        zerosum.append(1)
                        onesum.append(0)

                b[i][k]['0'] = np.sum(gamma_tarr * zerosum) / np.sum(xi[i,:,:])
                b[i][k]['1'] = np.sum(gamma_tarr * onesum) / np.sum(xi[i,:,:])
            for k in range(OBSERVATION_BINARY_COUNT+1, OBSERVATION_LENGTH):
                zerosum = []
                onesum = []
                twosum = []

                for t in range(0, timeStep):
                    obs = o[t][k]
                    if not obs:
                        zerosum.append(1)
                        onesum.append(0)
                        twosum.append(0)
                    elif obs == 1:
                        zerosum.append(0)
                        onesum.append(1)
                        twosum.append(0)
                    elif obs == 2:
                        zerosum.append(0)
                        onesum.append(0)
                        twosum.append(1)

                b[i][k]['0'] = np.sum(gamma_tarr * zerosum) / np.sum(xi[i,:,:])
                b[i][k]['1'] = np.sum(gamma_tarr * onesum) / np.sum(xi[i,:,:])
                b[i][k]['2'] = np.sum(gamma_tarr * twosum) / np.sum(xi[i,:,:])

        error = (np.abs(a-prev_a)).max() + (np.abs(b-prev_b)).max() 
        iters += 1            
        print ("Iteration: ", iters, " error: ", error, "P(O|lambda): ", np.sum(alpha[:,T-1]))
    
    return a, b, pi, alpha 

def initializeMatrices(statecount):
    # initializes the a and b matrices that will be used for HMM
    a = []
    b = []
    pi = [1]

    # initialize a using dirichlet distribution for every row
    a = np.random.dirichlet(np.ones(statecount),size=statecount)

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
        # dirichlet distribution is used to initialize the emission probabilities
        for j in range(1, 7):
            rand = np.random.dirichlet(np.ones(2),size=1)
            dct = {'0':rand[0][0], '1':rand[0][1]}
            tmp.append(dct)
        for j in range(7, 17):
            rand = np.random.dirichlet(np.ones(3),size=1)
            dct = {'0':rand[0][0], '1':rand[0][1], '2':rand[0][2]}
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
    
    # extract the features from pcap in the form of an array of Features objects
    features = extractor.extract_features(args.filename, args.interval)
    o = []
    # transform the features into list form that will be used by HMM
    for i in range(len(features)):
        o.append(features[i].getObsArray())
        
    # initialize a, b, pi with random probabilities
    (a, b, pi) = initializeMatrices(args.stateCount)

    # estimate the emission and transition probabilities using Baum-Welch alg.
    (a_est, b_est, pi_est, alpha_est) = baum_welch(a, b, pi, o)

    # get the proability of emittion observation o using viterbi alg.
    (path, delta, phi) = viterbi(a_est, b_est, o, pi_est)

if __name__ == "__main__":
    main()