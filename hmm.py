import numpy as np
import argparse
import datetime
import extractor
import copy
import pickle

TOLERANCE = 1e-5
ERR_TOLERANCE = 1e-6
MAX_ITERATIONS = 1e+5
OBSERVATION_LENGTH = 16
OBSERVATION_BINARY_COUNT = 6

OBSERVATION_PATH = '/home/btezergil/Desktop/research488/pickledObs/'

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
        prob *= b[state][i][str(o[i])]
    return prob

def getmaxdiff(b, prev_b):
    # gets the maximum difference of probability between b and prev_b
    allprobs = []
    for i in range(len(b)):
        for j in range(len(b[i])):
            newdct = b[i][j]
            olddct = prev_b[i][j]
            for k in range(len(newdct.values())):
                allprobs.append(abs(list(newdct.values())[k] - list(olddct.values())[k]))

    return max(allprobs)

def logtonormalb(b):
    for i in range(0, len(b)):
        for j in range(0, OBSERVATION_BINARY_COUNT):
            for k in range(0,2):
                #if b[i][j][str(k)]: 
                b[i][j][str(k)] = np.exp(b[i][j][str(k)])
        for j in range(OBSERVATION_BINARY_COUNT, OBSERVATION_LENGTH):
            for k in range(0,3):
                #if b[i][j][str(k)]: 
                b[i][j][str(k)] = np.exp(b[i][j][str(k)])

def normalize(a, b, pi):
    # EXPERIMENTAL
    # eliminates zero possibilities from a,b and pi arrays
    scaled = False
    for i in range(len(pi)):
        if pi[i] < TOLERANCE:
            pi[i] = TOLERANCE
            scaled = True
    if scaled:
        denom = np.sum(pi)
        for i in range(len(pi)):
            pi[i] /= denom

    scaled = False
    for i in range(len(a)):
        for j in range(len(a[0])):
            if a[i][j] < TOLERANCE:
                a[i][j] = TOLERANCE
                scaled = True
        if scaled:
            denom = np.sum(a[i])
            for j in range(len(a[0])):
                a[i][j] /= denom
            scaled = False

    scaled = False
    for i in range(0, len(b)):
        for j in range(0, OBSERVATION_BINARY_COUNT):
            for k in range(0,2):
                if b[i][j][str(k)] < TOLERANCE:  
                    b[i][j][str(k)] = TOLERANCE
                    scaled = True
            if scaled:
                denom = b[i][j]['0'] + b[i][j]['1']
                # scale back to 1
                for k in range(0,2):
                    b[i][j][str(k)] /= denom
            scaled = False
        for j in range(OBSERVATION_BINARY_COUNT, OBSERVATION_LENGTH):
            for k in range(0,3):
                if b[i][j][str(k)] < TOLERANCE:  
                    b[i][j][str(k)] = TOLERANCE
                    scaled = True
            if scaled:
                denom = b[i][j]['0'] + b[i][j]['1'] + b[i][j]['2']
                # scale back to 1
                for k in range(0,3):
                    b[i][j][str(k)] /= denom
            scaled = False

def mergeArrays(all_as, all_bs, all_pis):
    # EXPERIMENTAL
    # gets an array of a's, b's and pi's and merges them into one a,b,pi
    (a, b, pi) = initializeMatrices(len(all_pis[0]))

    for i in range(len(all_pis[0])):
        probsum = 0
        for j in range(len(all_pis)):
            probsum += all_pis[j][i]
        pi[i] = probsum / extractor.CAP_COUNT

    for i in range(len(all_as[0])):
        for j in range(len(all_as[0])):
            probsum = 0
            for k in range(len(all_as)):
                probsum += all_as[k][i][j]
            a[i][j] = probsum / extractor.CAP_COUNT

    for i in range(0, len(all_bs[0])):
        for j in range(0, OBSERVATION_BINARY_COUNT):
            for k in range(0,2):
                probsum = 0
                for l in range(len(all_bs)):
                    probsum += all_bs[l][i][j][str(k)]
                b[i][j][str(k)] = probsum / extractor.CAP_COUNT
        for j in range(OBSERVATION_BINARY_COUNT, OBSERVATION_LENGTH):
            for k in range(0,3):
                probsum = 0
                for l in range(len(all_bs)):
                    probsum += all_bs[l][i][j][str(k)]
                b[i][j][str(k)] = probsum / extractor.CAP_COUNT

    return a, b, pi


def forward(a, b, o, pi):
    # HMM forward algorithm implementation 

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]
    alpha = [[]]

    # initialization step
    for i in range(0, numberOfStates):
        alpha[0].append(pi[i] * getprob(b, o[0], i))

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

def forwardLog(a, b, o, pi):
    # HMM forward algorithm with log probabilities

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]
    alpha = [[]]

    # initialization step
    for i in range(0, numberOfStates):
        alpha[0].append(np.log(pi[i]) + np.log(getprob(b, o[0], i)))

    # inductive step
    for t in range(1, timeStep):
        tmp = []
        for i in range(0, numberOfStates):
            probsum = 0
            bmax = float('-inf')

            # find b = (max a_i)
            for j in range(0, numberOfStates):
                val = alpha[t-1][j] + np.log(a[j][i])
                if bmax < val:
                    bmax = val

            for j in range(0, numberOfStates):
                probsum += np.exp(alpha[t-1][j] + np.log(a[j][i]) - bmax)

            tmp.append(bmax + np.log(probsum) + np.log(getprob(b, o[t], i)))
        
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
        tmp = []
        for i in range(0, numberOfStates):
            probsum = 0
            for j in range(0, numberOfStates):
                # beta[0] is used here because since we fill it by using the procedure, the last element inserted is the result of the last iteration
                probsum += beta[0][j] * a[j][i] * getprob(b, o[t+1], j)
            tmp.append(probsum)
        beta.insert(0, tmp)

    return beta

def backwardLog(a, b, o):
    # HMM backward algorithm with log probabilities

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]

    beta = [[]]

    # initialization step
    for i in range(0, numberOfStates):
        beta[0].append(np.exp(1))

    # inductive step
    for t in range(timeStep-2, -1, -1):
        tmp = []
        for i in range(0, numberOfStates):
            probsum = 0
            bmax = float('-inf')

            for j in range(0, numberOfStates):
                # beta[0] is used here because since we fill it by using the procedure, the last element inserted is the result of the last iteration
                val = beta[0][j] + np.log(a[j][i]) + np.log(getprob(b, o[t+1], j))
                if bmax < val:
                    bmax = val

            for j in range(0, numberOfStates):
                # beta[0] is used here because since we fill it by using the procedure, the last element inserted is the result of the last iteration
                probsum += np.exp(beta[0][j] + np.log(a[j][i]) + np.log(getprob(b, o[t+1], j)) - bmax)
            tmp.append(bmax + np.log(probsum))

        beta.insert(0, tmp)

    return beta

def viterbi(a, b, o, pi):
    # HMM viterbi algorithm implementation

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]

    delta = np.zeros((numberOfStates, timeStep))
    phi = np.zeros((numberOfStates, timeStep))

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

def viterbiLog(a, b, o, pi):
    # HMM viterbi algorithm implementation

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]

    delta = np.zeros((numberOfStates, timeStep))
    phi = np.zeros((numberOfStates, timeStep))

    path = [0 for i in range(timeStep)]

    # initialization step
    for i in range(0, numberOfStates):
        phi[i,0] = 0
        delta[i,0] = np.log(pi[i]) + np.log(getprob(b, o[0], i))

    # inductive step
    for t in range(1, timeStep):
        for i in range(0, numberOfStates):
            maxarr = []
            # note that for phi and delta the same argument is used, delta has an extra factor and phi uses argmax instead of max, so we use the same list
            for j in range(0, numberOfStates):
                maxarr.append(delta[j,t-1] + np.log(a[j][i]))
            
            phi[i,t] = np.argmax(maxarr)
            delta[i,t] = np.max(maxarr) + np.log(getprob(b, o[t], i))

    path[-1] = np.argmax(delta[:,timeStep-1])

    for t in range(timeStep-3, -1, -1):
        path[t] = phi[int(path[t+1]),t+1]

    return path, delta, phi

def baum_welch(a, b, pi, o):

    numberOfStates = np.shape(a)[0]
    timeStep = np.shape(o)[0]

    xi = np.zeros((numberOfStates, numberOfStates, timeStep))

    iters = 0
    error = TOLERANCE + 10
    while iters < MAX_ITERATIONS and error > ERR_TOLERANCE:
        prev_a = np.copy(a)
        prev_b = copy.deepcopy(b)
        #prev_pi = np.copy(pi)

        #all_as = []
        #all_bs = []
        #all_pis = []

        #for ind in range(extractor.CAP_COUNT):
        #    o = obsArr[ind]
        #    a = prev_a
        #    b = prev_b
        #    pi = prev_pi

            # Estimate model parameters
        alpha = forwardLog(a, b, o, pi)
        beta = backwardLog(a, b, o)

        # Expectation step
        for t in range(0, timeStep-1):
            for i in range(0, numberOfStates):
                for j in range(0, numberOfStates):
                    xi[i,j,t] = alpha[t][i] + np.log(a[i][j]) + np.log(getprob(b, o[t+1], j)) + beta[t+1][j]
            xi[:,:,t] -= np.max(xi[:,:,t]) + np.log(np.sum(np.exp(xi[:,:,t] - np.max(xi[:,:,t]))))

        for i in range(0, numberOfStates):
            for j in range(0, numberOfStates):
                xi[i,j,timeStep-1] = alpha[timeStep-1][i] + np.log(a[i][j])
        xi[:,:,timeStep-1] -= np.max(xi[:,:,timeStep-1]) + np.log(np.sum(np.exp(xi[:,:,timeStep-1] - np.max(xi[:,:,timeStep-1]))))

        # Maximization step
        # note that a,b,pi values are in normal space, so we have to exp them back from log space
        for i in range(0, numberOfStates):
            pi[i] = np.max(xi[i,:,0]) + np.log(np.sum(np.exp(xi[i,:,0] - np.max(xi[i,:,0]))))

            for j in range(0, numberOfStates):
                a[i][j] = np.max(xi[i,j,:timeStep-1]) + np.log(np.sum(np.exp(xi[i,j,:timeStep-1] - np.max(xi[i,j,:timeStep-1])))) - np.max(xi[i,:,:timeStep-1]) - np.log(np.sum(np.exp(xi[i,:,:timeStep-1] - np.max(xi[i,:,:timeStep-1]))))

            denom = np.max(xi[i,:,:]) + np.log(np.sum(np.exp(xi[i,:,:] - np.max(xi[i,:,:]))))

            for k in range(0, OBSERVATION_BINARY_COUNT):
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

                tmpzeroarr = (np.array(xi[i,:,:]) * np.array(zerosum)).flatten()
                tmponearr = (np.array(xi[i,:,:]) * np.array(onesum)).flatten()
                zeroarr = []
                onearr = []
                # get 0's out from the arrays as they will not be counting and cause problems in the log space
                for l in range(0, len(tmpzeroarr)):
                    if tmpzeroarr[l]:
                        zeroarr.append(tmpzeroarr[l])
                    if tmponearr[l]:
                        onearr.append(tmponearr[l])
                zeroarr = np.array(zeroarr)
                onearr = np.array(onearr)

                if 1 in zerosum:
                    b[i][k]['0'] = np.max(zeroarr) + np.log(np.sum(np.exp(zeroarr - np.max(zeroarr)))) - denom
                else:
                    b[i][k]['0'] = float('-inf')
                if 1 in onesum:                    
                    b[i][k]['1'] = np.max(onearr) + np.log(np.sum(np.exp(onearr - np.max(onearr)))) - denom
                else:
                    b[i][k]['1'] = float('-inf')
            for k in range(OBSERVATION_BINARY_COUNT, OBSERVATION_LENGTH):
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

                tmpzeroarr = (np.array(xi[i,:,:]) * np.array(zerosum)).flatten()
                tmponearr = (np.array(xi[i,:,:]) * np.array(onesum)).flatten()
                tmptwoarr = (np.array(xi[i,:,:]) * np.array(twosum)).flatten()
                zeroarr = []
                onearr = []
                twoarr = []
                for l in range(0, len(tmpzeroarr)):
                    if tmpzeroarr[l]:
                        zeroarr.append(tmpzeroarr[l])
                    if tmponearr[l]:
                        onearr.append(tmponearr[l]) 
                    if tmptwoarr[l]:
                        twoarr.append(tmptwoarr[l])
                zeroarr = np.array(zeroarr)
                onearr = np.array(onearr)
                twoarr = np.array(twoarr)

                if 1 in zerosum :
                    b[i][k]['0'] = np.max(zeroarr) + np.log(np.sum(np.exp(zeroarr - np.max(zeroarr)))) - denom
                else:
                    b[i][k]['0'] = float('-inf')
                if 1 in onesum:
                    b[i][k]['1'] = np.max(onearr) + np.log(np.sum(np.exp(onearr - np.max(onearr)))) - denom
                else:
                    b[i][k]['1'] = float('-inf')
                if 1 in twosum:
                    b[i][k]['2'] = np.max(twoarr) + np.log(np.sum(np.exp(twoarr - np.max(twoarr)))) - denom
                else:
                    b[i][k]['2'] = float('-inf')

        # take exp of all values in a,b, and pi and take them back to normal probability space
        a = np.exp(a)
        pi = np.exp(pi)
        logtonormalb(b)
        normalize(a, b, pi)

        #all_as.append(a)
        #all_bs.append(b)
        #all_pis.append(pi)

        # merge all a,b,pi's into one array of their own
        #(a, b, pi) = mergeArrays(all_as, all_bs, all_pis)

        error = (np.abs(a-prev_a)).max() + getmaxdiff(b, prev_b) 
        iters += 1            
        print ("Iteration: ", iters, " error: ", error, "P(O|lambda): ", np.sum(np.exp(alpha[timeStep-1])))
    
    return a, b, pi, alpha 

def initializeMatrices(statecount):
    # initializes the a and b matrices that will be used for HMM
    b = []

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
    for _ in range(0, statecount):
        tmp = []
        # dirichlet distribution is used to initialize the emission probabilities
        for _ in range(0, OBSERVATION_BINARY_COUNT):
            rand = np.random.dirichlet(np.ones(2),size=1)
            dct = {'0':rand[0][0], '1':rand[0][1]}
            tmp.append(dct)
        for _ in range(OBSERVATION_BINARY_COUNT, OBSERVATION_LENGTH):
            rand = np.random.dirichlet(np.ones(3),size=1)
            dct = {'0':rand[0][0], '1':rand[0][1], '2':rand[0][2]}
            tmp.append(dct)
        b.append(tmp)

    # initialize pi, our starting state is always the first state
    pi = np.random.dirichlet(np.ones(statecount))

    return (a, b, pi)

def main():
    parser = argparse.ArgumentParser(description = 'Extract features from given pcap file.')
    parser.add_argument("filename", type = str)
    #parser.add_argument("appname", type = str)
    parser.add_argument("stateCount", type = int)
    parser.add_argument("interval", type = float)
    args = parser.parse_args()

    print("Extracting for {} with an interval of {} seconds\n".format(args.filename, args.interval))
    
    # extract the features from pcap in the form of an array of Features objects
    #features = extractor.extract_fileList(args.appname, args.interval)
    #obsArr = []
    # transform the features into list form that will be used by HMM
    #for i in range(extractor.CAP_COUNT):
    #    o = []
    #    for j in range(len(features[i])):
    #        o.append(features[i][j].getObsArray())
    #    obsArr.append(o)

    try:
        # get the features in pickled form
        with open(OBSERVATION_PATH + args.filename + '.pickle', 'rb') as f:
            o = pickle.load(f)
    except FileNotFoundError:
        # extract the features from pcap in the form of an array of Features objects
        features = extractor.extract_features(args.filename, args.interval)
        # transform the features into list form that will be used by HMM
        o = []
        for i in range(len(features)):
            o.append(features[i].getObsArray())
        
    # initialize a, b, pi with random probabilities
    (a, b, pi) = initializeMatrices(args.stateCount)

    # estimate the emission and transition probabilities using Baum-Welch alg.
    (a_est, b_est, pi_est, alpha_est) = baum_welch(a, b, pi, o)

    with open('data.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump((a_est, b_est, pi_est), f, pickle.HIGHEST_PROTOCOL)
    #(a_est, b_est, pi_est, alpha_est) = baum_welch(a, b, pi, obsArr)

    # get the proability of emittion observation o using viterbi alg.
    #(path, delta, phi) = viterbiLog(a_est, b_est, o, pi_est)
    #(path, delta, phi) = viterbiLog(a_est, b_est, obsArr, pi_est)

if __name__ == "__main__":
    main()