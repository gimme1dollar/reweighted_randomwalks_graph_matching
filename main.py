import numpy as np
import numpy.matlib as npm
#from graphs import scene_graph

'''
Most algorithm adopted from
% Minsu Cho, Jungmin Lee, and Kyoung Mu Lee, 
% Reweighted Random Walks for Graph Matching, 
% Proc. European Conference on Computer Vision (ECCV), 2010
% http://cv.snu.ac.kr/research/~RRWM/
'''

class RRWM:
    def __init__(self, G1 = [], G2 = [], M = None):
        self.G1 = G1
        self.G2 = G2
        self.M  = M
        self.make_group() # conflicting match groups in domain (size(M, 1) x nGroup)

    def make_group(self):
        nG1 = len(self.G1)
        nG2 = len(self.G2)

        E = np.ones((nG1, nG2))
        L = np.vstack((np.where(E)[1], np.where(E)[0]))

        nMatch = np.shape(L)[1] # nG1 * nG2

        featList = L[0]
        featList_unique = list(set(featList))
        nGroup = len(featList_unique)
        self.group1 = np.zeros((nMatch, nGroup))
        for i in range(nGroup):
            self.group1[np.where(featList == featList_unique[i]), i] = 1

        featList = L[1]
        featList_unique = list(set(featList))
        nGroup = len(featList_unique)
        self.group2 = np.zeros((nMatch, nGroup))
        for i in range(nGroup):
            self.group2[np.where(featList == featList_unique[i]), i] = 1

        return

    def get_id_from_group(self, group=None, idx1=None, ID1=None, idx2=None, ID2=None):
        # get_id_from_group
        if group is not None:
            nGroup = int(np.shape(group)[1])
            idx = []
            ID  = []
            for i in range(nGroup):
                tmp = np.where(group[:,i])
                if i == 0:
                    idx = tmp
                    ID  = i * np.ones(np.shape(tmp))
                else:
                    idx = np.hstack((idx, tmp))
                    ID  = np.hstack((ID, i * np.ones(np.shape(tmp))))
            return idx[0], ID[0]
        # get_id_from_group_slack
        else:
            nG1 = int(ID1[-1]) + 1
            nG2 = int(ID2[-1]) + 1

            maxIdx = int(np.shape(idx1)[0])
            addIdx = np.array([ i for i in range(maxIdx, maxIdx+nG2) ])
            addIdx = np.reshape(addIdx, (-1,1))

            idx1 = np.reshape(idx1, (-1,1))
            idx1 = np.vstack((idx1, addIdx))
            ID1  = np.reshape(ID1, (-1,1))
            ID1  = np.vstack((ID1, nG1 * np.ones((nG2,1))))

            addID   = np.array([i for i in range(0, nG2+1)])
            addID   = np.reshape(addID, (-1,1))
            tempIdx = np.zeros((maxIdx+nG2, 1))
            tempID  = np.zeros((maxIdx+nG2, 1))

            i, j = 0, 0
            while i < maxIdx-1:
                tempIdx[i+j] = idx2[i]
                tempID[i+j]  = ID2[i]
                if(ID2[i] != ID2[i+1]):
                    tempIdx[i+j+1] = addIdx[j]
                    tempID[i+j+1]  = addID[j]
                    j += 1
                i += 1

            while i <= maxIdx-1:
                tempIdx[i+j] = idx2[i]
                tempID[i+j]  = ID2[i]
                if i == maxIdx-1:
                    tempIdx[i+j+1] = addIdx[j]
                    tempID[i+j+1]  = addID[j]
                i += 1

            idx1 = [ int(idx1) for idx1 in np.reshape(idx1, -1) ]
            ID1  = [ int(id1) for id1 in np.reshape(ID1, -1)]
            idx2 = [ int(idx2) for idx2 in np.reshape(tempIdx, -1) ]
            ID2  = [ int(id2) for id2 in np.reshape(tempID, -1) ]

            return idx1, ID1, idx2, ID2, nG2 - nG1, nG2

    def get_conflict_matrix(self):
        group = np.hstack((self.group1,  self.group2))
        nMatch = np.shape(group)[0]

        conflict_matrix = np.zeros((nMatch, nMatch))
        for c in range(np.shape(group)[1]):
            idx = np.where(group[:, c])[0]
            for ii in idx:
                conflict_matrix[ii, np.where(group[:, c])[0]] = 1
        conflict_matrix = np.reshape(conflict_matrix, -1)
        for i in range(0, np.size(conflict_matrix), nMatch+1):
            conflict_matrix[i] = 0
        conflict_matrix = np.reshape(conflict_matrix, (nMatch, nMatch))

        not_conflict_matrix = np.zeros((nMatch, nMatch))
        not_conflict_matrix[np.where(conflict_matrix==0)] = 1

        return conflict_matrix, not_conflict_matrix

    def bistochastic_normalize_match_slack(self, pX,
                                           pIdx1, pID1, pIdx2, pID2,
                                           pTol, pDumDim, pDumVal, pMaxIter):
        N   = np.shape(pX)[0]
        pX  = np.reshape(pX, -1)

        dlt = pTol
        pX2 = np.zeros(N)
        pTemp = np.zeros(N)

        # number of groups
        nG1 = int(pID1[-1])
        nG2 = int(pID2[-1])

        iter = 0
        while(dlt >= pTol and iter < pMaxIter):
            iter+=1

            # copy current state
            for i in range(N):
                pX2[i] = pX[i]

            # update on domain 1
            pTemp[0] = pX[pIdx1[0]]
            for i in range(1, N, 1):
                if(pID1[i] == pID1[i-1]):
                    pTemp[i] = pTemp[i-1] + pX[pIdx1[i]]
                else:
                    pTemp[i] = pX[pIdx1[i]]
            for i in range(N-2, -1, -1):
                if(pID1[i] == pID1[i+1]):
                    pTemp[i] = pTemp[i+1]
            for i in range(N):
                pX[pIdx1[i]] /= pTemp[i]
            if(pDumDim == 1):
                for i in range(N-nG2, N, 1):
                    pX[i] *= pDumVal

            # update on domain 2
            pTemp[0] = pX[pIdx2[0]]
            for i in range(1, N, 1):
                if(pID2[i] == pID2[i-1]):
                    pTemp[i] = pTemp[i-1] + pX[pIdx2[i]]
                else:
                    pTemp[i] = pX[pIdx2[i]]
            for i in range(N-2, -1, -1):
                if(pID2[i] == pID2[i+1]):
                    pTemp[i] = pTemp[i+1]
            for i in range(N):
                pX[pIdx2[i]] /= pTemp[i]
            if(pDumDim == 2):
                for i in range(N-nG1, N, 1):
                    pX[i] *= pDumVal

            dlt = 0
            for i in range(N):
                dlt += (pX[i]-pX2[i])**2

        pX = np.reshape(pX, (N,1))
        return pX


    def run(self, c=0.15, amp_max=30, iterMax=300, thresConvergence=1e-30, tolC=1e-3):
        ## index (for normalization) for what??
        idx1, ID1 = self.get_id_from_group(self.group1)
        idx2, ID2 = self.get_id_from_group(self.group2)
        if ID1[-1] < ID2[-1]:
            idx1, ID1, idx2, ID2, dumVal, dumSize = self.get_id_from_group(None, idx1, ID1, idx2, ID2)
            dumDim = 1
        elif ID1[-1] > ID2[-1]:
            idx1, ID1, idx2, ID2, dumVal, dumSize = self.get_id_from_group(None, idx2, ID2, idx1, ID1)
            dumDim = 2
        else:
            dumDim, dumVal, dumSize = 0, 0, 0

        ## eleminate conflicting elements to prevent conflicting walks
        _, not_conf_mat = self.get_conflict_matrix()
        self.M *= not_conf_mat

        ## column-wise stochastic
        d    = [sum(col) for col in zip(*self.M)]
        maxD = max(d)
        M    = self.M / maxD

        ## main iteration
        nMatch = np.shape(self.M)[0]
        prev_score  = np.ones((nMatch, 1))/nMatch
        prev_score2 = prev_score
        prev_assign = np.ones((nMatch, 1))/nMatch

        bCont  = 1
        iter_i = 0
        while bCont and iter_i < iterMax:
            iter_i += 1

            # random walking with reweighted jumps
            cur_score = M @ (c*prev_score + (1-c)*prev_assign)

            sum_cur_score = sum(cur_score)
            if sum_cur_score > 0:
                cur_score = cur_score/sum_cur_score

            # update reweighted jump
            cur_assign = cur_score
            amp_value  = amp_max / max(cur_assign)
            cur_assign = np.exp(amp_value * cur_assign)

            # Sinkorn method of iterative bistocastic normalizations
            X_slack = cur_assign
            if dumSize > 0:
                X_slack = np.vstack((X_slack, dumVal * np.ones((dumSize, 1))))
            X_slack = self.bistochastic_normalize_match_slack(X_slack,
                                                              idx1, ID1, idx2, ID2,
                                                              tolC, dumDim, dumVal, 100)
            cur_assign = X_slack[:nMatch]

            sum_cur_assign = sum(cur_assign)
            if sum_cur_assign > 0:
                cur_assign = cur_assign / sum_cur_assign

            # check convergence
            diff1 = sum((cur_score - prev_score)**2)
            diff2 = sum((cur_score - prev_score2)**2)
            diff_min = min(diff1, diff2)
            if diff_min < thresConvergence:
                bCont = 0

            prev_score2 = prev_score
            prev_score  = cur_score
            prev_assign = cur_assign
        return cur_score

    def greedy_mapping(self, score):
        Xd = np.zeros((len(score),1))
        max_value, max_ind = np.max(score), np.argmax(score)
        while(max_value > 0):
            Xd[max_ind] = 1
            group1_ind = np.where(self.group1[max_ind, :])
            score_ind1 = np.where(self.group1[:, group1_ind[0]])[0]
            score[score_ind1] = 0
            group2_ind = np.where(self.group2[max_ind, :])
            score_ind2 = np.where(self.group2[:, group2_ind[0]])[0]
            score[score_ind2] = 0
            max_value, max_ind = np.max(score), np.argmax(score)
        return Xd

if __name__ == '__main__':
    for out in range(1, 2, 1):
        ## Generate two graphs
        nInlier, nOutlier = 2, out
        deformation = 0.15
        nP1, nP2 = nInlier, nInlier+nOutlier

        randperm = np.random.permutation(nP2)

        # G1
        G1 = np.random.rand(nP1, nP1)
        G1 = np.tril(G1, -1)
        G1 = G1 + np.transpose(G1)

        # G2
        G2 = np.random.rand(nP2, nP2)
        G2 = np.tril(G2, -1)
        G2 = G2 + np.transpose(G2)
        for row in range(nInlier):
            for col in range(nInlier):
                G2[randperm[row], randperm[col]] = G1[row, col]

        #N = np.random.rand(nP2, nP2)
        #N = deformation * np.tril(N, -1)
        #N = N + np.transpose(N)
        #G2 = G2 + N

        # M
        scale2D = .15
        M1 = npm.repmat(G1, nP2, nP2)
        M2 = np.kron(G2, np.ones((nP1, nP1)))
        M  = (M1 - M2) ** 2
        M  = np.exp(-M / scale2D)

        M[np.isnan(M)] = 0
        M = np.reshape(M, -1)
        for i in range(0, M.size, nP1 * nP2 +1):
            M[i] = 0
        M = np.reshape(M, (nP1 * nP2, nP1 * nP2))
        M = M + np.transpose(M)

        # GTBool
        GT_matrix = np.zeros((nP1, nP2))
        for i in range(nInlier):
            GT_matrix[i, randperm[i]] = 1
        GT_matrix = np.transpose(GT_matrix)
        GT_bool = np.reshape(GT_matrix, (-1, 1))

        ## RRWM on G1 & G2
        cRRWM = RRWM(G1=G1, G2=G2, M=M)

        Xraw = cRRWM.run()

        Xd = cRRWM.greedy_mapping(Xraw)
        #print(len(np.where(np.reshape(Xd, (nP2, nP1)))[0]))
        score = np.transpose(Xd) @ M @ Xd
        accuracy = np.transpose(Xd) @ GT_bool / sum(GT_bool)
        print(f"Exp with inlier {nInlier} & outier {nOutlier} \n"
              f"accuracy: {accuracy} \t score: {score}\n")