import numpy as np

# Read particles
def read_particles(path):
    
    f = open(path)
    n = [int(x) for x in f.readline().split()]

    ap = ()
    for i in range(0,n[0]):
        m = [int(x) for x in f.readline().split()]

        p = np.array([])
        for j in range(0,m[0]):
            pts = [float(x) for x in f.readline().split()]
            p = np.append(p,pts)

        p = p.reshape(m[0],3)
        ap = ap + (p,)
    
    return ap

# Neighbor sets
def neighbor_sets(p, Rn): 
    sets = np.array([np.array([j for j, p2 in enumerate(p) if np.linalg.norm(p1-p2)<Rn]) for i, p1 in enumerate(p)])    
    return(sets)


# Candidate sets
def candidate_sets(p1, p2, Rs): 
    sets = np.array([np.array([j for j, pc in enumerate(p2) if np.linalg.norm(pr-pc)<Rs]) for i, pr in enumerate(p1)])
    return(sets)

# Relaxation sets
def relaxation_sets(Sr, Sc, p1, p2, E, F, method):
    
    theta = ()
    for i, set in enumerate(Sc):
        thetaj = ()
        for id, j in enumerate(set):
            
            dij = p2[j] - p1[i]
            
            if method==1:
                thresh = E + F*np.linalg.norm(dij)
            elif method==2:
                thresh = F
            
            pairs = [(k,l) for k in Sr[i] for l in Sc[i] \
                     if np.linalg.norm((p2[j] - p1[i])-(p2[l] - p1[k]))<thresh \
                     and i!=k and j!=l]
                        
            thetaj = thetaj + (pairs,)
        theta = theta + (thetaj,)
    
    return(theta)

# Initialize probability matrices
def init_probability_matrices(Pij, Pi, Sc):
    
    for i, set in enumerate(Sc):
        m = np.shape(set)[0]
        for j in set:
            Pij[i][j] = 1./(m+1.)
        Pi[i] = 1./(m+1.)
        
    return Pij, Pi

# Normalize probability matrices
def norm_probability_matrices(Pij, Pi):
    
    for id, row in enumerate(Pij):
        sum = np.sum(row) + Pi[id]
        Pij[id] /= sum
        Pi[id] /= sum
        
    return Pij, Pi

# Iterations 1
def track(Sr, Sc, p1, p2, theta, n):
    
    Pij = np.zeros((np.shape(p1)[0], np.shape(p2)[0]), dtype='float32'); Pi = np.zeros(np.shape(p1)[0], dtype='float32');
    Pij2 = np.zeros((np.shape(p1)[0], np.shape(p2)[0]), dtype='float32'); Pi2 = np.zeros(np.shape(p2)[0], dtype='float32');
    Pij, Pi = init_probability_matrices(Pij, Pi, Sc); Pij2, Pi2 = init_probability_matrices(Pij2, Pi2, Sc);

    for N in range(0,n):
        for i, set in enumerate(Sc):
            for id, j in enumerate(set):

                sum = 0.
                for k, l in theta[i][id]:
                    sum += Pij[k][l]                    

                Pij2[i][j] = Pij[i][j]*(0.3 + 3.0*sum)
                
        Pij2, Pi2 = norm_probability_matrices(Pij2, Pi2)

        diff = np.sum(np.abs(Pij-Pij2)) + np.sum(np.abs(Pi-Pi2))
        # print diff

        Pij = np.array(Pij2)
        Pi = np.array(Pi2)
        
        if diff<0.01:
            break
    
    return Pij, Pi, diff, N
