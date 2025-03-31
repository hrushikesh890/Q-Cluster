import numpy as np
import random
import collections
from sklearn.cluster import KMeans
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator

def convert_data(data, n):
    """
    Converts data values by multiplying them by n.
    
    Args:
        data (dict): Dictionary containing data values
        n (int): Number to multiply values by
        
    Returns:
        dict: Dictionary with values multiplied by n
    """
    for k in data.keys():
        data[k] = int(data[k]*n)
    return data

def random_bitflip(ip, prob=0.45, seed = 10):
    ret = {}
    random.seed(seed)
    for z in ip.keys():
        for k in range(0, ip[z]):
            s = list(z)
            for i in range(0, len(z)):
                if random.random() < prob:
                    s[i] = '0' if s[i] == '1' else '1'
            s = "".join(s)
            ret[s] = ret.get(s, 0) + 1
    return collections.Counter(ret)

def multiply_dict(data, n):
    """
    Multiplies each value in the dictionary by n.
    
    Args:
        data (dict): Dictionary containing data values
        n (int): Number to multiply values by
    """
    new_data = {}
    for k in data.keys():
        new_data[k] = int(data[k]*n)
    return new_data

def generate_bitstring_dict(num_strings, num_bits, total_sum, seed=32):
    # Generate unique bitstrings
    random.seed(seed)
    bitstrings = set()
    while len(bitstrings) < num_strings:
        bitstrings.add(''.join(random.choice('01') for _ in range(num_bits)))
    
    bitstrings = list(bitstrings)

    # Generate random positive values that sum to total_sum
    partitions = sorted(random.sample(range(1, total_sum), num_strings - 1))
    if num_strings == 1:
        values = [total_sum]
    else:
        values = [partitions[0]] + [partitions[i] - partitions[i-1] for i in range(1, num_strings - 1)] + [total_sum - partitions[-1]]

    # Shuffle values to avoid predictable distribution
    random.shuffle(values)

    return dict(zip(bitstrings, values))

def restruct_data(ip, count):
    """
    Restructures data by repeating each key's list based on its count.
    
    Args:
        ip (dict): Dictionary mapping lists to counts
        count (int): Multiplier for counts
        
    Returns:
        np.array: Array containing repeated lists converted to integers
    """
    n = []
    for k in ip.keys():
        for i in range(int(ip[k]*count)):
            n.append(list(k))
    for i in range(len(n)):
        for j in range(len(n[i])):
            n[i][j] = int(n[i][j])
    return np.array(n)


class KMeans1DBitNA:
    """
    K-means clustering for binary strings with noise-aware distance metric.
    
    Args:
        n_clusters (int): Number of clusters
        max_iter (int): Maximum number of iterations
        seed (int): Random seed
        cent (list): Initial centroids
        threshold (float): Hamming distance threshold
        init_method (str): Method for initializing centroids ('random' or 'top_k')
    """
    def __init__(self, n_clusters, max_iter=300, seed=10, cent=None, threshold=None, init_method='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.cent = cent
        self.threshold = threshold  # Hamming distance threshold
        self.init_method = init_method  # Method for initializing centroids ('random' or 'top_k')

    def fit(self, X):
        """
        Fits the k-means model to the data.
        
        Args:
            X (np.array): Array of binary strings
            
        Returns:
            tuple: (centroids, labels) - Final centroids and cluster assignments
        """
        X_set = list(set([tuple(row) for row in X]))
        if self.cent is None:
            if self.init_method == 'random':
                self.centroids = random.sample(X_set, self.n_clusters)
            elif self.init_method == 'top_k':
                self.centroids = self._get_top_k_bitstrings(X, self.n_clusters)
        else:
            self.centroids = self.cent
        
        array_of_tuples = [tuple(row) for row in self.centroids]
        check_set = set(array_of_tuples)
        
        nits = 100
        nconv = 5
        np.random.seed(self.seed)
        while (len(check_set) != len(self.centroids)) and (nits > 0):
            if self.init_method == 'random':
                self.centroids = random.sample(X_set, self.n_clusters)
            nits -= 1
        if nits <= 0:
            print("failed to initialize centroids correctly")
        
        for _ in range(self.max_iter):
            labels = self._assign_labels(X)
            new_centroids = self._get_new_centroids(X, labels)
            if np.allclose(self.centroids, new_centroids):
                if nconv == 0:
                    break
                else:
                    nconv -= 1
            
            self.centroids = new_centroids
        
        self.labels_ = self._assign_labels(X)
        return self.centroids, self.labels_
    
    def _convert_to_dictionary(self, res):
        """
        Converts results to a dictionary counting occurrences.
        
        Args:
            res (np.array): Array of binary strings
            
        Returns:
            Counter: Dictionary counting string occurrences
        """
        ret = {}
        for i in range(len(res)):
            k = res[i]
            s = ''.join(map(str, k))
            if s in ret.keys():
                ret[s] += 1
            else:
                ret[s] = 1
                
            if self.labels_[i] != -1:
                c = ''.join(map(str, self.centroids[self.labels_[i]]))
                if s == c:
                    continue
                if c in ret.keys():
                    ret[c] += 1
                else:
                    ret[c] = 1
        
        return collections.Counter(ret)
    
    def _get_top_k_bitstrings(self, X, k):
        """
        Gets k most frequent bitstrings from data.
        
        Args:
            X (np.array): Array of binary strings
            k (int): Number of strings to return
            
        Returns:
            list: k most frequent binary strings
        """
        unique, counts = np.unique(X, axis=0, return_counts=True)
        bitstring_counts = list(zip(unique, counts))
        sorted_bitstrings = sorted(bitstring_counts, key=lambda x: x[1], reverse=True)
        top_k_bitstrings = [list(item[0]) for item in sorted_bitstrings[:k]]
        return top_k_bitstrings
    
    def _assign_labels(self, X):
        """
        Assigns cluster labels based on Hamming distance to centroids.
        
        Args:
            X (np.array): Array of binary strings
            
        Returns:
            np.array: Cluster assignments for each string
        """
        int_bitstrings = X.astype(int)
        int_centroids = np.array(self.centroids).astype(int)
        expanded_bitstrings = int_bitstrings[:, np.newaxis, :]
        xor_result = np.bitwise_xor(expanded_bitstrings, int_centroids)
        distances = np.sum(xor_result, axis=2)
        
        labels = np.full(len(X), -1)
        min_distances = np.min(distances, axis=1)
        closest_centroids = np.argmin(distances, axis=1)
        
        for i in range(len(X)):
            if self.threshold is None or min_distances[i] <= self.threshold:
                labels[i] = closest_centroids[i]
        
        return labels   
    
    def _get_majority_votes(self, array):
        """
        Gets majority vote for each bit position.
        
        Args:
            array (np.array): Array of binary strings
            
        Returns:
            list: Majority vote (0/1) for each position
        """
        majority_votes = []
        for i in range(array.shape[1]):
            c_0 = np.sum(array[:, i] == 0)
            c_1 = np.sum(array[:, i] == 1)
            majority_votes.append(0 if c_0 > c_1 else 1)
        return majority_votes

    def _get_new_centroids(self, X, labels):
        """
        Updates centroids based on cluster assignments.
        
        Args:
            X (np.array): Array of binary strings
            labels (np.array): Cluster assignments
            
        Returns:
            np.array: Updated centroids
        """
        nc = []
        for k in range(len(self.centroids)):
            x = X[labels == k]
            if len(x) > 0:
                nc.append(self._get_majority_votes(x))
            else:
                nc.append(self.centroids[k])

        check_set = set([tuple(row) for row in nc])
        if len(check_set) != len(nc):
            print("failed something is wrong")
        return np.array(nc)


class QCluster:
    """
    Quantum clustering algorithm for noisy binary strings.
    
    Args:
        max_clusters (int): Maximum number of clusters
        n (int): Number of samples
        max_iter (int): Maximum iterations per clustering
        tol (float): Convergence tolerance
        method (int): Clustering method
        prob (float): Noise probability
        init_method (str): Centroid initialization method
        mul (float): Threshold multiplier
        s_clus (int): Starting number of clusters
        logging (bool): Enable logging
    """
    def __init__(self, max_clusters, n, max_iter=300, tol=0.95, method=0, prob = 0.1, init_method = "top_k", mul = 1.0, s_clus = 1, logging = False):
        self.max_clusters = max_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = []
        self.centroids = None
        self.raw_labels = []
        self.method = method
        self.clusters = {}
        self.prob = prob
        self.n = n
        self.init_method = init_method
        self.mul = mul
        self.f_clus = 1
        self.n_labels = []
        self.s_clus = s_clus
        self.logging = logging
        self.log_data = {}
        
    def fit(self, data):
        """
        Fits clustering model to data.
        
        Args:
            data (np.array): Array of binary strings
            
        Returns:
            dict: Cluster assignments
        """
        n_clusters = self.s_clus
        prev_fid = -1
        curr_fid = 0
        prev_labels = []
        
        threshold = self.n*self.prob*(1 - self.prob)*self.mul
        it = 0
        if self.logging:
            self.log_data['threshold'] = threshold 
        while ((curr_fid < self.tol) and (n_clusters <= (self.max_clusters))):
            kmeans = KMeans1DBitNA(n_clusters, max_iter=self.max_iter, threshold=threshold, init_method=self.init_method)
            
            kmeans.fit(data)
            prev_labels = self.labels_
            self.raw_labels = kmeans.labels_
            self.centroids = kmeans.centroids
            self.labels_ = self._reshape_distribution(data, kmeans.centroids, collections.Counter(kmeans.labels_))
            
            if n_clusters > self.s_clus:
                prev_fid = curr_fid
                curr_fid = hellinger_fidelity(prev_labels, self.labels_)
                self.n_labels = self.labels_

            if self.logging:
                self.log_data[str(it)] = [ curr_fid, np.unique(self.raw_labels, return_counts=True), self.centroids, self.labels_]
                it+= 1
            n_clusters += 1
        self.f_clus = n_clusters - 2
        
        self.labels_ = prev_labels
        
        return self.labels_
    
    def fit_x(self, data, n_clusters):
        """
        Fits model with fixed number of clusters.
        
        Args:
            data (np.array): Array of binary strings
            n_clusters (int): Number of clusters
            
        Returns:
            dict: Cluster assignments
        """
        threshold = self.n*self.prob*(1 - self.prob)*self.mul
        kmeans = KMeans1DBitNA(n_clusters, max_iter=self.max_iter, threshold=threshold, init_method=self.init_method)       
        kmeans.fit(data)
        self.raw_labels = kmeans.labels_
            
        self.centroids = kmeans.centroids
        self.labels_ = self._reshape_distribution(data, kmeans.centroids, collections.Counter(kmeans.labels_))
        
        return self.labels_

    def hamming_distance(self, str1, str2):
        """
        Calculates Hamming distance between binary strings.
        
        Args:
            str1 (str): First binary string
            str2 (str): Second binary string
            
        Returns:
            int: Hamming distance
        """
        if len(str1) != len(str2):
            raise ValueError("Strings must be of equal length")
        distance = sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))
        return distance

    def _renormalize(self, ip):
        """
        Normalizes dictionary values to sum to 1.
        
        Args:
            ip (dict): Dictionary of values
            
        Returns:
            dict: Normalized dictionary
        """
        s = sum(ip.values())
        if s == 0:
            print("uhoh")
        return {k: v/s for k, v in ip.items()}

    def _reshape_distribution(self, X, centroids, lbl):
        """
        Reshapes distribution based on centroids and labels.
        
        Args:
            X (np.array): Data array
            centroids (list): Cluster centroids
            lbl (Counter): Label counts
            
        Returns:
            Counter: Reshaped distribution
        """
        d = collections.Counter()
        ip = []
        for x_s in X:
            ip.append("".join(str(bit) for bit in x_s))
        ip = collections.Counter(ip)
        ip = self._renormalize(ip)
        lbl = self._renormalize(lbl)
        new_dict = {}
        centroids_l = []
        for c in centroids:
            centroids_l.append("".join(str(bit) for bit in c))
        k = len(centroids_l)
        idx = 0
        if len(lbl) == 0:
            print("uhoh")
        
        for c in set(centroids_l):
            if c in ip.keys():
                if idx in lbl.keys():
                    new_dict[c] = lbl[idx]
                else:
                    new_dict[c] = ip[c]
            else:
                if idx in lbl.keys():
                    new_dict[c] = lbl[idx]
            idx += 1
        membership = {}
        tot = sum(ip.values())
        keyl = list(ip.keys())
        for s in ip.keys():
            mm = np.zeros(k+1)
            mm[k] = ip[s]/tot
            for i in range(len(centroids_l)):
                c = centroids_l[i]
                if s != c:
                    hd = self.hamming_distance(s, c)
                    psc = self.prob**hd
                    pc = new_dict[c]/tot
                    ps = (1-self.prob)**(self.n-hd)
                    mm[i] = psc*pc*ps
                else:
                    mm[i] = 0.0
            for i in range(len(centroids_l)):
                mm[k] -= mm[i]
            membership[s] = mm
        st = {}

        for i in range(len(centroids_l)):
            t = 0
            for s in membership.keys():
                t += membership[s][i]
            st[centroids_l[i]] = new_dict[centroids_l[i]]/tot
        
        op = {}
        for s in ip.keys():
            if s in centroids_l:
                op[s] = st[s]
            else:
                if membership[s][len(centroids_l)] > 0.0:
                    op[s] = membership[s][len(centroids_l)]
        return self._renormalize(op)


def define_complete_vec(ip_vector, nqbs):
    """
    Creates complete vector with all possible binary strings.
    
    Args:
        ip_vector (dict): Input vector
        nqbs (int): Number of qubits
        
    Returns:
        Counter: Complete vector
    """
    getbin = lambda x, n: format(x, 'b').zfill(n)
    out_vec = {}
    for i in range(0, 2**nqbs):
        key = getbin(i, nqbs)
        if (key in ip_vector):
            out_vec[key] = ip_vector[key]
        else:
            out_vec[key] = 0.0
    return collections.Counter(out_vec)

def renormalize(ip_vector):
    """
    Normalizes vector values to sum to 1.
    
    Args:
        ip_vector (dict): Input vector
        
    Returns:
        Counter: Normalized vector
    """
    s = 0
    for key in ip_vector.keys():
        s += ip_vector[key]
    
    for key in ip_vector.keys():
        ip_vector[key] = (ip_vector[key]/s)
    
    return collections.Counter(ip_vector)

def renormalize_inplace(ip_vector):
    """
    Normalizes vector values to sum to 1, creating new vector.
    
    Args:
        ip_vector (dict): Input vector
        
    Returns:
        Counter: Normalized vector
    """
    s = 0
    for key in ip_vector.keys():
        s += ip_vector[key]
    
    o_vec = {}
    for key in ip_vector.keys():
        o_vec[key] = (ip_vector[key]/s)
    
    return collections.Counter(o_vec)


def ideal_simulation(circ, shots = 32768, seed = 42):
    """
    Simulates ideal quantum circuit without noise.
    
    Args:
        circ: Quantum circuit
        shots (int): Number of shots
        seed (int): Random seed
        
    Returns:
        dict: Measurement counts
    """
    aer_sim = AerSimulator()
    
    sampler = Sampler(mode = aer_sim)
    options = sampler.options
    options.default_shots = shots
    result = sampler.run([circ]).result()
    kl = list(result[0].data.keys())
    counts = result[0].data[kl[0]].get_counts()
    return counts


def get_esp_modified_therm(qc, num_qubits, backend = None, backend_prop = None, gate_error = True, read_out = True):
    """
    Calculates modified thermal error rate for quantum circuit.
    
    Args:
        qc: Quantum circuit
        num_qubits (int): Number of qubits
        backend: Quantum backend
        backend_prop: Backend properties
        gate_error (bool): Include gate errors
        read_out (bool): Include readout errors
        
    Returns:
        float: Modified thermal error rate
    """
    if backend != None:
        backend_prop=backend.properties()
    else:
        backend_prop = backend_prop
    cx_reliability={}
    rz_reliability={}
    sx_reliability={}
    x_reliability={}
    readout_reliability={}
    cx_num={}
    cxnn = 0
    cxre = 1
    for ginfo in backend_prop.gates:
        if ginfo.gate=="cx" or ginfo.gate=="cz" or ginfo.gate=="ecr":
            for param in ginfo.parameters:
                if param.name=="gate_error":
                    g_reliab = 1.0 - (param.value)
                    break
            cx_reliability[(ginfo.qubits[0], ginfo.qubits[1])] = g_reliab
            cx_num[(ginfo.qubits[0], ginfo.qubits[1])] = 0
        if ginfo.gate=="rz":
            for param in ginfo.parameters:
                if param.name=="gate_error":
                    g_reliab = 1.0 - (param.value)
                    break
            rz_reliability[(ginfo.qubits[0])] = g_reliab
        if ginfo.gate=="sx":
            for param in ginfo.parameters:
                if param.name=="gate_error":
                    g_reliab = 1.0 - param.value
                    break
            sx_reliability[(ginfo.qubits[0])] = g_reliab
        if ginfo.gate=="x":
            for param in ginfo.parameters:
                if param.name=="gate_error":
                    g_reliab = 1.0 - param.value
                    break
            x_reliability[(ginfo.qubits[0])] = g_reliab
    for i in range(num_qubits):
        readout_reliability[(i)]=1.0-backend_prop.readout_error(i)
    for ginfo in backend_prop.gates:
        if ginfo.gate=="cx" or ginfo.gate=="cz" or ginfo.gate=="ecr":
            for param in ginfo.parameters:
                if param.name=="gate_length":
                    rate = 1/(np.mean([backend_prop.t1(ginfo.qubits[0]),backend_prop.t1(ginfo.qubits[1])]))
                    g_reliab = 1-np.exp(-param.value*(10**(-9))*rate)
                    break
            cx_reliability[(ginfo.qubits[0], ginfo.qubits[1])] -= g_reliab
        if ginfo.gate=="rz":
            for param in ginfo.parameters:
                if param.name=="gate_length":
                    rate = 1/(backend_prop.t1(ginfo.qubits[0]))
                    g_reliab = 1-np.exp(-param.value*(10**(-9))*rate)
                    break
            rz_reliability[(ginfo.qubits[0])] -=  g_reliab
        if ginfo.gate=="sx":
            for param in ginfo.parameters:
                if param.name=="gate_length":
                    rate = 1/(backend_prop.t1(ginfo.qubits[0]))
                    g_reliab = 1-np.exp(-param.value*(10**(-9))*rate)
                    break
            sx_reliability[(ginfo.qubits[0])] -=  g_reliab
        if ginfo.gate=="x":
            for param in ginfo.parameters:
                if param.name=="gate_length":
                    rate = 1/(backend_prop.t1(ginfo.qubits[0]))
                    g_reliab = 1 - np.exp(-param.value*(10**(-9))*rate)
                    break
            x_reliability[(ginfo.qubits[0])] -= g_reliab
    dag = circuit_to_dag(qc)
    esp=1
    for node in dag.op_nodes():
        if gate_error:
            if node.name == "rz":
                key=node.qargs[0]._index
                esp=esp*rz_reliability[key]
            elif node.name == "sx":
                key=node.qargs[0]._index
                esp=esp*sx_reliability[key]
            elif node.name == "x":
                key=node.qargs[0]._index
                esp=esp*x_reliability[key]
            elif node.name == "cx" or node.name == "ecr" or node.name == "cz":
                key=(node.qargs[0]._index, node.qargs[1]._index)
                esp=esp*cx_reliability[key]
        if ((read_out) and (node.name == "measure")):
            key=node.qargs[0]._index
            esp=esp*readout_reliability[key]
    return esp