from sklearn.utils import murmurhash3_32
import csv
import pandas as pd
import random
import time

def genNGrams(string, n):
    #case where n is greater than length of string
    if n > len(string):
        return []

    substrings = [string[i:i + n] for i in range(0, len(string) - n + 1)]
    
    return set(substrings)

def hashfunc(seed):
    def hash(key):
        return murmurhash3_32(key, seed)

    return hash

def uniHashFunc(seed, m):
    def hash(key):
        return murmurhash3_32(key, seed) % m

    return hash

def hash_func_to_bucket(k, R):
    coefficients = [random.randint(0, R - 1) for i in range(k)]
    constant = random.randint(0, R - 1)

    def hash_function(values):
        hash_value = sum(coefficient * value for coefficient, value in zip(coefficients, values))
        hash_value += constant
        return hash_value % R 

    return hash_function

def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2))



def MinHashFactory(seed):
    def MinHash (A , k ):
        out = []
        gram_size = 3
        hashes = []
        grams = genNGrams(A, gram_size)
        for i in range(k):
            hashes.append(hashfunc(seed=i))
        for hash in hashes:
            min_val = float('inf')
            for substring in grams:
                if hash(substring) < min_val:
                    min_val = hash(substring)
            out.append(min_val)
        return out

    return MinHash

s1 = "The mission statement of the WCSCC and area employers recognize the importance of good attendance on the job. Any student whose absences exceed 18 days is jeopardizing their opportunity for advanced placement as well as hindering his/her likelihood for successfully completing their program."
s2 = "The WCSCC’s mission statement and surrounding employers recognize the importance of great attendance. Any student who is absent more than 18 days will loose the opportunity for successfully completing their trade program."

# m = 1000

# s1_min_hash = MinHash(s1, m)
# s2_min_hash = MinHash(s2, m)

# collision_count = 0
# for i in range(m):
#     if s1_min_hash[i] == s2_min_hash[i]:
#         collision_count += 1

# estimated_similarity = collision_count / m

# s1_gram = genNGrams(s1, 3)
# s2_gram = genNGrams(s2, 3)

# actual_similarity = jaccard_similarity(s1_gram, s2_gram)

# print(f"Estimated Jaccard Similarity: {estimated_similarity}")
# print(f"Actual Jaccard Similarity: {actual_similarity}")

class HashTable ():

    def __init__ ( self , K , L , B , R ):
        self.K = K
        self.L = L
        self.B = B
        self.R = R
        self.hashes = []
        for i in range(self.L):
            self.hashes.append(MinHashFactory(i * 1000))
        self.bucket_mapper = hash_func_to_bucket(self.K, self.R)
        self.Table = []
        for i in range(self.R):
            col = []
            self.Table.append(col)
    
    #hashcodes a list of hashcodes of length R
    def insert ( self , hashcodes , id ):
        for i in range(len(hashcodes)):
            loc = self.bucket_mapper(hashcodes[i])

            if len(self.Table[loc]) < self.L:
                self.Table[loc].append(id)
    
    def lookup ( self , hashcodes ):
        out = []
        for i in range(len(hashcodes)):
            loc = self.bucket_mapper(hashcodes[i])
            out.extend(self.Table[loc])
        return set(out)

def insert(url, hashTable):
    hashcodes = []
    for i in range(hashTable.L):
        hashcodes.append(hashTable.hashes[i](url, hashTable.K))

    hashTable.insert(hashcodes, url)

def query(url, hashTable):
    hashcodes = []
    for i in range(hashTable.L):
        hashcodes.append(hashTable.hashes[i](url, hashTable.K))
    
    return hashTable.lookup(hashcodes)

data = pd.read_csv("aol.txt", sep="\t")
urllist = data.ClickURL.dropna().unique().tolist()

insertion_start = time.time()
hashTable = HashTable(2, 50, 64, 2**20)

for i in range(len(urllist)):
    insert(urllist[i], hashTable)

insertion_end = time.time()

print("Time taken to insert: " + str(insertion_start - insertion_end))

def taskOne(urllist, hashTable):
    begin = time.time()
    query_list = random.sample(urllist, 200)
    averages = {}
    for url in query_list:
        total = 0
        retrieved_urls = query(url, hashTable)

        for rurl in retrieved_urls:
            similarity = jaccard_similarity(genNGrams(url, 3), genNGrams(rurl, 3))
            total += similarity

        averages[url] = total/len(retrieved_urls)
    end = time.time()
    print("time taken on task one: " + str(end - begin))

    overall_unfiltered_avg = sum(averages.values())/len(averages.values())
    print("overall unfiltered average: " + str(overall_unfiltered_avg))
    print(averages)
    

taskOne(urllist, hashTable)

def taskTwo(urllist, hashTable):
    begin = time.time()
    query_list = random.sample(urllist, 200)
    pairwise_sim = {}
    for i in range(len(urllist)):
        temp_begin = time.time()
        url = urllist[i]
        temp_sim = {}
        for j in range(len(urllist)):
            if j != i:
                rurl = urllist[j]
                temp_sim[urllist[j]] = jaccard_similarity(genNGrams(url, 3), genNGrams(rurl, 3))
        pairwise_sim[url] = temp_sim
        temp_end = time.time()
        print("Query time: " + str(temp_begin - temp_end))
    end = time.time()
    print("total time: " + str(end - begin))
