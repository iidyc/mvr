import numpy as np
import faiss

f = open("build/embeddings.bin", "rb")
num_embeddings = np.fromfile(f, dtype=np.int32, count=1)[0]
d = np.fromfile(f, dtype=np.int32, count=1)[0]
d = int(d)
print(f"num_embeddings: {num_embeddings}, dimension: {d}")
num_embeddings = 1000000
embeddings = np.fromfile(f, dtype=np.float32, count=num_embeddings * d)
embeddings = embeddings.reshape(num_embeddings, d)
f.close()

centroids = faiss.IndexFlat(d)
index = faiss.IndexIVFFlat(centroids, d, 1024)
index.verbose = True
index.train(embeddings)
index.add(embeddings)