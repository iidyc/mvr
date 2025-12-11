import faiss
from faiss import IndexIVFFlat, IndexFlatL2
import numpy as np

f = open("../build/embeddings.bin", "rb")
num_embeddings = np.fromfile(f, dtype=np.int32, count=1)[0]
d = np.fromfile(f, dtype=np.int32, count=1)[0]
d = int(d)
print(f"num_embeddings: {num_embeddings}, dimension: {d}")
# num_embeddings = 1000000
embeddings = np.fromfile(f, dtype=np.float32, count=num_embeddings * d)
embeddings = embeddings.reshape(num_embeddings, d)
f.close()
print(f"embeddings shape: {embeddings.shape}")

res = faiss.StandardGpuResources()

centroids = faiss.GpuIndexFlatIP(res, d)
cp = faiss.ClusteringParameters()
cp.verbose = True
cp.niter = 20
kmeans = faiss.Clustering(d, 131072, cp)
kmeans.train(embeddings, centroids)

ivf = faiss.IndexIVFFlat(centroids, d, 131072, faiss.METRIC_INNER_PRODUCT)
ivf.make_direct_map(True)
ivf.verbose = True
ivf.add(embeddings)

ivf.quantizer = faiss.index_gpu_to_cpu(ivf.quantizer)
faiss.write_index(ivf, "../build/ivf_131072.faiss")