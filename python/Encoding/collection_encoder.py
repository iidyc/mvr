from colbert.indexing.collection_encoder import CollectionEncoder
from colbert.data import Collection
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
import os
import tqdm
import numpy as np
import torch

# args
checkpoint = 'colbert-ir/colbertv2.0'
collection = "/home/yanqichen_umass_edu/work/ColBERT/data/lotte/pooled/dev/collection.tsv"
embedding_filename = "/home/yanqichen_umass_edu/work/mvr/data/lotte/embeddings"
n_gpu = torch.cuda.device_count()

with Run().context(RunConfig(nranks=n_gpu)):
    offsets = []
    config = ColBERTConfig()
    checkpoint = Checkpoint(checkpoint, config)
    encoder = CollectionEncoder(config, checkpoint)
    collection = Collection(collection)
    batches = collection.enumerate_batches(rank=config.rank)
    for chunk_idx, offset, passages in tqdm.tqdm(batches, disable=config.rank > 0):
        offsets.append(offset)
        # Encode passages into embeddings with the checkpoint model
        embs, doclens = encoder.encode_passages(passages)
        numpy_32 = embs.numpy().astype("float32")
        np.save(os.path.join(embedding_filename, f"encoding{chunk_idx}_float32.npy"), numpy_32)
        np.save(os.path.join(embedding_filename, f"doclens{chunk_idx}.npy"), doclens)
        print(f'save embeddings chunkID {chunk_idx}')
    np.save(os.path.join(embedding_filename, f"offsets.npy"), np.array(offsets, dtype=np.int32)) 
