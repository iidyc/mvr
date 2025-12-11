#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/utils/utils.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>

void load_data(std::vector<float>& embeddings, std::vector<int>& doclens, int& num_embeddings, int& d, std::vector<int>& docid_map) {
    std::ifstream doc_lens_file("doc_lens.bin", std::ios::binary);
    int doclens_size;
    doc_lens_file.read(reinterpret_cast<char*>(&doclens_size), sizeof(int));
    doclens.resize(doclens_size);
    doc_lens_file.read(reinterpret_cast<char*>(doclens.data()), doclens.size() * sizeof(int));
    doc_lens_file.close();
    std::ifstream emb_file("embeddings.bin", std::ios::binary);
    emb_file.read(reinterpret_cast<char*>(&num_embeddings), sizeof(int));
    emb_file.read(reinterpret_cast<char*>(&d), sizeof(int));
    // num_embeddings = 1000000;
    embeddings.resize(size_t(num_embeddings) * d);
    emb_file.read(reinterpret_cast<char*>(embeddings.data()), embeddings.size() * sizeof(float));
    emb_file.close();
    docid_map.resize(num_embeddings);
    // for (int i = 0, offset = 0; i < doclens_size; ++i) {
    //     for (int j = 0; j < doclens[i]; ++j) {
    //         docid_map[offset + j] = i;
    //     }
    //     offset += doclens[i];
    // }
}

int main() {
    std::vector<float> embeddings;
    std::vector<int> doclens;
    int num_embeddings;
    int d;
    std::vector<int> docid_map;
    load_data(embeddings, doclens, num_embeddings, d, docid_map);
    faiss::IndexIVFFlat* ivf_index = (faiss::IndexIVFFlat*)faiss::read_index("ivf_hnsw_2097152.faiss");
    faiss::IndexIVFRaBitQ rabitq_index(ivf_index->quantizer, d, 2097152, faiss::METRIC_INNER_PRODUCT);
    std::vector<faiss::idx_t> list_nos;
    for (faiss::idx_t i = 0; i < ivf_index->ntotal; ++i) {
        faiss::idx_t lo = ivf_index->direct_map.get(i);
        list_nos.push_back(faiss::lo_listno(lo));
    }
    rabitq_index.make_direct_map(true);
    rabitq_index.is_trained = true;
    rabitq_index.add_core(num_embeddings, embeddings.data(), nullptr, list_nos.data());
    faiss::write_index(&rabitq_index, "ivf_rabitq_2097152.faiss");
}