#include <fstream>
#include <iostream>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/utils/distances.h>
#include <queue>
#include <omp.h>
#include "eval.h"

#include "RaBitQ-Library/include/rabitqlib/defines.hpp"
#include "RaBitQ-Library/include/rabitqlib/index/ivf/ivf.hpp"

void load_data(std::vector<float>& embeddings, int& num_embeddings, int& d) {
    std::ifstream emb_file("embeddings.bin", std::ios::binary);
    emb_file.read(reinterpret_cast<char*>(&num_embeddings), sizeof(int));
    emb_file.read(reinterpret_cast<char*>(&d), sizeof(int));
    // num_embeddings = 1000000;
    embeddings.resize(size_t(num_embeddings) * d);
    emb_file.read(reinterpret_cast<char*>(embeddings.data()), embeddings.size() * sizeof(float));
    emb_file.close();
    std::cout << "Loaded " << num_embeddings << " embeddings of dimension " << d << std::endl;
}

void load_query(std::vector<float>& Q, std::vector<int>& doclens, int& q_doclen, int& num_q, int& d) {
    std::ifstream doc_lens_file("doclens.bin", std::ios::binary);
    int doclens_size;
    doc_lens_file.read(reinterpret_cast<char*>(&doclens_size), sizeof(int));
    doclens.resize(doclens_size);
    doc_lens_file.read(reinterpret_cast<char*>(doclens.data()), doclens.size() * sizeof(int));
    doc_lens_file.close();
    std::cout << "Loaded " << doclens_size << " document lengths" << std::endl;
    std::ifstream qemb_file("query_embeddings.bin", std::ios::binary);
    qemb_file.read(reinterpret_cast<char*>(&num_q), sizeof(int));
    qemb_file.read(reinterpret_cast<char*>(&q_doclen), sizeof(int));
    qemb_file.read(reinterpret_cast<char*>(&d), sizeof(int));
    Q.resize(size_t(num_q) * q_doclen * d);
    qemb_file.read(reinterpret_cast<char*>(Q.data()), Q.size() * sizeof(float));
    qemb_file.close();
    std::cout << "Loaded " << num_q << " queries, each with " << q_doclen << " embeddings of dimension " << d << std::endl;
}

int main() {
    rabitqlib::ivf::IVF ivf;
    ivf.load("ivf_rabitq_2097152_5bits.index");
    std::vector<float> Q;
    std::vector<int> doclens;
    int q_doclen;
    int num_q;
    int d;
    load_query(Q, doclens, q_doclen, num_q, d);
    std::vector<float> embeddings;
    int num_embeddings;
    // load_data(embeddings, num_embeddings, d);
    // embedding id to document id mapping
    std::vector<int> docid_map;
    size_t cur_id = 0;
    for (int doclen : doclens) {
        for (int j = 0; j < doclen; ++j) {
            docid_map.push_back(cur_id);
        }
        cur_id += 1;
    }
    // document id to embedding id mapping
    std::vector<std::vector<int>> doc_to_emb(cur_id);
    for (size_t emb_id = 0; emb_id < docid_map.size(); ++emb_id) {
        int doc_id = docid_map[emb_id];
        doc_to_emb[doc_id].push_back(emb_id);
    }
    std::cout << "Total documents: " << cur_id << ", total embeddings: " << docid_map.size() << std::endl;
    std::vector<float> doc_dists(cur_id * q_doclen);
    std::vector<bool> doc_found(cur_id, false);
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < q_doclen; ++j) {
            float* qj = Q.data() + (i * q_doclen + j) * d;
            int nprobe = 8;
            std::vector<float> token_dists;
            std::vector<rabitqlib::PID> ids;
            ivf.gather_dists(qj, nprobe, token_dists, ids);
            for (size_t idx = 0; idx < ids.size(); ++idx) {
                int emb_id = ids[idx];
                float dist = token_dists[idx];
                int doc_id = docid_map[emb_id];
                doc_found[doc_id] = true;
                doc_dists[j * cur_id + doc_id] = std::max(doc_dists[j * cur_id + doc_id], dist);
            }
        }
    }
    std::vector<int> to_rerank_docs;
    for (int doc_id = 0; doc_id < cur_id; ++doc_id) {
        if (doc_found[doc_id]) {
            to_rerank_docs.push_back(doc_id);
        }
    }
    std::vector<float> doc_scores(cur_id, 0.0f);
    std::cout << "start reranking " << to_rerank_docs.size() << " documents." << std::endl;
    for (int j = 0; j < q_doclen; ++j) {
        float* qj = Q.data() + (0 * q_doclen + j) * d;
        for (size_t doc_id : to_rerank_docs) {
            doc_scores[doc_id] += doc_dists[j * cur_id + doc_id];
            // float max_token_score = -1e10;
            // for (size_t emb_id : doc_to_emb[doc_id]) {
            //     assert(emb_id < num_embeddings);
            //     float* emb_vec = embeddings.data() + emb_id * d;
            //     float dist = faiss::fvec_inner_product(qj, emb_vec, d);
            //     max_token_score = std::max(max_token_score, dist);
            // }
            // doc_scores[doc_id] += max_token_score;
        }
    }
    // get top-k documents
    int topk = 100;
    using DocScorePair = std::pair<float, int>;
    auto cmp = [](const DocScorePair& a, const DocScorePair& b) {
        return a.first < b.first; // max-heap
    };
    std::priority_queue<DocScorePair, std::vector<DocScorePair>, decltype(cmp)> max_heap(cmp);
    for (int doc_id : to_rerank_docs) {
        float score = doc_scores[doc_id];
        max_heap.emplace(score, doc_id);
    }
    std::vector<int> topk_results;
    for (int i = 0; i < topk && !max_heap.empty(); ++i) {
        topk_results.push_back(max_heap.top().second);
        max_heap.pop();
    }
    std::vector<std::vector<int>> retrieved(1);
    retrieved[0] = topk_results;
    auto ground_truth = read_gt_tsv(num_q, 1000);
    compute_recall(ground_truth, retrieved, topk);
    return 0;
}