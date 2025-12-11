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

void load_data(std::vector<float>& Q, std::vector<int>& doclens, int& q_doclen, int& num_q, int& d) {
    std::ifstream doc_lens_file("doclens.bin", std::ios::binary);
    int doclens_size;
    doc_lens_file.read(reinterpret_cast<char*>(&doclens_size), sizeof(int));
    doclens.resize(doclens_size);
    doc_lens_file.read(reinterpret_cast<char*>(doclens.data()), doclens.size() * sizeof(int));
    doc_lens_file.close();
    std::cout << "Loaded " << doclens_size << " document lengths" << std::endl;
    std::ifstream emb_file("query_embeddings.bin", std::ios::binary);
    emb_file.read(reinterpret_cast<char*>(&num_q), sizeof(int));
    emb_file.read(reinterpret_cast<char*>(&q_doclen), sizeof(int));
    emb_file.read(reinterpret_cast<char*>(&d), sizeof(int));
    Q.resize(size_t(num_q) * q_doclen * d);
    emb_file.read(reinterpret_cast<char*>(Q.data()), Q.size() * sizeof(float));
    emb_file.close();
    std::cout << "Loaded " << num_q << " queries, each with " << q_doclen << " embeddings of dimension " << d << std::endl;
}

int main() {
    faiss::IndexIVFRaBitQ* rabitq_index = (faiss::IndexIVFRaBitQ*)faiss::read_index("ivf_rabitq_2097152.faiss");
    // faiss::IndexIVFFlat* rabitq_index = (faiss::IndexIVFFlat*)faiss::read_index("ivf_hnsw_2097152.faiss");
    ((faiss::IndexHNSW*)rabitq_index->quantizer)->hnsw.efSearch = 20;
    // faiss::IndexIVFRaBitQ* rabitq_index = (faiss::IndexIVFRaBitQ*)faiss::read_index("ivf_rabitq_131072.faiss");
    // faiss::IndexIVFFlat* rabitq_index = (faiss::IndexIVFFlat*)faiss::read_index("ivf_131072.faiss");
    std::cout << rabitq_index->ntotal << " embeddings loaded in RaBitQ index." << std::endl;
    std::vector<float> Q;
    std::vector<int> doclens;
    int q_doclen;
    int num_q;
    int d;
    load_data(Q, doclens, q_doclen, num_q, d);
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
    // collect to rerank document ids
    std::vector<bool> doc_found(cur_id, false);
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < q_doclen; ++j) {
            float* qj = Q.data() + (i * q_doclen + j) * d;
            int nprobe = 8;
            std::vector<faiss::idx_t> assign(nprobe);
            rabitq_index->quantizer->assign(1, qj, assign.data(), nprobe);
            for (int p = 0; p < nprobe; ++p) {
                faiss::idx_t list_no = assign[p];
                faiss::InvertedLists::ScopedIds ids(rabitq_index->invlists, list_no);
                size_t list_size = rabitq_index->invlists->list_size(list_no);
                for (size_t k = 0; k < list_size; ++k) {
                    faiss::idx_t id = ids[k];
                    int doc_id = docid_map[id];
                    doc_found[doc_id] = true;
                }
            }
        }
    }
    int total_found = 0;
    for (bool found : doc_found) {
        if (found) {
            total_found += 1;
        }
    }
    // rabitq_index->qb = 4;
    auto dc = rabitq_index->get_distance_computer();
    std::cout << "Total found documents: " << total_found << std::endl;
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
        dc->set_query(qj);
#pragma omp parallel for
        for (int doc_id : to_rerank_docs) {
            float max_token_score = -1e10;
            for (int emb_id : doc_to_emb[doc_id]) {
                max_token_score = std::max(max_token_score, (*dc)(emb_id));
                // std::vector<float> emb_vec(d);
                // rabitq_index->reconstruct(emb_id, emb_vec.data());
                // float dist = faiss::fvec_inner_product(qj, emb_vec.data(), d);
                // max_token_score = std::max(max_token_score, dist);
            }
            doc_scores[doc_id] += max_token_score;
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
    // delete dc;
    delete rabitq_index;
    return 0;
}