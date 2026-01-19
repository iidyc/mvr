#include <iostream>
#include <vector>
#include <fstream>
#include <queue>

#include "../../RaBitQ-Library/include/rabitqlib/defines.hpp"
#include "../../RaBitQ-Library/include/rabitqlib/index/ivf/ivf.hpp"

std::vector<float> load_data(int& num_embeddings, int& d) {
    std::ifstream emb_file("embeddings.bin", std::ios::binary);
    emb_file.read(reinterpret_cast<char*>(&num_embeddings), sizeof(int));
    emb_file.read(reinterpret_cast<char*>(&d), sizeof(int));
    std::vector<float> embeddings(size_t(num_embeddings) * d);
    emb_file.read(reinterpret_cast<char*>(embeddings.data()), embeddings.size() * sizeof(float));
    emb_file.close();
    std::cout << ">>> Loaded " << num_embeddings << " embeddings of dimension " << d << std::endl;
    return std::move(embeddings);
}

std::vector<float> load_query(int& q_doclen, int& num_q, int& d) {
    std::ifstream qemb_file("query_embeddings.bin", std::ios::binary);
    qemb_file.read(reinterpret_cast<char*>(&num_q), sizeof(int));
    qemb_file.read(reinterpret_cast<char*>(&q_doclen), sizeof(int));
    qemb_file.read(reinterpret_cast<char*>(&d), sizeof(int));
    std::vector<float> Q(size_t(num_q) * q_doclen * d);
    qemb_file.read(reinterpret_cast<char*>(Q.data()), Q.size() * sizeof(float));
    qemb_file.close();
    std::cout << ">>> Loaded " << num_q << " queries, each with " << q_doclen << " embeddings of dimension " << d << std::endl;
    return std::move(Q);
}

std::vector<int> load_doclens() {
    std::ifstream doc_lens_file("doclens.bin", std::ios::binary);
    int doclens_size;
    doc_lens_file.read(reinterpret_cast<char*>(&doclens_size), sizeof(int));
    std::vector<int> doclens(doclens_size);
    doc_lens_file.read(reinterpret_cast<char*>(doclens.data()), doclens.size() * sizeof(int));
    doc_lens_file.close();
    std::cout << ">>> Loaded " << doclens_size << " document lengths" << std::endl;
    return std::move(doclens);
}

std::vector<size_t> build_docid_map(const std::vector<int>& doclens, int& num_docs) {
    std::vector<size_t> docid_map;
    num_docs = 0;
    for (int doclen : doclens) {
        for (int j = 0; j < doclen; ++j) {
            docid_map.push_back(num_docs);
        }
        num_docs += 1;
    }
    std::cout << ">>> Total documents: " << num_docs << ", total embeddings: " << docid_map.size() << std::endl;
    return std::move(docid_map);
}

std::vector<std::vector<size_t>> build_doc_to_emb_map(const std::vector<size_t>& docid_map, int num_docs) {
    std::vector<std::vector<size_t>> doc_to_emb(num_docs);
    for (size_t emb_id = 0; emb_id < docid_map.size(); ++emb_id) {
        size_t doc_id = docid_map[emb_id];
        doc_to_emb[doc_id].push_back(emb_id);
    }
    return std::move(doc_to_emb);
}

std::vector<size_t> gather_docids(
    rabitqlib::ivf::IVF& ivf, 
    int num_docs, 
    int q_doclen, 
    int d, 
    int qid,
    float* queries, 
    int nprobe, 
    std::vector<size_t>& docid_map
) {
    std::vector<bool> doc_found(num_docs, false);
    int dc_count = 0;
    for (int j = 0; j < q_doclen; ++j) {
        float* qj = queries + j * d;
        std::vector<rabitqlib::PID> ids;
        ivf.gather_ids(qj, nprobe, ids, qid + j);
        for (size_t idx = 0; idx < ids.size(); ++idx) {
            size_t emb_id = ids[idx];
            int doc_id = docid_map[emb_id];
            doc_found[doc_id] = true;
        }
    }
    std::vector<size_t> to_rerank_docs;
    for (int doc_id = 0; doc_id < num_docs; ++doc_id) {
        if (doc_found[doc_id]) {
            to_rerank_docs.push_back(doc_id);
        }
    }
    return to_rerank_docs;
}

std::vector<size_t> gather_docids_with_dists(
    rabitqlib::ivf::IVF& ivf, 
    int num_docs, 
    int q_doclen, 
    int d, 
    int qid,
    float* queries, 
    int nprobe, 
    std::vector<size_t>& docid_map,
    std::vector<float>& doc_dists_out
) {
    std::vector<bool> doc_found(num_docs, false);
    for (int j = 0; j < q_doclen; ++j) {
        float* qj = queries + j * d;
        std::vector<float> token_dists;
        std::vector<rabitqlib::PID> ids;
        ivf.gather_dists(qj, nprobe, token_dists, ids, qid + j);
        for (size_t idx = 0; idx < ids.size(); ++idx) {
            size_t emb_id = ids[idx];
            float dist = (2 - token_dists[idx]) / 2;
            int doc_id = docid_map[emb_id];
            doc_found[doc_id] = true;
            doc_dists_out[j * num_docs + doc_id] = std::max(doc_dists_out[j * num_docs + doc_id], dist);
        }
    }
    std::vector<size_t> to_rerank_docs;
    for (int doc_id = 0; doc_id < num_docs; ++doc_id) {
        if (doc_found[doc_id]) {
            to_rerank_docs.push_back(doc_id);
        }
    }
    return to_rerank_docs;
}

void rerank_rabitqex_dists(
    rabitqlib::ivf::IVF& ivf,
    int num_docs, 
    float* queries, 
    int q_doclen, 
    int d, 
    const std::vector<std::vector<size_t>>& doc_to_emb,
    const std::vector<size_t>& candidates, 
    int k, 
    std::vector<size_t>& id_out
) {
    auto dc = ivf.get_distance_computer();
    std::vector<float> doc_scores(num_docs, 0.0f);
    for (int j = 0; j < q_doclen; ++j) {
        float* qj = queries + j * d;
        dc->set_query(qj);
        for (size_t doc_id : candidates) {
            float max_token_score = -1e10;
            for (size_t emb_id : doc_to_emb[doc_id]) {
                float dist = (2 - (*dc)(emb_id)) / 2;
                max_token_score = std::max(max_token_score, dist);
            }
            doc_scores[doc_id] += max_token_score;
        }
    }
    using DocScorePair = std::pair<float, size_t>;
    auto cmp = [](const DocScorePair& a, const DocScorePair& b) {
        return a.first < b.first; // max-heap
    };
    std::priority_queue<DocScorePair, std::vector<DocScorePair>, decltype(cmp)> max_heap(cmp);
    for (size_t doc_id : candidates) {
        float score = doc_scores[doc_id];
        max_heap.emplace(score, doc_id);
    }
    for (int i = 0; i < k && !max_heap.empty(); ++i) {
        id_out.push_back(max_heap.top().second);
        max_heap.pop();
    }
}

void rerank_gathered_dists(
    rabitqlib::ivf::IVF& ivf,
    std::vector<float>& doc_dists,
    int num_docs, 
    int q_doclen, 
    const std::vector<size_t>& candidates, 
    int k, 
    std::vector<size_t>& id_out
) {
    std::vector<float> doc_scores(num_docs, 0.0f);
    for (int j = 0; j < q_doclen; ++j) {
        for (size_t doc_id : candidates) {
            doc_scores[doc_id] += doc_dists[j * num_docs + doc_id];
        }
    }
    using DocScorePair = std::pair<float, size_t>;
    auto cmp = [](const DocScorePair& a, const DocScorePair& b) {
        return a.first < b.first; // max-heap
    };
    std::priority_queue<DocScorePair, std::vector<DocScorePair>, decltype(cmp)> max_heap(cmp);
    for (size_t doc_id : candidates) {
        float score = doc_scores[doc_id];
        max_heap.emplace(score, doc_id);
    }
    for (int i = 0; i < k && !max_heap.empty(); ++i) {
        id_out.push_back(max_heap.top().second);
        max_heap.pop();
    }
}