#include "utils/utils.h"
#include "../eval.h"

int main() {
    // hyper-parameters
    int nprobe = 32;

    int k = 100;
    int num_d, num_q, d, q_doclen, num_docs;
    rabitqlib::ivf::IVF ivf;
    ivf.load("ivf_rabitq_2097152_5bits_l2.index");
    std::vector<float>               Q          = load_query(q_doclen, num_q, d);
    std::vector<int>                 doclens    = load_doclens();
    std::vector<size_t>              docid_map  = build_docid_map(doclens, num_docs);
    std::vector<std::vector<size_t>> doc_to_emb = build_doc_to_emb_map(docid_map, num_docs);

    ivf.centroid_dists_.resize(num_q * q_doclen);

    size_t dist_comps = 0;
    int nq = 100;
    std::vector<std::vector<size_t>> results(nq);
#pragma omp parallel for reduction(+:dist_comps)
    for (int qid = 0; qid < nq; ++qid) {
        std::vector<size_t> to_rerank_docs = gather_docids(ivf, num_docs, q_doclen, d, qid * q_doclen, Q.data() + qid * q_doclen * d, nprobe, docid_map);
        rerank_rabitqex_dists(ivf, num_docs, Q.data() + qid * q_doclen * d, q_doclen, d, doc_to_emb, to_rerank_docs, k, results[qid]);
        dist_comps += to_rerank_docs.size();
    }
    std::cout << ">>> Avg distance computations: " << dist_comps / nq << std::endl;
    auto ground_truth = read_gt_tsv(num_q, 1000);
    compute_recall(ground_truth, results, k);
}