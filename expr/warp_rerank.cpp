#include "utils/utils.h"
#include "../eval.h"

int main() {
    // hyper-parameters
    int nprobe = 128;
    int n_stage1 = 20000;

    int k = 100;
    int num_d, num_q, d, q_doclen, num_docs;
    rabitqlib::ivf::IVF ivf;
    ivf.load("ivf_rabitq_2097152_5bits_l2.index");
    std::vector<float>               Q          = load_query(q_doclen, num_q, d);
    std::vector<int>                 doclens    = load_doclens();
    std::vector<size_t>              docid_map  = build_docid_map(doclens, num_docs);
    std::vector<std::vector<size_t>> doc_to_emb = build_doc_to_emb_map(docid_map, num_docs);

    ivf.centroid_dists_.resize(num_q * q_doclen);

    int nq = 100;
    std::vector<std::vector<size_t>> results(nq);
    std::vector<Stats> stats(nq);
#pragma omp parallel for schedule(dynamic)
    for (int qid = 0; qid < nq; ++qid) {
        std::vector<float> doc_dists(num_docs * q_doclen, 0.0f);
        std::vector<size_t> to_rerank_docs = gather_docids_with_dists(ivf, num_docs, q_doclen, d, qid * q_doclen, Q.data() + qid * q_doclen * d, nprobe, docid_map, doc_dists, stats[qid]);
        std::vector<size_t> stage1_results;
        rerank_gathered_dists_impute(ivf, qid, doc_dists, num_docs, q_doclen, to_rerank_docs, n_stage1, stage1_results);
        rerank_rabitqex_dists(ivf, num_docs, Q.data() + qid * q_doclen * d, q_doclen, d, doc_to_emb, stage1_results, k, results[qid], stats[qid]);
    }
    auto ground_truth = read_gt_tsv(num_q, 1000);
    compute_recall(ground_truth, results, k);

    aggregate_stats(stats);
}