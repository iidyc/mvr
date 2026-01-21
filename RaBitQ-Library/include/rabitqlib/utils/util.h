#pragma once

#include <chrono>
#include <iostream>

struct Timer {
    std::chrono::_V2::system_clock::time_point s;
    std::chrono::_V2::system_clock::time_point e;
    std::chrono::duration<double> diff;

    void tick() {
        s = std::chrono::high_resolution_clock::now();
    }

    void tuck(std::string message, bool print = true) {
        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        if (print) {
            std::cout << "[" << diff.count() << " s] " << message << std::endl;
        }
    }
};

struct Stats {
    size_t gather_dist_comps = 0;
    size_t rerank_dist_comps = 0;
    double total_gather_time = 0.0;
    double gather_hnsw_time = 0.0;
    double gather_dist_time = 0.0;
    double gather_matrix_time = 0.0;
    double rerank_stage1_time = 0.0;
    double rerank_stage2_time = 0.0;
};

void aggregate_stats(const std::vector<Stats>& stats) {
    Stats total_stat;
    for (const auto& stat : stats) {
        total_stat.gather_dist_comps += stat.gather_dist_comps;
        total_stat.rerank_dist_comps += stat.rerank_dist_comps;
        total_stat.total_gather_time += stat.total_gather_time;
        total_stat.gather_hnsw_time += stat.gather_hnsw_time;
        total_stat.gather_dist_time += stat.gather_dist_time;
        total_stat.gather_matrix_time += stat.gather_matrix_time;
        total_stat.rerank_stage1_time += stat.rerank_stage1_time;
        total_stat.rerank_stage2_time += stat.rerank_stage2_time;
    }
    total_stat.gather_hnsw_time /= stats.size();
    total_stat.gather_dist_time /= stats.size();
    total_stat.total_gather_time /= stats.size();
    total_stat.gather_matrix_time /= stats.size();
    total_stat.rerank_stage1_time /= stats.size();
    total_stat.rerank_stage2_time /= stats.size();
    total_stat.gather_dist_comps /= stats.size();
    total_stat.rerank_dist_comps /= stats.size();

    std::cout << ">>> Average gather HNSW time: " << total_stat.gather_hnsw_time << " s" << std::endl;
    std::cout << ">>> Average gather distance time: " << total_stat.gather_dist_time << " s" << std::endl;
    std::cout << ">>> Average total gather time: " << total_stat.total_gather_time << " s" << std::endl;
    std::cout << ">>> Average gather matrix time: " << total_stat.gather_matrix_time << " s" << std::endl;
    std::cout << ">>> Average gather distance computations: " << total_stat.gather_dist_comps << std::endl;
    std::cout << ">>> Average rerank stage 1 time: " << total_stat.rerank_stage1_time << " s" << std::endl;
    std::cout << ">>> Average rerank stage 2 time: " << total_stat.rerank_stage2_time << " s" << std::endl;
    std::cout << ">>> Average rerank distance computations: " << total_stat.rerank_dist_comps << std::endl;
}