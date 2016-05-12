#ifndef TIMESTAT_H_INCLUDED
#define TIMESTAT_H_INCLUDED
#include <chrono>
#include <vector>

struct TimeStat
{
    float loss;
    std::chrono::high_resolution_clock::time_point time_start;
    std::chrono::high_resolution_clock::time_point time_end;
    std::chrono::high_resolution_clock::time_point start_time_stat() { return time_start = std::chrono::high_resolution_clock::now(); }
    std::chrono::high_resolution_clock::time_point end_time_stat() { return time_end = std::chrono::high_resolution_clock::now(); }
    TimeStat() :loss(0.f) {};
    float get_sum_E() { return loss; }
    long long get_time_cost_in_seconds()
    {
        std::chrono::seconds du = std::chrono::duration_cast<std::chrono::seconds>(time_end - time_start);
        return du.count();
    }
};

#endif