#pragma once

#include <array>
#include <vector>
#include <numeric>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

using uint = unsigned int;
using floating_t = double;
using prob_t = floating_t;
template<typename T, size_t sz> using array_t = std::array<T, sz>;

using pmf_t = std::vector<prob_t>;
using ProbVector = std::vector<prob_t>;

constexpr auto maxEta = 15u;

using StateSizedArray = array_t<floating_t, 1 << maxEta >;
using HalfStateSizedArray = array_t<floating_t, 1 << (maxEta-1) >;

template<class T> using xArray = array_t<T ,2>;
template<class T> using yArray = array_t<T, 2>;
