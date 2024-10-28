#pragma once
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <limits>
#include <cassert>
#include <algorithm>

#include "TypeDefinitions.h"

#ifdef __CUDACC__
#include <curand.h>
#include <curand_kernel.h>
#endif 

using std::clamp;

namespace Random
{
    static auto& generator()
    {
        static std::hash<std::thread::id> hasher;
        static thread_local std::mt19937 generator(clock() + hasher(std::this_thread::get_id()));
        return generator;
    }
}

template<class T = double>
T unirnd(T min, T max)
{
    static std::hash<std::thread::id> hasher;
    static thread_local std::mt19937 generator(clock() + hasher(std::this_thread::get_id()));
    std::uniform_real_distribution<T> distribution(min, max);
    return distribution(generator);
}

template<class T = double>
T uniintrnd(T min, T max)
{
    static std::hash<std::thread::id> hasher;
    static thread_local std::mt19937 generator(clock() + hasher(std::this_thread::get_id()));
    std::uniform_int_distribution<T> distribution(min, max);
    return distribution(generator); 
}

template <typename T = double>
CUDA_CALLABLE_MEMBER constexpr T eps_()
{
    return std::numeric_limits<T>::epsilon();
}

constexpr auto eps = eps_<prob_t>();

template<typename T = double>
CUDA_CALLABLE_MEMBER inline auto logeps()
{
#ifdef __CUDACC__
    return log(eps_<T>());
#else
    static auto L = log(eps_<T>());
    return L;
#endif
}

template<typename T>
CUDA_CALLABLE_MEMBER inline T safelog(const T& v)
{
    return v < eps_<T>() ? logeps<T>() : log(v);
}

template<typename T = double>
CUDA_CALLABLE_MEMBER inline auto log2eps()
{
#ifdef __CUDACC__
    return log2(eps_<T>());
#else
    static auto L = log2(eps_<T>());
    return L;
#endif
}

template<typename T>
CUDA_CALLABLE_MEMBER inline T safelog2(const T& v)
{
    return v < eps_<T>() ? log2eps<T>() : log2(v);
}

template<typename T>
CUDA_CALLABLE_MEMBER inline std::tuple<T,T> mean_variance(const T* p, size_t size)
{
    T variance = 0;
    T t = p[0];
    for (size_t i = 1; i < size; ++i)
    {
        t += p[i];
        const T diff = ((i + 1) * p[i]) - t;
        variance += (diff * diff) / ((i + 1.0) *i);
    }

    return { t / size, variance / (size - 1) };
}

template<typename T>
CUDA_CALLABLE_MEMBER inline double variance(const T* p, size_t size, double mean)
{
    auto variance = .0;
    for (size_t i = 0; i < size; ++i)
    {
        const auto diff = (p[i] - mean);
        variance += diff * diff;
    }
    return variance / (size - 1);
}

template<typename T>
CUDA_CALLABLE_MEMBER inline double stddev(const T* p, size_t size)
{
    T mean, variance;
    std::tie(mean, variance) = mean_variance(p, size);
    return sqrt(variance);
}

class Timer
{
protected:
    std::chrono::time_point<std::chrono::system_clock> m_startTime = std::chrono::system_clock::now();
    std::vector<std::pair<std::string, std::chrono::duration<double>>> m_processingTimes;

public:

    inline auto duration() const -> std::chrono::duration<double> { return std::chrono::system_clock::now() - m_startTime; }
    inline auto reset() -> std::chrono::duration<double>
    {
        const auto d = duration();
        m_startTime = std::chrono::system_clock::now();
        return d;
    }

    Timer() { reset(); }

    inline auto record(std::string name, bool bReset = false) -> std::chrono::duration<double>
    {
        auto d = duration();
        m_processingTimes.emplace_back(name, d);
        if (bReset)
            m_startTime = std::chrono::system_clock::now();
        return d;
    }

    inline auto lastRecord() const -> decltype(m_processingTimes)::value_type{ return m_processingTimes.back(); }
    inline auto lastRecordName() const -> decltype(m_processingTimes)::value_type::first_type{ return lastRecord().first; }
    inline auto lastRecordDuration() const -> decltype(m_processingTimes)::value_type::second_type{ return lastRecord().second; }
};
