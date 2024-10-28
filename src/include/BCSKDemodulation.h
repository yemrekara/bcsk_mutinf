#pragma once
#include <bitset>
#include <vector>
#include <algorithm>
#include <numeric>
#include "Utilities.h"

template<typename ProbArray>
inline auto convolveDiscreteDistributions(const ProbArray& pmf1, const ProbArray& pmf2)
{
    if (pmf1.empty())
        return pmf2;

    if (pmf2.empty())
        return pmf1;

    auto pmfSum = ProbArray(pmf1.size() + pmf2.size() - 1, .0f);
    for (auto i = 0u; i < pmf1.size(); ++i)
        for (auto j = 0u; j < pmf2.size(); ++j)
            pmfSum[i + j] += pmf1[i] * pmf2[j];
    return pmfSum;
}

template<typename ProbArray>
inline void calculateStatePMFVectorsRecursively(const std::vector<ProbArray>& pmfInput, const ProbArray& pmfCurrent, std::bitset<32> state, size_t b, std::vector<ProbArray>& pmfOutput, uint32_t maxPMFSize = 0)
{
    if (b >= pmfInput.size())
    {
        pmfOutput[state.to_ulong()] = pmfCurrent;
        return;
    }

    const auto& pmfLeft = pmfCurrent;
    calculateStatePMFVectorsRecursively(pmfInput, pmfLeft, state/*.set(b, false)*/, b + 1, pmfOutput, maxPMFSize);

    ProbArray pmfRight = convolveDiscreteDistributions(pmfCurrent, pmfInput[b]);
    if (maxPMFSize > 0)
        pmfRight.resize(maxPMFSize);
    calculateStatePMFVectorsRecursively(pmfInput, pmfRight, state.set(b, true), b + 1, pmfOutput, maxPMFSize);
};

template<typename ProbArray>
inline auto calculateStatePMFVectors(const std::vector<ProbArray>& pmfInput, uint32_t maxPMFSize = 0)
{
    auto pmfOut = std::vector<ProbArray>(1ull << pmfInput.size());
    calculateStatePMFVectorsRecursively(pmfInput, {}, 0, 0, pmfOut, maxPMFSize);
    return pmfOut;
}

class BinomialCoefficient
{
public:
    const uint32_t n = 0;
    
    BinomialCoefficient(const uint32_t n) :n(n)
    {
        m_L.resize(n + 1);
        std::transform(m_L.begin(), std::prev(m_L.end()), std::next(m_L.begin()), [i = 1](const double& v) mutable { return v + std::log(i++); });
    }

    double logVal(const uint32_t k) const
    {
        if (k > n)
            return 0;
        const auto itLast = m_L.rbegin();
        return *itLast - *std::next(itLast, k) - *std::next(m_L.begin(), k);
    }

    uint32_t operator[](const uint32_t k) const
    {
        return static_cast<uint32_t>(round(exp(logVal(k))));
    }

protected:
    std::vector<double> m_L;
};

class BinomialDistributionPMF
{
public:
    BinomialDistributionPMF(const uint32_t n, const double p) : m_p(p), m_coef(n)
    {
        m_log_p = safelog(p);
        m_log1mp = safelog(1 - p);
    }

    inline double operator[](const uint32_t k) const
    {
        if (k > m_coef.n)
            return 0;
        return exp(m_coef.logVal(k) + k * m_log_p + (m_coef.n - k)*m_log1mp);
    }

protected:
    double m_p;
    double m_log_p;
    double m_log1mp;
    BinomialCoefficient m_coef;
};

template<typename ObservationArray = std::vector<double>, typename ProbArray = std::vector<double>>
class BCSKDemodulation
{
protected:
    std::vector<ObservationArray> m_pObservation{};
public:
    const int32_t eta = 15;
    const int32_t nStates = 1 << eta;
    const int32_t mask;
    const uint32_t nMolecules = 50u;
    const uint32_t maxTau = nMolecules;

    const auto& observations() const
    {
        return m_pObservation;
    }

    const ObservationArray& observation(const size_t tau) const
    {
        return m_pObservation[tau];
    }

    BCSKDemodulation(const ProbArray& pHit, const uint32_t nMolecules) :
        eta(int32_t(pHit.size())),
        nStates(1 << int32_t(pHit.size())),
        mask(nStates - 1), 
        nMolecules(nMolecules), 
        maxTau(nMolecules - 1)
    {
        m_pObservation.resize(maxTau + 1);

        auto pmfBinom = std::vector<ProbArray>(eta);

#pragma omp parallel for
        for (auto h = 0; h < eta; ++h)
        {
            pmfBinom[h].resize(nMolecules + 1);
            BinomialDistributionPMF bd(nMolecules, pHit[h]);
            for (auto i = 0u; i <= nMolecules; ++i)
                pmfBinom[h][i] = bd[i];
        }

        auto tmp_mfState = calculateStatePMFVectors(pmfBinom, nMolecules + 1); //pmfState
#pragma omp parallel for
        for (auto state = 0; state < nStates; ++state)
            std::partial_sum(tmp_mfState[state].begin(), tmp_mfState[state].end(), tmp_mfState[state].begin());
        auto cmfState(std::move(tmp_mfState));

        for (auto tau = 0u; tau <= maxTau; ++tau)
            m_pObservation[tau][0] = 1;

#pragma omp parallel for
        for (auto state = 1; state < nStates; ++state)
        {
            const auto& cmfCurrent = cmfState[state];
            for (auto tau = 0u; tau <= maxTau; ++tau)
                m_pObservation[tau][state] = cmfCurrent[tau];
        }
        cmfState.clear();
    }

    static auto calculateStatePriors(const double px1, const int eta)
    {
        const double pRatio = 1.0 / px1 - 1.0;

        const auto nStates = 1 << eta;
        ObservationArray Pi;
        //ObservationArray Pi2(nStates);

        //const int maxvalModif = px1 > 0.5 ? 0 : -eta;

        //double sum = 0;
        const auto mult = std::pow(1 - px1, eta);
#pragma omp parallel for /*reduction(+:sum)*/
        for (auto i = 0; i < nStates; ++i)
        {
            //Pi2[i] = std::exp((int(std::bitset<maxEta>(nStates - 1 - i).count()) + maxvalModif)*log_pRatio);
            //Pi[i] = std::pow(pRatio, (int(std::bitset<32>(nStates - 1 - i).count()) + maxvalModif));
            Pi[i] = mult*std::pow(pRatio, -int(std::bitset<32>(i).count()));
            //sum += Pi[i];
        }
        
//#pragma omp parallel for
//        for (auto i = 0; i < nStates; ++i)
//        {
//            Pi[i] /= sum;
//        }
        return Pi;
    }

};
