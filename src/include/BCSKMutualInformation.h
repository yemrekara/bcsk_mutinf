#pragma once

#include "TypeDefinitions.h"
#include "Utilities.h"
#include "BCSKDemodulation.h"
#include <map>

//#define STORE_HY_HIST

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "helper_math.h"

using thrust::swap;

floating_t calculateDeltaHY_GPU(const bool y, const thrust::device_vector<floating_t>& Ny, const thrust::device_vector<floating_t>& alpha, cudaStream_t stream = nullptr);
void updateAlpha_GPU(const uint nStatesQuarter, const bool y, const xArray<thrust::device_vector<floating_t>>& My, const thrust::device_vector<floating_t>& alpha, thrust::device_vector<floating_t>& alpha_new, cudaStream_t stream = nullptr);
floating_t calculateHY_X_GPU(const int32_t nStates, const floating_t* O, const floating_t* Pi, cudaStream_t stream = nullptr);

template <typename T, size_t N>
__host__ __device__ void array_fill(std::array<T, N>& arr, const T& value) {
    for (size_t i = 0; i < N; ++i) {
        arr[i] = value;
    }
}
#else

template <typename T, size_t N>
void array_fill(std::array<T, N>& arr, const T& value) {
    arr.fill(value);
}

#define __host__
#define __device__

#endif

class ObservationsAndPriors
{
public:
    ObservationsAndPriors(const BCSKDemodulation<StateSizedArray, pmf_t>& bcskModel, const std::vector<prob_t>& px1s) :bcskModel(bcskModel), px1s(px1s)
    {
        const auto nTaus = bcskModel.observations().size();

        const auto nPx1s = px1s.size();
        for (int px1i = 0; px1i < nPx1s; ++px1i)
            Pis.push_back(BCSKDemodulation<StateSizedArray, pmf_t>::calculateStatePriors(px1s[px1i], bcskModel.eta));
        initGPU();
    }

    std::tuple<const floating_t*, const floating_t*> at(size_t tau, prob_t px1, bool gpu = false) const
    {
        static prob_t e = 1e-5;
        auto it = std::find_if(px1s.begin(), px1s.end(), [px1](prob_t f) {return f >= px1 - e && f <= px1 + e; });
        assert(it != px1s.end());
        return at(tau, size_t(std::distance(px1s.begin(), it)), gpu);
    }

    std::tuple<const floating_t*, const floating_t*> at(size_t tau, size_t px1i, bool gpu = false) const
    {
#ifdef ENABLE_CUDA
        if(gpu)
            return { thrust::raw_pointer_cast(&vec_d_O[tau][0]), thrust::raw_pointer_cast(&vec_d_Pi[px1i][0]) };
#endif
        return std::tuple<const floating_t*, const floating_t*>{ bcskModel.observation(tau).data(), Pis[px1i].data() };
    }

    int nStates() const
    {
        return bcskModel.nStates;
    }

    const floating_t* h_O(size_t tau) const
    {
        return bcskModel.observation(tau).data();
    }

protected:
    const BCSKDemodulation<StateSizedArray, pmf_t>& bcskModel;
    std::vector<prob_t> px1s;
    std::vector<StateSizedArray> Pis;

#ifdef ENABLE_CUDA
    void initGPU();
    std::vector<thrust::device_vector<floating_t>> vec_d_O; //Observations memory
    thrust::device_vector<floating_t*> d_O; //Pointers to observations
    std::vector<thrust::device_vector<floating_t>> vec_d_Pi; //Priors memory
    thrust::device_vector<floating_t*> d_Pi; //Pointers to priors
    thrust::device_vector<prob_t> d_px1s;
#else
    void initGPU() {}
#endif
};

constexpr unsigned nMolecules = 50u;

constexpr unsigned minIterations = 500u;
constexpr unsigned maxIterations = 100'000u;

constexpr unsigned stddevWindowSize = std::max(minIterations, 500u);
constexpr floating_t stddevThreshold = 0.0001;

struct MutualInformation
{
    prob_t px1;
    uint16_t tau;
    floating_t HY;
    floating_t HY_X;

    CUDA_CALLABLE_MEMBER auto value() const
    {
        return HY - HY_X;
    }

    CUDA_CALLABLE_MEMBER bool operator<(const MutualInformation& other) const
    {
        return this->value() < other.value();
    }

    std::string toString() const
    {
        return std::to_string(value()) + " at p(x=1)=" + std::to_string(px1) + " and Tau=" + std::to_string(tau);
    }

};

CUDA_CALLABLE_MEMBER inline void setMandNforState(const int state, const floating_t* O, const prob_t px1, const double px0, floating_t* M00, floating_t* M01, floating_t* M10, floating_t* M11, floating_t* N0, floating_t* N1)
{
    const auto state0 = (state << 1);
    const auto state1 = state0 + 1;

    auto& M00s = M00[state];
    auto& M01s = M01[state];
    auto& M10s = M10[state];
    auto& M11s = M11[state];
    auto& N0s = N0[state];
    auto& N1s = N1[state];

    M00s = O[state0] * px0;
    M01s = O[state1] * px1;
    M10s = px0 - M00s;
    M11s = px1 - M01s;

    //for (const auto y : { 0, 1 })
    //    for (const auto x : { 0, 1 })
    //        m_M[y][x][targetstate] = clamp(m_M[y][x][targetstate], 1e-7/*eps_<floating_t>()*/, floating_t{ 1.0 });
    M00s = clamp(M00s, eps_<floating_t>(), floating_t{ 1.0 });
    M01s = clamp(M01s, eps_<floating_t>(), floating_t{ 1.0 });
    M10s = clamp(M10s, eps_<floating_t>(), floating_t{ 1.0 });
    M11s = clamp(M11s, eps_<floating_t>(), floating_t{ 1.0 });

    N0s = clamp(M00s + M01s, floating_t{ .0 }, floating_t{ 1.0 });
    N1s = clamp(M10s + M11s, floating_t{ .0 }, floating_t{ 1.0 });

    if (N0s > eps_<floating_t>())
    {
        M00s /= N0s;
        M01s /= N0s;
    }

    if (N1s > eps_<floating_t>())
    {
        M10s /= N1s;
        M11s /= N1s;
    }
}

template<typename StorageArray>
class BCSKMutualInformationBaseStorage : public MutualInformation
{
public:
    CUDA_CALLABLE_MEMBER BCSKMutualInformationBaseStorage() : MutualInformation()
    {

    }

    /** Copy constructor */
    CUDA_CALLABLE_MEMBER BCSKMutualInformationBaseStorage(const BCSKMutualInformationBaseStorage& other) :
        MutualInformation(other)
    {
        this->HYbuffer = other.HYbuffer;
        this->m_N = other.m_N;
        this->m_M = other.m_M;
        this->m_alphaArrays = other.m_alphaArrays;
        this->m_alphaArrayIdx = other.m_alphaArrayIdx;

#ifdef STORE_HY_HIST
        histHY = other.histHY;
#endif

    }

    /** Move constructor */
    CUDA_CALLABLE_MEMBER BCSKMutualInformationBaseStorage(BCSKMutualInformationBaseStorage&& other) noexcept /* noexcept needed to enable optimizations in containers */
    {
        std::swap(this->HYbuffer, other.HYbuffer);
        std::swap(this->m_N, other.m_N);
        std::swap(this->m_M, other.m_M);
        std::swap(this->m_alphaArrays, other.m_alphaArrays);
        std::swap(this->m_alphaArrayIdx, other.m_alphaArrayIdx);

        MutualInformation::operator=(std::move(other));

#ifdef STORE_HY_HIST
        std::swap(histHY, other.histHY);
#endif
    }

    /** Destructor */
    CUDA_CALLABLE_MEMBER ~BCSKMutualInformationBaseStorage() noexcept /* explicitly specified destructors should be annotated noexcept as best-practice */
    {

    }

    /** Copy assignment operator */
    CUDA_CALLABLE_MEMBER BCSKMutualInformationBaseStorage& operator= (const BCSKMutualInformationBaseStorage& other)
    {
        BCSKMutualInformationBaseStorage tmp(other);         // re-use copy-constructor
        *this = std::move(tmp); // re-use move-assignment
        return *this;
    }

    CUDA_CALLABLE_MEMBER BCSKMutualInformationBaseStorage& operator=(BCSKMutualInformationBaseStorage&& other) noexcept
    {
        if (this == &other)
        {
            // take precautions against `foo = std::move(foo)`
            return *this;
        }
        std::swap(this->HYbuffer, other.HYbuffer);
        std::swap(this->m_N, other.m_N);
        std::swap(this->m_M, other.m_M);
        std::swap(this->m_alphaArrays, other.m_alphaArrays);
        std::swap(this->m_alphaArrayIdx, other.m_alphaArrayIdx);

        MutualInformation::operator=(std::move(other));

#ifdef STORE_HY_HIST
        std::swap(histHY, other.histHY);
#endif

        return *this;
    }

protected:
    int32_t nStates;
    int32_t mask;
    int32_t nStatesHalf;
    int32_t nStatesQuarter;

#ifdef STORE_HY_HIST
    array_t<floating_t, maxIterations>  histHY;
#endif

    array_t<floating_t, stddevWindowSize> HYbuffer;
    floating_t m_HYbuffer_sum = .0;
    floating_t m_HYbuffer_variance = .0;

    unsigned iteration = 0;

    //N[y][q'] <=> p(y|q')

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                             p(y,q|q')     p(y|q) p(q|q')    observation * transition                    //
    //  M[y][x][q'] = p(q|q',y) = ----------- = --------------- = --------------------------,  where q={q',x}  //
    //                              p(y|q')         p(y|q')               N[y][q']                             //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    yArray<xArray<StorageArray>> m_M;
    yArray<StorageArray> m_N;
    array_t<StorageArray, 2> m_alphaArrays;
    uint8_t m_alphaArrayIdx = 0;

    CUDA_CALLABLE_MEMBER bool updateAndCheckHYvariance()
    {
        constexpr auto N = double(stddevWindowSize);
#ifdef STORE_HY_HIST
        histHY[iteration - 1] = HY;
#endif
        const auto pos = iteration % this->HYbuffer.size();
        const auto oldHY = this->HYbuffer[pos];
        this->HYbuffer[pos] = this->HY;

        if (iteration == this->HYbuffer.size())
        {
            this->m_HYbuffer_sum = 0;
            for (const auto h : this->HYbuffer)
                this->m_HYbuffer_sum += h;
            //HYbuffer_variance = variance(HYbuffer.data(), HYbuffer.size());
            this->m_HYbuffer_variance = variance(this->HYbuffer.data(), this->HYbuffer.size(), this->m_HYbuffer_sum / N);
        }
        else if (iteration > this->HYbuffer.size())
        {
            const auto oldsum = this->m_HYbuffer_sum;
            const auto diff = this->HY - oldHY;
            this->m_HYbuffer_sum += diff;
            this->m_HYbuffer_variance += diff * (this->HY + oldHY - (oldsum + this->m_HYbuffer_sum) / N) / (N - 1.0);
        }

        if (iteration >= this->HYbuffer.size())
            return sqrt(this->m_HYbuffer_variance) < stddevThreshold;

        return false;
    }
};

template<bool bGPU> class BCSKMutualInformationBase;

template<> class BCSKMutualInformationBase<false> : public BCSKMutualInformationBaseStorage<HalfStateSizedArray>
{
protected:
    CUDA_CALLABLE_MEMBER void initArrays(const floating_t* O, size_t initAlpha_idx)
    {
        const auto px0 = 1 - px1;;
        auto& M00 = m_M[0][0];
        auto& M01 = m_M[0][1];
        auto& M10 = m_M[1][0];
        auto& M11 = m_M[1][1];
        auto& N0 = m_N[0];
        auto& N1 = m_N[1];
        // Populate M and N
        for (auto s = 0; s < nStatesHalf; ++s)
            setMandNforState(s, O, px1, px0, M00.data(), M01.data(), M10.data(), M11.data(), N0.data(), N1.data());

        // Initialize alpha
        auto& alpha = m_alphaArrays[m_alphaArrayIdx];
        array_fill(alpha, .0);
        alpha[initAlpha_idx] = 1;
    }

    CUDA_CALLABLE_MEMBER inline void updateAlpha(const bool y)
    {
        const auto& alpha = this->m_alphaArrays[this->m_alphaArrayIdx];
        auto& alpha_new = this->m_alphaArrays[1 - this->m_alphaArrayIdx];
        const auto& My = this->m_M[y];

        const auto* alpha2 = &alpha[this->nStatesQuarter];
        for (const auto b : { 0, 1 })
        {
            auto* alpha_target = &alpha_new[b];
            const auto& Myx = My[b];
            const auto* Myx2 = &Myx[this->nStatesQuarter];
            for (auto s = 0; s < this->nStatesQuarter; ++s)
                alpha_target[s << 1] = clamp(alpha[s] * Myx[s] + alpha2[s] * Myx2[s], .0, floating_t{ 2.0 });
        }
        this->m_alphaArrayIdx = 1 - this->m_alphaArrayIdx;
    }

    CUDA_CALLABLE_MEMBER inline floating_t calculateDeltaHY(const bool y) const
    {
        const auto& alpha = this->m_alphaArrays[this->m_alphaArrayIdx];
        auto deltaHY = floating_t{ .0 };

        const auto& Ny = this->m_N[y];
        for (auto s = 0; s < this->nStatesHalf; ++s)
            deltaHY += alpha[s] * Ny[s];

        deltaHY = -log2(clamp(deltaHY, eps_<floating_t>(), floating_t{ 1.0 }));
        return deltaHY;
    }

    CUDA_CALLABLE_MEMBER floating_t calculateHY_X(const int32_t nStates, const floating_t* O, const floating_t* Pi) const
    {
        // Calculate H(Y|X)
        floating_t HY_X = 0;
        for (auto state = 0; state < nStates; ++state)
        {
            const auto& p0 = O[state];
            const auto p1 = (floating_t{ 1.0 } -p0);
            const auto v0 = p0 < eps ? 0 : p0 * log2(p0);
            const auto v1 = p1 < eps ? 0 : p1 * log2(p1);
            // Pi[state]=\sum_{q'} \pi(q')*p(q|q')
            HY_X -= Pi[state] * (v0 + v1);
        }
        return HY_X;
    }
};

#ifdef ENABLE_CUDA
template<> class BCSKMutualInformationBase<true> : public BCSKMutualInformationBaseStorage<thrust::device_vector<floating_t>>
{
protected:
    cudaStream_t m_stream = nullptr;
    BCSKMutualInformationBase()
    {
        cudaStreamCreate(&m_stream);
    }
    
    ~BCSKMutualInformationBase()
    {
        cudaStreamDestroy(m_stream);
    }

    void initArrays(const floating_t* d_O, size_t initAlpha_idx);
    inline floating_t calculateDeltaHY(const bool y) const
    {
        return calculateDeltaHY_GPU(y, this->m_N[y], this->m_alphaArrays[this->m_alphaArrayIdx], m_stream);
    }

    inline void updateAlpha(const bool y)
    {
        updateAlpha_GPU(this->nStatesQuarter, y, this->m_M[y], this->m_alphaArrays[this->m_alphaArrayIdx], this->m_alphaArrays[1 - this->m_alphaArrayIdx], m_stream);
        m_alphaArrayIdx = 1 - m_alphaArrayIdx;
    }

    floating_t calculateHY_X(const int32_t nStates, const floating_t* O, const floating_t* Pi) const
    {
        return calculateHY_X_GPU(nStates, O, Pi, m_stream);
    }
};
#endif

template<bool bGPU = false>
class BCSKMutualInformation : public BCSKMutualInformationBase<bGPU>
{
public:
    using BCSKMutualInformationBase<bGPU>::BCSKMutualInformationBase;
    using BCSKMutualInformationBase<bGPU>::operator=;

    static BCSKMutualInformation<bGPU>& instance()
    {
        static thread_local BCSKMutualInformation<bGPU> B;
        return B;
    }

    int calculate(const BCSKDemodulation<StateSizedArray, pmf_t>& bcskModel, const uint16_t tau, const StateSizedArray& Pi, const prob_t px1)
    {
        return calculate(bcskModel.nStates, bcskModel.observation(tau), tau, Pi, px1);
    }

    int calculate(const int32_t nStates, const StateSizedArray& O, const uint16_t tau, const StateSizedArray& Pi, const prob_t px1)
    {
        return calculate(nStates, O.data(), tau, Pi.data(), px1);
    }

    CUDA_CALLABLE_MEMBER int calculate(const ObservationsAndPriors& OPi, const uint16_t tau, const prob_t px1)
    {
        const floating_t* O;
        const floating_t* Pi;
        std::tie(O, Pi) = OPi.at(tau, px1, bGPU);
        return calculate(OPi.nStates(), O, tau, Pi, px1, OPi.h_O(tau));
    }

    CUDA_CALLABLE_MEMBER int calculate(const int32_t nStates, const floating_t* O, const uint16_t tau, const floating_t* Pi, const prob_t px1, const floating_t* h_O = nullptr)
    {
        this->nStates = nStates;
        this->mask = nStates - 1;
        this->tau = tau;
        this->px1 = px1;
        this->nStatesHalf = nStates >> 1;
        this->nStatesQuarter = nStates >> 2;

        //O[q] = p(y=0|q) => p(y=1|q) = 1-O[q]
        this->HY_X = this->calculateHY_X(nStates, O, Pi);
        //return 0;
        return this->calculateHY(O, px1, bGPU ? h_O : O);
    }

protected:
    __host__ __device__ int calculateHY(const floating_t* O, const prob_t px1, const floating_t* h_O)
    {
        const auto px0 = 1 - px1;

#ifdef __CUDA_ARCH__
        curandStatePhilox4_32_10_t state;
        curand_init(clock64(), blockDim.x * blockIdx.x + threadIdx.x, 0, &state);
        const auto unif = [&]() { return curand_uniform(&state); };
        // Initial state
        decltype(this->nStates) s = ceilf(curand_uniform(&state)*this->nStates) - 1;
#else
        const auto unif = [&]() { return unirnd(.0, 1.0); };
        // Initial state
        auto s = uniintrnd(0, this->nStates - 1);
#endif

        const auto maskHalf = (this->nStates >> 1) - 1;
        const size_t initAlpha_idx = (s << 1) & maskHalf;

        this->initArrays(O, initAlpha_idx);


        array_fill(this->HYbuffer, .0);
        this->m_HYbuffer_sum = .0;
        this->m_HYbuffer_variance = .0;

        auto totalHY = .0;
        this->iteration = 0;
        for (this->iteration = 1; this->iteration <= maxIterations; ++this->iteration)
        {
            // Next state and observation y
            s = this->mask & ((s << 1) + (unif() > px0));
            const bool y = unif() > h_O[s];

            totalHY += this->calculateDeltaHY(y);
            this->HY = totalHY / this->iteration;

            if (this->iteration == maxIterations || this->updateAndCheckHYvariance())
                break;

            this->updateAlpha(y);
        }

        return this->iteration;
    }
};
