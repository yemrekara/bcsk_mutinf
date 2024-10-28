#ifdef ENABLE_CUDA

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include "helper_cuda.h"

#include "TypeDefinitions.h"
#include "BCSKMutualInformation.h"

using uint = unsigned int;

__global__ void setMandN_kernel(const uint nStatesHalf, const floating_t* const O, const prob_t px1, const prob_t px0,
    floating_t* M00, floating_t* M01, floating_t* M10, floating_t* M11, 
    floating_t* N0, floating_t* N1)
{
    int s = blockDim.x * blockIdx.x + threadIdx.x;

    if (s < nStatesHalf)
    {
        setMandNforState(s, O, px1, px0, M00, M01, M10, M11, N0, N1);
    }
}

__global__ void updateAlpha_kernel(const uint nStatesQuarter, const floating_t* const Myx, const floating_t* const alpha, floating_t* alpha_target)
{
    int s = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ const floating_t* Myx2;
    __shared__ const floating_t* alpha2;
    if (threadIdx.x == 0)
    {
        Myx2 = Myx + nStatesQuarter;
        alpha2 = alpha + nStatesQuarter;
    }
    __syncthreads();

    if (s < nStatesQuarter)
    {
        alpha_target[s << 1] = clamp(alpha[s] * Myx[s] + alpha2[s] * Myx2[s], .0, floating_t{ 2.0 });
    }
}

__global__ void calculateHY_X_helperkernel(const uint nStates, const floating_t* const O, const floating_t* const Pi, floating_t* const out)
{
    int state = blockDim.x * blockIdx.x + threadIdx.x;

    if (state < nStates)
    {
        const auto& p0 = O[state];
        const auto p1 = (floating_t{ 1.0 } - p0);
        const auto v0 = p0 < eps ? 0 : p0 * log2(p0);
        const auto v1 = p1 < eps ? 0 : p1 * log2(p1);
        // Pi[state]=\sum_{q'} \pi(q')*p(q|q')
        out[state] = Pi[state] * (v0 + v1);
    }
}

void BCSKMutualInformationBase<true>::initArrays(const floating_t* O, size_t initAlpha_idx)
{
    const auto nStatesHalf = nStates >> 1;

    for(const auto y: {0,1})
    {
        for (const auto x : { 0,1 })
            m_M[y][x].resize(nStatesHalf);
        m_N[y].resize(nStatesHalf);
    } 
    m_alphaArrays[0].resize(nStatesHalf);
    m_alphaArrays[1].resize(nStatesHalf);

    thrust::fill(m_alphaArrays[m_alphaArrayIdx].begin(), m_alphaArrays[m_alphaArrayIdx].end(), floating_t{ .0 });
    m_alphaArrays[m_alphaArrayIdx][initAlpha_idx] = floating_t{ 1.0 };

    auto* M00 = thrust::raw_pointer_cast(&m_M[0][0][0]);
    auto* M01 = thrust::raw_pointer_cast(&m_M[0][1][0]);
    auto* M10 = thrust::raw_pointer_cast(&m_M[1][0][0]);
    auto* M11 = thrust::raw_pointer_cast(&m_M[1][1][0]);
    auto* N0 = thrust::raw_pointer_cast(&m_N[0][0]);
    auto* N1 = thrust::raw_pointer_cast(&m_N[1][0]);

    const int threadsPerBlock = 1024;
    const int blocksPerGrid = (nStatesHalf + threadsPerBlock - 1) / threadsPerBlock;
    setMandN_kernel << <blocksPerGrid, threadsPerBlock >> > (nStatesHalf, O, px1, 1 - px1, M00, M01, M10, M11, N0, N1);

    checkCudaErrors_(cudaGetLastError());
    checkCudaErrors_(cudaDeviceSynchronize());
}

floating_t calculateDeltaHY_GPU(const bool y, const thrust::device_vector<floating_t>& Ny, const thrust::device_vector<floating_t>& alpha, cudaStream_t stream)
{
    floating_t deltaHY = thrust::inner_product(thrust::cuda::par.on(stream), Ny.begin(), Ny.end(), alpha.begin(), floating_t{ .0 });
    deltaHY = -log2(clamp(deltaHY, eps_<floating_t>(), floating_t{ 1.0 }));
    return deltaHY;
}

void updateAlpha_GPU(const uint nStatesQuarter, const bool y, const xArray<thrust::device_vector<floating_t>>& My, const thrust::device_vector<floating_t>& alpha, thrust::device_vector<floating_t>& alpha_new, cudaStream_t stream)
{
    const floating_t* p_alpha = thrust::raw_pointer_cast(&alpha[0]);
    floating_t* p_alpha_new = thrust::raw_pointer_cast(&alpha_new[0]);

    const int threadsPerBlock = 1024;
    const int blocksPerGrid = (nStatesQuarter + threadsPerBlock - 1) / threadsPerBlock;

    updateAlpha_kernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (nStatesQuarter, thrust::raw_pointer_cast(&My[0][0]), p_alpha, p_alpha_new);
    updateAlpha_kernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (nStatesQuarter, thrust::raw_pointer_cast(&My[1][0]), p_alpha, p_alpha_new + 1);


    checkCudaErrors_(cudaGetLastError());
    checkCudaErrors_(cudaDeviceSynchronize());
}

floating_t calculateHY_X_GPU(const int32_t nStates, const floating_t* O, const floating_t* Pi, cudaStream_t stream)
{
    thrust::device_vector<floating_t> out(nStates);
    const int threadsPerBlock = 1024;
    const int blocksPerGrid = (nStates + threadsPerBlock - 1) / threadsPerBlock;

    calculateHY_X_helperkernel << <blocksPerGrid, threadsPerBlock, 0, stream >> > (nStates, O, Pi, thrust::raw_pointer_cast(&out[0]));
 
    return -thrust::reduce(out.begin(), out.end(), floating_t{ .0 });
}

void ObservationsAndPriors::initGPU()
{
    const auto nTaus = bcskModel.observations().size();

    const auto nPx1s = px1s.size();
    for (int px1i = 0; px1i < nPx1s; ++px1i)
        Pis.push_back(BCSKDemodulation<StateSizedArray, pmf_t>::calculateStatePriors(px1s[px1i], bcskModel.eta));

    vec_d_O.resize(nTaus);
    d_O.resize(nTaus);
    vec_d_Pi.resize(nPx1s);
    d_Pi.resize(nPx1s);

    d_px1s.assign(px1s.begin(), px1s.end());

    // Upload Observation probabilities
    for (int tau = 0; tau < nTaus; ++tau)
    {
        const auto& O = bcskModel.observation(tau);
        vec_d_O[tau].assign(O.begin(), O.end());
        d_O[tau] = thrust::raw_pointer_cast(&vec_d_O[tau][0]);
    }

    // Upload State priors
    for (int px1i = 0; px1i < nPx1s; ++px1i)
    {
        vec_d_Pi[px1i].assign(Pis[px1i].begin(), Pis[px1i].end());
        auto* p = thrust::raw_pointer_cast(&vec_d_Pi[px1i][0]);
        d_Pi[px1i] = p;
    }
}

#endif