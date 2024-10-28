#include <vector>
#include <array>
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <chrono>

#include "TypeDefinitions.h"
#include "BCSKMutualInformation.h"
#include "Utilities.h"
#include <fstream>

#include "CLI11.hpp"
#include "rang.hpp"

#include <omp.h>
#include <map>

#ifdef ENABLE_CUDA
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#endif

floating_t mutinf(const BCSKDemodulation<StateSizedArray, pmf_t>& bcskModel, const std::vector<prob_t>& px1s);


int main(int argc, char** argv)
{
    constexpr unsigned eta = maxEta;// std::min(maxEta, 20u);

    CLI::App app{ "BCSK mutual information calculator" };

    auto bDoNotConcatenate = false;
    app.add_flag("-b,--bareoutput", bDoNotConcatenate, "Do not concatenate timestamp to the output filename");

    std::string outputFileName{ "bcskmi_out" };
    app.add_option("-f,--file", outputFileName, "Filename for the output csv file");

    std::vector<prob_t> px1s;
    constexpr auto px1Increment = 0.05;
    const auto px1sCount = size_t(std::ceil(1.0 / px1Increment) + 1);
    px1s.resize(px1sCount, 0.001);
    std::generate(std::next(px1s.begin()), px1s.end(), [p = .0, px1Increment]() mutable { return p += px1Increment; });
    px1s.back() = 0.999;

    app.add_option("-x,--px1", px1s, "Values for p(x=1)", true)->check(CLI::Range(.0, 1.0));

    std::vector<uint16_t> taus(nMolecules);
    std::iota(taus.begin(), taus.end(), 0);
    //taus.clear(); taus.push_back(29);
    app.add_option("-t,--tau", taus, "Values for tau", true)->check(CLI::Range(0u, nMolecules - 1));

    auto pHit = pmf_t{};
    app.add_option("-p,--phit", pHit, "Values for pHit", true)->required()->expected(-std::make_signed_t<decltype(eta)>(eta))->check(CLI::Range(.0, 1.0));

    int numthreads = omp_get_max_threads() * 0.75;
    app.add_option("-n,--numthreads", numthreads, "Number of threads", true)->check(CLI::Range(1, omp_get_max_threads()));

    auto bGPU = false;
#ifdef ENABLE_CUDA
    app.add_flag("-g,--gpu", bGPU, "Use GPU");
#endif

    //CLI11_PARSE(app, argc, argv);
    std::atexit([]() {std::cout << rang::style::reset; });
    try {
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e) {
        if (e.get_exit_code() != 0)
            std::cout << rang::bg::red;
        return app.exit(e);
    }

    if (!outputFileName.empty())
    {
        if (!bDoNotConcatenate)
        {
            time_t rawtime;
            char buffer[80];

            time(&rawtime);
            strftime(buffer, 80, "_%Y%m%d_%H%M%S", localtime(&rawtime));
            outputFileName += buffer;
        }
        outputFileName += ".csv";
    }

    const auto bWriteOutput = !outputFileName.empty();


    omp_set_num_threads(numthreads);
    omp_set_nested(false);

    auto printBoldGreen = [](std::string str)
    {
        std::cout << rang::style::reset << rang::fg::green << rang::style::bold << str << rang::style::reset;
    };

    auto printBoldMagenta = [](std::string str)
    {
        std::cout << rang::style::reset << rang::fg::magenta << rang::style::bold << str << rang::style::reset;
    };

    auto printBoldRed = [](std::string str)
    {
        std::cout << rang::style::reset << rang::fg::red << rang::style::bold << str << rang::style::reset;
    };

    pHit.resize(eta);
    printBoldGreen("---------------------------------------------------------------");
    std::cout << std::endl;

    std::ofstream outputFile;
    if (bWriteOutput)
    {
        outputFile.open(outputFileName, std::ofstream::out);
        if (!outputFile.is_open())
        {
            printBoldRed("Cannot open output file ");
            std::cout << outputFileName << std::endl;
            exit(-1);
        }
        printBoldGreen("Output file: ");
        std::cout << outputFileName << std::endl;

#ifndef STORE_HY_HIST
        outputFile << "px1; tau; HY; HY_X; Iterations; time" << std::endl;
#endif
    }

#ifdef ENABLE_CUDA
    if (bGPU)
    {
        printBoldGreen("GPU: Active");
        std::cout << std::endl;

        struct cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, 0);
        std::cout << "using " << properties.multiProcessorCount << " multiprocessors" << std::endl;
        std::cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "max threads per block: " << properties.maxThreadsPerBlock << std::endl;
        //system("pause");
    }
#endif
    printBoldGreen("Thread count: ");
    std::cout << omp_get_max_threads() << std::endl;
    printBoldGreen("eta: ");
    std::cout <<  eta << std::endl;
    printBoldGreen("nMolecules: ");
    std::cout << nMolecules << std::endl;
    printBoldGreen("Iteration count range: ");
    std::cout << minIterations << " - " << maxIterations << std::endl;
    printBoldGreen("Std Dev exit: ");
    std::cout << "Window size " << stddevWindowSize << " - Threshold " << stddevThreshold << std::endl;

    printBoldMagenta("p(x=1) array: ");
    std::cout << " [" + CLI::detail::join(px1s) + "]" << std::endl;
    printBoldMagenta("Tau array: ");
    std::cout << " [" + CLI::detail::join(taus) + "]" << std::endl;
    printBoldMagenta("pHit array: ");
    std::cout << " [" + CLI::detail::join(pHit) + "]" << std::endl;


    printBoldGreen("sizeof(BCSKMutualInformation): ");
    std::cout << "CPU: " << sizeof(BCSKMutualInformation<false>);
#ifdef ENABLE_CUDA
    std::cout << " - GPU: " << sizeof(BCSKMutualInformation<true>);
#endif
    std::cout << std::endl;

    Timer timer;

    using model_t = BCSKDemodulation<StateSizedArray, pmf_t>;
    const model_t bcskModel(pHit, nMolecules);

    const ObservationsAndPriors OPi(bcskModel, px1s);

    std::vector<MutualInformation> mutInf;
    for (int px1i = 0; px1i < px1s.size(); ++px1i)
        for (const auto tau : taus)
            mutInf.push_back(MutualInformation{ prob_t(px1i), tau, .0, .0 });


#pragma omp parallel for
    for (auto i = 0; i < mutInf.size(); ++i)
    {
        auto& mi = mutInf[i];
        int nIterations = 0;
        Timer timerInner;
        auto tau = mi.tau;
        auto px1i = mi.px1;
        auto px1 = px1s[px1i];
        mi.px1 = px1;
        if(bGPU)
        {
#ifdef ENABLE_CUDA
            auto& bcskMutInfCalculator = BCSKMutualInformation<true>::instance();
            nIterations = bcskMutInfCalculator.calculate(OPi, tau, px1);
            mi.HY = bcskMutInfCalculator.HY;
            mi.HY_X = bcskMutInfCalculator.HY_X;
#endif
        }
        else
        {
            auto& bcskMutInfCalculator = BCSKMutualInformation<false>::instance();
            nIterations = bcskMutInfCalculator.calculate(OPi, tau, px1);
            mi.HY = bcskMutInfCalculator.HY;
            mi.HY_X = bcskMutInfCalculator.HY_X;
        }

        const auto t = timerInner.duration().count();

//#pragma omp critical
//        {
//            std::cout << "p(x=1)=" << mi.px1 << "\tT=" << mi.tau << "\t" << nIterations << ": "/* << std::setprecision(3)*/ << mi.value() << "\t\t" << t << " s\t\t" << t * 1e6 / nIterations << " mus/it" << std::endl;
//        }

        if (bWriteOutput)
        {
            std::stringstream ss;
            ss << mi.px1 << ";" << mi.tau << ";" << mi.HY << ";" << mi.HY_X << ";" << nIterations << ";" << t << std::endl;
#pragma omp critical
            {
#ifdef STORE_HY_HIST
                outputFile << "hyhist=[";
                for (auto iterationNo = 0; iterationNo <= nIterations; ++iterationNo)
                    outputFile << bcskMutInfCalculator.histHY[iterationNo] - bcskMutInfCalculator.HY_X << " ";
                outputFile << "];";
                outputFile << std::endl << "plot(hyhist)";
                outputFile.close();
                exit(0);
#endif
                outputFile << ss.str();
            }
        }
    }

    outputFile.close();

    printBoldGreen("Elapsed time: ");
    std::cout << timer.duration().count() << " seconds" << std::endl;

    const auto maxMutInf = std::max_element(mutInf.begin(), mutInf.end());
    printBoldGreen("Maximum mutual information: ");
    std::cout << maxMutInf->toString() << std::endl;

    printBoldMagenta("---------------------------------------------------------------");
    std::cout << std::endl;
    //system("pause");
    return 0;
}

