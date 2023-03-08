/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <vector>
#include <cstdint>

#include <miopen/solver.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
#include <ck/library/tensor_operation_instance/gpu/quantization/grouped_convolution_forward_perlayer_quantization.hpp>
//#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>

#endif
MIOPEN_DECLARE_ENV_VAR(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_LAYER_QUANT_XDLOPS)

namespace miopen {
namespace solver {

using ActivationOp = ck::tensor_operation::element_wise::PassThrough;
using OElementOp = ck::tensor_operation::element_wise::Activation_Mul_Clamp<ActivationOp>;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_COMPOSABLEKERNEL
template <typename DataType>
using DeviceOpGFwdQuant = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD<
    2,
    ck::tensor_layout::convolution::GNHWC,
    ck::tensor_layout::convolution::GKYXC,
    ck::Tuple<>,
    ck::tensor_layout::convolution::GNHWK,
    DataType,
    DataType,
    ck::Tuple<>,
    DataType,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    OElementOp>;

template <typename DataType>
using DeviceOpGFwdQuantPtrs =
    ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOpGFwdQuant<DataType>>;

struct CKArgsGFwd
{
    CKArgsGFwd(const ProblemDescription& problem)
    {
        G        = ProblemInterpreter::GetGroupCountG(problem);
        N        = ProblemInterpreter::GetBatchN(problem);
        K        = ProblemInterpreter::GetOutputChannelK(problem);
        C        = ProblemInterpreter::GetInputChannelC(problem);
        C1       = C/G;
        Hi       = ProblemInterpreter::GetInputHeightHi(problem);
        Wi       = ProblemInterpreter::GetInputWidthWi(problem);
        Ho       = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo       = ProblemInterpreter::GetOutputWidthWo(problem);
        Y        = ProblemInterpreter::GetFilterHeightY(problem);
        X        = ProblemInterpreter::GetFilterWidthX(problem);
        input    = {G, N, Hi, Wi, C};
        output   = {G, N, Ho, Wo, K};
        weight   = {G, K, Y, X, C};
    /*std::cout<<"in struct"<<std::endl;
    std::cout<<"**********************************************G: "<<G<<std::endl;
    std::cout<<"**********************************************N: "<<N<<std::endl;
    std::cout<<"**********************************************K: "<<K<<std::endl;
    std::cout<<"**********************************************C: "<<C<<std::endl;
    std::cout<<"**********************************************Hi: "<<input[2]<<std::endl;
    std::cout<<"**********************************************Wi: "<<input[3]<<std::endl;
    std::cout<<"**********************************************Ho: "<<output[2]<<std::endl;
    std::cout<<"**********************************************Wo: "<<output[3]<<std::endl;
    std::cout<<"**********************************************Y: "<<weight[2]<<std::endl;
    std::cout<<"**********************************************X: "<<weight[3]<<std::endl;
    std::cout<<std::endl;*/
        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
        in_strides = {0, 0, 0, 0, 1};
        out_strides = {0, 0, 0, 0, 1};
        wei_strides = {0, 0, 0, 0, 1};
        std::partial_sum(rbegin(input),
                        std::prev(rend(input)),
                        std::next(rbegin(in_strides)),
                        std::multiplies<>{});
        std::partial_sum(rbegin(weight),
                        std::prev(rend(weight)),
                        std::next(rbegin(wei_strides)),
                        std::multiplies<>{});
        std::partial_sum(rbegin(output),
                        std::prev(rend(output)),
                        std::next(rbegin(out_strides)),
                        std::multiplies<>{});

        std::rotate(
            rbegin(input), std::next(rbegin(input)), std::next(rbegin(input), 3));
        std::rotate(
            rbegin(in_strides), std::next(rbegin(in_strides)), std::next(rbegin(in_strides), 3));
        std::rotate(
            rbegin(weight), std::next(rbegin(weight)), std::next(rbegin(weight), 3));
        std::rotate(
            rbegin(wei_strides), std::next(rbegin(wei_strides)), std::next(rbegin(wei_strides), 3));
        std::rotate(
            rbegin(output), std::next(rbegin(output)), std::next(rbegin(output), 3));
        std::rotate(
            rbegin(out_strides), std::next(rbegin(out_strides)), std::next(rbegin(out_strides), 3));
    /*std::cout<<"after computation"<<std::endl;
    std::cout<<"**********************************************input[0]: "<<input[0]<<std::endl;
    std::cout<<"**********************************************input[1]: "<<input[1]<<std::endl;
    std::cout<<"**********************************************input[2]: "<<input[2]<<std::endl;
    std::cout<<"**********************************************input[3]: "<<input[3]<<std::endl;
    std::cout<<"**********************************************input[4]: "<<input[4]<<std::endl;
    std::cout<<std::endl;*/
    }
    int G;
    int N;
    int K;
    int C;
    int C1;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    std::array<ck::index_t, 5> input;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> output;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> weight;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> strides;
    std::array<ck::index_t, 2> dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};
/*
struct CKArgsGFwd
{
    CKArgsGFwd(const ProblemDescription& problem)
    {
        G        = ProblemInterpreter::GetGroupCountG(problem);
        N        = ProblemInterpreter::GetBatchN(problem);
        K        = ProblemInterpreter::GetOutputChannelK(problem);
        C        = ProblemInterpreter::GetInputChannelC(problem);
        Hi       = ProblemInterpreter::GetInputHeightHi(problem);
        Wi       = ProblemInterpreter::GetInputWidthWi(problem);
        Ho       = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo       = ProblemInterpreter::GetOutputWidthWo(problem);
        Y        = ProblemInterpreter::GetFilterHeightY(problem);
        X        = ProblemInterpreter::GetFilterWidthX(problem);
        input    = {G, N, C, ProblemInterpreter::GetInputHeightHi(problem),
                 ProblemInterpreter::GetInputWidthWi(problem)};
        output   = {G, N, K, ProblemInterpreter::GetOutputHeightHo(problem),
                  ProblemInterpreter::GetOutputWidthWo(problem)};
        weight   = {G, K, C, ProblemInterpreter::GetFilterHeightY(problem),
                  ProblemInterpreter::GetFilterWidthX(problem)};
        in_strides  = {N * Hi * Wi * C, Hi * Wi * C, 1, Wi * C, C};
        out_strides = {N * Ho * Wo * C, Ho * Wo * C, 1, Wo * C, C};
        wei_strides = {K * Y * X * C, Y * X * C, 1, X * C, C};
        strides  = {ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                   ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        dilation = {ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                    ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding = {ProblemInterpreter::GetInputLeftPadH(problem),
                    ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding = {ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                    ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }
    int G;
    int N;
    int K;
    int C;
    int Hi;
    int Wi;
    int Ho;
    int Wo;
    int Y;
    int X;
    std::array<ck::index_t, 5> input;
    std::array<ck::index_t, 5> in_strides;
    std::array<ck::index_t, 5> output;
    std::array<ck::index_t, 5> out_strides;
    std::array<ck::index_t, 5> weight;
    std::array<ck::index_t, 5> wei_strides;
    std::array<ck::index_t, 2> strides;
    std::array<ck::index_t, 2> dilation;
    std::array<ck::index_t, 2> lPadding;
    std::array<ck::index_t, 2> rPadding;
};
*/
struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

using InDataType  = int8_t;
using WeiDataType = int8_t;
using OutDataType = int8_t;

using InLayout     = ck::tensor_layout::convolution::GNHWC;
using WeiLayout    = ck::tensor_layout::convolution::GKYXC;
using OutLayout    = ck::tensor_layout::convolution::GNHWK;
using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
/*
//static constexpr ck::index_t NumDimSpatial = 2;
static constexpr ck::index_t G             = 1;
static constexpr ck::index_t N             = 256;
static constexpr ck::index_t K             = 192;
static constexpr ck::index_t C             = 192;
static constexpr ck::index_t Y             = 3;
static constexpr ck::index_t X             = 3;
static constexpr ck::index_t Hi            = 28;
static constexpr ck::index_t Wi            = 28;
static constexpr ck::index_t Ho            = 28;
static constexpr ck::index_t Wo            = 28;
*/

template <typename DataType>
void PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops::Init(const ProblemDescription& problem)
{
    const auto args      = CKArgsGFwd{problem};
    const auto conv_ptrs = DeviceOpGFwdQuantPtrs<DataType>::GetInstances();
    assert(!conv_ptrs.empty());
    for(int i = 0; i < conv_ptrs.size(); i++)
    {
        auto argument_ptr = conv_ptrs[i]->MakeArgumentPointer(nullptr,
                                                              nullptr,
                                                              {},
                                                              nullptr,
                                                              args.input,
                                                              args.in_strides,
                                                              args.weight,
                                                              args.wei_strides,
                                                              {},
                                                              {},
                                                              args.output,
                                                              args.out_strides,
                                                              args.strides,
                                                              args.dilation,
                                                              args.lPadding,
                                                              args.rPadding,
                                                              {},
                                                              {},
                                                              OElementOp{1.0f, ActivationOp{}});
        if(conv_ptrs[i]->IsSupportedArgument(argument_ptr.get()))
        {
            valid_kernels.push_back(conv_ptrs[i]->GetTypeIdName());
        }
    }
    std::cout<<"valid_kernels.size()"<<valid_kernels.size()<<std::endl;
    assert(!valid_kernels.empty());
    this->index     = 0;
    this->kernel_id = valid_kernels[0];
}

template <typename DataType>
bool PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops::CheckIsSupportCKArgs(
    const ProblemDescription& problem) const
{
    const auto args      = CKArgsGFwd{problem};
    const auto conv_ptrs = DeviceOpGFwdQuantPtrs<DataType>::GetInstances();
    int i                = 0;
    for(; i < conv_ptrs.size(); i++)
    {
        if(conv_ptrs[i]->GetTypeIdName() == this->kernel_id)
        {
            break;
        }
    }
    if(i == valid_kernels.size())
    {
        return false;
    }
    auto argument_ptr = conv_ptrs[i]->MakeArgumentPointer(nullptr,
                                                          nullptr,
                                                          {},
                                                          nullptr,
                                                          args.input,
                                                          args.in_strides,
                                                          args.weight,
                                                          args.wei_strides,
                                                          {},
                                                          {},
                                                          args.output,
                                                          args.out_strides,
                                                          args.strides,
                                                          args.dilation,
                                                          args.lPadding,
                                                          args.rPadding,
                                                          {},
                                                          {},
                                                          OElementOp{1.0f, ActivationOp{}});
    return conv_ptrs[i]->IsSupportedArgument(argument_ptr.get());
}

template <typename DataType>
bool ConvHipImplicitGemmConvFwdLayerQuantXdlops::CheckCKApplicability(const ProblemDescription& problem) const
{
    const auto conv_ptrs = DeviceOpGFwdQuantPtrs<DataType>::GetInstances();
    assert(!conv_ptrs.empty());
    std::cout<<" conv_ptrs.size(): "<<conv_ptrs.size()<<std::endl;
    const auto args = CKArgsGFwd{problem};
    if(!std::all_of(args.strides.begin(), args.strides.end(), [&](auto x) { return x == 1; }))
        return false;
    /*
    std::array<ck::index_t, 5> in_lengths{G, N, C, Hi, Wi};
    std::array<ck::index_t, 5> in_strides{N * Hi * Wi * C, Hi * Wi * C, 1, Wi * C, C};
    std::array<ck::index_t, 5> weight_lengths{G, K, C, Y, X};
    std::array<ck::index_t, 5> weight_strides{K * Y * X * C, Y * X * C, 1, X * C, C};
    std::array<ck::index_t, 5> out_lengths{G, N, C, Ho, Wo};
    std::array<ck::index_t, 5> out_strides{N * Ho * Wo * C, Ho * Wo * C, 1, Wo * C, C};
    std::array<ck::index_t, 2> in_left_pad{1, 1};
    std::array<ck::index_t, 2> in_right_pad{1, 1};
    std::array<ck::index_t, 2> conv_strides{1, 1};
    std::array<ck::index_t, 2> conv_dilations{1, 1};

    SimpleDeviceMem in(sizeof(InDataType) * N * Hi * Wi * C);
    SimpleDeviceMem wei(sizeof(WeiDataType) * K * Y * X * C);
    SimpleDeviceMem out(sizeof(OutDataType) * N * Ho * Wo * K);*/
    for(int i = 0; i < conv_ptrs.size(); i++)
    {
        auto argument_ptr = conv_ptrs[i]->MakeArgumentPointer(nullptr,
                                                          nullptr,
                                                          {},
                                                          nullptr,
                                                          args.input,
                                                          args.in_strides,
                                                          args.weight,
                                                          args.wei_strides,
                                                          {},
                                                          {},
                                                          args.output,
                                                          args.out_strides,
                                                          args.strides,
                                                          args.dilation,
                                                          args.lPadding,
                                                          args.rPadding,
                                                          {},
                                                          {},
                                                          OElementOp{1.0f, ActivationOp{}});
        if(conv_ptrs[i]->IsSupportedArgument(argument_ptr.get()))
            return true;
    }
    std::cout<<"****error****"<<std::endl;
    return false;
}

template <typename DataType>
void ConvHipImplicitGemmConvFwdLayerQuantXdlops::RunCKSolution(
    const Handle& handle,
    const AnyInvokeParams& primitive_parameters,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops& config) const
{
    const auto args      = CKArgsGFwd{problem};
    const auto conv_ptrs = DeviceOpGFwdQuantPtrs<DataType>::GetInstances();
    int i                = 0;
    for(; i < conv_ptrs.size(); i++)
    {
        if(conv_ptrs[i]->GetTypeIdName() == config.kernel_id)
        {
            break;
        }
    }
    assert(i != conv_ptrs.size());
    auto& conv_ptr      = conv_ptrs.at(i);
    auto& data_ctx      = primitive_parameters.CastTo<conv::DataInvokeParams>();
    const auto& tensors = data_ctx.tensors;
    auto argument_ptr   = conv_ptr->MakeArgumentPointer(
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(tensors.in)),
        const_cast<void*>( // NOLINT (cppcoreguidelines-pro-type-const-cast)
            static_cast<const void*>(tensors.w)),
        {},
        static_cast<void*>(tensors.out),
        args.input,
        args.in_strides,
        args.weight,
        args.wei_strides,
        {},
        {},
        args.output,
        args.out_strides,
        args.strides,
        args.dilation,
        args.lPadding,
        args.rPadding,
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{},
        OElementOp{1.0f, ActivationOp{}}); // hard coded value. Need to change
    auto invoker_ptr            = conv_ptr->MakeInvokerPointer();
    const auto enable_profiling = handle.IsProfilingEnabled();

    std::cout<<"*********run ck solution**********"<<std::endl;
    auto& xDesc = tensors.inDesc;
    /*auto& wDesc = tensors.wDesc;
    auto& yDesc = tensors.outDesc;
    std::cout<<"**********************************************G: "<<args.G<<std::endl;
    std::cout<<"**********************************************N: "<<args.N<<std::endl;
    std::cout<<"**********************************************K: "<<args.K<<std::endl;
    std::cout<<"**********************************************C: "<<args.C<<std::endl;*/   
    std::cout<<"****************************problem.conv_problem.GetInLayout(): "<<problem.conv_problem.GetInLayout()<<std::endl;
    std::cout<<"****************************xDesc.GetLayout_str(): "<<xDesc.GetLayout_str()<<std::endl;
    /*std::cout<<"****************************yDesc.GetLayout_t(): "<<yDesc.GetLayout_t()<<std::endl;
    std::cout<<"****************************strides[0]: "<<args.in_strides[0]<<std::endl;
    std::cout<<"****************************strides[1]: "<<args.in_strides[1]<<std::endl;
    std::cout<<"****************************strides[2]: "<<args.in_strides[2]<<std::endl;
    std::cout<<"****************************strides[3]: "<<args.in_strides[3]<<std::endl;
    std::cout<<"****************************strides[4]: "<<args.in_strides[4]<<std::endl;
    std::cout<<"****************************xDesc.GetLengths()[0]: "<<xDesc.GetLengths()[0]<<std::endl;
    std::cout<<"****************************xDesc.GetLengths()[1]: "<<xDesc.GetLengths()[1]<<std::endl;
    std::cout<<"****************************xDesc.GetLengths()[2]: "<<xDesc.GetLengths()[2]<<std::endl;
    std::cout<<"****************************xDesc.GetLengths()[3]: "<<xDesc.GetLengths()[3]<<std::endl;
    std::cout<<"****************************wDesc.GetLengths()[0]: "<<wDesc.GetLengths()[0]<<std::endl;
    std::cout<<"****************************wDesc.GetLengths()[1]: "<<wDesc.GetLengths()[1]<<std::endl;
    std::cout<<"****************************wDesc.GetLengths()[2]: "<<wDesc.GetLengths()[2]<<std::endl;
    std::cout<<"****************************wDesc.GetLengths()[3]: "<<wDesc.GetLengths()[3]<<std::endl;*/

    float elapsed_time =
        invoker_ptr->Run(argument_ptr.get(), {handle.GetStream(), enable_profiling});
    std::cout<<" Run CK successfully!"<<std::endl;
    if(enable_profiling)
    {
        handle.ResetKernelTime();
        handle.AccumKernelTime(elapsed_time);
    }
}
#endif

void PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops::HeuristicInit(const ProblemDescription& problem)
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem;
#else
    std::cout<<"*****problem.conv_problem.GetInDataType(): "<<problem.conv_problem.GetInDataType()<<std::endl;
    switch(problem.conv_problem.GetInDataType())
    {
    //std::cout<<" "<<problem.conv_problem.GetInDataType()<<std::endl;
    case miopenInt8: Init<int8_t>(problem); break;
    case miopenHalf: //Init<ck::half_t>(problem); break;
    case miopenFloat: //Init<float>(problem); break;
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
#endif
}

bool PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        this->HeuristicInit(problem);
        assert(!valid_kernels.empty());
        return true;
    }
    if((index + 1) < valid_kernels.size())
    {
        ++index;
        this->kernel_id = this->valid_kernels[index];
        return true;
    }
    else
        return false;
}

bool PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops::IsValid(const ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem;
    return false;
#else
    switch(problem.conv_problem.GetInDataType())
    {
    case miopenInt8: return CheckIsSupportCKArgs<int8_t>(problem);
    case miopenHalf: //return CheckIsSupportCKArgs<ck::half_t>(problem);
    case miopenFloat: //return CheckIsSupportCKArgs<float>(problem);
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
    return false;
#endif
}

bool PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops::operator==(
    const PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops& other) const
{
    return this->kernel_id == other.kernel_id;
}

PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops
ConvHipImplicitGemmConvFwdLayerQuantXdlops::GetDefaultPerformanceConfig(const ProblemDescription& problem) const
{
    PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops pp;
    pp.HeuristicInit(problem);
    return pp;
}

bool ConvHipImplicitGemmConvFwdLayerQuantXdlops::IsValidPerformanceConfig(
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops& config) const
{
    return config.IsValid(problem);
}

PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops
ConvHipImplicitGemmConvFwdLayerQuantXdlops::Search(const ConvolutionContext& ctx,
                                     const ProblemDescription& problem,
                                     const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvHipImplicitGemmConvFwdLayerQuantXdlops::IsApplicable(const ConvolutionContext& ctx,
                                                const ProblemDescription& problem) const
{
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#else
    if(miopen::IsDisabled(MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_HIP_FWD_LAYER_QUANT_XDLOPS{}))
        return false;
    if(miopen::IsEnabled(MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC{}))
        return false;
    if(problem.conv_problem.GetInDataType() != problem.conv_problem.GetWeightsDataType() ||
       problem.conv_problem.GetWeightsDataType() != problem.conv_problem.GetOutDataType() ||
       problem.conv_problem.GetInDataType() != problem.conv_problem.GetOutDataType())
        return false;
    if(!problem.direction.IsForward())
        return false;
    if(!problem.Is2d())
        return false;
    if(!problem.IsLayoutNHWC())
    {   
        std::cout<<"not NHWC"<<std::endl;
        return false;
    }
    const std::string& arch = ctx.GetStream().GetDeviceName();
    if(!(arch == "gfx908" || arch == "gfx90a"))
        return false;
    switch(problem.conv_problem.GetInDataType())
    {
    case miopenInt8: return CheckCKApplicability<int8_t>(problem);
    case miopenHalf: //return CheckCKApplicability<ck::half_t>(problem);
    case miopenFloat: //return CheckCKApplicability<float>(problem);
    case miopenInt32:
    case miopenInt8x4:
    case miopenBFloat16:
    case miopenDouble: break;
    }
    return false;
#endif
}

ConvSolution ConvHipImplicitGemmConvFwdLayerQuantXdlops::GetSolution(
    const ConvolutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigHipImplicitGemmConvFwdLayerQuantXdlops& config) const
{
    std::ignore = ctx;
#if !MIOPEN_BACKEND_HIP || !MIOPEN_USE_COMPOSABLEKERNEL
    std::ignore = problem;
    std::ignore = config;
    return {};
#else
    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        std::ignore = kernels;
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            switch(problem.conv_problem.GetInDataType())
            {
            case miopenInt8:
                RunCKSolution<int8_t>(handle, primitive_parameters, problem, config);
                break;
            case miopenHalf:
                //RunCKSolution<ck::half_t>(handle, primitive_parameters, problem, config);
                //break;
            case miopenFloat:
                //RunCKSolution<float>(handle, primitive_parameters, problem, config);
                //break;
            case miopenInt32:
            case miopenInt8x4:
            case miopenBFloat16:
            case miopenDouble: break;
            }
        };
    };
    return result;
#endif
}

} // namespace solver
} // namespace miopen
