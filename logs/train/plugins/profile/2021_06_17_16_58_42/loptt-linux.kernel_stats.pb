
Ì
ªvoid DSE::regular_fft_clip<1, 2, 128, 16, 32, 1, float, float, float2>(float*, float2*, int, int3, int3, int, int3, int3, int, int, int, int, int, float, float, bool, int, float*, float*)*28¡æ$@¡æ$H¡æ$XbConv2DBackpropInputh

Ωvoid wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)*28Ç»!@Å–HÅ¯XbConv2DBackpropFilterh
∑
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*)*28øµ@ˇHÄÄXbConv2Dh
í
avoid DSE::vector_fft<1, 2, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)*28¢Ø@¢ØH¢ØXbConv2DBackpropInputh
ƒ
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*)*28Å‰@Ä HÅ∞XbConv2DBackpropInputh
≈
ìvoid tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<float, 256, 32, 32, false>(float const*, tensorflow::functor::Dimension<3>, float*)*28˝•@Ä HÄÄXbConv2DBackpropFilterh
u
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0*28Äê@Ä¯HÄÿXbConv2DBackpropInputh
‚
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ÅÃ@ÄHÄËbResourceApplyAdamh
ª
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Ä@Ä`HÄ¿	bResizeNearestNeighborh
«
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28¿á@Ä`H‡øbResizeNearestNeighborGradh
û
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)*28Ä‡@Ä‡HÄ‡XbConv2DBackpropInputh
∑
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28°º@‡HÄ¯	XbConv2DBackpropFilterh
J
&maxwell_scudnn_128x64_relu_small_nn_v1*28¡Á@¿óHÅ–XbConv2Dh
[
*maxwell_scudnn_128x32_stridedB_small_nn_v0*28ÄË@ÄËHÄËXbConv2DBackpropInputh
≈
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28·ˆ@ÅH¿ˇbReluGradh

I
maxwell_sgemm_128x64_nt*28¿’@Ä∏H‡›XbConv2DBackpropFilterh
©
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Åß@ÄHÄ∏XbConv2Dh
v
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Ç’
@ÄHÄàbBiasAddh

…void cudnn::ops::pooling_bw_kernel_max<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä∏
@ÄXHÄ‡bMaxPoolGradh
∂
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿∂
@ÄHˇØXbConv2DBackpropInputh
ü
˙void implicit_convolve_sgemm<float, float, 1024, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)*28¡≠
@¡≠
H¡≠
XbConv2Dh
n
=maxwell_scudnn_winograd_128x128_ldg1_ldg4_relu_tile424n_nt_v1*28·Â	@Å»H‡ùXbConv2DBackpropInputh
∂
ívoid transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*)*28Å¯@Ä8HÅËXbConv2Dh
a
=maxwell_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148m_nt_v1*28‡◊@‡◊H‡◊XbConv2Dh
d
2maxwell_scudnn_128x64_stridedB_splitK_medium_nn_v0*28Ä»@Ä»HÄ»XbConv2DBackpropFilterh
œ
™void gemv2T_kernel_val<int, int, float2, float2, float2, float2, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2> >(cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2>, float2, float2)*28Ä»@Ä»HÄ»XbConv2Dh
í
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)*28Ä®@ÄêHÄòXbConv2DBackpropFilterh
Ñ
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)*28Ä†@ÄêHÄêXbConv2Dh
ø
üvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 0>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 0>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28¡õ@ÄHÄÿbReluh

ƒ
ívoid transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*)*28Ç¯@Ä HÅàXbConv2DBackpropFilterh
n
=maxwell_scudnn_winograd_128x128_ldg1_ldg4_relu_tile442t_nt_v1*28·Ö@·ÖH·ÖXbConv2DBackpropInputh
a
=maxwell_scudnn_winograd_128x128_ldg1_ldg4_relu_tile442t_nt_v1*28Ä@ÄHÄXbConv2Dh
›
™void gemv2T_kernel_val<int, int, float2, float2, float2, float2, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2> >(cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2>, float2, float2)*28ÄÄ@ÄÄHÄÄXbConv2DBackpropFilterh
Æ
}void cudnn::winograd_nonfused::winogradWgradData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28ÅË@ÄxHÅàXbConv2DBackpropFilterh
ò
Âvoid cudnn::cnn::wgrad_alg1_engine<float, float, 128, 6, 7, 3, 3, 5, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)*28ÄË@ÄËHÄËXbConv2DBackpropFilterh
c
1maxwell_scudnn_128x64_stridedB_splitK_small_nn_v0*28Å®@Å®HÅ®XbConv2DBackpropFilterh
c
1maxwell_scudnn_128x64_stridedB_splitK_large_nn_v0*28Äê@ÄêHÄêXbConv2DBackpropFilterh
∑
Övoid gemv2N_kernel<int, int, float, float, float, float, 128, 4, 4, 4, 1, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28‡˛@‡˛H‡˛XbConv2DBackpropInputh
∞
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä@ÄHÄXbConv2DBackpropInputh
¨
âvoid pooling_fw_kernel_max_nhwc<float, float, 0, (cudnnNanPropagation_t)0, false>(PoolingParams, float*, float const*, float, float, int)*28Ä‡@Ä(HÄÄbMaxPoolh
n
=maxwell_scudnn_winograd_128x128_ldg1_ldg4_relu_tile418n_nt_v1*28Äæ@ÄæHÄæXbConv2DBackpropInputh
a
=maxwell_scudnn_winograd_128x128_ldg1_ldg4_relu_tile244t_nt_v1*28Ä∏@Ä∏HÄ∏XbConv2Dh
Ÿ
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28ÅØ@ÅGHÄ∏bBiasAddGradh
Æ
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28Çç@Ä8H¡ŒXbConv2DBackpropInputh
K
'maxwell_scudnn_128x128_relu_small_nn_v1*28Å¯@Å¯HÅ¯XbConv2Dh
K
'maxwell_scudnn_128x32_relu_medium_nn_v1*28ÅË@ÅËHÅËXbConv2Dh
∞
void cudnn::winograd_nonfused::winogradWgradDelta4x4<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28Äœ@ÄgHÄ¿XbConv2DBackpropFilterh
J
&maxwell_scudnn_128x32_relu_small_nn_v1*28Ä∞@Ä∞HÄ∞XbConv2Dh
w
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int)*28°é@†HÅhbBiasAddGradh
…
óvoid DSE::regular_fft_pad<0, 1, 128, 16, 32, 1, float, float, float2>(float2*, float*, int, int3, int3, int, int3, int3, int, int, int, int, int, bool)*28°Ü@†æHÅ»XbConv2DBackpropInputh
∏
Üvoid cudnn::winograd_nonfused::winogradWgradOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28Äˆ@ÄPHÄòXbConv2DBackpropFilterh
ë
avoid DSE::vector_fft<0, 1, 128, 8, 8, 1, float, float, float2>(float2*, float2*, int, int3, int3)*28†∆@ÅWHüÔXbConv2DBackpropInputh
;
maxwell_sgemm_128x64_nn*28Ä∏@Ä∏HÄ∏XbConv2Dh
r
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*28†¥@¿HÄòbMaxPoolGradh
6
sgemm_32x32x32_NN_vec*28†Ø@ÄH†∑bMatMulh
Õ
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Äê@Ä HÄÄbBiasAddGradh
6
sgemm_32x32x32_NT_vec*28Äé@ÄHÄ∞bMatMulh
˘
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28‡ˆ@ÄHÄ/bMulh
£
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28‡ı@‡ıH‡ıXbConv2Dh
°
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28Ä@Ä8HÄ∏XbConv2Dh
≠
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28‡Â@‡5HÄ@XbConv2DBackpropInputh
_
,void tensorflow::SetZero<float>(int, float*)*28Ä∏@ÄHÄpbResizeNearestNeighborGradh
∂
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28¿∂@ÄHÄPbSelecth
≥
évoid fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä∞@Ä∞HÄ∞XbConv2Dh
5
sgemm_32x32x32_TN_vec*28†Ø@ÄHÄPbMatMulh
¡
évoid fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä†@Ä†HÄ†XbConv2DBackpropFilterh
º
ñvoid tensorflow::functor::CleanupSegments<float*, float*, cub::Sum>(float*, float*, int, int, int, cub::Sum, std::iterator_traits<float*>::value_type)*28Äç@ÄHÄ bBiasAddGradh
®
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28ÄÄ@ÄÄHÄÄXbConv2Dh
˛
·void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Äx@ˇHÄbSubh
Ò
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Äx@ÄHÄPbCasth
w
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Äh@ÄhHÄhXbConv2DBackpropFilterh
v
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28·e@ÄHÄXbConv2Dh
•
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä`@Ä`HÄ`XbConv2Dh
 
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 2, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ÄP@ÄPHÄPXbConv2Dh
◊
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 512, 2, 512, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ÄP@ÄPHÄPXbConv2DBackpropInputh
ê
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ˇO@ˇHÄbAssignAddVariableOph
∏
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28°N@ÄH†bNegh
ÿ
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 4, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†F@†HÄ(XbConv2DBackpropFilterh
Á
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿C@†H‡bDivNoNanh
}
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*28Ä@@ÄHÄXbConv2DBackpropFilterh
≤
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Ä@@Ä@HÄ@XbConv2DBackpropInputh
 
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 8, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@@Ä@HÄ@XbConv2Dh
◊
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 8, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@@Ä@HÄ@XbConv2DBackpropInputh
Ç
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*28¿>@ÄH‡XbConv2DBackpropFilterh
ÿ
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 8, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28¿=@¿=H¿=XbConv2DBackpropFilterh
§
Övoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å8@ÄHÅ bAddV2h
 
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 9, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Å8@Å8HÅ8XbConv2Dh
≤
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä8@ˇHÄbRealDivh
ü
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28Ä8@Ä8HÄ8XbConv2Dh
◊
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 9, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä8@Ä8HÄ8XbConv2DBackpropInputh
ÿ
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 9, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä8@Ä8HÄ8XbConv2DBackpropFilterh
Æ
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ˇ7@ˇHÄbExph
œ
°void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long long const*, unsigned long long const*, tensorflow::random::PhiloxRandom, tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long long, tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float>)*28‡7@‡7H‡7bRandomStandardNormalh
™
çvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å0@ÄHÅbMulh
∂
ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä0@ÄHÄb	ZerosLikeh
◊
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0XbConv2DBackpropInputh
 
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 4, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0XbConv2Dh
◊
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 4, 256, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ/@ˇ/Hˇ/XbConv2DBackpropInputh
 
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 256, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡/@‡/H‡/XbConv2Dh
≤
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä/@Ä/HÄ/XbConv2DBackpropInputh

”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Å(@ÅHÄbPowh
Ú
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä(@ÄHÄbAddV2h
≠
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä(@Ä(HÄ(bAddNh
®
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä(@ÄHÄbBroadcastToh
‚
≈void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)*28Ä(@Ä(HÄ(bSumh
°
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28†%@¿H‡bTileh
¨
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å @ÄHÅbMulh
Â
¡void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b
LogicalAndh
i
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Ä @Ä HÄ XbConv2Dh
û
Ävoid tensorflow::functor::BlockReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, 256, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28Ä @ÄHÄbMeanh
º
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ˇ@ˇHˇbSigmoidh
ç
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ü@üHÄbAddNh
∏
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@¿H‡bAssignAddVariableOph
±
ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_sqrt_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_sqrt_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å@ÅHÅbSqrth
r
Dcask_cudnn::computeBOffsetsKernel(cask_cudnn::ComputeBOffsetsParams)*28Ä@ÄHÄXbConv2DBackpropInputh
Ω
ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄb
Reciprocalh
¥
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log1p_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log1p_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbLog1ph
Ç
Tvoid cask_cudnn::computeOffsetsKernel<true, false>(cask_cudnn::ComputeOffsetsParams)*28Ä@ÄHÄXbConv2DBackpropInputh
È
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::greater_equal<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::greater_equal<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†bGreaterEqualh
˝
ﬂvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Å@ÅHÅbCasth
˛
›void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_quotient_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbRealDivh

”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAddh
›	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAddNh
ö
˚void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAddV2h
›
¿void tensorflow::functor::RowReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28Ä@ÄHÄbSumh
∑
óvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_square_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_square_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28·@·H·bSquareh
§
ávoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbAddh