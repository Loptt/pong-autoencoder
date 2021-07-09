
›
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28êÄ#@êÄ#HêÄ#Xb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
l
maxwell_gcgemm_32x32_nt*28éÿ"@Å»HÇÄXb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
¥
€void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)*28êÿ!@êÿ!Hêÿ!Xb:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputh
Y
'maxwell_scudnn_128x128_relu_small_nn_v1*28äÄ@äÄHäÄbDecoder/conv5_dec/Reluh
¨
˘void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)*28àÄ@àÄHàÄbDecoder/conv4_dec/Reluh
¥
€void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)*28á–@á–Há–Xb:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputh
¨
˘void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)*28Áó@ÁóHÁóbDecoder/conv3_dec/Reluh
ø
Âvoid cudnn::cnn::wgrad_alg1_engine<float, float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)*28Ü¿@Ü¿HÜ¿Xb;gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilterh
ø
Âvoid cudnn::cnn::wgrad_alg1_engine<float, float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)*28á∞@á∞Há∞Xb;gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropFilterh
†
Àvoid fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28Ç–@ÄXHÄ†Xb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
o
maxwell_gcgemm_32x32_nt*28Öà@ÅÿHÅòXb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
Y
'maxwell_scudnn_128x128_relu_small_nn_v1*28Ö@ÖHÖbEncoder/conv5_enc/Reluh
Ù
üvoid fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28∆è@ÅXHÅÄXb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
ó
Ωvoid wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)*28Ö‡
@Ö‡
HÖ‡
Xb;gradient_tape/Encoder/conv1_enc/Conv2D/Conv2DBackpropFilterh
∂
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ö¿
@Ö¿
HÖ¿
bDecoder/conv1_dec/Reluh
ﬁ
™void gemv2T_kernel_val<int, int, float2, float2, float2, float2, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2> >(cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2>, float2, float2)*28Ñ¿
@Ñ¿
HÑ¿
XbDecoder/output/Conv2Dh
≥
Ÿvoid gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28Ö¿	@Ö¿	HÖ¿	Xb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
§
Àvoid fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28≈˜@ÅÄH¡øXb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
n
maxwell_cgemm_32x32_tn*28Ñò@ÑòHÑòXb:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputh
≤
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28‰œ@·◊HÉ¯Xb:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputh
∞
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28É†@É†HÉ†bigradient_tape/Decoder/output/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ø
Âvoid cudnn::cnn::wgrad_alg1_engine<float, float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)*28Éê@ÉêHÉêXb;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterh
å
Ÿvoid gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28„á@„áH„ábDecoder/conv1_dec/Reluh
›
Évoid cudnn::winograd_nonfused::winogradWgradDelta9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28Ñ‡@Ñ‡HÑ‡Xb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
ä
1maxwell_scudnn_128x64_stridedB_splitK_small_nn_v0*28Ñ»@Ñ»HÑ»Xb;gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropFilterh
”
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Éê@ÉêHÉêb,Decoder/upsamp1/resize/ResizeNearestNeighborh
Ì
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Ç‡@Ç‡HÇ‡b>gradient_tape/Decoder/upsamp1/resize/ResizeNearestNeighborGradh
Ø
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28É–@É–HÉ–bhgradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
∂
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)*28É¿@Å∞HÇêXb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
ì
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)*28É®@Å®HÇÄXbDecoder/output/Conv2Dh
≈
évoid fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Çò@ÇòHÇòXbEncoder/conv1_enc/Conv2Dh
o
maxwell_cgemm_32x32_tn*28Éê@ÉêHÉêXb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
˘
üvoid fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28ÉÄ@ÅòHÇËXb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ÉË@ÉËHÉËb%Adam/Adam/update_18/ResourceApplyAdamh
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28ÇË@ÇËHÇËb2gradient_tape/Encoder/maxpool5/MaxPool/MaxPoolGradh
Ë
ívoid transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*)*28Ç–@Ä HÇÿXb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ç»@Ç»HÇ»Xb;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterh
˜
üvoid fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Ç∏@ÅhHÅ†Xb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
≈
ívoid transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*)*28Å∏@Ä8HÅ»XbDecoder/output/Conv2Dh
å
Ÿvoid gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28Ç∞@Ç∞HÇ∞bDecoder/conv2_dec/Reluh
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Çò@ÇòHÇòb2gradient_tape/Encoder/maxpool1/MaxPool/MaxPoolGradh
o
maxwell_sgemm_128x64_nt*28Çà@ÇàHÇàXb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
≤
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28Ç@ÅËHÅàXb:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputh
≥
Ÿvoid gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28Å@ÅHÅXb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
∫
ávoid cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28‚Ô@‚ÔH‚ÔbDecoder/conv1_dec/Reluh
∫
ávoid cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28ÇË@ÇËHÇËbEncoder/conv3_enc/Reluh
·
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ç‡@Ç‡HÇ‡Xb:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputh
≥
Ÿvoid gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28Ç‡@Ç‡HÇ‡Xb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
‡
ávoid cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Å‡@Å‡HÅ‡Xb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
n
maxwell_cgemm_32x32_tn*28·ﬂ@·ﬂH·ﬂXb:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputh
∂
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28·◊@·◊H·◊bDecoder/conv2_dec/Reluh
≥
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28°¿@Å–H†Xb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
‹
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ç¿@Ç¿HÇ¿Xb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
n
maxwell_cgemm_32x32_tn*28Ç∏@Ç∏HÇ∏Xb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
Ê
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Å∏@Å∏HÅ∏b(gradient_tape/Encoder/conv1_enc/ReluGradh
I
maxwell_sgemm_128x64_nn*28Ç∞@Ç∞HÇ∞bEncoder/conv3_enc/Reluh
o
maxwell_sgemm_128x64_nt*28Ç∞@Ç∞HÇ∞Xb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
Å
™void gemv2T_kernel_val<int, int, float2, float2, float2, float2, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2> >(cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2>, float2, float2)*28Ç®@Ç®HÇ®Xb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
é
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ç®@Ç®HÇ®bEncoder/conv1_enc/BiasAddh
n
maxwell_cgemm_32x32_tn*28Å®@Å®HÅ®Xb:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputh
H
maxwell_cgemm_32x32_tn*28Ç†@Ç†HÇ†bEncoder/conv4_enc/Reluh
›
Évoid cudnn::winograd_nonfused::winogradWgradDelta9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28Ç†@Ç†HÇ†Xb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
∏
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ç†@Ç†HÇ†bDecoder/conv5_dec/Reluh
≤
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28Çò@ÅàHÅêXb:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputh
”
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Çò@ÇòHÇòb,Decoder/upsamp2/resize/ResizeNearestNeighborh
å
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28°ó@ÄàH°èbEncoder/conv4_enc/Reluh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Çê@ÇêHÇêblgradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
o
maxwell_cgemm_32x32_tn*28Åà@ÅàHÅàXb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
Ì
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28ÅÄ@ÅÄHÅÄb>gradient_tape/Decoder/upsamp2/resize/ResizeNearestNeighborGradh
H
maxwell_cgemm_32x32_tn*28Å¯@Å¯HÅ¯bEncoder/conv2_enc/Reluh
ﬁ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å¯@Å¯HÅ¯Xb:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputh
›
Évoid cudnn::winograd_nonfused::winogradWgradDelta9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28ÅË@ÅËHÅËXb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
∞
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28°Á@°ÁH°Ábkgradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28·◊@·◊H·◊b2gradient_tape/Encoder/maxpool4/MaxPool/MaxPoolGradh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¡Œ@¡ŒH¡Œb%Adam/Adam/update_20/ResourceApplyAdamh
£
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)*28Å»@Å»HÅ»XbEncoder/conv1_enc/Conv2Dh
ˆ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28°«@°«H°«b$Adam/Adam/update_8/ResourceApplyAdamh
“
üvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 0>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 0>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28Å¿@Å¿HÅ¿bEncoder/conv1_enc/Reluh
˝
 void fft2d_c2r_32x32<float, false, true, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28°∏@°∏H°∏bEncoder/conv2_enc/Reluh
ª
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Å∏@Å∏HÅ∏bEncoder/conv4_enc/Reluh
·
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Å∞@Å∞HÅ∞Xb:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputh
¯
üvoid fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Å∞@Å∞HÅ∞Xb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
“
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ç®@Ç®HÇ®bEncoder/maxpool1/MaxPoolh
•
Àvoid fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28Å®@Å®HÅ®Xb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
è
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Å†@Å†HÅ†bHDecoder/conv1_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
‚
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Åò@ÅòHÅòXb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
∫
ávoid cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Åê@ÅêHÅêbDecoder/conv2_dec/Reluh
‡
ávoid cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Åê@ÅêHÅêXb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
‹
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Åà@ÅàHÅàXb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
›
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Åà@ÅàHÅàXb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
∂
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28ÅÄ@ÅÄHÅÄbEncoder/conv3_enc/Reluh
ï
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)*28Å¯@ÅPHÄ®XbEncoder/conv1_enc/Conv2Dh
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Å@ÅHÅb2gradient_tape/Encoder/maxpool2/MaxPool/MaxPoolGradh
›
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å‡@Å‡HÅ‡Xb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä‡@Ä‡HÄ‡b2gradient_tape/Encoder/maxpool3/MaxPool/MaxPoolGradh
Ê
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Åÿ@ÅÿHÅÿb(gradient_tape/Encoder/conv2_enc/ReluGradh
û
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Åÿ@ÅÿHÅÿXb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
ü
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Åÿ@ÅÿHÅÿbXgradient_tape/Decoder/conv1_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
¬
évoid fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Å–@Å–HÅ–XbDecoder/output/Conv2Dh
Ì
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Å–@Å–HÅ–b>gradient_tape/Decoder/upsamp3/resize/ResizeNearestNeighborGradh
ﬁ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å–@Å–HÅ–Xb:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputh
”
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Ä–@Ä–HÄ–b,Decoder/upsamp3/resize/ResizeNearestNeighborh
∏
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å»@Å»HÅ»bEncoder/conv5_enc/Reluh
Â
évoid fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä¿@Ä¿HÄ¿Xb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
∏
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å∏@Å∏HÅ∏bDecoder/conv4_dec/Reluh
ﬁ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å∏@Å∏HÅ∏Xb:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputh
€
ãvoid cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float)*28Å∞@Å∞HÅ∞b3gradient_tape/Encoder/conv1_enc/BiasAdd/BiasAddGradh
“
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Å∞@Å∞HÅ∞bEncoder/maxpool2/MaxPoolh
°
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Å∞@Å∞HÅ∞Xb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
·
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28ÄØ@ÄØHÄØXb:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputh
“
üvoid fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Å®@Å®HÅ®bEncoder/conv2_enc/Reluh
Ê
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä®@Ä®HÄ®b(gradient_tape/Decoder/conv1_dec/ReluGradh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å†@Å†HÅ†Xb;gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilterh
§
Àvoid fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28Ä†@Ä†HÄ†Xb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä†@Ä†HÄ†Xb;gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropFilterh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Åò@ÅòHÅòblgradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
L
sgemm_32x32x32_NN_vec*28Äò@ÄòHÄòXbEncoder/bottleneck/MatMulh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Äò@ÄòHÄòb%Adam/Adam/update_22/ResourceApplyAdamh
ç
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Äê@ÄêHÄêbHDecoder/conv2_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
X
sgemm_32x32x32_NT_vec*28Äà@ÄàHÄàXb%gradient_tape/Decoder/decoding/MatMulh
Ê
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28ÄÄ@ÄÄHÄÄb(gradient_tape/Encoder/conv3_enc/ReluGradh
ı
üvoid fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28ÄÄ@ÄÄHÄÄXb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
´
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28ÄÄ@ÄÄHÄÄb3gradient_tape/Decoder/conv1_dec/BiasAdd/BiasAddGradh
Í
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Åx@ÅxHÅxb>gradient_tape/Decoder/upsamp4/resize/ResizeNearestNeighborGradh
®
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Äp@ÄpHÄpb3gradient_tape/Decoder/conv5_dec/BiasAdd/BiasAddGradh
®
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Äp@ÄpHÄpb3gradient_tape/Encoder/conv4_enc/BiasAdd/BiasAddGradh
≠
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Äp@ÄpHÄpbkgradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
–
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Äo@ÄoHÄob,Decoder/upsamp4/resize/ResizeNearestNeighborh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Åh@ÅhHÅhb$Adam/Adam/update_6/ResourceApplyAdamh

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Åh@ÅhHÅhb3gradient_tape/Encoder/conv2_enc/BiasAdd/BiasAddGradh
ö
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Åh@ÅhHÅhbXgradient_tape/Decoder/conv2_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
œ
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Äh@ÄhHÄhbEncoder/maxpool3/MaxPoolh
ı
üvoid fft2d_r2c_32x32<float, false, 5u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Å`@Å`HÅ`Xb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
ı
üvoid fft2d_r2c_32x32<float, false, 5u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Å`@Å`HÅ`Xb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
Æ
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ÅX@ÅXHÅXblgradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
œ
üvoid fft2d_r2c_32x32<float, false, 5u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28ÄX@ÄXHÄXbEncoder/conv2_enc/Reluh
®
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28ÄX@ÄXHÄXb3gradient_tape/Decoder/conv3_dec/BiasAdd/BiasAddGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ÄX@ÄXHÄXb%Adam/Adam/update_21/ResourceApplyAdamh
U
sgemm_32x32x32_TN_vec*28ÄP@ÄPHÄPb'gradient_tape/Decoder/decoding/MatMul_1h
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28ÄP@ÄPHÄPb(gradient_tape/Encoder/conv4_enc/ReluGradh
∑
ávoid cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28ÄP@ÄPHÄPbEncoder/conv3_enc/Reluh
›
ávoid cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28ÄP@ÄPHÄPXb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
Í
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28ÄP@ÄPHÄPb>gradient_tape/Decoder/upsamp5/resize/ResizeNearestNeighborGradh
Æ
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ÄP@ÄPHÄPblgradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
w
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28ÅH@ÅHHÅHbEncoder/conv4_enc/Reluh
≠
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ÄH@ÄHHÄHbAddNh
–
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28ÄH@ÄHHÄHb,Decoder/upsamp5/resize/ResizeNearestNeighborh
≠
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ÄH@ÄHHÄHbkgradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
œ
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28¿G@¿GH¿GbEncoder/maxpool4/MaxPoolh
·
ävoid cudnn::winograd_nonfused::winogradWgradOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28ÄG@ÄGHÄGXb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
”
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å@@Å@HÅ@b$gradient_tape/logistic_loss/Select_2h
·
ävoid cudnn::winograd_nonfused::winogradWgradOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28Å@@Å@HÅ@Xb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å@@Å@HÅ@Xb:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputh
Æ
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Å@@Å@HÅ@blgradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ò
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@@Ä@HÄ@b#gradient_tape/logistic_loss/mul/Mulh
ñ
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@@Ä@HÄ@b!gradient_tape/logistic_loss/mul_1h
•
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Ä@@Ä@HÄ@b0gradient_tape/Decoder/output/BiasAdd/BiasAddGradh

°void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long long const*, unsigned long long const*, tensorflow::random::PhiloxRandom, tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long long, tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float>)*28Ä@@Ä@HÄ@b5Encoder/sampling_1/random_normal/RandomStandardNormalh

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@@Ä@HÄ@b3gradient_tape/Encoder/conv3_enc/BiasAdd/BiasAddGradh

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@@Ä@HÄ@b3gradient_tape/Encoder/conv5_enc/BiasAdd/BiasAddGradh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@@Ä@HÄ@bDecoder/conv3_dec/Reluh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@@Ä@HÄ@Xb:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputh
§
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 4, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@@Ä@HÄ@bbgradient_tape/Encoder/maxpool5/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
≠
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@@Ä@HÄ@bkgradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
É
,void tensorflow::SetZero<float>(int, float*)*28‡?@‡?H‡?b>gradient_tape/Decoder/upsamp1/resize/ResizeNearestNeighborGradh
≠
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 4, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28†8@†8H†8bkgradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å8@Å8HÅ8Xb;gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropFilterh
ä
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Å8@Å8HÅ8bHDecoder/conv5_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
—
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä8@Ä8HÄ8b"gradient_tape/logistic_loss/Selecth
∑
ávoid cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä8@Ä8HÄ8bDecoder/conv2_dec/Reluh
›
ávoid cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä8@Ä8HÄ8Xb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
·
ävoid cudnn::winograd_nonfused::winogradWgradOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28Ä8@Ä8HÄ8Xb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
à
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä8@Ä8HÄ8bDecoder/output/BiasAddh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä8@Ä8HÄ8b$Adam/Adam/update_4/ResourceApplyAdamh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä8@Ä8HÄ8bEncoder/conv4_enc/Reluh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä8@Ä8HÄ8Xb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
å
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä8@Ä8HÄ8bJEncoder/maxpool5/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ö
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 4, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä8@Ä8HÄ8bXgradient_tape/Decoder/conv5_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Ü
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä7@Ä7HÄ7blogistic_loss/mulh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28¿6@¿6H¿6b%Adam/Adam/update_24/ResourceApplyAdamh
G
sgemm_32x32x32_NN_vec*28Ä0@Ä0HÄ0XbDecoder/decoding/MatMulh
W
sgemm_32x32x32_NT_vec*28Ä0@Ä0HÄ0Xb'gradient_tape/Encoder/bottleneck/MatMulh
˜
√void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::greater_equal<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::greater_equal<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä0@Ä0HÄ0blogistic_loss/GreaterEqualh
å
·void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä0@Ä0HÄ0blogistic_loss/subh
√
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä0@Ä0HÄ0blogistic_loss/Selecth
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä0@Ä0HÄ0b(gradient_tape/Decoder/conv2_dec/ReluGradh
û
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Ä0@Ä0HÄ0Xb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
ä
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0bHDecoder/conv3_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
‡
´void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*28·/@·/H·/bgradient_tape/DynamicStitchh
ä
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡/@‡/H‡/bHDecoder/conv4_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ˇ
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28¿.@¿.H¿.b3gradient_tape/Encoder/conv2_enc/BiasAdd/BiasAddGradh
∂
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å(@Å(HÅ(bgradient_tape/BroadcastToh
ˇ
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Å(@Å(HÅ(b3gradient_tape/Decoder/conv2_dec/BiasAdd/BiasAddGradh

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Å(@Å(HÅ(b3gradient_tape/Decoder/conv4_dec/BiasAdd/BiasAddGradh
S
sgemm_32x32x32_NT_vec*28Ä(@Ä(HÄ(Xb#gradient_tape/Encoder/z_mean/MatMulh
W
sgemm_32x32x32_TN_vec*28Ä(@Ä(HÄ(b)gradient_tape/Encoder/bottleneck/MatMul_1h
S
sgemm_32x32x32_TN_vec*28Ä(@Ä(HÄ(b%gradient_tape/Encoder/z_mean/MatMul_1h
º
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä(@Ä(HÄ(blogistic_loss/Exph
¬
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log1p_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_log1p_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä(@Ä(HÄ(blogistic_loss/Log1ph
≈
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä(@Ä(HÄ(blogistic_loss/Select_1h
‚
≈void cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)*28Ä(@Ä(HÄ(bSumh
œ
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä(@Ä(HÄ(bEncoder/maxpool5/MaxPoolh
‚
´void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*28Ä(@Ä(HÄ(bgradient_tape/DynamicStitch_1h
‚
´void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*28Ä(@Ä(HÄ(bgradient_tape/DynamicStitch_2h
Ò
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä(@Ä(HÄ(b"Adam/Adam/update/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä(@Ä(HÄ(b%Adam/Adam/update_16/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä(@Ä(HÄ(b$Adam/Adam/update_2/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä(@Ä(HÄ(b%Adam/Adam/update_23/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä(@Ä(HÄ(b%Adam/Adam/update_27/ResourceApplyAdamh

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä(@Ä(HÄ(b3gradient_tape/Decoder/conv2_dec/BiasAdd/BiasAddGradh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä(@Ä(HÄ(Xb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
ö
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä(@Ä(HÄ(bXgradient_tape/Decoder/conv3_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ö
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä(@Ä(HÄ(bXgradient_tape/Decoder/conv4_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
æ
Övoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28¿'@¿'H¿'bgradient_tape/logistic_loss/addh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä'@Ä'HÄ'bEncoder/conv3_enc/Reluh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28† @† H† bEncoder/conv2_enc/Reluh
î
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Å @Å HÅ bgradient_tape/logistic_loss/mulh
”
ïvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorSelectOp<Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å @Å HÅ b$gradient_tape/logistic_loss/Select_3h
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Å @Å HÅ b%Adam/Adam/update_29/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Å @Å HÅ b$Adam/Adam/update_9/ResourceApplyAdamh
Ÿ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å @Å HÅ Xb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
ÿ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å @Å HÅ Xb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
V
sgemm_32x32x32_NT_vec*28Ä @Ä HÄ Xb&gradient_tape/Encoder/z_log_var/MatMulh
V
sgemm_32x32x32_TN_vec*28Ä @Ä HÄ b(gradient_tape/Encoder/z_log_var/MatMul_1h
˙
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ blogistic_lossh
Ÿ
ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_inverse_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä @Ä HÄ b&gradient_tape/logistic_loss/Reciprocalh
¬
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä @Ä HÄ bgradient_tape/truediv_2h
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b(gradient_tape/Decoder/conv3_dec/ReluGradh
∑
ávoid cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä @Ä HÄ bDecoder/conv1_dec/Reluh
ù
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int)*28Ä @Ä HÄ b2gradient_tape/Decoder/decoding/BiasAdd/BiasAddGradh
ü
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int)*28Ä @Ä HÄ b4gradient_tape/Encoder/bottleneck/BiasAdd/BiasAddGradh
û
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int)*28Ä @Ä HÄ b3gradient_tape/Encoder/z_log_var/BiasAdd/BiasAddGradh
õ
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int)*28Ä @Ä HÄ b0gradient_tape/Encoder/z_mean/BiasAdd/BiasAddGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_10/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_11/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_17/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_19/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_25/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_26/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_28/ResourceApplyAdamh
Û
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä @Ä HÄ b3gradient_tape/Encoder/conv5_enc/BiasAdd/BiasAddGradh
ˇ
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä @Ä HÄ b3gradient_tape/Encoder/conv1_enc/BiasAdd/BiasAddGradh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ bDecoder/conv1_dec/Reluh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ bDecoder/conv2_dec/Reluh
∂
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ XbDecoder/output/Conv2Dh
π
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ XbEncoder/conv1_enc/Conv2Dh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28·@·H·Xb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
Û
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28†@†H†b3gradient_tape/Encoder/conv3_enc/BiasAdd/BiasAddGradh
ı
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAdam/Powh
™
çvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28†@†H†bmulh
É
,void tensorflow::SetZero<float>(int, float*)*28†@†H†b>gradient_tape/Decoder/upsamp2/resize/ResizeNearestNeighborGradh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†@†H†Xb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
E
sgemm_32x32x32_NN_vec*28Å@ÅHÅXbEncoder/z_mean/MatMulh
Ø
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å@ÅHÅbgradient_tape/Tileh
è
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Å@ÅHÅbAddN_1h
w
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Å@ÅHÅbEncoder/conv2_enc/Reluh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Å@ÅHÅb%Adam/Adam/update_14/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Å@ÅHÅb%Adam/Adam/update_15/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Å@ÅHÅb$Adam/Adam/update_5/ResourceApplyAdamh
§
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*28Ä@ÄHÄXb;gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropFilterh
H
sgemm_32x32x32_NN_vec*28Ä@ÄHÄXbEncoder/z_log_var/MatMulh
±
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/Tile_1h
Î
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbdiv_no_nan_1h
Î
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbdiv_no_nan_2h
˛
·void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbsubh
ü
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb*gradient_tape/Encoder/sampling_1/mul_1/Mulh
Ü
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbgradient_tape/mulh
”
ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄb&gradient_tape/logistic_loss/zeros_likeh
’
ìvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄb(gradient_tape/logistic_loss/zeros_like_1h
‘
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/logistic_loss/Negh
ÿ
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄb#gradient_tape/logistic_loss/sub/Negh
∆
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄblogistic_loss/Negh
¿
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/truedivh
¬
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/truediv_1h
Å
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbgradient_tape/Cast_1h
Ñ
ﬂvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAdam/Cast_1h
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb(gradient_tape/Decoder/conv4_dec/ReluGradh
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb(gradient_tape/Decoder/conv5_dec/ReluGradh
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb(gradient_tape/Encoder/conv5_enc/ReluGradh
è
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAddN_3h
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_2h
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_4h
∏
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/BroadcastTo_1h
Ñ
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28Ä@ÄHÄbDecoder/conv5_dec/Reluh
Ñ
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28Ä@ÄHÄbEncoder/conv5_enc/Reluh
{
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Ä@ÄHÄXbEncoder/conv1_enc/Conv2Dh
Ö
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Ä@ÄHÄbDecoder/decoding/BiasAddh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb$Adam/Adam/update_1/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_12/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_13/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb$Adam/Adam/update_3/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb$Adam/Adam/update_7/ResourceApplyAdamh
†
Ävoid tensorflow::functor::BlockReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, 256, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28Ä@ÄHÄbMean_1h
†
Ävoid tensorflow::functor::BlockReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, 256, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28Ä@ÄHÄbMean_2h
Û
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@ÄHÄb3gradient_tape/Decoder/conv4_dec/BiasAdd/BiasAddGradh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb;gradient_tape/Encoder/conv1_enc/Conv2D/Conv2DBackpropFilterh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
∫
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡bAssignAddVariableOp_6h
ﬂ
¿void tensorflow::functor::RowReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28‡@‡H‡bSum_1h
°
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*28¡@¡H¡Xb:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputh
Â
¡void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿b
LogicalAndh
Å
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿bgradient_tape/Cast_2h
ù
˚void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿@¿H¿bAdam/addh
É
,void tensorflow::SetZero<float>(int, float*)*28°@°H°b>gradient_tape/Decoder/upsamp4/resize/ResizeNearestNeighborGradh
©
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*28†@†H†Xb;gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropFilterh
¬
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†bAdam/Adam/AssignAddVariableOph
º
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28†@†H†bgradient_tape/Mul_1h
ù
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Å@ÅHÅb(gradient_tape/Encoder/sampling_1/mul/Mulh
á
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Å@ÅHÅbEncoder/bottleneck/BiasAddh
È
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb
div_no_nanh
Ä
·void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbsub_1h
˜
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb
Adam/Pow_1h
ç
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbEncoder/sampling_1/mul_1h
à
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbgradient_tape/Mul_2h
É
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbEncoder/sampling_1/addh
Ú
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbadd_1h
¡
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbEncoder/sampling_1/Exph
Æ
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbExph
Ω
çvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbEncoder/sampling_1/mulh
¢
Övoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbaddh
 
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/sub/Negh
Ã
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/sub_1/Negh
Õ
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄb$gradient_tape/Encoder/sampling_1/mulh
æ
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/mul/Mulh
∑
óvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_square_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_square_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbSquareh
ˇ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbgradient_tape/Casth
ê
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOph
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_1h
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_3h
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_5h
ﬂ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAddN_2h
°
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*28Ä@ÄHÄXb:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputh
x
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Ä@ÄHÄXbDecoder/output/Conv2Dh
Ü
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Ä@ÄHÄbEncoder/z_log_var/BiasAddh
É
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Ä@ÄHÄbEncoder/z_mean/BiasAddh
É
,void tensorflow::SetZero<float>(int, float*)*28Ä@ÄHÄb>gradient_tape/Decoder/upsamp3/resize/ResizeNearestNeighborGradh
É
,void tensorflow::SetZero<float>(int, float*)*28Ä@ÄHÄb>gradient_tape/Decoder/upsamp5/resize/ResizeNearestNeighborGradh