
Ü
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0*28é∞’@é∞’Hé∞’b#PerceptualNetwork/block1_conv2/Reluh
à
Dmaxwell_scudnn_winograd_128x128_ldg1_ldg4_mobile_relu_tile148t_nt_v0*28ã¯õ@ã¯õHã¯õb%PerceptualNetwork/block1_conv2/Relu_1h
ì
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28È∑Ü@È∑ÜHÈ∑Üb?gradient_tape/PerceptualNetwork/block1_pool/MaxPool/MaxPoolGradh
‡
ávoid cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Ëüz@ËüzHËüzXb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
¢
=maxwell_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148n_nt_v1*28àÿx@àÿxHàÿxXbGgradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputh
X
maxwell_sgemm_128x64_nn*28àu@àuHàub%PerceptualNetwork/block3_conv2/Relu_1h
√
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Üòa@ÜòaHÜòab#PerceptualNetwork/block3_conv2/Reluh
V
maxwell_sgemm_128x64_nn*28Ü–[@Ü–[HÜ–[b#PerceptualNetwork/block3_conv3/Reluh
Y
maxwell_sgemm_128x128_nn*28Ñ¿B@Ñ¿BHÑ¿Bb%PerceptualNetwork/block2_conv2/Relu_1h
}
maxwell_sgemm_128x128_nt*28Ñ¯>@Ñ¯>HÑ¯>XbGgradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputh
W
maxwell_sgemm_128x128_nn*28Ñ†:@Ñ†:HÑ†:b#PerceptualNetwork/block2_conv2/Reluh
ä
1maxwell_scudnn_128x64_stridedB_splitK_small_nn_v0*28ÉË5@ÉË5HÉË5Xb;gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropFilterh
X
maxwell_sgemm_128x64_nn*28ÑË3@ÑË3HÑË3b%PerceptualNetwork/block3_conv4/Relu_1h
V
maxwell_sgemm_128x64_nn*28ÑÄ3@ÑÄ3HÑÄ3b#PerceptualNetwork/block3_conv2/Reluh
V
maxwell_sgemm_128x64_nn*28ÉÄ3@ÉÄ3HÉÄ3b#PerceptualNetwork/block3_conv4/Reluh
X
maxwell_sgemm_128x64_nn*28É¯2@É¯2HÉ¯2b%PerceptualNetwork/block3_conv3/Relu_1h
|
maxwell_sgemm_128x64_nt*28Éÿ2@Éÿ2HÉÿ2XbGgradient_tape/PerceptualNetwork/block3_conv3/Conv2D/Conv2DBackpropInputh
|
maxwell_sgemm_128x64_nt*28Éê2@Éê2HÉê2XbGgradient_tape/PerceptualNetwork/block3_conv2/Conv2D/Conv2DBackpropInputh
|
maxwell_sgemm_128x64_nt*28Ñà2@Ñà2HÑà2XbGgradient_tape/PerceptualNetwork/block3_conv4/Conv2D/Conv2DBackpropInputh
«
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ç»#@Ç»#HÇ»#b%PerceptualNetwork/block3_conv4/Relu_1h
l
maxwell_gcgemm_32x32_nt*28É¿"@†¿HÄÄXb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
¡
€void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)*28ÇÄ @ÇÄ HÇÄ XbGgradient_tape/PerceptualNetwork/block1_conv1/Conv2D/Conv2DBackpropInputh
W
maxwell_sgemm_128x128_nn*28Ç‡@Ç‡HÇ‡b#PerceptualNetwork/block2_conv1/Reluh
Y
maxwell_sgemm_128x128_nn*28Ç–@Ç–HÇ–b%PerceptualNetwork/block2_conv1/Relu_1h
V
maxwell_sgemm_128x64_nn*28Ç∏@Ç∏HÇ∏b#PerceptualNetwork/block3_conv1/Reluh
X
maxwell_sgemm_128x64_nn*28Ç∞@Ç∞HÇ∞b%PerceptualNetwork/block3_conv1/Relu_1h
ı
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28ÇÄ@ÇÄHÇÄb7gradient_tape/PerceptualNetwork/block1_conv2/ReluGrad_1h
ı
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ç∏@Ç∏HÇ∏b7gradient_tape/PerceptualNetwork/block1_conv1/ReluGrad_1h
|
maxwell_sgemm_128x64_nt*28Åÿ@ÅÿHÅÿXbGgradient_tape/PerceptualNetwork/block3_conv1/Conv2D/Conv2DBackpropInputh
|
maxwell_sgemm_128x64_nt*28Ç¯@Ç¯HÇ¯XbGgradient_tape/PerceptualNetwork/block2_conv1/Conv2D/Conv2DBackpropInputh
Y
'maxwell_scudnn_128x128_relu_small_nn_v1*28Å¯@Å¯HÅ¯bDecoder/conv5_dec/Reluh
X
&maxwell_scudnn_128x32_relu_small_nn_v1*28¢Ë@¢ËH¢ËbDecoder/conv1_dec/Reluh
¨
˘void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)*28ÅÄ@ÅÄHÅÄbDecoder/conv4_dec/Reluh
ø
Âvoid cudnn::cnn::wgrad_alg1_engine<float, float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)*28Å–@Å–HÅ–Xb;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterh
≈
ívoid transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*)*28Ä–@Ä8HÄÿXbDecoder/output/Conv2Dh
g
&maxwell_scudnn_128x32_relu_small_nn_v1*28ÅÄ@ÅÄHÅÄb%PerceptualNetwork/block1_conv1/Relu_1h
e
&maxwell_scudnn_128x32_relu_small_nn_v1*28°ˇ@°ˇH°ˇb#PerceptualNetwork/block1_conv1/Reluh
æ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28¡«@¡«H¡«b#PerceptualNetwork/block2_conv2/Reluh
‰
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å¿@Å¿HÅ¿XbGgradient_tape/PerceptualNetwork/block2_conv1/Conv2D/Conv2DBackpropInputh
‰
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Åø@ÅøHÅøXbGgradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputh
¿
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ç∏@Ç∏HÇ∏b%PerceptualNetwork/block2_conv2/Relu_1h
¥
€void cudnn::detail::dgrad_engine<float, 512, 6, 5, 3, 3, 3, false>(int, int, int, float const*, int, float const*, int, float*, kernel_grad_params, unsigned long long, int, unsigned long long, int, float, int, int, int)*28Çÿ@ÇÿHÇÿXb:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputh
√
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Å®@Å®HÅ®b#PerceptualNetwork/block2_conv1/Reluh
≈
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Å®@Å®HÅ®b%PerceptualNetwork/block2_conv1/Relu_1h
√
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Å†@Å†HÅ†b#PerceptualNetwork/block2_conv2/Reluh
≈
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Å†@Å†HÅ†b%PerceptualNetwork/block2_conv2/Relu_1h
È
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Å†@Å†HÅ†XbGgradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputh
n
maxwell_cgemm_32x32_tn*28Å‡@Å‡HÅ‡Xb:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputh
ø
Âvoid cudnn::cnn::wgrad_alg1_engine<float, float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)*28Å∏@Å∏HÅ∏Xb;gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilterh
ﬂ
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ç@ÇHÇb%PerceptualNetwork/block1_pool/MaxPoolh
ı
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Å‡@Å‡HÅ‡b7gradient_tape/PerceptualNetwork/block2_conv1/ReluGrad_1h
·
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28¡Œ@¡ŒH¡Œb'PerceptualNetwork/block1_pool/MaxPool_1h
ı
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Å»@Å»HÅ»b7gradient_tape/PerceptualNetwork/block2_conv2/ReluGrad_1h
≤
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28Å»@ÄËHÅ‡
Xb:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputh
ø
Âvoid cudnn::cnn::wgrad_alg1_engine<float, float, 128, 5, 5, 3, 3, 3, false, false>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, float, int, int, int*, int*, int, int)*28Å∞@Å∞HÅ∞Xb;gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropFilterh
ê
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä»@Ä»HÄ»b?gradient_tape/PerceptualNetwork/block2_pool/MaxPool/MaxPoolGradh
†
Àvoid fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28Åà@ÄXHÄòXb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
Y
'maxwell_scudnn_128x128_relu_small_nn_v1*28ÄÄ@ÄÄHÄÄbEncoder/conv5_enc/Reluh
ê
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Å–@Å–HÅ–b?gradient_tape/PerceptualNetwork/block3_pool/MaxPool/MaxPoolGradh
æ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å¯
@Å¯
HÅ¯
b#PerceptualNetwork/block3_conv4/Reluh
¿
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å
@Å
HÅ
b%PerceptualNetwork/block3_conv4/Relu_1h
‰
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28ÅË
@ÅË
HÅË
XbGgradient_tape/PerceptualNetwork/block3_conv3/Conv2D/Conv2DBackpropInputh
æ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28ÄË
@ÄË
HÄË
b#PerceptualNetwork/block3_conv3/Reluh
¿
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å‡
@Å‡
HÅ‡
b%PerceptualNetwork/block3_conv3/Relu_1h
‰
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Åÿ
@Åÿ
HÅÿ
XbGgradient_tape/PerceptualNetwork/block3_conv2/Conv2D/Conv2DBackpropInputh
‰
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Åÿ
@Åÿ
HÅÿ
XbGgradient_tape/PerceptualNetwork/block3_conv4/Conv2D/Conv2DBackpropInputh
ó
Ωvoid wgrad_alg0_engine<float, 128, 5, 5, 3, 3, 3, false, 512>(int, int, int, float const*, int, float*, float const*, kernel_grad_params, unsigned long long, int, float, int, int, int, int)*28Åÿ
@Åÿ
HÅÿ
Xb;gradient_tape/Encoder/conv1_enc/Conv2D/Conv2DBackpropFilterh
‰
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28·◊
@·◊
H·◊
XbGgradient_tape/PerceptualNetwork/block3_conv1/Conv2D/Conv2DBackpropInputh
›
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å–
@Å–
HÅ–
Xb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
æ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä–
@Ä–
HÄ–
b#PerceptualNetwork/block3_conv2/Reluh
¿
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä–
@Ä–
HÄ–
b%PerceptualNetwork/block3_conv2/Relu_1h
ﬁ
™void gemv2T_kernel_val<int, int, float2, float2, float2, float2, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2> >(cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2>, float2, float2)*28Ä∏
@Ä∏
HÄ∏
XbDecoder/output/Conv2Dh
≈
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28·∂
@·∂
H·∂
b%PerceptualNetwork/block3_conv3/Relu_1h
Ù
üvoid fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28¡Ø
@Ä@HÄÄXb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
≈
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Äà
@Äà
HÄà
b%PerceptualNetwork/block3_conv2/Relu_1h
¨
˘void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)*28Äà
@Äà
HÄà
bEncoder/conv4_enc/Reluh
√
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28°Ä
@°Ä
H°Ä
b#PerceptualNetwork/block3_conv4/Reluh
≈
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28ÅÄ
@ÅÄ
HÅÄ
b%PerceptualNetwork/block3_conv1/Relu_1h
√
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28ÅÄ
@ÅÄ
HÅÄ
b#PerceptualNetwork/block3_conv2/Reluh
È
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28ÅÄ
@ÅÄ
HÅÄ
XbGgradient_tape/PerceptualNetwork/block3_conv4/Conv2D/Conv2DBackpropInputh
¨
˘void implicit_convolve_sgemm<float, float, 128, 5, 5, 3, 3, 3, 1, false, false, true>(int, int, int, float const*, int, float*, float const*, kernel_conv_params, unsigned long long, int, float, float, int, float const*, float const*, bool, int, int)*28ÅÄ
@ÅÄ
HÅÄ
bEncoder/conv3_enc/Reluh
√
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28ÄÄ
@ÄÄ
HÄÄ
b#PerceptualNetwork/block3_conv3/Reluh
≈
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Å¯	@Å¯	HÅ¯	b%PerceptualNetwork/block3_conv4/Relu_1h
√
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Ä¯	@Ä¯	HÄ¯	b#PerceptualNetwork/block3_conv1/Reluh
È
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Å	@Å	HÅ	XbGgradient_tape/PerceptualNetwork/block3_conv2/Conv2D/Conv2DBackpropInputh
È
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Ä	@Ä	HÄ	XbGgradient_tape/PerceptualNetwork/block3_conv3/Conv2D/Conv2DBackpropInputh
≥
Ÿvoid gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28Å∞	@Å∞	HÅ∞	Xb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
æ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å¯@Å¯HÅ¯b#PerceptualNetwork/block2_conv1/Reluh
¿
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å¯@Å¯HÅ¯b%PerceptualNetwork/block2_conv1/Relu_1h
È
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Ä¿@Ä¿HÄ¿XbGgradient_tape/PerceptualNetwork/block2_conv1/Conv2D/Conv2DBackpropInputh
o
maxwell_sgemm_128x64_nt*28Äò@ÄòHÄòXb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
n
maxwell_cgemm_32x32_tn*28Åê@ÅêHÅêXb:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputh
ˇ
¿void tensorflow::functor::RowReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28ÅÄ@ÅÄHÅÄb"gradient_tape/grayscale_to_rgb/Sumh
·
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28·œ@·œH·œb'PerceptualNetwork/block2_pool/MaxPool_1h
≤
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28†œ@†œHÄÄXb:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputh
∞
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Å∞@Å∞HÅ∞bigradient_tape/Decoder/output/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ﬂ
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Äê@ÄêHÄêb%PerceptualNetwork/block2_pool/MaxPoolh
›
Évoid cudnn::winograd_nonfused::winogradWgradDelta9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28Å‡@Å‡HÅ‡Xb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
ı
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Åœ@ÅœHÅœb7gradient_tape/PerceptualNetwork/block3_conv3/ReluGrad_1h
ı
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28‡«@‡«H‡«b7gradient_tape/PerceptualNetwork/block3_conv2/ReluGrad_1h
ı
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28°¿@°¿H°¿b7gradient_tape/PerceptualNetwork/block3_conv1/ReluGrad_1h
ı
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Å¿@Å¿HÅ¿b7gradient_tape/PerceptualNetwork/block3_conv4/ReluGrad_1h
”
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Äò@ÄòHÄòb,Decoder/upsamp1/resize/ResizeNearestNeighborh
Ì
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28ÄË@ÄËHÄËb>gradient_tape/Decoder/upsamp1/resize/ResizeNearestNeighborGradh
≈
évoid fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä–@Ä–HÄ–XbEncoder/conv1_enc/Conv2Dh
Ø
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä–@Ä–HÄ–bhgradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
‹
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å»@Å»HÅ»Xb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
æ
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Å¿@Å¿HÅ¿b#PerceptualNetwork/block3_conv1/Reluh
¿
void cudnn::winograd_nonfused::winogradForwardData4x4<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä¿@Ä¿HÄ¿b%PerceptualNetwork/block3_conv1/Relu_1h
∂
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)*28Ä∞@Ä∞HÄÄXb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
È
Évoid cudnn::winograd_nonfused::winogradForwardOutput4x4<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Å®@Å®HÅ®XbGgradient_tape/PerceptualNetwork/block3_conv1/Conv2D/Conv2DBackpropInputh
ì
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)*28Ä†@Ä®HÄ¯XbDecoder/output/Conv2Dh
≈
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Åê@ÅêHÅêb%PerceptualNetwork/block3_conv2/Relu_1h
o
maxwell_cgemm_32x32_tn*28Äê@ÄêHÄêXb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
√
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Äê@ÄêHÄêb#PerceptualNetwork/block3_conv4/Reluh
È
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Äê@ÄêHÄêXbGgradient_tape/PerceptualNetwork/block3_conv3/Conv2D/Conv2DBackpropInputh
È
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28†à@†àH†àXbGgradient_tape/PerceptualNetwork/block3_conv2/Conv2D/Conv2DBackpropInputh
√
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Äà@ÄàHÄàb#PerceptualNetwork/block3_conv3/Reluh
˘
üvoid fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Äà@ÄòHÄXb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
≈
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28ÄÄ@ÄÄHÄÄb%PerceptualNetwork/block3_conv4/Relu_1h
È
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28ÄÄ@ÄÄHÄÄXbGgradient_tape/PerceptualNetwork/block3_conv4/Conv2D/Conv2DBackpropInputh
≈
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Å¯@Å¯HÅ¯b%PerceptualNetwork/block3_conv3/Relu_1h
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28ÄË@ÄËHÄËb2gradient_tape/Encoder/maxpool5/MaxPool/MaxPoolGradh
«
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å‡@Å‡HÅ‡b%PerceptualNetwork/block3_conv3/Relu_1h
Î
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä‡@Ä‡HÄ‡XbGgradient_tape/PerceptualNetwork/block3_conv4/Conv2D/Conv2DBackpropInputh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Äﬂ@ÄﬂHÄﬂb%Adam/Adam/update_18/ResourceApplyAdamh
≈
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Åÿ@ÅÿHÅÿb#PerceptualNetwork/block3_conv2/Reluh
≈
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Åÿ@ÅÿHÅÿb#PerceptualNetwork/block3_conv4/Reluh
«
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äÿ@ÄÿHÄÿb%PerceptualNetwork/block3_conv2/Relu_1h
≈
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äÿ@ÄÿHÄÿb#PerceptualNetwork/block3_conv3/Reluh
Î
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28°–@°–H°–XbGgradient_tape/PerceptualNetwork/block3_conv3/Conv2D/Conv2DBackpropInputh
Ë
ívoid transpose_readWrite_alignment_kernel<float2, float2, 1, false, 6, 4, 4>(cublasTransposeParams<float2>, float2 const*, float2*, float2 const*)*28Ä–@Ä HÄ–Xb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
Î
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡œ@‡œH‡œXbGgradient_tape/PerceptualNetwork/block3_conv2/Conv2D/Conv2DBackpropInputh
å
Ÿvoid gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28Å®@Å®HÅ®bDecoder/conv2_dec/Reluh
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä®@Ä®HÄ®b2gradient_tape/Encoder/maxpool1/MaxPool/MaxPoolGradh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä†@Ä†HÄ†Xb;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterh
‡
ávoid cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28‡á@‡áH‡áXb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
≤
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28·Ü@ÅËH‡ûXb:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputh
o
maxwell_sgemm_128x64_nt*28°Ä@°ÄH°ÄXb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
å
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28ÄÄ@ÄHÄêbDecoder/conv3_dec/Reluh
≥
Ÿvoid gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28Ä¯@Ä¯HÄ¯Xb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
·
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä@ÄHÄXb:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputh
·
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28ÅË@ÅËHÅËXb:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputh
≥
Ÿvoid gemmSN_NN_kernel<float, 128, 2, 4, 8, 4, 4, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float> >(cublasGemmSmallNParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)*28Ä‡@Ä‡HÄ‡Xb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
n
maxwell_cgemm_32x32_tn*28Åÿ@ÅÿHÅÿXb:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputh
H
maxwell_cgemm_32x32_tn*28Äÿ@ÄÿHÄÿbDecoder/conv3_dec/Reluh
∂
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Äÿ@ÄÿHÄÿbDecoder/conv2_dec/Reluh
ﬂ
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä»@Ä»HÄ»b%PerceptualNetwork/block3_pool/MaxPoolh
Ê
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Å¿@Å¿HÅ¿b(gradient_tape/Encoder/conv1_enc/ReluGradh
·
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Å¿@Å¿HÅ¿b'PerceptualNetwork/block3_pool/MaxPool_1h
‹
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Ä¿@Ä¿HÄ¿Xb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
n
maxwell_cgemm_32x32_tn*28Ä∏@Ä∏HÄ∏Xb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
o
maxwell_sgemm_128x64_nt*28Ä∏@Ä∏HÄ∏Xb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
é
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28Ä∞@Ä∞HÄ∞bEncoder/conv1_enc/BiasAddh
›
Évoid cudnn::winograd_nonfused::winogradWgradDelta9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28Å®@Å®HÅ®Xb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
o
maxwell_cgemm_32x32_tn*28Ä†@Ä†HÄ†Xb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
n
maxwell_cgemm_32x32_tn*28Äò@ÄòHÄòXb:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputh
Å
™void gemv2T_kernel_val<int, int, float2, float2, float2, float2, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2> >(cublasGemvParams<cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2 const>, cublasGemvTensorStridedBatched<float2>, float2>, float2, float2)*28Äò@ÄòHÄòXb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
”
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Äò@ÄòHÄòb,Decoder/upsamp2/resize/ResizeNearestNeighborh
≤
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28ˇè@ÄàHˇáXb:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Äà@ÄàHÄàblgradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Ì
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28ÄÄ@ÄÄHÄÄb>gradient_tape/Decoder/upsamp2/resize/ResizeNearestNeighborGradh
ﬁ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ÄÄ@ÄÄHÄÄXb:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputh
è
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ÄÄ@ÄÄHÄÄbHDecoder/conv1_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
∞
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ÄÄ@ÄÄHÄÄbkgradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
≥
Zvoid fft2d_r2c_16x16<float>(float2*, float const*, int, int, int, int, int, int, int, int)*28Ä¯@Ä†HÄÿXb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
∏
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä¯@Ä¯HÄ¯bDecoder/conv5_dec/Reluh
H
maxwell_cgemm_32x32_tn*28¿˜@¿˜H¿˜bEncoder/conv2_enc/Reluh
›
Évoid cudnn::winograd_nonfused::winogradWgradDelta9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDeltaParams<float, float>)*28Ä@ÄHÄXb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
£
mvoid pointwise_mult_and_sum_complex<float2, 8, 4>(float2*, float2*, float2*, int, int, int, int, int, float2)*28Å–@Å–HÅ–XbEncoder/conv1_enc/Conv2Dh
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä–@Ä–HÄ–b2gradient_tape/Encoder/maxpool4/MaxPool/MaxPoolGradh
È
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28°œ@°œH°œXbGgradient_tape/PerceptualNetwork/block3_conv1/Conv2D/Conv2DBackpropInputh
≈
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Å»@Å»HÅ»b%PerceptualNetwork/block3_conv1/Relu_1h
√
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä»@Ä»HÄ»b#PerceptualNetwork/block3_conv1/Reluh
ˆ
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä»@Ä»HÄ»b$Adam/Adam/update_8/ResourceApplyAdamh
˝
 void fft2d_c2r_32x32<float, false, true, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28Ä¿@Ä¿HÄ¿bEncoder/conv2_enc/Reluh
“
üvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 0>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 0>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*28¿ø@¿øH¿øbEncoder/conv1_enc/Reluh
·
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä∏@Ä∏HÄ∏Xb:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputh
¯
üvoid fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Ä∏@Ä∏HÄ∏Xb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä∏@Ä∏HÄ∏b%Adam/Adam/update_20/ResourceApplyAdamh
‚
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä∞@Ä∞HÄ∞Xb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
“
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Ä†@Ä†HÄ†bEncoder/maxpool1/MaxPoolh
•
Àvoid fft2d_c2r_32x32<float, false, false, 1u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28Ä†@Ä†HÄ†Xb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
¿
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 2, 1024, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä†@Ä†HÄ†bygradient_tape/PerceptualNetwork/block1_conv1/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Î
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Åò@ÅòHÅòXbGgradient_tape/PerceptualNetwork/block3_conv1/Conv2D/Conv2DBackpropInputh
û
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Åò@ÅòHÅòbWPerceptualNetwork/block1_conv1/Conv2D_1-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
∫
ávoid cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Äò@ÄòHÄòbDecoder/conv2_dec/Reluh
‡
ávoid cudnn::winograd_nonfused::winogradForwardOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradOutputParams<float, float>)*28Äê@ÄêHÄêXb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
≈
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äê@ÄêHÄêb#PerceptualNetwork/block3_conv1/Reluh
«
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äê@ÄêHÄêb%PerceptualNetwork/block3_conv1/Relu_1h
‹
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Äà@ÄàHÄàXb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
›
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Äà@ÄàHÄàXb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
ï
`void fft2d_r2c_64x64<float, true>(float2*, float const*, int, int, int, int, int, int, int, int)*28†Ä@ÄPH†∞XbEncoder/conv1_enc/Conv2Dh
°
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28ÄÄ@ÄÄHÄÄXb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
∞
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å¯@Å¯HÅ¯bgrayscale_to_rgbh
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Å@ÅHÅb2gradient_tape/Encoder/maxpool2/MaxPool/MaxPoolGradh
ü
™void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 1024, 1024, 2, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Å‡@Å‡HÅ‡bXgradient_tape/Decoder/conv1_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Ê
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä‡@Ä‡HÄ‡b(gradient_tape/Encoder/conv2_enc/ReluGradh
É
¥void cudnn::pooling_bw_kernel_max_nchw_fully_packed_small<float, float, 2, (cudnnNanPropagation_t)0>(cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Äÿ@ÄÿHÄÿb2gradient_tape/Encoder/maxpool3/MaxPool/MaxPoolGradh
›
Évoid cudnn::winograd_nonfused::winogradForwardData9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradDataParams<float, float>)*28Äÿ@ÄÿHÄÿXb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
˜
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Äÿ@ÄÿHÄÿb%Adam/Adam/update_22/ResourceApplyAdamh
ù
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Äÿ@ÄÿHÄÿbXgradient_tape/Decoder/conv2_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
¬
évoid fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28¿◊@¿◊H¿◊XbDecoder/output/Conv2Dh
”
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Ä◊@Ä◊HÄ◊b,Decoder/upsamp3/resize/ResizeNearestNeighborh
Ì
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Ä–@Ä–HÄ–b>gradient_tape/Decoder/upsamp3/resize/ResizeNearestNeighborGradh
∏
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å»@Å»HÅ»bEncoder/conv5_enc/Reluh
è
·void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä»@Ä»HÄ»bgradient_tape/subh
Â
évoid fft2d_c2r_64x64<float, false, true>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä»@Ä»HÄ»Xb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
û
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Ä»@Ä»HÄ»Xb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
ú
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä»@Ä»HÄ»bWPerceptualNetwork/block3_pool/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
ﬁ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä¿@Ä¿HÄ¿Xb:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputh
û
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä¿@Ä¿HÄ¿bYPerceptualNetwork/block3_pool/MaxPool_1-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
¥
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä¿@Ä¿HÄ¿bogradient_tape/PerceptualNetwork/block3_pool/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ª
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä∏@Ä∏HÄ∏bDecoder/conv3_dec/Reluh
·
àvoid fft2d_c2r_16x16<float, false>(float*, float2*, int, int, int, int, int, int, int, int, int, int, float, float, int, float*, float*)*28Ä∏@Ä∏HÄ∏Xb:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputh
ﬁ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä∏@Ä∏HÄ∏Xb:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputh
∏
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ˇ∑@ˇ∑Hˇ∑bDecoder/conv4_dec/Reluh
∫
ﬂvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sigmoid_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sigmoid_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä∞@Ä∞HÄ∞b>gradient_tape/PerceptualNetwork/activation/Sigmoid/SigmoidGradh
€
ãvoid cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, cub::Sum>::Policy600, float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float>(float const*, float*, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, cub::Sum, float)*28Ä∞@Ä∞HÄ∞b3gradient_tape/Encoder/conv1_enc/BiasAdd/BiasAddGradh
≈
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28¿Ø@¿ØH¿Øb%PerceptualNetwork/block2_conv2/Relu_1h
√
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28ÄØ@ÄØHÄØb#PerceptualNetwork/block2_conv2/Reluh
“
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28°®@°®H°®bEncoder/maxpool2/MaxPoolh
ë
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_squared_difference_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä®@Ä®HÄ®bSquaredDifferenceh
Ê
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä®@Ä®HÄ®b(gradient_tape/Decoder/conv1_dec/ReluGradh
È
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä®@Ä®HÄ®XbGgradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputh
“
üvoid fft2d_r2c_32x32<float, false, 0u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Ä®@Ä®HÄ®bEncoder/conv2_enc/Reluh
§
Àvoid fft2d_c2r_32x32<float, false, false, 0u, false, false>(float*, float2 const*, int, int, int, int, int, int, int, int, int, float, float, cudnn::reduced_divisor, bool, float*, float*, int2, int, int)*28Ä†@Ä†HÄ†Xb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28†ò@†òH†òXb;gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropFilterh
L
sgemm_32x32x32_NN_vec*28Äò@ÄòHÄòXbEncoder/bottleneck/MatMulh
ã
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Äò@ÄòHÄòbgradient_tape/mul_1h
ﬂ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äò@ÄòHÄòXb;gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilterh
±
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Åê@ÅêHÅêblgradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
X
sgemm_32x32x32_NT_vec*28Äê@ÄêHÄêXb%gradient_tape/Decoder/decoding/MatMulh
Î
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äê@ÄêHÄêXbGgradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputh
”
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28Äà@ÄàHÄàb,Decoder/upsamp4/resize/ResizeNearestNeighborh
≈
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äà@ÄàHÄàb#PerceptualNetwork/block2_conv2/Reluh
«
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äà@ÄàHÄàb%PerceptualNetwork/block2_conv2/Relu_1h
´
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28ˇá@ˇáHˇáb3gradient_tape/Decoder/conv1_dec/BiasAdd/BiasAddGradh
ª
çvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28‡á@‡áH‡ábgradient_tape/Mulh
Ê
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28ÄÄ@ÄÄHÄÄb(gradient_tape/Encoder/conv3_enc/ReluGradh
ı
üvoid fft2d_r2c_32x32<float, false, 1u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28ÄÄ@ÄÄHÄÄXb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
Û
£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28ÄÄ@ÄÄHÄÄb3gradient_tape/Encoder/conv2_enc/BiasAdd/BiasAddGradh
®
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Äx@ÄxHÄxb3gradient_tape/Encoder/conv4_enc/BiasAdd/BiasAddGradh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Äx@ÄxHÄxbDecoder/conv3_dec/Reluh
ä
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28‡w@‡wH‡wbHDecoder/conv2_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Í
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28†p@†pH†pb>gradient_tape/Decoder/upsamp4/resize/ResizeNearestNeighborGradh
≠
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Äp@ÄpHÄpbkgradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Äh@ÄhHÄhb$Adam/Adam/update_6/ResourceApplyAdamh
Ÿ
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Å`@Å`HÅ`b$PerceptualNetwork/activation/Sigmoidh
¿
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä`@Ä`HÄ`b#PerceptualNetwork/block2_conv1/Reluh
¬
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä`@Ä`HÄ`b%PerceptualNetwork/block2_conv1/Relu_1h
Ê
Évoid cudnn::winograd_nonfused::winogradForwardFilter4x4<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä`@Ä`HÄ`XbGgradient_tape/PerceptualNetwork/block2_conv1/Conv2D/Conv2DBackpropInputh
œ
üvoid fft2d_r2c_32x32<float, false, 5u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Ä`@Ä`HÄ`bEncoder/conv2_enc/Reluh
ı
üvoid fft2d_r2c_32x32<float, false, 5u, false>(float2*, float const*, int, int, int, int, int, int, int, int, int, cudnn::reduced_divisor, bool, int2, int, int)*28Ä`@Ä`HÄ`Xb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
®
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Ä`@Ä`HÄ`b3gradient_tape/Decoder/conv3_dec/BiasAdd/BiasAddGradh
®
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28Ä`@Ä`HÄ`b3gradient_tape/Decoder/conv4_dec/BiasAdd/BiasAddGradh
Æ
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä`@Ä`HÄ`blgradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
€
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28¿_@¿_H¿_b&PerceptualNetwork/activation/Sigmoid_1h
Ø
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 4, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 4ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 4, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ÄX@ÄXHÄXbgrayscale_to_rgb_1h
®
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28ÄX@ÄXHÄXb3gradient_tape/Decoder/conv2_dec/BiasAdd/BiasAddGradh
Æ
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ÄX@ÄXHÄXblgradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
¬
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡W@‡WH‡Wb#PerceptualNetwork/block2_conv1/Reluh
Ë
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28‡W@‡WH‡WXbGgradient_tape/PerceptualNetwork/block2_conv1/Conv2D/Conv2DBackpropInputh
Í
ívoid tensorflow::(anonymous namespace)::ResizeNearestNeighborBackwardNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28ÅP@ÅPHÅPb>gradient_tape/Decoder/upsamp5/resize/ResizeNearestNeighborGradh
≠
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ÅP@ÅPHÅPbkgradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28ÄP@ÄPHÄPb(gradient_tape/Encoder/conv4_enc/ReluGradh
∂
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ÄP@ÄPHÄPbgradient_tape/BroadcastToh
œ
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28ÄP@ÄPHÄPbEncoder/maxpool3/MaxPoolh
›
ávoid cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28ÄP@ÄPHÄPXb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
–
ävoid tensorflow::(anonymous namespace)::ResizeNearestNeighborNHWC<float>(int, float const*, int, int, int, int, int, float, float, float*)*28ÄP@ÄPHÄPb,Decoder/upsamp5/resize/ResizeNearestNeighborh
ƒ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ÄP@ÄPHÄPb%PerceptualNetwork/block2_conv1/Relu_1h
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28¿O@¿OH¿ObAssignAddVariableOp_4h

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28ÅH@ÅHHÅHb3gradient_tape/Encoder/conv3_enc/BiasAdd/BiasAddGradh
U
sgemm_32x32x32_TN_vec*28ÄH@ÄHHÄHb'gradient_tape/Decoder/decoding/MatMul_1h
¬
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28ÄH@ÄHHÄHbgradient_tape/truediv_1h
·
ävoid cudnn::winograd_nonfused::winogradWgradOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28ÄH@ÄHHÄHXb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
w
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28ÄH@ÄHHÄHbDecoder/conv3_dec/Reluh
•
\void tensorflow::BiasGradNCHW_SharedAtomics<float>(float const*, float*, int, int, int, int)*28ÄH@ÄHHÄHb0gradient_tape/Decoder/output/BiasAdd/BiasAddGradh
Ë
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å@@Å@HÅ@XbGgradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputh
∆
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@@Ä@HÄ@bgradient_tape/Negh
°
Évoid cub::DeviceSegmentedReduceKernel<cub::DeviceReducePolicy<float, float, int, tensorflow::functor::Sum<float> >::Policy600, float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float>(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, cub::TransformInputIterator<int, tensorflow::functor::RowOffset, cub::CountingInputIterator<int, long>, long>, int, tensorflow::functor::Sum<float>, float)*28Ä@@Ä@HÄ@bMeanh
Æ
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*28Ä@@Ä@HÄ@XbGgradient_tape/PerceptualNetwork/block1_conv1/Conv2D/Conv2DBackpropInputh
‡
~void cudnn::winograd::generateWinogradTilesKernel<0, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28Ä@@Ä@HÄ@XbGgradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputh
∫
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28Ä@@Ä@HÄ@b#PerceptualNetwork/block1_conv2/Reluh
º
~void cudnn::winograd::generateWinogradTilesKernel<1, float, float>(cudnn::winograd::GenerateWinogradTilesParams<float, float>)*28Ä@@Ä@HÄ@b%PerceptualNetwork/block1_conv2/Relu_1h
·
ävoid cudnn::winograd_nonfused::winogradWgradOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28Ä@@Ä@HÄ@Xb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
Ó
°void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long long const*, unsigned long long const*, tensorflow::random::PhiloxRandom, tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long long, tensorflow::random::NormalDistribution<tensorflow::random::PhiloxRandom, float>)*28Ä@@Ä@HÄ@b3Encoder/sampling/random_normal/RandomStandardNormalh

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@@Ä@HÄ@b3gradient_tape/Decoder/conv5_dec/BiasAdd/BiasAddGradh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@@Ä@HÄ@bEncoder/conv4_enc/Reluh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@@Ä@HÄ@Xb;gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropFilterh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@@Ä@HÄ@Xb:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputh
ö
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 4, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@@Ä@HÄ@bXgradient_tape/Decoder/conv5_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
§
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 4, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@@Ä@HÄ@bbgradient_tape/Encoder/maxpool5/MaxPool/MaxPoolGrad-2-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
ä
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@@Ä@HÄ@bHDecoder/conv3_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≠
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä@@Ä@HÄ@bkgradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿>@¿>H¿>Xb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Å8@Å8HÅ8b$Adam/Adam/update_4/ResourceApplyAdamh
∑
ávoid cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä8@Ä8HÄ8bDecoder/conv2_dec/Reluh
›
ávoid cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä8@Ä8HÄ8Xb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
É
,void tensorflow::SetZero<float>(int, float*)*28Ä8@Ä8HÄ8b>gradient_tape/Decoder/upsamp1/resize/ResizeNearestNeighborGradh

£void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä8@Ä8HÄ8b3gradient_tape/Encoder/conv5_enc/BiasAdd/BiasAddGradh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä8@Ä8HÄ8Xb:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputh
ä
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä8@Ä8HÄ8bHDecoder/conv5_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
Æ
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä8@Ä8HÄ8blgradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28ˇ7@ˇ7Hˇ7b(gradient_tape/Decoder/conv2_dec/ReluGradh
œ
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28ˇ7@ˇ7Hˇ7bEncoder/maxpool4/MaxPoolh
å
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 4, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ7@ˇ7Hˇ7bJEncoder/maxpool5/MaxPool-0-0-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
≠
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 4, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ7@ˇ7Hˇ7bkgradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
¬
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28¿7@¿7H¿7b#PerceptualNetwork/block1_conv2/Reluh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28†7@†7H†7b%Adam/Adam/update_24/ResourceApplyAdamh
·
ävoid cudnn::winograd_nonfused::winogradWgradOutput9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradWgradOutputParams<float, float>)*28Å7@Å7HÅ7Xb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
©
ﬂvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sigmoid_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sigmoid_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Å0@Å0HÅ0b0gradient_tape/Decoder/output/Sigmoid/SigmoidGradh
œ
ùvoid cudnn::ops::pooling_fw_4d_kernel<float, float, cudnn::maxpooling_func<float, (cudnnNanPropagation_t)0>, (cudnnPoolingMode_t)0, false>(cudnnTensorStruct, float const*, cudnnTensorStruct, float*, cudnnPoolingStruct, float, float, int, cudnn::reduced_divisor, cudnn::reduced_divisor)*28Å0@Å0HÅ0bEncoder/maxpool5/MaxPoolh
û
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Å0@Å0HÅ0Xb;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterh
G
sgemm_32x32x32_NN_vec*28Ä0@Ä0HÄ0XbDecoder/decoding/MatMulh
W
sgemm_32x32x32_NT_vec*28Ä0@Ä0HÄ0Xb'gradient_tape/Encoder/bottleneck/MatMulh
Ò
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä0@Ä0HÄ0b"Adam/Adam/update/ResourceApplyAdamh
ƒ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä0@Ä0HÄ0b%PerceptualNetwork/block1_conv2/Relu_1h
ä
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä0@Ä0HÄ0bHDecoder/conv4_dec/Relu-0-1-TransposeNCHWToNHWC-LayoutOptimizer:Transposeh
à
Yvoid tensorflow::BiasNCHWKernel<float>(int, float const*, float const*, float*, int, int)*28ˇ/@ˇ/Hˇ/bDecoder/output/BiasAddh
ˇ
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28‡.@‡.H‡.b3gradient_tape/Encoder/conv1_enc/BiasAdd/BiasAddGradh
¸
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28†(@†(H†(b0gradient_tape/Encoder/z_mean/BiasAdd/BiasAddGradh
‚
´void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*28Å(@Å(HÅ(bgradient_tape/DynamicStitch_1h
V
sgemm_32x32x32_NT_vec*28Ä(@Ä(HÄ(Xb&gradient_tape/Encoder/z_log_var/MatMulh
À
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_logistic_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä(@Ä(HÄ(bDecoder/output/Sigmoidh
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä(@Ä(HÄ(b(gradient_tape/Decoder/conv3_dec/ReluGradh
›
ávoid cudnn::winograd_nonfused::winogradForwardFilter9x9_5x5<float, float>(cudnn::winograd_nonfused::WinogradFilterParams<float, float>)*28Ä(@Ä(HÄ(Xb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
‡
´void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*28Ä(@Ä(HÄ(bgradient_tape/DynamicStitchh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä(@Ä(HÄ(b%Adam/Adam/update_10/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä(@Ä(HÄ(b$Adam/Adam/update_2/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä(@Ä(HÄ(b%Adam/Adam/update_28/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä(@Ä(HÄ(b$Adam/Adam/update_9/ResourceApplyAdamh
ˇ
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä(@Ä(HÄ(b3gradient_tape/Encoder/conv2_enc/BiasAdd/BiasAddGradh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä(@Ä(HÄ(bDecoder/conv2_dec/Reluh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä(@Ä(HÄ(bEncoder/conv3_enc/Reluh
π
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä(@Ä(HÄ(XbEncoder/conv1_enc/Conv2Dh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä(@Ä(HÄ(Xb;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterh
‹
övoid tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä(@Ä(HÄ(b(gradient_tape/grayscale_to_rgb/transposeh
ö
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28Ä(@Ä(HÄ(bXgradient_tape/Decoder/conv4_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
S
sgemm_32x32x32_NT_vec*28ˇ'@ˇ'Hˇ'Xb#gradient_tape/Encoder/z_mean/MatMulh
ö
®void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*28ˇ'@ˇ'Hˇ'bXgradient_tape/Decoder/conv3_dec/ReluGrad-0-TransposeNHWCToNCHW-LayoutOptimizer:Transposeh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Å @Å HÅ bEncoder/conv2_enc/Reluh
H
sgemm_32x32x32_NN_vec*28Ä @Ä HÄ XbEncoder/z_log_var/MatMulh
W
sgemm_32x32x32_TN_vec*28Ä @Ä HÄ b)gradient_tape/Encoder/bottleneck/MatMul_1h
V
sgemm_32x32x32_TN_vec*28Ä @Ä HÄ b(gradient_tape/Encoder/z_log_var/MatMul_1h
ı
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ bAdam/Powh
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä @Ä HÄ b(gradient_tape/Encoder/conv5_enc/ReluGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_11/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_19/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä @Ä HÄ b%Adam/Adam/update_26/ResourceApplyAdamh
†
Ävoid tensorflow::functor::BlockReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, 256, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28Ä @Ä HÄ bMean_1h
Û
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä @Ä HÄ b3gradient_tape/Decoder/conv5_dec/BiasAdd/BiasAddGradh
Ú
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä @Ä HÄ b2gradient_tape/Decoder/decoding/BiasAdd/BiasAddGradh
µ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ bDecoder/conv1_dec/Reluh
¬
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ b#PerceptualNetwork/block1_conv1/Reluh
Ÿ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb;gradient_tape/Encoder/conv1_enc/Conv2D/Conv2DBackpropFilterh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb;gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropFilterh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ Xb:gradient_tape/Encoder/conv3_enc/Conv2D/Conv2DBackpropInputh
Ë
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä @Ä HÄ XbGgradient_tape/PerceptualNetwork/block1_conv1/Conv2D/Conv2DBackpropInputh
E
sgemm_32x32x32_NN_vec*28ˇ@ˇHˇXbEncoder/z_mean/MatMulh
S
sgemm_32x32x32_TN_vec*28ˇ@ˇHˇb%gradient_tape/Encoder/z_mean/MatMul_1h
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28ˇ@ˇHˇXb:gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropInputh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_27/ResourceApplyAdamh
Ñ
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28†@†H†bEncoder/conv5_enc/Reluh
à
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Å@ÅHÅbgradient_tape/mul_2h
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Å@ÅHÅb%Adam/Adam/update_17/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Å@ÅHÅb$Adam/Adam/update_5/ResourceApplyAdamh
§
Ncask_cudnn::computeWgradBOffsetsKernel(cask_cudnn::ComputeWgradBOffsetsParams)*28Ä@ÄHÄXb;gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropFilterh
©
Scask_cudnn::computeWgradSplitKOffsetsKernel(cask_cudnn::ComputeSplitKOffsetsParams)*28Ä@ÄHÄXb;gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropFilterh
Ø
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/Tileh
±
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/Tile_1h
È
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb
div_no_nanh
Î
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbdiv_no_nan_1h
¿
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_quotient_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/truedivh
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb(gradient_tape/Decoder/conv5_dec/ReluGradh
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_2h
∏
Évoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/BroadcastTo_1h
ù
˚void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long, long long>, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAdam/addh
Ñ
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28Ä@ÄHÄbDecoder/conv1_dec/Reluh
Ñ
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28Ä@ÄHÄbDecoder/conv5_dec/Reluh
ë
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28Ä@ÄHÄb#PerceptualNetwork/block1_conv1/Reluh
ì
Uvoid cask_cudnn::computeOffsetsKernel<false, false>(cask_cudnn::ComputeOffsetsParams)*28Ä@ÄHÄb%PerceptualNetwork/block1_conv1/Relu_1h
°
Lvoid cudnn::ops::scalePackedTensor_kernel<float, float>(long, float*, float)*28Ä@ÄHÄXb:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputh
w
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Ä@ÄHÄbEncoder/conv2_enc/Reluh
Ö
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Ä@ÄHÄbDecoder/decoding/BiasAddh
á
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Ä@ÄHÄbEncoder/bottleneck/BiasAddh
É
,void tensorflow::SetZero<float>(int, float*)*28Ä@ÄHÄb>gradient_tape/Decoder/upsamp2/resize/ResizeNearestNeighborGradh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb$Adam/Adam/update_1/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_12/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_13/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_14/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_15/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_21/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_23/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_25/ResourceApplyAdamh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb%Adam/Adam/update_29/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb$Adam/Adam/update_3/ResourceApplyAdamh
Û
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28Ä@ÄHÄb$Adam/Adam/update_7/ResourceApplyAdamh
ﬂ
¬void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28Ä@ÄHÄbSumh
Û
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@ÄHÄb3gradient_tape/Encoder/conv3_enc/BiasAdd/BiasAddGradh
Û
¶void tensorflow::functor::ColumnReduceKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@ÄHÄb3gradient_tape/Encoder/conv5_enc/BiasAdd/BiasAddGradh
Ä
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@ÄHÄb4gradient_tape/Encoder/bottleneck/BiasAdd/BiasAddGradh
ˇ
≤void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)*28Ä@ÄHÄb3gradient_tape/Encoder/z_log_var/BiasAdd/BiasAddGradh
ﬂ
¿void tensorflow::functor::RowReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*28Ä@ÄHÄbSum_1h
ƒ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄb%PerceptualNetwork/block1_conv1/Relu_1h
∂
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXbDecoder/output/Conv2Dh
‹
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputh
ÿ
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputh
€
Övoid tensorflow::functor::ShuffleInTensor3Simple<float, 2, 1, 0, false>(int, float const*, tensorflow::functor::Dimension<3>, float*)*28Ä@ÄHÄXb:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputh
„
°void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*28ˇ@ˇHˇb(gradient_tape/Decoder/conv4_dec/ReluGradh
Ù
µvoid tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool)*28ˇ@ˇHˇb%Adam/Adam/update_16/ResourceApplyAdamh
Â
¡void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<bool, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_boolean_and_op, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<bool const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡b
LogicalAndh
Î
≈void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28‡@‡H‡bdiv_no_nan_2h
ﬂ	
ø	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ﬂ@ﬂHﬂbAddN_1h
É
,void tensorflow::SetZero<float>(int, float*)*28‡@‡H‡b>gradient_tape/Decoder/upsamp3/resize/ResizeNearestNeighborGradh
Å
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†bEncoder/sampling/addh
ø
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28†@†H†bEncoder/sampling/Exph
Å
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28†@†H†bgradient_tape/Cast_1h
É
,void tensorflow::SetZero<float>(int, float*)*28Å@ÅHÅb>gradient_tape/Decoder/upsamp5/resize/ResizeNearestNeighborGradh
˛
·void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbsubh
Ä
·void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbsub_1h
˜
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_pow_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb
Adam/Pow_1h
ã
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbEncoder/sampling/mul_1h
ù
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄb(gradient_tape/Encoder/sampling/mul_1/Mulh
Ú
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbadd_1h
Æ
ëvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_exp_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbExph
ª
çvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbEncoder/sampling/mulh
™
çvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbmulh
¢
Övoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_left<float, float, Eigen::internal::scalar_sum_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbaddh
 
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/sub/Negh
Ã
õvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_opposite_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/sub_1/Negh
À
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄb"gradient_tape/Encoder/sampling/mulh
º
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/Mul_3h
æ
èvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_right<float, float, Eigen::internal::scalar_product_op<float, float>, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbgradient_tape/mul/Mulh
∑
óvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_square_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseUnaryOp<Eigen::internal::scalar_square_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*28Ä@ÄHÄbSquareh
ˇ
”void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<int const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbgradient_tape/Casth
Ñ
ﬂvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorConversionOp<float, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAdam/Cast_1h
ç
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAddNh
è
Ôvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAddN_2h
ê
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOph
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_1h
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAssignAddVariableOp_3h
¬
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28Ä@ÄHÄbAdam/Adam/AssignAddVariableOph
x
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Ä@ÄHÄXbDecoder/output/Conv2Dh
{
Hvoid flip_filter<float, float>(float*, float const*, int, int, int, int)*28Ä@ÄHÄXbEncoder/conv1_enc/Conv2Dh
Ü
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Ä@ÄHÄbEncoder/z_log_var/BiasAddh
É
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*28Ä@ÄHÄbEncoder/z_mean/BiasAddh
É
,void tensorflow::SetZero<float>(int, float*)*28Ä@ÄHÄb>gradient_tape/Decoder/upsamp4/resize/ResizeNearestNeighborGradh
õ
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ˇ@ˇHˇb&gradient_tape/Encoder/sampling/mul/Mulh
à
€void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float, float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ˇ@ˇHˇbgradient_tape/Mul_4h
í
„void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ˇ@ˇHˇbAssignAddVariableOp_5h
∫
ãvoid Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long long const, long long const>, Eigen::TensorMap<Eigen::Tensor<long long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*28ˇ@ˇHˇbAssignAddVariableOp_6h