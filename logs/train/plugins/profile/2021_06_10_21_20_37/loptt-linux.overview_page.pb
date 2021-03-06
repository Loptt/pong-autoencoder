?	r?z?f?N@r?z?f?N@!r?z?f?N@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-r?z?f?N@?? k՞@1ڎ???;K@A?x?JxB??Iڐf?@*	/?$F??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator u?)@!?qBX*?X@) u?)@1?qBX*?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchиp $??!??p?5???)иp $??1??p?5???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?RD?U???!?jx?????)^?/?ۆ?1C??????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?0E?4.@!?p?qr?X@)?0??Zq?1???Cf ??:Preprocessing2F
Iterator::Modely??"????!??G?F??)?????l?1???,Ib??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?E?cI:&@QGW?Ӷ8V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?? k՞@?? k՞@!?? k՞@      ??!       "	ڎ???;K@ڎ???;K@!ڎ???;K@*      ??!       2	?x?JxB???x?JxB??!?x?JxB??:	ڐf?@ڐf?@!ڐf?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?E?cI:&@yGW?Ӷ8V@?"E
#PerceptualNetwork/block1_conv2/Relu_FusedConv2D??U5???!??U5???"E
#PerceptualNetwork/block3_conv2/Relu_FusedConv2DFj??E???!!L5?۽?"G
%PerceptualNetwork/block1_conv2/Relu_1_FusedConv2D?X;v%??!J?P)*???"G
%PerceptualNetwork/block3_conv2/Relu_1_FusedConv2DF?T\?ئ?!?#f a???"e
:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput?????!?h?C????0"`
?gradient_tape/PerceptualNetwork/block1_pool/MaxPool/MaxPoolGradMaxPoolGrad?賫_???!b?:????"E
#PerceptualNetwork/block3_conv3/Relu_FusedConv2D.U??Т?!???????"r
Ggradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?9?ڼ??!G??H_???0"G
%PerceptualNetwork/block3_conv4/Relu_1_FusedConv2DN????m??!??<?m??"G
%PerceptualNetwork/block2_conv2/Relu_1_FusedConv2D?uWF????!?-?	?f??Q      Y@Y?+Q?!@a^?ڕ??V@q?????y?Y??L?v?"?

both?Your program is POTENTIALLY input-bound because 7.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 