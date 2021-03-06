?	??f??[@??f??[@!??f??[@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??f??[@|??c??Y@1?o&??	@A???W??Ij??h?M@*	?/?^?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??>?@!?&=???X@)??>?@1?&=???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?e??S9??!???u???)?e??S9??1???u???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?j??դ?!<????)??b????1?Qݹ8p??:Preprocessing2F
Iterator::Model?ؖg)??!?z????)???<HO??1??s?%???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??"???@!??5??X@)	n?l??[?1?`M^駟?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?B?GX@Q#>???@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|??c??Y@|??c??Y@!|??c??Y@      ??!       "	?o&??	@?o&??	@!?o&??	@*      ??!       2	???W?????W??!???W??:	j??h?M@j??h?M@!j??h?M@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?B?GX@y#>???@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput?L?pt???!?L?pt???0"8
Decoder/conv5_dec/Relu_FusedConv2D???L???!L#?ށ`??"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput??.?*i??!iIM??0"8
Decoder/conv4_dec/Relu_FusedConv2Dv?`%e؟?!<?PK???"3
Decoder/output/Conv2DConv2Da4C??ٞ?!?0?????0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Wk?}֜?!ƛ???{??0"g
;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter&\K_%|??!KpA???0"S
2gradient_tape/Encoder/maxpool5/MaxPool/MaxPoolGradMaxPoolGrad?C?8&??!᧰??g??"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterЗ?"&??!?$J?T???0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterКQ|r??!?>Z?|???0Q      Y@Y??8??8@a?q?q?W@q?4????F@y??l'?o??"?
both?Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?45.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 