?	ĕ?wF?P@ĕ?wF?P@!ĕ?wF?P@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-ĕ?wF?P@????@L@1???s]@A??=x???I	N} y@*	??/=??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?.??[<@!???(i?X@)?.??[<@1???(i?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchۅ?:????!p??e??)ۅ?:????1p??e??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?$????!?b?ȩ??)?C?bԵ??1?E??r???:Preprocessing2F
Iterator::Model??"ڎ??!n??R????))??qh?1???)???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?IӠh>@!?fZ???X@)w?ِf`?1L????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 82.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?<ę?V@Q'??1?"@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????@L@????@L@!????@L@      ??!       "	???s]@???s]@!???s]@*      ??!       2	??=x?????=x???!??=x???:		N} y@	N} y@!	N} y@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?<ę?V@y'??1?"@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInputk!ǀ???!k!ǀ???0"3
Decoder/output/Conv2DConv2D?c???­?!j??;????0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterI?,??.??!??K?C??0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?K?Ts???!?[9>p=??0"g
;gradient_tape/Encoder/conv1_enc/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter[Y?8i/??!???f???0"8
Decoder/conv1_dec/Relu_FusedConv2D???????!?P??Z??"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltereD?{ڕ?!%Dlx???0"?
igradient_tape/Decoder/output/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown?]%?g???!?z?????"8
Decoder/conv4_dec/Relu_FusedConv2D^?;???!??v????"e
:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputJ?@?rؑ?!߱Z?i6??0Q      Y@Yk?4w?_@aɳ???W@q?lzJ?;@y[kd?]??"?
both?Your program is POTENTIALLY input-bound because 82.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?27.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 