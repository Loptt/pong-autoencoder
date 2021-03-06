?	?~?:p	\@?~?:p	\@!?~?:p	\@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?~?:p	\@[a?^CX@1s?FZ*?'@A??N@a??I7?嶭@*	??x??ɻ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorI?s
?C@!X?:͎?X@)I?s
?C@1X?:͎?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??Y?e??!R??????)??Y?e??1R??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Ӻj???!Ȓ??)yx??e??1????????:Preprocessing2F
Iterator::Model??8?#+??!??H%[??)A??ǘ?f?1C?i?$???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap#h?$?E@!?nݵI?X@)ۥ???_?1X?'????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 85.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noId?Q?2\V@Q?Tq?k%@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	[a?^CX@[a?^CX@![a?^CX@      ??!       "	s?FZ*?'@s?FZ*?'@!s?FZ*?'@*      ??!       2	??N@a????N@a??!??N@a??:	7?嶭@7?嶭@!7?嶭@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qd?Q?2\V@y?Tq?k%@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput?b9?C??!?b9?C??0"?
igradient_tape/Decoder/output/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown?O鹿???!ض??????"3
Decoder/output/Conv2DConv2D?2?{??!Za??ҟ??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?E??-]??!ly???{??0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?x=???!?z?????0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputi????L??!????ݴ??0"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??o ??!/???V??0"8
Decoder/conv5_dec/Relu_FusedConv2D?a?A?Ù?!L??????"8
Decoder/conv4_dec/Relu_FusedConv2Dn?!O'???!????l??"g
;gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???P{??!l???7???0Q      Y@Y1?f?t@a???y??W@qDn??nC@y??$D7??"?
both?Your program is POTENTIALLY input-bound because 85.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?38.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 