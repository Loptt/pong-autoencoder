?	?jdWZz@@?jdWZz@@!?jdWZz@@	~???0??~???0??!~???0??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?jdWZz@@C?l?P@1?9???8%@A?~?{1@I?U?@ة@Y?Q?????*	/?$???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorrQ-"?1&@!VT"???X@)rQ-"?1&@1VT"???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???<?;??!?o?P???)???<?;??1?o?P???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???	ۭ?!,?????)?????>??1?0y,????:Preprocessing2F
Iterator::Modely$^????!??????)?%:?,B??1??KpQ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap? x|3&@!???r??X@)@?5_%o?19?<?tn??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???0??I?????P@QK??C@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	C?l?P@C?l?P@!C?l?P@      ??!       "	?9???8%@?9???8%@!?9???8%@*      ??!       2	?~?{1@?~?{1@!?~?{1@:	?U?@ة@?U?@ة@!?U?@ة@B      ??!       J	?Q??????Q?????!?Q?????R      ??!       Z	?Q??????Q?????!?Q?????b      ??!       JGPUY???0??b q?????P@yK??C@@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput???!???!???!???0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputK??Y??!???~???0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Ë*Gӣ?!???I????0"8
Decoder/conv5_dec/Relu_FusedConv2Dӱ?????!+?E/????"K
$Adam/Adam/update_8/ResourceApplyAdamResourceApplyAdam6?Y?Ԣ?!?%.??%??"3
Decoder/output/Conv2DConv2D?????ߠ?!$+??A??0"e
:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputںm?,L??!v?X>_K??0"8
Decoder/conv1_dec/Relu_FusedConv2D?R??Œ??!?p'??D??"e
:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput????;???!??R?2??0"8
Decoder/conv4_dec/Relu_FusedConv2D??
?^7??!!??>e??Q      Y@YG????@a?g??#?W@q???2?E@y^???ߣ?"?
both?Your program is POTENTIALLY input-bound because 7.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?43.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 