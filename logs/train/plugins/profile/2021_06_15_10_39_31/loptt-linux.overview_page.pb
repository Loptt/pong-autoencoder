?	?????L@?????L@!?????L@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?????L@ӟ?H?@1???OB@AWZF?=-'@I]T?dA@*	?C?l??e@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?+?V]???!???..I@)?+?V]???1???..I@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism::ZՒ??!l?2?j?T@).???<??1QI?=?V@@:Preprocessing2F
Iterator::Model?|	^??!      Y@)8?{?5Z??1SF4'U?0@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?62.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?^??KV@Q??? ?%@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ӟ?H?@ӟ?H?@!ӟ?H?@      ??!       "	???OB@???OB@!???OB@*      ??!       2	WZF?=-'@WZF?=-'@!WZF?=-'@:	]T?dA@]T?dA@!]T?dA@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?^??KV@y??? ?%@?"e
:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput???A?1??!???A?1??0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputC3iA???!rl?5??0"8
Decoder/conv5_dec/Relu_FusedConv2Dq??滢??!^???I???"e
:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInputw??#????!<??}???0"8
Decoder/conv4_dec/Relu_FusedConv2D??#jՠ??!p5??~??"8
Encoder/conv5_enc/Relu_FusedConv2D(??U????!] ?v???"e
:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput?"l!??!???????0"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput??;?T???!????.???0"e
:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInput??!?oW??!??֥O??0"8
Decoder/conv3_dec/Relu_FusedConv2D߰k,???!??^?H???Q      Y@Yf?t??@a?y??D?W@qy5d?zX@yF9????"?
both?Your program is POTENTIALLY input-bound because 6.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?62.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 