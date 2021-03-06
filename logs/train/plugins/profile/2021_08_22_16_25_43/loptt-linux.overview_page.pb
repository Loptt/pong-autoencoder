?	t|?8c?D@t|?8c?D@!t|?8c?D@	g?G?1??g?G?1??!g?G?1??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6t|?8c?D@?F??s@1?fd???@AX?2ı?;@I??N?0?@Y?d???*	?Zd;?I@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???7/N??!@?ٵJ@)???7/N??1@?ٵJ@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??=x?Ҧ?!?/2Ӫ?U@)????W??1`Q??|]@@:Preprocessing2F
Iterator::Model^H??0~??!      Y@)?HJzZ}?1??nf??+@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?12.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9g?G?1??I??$%DU@Q%y?r?-@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?F??s@?F??s@!?F??s@      ??!       "	?fd???@?fd???@!?fd???@*      ??!       2	X?2ı?;@X?2ı?;@!X?2ı?;@:	??N?0?@??N?0?@!??N?0?@B      ??!       J	?d????d???!?d???R      ??!       Z	?d????d???!?d???b      ??!       JGPUYg?G?1??b q??$%DU@y%y?r?-@?"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter9?B"?9??!9?B"?9??0"3
Decoder/output/Conv2DConv2DBx???! j]	Z#??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??vIS??!?"?O,8??0"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput???jẈ?!??^*B+??0"8
Decoder/conv4_dec/Relu_FusedConv2D???<7ޚ?!?x?	???"e
:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInput????k??!=??>J??0"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter~??3}9??!mJ8?????0"g
;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!*??>?8??0"g
;gradient_tape/Encoder/conv1_enc/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterƶ???S??!?e?w????0"8
Decoder/conv1_dec/Relu_FusedConv2D6??3?(??!?0?zn ??Q      Y@Y??|??|@a?3X?3?W@qX?j??X@y\g`7?q??"?
both?Your program is POTENTIALLY input-bound because 6.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?12.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?98.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 