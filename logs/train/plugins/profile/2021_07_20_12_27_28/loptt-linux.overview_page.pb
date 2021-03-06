?	??m??e%@??m??e%@!??m??e%@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??m??e%@֋??hW @1??3?~@A?b?dU???I?_???@*	?(\??A?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorP??ô@!??h?ϭX@)P??ô@1??h?ϭX@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???\7???!?Mq??k??)???\7???1?Mq??k??:Preprocessing2F
Iterator::Model?^?"????!??l???)??ۻ}??1)n?2J???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??F????!??|??Q??)???q??1?ׇ(S7??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapV]??@!mM??g?X@)o,(?4j?1d??۾??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?46.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI~?.G?bP@Q??q?:A@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	֋??hW @֋??hW @!֋??hW @      ??!       "	??3?~@??3?~@!??3?~@*      ??!       2	?b?dU????b?dU???!?b?dU???:	?_???@?_???@!?_???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q~?.G?bP@y??q?:A@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInputR?!?A7??!R?!?A7??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??cJ???!Mq??????0"3
Decoder/output/Conv2DConv2Dobϧq??!B??????0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?9:0aq??!??$h???0"e
:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputש?Ҟ?!?????y??0"8
Decoder/conv4_dec/Relu_FusedConv2D??.?Ϟ?!vj?H?S??"8
Decoder/conv3_dec/Relu_FusedConv2D
nz<?ϛ?!?X?f??"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??YZ?A??!&v????0"g
;gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???q8??!?%??j???0"g
;gradient_tape/Encoder/conv1_enc/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterƯ??#??!??o??0Q      Y@Yd+????@aJݗ?V?W@q??(ڪ??y??d?U???"?

both?Your program is POTENTIALLY input-bound because 19.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?46.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 