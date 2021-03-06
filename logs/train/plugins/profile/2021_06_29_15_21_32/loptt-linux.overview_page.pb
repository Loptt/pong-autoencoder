?	?j????3@?j????3@!?j????3@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?j????3@l?<:@1 |?&@Ap@KW????I?'?$?z@*	???K???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorF?xx"@!?2??թX@)F?xx"@1?2??թX@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??lXSY??!a~hא???)??9d???1UC??@??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapd???7@!?C?X@)???|	??1e=X^m??:Preprocessing2F
Iterator::Model??????!
y:?y???)2 Tq???1R??O?Y??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch[{??B??!m?"#??)[{??B??1m?"#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?20.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??gCnE@Q68????L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	l?<:@l?<:@!l?<:@      ??!       "	 |?&@ |?&@! |?&@*      ??!       2	p@KW????p@KW????!p@KW????:	?'?$?z@?'?$?z@!?'?$?z@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??gCnE@y68????L@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput9?ܵ??!9?ܵ??0"3
Decoder/output/Conv2DConv2D???vʳ??!?&?Og???0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterr?@????!;F?e?T??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?"$?z???!?ʀ?re??0"8
Decoder/conv5_dec/Relu_FusedConv2D?'H???!qI	???"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput?6+??0??!????????0"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter7???߿??!?H???4??0"C
%Decoder/coordinate_channel2d_1/concatConcatV2?܌v????!?g?????"8
Decoder/conv4_dec/Relu_FusedConv2D?wyؠ??!?? ????"8
Encoder/conv5_enc/Relu_FusedConv2D??O?MȐ?!)?? D???Q      Y@Y˛;??!@a??xj?V@q????????y?ű0ƒ?"?

both?Your program is POTENTIALLY input-bound because 21.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?20.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 