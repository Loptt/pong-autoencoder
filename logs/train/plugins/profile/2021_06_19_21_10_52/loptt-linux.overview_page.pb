?	&?B??5@&?B??5@!&?B??5@	ܟ?M)g??ܟ?M)g??!ܟ?M)g??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6&?B??5@L????h@1?9z?&.@A?ɐ??I()? F@Y`2?CP??*	sh???l?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorF#?W<?@!?z6?5?X@)F#?W<?@1?z6?5?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??.????!x??/??)??.????1x??/??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?"2?⍤?!?g)?`??)??5Φ#??1??????:Preprocessing2F
Iterator::Model???D???!?康????){?Fw;s?1??r????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap$?@??@!4????X@)??V?I?k?1w0|Ђi??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?13.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ݟ?M)g??I?-???>@Q??BQ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	L????h@L????h@!L????h@      ??!       "	?9z?&.@?9z?&.@!?9z?&.@*      ??!       2	?ɐ???ɐ??!?ɐ??:	()? F@()? F@!()? F@B      ??!       J	`2?CP??`2?CP??!`2?CP??R      ??!       Z	`2?CP??`2?CP??!`2?CP??b      ??!       JGPUYݟ?M)g??b q?-???>@y??BQ@?"E
#PerceptualNetwork/block1_conv2/Relu_FusedConv2D?/6????!?/6????"G
%PerceptualNetwork/block1_conv2/Relu_1_FusedConv2DZ????{??!??r˹??"r
Ggradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput<?Q?j??!?=jw??0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Yr??!?Jd??S??0"E
#PerceptualNetwork/block2_conv2/Relu_FusedConv2DF?8????!Q7;4-??"G
%PerceptualNetwork/block2_conv2/Relu_1_FusedConv2D?B??w??!?
-?/??"r
Ggradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?]?????!f?GPw"??0"G
%PerceptualNetwork/block2_conv1/Relu_1_FusedConv2D???????!?c@?I7??"E
#PerceptualNetwork/block2_conv1/Relu_FusedConv2D;S?u?p??!>N?AhE??"r
Ggradient_tape/PerceptualNetwork/block2_conv1/Conv2D/Conv2DBackpropInputConv2DBackpropInput?D?-?:??!???)??0Q      Y@YvMr	?$@a=Q?ѾjV@q?`t?F???y̋f/???"?

both?Your program is POTENTIALLY input-bound because 16.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?13.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 