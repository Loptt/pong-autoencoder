?	?B???`@?B???`@!?B???`@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?B???`@?q??r?]@1?G'@A??uR_???I?Ov@*	?&1ܳ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorfI??Z&@!/y?Y?X@)fI??Z&@1/y?Y?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??,?s???!?}?Ȏ??)
?_??͘?1Нn?}??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????@g??!?*?????)????@g??1?*?????:Preprocessing2F
Iterator::Model?,C????!??i??"??)??f?v?d?1wǀI?<??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapi?x?J(@!?,?غ?X@)?#????^?1蝥???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI????V@Q7?!?1_!@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q??r?]@?q??r?]@!?q??r?]@      ??!       "	?G'@?G'@!?G'@*      ??!       2	??uR_?????uR_???!??uR_???:	?Ov@?Ov@!?Ov@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????V@y7?!?1_!@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput?/?e???!?/?e???0"3
Decoder/output/Conv2DConv2D?<2C??!?טZ??0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?D?A???!?Q?A???0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter6Q??j9??!?;?bnA??0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput՘9?<???!L??1????0"8
Decoder/conv5_dec/Relu_FusedConv2D??????!?/Y??s??"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?=?J?(??!|??7???0"8
Decoder/conv4_dec/Relu_FusedConv2D???}??!??rX8??"C
%Decoder/coordinate_channel2d_9/concatConcatV2??c?b??!????In??"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter\??y+??!Z?~???0Q      Y@Y˛;??!@a??xj?V@q????%-I@y??fz???"?

both?Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?50.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 