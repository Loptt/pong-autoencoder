?	?y?CnT@?y?CnT@!?y?CnT@	???n?n?????n?n??!???n?n??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?y?CnT@?????R@1j??ߪ@A?#EdXţ?I?v??-u??Y^M??????*	?z?gײ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?w?7N
@!.????X@)?w?7N
@1.????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch? ?bG???!?5?)???)? ?bG???1?5?)???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??\????!#?:2?1??)?}?<???1qKdE???:Preprocessing2F
Iterator::Model???????!????#??)OYM?]g?1??_XF??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMaplЗ??@!	??p?X@)?\??7?e?1???(????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???n?n??I???k?W@Q???Pz?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????R@?????R@!?????R@      ??!       "	j??ߪ@j??ߪ@!j??ߪ@*      ??!       2	?#EdXţ??#EdXţ?!?#EdXţ?:	?v??-u???v??-u??!?v??-u??B      ??!       J	^M??????^M??????!^M??????R      ??!       Z	^M??????^M??????!^M??????b      ??!       JGPUY???n?n??b q???k?W@y???Pz?@?"3
Decoder/output/Conv2DConv2D???¥b??!???¥b??0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter[H7,??! s?nG??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?V?ڧ?!b???=??0"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput,||?!(??!m?g????0"8
Decoder/conv4_dec/Relu_FusedConv2DH?l%b]??!?44?S??"e
:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInput#?X??o??!=(???0??0"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter9?@??k??!8d^????0"8
Decoder/conv1_dec/Relu_FusedConv2De?5?W???!W?7?z??"8
Encoder/conv4_enc/Relu_FusedConv2D͠l?0X??!d[???p??"e
:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputK????V??!???wk???0Q      Y@Y7?<7?<@a<7?<7?W@q]?b/?B@y"1????"?

both?Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?37.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 