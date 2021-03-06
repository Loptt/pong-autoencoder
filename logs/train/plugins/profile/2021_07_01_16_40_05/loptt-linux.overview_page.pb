?	E?*kod@E?*kod@!E?*kod@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-E?*kod@;???4b@1????*@AL?1?=B??I?^??@*	^?IBu?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorgd???@!??b???X@)gd???@1??b???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch"?4???!:???)??)"?4???1:???)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismR?=?N??!2h??zm??)*??% ???1??m?`??:Preprocessing2F
Iterator::Model??v????!BӋ?Y???)$????l?1~X?^?&??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??]???@!Y??L??X@)겘?|\[?1{?Q{S??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI枨?V@Q???*? @Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;???4b@;???4b@!;???4b@      ??!       "	????*@????*@!????*@*      ??!       2	L?1?=B??L?1?=B??!L?1?=B??:	?^??@?^??@!?^??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q枨?V@y???*? @?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInputCU<w??!CU<w??0"3
Decoder/output/Conv2DConv2D+I????!?H???0"8
Decoder/conv2_dec/Relu_FusedConv2D|???ۤ?!???????"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterEtξ???!?Ȱ?????0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter1?	*??!%??o.???0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInpute??????! #Y-???0"8
Decoder/conv5_dec/Relu_FusedConv2Dg??s??!??]P>??"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterkvѪPӖ?!?[ss??0"8
Decoder/conv4_dec/Relu_FusedConv2D???W#???!谊?%???"C
%Decoder/coordinate_channel2d_3/concatConcatV2%Q?,?
??!??V????Q      Y@Y˛;??!@a??xj?V@qk?1KuH@y/<????"?

both?Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?48.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 