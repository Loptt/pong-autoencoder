?	Uj?@a@Uj?@a@!Uj?@a@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Uj?@a@?B??^@1???_{)@A@?3iSu??I???uo?
@*	Zd;???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator@?j??@!:)?@??X@)@?j??@1:)?@??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchD???XP??!? 2??)D???XP??1? 2??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?h?????!???????)p??-??1BR`??D??:Preprocessing2F
Iterator::Model?+?????!? ?????)???mj?1?Q?,?3??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???=??@!}???X@)a??q6]?1?2ߺ#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI????a?V@Qy????"@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?B??^@?B??^@!?B??^@      ??!       "	???_{)@???_{)@!???_{)@*      ??!       2	@?3iSu??@?3iSu??!@?3iSu??:	???uo?
@???uo?
@!???uo?
@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????a?V@yy????"@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput?I???>??!?I???>??0"3
Decoder/output/Conv2DConv2D????s???!Bwl?'???0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??i??ѣ?!?p??Sj??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!3? ?????0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputN?3????!????_???0"8
Decoder/conv5_dec/Relu_FusedConv2D4cT㭗?!"Y?=??"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter<???뫗?!??}????0"8
Decoder/conv4_dec/Relu_FusedConv2D?7rܱ??!A??D???"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?v??m???!?~?1??0"C
%Decoder/coordinate_channel2d_7/concatConcatV2:?Nfb??!?9?}W??Q      Y@Y˛;??!@a??xj?V@q?k3n3JH@yW??5???"?

both?Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?48.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 