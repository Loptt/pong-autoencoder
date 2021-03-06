?	0?k???R@0?k???R@!0?k???R@	??@c?????@c???!??@c???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails60?k???R@ol?`@1?+??=@A?ND??C@I~6rݔ? @YW??el???*	R????E@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?E? ??!wk?0{L@)?E? ??1wk?0{L@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Ƽ?8d??!W???c?U@)?iT?d??1l??u??>@:Preprocessing2F
Iterator::ModelJ?i?W??!      Y@)H???w?1H??R?l*@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??@c???I%??7SuN@Qê?'ՆC@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ol?`@ol?`@!ol?`@      ??!       "	?+??=@?+??=@!?+??=@*      ??!       2	?ND??C@?ND??C@!?ND??C@:	~6rݔ? @~6rݔ? @!~6rݔ? @B      ??!       J	W??el???W??el???!W??el???R      ??!       Z	W??el???W??el???!W??el???b      ??!       JGPUY??@c???b q%??7SuN@yê?'ՆC@?"r
Ggradient_tape/PerceptualNetwork/block1_conv1/Conv2D/Conv2DBackpropInputConv2DBackpropInputTݕ?k???!Tݕ?k???0"E
#PerceptualNetwork/block1_conv2/Relu_FusedConv2D	??pa??!?e?`$???"G
%PerceptualNetwork/block1_conv2/Relu_1_FusedConv2D˛?O?@??!ߙ[Dȣ??"r
Ggradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?? coѪ?!̺#$X??0"E
#PerceptualNetwork/block2_conv2/Relu_FusedConv2D^?	tI??!RS?@u??"G
%PerceptualNetwork/block2_conv2/Relu_1_FusedConv2D?z???ߥ?!?4+C21??"r
Ggradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput???<@???!F??J????0"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?1???&??!wxt????0"G
%PerceptualNetwork/block2_conv1/Relu_1_FusedConv2D?V?{????!L#?Na	??"E
#PerceptualNetwork/block2_conv1/Relu_FusedConv2Db?\-????!R??!???Q      Y@Y?e. @aN=?]3?V@q?V?G?X@y?)N????"?

both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?98.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 