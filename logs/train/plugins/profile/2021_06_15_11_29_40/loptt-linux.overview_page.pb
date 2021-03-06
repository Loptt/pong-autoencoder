?	?UfJk`@?UfJk`@!?UfJk`@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?UfJk`@:?6U?f^@1???4?@A4??`??I0עh?@*	!?rhѽ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???b(@!&H?m??X@)???b(@1&H?m??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch/ܹ0ҋ??!??j?Z<??)/ܹ0ҋ??1??j?Z<??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism7?Nx	N??!M?d?????)|`?? ??1c^??_r??:Preprocessing2F
Iterator::Model+5{???!???????)zލ?Ai?1钑????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap"??ƽ@!??CA??X@)??O?d?1??㼛ޢ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIjf?c?W@Qby?)?Y@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	:?6U?f^@:?6U?f^@!:?6U?f^@      ??!       "	???4?@???4?@!???4?@*      ??!       2	4??`??4??`??!4??`??:	0עh?@0עh?@!0עh?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qjf?c?W@yby?)?Y@?"e
:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput??[??m??!??[??m??0"e
:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput?.?????!k`Ü????0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputU?0C??!?Rl????0"8
Decoder/conv5_dec/Relu_FusedConv2DG|F????!??-0?H??"e
:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInput<Ace???!?*ڜL??0"8
Decoder/conv4_dec/Relu_FusedConv2D?#?+X??!!|7??"8
Encoder/conv5_enc/Relu_FusedConv2Dh?Fߎ=??!??d??~??"e
:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputD?h_Ę?!&????0"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput??
o?n??!K??????0"e
:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInputJ?E?7??!hz???
??0Q      Y@Y6?d?M6@am??&??W@qE9I<??3@y???????"?

both?Your program is POTENTIALLY input-bound because 92.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?19.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 