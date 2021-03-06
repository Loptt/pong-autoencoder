?	???.[@???.[@!???.[@	v@b}?v@b}?!v@b}?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???.[@! _BlW@1uF^?)@A??R%?ޢ?IV???؟@Y?}͑??*	ףp=ꄽ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????c?@!??r??X@)????c?@1??r??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchWBwI???!L?tsB??)WBwI???1L?tsB??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??Դ?i??!???5'??)X???ާ??1,?????:Preprocessing2F
Iterator::Model4`?_???!?????)-?\o??p?1?B?͏??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??nJ?@!??????X@)??IӠh^?1???T&??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9v@b}?I~'??aV@Q<3?O'@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	! _BlW@! _BlW@!! _BlW@      ??!       "	uF^?)@uF^?)@!uF^?)@*      ??!       2	??R%?ޢ???R%?ޢ?!??R%?ޢ?:	V???؟@V???؟@!V???؟@B      ??!       J	?}͑???}͑??!?}͑??R      ??!       Z	?}͑???}͑??!?}͑??b      ??!       JGPUYv@b}?b q~'??aV@y<3?O'@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput"??)y???!"??)y???0"e
:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput??<???!?pZ??J??0"g
;gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?' Y*??!?z??Z??0"3
Decoder/output/Conv2DConv2D?-?ns??!;?H??Y??0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput??
?yG??!Ӡ?????0"g
;gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterMZ?҆??!,?xņ??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterVG??M??!5?l????0"8
Decoder/conv5_dec/Relu_FusedConv2DcL(????!?>,L;??"8
Decoder/conv4_dec/Relu_FusedConv2D?uޤ0 ??!L?`A??"8
Decoder/conv1_dec/Relu_FusedConv2D4???]f??!?7>????Q      Y@YB躍`@a?~Q$??W@qK?LÓB@y?Y?V???"?

both?Your program is POTENTIALLY input-bound because 86.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?37.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 