?	[%X?Z@[%X?Z@![%X?Z@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-[%X?Z@?E{??YW@1??n??C(@A?a?????I?*?WY???*	d;?O???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?$y??@!???:0?X@)?$y??@1???:0?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch霟?8???!?,?\[R??)霟?8???1?,?\[R??:Preprocessing2F
Iterator::Model???????!??&???)?9?S?ɒ?1qYm?$??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism7T??7???!ԠC???)7ݲC?Æ?1??n?'???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??e???@!Pe?˯X@)I?s
??a?1?!C˓ݤ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIt6TF=.V@Q\L^??&@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?E{??YW@?E{??YW@!?E{??YW@      ??!       "	??n??C(@??n??C(@!??n??C(@*      ??!       2	?a??????a?????!?a?????:	?*?WY????*?WY???!?*?WY???B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qt6TF=.V@y\L^??&@?"3
Decoder/output/Conv2DConv2DH?M;???!H?M;???0"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput??KD??!pFf?C??0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputɇ?~ި?!b(^?M??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltergU?}???!???sc??0"8
Decoder/conv5_dec/Relu_FusedConv2D$????ʢ?!?1r?a??"e
:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput	????E??!???v???0"8
Decoder/conv4_dec/Relu_FusedConv2D???????!!?Hȿ??"8
Decoder/conv1_dec/Relu_FusedConv2D%	????!???#ɩ??"e
:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInputD?z?Y(??!??ѽN|??0"e
:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInput?????U??!???????0Q      Y@YB躍`@a?~Q$??W@q?^s??F@y??1dp
??"?

both?Your program is POTENTIALLY input-bound because 86.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?45.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 