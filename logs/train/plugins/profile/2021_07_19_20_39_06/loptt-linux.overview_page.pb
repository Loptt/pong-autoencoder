?	d?C?]@d?C?]@!d?C?]@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-d?C?]@?S???Y@1?6qr?&@A??s?v???I?4f?@*	??"?.?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorP8??L?@!?!???X@)P8??L?@1?!???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch;?O Š?!?u??du??);?O Š?1?u??du??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?\?	???!???Ϯ??)@ޫV&???1? 5$????:Preprocessing2F
Iterator::Model^/M????!B9z?D??)'??bg?1?W$?_??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???[v?@!??iv?X@)???]/Ma?1?K??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?B=Z?V@Q_?M.?"@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S???Y@?S???Y@!?S???Y@      ??!       "	?6qr?&@?6qr?&@!?6qr?&@*      ??!       2	??s?v?????s?v???!??s?v???:	?4f?@?4f?@!?4f?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?B=Z?V@y_?M.?"@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInputG~l6???!G~l6???0"3
Decoder/output/Conv2DConv2DQ.?E侦?!ۉ?G????0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter.Oe?\??!?ݲi?B??0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?J.?zK??!(8Ѫ??0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput?+&|?'??!???????0"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilteru?u???!?o]?Wq??0"8
Decoder/conv5_dec/Relu_FusedConv2DF&???=??!???2%??"8
Decoder/conv4_dec/Relu_FusedConv2D/6s??;??!X5"?????"g
;gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter.??3??!{??(c	??0"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ZӗS???!$?b?S??0Q      Y@Y1?f?t@a???y??W@q?)?D@y?g?????"?

both?Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?41.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 