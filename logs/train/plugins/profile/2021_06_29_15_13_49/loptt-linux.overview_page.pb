?	N(D?!?-@N(D?!?-@!N(D?!?-@	?s?uO???s?uO??!?s?uO??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6N(D?!?-@?3K?T??1?f???"@A?'I?L???I?{H??_@Yo?[t???*	J+???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatoru ???G@!??9???X@)u ???G@1??9???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?b??*3??!/+??!???)?b??*3??1/+??!???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??k????!J??w7??)?B ?8???1e??N????:Preprocessing2F
Iterator::Model?<??tZ??!
%2xa???)???4??1?"?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?k?}?J@!??=?X@)؞Y??f?1H?V2??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?24.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?s?uO??IP?R!P?B@Q???#O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?3K?T???3K?T??!?3K?T??      ??!       "	?f???"@?f???"@!?f???"@*      ??!       2	?'I?L????'I?L???!?'I?L???:	?{H??_@?{H??_@!?{H??_@B      ??!       J	o?[t???o?[t???!o?[t???R      ??!       Z	o?[t???o?[t???!o?[t???b      ??!       JGPUY?s?uO??b qP?R!P?B@y???#O@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput3?>????!3?>????0"3
Decoder/output/Conv2DConv2D?M?@?R??!???N????0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter( S???!????Z??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterƩ?????!/J)D??0"8
Decoder/conv5_dec/Relu_FusedConv2D?"?U????!Yδ \??"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput?KPꗠ?!]????n??0"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?v??????!̷PWY=??0"8
Decoder/conv4_dec/Relu_FusedConv2D?L1a4??!??c?L???"8
Encoder/conv5_enc/Relu_FusedConv2DR?vkn??!???T3??"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterBE?|??!r??tV??0Q      Y@Y1?f?t@a???y??W@q??"?F??y?a-p???"?

both?Your program is POTENTIALLY input-bound because 13.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?24.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 