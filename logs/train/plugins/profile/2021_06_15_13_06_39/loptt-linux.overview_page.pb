?	?C?H0;@?C?H0;@!?C?H0;@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?C?H0;@&????E,@1Jm 6?@A? ?Ъ?I?g?p@*	Zd;?/?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?Q??K @!d?-e??X@)?Q??K @1d?-e??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch"??`???!?YƧ???)"??`???1?YƧ???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismՑ#?????!]/#???)??}?<??1f	?<M??:Preprocessing2F
Iterator::Model?????=??!??H.???)????oa}?1??LY`??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???*lN @!??n???X@)
??ϛ?t?1??Z?I??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?27.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??0?S@Qe???=?4@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	&????E,@&????E,@!&????E,@      ??!       "	Jm 6?@Jm 6?@!Jm 6?@*      ??!       2	? ?Ъ?? ?Ъ?!? ?Ъ?:	?g?p@?g?p@!?g?p@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??0?S@ye???=?4@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInputR????4??!R????4??0"3
Decoder/output/Conv2DConv2D?y???!?52???0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterf??Q???!???N"a??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????j??!????|??0"g
;gradient_tape/Encoder/conv1_enc/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter_a?qC??!{??곲??0"8
Decoder/conv1_dec/Relu_FusedConv2D?V?x;5??!?_N???"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilters???o??!7?1 m??0"?
igradient_tape/Decoder/output/Conv2D/Conv2DBackpropInput-0-0-TransposeNCHWToNHWC-LayoutOptimizer:TransposeUnknown??hU????!??>7͛??"8
Decoder/conv4_dec/Relu_FusedConv2Da?3??}??!??!?????"e
:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput???Tđ?!??nV????0Q      Y@Yk?4w?_@aɳ???W@qîGdV-??yH??u???"?

both?Your program is POTENTIALLY input-bound because 52.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?27.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 