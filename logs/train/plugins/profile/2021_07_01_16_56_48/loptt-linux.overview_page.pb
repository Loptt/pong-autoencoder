?	?u?!?1@?u?!?1@!?u?!?1@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?u?!?1@?K???
@1p?4(?'$@A?'?$隙?I?c]?FS@*	??? ???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorG?J??1@!3*??t?X@)G?J??1@13*??t?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?N?j???!R???v???)?N?j???1R???v???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism ?	F???!-?P=????){??v? ??1??[U??:Preprocessing2F
Iterator::Model???Y???!?????S??)?ם?<?|?1?????F??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapt??Y5@!?A???X@)?h9?Cmk?1J????ة?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?23.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIU?`? IE@Q????L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?K???
@?K???
@!?K???
@      ??!       "	p?4(?'$@p?4(?'$@!p?4(?'$@*      ??!       2	?'?$隙??'?$隙?!?'?$隙?:	?c]?FS@?c]?FS@!?c]?FS@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qU?`? IE@y????L@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput?ql9???!?ql9???0"3
Decoder/output/Conv2DConv2D??q? ??!&_??>???0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter~??F8??!?Q9(???0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??i% ??!BUL?,Q??0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput????9??!s
?\bX??0"8
Decoder/conv5_dec/Relu_FusedConv2D?? ????!??|? ??"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterQU??"???!???????0"8
Decoder/conv4_dec/Relu_FusedConv2D?xm.-???!????'??"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterB'??R6??!$??[??0"8
Encoder/conv5_enc/Relu_FusedConv2D??պt???!??=?=???Q      Y@Y1?f?t@a???y??W@q4e??o??y??z??y??"?

both?Your program is POTENTIALLY input-bound because 19.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?23.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 