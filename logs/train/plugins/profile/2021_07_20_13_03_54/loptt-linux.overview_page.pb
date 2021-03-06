?	?`6??)@?`6??)@!?`6??)@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?`6??)@(a??_?@1?͌~4?@A}??O9&??I?im?{@*	e;?O?-?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??E_A?@!z?=Z?X@)??E_A?@1z?=Z?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?qQ-"???!?<????)?qQ-"???1?<????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?????U??!/?k\???)?q?_!??1?Z?? Q??:Preprocessing2F
Iterator::Modelv5y?j??!??鹏???){?%T??1?S?7?|??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?q?߅?@!x,????X@)?v?$j?1?Əe??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?37.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIƗX??K@Q:h??EF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	(a??_?@(a??_?@!(a??_?@      ??!       "	?͌~4?@?͌~4?@!?͌~4?@*      ??!       2	}??O9&??}??O9&??!}??O9&??:	?im?{@?im?{@!?im?{@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qƗX??K@y:h??EF@?"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput?E?D???!?E?D???0"8
Decoder/conv5_dec/Relu_FusedConv2D??%?????!ؖ?λ?"8
Decoder/conv4_dec/Relu_FusedConv2D???l?֨?!??;????"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterX5p}???!n
I?Z??0"g
;gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter:O_Ǥ?!<?)h???0"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput?? ?2N??!?Rm????0"e
:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInput?l???!??1?'??0"e
:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput5??]?V??!??? ??0"8
Encoder/conv5_enc/Relu_FusedConv2D???yާ??!O?|~???"S
2gradient_tape/Encoder/maxpool5/MaxPool/MaxPoolGradMaxPoolGrad???<????!Z?IU????Q      Y@Y??8??8@a?q?q?W@qFe?ʪ??y ?%<UU??"?

both?Your program is POTENTIALLY input-bound because 17.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?37.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 