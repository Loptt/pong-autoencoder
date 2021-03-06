?	??K8?.@??K8?.@!??K8?.@	????????????????!????????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??K8?.@???uZ@1?}?e??"@AGɫsȞ?I?^?@YY??????*	?O????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorѱ?J\G@!??_F??X@)ѱ?J\G@1??_F??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchr?30????!?????)??)r?30????1?????)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?&p?n??!a-E???)@?0`?U??1|?ʦ????:Preprocessing2F
Iterator::Model?CP5z5??!%6?˯??)??R?h|?1M?1?"???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?p??K@!'??@??X@)???2#r?1????ӟ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 16.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?20.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????????I?E?><?B@Q?????"O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???uZ@???uZ@!???uZ@      ??!       "	?}?e??"@?}?e??"@!?}?e??"@*      ??!       2	GɫsȞ?GɫsȞ?!GɫsȞ?:	?^?@?^?@!?^?@B      ??!       J	Y??????Y??????!Y??????R      ??!       Z	Y??????Y??????!Y??????b      ??!       JGPUY????????b q?E?><?B@y?????"O@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput???;I???!???;I???0"3
Decoder/output/Conv2DConv2DQ??H??!<?.J???0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter΢N.???!??P?g??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???ա???!+??E??0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput0?lz???!:?2??E??0"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Aປ?!?)Cσ??0"8
Decoder/conv5_dec/Relu_FusedConv2D???rG??!???????"8
Decoder/conv4_dec/Relu_FusedConv2DXQ?r???!?
%?D??"g
;gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterb?x?k???!e??????0"g
;gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterw?????!?ұ4????0Q      Y@Y1?f?t@a???y??W@qF?^a??y????k???"?

both?Your program is POTENTIALLY input-bound because 16.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?20.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 