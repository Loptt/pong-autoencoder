?$	?????3d@4?!??F@????J/`@!?q }8h@	!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-????J/`@8gDi?P@1d@?z??$@AJ?%r??@I3?뤾?H@"p
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?q }8h@K!?K?3@1*X?l:?G@ATs??P?@I?~1[?^@*	?"????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorU??N?`W@!`?4??X@)U??N?`W@1`?4??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?a̖ۢ?!???^??)?a̖ۢ?1???^??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?A_z?s??!????ק??)???2?6??1?>66????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?????`W@!nY?BT?X@)?????w?1?K??"Ly?:Preprocessing2F
Iterator::Model?BW"P???!??4??]??)p	???Jt?1??1??u?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 26.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?53.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI????~T@Q?w??$2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	p@KW??E@???t?@@K!?K?3@!8gDi?P@	!       "$	C??K8=@??BWc?:@d@?z??$@!*X?l:?G@*	!       2$	O???@@w}n̘{??J?%r??@!Ts??P?@:$	Sv?AݧU@?%p??J@3?뤾?H@!?~1[?^@B	!       J	!       R	!       Z	!       b	!       JGPUb q????~T@y?w??$2@?">
Conv2DBackpropInputConv2DBackpropInput	ԧ????!	ԧ????0"@
Conv2DBackpropFilterConv2DBackpropFilter3D|T[)??!????0"$
Conv2DConv2D??̭o??!?<|D???0"(
gradients/AddNAddN[??x????!+?b??"@
ResizeNearestNeighborResizeNearestNeighbor!?wb?x?!?kv????"H
ResizeNearestNeighborGradResizeNearestNeighborGrad?,=r?x?!???q???"*
gradients/AddN_1AddN?ʜ?̀x?!A?Y?o??"8
ResourceApplyAdamResourceApplyAdam?b?H?w?!W[?D-??"D
*ArithmeticOptimizer/AddOpsRewrite_add_2320AddN<C}?\?w?!qESg???"*
gradients/AddN_2AddNg??J?v?!?%Q[????Q      Y@Y??`?VZ??a?|v???X@q?'???@y?(?)??2@"?
both?Your program is POTENTIALLY input-bound because 26.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?53.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?18.8% of Op time on the device used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 