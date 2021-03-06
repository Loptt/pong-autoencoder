?	?a??mAb@?a??mAb@!?a??mAb@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?a??mAb@??U?)U@1S[? ?3@A?}(F@I? ϠC@*	!?rh??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??AA)?;@!??j;+?X@)??AA)?;@1??j;+?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchO!W?Y??!!"???K??)O!W?Y??1!"???K??:Preprocessing2F
Iterator::Model??jGq???!\?Zs????)Q0c
֘?1??Wh??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??=]ݱ??!1?:vG??)???????1@?sW????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap)???^?;@!???>B?X@)_???:Ts?1]2p??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 58.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?26.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???-?U@Q??S??*@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??U?)U@??U?)U@!??U?)U@      ??!       "	S[? ?3@S[? ?3@!S[? ?3@*      ??!       2	?}(F@?}(F@!?}(F@:	? ϠC@? ϠC@!? ϠC@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???-?U@y??S??*@?">
Conv2DBackpropInputConv2DBackpropInput?]P????!?]P????0"@
Conv2DBackpropFilterConv2DBackpropFilterUDp?hb??!???'??0"$
Conv2DConv2D???M???!?Zgpm???0"D
*ArithmeticOptimizer/AddOpsRewrite_add_2320AddNSU?͒?!ί̡H???"H
ResizeNearestNeighborGradResizeNearestNeighborGradg??yɑ?!t(l)????"@
ResizeNearestNeighborResizeNearestNeighborr??VH??!?%??^???"8
ResourceApplyAdamResourceApplyAdam??6Ǟ8??!;??u????",
MaxPoolGradMaxPoolGradAZ_?G??!?_?%???"&
ReluGradReluGradR?2?????!4??i??",
BiasAddGradBiasAddGrad??>?Eg??!??M?]??Q      Y@Y????a?@aŴ??iW@q+??`??2@yjsY???J@"?
both?Your program is POTENTIALLY input-bound because 58.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?26.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?18.8% of Op time on the host used eager execution. 53.8% of Op time on the device used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 