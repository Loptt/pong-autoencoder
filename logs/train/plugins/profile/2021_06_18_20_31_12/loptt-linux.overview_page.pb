?  *	1?"%?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?מY?=@!?M?0?X@)?מY?=@1?M?0?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??\??ʎ?!?6?O˩?)??\??ʎ?1?6?O˩?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism"¿3??!?u?&???)??"?????1ܴ???l??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapj??%?=@!X??X@)1zn?+q?1????6Č?:Preprocessing2F
Iterator::Model???f?8??!????'???)i;???.h?1?W?H?A??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.?">
Conv2DBackpropInputConv2DBackpropInput?~f??N??!?~f??N??0"@
Conv2DBackpropFilterConv2DBackpropFilter?cަM??!m???Ӻ??0"$
Conv2DConv2Dx??S?^??!???~???0"8
ResourceApplyAdamResourceApplyAdam???gJ???!%?$oӠ??"H
ResizeNearestNeighborGradResizeNearestNeighborGrad-??????!?,A5s???"@
ResizeNearestNeighborResizeNearestNeighborWn??Ft??! ?]i???",
MaxPoolGradMaxPoolGrad?Q?????!(@`E?>??"&
ReluGradReluGrad??_sC9??! ??`????",
BiasAddGradBiasAddGrad??S?s??!?Ϙ#\??"$
BiasAddBiasAdd?ݵ?6R??!?~N?????Y?X?#??O@a9??u<B@q,?ŠJ2@y      Y@"?	
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?18.1% of Op time on the host used eager execution. 100.0% of Op time on the device used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.