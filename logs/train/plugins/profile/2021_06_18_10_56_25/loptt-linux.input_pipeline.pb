$	?????3d@4?!??F@????J/`@!?q }8h@	!       "n
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
$	p@KW??E@???t?@@K!?K?3@!8gDi?P@	!       "$	C??K8=@??BWc?:@d@?z??$@!*X?l:?G@*	!       2$	O???@@w}n̘{??J?%r??@!Ts??P?@:$	Sv?AݧU@?%p??J@3?뤾?H@!?~1[?^@B	!       J	!       R	!       Z	!       b	!       JGPUb q????~T@y?w??$2@