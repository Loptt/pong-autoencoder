	?????=@?????=@!?????=@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?????=@?b??^?@1T?? ?*@A????Or??I??Ϝ??'@*	?5^????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???j?-@!?i4ե?X@)???j?-@1?i4ե?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchv??????!???8??)v??????1???8??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?>rk?m??!?r?m`??)?yUg????1덹??:Preprocessing2F
Iterator::Model???H?,??!C?a?????)?(B?v?u?1??x?S??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?? v?0@!?<?>j?X@)???Z(i?1?,??L#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?39.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI6????K@Q?%?o11F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?b??^?@?b??^?@!?b??^?@      ??!       "	T?? ?*@T?? ?*@!T?? ?*@*      ??!       2	????Or??????Or??!????Or??:	??Ϝ??'@??Ϝ??'@!??Ϝ??'@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q6????K@y?%?o11F@