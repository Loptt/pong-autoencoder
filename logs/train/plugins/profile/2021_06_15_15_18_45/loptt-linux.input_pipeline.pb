	?5??dN@?5??dN@!?5??dN@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?5??dN@???
?I@1?????X@A?Z?!???I???]@*	-?????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?P??@!?T \?X@)?P??@1?T \?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?LnY??!0????)?LnY??10????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismv?1<???!??:????)?????&??1?چFN5??:Preprocessing2F
Iterator::Model:τ&???!\O? Ƹ??)??.?.i?1??J[???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?"N'??@!a!?s??X@)Z.??S\?1?d?%???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 85.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI8I???dV@QB?5?C?$@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???
?I@???
?I@!???
?I@      ??!       "	?????X@?????X@!?????X@*      ??!       2	?Z?!????Z?!???!?Z?!???:	???]@???]@!???]@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q8I???dV@yB?5?C?$@