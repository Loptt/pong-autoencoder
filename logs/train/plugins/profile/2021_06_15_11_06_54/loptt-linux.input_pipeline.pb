	???T?+P@???T?+P@!???T?+P@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-???T?+P@??f? L@1?xy:W?@A?c?ߣ?I??z6+@*	?????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?B]r@!?XA)?X@)?B]r@1?XA)?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch7?xͫ:??!??????)7?xͫ:??1??????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism^??????!|??????)	?L?n??1愗?x??:Preprocessing2F
Iterator::Model???_???!??@B???)?KU??o?1W??wu??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapdʇ?jt@!?~?{U?X@)??-?l`?1?K???a??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 87.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI*?4j,?V@Q??Y???!@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??f? L@??f? L@!??f? L@      ??!       "	?xy:W?@?xy:W?@!?xy:W?@*      ??!       2	?c?ߣ??c?ߣ?!?c?ߣ?:	??z6+@??z6+@!??z6+@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q*?4j,?V@y??Y???!@