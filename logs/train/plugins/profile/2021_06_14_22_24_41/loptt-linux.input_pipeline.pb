	@?#H?2O@@?#H?2O@!@?#H?2O@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-@?#H?2O@|&??i?I@1??*?\@A????̓??Iu??<~@*	q=
???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????H@!J????X@)????H@1J????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch*??g\??!&?;k????)*??g\??1&?;k????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??????!9FL?=???)܂??????1??
???:Preprocessing2F
Iterator::Model??+,???!??jj"q??)?#?@?x?13??K??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapn5???K@!5TVv;?X@)?vöEi?1P3????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 83.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?7.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?L???V@Q??]??"@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	|&??i?I@|&??i?I@!|&??i?I@      ??!       "	??*?\@??*?\@!??*?\@*      ??!       2	????̓??????̓??!????̓??:	u??<~@u??<~@!u??<~@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?L???V@y??]??"@