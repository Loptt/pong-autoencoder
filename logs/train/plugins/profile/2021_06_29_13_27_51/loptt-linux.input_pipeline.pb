	?!?k^i2@?!?k^i2@!?!?k^i2@	($-???($-???!($-???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?!?k^i2@???_?@1?&l??$@Ax??,???IY???@Y}>ʈ@??*	=
ףМ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator3????@!w[_#??X@)3????@1w[_#??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch? ??*???!?D?????)? ??*???1?D?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism`"ĕ??!:??#)??)?h8en??1??=?0??:Preprocessing2F
Iterator::Model??`?d??!C?Y\???)~t??gy~?1.9شS6??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??(_??@!?ә??X@)????9]f?18}??YC??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?23.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9($-???I?A&ϴE@Q????0&L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???_?@???_?@!???_?@      ??!       "	?&l??$@?&l??$@!?&l??$@*      ??!       2	x??,???x??,???!x??,???:	Y???@Y???@!Y???@B      ??!       J	}>ʈ@??}>ʈ@??!}>ʈ@??R      ??!       Z	}>ʈ@??}>ʈ@??!}>ʈ@??b      ??!       JGPUY($-???b q?A&ϴE@y????0&L@