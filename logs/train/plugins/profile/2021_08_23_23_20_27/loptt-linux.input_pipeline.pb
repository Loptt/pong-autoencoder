	?lmW@?lmW@!?lmW@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?lmW@@?C?]U@1_???j@Ak??**??IXr?? @*	,???ǳ?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??????@!Z???Z?X@)??????@1Z???Z?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??S?K??!????????)??S?K??1????????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism|ds?<G??!IWO????)?5x_???1|?Bk?x??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapÞv?k?@!2s????X@)9Q???k?1w^?r???:Preprocessing2F
Iterator::Model?P????!?3c?YG??)????!9i?1????v??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 91.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?L(?_W@Q1{?6@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	@?C?]U@@?C?]U@!@?C?]U@      ??!       "	_???j@_???j@!_???j@*      ??!       2	k??**??k??**??!k??**??:	Xr?? @Xr?? @!Xr?? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?L(?_W@y1{?6@