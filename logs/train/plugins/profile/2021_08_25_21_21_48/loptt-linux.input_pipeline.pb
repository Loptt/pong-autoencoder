	Uj?@a@Uj?@a@!Uj?@a@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Uj?@a@?B??^@1???_{)@A@?3iSu??I???uo?
@*	Zd;???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator@?j??@!:)?@??X@)@?j??@1:)?@??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchD???XP??!? 2??)D???XP??1? 2??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?h?????!???????)p??-??1BR`??D??:Preprocessing2F
Iterator::Model?+?????!? ?????)???mj?1?Q?,?3??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???=??@!}???X@)a??q6]?1?2ߺ#??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI????a?V@Qy????"@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?B??^@?B??^@!?B??^@      ??!       "	???_{)@???_{)@!???_{)@*      ??!       2	@?3iSu??@?3iSu??!@?3iSu??:	???uo?
@???uo?
@!???uo?
@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????a?V@yy????"@