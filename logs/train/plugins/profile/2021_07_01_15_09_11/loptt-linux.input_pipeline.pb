	\?	???d@\?	???d@!\?	???d@	ā;Z?
??ā;Z?
??!ā;Z?
??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6\?	???d@?!8.c?b@1?%9`W?)@A????B???I??o???@Y@??T???*	?rh?m??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorC???? @!????׫X@)C???? @1????׫X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??HV??!:??v????)??HV??1:??v????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?|~!??!D??]$??)?f?????1?ވ?`??:Preprocessing2F
Iterator::Modelt	??????!ɯ??U??)()? ?l?1M?\v3??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??W??"@!A?????X@)??ǘ??`?1?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 90.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ā;Z?
??I??&?W@QS/??I?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?!8.c?b@?!8.c?b@!?!8.c?b@      ??!       "	?%9`W?)@?%9`W?)@!?%9`W?)@*      ??!       2	????B???????B???!????B???:	??o???@??o???@!??o???@B      ??!       J	@??T???@??T???!@??T???R      ??!       Z	@??T???@??T???!@??T???b      ??!       JGPUYā;Z?
??b q??&?W@yS/??I?@