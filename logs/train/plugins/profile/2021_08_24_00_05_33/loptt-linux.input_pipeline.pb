	?y?CnT@?y?CnT@!?y?CnT@	???n?n?????n?n??!???n?n??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?y?CnT@?????R@1j??ߪ@A?#EdXţ?I?v??-u??Y^M??????*	?z?gײ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?w?7N
@!.????X@)?w?7N
@1.????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch? ?bG???!?5?)???)? ?bG???1?5?)???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??\????!#?:2?1??)?}?<???1qKdE???:Preprocessing2F
Iterator::Model???????!????#??)OYM?]g?1??_XF??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMaplЗ??@!	??p?X@)?\??7?e?1???(????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???n?n??I???k?W@Q???Pz?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????R@?????R@!?????R@      ??!       "	j??ߪ@j??ߪ@!j??ߪ@*      ??!       2	?#EdXţ??#EdXţ?!?#EdXţ?:	?v??-u???v??-u??!?v??-u??B      ??!       J	^M??????^M??????!^M??????R      ??!       Z	^M??????^M??????!^M??????b      ??!       JGPUY???n?n??b q???k?W@y???Pz?@