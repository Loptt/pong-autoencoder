	?B???`@?B???`@!?B???`@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?B???`@?q??r?]@1?G'@A??uR_???I?Ov@*	?&1ܳ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorfI??Z&@!/y?Y?X@)fI??Z&@1/y?Y?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??,?s???!?}?Ȏ??)
?_??͘?1Нn?}??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????@g??!?*?????)????@g??1?*?????:Preprocessing2F
Iterator::Model?,C????!??i??"??)??f?v?d?1wǀI?<??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapi?x?J(@!?,?غ?X@)?#????^?1蝥???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI????V@Q7?!?1_!@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q??r?]@?q??r?]@!?q??r?]@      ??!       "	?G'@?G'@!?G'@*      ??!       2	??uR_?????uR_???!??uR_???:	?Ov@?Ov@!?Ov@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q????V@y7?!?1_!@