	???.[@???.[@!???.[@	v@b}?v@b}?!v@b}?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6???.[@! _BlW@1uF^?)@A??R%?ޢ?IV???؟@Y?}͑??*	ףp=ꄽ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????c?@!??r??X@)????c?@1??r??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchWBwI???!L?tsB??)WBwI???1L?tsB??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??Դ?i??!???5'??)X???ާ??1,?????:Preprocessing2F
Iterator::Model4`?_???!?????)-?\o??p?1?B?͏??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??nJ?@!??????X@)??IӠh^?1???T&??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 86.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9v@b}?I~'??aV@Q<3?O'@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	! _BlW@! _BlW@!! _BlW@      ??!       "	uF^?)@uF^?)@!uF^?)@*      ??!       2	??R%?ޢ???R%?ޢ?!??R%?ޢ?:	V???؟@V???؟@!V???؟@B      ??!       J	?}͑???}͑??!?}͑??R      ??!       Z	?}͑???}͑??!?}͑??b      ??!       JGPUYv@b}?b q~'??aV@y<3?O'@