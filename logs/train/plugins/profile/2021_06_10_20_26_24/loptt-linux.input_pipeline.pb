	0?k???R@0?k???R@!0?k???R@	??@c?????@c???!??@c???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails60?k???R@ol?`@1?+??=@A?ND??C@I~6rݔ? @YW??el???*	R????E@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?E? ??!wk?0{L@)?E? ??1wk?0{L@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Ƽ?8d??!W???c?U@)?iT?d??1l??u??>@:Preprocessing2F
Iterator::ModelJ?i?W??!      Y@)H???w?1H??R?l*@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??@c???I%??7SuN@Qê?'ՆC@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	ol?`@ol?`@!ol?`@      ??!       "	?+??=@?+??=@!?+??=@*      ??!       2	?ND??C@?ND??C@!?ND??C@:	~6rݔ? @~6rݔ? @!~6rݔ? @B      ??!       J	W??el???W??el???!W??el???R      ??!       Z	W??el???W??el???!W??el???b      ??!       JGPUY??@c???b q%??7SuN@yê?'ՆC@