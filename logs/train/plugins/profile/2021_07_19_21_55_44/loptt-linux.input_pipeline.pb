	?z2??[@?z2??[@!?z2??[@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?z2??[@9`W???X@1j?t?T@AV???4??I?J?(?@*	?l??i?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorT???M@! e1M??X@)T???M@1 e1M??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch? %̴??!Y?SWu??)? %̴??1Y?SWu??:Preprocessing2F
Iterator::Model?˵hڮ?!?ߴ=???)?Gqh??1?O?????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism()? ???!??)I???)R_?vj.??13!?Ư??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?.6?R@!?,	נ?X@)?? =Eq?1?_'???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 91.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??????W@Q???0ԁ@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	9`W???X@9`W???X@!9`W???X@      ??!       "	j?t?T@j?t?T@!j?t?T@*      ??!       2	V???4??V???4??!V???4??:	?J?(?@?J?(?@!?J?(?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??????W@y???0ԁ@