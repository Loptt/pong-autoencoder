	34??u\@34??u\@!34??u\@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-34??u\@?[?nK?Y@1?0Bx?!@A??f?|??I????@*	i??|?5?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorR????a@!0?@?A?X@)R????a@10?@?A?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchp[[x^*??!?4+???)p[[x^*??1?4+???:Preprocessing2F
Iterator::Model???׷?!? ??{??)
?]?V??1?e?6e??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism0g?+????!????.???)??E|'f??1M?,????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?1?	?d@!|H??X@)?N"¿j?1?_"9?b??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 90.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?Iv??W@Q?f??O?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?[?nK?Y@?[?nK?Y@!?[?nK?Y@      ??!       "	?0Bx?!@?0Bx?!@!?0Bx?!@*      ??!       2	??f?|????f?|??!??f?|??:	????@????@!????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Iv??W@y?f??O?@