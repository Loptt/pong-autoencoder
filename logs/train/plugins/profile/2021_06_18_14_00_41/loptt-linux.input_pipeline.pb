	??q4GQi@??q4GQi@!??q4GQi@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-??q4GQi@˄_??^@1?)??y7@A`?5?!:@I???~39J@*	V-?M??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?c???s?@!??/?{?X@)?c???s?@1??/?{?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch]?6??n??!???5?α?)]?6??n??1???5?α?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???0??!5?V????)pA?,_??1nv?????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??E??t?@!??%|[?X@)ۈ'??q?1??"?~???:Preprocessing2F
Iterator::Model??"[A??!d?i???)???W?q?1N??Q-???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 59.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?25.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???T6V@Q氃XM.'@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	˄_??^@˄_??^@!˄_??^@      ??!       "	?)??y7@?)??y7@!?)??y7@*      ??!       2	`?5?!:@`?5?!:@!`?5?!:@:	???~39J@???~39J@!???~39J@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???T6V@y氃XM.'@