	\t??z37@\t??z37@!\t??z37@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-\t??z37@???ץ@18??9@?%@A??R?h??IRE?*?@*	????X??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator????>9 @!??<??X@)????>9 @1??<??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism}v?uŌ??!?f?Ln???)?-?熦??1g_??A???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch4??s??!|ۈ?5G??)4??s??1|ۈ?5G??:Preprocessing2F
Iterator::Modely?@e????!?u?*??)?W?ۼ??1z2fv!??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??K?A; @!??Ū?X@)????p?1??5?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 27.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?25.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?FxS~?J@Q????MG@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???ץ@???ץ@!???ץ@      ??!       "	8??9@?%@8??9@?%@!8??9@?%@*      ??!       2	??R?h????R?h??!??R?h??:	RE?*?@RE?*?@!RE?*?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?FxS~?J@y????MG@