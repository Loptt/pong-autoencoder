	r?z?f?N@r?z?f?N@!r?z?f?N@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-r?z?f?N@?? k՞@1ڎ???;K@A?x?JxB??Iڐf?@*	/?$F??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator u?)@!?qBX*?X@) u?)@1?qBX*?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchиp $??!??p?5???)иp $??1??p?5???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?RD?U???!?jx?????)^?/?ۆ?1C??????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?0E?4.@!?p?qr?X@)?0??Zq?1???Cf ??:Preprocessing2F
Iterator::Modely??"????!??G?F??)?????l?1???,Ib??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?E?cI:&@QGW?Ӷ8V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?? k՞@?? k՞@!?? k՞@      ??!       "	ڎ???;K@ڎ???;K@!ڎ???;K@*      ??!       2	?x?JxB???x?JxB??!?x?JxB??:	ڐf?@ڐf?@!ڐf?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?E?cI:&@yGW?Ӷ8V@