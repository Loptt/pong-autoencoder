		4??`@	4??`@!	4??`@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-	4??`@???]@1.T????%@Ah?.?K??IC:<??@*	????2:?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?/g??@!?5o?X@)?/g??@1?5o?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?f??I}??!?6????)?f??I}??1?6????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismP?Lۿ???!???,??)<??kЇ?1?nWnؘ??:Preprocessing2F
Iterator::Model??????!?]?6r???)4Lm???n?1?i?;????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap	???c?@!DK?U?X@)?4`??ie?1
???c???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI? z???V@Q??.d?? @Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???]@???]@!???]@      ??!       "	.T????%@.T????%@!.T????%@*      ??!       2	h?.?K??h?.?K??!h?.?K??:	C:<??@C:<??@!C:<??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q? z???V@y??.d?? @