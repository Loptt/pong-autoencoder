	?`6??)@?`6??)@!?`6??)@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?`6??)@(a??_?@1?͌~4?@A}??O9&??I?im?{@*	e;?O?-?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??E_A?@!z?=Z?X@)??E_A?@1z?=Z?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?qQ-"???!?<????)?qQ-"???1?<????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?????U??!/?k\???)?q?_!??1?Z?? Q??:Preprocessing2F
Iterator::Modelv5y?j??!??鹏???){?%T??1?S?7?|??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?q?߅?@!x,????X@)?v?$j?1?Əe??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?37.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIƗX??K@Q:h??EF@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	(a??_?@(a??_?@!(a??_?@      ??!       "	?͌~4?@?͌~4?@!?͌~4?@*      ??!       2	}??O9&??}??O9&??!}??O9&??:	?im?{@?im?{@!?im?{@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qƗX??K@y:h??EF@