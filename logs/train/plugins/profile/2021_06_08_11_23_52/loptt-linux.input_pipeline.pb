	?x]P@?x]P@!?x]P@	?֬??????֬?????!?֬?????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?x]P@?d ??v@1???
G?(@A??? ?T3@I9'0??8@Y??Z?a/??*	?n???^@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?<֌??!kS? ??L@)?<֌??1kS? ??L@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismF"4?????!?UPxQV@)_??W???1?ss????@:Preprocessing2F
Iterator::Modell#??fF??!      Y@)-	PS?֊?1??W}=t%@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 12.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?38.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?֬?????I?L?D?-T@Q!t?"3@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?d ??v@?d ??v@!?d ??v@      ??!       "	???
G?(@???
G?(@!???
G?(@*      ??!       2	??? ?T3@??? ?T3@!??? ?T3@:	9'0??8@9'0??8@!9'0??8@B      ??!       J	??Z?a/????Z?a/??!??Z?a/??R      ??!       Z	??Z?a/????Z?a/??!??Z?a/??b      ??!       JGPUY?֬?????b q?L?D?-T@y!t?"3@