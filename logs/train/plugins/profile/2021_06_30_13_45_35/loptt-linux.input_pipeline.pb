	P?eoE;@P?eoE;@!P?eoE;@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-P?eoE;@>?#d?@1qZ????%@A??w???I??Y?'@*	;?O???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?"?tu?@!호???X@)?"?tu?@1호???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?Y-??D??!9:HE&???)?Y-??D??19:HE&???:Preprocessing2F
Iterator::Model?$???!??U?$??)X????W??1?(U?>??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??f?R@??!????(??)?>??,???1>R??b??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapM!u;?@!\?"?o?X@)?X?O0n?1?{{PCȫ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?42.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIA=???M@Q????>$D@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>?#d?@>?#d?@!>?#d?@      ??!       "	qZ????%@qZ????%@!qZ????%@*      ??!       2	??w?????w???!??w???:	??Y?'@??Y?'@!??Y?'@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qA=???M@y????>$D@