	?u?!?1@?u?!?1@!?u?!?1@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?u?!?1@?K???
@1p?4(?'$@A?'?$隙?I?c]?FS@*	??? ???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorG?J??1@!3*??t?X@)G?J??1@13*??t?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?N?j???!R???v???)?N?j???1R???v???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism ?	F???!-?P=????){??v? ??1??[U??:Preprocessing2F
Iterator::Model???Y???!?????S??)?ם?<?|?1?????F??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapt??Y5@!?A???X@)?h9?Cmk?1J????ة?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?23.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIU?`? IE@Q????L@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?K???
@?K???
@!?K???
@      ??!       "	p?4(?'$@p?4(?'$@!p?4(?'$@*      ??!       2	?'?$隙??'?$隙?!?'?$隙?:	?c]?FS@?c]?FS@!?c]?FS@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qU?`? IE@y????L@