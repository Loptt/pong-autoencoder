	d?C?]@d?C?]@!d?C?]@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-d?C?]@?S???Y@1?6qr?&@A??s?v???I?4f?@*	??"?.?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorP8??L?@!?!???X@)P8??L?@1?!???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch;?O Š?!?u??du??);?O Š?1?u??du??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?\?	???!???Ϯ??)@ޫV&???1? 5$????:Preprocessing2F
Iterator::Model^/M????!B9z?D??)'??bg?1?W$?_??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???[v?@!??iv?X@)???]/Ma?1?K??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 88.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?B=Z?V@Q_?M.?"@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?S???Y@?S???Y@!?S???Y@      ??!       "	?6qr?&@?6qr?&@!?6qr?&@*      ??!       2	??s?v?????s?v???!??s?v???:	?4f?@?4f?@!?4f?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?B=Z?V@y_?M.?"@