	?ECƣ?\@?ECƣ?\@!?ECƣ?\@	?^l=?ԃ??^l=?ԃ?!?^l=?ԃ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?ECƣ?\@x???-Y@1?N???.&@A'N?w(
??I?n??@Y?Ƅ?K???*	?z?ɳ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorԻx?n@!??I???X@)Իx?n@1??I???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?$?z?ۡ?!?o*J	??)?$?z?ۡ?1?o*J	??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismOv3???!??o޲q??)(G?`Ƅ?1W?Q????:Preprocessing2F
Iterator::Model8?0C㉨?!(?F	?G??)???9?g?1?hp??]??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapi??r?@!r??p?X@)-?\o??`?1?LJ????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?^l=?ԃ?I?P??I?V@Q ??h#@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	x???-Y@x???-Y@!x???-Y@      ??!       "	?N???.&@?N???.&@!?N???.&@*      ??!       2	'N?w(
??'N?w(
??!'N?w(
??:	?n??@?n??@!?n??@B      ??!       J	?Ƅ?K????Ƅ?K???!?Ƅ?K???R      ??!       Z	?Ƅ?K????Ƅ?K???!?Ƅ?K???b      ??!       JGPUY?^l=?ԃ?b q?P??I?V@y ??h#@