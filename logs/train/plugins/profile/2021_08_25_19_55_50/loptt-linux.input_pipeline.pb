	?????Mb@?????Mb@!?????Mb@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?????Mb@?????`@19??!?%@A?b?D(??I {?\?@*	.???/?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??.?uw@!?J?q-?X@)??.?uw@1?J?q-?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?~? ???!4u?M$??)?~? ???14u?M$??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism(???֣?!A??&???)?Z?Qf??1.??????:Preprocessing2F
Iterator::ModeloEb????!PG?Ȭ???)~t??gyn?1w??5d??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?uiy@!q?n???X@)?E?n?1_?1????٣?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 90.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI]QR<L"W@Q6??:<?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????`@?????`@!?????`@      ??!       "	9??!?%@9??!?%@!9??!?%@*      ??!       2	?b?D(???b?D(??!?b?D(??:	 {?\?@ {?\?@! {?\?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q]QR<L"W@y6??:<?@