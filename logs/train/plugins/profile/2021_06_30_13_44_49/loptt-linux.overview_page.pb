?	~V?)?;1@~V?)?;1@!~V?)?;1@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-~V?)?;1@???? |	@1????z#@A??p???I^J]2?@*	??????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorj4??@!??7c??X@)j4??@1??7c??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?#K???!LE?bz&??)?#K???1LE?bz&??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?c[????!ֽ?????)iƢ??d??1^6?P?7??:Preprocessing2F
Iterator::Model?t?i???!ؑ+]F??)n??KX{?1?N췼?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???ŝ@!ܨE?s?X@)~9?]?f?1??nP+??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?24.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?--???E@Q1???~BL@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???? |	@???? |	@!???? |	@      ??!       "	????z#@????z#@!????z#@*      ??!       2	??p?????p???!??p???:	^J]2?@^J]2?@!^J]2?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?--???E@y1???~BL@?"m
>gradient_tape/Decoder/upsamp1/resize/ResizeNearestNeighborGradResizeNearestNeighborGrad!9?????!!9?????"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Ҍ??9??!??:????0"3
Decoder/output/Conv2DConv2De??ި?!]??&?$??0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??Kɼ??!}?NPQ???0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput??t:-??!_q??????0"8
Decoder/conv5_dec/Relu_FusedConv2D??? ?՚?!???SO??"8
Decoder/conv4_dec/Relu_FusedConv2D4??/?/??!?V?sL???"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterD??????!??ϗt8??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?MRg?'??!?
+????0"e
:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput??k?4͓?!???????0Q      Y@Y/??s7@am??ǈ?W@q??U????y?e3???"?

both?Your program is POTENTIALLY input-bound because 18.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?24.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 