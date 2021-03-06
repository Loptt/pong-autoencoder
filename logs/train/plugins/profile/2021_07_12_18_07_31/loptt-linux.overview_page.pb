?	Fy?尘S@Fy?尘S@!Fy?尘S@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Fy?尘S@C?+F7@1͑?_-@Ah?RD?U??I???FuFD@*	B`??"?c@2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchV???4???!:A?=&?J@)V???4???1:A?=&?J@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Ƅ?K???!DJ_??T@)?o{??v??1??v?C=@:Preprocessing2F
Iterator::Model?;O<g??!      Y@)?S???1?ւj??0@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 29.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?51.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI???2?_T@Q|?=4{?2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	C?+F7@C?+F7@!C?+F7@      ??!       "	͑?_-@͑?_-@!͑?_-@*      ??!       2	h?RD?U??h?RD?U??!h?RD?U??:	???FuFD@???FuFD@!???FuFD@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???2?_T@y|?=4{?2@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInputY??6???!Y??6???0"3
Decoder/output/Conv2DConv2Dz?x??	??!?C(?W???0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?
d`q??!?$??????0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??p?#ա?!?=(+??0"8
Decoder/conv5_dec/Relu_FusedConv2D???????!,݂Γ???"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput??-????!g?????0"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter,?3?<??!ھ?Z????0"8
Decoder/conv4_dec/Relu_FusedConv2D??g????!WK8??E??"g
;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter{<{֒?!	??s??0"C
%Decoder/coordinate_channel2d_1/concatConcatV2?p+_?Y??!??~?x??Q      Y@Yuk~X? @a?2?tk?V@q5?O?@y?L?%?(??"?

both?Your program is POTENTIALLY input-bound because 29.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?51.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 