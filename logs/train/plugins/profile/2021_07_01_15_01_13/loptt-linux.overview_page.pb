?	C???q@@C???q@@!C???q@@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C???q@@????x@1??);?0(@A??(w+@I????@*		?Zi?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator&?"?d?@!6?2???X@)&?"?d?@16?2???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchf?L2r??!?W?8???)f?L2r??1?W?8???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?E(?????!Q??????)?oB@??1v???U???:Preprocessing2F
Iterator::Model?wb֋???!????X??){O崧?|?1W????!??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?sCS?@!??ō??X@)u?^?1??u????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?12.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI)u?8?O@Q׊ ??cB@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????x@????x@!????x@      ??!       "	??);?0(@??);?0(@!??);?0(@*      ??!       2	??(w+@??(w+@!??(w+@:	????@????@!????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q)u?8?O@y׊ ??cB@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInputt&?`S1??!t&?`S1??0"3
Decoder/output/Conv2DConv2D???}?|??!yC-?G???0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Х&?Ԥ?!??e߂??0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter=??a]>??!p:????0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput</2YG??!?ͦ??0"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter$8??A{??!???????0"8
Decoder/conv5_dec/Relu_FusedConv2Dc?????!??P??`??"8
Decoder/conv4_dec/Relu_FusedConv2DHm??%???!?>?D???"C
%Decoder/coordinate_channel2d_1/concatConcatV2l? 7????!M?????"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??x????!?-?ҿ???0Q      Y@Y˛;??!@a??xj?V@q3?|??I@y?~`TP??"?
both?Your program is POTENTIALLY input-bound because 8.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?12.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?51.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 