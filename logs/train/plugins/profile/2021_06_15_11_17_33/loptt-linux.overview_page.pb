?	?<?6M@?<?6M@!?<?6M@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?<?6M@?@?G@1?2?,%@AŪA?۽??IM??.?@*	E????l?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??S? ?@!f?c9?X@)??S? ?@1f?c9?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Ü?M??!??o.q???)??OVW??1?m]?]??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??6?4D??!a???6???)??6?4D??1a???6???:Preprocessing2F
Iterator::Model??Kǜ??! ?ц???)?$?j?d?1???*Za??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??j?#?@!Q?]???X@)?M???a?1U????[??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 82.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noID????V@Q?-?1W?#@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?@?G@?@?G@!?@?G@      ??!       "	?2?,%@?2?,%@!?2?,%@*      ??!       2	ŪA?۽??ŪA?۽??!ŪA?۽??:	M??.?@M??.?@!M??.?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qD????V@y?-?1W?#@?"e
:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput&?l???!&?l???0"8
Decoder/conv5_dec/Relu_FusedConv2D?՜??/??!p?RK~??"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput???????!?Ҡ???0"e
:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInput??8a
??!?X??C???0"8
Decoder/conv4_dec/Relu_FusedConv2D	??ª?!???Ib;??"8
Encoder/conv5_enc/Relu_FusedConv2D???婤?!f?D?????"e
:gradient_tape/Decoder/conv3_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput????????!N??%????0"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput;??q?;??!???KD??0"e
:gradient_tape/Encoder/conv4_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInputد?7iښ?!z:p????0"8
Decoder/conv3_dec/Relu_FusedConv2D?ed?[???!h??-{??Q      Y@Y6?d?M6@am??&??W@q4o??o=!@yck*^o??"?

both?Your program is POTENTIALLY input-bound because 82.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 