?	?@?S??0@?@?S??0@!?@?S??0@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?@?S??0@??6?@1????I'@A???)????Iu?8F??@*	?z??ɺ@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??4}6@!??ZN?X@)??4}6@1??ZN?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??I???!M#K???)??I???1M#K???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismvS?k%t??!?#hY8`??)>>!;oc??1? DU????:Preprocessing2F
Iterator::Model???:TS??!?MTE???)!yv?v?1st-?g???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap????u9@!\dWu?X@)?????g?1??ڨ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 15.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?14.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI4n??>@Qs???~Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??6?@??6?@!??6?@      ??!       "	????I'@????I'@!????I'@*      ??!       2	???)???????)????!???)????:	u?8F??@u?8F??@!u?8F??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q4n??>@ys???~Q@?"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput隝?!???!隝?!???0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltereǥ#F???!'??4Q??0"e
:gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Co˲̪?!?}?`??0"8
Decoder/conv5_dec/Relu_FusedConv2D??r????!???p??"e
:gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput??:?3??!3?8?g??0"3
Decoder/output/Conv2DConv2D?t e3??!?A?%In??0"8
Decoder/conv1_dec/Relu_FusedConv2D?m?R???!???NNm??"e
:gradient_tape/Decoder/conv4_dec/Conv2D/Conv2DBackpropInputConv2DBackpropInput??i?????!B2h??V??0"8
Decoder/conv4_dec/Relu_FusedConv2Dum??!?????2??"e
:gradient_tape/Encoder/conv5_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInput???9????!???????0Q      Y@YG????@a?g??#?W@q̉s?$??y4?'?????"?

both?Your program is POTENTIALLY input-bound because 15.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?14.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 