?	?L?:y@@?L?:y@@!?L?:y@@	=??????=??????!=??????"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?L?:y@@XƆn? @1:ZՒ~@A	5C?(?4@I?.4?id@Y???9??*	?Zd;K@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch!?????!e???P@)!?????1e???P@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism0+?~N??!???	"xV@)>?h??i??1]i?t?6@:Preprocessing2F
Iterator::Model???n,(??!      Y@)?C?.l?v?1Z???>$@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?11.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9=??????ImLO?RT@Q?c?2@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	XƆn? @XƆn? @!XƆn? @      ??!       "	:ZՒ~@:ZՒ~@!:ZՒ~@*      ??!       2		5C?(?4@	5C?(?4@!	5C?(?4@:	?.4?id@?.4?id@!?.4?id@B      ??!       J	???9?????9??!???9??R      ??!       Z	???9?????9??!???9??b      ??!       JGPUY=??????b qmLO?RT@y?c?2@?"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???`??!???`??0"3
Decoder/output/Conv2DConv2D^??i??!~?????0"g
;gradient_tape/Decoder/conv1_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?]?2ބ??!?X;?????0"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput??mHb??!?;??????0"8
Decoder/conv4_dec/Relu_FusedConv2D<-?a????!??ۖ;:??"e
:gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropInputConv2DBackpropInput?+ʘr???!???????0"g
;gradient_tape/Decoder/conv2_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter߭??????!^?(? ???0"g
;gradient_tape/Encoder/conv2_enc/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterH?rf??!?S?? ??0"8
Decoder/conv1_dec/Relu_FusedConv2D)????!b?##?`??"8
Encoder/conv4_enc/Relu_FusedConv2Dz?t??]??!:
?̶??Q      Y@Y??|??|@a?3X?3?W@q?????X@y4?H?Jة?"?
both?Your program is POTENTIALLY input-bound because 6.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?11.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?98.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 