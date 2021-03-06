?	?r?w??3@?r?w??3@!?r?w??3@	????{u??????{u??!????{u??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?r?w??3@f?????@1>???4 ,@A      ??I|,}??:@Y????????*	?S??[??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???#b?@!??????X@)???#b?@1??????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???????!ĵ?f}???)???????1ĵ?f}???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???l ??!?k?-?2??)=?!7???1?kۍ)???:Preprocessing2F
Iterator::Model9{????!???????)˟ov?1Ep???ڶ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapq;4,F?@!??d ??X@)?Ŧ?B g?1?@~????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 13.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?14.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9????{u??Ix???gs<@Q?c!";?Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	f?????@f?????@!f?????@      ??!       "	>???4 ,@>???4 ,@!>???4 ,@*      ??!       2	      ??      ??!      ??:	|,}??:@|,}??:@!|,}??:@B      ??!       J	????????????????!????????R      ??!       Z	????????????????!????????b      ??!       JGPUY????{u??b qx???gs<@y?c!";?Q@?"E
#PerceptualNetwork/block1_conv2/Relu_FusedConv2D??C:???!??C:???"G
%PerceptualNetwork/block1_conv2/Relu_1_FusedConv2D˻?"q???!4?)??Ȼ?"r
Ggradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInputJG?7C&??!lE?????0"r
Ggradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput훗?Z|??!g,?X???0"E
#PerceptualNetwork/block2_conv2/Relu_FusedConv2D???̐h??!fTF?s??"G
%PerceptualNetwork/block2_conv2/Relu_1_FusedConv2D?6?f?_??!B{?2???"`
?gradient_tape/PerceptualNetwork/block1_pool/MaxPool/MaxPoolGradMaxPoolGrad??.?}???!BM?????"E
#PerceptualNetwork/block2_conv1/Relu_FusedConv2D?e????!?p?QiG??"G
%PerceptualNetwork/block2_conv1/Relu_1_FusedConv2D????=??!?N??o??"r
Ggradient_tape/PerceptualNetwork/block2_conv1/Conv2D/Conv2DBackpropInputConv2DBackpropInput{?k????!??Jn??0Q      Y@Y?Kh/??$@a???KhV@q?W'?z??y?[??Y???"?

both?Your program is POTENTIALLY input-bound because 13.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?14.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 