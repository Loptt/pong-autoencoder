?	?g???!6@?g???!6@!?g???!6@	>u~?c???>u~?c???!>u~?c???"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?g???!6@????V@1?B?ʠJ-@A???խ???I=D?;??
@Y??`?$???*	P??n???@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorc_??`@!?????X@)c_??`@1?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch
If????!??EP=|??)
If????1??EP=|??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?r?4???!??G?g??)e???? ??1? ????:Preprocessing2F
Iterator::Modelȱ?ᘭ?!??Г@??)'??bw?1qtAB?ȶ?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapqU?wE@!	?_?~?X@);8؛?s?1?`)=?W??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?15.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9>u~?c???IPo$J}@@Qn?.??P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????V@????V@!????V@      ??!       "	?B?ʠJ-@?B?ʠJ-@!?B?ʠJ-@*      ??!       2	???խ??????խ???!???խ???:	=D?;??
@=D?;??
@!=D?;??
@B      ??!       J	??`?$?????`?$???!??`?$???R      ??!       Z	??`?$?????`?$???!??`?$???b      ??!       JGPUY>u~?c???b qPo$J}@@yn?.??P@?"E
#PerceptualNetwork/block1_conv2/Relu_FusedConv2D???PG>??!???PG>??"G
%PerceptualNetwork/block1_conv2/Relu_1_FusedConv2D8?|?+*??!???y94??"r
Ggradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?i 4d???!]???????0"E
#PerceptualNetwork/block2_conv2/Relu_FusedConv2D(ͮ?????!????????"G
%PerceptualNetwork/block2_conv2/Relu_1_FusedConv2D???߷??!o??????"r
Ggradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??2]8D??!???W??0"d
8gradient_tape/Decoder/output/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???S????!?`?????0"E
#PerceptualNetwork/block2_conv1/Relu_FusedConv2D?x?Hǡ?!͏[t"??"G
%PerceptualNetwork/block2_conv1/Relu_1_FusedConv2D_?<????!????A??"r
Ggradient_tape/PerceptualNetwork/block2_conv1/Conv2D/Conv2DBackpropInputConv2DBackpropInputO??o+???!t???!1??0Q      Y@Y?Kh/??$@a???KhV@qs?f????y?x?j???"?

both?Your program is POTENTIALLY input-bound because 17.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?15.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 