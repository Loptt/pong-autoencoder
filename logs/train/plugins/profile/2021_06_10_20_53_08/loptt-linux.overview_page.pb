?	??[?8@@??[?8@@!??[?8@@	???x.?????x.??!???x.??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??[?8@@??߆/	@1?W?2?U;@AK?b??¢?I??A_z???Y7???0??*	X9??6??@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?M?q"@!???,?X@)?M?q"@1???,?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetche??k]j??!??^?h??)e??k]j??1??^?h??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????`??!A??{gC??)?@??ǘ??1I???_<??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap??/g&@![+C?X@)???z?2q?1$?>l?Y??:Preprocessing2F
Iterator::Modelɑ???ˢ?!?Rjw|???)??#?k?1?E?O???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???x.??I????./@Q?'2?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??߆/	@??߆/	@!??߆/	@      ??!       "	?W?2?U;@?W?2?U;@!?W?2?U;@*      ??!       2	K?b??¢?K?b??¢?!K?b??¢?:	??A_z?????A_z???!??A_z???B      ??!       J	7???0??7???0??!7???0??R      ??!       Z	7???0??7???0??!7???0??b      ??!       JGPUY???x.??b q????./@y?'2?U@?"r
Ggradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput????J??!????J??0"E
#PerceptualNetwork/block1_conv2/Relu_FusedConv2Dԩ?ھ??!?y??*??"G
%PerceptualNetwork/block1_conv2/Relu_1_FusedConv2D?qY????!H??T????";
"gradient_tape/grayscale_to_rgb/SumSumd?.}???!?Q??Io??"G
%PerceptualNetwork/block2_conv2/Relu_1_FusedConv2D???;<צ?!?&IS???"E
#PerceptualNetwork/block2_conv2/Relu_FusedConv2D? /?=զ?!??4???"r
Ggradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput???i}??!?? ռ??0"G
%PerceptualNetwork/block2_conv1/Relu_1_FusedConv2D?x4??!z?bd???"E
#PerceptualNetwork/block2_conv1/Relu_FusedConv2D*;)?B???!-9E????"b
7gradient_tape/Decoder/output/Conv2D/Conv2DBackpropInputConv2DBackpropInput?8?rh??!??ofY??0Q      Y@Y/?袋."@a?袋.?V@q?0?1Z??y???2?~?"?

both?Your program is POTENTIALLY input-bound because 9.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?5.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 