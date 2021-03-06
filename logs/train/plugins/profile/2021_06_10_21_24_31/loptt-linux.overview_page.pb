?	^?9?S?f@^?9?S?f@!^?9?S?f@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-^?9?S?f@;???]@1?f??IM@Ag?!?{??I???s@*	9??v~?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generatorr???@!??~??X@)r???@1??~??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?]L3????!?O̅?7??)?]L3????1?O̅?7??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism5?磌???!6?
U???)??)x
??1`9,e??:Preprocessing2F
Iterator::Model????=??!?ob?F???)}??z?Vh?1??????:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap5???@!!;Sr??X@)?n???a?1=????ǡ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 65.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?K???P@Q?h??@@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	;???]@;???]@!;???]@      ??!       "	?f??IM@?f??IM@!?f??IM@*      ??!       2	g?!?{??g?!?{??!g?!?{??:	???s@???s@!???s@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?K???P@y?h??@@?"r
Ggradient_tape/PerceptualNetwork/block2_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInputǜvٗ??!ǜvٗ??0"r
Ggradient_tape/PerceptualNetwork/block1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?v?YФ??!,?? u??0"G
%PerceptualNetwork/block2_conv2/Relu_1_FusedConv2D)?֚=???!"?58p???"E
#PerceptualNetwork/block1_conv2/Relu_FusedConv2D???q٩?!0y?=????"G
%PerceptualNetwork/block1_conv2/Relu_1_FusedConv2D??m?\??!H?dKy???"E
#PerceptualNetwork/block3_conv3/Relu_FusedConv2Dtn/٦?!ɨR1????"E
#PerceptualNetwork/block3_conv2/Relu_FusedConv2D?%3???!~5?C!??"g
;gradient_tape/Decoder/conv5_dec/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ò?????!?e뱻v??0"E
#PerceptualNetwork/block2_conv2/Relu_FusedConv2D???????!p?'E????"G
%PerceptualNetwork/block3_conv2/Relu_1_FusedConv2DR?+W???!?R?J?H??Q      Y@Y?+Q?!@a^?ڕ??V@q0f?g?+6@y??TK??|?"?

both?Your program is POTENTIALLY input-bound because 65.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?22.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 