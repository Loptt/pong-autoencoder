??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv1_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1_enc/kernel
}
$conv1_enc/kernel/Read/ReadVariableOpReadVariableOpconv1_enc/kernel*&
_output_shapes
:*
dtype0
t
conv1_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_enc/bias
m
"conv1_enc/bias/Read/ReadVariableOpReadVariableOpconv1_enc/bias*
_output_shapes
:*
dtype0
?
conv2_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2_enc/kernel
}
$conv2_enc/kernel/Read/ReadVariableOpReadVariableOpconv2_enc/kernel*&
_output_shapes
: *
dtype0
t
conv2_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2_enc/bias
m
"conv2_enc/bias/Read/ReadVariableOpReadVariableOpconv2_enc/bias*
_output_shapes
: *
dtype0
?
conv3_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv3_enc/kernel
}
$conv3_enc/kernel/Read/ReadVariableOpReadVariableOpconv3_enc/kernel*&
_output_shapes
: @*
dtype0
t
conv3_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3_enc/bias
m
"conv3_enc/bias/Read/ReadVariableOpReadVariableOpconv3_enc/bias*
_output_shapes
:@*
dtype0
?
conv4_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv4_enc/kernel
~
$conv4_enc/kernel/Read/ReadVariableOpReadVariableOpconv4_enc/kernel*'
_output_shapes
:@?*
dtype0
u
conv4_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv4_enc/bias
n
"conv4_enc/bias/Read/ReadVariableOpReadVariableOpconv4_enc/bias*
_output_shapes	
:?*
dtype0

bottleneck/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*"
shared_namebottleneck/kernel
x
%bottleneck/kernel/Read/ReadVariableOpReadVariableOpbottleneck/kernel*
_output_shapes
:	?*
dtype0
v
bottleneck/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebottleneck/bias
o
#bottleneck/bias/Read/ReadVariableOpReadVariableOpbottleneck/bias*
_output_shapes
:*
dtype0
{
decoding/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedecoding/kernel
t
#decoding/kernel/Read/ReadVariableOpReadVariableOpdecoding/kernel*
_output_shapes
:	?*
dtype0
s
decoding/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedecoding/bias
l
!decoding/bias/Read/ReadVariableOpReadVariableOpdecoding/bias*
_output_shapes	
:?*
dtype0
?
conv4_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv4_dec/kernel

$conv4_dec/kernel/Read/ReadVariableOpReadVariableOpconv4_dec/kernel*(
_output_shapes
:??*
dtype0
u
conv4_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv4_dec/bias
n
"conv4_dec/bias/Read/ReadVariableOpReadVariableOpconv4_dec/bias*
_output_shapes	
:?*
dtype0
?
conv3_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*!
shared_nameconv3_dec/kernel
~
$conv3_dec/kernel/Read/ReadVariableOpReadVariableOpconv3_dec/kernel*'
_output_shapes
:?@*
dtype0
t
conv3_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv3_dec/bias
m
"conv3_dec/bias/Read/ReadVariableOpReadVariableOpconv3_dec/bias*
_output_shapes
:@*
dtype0
?
conv2_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv2_dec/kernel
}
$conv2_dec/kernel/Read/ReadVariableOpReadVariableOpconv2_dec/kernel*&
_output_shapes
:@ *
dtype0
t
conv2_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2_dec/bias
m
"conv2_dec/bias/Read/ReadVariableOpReadVariableOpconv2_dec/bias*
_output_shapes
: *
dtype0
?
conv1_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv1_dec/kernel
}
$conv1_dec/kernel/Read/ReadVariableOpReadVariableOpconv1_dec/kernel*&
_output_shapes
: *
dtype0
t
conv1_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_dec/bias
m
"conv1_dec/bias/Read/ReadVariableOpReadVariableOpconv1_dec/bias*
_output_shapes
:*
dtype0
~
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/kernel
w
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*&
_output_shapes
:*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/conv1_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1_enc/kernel/m
?
+Adam/conv1_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv1_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_enc/bias/m
{
)Adam/conv1_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2_enc/kernel/m
?
+Adam/conv2_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2_enc/bias/m
{
)Adam/conv2_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv3_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv3_enc/kernel/m
?
+Adam/conv3_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv3_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv3_enc/bias/m
{
)Adam/conv3_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv4_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv4_enc/kernel/m
?
+Adam/conv4_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/kernel/m*'
_output_shapes
:@?*
dtype0
?
Adam/conv4_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv4_enc/bias/m
|
)Adam/conv4_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/bottleneck/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameAdam/bottleneck/kernel/m
?
,Adam/bottleneck/kernel/m/Read/ReadVariableOpReadVariableOpAdam/bottleneck/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/bottleneck/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/bottleneck/bias/m
}
*Adam/bottleneck/bias/m/Read/ReadVariableOpReadVariableOpAdam/bottleneck/bias/m*
_output_shapes
:*
dtype0
?
Adam/decoding/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/decoding/kernel/m
?
*Adam/decoding/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoding/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/decoding/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/decoding/bias/m
z
(Adam/decoding/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoding/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv4_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv4_dec/kernel/m
?
+Adam/conv4_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv4_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv4_dec/bias/m
|
)Adam/conv4_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv3_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*(
shared_nameAdam/conv3_dec/kernel/m
?
+Adam/conv3_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/kernel/m*'
_output_shapes
:?@*
dtype0
?
Adam/conv3_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv3_dec/bias/m
{
)Adam/conv3_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2_dec/kernel/m
?
+Adam/conv2_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/kernel/m*&
_output_shapes
:@ *
dtype0
?
Adam/conv2_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2_dec/bias/m
{
)Adam/conv2_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv1_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1_dec/kernel/m
?
+Adam/conv1_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv1_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_dec/bias/m
{
)Adam/conv1_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/bias/m*
_output_shapes
:*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output/kernel/m
?
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv1_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1_enc/kernel/v
?
+Adam/conv1_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv1_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_enc/bias/v
{
)Adam/conv1_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2_enc/kernel/v
?
+Adam/conv2_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2_enc/bias/v
{
)Adam/conv2_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv3_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv3_enc/kernel/v
?
+Adam/conv3_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv3_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv3_enc/bias/v
{
)Adam/conv3_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv4_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv4_enc/kernel/v
?
+Adam/conv4_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/kernel/v*'
_output_shapes
:@?*
dtype0
?
Adam/conv4_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv4_enc/bias/v
|
)Adam/conv4_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/bottleneck/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameAdam/bottleneck/kernel/v
?
,Adam/bottleneck/kernel/v/Read/ReadVariableOpReadVariableOpAdam/bottleneck/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/bottleneck/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/bottleneck/bias/v
}
*Adam/bottleneck/bias/v/Read/ReadVariableOpReadVariableOpAdam/bottleneck/bias/v*
_output_shapes
:*
dtype0
?
Adam/decoding/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/decoding/kernel/v
?
*Adam/decoding/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoding/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/decoding/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/decoding/bias/v
z
(Adam/decoding/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoding/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv4_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv4_dec/kernel/v
?
+Adam/conv4_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv4_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv4_dec/bias/v
|
)Adam/conv4_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv3_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*(
shared_nameAdam/conv3_dec/kernel/v
?
+Adam/conv3_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/kernel/v*'
_output_shapes
:?@*
dtype0
?
Adam/conv3_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv3_dec/bias/v
{
)Adam/conv3_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2_dec/kernel/v
?
+Adam/conv2_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/kernel/v*&
_output_shapes
:@ *
dtype0
?
Adam/conv2_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2_dec/bias/v
{
)Adam/conv2_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv1_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1_dec/kernel/v
?
+Adam/conv1_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv1_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_dec/bias/v
{
)Adam/conv1_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/bias/v*
_output_shapes
:*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output/kernel/v
?
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B܇ Bԇ
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
 
?

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
regularization_losses
trainable_variables
	variables
	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
 layer_with_weights-3
 layer-7
!layer-8
"layer_with_weights-4
"layer-9
#layer-10
$layer_with_weights-5
$layer-11
%regularization_losses
&trainable_variables
'	variables
(	keras_api
?
)iter

*beta_1

+beta_2
	,decay
-learning_rate.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?
 
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21
?
Dlayer_metrics
regularization_losses
trainable_variables

Elayers
Flayer_regularization_losses
Gmetrics
Hnon_trainable_variables
	variables
 
 
h

.kernel
/bias
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
R
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
h

0kernel
1bias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
R
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
h

2kernel
3bias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
R
]	variables
^regularization_losses
_trainable_variables
`	keras_api
h

4kernel
5bias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
R
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
R
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
h

6kernel
7bias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
 
F
.0
/1
02
13
24
35
46
57
68
79
F
.0
/1
02
13
24
35
46
57
68
79
?
qlayer_metrics
regularization_losses
trainable_variables

rlayers
slayer_regularization_losses
tmetrics
unon_trainable_variables
	variables
 
h

8kernel
9bias
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
R
z	variables
{regularization_losses
|trainable_variables
}	keras_api
j

:kernel
;bias
~	variables
regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

<kernel
=bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

>kernel
?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

@kernel
Abias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
V
?	variables
?regularization_losses
?trainable_variables
?	keras_api
l

Bkernel
Cbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
 
V
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11
V
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11
?
?layer_metrics
%regularization_losses
&trainable_variables
?layers
 ?layer_regularization_losses
?metrics
?non_trainable_variables
'	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv1_enc/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv1_enc/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2_enc/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2_enc/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv3_enc/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv3_enc/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv4_enc/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv4_enc/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbottleneck/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbottleneck/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdecoding/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdecoding/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv4_dec/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv4_dec/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3_dec/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3_dec/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2_dec/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2_dec/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1_dec/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1_dec/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEoutput/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEoutput/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
 

?0
 

.0
/1
 

.0
/1
?
?layer_metrics
I	variables
Jregularization_losses
Ktrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
M	variables
Nregularization_losses
Otrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses

00
11
 

00
11
?
?layer_metrics
Q	variables
Rregularization_losses
Strainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
U	variables
Vregularization_losses
Wtrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses

20
31
 

20
31
?
?layer_metrics
Y	variables
Zregularization_losses
[trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
]	variables
^regularization_losses
_trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses

40
51
 

40
51
?
?layer_metrics
a	variables
bregularization_losses
ctrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
e	variables
fregularization_losses
gtrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
i	variables
jregularization_losses
ktrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses

60
71
 

60
71
?
?layer_metrics
m	variables
nregularization_losses
otrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
N

0
1
2
3
4
5
6
7
8
9
10
 
 
 

80
91
 

80
91
?
?layer_metrics
v	variables
wregularization_losses
xtrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
z	variables
{regularization_losses
|trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses

:0
;1
 

:0
;1
?
?layer_metrics
~	variables
regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses

<0
=1
 

<0
=1
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses

>0
?1
 

>0
?1
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses

@0
A1
 

@0
A1
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
 
 
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses

B0
C1
 

B0
C1
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
 
V
0
1
2
3
4
5
6
 7
!8
"9
#10
$11
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
yw
VARIABLE_VALUEAdam/conv1_enc/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1_enc/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2_enc/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2_enc/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv3_enc/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3_enc/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv4_enc/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4_enc/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/bottleneck/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/bottleneck/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/decoding/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/decoding/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv4_dec/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv4_dec/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3_dec/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3_dec/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2_dec/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2_dec/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1_dec/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1_dec/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/output/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/output/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv1_enc/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv1_enc/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2_enc/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2_enc/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv3_enc/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv3_enc/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv4_enc/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv4_enc/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/bottleneck/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/bottleneck/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/decoding/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/decoding/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv4_dec/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv4_dec/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3_dec/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3_dec/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2_dec/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2_dec/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1_dec/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1_dec/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/output/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/output/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_encoderPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_encoderconv1_enc/kernelconv1_enc/biasconv2_enc/kernelconv2_enc/biasconv3_enc/kernelconv3_enc/biasconv4_enc/kernelconv4_enc/biasbottleneck/kernelbottleneck/biasdecoding/kerneldecoding/biasconv4_dec/kernelconv4_dec/biasconv3_dec/kernelconv3_dec/biasconv2_dec/kernelconv2_dec/biasconv1_dec/kernelconv1_dec/biasoutput/kerneloutput/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_84535
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv1_enc/kernel/Read/ReadVariableOp"conv1_enc/bias/Read/ReadVariableOp$conv2_enc/kernel/Read/ReadVariableOp"conv2_enc/bias/Read/ReadVariableOp$conv3_enc/kernel/Read/ReadVariableOp"conv3_enc/bias/Read/ReadVariableOp$conv4_enc/kernel/Read/ReadVariableOp"conv4_enc/bias/Read/ReadVariableOp%bottleneck/kernel/Read/ReadVariableOp#bottleneck/bias/Read/ReadVariableOp#decoding/kernel/Read/ReadVariableOp!decoding/bias/Read/ReadVariableOp$conv4_dec/kernel/Read/ReadVariableOp"conv4_dec/bias/Read/ReadVariableOp$conv3_dec/kernel/Read/ReadVariableOp"conv3_dec/bias/Read/ReadVariableOp$conv2_dec/kernel/Read/ReadVariableOp"conv2_dec/bias/Read/ReadVariableOp$conv1_dec/kernel/Read/ReadVariableOp"conv1_dec/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv1_enc/kernel/m/Read/ReadVariableOp)Adam/conv1_enc/bias/m/Read/ReadVariableOp+Adam/conv2_enc/kernel/m/Read/ReadVariableOp)Adam/conv2_enc/bias/m/Read/ReadVariableOp+Adam/conv3_enc/kernel/m/Read/ReadVariableOp)Adam/conv3_enc/bias/m/Read/ReadVariableOp+Adam/conv4_enc/kernel/m/Read/ReadVariableOp)Adam/conv4_enc/bias/m/Read/ReadVariableOp,Adam/bottleneck/kernel/m/Read/ReadVariableOp*Adam/bottleneck/bias/m/Read/ReadVariableOp*Adam/decoding/kernel/m/Read/ReadVariableOp(Adam/decoding/bias/m/Read/ReadVariableOp+Adam/conv4_dec/kernel/m/Read/ReadVariableOp)Adam/conv4_dec/bias/m/Read/ReadVariableOp+Adam/conv3_dec/kernel/m/Read/ReadVariableOp)Adam/conv3_dec/bias/m/Read/ReadVariableOp+Adam/conv2_dec/kernel/m/Read/ReadVariableOp)Adam/conv2_dec/bias/m/Read/ReadVariableOp+Adam/conv1_dec/kernel/m/Read/ReadVariableOp)Adam/conv1_dec/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv1_enc/kernel/v/Read/ReadVariableOp)Adam/conv1_enc/bias/v/Read/ReadVariableOp+Adam/conv2_enc/kernel/v/Read/ReadVariableOp)Adam/conv2_enc/bias/v/Read/ReadVariableOp+Adam/conv3_enc/kernel/v/Read/ReadVariableOp)Adam/conv3_enc/bias/v/Read/ReadVariableOp+Adam/conv4_enc/kernel/v/Read/ReadVariableOp)Adam/conv4_enc/bias/v/Read/ReadVariableOp,Adam/bottleneck/kernel/v/Read/ReadVariableOp*Adam/bottleneck/bias/v/Read/ReadVariableOp*Adam/decoding/kernel/v/Read/ReadVariableOp(Adam/decoding/bias/v/Read/ReadVariableOp+Adam/conv4_dec/kernel/v/Read/ReadVariableOp)Adam/conv4_dec/bias/v/Read/ReadVariableOp+Adam/conv3_dec/kernel/v/Read/ReadVariableOp)Adam/conv3_dec/bias/v/Read/ReadVariableOp+Adam/conv2_dec/kernel/v/Read/ReadVariableOp)Adam/conv2_dec/bias/v/Read/ReadVariableOp+Adam/conv1_dec/kernel/v/Read/ReadVariableOp)Adam/conv1_dec/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*V
TinO
M2K	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_85747
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv1_enc/kernelconv1_enc/biasconv2_enc/kernelconv2_enc/biasconv3_enc/kernelconv3_enc/biasconv4_enc/kernelconv4_enc/biasbottleneck/kernelbottleneck/biasdecoding/kerneldecoding/biasconv4_dec/kernelconv4_dec/biasconv3_dec/kernelconv3_dec/biasconv2_dec/kernelconv2_dec/biasconv1_dec/kernelconv1_dec/biasoutput/kerneloutput/biastotalcountAdam/conv1_enc/kernel/mAdam/conv1_enc/bias/mAdam/conv2_enc/kernel/mAdam/conv2_enc/bias/mAdam/conv3_enc/kernel/mAdam/conv3_enc/bias/mAdam/conv4_enc/kernel/mAdam/conv4_enc/bias/mAdam/bottleneck/kernel/mAdam/bottleneck/bias/mAdam/decoding/kernel/mAdam/decoding/bias/mAdam/conv4_dec/kernel/mAdam/conv4_dec/bias/mAdam/conv3_dec/kernel/mAdam/conv3_dec/bias/mAdam/conv2_dec/kernel/mAdam/conv2_dec/bias/mAdam/conv1_dec/kernel/mAdam/conv1_dec/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv1_enc/kernel/vAdam/conv1_enc/bias/vAdam/conv2_enc/kernel/vAdam/conv2_enc/bias/vAdam/conv3_enc/kernel/vAdam/conv3_enc/bias/vAdam/conv4_enc/kernel/vAdam/conv4_enc/bias/vAdam/bottleneck/kernel/vAdam/bottleneck/bias/vAdam/decoding/kernel/vAdam/decoding/bias/vAdam/conv4_dec/kernel/vAdam/conv4_dec/bias/vAdam/conv3_dec/kernel/vAdam/conv3_dec/bias/vAdam/conv2_dec/kernel/vAdam/conv2_dec/bias/vAdam/conv1_dec/kernel/vAdam/conv1_dec/bias/vAdam/output/kernel/vAdam/output/bias/v*U
TinN
L2J*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_85976??
?
_
C__inference_maxpool3_layer_call_and_return_conditional_losses_83296

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv1_enc_layer_call_and_return_conditional_losses_83329

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_upsamp3_layer_call_and_return_conditional_losses_83656

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2_dec_layer_call_and_return_conditional_losses_83819

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
_
C__inference_maxpool4_layer_call_and_return_conditional_losses_83308

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
'__inference_Decoder_layer_call_fn_85228

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_839732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_conv3_dec_layer_call_fn_85445

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_dec_layer_call_and_return_conditional_losses_837912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2_dec_layer_call_and_return_conditional_losses_85456

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
D
(__inference_maxpool4_layer_call_fn_83314

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool4_layer_call_and_return_conditional_losses_833082
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_85400

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
'__inference_Decoder_layer_call_fn_84068
input_decoder
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_decoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_840412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameinput_decoder
?:
?
B__inference_Encoder_layer_call_and_return_conditional_losses_84931

inputs,
(conv1_enc_conv2d_readvariableop_resource-
)conv1_enc_biasadd_readvariableop_resource,
(conv2_enc_conv2d_readvariableop_resource-
)conv2_enc_biasadd_readvariableop_resource,
(conv3_enc_conv2d_readvariableop_resource-
)conv3_enc_biasadd_readvariableop_resource,
(conv4_enc_conv2d_readvariableop_resource-
)conv4_enc_biasadd_readvariableop_resource-
)bottleneck_matmul_readvariableop_resource.
*bottleneck_biasadd_readvariableop_resource
identity??!bottleneck/BiasAdd/ReadVariableOp? bottleneck/MatMul/ReadVariableOp? conv1_enc/BiasAdd/ReadVariableOp?conv1_enc/Conv2D/ReadVariableOp? conv2_enc/BiasAdd/ReadVariableOp?conv2_enc/Conv2D/ReadVariableOp? conv3_enc/BiasAdd/ReadVariableOp?conv3_enc/Conv2D/ReadVariableOp? conv4_enc/BiasAdd/ReadVariableOp?conv4_enc/Conv2D/ReadVariableOp?
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOp?
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1_enc/Conv2D?
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOp?
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv1_enc/Relu?
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:?????????

*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool?
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_enc/Conv2D/ReadVariableOp?
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingSAME*
strides
2
conv2_enc/Conv2D?
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_enc/BiasAdd/ReadVariableOp?
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2_enc/Relu?
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPool?
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv3_enc/Conv2D/ReadVariableOp?
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv3_enc/Conv2D?
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_enc/BiasAdd/ReadVariableOp?
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv3_enc/Relu?
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPool?
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv4_enc/Conv2D/ReadVariableOp?
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_enc/Conv2D?
 conv4_enc/BiasAdd/ReadVariableOpReadVariableOp)conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv4_enc/BiasAdd/ReadVariableOp?
conv4_enc/BiasAddBiasAddconv4_enc/Conv2D:output:0(conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_enc/BiasAdd
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_enc/Relu?
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
maxpool4/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapemaxpool4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 bottleneck/MatMul/ReadVariableOp?
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/MatMul?
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp?
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/BiasAdd?
IdentityIdentitybottleneck/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2F
!bottleneck/BiasAdd/ReadVariableOp!bottleneck/BiasAdd/ReadVariableOp2D
 bottleneck/MatMul/ReadVariableOp bottleneck/MatMul/ReadVariableOp2D
 conv1_enc/BiasAdd/ReadVariableOp conv1_enc/BiasAdd/ReadVariableOp2B
conv1_enc/Conv2D/ReadVariableOpconv1_enc/Conv2D/ReadVariableOp2D
 conv2_enc/BiasAdd/ReadVariableOp conv2_enc/BiasAdd/ReadVariableOp2B
conv2_enc/Conv2D/ReadVariableOpconv2_enc/Conv2D/ReadVariableOp2D
 conv3_enc/BiasAdd/ReadVariableOp conv3_enc/BiasAdd/ReadVariableOp2B
conv3_enc/Conv2D/ReadVariableOpconv3_enc/Conv2D/ReadVariableOp2D
 conv4_enc/BiasAdd/ReadVariableOp conv4_enc/BiasAdd/ReadVariableOp2B
conv4_enc/Conv2D/ReadVariableOpconv4_enc/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
'__inference_Decoder_layer_call_fn_84000
input_decoder
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_decoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_839732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameinput_decoder
?
?
>__inference_CAE_layer_call_and_return_conditional_losses_84227
input_encoder
encoder_84122
encoder_84124
encoder_84126
encoder_84128
encoder_84130
encoder_84132
encoder_84134
encoder_84136
encoder_84138
encoder_84140
decoder_84201
decoder_84203
decoder_84205
decoder_84207
decoder_84209
decoder_84211
decoder_84213
decoder_84215
decoder_84217
decoder_84219
decoder_84221
decoder_84223
identity??Decoder/StatefulPartitionedCall?Encoder/StatefulPartitionedCall?
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_encoderencoder_84122encoder_84124encoder_84126encoder_84128encoder_84130encoder_84132encoder_84134encoder_84136encoder_84138encoder_84140*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_835422!
Encoder/StatefulPartitionedCall?
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_84201decoder_84203decoder_84205decoder_84207decoder_84209decoder_84211decoder_84213decoder_84215decoder_84217decoder_84219decoder_84221decoder_84223*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_839732!
Decoder/StatefulPartitionedCall?
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?
~
)__inference_conv1_enc_layer_call_fn_85277

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_enc_layer_call_and_return_conditional_losses_833292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
D
(__inference_maxpool1_layer_call_fn_83278

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool1_layer_call_and_return_conditional_losses_832722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_Encoder_layer_call_fn_85000

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_835422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv3_dec_layer_call_and_return_conditional_losses_85436

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?0
?
B__inference_Decoder_layer_call_and_return_conditional_losses_83892
input_decoder
decoding_83725
decoding_83727
conv4_dec_83774
conv4_dec_83776
conv3_dec_83802
conv3_dec_83804
conv2_dec_83830
conv2_dec_83832
conv1_dec_83858
conv1_dec_83860
output_83886
output_83888
identity??!conv1_dec/StatefulPartitionedCall?!conv2_dec/StatefulPartitionedCall?!conv3_dec/StatefulPartitionedCall?!conv4_dec/StatefulPartitionedCall? decoding/StatefulPartitionedCall?output/StatefulPartitionedCall?
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_83725decoding_83727*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoding_layer_call_and_return_conditional_losses_837142"
 decoding/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_837442
reshape/PartitionedCall?
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_83774conv4_dec_83776*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_dec_layer_call_and_return_conditional_losses_837632#
!conv4_dec/StatefulPartitionedCall?
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp4_layer_call_and_return_conditional_losses_836372
upsamp4/PartitionedCall?
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_83802conv3_dec_83804*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_dec_layer_call_and_return_conditional_losses_837912#
!conv3_dec/StatefulPartitionedCall?
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp3_layer_call_and_return_conditional_losses_836562
upsamp3/PartitionedCall?
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_83830conv2_dec_83832*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_dec_layer_call_and_return_conditional_losses_838192#
!conv2_dec/StatefulPartitionedCall?
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp2_layer_call_and_return_conditional_losses_836752
upsamp2/PartitionedCall?
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_83858conv1_dec_83860*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_dec_layer_call_and_return_conditional_losses_838472#
!conv1_dec/StatefulPartitionedCall?
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp1_layer_call_and_return_conditional_losses_836942
upsamp1/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_83886output_83888*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_838752 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameinput_decoder
?o
?
B__inference_Decoder_layer_call_and_return_conditional_losses_85199

inputs+
'decoding_matmul_readvariableop_resource,
(decoding_biasadd_readvariableop_resource,
(conv4_dec_conv2d_readvariableop_resource-
)conv4_dec_biasadd_readvariableop_resource,
(conv3_dec_conv2d_readvariableop_resource-
)conv3_dec_biasadd_readvariableop_resource,
(conv2_dec_conv2d_readvariableop_resource-
)conv2_dec_biasadd_readvariableop_resource,
(conv1_dec_conv2d_readvariableop_resource-
)conv1_dec_biasadd_readvariableop_resource)
%output_conv2d_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity?? conv1_dec/BiasAdd/ReadVariableOp?conv1_dec/Conv2D/ReadVariableOp? conv2_dec/BiasAdd/ReadVariableOp?conv2_dec/Conv2D/ReadVariableOp? conv3_dec/BiasAdd/ReadVariableOp?conv3_dec/Conv2D/ReadVariableOp? conv4_dec/BiasAdd/ReadVariableOp?conv4_dec/Conv2D/ReadVariableOp?decoding/BiasAdd/ReadVariableOp?decoding/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/Conv2D/ReadVariableOp?
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
decoding/MatMul/ReadVariableOp?
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoding/MatMul?
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
decoding/BiasAdd/ReadVariableOp?
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoding/BiasAddg
reshape/ShapeShapedecoding/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedecoding/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshape?
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv4_dec/Conv2D/ReadVariableOp?
conv4_dec/Conv2DConv2Dreshape/Reshape:output:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_dec/Conv2D?
 conv4_dec/BiasAdd/ReadVariableOpReadVariableOp)conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv4_dec/BiasAdd/ReadVariableOp?
conv4_dec/BiasAddBiasAddconv4_dec/Conv2D:output:0(conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_dec/BiasAdd
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_dec/Reluj
upsamp4/ShapeShapeconv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp4/Shape?
upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack?
upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack_1?
upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack_2?
upsamp4/strided_sliceStridedSliceupsamp4/Shape:output:0$upsamp4/strided_slice/stack:output:0&upsamp4/strided_slice/stack_1:output:0&upsamp4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp4/strided_sliceo
upsamp4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp4/Const~
upsamp4/mulMulupsamp4/strided_slice:output:0upsamp4/Const:output:0*
T0*
_output_shapes
:2
upsamp4/mul?
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighbor?
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv3_dec/Conv2D/ReadVariableOp?
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv3_dec/Conv2D?
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_dec/BiasAdd/ReadVariableOp?
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv3_dec/Reluj
upsamp3/ShapeShapeconv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp3/Shape?
upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack?
upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack_1?
upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack_2?
upsamp3/strided_sliceStridedSliceupsamp3/Shape:output:0$upsamp3/strided_slice/stack:output:0&upsamp3/strided_slice/stack_1:output:0&upsamp3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp3/strided_sliceo
upsamp3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp3/Const~
upsamp3/mulMulupsamp3/strided_slice:output:0upsamp3/Const:output:0*
T0*
_output_shapes
:2
upsamp3/mul?
$upsamp3/resize/ResizeNearestNeighborResizeNearestNeighborconv3_dec/Relu:activations:0upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighbor?
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2_dec/Conv2D/ReadVariableOp?
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2_dec/Conv2D?
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_dec/BiasAdd/ReadVariableOp?
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2_dec/Reluj
upsamp2/ShapeShapeconv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp2/Shape?
upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack?
upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack_1?
upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack_2?
upsamp2/strided_sliceStridedSliceupsamp2/Shape:output:0$upsamp2/strided_slice/stack:output:0&upsamp2/strided_slice/stack_1:output:0&upsamp2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp2/strided_sliceo
upsamp2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp2/Const~
upsamp2/mulMulupsamp2/strided_slice:output:0upsamp2/Const:output:0*
T0*
_output_shapes
:2
upsamp2/mul?
$upsamp2/resize/ResizeNearestNeighborResizeNearestNeighborconv2_dec/Relu:activations:0upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighbor?
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv1_dec/Conv2D/ReadVariableOp?
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1_dec/Conv2D?
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOp?
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv1_dec/Reluj
upsamp1/ShapeShapeconv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp1/Shape?
upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack?
upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack_1?
upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack_2?
upsamp1/strided_sliceStridedSliceupsamp1/Shape:output:0$upsamp1/strided_slice/stack:output:0&upsamp1/strided_slice/stack_1:output:0&upsamp1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp1/strided_sliceo
upsamp1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp1/Const~
upsamp1/mulMulupsamp1/strided_slice:output:0upsamp1/Const:output:0*
T0*
_output_shapes
:2
upsamp1/mul?
$upsamp1/resize/ResizeNearestNeighborResizeNearestNeighborconv1_dec/Relu:activations:0upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighbor?
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
output/Conv2D/ReadVariableOp?
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
output/Conv2D?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
output/Sigmoid?
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2D
 conv1_dec/BiasAdd/ReadVariableOp conv1_dec/BiasAdd/ReadVariableOp2B
conv1_dec/Conv2D/ReadVariableOpconv1_dec/Conv2D/ReadVariableOp2D
 conv2_dec/BiasAdd/ReadVariableOp conv2_dec/BiasAdd/ReadVariableOp2B
conv2_dec/Conv2D/ReadVariableOpconv2_dec/Conv2D/ReadVariableOp2D
 conv3_dec/BiasAdd/ReadVariableOp conv3_dec/BiasAdd/ReadVariableOp2B
conv3_dec/Conv2D/ReadVariableOpconv3_dec/Conv2D/ReadVariableOp2D
 conv4_dec/BiasAdd/ReadVariableOp conv4_dec/BiasAdd/ReadVariableOp2B
conv4_dec/Conv2D/ReadVariableOpconv4_dec/Conv2D/ReadVariableOp2B
decoding/BiasAdd/ReadVariableOpdecoding/BiasAdd/ReadVariableOp2@
decoding/MatMul/ReadVariableOpdecoding/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/Conv2D/ReadVariableOpoutput/Conv2D/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_conv2_enc_layer_call_fn_85297

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_enc_layer_call_and_return_conditional_losses_833572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?
^
B__inference_upsamp4_layer_call_and_return_conditional_losses_83637

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2_enc_layer_call_and_return_conditional_losses_83357

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
?+
?
B__inference_Encoder_layer_call_and_return_conditional_losses_83471
input_encoder
conv1_enc_83340
conv1_enc_83342
conv2_enc_83368
conv2_enc_83370
conv3_enc_83396
conv3_enc_83398
conv4_enc_83424
conv4_enc_83426
bottleneck_83465
bottleneck_83467
identity??"bottleneck/StatefulPartitionedCall?!conv1_enc/StatefulPartitionedCall?!conv2_enc/StatefulPartitionedCall?!conv3_enc/StatefulPartitionedCall?!conv4_enc/StatefulPartitionedCall?
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_83340conv1_enc_83342*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_enc_layer_call_and_return_conditional_losses_833292#
!conv1_enc/StatefulPartitionedCall?
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool1_layer_call_and_return_conditional_losses_832722
maxpool1/PartitionedCall?
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_83368conv2_enc_83370*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_enc_layer_call_and_return_conditional_losses_833572#
!conv2_enc/StatefulPartitionedCall?
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool2_layer_call_and_return_conditional_losses_832842
maxpool2/PartitionedCall?
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_83396conv3_enc_83398*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_enc_layer_call_and_return_conditional_losses_833852#
!conv3_enc/StatefulPartitionedCall?
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool3_layer_call_and_return_conditional_losses_832962
maxpool3/PartitionedCall?
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_83424conv4_enc_83426*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_enc_layer_call_and_return_conditional_losses_834132#
!conv4_enc/StatefulPartitionedCall?
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool4_layer_call_and_return_conditional_losses_833082
maxpool4/PartitionedCall?
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_834362
flatten/PartitionedCall?
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_83465bottleneck_83467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_bottleneck_layer_call_and_return_conditional_losses_834542$
"bottleneck/StatefulPartitionedCall?
IdentityIdentity+bottleneck/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?

?
D__inference_conv3_enc_layer_call_and_return_conditional_losses_83385

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
A__inference_output_layer_call_and_return_conditional_losses_85496

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
'__inference_Encoder_layer_call_fn_83565
input_encoder
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_encoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_835422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?
{
&__inference_output_layer_call_fn_85505

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_838752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
>__inference_CAE_layer_call_and_return_conditional_losses_84429

inputs
encoder_84382
encoder_84384
encoder_84386
encoder_84388
encoder_84390
encoder_84392
encoder_84394
encoder_84396
encoder_84398
encoder_84400
decoder_84403
decoder_84405
decoder_84407
decoder_84409
decoder_84411
decoder_84413
decoder_84415
decoder_84417
decoder_84419
decoder_84421
decoder_84423
decoder_84425
identity??Decoder/StatefulPartitionedCall?Encoder/StatefulPartitionedCall?
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_84382encoder_84384encoder_84386encoder_84388encoder_84390encoder_84392encoder_84394encoder_84396encoder_84398encoder_84400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_836012!
Encoder/StatefulPartitionedCall?
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_84403decoder_84405decoder_84407decoder_84409decoder_84411decoder_84413decoder_84415decoder_84417decoder_84419decoder_84421decoder_84423decoder_84425*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_840412!
Decoder/StatefulPartitionedCall?
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
'__inference_Encoder_layer_call_fn_83624
input_encoder
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_encoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_836012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?
~
)__inference_conv4_enc_layer_call_fn_85337

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_enc_layer_call_and_return_conditional_losses_834132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
C__inference_decoding_layer_call_and_return_conditional_losses_83714

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_maxpool1_layer_call_and_return_conditional_losses_83272

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
__inference__traced_save_85747
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop/
+savev2_conv1_enc_kernel_read_readvariableop-
)savev2_conv1_enc_bias_read_readvariableop/
+savev2_conv2_enc_kernel_read_readvariableop-
)savev2_conv2_enc_bias_read_readvariableop/
+savev2_conv3_enc_kernel_read_readvariableop-
)savev2_conv3_enc_bias_read_readvariableop/
+savev2_conv4_enc_kernel_read_readvariableop-
)savev2_conv4_enc_bias_read_readvariableop0
,savev2_bottleneck_kernel_read_readvariableop.
*savev2_bottleneck_bias_read_readvariableop.
*savev2_decoding_kernel_read_readvariableop,
(savev2_decoding_bias_read_readvariableop/
+savev2_conv4_dec_kernel_read_readvariableop-
)savev2_conv4_dec_bias_read_readvariableop/
+savev2_conv3_dec_kernel_read_readvariableop-
)savev2_conv3_dec_bias_read_readvariableop/
+savev2_conv2_dec_kernel_read_readvariableop-
)savev2_conv2_dec_bias_read_readvariableop/
+savev2_conv1_dec_kernel_read_readvariableop-
)savev2_conv1_dec_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv1_enc_kernel_m_read_readvariableop4
0savev2_adam_conv1_enc_bias_m_read_readvariableop6
2savev2_adam_conv2_enc_kernel_m_read_readvariableop4
0savev2_adam_conv2_enc_bias_m_read_readvariableop6
2savev2_adam_conv3_enc_kernel_m_read_readvariableop4
0savev2_adam_conv3_enc_bias_m_read_readvariableop6
2savev2_adam_conv4_enc_kernel_m_read_readvariableop4
0savev2_adam_conv4_enc_bias_m_read_readvariableop7
3savev2_adam_bottleneck_kernel_m_read_readvariableop5
1savev2_adam_bottleneck_bias_m_read_readvariableop5
1savev2_adam_decoding_kernel_m_read_readvariableop3
/savev2_adam_decoding_bias_m_read_readvariableop6
2savev2_adam_conv4_dec_kernel_m_read_readvariableop4
0savev2_adam_conv4_dec_bias_m_read_readvariableop6
2savev2_adam_conv3_dec_kernel_m_read_readvariableop4
0savev2_adam_conv3_dec_bias_m_read_readvariableop6
2savev2_adam_conv2_dec_kernel_m_read_readvariableop4
0savev2_adam_conv2_dec_bias_m_read_readvariableop6
2savev2_adam_conv1_dec_kernel_m_read_readvariableop4
0savev2_adam_conv1_dec_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop6
2savev2_adam_conv1_enc_kernel_v_read_readvariableop4
0savev2_adam_conv1_enc_bias_v_read_readvariableop6
2savev2_adam_conv2_enc_kernel_v_read_readvariableop4
0savev2_adam_conv2_enc_bias_v_read_readvariableop6
2savev2_adam_conv3_enc_kernel_v_read_readvariableop4
0savev2_adam_conv3_enc_bias_v_read_readvariableop6
2savev2_adam_conv4_enc_kernel_v_read_readvariableop4
0savev2_adam_conv4_enc_bias_v_read_readvariableop7
3savev2_adam_bottleneck_kernel_v_read_readvariableop5
1savev2_adam_bottleneck_bias_v_read_readvariableop5
1savev2_adam_decoding_kernel_v_read_readvariableop3
/savev2_adam_decoding_bias_v_read_readvariableop6
2savev2_adam_conv4_dec_kernel_v_read_readvariableop4
0savev2_adam_conv4_dec_bias_v_read_readvariableop6
2savev2_adam_conv3_dec_kernel_v_read_readvariableop4
0savev2_adam_conv3_dec_bias_v_read_readvariableop6
2savev2_adam_conv2_dec_kernel_v_read_readvariableop4
0savev2_adam_conv2_dec_bias_v_read_readvariableop6
2savev2_adam_conv1_dec_kernel_v_read_readvariableop4
0savev2_adam_conv1_dec_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?&
value?&B?&JB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv1_enc_kernel_read_readvariableop)savev2_conv1_enc_bias_read_readvariableop+savev2_conv2_enc_kernel_read_readvariableop)savev2_conv2_enc_bias_read_readvariableop+savev2_conv3_enc_kernel_read_readvariableop)savev2_conv3_enc_bias_read_readvariableop+savev2_conv4_enc_kernel_read_readvariableop)savev2_conv4_enc_bias_read_readvariableop,savev2_bottleneck_kernel_read_readvariableop*savev2_bottleneck_bias_read_readvariableop*savev2_decoding_kernel_read_readvariableop(savev2_decoding_bias_read_readvariableop+savev2_conv4_dec_kernel_read_readvariableop)savev2_conv4_dec_bias_read_readvariableop+savev2_conv3_dec_kernel_read_readvariableop)savev2_conv3_dec_bias_read_readvariableop+savev2_conv2_dec_kernel_read_readvariableop)savev2_conv2_dec_bias_read_readvariableop+savev2_conv1_dec_kernel_read_readvariableop)savev2_conv1_dec_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv1_enc_kernel_m_read_readvariableop0savev2_adam_conv1_enc_bias_m_read_readvariableop2savev2_adam_conv2_enc_kernel_m_read_readvariableop0savev2_adam_conv2_enc_bias_m_read_readvariableop2savev2_adam_conv3_enc_kernel_m_read_readvariableop0savev2_adam_conv3_enc_bias_m_read_readvariableop2savev2_adam_conv4_enc_kernel_m_read_readvariableop0savev2_adam_conv4_enc_bias_m_read_readvariableop3savev2_adam_bottleneck_kernel_m_read_readvariableop1savev2_adam_bottleneck_bias_m_read_readvariableop1savev2_adam_decoding_kernel_m_read_readvariableop/savev2_adam_decoding_bias_m_read_readvariableop2savev2_adam_conv4_dec_kernel_m_read_readvariableop0savev2_adam_conv4_dec_bias_m_read_readvariableop2savev2_adam_conv3_dec_kernel_m_read_readvariableop0savev2_adam_conv3_dec_bias_m_read_readvariableop2savev2_adam_conv2_dec_kernel_m_read_readvariableop0savev2_adam_conv2_dec_bias_m_read_readvariableop2savev2_adam_conv1_dec_kernel_m_read_readvariableop0savev2_adam_conv1_dec_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv1_enc_kernel_v_read_readvariableop0savev2_adam_conv1_enc_bias_v_read_readvariableop2savev2_adam_conv2_enc_kernel_v_read_readvariableop0savev2_adam_conv2_enc_bias_v_read_readvariableop2savev2_adam_conv3_enc_kernel_v_read_readvariableop0savev2_adam_conv3_enc_bias_v_read_readvariableop2savev2_adam_conv4_enc_kernel_v_read_readvariableop0savev2_adam_conv4_enc_bias_v_read_readvariableop3savev2_adam_bottleneck_kernel_v_read_readvariableop1savev2_adam_bottleneck_bias_v_read_readvariableop1savev2_adam_decoding_kernel_v_read_readvariableop/savev2_adam_decoding_bias_v_read_readvariableop2savev2_adam_conv4_dec_kernel_v_read_readvariableop0savev2_adam_conv4_dec_bias_v_read_readvariableop2savev2_adam_conv3_dec_kernel_v_read_readvariableop0savev2_adam_conv3_dec_bias_v_read_readvariableop2savev2_adam_conv2_dec_kernel_v_read_readvariableop0savev2_adam_conv2_dec_bias_v_read_readvariableop2savev2_adam_conv1_dec_kernel_v_read_readvariableop0savev2_adam_conv1_dec_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *X
dtypesN
L2J	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : ::: : : @:@:@?:?:	?::	?:?:??:?:?@:@:@ : : :::: : ::: : : @:@:@?:?:	?::	?:?:??:?:?@:@:@ : : :::::: : : @:@:@?:?:	?::	?:?:??:?:?@:@:@ : : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 	

_output_shapes
: :,
(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:-)
'
_output_shapes
:?@: 

_output_shapes
:@:,(
&
_output_shapes
:@ : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
: : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@:-$)
'
_output_shapes
:@?:!%

_output_shapes	
:?:%&!

_output_shapes
:	?: '

_output_shapes
::%(!

_output_shapes
:	?:!)

_output_shapes	
:?:.**
(
_output_shapes
:??:!+

_output_shapes	
:?:-,)
'
_output_shapes
:?@: -

_output_shapes
:@:,.(
&
_output_shapes
:@ : /

_output_shapes
: :,0(
&
_output_shapes
: : 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
: @: 9

_output_shapes
:@:-:)
'
_output_shapes
:@?:!;

_output_shapes	
:?:%<!

_output_shapes
:	?: =

_output_shapes
::%>!

_output_shapes
:	?:!?

_output_shapes	
:?:.@*
(
_output_shapes
:??:!A

_output_shapes	
:?:-B)
'
_output_shapes
:?@: C

_output_shapes
:@:,D(
&
_output_shapes
:@ : E

_output_shapes
: :,F(
&
_output_shapes
: : G

_output_shapes
::,H(
&
_output_shapes
:: I

_output_shapes
::J

_output_shapes
: 
??
?
>__inference_CAE_layer_call_and_return_conditional_losses_84789

inputs4
0encoder_conv1_enc_conv2d_readvariableop_resource5
1encoder_conv1_enc_biasadd_readvariableop_resource4
0encoder_conv2_enc_conv2d_readvariableop_resource5
1encoder_conv2_enc_biasadd_readvariableop_resource4
0encoder_conv3_enc_conv2d_readvariableop_resource5
1encoder_conv3_enc_biasadd_readvariableop_resource4
0encoder_conv4_enc_conv2d_readvariableop_resource5
1encoder_conv4_enc_biasadd_readvariableop_resource5
1encoder_bottleneck_matmul_readvariableop_resource6
2encoder_bottleneck_biasadd_readvariableop_resource3
/decoder_decoding_matmul_readvariableop_resource4
0decoder_decoding_biasadd_readvariableop_resource4
0decoder_conv4_dec_conv2d_readvariableop_resource5
1decoder_conv4_dec_biasadd_readvariableop_resource4
0decoder_conv3_dec_conv2d_readvariableop_resource5
1decoder_conv3_dec_biasadd_readvariableop_resource4
0decoder_conv2_dec_conv2d_readvariableop_resource5
1decoder_conv2_dec_biasadd_readvariableop_resource4
0decoder_conv1_dec_conv2d_readvariableop_resource5
1decoder_conv1_dec_biasadd_readvariableop_resource1
-decoder_output_conv2d_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identity??(Decoder/conv1_dec/BiasAdd/ReadVariableOp?'Decoder/conv1_dec/Conv2D/ReadVariableOp?(Decoder/conv2_dec/BiasAdd/ReadVariableOp?'Decoder/conv2_dec/Conv2D/ReadVariableOp?(Decoder/conv3_dec/BiasAdd/ReadVariableOp?'Decoder/conv3_dec/Conv2D/ReadVariableOp?(Decoder/conv4_dec/BiasAdd/ReadVariableOp?'Decoder/conv4_dec/Conv2D/ReadVariableOp?'Decoder/decoding/BiasAdd/ReadVariableOp?&Decoder/decoding/MatMul/ReadVariableOp?%Decoder/output/BiasAdd/ReadVariableOp?$Decoder/output/Conv2D/ReadVariableOp?)Encoder/bottleneck/BiasAdd/ReadVariableOp?(Encoder/bottleneck/MatMul/ReadVariableOp?(Encoder/conv1_enc/BiasAdd/ReadVariableOp?'Encoder/conv1_enc/Conv2D/ReadVariableOp?(Encoder/conv2_enc/BiasAdd/ReadVariableOp?'Encoder/conv2_enc/Conv2D/ReadVariableOp?(Encoder/conv3_enc/BiasAdd/ReadVariableOp?'Encoder/conv3_enc/Conv2D/ReadVariableOp?(Encoder/conv4_enc/BiasAdd/ReadVariableOp?'Encoder/conv4_enc/Conv2D/ReadVariableOp?
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOp?
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2D?
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOp?
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Encoder/conv1_enc/BiasAdd?
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Encoder/conv1_enc/Relu?
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:?????????

*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPool?
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOp?
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2D?
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOp?
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
Encoder/conv2_enc/BiasAdd?
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
Encoder/conv2_enc/Relu?
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPool?
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOp?
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2D?
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOp?
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Encoder/conv3_enc/BiasAdd?
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Encoder/conv3_enc/Relu?
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPool?
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOp?
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2D?
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOp?
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
Encoder/conv4_enc/BiasAdd?
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Encoder/conv4_enc/Relu?
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool4/MaxPool
Encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Encoder/flatten/Const?
Encoder/flatten/ReshapeReshape!Encoder/maxpool4/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
Encoder/flatten/Reshape?
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOp?
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/MatMul?
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOp?
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/BiasAdd?
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOp?
Decoder/decoding/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Decoder/decoding/MatMul?
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOp?
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Decoder/decoding/BiasAdd
Decoder/reshape/ShapeShape!Decoder/decoding/BiasAdd:output:0*
T0*
_output_shapes
:2
Decoder/reshape/Shape?
#Decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Decoder/reshape/strided_slice/stack?
%Decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/reshape/strided_slice/stack_1?
%Decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/reshape/strided_slice/stack_2?
Decoder/reshape/strided_sliceStridedSliceDecoder/reshape/Shape:output:0,Decoder/reshape/strided_slice/stack:output:0.Decoder/reshape/strided_slice/stack_1:output:0.Decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Decoder/reshape/strided_slice?
Decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
Decoder/reshape/Reshape/shape/1?
Decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
Decoder/reshape/Reshape/shape/2?
Decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2!
Decoder/reshape/Reshape/shape/3?
Decoder/reshape/Reshape/shapePack&Decoder/reshape/strided_slice:output:0(Decoder/reshape/Reshape/shape/1:output:0(Decoder/reshape/Reshape/shape/2:output:0(Decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Decoder/reshape/Reshape/shape?
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
Decoder/reshape/Reshape?
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOp?
Decoder/conv4_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2D?
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOp?
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
Decoder/conv4_dec/BiasAdd?
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Decoder/conv4_dec/Relu?
Decoder/upsamp4/ShapeShape$Decoder/conv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp4/Shape?
#Decoder/upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp4/strided_slice/stack?
%Decoder/upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp4/strided_slice/stack_1?
%Decoder/upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp4/strided_slice/stack_2?
Decoder/upsamp4/strided_sliceStridedSliceDecoder/upsamp4/Shape:output:0,Decoder/upsamp4/strided_slice/stack:output:0.Decoder/upsamp4/strided_slice/stack_1:output:0.Decoder/upsamp4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp4/strided_slice
Decoder/upsamp4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp4/Const?
Decoder/upsamp4/mulMul&Decoder/upsamp4/strided_slice:output:0Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp4/mul?
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighbor?
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp?
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2D?
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOp?
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Decoder/conv3_dec/BiasAdd?
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Decoder/conv3_dec/Relu?
Decoder/upsamp3/ShapeShape$Decoder/conv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp3/Shape?
#Decoder/upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp3/strided_slice/stack?
%Decoder/upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp3/strided_slice/stack_1?
%Decoder/upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp3/strided_slice/stack_2?
Decoder/upsamp3/strided_sliceStridedSliceDecoder/upsamp3/Shape:output:0,Decoder/upsamp3/strided_slice/stack:output:0.Decoder/upsamp3/strided_slice/stack_1:output:0.Decoder/upsamp3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp3/strided_slice
Decoder/upsamp3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp3/Const?
Decoder/upsamp3/mulMul&Decoder/upsamp3/strided_slice:output:0Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp3/mul?
,Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv3_dec/Relu:activations:0Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighbor?
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp?
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2D?
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOp?
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv2_dec/BiasAdd?
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv2_dec/Relu?
Decoder/upsamp2/ShapeShape$Decoder/conv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp2/Shape?
#Decoder/upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp2/strided_slice/stack?
%Decoder/upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp2/strided_slice/stack_1?
%Decoder/upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp2/strided_slice/stack_2?
Decoder/upsamp2/strided_sliceStridedSliceDecoder/upsamp2/Shape:output:0,Decoder/upsamp2/strided_slice/stack:output:0.Decoder/upsamp2/strided_slice/stack_1:output:0.Decoder/upsamp2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp2/strided_slice
Decoder/upsamp2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp2/Const?
Decoder/upsamp2/mulMul&Decoder/upsamp2/strided_slice:output:0Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp2/mul?
,Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv2_dec/Relu:activations:0Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighbor?
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp?
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2D?
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOp?
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Decoder/conv1_dec/BiasAdd?
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Decoder/conv1_dec/Relu?
Decoder/upsamp1/ShapeShape$Decoder/conv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp1/Shape?
#Decoder/upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp1/strided_slice/stack?
%Decoder/upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp1/strided_slice/stack_1?
%Decoder/upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp1/strided_slice/stack_2?
Decoder/upsamp1/strided_sliceStridedSliceDecoder/upsamp1/Shape:output:0,Decoder/upsamp1/strided_slice/stack:output:0.Decoder/upsamp1/strided_slice/stack_1:output:0.Decoder/upsamp1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp1/strided_slice
Decoder/upsamp1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp1/Const?
Decoder/upsamp1/mulMul&Decoder/upsamp1/strided_slice:output:0Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp1/mul?
,Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv1_dec/Relu:activations:0Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighbor?
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp?
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Decoder/output/Conv2D?
%Decoder/output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Decoder/output/BiasAdd/ReadVariableOp?
Decoder/output/BiasAddBiasAddDecoder/output/Conv2D:output:0-Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Decoder/output/BiasAdd?
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Decoder/output/Sigmoid?
IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::2T
(Decoder/conv1_dec/BiasAdd/ReadVariableOp(Decoder/conv1_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv1_dec/Conv2D/ReadVariableOp'Decoder/conv1_dec/Conv2D/ReadVariableOp2T
(Decoder/conv2_dec/BiasAdd/ReadVariableOp(Decoder/conv2_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv2_dec/Conv2D/ReadVariableOp'Decoder/conv2_dec/Conv2D/ReadVariableOp2T
(Decoder/conv3_dec/BiasAdd/ReadVariableOp(Decoder/conv3_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv3_dec/Conv2D/ReadVariableOp'Decoder/conv3_dec/Conv2D/ReadVariableOp2T
(Decoder/conv4_dec/BiasAdd/ReadVariableOp(Decoder/conv4_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv4_dec/Conv2D/ReadVariableOp'Decoder/conv4_dec/Conv2D/ReadVariableOp2R
'Decoder/decoding/BiasAdd/ReadVariableOp'Decoder/decoding/BiasAdd/ReadVariableOp2P
&Decoder/decoding/MatMul/ReadVariableOp&Decoder/decoding/MatMul/ReadVariableOp2N
%Decoder/output/BiasAdd/ReadVariableOp%Decoder/output/BiasAdd/ReadVariableOp2L
$Decoder/output/Conv2D/ReadVariableOp$Decoder/output/Conv2D/ReadVariableOp2V
)Encoder/bottleneck/BiasAdd/ReadVariableOp)Encoder/bottleneck/BiasAdd/ReadVariableOp2T
(Encoder/bottleneck/MatMul/ReadVariableOp(Encoder/bottleneck/MatMul/ReadVariableOp2T
(Encoder/conv1_enc/BiasAdd/ReadVariableOp(Encoder/conv1_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv1_enc/Conv2D/ReadVariableOp'Encoder/conv1_enc/Conv2D/ReadVariableOp2T
(Encoder/conv2_enc/BiasAdd/ReadVariableOp(Encoder/conv2_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv2_enc/Conv2D/ReadVariableOp'Encoder/conv2_enc/Conv2D/ReadVariableOp2T
(Encoder/conv3_enc/BiasAdd/ReadVariableOp(Encoder/conv3_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv3_enc/Conv2D/ReadVariableOp'Encoder/conv3_enc/Conv2D/ReadVariableOp2T
(Encoder/conv4_enc/BiasAdd/ReadVariableOp(Encoder/conv4_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv4_enc/Conv2D/ReadVariableOp'Encoder/conv4_enc/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_85343

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_CAE_layer_call_fn_84838

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CAE_layer_call_and_return_conditional_losses_843302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_maxpool2_layer_call_and_return_conditional_losses_83284

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_output_layer_call_and_return_conditional_losses_83875

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

*__inference_bottleneck_layer_call_fn_85367

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_bottleneck_layer_call_and_return_conditional_losses_834542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?+
?
B__inference_Encoder_layer_call_and_return_conditional_losses_83601

inputs
conv1_enc_83570
conv1_enc_83572
conv2_enc_83576
conv2_enc_83578
conv3_enc_83582
conv3_enc_83584
conv4_enc_83588
conv4_enc_83590
bottleneck_83595
bottleneck_83597
identity??"bottleneck/StatefulPartitionedCall?!conv1_enc/StatefulPartitionedCall?!conv2_enc/StatefulPartitionedCall?!conv3_enc/StatefulPartitionedCall?!conv4_enc/StatefulPartitionedCall?
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_83570conv1_enc_83572*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_enc_layer_call_and_return_conditional_losses_833292#
!conv1_enc/StatefulPartitionedCall?
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool1_layer_call_and_return_conditional_losses_832722
maxpool1/PartitionedCall?
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_83576conv2_enc_83578*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_enc_layer_call_and_return_conditional_losses_833572#
!conv2_enc/StatefulPartitionedCall?
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool2_layer_call_and_return_conditional_losses_832842
maxpool2/PartitionedCall?
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_83582conv3_enc_83584*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_enc_layer_call_and_return_conditional_losses_833852#
!conv3_enc/StatefulPartitionedCall?
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool3_layer_call_and_return_conditional_losses_832962
maxpool3/PartitionedCall?
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_83588conv4_enc_83590*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_enc_layer_call_and_return_conditional_losses_834132#
!conv4_enc/StatefulPartitionedCall?
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool4_layer_call_and_return_conditional_losses_833082
maxpool4/PartitionedCall?
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_834362
flatten/PartitionedCall?
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_83595bottleneck_83597*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_bottleneck_layer_call_and_return_conditional_losses_834542$
"bottleneck/StatefulPartitionedCall?
IdentityIdentity+bottleneck/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv3_dec_layer_call_and_return_conditional_losses_83791

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_upsamp1_layer_call_and_return_conditional_losses_83694

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_CAE_layer_call_fn_84887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CAE_layer_call_and_return_conditional_losses_844292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_bottleneck_layer_call_and_return_conditional_losses_83454

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
~
)__inference_conv4_dec_layer_call_fn_85425

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_dec_layer_call_and_return_conditional_losses_837632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
#__inference_CAE_layer_call_fn_84377
input_encoder
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_encoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CAE_layer_call_and_return_conditional_losses_843302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?
C
'__inference_flatten_layer_call_fn_85348

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_834362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_maxpool2_layer_call_fn_83290

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool2_layer_call_and_return_conditional_losses_832842
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
#__inference_CAE_layer_call_fn_84476
input_encoder
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_encoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *G
fBR@
>__inference_CAE_layer_call_and_return_conditional_losses_844292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?:
?
B__inference_Encoder_layer_call_and_return_conditional_losses_84975

inputs,
(conv1_enc_conv2d_readvariableop_resource-
)conv1_enc_biasadd_readvariableop_resource,
(conv2_enc_conv2d_readvariableop_resource-
)conv2_enc_biasadd_readvariableop_resource,
(conv3_enc_conv2d_readvariableop_resource-
)conv3_enc_biasadd_readvariableop_resource,
(conv4_enc_conv2d_readvariableop_resource-
)conv4_enc_biasadd_readvariableop_resource-
)bottleneck_matmul_readvariableop_resource.
*bottleneck_biasadd_readvariableop_resource
identity??!bottleneck/BiasAdd/ReadVariableOp? bottleneck/MatMul/ReadVariableOp? conv1_enc/BiasAdd/ReadVariableOp?conv1_enc/Conv2D/ReadVariableOp? conv2_enc/BiasAdd/ReadVariableOp?conv2_enc/Conv2D/ReadVariableOp? conv3_enc/BiasAdd/ReadVariableOp?conv3_enc/Conv2D/ReadVariableOp? conv4_enc/BiasAdd/ReadVariableOp?conv4_enc/Conv2D/ReadVariableOp?
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOp?
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1_enc/Conv2D?
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOp?
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv1_enc/Relu?
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:?????????

*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool?
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_enc/Conv2D/ReadVariableOp?
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingSAME*
strides
2
conv2_enc/Conv2D?
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_enc/BiasAdd/ReadVariableOp?
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
conv2_enc/Relu?
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPool?
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv3_enc/Conv2D/ReadVariableOp?
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv3_enc/Conv2D?
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_enc/BiasAdd/ReadVariableOp?
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv3_enc/Relu?
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPool?
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02!
conv4_enc/Conv2D/ReadVariableOp?
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_enc/Conv2D?
 conv4_enc/BiasAdd/ReadVariableOpReadVariableOp)conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv4_enc/BiasAdd/ReadVariableOp?
conv4_enc/BiasAddBiasAddconv4_enc/Conv2D:output:0(conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_enc/BiasAdd
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_enc/Relu?
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
maxpool4/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshapemaxpool4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02"
 bottleneck/MatMul/ReadVariableOp?
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/MatMul?
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp?
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/BiasAdd?
IdentityIdentitybottleneck/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2F
!bottleneck/BiasAdd/ReadVariableOp!bottleneck/BiasAdd/ReadVariableOp2D
 bottleneck/MatMul/ReadVariableOp bottleneck/MatMul/ReadVariableOp2D
 conv1_enc/BiasAdd/ReadVariableOp conv1_enc/BiasAdd/ReadVariableOp2B
conv1_enc/Conv2D/ReadVariableOpconv1_enc/Conv2D/ReadVariableOp2D
 conv2_enc/BiasAdd/ReadVariableOp conv2_enc/BiasAdd/ReadVariableOp2B
conv2_enc/Conv2D/ReadVariableOpconv2_enc/Conv2D/ReadVariableOp2D
 conv3_enc/BiasAdd/ReadVariableOp conv3_enc/BiasAdd/ReadVariableOp2B
conv3_enc/Conv2D/ReadVariableOpconv3_enc/Conv2D/ReadVariableOp2D
 conv4_enc/BiasAdd/ReadVariableOp conv4_enc/BiasAdd/ReadVariableOp2B
conv4_enc/Conv2D/ReadVariableOpconv4_enc/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_upsamp3_layer_call_fn_83662

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp3_layer_call_and_return_conditional_losses_836562
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_upsamp2_layer_call_and_return_conditional_losses_83675

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
C
'__inference_reshape_layer_call_fn_85405

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_837442
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?&
!__inference__traced_restore_85976
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate'
#assignvariableop_5_conv1_enc_kernel%
!assignvariableop_6_conv1_enc_bias'
#assignvariableop_7_conv2_enc_kernel%
!assignvariableop_8_conv2_enc_bias'
#assignvariableop_9_conv3_enc_kernel&
"assignvariableop_10_conv3_enc_bias(
$assignvariableop_11_conv4_enc_kernel&
"assignvariableop_12_conv4_enc_bias)
%assignvariableop_13_bottleneck_kernel'
#assignvariableop_14_bottleneck_bias'
#assignvariableop_15_decoding_kernel%
!assignvariableop_16_decoding_bias(
$assignvariableop_17_conv4_dec_kernel&
"assignvariableop_18_conv4_dec_bias(
$assignvariableop_19_conv3_dec_kernel&
"assignvariableop_20_conv3_dec_bias(
$assignvariableop_21_conv2_dec_kernel&
"assignvariableop_22_conv2_dec_bias(
$assignvariableop_23_conv1_dec_kernel&
"assignvariableop_24_conv1_dec_bias%
!assignvariableop_25_output_kernel#
assignvariableop_26_output_bias
assignvariableop_27_total
assignvariableop_28_count/
+assignvariableop_29_adam_conv1_enc_kernel_m-
)assignvariableop_30_adam_conv1_enc_bias_m/
+assignvariableop_31_adam_conv2_enc_kernel_m-
)assignvariableop_32_adam_conv2_enc_bias_m/
+assignvariableop_33_adam_conv3_enc_kernel_m-
)assignvariableop_34_adam_conv3_enc_bias_m/
+assignvariableop_35_adam_conv4_enc_kernel_m-
)assignvariableop_36_adam_conv4_enc_bias_m0
,assignvariableop_37_adam_bottleneck_kernel_m.
*assignvariableop_38_adam_bottleneck_bias_m.
*assignvariableop_39_adam_decoding_kernel_m,
(assignvariableop_40_adam_decoding_bias_m/
+assignvariableop_41_adam_conv4_dec_kernel_m-
)assignvariableop_42_adam_conv4_dec_bias_m/
+assignvariableop_43_adam_conv3_dec_kernel_m-
)assignvariableop_44_adam_conv3_dec_bias_m/
+assignvariableop_45_adam_conv2_dec_kernel_m-
)assignvariableop_46_adam_conv2_dec_bias_m/
+assignvariableop_47_adam_conv1_dec_kernel_m-
)assignvariableop_48_adam_conv1_dec_bias_m,
(assignvariableop_49_adam_output_kernel_m*
&assignvariableop_50_adam_output_bias_m/
+assignvariableop_51_adam_conv1_enc_kernel_v-
)assignvariableop_52_adam_conv1_enc_bias_v/
+assignvariableop_53_adam_conv2_enc_kernel_v-
)assignvariableop_54_adam_conv2_enc_bias_v/
+assignvariableop_55_adam_conv3_enc_kernel_v-
)assignvariableop_56_adam_conv3_enc_bias_v/
+assignvariableop_57_adam_conv4_enc_kernel_v-
)assignvariableop_58_adam_conv4_enc_bias_v0
,assignvariableop_59_adam_bottleneck_kernel_v.
*assignvariableop_60_adam_bottleneck_bias_v.
*assignvariableop_61_adam_decoding_kernel_v,
(assignvariableop_62_adam_decoding_bias_v/
+assignvariableop_63_adam_conv4_dec_kernel_v-
)assignvariableop_64_adam_conv4_dec_bias_v/
+assignvariableop_65_adam_conv3_dec_kernel_v-
)assignvariableop_66_adam_conv3_dec_bias_v/
+assignvariableop_67_adam_conv2_dec_kernel_v-
)assignvariableop_68_adam_conv2_dec_bias_v/
+assignvariableop_69_adam_conv1_dec_kernel_v-
)assignvariableop_70_adam_conv1_dec_bias_v,
(assignvariableop_71_adam_output_kernel_v*
&assignvariableop_72_adam_output_bias_v
identity_74??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_8?AssignVariableOp_9?'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?&
value?&B?&JB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:J*
dtype0*?
value?B?JB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*X
dtypesN
L2J	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv1_enc_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_conv1_enc_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_conv2_enc_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_conv2_enc_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv3_enc_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv3_enc_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv4_enc_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv4_enc_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp%assignvariableop_13_bottleneck_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_bottleneck_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_decoding_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_decoding_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv4_dec_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv4_dec_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv3_dec_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp"assignvariableop_20_conv3_dec_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp$assignvariableop_21_conv2_dec_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp"assignvariableop_22_conv2_dec_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp$assignvariableop_23_conv1_dec_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp"assignvariableop_24_conv1_dec_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_output_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_output_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv1_enc_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv1_enc_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2_enc_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2_enc_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv3_enc_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv3_enc_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv4_enc_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv4_enc_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp,assignvariableop_37_adam_bottleneck_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp*assignvariableop_38_adam_bottleneck_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_decoding_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_decoding_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv4_dec_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv4_dec_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv3_dec_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv3_dec_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2_dec_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2_dec_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv1_dec_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv1_dec_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_output_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_output_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv1_enc_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv1_enc_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2_enc_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2_enc_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv3_enc_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv3_enc_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv4_enc_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv4_enc_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_bottleneck_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_bottleneck_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_decoding_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_decoding_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv4_dec_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv4_dec_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv3_dec_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv3_dec_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2_dec_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2_dec_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv1_dec_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv1_dec_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp(assignvariableop_71_adam_output_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp&assignvariableop_72_adam_output_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_729
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_73Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_73?
Identity_74IdentityIdentity_73:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_74"#
identity_74Identity_74:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
>__inference_CAE_layer_call_and_return_conditional_losses_84277
input_encoder
encoder_84230
encoder_84232
encoder_84234
encoder_84236
encoder_84238
encoder_84240
encoder_84242
encoder_84244
encoder_84246
encoder_84248
decoder_84251
decoder_84253
decoder_84255
decoder_84257
decoder_84259
decoder_84261
decoder_84263
decoder_84265
decoder_84267
decoder_84269
decoder_84271
decoder_84273
identity??Decoder/StatefulPartitionedCall?Encoder/StatefulPartitionedCall?
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_encoderencoder_84230encoder_84232encoder_84234encoder_84236encoder_84238encoder_84240encoder_84242encoder_84244encoder_84246encoder_84248*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_836012!
Encoder/StatefulPartitionedCall?
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_84251decoder_84253decoder_84255decoder_84257decoder_84259decoder_84261decoder_84263decoder_84265decoder_84267decoder_84269decoder_84271decoder_84273*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_840412!
Decoder/StatefulPartitionedCall?
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?

?
D__inference_conv4_dec_layer_call_and_return_conditional_losses_85416

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
C
'__inference_upsamp2_layer_call_fn_83681

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp2_layer_call_and_return_conditional_losses_836752
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
>__inference_CAE_layer_call_and_return_conditional_losses_84330

inputs
encoder_84283
encoder_84285
encoder_84287
encoder_84289
encoder_84291
encoder_84293
encoder_84295
encoder_84297
encoder_84299
encoder_84301
decoder_84304
decoder_84306
decoder_84308
decoder_84310
decoder_84312
decoder_84314
decoder_84316
decoder_84318
decoder_84320
decoder_84322
decoder_84324
decoder_84326
identity??Decoder/StatefulPartitionedCall?Encoder/StatefulPartitionedCall?
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_84283encoder_84285encoder_84287encoder_84289encoder_84291encoder_84293encoder_84295encoder_84297encoder_84299encoder_84301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_835422!
Encoder/StatefulPartitionedCall?
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:0decoder_84304decoder_84306decoder_84308decoder_84310decoder_84312decoder_84314decoder_84316decoder_84318decoder_84320decoder_84322decoder_84324decoder_84326*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_839732!
Decoder/StatefulPartitionedCall?
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv1_dec_layer_call_and_return_conditional_losses_85476

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
C
'__inference_upsamp4_layer_call_fn_83643

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp4_layer_call_and_return_conditional_losses_836372
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_83436

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_conv4_enc_layer_call_and_return_conditional_losses_85328

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
D__inference_conv1_enc_layer_call_and_return_conditional_losses_85268

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
B__inference_Decoder_layer_call_and_return_conditional_losses_84041

inputs
decoding_84005
decoding_84007
conv4_dec_84011
conv4_dec_84013
conv3_dec_84017
conv3_dec_84019
conv2_dec_84023
conv2_dec_84025
conv1_dec_84029
conv1_dec_84031
output_84035
output_84037
identity??!conv1_dec/StatefulPartitionedCall?!conv2_dec/StatefulPartitionedCall?!conv3_dec/StatefulPartitionedCall?!conv4_dec/StatefulPartitionedCall? decoding/StatefulPartitionedCall?output/StatefulPartitionedCall?
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_84005decoding_84007*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoding_layer_call_and_return_conditional_losses_837142"
 decoding/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_837442
reshape/PartitionedCall?
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_84011conv4_dec_84013*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_dec_layer_call_and_return_conditional_losses_837632#
!conv4_dec/StatefulPartitionedCall?
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp4_layer_call_and_return_conditional_losses_836372
upsamp4/PartitionedCall?
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_84017conv3_dec_84019*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_dec_layer_call_and_return_conditional_losses_837912#
!conv3_dec/StatefulPartitionedCall?
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp3_layer_call_and_return_conditional_losses_836562
upsamp3/PartitionedCall?
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_84023conv2_dec_84025*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_dec_layer_call_and_return_conditional_losses_838192#
!conv2_dec/StatefulPartitionedCall?
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp2_layer_call_and_return_conditional_losses_836752
upsamp2/PartitionedCall?
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_84029conv1_dec_84031*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_dec_layer_call_and_return_conditional_losses_838472#
!conv1_dec/StatefulPartitionedCall?
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp1_layer_call_and_return_conditional_losses_836942
upsamp1/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_84035output_84037*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_838752 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_conv3_enc_layer_call_and_return_conditional_losses_85308

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv4_enc_layer_call_and_return_conditional_losses_83413

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_84535
input_encoder
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_encoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*8
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_832662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?
~
)__inference_conv3_enc_layer_call_fn_85317

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_enc_layer_call_and_return_conditional_losses_833852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?+
?
B__inference_Encoder_layer_call_and_return_conditional_losses_83542

inputs
conv1_enc_83511
conv1_enc_83513
conv2_enc_83517
conv2_enc_83519
conv3_enc_83523
conv3_enc_83525
conv4_enc_83529
conv4_enc_83531
bottleneck_83536
bottleneck_83538
identity??"bottleneck/StatefulPartitionedCall?!conv1_enc/StatefulPartitionedCall?!conv2_enc/StatefulPartitionedCall?!conv3_enc/StatefulPartitionedCall?!conv4_enc/StatefulPartitionedCall?
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_83511conv1_enc_83513*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_enc_layer_call_and_return_conditional_losses_833292#
!conv1_enc/StatefulPartitionedCall?
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool1_layer_call_and_return_conditional_losses_832722
maxpool1/PartitionedCall?
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_83517conv2_enc_83519*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_enc_layer_call_and_return_conditional_losses_833572#
!conv2_enc/StatefulPartitionedCall?
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool2_layer_call_and_return_conditional_losses_832842
maxpool2/PartitionedCall?
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_83523conv3_enc_83525*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_enc_layer_call_and_return_conditional_losses_833852#
!conv3_enc/StatefulPartitionedCall?
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool3_layer_call_and_return_conditional_losses_832962
maxpool3/PartitionedCall?
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_83529conv4_enc_83531*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_enc_layer_call_and_return_conditional_losses_834132#
!conv4_enc/StatefulPartitionedCall?
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool4_layer_call_and_return_conditional_losses_833082
maxpool4/PartitionedCall?
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_834362
flatten/PartitionedCall?
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_83536bottleneck_83538*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_bottleneck_layer_call_and_return_conditional_losses_834542$
"bottleneck/StatefulPartitionedCall?
IdentityIdentity+bottleneck/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_conv4_dec_layer_call_and_return_conditional_losses_83763

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv1_dec_layer_call_and_return_conditional_losses_83847

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
'__inference_Decoder_layer_call_fn_85257

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Decoder_layer_call_and_return_conditional_losses_840412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?0
?
B__inference_Decoder_layer_call_and_return_conditional_losses_83973

inputs
decoding_83937
decoding_83939
conv4_dec_83943
conv4_dec_83945
conv3_dec_83949
conv3_dec_83951
conv2_dec_83955
conv2_dec_83957
conv1_dec_83961
conv1_dec_83963
output_83967
output_83969
identity??!conv1_dec/StatefulPartitionedCall?!conv2_dec/StatefulPartitionedCall?!conv3_dec/StatefulPartitionedCall?!conv4_dec/StatefulPartitionedCall? decoding/StatefulPartitionedCall?output/StatefulPartitionedCall?
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_83937decoding_83939*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoding_layer_call_and_return_conditional_losses_837142"
 decoding/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_837442
reshape/PartitionedCall?
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_83943conv4_dec_83945*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_dec_layer_call_and_return_conditional_losses_837632#
!conv4_dec/StatefulPartitionedCall?
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp4_layer_call_and_return_conditional_losses_836372
upsamp4/PartitionedCall?
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_83949conv3_dec_83951*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_dec_layer_call_and_return_conditional_losses_837912#
!conv3_dec/StatefulPartitionedCall?
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp3_layer_call_and_return_conditional_losses_836562
upsamp3/PartitionedCall?
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_83955conv2_dec_83957*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_dec_layer_call_and_return_conditional_losses_838192#
!conv2_dec/StatefulPartitionedCall?
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp2_layer_call_and_return_conditional_losses_836752
upsamp2/PartitionedCall?
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_83961conv1_dec_83963*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_dec_layer_call_and_return_conditional_losses_838472#
!conv1_dec/StatefulPartitionedCall?
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp1_layer_call_and_return_conditional_losses_836942
upsamp1/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_83967output_83969*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_838752 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
C__inference_decoding_layer_call_and_return_conditional_losses_85377

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
}
(__inference_decoding_layer_call_fn_85386

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoding_layer_call_and_return_conditional_losses_837142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_upsamp1_layer_call_fn_83700

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp1_layer_call_and_return_conditional_losses_836942
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2_dec_layer_call_fn_85465

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_dec_layer_call_and_return_conditional_losses_838192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
~
)__inference_conv1_dec_layer_call_fn_85485

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_dec_layer_call_and_return_conditional_losses_838472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?o
?
B__inference_Decoder_layer_call_and_return_conditional_losses_85112

inputs+
'decoding_matmul_readvariableop_resource,
(decoding_biasadd_readvariableop_resource,
(conv4_dec_conv2d_readvariableop_resource-
)conv4_dec_biasadd_readvariableop_resource,
(conv3_dec_conv2d_readvariableop_resource-
)conv3_dec_biasadd_readvariableop_resource,
(conv2_dec_conv2d_readvariableop_resource-
)conv2_dec_biasadd_readvariableop_resource,
(conv1_dec_conv2d_readvariableop_resource-
)conv1_dec_biasadd_readvariableop_resource)
%output_conv2d_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity?? conv1_dec/BiasAdd/ReadVariableOp?conv1_dec/Conv2D/ReadVariableOp? conv2_dec/BiasAdd/ReadVariableOp?conv2_dec/Conv2D/ReadVariableOp? conv3_dec/BiasAdd/ReadVariableOp?conv3_dec/Conv2D/ReadVariableOp? conv4_dec/BiasAdd/ReadVariableOp?conv4_dec/Conv2D/ReadVariableOp?decoding/BiasAdd/ReadVariableOp?decoding/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/Conv2D/ReadVariableOp?
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
decoding/MatMul/ReadVariableOp?
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoding/MatMul?
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
decoding/BiasAdd/ReadVariableOp?
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoding/BiasAddg
reshape/ShapeShapedecoding/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedecoding/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshape?
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv4_dec/Conv2D/ReadVariableOp?
conv4_dec/Conv2DConv2Dreshape/Reshape:output:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv4_dec/Conv2D?
 conv4_dec/BiasAdd/ReadVariableOpReadVariableOp)conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv4_dec/BiasAdd/ReadVariableOp?
conv4_dec/BiasAddBiasAddconv4_dec/Conv2D:output:0(conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv4_dec/BiasAdd
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv4_dec/Reluj
upsamp4/ShapeShapeconv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp4/Shape?
upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack?
upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack_1?
upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack_2?
upsamp4/strided_sliceStridedSliceupsamp4/Shape:output:0$upsamp4/strided_slice/stack:output:0&upsamp4/strided_slice/stack_1:output:0&upsamp4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp4/strided_sliceo
upsamp4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp4/Const~
upsamp4/mulMulupsamp4/strided_slice:output:0upsamp4/Const:output:0*
T0*
_output_shapes
:2
upsamp4/mul?
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighbor?
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02!
conv3_dec/Conv2D/ReadVariableOp?
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv3_dec/Conv2D?
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_dec/BiasAdd/ReadVariableOp?
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv3_dec/Reluj
upsamp3/ShapeShapeconv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp3/Shape?
upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack?
upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack_1?
upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack_2?
upsamp3/strided_sliceStridedSliceupsamp3/Shape:output:0$upsamp3/strided_slice/stack:output:0&upsamp3/strided_slice/stack_1:output:0&upsamp3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp3/strided_sliceo
upsamp3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp3/Const~
upsamp3/mulMulupsamp3/strided_slice:output:0upsamp3/Const:output:0*
T0*
_output_shapes
:2
upsamp3/mul?
$upsamp3/resize/ResizeNearestNeighborResizeNearestNeighborconv3_dec/Relu:activations:0upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighbor?
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2_dec/Conv2D/ReadVariableOp?
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2_dec/Conv2D?
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_dec/BiasAdd/ReadVariableOp?
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2_dec/Reluj
upsamp2/ShapeShapeconv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp2/Shape?
upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack?
upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack_1?
upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack_2?
upsamp2/strided_sliceStridedSliceupsamp2/Shape:output:0$upsamp2/strided_slice/stack:output:0&upsamp2/strided_slice/stack_1:output:0&upsamp2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp2/strided_sliceo
upsamp2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp2/Const~
upsamp2/mulMulupsamp2/strided_slice:output:0upsamp2/Const:output:0*
T0*
_output_shapes
:2
upsamp2/mul?
$upsamp2/resize/ResizeNearestNeighborResizeNearestNeighborconv2_dec/Relu:activations:0upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighbor?
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv1_dec/Conv2D/ReadVariableOp?
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1_dec/Conv2D?
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOp?
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv1_dec/Reluj
upsamp1/ShapeShapeconv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp1/Shape?
upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack?
upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack_1?
upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack_2?
upsamp1/strided_sliceStridedSliceupsamp1/Shape:output:0$upsamp1/strided_slice/stack:output:0&upsamp1/strided_slice/stack_1:output:0&upsamp1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp1/strided_sliceo
upsamp1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp1/Const~
upsamp1/mulMulupsamp1/strided_slice:output:0upsamp1/Const:output:0*
T0*
_output_shapes
:2
upsamp1/mul?
$upsamp1/resize/ResizeNearestNeighborResizeNearestNeighborconv1_dec/Relu:activations:0upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighbor?
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
output/Conv2D/ReadVariableOp?
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
output/Conv2D?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
output/Sigmoid?
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2D
 conv1_dec/BiasAdd/ReadVariableOp conv1_dec/BiasAdd/ReadVariableOp2B
conv1_dec/Conv2D/ReadVariableOpconv1_dec/Conv2D/ReadVariableOp2D
 conv2_dec/BiasAdd/ReadVariableOp conv2_dec/BiasAdd/ReadVariableOp2B
conv2_dec/Conv2D/ReadVariableOpconv2_dec/Conv2D/ReadVariableOp2D
 conv3_dec/BiasAdd/ReadVariableOp conv3_dec/BiasAdd/ReadVariableOp2B
conv3_dec/Conv2D/ReadVariableOpconv3_dec/Conv2D/ReadVariableOp2D
 conv4_dec/BiasAdd/ReadVariableOp conv4_dec/BiasAdd/ReadVariableOp2B
conv4_dec/Conv2D/ReadVariableOpconv4_dec/Conv2D/ReadVariableOp2B
decoding/BiasAdd/ReadVariableOpdecoding/BiasAdd/ReadVariableOp2@
decoding/MatMul/ReadVariableOpdecoding/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/Conv2D/ReadVariableOpoutput/Conv2D/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_conv2_enc_layer_call_and_return_conditional_losses_85288

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????


 
_user_specified_nameinputs
??
?
>__inference_CAE_layer_call_and_return_conditional_losses_84662

inputs4
0encoder_conv1_enc_conv2d_readvariableop_resource5
1encoder_conv1_enc_biasadd_readvariableop_resource4
0encoder_conv2_enc_conv2d_readvariableop_resource5
1encoder_conv2_enc_biasadd_readvariableop_resource4
0encoder_conv3_enc_conv2d_readvariableop_resource5
1encoder_conv3_enc_biasadd_readvariableop_resource4
0encoder_conv4_enc_conv2d_readvariableop_resource5
1encoder_conv4_enc_biasadd_readvariableop_resource5
1encoder_bottleneck_matmul_readvariableop_resource6
2encoder_bottleneck_biasadd_readvariableop_resource3
/decoder_decoding_matmul_readvariableop_resource4
0decoder_decoding_biasadd_readvariableop_resource4
0decoder_conv4_dec_conv2d_readvariableop_resource5
1decoder_conv4_dec_biasadd_readvariableop_resource4
0decoder_conv3_dec_conv2d_readvariableop_resource5
1decoder_conv3_dec_biasadd_readvariableop_resource4
0decoder_conv2_dec_conv2d_readvariableop_resource5
1decoder_conv2_dec_biasadd_readvariableop_resource4
0decoder_conv1_dec_conv2d_readvariableop_resource5
1decoder_conv1_dec_biasadd_readvariableop_resource1
-decoder_output_conv2d_readvariableop_resource2
.decoder_output_biasadd_readvariableop_resource
identity??(Decoder/conv1_dec/BiasAdd/ReadVariableOp?'Decoder/conv1_dec/Conv2D/ReadVariableOp?(Decoder/conv2_dec/BiasAdd/ReadVariableOp?'Decoder/conv2_dec/Conv2D/ReadVariableOp?(Decoder/conv3_dec/BiasAdd/ReadVariableOp?'Decoder/conv3_dec/Conv2D/ReadVariableOp?(Decoder/conv4_dec/BiasAdd/ReadVariableOp?'Decoder/conv4_dec/Conv2D/ReadVariableOp?'Decoder/decoding/BiasAdd/ReadVariableOp?&Decoder/decoding/MatMul/ReadVariableOp?%Decoder/output/BiasAdd/ReadVariableOp?$Decoder/output/Conv2D/ReadVariableOp?)Encoder/bottleneck/BiasAdd/ReadVariableOp?(Encoder/bottleneck/MatMul/ReadVariableOp?(Encoder/conv1_enc/BiasAdd/ReadVariableOp?'Encoder/conv1_enc/Conv2D/ReadVariableOp?(Encoder/conv2_enc/BiasAdd/ReadVariableOp?'Encoder/conv2_enc/Conv2D/ReadVariableOp?(Encoder/conv3_enc/BiasAdd/ReadVariableOp?'Encoder/conv3_enc/Conv2D/ReadVariableOp?(Encoder/conv4_enc/BiasAdd/ReadVariableOp?'Encoder/conv4_enc/Conv2D/ReadVariableOp?
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOp?
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2D?
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOp?
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Encoder/conv1_enc/BiasAdd?
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Encoder/conv1_enc/Relu?
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:?????????

*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPool?
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOp?
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2D?
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOp?
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
Encoder/conv2_enc/BiasAdd?
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
Encoder/conv2_enc/Relu?
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPool?
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOp?
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2D?
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOp?
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Encoder/conv3_enc/BiasAdd?
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Encoder/conv3_enc/Relu?
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPool?
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOp?
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2D?
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOp?
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
Encoder/conv4_enc/BiasAdd?
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Encoder/conv4_enc/Relu?
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool4/MaxPool
Encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Encoder/flatten/Const?
Encoder/flatten/ReshapeReshape!Encoder/maxpool4/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
Encoder/flatten/Reshape?
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOp?
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/MatMul?
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOp?
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/BiasAdd?
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOp?
Decoder/decoding/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Decoder/decoding/MatMul?
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOp?
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Decoder/decoding/BiasAdd
Decoder/reshape/ShapeShape!Decoder/decoding/BiasAdd:output:0*
T0*
_output_shapes
:2
Decoder/reshape/Shape?
#Decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Decoder/reshape/strided_slice/stack?
%Decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/reshape/strided_slice/stack_1?
%Decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/reshape/strided_slice/stack_2?
Decoder/reshape/strided_sliceStridedSliceDecoder/reshape/Shape:output:0,Decoder/reshape/strided_slice/stack:output:0.Decoder/reshape/strided_slice/stack_1:output:0.Decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Decoder/reshape/strided_slice?
Decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
Decoder/reshape/Reshape/shape/1?
Decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
Decoder/reshape/Reshape/shape/2?
Decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2!
Decoder/reshape/Reshape/shape/3?
Decoder/reshape/Reshape/shapePack&Decoder/reshape/strided_slice:output:0(Decoder/reshape/Reshape/shape/1:output:0(Decoder/reshape/Reshape/shape/2:output:0(Decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Decoder/reshape/Reshape/shape?
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
Decoder/reshape/Reshape?
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOp?
Decoder/conv4_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2D?
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOp?
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
Decoder/conv4_dec/BiasAdd?
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Decoder/conv4_dec/Relu?
Decoder/upsamp4/ShapeShape$Decoder/conv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp4/Shape?
#Decoder/upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp4/strided_slice/stack?
%Decoder/upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp4/strided_slice/stack_1?
%Decoder/upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp4/strided_slice/stack_2?
Decoder/upsamp4/strided_sliceStridedSliceDecoder/upsamp4/Shape:output:0,Decoder/upsamp4/strided_slice/stack:output:0.Decoder/upsamp4/strided_slice/stack_1:output:0.Decoder/upsamp4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp4/strided_slice
Decoder/upsamp4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp4/Const?
Decoder/upsamp4/mulMul&Decoder/upsamp4/strided_slice:output:0Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp4/mul?
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighbor?
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp?
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2D?
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOp?
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Decoder/conv3_dec/BiasAdd?
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Decoder/conv3_dec/Relu?
Decoder/upsamp3/ShapeShape$Decoder/conv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp3/Shape?
#Decoder/upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp3/strided_slice/stack?
%Decoder/upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp3/strided_slice/stack_1?
%Decoder/upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp3/strided_slice/stack_2?
Decoder/upsamp3/strided_sliceStridedSliceDecoder/upsamp3/Shape:output:0,Decoder/upsamp3/strided_slice/stack:output:0.Decoder/upsamp3/strided_slice/stack_1:output:0.Decoder/upsamp3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp3/strided_slice
Decoder/upsamp3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp3/Const?
Decoder/upsamp3/mulMul&Decoder/upsamp3/strided_slice:output:0Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp3/mul?
,Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv3_dec/Relu:activations:0Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighbor?
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp?
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2D?
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOp?
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv2_dec/BiasAdd?
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv2_dec/Relu?
Decoder/upsamp2/ShapeShape$Decoder/conv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp2/Shape?
#Decoder/upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp2/strided_slice/stack?
%Decoder/upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp2/strided_slice/stack_1?
%Decoder/upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp2/strided_slice/stack_2?
Decoder/upsamp2/strided_sliceStridedSliceDecoder/upsamp2/Shape:output:0,Decoder/upsamp2/strided_slice/stack:output:0.Decoder/upsamp2/strided_slice/stack_1:output:0.Decoder/upsamp2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp2/strided_slice
Decoder/upsamp2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp2/Const?
Decoder/upsamp2/mulMul&Decoder/upsamp2/strided_slice:output:0Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp2/mul?
,Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv2_dec/Relu:activations:0Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighbor?
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp?
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2D?
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOp?
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Decoder/conv1_dec/BiasAdd?
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Decoder/conv1_dec/Relu?
Decoder/upsamp1/ShapeShape$Decoder/conv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp1/Shape?
#Decoder/upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp1/strided_slice/stack?
%Decoder/upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp1/strided_slice/stack_1?
%Decoder/upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp1/strided_slice/stack_2?
Decoder/upsamp1/strided_sliceStridedSliceDecoder/upsamp1/Shape:output:0,Decoder/upsamp1/strided_slice/stack:output:0.Decoder/upsamp1/strided_slice/stack_1:output:0.Decoder/upsamp1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp1/strided_slice
Decoder/upsamp1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp1/Const?
Decoder/upsamp1/mulMul&Decoder/upsamp1/strided_slice:output:0Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp1/mul?
,Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv1_dec/Relu:activations:0Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighbor?
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp?
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Decoder/output/Conv2D?
%Decoder/output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Decoder/output/BiasAdd/ReadVariableOp?
Decoder/output/BiasAddBiasAddDecoder/output/Conv2D:output:0-Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Decoder/output/BiasAdd?
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Decoder/output/Sigmoid?
IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::2T
(Decoder/conv1_dec/BiasAdd/ReadVariableOp(Decoder/conv1_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv1_dec/Conv2D/ReadVariableOp'Decoder/conv1_dec/Conv2D/ReadVariableOp2T
(Decoder/conv2_dec/BiasAdd/ReadVariableOp(Decoder/conv2_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv2_dec/Conv2D/ReadVariableOp'Decoder/conv2_dec/Conv2D/ReadVariableOp2T
(Decoder/conv3_dec/BiasAdd/ReadVariableOp(Decoder/conv3_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv3_dec/Conv2D/ReadVariableOp'Decoder/conv3_dec/Conv2D/ReadVariableOp2T
(Decoder/conv4_dec/BiasAdd/ReadVariableOp(Decoder/conv4_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv4_dec/Conv2D/ReadVariableOp'Decoder/conv4_dec/Conv2D/ReadVariableOp2R
'Decoder/decoding/BiasAdd/ReadVariableOp'Decoder/decoding/BiasAdd/ReadVariableOp2P
&Decoder/decoding/MatMul/ReadVariableOp&Decoder/decoding/MatMul/ReadVariableOp2N
%Decoder/output/BiasAdd/ReadVariableOp%Decoder/output/BiasAdd/ReadVariableOp2L
$Decoder/output/Conv2D/ReadVariableOp$Decoder/output/Conv2D/ReadVariableOp2V
)Encoder/bottleneck/BiasAdd/ReadVariableOp)Encoder/bottleneck/BiasAdd/ReadVariableOp2T
(Encoder/bottleneck/MatMul/ReadVariableOp(Encoder/bottleneck/MatMul/ReadVariableOp2T
(Encoder/conv1_enc/BiasAdd/ReadVariableOp(Encoder/conv1_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv1_enc/Conv2D/ReadVariableOp'Encoder/conv1_enc/Conv2D/ReadVariableOp2T
(Encoder/conv2_enc/BiasAdd/ReadVariableOp(Encoder/conv2_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv2_enc/Conv2D/ReadVariableOp'Encoder/conv2_enc/Conv2D/ReadVariableOp2T
(Encoder/conv3_enc/BiasAdd/ReadVariableOp(Encoder/conv3_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv3_enc/Conv2D/ReadVariableOp'Encoder/conv3_enc/Conv2D/ReadVariableOp2T
(Encoder/conv4_enc/BiasAdd/ReadVariableOp(Encoder/conv4_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv4_enc/Conv2D/ReadVariableOp'Encoder/conv4_enc/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
D
(__inference_maxpool3_layer_call_fn_83302

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool3_layer_call_and_return_conditional_losses_832962
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?+
?
B__inference_Encoder_layer_call_and_return_conditional_losses_83505
input_encoder
conv1_enc_83474
conv1_enc_83476
conv2_enc_83480
conv2_enc_83482
conv3_enc_83486
conv3_enc_83488
conv4_enc_83492
conv4_enc_83494
bottleneck_83499
bottleneck_83501
identity??"bottleneck/StatefulPartitionedCall?!conv1_enc/StatefulPartitionedCall?!conv2_enc/StatefulPartitionedCall?!conv3_enc/StatefulPartitionedCall?!conv4_enc/StatefulPartitionedCall?
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_83474conv1_enc_83476*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_enc_layer_call_and_return_conditional_losses_833292#
!conv1_enc/StatefulPartitionedCall?
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool1_layer_call_and_return_conditional_losses_832722
maxpool1/PartitionedCall?
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_83480conv2_enc_83482*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_enc_layer_call_and_return_conditional_losses_833572#
!conv2_enc/StatefulPartitionedCall?
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool2_layer_call_and_return_conditional_losses_832842
maxpool2/PartitionedCall?
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_83486conv3_enc_83488*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_enc_layer_call_and_return_conditional_losses_833852#
!conv3_enc/StatefulPartitionedCall?
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool3_layer_call_and_return_conditional_losses_832962
maxpool3/PartitionedCall?
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_83492conv4_enc_83494*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_enc_layer_call_and_return_conditional_losses_834132#
!conv4_enc/StatefulPartitionedCall?
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_maxpool4_layer_call_and_return_conditional_losses_833082
maxpool4/PartitionedCall?
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_834362
flatten/PartitionedCall?
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_83499bottleneck_83501*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_bottleneck_layer_call_and_return_conditional_losses_834542$
"bottleneck/StatefulPartitionedCall?
IdentityIdentity+bottleneck/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?0
?
B__inference_Decoder_layer_call_and_return_conditional_losses_83931
input_decoder
decoding_83895
decoding_83897
conv4_dec_83901
conv4_dec_83903
conv3_dec_83907
conv3_dec_83909
conv2_dec_83913
conv2_dec_83915
conv1_dec_83919
conv1_dec_83921
output_83925
output_83927
identity??!conv1_dec/StatefulPartitionedCall?!conv2_dec/StatefulPartitionedCall?!conv3_dec/StatefulPartitionedCall?!conv4_dec/StatefulPartitionedCall? decoding/StatefulPartitionedCall?output/StatefulPartitionedCall?
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_83895decoding_83897*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_decoding_layer_call_and_return_conditional_losses_837142"
 decoding/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_837442
reshape/PartitionedCall?
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_83901conv4_dec_83903*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv4_dec_layer_call_and_return_conditional_losses_837632#
!conv4_dec/StatefulPartitionedCall?
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp4_layer_call_and_return_conditional_losses_836372
upsamp4/PartitionedCall?
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_83907conv3_dec_83909*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv3_dec_layer_call_and_return_conditional_losses_837912#
!conv3_dec/StatefulPartitionedCall?
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp3_layer_call_and_return_conditional_losses_836562
upsamp3/PartitionedCall?
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_83913conv2_dec_83915*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2_dec_layer_call_and_return_conditional_losses_838192#
!conv2_dec/StatefulPartitionedCall?
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp2_layer_call_and_return_conditional_losses_836752
upsamp2/PartitionedCall?
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_83919conv1_dec_83921*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv1_dec_layer_call_and_return_conditional_losses_838472#
!conv1_dec/StatefulPartitionedCall?
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_upsamp1_layer_call_and_return_conditional_losses_836942
upsamp1/PartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_83925output_83927*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_output_layer_call_and_return_conditional_losses_838752 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameinput_decoder
?
?
'__inference_Encoder_layer_call_fn_85025

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_Encoder_layer_call_and_return_conditional_losses_836012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_83744

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_83266
input_encoder8
4cae_encoder_conv1_enc_conv2d_readvariableop_resource9
5cae_encoder_conv1_enc_biasadd_readvariableop_resource8
4cae_encoder_conv2_enc_conv2d_readvariableop_resource9
5cae_encoder_conv2_enc_biasadd_readvariableop_resource8
4cae_encoder_conv3_enc_conv2d_readvariableop_resource9
5cae_encoder_conv3_enc_biasadd_readvariableop_resource8
4cae_encoder_conv4_enc_conv2d_readvariableop_resource9
5cae_encoder_conv4_enc_biasadd_readvariableop_resource9
5cae_encoder_bottleneck_matmul_readvariableop_resource:
6cae_encoder_bottleneck_biasadd_readvariableop_resource7
3cae_decoder_decoding_matmul_readvariableop_resource8
4cae_decoder_decoding_biasadd_readvariableop_resource8
4cae_decoder_conv4_dec_conv2d_readvariableop_resource9
5cae_decoder_conv4_dec_biasadd_readvariableop_resource8
4cae_decoder_conv3_dec_conv2d_readvariableop_resource9
5cae_decoder_conv3_dec_biasadd_readvariableop_resource8
4cae_decoder_conv2_dec_conv2d_readvariableop_resource9
5cae_decoder_conv2_dec_biasadd_readvariableop_resource8
4cae_decoder_conv1_dec_conv2d_readvariableop_resource9
5cae_decoder_conv1_dec_biasadd_readvariableop_resource5
1cae_decoder_output_conv2d_readvariableop_resource6
2cae_decoder_output_biasadd_readvariableop_resource
identity??,CAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp?+CAE/Decoder/conv1_dec/Conv2D/ReadVariableOp?,CAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp?+CAE/Decoder/conv2_dec/Conv2D/ReadVariableOp?,CAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp?+CAE/Decoder/conv3_dec/Conv2D/ReadVariableOp?,CAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp?+CAE/Decoder/conv4_dec/Conv2D/ReadVariableOp?+CAE/Decoder/decoding/BiasAdd/ReadVariableOp?*CAE/Decoder/decoding/MatMul/ReadVariableOp?)CAE/Decoder/output/BiasAdd/ReadVariableOp?(CAE/Decoder/output/Conv2D/ReadVariableOp?-CAE/Encoder/bottleneck/BiasAdd/ReadVariableOp?,CAE/Encoder/bottleneck/MatMul/ReadVariableOp?,CAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp?+CAE/Encoder/conv1_enc/Conv2D/ReadVariableOp?,CAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp?+CAE/Encoder/conv2_enc/Conv2D/ReadVariableOp?,CAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp?+CAE/Encoder/conv3_enc/Conv2D/ReadVariableOp?,CAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp?+CAE/Encoder/conv4_enc/Conv2D/ReadVariableOp?
+CAE/Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp4cae_encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+CAE/Encoder/conv1_enc/Conv2D/ReadVariableOp?
CAE/Encoder/conv1_enc/Conv2DConv2Dinput_encoder3CAE/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
CAE/Encoder/conv1_enc/Conv2D?
,CAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp5cae_encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,CAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp?
CAE/Encoder/conv1_enc/BiasAddBiasAdd%CAE/Encoder/conv1_enc/Conv2D:output:04CAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
CAE/Encoder/conv1_enc/BiasAdd?
CAE/Encoder/conv1_enc/ReluRelu&CAE/Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
CAE/Encoder/conv1_enc/Relu?
CAE/Encoder/maxpool1/MaxPoolMaxPool(CAE/Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:?????????

*
ksize
*
paddingSAME*
strides
2
CAE/Encoder/maxpool1/MaxPool?
+CAE/Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp4cae_encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+CAE/Encoder/conv2_enc/Conv2D/ReadVariableOp?
CAE/Encoder/conv2_enc/Conv2DConv2D%CAE/Encoder/maxpool1/MaxPool:output:03CAE/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 *
paddingSAME*
strides
2
CAE/Encoder/conv2_enc/Conv2D?
,CAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp5cae_encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,CAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp?
CAE/Encoder/conv2_enc/BiasAddBiasAdd%CAE/Encoder/conv2_enc/Conv2D:output:04CAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

 2
CAE/Encoder/conv2_enc/BiasAdd?
CAE/Encoder/conv2_enc/ReluRelu&CAE/Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

 2
CAE/Encoder/conv2_enc/Relu?
CAE/Encoder/maxpool2/MaxPoolMaxPool(CAE/Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
CAE/Encoder/maxpool2/MaxPool?
+CAE/Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp4cae_encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+CAE/Encoder/conv3_enc/Conv2D/ReadVariableOp?
CAE/Encoder/conv3_enc/Conv2DConv2D%CAE/Encoder/maxpool2/MaxPool:output:03CAE/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
CAE/Encoder/conv3_enc/Conv2D?
,CAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp5cae_encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,CAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp?
CAE/Encoder/conv3_enc/BiasAddBiasAdd%CAE/Encoder/conv3_enc/Conv2D:output:04CAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
CAE/Encoder/conv3_enc/BiasAdd?
CAE/Encoder/conv3_enc/ReluRelu&CAE/Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
CAE/Encoder/conv3_enc/Relu?
CAE/Encoder/maxpool3/MaxPoolMaxPool(CAE/Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
CAE/Encoder/maxpool3/MaxPool?
+CAE/Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp4cae_encoder_conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype02-
+CAE/Encoder/conv4_enc/Conv2D/ReadVariableOp?
CAE/Encoder/conv4_enc/Conv2DConv2D%CAE/Encoder/maxpool3/MaxPool:output:03CAE/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
CAE/Encoder/conv4_enc/Conv2D?
,CAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp5cae_encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,CAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp?
CAE/Encoder/conv4_enc/BiasAddBiasAdd%CAE/Encoder/conv4_enc/Conv2D:output:04CAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
CAE/Encoder/conv4_enc/BiasAdd?
CAE/Encoder/conv4_enc/ReluRelu&CAE/Encoder/conv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
CAE/Encoder/conv4_enc/Relu?
CAE/Encoder/maxpool4/MaxPoolMaxPool(CAE/Encoder/conv4_enc/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingSAME*
strides
2
CAE/Encoder/maxpool4/MaxPool?
CAE/Encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
CAE/Encoder/flatten/Const?
CAE/Encoder/flatten/ReshapeReshape%CAE/Encoder/maxpool4/MaxPool:output:0"CAE/Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
CAE/Encoder/flatten/Reshape?
,CAE/Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp5cae_encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,CAE/Encoder/bottleneck/MatMul/ReadVariableOp?
CAE/Encoder/bottleneck/MatMulMatMul$CAE/Encoder/flatten/Reshape:output:04CAE/Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
CAE/Encoder/bottleneck/MatMul?
-CAE/Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp6cae_encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-CAE/Encoder/bottleneck/BiasAdd/ReadVariableOp?
CAE/Encoder/bottleneck/BiasAddBiasAdd'CAE/Encoder/bottleneck/MatMul:product:05CAE/Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
CAE/Encoder/bottleneck/BiasAdd?
*CAE/Decoder/decoding/MatMul/ReadVariableOpReadVariableOp3cae_decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02,
*CAE/Decoder/decoding/MatMul/ReadVariableOp?
CAE/Decoder/decoding/MatMulMatMul'CAE/Encoder/bottleneck/BiasAdd:output:02CAE/Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CAE/Decoder/decoding/MatMul?
+CAE/Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp4cae_decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+CAE/Decoder/decoding/BiasAdd/ReadVariableOp?
CAE/Decoder/decoding/BiasAddBiasAdd%CAE/Decoder/decoding/MatMul:product:03CAE/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
CAE/Decoder/decoding/BiasAdd?
CAE/Decoder/reshape/ShapeShape%CAE/Decoder/decoding/BiasAdd:output:0*
T0*
_output_shapes
:2
CAE/Decoder/reshape/Shape?
'CAE/Decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'CAE/Decoder/reshape/strided_slice/stack?
)CAE/Decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/reshape/strided_slice/stack_1?
)CAE/Decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/reshape/strided_slice/stack_2?
!CAE/Decoder/reshape/strided_sliceStridedSlice"CAE/Decoder/reshape/Shape:output:00CAE/Decoder/reshape/strided_slice/stack:output:02CAE/Decoder/reshape/strided_slice/stack_1:output:02CAE/Decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!CAE/Decoder/reshape/strided_slice?
#CAE/Decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#CAE/Decoder/reshape/Reshape/shape/1?
#CAE/Decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#CAE/Decoder/reshape/Reshape/shape/2?
#CAE/Decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2%
#CAE/Decoder/reshape/Reshape/shape/3?
!CAE/Decoder/reshape/Reshape/shapePack*CAE/Decoder/reshape/strided_slice:output:0,CAE/Decoder/reshape/Reshape/shape/1:output:0,CAE/Decoder/reshape/Reshape/shape/2:output:0,CAE/Decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!CAE/Decoder/reshape/Reshape/shape?
CAE/Decoder/reshape/ReshapeReshape%CAE/Decoder/decoding/BiasAdd:output:0*CAE/Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
CAE/Decoder/reshape/Reshape?
+CAE/Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp4cae_decoder_conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+CAE/Decoder/conv4_dec/Conv2D/ReadVariableOp?
CAE/Decoder/conv4_dec/Conv2DConv2D$CAE/Decoder/reshape/Reshape:output:03CAE/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
CAE/Decoder/conv4_dec/Conv2D?
,CAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp5cae_decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,CAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp?
CAE/Decoder/conv4_dec/BiasAddBiasAdd%CAE/Decoder/conv4_dec/Conv2D:output:04CAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
CAE/Decoder/conv4_dec/BiasAdd?
CAE/Decoder/conv4_dec/ReluRelu&CAE/Decoder/conv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
CAE/Decoder/conv4_dec/Relu?
CAE/Decoder/upsamp4/ShapeShape(CAE/Decoder/conv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
CAE/Decoder/upsamp4/Shape?
'CAE/Decoder/upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'CAE/Decoder/upsamp4/strided_slice/stack?
)CAE/Decoder/upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/upsamp4/strided_slice/stack_1?
)CAE/Decoder/upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/upsamp4/strided_slice/stack_2?
!CAE/Decoder/upsamp4/strided_sliceStridedSlice"CAE/Decoder/upsamp4/Shape:output:00CAE/Decoder/upsamp4/strided_slice/stack:output:02CAE/Decoder/upsamp4/strided_slice/stack_1:output:02CAE/Decoder/upsamp4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!CAE/Decoder/upsamp4/strided_slice?
CAE/Decoder/upsamp4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
CAE/Decoder/upsamp4/Const?
CAE/Decoder/upsamp4/mulMul*CAE/Decoder/upsamp4/strided_slice:output:0"CAE/Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
CAE/Decoder/upsamp4/mul?
0CAE/Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor(CAE/Decoder/conv4_dec/Relu:activations:0CAE/Decoder/upsamp4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(22
0CAE/Decoder/upsamp4/resize/ResizeNearestNeighbor?
+CAE/Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp4cae_decoder_conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype02-
+CAE/Decoder/conv3_dec/Conv2D/ReadVariableOp?
CAE/Decoder/conv3_dec/Conv2DConv2DACAE/Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:03CAE/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
CAE/Decoder/conv3_dec/Conv2D?
,CAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp5cae_decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,CAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp?
CAE/Decoder/conv3_dec/BiasAddBiasAdd%CAE/Decoder/conv3_dec/Conv2D:output:04CAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
CAE/Decoder/conv3_dec/BiasAdd?
CAE/Decoder/conv3_dec/ReluRelu&CAE/Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
CAE/Decoder/conv3_dec/Relu?
CAE/Decoder/upsamp3/ShapeShape(CAE/Decoder/conv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
CAE/Decoder/upsamp3/Shape?
'CAE/Decoder/upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'CAE/Decoder/upsamp3/strided_slice/stack?
)CAE/Decoder/upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/upsamp3/strided_slice/stack_1?
)CAE/Decoder/upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/upsamp3/strided_slice/stack_2?
!CAE/Decoder/upsamp3/strided_sliceStridedSlice"CAE/Decoder/upsamp3/Shape:output:00CAE/Decoder/upsamp3/strided_slice/stack:output:02CAE/Decoder/upsamp3/strided_slice/stack_1:output:02CAE/Decoder/upsamp3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!CAE/Decoder/upsamp3/strided_slice?
CAE/Decoder/upsamp3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
CAE/Decoder/upsamp3/Const?
CAE/Decoder/upsamp3/mulMul*CAE/Decoder/upsamp3/strided_slice:output:0"CAE/Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
CAE/Decoder/upsamp3/mul?
0CAE/Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor(CAE/Decoder/conv3_dec/Relu:activations:0CAE/Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(22
0CAE/Decoder/upsamp3/resize/ResizeNearestNeighbor?
+CAE/Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp4cae_decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02-
+CAE/Decoder/conv2_dec/Conv2D/ReadVariableOp?
CAE/Decoder/conv2_dec/Conv2DConv2DACAE/Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:03CAE/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
CAE/Decoder/conv2_dec/Conv2D?
,CAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp5cae_decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,CAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp?
CAE/Decoder/conv2_dec/BiasAddBiasAdd%CAE/Decoder/conv2_dec/Conv2D:output:04CAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
CAE/Decoder/conv2_dec/BiasAdd?
CAE/Decoder/conv2_dec/ReluRelu&CAE/Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
CAE/Decoder/conv2_dec/Relu?
CAE/Decoder/upsamp2/ShapeShape(CAE/Decoder/conv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
CAE/Decoder/upsamp2/Shape?
'CAE/Decoder/upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'CAE/Decoder/upsamp2/strided_slice/stack?
)CAE/Decoder/upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/upsamp2/strided_slice/stack_1?
)CAE/Decoder/upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/upsamp2/strided_slice/stack_2?
!CAE/Decoder/upsamp2/strided_sliceStridedSlice"CAE/Decoder/upsamp2/Shape:output:00CAE/Decoder/upsamp2/strided_slice/stack:output:02CAE/Decoder/upsamp2/strided_slice/stack_1:output:02CAE/Decoder/upsamp2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!CAE/Decoder/upsamp2/strided_slice?
CAE/Decoder/upsamp2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
CAE/Decoder/upsamp2/Const?
CAE/Decoder/upsamp2/mulMul*CAE/Decoder/upsamp2/strided_slice:output:0"CAE/Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
CAE/Decoder/upsamp2/mul?
0CAE/Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor(CAE/Decoder/conv2_dec/Relu:activations:0CAE/Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(22
0CAE/Decoder/upsamp2/resize/ResizeNearestNeighbor?
+CAE/Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp4cae_decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+CAE/Decoder/conv1_dec/Conv2D/ReadVariableOp?
CAE/Decoder/conv1_dec/Conv2DConv2DACAE/Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:03CAE/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
CAE/Decoder/conv1_dec/Conv2D?
,CAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp5cae_decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,CAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp?
CAE/Decoder/conv1_dec/BiasAddBiasAdd%CAE/Decoder/conv1_dec/Conv2D:output:04CAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
CAE/Decoder/conv1_dec/BiasAdd?
CAE/Decoder/conv1_dec/ReluRelu&CAE/Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
CAE/Decoder/conv1_dec/Relu?
CAE/Decoder/upsamp1/ShapeShape(CAE/Decoder/conv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
CAE/Decoder/upsamp1/Shape?
'CAE/Decoder/upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'CAE/Decoder/upsamp1/strided_slice/stack?
)CAE/Decoder/upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/upsamp1/strided_slice/stack_1?
)CAE/Decoder/upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)CAE/Decoder/upsamp1/strided_slice/stack_2?
!CAE/Decoder/upsamp1/strided_sliceStridedSlice"CAE/Decoder/upsamp1/Shape:output:00CAE/Decoder/upsamp1/strided_slice/stack:output:02CAE/Decoder/upsamp1/strided_slice/stack_1:output:02CAE/Decoder/upsamp1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!CAE/Decoder/upsamp1/strided_slice?
CAE/Decoder/upsamp1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
CAE/Decoder/upsamp1/Const?
CAE/Decoder/upsamp1/mulMul*CAE/Decoder/upsamp1/strided_slice:output:0"CAE/Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
CAE/Decoder/upsamp1/mul?
0CAE/Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor(CAE/Decoder/conv1_dec/Relu:activations:0CAE/Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????  *
half_pixel_centers(22
0CAE/Decoder/upsamp1/resize/ResizeNearestNeighbor?
(CAE/Decoder/output/Conv2D/ReadVariableOpReadVariableOp1cae_decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(CAE/Decoder/output/Conv2D/ReadVariableOp?
CAE/Decoder/output/Conv2DConv2DACAE/Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:00CAE/Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
CAE/Decoder/output/Conv2D?
)CAE/Decoder/output/BiasAdd/ReadVariableOpReadVariableOp2cae_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)CAE/Decoder/output/BiasAdd/ReadVariableOp?
CAE/Decoder/output/BiasAddBiasAdd"CAE/Decoder/output/Conv2D:output:01CAE/Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
CAE/Decoder/output/BiasAdd?
CAE/Decoder/output/SigmoidSigmoid#CAE/Decoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
CAE/Decoder/output/Sigmoid?
IdentityIdentityCAE/Decoder/output/Sigmoid:y:0-^CAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp,^CAE/Decoder/conv1_dec/Conv2D/ReadVariableOp-^CAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp,^CAE/Decoder/conv2_dec/Conv2D/ReadVariableOp-^CAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp,^CAE/Decoder/conv3_dec/Conv2D/ReadVariableOp-^CAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp,^CAE/Decoder/conv4_dec/Conv2D/ReadVariableOp,^CAE/Decoder/decoding/BiasAdd/ReadVariableOp+^CAE/Decoder/decoding/MatMul/ReadVariableOp*^CAE/Decoder/output/BiasAdd/ReadVariableOp)^CAE/Decoder/output/Conv2D/ReadVariableOp.^CAE/Encoder/bottleneck/BiasAdd/ReadVariableOp-^CAE/Encoder/bottleneck/MatMul/ReadVariableOp-^CAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp,^CAE/Encoder/conv1_enc/Conv2D/ReadVariableOp-^CAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp,^CAE/Encoder/conv2_enc/Conv2D/ReadVariableOp-^CAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp,^CAE/Encoder/conv3_enc/Conv2D/ReadVariableOp-^CAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp,^CAE/Encoder/conv4_enc/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapesu
s:?????????::::::::::::::::::::::2\
,CAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp,CAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp2Z
+CAE/Decoder/conv1_dec/Conv2D/ReadVariableOp+CAE/Decoder/conv1_dec/Conv2D/ReadVariableOp2\
,CAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp,CAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp2Z
+CAE/Decoder/conv2_dec/Conv2D/ReadVariableOp+CAE/Decoder/conv2_dec/Conv2D/ReadVariableOp2\
,CAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp,CAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp2Z
+CAE/Decoder/conv3_dec/Conv2D/ReadVariableOp+CAE/Decoder/conv3_dec/Conv2D/ReadVariableOp2\
,CAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp,CAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp2Z
+CAE/Decoder/conv4_dec/Conv2D/ReadVariableOp+CAE/Decoder/conv4_dec/Conv2D/ReadVariableOp2Z
+CAE/Decoder/decoding/BiasAdd/ReadVariableOp+CAE/Decoder/decoding/BiasAdd/ReadVariableOp2X
*CAE/Decoder/decoding/MatMul/ReadVariableOp*CAE/Decoder/decoding/MatMul/ReadVariableOp2V
)CAE/Decoder/output/BiasAdd/ReadVariableOp)CAE/Decoder/output/BiasAdd/ReadVariableOp2T
(CAE/Decoder/output/Conv2D/ReadVariableOp(CAE/Decoder/output/Conv2D/ReadVariableOp2^
-CAE/Encoder/bottleneck/BiasAdd/ReadVariableOp-CAE/Encoder/bottleneck/BiasAdd/ReadVariableOp2\
,CAE/Encoder/bottleneck/MatMul/ReadVariableOp,CAE/Encoder/bottleneck/MatMul/ReadVariableOp2\
,CAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp,CAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp2Z
+CAE/Encoder/conv1_enc/Conv2D/ReadVariableOp+CAE/Encoder/conv1_enc/Conv2D/ReadVariableOp2\
,CAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp,CAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp2Z
+CAE/Encoder/conv2_enc/Conv2D/ReadVariableOp+CAE/Encoder/conv2_enc/Conv2D/ReadVariableOp2\
,CAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp,CAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp2Z
+CAE/Encoder/conv3_enc/Conv2D/ReadVariableOp+CAE/Encoder/conv3_enc/Conv2D/ReadVariableOp2\
,CAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp,CAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp2Z
+CAE/Encoder/conv4_enc/Conv2D/ReadVariableOp+CAE/Encoder/conv4_enc/Conv2D/ReadVariableOp:^ Z
/
_output_shapes
:?????????
'
_user_specified_nameinput_encoder
?	
?
E__inference_bottleneck_layer_call_and_return_conditional_losses_85358

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
O
input_encoder>
serving_default_input_encoder:0?????????C
Decoder8
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "Functional", "name": "CAE", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "CAE", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["bottleneck", 0, 0]]}, "name": "Encoder", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [13, 13]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}, "name": "Decoder", "inbound_nodes": [[["Encoder", 1, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["Decoder", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "CAE", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["bottleneck", 0, 0]]}, "name": "Encoder", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [13, 13]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}, "name": "Decoder", "inbound_nodes": [[["Encoder", 1, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["Decoder", 1, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_encoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}}
?[

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?X
_tf_keras_network?X{"class_name": "Functional", "name": "Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["bottleneck", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["bottleneck", 0, 0]]}}}
?d
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
 layer_with_weights-3
 layer-7
!layer-8
"layer_with_weights-4
"layer-9
#layer-10
$layer_with_weights-5
$layer-11
%regularization_losses
&trainable_variables
'	variables
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?`
_tf_keras_network?`{"class_name": "Functional", "name": "Decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [13, 13]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 16]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [13, 13]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}}}
?
)iter

*beta_1

+beta_2
	,decay
-learning_rate.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m??m?@m?Am?Bm?Cm?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v??v?@v?Av?Bv?Cv?"
	optimizer
 "
trackable_list_wrapper
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21"
trackable_list_wrapper
?
.0
/1
02
13
24
35
46
57
68
79
810
911
:12
;13
<14
=15
>16
?17
@18
A19
B20
C21"
trackable_list_wrapper
?
Dlayer_metrics
regularization_losses
trainable_variables

Elayers
Flayer_regularization_losses
Gmetrics
Hnon_trainable_variables
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_encoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}}
?	

.kernel
/bias
I	variables
Jregularization_losses
Ktrainable_variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 1]}}
?
M	variables
Nregularization_losses
Otrainable_variables
P	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxpool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

0kernel
1bias
Q	variables
Rregularization_losses
Strainable_variables
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 16]}}
?
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxpool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

2kernel
3bias
Y	variables
Zregularization_losses
[trainable_variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv3_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 32]}}
?
]	variables
^regularization_losses
_trainable_variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxpool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

4kernel
5bias
a	variables
bregularization_losses
ctrainable_variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv4_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 64]}}
?
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "maxpool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

6kernel
7bias
m	variables
nregularization_losses
otrainable_variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "bottleneck", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
 "
trackable_list_wrapper
f
.0
/1
02
13
24
35
46
57
68
79"
trackable_list_wrapper
f
.0
/1
02
13
24
35
46
57
68
79"
trackable_list_wrapper
?
qlayer_metrics
regularization_losses
trainable_variables

rlayers
slayer_regularization_losses
tmetrics
unon_trainable_variables
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_decoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}}
?

8kernel
9bias
v	variables
wregularization_losses
xtrainable_variables
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "decoding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
z	variables
{regularization_losses
|trainable_variables
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 128]}}}
?	

:kernel
;bias
~	variables
regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv4_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 128]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "upsamp4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

<kernel
=bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv3_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "upsamp3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

>kernel
?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "upsamp2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

@kernel
Abias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
?
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "upsamp1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

Bkernel
Cbias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [13, 13]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
 "
trackable_list_wrapper
v
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11"
trackable_list_wrapper
v
80
91
:2
;3
<4
=5
>6
?7
@8
A9
B10
C11"
trackable_list_wrapper
?
?layer_metrics
%regularization_losses
&trainable_variables
?layers
 ?layer_regularization_losses
?metrics
?non_trainable_variables
'	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(2conv1_enc/kernel
:2conv1_enc/bias
*:( 2conv2_enc/kernel
: 2conv2_enc/bias
*:( @2conv3_enc/kernel
:@2conv3_enc/bias
+:)@?2conv4_enc/kernel
:?2conv4_enc/bias
$:"	?2bottleneck/kernel
:2bottleneck/bias
": 	?2decoding/kernel
:?2decoding/bias
,:*??2conv4_dec/kernel
:?2conv4_dec/bias
+:)?@2conv3_dec/kernel
:@2conv3_dec/bias
*:(@ 2conv2_dec/kernel
: 2conv2_dec/bias
*:( 2conv1_dec/kernel
:2conv1_dec/bias
':%2output/kernel
:2output/bias
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
?layer_metrics
I	variables
Jregularization_losses
Ktrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
M	variables
Nregularization_losses
Otrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
?layer_metrics
Q	variables
Rregularization_losses
Strainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
U	variables
Vregularization_losses
Wtrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
?layer_metrics
Y	variables
Zregularization_losses
[trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
]	variables
^regularization_losses
_trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
?layer_metrics
a	variables
bregularization_losses
ctrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
e	variables
fregularization_losses
gtrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
i	variables
jregularization_losses
ktrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
?layer_metrics
m	variables
nregularization_losses
otrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
n

0
1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
?layer_metrics
v	variables
wregularization_losses
xtrainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
z	variables
{regularization_losses
|trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
?layer_metrics
~	variables
regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
?layer_metrics
?	variables
?regularization_losses
?trainable_variables
?layers
?metrics
?non_trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
v
0
1
2
3
4
5
6
 7
!8
"9
#10
$11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
/:-2Adam/conv1_enc/kernel/m
!:2Adam/conv1_enc/bias/m
/:- 2Adam/conv2_enc/kernel/m
!: 2Adam/conv2_enc/bias/m
/:- @2Adam/conv3_enc/kernel/m
!:@2Adam/conv3_enc/bias/m
0:.@?2Adam/conv4_enc/kernel/m
": ?2Adam/conv4_enc/bias/m
):'	?2Adam/bottleneck/kernel/m
": 2Adam/bottleneck/bias/m
':%	?2Adam/decoding/kernel/m
!:?2Adam/decoding/bias/m
1:/??2Adam/conv4_dec/kernel/m
": ?2Adam/conv4_dec/bias/m
0:.?@2Adam/conv3_dec/kernel/m
!:@2Adam/conv3_dec/bias/m
/:-@ 2Adam/conv2_dec/kernel/m
!: 2Adam/conv2_dec/bias/m
/:- 2Adam/conv1_dec/kernel/m
!:2Adam/conv1_dec/bias/m
,:*2Adam/output/kernel/m
:2Adam/output/bias/m
/:-2Adam/conv1_enc/kernel/v
!:2Adam/conv1_enc/bias/v
/:- 2Adam/conv2_enc/kernel/v
!: 2Adam/conv2_enc/bias/v
/:- @2Adam/conv3_enc/kernel/v
!:@2Adam/conv3_enc/bias/v
0:.@?2Adam/conv4_enc/kernel/v
": ?2Adam/conv4_enc/bias/v
):'	?2Adam/bottleneck/kernel/v
": 2Adam/bottleneck/bias/v
':%	?2Adam/decoding/kernel/v
!:?2Adam/decoding/bias/v
1:/??2Adam/conv4_dec/kernel/v
": ?2Adam/conv4_dec/bias/v
0:.?@2Adam/conv3_dec/kernel/v
!:@2Adam/conv3_dec/bias/v
/:-@ 2Adam/conv2_dec/kernel/v
!: 2Adam/conv2_dec/bias/v
/:- 2Adam/conv1_dec/kernel/v
!:2Adam/conv1_dec/bias/v
,:*2Adam/output/kernel/v
:2Adam/output/bias/v
?2?
 __inference__wrapped_model_83266?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *4?1
/?,
input_encoder?????????
?2?
#__inference_CAE_layer_call_fn_84887
#__inference_CAE_layer_call_fn_84377
#__inference_CAE_layer_call_fn_84838
#__inference_CAE_layer_call_fn_84476?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_CAE_layer_call_and_return_conditional_losses_84277
>__inference_CAE_layer_call_and_return_conditional_losses_84789
>__inference_CAE_layer_call_and_return_conditional_losses_84662
>__inference_CAE_layer_call_and_return_conditional_losses_84227?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_Encoder_layer_call_fn_85025
'__inference_Encoder_layer_call_fn_83565
'__inference_Encoder_layer_call_fn_85000
'__inference_Encoder_layer_call_fn_83624?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_Encoder_layer_call_and_return_conditional_losses_84975
B__inference_Encoder_layer_call_and_return_conditional_losses_84931
B__inference_Encoder_layer_call_and_return_conditional_losses_83505
B__inference_Encoder_layer_call_and_return_conditional_losses_83471?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
'__inference_Decoder_layer_call_fn_85257
'__inference_Decoder_layer_call_fn_84000
'__inference_Decoder_layer_call_fn_85228
'__inference_Decoder_layer_call_fn_84068?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
B__inference_Decoder_layer_call_and_return_conditional_losses_85199
B__inference_Decoder_layer_call_and_return_conditional_losses_85112
B__inference_Decoder_layer_call_and_return_conditional_losses_83892
B__inference_Decoder_layer_call_and_return_conditional_losses_83931?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_84535input_encoder"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv1_enc_layer_call_fn_85277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv1_enc_layer_call_and_return_conditional_losses_85268?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_maxpool1_layer_call_fn_83278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_maxpool1_layer_call_and_return_conditional_losses_83272?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv2_enc_layer_call_fn_85297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2_enc_layer_call_and_return_conditional_losses_85288?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_maxpool2_layer_call_fn_83290?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_maxpool2_layer_call_and_return_conditional_losses_83284?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv3_enc_layer_call_fn_85317?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv3_enc_layer_call_and_return_conditional_losses_85308?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_maxpool3_layer_call_fn_83302?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_maxpool3_layer_call_and_return_conditional_losses_83296?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv4_enc_layer_call_fn_85337?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv4_enc_layer_call_and_return_conditional_losses_85328?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_maxpool4_layer_call_fn_83314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
C__inference_maxpool4_layer_call_and_return_conditional_losses_83308?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
'__inference_flatten_layer_call_fn_85348?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_flatten_layer_call_and_return_conditional_losses_85343?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_bottleneck_layer_call_fn_85367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_bottleneck_layer_call_and_return_conditional_losses_85358?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_decoding_layer_call_fn_85386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_decoding_layer_call_and_return_conditional_losses_85377?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_reshape_layer_call_fn_85405?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_reshape_layer_call_and_return_conditional_losses_85400?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv4_dec_layer_call_fn_85425?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv4_dec_layer_call_and_return_conditional_losses_85416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_upsamp4_layer_call_fn_83643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_upsamp4_layer_call_and_return_conditional_losses_83637?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv3_dec_layer_call_fn_85445?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv3_dec_layer_call_and_return_conditional_losses_85436?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_upsamp3_layer_call_fn_83662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_upsamp3_layer_call_and_return_conditional_losses_83656?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv2_dec_layer_call_fn_85465?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2_dec_layer_call_and_return_conditional_losses_85456?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_upsamp2_layer_call_fn_83681?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_upsamp2_layer_call_and_return_conditional_losses_83675?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv1_dec_layer_call_fn_85485?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv1_dec_layer_call_and_return_conditional_losses_85476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_upsamp1_layer_call_fn_83700?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
B__inference_upsamp1_layer_call_and_return_conditional_losses_83694?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_output_layer_call_fn_85505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_output_layer_call_and_return_conditional_losses_85496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
>__inference_CAE_layer_call_and_return_conditional_losses_84227?./0123456789:;<=>?@ABCF?C
<?9
/?,
input_encoder?????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
>__inference_CAE_layer_call_and_return_conditional_losses_84277?./0123456789:;<=>?@ABCF?C
<?9
/?,
input_encoder?????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
>__inference_CAE_layer_call_and_return_conditional_losses_84662?./0123456789:;<=>?@ABC??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
>__inference_CAE_layer_call_and_return_conditional_losses_84789?./0123456789:;<=>?@ABC??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
#__inference_CAE_layer_call_fn_84377?./0123456789:;<=>?@ABCF?C
<?9
/?,
input_encoder?????????
p

 
? "2?/+????????????????????????????
#__inference_CAE_layer_call_fn_84476?./0123456789:;<=>?@ABCF?C
<?9
/?,
input_encoder?????????
p 

 
? "2?/+????????????????????????????
#__inference_CAE_layer_call_fn_84838?./0123456789:;<=>?@ABC??<
5?2
(?%
inputs?????????
p

 
? "2?/+????????????????????????????
#__inference_CAE_layer_call_fn_84887?./0123456789:;<=>?@ABC??<
5?2
(?%
inputs?????????
p 

 
? "2?/+????????????????????????????
B__inference_Decoder_layer_call_and_return_conditional_losses_83892?89:;<=>?@ABC>?;
4?1
'?$
input_decoder?????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
B__inference_Decoder_layer_call_and_return_conditional_losses_83931?89:;<=>?@ABC>?;
4?1
'?$
input_decoder?????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
B__inference_Decoder_layer_call_and_return_conditional_losses_85112v89:;<=>?@ABC7?4
-?*
 ?
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
B__inference_Decoder_layer_call_and_return_conditional_losses_85199v89:;<=>?@ABC7?4
-?*
 ?
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
'__inference_Decoder_layer_call_fn_84000?89:;<=>?@ABC>?;
4?1
'?$
input_decoder?????????
p

 
? "2?/+????????????????????????????
'__inference_Decoder_layer_call_fn_84068?89:;<=>?@ABC>?;
4?1
'?$
input_decoder?????????
p 

 
? "2?/+????????????????????????????
'__inference_Decoder_layer_call_fn_85228{89:;<=>?@ABC7?4
-?*
 ?
inputs?????????
p

 
? "2?/+????????????????????????????
'__inference_Decoder_layer_call_fn_85257{89:;<=>?@ABC7?4
-?*
 ?
inputs?????????
p 

 
? "2?/+????????????????????????????
B__inference_Encoder_layer_call_and_return_conditional_losses_83471{
./01234567F?C
<?9
/?,
input_encoder?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_Encoder_layer_call_and_return_conditional_losses_83505{
./01234567F?C
<?9
/?,
input_encoder?????????
p 

 
? "%?"
?
0?????????
? ?
B__inference_Encoder_layer_call_and_return_conditional_losses_84931t
./01234567??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????
? ?
B__inference_Encoder_layer_call_and_return_conditional_losses_84975t
./01234567??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
'__inference_Encoder_layer_call_fn_83565n
./01234567F?C
<?9
/?,
input_encoder?????????
p

 
? "???????????
'__inference_Encoder_layer_call_fn_83624n
./01234567F?C
<?9
/?,
input_encoder?????????
p 

 
? "???????????
'__inference_Encoder_layer_call_fn_85000g
./01234567??<
5?2
(?%
inputs?????????
p

 
? "???????????
'__inference_Encoder_layer_call_fn_85025g
./01234567??<
5?2
(?%
inputs?????????
p 

 
? "???????????
 __inference__wrapped_model_83266?./0123456789:;<=>?@ABC>?;
4?1
/?,
input_encoder?????????
? "9?6
4
Decoder)?&
Decoder??????????
E__inference_bottleneck_layer_call_and_return_conditional_losses_85358]670?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ~
*__inference_bottleneck_layer_call_fn_85367P670?-
&?#
!?
inputs??????????
? "???????????
D__inference_conv1_dec_layer_call_and_return_conditional_losses_85476?@AI?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
)__inference_conv1_dec_layer_call_fn_85485?@AI?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
D__inference_conv1_enc_layer_call_and_return_conditional_losses_85268l./7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv1_enc_layer_call_fn_85277_./7?4
-?*
(?%
inputs?????????
? " ???????????
D__inference_conv2_dec_layer_call_and_return_conditional_losses_85456?>?I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
)__inference_conv2_dec_layer_call_fn_85465?>?I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
D__inference_conv2_enc_layer_call_and_return_conditional_losses_85288l017?4
-?*
(?%
inputs?????????


? "-?*
#? 
0?????????

 
? ?
)__inference_conv2_enc_layer_call_fn_85297_017?4
-?*
(?%
inputs?????????


? " ??????????

 ?
D__inference_conv3_dec_layer_call_and_return_conditional_losses_85436?<=J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
)__inference_conv3_dec_layer_call_fn_85445?<=J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
D__inference_conv3_enc_layer_call_and_return_conditional_losses_85308l237?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv3_enc_layer_call_fn_85317_237?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_conv4_dec_layer_call_and_return_conditional_losses_85416n:;8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
)__inference_conv4_dec_layer_call_fn_85425a:;8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_conv4_enc_layer_call_and_return_conditional_losses_85328m457?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
)__inference_conv4_enc_layer_call_fn_85337`457?4
-?*
(?%
inputs?????????@
? "!????????????
C__inference_decoding_layer_call_and_return_conditional_losses_85377]89/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? |
(__inference_decoding_layer_call_fn_85386P89/?,
%?"
 ?
inputs?????????
? "????????????
B__inference_flatten_layer_call_and_return_conditional_losses_85343b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
'__inference_flatten_layer_call_fn_85348U8?5
.?+
)?&
inputs??????????
? "????????????
C__inference_maxpool1_layer_call_and_return_conditional_losses_83272?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_maxpool1_layer_call_fn_83278?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_maxpool2_layer_call_and_return_conditional_losses_83284?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_maxpool2_layer_call_fn_83290?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_maxpool3_layer_call_and_return_conditional_losses_83296?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_maxpool3_layer_call_fn_83302?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_maxpool4_layer_call_and_return_conditional_losses_83308?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
(__inference_maxpool4_layer_call_fn_83314?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_output_layer_call_and_return_conditional_losses_85496?BCI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
&__inference_output_layer_call_fn_85505?BCI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
B__inference_reshape_layer_call_and_return_conditional_losses_85400b0?-
&?#
!?
inputs??????????
? ".?+
$?!
0??????????
? ?
'__inference_reshape_layer_call_fn_85405U0?-
&?#
!?
inputs??????????
? "!????????????
#__inference_signature_wrapper_84535?./0123456789:;<=>?@ABCO?L
? 
E?B
@
input_encoder/?,
input_encoder?????????"9?6
4
Decoder)?&
Decoder??????????
B__inference_upsamp1_layer_call_and_return_conditional_losses_83694?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
'__inference_upsamp1_layer_call_fn_83700?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_upsamp2_layer_call_and_return_conditional_losses_83675?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
'__inference_upsamp2_layer_call_fn_83681?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_upsamp3_layer_call_and_return_conditional_losses_83656?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
'__inference_upsamp3_layer_call_fn_83662?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_upsamp4_layer_call_and_return_conditional_losses_83637?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
'__inference_upsamp4_layer_call_fn_83643?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????