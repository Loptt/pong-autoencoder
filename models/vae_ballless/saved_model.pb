!
¼
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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

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
,
Exp
x"T
y"T"
Ttype:

2
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

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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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

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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ÚÅ
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
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

conv1_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1_enc/kernel
}
$conv1_enc/kernel/Read/ReadVariableOpReadVariableOpconv1_enc/kernel*&
_output_shapes
:*
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

conv2_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2_enc/kernel
}
$conv2_enc/kernel/Read/ReadVariableOpReadVariableOpconv2_enc/kernel*&
_output_shapes
: *
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

conv3_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv3_enc/kernel
}
$conv3_enc/kernel/Read/ReadVariableOpReadVariableOpconv3_enc/kernel*&
_output_shapes
: @*
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

conv4_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv4_enc/kernel
~
$conv4_enc/kernel/Read/ReadVariableOpReadVariableOpconv4_enc/kernel*'
_output_shapes
:@*
dtype0
u
conv4_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv4_enc/bias
n
"conv4_enc/bias/Read/ReadVariableOpReadVariableOpconv4_enc/bias*
_output_shapes	
:*
dtype0

bottleneck/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namebottleneck/kernel
x
%bottleneck/kernel/Read/ReadVariableOpReadVariableOpbottleneck/kernel*
_output_shapes
:	*
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
v
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namez_mean/kernel
o
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes

:*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
|
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namez_log_var/kernel
u
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes

:*
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:*
dtype0
{
decoding/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedecoding/kernel
t
#decoding/kernel/Read/ReadVariableOpReadVariableOpdecoding/kernel*
_output_shapes
:	*
dtype0
s
decoding/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedecoding/bias
l
!decoding/bias/Read/ReadVariableOpReadVariableOpdecoding/bias*
_output_shapes	
:*
dtype0

conv4_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv4_dec/kernel

$conv4_dec/kernel/Read/ReadVariableOpReadVariableOpconv4_dec/kernel*(
_output_shapes
:*
dtype0
u
conv4_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv4_dec/bias
n
"conv4_dec/bias/Read/ReadVariableOpReadVariableOpconv4_dec/bias*
_output_shapes	
:*
dtype0

conv3_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3_dec/kernel
~
$conv3_dec/kernel/Read/ReadVariableOpReadVariableOpconv3_dec/kernel*'
_output_shapes
:@*
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

conv2_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv2_dec/kernel
}
$conv2_dec/kernel/Read/ReadVariableOpReadVariableOpconv2_dec/kernel*&
_output_shapes
:@ *
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

conv1_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv1_dec/kernel
}
$conv1_dec/kernel/Read/ReadVariableOpReadVariableOpconv1_dec/kernel*&
_output_shapes
: *
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

Adam/conv1_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1_enc/kernel/m

+Adam/conv1_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/kernel/m*&
_output_shapes
:*
dtype0

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

Adam/conv2_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2_enc/kernel/m

+Adam/conv2_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/kernel/m*&
_output_shapes
: *
dtype0

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

Adam/conv3_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv3_enc/kernel/m

+Adam/conv3_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/kernel/m*&
_output_shapes
: @*
dtype0

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

Adam/conv4_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv4_enc/kernel/m

+Adam/conv4_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv4_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv4_enc/bias/m
|
)Adam/conv4_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/bias/m*
_output_shapes	
:*
dtype0

Adam/bottleneck/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/bottleneck/kernel/m

,Adam/bottleneck/kernel/m/Read/ReadVariableOpReadVariableOpAdam/bottleneck/kernel/m*
_output_shapes
:	*
dtype0

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

Adam/z_mean/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/z_mean/kernel/m
}
(Adam/z_mean/kernel/m/Read/ReadVariableOpReadVariableOpAdam/z_mean/kernel/m*
_output_shapes

:*
dtype0
|
Adam/z_mean/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/z_mean/bias/m
u
&Adam/z_mean/bias/m/Read/ReadVariableOpReadVariableOpAdam/z_mean/bias/m*
_output_shapes
:*
dtype0

Adam/z_log_var/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/z_log_var/kernel/m

+Adam/z_log_var/kernel/m/Read/ReadVariableOpReadVariableOpAdam/z_log_var/kernel/m*
_output_shapes

:*
dtype0

Adam/z_log_var/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/z_log_var/bias/m
{
)Adam/z_log_var/bias/m/Read/ReadVariableOpReadVariableOpAdam/z_log_var/bias/m*
_output_shapes
:*
dtype0

Adam/decoding/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/decoding/kernel/m

*Adam/decoding/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoding/kernel/m*
_output_shapes
:	*
dtype0

Adam/decoding/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/decoding/bias/m
z
(Adam/decoding/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoding/bias/m*
_output_shapes	
:*
dtype0

Adam/conv4_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv4_dec/kernel/m

+Adam/conv4_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv4_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv4_dec/bias/m
|
)Adam/conv4_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/bias/m*
_output_shapes	
:*
dtype0

Adam/conv3_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv3_dec/kernel/m

+Adam/conv3_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/kernel/m*'
_output_shapes
:@*
dtype0

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

Adam/conv2_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2_dec/kernel/m

+Adam/conv2_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/kernel/m*&
_output_shapes
:@ *
dtype0

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

Adam/conv1_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1_dec/kernel/m

+Adam/conv1_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/kernel/m*&
_output_shapes
: *
dtype0

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

Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output/kernel/m

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

Adam/conv1_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1_enc/kernel/v

+Adam/conv1_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/kernel/v*&
_output_shapes
:*
dtype0

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

Adam/conv2_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2_enc/kernel/v

+Adam/conv2_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/kernel/v*&
_output_shapes
: *
dtype0

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

Adam/conv3_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv3_enc/kernel/v

+Adam/conv3_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/kernel/v*&
_output_shapes
: @*
dtype0

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

Adam/conv4_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv4_enc/kernel/v

+Adam/conv4_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv4_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv4_enc/bias/v
|
)Adam/conv4_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/bias/v*
_output_shapes	
:*
dtype0

Adam/bottleneck/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/bottleneck/kernel/v

,Adam/bottleneck/kernel/v/Read/ReadVariableOpReadVariableOpAdam/bottleneck/kernel/v*
_output_shapes
:	*
dtype0

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

Adam/z_mean/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/z_mean/kernel/v
}
(Adam/z_mean/kernel/v/Read/ReadVariableOpReadVariableOpAdam/z_mean/kernel/v*
_output_shapes

:*
dtype0
|
Adam/z_mean/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/z_mean/bias/v
u
&Adam/z_mean/bias/v/Read/ReadVariableOpReadVariableOpAdam/z_mean/bias/v*
_output_shapes
:*
dtype0

Adam/z_log_var/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/z_log_var/kernel/v

+Adam/z_log_var/kernel/v/Read/ReadVariableOpReadVariableOpAdam/z_log_var/kernel/v*
_output_shapes

:*
dtype0

Adam/z_log_var/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/z_log_var/bias/v
{
)Adam/z_log_var/bias/v/Read/ReadVariableOpReadVariableOpAdam/z_log_var/bias/v*
_output_shapes
:*
dtype0

Adam/decoding/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/decoding/kernel/v

*Adam/decoding/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoding/kernel/v*
_output_shapes
:	*
dtype0

Adam/decoding/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/decoding/bias/v
z
(Adam/decoding/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoding/bias/v*
_output_shapes	
:*
dtype0

Adam/conv4_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv4_dec/kernel/v

+Adam/conv4_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv4_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv4_dec/bias/v
|
)Adam/conv4_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/bias/v*
_output_shapes	
:*
dtype0

Adam/conv3_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv3_dec/kernel/v

+Adam/conv3_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/kernel/v*'
_output_shapes
:@*
dtype0

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

Adam/conv2_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2_dec/kernel/v

+Adam/conv2_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/kernel/v*&
_output_shapes
:@ *
dtype0

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

Adam/conv1_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1_dec/kernel/v

+Adam/conv1_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/kernel/v*&
_output_shapes
: *
dtype0

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

Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output/kernel/v

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
ª¢
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ä¡
valueÙ¡BÕ¡ BÍ¡
ã
encoder
decoder
total_loss_tracker
reconstruction_loss_tracker
kl_loss_tracker
	optimizer
loss
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
Â
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
trainable_variables
regularization_losses
	variables
	keras_api

layer-0
 layer_with_weights-0
 layer-1
!layer-2
"layer_with_weights-1
"layer-3
#layer-4
$layer_with_weights-2
$layer-5
%layer-6
&layer_with_weights-3
&layer-7
'layer-8
(layer_with_weights-4
(layer-9
)layer-10
*layer_with_weights-5
*layer-11
+trainable_variables
,regularization_losses
-	variables
.	keras_api
4
	/total
	0count
1	variables
2	keras_api
4
	3total
	4count
5	variables
6	keras_api
4
	7total
	8count
9	variables
:	keras_api
È
;iter

<beta_1

=beta_2
	>decay
?learning_rate@mÁAmÂBmÃCmÄDmÅEmÆFmÇGmÈHmÉImÊJmËKmÌLmÍMmÎNmÏOmÐPmÑQmÒRmÓSmÔTmÕUmÖVm×WmØXmÙYmÚ@vÛAvÜBvÝCvÞDvßEvàFváGvâHvãIväJvåKvæLvçMvèNvéOvêPvëQvìRvíSvîTvïUvðVvñWvòXvóYvô
 
Æ
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15
P16
Q17
R18
S19
T20
U21
V22
W23
X24
Y25
 
ö
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15
P16
Q17
R18
S19
T20
U21
V22
W23
X24
Y25
/26
027
328
429
730
831
­
Zlayer_metrics
[metrics
trainable_variables
\layer_regularization_losses

]layers
	regularization_losses
^non_trainable_variables

	variables
 
 
h

@kernel
Abias
_trainable_variables
`regularization_losses
a	variables
b	keras_api
R
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
h

Bkernel
Cbias
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
R
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
h

Dkernel
Ebias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
R
strainable_variables
tregularization_losses
u	variables
v	keras_api
h

Fkernel
Gbias
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
R
{trainable_variables
|regularization_losses
}	variables
~	keras_api
U
trainable_variables
regularization_losses
	variables
	keras_api
l

Hkernel
Ibias
trainable_variables
regularization_losses
	variables
	keras_api
l

Jkernel
Kbias
trainable_variables
regularization_losses
	variables
	keras_api
l

Lkernel
Mbias
trainable_variables
regularization_losses
	variables
	keras_api
V
trainable_variables
regularization_losses
	variables
	keras_api
f
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
 
f
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
²
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables
 
l

Nkernel
Obias
trainable_variables
regularization_losses
	variables
	keras_api
V
trainable_variables
regularization_losses
	variables
	keras_api
l

Pkernel
Qbias
 trainable_variables
¡regularization_losses
¢	variables
£	keras_api
V
¤trainable_variables
¥regularization_losses
¦	variables
§	keras_api
l

Rkernel
Sbias
¨trainable_variables
©regularization_losses
ª	variables
«	keras_api
V
¬trainable_variables
­regularization_losses
®	variables
¯	keras_api
l

Tkernel
Ubias
°trainable_variables
±regularization_losses
²	variables
³	keras_api
V
´trainable_variables
µregularization_losses
¶	variables
·	keras_api
l

Vkernel
Wbias
¸trainable_variables
¹regularization_losses
º	variables
»	keras_api
V
¼trainable_variables
½regularization_losses
¾	variables
¿	keras_api
l

Xkernel
Ybias
Àtrainable_variables
Áregularization_losses
Â	variables
Ã	keras_api
V
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
 
V
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11
²
Älayer_metrics
Åmetrics
+trainable_variables
 Ælayer_regularization_losses
Çlayers
,regularization_losses
Ènon_trainable_variables
-	variables
NL
VARIABLE_VALUEtotal3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEcount3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUE

/0
01

1	variables
YW
VARIABLE_VALUEtotal_1<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEcount_1<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUE

30
41

5	variables
MK
VARIABLE_VALUEtotal_20kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEcount_20kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUE

70
81

9	variables
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
TR
VARIABLE_VALUEz_mean/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEz_mean/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEz_log_var/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEz_log_var/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdecoding/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdecoding/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv4_dec/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv4_dec/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3_dec/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3_dec/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2_dec/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2_dec/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1_dec/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1_dec/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEoutput/kernel1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEoutput/bias1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
6

total_loss
reconstruction_loss
kl_loss

0
1
2
 

0
1
*
/0
01
32
43
74
85

@0
A1
 

@0
A1
²
Élayer_metrics
Êmetrics
_trainable_variables
 Ëlayer_regularization_losses
Ìlayers
`regularization_losses
Ínon_trainable_variables
a	variables
 
 
 
²
Îlayer_metrics
Ïmetrics
ctrainable_variables
 Ðlayer_regularization_losses
Ñlayers
dregularization_losses
Ònon_trainable_variables
e	variables

B0
C1
 

B0
C1
²
Ólayer_metrics
Ômetrics
gtrainable_variables
 Õlayer_regularization_losses
Ölayers
hregularization_losses
×non_trainable_variables
i	variables
 
 
 
²
Ølayer_metrics
Ùmetrics
ktrainable_variables
 Úlayer_regularization_losses
Ûlayers
lregularization_losses
Ünon_trainable_variables
m	variables

D0
E1
 

D0
E1
²
Ýlayer_metrics
Þmetrics
otrainable_variables
 ßlayer_regularization_losses
àlayers
pregularization_losses
ánon_trainable_variables
q	variables
 
 
 
²
âlayer_metrics
ãmetrics
strainable_variables
 älayer_regularization_losses
ålayers
tregularization_losses
ænon_trainable_variables
u	variables

F0
G1
 

F0
G1
²
çlayer_metrics
èmetrics
wtrainable_variables
 élayer_regularization_losses
êlayers
xregularization_losses
ënon_trainable_variables
y	variables
 
 
 
²
ìlayer_metrics
ímetrics
{trainable_variables
 îlayer_regularization_losses
ïlayers
|regularization_losses
ðnon_trainable_variables
}	variables
 
 
 
´
ñlayer_metrics
òmetrics
trainable_variables
 ólayer_regularization_losses
ôlayers
regularization_losses
õnon_trainable_variables
	variables

H0
I1
 

H0
I1
µ
ölayer_metrics
÷metrics
trainable_variables
 ølayer_regularization_losses
ùlayers
regularization_losses
únon_trainable_variables
	variables

J0
K1
 

J0
K1
µ
ûlayer_metrics
ümetrics
trainable_variables
 ýlayer_regularization_losses
þlayers
regularization_losses
ÿnon_trainable_variables
	variables

L0
M1
 

L0
M1
µ
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables
 
 
 
µ
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables
 
 
 
f
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 

N0
O1
 

N0
O1
µ
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables
 
 
 
µ
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables

P0
Q1
 

P0
Q1
µ
layer_metrics
metrics
 trainable_variables
 layer_regularization_losses
layers
¡regularization_losses
non_trainable_variables
¢	variables
 
 
 
µ
layer_metrics
metrics
¤trainable_variables
 layer_regularization_losses
layers
¥regularization_losses
non_trainable_variables
¦	variables

R0
S1
 

R0
S1
µ
layer_metrics
metrics
¨trainable_variables
  layer_regularization_losses
¡layers
©regularization_losses
¢non_trainable_variables
ª	variables
 
 
 
µ
£layer_metrics
¤metrics
¬trainable_variables
 ¥layer_regularization_losses
¦layers
­regularization_losses
§non_trainable_variables
®	variables

T0
U1
 

T0
U1
µ
¨layer_metrics
©metrics
°trainable_variables
 ªlayer_regularization_losses
«layers
±regularization_losses
¬non_trainable_variables
²	variables
 
 
 
µ
­layer_metrics
®metrics
´trainable_variables
 ¯layer_regularization_losses
°layers
µregularization_losses
±non_trainable_variables
¶	variables

V0
W1
 

V0
W1
µ
²layer_metrics
³metrics
¸trainable_variables
 ´layer_regularization_losses
µlayers
¹regularization_losses
¶non_trainable_variables
º	variables
 
 
 
µ
·layer_metrics
¸metrics
¼trainable_variables
 ¹layer_regularization_losses
ºlayers
½regularization_losses
»non_trainable_variables
¾	variables

X0
Y1
 

X0
Y1
µ
¼layer_metrics
½metrics
Àtrainable_variables
 ¾layer_regularization_losses
¿layers
Áregularization_losses
Ànon_trainable_variables
Â	variables
 
 
 
V
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11
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
wu
VARIABLE_VALUEAdam/z_mean/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/z_mean/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/z_log_var/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/z_log_var/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/decoding/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/decoding/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv4_dec/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv4_dec/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3_dec/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3_dec/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2_dec/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2_dec/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1_dec/kernel/mMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1_dec/bias/mMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/output/kernel/mMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/output/bias/mMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
wu
VARIABLE_VALUEAdam/z_mean/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/z_mean/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/z_log_var/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/z_log_var/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/decoding/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/decoding/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv4_dec/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv4_dec/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3_dec/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3_dec/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2_dec/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2_dec/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1_dec/kernel/vMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1_dec/bias/vMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/output/kernel/vMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/output/bias/vMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1_enc/kernelconv1_enc/biasconv2_enc/kernelconv2_enc/biasconv3_enc/kernelconv3_enc/biasconv4_enc/kernelconv4_enc/biasbottleneck/kernelbottleneck/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/biasdecoding/kerneldecoding/biasconv4_dec/kernelconv4_dec/biasconv3_dec/kernelconv3_dec/biasconv2_dec/kernelconv2_dec/biasconv1_dec/kernelconv1_dec/biasoutput/kerneloutput/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_396918
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ø
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv1_enc/kernel/Read/ReadVariableOp"conv1_enc/bias/Read/ReadVariableOp$conv2_enc/kernel/Read/ReadVariableOp"conv2_enc/bias/Read/ReadVariableOp$conv3_enc/kernel/Read/ReadVariableOp"conv3_enc/bias/Read/ReadVariableOp$conv4_enc/kernel/Read/ReadVariableOp"conv4_enc/bias/Read/ReadVariableOp%bottleneck/kernel/Read/ReadVariableOp#bottleneck/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOp#decoding/kernel/Read/ReadVariableOp!decoding/bias/Read/ReadVariableOp$conv4_dec/kernel/Read/ReadVariableOp"conv4_dec/bias/Read/ReadVariableOp$conv3_dec/kernel/Read/ReadVariableOp"conv3_dec/bias/Read/ReadVariableOp$conv2_dec/kernel/Read/ReadVariableOp"conv2_dec/bias/Read/ReadVariableOp$conv1_dec/kernel/Read/ReadVariableOp"conv1_dec/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOp+Adam/conv1_enc/kernel/m/Read/ReadVariableOp)Adam/conv1_enc/bias/m/Read/ReadVariableOp+Adam/conv2_enc/kernel/m/Read/ReadVariableOp)Adam/conv2_enc/bias/m/Read/ReadVariableOp+Adam/conv3_enc/kernel/m/Read/ReadVariableOp)Adam/conv3_enc/bias/m/Read/ReadVariableOp+Adam/conv4_enc/kernel/m/Read/ReadVariableOp)Adam/conv4_enc/bias/m/Read/ReadVariableOp,Adam/bottleneck/kernel/m/Read/ReadVariableOp*Adam/bottleneck/bias/m/Read/ReadVariableOp(Adam/z_mean/kernel/m/Read/ReadVariableOp&Adam/z_mean/bias/m/Read/ReadVariableOp+Adam/z_log_var/kernel/m/Read/ReadVariableOp)Adam/z_log_var/bias/m/Read/ReadVariableOp*Adam/decoding/kernel/m/Read/ReadVariableOp(Adam/decoding/bias/m/Read/ReadVariableOp+Adam/conv4_dec/kernel/m/Read/ReadVariableOp)Adam/conv4_dec/bias/m/Read/ReadVariableOp+Adam/conv3_dec/kernel/m/Read/ReadVariableOp)Adam/conv3_dec/bias/m/Read/ReadVariableOp+Adam/conv2_dec/kernel/m/Read/ReadVariableOp)Adam/conv2_dec/bias/m/Read/ReadVariableOp+Adam/conv1_dec/kernel/m/Read/ReadVariableOp)Adam/conv1_dec/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv1_enc/kernel/v/Read/ReadVariableOp)Adam/conv1_enc/bias/v/Read/ReadVariableOp+Adam/conv2_enc/kernel/v/Read/ReadVariableOp)Adam/conv2_enc/bias/v/Read/ReadVariableOp+Adam/conv3_enc/kernel/v/Read/ReadVariableOp)Adam/conv3_enc/bias/v/Read/ReadVariableOp+Adam/conv4_enc/kernel/v/Read/ReadVariableOp)Adam/conv4_enc/bias/v/Read/ReadVariableOp,Adam/bottleneck/kernel/v/Read/ReadVariableOp*Adam/bottleneck/bias/v/Read/ReadVariableOp(Adam/z_mean/kernel/v/Read/ReadVariableOp&Adam/z_mean/bias/v/Read/ReadVariableOp+Adam/z_log_var/kernel/v/Read/ReadVariableOp)Adam/z_log_var/bias/v/Read/ReadVariableOp*Adam/decoding/kernel/v/Read/ReadVariableOp(Adam/decoding/bias/v/Read/ReadVariableOp+Adam/conv4_dec/kernel/v/Read/ReadVariableOp)Adam/conv4_dec/bias/v/Read/ReadVariableOp+Adam/conv3_dec/kernel/v/Read/ReadVariableOp)Adam/conv3_dec/bias/v/Read/ReadVariableOp+Adam/conv2_dec/kernel/v/Read/ReadVariableOp)Adam/conv2_dec/bias/v/Read/ReadVariableOp+Adam/conv1_dec/kernel/v/Read/ReadVariableOp)Adam/conv1_dec/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*f
Tin_
]2[	*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_398424
ß
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotalcounttotal_1count_1total_2count_2	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv1_enc/kernelconv1_enc/biasconv2_enc/kernelconv2_enc/biasconv3_enc/kernelconv3_enc/biasconv4_enc/kernelconv4_enc/biasbottleneck/kernelbottleneck/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/biasdecoding/kerneldecoding/biasconv4_dec/kernelconv4_dec/biasconv3_dec/kernelconv3_dec/biasconv2_dec/kernelconv2_dec/biasconv1_dec/kernelconv1_dec/biasoutput/kerneloutput/biasAdam/conv1_enc/kernel/mAdam/conv1_enc/bias/mAdam/conv2_enc/kernel/mAdam/conv2_enc/bias/mAdam/conv3_enc/kernel/mAdam/conv3_enc/bias/mAdam/conv4_enc/kernel/mAdam/conv4_enc/bias/mAdam/bottleneck/kernel/mAdam/bottleneck/bias/mAdam/z_mean/kernel/mAdam/z_mean/bias/mAdam/z_log_var/kernel/mAdam/z_log_var/bias/mAdam/decoding/kernel/mAdam/decoding/bias/mAdam/conv4_dec/kernel/mAdam/conv4_dec/bias/mAdam/conv3_dec/kernel/mAdam/conv3_dec/bias/mAdam/conv2_dec/kernel/mAdam/conv2_dec/bias/mAdam/conv1_dec/kernel/mAdam/conv1_dec/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv1_enc/kernel/vAdam/conv1_enc/bias/vAdam/conv2_enc/kernel/vAdam/conv2_enc/bias/vAdam/conv3_enc/kernel/vAdam/conv3_enc/bias/vAdam/conv4_enc/kernel/vAdam/conv4_enc/bias/vAdam/bottleneck/kernel/vAdam/bottleneck/bias/vAdam/z_mean/kernel/vAdam/z_mean/bias/vAdam/z_log_var/kernel/vAdam/z_log_var/bias/vAdam/decoding/kernel/vAdam/decoding/bias/vAdam/conv4_dec/kernel/vAdam/conv4_dec/bias/vAdam/conv3_dec/kernel/vAdam/conv3_dec/bias/vAdam/conv2_dec/kernel/vAdam/conv2_dec/bias/vAdam/conv1_dec/kernel/vAdam/conv1_dec/bias/vAdam/output/kernel/vAdam/output/bias/v*e
Tin^
\2Z*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_398701¯»

_
C__inference_upsamp1_layer_call_and_return_conditional_losses_396049

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Þ
E__inference_conv2_dec_layer_call_and_return_conditional_losses_396174

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
Þ
E__inference_z_log_var_layer_call_and_return_conditional_losses_395709

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ý
D__inference_decoding_layer_call_and_return_conditional_losses_396069

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
E
)__inference_maxpool3_layer_call_fn_395505

inputs
identityè
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_3954992
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
ð
$__inference_VAE_layer_call_fn_397352

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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCallÓ
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_VAE_layer_call_and_return_conditional_losses_3967392
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
t
+__inference_sampling_1_layer_call_fn_397996
inputs_0
inputs_1
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_3957512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
	
Û
B__inference_z_mean_layer_call_and_return_conditional_losses_395683

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
½
C__inference_Decoder_layer_call_and_return_conditional_losses_396247
input_decoder
decoding_396080
decoding_396082
conv4_dec_396129
conv4_dec_396131
conv3_dec_396157
conv3_dec_396159
conv2_dec_396185
conv2_dec_396187
conv1_dec_396213
conv1_dec_396215
output_396241
output_396243
identity¢!conv1_dec/StatefulPartitionedCall¢!conv2_dec/StatefulPartitionedCall¢!conv3_dec/StatefulPartitionedCall¢!conv4_dec/StatefulPartitionedCall¢ decoding/StatefulPartitionedCall¢output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_396080decoding_396082*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_3960692"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3960992
reshape/PartitionedCall¿
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_396129conv4_dec_396131*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_3961182#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_3959922
upsamp4/PartitionedCallÐ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_396157conv3_dec_396159*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_3961462#
!conv3_dec/StatefulPartitionedCall
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_3960112
upsamp3/PartitionedCallÐ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_396185conv2_dec_396187*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_3961742#
!conv2_dec/StatefulPartitionedCall
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_3960302
upsamp2/PartitionedCallÐ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_396213conv1_dec_396215*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_3962022#
!conv1_dec/StatefulPartitionedCall
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_3960492
upsamp1/PartitionedCallÁ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_396241output_396243*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3962302 
output/StatefulPartitionedCallé
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinput_decoder


*__inference_conv4_enc_layer_call_fn_397896

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_3956162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
1
¶
C__inference_Decoder_layer_call_and_return_conditional_losses_396396

inputs
decoding_396360
decoding_396362
conv4_dec_396366
conv4_dec_396368
conv3_dec_396372
conv3_dec_396374
conv2_dec_396378
conv2_dec_396380
conv1_dec_396384
conv1_dec_396386
output_396390
output_396392
identity¢!conv1_dec/StatefulPartitionedCall¢!conv2_dec/StatefulPartitionedCall¢!conv3_dec/StatefulPartitionedCall¢!conv4_dec/StatefulPartitionedCall¢ decoding/StatefulPartitionedCall¢output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_396360decoding_396362*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_3960692"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3960992
reshape/PartitionedCall¿
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_396366conv4_dec_396368*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_3961182#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_3959922
upsamp4/PartitionedCallÐ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_396372conv3_dec_396374*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_3961462#
!conv3_dec/StatefulPartitionedCall
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_3960112
upsamp3/PartitionedCallÐ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_396378conv2_dec_396380*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_3961742#
!conv2_dec/StatefulPartitionedCall
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_3960302
upsamp2/PartitionedCallÐ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_396384conv1_dec_396386*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_3962022#
!conv1_dec/StatefulPartitionedCall
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_3960492
upsamp1/PartitionedCallÁ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_396390output_396392*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3962302 
output/StatefulPartitionedCallé
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

Þ
E__inference_conv3_enc_layer_call_and_return_conditional_losses_395588

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ï

Þ
E__inference_conv2_enc_layer_call_and_return_conditional_losses_395560

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
Æõ
 
?__inference_VAE_layer_call_and_return_conditional_losses_397238

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
2encoder_bottleneck_biasadd_readvariableop_resource1
-encoder_z_mean_matmul_readvariableop_resource2
.encoder_z_mean_biasadd_readvariableop_resource4
0encoder_z_log_var_matmul_readvariableop_resource5
1encoder_z_log_var_biasadd_readvariableop_resource3
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
identity¢(Decoder/conv1_dec/BiasAdd/ReadVariableOp¢'Decoder/conv1_dec/Conv2D/ReadVariableOp¢(Decoder/conv2_dec/BiasAdd/ReadVariableOp¢'Decoder/conv2_dec/Conv2D/ReadVariableOp¢(Decoder/conv3_dec/BiasAdd/ReadVariableOp¢'Decoder/conv3_dec/Conv2D/ReadVariableOp¢(Decoder/conv4_dec/BiasAdd/ReadVariableOp¢'Decoder/conv4_dec/Conv2D/ReadVariableOp¢'Decoder/decoding/BiasAdd/ReadVariableOp¢&Decoder/decoding/MatMul/ReadVariableOp¢%Decoder/output/BiasAdd/ReadVariableOp¢$Decoder/output/Conv2D/ReadVariableOp¢)Encoder/bottleneck/BiasAdd/ReadVariableOp¢(Encoder/bottleneck/MatMul/ReadVariableOp¢(Encoder/conv1_enc/BiasAdd/ReadVariableOp¢'Encoder/conv1_enc/Conv2D/ReadVariableOp¢(Encoder/conv2_enc/BiasAdd/ReadVariableOp¢'Encoder/conv2_enc/Conv2D/ReadVariableOp¢(Encoder/conv3_enc/BiasAdd/ReadVariableOp¢'Encoder/conv3_enc/Conv2D/ReadVariableOp¢(Encoder/conv4_enc/BiasAdd/ReadVariableOp¢'Encoder/conv4_enc/Conv2D/ReadVariableOp¢(Encoder/z_log_var/BiasAdd/ReadVariableOp¢'Encoder/z_log_var/MatMul/ReadVariableOp¢%Encoder/z_mean/BiasAdd/ReadVariableOp¢$Encoder/z_mean/MatMul/ReadVariableOpË
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOpÙ
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2DÂ
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOpÐ
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/conv1_enc/BiasAdd
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/conv1_enc/ReluÑ
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPoolË
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOpô
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2DÂ
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOpÐ
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
Encoder/conv2_enc/BiasAdd
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
Encoder/conv2_enc/ReluÑ
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPoolË
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOpô
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2DÂ
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOpÐ
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Encoder/conv3_enc/BiasAdd
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Encoder/conv3_enc/ReluÑ
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPoolÌ
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOpõ
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2DÃ
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOpÑ
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/conv4_enc/BiasAdd
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/conv4_enc/ReluÒ
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB"ÿÿÿÿ   2
Encoder/flatten/Const³
Encoder/flatten/ReshapeReshape!Encoder/maxpool4/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/flatten/ReshapeÇ
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOpÆ
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/bottleneck/MatMulÅ
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOpÍ
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/bottleneck/BiasAddº
$Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Encoder/z_mean/MatMul/ReadVariableOp½
Encoder/z_mean/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0,Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/z_mean/MatMul¹
%Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Encoder/z_mean/BiasAdd/ReadVariableOp½
Encoder/z_mean/BiasAddBiasAddEncoder/z_mean/MatMul:product:0-Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/z_mean/BiasAddÃ
'Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Encoder/z_log_var/MatMul/ReadVariableOpÆ
Encoder/z_log_var/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/z_log_var/MatMulÂ
(Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/z_log_var/BiasAdd/ReadVariableOpÉ
Encoder/z_log_var/BiasAddBiasAdd"Encoder/z_log_var/MatMul:product:00Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/z_log_var/BiasAdd
Encoder/sampling_1/ShapeShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling_1/Shape
&Encoder/sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Encoder/sampling_1/strided_slice/stack
(Encoder/sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_1/strided_slice/stack_1
(Encoder/sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_1/strided_slice/stack_2Ô
 Encoder/sampling_1/strided_sliceStridedSlice!Encoder/sampling_1/Shape:output:0/Encoder/sampling_1/strided_slice/stack:output:01Encoder/sampling_1/strided_slice/stack_1:output:01Encoder/sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Encoder/sampling_1/strided_slice
Encoder/sampling_1/Shape_1ShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling_1/Shape_1
(Encoder/sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_1/strided_slice_1/stack¢
*Encoder/sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*Encoder/sampling_1/strided_slice_1/stack_1¢
*Encoder/sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*Encoder/sampling_1/strided_slice_1/stack_2à
"Encoder/sampling_1/strided_slice_1StridedSlice#Encoder/sampling_1/Shape_1:output:01Encoder/sampling_1/strided_slice_1/stack:output:03Encoder/sampling_1/strided_slice_1/stack_1:output:03Encoder/sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"Encoder/sampling_1/strided_slice_1Þ
&Encoder/sampling_1/random_normal/shapePack)Encoder/sampling_1/strided_slice:output:0+Encoder/sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Encoder/sampling_1/random_normal/shape
%Encoder/sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%Encoder/sampling_1/random_normal/mean
'Encoder/sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'Encoder/sampling_1/random_normal/stddev
5Encoder/sampling_1/random_normal/RandomStandardNormalRandomStandardNormal/Encoder/sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2¦í·27
5Encoder/sampling_1/random_normal/RandomStandardNormal
$Encoder/sampling_1/random_normal/mulMul>Encoder/sampling_1/random_normal/RandomStandardNormal:output:00Encoder/sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$Encoder/sampling_1/random_normal/mulà
 Encoder/sampling_1/random_normalAdd(Encoder/sampling_1/random_normal/mul:z:0.Encoder/sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Encoder/sampling_1/random_normaly
Encoder/sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Encoder/sampling_1/mul/x°
Encoder/sampling_1/mulMul!Encoder/sampling_1/mul/x:output:0"Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/sampling_1/mul
Encoder/sampling_1/ExpExpEncoder/sampling_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/sampling_1/Exp¯
Encoder/sampling_1/mul_1MulEncoder/sampling_1/Exp:y:0$Encoder/sampling_1/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/sampling_1/mul_1ª
Encoder/sampling_1/addAddV2Encoder/z_mean/BiasAdd:output:0Encoder/sampling_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/sampling_1/addÁ
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOp»
Decoder/decoding/MatMulMatMulEncoder/sampling_1/add:z:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/decoding/MatMulÀ
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOpÆ
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/decoding/BiasAdd
Decoder/reshape/ShapeShape!Decoder/decoding/BiasAdd:output:0*
T0*
_output_shapes
:2
Decoder/reshape/Shape
#Decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Decoder/reshape/strided_slice/stack
%Decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/reshape/strided_slice/stack_1
%Decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/reshape/strided_slice/stack_2Â
Decoder/reshape/strided_sliceStridedSliceDecoder/reshape/Shape:output:0,Decoder/reshape/strided_slice/stack:output:0.Decoder/reshape/strided_slice/stack_1:output:0.Decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Decoder/reshape/strided_slice
Decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
Decoder/reshape/Reshape/shape/1
Decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
Decoder/reshape/Reshape/shape/2
Decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2!
Decoder/reshape/Reshape/shape/3
Decoder/reshape/Reshape/shapePack&Decoder/reshape/strided_slice:output:0(Decoder/reshape/Reshape/shape/1:output:0(Decoder/reshape/Reshape/shape/2:output:0(Decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Decoder/reshape/Reshape/shapeÃ
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/reshape/ReshapeÍ
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOpô
Decoder/conv4_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2DÃ
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOpÑ
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/conv4_dec/BiasAdd
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/conv4_dec/Relu
Decoder/upsamp4/ShapeShape$Decoder/conv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp4/Shape
#Decoder/upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp4/strided_slice/stack
%Decoder/upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp4/strided_slice/stack_1
%Decoder/upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp4/strided_slice/stack_2®
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
Decoder/upsamp4/Const
Decoder/upsamp4/mulMul&Decoder/upsamp4/strided_slice:output:0Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp4/mul
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighborÌ
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2DÂ
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOpÐ
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Decoder/conv3_dec/BiasAdd
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Decoder/conv3_dec/Relu
Decoder/upsamp3/ShapeShape$Decoder/conv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp3/Shape
#Decoder/upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp3/strided_slice/stack
%Decoder/upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp3/strided_slice/stack_1
%Decoder/upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp3/strided_slice/stack_2®
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
Decoder/upsamp3/Const
Decoder/upsamp3/mulMul&Decoder/upsamp3/strided_slice:output:0Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp3/mul
,Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv3_dec/Relu:activations:0Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighborË
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2DÂ
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOpÐ
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Decoder/conv2_dec/BiasAdd
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Decoder/conv2_dec/Relu
Decoder/upsamp2/ShapeShape$Decoder/conv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp2/Shape
#Decoder/upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp2/strided_slice/stack
%Decoder/upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp2/strided_slice/stack_1
%Decoder/upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp2/strided_slice/stack_2®
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
Decoder/upsamp2/Const
Decoder/upsamp2/mulMul&Decoder/upsamp2/strided_slice:output:0Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp2/mul
,Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv2_dec/Relu:activations:0Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighborË
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2DÂ
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOpÐ
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/conv1_dec/BiasAdd
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/conv1_dec/Relu
Decoder/upsamp1/ShapeShape$Decoder/conv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp1/Shape
#Decoder/upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp1/strided_slice/stack
%Decoder/upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp1/strided_slice/stack_1
%Decoder/upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp1/strided_slice/stack_2®
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
Decoder/upsamp1/Const
Decoder/upsamp1/mulMul&Decoder/upsamp1/strided_slice:output:0Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp1/mul
,Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv1_dec/Relu:activations:0Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighborÂ
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Decoder/output/Conv2D¹
%Decoder/output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Decoder/output/BiasAdd/ReadVariableOpÄ
Decoder/output/BiasAddBiasAddDecoder/output/Conv2D:output:0-Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/output/BiasAdd
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/output/Sigmoid»	
IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp)^Encoder/z_log_var/BiasAdd/ReadVariableOp(^Encoder/z_log_var/MatMul/ReadVariableOp&^Encoder/z_mean/BiasAdd/ReadVariableOp%^Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2T
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
'Encoder/conv4_enc/Conv2D/ReadVariableOp'Encoder/conv4_enc/Conv2D/ReadVariableOp2T
(Encoder/z_log_var/BiasAdd/ReadVariableOp(Encoder/z_log_var/BiasAdd/ReadVariableOp2R
'Encoder/z_log_var/MatMul/ReadVariableOp'Encoder/z_log_var/MatMul/ReadVariableOp2N
%Encoder/z_mean/BiasAdd/ReadVariableOp%Encoder/z_mean/BiasAdd/ReadVariableOp2L
$Encoder/z_mean/MatMul/ReadVariableOp$Encoder/z_mean/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
C__inference_upsamp2_layer_call_and_return_conditional_losses_396030

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
üo
è	
C__inference_Encoder_layer_call_and_return_conditional_losses_397510

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
*bottleneck_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2¢!bottleneck/BiasAdd/ReadVariableOp¢ bottleneck/MatMul/ReadVariableOp¢ conv1_enc/BiasAdd/ReadVariableOp¢conv1_enc/Conv2D/ReadVariableOp¢ conv2_enc/BiasAdd/ReadVariableOp¢conv2_enc/Conv2D/ReadVariableOp¢ conv3_enc/BiasAdd/ReadVariableOp¢conv3_enc/Conv2D/ReadVariableOp¢ conv4_enc/BiasAdd/ReadVariableOp¢conv4_enc/Conv2D/ReadVariableOp¢ z_log_var/BiasAdd/ReadVariableOp¢z_log_var/MatMul/ReadVariableOp¢z_mean/BiasAdd/ReadVariableOp¢z_mean/MatMul/ReadVariableOp³
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOpÁ
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1_enc/Conv2Dª
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOp°
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1_enc/Relu¹
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool³
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_enc/Conv2D/ReadVariableOpÔ
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingSAME*
strides
2
conv2_enc/Conv2Dª
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_enc/BiasAdd/ReadVariableOp°
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2_enc/Relu¹
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPool³
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv3_enc/Conv2D/ReadVariableOpÔ
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv3_enc/Conv2Dª
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_enc/BiasAdd/ReadVariableOp°
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv3_enc/Relu¹
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPool´
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv4_enc/Conv2D/ReadVariableOpÕ
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv4_enc/Conv2D«
 conv4_enc/BiasAdd/ReadVariableOpReadVariableOp)conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv4_enc/BiasAdd/ReadVariableOp±
conv4_enc/BiasAddBiasAddconv4_enc/Conv2D:output:0(conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv4_enc/BiasAdd
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv4_enc/Reluº
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapemaxpool4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¯
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 bottleneck/MatMul/ReadVariableOp¦
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bottleneck/MatMul­
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp­
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bottleneck/BiasAdd¢
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMulbottleneck/BiasAdd:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
z_mean/MatMul¡
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
z_mean/BiasAdd«
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
z_log_var/MatMul/ReadVariableOp¦
z_log_var/MatMulMatMulbottleneck/BiasAdd:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
z_log_var/MatMulª
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp©
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
z_log_var/BiasAddk
sampling_1/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape
sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
sampling_1/strided_slice/stack
 sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_1
 sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_2¤
sampling_1/strided_sliceStridedSlicesampling_1/Shape:output:0'sampling_1/strided_slice/stack:output:0)sampling_1/strided_slice/stack_1:output:0)sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_sliceo
sampling_1/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape_1
 sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice_1/stack
"sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_1
"sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_2°
sampling_1/strided_slice_1StridedSlicesampling_1/Shape_1:output:0)sampling_1/strided_slice_1/stack:output:0+sampling_1/strided_slice_1/stack_1:output:0+sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_slice_1¾
sampling_1/random_normal/shapePack!sampling_1/strided_slice:output:0#sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2 
sampling_1/random_normal/shape
sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling_1/random_normal/mean
sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
sampling_1/random_normal/stddev
-sampling_1/random_normal/RandomStandardNormalRandomStandardNormal'sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2º2/
-sampling_1/random_normal/RandomStandardNormalà
sampling_1/random_normal/mulMul6sampling_1/random_normal/RandomStandardNormal:output:0(sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sampling_1/random_normal/mulÀ
sampling_1/random_normalAdd sampling_1/random_normal/mul:z:0&sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sampling_1/random_normali
sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling_1/mul/x
sampling_1/mulMulsampling_1/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sampling_1/mulm
sampling_1/ExpExpsampling_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sampling_1/Exp
sampling_1/mul_1Mulsampling_1/Exp:y:0sampling_1/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sampling_1/mul_1
sampling_1/addAddV2z_mean/BiasAdd:output:0sampling_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sampling_1/addÊ
IdentityIdentityz_mean/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÑ

Identity_1Identityz_log_var/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1É

Identity_2Identitysampling_1/add:z:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::2F
!bottleneck/BiasAdd/ReadVariableOp!bottleneck/BiasAdd/ReadVariableOp2D
 bottleneck/MatMul/ReadVariableOp bottleneck/MatMul/ReadVariableOp2D
 conv1_enc/BiasAdd/ReadVariableOp conv1_enc/BiasAdd/ReadVariableOp2B
conv1_enc/Conv2D/ReadVariableOpconv1_enc/Conv2D/ReadVariableOp2D
 conv2_enc/BiasAdd/ReadVariableOp conv2_enc/BiasAdd/ReadVariableOp2B
conv2_enc/Conv2D/ReadVariableOpconv2_enc/Conv2D/ReadVariableOp2D
 conv3_enc/BiasAdd/ReadVariableOp conv3_enc/BiasAdd/ReadVariableOp2B
conv3_enc/Conv2D/ReadVariableOpconv3_enc/Conv2D/ReadVariableOp2D
 conv4_enc/BiasAdd/ReadVariableOp conv4_enc/BiasAdd/ReadVariableOp2B
conv4_enc/Conv2D/ReadVariableOpconv4_enc/Conv2D/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
D
(__inference_upsamp2_layer_call_fn_396036

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_3960302
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
C__inference_upsamp3_layer_call_and_return_conditional_losses_396011

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
`
D__inference_maxpool3_layer_call_and_return_conditional_losses_395499

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Þ
E__inference_conv2_dec_layer_call_and_return_conditional_losses_398085

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
½
_
C__inference_flatten_layer_call_and_return_conditional_losses_395639

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

_
C__inference_upsamp4_layer_call_and_return_conditional_losses_395992

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
strided_slice/stack_2Î
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
mulÕ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2
resize/ResizeNearestNeighbor¤
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
1
½
C__inference_Decoder_layer_call_and_return_conditional_losses_396286
input_decoder
decoding_396250
decoding_396252
conv4_dec_396256
conv4_dec_396258
conv3_dec_396262
conv3_dec_396264
conv2_dec_396268
conv2_dec_396270
conv1_dec_396274
conv1_dec_396276
output_396280
output_396282
identity¢!conv1_dec/StatefulPartitionedCall¢!conv2_dec/StatefulPartitionedCall¢!conv3_dec/StatefulPartitionedCall¢!conv4_dec/StatefulPartitionedCall¢ decoding/StatefulPartitionedCall¢output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_396250decoding_396252*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_3960692"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3960992
reshape/PartitionedCall¿
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_396256conv4_dec_396258*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_3961182#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_3959922
upsamp4/PartitionedCallÐ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_396262conv3_dec_396264*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_3961462#
!conv3_dec/StatefulPartitionedCall
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_3960112
upsamp3/PartitionedCallÐ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_396268conv2_dec_396270*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_3961742#
!conv2_dec/StatefulPartitionedCall
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_3960302
upsamp2/PartitionedCallÐ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_396274conv1_dec_396276*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_3962022#
!conv1_dec/StatefulPartitionedCall
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_3960492
upsamp1/PartitionedCallÁ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_396280output_396282*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3962302 
output/StatefulPartitionedCallé
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinput_decoder
á
~
)__inference_decoding_layer_call_fn_398015

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_3960692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Û
B__inference_z_mean_layer_call_and_return_conditional_losses_397936

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
A
Ñ
C__inference_Encoder_layer_call_and_return_conditional_losses_395860

inputs
conv1_enc_395816
conv1_enc_395818
conv2_enc_395822
conv2_enc_395824
conv3_enc_395828
conv3_enc_395830
conv4_enc_395834
conv4_enc_395836
bottleneck_395841
bottleneck_395843
z_mean_395846
z_mean_395848
z_log_var_395851
z_log_var_395853
identity

identity_1

identity_2¢"bottleneck/StatefulPartitionedCall¢!conv1_enc/StatefulPartitionedCall¢!conv2_enc/StatefulPartitionedCall¢!conv3_enc/StatefulPartitionedCall¢!conv4_enc/StatefulPartitionedCall¢"sampling_1/StatefulPartitionedCall¢!z_log_var/StatefulPartitionedCall¢z_mean/StatefulPartitionedCall¤
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_395816conv1_enc_395818*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_3955322#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_3954752
maxpool1/PartitionedCall¿
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_395822conv2_enc_395824*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_3955602#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_3954872
maxpool2/PartitionedCall¿
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_395828conv3_enc_395830*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_3955882#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_3954992
maxpool3/PartitionedCallÀ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_395834conv4_enc_395836*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_3956162#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_3955112
maxpool4/PartitionedCallð
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3956392
flatten/PartitionedCall»
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_395841bottleneck_395843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_3956572$
"bottleneck/StatefulPartitionedCall²
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_395846z_mean_395848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_3956832 
z_mean/StatefulPartitionedCallÁ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_395851z_log_var_395853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_3957092#
!z_log_var/StatefulPartitionedCallÃ
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_3957512$
"sampling_1/StatefulPartitionedCall
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¡

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1¢

Identity_2Identity+sampling_1/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Éo

C__inference_Decoder_layer_call_and_return_conditional_losses_397671

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
identity¢ conv1_dec/BiasAdd/ReadVariableOp¢conv1_dec/Conv2D/ReadVariableOp¢ conv2_dec/BiasAdd/ReadVariableOp¢conv2_dec/Conv2D/ReadVariableOp¢ conv3_dec/BiasAdd/ReadVariableOp¢conv3_dec/Conv2D/ReadVariableOp¢ conv4_dec/BiasAdd/ReadVariableOp¢conv4_dec/Conv2D/ReadVariableOp¢decoding/BiasAdd/ReadVariableOp¢decoding/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/Conv2D/ReadVariableOp©
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
decoding/MatMul/ReadVariableOp
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
decoding/MatMul¨
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
decoding/BiasAdd/ReadVariableOp¦
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
decoding/BiasAddg
reshape/ShapeShapedecoding/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
B :2
reshape/Reshape/shape/3ê
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape£
reshape/ReshapeReshapedecoding/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/Reshapeµ
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv4_dec/Conv2D/ReadVariableOpÔ
conv4_dec/Conv2DConv2Dreshape/Reshape:output:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv4_dec/Conv2D«
 conv4_dec/BiasAdd/ReadVariableOpReadVariableOp)conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv4_dec/BiasAdd/ReadVariableOp±
conv4_dec/BiasAddBiasAddconv4_dec/Conv2D:output:0(conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv4_dec/BiasAdd
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv4_dec/Reluj
upsamp4/ShapeShapeconv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp4/Shape
upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack
upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack_1
upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack_2þ
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
upsamp4/mulé
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighbor´
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv3_dec/Conv2D/ReadVariableOpð
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv3_dec/Conv2Dª
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_dec/BiasAdd/ReadVariableOp°
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv3_dec/Reluj
upsamp3/ShapeShapeconv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp3/Shape
upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack
upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack_1
upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack_2þ
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
upsamp3/mulè
$upsamp3/resize/ResizeNearestNeighborResizeNearestNeighborconv3_dec/Relu:activations:0upsamp3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighbor³
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2_dec/Conv2D/ReadVariableOpð
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2_dec/Conv2Dª
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_dec/BiasAdd/ReadVariableOp°
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2_dec/Reluj
upsamp2/ShapeShapeconv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp2/Shape
upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack
upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack_1
upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack_2þ
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
upsamp2/mulè
$upsamp2/resize/ResizeNearestNeighborResizeNearestNeighborconv2_dec/Relu:activations:0upsamp2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighbor³
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv1_dec/Conv2D/ReadVariableOpð
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1_dec/Conv2Dª
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOp°
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1_dec/Reluj
upsamp1/ShapeShapeconv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp1/Shape
upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack
upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack_1
upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack_2þ
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
upsamp1/mulè
$upsamp1/resize/ResizeNearestNeighborResizeNearestNeighborconv1_dec/Relu:activations:0upsamp1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighborª
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
output/Conv2D/ReadVariableOpè
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
output/Conv2D¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp¤
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/Sigmoid
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2D
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Çð
.
"__inference__traced_restore_398701
file_prefix
assignvariableop_total
assignvariableop_1_count
assignvariableop_2_total_1
assignvariableop_3_count_1
assignvariableop_4_total_2
assignvariableop_5_count_2 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate(
$assignvariableop_11_conv1_enc_kernel&
"assignvariableop_12_conv1_enc_bias(
$assignvariableop_13_conv2_enc_kernel&
"assignvariableop_14_conv2_enc_bias(
$assignvariableop_15_conv3_enc_kernel&
"assignvariableop_16_conv3_enc_bias(
$assignvariableop_17_conv4_enc_kernel&
"assignvariableop_18_conv4_enc_bias)
%assignvariableop_19_bottleneck_kernel'
#assignvariableop_20_bottleneck_bias%
!assignvariableop_21_z_mean_kernel#
assignvariableop_22_z_mean_bias(
$assignvariableop_23_z_log_var_kernel&
"assignvariableop_24_z_log_var_bias'
#assignvariableop_25_decoding_kernel%
!assignvariableop_26_decoding_bias(
$assignvariableop_27_conv4_dec_kernel&
"assignvariableop_28_conv4_dec_bias(
$assignvariableop_29_conv3_dec_kernel&
"assignvariableop_30_conv3_dec_bias(
$assignvariableop_31_conv2_dec_kernel&
"assignvariableop_32_conv2_dec_bias(
$assignvariableop_33_conv1_dec_kernel&
"assignvariableop_34_conv1_dec_bias%
!assignvariableop_35_output_kernel#
assignvariableop_36_output_bias/
+assignvariableop_37_adam_conv1_enc_kernel_m-
)assignvariableop_38_adam_conv1_enc_bias_m/
+assignvariableop_39_adam_conv2_enc_kernel_m-
)assignvariableop_40_adam_conv2_enc_bias_m/
+assignvariableop_41_adam_conv3_enc_kernel_m-
)assignvariableop_42_adam_conv3_enc_bias_m/
+assignvariableop_43_adam_conv4_enc_kernel_m-
)assignvariableop_44_adam_conv4_enc_bias_m0
,assignvariableop_45_adam_bottleneck_kernel_m.
*assignvariableop_46_adam_bottleneck_bias_m,
(assignvariableop_47_adam_z_mean_kernel_m*
&assignvariableop_48_adam_z_mean_bias_m/
+assignvariableop_49_adam_z_log_var_kernel_m-
)assignvariableop_50_adam_z_log_var_bias_m.
*assignvariableop_51_adam_decoding_kernel_m,
(assignvariableop_52_adam_decoding_bias_m/
+assignvariableop_53_adam_conv4_dec_kernel_m-
)assignvariableop_54_adam_conv4_dec_bias_m/
+assignvariableop_55_adam_conv3_dec_kernel_m-
)assignvariableop_56_adam_conv3_dec_bias_m/
+assignvariableop_57_adam_conv2_dec_kernel_m-
)assignvariableop_58_adam_conv2_dec_bias_m/
+assignvariableop_59_adam_conv1_dec_kernel_m-
)assignvariableop_60_adam_conv1_dec_bias_m,
(assignvariableop_61_adam_output_kernel_m*
&assignvariableop_62_adam_output_bias_m/
+assignvariableop_63_adam_conv1_enc_kernel_v-
)assignvariableop_64_adam_conv1_enc_bias_v/
+assignvariableop_65_adam_conv2_enc_kernel_v-
)assignvariableop_66_adam_conv2_enc_bias_v/
+assignvariableop_67_adam_conv3_enc_kernel_v-
)assignvariableop_68_adam_conv3_enc_bias_v/
+assignvariableop_69_adam_conv4_enc_kernel_v-
)assignvariableop_70_adam_conv4_enc_bias_v0
,assignvariableop_71_adam_bottleneck_kernel_v.
*assignvariableop_72_adam_bottleneck_bias_v,
(assignvariableop_73_adam_z_mean_kernel_v*
&assignvariableop_74_adam_z_mean_bias_v/
+assignvariableop_75_adam_z_log_var_kernel_v-
)assignvariableop_76_adam_z_log_var_bias_v.
*assignvariableop_77_adam_decoding_kernel_v,
(assignvariableop_78_adam_decoding_bias_v/
+assignvariableop_79_adam_conv4_dec_kernel_v-
)assignvariableop_80_adam_conv4_dec_bias_v/
+assignvariableop_81_adam_conv3_dec_kernel_v-
)assignvariableop_82_adam_conv3_dec_bias_v/
+assignvariableop_83_adam_conv2_dec_kernel_v-
)assignvariableop_84_adam_conv2_dec_bias_v/
+assignvariableop_85_adam_conv1_dec_kernel_v-
)assignvariableop_86_adam_conv1_dec_bias_v,
(assignvariableop_87_adam_output_kernel_v*
&assignvariableop_88_adam_output_bias_v
identity_90¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_9ò/
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*þ.
valueô.Bñ.ZB3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÅ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*É
value¿B¼ZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesð
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*þ
_output_shapesë
è::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*h
dtypes^
\2Z	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_totalIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_countIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2
AssignVariableOp_2AssignVariableOpassignvariableop_2_total_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOpassignvariableop_3_count_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_total_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_count_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6¡
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¢
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10®
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¬
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv1_enc_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ª
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv1_enc_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¬
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2_enc_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ª
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2_enc_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¬
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv3_enc_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ª
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv3_enc_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¬
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv4_enc_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ª
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv4_enc_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19­
AssignVariableOp_19AssignVariableOp%assignvariableop_19_bottleneck_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20«
AssignVariableOp_20AssignVariableOp#assignvariableop_20_bottleneck_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21©
AssignVariableOp_21AssignVariableOp!assignvariableop_21_z_mean_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22§
AssignVariableOp_22AssignVariableOpassignvariableop_22_z_mean_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¬
AssignVariableOp_23AssignVariableOp$assignvariableop_23_z_log_var_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ª
AssignVariableOp_24AssignVariableOp"assignvariableop_24_z_log_var_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25«
AssignVariableOp_25AssignVariableOp#assignvariableop_25_decoding_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26©
AssignVariableOp_26AssignVariableOp!assignvariableop_26_decoding_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¬
AssignVariableOp_27AssignVariableOp$assignvariableop_27_conv4_dec_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ª
AssignVariableOp_28AssignVariableOp"assignvariableop_28_conv4_dec_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¬
AssignVariableOp_29AssignVariableOp$assignvariableop_29_conv3_dec_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ª
AssignVariableOp_30AssignVariableOp"assignvariableop_30_conv3_dec_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¬
AssignVariableOp_31AssignVariableOp$assignvariableop_31_conv2_dec_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ª
AssignVariableOp_32AssignVariableOp"assignvariableop_32_conv2_dec_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¬
AssignVariableOp_33AssignVariableOp$assignvariableop_33_conv1_dec_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ª
AssignVariableOp_34AssignVariableOp"assignvariableop_34_conv1_dec_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35©
AssignVariableOp_35AssignVariableOp!assignvariableop_35_output_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36§
AssignVariableOp_36AssignVariableOpassignvariableop_36_output_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37³
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv1_enc_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv1_enc_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39³
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2_enc_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40±
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2_enc_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41³
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv3_enc_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42±
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv3_enc_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43³
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv4_enc_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44±
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv4_enc_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45´
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_bottleneck_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46²
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_bottleneck_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47°
AssignVariableOp_47AssignVariableOp(assignvariableop_47_adam_z_mean_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48®
AssignVariableOp_48AssignVariableOp&assignvariableop_48_adam_z_mean_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49³
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_z_log_var_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_z_log_var_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51²
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_decoding_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52°
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_decoding_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53³
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv4_dec_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54±
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv4_dec_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55³
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv3_dec_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56±
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv3_dec_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57³
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2_dec_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58±
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2_dec_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59³
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv1_dec_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60±
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv1_dec_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61°
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_output_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62®
AssignVariableOp_62AssignVariableOp&assignvariableop_62_adam_output_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63³
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv1_enc_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64±
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv1_enc_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65³
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2_enc_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66±
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2_enc_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67³
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv3_enc_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68±
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv3_enc_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69³
AssignVariableOp_69AssignVariableOp+assignvariableop_69_adam_conv4_enc_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70±
AssignVariableOp_70AssignVariableOp)assignvariableop_70_adam_conv4_enc_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71´
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_bottleneck_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72²
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_bottleneck_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73°
AssignVariableOp_73AssignVariableOp(assignvariableop_73_adam_z_mean_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74®
AssignVariableOp_74AssignVariableOp&assignvariableop_74_adam_z_mean_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75³
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_z_log_var_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76±
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_z_log_var_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77²
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_decoding_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78°
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_decoding_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79³
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_conv4_dec_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80±
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_conv4_dec_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81³
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_conv3_dec_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82±
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_conv3_dec_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83³
AssignVariableOp_83AssignVariableOp+assignvariableop_83_adam_conv2_dec_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84±
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adam_conv2_dec_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85³
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_conv1_dec_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86±
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_conv1_dec_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87°
AssignVariableOp_87AssignVariableOp(assignvariableop_87_adam_output_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88®
AssignVariableOp_88AssignVariableOp&assignvariableop_88_adam_output_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_889
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_89Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_89÷
Identity_90IdentityIdentity_89:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_90"#
identity_90Identity_90:output:0*û
_input_shapesé
æ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
é
_
C__inference_reshape_layer_call_and_return_conditional_losses_396099

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
strided_slice/stack_2â
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
B :2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Û
B__inference_output_layer_call_and_return_conditional_losses_398125

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidª
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ

Þ
E__inference_conv4_enc_layer_call_and_return_conditional_losses_395616

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û
|
'__inference_z_mean_layer_call_fn_397945

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_3956832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
ª
?__inference_VAE_layer_call_and_return_conditional_losses_396739

inputs
encoder_396682
encoder_396684
encoder_396686
encoder_396688
encoder_396690
encoder_396692
encoder_396694
encoder_396696
encoder_396698
encoder_396700
encoder_396702
encoder_396704
encoder_396706
encoder_396708
decoder_396713
decoder_396715
decoder_396717
decoder_396719
decoder_396721
decoder_396723
decoder_396725
decoder_396727
decoder_396729
decoder_396731
decoder_396733
decoder_396735
identity¢Decoder/StatefulPartitionedCall¢Encoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_396682encoder_396684encoder_396686encoder_396688encoder_396690encoder_396692encoder_396694encoder_396696encoder_396698encoder_396700encoder_396702encoder_396704encoder_396706encoder_396708*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_3959442!
Encoder/StatefulPartitionedCall
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_396713decoder_396715decoder_396717decoder_396719decoder_396721decoder_396723decoder_396725decoder_396727decoder_396729decoder_396731decoder_396733decoder_396735*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_3963962!
Decoder/StatefulPartitionedCallÚ
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ñ
$__inference_VAE_layer_call_fn_396851
input_1
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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_VAE_layer_call_and_return_conditional_losses_3967392
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ã
|
'__inference_output_layer_call_fn_398134

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3962302
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
Û
(__inference_Encoder_layer_call_fn_395979
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

unknown_12
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_encoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_3959442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinput_encoder
§
D
(__inference_flatten_layer_call_fn_397907

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3956392
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
`
D__inference_maxpool4_layer_call_and_return_conditional_losses_395511

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
A
Ñ
C__inference_Encoder_layer_call_and_return_conditional_losses_395944

inputs
conv1_enc_395900
conv1_enc_395902
conv2_enc_395906
conv2_enc_395908
conv3_enc_395912
conv3_enc_395914
conv4_enc_395918
conv4_enc_395920
bottleneck_395925
bottleneck_395927
z_mean_395930
z_mean_395932
z_log_var_395935
z_log_var_395937
identity

identity_1

identity_2¢"bottleneck/StatefulPartitionedCall¢!conv1_enc/StatefulPartitionedCall¢!conv2_enc/StatefulPartitionedCall¢!conv3_enc/StatefulPartitionedCall¢!conv4_enc/StatefulPartitionedCall¢"sampling_1/StatefulPartitionedCall¢!z_log_var/StatefulPartitionedCall¢z_mean/StatefulPartitionedCall¤
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_395900conv1_enc_395902*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_3955322#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_3954752
maxpool1/PartitionedCall¿
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_395906conv2_enc_395908*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_3955602#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_3954872
maxpool2/PartitionedCall¿
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_395912conv3_enc_395914*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_3955882#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_3954992
maxpool3/PartitionedCallÀ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_395918conv4_enc_395920*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_3956162#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_3955112
maxpool4/PartitionedCallð
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3956392
flatten/PartitionedCall»
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_395925bottleneck_395927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_3956572$
"bottleneck/StatefulPartitionedCall²
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_395930z_mean_395932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_3956832 
z_mean/StatefulPartitionedCallÁ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_395935z_log_var_395937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_3957092#
!z_log_var/StatefulPartitionedCallÃ
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_3957512$
"sampling_1/StatefulPartitionedCall
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¡

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1¢

Identity_2Identity+sampling_1/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
Ó
!__inference__wrapped_model_395469
input_18
4vae_encoder_conv1_enc_conv2d_readvariableop_resource9
5vae_encoder_conv1_enc_biasadd_readvariableop_resource8
4vae_encoder_conv2_enc_conv2d_readvariableop_resource9
5vae_encoder_conv2_enc_biasadd_readvariableop_resource8
4vae_encoder_conv3_enc_conv2d_readvariableop_resource9
5vae_encoder_conv3_enc_biasadd_readvariableop_resource8
4vae_encoder_conv4_enc_conv2d_readvariableop_resource9
5vae_encoder_conv4_enc_biasadd_readvariableop_resource9
5vae_encoder_bottleneck_matmul_readvariableop_resource:
6vae_encoder_bottleneck_biasadd_readvariableop_resource5
1vae_encoder_z_mean_matmul_readvariableop_resource6
2vae_encoder_z_mean_biasadd_readvariableop_resource8
4vae_encoder_z_log_var_matmul_readvariableop_resource9
5vae_encoder_z_log_var_biasadd_readvariableop_resource7
3vae_decoder_decoding_matmul_readvariableop_resource8
4vae_decoder_decoding_biasadd_readvariableop_resource8
4vae_decoder_conv4_dec_conv2d_readvariableop_resource9
5vae_decoder_conv4_dec_biasadd_readvariableop_resource8
4vae_decoder_conv3_dec_conv2d_readvariableop_resource9
5vae_decoder_conv3_dec_biasadd_readvariableop_resource8
4vae_decoder_conv2_dec_conv2d_readvariableop_resource9
5vae_decoder_conv2_dec_biasadd_readvariableop_resource8
4vae_decoder_conv1_dec_conv2d_readvariableop_resource9
5vae_decoder_conv1_dec_biasadd_readvariableop_resource5
1vae_decoder_output_conv2d_readvariableop_resource6
2vae_decoder_output_biasadd_readvariableop_resource
identity¢,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp¢+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp¢,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp¢+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp¢,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp¢+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp¢,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp¢+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp¢+VAE/Decoder/decoding/BiasAdd/ReadVariableOp¢*VAE/Decoder/decoding/MatMul/ReadVariableOp¢)VAE/Decoder/output/BiasAdd/ReadVariableOp¢(VAE/Decoder/output/Conv2D/ReadVariableOp¢-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp¢,VAE/Encoder/bottleneck/MatMul/ReadVariableOp¢,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp¢+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp¢,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp¢+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp¢,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp¢+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp¢,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp¢+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp¢,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp¢+VAE/Encoder/z_log_var/MatMul/ReadVariableOp¢)VAE/Encoder/z_mean/BiasAdd/ReadVariableOp¢(VAE/Encoder/z_mean/MatMul/ReadVariableOp×
+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOpæ
VAE/Encoder/conv1_enc/Conv2DConv2Dinput_13VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
VAE/Encoder/conv1_enc/Conv2DÎ
,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpà
VAE/Encoder/conv1_enc/BiasAddBiasAdd%VAE/Encoder/conv1_enc/Conv2D:output:04VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/conv1_enc/BiasAdd¢
VAE/Encoder/conv1_enc/ReluRelu&VAE/Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/conv1_enc/ReluÝ
VAE/Encoder/maxpool1/MaxPoolMaxPool(VAE/Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool1/MaxPool×
+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv2_enc/Conv2DConv2D%VAE/Encoder/maxpool1/MaxPool:output:03VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingSAME*
strides
2
VAE/Encoder/conv2_enc/Conv2DÎ
,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpà
VAE/Encoder/conv2_enc/BiasAddBiasAdd%VAE/Encoder/conv2_enc/Conv2D:output:04VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
VAE/Encoder/conv2_enc/BiasAdd¢
VAE/Encoder/conv2_enc/ReluRelu&VAE/Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
VAE/Encoder/conv2_enc/ReluÝ
VAE/Encoder/maxpool2/MaxPoolMaxPool(VAE/Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool2/MaxPool×
+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv3_enc/Conv2DConv2D%VAE/Encoder/maxpool2/MaxPool:output:03VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
VAE/Encoder/conv3_enc/Conv2DÎ
,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpà
VAE/Encoder/conv3_enc/BiasAddBiasAdd%VAE/Encoder/conv3_enc/Conv2D:output:04VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
VAE/Encoder/conv3_enc/BiasAdd¢
VAE/Encoder/conv3_enc/ReluRelu&VAE/Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
VAE/Encoder/conv3_enc/ReluÝ
VAE/Encoder/maxpool3/MaxPoolMaxPool(VAE/Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool3/MaxPoolØ
+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv4_enc/Conv2DConv2D%VAE/Encoder/maxpool3/MaxPool:output:03VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
VAE/Encoder/conv4_enc/Conv2DÏ
,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpá
VAE/Encoder/conv4_enc/BiasAddBiasAdd%VAE/Encoder/conv4_enc/Conv2D:output:04VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/conv4_enc/BiasAdd£
VAE/Encoder/conv4_enc/ReluRelu&VAE/Encoder/conv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/conv4_enc/ReluÞ
VAE/Encoder/maxpool4/MaxPoolMaxPool(VAE/Encoder/conv4_enc/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool4/MaxPool
VAE/Encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
VAE/Encoder/flatten/ConstÃ
VAE/Encoder/flatten/ReshapeReshape%VAE/Encoder/maxpool4/MaxPool:output:0"VAE/Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/flatten/ReshapeÓ
,VAE/Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp5vae_encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02.
,VAE/Encoder/bottleneck/MatMul/ReadVariableOpÖ
VAE/Encoder/bottleneck/MatMulMatMul$VAE/Encoder/flatten/Reshape:output:04VAE/Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/bottleneck/MatMulÑ
-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp6vae_encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOpÝ
VAE/Encoder/bottleneck/BiasAddBiasAdd'VAE/Encoder/bottleneck/MatMul:product:05VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
VAE/Encoder/bottleneck/BiasAddÆ
(VAE/Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp1vae_encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(VAE/Encoder/z_mean/MatMul/ReadVariableOpÍ
VAE/Encoder/z_mean/MatMulMatMul'VAE/Encoder/bottleneck/BiasAdd:output:00VAE/Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/z_mean/MatMulÅ
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp2vae_encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOpÍ
VAE/Encoder/z_mean/BiasAddBiasAdd#VAE/Encoder/z_mean/MatMul:product:01VAE/Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/z_mean/BiasAddÏ
+VAE/Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp4vae_encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+VAE/Encoder/z_log_var/MatMul/ReadVariableOpÖ
VAE/Encoder/z_log_var/MatMulMatMul'VAE/Encoder/bottleneck/BiasAdd:output:03VAE/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/z_log_var/MatMulÎ
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOpÙ
VAE/Encoder/z_log_var/BiasAddBiasAdd&VAE/Encoder/z_log_var/MatMul:product:04VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/z_log_var/BiasAdd
VAE/Encoder/sampling_1/ShapeShape#VAE/Encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
VAE/Encoder/sampling_1/Shape¢
*VAE/Encoder/sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*VAE/Encoder/sampling_1/strided_slice/stack¦
,VAE/Encoder/sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling_1/strided_slice/stack_1¦
,VAE/Encoder/sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling_1/strided_slice/stack_2ì
$VAE/Encoder/sampling_1/strided_sliceStridedSlice%VAE/Encoder/sampling_1/Shape:output:03VAE/Encoder/sampling_1/strided_slice/stack:output:05VAE/Encoder/sampling_1/strided_slice/stack_1:output:05VAE/Encoder/sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$VAE/Encoder/sampling_1/strided_slice
VAE/Encoder/sampling_1/Shape_1Shape#VAE/Encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2 
VAE/Encoder/sampling_1/Shape_1¦
,VAE/Encoder/sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling_1/strided_slice_1/stackª
.VAE/Encoder/sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.VAE/Encoder/sampling_1/strided_slice_1/stack_1ª
.VAE/Encoder/sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.VAE/Encoder/sampling_1/strided_slice_1/stack_2ø
&VAE/Encoder/sampling_1/strided_slice_1StridedSlice'VAE/Encoder/sampling_1/Shape_1:output:05VAE/Encoder/sampling_1/strided_slice_1/stack:output:07VAE/Encoder/sampling_1/strided_slice_1/stack_1:output:07VAE/Encoder/sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&VAE/Encoder/sampling_1/strided_slice_1î
*VAE/Encoder/sampling_1/random_normal/shapePack-VAE/Encoder/sampling_1/strided_slice:output:0/VAE/Encoder/sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2,
*VAE/Encoder/sampling_1/random_normal/shape
)VAE/Encoder/sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)VAE/Encoder/sampling_1/random_normal/mean
+VAE/Encoder/sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+VAE/Encoder/sampling_1/random_normal/stddevª
9VAE/Encoder/sampling_1/random_normal/RandomStandardNormalRandomStandardNormal3VAE/Encoder/sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2èº2;
9VAE/Encoder/sampling_1/random_normal/RandomStandardNormal
(VAE/Encoder/sampling_1/random_normal/mulMulBVAE/Encoder/sampling_1/random_normal/RandomStandardNormal:output:04VAE/Encoder/sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(VAE/Encoder/sampling_1/random_normal/mulð
$VAE/Encoder/sampling_1/random_normalAdd,VAE/Encoder/sampling_1/random_normal/mul:z:02VAE/Encoder/sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$VAE/Encoder/sampling_1/random_normal
VAE/Encoder/sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
VAE/Encoder/sampling_1/mul/xÀ
VAE/Encoder/sampling_1/mulMul%VAE/Encoder/sampling_1/mul/x:output:0&VAE/Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/sampling_1/mul
VAE/Encoder/sampling_1/ExpExpVAE/Encoder/sampling_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/sampling_1/Exp¿
VAE/Encoder/sampling_1/mul_1MulVAE/Encoder/sampling_1/Exp:y:0(VAE/Encoder/sampling_1/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/sampling_1/mul_1º
VAE/Encoder/sampling_1/addAddV2#VAE/Encoder/z_mean/BiasAdd:output:0 VAE/Encoder/sampling_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Encoder/sampling_1/addÍ
*VAE/Decoder/decoding/MatMul/ReadVariableOpReadVariableOp3vae_decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*VAE/Decoder/decoding/MatMul/ReadVariableOpË
VAE/Decoder/decoding/MatMulMatMulVAE/Encoder/sampling_1/add:z:02VAE/Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Decoder/decoding/MatMulÌ
+VAE/Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp4vae_decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+VAE/Decoder/decoding/BiasAdd/ReadVariableOpÖ
VAE/Decoder/decoding/BiasAddBiasAdd%VAE/Decoder/decoding/MatMul:product:03VAE/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Decoder/decoding/BiasAdd
VAE/Decoder/reshape/ShapeShape%VAE/Decoder/decoding/BiasAdd:output:0*
T0*
_output_shapes
:2
VAE/Decoder/reshape/Shape
'VAE/Decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'VAE/Decoder/reshape/strided_slice/stack 
)VAE/Decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/reshape/strided_slice/stack_1 
)VAE/Decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/reshape/strided_slice/stack_2Ú
!VAE/Decoder/reshape/strided_sliceStridedSlice"VAE/Decoder/reshape/Shape:output:00VAE/Decoder/reshape/strided_slice/stack:output:02VAE/Decoder/reshape/strided_slice/stack_1:output:02VAE/Decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!VAE/Decoder/reshape/strided_slice
#VAE/Decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#VAE/Decoder/reshape/Reshape/shape/1
#VAE/Decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#VAE/Decoder/reshape/Reshape/shape/2
#VAE/Decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2%
#VAE/Decoder/reshape/Reshape/shape/3²
!VAE/Decoder/reshape/Reshape/shapePack*VAE/Decoder/reshape/strided_slice:output:0,VAE/Decoder/reshape/Reshape/shape/1:output:0,VAE/Decoder/reshape/Reshape/shape/2:output:0,VAE/Decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!VAE/Decoder/reshape/Reshape/shapeÓ
VAE/Decoder/reshape/ReshapeReshape%VAE/Decoder/decoding/BiasAdd:output:0*VAE/Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Decoder/reshape/ReshapeÙ
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp
VAE/Decoder/conv4_dec/Conv2DConv2D$VAE/Decoder/reshape/Reshape:output:03VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
VAE/Decoder/conv4_dec/Conv2DÏ
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpá
VAE/Decoder/conv4_dec/BiasAddBiasAdd%VAE/Decoder/conv4_dec/Conv2D:output:04VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Decoder/conv4_dec/BiasAdd£
VAE/Decoder/conv4_dec/ReluRelu&VAE/Decoder/conv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Decoder/conv4_dec/Relu
VAE/Decoder/upsamp4/ShapeShape(VAE/Decoder/conv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp4/Shape
'VAE/Decoder/upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'VAE/Decoder/upsamp4/strided_slice/stack 
)VAE/Decoder/upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp4/strided_slice/stack_1 
)VAE/Decoder/upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp4/strided_slice/stack_2Æ
!VAE/Decoder/upsamp4/strided_sliceStridedSlice"VAE/Decoder/upsamp4/Shape:output:00VAE/Decoder/upsamp4/strided_slice/stack:output:02VAE/Decoder/upsamp4/strided_slice/stack_1:output:02VAE/Decoder/upsamp4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!VAE/Decoder/upsamp4/strided_slice
VAE/Decoder/upsamp4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
VAE/Decoder/upsamp4/Const®
VAE/Decoder/upsamp4/mulMul*VAE/Decoder/upsamp4/strided_slice:output:0"VAE/Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp4/mul
0VAE/Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv4_dec/Relu:activations:0VAE/Decoder/upsamp4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(22
0VAE/Decoder/upsamp4/resize/ResizeNearestNeighborØ
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv3_dec/Conv2DConv2DAVAE/Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
VAE/Decoder/conv3_dec/Conv2DÎ
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpà
VAE/Decoder/conv3_dec/BiasAddBiasAdd%VAE/Decoder/conv3_dec/Conv2D:output:04VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
VAE/Decoder/conv3_dec/BiasAdd¢
VAE/Decoder/conv3_dec/ReluRelu&VAE/Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
VAE/Decoder/conv3_dec/Relu
VAE/Decoder/upsamp3/ShapeShape(VAE/Decoder/conv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp3/Shape
'VAE/Decoder/upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'VAE/Decoder/upsamp3/strided_slice/stack 
)VAE/Decoder/upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp3/strided_slice/stack_1 
)VAE/Decoder/upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp3/strided_slice/stack_2Æ
!VAE/Decoder/upsamp3/strided_sliceStridedSlice"VAE/Decoder/upsamp3/Shape:output:00VAE/Decoder/upsamp3/strided_slice/stack:output:02VAE/Decoder/upsamp3/strided_slice/stack_1:output:02VAE/Decoder/upsamp3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!VAE/Decoder/upsamp3/strided_slice
VAE/Decoder/upsamp3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
VAE/Decoder/upsamp3/Const®
VAE/Decoder/upsamp3/mulMul*VAE/Decoder/upsamp3/strided_slice:output:0"VAE/Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp3/mul
0VAE/Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv3_dec/Relu:activations:0VAE/Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(22
0VAE/Decoder/upsamp3/resize/ResizeNearestNeighbor×
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02-
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv2_dec/Conv2DConv2DAVAE/Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
VAE/Decoder/conv2_dec/Conv2DÎ
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpà
VAE/Decoder/conv2_dec/BiasAddBiasAdd%VAE/Decoder/conv2_dec/Conv2D:output:04VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
VAE/Decoder/conv2_dec/BiasAdd¢
VAE/Decoder/conv2_dec/ReluRelu&VAE/Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
VAE/Decoder/conv2_dec/Relu
VAE/Decoder/upsamp2/ShapeShape(VAE/Decoder/conv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp2/Shape
'VAE/Decoder/upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'VAE/Decoder/upsamp2/strided_slice/stack 
)VAE/Decoder/upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp2/strided_slice/stack_1 
)VAE/Decoder/upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp2/strided_slice/stack_2Æ
!VAE/Decoder/upsamp2/strided_sliceStridedSlice"VAE/Decoder/upsamp2/Shape:output:00VAE/Decoder/upsamp2/strided_slice/stack:output:02VAE/Decoder/upsamp2/strided_slice/stack_1:output:02VAE/Decoder/upsamp2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!VAE/Decoder/upsamp2/strided_slice
VAE/Decoder/upsamp2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
VAE/Decoder/upsamp2/Const®
VAE/Decoder/upsamp2/mulMul*VAE/Decoder/upsamp2/strided_slice:output:0"VAE/Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp2/mul
0VAE/Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv2_dec/Relu:activations:0VAE/Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(22
0VAE/Decoder/upsamp2/resize/ResizeNearestNeighbor×
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv1_dec/Conv2DConv2DAVAE/Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
VAE/Decoder/conv1_dec/Conv2DÎ
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpà
VAE/Decoder/conv1_dec/BiasAddBiasAdd%VAE/Decoder/conv1_dec/Conv2D:output:04VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Decoder/conv1_dec/BiasAdd¢
VAE/Decoder/conv1_dec/ReluRelu&VAE/Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Decoder/conv1_dec/Relu
VAE/Decoder/upsamp1/ShapeShape(VAE/Decoder/conv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp1/Shape
'VAE/Decoder/upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'VAE/Decoder/upsamp1/strided_slice/stack 
)VAE/Decoder/upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp1/strided_slice/stack_1 
)VAE/Decoder/upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp1/strided_slice/stack_2Æ
!VAE/Decoder/upsamp1/strided_sliceStridedSlice"VAE/Decoder/upsamp1/Shape:output:00VAE/Decoder/upsamp1/strided_slice/stack:output:02VAE/Decoder/upsamp1/strided_slice/stack_1:output:02VAE/Decoder/upsamp1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!VAE/Decoder/upsamp1/strided_slice
VAE/Decoder/upsamp1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
VAE/Decoder/upsamp1/Const®
VAE/Decoder/upsamp1/mulMul*VAE/Decoder/upsamp1/strided_slice:output:0"VAE/Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp1/mul
0VAE/Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv1_dec/Relu:activations:0VAE/Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(22
0VAE/Decoder/upsamp1/resize/ResizeNearestNeighborÎ
(VAE/Decoder/output/Conv2D/ReadVariableOpReadVariableOp1vae_decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(VAE/Decoder/output/Conv2D/ReadVariableOp
VAE/Decoder/output/Conv2DConv2DAVAE/Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:00VAE/Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
VAE/Decoder/output/Conv2DÅ
)VAE/Decoder/output/BiasAdd/ReadVariableOpReadVariableOp2vae_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)VAE/Decoder/output/BiasAdd/ReadVariableOpÔ
VAE/Decoder/output/BiasAddBiasAdd"VAE/Decoder/output/Conv2D:output:01VAE/Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Decoder/output/BiasAdd¢
VAE/Decoder/output/SigmoidSigmoid#VAE/Decoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
VAE/Decoder/output/Sigmoid§

IdentityIdentityVAE/Decoder/output/Sigmoid:y:0-^VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp,^VAE/Decoder/decoding/BiasAdd/ReadVariableOp+^VAE/Decoder/decoding/MatMul/ReadVariableOp*^VAE/Decoder/output/BiasAdd/ReadVariableOp)^VAE/Decoder/output/Conv2D/ReadVariableOp.^VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp-^VAE/Encoder/bottleneck/MatMul/ReadVariableOp-^VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp-^VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp,^VAE/Encoder/z_log_var/MatMul/ReadVariableOp*^VAE/Encoder/z_mean/BiasAdd/ReadVariableOp)^VAE/Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2\
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp2Z
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp2\
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp2Z
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp2\
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp2Z
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp2\
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp2Z
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp2Z
+VAE/Decoder/decoding/BiasAdd/ReadVariableOp+VAE/Decoder/decoding/BiasAdd/ReadVariableOp2X
*VAE/Decoder/decoding/MatMul/ReadVariableOp*VAE/Decoder/decoding/MatMul/ReadVariableOp2V
)VAE/Decoder/output/BiasAdd/ReadVariableOp)VAE/Decoder/output/BiasAdd/ReadVariableOp2T
(VAE/Decoder/output/Conv2D/ReadVariableOp(VAE/Decoder/output/Conv2D/ReadVariableOp2^
-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp2\
,VAE/Encoder/bottleneck/MatMul/ReadVariableOp,VAE/Encoder/bottleneck/MatMul/ReadVariableOp2\
,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp2Z
+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp2\
,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp2Z
+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp2\
,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp2Z
+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp2\
,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp2Z
+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp2\
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp2Z
+VAE/Encoder/z_log_var/MatMul/ReadVariableOp+VAE/Encoder/z_log_var/MatMul/ReadVariableOp2V
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOp)VAE/Encoder/z_mean/BiasAdd/ReadVariableOp2T
(VAE/Encoder/z_mean/MatMul/ReadVariableOp(VAE/Encoder/z_mean/MatMul/ReadVariableOp:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
1
¶
C__inference_Decoder_layer_call_and_return_conditional_losses_396328

inputs
decoding_396292
decoding_396294
conv4_dec_396298
conv4_dec_396300
conv3_dec_396304
conv3_dec_396306
conv2_dec_396310
conv2_dec_396312
conv1_dec_396316
conv1_dec_396318
output_396322
output_396324
identity¢!conv1_dec/StatefulPartitionedCall¢!conv2_dec/StatefulPartitionedCall¢!conv3_dec/StatefulPartitionedCall¢!conv4_dec/StatefulPartitionedCall¢ decoding/StatefulPartitionedCall¢output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_396292decoding_396294*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_3960692"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3960992
reshape/PartitionedCall¿
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_396298conv4_dec_396300*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_3961182#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_3959922
upsamp4/PartitionedCallÐ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_396304conv3_dec_396306*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_3961462#
!conv3_dec/StatefulPartitionedCall
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_3960112
upsamp3/PartitionedCallÐ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_396310conv2_dec_396312*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_3961742#
!conv2_dec/StatefulPartitionedCall
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_3960302
upsamp2/PartitionedCallÐ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_396316conv1_dec_396318*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_3962022#
!conv1_dec/StatefulPartitionedCall
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_3960492
upsamp1/PartitionedCallÁ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_396322output_396324*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_3962302 
output/StatefulPartitionedCallé
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
ñ
$__inference_VAE_layer_call_fn_396794
input_1
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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_VAE_layer_call_and_return_conditional_losses_3967392
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Á
Û
(__inference_Encoder_layer_call_fn_395895
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

unknown_12
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinput_encoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_3958602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinput_encoder
ù
`
D__inference_maxpool2_layer_call_and_return_conditional_losses_395487

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æõ
 
?__inference_VAE_layer_call_and_return_conditional_losses_397078

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
2encoder_bottleneck_biasadd_readvariableop_resource1
-encoder_z_mean_matmul_readvariableop_resource2
.encoder_z_mean_biasadd_readvariableop_resource4
0encoder_z_log_var_matmul_readvariableop_resource5
1encoder_z_log_var_biasadd_readvariableop_resource3
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
identity¢(Decoder/conv1_dec/BiasAdd/ReadVariableOp¢'Decoder/conv1_dec/Conv2D/ReadVariableOp¢(Decoder/conv2_dec/BiasAdd/ReadVariableOp¢'Decoder/conv2_dec/Conv2D/ReadVariableOp¢(Decoder/conv3_dec/BiasAdd/ReadVariableOp¢'Decoder/conv3_dec/Conv2D/ReadVariableOp¢(Decoder/conv4_dec/BiasAdd/ReadVariableOp¢'Decoder/conv4_dec/Conv2D/ReadVariableOp¢'Decoder/decoding/BiasAdd/ReadVariableOp¢&Decoder/decoding/MatMul/ReadVariableOp¢%Decoder/output/BiasAdd/ReadVariableOp¢$Decoder/output/Conv2D/ReadVariableOp¢)Encoder/bottleneck/BiasAdd/ReadVariableOp¢(Encoder/bottleneck/MatMul/ReadVariableOp¢(Encoder/conv1_enc/BiasAdd/ReadVariableOp¢'Encoder/conv1_enc/Conv2D/ReadVariableOp¢(Encoder/conv2_enc/BiasAdd/ReadVariableOp¢'Encoder/conv2_enc/Conv2D/ReadVariableOp¢(Encoder/conv3_enc/BiasAdd/ReadVariableOp¢'Encoder/conv3_enc/Conv2D/ReadVariableOp¢(Encoder/conv4_enc/BiasAdd/ReadVariableOp¢'Encoder/conv4_enc/Conv2D/ReadVariableOp¢(Encoder/z_log_var/BiasAdd/ReadVariableOp¢'Encoder/z_log_var/MatMul/ReadVariableOp¢%Encoder/z_mean/BiasAdd/ReadVariableOp¢$Encoder/z_mean/MatMul/ReadVariableOpË
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOpÙ
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2DÂ
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOpÐ
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/conv1_enc/BiasAdd
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/conv1_enc/ReluÑ
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPoolË
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOpô
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2DÂ
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOpÐ
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
Encoder/conv2_enc/BiasAdd
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
Encoder/conv2_enc/ReluÑ
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPoolË
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOpô
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2DÂ
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOpÐ
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Encoder/conv3_enc/BiasAdd
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Encoder/conv3_enc/ReluÑ
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPoolÌ
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOpõ
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2DÃ
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOpÑ
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/conv4_enc/BiasAdd
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/conv4_enc/ReluÒ
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB"ÿÿÿÿ   2
Encoder/flatten/Const³
Encoder/flatten/ReshapeReshape!Encoder/maxpool4/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/flatten/ReshapeÇ
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOpÆ
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/bottleneck/MatMulÅ
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOpÍ
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/bottleneck/BiasAddº
$Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Encoder/z_mean/MatMul/ReadVariableOp½
Encoder/z_mean/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0,Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/z_mean/MatMul¹
%Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Encoder/z_mean/BiasAdd/ReadVariableOp½
Encoder/z_mean/BiasAddBiasAddEncoder/z_mean/MatMul:product:0-Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/z_mean/BiasAddÃ
'Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Encoder/z_log_var/MatMul/ReadVariableOpÆ
Encoder/z_log_var/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/z_log_var/MatMulÂ
(Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/z_log_var/BiasAdd/ReadVariableOpÉ
Encoder/z_log_var/BiasAddBiasAdd"Encoder/z_log_var/MatMul:product:00Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/z_log_var/BiasAdd
Encoder/sampling_1/ShapeShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling_1/Shape
&Encoder/sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Encoder/sampling_1/strided_slice/stack
(Encoder/sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_1/strided_slice/stack_1
(Encoder/sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_1/strided_slice/stack_2Ô
 Encoder/sampling_1/strided_sliceStridedSlice!Encoder/sampling_1/Shape:output:0/Encoder/sampling_1/strided_slice/stack:output:01Encoder/sampling_1/strided_slice/stack_1:output:01Encoder/sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Encoder/sampling_1/strided_slice
Encoder/sampling_1/Shape_1ShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling_1/Shape_1
(Encoder/sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_1/strided_slice_1/stack¢
*Encoder/sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*Encoder/sampling_1/strided_slice_1/stack_1¢
*Encoder/sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*Encoder/sampling_1/strided_slice_1/stack_2à
"Encoder/sampling_1/strided_slice_1StridedSlice#Encoder/sampling_1/Shape_1:output:01Encoder/sampling_1/strided_slice_1/stack:output:03Encoder/sampling_1/strided_slice_1/stack_1:output:03Encoder/sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"Encoder/sampling_1/strided_slice_1Þ
&Encoder/sampling_1/random_normal/shapePack)Encoder/sampling_1/strided_slice:output:0+Encoder/sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Encoder/sampling_1/random_normal/shape
%Encoder/sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%Encoder/sampling_1/random_normal/mean
'Encoder/sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'Encoder/sampling_1/random_normal/stddev
5Encoder/sampling_1/random_normal/RandomStandardNormalRandomStandardNormal/Encoder/sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Òè¤27
5Encoder/sampling_1/random_normal/RandomStandardNormal
$Encoder/sampling_1/random_normal/mulMul>Encoder/sampling_1/random_normal/RandomStandardNormal:output:00Encoder/sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$Encoder/sampling_1/random_normal/mulà
 Encoder/sampling_1/random_normalAdd(Encoder/sampling_1/random_normal/mul:z:0.Encoder/sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 Encoder/sampling_1/random_normaly
Encoder/sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Encoder/sampling_1/mul/x°
Encoder/sampling_1/mulMul!Encoder/sampling_1/mul/x:output:0"Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/sampling_1/mul
Encoder/sampling_1/ExpExpEncoder/sampling_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/sampling_1/Exp¯
Encoder/sampling_1/mul_1MulEncoder/sampling_1/Exp:y:0$Encoder/sampling_1/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/sampling_1/mul_1ª
Encoder/sampling_1/addAddV2Encoder/z_mean/BiasAdd:output:0Encoder/sampling_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Encoder/sampling_1/addÁ
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOp»
Decoder/decoding/MatMulMatMulEncoder/sampling_1/add:z:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/decoding/MatMulÀ
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOpÆ
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/decoding/BiasAdd
Decoder/reshape/ShapeShape!Decoder/decoding/BiasAdd:output:0*
T0*
_output_shapes
:2
Decoder/reshape/Shape
#Decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#Decoder/reshape/strided_slice/stack
%Decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/reshape/strided_slice/stack_1
%Decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/reshape/strided_slice/stack_2Â
Decoder/reshape/strided_sliceStridedSliceDecoder/reshape/Shape:output:0,Decoder/reshape/strided_slice/stack:output:0.Decoder/reshape/strided_slice/stack_1:output:0.Decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Decoder/reshape/strided_slice
Decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
Decoder/reshape/Reshape/shape/1
Decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
Decoder/reshape/Reshape/shape/2
Decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2!
Decoder/reshape/Reshape/shape/3
Decoder/reshape/Reshape/shapePack&Decoder/reshape/strided_slice:output:0(Decoder/reshape/Reshape/shape/1:output:0(Decoder/reshape/Reshape/shape/2:output:0(Decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Decoder/reshape/Reshape/shapeÃ
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/reshape/ReshapeÍ
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOpô
Decoder/conv4_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2DÃ
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOpÑ
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/conv4_dec/BiasAdd
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/conv4_dec/Relu
Decoder/upsamp4/ShapeShape$Decoder/conv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp4/Shape
#Decoder/upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp4/strided_slice/stack
%Decoder/upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp4/strided_slice/stack_1
%Decoder/upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp4/strided_slice/stack_2®
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
Decoder/upsamp4/Const
Decoder/upsamp4/mulMul&Decoder/upsamp4/strided_slice:output:0Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp4/mul
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighborÌ
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2DÂ
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOpÐ
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Decoder/conv3_dec/BiasAdd
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Decoder/conv3_dec/Relu
Decoder/upsamp3/ShapeShape$Decoder/conv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp3/Shape
#Decoder/upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp3/strided_slice/stack
%Decoder/upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp3/strided_slice/stack_1
%Decoder/upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp3/strided_slice/stack_2®
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
Decoder/upsamp3/Const
Decoder/upsamp3/mulMul&Decoder/upsamp3/strided_slice:output:0Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp3/mul
,Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv3_dec/Relu:activations:0Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighborË
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2DÂ
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOpÐ
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Decoder/conv2_dec/BiasAdd
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Decoder/conv2_dec/Relu
Decoder/upsamp2/ShapeShape$Decoder/conv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp2/Shape
#Decoder/upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp2/strided_slice/stack
%Decoder/upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp2/strided_slice/stack_1
%Decoder/upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp2/strided_slice/stack_2®
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
Decoder/upsamp2/Const
Decoder/upsamp2/mulMul&Decoder/upsamp2/strided_slice:output:0Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp2/mul
,Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv2_dec/Relu:activations:0Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighborË
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2DÂ
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOpÐ
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/conv1_dec/BiasAdd
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/conv1_dec/Relu
Decoder/upsamp1/ShapeShape$Decoder/conv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp1/Shape
#Decoder/upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp1/strided_slice/stack
%Decoder/upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp1/strided_slice/stack_1
%Decoder/upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp1/strided_slice/stack_2®
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
Decoder/upsamp1/Const
Decoder/upsamp1/mulMul&Decoder/upsamp1/strided_slice:output:0Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp1/mul
,Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv1_dec/Relu:activations:0Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighborÂ
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Decoder/output/Conv2D¹
%Decoder/output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Decoder/output/BiasAdd/ReadVariableOpÄ
Decoder/output/BiasAddBiasAddDecoder/output/Conv2D:output:0-Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/output/BiasAdd
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Decoder/output/Sigmoid»	
IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp)^Encoder/z_log_var/BiasAdd/ReadVariableOp(^Encoder/z_log_var/MatMul/ReadVariableOp&^Encoder/z_mean/BiasAdd/ReadVariableOp%^Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2T
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
'Encoder/conv4_enc/Conv2D/ReadVariableOp'Encoder/conv4_enc/Conv2D/ReadVariableOp2T
(Encoder/z_log_var/BiasAdd/ReadVariableOp(Encoder/z_log_var/BiasAdd/ReadVariableOp2R
'Encoder/z_log_var/MatMul/ReadVariableOp'Encoder/z_log_var/MatMul/ReadVariableOp2N
%Encoder/z_mean/BiasAdd/ReadVariableOp%Encoder/z_mean/BiasAdd/ReadVariableOp2L
$Encoder/z_mean/MatMul/ReadVariableOp$Encoder/z_mean/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ªA
Ø
C__inference_Encoder_layer_call_and_return_conditional_losses_395763
input_encoder
conv1_enc_395543
conv1_enc_395545
conv2_enc_395571
conv2_enc_395573
conv3_enc_395599
conv3_enc_395601
conv4_enc_395627
conv4_enc_395629
bottleneck_395668
bottleneck_395670
z_mean_395694
z_mean_395696
z_log_var_395720
z_log_var_395722
identity

identity_1

identity_2¢"bottleneck/StatefulPartitionedCall¢!conv1_enc/StatefulPartitionedCall¢!conv2_enc/StatefulPartitionedCall¢!conv3_enc/StatefulPartitionedCall¢!conv4_enc/StatefulPartitionedCall¢"sampling_1/StatefulPartitionedCall¢!z_log_var/StatefulPartitionedCall¢z_mean/StatefulPartitionedCall«
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_395543conv1_enc_395545*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_3955322#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_3954752
maxpool1/PartitionedCall¿
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_395571conv2_enc_395573*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_3955602#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_3954872
maxpool2/PartitionedCall¿
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_395599conv3_enc_395601*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_3955882#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_3954992
maxpool3/PartitionedCallÀ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_395627conv4_enc_395629*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_3956162#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_3955112
maxpool4/PartitionedCallð
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3956392
flatten/PartitionedCall»
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_395668bottleneck_395670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_3956572$
"bottleneck/StatefulPartitionedCall²
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_395694z_mean_395696*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_3956832 
z_mean/StatefulPartitionedCallÁ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_395720z_log_var_395722*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_3957092#
!z_log_var/StatefulPartitionedCallÃ
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_3957512$
"sampling_1/StatefulPartitionedCall
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¡

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1¢

Identity_2Identity+sampling_1/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinput_encoder
	
ß
F__inference_bottleneck_layer_call_and_return_conditional_losses_397917

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2_enc_layer_call_fn_397856

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_3955602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ

::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
¾
Þ
E__inference_conv3_dec_layer_call_and_return_conditional_losses_398065

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
_
C__inference_reshape_layer_call_and_return_conditional_losses_398029

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
strided_slice/stack_2â
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
B :2
Reshape/shape/3º
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô	

(__inference_Decoder_layer_call_fn_397816

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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_3963962
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
(__inference_Encoder_layer_call_fn_397547

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

unknown_12
identity

identity_1

identity_2¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_3958602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

Þ
E__inference_conv1_enc_layer_call_and_return_conditional_losses_395532

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Û
B__inference_output_layer_call_and_return_conditional_losses_396230

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
Sigmoidª
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë

*__inference_conv3_dec_layer_call_fn_398074

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_3961462
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

s
F__inference_sampling_1_layer_call_and_return_conditional_losses_395751

inputs
inputs_1
identityD
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevä
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ÆÃ82$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
E
)__inference_maxpool1_layer_call_fn_395481

inputs
identityè
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_3954752
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
D
(__inference_upsamp1_layer_call_fn_396055

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_3960492
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

*__inference_conv2_dec_layer_call_fn_398094

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_3961742
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ªA
Ø
C__inference_Encoder_layer_call_and_return_conditional_losses_395810
input_encoder
conv1_enc_395766
conv1_enc_395768
conv2_enc_395772
conv2_enc_395774
conv3_enc_395778
conv3_enc_395780
conv4_enc_395784
conv4_enc_395786
bottleneck_395791
bottleneck_395793
z_mean_395796
z_mean_395798
z_log_var_395801
z_log_var_395803
identity

identity_1

identity_2¢"bottleneck/StatefulPartitionedCall¢!conv1_enc/StatefulPartitionedCall¢!conv2_enc/StatefulPartitionedCall¢!conv3_enc/StatefulPartitionedCall¢!conv4_enc/StatefulPartitionedCall¢"sampling_1/StatefulPartitionedCall¢!z_log_var/StatefulPartitionedCall¢z_mean/StatefulPartitionedCall«
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_395766conv1_enc_395768*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_3955322#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_3954752
maxpool1/PartitionedCall¿
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_395772conv2_enc_395774*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_3955602#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_3954872
maxpool2/PartitionedCall¿
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_395778conv3_enc_395780*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_3955882#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_3954992
maxpool3/PartitionedCallÀ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_395784conv4_enc_395786*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_3956162#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_3955112
maxpool4/PartitionedCallð
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3956392
flatten/PartitionedCall»
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_395791bottleneck_395793*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_3956572$
"bottleneck/StatefulPartitionedCall²
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_395796z_mean_395798*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_3956832 
z_mean/StatefulPartitionedCallÁ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_395801z_log_var_395803*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_3957092#
!z_log_var/StatefulPartitionedCallÃ
"sampling_1/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_sampling_1_layer_call_and_return_conditional_losses_3957512$
"sampling_1/StatefulPartitionedCall
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¡

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1¢

Identity_2Identity+sampling_1/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_1/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2H
"sampling_1/StatefulPartitionedCall"sampling_1/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:^ Z
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinput_encoder


*__inference_conv3_enc_layer_call_fn_397876

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_3955882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
ß
F__inference_bottleneck_layer_call_and_return_conditional_losses_395657

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

*__inference_conv1_dec_layer_call_fn_398114

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_3962022
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
½
_
C__inference_flatten_layer_call_and_return_conditional_losses_397902

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø

Þ
E__inference_conv4_dec_layer_call_and_return_conditional_losses_398045

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

Þ
E__inference_conv3_enc_layer_call_and_return_conditional_losses_397867

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
üo
è	
C__inference_Encoder_layer_call_and_return_conditional_losses_397431

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
*bottleneck_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2¢!bottleneck/BiasAdd/ReadVariableOp¢ bottleneck/MatMul/ReadVariableOp¢ conv1_enc/BiasAdd/ReadVariableOp¢conv1_enc/Conv2D/ReadVariableOp¢ conv2_enc/BiasAdd/ReadVariableOp¢conv2_enc/Conv2D/ReadVariableOp¢ conv3_enc/BiasAdd/ReadVariableOp¢conv3_enc/Conv2D/ReadVariableOp¢ conv4_enc/BiasAdd/ReadVariableOp¢conv4_enc/Conv2D/ReadVariableOp¢ z_log_var/BiasAdd/ReadVariableOp¢z_log_var/MatMul/ReadVariableOp¢z_mean/BiasAdd/ReadVariableOp¢z_mean/MatMul/ReadVariableOp³
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOpÁ
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1_enc/Conv2Dª
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOp°
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1_enc/Relu¹
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool³
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_enc/Conv2D/ReadVariableOpÔ
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingSAME*
strides
2
conv2_enc/Conv2Dª
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_enc/BiasAdd/ReadVariableOp°
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
conv2_enc/Relu¹
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPool³
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv3_enc/Conv2D/ReadVariableOpÔ
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv3_enc/Conv2Dª
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_enc/BiasAdd/ReadVariableOp°
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv3_enc/Relu¹
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPool´
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv4_enc/Conv2D/ReadVariableOpÕ
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv4_enc/Conv2D«
 conv4_enc/BiasAdd/ReadVariableOpReadVariableOp)conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv4_enc/BiasAdd/ReadVariableOp±
conv4_enc/BiasAddBiasAddconv4_enc/Conv2D:output:0(conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv4_enc/BiasAdd
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv4_enc/Reluº
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapemaxpool4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¯
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 bottleneck/MatMul/ReadVariableOp¦
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bottleneck/MatMul­
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp­
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
bottleneck/BiasAdd¢
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMulbottleneck/BiasAdd:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
z_mean/MatMul¡
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
z_mean/BiasAdd«
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
z_log_var/MatMul/ReadVariableOp¦
z_log_var/MatMulMatMulbottleneck/BiasAdd:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
z_log_var/MatMulª
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp©
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
z_log_var/BiasAddk
sampling_1/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape
sampling_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
sampling_1/strided_slice/stack
 sampling_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_1
 sampling_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice/stack_2¤
sampling_1/strided_sliceStridedSlicesampling_1/Shape:output:0'sampling_1/strided_slice/stack:output:0)sampling_1/strided_slice/stack_1:output:0)sampling_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_sliceo
sampling_1/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_1/Shape_1
 sampling_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_1/strided_slice_1/stack
"sampling_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_1
"sampling_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_1/strided_slice_1/stack_2°
sampling_1/strided_slice_1StridedSlicesampling_1/Shape_1:output:0)sampling_1/strided_slice_1/stack:output:0+sampling_1/strided_slice_1/stack_1:output:0+sampling_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_1/strided_slice_1¾
sampling_1/random_normal/shapePack!sampling_1/strided_slice:output:0#sampling_1/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2 
sampling_1/random_normal/shape
sampling_1/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling_1/random_normal/mean
sampling_1/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
sampling_1/random_normal/stddev
-sampling_1/random_normal/RandomStandardNormalRandomStandardNormal'sampling_1/random_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2Äí¬2/
-sampling_1/random_normal/RandomStandardNormalà
sampling_1/random_normal/mulMul6sampling_1/random_normal/RandomStandardNormal:output:0(sampling_1/random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sampling_1/random_normal/mulÀ
sampling_1/random_normalAdd sampling_1/random_normal/mul:z:0&sampling_1/random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sampling_1/random_normali
sampling_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling_1/mul/x
sampling_1/mulMulsampling_1/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sampling_1/mulm
sampling_1/ExpExpsampling_1/mul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sampling_1/Exp
sampling_1/mul_1Mulsampling_1/Exp:y:0sampling_1/random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sampling_1/mul_1
sampling_1/addAddV2z_mean/BiasAdd:output:0sampling_1/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sampling_1/addÊ
IdentityIdentityz_mean/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÑ

Identity_1Identityz_log_var/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1É

Identity_2Identitysampling_1/add:z:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::2F
!bottleneck/BiasAdd/ReadVariableOp!bottleneck/BiasAdd/ReadVariableOp2D
 bottleneck/MatMul/ReadVariableOp bottleneck/MatMul/ReadVariableOp2D
 conv1_enc/BiasAdd/ReadVariableOp conv1_enc/BiasAdd/ReadVariableOp2B
conv1_enc/Conv2D/ReadVariableOpconv1_enc/Conv2D/ReadVariableOp2D
 conv2_enc/BiasAdd/ReadVariableOp conv2_enc/BiasAdd/ReadVariableOp2B
conv2_enc/Conv2D/ReadVariableOpconv2_enc/Conv2D/ReadVariableOp2D
 conv3_enc/BiasAdd/ReadVariableOp conv3_enc/BiasAdd/ReadVariableOp2B
conv3_enc/Conv2D/ReadVariableOpconv3_enc/Conv2D/ReadVariableOp2D
 conv4_enc/BiasAdd/ReadVariableOp conv4_enc/BiasAdd/ReadVariableOp2B
conv4_enc/Conv2D/ReadVariableOpconv4_enc/Conv2D/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é	

(__inference_Decoder_layer_call_fn_396355
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_decoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_3963282
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinput_decoder
ù
`
D__inference_maxpool1_layer_call_and_return_conditional_losses_395475

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv1_enc_layer_call_fn_397836

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_3955322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Þ
E__inference_conv1_dec_layer_call_and_return_conditional_losses_396202

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
	
Þ
E__inference_z_log_var_layer_call_and_return_conditional_losses_397955

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
«
?__inference_VAE_layer_call_and_return_conditional_losses_396676
input_1
encoder_396619
encoder_396621
encoder_396623
encoder_396625
encoder_396627
encoder_396629
encoder_396631
encoder_396633
encoder_396635
encoder_396637
encoder_396639
encoder_396641
encoder_396643
encoder_396645
decoder_396650
decoder_396652
decoder_396654
decoder_396656
decoder_396658
decoder_396660
decoder_396662
decoder_396664
decoder_396666
decoder_396668
decoder_396670
decoder_396672
identity¢Decoder/StatefulPartitionedCall¢Encoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_396619encoder_396621encoder_396623encoder_396625encoder_396627encoder_396629encoder_396631encoder_396633encoder_396635encoder_396637encoder_396639encoder_396641encoder_396643encoder_396645*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_3959442!
Encoder/StatefulPartitionedCall
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_396650decoder_396652decoder_396654decoder_396656decoder_396658decoder_396660decoder_396662decoder_396664decoder_396666decoder_396668decoder_396670decoder_396672*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_3963962!
Decoder/StatefulPartitionedCallÚ
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

u
F__inference_sampling_1_layer_call_and_return_conditional_losses_397990
inputs_0
inputs_1
identityF
ShapeShapeinputs_0*
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
random_normal/shapePackstrided_slice:output:0strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
random_normal/shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_normal/stddevä
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
seed±ÿå)*
seed2ó_2$
"random_normal/RandomStandardNormal´
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_normalS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x]
mulMulmul/x:output:0inputs_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
 
D
(__inference_upsamp4_layer_call_fn_395998

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_3959922
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

Þ
E__inference_conv2_enc_layer_call_and_return_conditional_losses_397847

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ

::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


 
_user_specified_nameinputs
Õ

Þ
E__inference_conv4_enc_layer_call_and_return_conditional_losses_397887

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¢
E
)__inference_maxpool4_layer_call_fn_395517

inputs
identityè
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_3955112
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
Þ
E__inference_conv1_dec_layer_call_and_return_conditional_losses_398105

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Éo

C__inference_Decoder_layer_call_and_return_conditional_losses_397758

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
identity¢ conv1_dec/BiasAdd/ReadVariableOp¢conv1_dec/Conv2D/ReadVariableOp¢ conv2_dec/BiasAdd/ReadVariableOp¢conv2_dec/Conv2D/ReadVariableOp¢ conv3_dec/BiasAdd/ReadVariableOp¢conv3_dec/Conv2D/ReadVariableOp¢ conv4_dec/BiasAdd/ReadVariableOp¢conv4_dec/Conv2D/ReadVariableOp¢decoding/BiasAdd/ReadVariableOp¢decoding/MatMul/ReadVariableOp¢output/BiasAdd/ReadVariableOp¢output/Conv2D/ReadVariableOp©
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
decoding/MatMul/ReadVariableOp
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
decoding/MatMul¨
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
decoding/BiasAdd/ReadVariableOp¦
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
decoding/BiasAddg
reshape/ShapeShapedecoding/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
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
B :2
reshape/Reshape/shape/3ê
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape£
reshape/ReshapeReshapedecoding/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/Reshapeµ
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv4_dec/Conv2D/ReadVariableOpÔ
conv4_dec/Conv2DConv2Dreshape/Reshape:output:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv4_dec/Conv2D«
 conv4_dec/BiasAdd/ReadVariableOpReadVariableOp)conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv4_dec/BiasAdd/ReadVariableOp±
conv4_dec/BiasAddBiasAddconv4_dec/Conv2D:output:0(conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv4_dec/BiasAdd
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv4_dec/Reluj
upsamp4/ShapeShapeconv4_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp4/Shape
upsamp4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack
upsamp4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack_1
upsamp4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp4/strided_slice/stack_2þ
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
upsamp4/mulé
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighbor´
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv3_dec/Conv2D/ReadVariableOpð
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv3_dec/Conv2Dª
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_dec/BiasAdd/ReadVariableOp°
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv3_dec/Reluj
upsamp3/ShapeShapeconv3_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp3/Shape
upsamp3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack
upsamp3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack_1
upsamp3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp3/strided_slice/stack_2þ
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
upsamp3/mulè
$upsamp3/resize/ResizeNearestNeighborResizeNearestNeighborconv3_dec/Relu:activations:0upsamp3/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighbor³
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2_dec/Conv2D/ReadVariableOpð
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2_dec/Conv2Dª
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_dec/BiasAdd/ReadVariableOp°
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2_dec/Reluj
upsamp2/ShapeShapeconv2_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp2/Shape
upsamp2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack
upsamp2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack_1
upsamp2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp2/strided_slice/stack_2þ
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
upsamp2/mulè
$upsamp2/resize/ResizeNearestNeighborResizeNearestNeighborconv2_dec/Relu:activations:0upsamp2/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighbor³
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv1_dec/Conv2D/ReadVariableOpð
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv1_dec/Conv2Dª
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOp°
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv1_dec/Reluj
upsamp1/ShapeShapeconv1_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp1/Shape
upsamp1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack
upsamp1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack_1
upsamp1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp1/strided_slice/stack_2þ
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
upsamp1/mulè
$upsamp1/resize/ResizeNearestNeighborResizeNearestNeighborconv1_dec/Relu:activations:0upsamp1/mul:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighborª
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
output/Conv2D/ReadVariableOpè
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
2
output/Conv2D¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp¤
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/Sigmoid
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::2D
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
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
«
?__inference_VAE_layer_call_and_return_conditional_losses_396616
input_1
encoder_396501
encoder_396503
encoder_396505
encoder_396507
encoder_396509
encoder_396511
encoder_396513
encoder_396515
encoder_396517
encoder_396519
encoder_396521
encoder_396523
encoder_396525
encoder_396527
decoder_396590
decoder_396592
decoder_396594
decoder_396596
decoder_396598
decoder_396600
decoder_396602
decoder_396604
decoder_396606
decoder_396608
decoder_396610
decoder_396612
identity¢Decoder/StatefulPartitionedCall¢Encoder/StatefulPartitionedCall
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_396501encoder_396503encoder_396505encoder_396507encoder_396509encoder_396511encoder_396513encoder_396515encoder_396517encoder_396519encoder_396521encoder_396523encoder_396525encoder_396527*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_3958602!
Encoder/StatefulPartitionedCall
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_396590decoder_396592decoder_396594decoder_396596decoder_396598decoder_396600decoder_396602decoder_396604decoder_396606decoder_396608decoder_396610decoder_396612*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_3963282!
Decoder/StatefulPartitionedCallÚ
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ø

Þ
E__inference_conv4_dec_layer_call_and_return_conditional_losses_396118

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
ñ
$__inference_signature_wrapper_396918
input_1
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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_3954692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¬
Ô
(__inference_Encoder_layer_call_fn_397584

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

unknown_12
identity

identity_1

identity_2¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_3959442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:ÿÿÿÿÿÿÿÿÿ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
D
(__inference_upsamp3_layer_call_fn_396017

inputs
identityç
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_3960112
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á

*__inference_z_log_var_layer_call_fn_397964

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_3957092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô	

(__inference_Decoder_layer_call_fn_397787

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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_3963282
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢
E
)__inference_maxpool2_layer_call_fn_395493

inputs
identityè
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_3954872
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv4_dec_layer_call_fn_398054

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_3961182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

Þ
E__inference_conv1_enc_layer_call_and_return_conditional_losses_397827

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
Ý
D__inference_decoding_layer_call_and_return_conditional_losses_398006

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
Þ
E__inference_conv3_dec_layer_call_and_return_conditional_losses_396146

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpµ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ

+__inference_bottleneck_layer_call_fn_397926

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_3956572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç¨
 $
__inference__traced_save_398424
file_prefix$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop(
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
*savev2_bottleneck_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop.
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
&savev2_output_bias_read_readvariableop6
2savev2_adam_conv1_enc_kernel_m_read_readvariableop4
0savev2_adam_conv1_enc_bias_m_read_readvariableop6
2savev2_adam_conv2_enc_kernel_m_read_readvariableop4
0savev2_adam_conv2_enc_bias_m_read_readvariableop6
2savev2_adam_conv3_enc_kernel_m_read_readvariableop4
0savev2_adam_conv3_enc_bias_m_read_readvariableop6
2savev2_adam_conv4_enc_kernel_m_read_readvariableop4
0savev2_adam_conv4_enc_bias_m_read_readvariableop7
3savev2_adam_bottleneck_kernel_m_read_readvariableop5
1savev2_adam_bottleneck_bias_m_read_readvariableop3
/savev2_adam_z_mean_kernel_m_read_readvariableop1
-savev2_adam_z_mean_bias_m_read_readvariableop6
2savev2_adam_z_log_var_kernel_m_read_readvariableop4
0savev2_adam_z_log_var_bias_m_read_readvariableop5
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
1savev2_adam_bottleneck_bias_v_read_readvariableop3
/savev2_adam_z_mean_kernel_v_read_readvariableop1
-savev2_adam_z_mean_bias_v_read_readvariableop6
2savev2_adam_z_log_var_kernel_v_read_readvariableop4
0savev2_adam_z_log_var_bias_v_read_readvariableop5
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

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameì/
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*þ.
valueô.Bñ.ZB3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¿
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*É
value¿B¼ZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesã"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv1_enc_kernel_read_readvariableop)savev2_conv1_enc_bias_read_readvariableop+savev2_conv2_enc_kernel_read_readvariableop)savev2_conv2_enc_bias_read_readvariableop+savev2_conv3_enc_kernel_read_readvariableop)savev2_conv3_enc_bias_read_readvariableop+savev2_conv4_enc_kernel_read_readvariableop)savev2_conv4_enc_bias_read_readvariableop,savev2_bottleneck_kernel_read_readvariableop*savev2_bottleneck_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableop*savev2_decoding_kernel_read_readvariableop(savev2_decoding_bias_read_readvariableop+savev2_conv4_dec_kernel_read_readvariableop)savev2_conv4_dec_bias_read_readvariableop+savev2_conv3_dec_kernel_read_readvariableop)savev2_conv3_dec_bias_read_readvariableop+savev2_conv2_dec_kernel_read_readvariableop)savev2_conv2_dec_bias_read_readvariableop+savev2_conv1_dec_kernel_read_readvariableop)savev2_conv1_dec_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop2savev2_adam_conv1_enc_kernel_m_read_readvariableop0savev2_adam_conv1_enc_bias_m_read_readvariableop2savev2_adam_conv2_enc_kernel_m_read_readvariableop0savev2_adam_conv2_enc_bias_m_read_readvariableop2savev2_adam_conv3_enc_kernel_m_read_readvariableop0savev2_adam_conv3_enc_bias_m_read_readvariableop2savev2_adam_conv4_enc_kernel_m_read_readvariableop0savev2_adam_conv4_enc_bias_m_read_readvariableop3savev2_adam_bottleneck_kernel_m_read_readvariableop1savev2_adam_bottleneck_bias_m_read_readvariableop/savev2_adam_z_mean_kernel_m_read_readvariableop-savev2_adam_z_mean_bias_m_read_readvariableop2savev2_adam_z_log_var_kernel_m_read_readvariableop0savev2_adam_z_log_var_bias_m_read_readvariableop1savev2_adam_decoding_kernel_m_read_readvariableop/savev2_adam_decoding_bias_m_read_readvariableop2savev2_adam_conv4_dec_kernel_m_read_readvariableop0savev2_adam_conv4_dec_bias_m_read_readvariableop2savev2_adam_conv3_dec_kernel_m_read_readvariableop0savev2_adam_conv3_dec_bias_m_read_readvariableop2savev2_adam_conv2_dec_kernel_m_read_readvariableop0savev2_adam_conv2_dec_bias_m_read_readvariableop2savev2_adam_conv1_dec_kernel_m_read_readvariableop0savev2_adam_conv1_dec_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv1_enc_kernel_v_read_readvariableop0savev2_adam_conv1_enc_bias_v_read_readvariableop2savev2_adam_conv2_enc_kernel_v_read_readvariableop0savev2_adam_conv2_enc_bias_v_read_readvariableop2savev2_adam_conv3_enc_kernel_v_read_readvariableop0savev2_adam_conv3_enc_bias_v_read_readvariableop2savev2_adam_conv4_enc_kernel_v_read_readvariableop0savev2_adam_conv4_enc_bias_v_read_readvariableop3savev2_adam_bottleneck_kernel_v_read_readvariableop1savev2_adam_bottleneck_bias_v_read_readvariableop/savev2_adam_z_mean_kernel_v_read_readvariableop-savev2_adam_z_mean_bias_v_read_readvariableop2savev2_adam_z_log_var_kernel_v_read_readvariableop0savev2_adam_z_log_var_bias_v_read_readvariableop1savev2_adam_decoding_kernel_v_read_readvariableop/savev2_adam_decoding_bias_v_read_readvariableop2savev2_adam_conv4_dec_kernel_v_read_readvariableop0savev2_adam_conv4_dec_bias_v_read_readvariableop2savev2_adam_conv3_dec_kernel_v_read_readvariableop0savev2_adam_conv3_dec_bias_v_read_readvariableop2savev2_adam_conv2_dec_kernel_v_read_readvariableop0savev2_adam_conv2_dec_bias_v_read_readvariableop2savev2_adam_conv1_dec_kernel_v_read_readvariableop0savev2_adam_conv1_dec_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *h
dtypes^
\2Z	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapes
ý: : : : : : : : : : : : ::: : : @:@:@::	::::::	::::@:@:@ : : :::::: : : @:@:@::	::::::	::::@:@:@ : : :::::: : : @:@:@::	::::::	::::@:@:@ : : :::: 2(
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@:, (
&
_output_shapes
:@ : !

_output_shapes
: :,"(
&
_output_shapes
: : #

_output_shapes
::,$(
&
_output_shapes
:: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
: : )

_output_shapes
: :,*(
&
_output_shapes
: @: +

_output_shapes
:@:-,)
'
_output_shapes
:@:!-

_output_shapes	
::%.!

_output_shapes
:	: /

_output_shapes
::$0 

_output_shapes

:: 1

_output_shapes
::$2 

_output_shapes

:: 3

_output_shapes
::%4!

_output_shapes
:	:!5

_output_shapes	
::.6*
(
_output_shapes
::!7

_output_shapes	
::-8)
'
_output_shapes
:@: 9

_output_shapes
:@:,:(
&
_output_shapes
:@ : ;

_output_shapes
: :,<(
&
_output_shapes
: : =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
: : C

_output_shapes
: :,D(
&
_output_shapes
: @: E

_output_shapes
:@:-F)
'
_output_shapes
:@:!G

_output_shapes	
::%H!

_output_shapes
:	: I

_output_shapes
::$J 

_output_shapes

:: K

_output_shapes
::$L 

_output_shapes

:: M

_output_shapes
::%N!

_output_shapes
:	:!O

_output_shapes	
::.P*
(
_output_shapes
::!Q

_output_shapes	
::-R)
'
_output_shapes
:@: S

_output_shapes
:@:,T(
&
_output_shapes
:@ : U

_output_shapes
: :,V(
&
_output_shapes
: : W

_output_shapes
::,X(
&
_output_shapes
:: Y

_output_shapes
::Z

_output_shapes
: 
§
D
(__inference_reshape_layer_call_fn_398034

inputs
identityÍ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3960992
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
ð
$__inference_VAE_layer_call_fn_397295

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

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCallÓ
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
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_VAE_layer_call_and_return_conditional_losses_3967392
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é	

(__inference_Decoder_layer_call_fn_396423
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_decoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_3963962
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameinput_decoder"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*»
serving_default§
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿD
output_18
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:§
ý
encoder
decoder
total_loss_tracker
reconstruction_loss_tracker
kl_loss_tracker
	optimizer
loss
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
õ_default_save_signature
+ö&call_and_return_all_conditional_losses
÷__call__"½
_tf_keras_model£{"class_name": "VAEInternal", "name": "VAE", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "VAEInternal"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Óo
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
layer-8
layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
trainable_variables
regularization_losses
	variables
	keras_api
+ø&call_and_return_all_conditional_losses
ù__call__"Òk
_tf_keras_network¶k{"class_name": "Functional", "name": "Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling_1", "trainable": true, "dtype": "float32"}, "name": "sampling_1", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling_1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling_1", "trainable": true, "dtype": "float32"}, "name": "sampling_1", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling_1", 0, 0]]}}}
¿d
layer-0
 layer_with_weights-0
 layer-1
!layer-2
"layer_with_weights-1
"layer-3
#layer-4
$layer_with_weights-2
$layer-5
%layer-6
&layer_with_weights-3
&layer-7
'layer-8
(layer_with_weights-4
(layer-9
)layer-10
*layer_with_weights-5
*layer-11
+trainable_variables
,regularization_losses
-	variables
.	keras_api
+ú&call_and_return_all_conditional_losses
û__call__"ô`
_tf_keras_networkØ`{"class_name": "Functional", "name": "Decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [13, 13]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 16]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [13, 13]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}}}
Ç
	/total
	0count
1	variables
2	keras_api"
_tf_keras_metricv{"class_name": "Mean", "name": "total_loss", "dtype": "float32", "config": {"name": "total_loss", "dtype": "float32"}}
Ú
	3total
	4count
5	variables
6	keras_api"£
_tf_keras_metric{"class_name": "Mean", "name": "reconstruction_loss", "dtype": "float32", "config": {"name": "reconstruction_loss", "dtype": "float32"}}
Á
	7total
	8count
9	variables
:	keras_api"
_tf_keras_metricp{"class_name": "Mean", "name": "kl_loss", "dtype": "float32", "config": {"name": "kl_loss", "dtype": "float32"}}
Û
;iter

<beta_1

=beta_2
	>decay
?learning_rate@mÁAmÂBmÃCmÄDmÅEmÆFmÇGmÈHmÉImÊJmËKmÌLmÍMmÎNmÏOmÐPmÑQmÒRmÓSmÔTmÕUmÖVm×WmØXmÙYmÚ@vÛAvÜBvÝCvÞDvßEvàFváGvâHvãIväJvåKvæLvçMvèNvéOvêPvëQvìRvíSvîTvïUvðVvñWvòXvóYvô"
	optimizer
 "
trackable_dict_wrapper
æ
@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15
P16
Q17
R18
S19
T20
U21
V22
W23
X24
Y25"
trackable_list_wrapper
 "
trackable_list_wrapper

@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13
N14
O15
P16
Q17
R18
S19
T20
U21
V22
W23
X24
Y25
/26
027
328
429
730
831"
trackable_list_wrapper
Î
Zlayer_metrics
[metrics
trainable_variables
\layer_regularization_losses

]layers
	regularization_losses
^non_trainable_variables

	variables
÷__call__
õ_default_save_signature
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
-
üserving_default"
signature_map
"
_tf_keras_input_layerâ{"class_name": "InputLayer", "name": "input_encoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 20, 20, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}}
ô	

@kernel
Abias
_trainable_variables
`regularization_losses
a	variables
b	keras_api
+ý&call_and_return_all_conditional_losses
þ__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv1_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 1]}}
ò
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
+ÿ&call_and_return_all_conditional_losses
__call__"á
_tf_keras_layerÇ{"class_name": "MaxPooling2D", "name": "maxpool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ö	

Bkernel
Cbias
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
+&call_and_return_all_conditional_losses
__call__"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv2_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 16]}}
ò
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
+&call_and_return_all_conditional_losses
__call__"á
_tf_keras_layerÇ{"class_name": "MaxPooling2D", "name": "maxpool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ô	

Dkernel
Ebias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv3_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 32]}}
ò
strainable_variables
tregularization_losses
u	variables
v	keras_api
+&call_and_return_all_conditional_losses
__call__"á
_tf_keras_layerÇ{"class_name": "MaxPooling2D", "name": "maxpool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
õ	

Fkernel
Gbias
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv4_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 64]}}
ò
{trainable_variables
|regularization_losses
}	variables
~	keras_api
+&call_and_return_all_conditional_losses
__call__"á
_tf_keras_layerÇ{"class_name": "MaxPooling2D", "name": "maxpool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ç
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


Hkernel
Ibias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Õ
_tf_keras_layer»{"class_name": "Dense", "name": "bottleneck", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
ö

Jkernel
Kbias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ë
_tf_keras_layer±{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ü

Lkernel
Mbias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
¿
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ª
_tf_keras_layer{"class_name": "Sampling", "name": "sampling_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "sampling_1", "trainable": true, "dtype": "float32"}}

@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13"
trackable_list_wrapper
 "
trackable_list_wrapper

@0
A1
B2
C3
D4
E5
F6
G7
H8
I9
J10
K11
L12
M13"
trackable_list_wrapper
µ
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables
ù__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
÷"ô
_tf_keras_input_layerÔ{"class_name": "InputLayer", "name": "input_decoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}}
û

Nkernel
Obias
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "decoding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
û
trainable_variables
regularization_losses
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"æ
_tf_keras_layerÌ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 2, 128]}}}
û	

Pkernel
Qbias
 trainable_variables
¡regularization_losses
¢	variables
£	keras_api
+&call_and_return_all_conditional_losses
__call__"Ð
_tf_keras_layer¶{"class_name": "Conv2D", "name": "conv4_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 2, 128]}}
¿
¤trainable_variables
¥regularization_losses
¦	variables
§	keras_api
+&call_and_return_all_conditional_losses
__call__"ª
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ú	

Rkernel
Sbias
¨trainable_variables
©regularization_losses
ª	variables
«	keras_api
+&call_and_return_all_conditional_losses
 __call__"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv3_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 128]}}
¿
¬trainable_variables
­regularization_losses
®	variables
¯	keras_api
+¡&call_and_return_all_conditional_losses
¢__call__"ª
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ø	

Tkernel
Ubias
°trainable_variables
±regularization_losses
²	variables
³	keras_api
+£&call_and_return_all_conditional_losses
¤__call__"Í
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
¿
´trainable_variables
µregularization_losses
¶	variables
·	keras_api
+¥&call_and_return_all_conditional_losses
¦__call__"ª
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ú	

Vkernel
Wbias
¸trainable_variables
¹regularization_losses
º	variables
»	keras_api
+§&call_and_return_all_conditional_losses
¨__call__"Ï
_tf_keras_layerµ{"class_name": "Conv2D", "name": "conv1_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
¿
¼trainable_variables
½regularization_losses
¾	variables
¿	keras_api
+©&call_and_return_all_conditional_losses
ª__call__"ª
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ù	

Xkernel
Ybias
Àtrainable_variables
Áregularization_losses
Â	variables
Ã	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [13, 13]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 16]}}
v
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11"
trackable_list_wrapper
 "
trackable_list_wrapper
v
N0
O1
P2
Q3
R4
S5
T6
U7
V8
W9
X10
Y11"
trackable_list_wrapper
µ
Älayer_metrics
Åmetrics
+trainable_variables
 Ælayer_regularization_losses
Çlayers
,regularization_losses
Ènon_trainable_variables
-	variables
û__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
:  (2total
:  (2count
.
/0
01"
trackable_list_wrapper
-
1	variables"
_generic_user_object
:  (2total
:  (2count
.
30
41"
trackable_list_wrapper
-
5	variables"
_generic_user_object
:  (2total
:  (2count
.
70
81"
trackable_list_wrapper
-
9	variables"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(2conv1_enc/kernel
:2conv1_enc/bias
*:( 2conv2_enc/kernel
: 2conv2_enc/bias
*:( @2conv3_enc/kernel
:@2conv3_enc/bias
+:)@2conv4_enc/kernel
:2conv4_enc/bias
$:"	2bottleneck/kernel
:2bottleneck/bias
:2z_mean/kernel
:2z_mean/bias
": 2z_log_var/kernel
:2z_log_var/bias
": 	2decoding/kernel
:2decoding/bias
,:*2conv4_dec/kernel
:2conv4_dec/bias
+:)@2conv3_dec/kernel
:@2conv3_dec/bias
*:(@ 2conv2_dec/kernel
: 2conv2_dec/bias
*:( 2conv1_dec/kernel
:2conv1_dec/bias
':%2output/kernel
:2output/bias
V

total_loss
reconstruction_loss
kl_loss"
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
J
/0
01
32
43
74
85"
trackable_list_wrapper
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
µ
Élayer_metrics
Êmetrics
_trainable_variables
 Ëlayer_regularization_losses
Ìlayers
`regularization_losses
Ínon_trainable_variables
a	variables
þ__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Îlayer_metrics
Ïmetrics
ctrainable_variables
 Ðlayer_regularization_losses
Ñlayers
dregularization_losses
Ònon_trainable_variables
e	variables
__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
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
µ
Ólayer_metrics
Ômetrics
gtrainable_variables
 Õlayer_regularization_losses
Ölayers
hregularization_losses
×non_trainable_variables
i	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ølayer_metrics
Ùmetrics
ktrainable_variables
 Úlayer_regularization_losses
Ûlayers
lregularization_losses
Ünon_trainable_variables
m	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
µ
Ýlayer_metrics
Þmetrics
otrainable_variables
 ßlayer_regularization_losses
àlayers
pregularization_losses
ánon_trainable_variables
q	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
âlayer_metrics
ãmetrics
strainable_variables
 älayer_regularization_losses
ålayers
tregularization_losses
ænon_trainable_variables
u	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
µ
çlayer_metrics
èmetrics
wtrainable_variables
 élayer_regularization_losses
êlayers
xregularization_losses
ënon_trainable_variables
y	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ìlayer_metrics
ímetrics
{trainable_variables
 îlayer_regularization_losses
ïlayers
|regularization_losses
ðnon_trainable_variables
}	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
·
ñlayer_metrics
òmetrics
trainable_variables
 ólayer_regularization_losses
ôlayers
regularization_losses
õnon_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
¸
ölayer_metrics
÷metrics
trainable_variables
 ølayer_regularization_losses
ùlayers
regularization_losses
únon_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
¸
ûlayer_metrics
ümetrics
trainable_variables
 ýlayer_regularization_losses
þlayers
regularization_losses
ÿnon_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
¸
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
¸
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
metrics
trainable_variables
 layer_regularization_losses
layers
regularization_losses
non_trainable_variables
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
¸
layer_metrics
metrics
 trainable_variables
 layer_regularization_losses
layers
¡regularization_losses
non_trainable_variables
¢	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layer_metrics
metrics
¤trainable_variables
 layer_regularization_losses
layers
¥regularization_losses
non_trainable_variables
¦	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
¸
layer_metrics
metrics
¨trainable_variables
  layer_regularization_losses
¡layers
©regularization_losses
¢non_trainable_variables
ª	variables
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
£layer_metrics
¤metrics
¬trainable_variables
 ¥layer_regularization_losses
¦layers
­regularization_losses
§non_trainable_variables
®	variables
¢__call__
+¡&call_and_return_all_conditional_losses
'¡"call_and_return_conditional_losses"
_generic_user_object
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
¸
¨layer_metrics
©metrics
°trainable_variables
 ªlayer_regularization_losses
«layers
±regularization_losses
¬non_trainable_variables
²	variables
¤__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
­layer_metrics
®metrics
´trainable_variables
 ¯layer_regularization_losses
°layers
µregularization_losses
±non_trainable_variables
¶	variables
¦__call__
+¥&call_and_return_all_conditional_losses
'¥"call_and_return_conditional_losses"
_generic_user_object
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
¸
²layer_metrics
³metrics
¸trainable_variables
 ´layer_regularization_losses
µlayers
¹regularization_losses
¶non_trainable_variables
º	variables
¨__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
·layer_metrics
¸metrics
¼trainable_variables
 ¹layer_regularization_losses
ºlayers
½regularization_losses
»non_trainable_variables
¾	variables
ª__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
¸
¼layer_metrics
½metrics
Àtrainable_variables
 ¾layer_regularization_losses
¿layers
Áregularization_losses
Ànon_trainable_variables
Â	variables
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
 1
!2
"3
#4
$5
%6
&7
'8
(9
)10
*11"
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
/:-2Adam/conv1_enc/kernel/m
!:2Adam/conv1_enc/bias/m
/:- 2Adam/conv2_enc/kernel/m
!: 2Adam/conv2_enc/bias/m
/:- @2Adam/conv3_enc/kernel/m
!:@2Adam/conv3_enc/bias/m
0:.@2Adam/conv4_enc/kernel/m
": 2Adam/conv4_enc/bias/m
):'	2Adam/bottleneck/kernel/m
": 2Adam/bottleneck/bias/m
$:"2Adam/z_mean/kernel/m
:2Adam/z_mean/bias/m
':%2Adam/z_log_var/kernel/m
!:2Adam/z_log_var/bias/m
':%	2Adam/decoding/kernel/m
!:2Adam/decoding/bias/m
1:/2Adam/conv4_dec/kernel/m
": 2Adam/conv4_dec/bias/m
0:.@2Adam/conv3_dec/kernel/m
!:@2Adam/conv3_dec/bias/m
/:-@ 2Adam/conv2_dec/kernel/m
!: 2Adam/conv2_dec/bias/m
/:- 2Adam/conv1_dec/kernel/m
!:2Adam/conv1_dec/bias/m
,:*2Adam/output/kernel/m
:2Adam/output/bias/m
/:-2Adam/conv1_enc/kernel/v
!:2Adam/conv1_enc/bias/v
/:- 2Adam/conv2_enc/kernel/v
!: 2Adam/conv2_enc/bias/v
/:- @2Adam/conv3_enc/kernel/v
!:@2Adam/conv3_enc/bias/v
0:.@2Adam/conv4_enc/kernel/v
": 2Adam/conv4_enc/bias/v
):'	2Adam/bottleneck/kernel/v
": 2Adam/bottleneck/bias/v
$:"2Adam/z_mean/kernel/v
:2Adam/z_mean/bias/v
':%2Adam/z_log_var/kernel/v
!:2Adam/z_log_var/bias/v
':%	2Adam/decoding/kernel/v
!:2Adam/decoding/bias/v
1:/2Adam/conv4_dec/kernel/v
": 2Adam/conv4_dec/bias/v
0:.@2Adam/conv3_dec/kernel/v
!:@2Adam/conv3_dec/bias/v
/:-@ 2Adam/conv2_dec/kernel/v
!: 2Adam/conv2_dec/bias/v
/:- 2Adam/conv1_dec/kernel/v
!:2Adam/conv1_dec/bias/v
,:*2Adam/output/kernel/v
:2Adam/output/bias/v
ç2ä
!__inference__wrapped_model_395469¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
½2º
?__inference_VAE_layer_call_and_return_conditional_losses_396616
?__inference_VAE_layer_call_and_return_conditional_losses_396676
?__inference_VAE_layer_call_and_return_conditional_losses_397238
?__inference_VAE_layer_call_and_return_conditional_losses_397078³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
$__inference_VAE_layer_call_fn_397352
$__inference_VAE_layer_call_fn_397295
$__inference_VAE_layer_call_fn_396794
$__inference_VAE_layer_call_fn_396851³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ú2×
C__inference_Encoder_layer_call_and_return_conditional_losses_397431
C__inference_Encoder_layer_call_and_return_conditional_losses_395810
C__inference_Encoder_layer_call_and_return_conditional_losses_397510
C__inference_Encoder_layer_call_and_return_conditional_losses_395763À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
(__inference_Encoder_layer_call_fn_397584
(__inference_Encoder_layer_call_fn_395895
(__inference_Encoder_layer_call_fn_395979
(__inference_Encoder_layer_call_fn_397547À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ú2×
C__inference_Decoder_layer_call_and_return_conditional_losses_397671
C__inference_Decoder_layer_call_and_return_conditional_losses_397758
C__inference_Decoder_layer_call_and_return_conditional_losses_396247
C__inference_Decoder_layer_call_and_return_conditional_losses_396286À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
(__inference_Decoder_layer_call_fn_396423
(__inference_Decoder_layer_call_fn_396355
(__inference_Decoder_layer_call_fn_397787
(__inference_Decoder_layer_call_fn_397816À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ËBÈ
$__inference_signature_wrapper_396918input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv1_enc_layer_call_and_return_conditional_losses_397827¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv1_enc_layer_call_fn_397836¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬2©
D__inference_maxpool1_layer_call_and_return_conditional_losses_395475à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
)__inference_maxpool1_layer_call_fn_395481à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv2_enc_layer_call_and_return_conditional_losses_397847¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2_enc_layer_call_fn_397856¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬2©
D__inference_maxpool2_layer_call_and_return_conditional_losses_395487à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
)__inference_maxpool2_layer_call_fn_395493à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv3_enc_layer_call_and_return_conditional_losses_397867¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv3_enc_layer_call_fn_397876¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬2©
D__inference_maxpool3_layer_call_and_return_conditional_losses_395499à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
)__inference_maxpool3_layer_call_fn_395505à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv4_enc_layer_call_and_return_conditional_losses_397887¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv4_enc_layer_call_fn_397896¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬2©
D__inference_maxpool4_layer_call_and_return_conditional_losses_395511à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
)__inference_maxpool4_layer_call_fn_395517à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_397902¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_flatten_layer_call_fn_397907¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_bottleneck_layer_call_and_return_conditional_losses_397917¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_bottleneck_layer_call_fn_397926¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_z_mean_layer_call_and_return_conditional_losses_397936¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_z_mean_layer_call_fn_397945¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_z_log_var_layer_call_and_return_conditional_losses_397955¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_z_log_var_layer_call_fn_397964¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð2í
F__inference_sampling_1_layer_call_and_return_conditional_losses_397990¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_sampling_1_layer_call_fn_397996¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_decoding_layer_call_and_return_conditional_losses_398006¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_decoding_layer_call_fn_398015¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_reshape_layer_call_and_return_conditional_losses_398029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_reshape_layer_call_fn_398034¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ï2ì
E__inference_conv4_dec_layer_call_and_return_conditional_losses_398045¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv4_dec_layer_call_fn_398054¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
«2¨
C__inference_upsamp4_layer_call_and_return_conditional_losses_395992à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
(__inference_upsamp4_layer_call_fn_395998à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv3_dec_layer_call_and_return_conditional_losses_398065¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv3_dec_layer_call_fn_398074¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
«2¨
C__inference_upsamp3_layer_call_and_return_conditional_losses_396011à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
(__inference_upsamp3_layer_call_fn_396017à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv2_dec_layer_call_and_return_conditional_losses_398085¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv2_dec_layer_call_fn_398094¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
«2¨
C__inference_upsamp2_layer_call_and_return_conditional_losses_396030à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
(__inference_upsamp2_layer_call_fn_396036à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ï2ì
E__inference_conv1_dec_layer_call_and_return_conditional_losses_398105¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ô2Ñ
*__inference_conv1_dec_layer_call_fn_398114¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
«2¨
C__inference_upsamp1_layer_call_and_return_conditional_losses_396049à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
(__inference_upsamp1_layer_call_fn_396055à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ì2é
B__inference_output_layer_call_and_return_conditional_losses_398125¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_output_layer_call_fn_398134¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ×
C__inference_Decoder_layer_call_and_return_conditional_losses_396247NOPQRSTUVWXY>¢;
4¢1
'$
input_decoderÿÿÿÿÿÿÿÿÿ
p

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ×
C__inference_Decoder_layer_call_and_return_conditional_losses_396286NOPQRSTUVWXY>¢;
4¢1
'$
input_decoderÿÿÿÿÿÿÿÿÿ
p 

 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ½
C__inference_Decoder_layer_call_and_return_conditional_losses_397671vNOPQRSTUVWXY7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ½
C__inference_Decoder_layer_call_and_return_conditional_losses_397758vNOPQRSTUVWXY7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ¯
(__inference_Decoder_layer_call_fn_396355NOPQRSTUVWXY>¢;
4¢1
'$
input_decoderÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¯
(__inference_Decoder_layer_call_fn_396423NOPQRSTUVWXY>¢;
4¢1
'$
input_decoderÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
(__inference_Decoder_layer_call_fn_397787{NOPQRSTUVWXY7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ§
(__inference_Decoder_layer_call_fn_397816{NOPQRSTUVWXY7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
C__inference_Encoder_layer_call_and_return_conditional_losses_395763Ä@ABCDEFGHIJKLMF¢C
<¢9
/,
input_encoderÿÿÿÿÿÿÿÿÿ
p

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
C__inference_Encoder_layer_call_and_return_conditional_losses_395810Ä@ABCDEFGHIJKLMF¢C
<¢9
/,
input_encoderÿÿÿÿÿÿÿÿÿ
p 

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
C__inference_Encoder_layer_call_and_return_conditional_losses_397431½@ABCDEFGHIJKLM?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 
C__inference_Encoder_layer_call_and_return_conditional_losses_397510½@ABCDEFGHIJKLM?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "j¢g
`]

0/0ÿÿÿÿÿÿÿÿÿ

0/1ÿÿÿÿÿÿÿÿÿ

0/2ÿÿÿÿÿÿÿÿÿ
 á
(__inference_Encoder_layer_call_fn_395895´@ABCDEFGHIJKLMF¢C
<¢9
/,
input_encoderÿÿÿÿÿÿÿÿÿ
p

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿá
(__inference_Encoder_layer_call_fn_395979´@ABCDEFGHIJKLMF¢C
<¢9
/,
input_encoderÿÿÿÿÿÿÿÿÿ
p 

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿÚ
(__inference_Encoder_layer_call_fn_397547­@ABCDEFGHIJKLM?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿÚ
(__inference_Encoder_layer_call_fn_397584­@ABCDEFGHIJKLM?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ZW

0ÿÿÿÿÿÿÿÿÿ

1ÿÿÿÿÿÿÿÿÿ

2ÿÿÿÿÿÿÿÿÿß
?__inference_VAE_layer_call_and_return_conditional_losses_396616@ABCDEFGHIJKLMNOPQRSTUVWXY<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ß
?__inference_VAE_layer_call_and_return_conditional_losses_396676@ABCDEFGHIJKLMNOPQRSTUVWXY<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ì
?__inference_VAE_layer_call_and_return_conditional_losses_397078@ABCDEFGHIJKLMNOPQRSTUVWXY;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ì
?__inference_VAE_layer_call_and_return_conditional_losses_397238@ABCDEFGHIJKLMNOPQRSTUVWXY;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ·
$__inference_VAE_layer_call_fn_396794@ABCDEFGHIJKLMNOPQRSTUVWXY<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
$__inference_VAE_layer_call_fn_396851@ABCDEFGHIJKLMNOPQRSTUVWXY<¢9
2¢/
)&
input_1ÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¶
$__inference_VAE_layer_call_fn_397295@ABCDEFGHIJKLMNOPQRSTUVWXY;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¶
$__inference_VAE_layer_call_fn_397352@ABCDEFGHIJKLMNOPQRSTUVWXY;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
!__inference__wrapped_model_395469@ABCDEFGHIJKLMNOPQRSTUVWXY8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
output_1*'
output_1ÿÿÿÿÿÿÿÿÿ§
F__inference_bottleneck_layer_call_and_return_conditional_losses_397917]HI0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_bottleneck_layer_call_fn_397926PHI0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÚ
E__inference_conv1_dec_layer_call_and_return_conditional_losses_398105VWI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ²
*__inference_conv1_dec_layer_call_fn_398114VWI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿµ
E__inference_conv1_enc_layer_call_and_return_conditional_losses_397827l@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv1_enc_layer_call_fn_397836_@A7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿÚ
E__inference_conv2_dec_layer_call_and_return_conditional_losses_398085TUI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ²
*__inference_conv2_dec_layer_call_fn_398094TUI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ µ
E__inference_conv2_enc_layer_call_and_return_conditional_losses_397847lBC7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ


ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ

 
 
*__inference_conv2_enc_layer_call_fn_397856_BC7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ


ª " ÿÿÿÿÿÿÿÿÿ

 Û
E__inference_conv3_dec_layer_call_and_return_conditional_losses_398065RSJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ³
*__inference_conv3_dec_layer_call_fn_398074RSJ¢G
@¢=
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@µ
E__inference_conv3_enc_layer_call_and_return_conditional_losses_397867lDE7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv3_enc_layer_call_fn_397876_DE7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ@·
E__inference_conv4_dec_layer_call_and_return_conditional_losses_398045nPQ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv4_dec_layer_call_fn_398054aPQ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¶
E__inference_conv4_enc_layer_call_and_return_conditional_losses_397887mFG7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_conv4_enc_layer_call_fn_397896`FG7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ¥
D__inference_decoding_layer_call_and_return_conditional_losses_398006]NO/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
)__inference_decoding_layer_call_fn_398015PNO/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
C__inference_flatten_layer_call_and_return_conditional_losses_397902b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_flatten_layer_call_fn_397907U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿç
D__inference_maxpool1_layer_call_and_return_conditional_losses_395475R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
)__inference_maxpool1_layer_call_fn_395481R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
D__inference_maxpool2_layer_call_and_return_conditional_losses_395487R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
)__inference_maxpool2_layer_call_fn_395493R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
D__inference_maxpool3_layer_call_and_return_conditional_losses_395499R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
)__inference_maxpool3_layer_call_fn_395505R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
D__inference_maxpool4_layer_call_and_return_conditional_losses_395511R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
)__inference_maxpool4_layer_call_fn_395517R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ×
B__inference_output_layer_call_and_return_conditional_losses_398125XYI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¯
'__inference_output_layer_call_fn_398134XYI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
C__inference_reshape_layer_call_and_return_conditional_losses_398029b0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
(__inference_reshape_layer_call_fn_398034U0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿÎ
F__inference_sampling_1_layer_call_and_return_conditional_losses_397990Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
+__inference_sampling_1_layer_call_fn_397996vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÇ
$__inference_signature_wrapper_396918@ABCDEFGHIJKLMNOPQRSTUVWXYC¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ";ª8
6
output_1*'
output_1ÿÿÿÿÿÿÿÿÿæ
C__inference_upsamp1_layer_call_and_return_conditional_losses_396049R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
(__inference_upsamp1_layer_call_fn_396055R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
C__inference_upsamp2_layer_call_and_return_conditional_losses_396030R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
(__inference_upsamp2_layer_call_fn_396036R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
C__inference_upsamp3_layer_call_and_return_conditional_losses_396011R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
(__inference_upsamp3_layer_call_fn_396017R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿæ
C__inference_upsamp4_layer_call_and_return_conditional_losses_395992R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
(__inference_upsamp4_layer_call_fn_395998R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
E__inference_z_log_var_layer_call_and_return_conditional_losses_397955\LM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
*__inference_z_log_var_layer_call_fn_397964OLM/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¢
B__inference_z_mean_layer_call_and_return_conditional_losses_397936\JK/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_z_mean_layer_call_fn_397945OJK/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ