μό 
Ό
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
Ύ
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
φ
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8Έ
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

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

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

conv4_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv4_enc/kernel
~
$conv4_enc/kernel/Read/ReadVariableOpReadVariableOpconv4_enc/kernel*'
_output_shapes
:@*
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
shape:		*"
shared_namebottleneck/kernel
x
%bottleneck/kernel/Read/ReadVariableOpReadVariableOpbottleneck/kernel*
_output_shapes
:		*
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
shape:		* 
shared_namedecoding/kernel
t
#decoding/kernel/Read/ReadVariableOpReadVariableOpdecoding/kernel*
_output_shapes
:		*
dtype0
s
decoding/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedecoding/bias
l
!decoding/bias/Read/ReadVariableOpReadVariableOpdecoding/bias*
_output_shapes	
:	*
dtype0

conv4_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv4_dec/kernel

$conv4_dec/kernel/Read/ReadVariableOpReadVariableOpconv4_dec/kernel*(
_output_shapes
:*
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
shape:@*!
shared_nameconv3_dec/kernel
~
$conv3_dec/kernel/Read/ReadVariableOpReadVariableOpconv3_dec/kernel*'
_output_shapes
:@*
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

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
shape:		*
shared_nameoutput/kernel
w
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*&
_output_shapes
:		*
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
shape:*(
shared_nameAdam/conv1_enc/kernel/m

+Adam/conv1_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/kernel/m*&
_output_shapes
:*
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
shape: *(
shared_nameAdam/conv2_enc/kernel/m

+Adam/conv2_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/kernel/m*&
_output_shapes
: *
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
shape: @*(
shared_nameAdam/conv3_enc/kernel/m

+Adam/conv3_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/kernel/m*&
_output_shapes
: @*
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
shape:@*(
shared_nameAdam/conv4_enc/kernel/m

+Adam/conv4_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/kernel/m*'
_output_shapes
:@*
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
shape:		*)
shared_nameAdam/bottleneck/kernel/m

,Adam/bottleneck/kernel/m/Read/ReadVariableOpReadVariableOpAdam/bottleneck/kernel/m*
_output_shapes
:		*
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
shape:		*'
shared_nameAdam/decoding/kernel/m

*Adam/decoding/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoding/kernel/m*
_output_shapes
:		*
dtype0

Adam/decoding/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/decoding/bias/m
z
(Adam/decoding/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoding/bias/m*
_output_shapes	
:	*
dtype0

Adam/conv4_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv4_dec/kernel/m

+Adam/conv4_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/kernel/m*(
_output_shapes
:*
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
shape:@*(
shared_nameAdam/conv3_dec/kernel/m

+Adam/conv3_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/kernel/m*'
_output_shapes
:@*
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
shape:@ *(
shared_nameAdam/conv2_dec/kernel/m

+Adam/conv2_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/kernel/m*&
_output_shapes
:@ *
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
shape: *(
shared_nameAdam/conv1_dec/kernel/m

+Adam/conv1_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/kernel/m*&
_output_shapes
: *
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
shape:		*%
shared_nameAdam/output/kernel/m

(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*&
_output_shapes
:		*
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
shape:*(
shared_nameAdam/conv1_enc/kernel/v

+Adam/conv1_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/kernel/v*&
_output_shapes
:*
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
shape: *(
shared_nameAdam/conv2_enc/kernel/v

+Adam/conv2_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/kernel/v*&
_output_shapes
: *
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
shape: @*(
shared_nameAdam/conv3_enc/kernel/v

+Adam/conv3_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/kernel/v*&
_output_shapes
: @*
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
shape:@*(
shared_nameAdam/conv4_enc/kernel/v

+Adam/conv4_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/kernel/v*'
_output_shapes
:@*
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
shape:		*)
shared_nameAdam/bottleneck/kernel/v

,Adam/bottleneck/kernel/v/Read/ReadVariableOpReadVariableOpAdam/bottleneck/kernel/v*
_output_shapes
:		*
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
shape:		*'
shared_nameAdam/decoding/kernel/v

*Adam/decoding/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoding/kernel/v*
_output_shapes
:		*
dtype0

Adam/decoding/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/decoding/bias/v
z
(Adam/decoding/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoding/bias/v*
_output_shapes	
:	*
dtype0

Adam/conv4_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv4_dec/kernel/v

+Adam/conv4_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/kernel/v*(
_output_shapes
:*
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
shape:@*(
shared_nameAdam/conv3_dec/kernel/v

+Adam/conv3_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/kernel/v*'
_output_shapes
:@*
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
shape:@ *(
shared_nameAdam/conv2_dec/kernel/v

+Adam/conv2_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/kernel/v*&
_output_shapes
:@ *
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
shape: *(
shared_nameAdam/conv1_dec/kernel/v

+Adam/conv1_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/kernel/v*&
_output_shapes
: *
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
shape:		*%
shared_nameAdam/output/kernel/v

(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*&
_output_shapes
:		*
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

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ψ
valueΝBΙ BΑ
γ
encoder
decoder
total_loss_tracker
reconstruction_loss_tracker
kl_loss_tracker
	optimizer
loss
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
Β
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
	variables
trainable_variables
regularization_losses
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
+	variables
,trainable_variables
-regularization_losses
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
Θ
;iter

<beta_1

=beta_2
	>decay
?learning_rate@mΑAmΒBmΓCmΔDmΕEmΖFmΗGmΘHmΙImΚJmΛKmΜLmΝMmΞNmΟOmΠPmΡQm?RmΣSmΤTmΥUmΦVmΧWmΨXmΩYmΪ@vΫAvάBvέCvήDvίEvΰFvαGvβHvγIvδJvεKvζLvηMvθNvιOvκPvλQvμRvνSvξTvοUvπVvρWvςXvσYvτ
 
φ
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
Ζ
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
­
	variables

Zlayers
	trainable_variables
[non_trainable_variables
\metrics

regularization_losses
]layer_metrics
^layer_regularization_losses
 
 
h

@kernel
Abias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
R
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

Bkernel
Cbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
R
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
h

Dkernel
Ebias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
R
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
h

Fkernel
Gbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
R
{	variables
|trainable_variables
}regularization_losses
~	keras_api
U
	variables
trainable_variables
regularization_losses
	keras_api
l

Hkernel
Ibias
	variables
trainable_variables
regularization_losses
	keras_api
l

Jkernel
Kbias
	variables
trainable_variables
regularization_losses
	keras_api
l

Lkernel
Mbias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
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
²
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses
 
l

Nkernel
Obias
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
l

Pkernel
Qbias
 	variables
‘trainable_variables
’regularization_losses
£	keras_api
V
€	variables
₯trainable_variables
¦regularization_losses
§	keras_api
l

Rkernel
Sbias
¨	variables
©trainable_variables
ͺregularization_losses
«	keras_api
V
¬	variables
­trainable_variables
?regularization_losses
―	keras_api
l

Tkernel
Ubias
°	variables
±trainable_variables
²regularization_losses
³	keras_api
V
΄	variables
΅trainable_variables
Άregularization_losses
·	keras_api
l

Vkernel
Wbias
Έ	variables
Ήtrainable_variables
Ίregularization_losses
»	keras_api
V
Ό	variables
½trainable_variables
Ύregularization_losses
Ώ	keras_api
l

Xkernel
Ybias
ΐ	variables
Αtrainable_variables
Βregularization_losses
Γ	keras_api
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
²
+	variables
Δlayers
,trainable_variables
Εnon_trainable_variables
Ζmetrics
-regularization_losses
Ηlayer_metrics
 Θlayer_regularization_losses
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
LJ
VARIABLE_VALUEconv1_enc/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv1_enc/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2_enc/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2_enc/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv3_enc/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv3_enc/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv4_enc/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv4_enc/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEbottleneck/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEbottleneck/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEz_mean/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEz_mean/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEz_log_var/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEz_log_var/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEdecoding/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdecoding/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv4_dec/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv4_dec/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv3_dec/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv3_dec/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv2_dec/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2_dec/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEconv1_dec/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv1_dec/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEoutput/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEoutput/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
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

0
1
2
6

total_loss
reconstruction_loss
kl_loss
 

@0
A1

@0
A1
 
²
_	variables
Ιlayers
`trainable_variables
Κnon_trainable_variables
Λmetrics
aregularization_losses
Μlayer_metrics
 Νlayer_regularization_losses
 
 
 
²
c	variables
Ξlayers
dtrainable_variables
Οnon_trainable_variables
Πmetrics
eregularization_losses
Ρlayer_metrics
 ?layer_regularization_losses

B0
C1

B0
C1
 
²
g	variables
Σlayers
htrainable_variables
Τnon_trainable_variables
Υmetrics
iregularization_losses
Φlayer_metrics
 Χlayer_regularization_losses
 
 
 
²
k	variables
Ψlayers
ltrainable_variables
Ωnon_trainable_variables
Ϊmetrics
mregularization_losses
Ϋlayer_metrics
 άlayer_regularization_losses

D0
E1

D0
E1
 
²
o	variables
έlayers
ptrainable_variables
ήnon_trainable_variables
ίmetrics
qregularization_losses
ΰlayer_metrics
 αlayer_regularization_losses
 
 
 
²
s	variables
βlayers
ttrainable_variables
γnon_trainable_variables
δmetrics
uregularization_losses
εlayer_metrics
 ζlayer_regularization_losses

F0
G1

F0
G1
 
²
w	variables
ηlayers
xtrainable_variables
θnon_trainable_variables
ιmetrics
yregularization_losses
κlayer_metrics
 λlayer_regularization_losses
 
 
 
²
{	variables
μlayers
|trainable_variables
νnon_trainable_variables
ξmetrics
}regularization_losses
οlayer_metrics
 πlayer_regularization_losses
 
 
 
΄
	variables
ρlayers
trainable_variables
ςnon_trainable_variables
σmetrics
regularization_losses
τlayer_metrics
 υlayer_regularization_losses

H0
I1

H0
I1
 
΅
	variables
φlayers
trainable_variables
χnon_trainable_variables
ψmetrics
regularization_losses
ωlayer_metrics
 ϊlayer_regularization_losses

J0
K1

J0
K1
 
΅
	variables
ϋlayers
trainable_variables
όnon_trainable_variables
ύmetrics
regularization_losses
ώlayer_metrics
 ?layer_regularization_losses

L0
M1

L0
M1
 
΅
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses
 
 
 
΅
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses
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
 
 
 

N0
O1

N0
O1
 
΅
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses
 
 
 
΅
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses

P0
Q1

P0
Q1
 
΅
 	variables
layers
‘trainable_variables
non_trainable_variables
metrics
’regularization_losses
layer_metrics
 layer_regularization_losses
 
 
 
΅
€	variables
layers
₯trainable_variables
non_trainable_variables
metrics
¦regularization_losses
layer_metrics
 layer_regularization_losses

R0
S1

R0
S1
 
΅
¨	variables
layers
©trainable_variables
non_trainable_variables
 metrics
ͺregularization_losses
‘layer_metrics
 ’layer_regularization_losses
 
 
 
΅
¬	variables
£layers
­trainable_variables
€non_trainable_variables
₯metrics
?regularization_losses
¦layer_metrics
 §layer_regularization_losses

T0
U1

T0
U1
 
΅
°	variables
¨layers
±trainable_variables
©non_trainable_variables
ͺmetrics
²regularization_losses
«layer_metrics
 ¬layer_regularization_losses
 
 
 
΅
΄	variables
­layers
΅trainable_variables
?non_trainable_variables
―metrics
Άregularization_losses
°layer_metrics
 ±layer_regularization_losses

V0
W1

V0
W1
 
΅
Έ	variables
²layers
Ήtrainable_variables
³non_trainable_variables
΄metrics
Ίregularization_losses
΅layer_metrics
 Άlayer_regularization_losses
 
 
 
΅
Ό	variables
·layers
½trainable_variables
Έnon_trainable_variables
Ήmetrics
Ύregularization_losses
Ίlayer_metrics
 »layer_regularization_losses

X0
Y1

X0
Y1
 
΅
ΐ	variables
Όlayers
Αtrainable_variables
½non_trainable_variables
Ύmetrics
Βregularization_losses
Ώlayer_metrics
 ΐlayer_regularization_losses
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
 
 
 
om
VARIABLE_VALUEAdam/conv1_enc/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv1_enc/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2_enc/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2_enc/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv3_enc/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv3_enc/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv4_enc/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv4_enc/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bottleneck/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/bottleneck/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/z_mean/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/z_mean/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/z_log_var/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/z_log_var/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/decoding/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/decoding/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv4_dec/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv4_dec/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv3_dec/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv3_dec/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2_dec/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2_dec/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv1_dec/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1_dec/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/output/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/output/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv1_enc/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv1_enc/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2_enc/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2_enc/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv3_enc/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv3_enc/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv4_enc/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv4_enc/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bottleneck/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/bottleneck/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/z_mean/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/z_mean/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/z_log_var/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/z_log_var/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/decoding/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/decoding/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv4_dec/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv4_dec/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv3_dec/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv3_dec/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv2_dec/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2_dec/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/conv1_dec/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv1_dec/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/output/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/output/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:?????????((*
dtype0*$
shape:?????????((
 
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1_enc/kernelconv1_enc/biasconv2_enc/kernelconv2_enc/biasconv3_enc/kernelconv3_enc/biasconv4_enc/kernelconv4_enc/biasbottleneck/kernelbottleneck/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/biasdecoding/kerneldecoding/biasconv4_dec/kernelconv4_dec/biasconv3_dec/kernelconv3_dec/biasconv2_dec/kernelconv2_dec/biasconv1_dec/kernelconv1_dec/biasoutput/kerneloutput/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????((*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1231182
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ω
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_1232688
ΰ
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1232965υ³
€
F
*__inference_maxpool4_layer_call_fn_1229781

inputs
identityι
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
GPU2*0J 8 *N
fIRG
E__inference_maxpool4_layer_call_and_return_conditional_losses_12297752
PartitionedCall
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
	
ή
E__inference_decoding_layer_call_and_return_conditional_losses_1230333

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????	2

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
€
F
*__inference_maxpool3_layer_call_fn_1229769

inputs
identityι
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
GPU2*0J 8 *N
fIRG
E__inference_maxpool3_layer_call_and_return_conditional_losses_12297632
PartitionedCall
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
€
F
*__inference_maxpool2_layer_call_fn_1229757

inputs
identityι
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
GPU2*0J 8 *N
fIRG
E__inference_maxpool2_layer_call_and_return_conditional_losses_12297512
PartitionedCall
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
	
ί
F__inference_z_log_var_layer_call_and_return_conditional_losses_1232219

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
ΐ1
Κ
D__inference_Decoder_layer_call_and_return_conditional_losses_1230550
input_decoder
decoding_1230514
decoding_1230516
conv4_dec_1230520
conv4_dec_1230522
conv3_dec_1230526
conv3_dec_1230528
conv2_dec_1230532
conv2_dec_1230534
conv1_dec_1230538
conv1_dec_1230540
output_1230544
output_1230546
identity’!conv1_dec/StatefulPartitionedCall’!conv2_dec/StatefulPartitionedCall’!conv3_dec/StatefulPartitionedCall’!conv4_dec/StatefulPartitionedCall’ decoding/StatefulPartitionedCall’output/StatefulPartitionedCall’
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_1230514decoding_1230516*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoding_layer_call_and_return_conditional_losses_12303332"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_12303632
reshape/PartitionedCallΒ
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_1230520conv4_dec_1230522*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_dec_layer_call_and_return_conditional_losses_12303822#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_upsamp4_layer_call_and_return_conditional_losses_12302562
upsamp4/PartitionedCallΣ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_1230526conv3_dec_1230528*
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
GPU2*0J 8 *O
fJRH
F__inference_conv3_dec_layer_call_and_return_conditional_losses_12304102#
!conv3_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp3_layer_call_and_return_conditional_losses_12302752
upsamp3/PartitionedCallΣ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_1230532conv2_dec_1230534*
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
GPU2*0J 8 *O
fJRH
F__inference_conv2_dec_layer_call_and_return_conditional_losses_12304382#
!conv2_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp2_layer_call_and_return_conditional_losses_12302942
upsamp2/PartitionedCallΣ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_1230538conv1_dec_1230540*
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
GPU2*0J 8 *O
fJRH
F__inference_conv1_dec_layer_call_and_return_conditional_losses_12304662#
!conv1_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp1_layer_call_and_return_conditional_losses_12303132
upsamp1/PartitionedCallΔ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_1230544output_1230546*
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
GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_12304942 
output/StatefulPartitionedCallι
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
ϊ
a
E__inference_maxpool4_layer_call_and_return_conditional_losses_1229775

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool
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
Π

ί
F__inference_conv1_enc_layer_call_and_return_conditional_losses_1229796

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
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
:?????????((2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????((2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????((::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs

`
D__inference_upsamp2_layer_call_and_return_conditional_losses_1230294

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
strided_slice/stack_2Ξ
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
mulΥ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor€
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


+__inference_conv1_enc_layer_call_fn_1232100

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????((*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_enc_layer_call_and_return_conditional_losses_12297962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????((2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????((::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs


+__inference_conv4_enc_layer_call_fn_1232160

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_enc_layer_call_and_return_conditional_losses_12298802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Γ
ά
)__inference_Encoder_layer_call_fn_1230243
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

identity_2’StatefulPartitionedCallΕ
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
9:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_12302082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????((
'
_user_specified_nameinput_encoder
Ώ
ί
F__inference_conv3_dec_layer_call_and_return_conditional_losses_1232329

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
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
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
έ
}
(__inference_z_mean_layer_call_fn_1232209

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallφ
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
GPU2*0J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_12299472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs

`
D__inference_upsamp3_layer_call_and_return_conditional_losses_1230275

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
strided_slice/stack_2Ξ
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
mulΥ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor€
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
Υ
)__inference_Encoder_layer_call_fn_1231811

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

identity_2’StatefulPartitionedCallΎ
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
9:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_12301242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
Ϋ
Τ
"__inference__wrapped_model_1229733
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
identity’,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp’+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp’,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp’+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp’,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp’+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp’,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp’+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp’+VAE/Decoder/decoding/BiasAdd/ReadVariableOp’*VAE/Decoder/decoding/MatMul/ReadVariableOp’)VAE/Decoder/output/BiasAdd/ReadVariableOp’(VAE/Decoder/output/Conv2D/ReadVariableOp’-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp’,VAE/Encoder/bottleneck/MatMul/ReadVariableOp’,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp’+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp’,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp’+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp’,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp’+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp’,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp’+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp’,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp’+VAE/Encoder/z_log_var/MatMul/ReadVariableOp’)VAE/Encoder/z_mean/BiasAdd/ReadVariableOp’(VAE/Encoder/z_mean/MatMul/ReadVariableOpΧ
+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOpζ
VAE/Encoder/conv1_enc/Conv2DConv2Dinput_13VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingSAME*
strides
2
VAE/Encoder/conv1_enc/Conv2DΞ
,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpΰ
VAE/Encoder/conv1_enc/BiasAddBiasAdd%VAE/Encoder/conv1_enc/Conv2D:output:04VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
VAE/Encoder/conv1_enc/BiasAdd’
VAE/Encoder/conv1_enc/ReluRelu&VAE/Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
VAE/Encoder/conv1_enc/Reluέ
VAE/Encoder/maxpool1/MaxPoolMaxPool(VAE/Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool1/MaxPoolΧ
+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv2_enc/Conv2DConv2D%VAE/Encoder/maxpool1/MaxPool:output:03VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
VAE/Encoder/conv2_enc/Conv2DΞ
,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpΰ
VAE/Encoder/conv2_enc/BiasAddBiasAdd%VAE/Encoder/conv2_enc/Conv2D:output:04VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
VAE/Encoder/conv2_enc/BiasAdd’
VAE/Encoder/conv2_enc/ReluRelu&VAE/Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
VAE/Encoder/conv2_enc/Reluέ
VAE/Encoder/maxpool2/MaxPoolMaxPool(VAE/Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:?????????

 *
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool2/MaxPoolΧ
+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv3_enc/Conv2DConv2D%VAE/Encoder/maxpool2/MaxPool:output:03VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
paddingSAME*
strides
2
VAE/Encoder/conv3_enc/Conv2DΞ
,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpΰ
VAE/Encoder/conv3_enc/BiasAddBiasAdd%VAE/Encoder/conv3_enc/Conv2D:output:04VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@2
VAE/Encoder/conv3_enc/BiasAdd’
VAE/Encoder/conv3_enc/ReluRelu&VAE/Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

@2
VAE/Encoder/conv3_enc/Reluέ
VAE/Encoder/maxpool3/MaxPoolMaxPool(VAE/Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool3/MaxPoolΨ
+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv4_enc/Conv2DConv2D%VAE/Encoder/maxpool3/MaxPool:output:03VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
VAE/Encoder/conv4_enc/Conv2DΟ
,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpα
VAE/Encoder/conv4_enc/BiasAddBiasAdd%VAE/Encoder/conv4_enc/Conv2D:output:04VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
VAE/Encoder/conv4_enc/BiasAdd£
VAE/Encoder/conv4_enc/ReluRelu&VAE/Encoder/conv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
VAE/Encoder/conv4_enc/Reluή
VAE/Encoder/maxpool4/MaxPoolMaxPool(VAE/Encoder/conv4_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????  2
VAE/Encoder/flatten/ConstΓ
VAE/Encoder/flatten/ReshapeReshape%VAE/Encoder/maxpool4/MaxPool:output:0"VAE/Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:?????????	2
VAE/Encoder/flatten/ReshapeΣ
,VAE/Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp5vae_encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02.
,VAE/Encoder/bottleneck/MatMul/ReadVariableOpΦ
VAE/Encoder/bottleneck/MatMulMatMul$VAE/Encoder/flatten/Reshape:output:04VAE/Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/bottleneck/MatMulΡ
-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp6vae_encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOpέ
VAE/Encoder/bottleneck/BiasAddBiasAdd'VAE/Encoder/bottleneck/MatMul:product:05VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
VAE/Encoder/bottleneck/BiasAddΖ
(VAE/Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp1vae_encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(VAE/Encoder/z_mean/MatMul/ReadVariableOpΝ
VAE/Encoder/z_mean/MatMulMatMul'VAE/Encoder/bottleneck/BiasAdd:output:00VAE/Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/z_mean/MatMulΕ
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp2vae_encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOpΝ
VAE/Encoder/z_mean/BiasAddBiasAdd#VAE/Encoder/z_mean/MatMul:product:01VAE/Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/z_mean/BiasAddΟ
+VAE/Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp4vae_encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+VAE/Encoder/z_log_var/MatMul/ReadVariableOpΦ
VAE/Encoder/z_log_var/MatMulMatMul'VAE/Encoder/bottleneck/BiasAdd:output:03VAE/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/z_log_var/MatMulΞ
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOpΩ
VAE/Encoder/z_log_var/BiasAddBiasAdd&VAE/Encoder/z_log_var/MatMul:product:04VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/z_log_var/BiasAdd
VAE/Encoder/sampling_5/ShapeShape#VAE/Encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
VAE/Encoder/sampling_5/Shape’
*VAE/Encoder/sampling_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*VAE/Encoder/sampling_5/strided_slice/stack¦
,VAE/Encoder/sampling_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling_5/strided_slice/stack_1¦
,VAE/Encoder/sampling_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling_5/strided_slice/stack_2μ
$VAE/Encoder/sampling_5/strided_sliceStridedSlice%VAE/Encoder/sampling_5/Shape:output:03VAE/Encoder/sampling_5/strided_slice/stack:output:05VAE/Encoder/sampling_5/strided_slice/stack_1:output:05VAE/Encoder/sampling_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$VAE/Encoder/sampling_5/strided_slice
VAE/Encoder/sampling_5/Shape_1Shape#VAE/Encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2 
VAE/Encoder/sampling_5/Shape_1¦
,VAE/Encoder/sampling_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling_5/strided_slice_1/stackͺ
.VAE/Encoder/sampling_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.VAE/Encoder/sampling_5/strided_slice_1/stack_1ͺ
.VAE/Encoder/sampling_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.VAE/Encoder/sampling_5/strided_slice_1/stack_2ψ
&VAE/Encoder/sampling_5/strided_slice_1StridedSlice'VAE/Encoder/sampling_5/Shape_1:output:05VAE/Encoder/sampling_5/strided_slice_1/stack:output:07VAE/Encoder/sampling_5/strided_slice_1/stack_1:output:07VAE/Encoder/sampling_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&VAE/Encoder/sampling_5/strided_slice_1ξ
*VAE/Encoder/sampling_5/random_normal/shapePack-VAE/Encoder/sampling_5/strided_slice:output:0/VAE/Encoder/sampling_5/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2,
*VAE/Encoder/sampling_5/random_normal/shape
)VAE/Encoder/sampling_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)VAE/Encoder/sampling_5/random_normal/mean
+VAE/Encoder/sampling_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+VAE/Encoder/sampling_5/random_normal/stddevͺ
9VAE/Encoder/sampling_5/random_normal/RandomStandardNormalRandomStandardNormal3VAE/Encoder/sampling_5/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed2μ€2;
9VAE/Encoder/sampling_5/random_normal/RandomStandardNormal
(VAE/Encoder/sampling_5/random_normal/mulMulBVAE/Encoder/sampling_5/random_normal/RandomStandardNormal:output:04VAE/Encoder/sampling_5/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2*
(VAE/Encoder/sampling_5/random_normal/mulπ
$VAE/Encoder/sampling_5/random_normalAdd,VAE/Encoder/sampling_5/random_normal/mul:z:02VAE/Encoder/sampling_5/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2&
$VAE/Encoder/sampling_5/random_normal
VAE/Encoder/sampling_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
VAE/Encoder/sampling_5/mul/xΐ
VAE/Encoder/sampling_5/mulMul%VAE/Encoder/sampling_5/mul/x:output:0&VAE/Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/sampling_5/mul
VAE/Encoder/sampling_5/ExpExpVAE/Encoder/sampling_5/mul:z:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/sampling_5/ExpΏ
VAE/Encoder/sampling_5/mul_1MulVAE/Encoder/sampling_5/Exp:y:0(VAE/Encoder/sampling_5/random_normal:z:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/sampling_5/mul_1Ί
VAE/Encoder/sampling_5/addAddV2#VAE/Encoder/z_mean/BiasAdd:output:0 VAE/Encoder/sampling_5/mul_1:z:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/sampling_5/addΝ
*VAE/Decoder/decoding/MatMul/ReadVariableOpReadVariableOp3vae_decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02,
*VAE/Decoder/decoding/MatMul/ReadVariableOpΛ
VAE/Decoder/decoding/MatMulMatMulVAE/Encoder/sampling_5/add:z:02VAE/Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
VAE/Decoder/decoding/MatMulΜ
+VAE/Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp4vae_decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:	*
dtype02-
+VAE/Decoder/decoding/BiasAdd/ReadVariableOpΦ
VAE/Decoder/decoding/BiasAddBiasAdd%VAE/Decoder/decoding/MatMul:product:03VAE/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
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
)VAE/Decoder/reshape/strided_slice/stack_2Ϊ
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
value	B :2%
#VAE/Decoder/reshape/Reshape/shape/1
#VAE/Decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
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
!VAE/Decoder/reshape/Reshape/shapeΣ
VAE/Decoder/reshape/ReshapeReshape%VAE/Decoder/decoding/BiasAdd:output:0*VAE/Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????2
VAE/Decoder/reshape/ReshapeΩ
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp
VAE/Decoder/conv4_dec/Conv2DConv2D$VAE/Decoder/reshape/Reshape:output:03VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
VAE/Decoder/conv4_dec/Conv2DΟ
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpα
VAE/Decoder/conv4_dec/BiasAddBiasAdd%VAE/Decoder/conv4_dec/Conv2D:output:04VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
VAE/Decoder/conv4_dec/BiasAdd£
VAE/Decoder/conv4_dec/ReluRelu&VAE/Decoder/conv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
)VAE/Decoder/upsamp4/strided_slice/stack_2Ζ
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
VAE/Decoder/upsamp4/Const?
VAE/Decoder/upsamp4/mulMul*VAE/Decoder/upsamp4/strided_slice:output:0"VAE/Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp4/mul
0VAE/Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv4_dec/Relu:activations:0VAE/Decoder/upsamp4/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(22
0VAE/Decoder/upsamp4/resize/ResizeNearestNeighborΨ
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv3_dec/Conv2DConv2DAVAE/Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
VAE/Decoder/conv3_dec/Conv2DΞ
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpΰ
VAE/Decoder/conv3_dec/BiasAddBiasAdd%VAE/Decoder/conv3_dec/Conv2D:output:04VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
VAE/Decoder/conv3_dec/BiasAdd’
VAE/Decoder/conv3_dec/ReluRelu&VAE/Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
)VAE/Decoder/upsamp3/strided_slice/stack_2Ζ
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
VAE/Decoder/upsamp3/Const?
VAE/Decoder/upsamp3/mulMul*VAE/Decoder/upsamp3/strided_slice:output:0"VAE/Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp3/mul
0VAE/Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv3_dec/Relu:activations:0VAE/Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(22
0VAE/Decoder/upsamp3/resize/ResizeNearestNeighborΧ
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02-
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv2_dec/Conv2DConv2DAVAE/Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
VAE/Decoder/conv2_dec/Conv2DΞ
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpΰ
VAE/Decoder/conv2_dec/BiasAddBiasAdd%VAE/Decoder/conv2_dec/Conv2D:output:04VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
VAE/Decoder/conv2_dec/BiasAdd’
VAE/Decoder/conv2_dec/ReluRelu&VAE/Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
)VAE/Decoder/upsamp2/strided_slice/stack_2Ζ
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
VAE/Decoder/upsamp2/Const?
VAE/Decoder/upsamp2/mulMul*VAE/Decoder/upsamp2/strided_slice:output:0"VAE/Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp2/mul
0VAE/Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv2_dec/Relu:activations:0VAE/Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(22
0VAE/Decoder/upsamp2/resize/ResizeNearestNeighborΧ
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv1_dec/Conv2DConv2DAVAE/Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
VAE/Decoder/conv1_dec/Conv2DΞ
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpΰ
VAE/Decoder/conv1_dec/BiasAddBiasAdd%VAE/Decoder/conv1_dec/Conv2D:output:04VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
VAE/Decoder/conv1_dec/BiasAdd’
VAE/Decoder/conv1_dec/ReluRelu&VAE/Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
)VAE/Decoder/upsamp1/strided_slice/stack_2Ζ
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
VAE/Decoder/upsamp1/Const?
VAE/Decoder/upsamp1/mulMul*VAE/Decoder/upsamp1/strided_slice:output:0"VAE/Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp1/mul
0VAE/Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv1_dec/Relu:activations:0VAE/Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????00*
half_pixel_centers(22
0VAE/Decoder/upsamp1/resize/ResizeNearestNeighborΞ
(VAE/Decoder/output/Conv2D/ReadVariableOpReadVariableOp1vae_decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02*
(VAE/Decoder/output/Conv2D/ReadVariableOp
VAE/Decoder/output/Conv2DConv2DAVAE/Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:00VAE/Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingVALID*
strides
2
VAE/Decoder/output/Conv2DΕ
)VAE/Decoder/output/BiasAdd/ReadVariableOpReadVariableOp2vae_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)VAE/Decoder/output/BiasAdd/ReadVariableOpΤ
VAE/Decoder/output/BiasAddBiasAdd"VAE/Decoder/output/Conv2D:output:01VAE/Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
VAE/Decoder/output/BiasAdd’
VAE/Decoder/output/SigmoidSigmoid#VAE/Decoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
VAE/Decoder/output/Sigmoid§

IdentityIdentityVAE/Decoder/output/Sigmoid:y:0-^VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp,^VAE/Decoder/decoding/BiasAdd/ReadVariableOp+^VAE/Decoder/decoding/MatMul/ReadVariableOp*^VAE/Decoder/output/BiasAdd/ReadVariableOp)^VAE/Decoder/output/Conv2D/ReadVariableOp.^VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp-^VAE/Encoder/bottleneck/MatMul/ReadVariableOp-^VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp-^VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp,^VAE/Encoder/z_log_var/MatMul/ReadVariableOp*^VAE/Encoder/z_mean/BiasAdd/ReadVariableOp)^VAE/Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????((2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::2\
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
:?????????((
!
_user_specified_name	input_1
ω
Ζ
@__inference_VAE_layer_call_and_return_conditional_losses_1230880
input_1
encoder_1230765
encoder_1230767
encoder_1230769
encoder_1230771
encoder_1230773
encoder_1230775
encoder_1230777
encoder_1230779
encoder_1230781
encoder_1230783
encoder_1230785
encoder_1230787
encoder_1230789
encoder_1230791
decoder_1230854
decoder_1230856
decoder_1230858
decoder_1230860
decoder_1230862
decoder_1230864
decoder_1230866
decoder_1230868
decoder_1230870
decoder_1230872
decoder_1230874
decoder_1230876
identity’Decoder/StatefulPartitionedCall’Encoder/StatefulPartitionedCall’
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_1230765encoder_1230767encoder_1230769encoder_1230771encoder_1230773encoder_1230775encoder_1230777encoder_1230779encoder_1230781encoder_1230783encoder_1230785encoder_1230787encoder_1230789encoder_1230791*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_12301242!
Encoder/StatefulPartitionedCall
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_1230854decoder_1230856decoder_1230858decoder_1230860decoder_1230862decoder_1230864decoder_1230866decoder_1230868decoder_1230870decoder_1230872decoder_1230874decoder_1230876*
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
GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_12305922!
Decoder/StatefulPartitionedCallΪ
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_1
?
Υ
)__inference_Encoder_layer_call_fn_1231848

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

identity_2’StatefulPartitionedCallΎ
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
9:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_12302082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
ω
Ζ
@__inference_VAE_layer_call_and_return_conditional_losses_1230940
input_1
encoder_1230883
encoder_1230885
encoder_1230887
encoder_1230889
encoder_1230891
encoder_1230893
encoder_1230895
encoder_1230897
encoder_1230899
encoder_1230901
encoder_1230903
encoder_1230905
encoder_1230907
encoder_1230909
decoder_1230914
decoder_1230916
decoder_1230918
decoder_1230920
decoder_1230922
decoder_1230924
decoder_1230926
decoder_1230928
decoder_1230930
decoder_1230932
decoder_1230934
decoder_1230936
identity’Decoder/StatefulPartitionedCall’Encoder/StatefulPartitionedCall’
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_1230883encoder_1230885encoder_1230887encoder_1230889encoder_1230891encoder_1230893encoder_1230895encoder_1230897encoder_1230899encoder_1230901encoder_1230903encoder_1230905encoder_1230907encoder_1230909*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_12302082!
Encoder/StatefulPartitionedCall
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_1230914decoder_1230916decoder_1230918decoder_1230920decoder_1230922decoder_1230924decoder_1230926decoder_1230928decoder_1230930decoder_1230932decoder_1230934decoder_1230936*
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
GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_12306602!
Decoder/StatefulPartitionedCallΪ
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_1
	
ΰ
G__inference_bottleneck_layer_call_and_return_conditional_losses_1232181

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????	
 
_user_specified_nameinputs
	
ί
F__inference_z_log_var_layer_call_and_return_conditional_losses_1229973

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
λ	

)__inference_Decoder_layer_call_fn_1230619
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
identity’StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_12305922
StatefulPartitionedCall¨
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

`
D__inference_upsamp4_layer_call_and_return_conditional_losses_1230256

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
strided_slice/stack_2Ξ
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
mulΥ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor€
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
€
F
*__inference_maxpool1_layer_call_fn_1229745

inputs
identityι
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
GPU2*0J 8 *N
fIRG
E__inference_maxpool1_layer_call_and_return_conditional_losses_12297392
PartitionedCall
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
Ώ
ί
F__inference_conv3_dec_layer_call_and_return_conditional_losses_1230410

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
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
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ύ
`
D__inference_flatten_layer_call_and_return_conditional_losses_1229903

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs

t
G__inference_sampling_5_layer_call_and_return_conditional_losses_1230015

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
strided_slice/stack_2β
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
strided_slice_1/stack_2ξ
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
random_normal/stddevδ
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed2Ψ+2$
"random_normal/RandomStandardNormal΄
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
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
:?????????2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
©
E
)__inference_flatten_layer_call_fn_1232171

inputs
identityΖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_12299032
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ΏA
ΰ
D__inference_Encoder_layer_call_and_return_conditional_losses_1230124

inputs
conv1_enc_1230080
conv1_enc_1230082
conv2_enc_1230086
conv2_enc_1230088
conv3_enc_1230092
conv3_enc_1230094
conv4_enc_1230098
conv4_enc_1230100
bottleneck_1230105
bottleneck_1230107
z_mean_1230110
z_mean_1230112
z_log_var_1230115
z_log_var_1230117
identity

identity_1

identity_2’"bottleneck/StatefulPartitionedCall’!conv1_enc/StatefulPartitionedCall’!conv2_enc/StatefulPartitionedCall’!conv3_enc/StatefulPartitionedCall’!conv4_enc/StatefulPartitionedCall’"sampling_5/StatefulPartitionedCall’!z_log_var/StatefulPartitionedCall’z_mean/StatefulPartitionedCall§
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_1230080conv1_enc_1230082*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????((*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_enc_layer_call_and_return_conditional_losses_12297962#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool1_layer_call_and_return_conditional_losses_12297392
maxpool1/PartitionedCallΒ
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_1230086conv2_enc_1230088*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_enc_layer_call_and_return_conditional_losses_12298242#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool2_layer_call_and_return_conditional_losses_12297512
maxpool2/PartitionedCallΒ
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_1230092conv3_enc_1230094*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv3_enc_layer_call_and_return_conditional_losses_12298522#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool3_layer_call_and_return_conditional_losses_12297632
maxpool3/PartitionedCallΓ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_1230098conv4_enc_1230100*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_enc_layer_call_and_return_conditional_losses_12298802#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool4_layer_call_and_return_conditional_losses_12297752
maxpool4/PartitionedCallρ
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_12299032
flatten/PartitionedCallΎ
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_1230105bottleneck_1230107*
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
GPU2*0J 8 *P
fKRI
G__inference_bottleneck_layer_call_and_return_conditional_losses_12299212$
"bottleneck/StatefulPartitionedCall΅
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_1230110z_mean_1230112*
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
GPU2*0J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_12299472 
z_mean/StatefulPartitionedCallΔ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_1230115z_log_var_1230117*
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
GPU2*0J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_12299732#
!z_log_var/StatefulPartitionedCallΔ
"sampling_5/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sampling_5_layer_call_and_return_conditional_losses_12300152$
"sampling_5/StatefulPartitionedCall
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity‘

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1’

Identity_2Identity+sampling_5/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2H
"sampling_5/StatefulPartitionedCall"sampling_5/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
Ω

ί
F__inference_conv4_dec_layer_call_and_return_conditional_losses_1230382

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
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
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ϊ
a
E__inference_maxpool2_layer_call_and_return_conditional_losses_1229751

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool
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

v
G__inference_sampling_5_layer_call_and_return_conditional_losses_1232254
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
strided_slice/stack_2β
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
strided_slice_1/stack_2ξ
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
random_normal/stddevδ
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed2g2$
"random_normal/RandomStandardNormal΄
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
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
:?????????2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
	
ά
C__inference_z_mean_layer_call_and_return_conditional_losses_1232200

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
Φ

ί
F__inference_conv4_enc_layer_call_and_return_conditional_losses_1232151

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
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
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Μ

+__inference_conv1_dec_layer_call_fn_1232378

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
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
GPU2*0J 8 *O
fJRH
F__inference_conv1_dec_layer_call_and_return_conditional_losses_12304662
StatefulPartitionedCall¨
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
ϊ
a
E__inference_maxpool3_layer_call_and_return_conditional_losses_1229763

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool
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
½
ρ
%__inference_VAE_layer_call_fn_1231559

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
identity’StatefulPartitionedCallΤ
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
-:+???????????????????????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_VAE_layer_call_and_return_conditional_losses_12310032
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
Ξ

+__inference_conv3_dec_layer_call_fn_1232338

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
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
GPU2*0J 8 *O
fJRH
F__inference_conv3_dec_layer_call_and_return_conditional_losses_12304102
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Π

ί
F__inference_conv2_enc_layer_call_and_return_conditional_losses_1232111

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
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
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζυ
‘
@__inference_VAE_layer_call_and_return_conditional_losses_1231342

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
identity’(Decoder/conv1_dec/BiasAdd/ReadVariableOp’'Decoder/conv1_dec/Conv2D/ReadVariableOp’(Decoder/conv2_dec/BiasAdd/ReadVariableOp’'Decoder/conv2_dec/Conv2D/ReadVariableOp’(Decoder/conv3_dec/BiasAdd/ReadVariableOp’'Decoder/conv3_dec/Conv2D/ReadVariableOp’(Decoder/conv4_dec/BiasAdd/ReadVariableOp’'Decoder/conv4_dec/Conv2D/ReadVariableOp’'Decoder/decoding/BiasAdd/ReadVariableOp’&Decoder/decoding/MatMul/ReadVariableOp’%Decoder/output/BiasAdd/ReadVariableOp’$Decoder/output/Conv2D/ReadVariableOp’)Encoder/bottleneck/BiasAdd/ReadVariableOp’(Encoder/bottleneck/MatMul/ReadVariableOp’(Encoder/conv1_enc/BiasAdd/ReadVariableOp’'Encoder/conv1_enc/Conv2D/ReadVariableOp’(Encoder/conv2_enc/BiasAdd/ReadVariableOp’'Encoder/conv2_enc/Conv2D/ReadVariableOp’(Encoder/conv3_enc/BiasAdd/ReadVariableOp’'Encoder/conv3_enc/Conv2D/ReadVariableOp’(Encoder/conv4_enc/BiasAdd/ReadVariableOp’'Encoder/conv4_enc/Conv2D/ReadVariableOp’(Encoder/z_log_var/BiasAdd/ReadVariableOp’'Encoder/z_log_var/MatMul/ReadVariableOp’%Encoder/z_mean/BiasAdd/ReadVariableOp’$Encoder/z_mean/MatMul/ReadVariableOpΛ
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOpΩ
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2DΒ
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
Encoder/conv1_enc/BiasAdd
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
Encoder/conv1_enc/ReluΡ
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPoolΛ
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOpτ
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2DΒ
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Encoder/conv2_enc/BiasAdd
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Encoder/conv2_enc/ReluΡ
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:?????????

 *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPoolΛ
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOpτ
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2DΒ
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@2
Encoder/conv3_enc/BiasAdd
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

@2
Encoder/conv3_enc/ReluΡ
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPoolΜ
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOpυ
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2DΓ
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOpΡ
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Encoder/conv4_enc/BiasAdd
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Encoder/conv4_enc/Relu?
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????  2
Encoder/flatten/Const³
Encoder/flatten/ReshapeReshape!Encoder/maxpool4/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:?????????	2
Encoder/flatten/ReshapeΗ
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOpΖ
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/MatMulΕ
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOpΝ
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/BiasAddΊ
$Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Encoder/z_mean/MatMul/ReadVariableOp½
Encoder/z_mean/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0,Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_mean/MatMulΉ
%Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Encoder/z_mean/BiasAdd/ReadVariableOp½
Encoder/z_mean/BiasAddBiasAddEncoder/z_mean/MatMul:product:0-Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_mean/BiasAddΓ
'Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Encoder/z_log_var/MatMul/ReadVariableOpΖ
Encoder/z_log_var/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_log_var/MatMulΒ
(Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/z_log_var/BiasAdd/ReadVariableOpΙ
Encoder/z_log_var/BiasAddBiasAdd"Encoder/z_log_var/MatMul:product:00Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_log_var/BiasAdd
Encoder/sampling_5/ShapeShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling_5/Shape
&Encoder/sampling_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Encoder/sampling_5/strided_slice/stack
(Encoder/sampling_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_5/strided_slice/stack_1
(Encoder/sampling_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_5/strided_slice/stack_2Τ
 Encoder/sampling_5/strided_sliceStridedSlice!Encoder/sampling_5/Shape:output:0/Encoder/sampling_5/strided_slice/stack:output:01Encoder/sampling_5/strided_slice/stack_1:output:01Encoder/sampling_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Encoder/sampling_5/strided_slice
Encoder/sampling_5/Shape_1ShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling_5/Shape_1
(Encoder/sampling_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_5/strided_slice_1/stack’
*Encoder/sampling_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*Encoder/sampling_5/strided_slice_1/stack_1’
*Encoder/sampling_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*Encoder/sampling_5/strided_slice_1/stack_2ΰ
"Encoder/sampling_5/strided_slice_1StridedSlice#Encoder/sampling_5/Shape_1:output:01Encoder/sampling_5/strided_slice_1/stack:output:03Encoder/sampling_5/strided_slice_1/stack_1:output:03Encoder/sampling_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"Encoder/sampling_5/strided_slice_1ή
&Encoder/sampling_5/random_normal/shapePack)Encoder/sampling_5/strided_slice:output:0+Encoder/sampling_5/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Encoder/sampling_5/random_normal/shape
%Encoder/sampling_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%Encoder/sampling_5/random_normal/mean
'Encoder/sampling_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'Encoder/sampling_5/random_normal/stddev
5Encoder/sampling_5/random_normal/RandomStandardNormalRandomStandardNormal/Encoder/sampling_5/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed2?27
5Encoder/sampling_5/random_normal/RandomStandardNormal
$Encoder/sampling_5/random_normal/mulMul>Encoder/sampling_5/random_normal/RandomStandardNormal:output:00Encoder/sampling_5/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2&
$Encoder/sampling_5/random_normal/mulΰ
 Encoder/sampling_5/random_normalAdd(Encoder/sampling_5/random_normal/mul:z:0.Encoder/sampling_5/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2"
 Encoder/sampling_5/random_normaly
Encoder/sampling_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Encoder/sampling_5/mul/x°
Encoder/sampling_5/mulMul!Encoder/sampling_5/mul/x:output:0"Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling_5/mul
Encoder/sampling_5/ExpExpEncoder/sampling_5/mul:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling_5/Exp―
Encoder/sampling_5/mul_1MulEncoder/sampling_5/Exp:y:0$Encoder/sampling_5/random_normal:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling_5/mul_1ͺ
Encoder/sampling_5/addAddV2Encoder/z_mean/BiasAdd:output:0Encoder/sampling_5/mul_1:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling_5/addΑ
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOp»
Decoder/decoding/MatMulMatMulEncoder/sampling_5/add:z:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
Decoder/decoding/MatMulΐ
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:	*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOpΖ
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
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
%Decoder/reshape/strided_slice/stack_2Β
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
value	B :2!
Decoder/reshape/Reshape/shape/1
Decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
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
Decoder/reshape/Reshape/shapeΓ
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????2
Decoder/reshape/ReshapeΝ
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOpτ
Decoder/conv4_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2DΓ
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOpΡ
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Decoder/conv4_dec/BiasAdd
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
Decoder/upsamp4/Const
Decoder/upsamp4/mulMul&Decoder/upsamp4/strided_slice:output:0Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp4/mul
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighborΜ
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2DΒ
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Decoder/conv3_dec/BiasAdd
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
Decoder/upsamp3/Const
Decoder/upsamp3/mulMul&Decoder/upsamp3/strided_slice:output:0Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp3/mul
,Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv3_dec/Relu:activations:0Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighborΛ
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2DΒ
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv2_dec/BiasAdd
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
Decoder/upsamp2/Const
Decoder/upsamp2/mulMul&Decoder/upsamp2/strided_slice:output:0Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp2/mul
,Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv2_dec/Relu:activations:0Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighborΛ
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2DΒ
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Decoder/conv1_dec/BiasAdd
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
Decoder/upsamp1/Const
Decoder/upsamp1/mulMul&Decoder/upsamp1/strided_slice:output:0Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp1/mul
,Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv1_dec/Relu:activations:0Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????00*
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighborΒ
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingVALID*
strides
2
Decoder/output/Conv2DΉ
%Decoder/output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Decoder/output/BiasAdd/ReadVariableOpΔ
Decoder/output/BiasAddBiasAddDecoder/output/Conv2D:output:0-Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
Decoder/output/BiasAdd
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
Decoder/output/Sigmoid»	
IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp)^Encoder/z_log_var/BiasAdd/ReadVariableOp(^Encoder/z_log_var/MatMul/ReadVariableOp&^Encoder/z_mean/BiasAdd/ReadVariableOp%^Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????((2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::2T
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
:?????????((
 
_user_specified_nameinputs


+__inference_conv3_enc_layer_call_fn_1232140

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
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
@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv3_enc_layer_call_and_return_conditional_losses_12298522
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????

@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

 ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????

 
 
_user_specified_nameinputs
Ό
ά
C__inference_output_layer_call_and_return_conditional_losses_1230494

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
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
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoidͺ
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
«1
Γ
D__inference_Decoder_layer_call_and_return_conditional_losses_1230592

inputs
decoding_1230556
decoding_1230558
conv4_dec_1230562
conv4_dec_1230564
conv3_dec_1230568
conv3_dec_1230570
conv2_dec_1230574
conv2_dec_1230576
conv1_dec_1230580
conv1_dec_1230582
output_1230586
output_1230588
identity’!conv1_dec/StatefulPartitionedCall’!conv2_dec/StatefulPartitionedCall’!conv3_dec/StatefulPartitionedCall’!conv4_dec/StatefulPartitionedCall’ decoding/StatefulPartitionedCall’output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_1230556decoding_1230558*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoding_layer_call_and_return_conditional_losses_12303332"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_12303632
reshape/PartitionedCallΒ
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_1230562conv4_dec_1230564*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_dec_layer_call_and_return_conditional_losses_12303822#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_upsamp4_layer_call_and_return_conditional_losses_12302562
upsamp4/PartitionedCallΣ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_1230568conv3_dec_1230570*
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
GPU2*0J 8 *O
fJRH
F__inference_conv3_dec_layer_call_and_return_conditional_losses_12304102#
!conv3_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp3_layer_call_and_return_conditional_losses_12302752
upsamp3/PartitionedCallΣ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_1230574conv2_dec_1230576*
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
GPU2*0J 8 *O
fJRH
F__inference_conv2_dec_layer_call_and_return_conditional_losses_12304382#
!conv2_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp2_layer_call_and_return_conditional_losses_12302942
upsamp2/PartitionedCallΣ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_1230580conv1_dec_1230582*
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
GPU2*0J 8 *O
fJRH
F__inference_conv1_dec_layer_call_and_return_conditional_losses_12304662#
!conv1_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp1_layer_call_and_return_conditional_losses_12303132
upsamp1/PartitionedCallΔ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_1230586output_1230588*
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
GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_12304942 
output/StatefulPartitionedCallι
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
Ύ
`
D__inference_flatten_layer_call_and_return_conditional_losses_1232166

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
	
ά
C__inference_z_mean_layer_call_and_return_conditional_losses_1229947

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

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
γ

*__inference_decoding_layer_call_fn_1232279

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallω
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoding_layer_call_and_return_conditional_losses_12303332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ΏA
ΰ
D__inference_Encoder_layer_call_and_return_conditional_losses_1230208

inputs
conv1_enc_1230164
conv1_enc_1230166
conv2_enc_1230170
conv2_enc_1230172
conv3_enc_1230176
conv3_enc_1230178
conv4_enc_1230182
conv4_enc_1230184
bottleneck_1230189
bottleneck_1230191
z_mean_1230194
z_mean_1230196
z_log_var_1230199
z_log_var_1230201
identity

identity_1

identity_2’"bottleneck/StatefulPartitionedCall’!conv1_enc/StatefulPartitionedCall’!conv2_enc/StatefulPartitionedCall’!conv3_enc/StatefulPartitionedCall’!conv4_enc/StatefulPartitionedCall’"sampling_5/StatefulPartitionedCall’!z_log_var/StatefulPartitionedCall’z_mean/StatefulPartitionedCall§
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_1230164conv1_enc_1230166*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????((*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_enc_layer_call_and_return_conditional_losses_12297962#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool1_layer_call_and_return_conditional_losses_12297392
maxpool1/PartitionedCallΒ
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_1230170conv2_enc_1230172*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_enc_layer_call_and_return_conditional_losses_12298242#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool2_layer_call_and_return_conditional_losses_12297512
maxpool2/PartitionedCallΒ
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_1230176conv3_enc_1230178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv3_enc_layer_call_and_return_conditional_losses_12298522#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool3_layer_call_and_return_conditional_losses_12297632
maxpool3/PartitionedCallΓ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_1230182conv4_enc_1230184*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_enc_layer_call_and_return_conditional_losses_12298802#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool4_layer_call_and_return_conditional_losses_12297752
maxpool4/PartitionedCallρ
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_12299032
flatten/PartitionedCallΎ
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_1230189bottleneck_1230191*
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
GPU2*0J 8 *P
fKRI
G__inference_bottleneck_layer_call_and_return_conditional_losses_12299212$
"bottleneck/StatefulPartitionedCall΅
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_1230194z_mean_1230196*
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
GPU2*0J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_12299472 
z_mean/StatefulPartitionedCallΔ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_1230199z_log_var_1230201*
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
GPU2*0J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_12299732#
!z_log_var/StatefulPartitionedCallΔ
"sampling_5/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sampling_5_layer_call_and_return_conditional_losses_12300152$
"sampling_5/StatefulPartitionedCall
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity‘

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1’

Identity_2Identity+sampling_5/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2H
"sampling_5/StatefulPartitionedCall"sampling_5/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
θ

,__inference_bottleneck_layer_call_fn_1232190

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallϊ
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
GPU2*0J 8 *P
fKRI
G__inference_bottleneck_layer_call_and_return_conditional_losses_12299212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????	
 
_user_specified_nameinputs
Π

ί
F__inference_conv3_enc_layer_call_and_return_conditional_losses_1229852

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
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
:?????????

@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????

 
 
_user_specified_nameinputs
Μ

+__inference_conv2_dec_layer_call_fn_1232358

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
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
GPU2*0J 8 *O
fJRH
F__inference_conv2_dec_layer_call_and_return_conditional_losses_12304382
StatefulPartitionedCall¨
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
όo
ι	
D__inference_Encoder_layer_call_and_return_conditional_losses_1231774

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

identity_2’!bottleneck/BiasAdd/ReadVariableOp’ bottleneck/MatMul/ReadVariableOp’ conv1_enc/BiasAdd/ReadVariableOp’conv1_enc/Conv2D/ReadVariableOp’ conv2_enc/BiasAdd/ReadVariableOp’conv2_enc/Conv2D/ReadVariableOp’ conv3_enc/BiasAdd/ReadVariableOp’conv3_enc/Conv2D/ReadVariableOp’ conv4_enc/BiasAdd/ReadVariableOp’conv4_enc/Conv2D/ReadVariableOp’ z_log_var/BiasAdd/ReadVariableOp’z_log_var/MatMul/ReadVariableOp’z_mean/BiasAdd/ReadVariableOp’z_mean/MatMul/ReadVariableOp³
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOpΑ
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingSAME*
strides
2
conv1_enc/Conv2Dͺ
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOp°
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
conv1_enc/ReluΉ
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool³
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_enc/Conv2D/ReadVariableOpΤ
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2_enc/Conv2Dͺ
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_enc/BiasAdd/ReadVariableOp°
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2_enc/ReluΉ
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:?????????

 *
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPool³
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv3_enc/Conv2D/ReadVariableOpΤ
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
paddingSAME*
strides
2
conv3_enc/Conv2Dͺ
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_enc/BiasAdd/ReadVariableOp°
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

@2
conv3_enc/ReluΉ
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPool΄
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv4_enc/Conv2D/ReadVariableOpΥ
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
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
:?????????2
conv4_enc/BiasAdd
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
conv4_enc/ReluΊ
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????  2
flatten/Const
flatten/ReshapeReshapemaxpool4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????	2
flatten/Reshape―
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02"
 bottleneck/MatMul/ReadVariableOp¦
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/MatMul­
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp­
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/BiasAdd’
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMulbottleneck/BiasAdd:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/MatMul‘
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/BiasAdd«
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
z_log_var/MatMul/ReadVariableOp¦
z_log_var/MatMulMatMulbottleneck/BiasAdd:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/MatMulͺ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp©
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/BiasAddk
sampling_5/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_5/Shape
sampling_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
sampling_5/strided_slice/stack
 sampling_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_5/strided_slice/stack_1
 sampling_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_5/strided_slice/stack_2€
sampling_5/strided_sliceStridedSlicesampling_5/Shape:output:0'sampling_5/strided_slice/stack:output:0)sampling_5/strided_slice/stack_1:output:0)sampling_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_5/strided_sliceo
sampling_5/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_5/Shape_1
 sampling_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_5/strided_slice_1/stack
"sampling_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_5/strided_slice_1/stack_1
"sampling_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_5/strided_slice_1/stack_2°
sampling_5/strided_slice_1StridedSlicesampling_5/Shape_1:output:0)sampling_5/strided_slice_1/stack:output:0+sampling_5/strided_slice_1/stack_1:output:0+sampling_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_5/strided_slice_1Ύ
sampling_5/random_normal/shapePack!sampling_5/strided_slice:output:0#sampling_5/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2 
sampling_5/random_normal/shape
sampling_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling_5/random_normal/mean
sampling_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
sampling_5/random_normal/stddev
-sampling_5/random_normal/RandomStandardNormalRandomStandardNormal'sampling_5/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed2Φλ2/
-sampling_5/random_normal/RandomStandardNormalΰ
sampling_5/random_normal/mulMul6sampling_5/random_normal/RandomStandardNormal:output:0(sampling_5/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
sampling_5/random_normal/mulΐ
sampling_5/random_normalAdd sampling_5/random_normal/mul:z:0&sampling_5/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
sampling_5/random_normali
sampling_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling_5/mul/x
sampling_5/mulMulsampling_5/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sampling_5/mulm
sampling_5/ExpExpsampling_5/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling_5/Exp
sampling_5/mul_1Mulsampling_5/Exp:y:0sampling_5/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling_5/mul_1
sampling_5/addAddV2z_mean/BiasAdd:output:0sampling_5/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling_5/addΚ
IdentityIdentityz_mean/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

IdentityΡ

Identity_1Identityz_log_var/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1Ι

Identity_2Identitysampling_5/add:z:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::2F
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
:?????????((
 
_user_specified_nameinputs
λ	

)__inference_Decoder_layer_call_fn_1230687
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
identity’StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_12306602
StatefulPartitionedCall¨
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
ύo
ι	
D__inference_Encoder_layer_call_and_return_conditional_losses_1231695

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

identity_2’!bottleneck/BiasAdd/ReadVariableOp’ bottleneck/MatMul/ReadVariableOp’ conv1_enc/BiasAdd/ReadVariableOp’conv1_enc/Conv2D/ReadVariableOp’ conv2_enc/BiasAdd/ReadVariableOp’conv2_enc/Conv2D/ReadVariableOp’ conv3_enc/BiasAdd/ReadVariableOp’conv3_enc/Conv2D/ReadVariableOp’ conv4_enc/BiasAdd/ReadVariableOp’conv4_enc/Conv2D/ReadVariableOp’ z_log_var/BiasAdd/ReadVariableOp’z_log_var/MatMul/ReadVariableOp’z_mean/BiasAdd/ReadVariableOp’z_mean/MatMul/ReadVariableOp³
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOpΑ
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingSAME*
strides
2
conv1_enc/Conv2Dͺ
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOp°
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
conv1_enc/ReluΉ
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool³
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_enc/Conv2D/ReadVariableOpΤ
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2_enc/Conv2Dͺ
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_enc/BiasAdd/ReadVariableOp°
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2_enc/ReluΉ
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:?????????

 *
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPool³
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv3_enc/Conv2D/ReadVariableOpΤ
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
paddingSAME*
strides
2
conv3_enc/Conv2Dͺ
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_enc/BiasAdd/ReadVariableOp°
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

@2
conv3_enc/ReluΉ
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPool΄
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv4_enc/Conv2D/ReadVariableOpΥ
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
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
:?????????2
conv4_enc/BiasAdd
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
conv4_enc/ReluΊ
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????  2
flatten/Const
flatten/ReshapeReshapemaxpool4/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????	2
flatten/Reshape―
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02"
 bottleneck/MatMul/ReadVariableOp¦
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/MatMul­
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp­
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/BiasAdd’
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMulbottleneck/BiasAdd:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/MatMul‘
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/BiasAdd«
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
z_log_var/MatMul/ReadVariableOp¦
z_log_var/MatMulMatMulbottleneck/BiasAdd:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/MatMulͺ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp©
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/BiasAddk
sampling_5/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_5/Shape
sampling_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
sampling_5/strided_slice/stack
 sampling_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_5/strided_slice/stack_1
 sampling_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_5/strided_slice/stack_2€
sampling_5/strided_sliceStridedSlicesampling_5/Shape:output:0'sampling_5/strided_slice/stack:output:0)sampling_5/strided_slice/stack_1:output:0)sampling_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_5/strided_sliceo
sampling_5/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling_5/Shape_1
 sampling_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2"
 sampling_5/strided_slice_1/stack
"sampling_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_5/strided_slice_1/stack_1
"sampling_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sampling_5/strided_slice_1/stack_2°
sampling_5/strided_slice_1StridedSlicesampling_5/Shape_1:output:0)sampling_5/strided_slice_1/stack:output:0+sampling_5/strided_slice_1/stack_1:output:0+sampling_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling_5/strided_slice_1Ύ
sampling_5/random_normal/shapePack!sampling_5/strided_slice:output:0#sampling_5/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2 
sampling_5/random_normal/shape
sampling_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling_5/random_normal/mean
sampling_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
sampling_5/random_normal/stddev
-sampling_5/random_normal/RandomStandardNormalRandomStandardNormal'sampling_5/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed2Βδ2/
-sampling_5/random_normal/RandomStandardNormalΰ
sampling_5/random_normal/mulMul6sampling_5/random_normal/RandomStandardNormal:output:0(sampling_5/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
sampling_5/random_normal/mulΐ
sampling_5/random_normalAdd sampling_5/random_normal/mul:z:0&sampling_5/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
sampling_5/random_normali
sampling_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling_5/mul/x
sampling_5/mulMulsampling_5/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sampling_5/mulm
sampling_5/ExpExpsampling_5/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling_5/Exp
sampling_5/mul_1Mulsampling_5/Exp:y:0sampling_5/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling_5/mul_1
sampling_5/addAddV2z_mean/BiasAdd:output:0sampling_5/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling_5/addΚ
IdentityIdentityz_mean/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

IdentityΡ

Identity_1Identityz_log_var/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1Ι

Identity_2Identitysampling_5/add:z:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::2F
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
:?????????((
 
_user_specified_nameinputs


+__inference_conv4_dec_layer_call_fn_1232318

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_dec_layer_call_and_return_conditional_losses_12303822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Φ

ί
F__inference_conv4_enc_layer_call_and_return_conditional_losses_1229880

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
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
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
ά’
‘$
 __inference__traced_save_1232688
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

identity_1’MergeV2Checkpoints
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
ShardedFilenameΰ)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*ς(
valueθ(Bε(ZB3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesΏ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*Ι
valueΏBΌZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesγ"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv1_enc_kernel_read_readvariableop)savev2_conv1_enc_bias_read_readvariableop+savev2_conv2_enc_kernel_read_readvariableop)savev2_conv2_enc_bias_read_readvariableop+savev2_conv3_enc_kernel_read_readvariableop)savev2_conv3_enc_bias_read_readvariableop+savev2_conv4_enc_kernel_read_readvariableop)savev2_conv4_enc_bias_read_readvariableop,savev2_bottleneck_kernel_read_readvariableop*savev2_bottleneck_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableop*savev2_decoding_kernel_read_readvariableop(savev2_decoding_bias_read_readvariableop+savev2_conv4_dec_kernel_read_readvariableop)savev2_conv4_dec_bias_read_readvariableop+savev2_conv3_dec_kernel_read_readvariableop)savev2_conv3_dec_bias_read_readvariableop+savev2_conv2_dec_kernel_read_readvariableop)savev2_conv2_dec_bias_read_readvariableop+savev2_conv1_dec_kernel_read_readvariableop)savev2_conv1_dec_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop2savev2_adam_conv1_enc_kernel_m_read_readvariableop0savev2_adam_conv1_enc_bias_m_read_readvariableop2savev2_adam_conv2_enc_kernel_m_read_readvariableop0savev2_adam_conv2_enc_bias_m_read_readvariableop2savev2_adam_conv3_enc_kernel_m_read_readvariableop0savev2_adam_conv3_enc_bias_m_read_readvariableop2savev2_adam_conv4_enc_kernel_m_read_readvariableop0savev2_adam_conv4_enc_bias_m_read_readvariableop3savev2_adam_bottleneck_kernel_m_read_readvariableop1savev2_adam_bottleneck_bias_m_read_readvariableop/savev2_adam_z_mean_kernel_m_read_readvariableop-savev2_adam_z_mean_bias_m_read_readvariableop2savev2_adam_z_log_var_kernel_m_read_readvariableop0savev2_adam_z_log_var_bias_m_read_readvariableop1savev2_adam_decoding_kernel_m_read_readvariableop/savev2_adam_decoding_bias_m_read_readvariableop2savev2_adam_conv4_dec_kernel_m_read_readvariableop0savev2_adam_conv4_dec_bias_m_read_readvariableop2savev2_adam_conv3_dec_kernel_m_read_readvariableop0savev2_adam_conv3_dec_bias_m_read_readvariableop2savev2_adam_conv2_dec_kernel_m_read_readvariableop0savev2_adam_conv2_dec_bias_m_read_readvariableop2savev2_adam_conv1_dec_kernel_m_read_readvariableop0savev2_adam_conv1_dec_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv1_enc_kernel_v_read_readvariableop0savev2_adam_conv1_enc_bias_v_read_readvariableop2savev2_adam_conv2_enc_kernel_v_read_readvariableop0savev2_adam_conv2_enc_bias_v_read_readvariableop2savev2_adam_conv3_enc_kernel_v_read_readvariableop0savev2_adam_conv3_enc_bias_v_read_readvariableop2savev2_adam_conv4_enc_kernel_v_read_readvariableop0savev2_adam_conv4_enc_bias_v_read_readvariableop3savev2_adam_bottleneck_kernel_v_read_readvariableop1savev2_adam_bottleneck_bias_v_read_readvariableop/savev2_adam_z_mean_kernel_v_read_readvariableop-savev2_adam_z_mean_bias_v_read_readvariableop2savev2_adam_z_log_var_kernel_v_read_readvariableop0savev2_adam_z_log_var_bias_v_read_readvariableop1savev2_adam_decoding_kernel_v_read_readvariableop/savev2_adam_decoding_bias_v_read_readvariableop2savev2_adam_conv4_dec_kernel_v_read_readvariableop0savev2_adam_conv4_dec_bias_v_read_readvariableop2savev2_adam_conv3_dec_kernel_v_read_readvariableop0savev2_adam_conv3_dec_bias_v_read_readvariableop2savev2_adam_conv2_dec_kernel_v_read_readvariableop0savev2_adam_conv2_dec_bias_v_read_readvariableop2savev2_adam_conv1_dec_kernel_v_read_readvariableop0savev2_adam_conv1_dec_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *h
dtypes^
\2Z	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
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
ύ: : : : : : : : : : : : ::: : : @:@:@::		::::::		:	:::@:@:@ : : ::		:::: : : @:@:@::		::::::		:	:::@:@:@ : : ::		:::: : : @:@:@::		::::::		:	:::@:@:@ : : ::		:: 2(
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
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::%!

_output_shapes
:		: 
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
:		:!

_output_shapes	
:	:.*
(
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@:, (
&
_output_shapes
:@ : !

_output_shapes
: :,"(
&
_output_shapes
: : #

_output_shapes
::,$(
&
_output_shapes
:		: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
: : )

_output_shapes
: :,*(
&
_output_shapes
: @: +

_output_shapes
:@:-,)
'
_output_shapes
:@:!-

_output_shapes	
::%.!

_output_shapes
:		: /
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
:		:!5

_output_shapes	
:	:.6*
(
_output_shapes
::!7

_output_shapes	
::-8)
'
_output_shapes
:@: 9

_output_shapes
:@:,:(
&
_output_shapes
:@ : ;

_output_shapes
: :,<(
&
_output_shapes
: : =

_output_shapes
::,>(
&
_output_shapes
:		: ?

_output_shapes
::,@(
&
_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
: : C

_output_shapes
: :,D(
&
_output_shapes
: @: E

_output_shapes
:@:-F)
'
_output_shapes
:@:!G

_output_shapes	
::%H!

_output_shapes
:		: I
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
:		:!O

_output_shapes	
:	:.P*
(
_output_shapes
::!Q

_output_shapes	
::-R)
'
_output_shapes
:@: S

_output_shapes
:@:,T(
&
_output_shapes
:@ : U

_output_shapes
: :,V(
&
_output_shapes
: : W

_output_shapes
::,X(
&
_output_shapes
:		: Y

_output_shapes
::Z

_output_shapes
: 
φ
Ε
@__inference_VAE_layer_call_and_return_conditional_losses_1231003

inputs
encoder_1230946
encoder_1230948
encoder_1230950
encoder_1230952
encoder_1230954
encoder_1230956
encoder_1230958
encoder_1230960
encoder_1230962
encoder_1230964
encoder_1230966
encoder_1230968
encoder_1230970
encoder_1230972
decoder_1230977
decoder_1230979
decoder_1230981
decoder_1230983
decoder_1230985
decoder_1230987
decoder_1230989
decoder_1230991
decoder_1230993
decoder_1230995
decoder_1230997
decoder_1230999
identity’Decoder/StatefulPartitionedCall’Encoder/StatefulPartitionedCall‘
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_1230946encoder_1230948encoder_1230950encoder_1230952encoder_1230954encoder_1230956encoder_1230958encoder_1230960encoder_1230962encoder_1230964encoder_1230966encoder_1230968encoder_1230970encoder_1230972*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_12302082!
Encoder/StatefulPartitionedCall
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_1230977decoder_1230979decoder_1230981decoder_1230983decoder_1230985decoder_1230987decoder_1230989decoder_1230991decoder_1230993decoder_1230995decoder_1230997decoder_1230999*
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
GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_12306602!
Decoder/StatefulPartitionedCallΪ
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
ώ
ς
%__inference_signature_wrapper_1231182
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
identity’StatefulPartitionedCall₯
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
:?????????((*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_12297332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????((2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_1
«1
Γ
D__inference_Decoder_layer_call_and_return_conditional_losses_1230660

inputs
decoding_1230624
decoding_1230626
conv4_dec_1230630
conv4_dec_1230632
conv3_dec_1230636
conv3_dec_1230638
conv2_dec_1230642
conv2_dec_1230644
conv1_dec_1230648
conv1_dec_1230650
output_1230654
output_1230656
identity’!conv1_dec/StatefulPartitionedCall’!conv2_dec/StatefulPartitionedCall’!conv3_dec/StatefulPartitionedCall’!conv4_dec/StatefulPartitionedCall’ decoding/StatefulPartitionedCall’output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_1230624decoding_1230626*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoding_layer_call_and_return_conditional_losses_12303332"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_12303632
reshape/PartitionedCallΒ
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_1230630conv4_dec_1230632*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_dec_layer_call_and_return_conditional_losses_12303822#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_upsamp4_layer_call_and_return_conditional_losses_12302562
upsamp4/PartitionedCallΣ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_1230636conv3_dec_1230638*
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
GPU2*0J 8 *O
fJRH
F__inference_conv3_dec_layer_call_and_return_conditional_losses_12304102#
!conv3_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp3_layer_call_and_return_conditional_losses_12302752
upsamp3/PartitionedCallΣ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_1230642conv2_dec_1230644*
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
GPU2*0J 8 *O
fJRH
F__inference_conv2_dec_layer_call_and_return_conditional_losses_12304382#
!conv2_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp2_layer_call_and_return_conditional_losses_12302942
upsamp2/PartitionedCallΣ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_1230648conv1_dec_1230650*
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
GPU2*0J 8 *O
fJRH
F__inference_conv1_dec_layer_call_and_return_conditional_losses_12304662#
!conv1_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp1_layer_call_and_return_conditional_losses_12303132
upsamp1/PartitionedCallΔ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_1230654output_1230656*
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
GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_12304942 
output/StatefulPartitionedCallι
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
ϊ
a
E__inference_maxpool1_layer_call_and_return_conditional_losses_1229739

inputs
identity¬
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool
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
Π

ί
F__inference_conv2_enc_layer_call_and_return_conditional_losses_1229824

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
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
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζυ
‘
@__inference_VAE_layer_call_and_return_conditional_losses_1231502

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
identity’(Decoder/conv1_dec/BiasAdd/ReadVariableOp’'Decoder/conv1_dec/Conv2D/ReadVariableOp’(Decoder/conv2_dec/BiasAdd/ReadVariableOp’'Decoder/conv2_dec/Conv2D/ReadVariableOp’(Decoder/conv3_dec/BiasAdd/ReadVariableOp’'Decoder/conv3_dec/Conv2D/ReadVariableOp’(Decoder/conv4_dec/BiasAdd/ReadVariableOp’'Decoder/conv4_dec/Conv2D/ReadVariableOp’'Decoder/decoding/BiasAdd/ReadVariableOp’&Decoder/decoding/MatMul/ReadVariableOp’%Decoder/output/BiasAdd/ReadVariableOp’$Decoder/output/Conv2D/ReadVariableOp’)Encoder/bottleneck/BiasAdd/ReadVariableOp’(Encoder/bottleneck/MatMul/ReadVariableOp’(Encoder/conv1_enc/BiasAdd/ReadVariableOp’'Encoder/conv1_enc/Conv2D/ReadVariableOp’(Encoder/conv2_enc/BiasAdd/ReadVariableOp’'Encoder/conv2_enc/Conv2D/ReadVariableOp’(Encoder/conv3_enc/BiasAdd/ReadVariableOp’'Encoder/conv3_enc/Conv2D/ReadVariableOp’(Encoder/conv4_enc/BiasAdd/ReadVariableOp’'Encoder/conv4_enc/Conv2D/ReadVariableOp’(Encoder/z_log_var/BiasAdd/ReadVariableOp’'Encoder/z_log_var/MatMul/ReadVariableOp’%Encoder/z_mean/BiasAdd/ReadVariableOp’$Encoder/z_mean/MatMul/ReadVariableOpΛ
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOpΩ
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2DΒ
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
Encoder/conv1_enc/BiasAdd
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
Encoder/conv1_enc/ReluΡ
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPoolΛ
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOpτ
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2DΒ
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Encoder/conv2_enc/BiasAdd
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Encoder/conv2_enc/ReluΡ
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:?????????

 *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPoolΛ
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOpτ
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2DΒ
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@2
Encoder/conv3_enc/BiasAdd
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????

@2
Encoder/conv3_enc/ReluΡ
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPoolΜ
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOpυ
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2DΓ
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOpΡ
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Encoder/conv4_enc/BiasAdd
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Encoder/conv4_enc/Relu?
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????  2
Encoder/flatten/Const³
Encoder/flatten/ReshapeReshape!Encoder/maxpool4/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:?????????	2
Encoder/flatten/ReshapeΗ
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOpΖ
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/MatMulΕ
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOpΝ
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/BiasAddΊ
$Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Encoder/z_mean/MatMul/ReadVariableOp½
Encoder/z_mean/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0,Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_mean/MatMulΉ
%Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Encoder/z_mean/BiasAdd/ReadVariableOp½
Encoder/z_mean/BiasAddBiasAddEncoder/z_mean/MatMul:product:0-Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_mean/BiasAddΓ
'Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Encoder/z_log_var/MatMul/ReadVariableOpΖ
Encoder/z_log_var/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_log_var/MatMulΒ
(Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/z_log_var/BiasAdd/ReadVariableOpΙ
Encoder/z_log_var/BiasAddBiasAdd"Encoder/z_log_var/MatMul:product:00Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_log_var/BiasAdd
Encoder/sampling_5/ShapeShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling_5/Shape
&Encoder/sampling_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&Encoder/sampling_5/strided_slice/stack
(Encoder/sampling_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_5/strided_slice/stack_1
(Encoder/sampling_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_5/strided_slice/stack_2Τ
 Encoder/sampling_5/strided_sliceStridedSlice!Encoder/sampling_5/Shape:output:0/Encoder/sampling_5/strided_slice/stack:output:01Encoder/sampling_5/strided_slice/stack_1:output:01Encoder/sampling_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Encoder/sampling_5/strided_slice
Encoder/sampling_5/Shape_1ShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling_5/Shape_1
(Encoder/sampling_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling_5/strided_slice_1/stack’
*Encoder/sampling_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*Encoder/sampling_5/strided_slice_1/stack_1’
*Encoder/sampling_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*Encoder/sampling_5/strided_slice_1/stack_2ΰ
"Encoder/sampling_5/strided_slice_1StridedSlice#Encoder/sampling_5/Shape_1:output:01Encoder/sampling_5/strided_slice_1/stack:output:03Encoder/sampling_5/strided_slice_1/stack_1:output:03Encoder/sampling_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"Encoder/sampling_5/strided_slice_1ή
&Encoder/sampling_5/random_normal/shapePack)Encoder/sampling_5/strided_slice:output:0+Encoder/sampling_5/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2(
&Encoder/sampling_5/random_normal/shape
%Encoder/sampling_5/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2'
%Encoder/sampling_5/random_normal/mean
'Encoder/sampling_5/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'Encoder/sampling_5/random_normal/stddev
5Encoder/sampling_5/random_normal/RandomStandardNormalRandomStandardNormal/Encoder/sampling_5/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed2¦ 27
5Encoder/sampling_5/random_normal/RandomStandardNormal
$Encoder/sampling_5/random_normal/mulMul>Encoder/sampling_5/random_normal/RandomStandardNormal:output:00Encoder/sampling_5/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2&
$Encoder/sampling_5/random_normal/mulΰ
 Encoder/sampling_5/random_normalAdd(Encoder/sampling_5/random_normal/mul:z:0.Encoder/sampling_5/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2"
 Encoder/sampling_5/random_normaly
Encoder/sampling_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Encoder/sampling_5/mul/x°
Encoder/sampling_5/mulMul!Encoder/sampling_5/mul/x:output:0"Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling_5/mul
Encoder/sampling_5/ExpExpEncoder/sampling_5/mul:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling_5/Exp―
Encoder/sampling_5/mul_1MulEncoder/sampling_5/Exp:y:0$Encoder/sampling_5/random_normal:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling_5/mul_1ͺ
Encoder/sampling_5/addAddV2Encoder/z_mean/BiasAdd:output:0Encoder/sampling_5/mul_1:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling_5/addΑ
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOp»
Decoder/decoding/MatMulMatMulEncoder/sampling_5/add:z:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
Decoder/decoding/MatMulΐ
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:	*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOpΖ
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
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
%Decoder/reshape/strided_slice/stack_2Β
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
value	B :2!
Decoder/reshape/Reshape/shape/1
Decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
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
Decoder/reshape/Reshape/shapeΓ
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????2
Decoder/reshape/ReshapeΝ
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOpτ
Decoder/conv4_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2DΓ
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOpΡ
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Decoder/conv4_dec/BiasAdd
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
Decoder/upsamp4/Const
Decoder/upsamp4/mulMul&Decoder/upsamp4/strided_slice:output:0Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp4/mul
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighborΜ
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2DΒ
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Decoder/conv3_dec/BiasAdd
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
Decoder/upsamp3/Const
Decoder/upsamp3/mulMul&Decoder/upsamp3/strided_slice:output:0Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp3/mul
,Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv3_dec/Relu:activations:0Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighborΛ
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2DΒ
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv2_dec/BiasAdd
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
Decoder/upsamp2/Const
Decoder/upsamp2/mulMul&Decoder/upsamp2/strided_slice:output:0Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp2/mul
,Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv2_dec/Relu:activations:0Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighborΛ
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2DΒ
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Decoder/conv1_dec/BiasAdd
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
Decoder/upsamp1/Const
Decoder/upsamp1/mulMul&Decoder/upsamp1/strided_slice:output:0Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp1/mul
,Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv1_dec/Relu:activations:0Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????00*
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighborΒ
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingVALID*
strides
2
Decoder/output/Conv2DΉ
%Decoder/output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Decoder/output/BiasAdd/ReadVariableOpΔ
Decoder/output/BiasAddBiasAddDecoder/output/Conv2D:output:0-Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
Decoder/output/BiasAdd
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
Decoder/output/Sigmoid»	
IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp)^Encoder/z_log_var/BiasAdd/ReadVariableOp(^Encoder/z_log_var/MatMul/ReadVariableOp&^Encoder/z_mean/BiasAdd/ReadVariableOp%^Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????((2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::2T
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
:?????????((
 
_user_specified_nameinputs
Κo

D__inference_Decoder_layer_call_and_return_conditional_losses_1231935

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
identity’ conv1_dec/BiasAdd/ReadVariableOp’conv1_dec/Conv2D/ReadVariableOp’ conv2_dec/BiasAdd/ReadVariableOp’conv2_dec/Conv2D/ReadVariableOp’ conv3_dec/BiasAdd/ReadVariableOp’conv3_dec/Conv2D/ReadVariableOp’ conv4_dec/BiasAdd/ReadVariableOp’conv4_dec/Conv2D/ReadVariableOp’decoding/BiasAdd/ReadVariableOp’decoding/MatMul/ReadVariableOp’output/BiasAdd/ReadVariableOp’output/Conv2D/ReadVariableOp©
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02 
decoding/MatMul/ReadVariableOp
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
decoding/MatMul¨
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:	*
dtype02!
decoding/BiasAdd/ReadVariableOp¦
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
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
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
reshape/Reshape/shape/3κ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape£
reshape/ReshapeReshapedecoding/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????2
reshape/Reshape΅
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv4_dec/Conv2D/ReadVariableOpΤ
conv4_dec/Conv2DConv2Dreshape/Reshape:output:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
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
:?????????2
conv4_dec/BiasAdd
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
upsamp4/strided_slice/stack_2ώ
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
upsamp4/mulι
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighbor΄
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv3_dec/Conv2D/ReadVariableOpπ
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv3_dec/Conv2Dͺ
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_dec/BiasAdd/ReadVariableOp°
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
upsamp3/strided_slice/stack_2ώ
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
upsamp3/mulθ
$upsamp3/resize/ResizeNearestNeighborResizeNearestNeighborconv3_dec/Relu:activations:0upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighbor³
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2_dec/Conv2D/ReadVariableOpπ
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2_dec/Conv2Dͺ
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_dec/BiasAdd/ReadVariableOp°
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
upsamp2/strided_slice/stack_2ώ
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
upsamp2/mulθ
$upsamp2/resize/ResizeNearestNeighborResizeNearestNeighborconv2_dec/Relu:activations:0upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighbor³
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv1_dec/Conv2D/ReadVariableOpπ
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1_dec/Conv2Dͺ
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOp°
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
upsamp1/strided_slice/stack_2ώ
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
upsamp1/mulθ
$upsamp1/resize/ResizeNearestNeighborResizeNearestNeighborconv1_dec/Relu:activations:0upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????00*
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighborͺ
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
output/Conv2D/ReadVariableOpθ
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingVALID*
strides
2
output/Conv2D‘
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp€
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
output/Sigmoid
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????((2

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
Ω

ί
F__inference_conv4_dec_layer_call_and_return_conditional_losses_1232309

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
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
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Π

ί
F__inference_conv3_enc_layer_call_and_return_conditional_losses_1232131

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????

@*
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
:?????????

@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????

@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????

@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????

 ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????

 
 
_user_specified_nameinputs
	
ΰ
G__inference_bottleneck_layer_call_and_return_conditional_losses_1229921

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????	
 
_user_specified_nameinputs
Ό
ά
C__inference_output_layer_call_and_return_conditional_losses_1232389

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
Conv2D/ReadVariableOpΆ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
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
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoidͺ
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
½
ρ
%__inference_VAE_layer_call_fn_1231616

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
identity’StatefulPartitionedCallΤ
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
-:+???????????????????????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_VAE_layer_call_and_return_conditional_losses_12310032
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
’
E
)__inference_upsamp1_layer_call_fn_1230319

inputs
identityθ
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp1_layer_call_and_return_conditional_losses_12303132
PartitionedCall
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
ΐ
ς
%__inference_VAE_layer_call_fn_1231115
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
identity’StatefulPartitionedCallΥ
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
-:+???????????????????????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_VAE_layer_call_and_return_conditional_losses_12310032
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_1


+__inference_conv2_enc_layer_call_fn_1232120

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_enc_layer_call_and_return_conditional_losses_12298242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
κ
`
D__inference_reshape_layer_call_and_return_conditional_losses_1232293

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
strided_slice/stack_2β
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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape/shape/3Ί
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????	:P L
(
_output_shapes
:?????????	
 
_user_specified_nameinputs

`
D__inference_upsamp1_layer_call_and_return_conditional_losses_1230313

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
strided_slice/stack_2Ξ
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
mulΥ
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor€
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
Φ	

)__inference_Decoder_layer_call_fn_1232080

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
identity’StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_12306602
StatefulPartitionedCall¨
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
Όκ
.
#__inference__traced_restore_1232965
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
identity_90’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_69’AssignVariableOp_7’AssignVariableOp_70’AssignVariableOp_71’AssignVariableOp_72’AssignVariableOp_73’AssignVariableOp_74’AssignVariableOp_75’AssignVariableOp_76’AssignVariableOp_77’AssignVariableOp_78’AssignVariableOp_79’AssignVariableOp_8’AssignVariableOp_80’AssignVariableOp_81’AssignVariableOp_82’AssignVariableOp_83’AssignVariableOp_84’AssignVariableOp_85’AssignVariableOp_86’AssignVariableOp_87’AssignVariableOp_88’AssignVariableOp_9ζ)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*ς(
valueθ(Bε(ZB3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesΕ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Z*
dtype0*Ι
valueΏBΌZB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesπ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ώ
_output_shapesλ
θ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*h
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

Identity_6‘
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

Identity_9’
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
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
Identity_12ͺ
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
Identity_14ͺ
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
Identity_16ͺ
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
Identity_18ͺ
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
Identity_24ͺ
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
Identity_28ͺ
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
Identity_30ͺ
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
Identity_32ͺ
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
Identity_34ͺ
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
Identity_45΄
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
Identity_48?
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
Identity_62?
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
Identity_71΄
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
Identity_74?
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
Identity_88?
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
Identity_89χ
Identity_90IdentityIdentity_89:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_90"#
identity_90Identity_90:output:0*ϋ
_input_shapesι
ζ: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
Ό
ί
F__inference_conv1_dec_layer_call_and_return_conditional_losses_1232369

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
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
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu±
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
	
ή
E__inference_decoding_layer_call_and_return_conditional_losses_1232270

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????	2

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
ΐ
ς
%__inference_VAE_layer_call_fn_1231058
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
identity’StatefulPartitionedCallΥ
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
-:+???????????????????????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_VAE_layer_call_and_return_conditional_losses_12310032
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*
_input_shapes
:?????????((::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????((
!
_user_specified_name	input_1
Π

ί
F__inference_conv1_enc_layer_call_and_return_conditional_losses_1232091

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
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
:?????????((2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????((2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????((::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????((
 
_user_specified_nameinputs
’
E
)__inference_upsamp3_layer_call_fn_1230281

inputs
identityθ
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp3_layer_call_and_return_conditional_losses_12302752
PartitionedCall
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
ΤA
η
D__inference_Encoder_layer_call_and_return_conditional_losses_1230074
input_encoder
conv1_enc_1230030
conv1_enc_1230032
conv2_enc_1230036
conv2_enc_1230038
conv3_enc_1230042
conv3_enc_1230044
conv4_enc_1230048
conv4_enc_1230050
bottleneck_1230055
bottleneck_1230057
z_mean_1230060
z_mean_1230062
z_log_var_1230065
z_log_var_1230067
identity

identity_1

identity_2’"bottleneck/StatefulPartitionedCall’!conv1_enc/StatefulPartitionedCall’!conv2_enc/StatefulPartitionedCall’!conv3_enc/StatefulPartitionedCall’!conv4_enc/StatefulPartitionedCall’"sampling_5/StatefulPartitionedCall’!z_log_var/StatefulPartitionedCall’z_mean/StatefulPartitionedCall?
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_1230030conv1_enc_1230032*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????((*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_enc_layer_call_and_return_conditional_losses_12297962#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool1_layer_call_and_return_conditional_losses_12297392
maxpool1/PartitionedCallΒ
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_1230036conv2_enc_1230038*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_enc_layer_call_and_return_conditional_losses_12298242#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool2_layer_call_and_return_conditional_losses_12297512
maxpool2/PartitionedCallΒ
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_1230042conv3_enc_1230044*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv3_enc_layer_call_and_return_conditional_losses_12298522#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool3_layer_call_and_return_conditional_losses_12297632
maxpool3/PartitionedCallΓ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_1230048conv4_enc_1230050*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_enc_layer_call_and_return_conditional_losses_12298802#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool4_layer_call_and_return_conditional_losses_12297752
maxpool4/PartitionedCallρ
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_12299032
flatten/PartitionedCallΎ
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_1230055bottleneck_1230057*
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
GPU2*0J 8 *P
fKRI
G__inference_bottleneck_layer_call_and_return_conditional_losses_12299212$
"bottleneck/StatefulPartitionedCall΅
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_1230060z_mean_1230062*
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
GPU2*0J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_12299472 
z_mean/StatefulPartitionedCallΔ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_1230065z_log_var_1230067*
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
GPU2*0J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_12299732#
!z_log_var/StatefulPartitionedCallΔ
"sampling_5/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sampling_5_layer_call_and_return_conditional_losses_12300152$
"sampling_5/StatefulPartitionedCall
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity‘

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1’

Identity_2Identity+sampling_5/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2H
"sampling_5/StatefulPartitionedCall"sampling_5/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????((
'
_user_specified_nameinput_encoder
©
E
)__inference_reshape_layer_call_fn_1232298

inputs
identityΞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_12303632
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????	:P L
(
_output_shapes
:?????????	
 
_user_specified_nameinputs
κ
`
D__inference_reshape_layer_call_and_return_conditional_losses_1230363

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
strided_slice/stack_2β
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
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape/shape/3Ί
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:?????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????	:P L
(
_output_shapes
:?????????	
 
_user_specified_nameinputs
Ό
ί
F__inference_conv2_dec_layer_call_and_return_conditional_losses_1232349

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
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
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu±
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
Κo

D__inference_Decoder_layer_call_and_return_conditional_losses_1232022

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
identity’ conv1_dec/BiasAdd/ReadVariableOp’conv1_dec/Conv2D/ReadVariableOp’ conv2_dec/BiasAdd/ReadVariableOp’conv2_dec/Conv2D/ReadVariableOp’ conv3_dec/BiasAdd/ReadVariableOp’conv3_dec/Conv2D/ReadVariableOp’ conv4_dec/BiasAdd/ReadVariableOp’conv4_dec/Conv2D/ReadVariableOp’decoding/BiasAdd/ReadVariableOp’decoding/MatMul/ReadVariableOp’output/BiasAdd/ReadVariableOp’output/Conv2D/ReadVariableOp©
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02 
decoding/MatMul/ReadVariableOp
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
decoding/MatMul¨
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:	*
dtype02!
decoding/BiasAdd/ReadVariableOp¦
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????	2
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
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
reshape/Reshape/shape/3κ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape£
reshape/ReshapeReshapedecoding/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????2
reshape/Reshape΅
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv4_dec/Conv2D/ReadVariableOpΤ
conv4_dec/Conv2DConv2Dreshape/Reshape:output:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
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
:?????????2
conv4_dec/BiasAdd
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
upsamp4/strided_slice/stack_2ώ
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
upsamp4/mulι
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighbor΄
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv3_dec/Conv2D/ReadVariableOpπ
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv3_dec/Conv2Dͺ
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv3_dec/BiasAdd/ReadVariableOp°
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
upsamp3/strided_slice/stack_2ώ
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
upsamp3/mulθ
$upsamp3/resize/ResizeNearestNeighborResizeNearestNeighborconv3_dec/Relu:activations:0upsamp3/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighbor³
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2_dec/Conv2D/ReadVariableOpπ
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2_dec/Conv2Dͺ
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2_dec/BiasAdd/ReadVariableOp°
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
upsamp2/strided_slice/stack_2ώ
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
upsamp2/mulθ
$upsamp2/resize/ResizeNearestNeighborResizeNearestNeighborconv2_dec/Relu:activations:0upsamp2/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighbor³
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv1_dec/Conv2D/ReadVariableOpπ
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv1_dec/Conv2Dͺ
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOp°
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
upsamp1/strided_slice/stack_2ώ
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
upsamp1/mulθ
$upsamp1/resize/ResizeNearestNeighborResizeNearestNeighborconv1_dec/Relu:activations:0upsamp1/mul:z:0*
T0*/
_output_shapes
:?????????00*
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighborͺ
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype02
output/Conv2D/ReadVariableOpθ
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((*
paddingVALID*
strides
2
output/Conv2D‘
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp€
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????((2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:?????????((2
output/Sigmoid
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????((2

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
Ε
}
(__inference_output_layer_call_fn_1232398

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
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
GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_12304942
StatefulPartitionedCall¨
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
Ό
ί
F__inference_conv2_dec_layer_call_and_return_conditional_losses_1230438

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
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
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu±
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
Γ
ά
)__inference_Encoder_layer_call_fn_1230159
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

identity_2’StatefulPartitionedCallΕ
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
9:?????????:?????????:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_Encoder_layer_call_and_return_conditional_losses_12301242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????((
'
_user_specified_nameinput_encoder
δ

+__inference_z_log_var_layer_call_fn_1232228

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallω
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
GPU2*0J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_12299732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ΐ1
Κ
D__inference_Decoder_layer_call_and_return_conditional_losses_1230511
input_decoder
decoding_1230344
decoding_1230346
conv4_dec_1230393
conv4_dec_1230395
conv3_dec_1230421
conv3_dec_1230423
conv2_dec_1230449
conv2_dec_1230451
conv1_dec_1230477
conv1_dec_1230479
output_1230505
output_1230507
identity’!conv1_dec/StatefulPartitionedCall’!conv2_dec/StatefulPartitionedCall’!conv3_dec/StatefulPartitionedCall’!conv4_dec/StatefulPartitionedCall’ decoding/StatefulPartitionedCall’output/StatefulPartitionedCall’
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_1230344decoding_1230346*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_decoding_layer_call_and_return_conditional_losses_12303332"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_12303632
reshape/PartitionedCallΒ
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv4_dec_1230393conv4_dec_1230395*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_dec_layer_call_and_return_conditional_losses_12303822#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_upsamp4_layer_call_and_return_conditional_losses_12302562
upsamp4/PartitionedCallΣ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_1230421conv3_dec_1230423*
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
GPU2*0J 8 *O
fJRH
F__inference_conv3_dec_layer_call_and_return_conditional_losses_12304102#
!conv3_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp3_layer_call_and_return_conditional_losses_12302752
upsamp3/PartitionedCallΣ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_1230449conv2_dec_1230451*
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
GPU2*0J 8 *O
fJRH
F__inference_conv2_dec_layer_call_and_return_conditional_losses_12304382#
!conv2_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp2_layer_call_and_return_conditional_losses_12302942
upsamp2/PartitionedCallΣ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_1230477conv1_dec_1230479*
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
GPU2*0J 8 *O
fJRH
F__inference_conv1_dec_layer_call_and_return_conditional_losses_12304662#
!conv1_dec/StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp1_layer_call_and_return_conditional_losses_12303132
upsamp1/PartitionedCallΔ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_1230505output_1230507*
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
GPU2*0J 8 *L
fGRE
C__inference_output_layer_call_and_return_conditional_losses_12304942 
output/StatefulPartitionedCallι
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
Ό
ί
F__inference_conv1_dec_layer_call_and_return_conditional_losses_1230466

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
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
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu±
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
ΤA
η
D__inference_Encoder_layer_call_and_return_conditional_losses_1230027
input_encoder
conv1_enc_1229807
conv1_enc_1229809
conv2_enc_1229835
conv2_enc_1229837
conv3_enc_1229863
conv3_enc_1229865
conv4_enc_1229891
conv4_enc_1229893
bottleneck_1229932
bottleneck_1229934
z_mean_1229958
z_mean_1229960
z_log_var_1229984
z_log_var_1229986
identity

identity_1

identity_2’"bottleneck/StatefulPartitionedCall’!conv1_enc/StatefulPartitionedCall’!conv2_enc/StatefulPartitionedCall’!conv3_enc/StatefulPartitionedCall’!conv4_enc/StatefulPartitionedCall’"sampling_5/StatefulPartitionedCall’!z_log_var/StatefulPartitionedCall’z_mean/StatefulPartitionedCall?
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_1229807conv1_enc_1229809*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????((*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv1_enc_layer_call_and_return_conditional_losses_12297962#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool1_layer_call_and_return_conditional_losses_12297392
maxpool1/PartitionedCallΒ
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_1229835conv2_enc_1229837*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2_enc_layer_call_and_return_conditional_losses_12298242#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool2_layer_call_and_return_conditional_losses_12297512
maxpool2/PartitionedCallΒ
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_1229863conv3_enc_1229865*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????

@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv3_enc_layer_call_and_return_conditional_losses_12298522#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool3_layer_call_and_return_conditional_losses_12297632
maxpool3/PartitionedCallΓ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_1229891conv4_enc_1229893*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv4_enc_layer_call_and_return_conditional_losses_12298802#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_maxpool4_layer_call_and_return_conditional_losses_12297752
maxpool4/PartitionedCallρ
flatten/PartitionedCallPartitionedCall!maxpool4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_12299032
flatten/PartitionedCallΎ
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_1229932bottleneck_1229934*
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
GPU2*0J 8 *P
fKRI
G__inference_bottleneck_layer_call_and_return_conditional_losses_12299212$
"bottleneck/StatefulPartitionedCall΅
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_1229958z_mean_1229960*
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
GPU2*0J 8 *L
fGRE
C__inference_z_mean_layer_call_and_return_conditional_losses_12299472 
z_mean/StatefulPartitionedCallΔ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_1229984z_log_var_1229986*
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
GPU2*0J 8 *O
fJRH
F__inference_z_log_var_layer_call_and_return_conditional_losses_12299732#
!z_log_var/StatefulPartitionedCallΔ
"sampling_5/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sampling_5_layer_call_and_return_conditional_losses_12300152$
"sampling_5/StatefulPartitionedCall
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity‘

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1’

Identity_2Identity+sampling_5/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall#^sampling_5/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*f
_input_shapesU
S:?????????((::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2H
"sampling_5/StatefulPartitionedCall"sampling_5/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:^ Z
/
_output_shapes
:?????????((
'
_user_specified_nameinput_encoder
Φ	

)__inference_Decoder_layer_call_fn_1232051

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
identity’StatefulPartitionedCall
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
GPU2*0J 8 *M
fHRF
D__inference_Decoder_layer_call_and_return_conditional_losses_12305922
StatefulPartitionedCall¨
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
’
E
)__inference_upsamp2_layer_call_fn_1230300

inputs
identityθ
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp2_layer_call_and_return_conditional_losses_12302942
PartitionedCall
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
’
E
)__inference_upsamp4_layer_call_fn_1230262

inputs
identityθ
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
GPU2*0J 8 *M
fHRF
D__inference_upsamp4_layer_call_and_return_conditional_losses_12302562
PartitionedCall
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
¬
u
,__inference_sampling_5_layer_call_fn_1232260
inputs_0
inputs_1
identity’StatefulPartitionedCallν
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sampling_5_layer_call_and_return_conditional_losses_12300152
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1"±L
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
serving_default_input_1:0?????????((D
output_18
StatefulPartitionedCall:0?????????((tensorflow/serving/predict:‘¨
ύ
encoder
decoder
total_loss_tracker
reconstruction_loss_tracker
kl_loss_tracker
	optimizer
loss
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
υ_default_save_signature
+φ&call_and_return_all_conditional_losses
χ__call__"½
_tf_keras_model£{"class_name": "VAEInternal", "name": "VAE", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "VAEInternal"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Σo
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
	variables
trainable_variables
regularization_losses
	keras_api
+ψ&call_and_return_all_conditional_losses
ω__call__"?k
_tf_keras_networkΆk{"class_name": "Functional", "name": "Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 40, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling_5", "trainable": true, "dtype": "float32"}, "name": "sampling_5", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling_5", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 40, 40, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 40, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 40, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling_5", "trainable": true, "dtype": "float32"}, "name": "sampling_5", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling_5", 0, 0]]}}}
½d
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
+	variables
,trainable_variables
-regularization_losses
.	keras_api
+ϊ&call_and_return_all_conditional_losses
ϋ__call__"ς`
_tf_keras_networkΦ`{"class_name": "Functional", "name": "Decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 1152, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 16]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 1152, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}}}
Η
	/total
	0count
1	variables
2	keras_api"
_tf_keras_metricv{"class_name": "Mean", "name": "total_loss", "dtype": "float32", "config": {"name": "total_loss", "dtype": "float32"}}
Ϊ
	3total
	4count
5	variables
6	keras_api"£
_tf_keras_metric{"class_name": "Mean", "name": "reconstruction_loss", "dtype": "float32", "config": {"name": "reconstruction_loss", "dtype": "float32"}}
Α
	7total
	8count
9	variables
:	keras_api"
_tf_keras_metricp{"class_name": "Mean", "name": "kl_loss", "dtype": "float32", "config": {"name": "kl_loss", "dtype": "float32"}}
Ϋ
;iter

<beta_1

=beta_2
	>decay
?learning_rate@mΑAmΒBmΓCmΔDmΕEmΖFmΗGmΘHmΙImΚJmΛKmΜLmΝMmΞNmΟOmΠPmΡQm?RmΣSmΤTmΥUmΦVmΧWmΨXmΩYmΪ@vΫAvάBvέCvήDvίEvΰFvαGvβHvγIvδJvεKvζLvηMvθNvιOvκPvλQvμRvνSvξTvοUvπVvρWvςXvσYvτ"
	optimizer
 "
trackable_dict_wrapper
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
ζ
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
Ξ
	variables

Zlayers
	trainable_variables
[non_trainable_variables
\metrics

regularization_losses
]layer_metrics
^layer_regularization_losses
χ__call__
υ_default_save_signature
+φ&call_and_return_all_conditional_losses
'φ"call_and_return_conditional_losses"
_generic_user_object
-
όserving_default"
signature_map
"
_tf_keras_input_layerβ{"class_name": "InputLayer", "name": "input_encoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 40, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 40, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}}
τ	

@kernel
Abias
_	variables
`trainable_variables
aregularization_losses
b	keras_api
+ύ&call_and_return_all_conditional_losses
ώ__call__"Ν
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv1_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 40, 1]}}
ς
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
+?&call_and_return_all_conditional_losses
__call__"α
_tf_keras_layerΗ{"class_name": "MaxPooling2D", "name": "maxpool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
φ	

Bkernel
Cbias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
+&call_and_return_all_conditional_losses
__call__"Ο
_tf_keras_layer΅{"class_name": "Conv2D", "name": "conv2_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 16]}}
ς
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
+&call_and_return_all_conditional_losses
__call__"α
_tf_keras_layerΗ{"class_name": "MaxPooling2D", "name": "maxpool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
φ	

Dkernel
Ebias
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
+&call_and_return_all_conditional_losses
__call__"Ο
_tf_keras_layer΅{"class_name": "Conv2D", "name": "conv3_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 32]}}
ς
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
+&call_and_return_all_conditional_losses
__call__"α
_tf_keras_layerΗ{"class_name": "MaxPooling2D", "name": "maxpool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
υ	

Fkernel
Gbias
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
+&call_and_return_all_conditional_losses
__call__"Ξ
_tf_keras_layer΄{"class_name": "Conv2D", "name": "conv4_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 64]}}
ς
{	variables
|trainable_variables
}regularization_losses
~	keras_api
+&call_and_return_all_conditional_losses
__call__"α
_tf_keras_layerΗ{"class_name": "MaxPooling2D", "name": "maxpool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
η
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Σ
_tf_keras_layerΉ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


Hkernel
Ibias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Χ
_tf_keras_layer½{"class_name": "Dense", "name": "bottleneck", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
φ

Jkernel
Kbias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Λ
_tf_keras_layer±{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ό

Lkernel
Mbias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ρ
_tf_keras_layer·{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
Ώ
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ͺ
_tf_keras_layer{"class_name": "Sampling", "name": "sampling_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "sampling_5", "trainable": true, "dtype": "float32"}}
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
΅
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses
ω__call__
+ψ&call_and_return_all_conditional_losses
'ψ"call_and_return_conditional_losses"
_generic_user_object
χ"τ
_tf_keras_input_layerΤ{"class_name": "InputLayer", "name": "input_decoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}}
ό

Nkernel
Obias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ρ
_tf_keras_layer·{"class_name": "Dense", "name": "decoding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 1152, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ϋ
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ζ
_tf_keras_layerΜ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [3, 3, 128]}}}
ϋ	

Pkernel
Qbias
 	variables
‘trainable_variables
’regularization_losses
£	keras_api
+&call_and_return_all_conditional_losses
__call__"Π
_tf_keras_layerΆ{"class_name": "Conv2D", "name": "conv4_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 128]}}
Ώ
€	variables
₯trainable_variables
¦regularization_losses
§	keras_api
+&call_and_return_all_conditional_losses
__call__"ͺ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ϊ	

Rkernel
Sbias
¨	variables
©trainable_variables
ͺregularization_losses
«	keras_api
+&call_and_return_all_conditional_losses
 __call__"Ο
_tf_keras_layer΅{"class_name": "Conv2D", "name": "conv3_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 128]}}
Ώ
¬	variables
­trainable_variables
?regularization_losses
―	keras_api
+‘&call_and_return_all_conditional_losses
’__call__"ͺ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ϊ	

Tkernel
Ubias
°	variables
±trainable_variables
²regularization_losses
³	keras_api
+£&call_and_return_all_conditional_losses
€__call__"Ο
_tf_keras_layer΅{"class_name": "Conv2D", "name": "conv2_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 64]}}
Ώ
΄	variables
΅trainable_variables
Άregularization_losses
·	keras_api
+₯&call_and_return_all_conditional_losses
¦__call__"ͺ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ϊ	

Vkernel
Wbias
Έ	variables
Ήtrainable_variables
Ίregularization_losses
»	keras_api
+§&call_and_return_all_conditional_losses
¨__call__"Ο
_tf_keras_layer΅{"class_name": "Conv2D", "name": "conv1_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 32]}}
Ώ
Ό	variables
½trainable_variables
Ύregularization_losses
Ώ	keras_api
+©&call_and_return_all_conditional_losses
ͺ__call__"ͺ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
χ	

Xkernel
Ybias
ΐ	variables
Αtrainable_variables
Βregularization_losses
Γ	keras_api
+«&call_and_return_all_conditional_losses
¬__call__"Μ
_tf_keras_layer²{"class_name": "Conv2D", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [9, 9]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 16]}}
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
΅
+	variables
Δlayers
,trainable_variables
Εnon_trainable_variables
Ζmetrics
-regularization_losses
Ηlayer_metrics
 Θlayer_regularization_losses
ϋ__call__
+ϊ&call_and_return_all_conditional_losses
'ϊ"call_and_return_conditional_losses"
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
*:(2conv1_enc/kernel
:2conv1_enc/bias
*:( 2conv2_enc/kernel
: 2conv2_enc/bias
*:( @2conv3_enc/kernel
:@2conv3_enc/bias
+:)@2conv4_enc/kernel
:2conv4_enc/bias
$:"		2bottleneck/kernel
:2bottleneck/bias
:2z_mean/kernel
:2z_mean/bias
": 2z_log_var/kernel
:2z_log_var/bias
": 		2decoding/kernel
:	2decoding/bias
,:*2conv4_dec/kernel
:2conv4_dec/bias
+:)@2conv3_dec/kernel
:@2conv3_dec/bias
*:(@ 2conv2_dec/kernel
: 2conv2_dec/bias
*:( 2conv1_dec/kernel
:2conv1_dec/bias
':%		2output/kernel
:2output/bias
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
5
0
1
2"
trackable_list_wrapper
V

total_loss
reconstruction_loss
kl_loss"
trackable_dict_wrapper
 "
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
_	variables
Ιlayers
`trainable_variables
Κnon_trainable_variables
Λmetrics
aregularization_losses
Μlayer_metrics
 Νlayer_regularization_losses
ώ__call__
+ύ&call_and_return_all_conditional_losses
'ύ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
c	variables
Ξlayers
dtrainable_variables
Οnon_trainable_variables
Πmetrics
eregularization_losses
Ρlayer_metrics
 ?layer_regularization_losses
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
g	variables
Σlayers
htrainable_variables
Τnon_trainable_variables
Υmetrics
iregularization_losses
Φlayer_metrics
 Χlayer_regularization_losses
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
΅
k	variables
Ψlayers
ltrainable_variables
Ωnon_trainable_variables
Ϊmetrics
mregularization_losses
Ϋlayer_metrics
 άlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
o	variables
έlayers
ptrainable_variables
ήnon_trainable_variables
ίmetrics
qregularization_losses
ΰlayer_metrics
 αlayer_regularization_losses
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
΅
s	variables
βlayers
ttrainable_variables
γnon_trainable_variables
δmetrics
uregularization_losses
εlayer_metrics
 ζlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
w	variables
ηlayers
xtrainable_variables
θnon_trainable_variables
ιmetrics
yregularization_losses
κlayer_metrics
 λlayer_regularization_losses
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
΅
{	variables
μlayers
|trainable_variables
νnon_trainable_variables
ξmetrics
}regularization_losses
οlayer_metrics
 πlayer_regularization_losses
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
	variables
ρlayers
trainable_variables
ςnon_trainable_variables
σmetrics
regularization_losses
τlayer_metrics
 υlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
φlayers
trainable_variables
χnon_trainable_variables
ψmetrics
regularization_losses
ωlayer_metrics
 ϊlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
ϋlayers
trainable_variables
όnon_trainable_variables
ύmetrics
regularization_losses
ώlayer_metrics
 ?layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses
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
Έ
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses
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
Έ
	variables
layers
trainable_variables
non_trainable_variables
metrics
regularization_losses
layer_metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 	variables
layers
‘trainable_variables
non_trainable_variables
metrics
’regularization_losses
layer_metrics
 layer_regularization_losses
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
Έ
€	variables
layers
₯trainable_variables
non_trainable_variables
metrics
¦regularization_losses
layer_metrics
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
¨	variables
layers
©trainable_variables
non_trainable_variables
 metrics
ͺregularization_losses
‘layer_metrics
 ’layer_regularization_losses
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
Έ
¬	variables
£layers
­trainable_variables
€non_trainable_variables
₯metrics
?regularization_losses
¦layer_metrics
 §layer_regularization_losses
’__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
°	variables
¨layers
±trainable_variables
©non_trainable_variables
ͺmetrics
²regularization_losses
«layer_metrics
 ¬layer_regularization_losses
€__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
΄	variables
­layers
΅trainable_variables
?non_trainable_variables
―metrics
Άregularization_losses
°layer_metrics
 ±layer_regularization_losses
¦__call__
+₯&call_and_return_all_conditional_losses
'₯"call_and_return_conditional_losses"
_generic_user_object
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Έ	variables
²layers
Ήtrainable_variables
³non_trainable_variables
΄metrics
Ίregularization_losses
΅layer_metrics
 Άlayer_regularization_losses
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
Έ
Ό	variables
·layers
½trainable_variables
Έnon_trainable_variables
Ήmetrics
Ύregularization_losses
Ίlayer_metrics
 »layer_regularization_losses
ͺ__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ΐ	variables
Όlayers
Αtrainable_variables
½non_trainable_variables
Ύmetrics
Βregularization_losses
Ώlayer_metrics
 ΐlayer_regularization_losses
¬__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
/:-2Adam/conv1_enc/kernel/m
!:2Adam/conv1_enc/bias/m
/:- 2Adam/conv2_enc/kernel/m
!: 2Adam/conv2_enc/bias/m
/:- @2Adam/conv3_enc/kernel/m
!:@2Adam/conv3_enc/bias/m
0:.@2Adam/conv4_enc/kernel/m
": 2Adam/conv4_enc/bias/m
):'		2Adam/bottleneck/kernel/m
": 2Adam/bottleneck/bias/m
$:"2Adam/z_mean/kernel/m
:2Adam/z_mean/bias/m
':%2Adam/z_log_var/kernel/m
!:2Adam/z_log_var/bias/m
':%		2Adam/decoding/kernel/m
!:	2Adam/decoding/bias/m
1:/2Adam/conv4_dec/kernel/m
": 2Adam/conv4_dec/bias/m
0:.@2Adam/conv3_dec/kernel/m
!:@2Adam/conv3_dec/bias/m
/:-@ 2Adam/conv2_dec/kernel/m
!: 2Adam/conv2_dec/bias/m
/:- 2Adam/conv1_dec/kernel/m
!:2Adam/conv1_dec/bias/m
,:*		2Adam/output/kernel/m
:2Adam/output/bias/m
/:-2Adam/conv1_enc/kernel/v
!:2Adam/conv1_enc/bias/v
/:- 2Adam/conv2_enc/kernel/v
!: 2Adam/conv2_enc/bias/v
/:- @2Adam/conv3_enc/kernel/v
!:@2Adam/conv3_enc/bias/v
0:.@2Adam/conv4_enc/kernel/v
": 2Adam/conv4_enc/bias/v
):'		2Adam/bottleneck/kernel/v
": 2Adam/bottleneck/bias/v
$:"2Adam/z_mean/kernel/v
:2Adam/z_mean/bias/v
':%2Adam/z_log_var/kernel/v
!:2Adam/z_log_var/bias/v
':%		2Adam/decoding/kernel/v
!:	2Adam/decoding/bias/v
1:/2Adam/conv4_dec/kernel/v
": 2Adam/conv4_dec/bias/v
0:.@2Adam/conv3_dec/kernel/v
!:@2Adam/conv3_dec/bias/v
/:-@ 2Adam/conv2_dec/kernel/v
!: 2Adam/conv2_dec/bias/v
/:- 2Adam/conv1_dec/kernel/v
!:2Adam/conv1_dec/bias/v
,:*		2Adam/output/kernel/v
:2Adam/output/bias/v
θ2ε
"__inference__wrapped_model_1229733Ύ
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
annotationsͺ *.’+
)&
input_1?????????((
Α2Ύ
@__inference_VAE_layer_call_and_return_conditional_losses_1231342
@__inference_VAE_layer_call_and_return_conditional_losses_1230880
@__inference_VAE_layer_call_and_return_conditional_losses_1231502
@__inference_VAE_layer_call_and_return_conditional_losses_1230940³
ͺ²¦
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
annotationsͺ *
 
Υ2?
%__inference_VAE_layer_call_fn_1231616
%__inference_VAE_layer_call_fn_1231115
%__inference_VAE_layer_call_fn_1231058
%__inference_VAE_layer_call_fn_1231559³
ͺ²¦
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
annotationsͺ *
 
ή2Ϋ
D__inference_Encoder_layer_call_and_return_conditional_losses_1231695
D__inference_Encoder_layer_call_and_return_conditional_losses_1231774
D__inference_Encoder_layer_call_and_return_conditional_losses_1230074
D__inference_Encoder_layer_call_and_return_conditional_losses_1230027ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ς2ο
)__inference_Encoder_layer_call_fn_1231811
)__inference_Encoder_layer_call_fn_1231848
)__inference_Encoder_layer_call_fn_1230243
)__inference_Encoder_layer_call_fn_1230159ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ή2Ϋ
D__inference_Decoder_layer_call_and_return_conditional_losses_1230550
D__inference_Decoder_layer_call_and_return_conditional_losses_1231935
D__inference_Decoder_layer_call_and_return_conditional_losses_1230511
D__inference_Decoder_layer_call_and_return_conditional_losses_1232022ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ς2ο
)__inference_Decoder_layer_call_fn_1230619
)__inference_Decoder_layer_call_fn_1230687
)__inference_Decoder_layer_call_fn_1232080
)__inference_Decoder_layer_call_fn_1232051ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ΜBΙ
%__inference_signature_wrapper_1231182input_1"
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
annotationsͺ *
 
π2ν
F__inference_conv1_enc_layer_call_and_return_conditional_losses_1232091’
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
annotationsͺ *
 
Υ2?
+__inference_conv1_enc_layer_call_fn_1232100’
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
annotationsͺ *
 
­2ͺ
E__inference_maxpool1_layer_call_and_return_conditional_losses_1229739ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
*__inference_maxpool1_layer_call_fn_1229745ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
π2ν
F__inference_conv2_enc_layer_call_and_return_conditional_losses_1232111’
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
annotationsͺ *
 
Υ2?
+__inference_conv2_enc_layer_call_fn_1232120’
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
annotationsͺ *
 
­2ͺ
E__inference_maxpool2_layer_call_and_return_conditional_losses_1229751ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
*__inference_maxpool2_layer_call_fn_1229757ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
π2ν
F__inference_conv3_enc_layer_call_and_return_conditional_losses_1232131’
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
annotationsͺ *
 
Υ2?
+__inference_conv3_enc_layer_call_fn_1232140’
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
annotationsͺ *
 
­2ͺ
E__inference_maxpool3_layer_call_and_return_conditional_losses_1229763ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
*__inference_maxpool3_layer_call_fn_1229769ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
π2ν
F__inference_conv4_enc_layer_call_and_return_conditional_losses_1232151’
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
annotationsͺ *
 
Υ2?
+__inference_conv4_enc_layer_call_fn_1232160’
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
annotationsͺ *
 
­2ͺ
E__inference_maxpool4_layer_call_and_return_conditional_losses_1229775ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
*__inference_maxpool4_layer_call_fn_1229781ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
ξ2λ
D__inference_flatten_layer_call_and_return_conditional_losses_1232166’
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
annotationsͺ *
 
Σ2Π
)__inference_flatten_layer_call_fn_1232171’
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
annotationsͺ *
 
ρ2ξ
G__inference_bottleneck_layer_call_and_return_conditional_losses_1232181’
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
annotationsͺ *
 
Φ2Σ
,__inference_bottleneck_layer_call_fn_1232190’
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
annotationsͺ *
 
ν2κ
C__inference_z_mean_layer_call_and_return_conditional_losses_1232200’
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
annotationsͺ *
 
?2Ο
(__inference_z_mean_layer_call_fn_1232209’
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
annotationsͺ *
 
π2ν
F__inference_z_log_var_layer_call_and_return_conditional_losses_1232219’
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
annotationsͺ *
 
Υ2?
+__inference_z_log_var_layer_call_fn_1232228’
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
annotationsͺ *
 
ρ2ξ
G__inference_sampling_5_layer_call_and_return_conditional_losses_1232254’
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
annotationsͺ *
 
Φ2Σ
,__inference_sampling_5_layer_call_fn_1232260’
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
annotationsͺ *
 
ο2μ
E__inference_decoding_layer_call_and_return_conditional_losses_1232270’
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
annotationsͺ *
 
Τ2Ρ
*__inference_decoding_layer_call_fn_1232279’
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
annotationsͺ *
 
ξ2λ
D__inference_reshape_layer_call_and_return_conditional_losses_1232293’
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
annotationsͺ *
 
Σ2Π
)__inference_reshape_layer_call_fn_1232298’
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
annotationsͺ *
 
π2ν
F__inference_conv4_dec_layer_call_and_return_conditional_losses_1232309’
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
annotationsͺ *
 
Υ2?
+__inference_conv4_dec_layer_call_fn_1232318’
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
annotationsͺ *
 
¬2©
D__inference_upsamp4_layer_call_and_return_conditional_losses_1230256ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
)__inference_upsamp4_layer_call_fn_1230262ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
π2ν
F__inference_conv3_dec_layer_call_and_return_conditional_losses_1232329’
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
annotationsͺ *
 
Υ2?
+__inference_conv3_dec_layer_call_fn_1232338’
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
annotationsͺ *
 
¬2©
D__inference_upsamp3_layer_call_and_return_conditional_losses_1230275ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
)__inference_upsamp3_layer_call_fn_1230281ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
π2ν
F__inference_conv2_dec_layer_call_and_return_conditional_losses_1232349’
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
annotationsͺ *
 
Υ2?
+__inference_conv2_dec_layer_call_fn_1232358’
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
annotationsͺ *
 
¬2©
D__inference_upsamp2_layer_call_and_return_conditional_losses_1230294ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
)__inference_upsamp2_layer_call_fn_1230300ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
π2ν
F__inference_conv1_dec_layer_call_and_return_conditional_losses_1232369’
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
annotationsͺ *
 
Υ2?
+__inference_conv1_dec_layer_call_fn_1232378’
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
annotationsͺ *
 
¬2©
D__inference_upsamp1_layer_call_and_return_conditional_losses_1230313ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
)__inference_upsamp1_layer_call_fn_1230319ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
ν2κ
C__inference_output_layer_call_and_return_conditional_losses_1232389’
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
annotationsͺ *
 
?2Ο
(__inference_output_layer_call_fn_1232398’
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
annotationsͺ *
 Ψ
D__inference_Decoder_layer_call_and_return_conditional_losses_1230511NOPQRSTUVWXY>’;
4’1
'$
input_decoder?????????
p

 
ͺ "?’<
52
0+???????????????????????????
 Ψ
D__inference_Decoder_layer_call_and_return_conditional_losses_1230550NOPQRSTUVWXY>’;
4’1
'$
input_decoder?????????
p 

 
ͺ "?’<
52
0+???????????????????????????
 Ύ
D__inference_Decoder_layer_call_and_return_conditional_losses_1231935vNOPQRSTUVWXY7’4
-’*
 
inputs?????????
p

 
ͺ "-’*
# 
0?????????((
 Ύ
D__inference_Decoder_layer_call_and_return_conditional_losses_1232022vNOPQRSTUVWXY7’4
-’*
 
inputs?????????
p 

 
ͺ "-’*
# 
0?????????((
 °
)__inference_Decoder_layer_call_fn_1230619NOPQRSTUVWXY>’;
4’1
'$
input_decoder?????????
p

 
ͺ "2/+???????????????????????????°
)__inference_Decoder_layer_call_fn_1230687NOPQRSTUVWXY>’;
4’1
'$
input_decoder?????????
p 

 
ͺ "2/+???????????????????????????¨
)__inference_Decoder_layer_call_fn_1232051{NOPQRSTUVWXY7’4
-’*
 
inputs?????????
p

 
ͺ "2/+???????????????????????????¨
)__inference_Decoder_layer_call_fn_1232080{NOPQRSTUVWXY7’4
-’*
 
inputs?????????
p 

 
ͺ "2/+???????????????????????????
D__inference_Encoder_layer_call_and_return_conditional_losses_1230027Δ@ABCDEFGHIJKLMF’C
<’9
/,
input_encoder?????????((
p

 
ͺ "j’g
`]

0/0?????????

0/1?????????

0/2?????????
 
D__inference_Encoder_layer_call_and_return_conditional_losses_1230074Δ@ABCDEFGHIJKLMF’C
<’9
/,
input_encoder?????????((
p 

 
ͺ "j’g
`]

0/0?????????

0/1?????????

0/2?????????
 
D__inference_Encoder_layer_call_and_return_conditional_losses_1231695½@ABCDEFGHIJKLM?’<
5’2
(%
inputs?????????((
p

 
ͺ "j’g
`]

0/0?????????

0/1?????????

0/2?????????
 
D__inference_Encoder_layer_call_and_return_conditional_losses_1231774½@ABCDEFGHIJKLM?’<
5’2
(%
inputs?????????((
p 

 
ͺ "j’g
`]

0/0?????????

0/1?????????

0/2?????????
 β
)__inference_Encoder_layer_call_fn_1230159΄@ABCDEFGHIJKLMF’C
<’9
/,
input_encoder?????????((
p

 
ͺ "ZW

0?????????

1?????????

2?????????β
)__inference_Encoder_layer_call_fn_1230243΄@ABCDEFGHIJKLMF’C
<’9
/,
input_encoder?????????((
p 

 
ͺ "ZW

0?????????

1?????????

2?????????Ϋ
)__inference_Encoder_layer_call_fn_1231811­@ABCDEFGHIJKLM?’<
5’2
(%
inputs?????????((
p

 
ͺ "ZW

0?????????

1?????????

2?????????Ϋ
)__inference_Encoder_layer_call_fn_1231848­@ABCDEFGHIJKLM?’<
5’2
(%
inputs?????????((
p 

 
ͺ "ZW

0?????????

1?????????

2?????????ΰ
@__inference_VAE_layer_call_and_return_conditional_losses_1230880@ABCDEFGHIJKLMNOPQRSTUVWXY<’9
2’/
)&
input_1?????????((
p
ͺ "?’<
52
0+???????????????????????????
 ΰ
@__inference_VAE_layer_call_and_return_conditional_losses_1230940@ABCDEFGHIJKLMNOPQRSTUVWXY<’9
2’/
)&
input_1?????????((
p 
ͺ "?’<
52
0+???????????????????????????
 Ν
@__inference_VAE_layer_call_and_return_conditional_losses_1231342@ABCDEFGHIJKLMNOPQRSTUVWXY;’8
1’.
(%
inputs?????????((
p
ͺ "-’*
# 
0?????????((
 Ν
@__inference_VAE_layer_call_and_return_conditional_losses_1231502@ABCDEFGHIJKLMNOPQRSTUVWXY;’8
1’.
(%
inputs?????????((
p 
ͺ "-’*
# 
0?????????((
 Έ
%__inference_VAE_layer_call_fn_1231058@ABCDEFGHIJKLMNOPQRSTUVWXY<’9
2’/
)&
input_1?????????((
p
ͺ "2/+???????????????????????????Έ
%__inference_VAE_layer_call_fn_1231115@ABCDEFGHIJKLMNOPQRSTUVWXY<’9
2’/
)&
input_1?????????((
p 
ͺ "2/+???????????????????????????·
%__inference_VAE_layer_call_fn_1231559@ABCDEFGHIJKLMNOPQRSTUVWXY;’8
1’.
(%
inputs?????????((
p
ͺ "2/+???????????????????????????·
%__inference_VAE_layer_call_fn_1231616@ABCDEFGHIJKLMNOPQRSTUVWXY;’8
1’.
(%
inputs?????????((
p 
ͺ "2/+???????????????????????????Ί
"__inference__wrapped_model_1229733@ABCDEFGHIJKLMNOPQRSTUVWXY8’5
.’+
)&
input_1?????????((
ͺ ";ͺ8
6
output_1*'
output_1?????????((¨
G__inference_bottleneck_layer_call_and_return_conditional_losses_1232181]HI0’-
&’#
!
inputs?????????	
ͺ "%’"

0?????????
 
,__inference_bottleneck_layer_call_fn_1232190PHI0’-
&’#
!
inputs?????????	
ͺ "?????????Ϋ
F__inference_conv1_dec_layer_call_and_return_conditional_losses_1232369VWI’F
?’<
:7
inputs+??????????????????????????? 
ͺ "?’<
52
0+???????????????????????????
 ³
+__inference_conv1_dec_layer_call_fn_1232378VWI’F
?’<
:7
inputs+??????????????????????????? 
ͺ "2/+???????????????????????????Ά
F__inference_conv1_enc_layer_call_and_return_conditional_losses_1232091l@A7’4
-’*
(%
inputs?????????((
ͺ "-’*
# 
0?????????((
 
+__inference_conv1_enc_layer_call_fn_1232100_@A7’4
-’*
(%
inputs?????????((
ͺ " ?????????((Ϋ
F__inference_conv2_dec_layer_call_and_return_conditional_losses_1232349TUI’F
?’<
:7
inputs+???????????????????????????@
ͺ "?’<
52
0+??????????????????????????? 
 ³
+__inference_conv2_dec_layer_call_fn_1232358TUI’F
?’<
:7
inputs+???????????????????????????@
ͺ "2/+??????????????????????????? Ά
F__inference_conv2_enc_layer_call_and_return_conditional_losses_1232111lBC7’4
-’*
(%
inputs?????????
ͺ "-’*
# 
0????????? 
 
+__inference_conv2_enc_layer_call_fn_1232120_BC7’4
-’*
(%
inputs?????????
ͺ " ????????? ά
F__inference_conv3_dec_layer_call_and_return_conditional_losses_1232329RSJ’G
@’=
;8
inputs,???????????????????????????
ͺ "?’<
52
0+???????????????????????????@
 ΄
+__inference_conv3_dec_layer_call_fn_1232338RSJ’G
@’=
;8
inputs,???????????????????????????
ͺ "2/+???????????????????????????@Ά
F__inference_conv3_enc_layer_call_and_return_conditional_losses_1232131lDE7’4
-’*
(%
inputs?????????

 
ͺ "-’*
# 
0?????????

@
 
+__inference_conv3_enc_layer_call_fn_1232140_DE7’4
-’*
(%
inputs?????????

 
ͺ " ?????????

@Έ
F__inference_conv4_dec_layer_call_and_return_conditional_losses_1232309nPQ8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
+__inference_conv4_dec_layer_call_fn_1232318aPQ8’5
.’+
)&
inputs?????????
ͺ "!?????????·
F__inference_conv4_enc_layer_call_and_return_conditional_losses_1232151mFG7’4
-’*
(%
inputs?????????@
ͺ ".’+
$!
0?????????
 
+__inference_conv4_enc_layer_call_fn_1232160`FG7’4
-’*
(%
inputs?????????@
ͺ "!?????????¦
E__inference_decoding_layer_call_and_return_conditional_losses_1232270]NO/’,
%’"
 
inputs?????????
ͺ "&’#

0?????????	
 ~
*__inference_decoding_layer_call_fn_1232279PNO/’,
%’"
 
inputs?????????
ͺ "?????????	ͺ
D__inference_flatten_layer_call_and_return_conditional_losses_1232166b8’5
.’+
)&
inputs?????????
ͺ "&’#

0?????????	
 
)__inference_flatten_layer_call_fn_1232171U8’5
.’+
)&
inputs?????????
ͺ "?????????	θ
E__inference_maxpool1_layer_call_and_return_conditional_losses_1229739R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 ΐ
*__inference_maxpool1_layer_call_fn_1229745R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????θ
E__inference_maxpool2_layer_call_and_return_conditional_losses_1229751R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 ΐ
*__inference_maxpool2_layer_call_fn_1229757R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????θ
E__inference_maxpool3_layer_call_and_return_conditional_losses_1229763R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 ΐ
*__inference_maxpool3_layer_call_fn_1229769R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????θ
E__inference_maxpool4_layer_call_and_return_conditional_losses_1229775R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 ΐ
*__inference_maxpool4_layer_call_fn_1229781R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????Ψ
C__inference_output_layer_call_and_return_conditional_losses_1232389XYI’F
?’<
:7
inputs+???????????????????????????
ͺ "?’<
52
0+???????????????????????????
 °
(__inference_output_layer_call_fn_1232398XYI’F
?’<
:7
inputs+???????????????????????????
ͺ "2/+???????????????????????????ͺ
D__inference_reshape_layer_call_and_return_conditional_losses_1232293b0’-
&’#
!
inputs?????????	
ͺ ".’+
$!
0?????????
 
)__inference_reshape_layer_call_fn_1232298U0’-
&’#
!
inputs?????????	
ͺ "!?????????Ο
G__inference_sampling_5_layer_call_and_return_conditional_losses_1232254Z’W
P’M
KH
"
inputs/0?????????
"
inputs/1?????????
ͺ "%’"

0?????????
 ¦
,__inference_sampling_5_layer_call_fn_1232260vZ’W
P’M
KH
"
inputs/0?????????
"
inputs/1?????????
ͺ "?????????Θ
%__inference_signature_wrapper_1231182@ABCDEFGHIJKLMNOPQRSTUVWXYC’@
’ 
9ͺ6
4
input_1)&
input_1?????????((";ͺ8
6
output_1*'
output_1?????????((η
D__inference_upsamp1_layer_call_and_return_conditional_losses_1230313R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ώ
)__inference_upsamp1_layer_call_fn_1230319R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????η
D__inference_upsamp2_layer_call_and_return_conditional_losses_1230294R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ώ
)__inference_upsamp2_layer_call_fn_1230300R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????η
D__inference_upsamp3_layer_call_and_return_conditional_losses_1230275R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ώ
)__inference_upsamp3_layer_call_fn_1230281R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????η
D__inference_upsamp4_layer_call_and_return_conditional_losses_1230256R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ώ
)__inference_upsamp4_layer_call_fn_1230262R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????¦
F__inference_z_log_var_layer_call_and_return_conditional_losses_1232219\LM/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 ~
+__inference_z_log_var_layer_call_fn_1232228OLM/’,
%’"
 
inputs?????????
ͺ "?????????£
C__inference_z_mean_layer_call_and_return_conditional_losses_1232200\JK/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 {
(__inference_z_mean_layer_call_fn_1232209OJK/’,
%’"
 
inputs?????????
ͺ "?????????