ча%
М
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
О
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
і
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ј
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
shape:*!
shared_nameconv1_enc/kernel
}
$conv1_enc/kernel/Read/ReadVariableOpReadVariableOpconv1_enc/kernel*&
_output_shapes
:*
dtype0
t
conv1_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_enc/bias
m
"conv1_enc/bias/Read/ReadVariableOpReadVariableOpconv1_enc/bias*
_output_shapes
:*
dtype0

conv2_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2_enc/kernel
}
$conv2_enc/kernel/Read/ReadVariableOpReadVariableOpconv2_enc/kernel*&
_output_shapes
:*
dtype0
t
conv2_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2_enc/bias
m
"conv2_enc/bias/Read/ReadVariableOpReadVariableOpconv2_enc/bias*
_output_shapes
:*
dtype0

conv3_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv3_enc/kernel
}
$conv3_enc/kernel/Read/ReadVariableOpReadVariableOpconv3_enc/kernel*&
_output_shapes
: *
dtype0
t
conv3_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3_enc/bias
m
"conv3_enc/bias/Read/ReadVariableOpReadVariableOpconv3_enc/bias*
_output_shapes
: *
dtype0

conv4_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv4_enc/kernel
}
$conv4_enc/kernel/Read/ReadVariableOpReadVariableOpconv4_enc/kernel*&
_output_shapes
: @*
dtype0
t
conv4_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv4_enc/bias
m
"conv4_enc/bias/Read/ReadVariableOpReadVariableOpconv4_enc/bias*
_output_shapes
:@*
dtype0

conv5_enc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv5_enc/kernel
~
$conv5_enc/kernel/Read/ReadVariableOpReadVariableOpconv5_enc/kernel*'
_output_shapes
:@*
dtype0
u
conv5_enc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv5_enc/bias
n
"conv5_enc/bias/Read/ReadVariableOpReadVariableOpconv5_enc/bias*
_output_shapes	
:*
dtype0

bottleneck/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namebottleneck/kernel
x
%bottleneck/kernel/Read/ReadVariableOpReadVariableOpbottleneck/kernel*
_output_shapes
:	*
dtype0
v
bottleneck/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namebottleneck/bias
o
#bottleneck/bias/Read/ReadVariableOpReadVariableOpbottleneck/bias*
_output_shapes
:*
dtype0
v
z_mean/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namez_mean/kernel
o
!z_mean/kernel/Read/ReadVariableOpReadVariableOpz_mean/kernel*
_output_shapes

:*
dtype0
n
z_mean/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_mean/bias
g
z_mean/bias/Read/ReadVariableOpReadVariableOpz_mean/bias*
_output_shapes
:*
dtype0
|
z_log_var/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namez_log_var/kernel
u
$z_log_var/kernel/Read/ReadVariableOpReadVariableOpz_log_var/kernel*
_output_shapes

:*
dtype0
t
z_log_var/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namez_log_var/bias
m
"z_log_var/bias/Read/ReadVariableOpReadVariableOpz_log_var/bias*
_output_shapes
:*
dtype0
{
decoding/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedecoding/kernel
t
#decoding/kernel/Read/ReadVariableOpReadVariableOpdecoding/kernel*
_output_shapes
:	*
dtype0
s
decoding/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedecoding/bias
l
!decoding/bias/Read/ReadVariableOpReadVariableOpdecoding/bias*
_output_shapes	
:*
dtype0

conv5_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv5_dec/kernel

$conv5_dec/kernel/Read/ReadVariableOpReadVariableOpconv5_dec/kernel*(
_output_shapes
:*
dtype0
u
conv5_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv5_dec/bias
n
"conv5_dec/bias/Read/ReadVariableOpReadVariableOpconv5_dec/bias*
_output_shapes	
:*
dtype0

conv4_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv4_dec/kernel
~
$conv4_dec/kernel/Read/ReadVariableOpReadVariableOpconv4_dec/kernel*'
_output_shapes
:@*
dtype0
t
conv4_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv4_dec/bias
m
"conv4_dec/bias/Read/ReadVariableOpReadVariableOpconv4_dec/bias*
_output_shapes
:@*
dtype0

conv3_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv3_dec/kernel
}
$conv3_dec/kernel/Read/ReadVariableOpReadVariableOpconv3_dec/kernel*&
_output_shapes
:@ *
dtype0
t
conv3_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv3_dec/bias
m
"conv3_dec/bias/Read/ReadVariableOpReadVariableOpconv3_dec/bias*
_output_shapes
: *
dtype0

conv2_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2_dec/kernel
}
$conv2_dec/kernel/Read/ReadVariableOpReadVariableOpconv2_dec/kernel*&
_output_shapes
: *
dtype0
t
conv2_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2_dec/bias
m
"conv2_dec/bias/Read/ReadVariableOpReadVariableOpconv2_dec/bias*
_output_shapes
:*
dtype0

conv1_dec/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv1_dec/kernel
}
$conv1_dec/kernel/Read/ReadVariableOpReadVariableOpconv1_dec/kernel*&
_output_shapes
:*
dtype0
t
conv1_dec/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1_dec/bias
m
"conv1_dec/bias/Read/ReadVariableOpReadVariableOpconv1_dec/bias*
_output_shapes
:*
dtype0
~
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/kernel
w
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*&
_output_shapes
:*
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
shape:*(
shared_nameAdam/conv1_enc/kernel/m

+Adam/conv1_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv1_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_enc/bias/m
{
)Adam/conv1_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/bias/m*
_output_shapes
:*
dtype0

Adam/conv2_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2_enc/kernel/m

+Adam/conv2_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2_enc/bias/m
{
)Adam/conv2_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/bias/m*
_output_shapes
:*
dtype0

Adam/conv3_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv3_enc/kernel/m

+Adam/conv3_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv3_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3_enc/bias/m
{
)Adam/conv3_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/bias/m*
_output_shapes
: *
dtype0

Adam/conv4_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv4_enc/kernel/m

+Adam/conv4_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv4_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv4_enc/bias/m
{
)Adam/conv4_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/bias/m*
_output_shapes
:@*
dtype0

Adam/conv5_enc/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv5_enc/kernel/m

+Adam/conv5_enc/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv5_enc/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv5_enc/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv5_enc/bias/m
|
)Adam/conv5_enc/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv5_enc/bias/m*
_output_shapes	
:*
dtype0

Adam/bottleneck/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/bottleneck/kernel/m

,Adam/bottleneck/kernel/m/Read/ReadVariableOpReadVariableOpAdam/bottleneck/kernel/m*
_output_shapes
:	*
dtype0

Adam/bottleneck/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/bottleneck/bias/m
}
*Adam/bottleneck/bias/m/Read/ReadVariableOpReadVariableOpAdam/bottleneck/bias/m*
_output_shapes
:*
dtype0

Adam/z_mean/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/z_mean/kernel/m
}
(Adam/z_mean/kernel/m/Read/ReadVariableOpReadVariableOpAdam/z_mean/kernel/m*
_output_shapes

:*
dtype0
|
Adam/z_mean/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/z_mean/bias/m
u
&Adam/z_mean/bias/m/Read/ReadVariableOpReadVariableOpAdam/z_mean/bias/m*
_output_shapes
:*
dtype0

Adam/z_log_var/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/z_log_var/kernel/m

+Adam/z_log_var/kernel/m/Read/ReadVariableOpReadVariableOpAdam/z_log_var/kernel/m*
_output_shapes

:*
dtype0

Adam/z_log_var/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/z_log_var/bias/m
{
)Adam/z_log_var/bias/m/Read/ReadVariableOpReadVariableOpAdam/z_log_var/bias/m*
_output_shapes
:*
dtype0

Adam/decoding/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/decoding/kernel/m

*Adam/decoding/kernel/m/Read/ReadVariableOpReadVariableOpAdam/decoding/kernel/m*
_output_shapes
:	*
dtype0

Adam/decoding/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/decoding/bias/m
z
(Adam/decoding/bias/m/Read/ReadVariableOpReadVariableOpAdam/decoding/bias/m*
_output_shapes	
:*
dtype0

Adam/conv5_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv5_dec/kernel/m

+Adam/conv5_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv5_dec/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv5_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv5_dec/bias/m
|
)Adam/conv5_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv5_dec/bias/m*
_output_shapes	
:*
dtype0

Adam/conv4_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv4_dec/kernel/m

+Adam/conv4_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv4_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv4_dec/bias/m
{
)Adam/conv4_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/bias/m*
_output_shapes
:@*
dtype0

Adam/conv3_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv3_dec/kernel/m

+Adam/conv3_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/kernel/m*&
_output_shapes
:@ *
dtype0

Adam/conv3_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3_dec/bias/m
{
)Adam/conv3_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/bias/m*
_output_shapes
: *
dtype0

Adam/conv2_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2_dec/kernel/m

+Adam/conv2_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2_dec/bias/m
{
)Adam/conv2_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/bias/m*
_output_shapes
:*
dtype0

Adam/conv1_dec/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1_dec/kernel/m

+Adam/conv1_dec/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv1_dec/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_dec/bias/m
{
)Adam/conv1_dec/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/bias/m*
_output_shapes
:*
dtype0

Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output/kernel/m

(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*&
_output_shapes
:*
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
shape:*(
shared_nameAdam/conv1_enc/kernel/v

+Adam/conv1_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv1_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_enc/bias/v
{
)Adam/conv1_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_enc/bias/v*
_output_shapes
:*
dtype0

Adam/conv2_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2_enc/kernel/v

+Adam/conv2_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2_enc/bias/v
{
)Adam/conv2_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2_enc/bias/v*
_output_shapes
:*
dtype0

Adam/conv3_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv3_enc/kernel/v

+Adam/conv3_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv3_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3_enc/bias/v
{
)Adam/conv3_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3_enc/bias/v*
_output_shapes
: *
dtype0

Adam/conv4_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv4_enc/kernel/v

+Adam/conv4_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv4_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv4_enc/bias/v
{
)Adam/conv4_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4_enc/bias/v*
_output_shapes
:@*
dtype0

Adam/conv5_enc/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv5_enc/kernel/v

+Adam/conv5_enc/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv5_enc/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv5_enc/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv5_enc/bias/v
|
)Adam/conv5_enc/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv5_enc/bias/v*
_output_shapes	
:*
dtype0

Adam/bottleneck/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/bottleneck/kernel/v

,Adam/bottleneck/kernel/v/Read/ReadVariableOpReadVariableOpAdam/bottleneck/kernel/v*
_output_shapes
:	*
dtype0

Adam/bottleneck/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/bottleneck/bias/v
}
*Adam/bottleneck/bias/v/Read/ReadVariableOpReadVariableOpAdam/bottleneck/bias/v*
_output_shapes
:*
dtype0

Adam/z_mean/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameAdam/z_mean/kernel/v
}
(Adam/z_mean/kernel/v/Read/ReadVariableOpReadVariableOpAdam/z_mean/kernel/v*
_output_shapes

:*
dtype0
|
Adam/z_mean/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/z_mean/bias/v
u
&Adam/z_mean/bias/v/Read/ReadVariableOpReadVariableOpAdam/z_mean/bias/v*
_output_shapes
:*
dtype0

Adam/z_log_var/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/z_log_var/kernel/v

+Adam/z_log_var/kernel/v/Read/ReadVariableOpReadVariableOpAdam/z_log_var/kernel/v*
_output_shapes

:*
dtype0

Adam/z_log_var/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/z_log_var/bias/v
{
)Adam/z_log_var/bias/v/Read/ReadVariableOpReadVariableOpAdam/z_log_var/bias/v*
_output_shapes
:*
dtype0

Adam/decoding/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/decoding/kernel/v

*Adam/decoding/kernel/v/Read/ReadVariableOpReadVariableOpAdam/decoding/kernel/v*
_output_shapes
:	*
dtype0

Adam/decoding/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/decoding/bias/v
z
(Adam/decoding/bias/v/Read/ReadVariableOpReadVariableOpAdam/decoding/bias/v*
_output_shapes	
:*
dtype0

Adam/conv5_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv5_dec/kernel/v

+Adam/conv5_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv5_dec/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv5_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv5_dec/bias/v
|
)Adam/conv5_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv5_dec/bias/v*
_output_shapes	
:*
dtype0

Adam/conv4_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv4_dec/kernel/v

+Adam/conv4_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv4_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv4_dec/bias/v
{
)Adam/conv4_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv4_dec/bias/v*
_output_shapes
:@*
dtype0

Adam/conv3_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv3_dec/kernel/v

+Adam/conv3_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/kernel/v*&
_output_shapes
:@ *
dtype0

Adam/conv3_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv3_dec/bias/v
{
)Adam/conv3_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3_dec/bias/v*
_output_shapes
: *
dtype0

Adam/conv2_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2_dec/kernel/v

+Adam/conv2_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2_dec/bias/v
{
)Adam/conv2_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2_dec/bias/v*
_output_shapes
:*
dtype0

Adam/conv1_dec/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv1_dec/kernel/v

+Adam/conv1_dec/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv1_dec/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv1_dec/bias/v
{
)Adam/conv1_dec/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_dec/bias/v*
_output_shapes
:*
dtype0

Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/output/kernel/v

(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*&
_output_shapes
:*
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
АЙ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ъИ
valueпИBлИ BгИ
у
encoder
decoder
total_loss_tracker
reconstruction_loss_tracker
kl_loss_tracker
	optimizer
loss
trainable_variables
		variables

regularization_losses
	keras_api

signatures
ј
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
layer_with_weights-4
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
trainable_variables
	variables
regularization_losses
 	keras_api
Т
!layer-0
"layer_with_weights-0
"layer-1
#layer-2
$layer_with_weights-1
$layer-3
%layer-4
&layer_with_weights-2
&layer-5
'layer-6
(layer_with_weights-3
(layer-7
)layer-8
*layer_with_weights-4
*layer-9
+layer-10
,layer_with_weights-5
,layer-11
-layer-12
.layer_with_weights-6
.layer-13
/trainable_variables
0	variables
1regularization_losses
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
4
	;total
	<count
=	variables
>	keras_api

?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rateDmэEmюFmяGm№HmёImђJmѓKmєLmѕMmіNmїOmјPmљQmњRmћSmќTm§UmўVmџWmXmYmZm[m\m]m^m_m`mamDvEvFvGvHvIvJvKvLvMvNvOvPvQvRvSvTvUvVvWvXvYv ZvЁ[vЂ\vЃ]vЄ^vЅ_vІ`vЇavЈ
 
ц
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29

D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
330
431
732
833
;34
<35
 
­
trainable_variables

blayers
		variables

regularization_losses
cnon_trainable_variables
dmetrics
elayer_regularization_losses
flayer_metrics
 
 
h

Dkernel
Ebias
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
R
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
h

Fkernel
Gbias
otrainable_variables
p	variables
qregularization_losses
r	keras_api
R
strainable_variables
t	variables
uregularization_losses
v	keras_api
h

Hkernel
Ibias
wtrainable_variables
x	variables
yregularization_losses
z	keras_api
R
{trainable_variables
|	variables
}regularization_losses
~	keras_api
k

Jkernel
Kbias
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
l

Lkernel
Mbias
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
l

Nkernel
Obias
trainable_variables
	variables
regularization_losses
	keras_api
l

Pkernel
Qbias
trainable_variables
	variables
regularization_losses
	keras_api
l

Rkernel
Sbias
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
 	variables
Ёregularization_losses
Ђ	keras_api
v
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
v
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
 
В
trainable_variables
Ѓlayers
	variables
regularization_losses
Єnon_trainable_variables
Ѕmetrics
 Іlayer_regularization_losses
Їlayer_metrics
 
l

Tkernel
Ubias
Јtrainable_variables
Љ	variables
Њregularization_losses
Ћ	keras_api
V
Ќtrainable_variables
­	variables
Ўregularization_losses
Џ	keras_api
l

Vkernel
Wbias
Аtrainable_variables
Б	variables
Вregularization_losses
Г	keras_api
V
Дtrainable_variables
Е	variables
Жregularization_losses
З	keras_api
l

Xkernel
Ybias
Иtrainable_variables
Й	variables
Кregularization_losses
Л	keras_api
V
Мtrainable_variables
Н	variables
Оregularization_losses
П	keras_api
l

Zkernel
[bias
Рtrainable_variables
С	variables
Тregularization_losses
У	keras_api
V
Фtrainable_variables
Х	variables
Цregularization_losses
Ч	keras_api
l

\kernel
]bias
Шtrainable_variables
Щ	variables
Ъregularization_losses
Ы	keras_api
V
Ьtrainable_variables
Э	variables
Юregularization_losses
Я	keras_api
l

^kernel
_bias
аtrainable_variables
б	variables
вregularization_losses
г	keras_api
V
дtrainable_variables
е	variables
жregularization_losses
з	keras_api
l

`kernel
abias
иtrainable_variables
й	variables
кregularization_losses
л	keras_api
f
T0
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11
`12
a13
f
T0
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11
`12
a13
 
В
/trainable_variables
мlayers
0	variables
1regularization_losses
нnon_trainable_variables
оmetrics
 пlayer_regularization_losses
рlayer_metrics
NL
VARIABLE_VALUEtotal3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEcount3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUE

30
41

5	variables
YW
VARIABLE_VALUEtotal_1<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEcount_1<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUE

70
81

9	variables
MK
VARIABLE_VALUEtotal_20kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEcount_20kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUE

;0
<1

=	variables
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
VT
VARIABLE_VALUEconv5_enc/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv5_enc/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbottleneck/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEbottleneck/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEz_mean/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEz_mean/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEz_log_var/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEz_log_var/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdecoding/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdecoding/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv5_dec/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv5_dec/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv4_dec/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv4_dec/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv3_dec/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv3_dec/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2_dec/kernel1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2_dec/bias1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv1_dec/kernel1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv1_dec/bias1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEoutput/kernel1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEoutput/bias1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUE

0
1
*
30
41
72
83
;4
<5

0
1
2
 
6

total_loss
reconstruction_loss
kl_loss

D0
E1

D0
E1
 
В
gtrainable_variables
сlayers
h	variables
iregularization_losses
тnon_trainable_variables
уmetrics
 фlayer_regularization_losses
хlayer_metrics
 
 
 
В
ktrainable_variables
цlayers
l	variables
mregularization_losses
чnon_trainable_variables
шmetrics
 щlayer_regularization_losses
ъlayer_metrics

F0
G1

F0
G1
 
В
otrainable_variables
ыlayers
p	variables
qregularization_losses
ьnon_trainable_variables
эmetrics
 юlayer_regularization_losses
яlayer_metrics
 
 
 
В
strainable_variables
№layers
t	variables
uregularization_losses
ёnon_trainable_variables
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics

H0
I1

H0
I1
 
В
wtrainable_variables
ѕlayers
x	variables
yregularization_losses
іnon_trainable_variables
їmetrics
 јlayer_regularization_losses
љlayer_metrics
 
 
 
В
{trainable_variables
њlayers
|	variables
}regularization_losses
ћnon_trainable_variables
ќmetrics
 §layer_regularization_losses
ўlayer_metrics

J0
K1

J0
K1
 
Д
trainable_variables
џlayers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
 
 
 
Е
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics

L0
M1

L0
M1
 
Е
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
 
 
 
Е
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
 
 
 
Е
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics

N0
O1

N0
O1
 
Е
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics

P0
Q1

P0
Q1
 
Е
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
  layer_regularization_losses
Ёlayer_metrics

R0
S1

R0
S1
 
Е
trainable_variables
Ђlayers
	variables
regularization_losses
Ѓnon_trainable_variables
Єmetrics
 Ѕlayer_regularization_losses
Іlayer_metrics
 
 
 
Е
trainable_variables
Їlayers
 	variables
Ёregularization_losses
Јnon_trainable_variables
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
v
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
14
15
 
 
 
 

T0
U1

T0
U1
 
Е
Јtrainable_variables
Ќlayers
Љ	variables
Њregularization_losses
­non_trainable_variables
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
 
 
 
Е
Ќtrainable_variables
Бlayers
­	variables
Ўregularization_losses
Вnon_trainable_variables
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics

V0
W1

V0
W1
 
Е
Аtrainable_variables
Жlayers
Б	variables
Вregularization_losses
Зnon_trainable_variables
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
 
 
 
Е
Дtrainable_variables
Лlayers
Е	variables
Жregularization_losses
Мnon_trainable_variables
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics

X0
Y1

X0
Y1
 
Е
Иtrainable_variables
Рlayers
Й	variables
Кregularization_losses
Сnon_trainable_variables
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
 
 
 
Е
Мtrainable_variables
Хlayers
Н	variables
Оregularization_losses
Цnon_trainable_variables
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics

Z0
[1

Z0
[1
 
Е
Рtrainable_variables
Ъlayers
С	variables
Тregularization_losses
Ыnon_trainable_variables
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
 
 
 
Е
Фtrainable_variables
Яlayers
Х	variables
Цregularization_losses
аnon_trainable_variables
бmetrics
 вlayer_regularization_losses
гlayer_metrics

\0
]1

\0
]1
 
Е
Шtrainable_variables
дlayers
Щ	variables
Ъregularization_losses
еnon_trainable_variables
жmetrics
 зlayer_regularization_losses
иlayer_metrics
 
 
 
Е
Ьtrainable_variables
йlayers
Э	variables
Юregularization_losses
кnon_trainable_variables
лmetrics
 мlayer_regularization_losses
нlayer_metrics

^0
_1

^0
_1
 
Е
аtrainable_variables
оlayers
б	variables
вregularization_losses
пnon_trainable_variables
рmetrics
 сlayer_regularization_losses
тlayer_metrics
 
 
 
Е
дtrainable_variables
уlayers
е	variables
жregularization_losses
фnon_trainable_variables
хmetrics
 цlayer_regularization_losses
чlayer_metrics

`0
a1

`0
a1
 
Е
иtrainable_variables
шlayers
й	variables
кregularization_losses
щnon_trainable_variables
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
f
!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
yw
VARIABLE_VALUEAdam/conv5_enc/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv5_enc/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/bottleneck/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/bottleneck/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/z_mean/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/z_mean/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/z_log_var/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/z_log_var/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/decoding/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/decoding/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv5_dec/kernel/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv5_dec/bias/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv4_dec/kernel/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv4_dec/bias/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3_dec/kernel/mMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3_dec/bias/mMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2_dec/kernel/mMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2_dec/bias/mMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1_dec/kernel/mMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1_dec/bias/mMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/output/kernel/mMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/output/bias/mMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
yw
VARIABLE_VALUEAdam/conv5_enc/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv5_enc/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/bottleneck/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/bottleneck/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/z_mean/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/z_mean/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/z_log_var/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/z_log_var/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/decoding/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/decoding/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv5_dec/kernel/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv5_dec/bias/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv4_dec/kernel/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv4_dec/bias/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv3_dec/kernel/vMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv3_dec/bias/vMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2_dec/kernel/vMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2_dec/bias/vMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv1_dec/kernel/vMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv1_dec/bias/vMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/output/kernel/vMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/output/bias/vMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:џџџџџџџџџ(*
dtype0*$
shape:џџџџџџџџџ(
ы
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1_enc/kernelconv1_enc/biasconv2_enc/kernelconv2_enc/biasconv3_enc/kernelconv3_enc/biasconv4_enc/kernelconv4_enc/biasconv5_enc/kernelconv5_enc/biasbottleneck/kernelbottleneck/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/biasdecoding/kerneldecoding/biasconv5_dec/kernelconv5_dec/biasconv4_dec/kernelconv4_dec/biasconv3_dec/kernelconv3_dec/biasconv2_dec/kernelconv2_dec/biasconv1_dec/kernelconv1_dec/biasoutput/kerneloutput/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_150762
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
и"
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenametotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp$conv1_enc/kernel/Read/ReadVariableOp"conv1_enc/bias/Read/ReadVariableOp$conv2_enc/kernel/Read/ReadVariableOp"conv2_enc/bias/Read/ReadVariableOp$conv3_enc/kernel/Read/ReadVariableOp"conv3_enc/bias/Read/ReadVariableOp$conv4_enc/kernel/Read/ReadVariableOp"conv4_enc/bias/Read/ReadVariableOp$conv5_enc/kernel/Read/ReadVariableOp"conv5_enc/bias/Read/ReadVariableOp%bottleneck/kernel/Read/ReadVariableOp#bottleneck/bias/Read/ReadVariableOp!z_mean/kernel/Read/ReadVariableOpz_mean/bias/Read/ReadVariableOp$z_log_var/kernel/Read/ReadVariableOp"z_log_var/bias/Read/ReadVariableOp#decoding/kernel/Read/ReadVariableOp!decoding/bias/Read/ReadVariableOp$conv5_dec/kernel/Read/ReadVariableOp"conv5_dec/bias/Read/ReadVariableOp$conv4_dec/kernel/Read/ReadVariableOp"conv4_dec/bias/Read/ReadVariableOp$conv3_dec/kernel/Read/ReadVariableOp"conv3_dec/bias/Read/ReadVariableOp$conv2_dec/kernel/Read/ReadVariableOp"conv2_dec/bias/Read/ReadVariableOp$conv1_dec/kernel/Read/ReadVariableOp"conv1_dec/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOp+Adam/conv1_enc/kernel/m/Read/ReadVariableOp)Adam/conv1_enc/bias/m/Read/ReadVariableOp+Adam/conv2_enc/kernel/m/Read/ReadVariableOp)Adam/conv2_enc/bias/m/Read/ReadVariableOp+Adam/conv3_enc/kernel/m/Read/ReadVariableOp)Adam/conv3_enc/bias/m/Read/ReadVariableOp+Adam/conv4_enc/kernel/m/Read/ReadVariableOp)Adam/conv4_enc/bias/m/Read/ReadVariableOp+Adam/conv5_enc/kernel/m/Read/ReadVariableOp)Adam/conv5_enc/bias/m/Read/ReadVariableOp,Adam/bottleneck/kernel/m/Read/ReadVariableOp*Adam/bottleneck/bias/m/Read/ReadVariableOp(Adam/z_mean/kernel/m/Read/ReadVariableOp&Adam/z_mean/bias/m/Read/ReadVariableOp+Adam/z_log_var/kernel/m/Read/ReadVariableOp)Adam/z_log_var/bias/m/Read/ReadVariableOp*Adam/decoding/kernel/m/Read/ReadVariableOp(Adam/decoding/bias/m/Read/ReadVariableOp+Adam/conv5_dec/kernel/m/Read/ReadVariableOp)Adam/conv5_dec/bias/m/Read/ReadVariableOp+Adam/conv4_dec/kernel/m/Read/ReadVariableOp)Adam/conv4_dec/bias/m/Read/ReadVariableOp+Adam/conv3_dec/kernel/m/Read/ReadVariableOp)Adam/conv3_dec/bias/m/Read/ReadVariableOp+Adam/conv2_dec/kernel/m/Read/ReadVariableOp)Adam/conv2_dec/bias/m/Read/ReadVariableOp+Adam/conv1_dec/kernel/m/Read/ReadVariableOp)Adam/conv1_dec/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp+Adam/conv1_enc/kernel/v/Read/ReadVariableOp)Adam/conv1_enc/bias/v/Read/ReadVariableOp+Adam/conv2_enc/kernel/v/Read/ReadVariableOp)Adam/conv2_enc/bias/v/Read/ReadVariableOp+Adam/conv3_enc/kernel/v/Read/ReadVariableOp)Adam/conv3_enc/bias/v/Read/ReadVariableOp+Adam/conv4_enc/kernel/v/Read/ReadVariableOp)Adam/conv4_enc/bias/v/Read/ReadVariableOp+Adam/conv5_enc/kernel/v/Read/ReadVariableOp)Adam/conv5_enc/bias/v/Read/ReadVariableOp,Adam/bottleneck/kernel/v/Read/ReadVariableOp*Adam/bottleneck/bias/v/Read/ReadVariableOp(Adam/z_mean/kernel/v/Read/ReadVariableOp&Adam/z_mean/bias/v/Read/ReadVariableOp+Adam/z_log_var/kernel/v/Read/ReadVariableOp)Adam/z_log_var/bias/v/Read/ReadVariableOp*Adam/decoding/kernel/v/Read/ReadVariableOp(Adam/decoding/bias/v/Read/ReadVariableOp+Adam/conv5_dec/kernel/v/Read/ReadVariableOp)Adam/conv5_dec/bias/v/Read/ReadVariableOp+Adam/conv4_dec/kernel/v/Read/ReadVariableOp)Adam/conv4_dec/bias/v/Read/ReadVariableOp+Adam/conv3_dec/kernel/v/Read/ReadVariableOp)Adam/conv3_dec/bias/v/Read/ReadVariableOp+Adam/conv2_dec/kernel/v/Read/ReadVariableOp)Adam/conv2_dec/bias/v/Read/ReadVariableOp+Adam/conv1_dec/kernel/v/Read/ReadVariableOp)Adam/conv1_dec/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*r
Tink
i2g	*
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
__inference__traced_save_152468
я
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenametotalcounttotal_1count_1total_2count_2	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv1_enc/kernelconv1_enc/biasconv2_enc/kernelconv2_enc/biasconv3_enc/kernelconv3_enc/biasconv4_enc/kernelconv4_enc/biasconv5_enc/kernelconv5_enc/biasbottleneck/kernelbottleneck/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/biasdecoding/kerneldecoding/biasconv5_dec/kernelconv5_dec/biasconv4_dec/kernelconv4_dec/biasconv3_dec/kernelconv3_dec/biasconv2_dec/kernelconv2_dec/biasconv1_dec/kernelconv1_dec/biasoutput/kerneloutput/biasAdam/conv1_enc/kernel/mAdam/conv1_enc/bias/mAdam/conv2_enc/kernel/mAdam/conv2_enc/bias/mAdam/conv3_enc/kernel/mAdam/conv3_enc/bias/mAdam/conv4_enc/kernel/mAdam/conv4_enc/bias/mAdam/conv5_enc/kernel/mAdam/conv5_enc/bias/mAdam/bottleneck/kernel/mAdam/bottleneck/bias/mAdam/z_mean/kernel/mAdam/z_mean/bias/mAdam/z_log_var/kernel/mAdam/z_log_var/bias/mAdam/decoding/kernel/mAdam/decoding/bias/mAdam/conv5_dec/kernel/mAdam/conv5_dec/bias/mAdam/conv4_dec/kernel/mAdam/conv4_dec/bias/mAdam/conv3_dec/kernel/mAdam/conv3_dec/bias/mAdam/conv2_dec/kernel/mAdam/conv2_dec/bias/mAdam/conv1_dec/kernel/mAdam/conv1_dec/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv1_enc/kernel/vAdam/conv1_enc/bias/vAdam/conv2_enc/kernel/vAdam/conv2_enc/bias/vAdam/conv3_enc/kernel/vAdam/conv3_enc/bias/vAdam/conv4_enc/kernel/vAdam/conv4_enc/bias/vAdam/conv5_enc/kernel/vAdam/conv5_enc/bias/vAdam/bottleneck/kernel/vAdam/bottleneck/bias/vAdam/z_mean/kernel/vAdam/z_mean/bias/vAdam/z_log_var/kernel/vAdam/z_log_var/bias/vAdam/decoding/kernel/vAdam/decoding/bias/vAdam/conv5_dec/kernel/vAdam/conv5_dec/bias/vAdam/conv4_dec/kernel/vAdam/conv4_dec/bias/vAdam/conv3_dec/kernel/vAdam/conv3_dec/bias/vAdam/conv2_dec/kernel/vAdam/conv2_dec/bias/vAdam/conv1_dec/kernel/vAdam/conv1_dec/bias/vAdam/output/kernel/vAdam/output/bias/v*q
Tinj
h2f*
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
"__inference__traced_restore_152781ж

_
C__inference_upsamp1_layer_call_and_return_conditional_losses_149775

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
strided_slice/stack_2Ю
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
mulе
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(2
resize/ResizeNearestNeighborЄ
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
У
А
$__inference_VAE_layer_call_fn_151193

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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityЂStatefulPartitionedCall
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_VAE_layer_call_and_return_conditional_losses_1505592
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
с
~
)__inference_decoding_layer_call_fn_152003

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_1497952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
љ
`
D__inference_maxpool4_layer_call_and_return_conditional_losses_149152

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ц
Б
$__inference_VAE_layer_call_fn_150687
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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityЂStatefulPartitionedCall
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_VAE_layer_call_and_return_conditional_losses_1505592
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ(
!
_user_specified_name	input_1
 
D
(__inference_upsamp4_layer_call_fn_149724

inputs
identityч
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г9

C__inference_Decoder_layer_call_and_return_conditional_losses_150094

inputs
decoding_150052
decoding_150054
conv5_dec_150058
conv5_dec_150060
conv4_dec_150064
conv4_dec_150066
conv3_dec_150070
conv3_dec_150072
conv2_dec_150076
conv2_dec_150078
conv1_dec_150082
conv1_dec_150084
output_150088
output_150090
identityЂ!conv1_dec/StatefulPartitionedCallЂ!conv2_dec/StatefulPartitionedCallЂ!conv3_dec/StatefulPartitionedCallЂ!conv4_dec/StatefulPartitionedCallЂ!conv5_dec/StatefulPartitionedCallЂ decoding/StatefulPartitionedCallЂoutput/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_150052decoding_150054*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_1497952"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1498252
reshape/PartitionedCallП
!conv5_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv5_dec_150058conv5_dec_150060*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_dec_layer_call_and_return_conditional_losses_1498442#
!conv5_dec/StatefulPartitionedCall
upsamp5/PartitionedCallPartitionedCall*conv5_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
upsamp5/PartitionedCallа
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp5/PartitionedCall:output:0conv4_dec_150064conv4_dec_150066*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_1498722#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
upsamp4/PartitionedCallа
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_150070conv3_dec_150072*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_1499002#
!conv3_dec/StatefulPartitionedCall
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
upsamp3/PartitionedCallа
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_150076conv2_dec_150078*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_1499282#
!conv2_dec/StatefulPartitionedCall
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
upsamp2/PartitionedCallа
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_150082conv1_dec_150084*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_1499562#
!conv1_dec/StatefulPartitionedCall
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
upsamp1/PartitionedCallС
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_150088output_150090*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall"^conv5_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2F
!conv5_dec/StatefulPartitionedCall!conv5_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
№
є
(__inference_Encoder_layer_call_fn_151514

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

unknown_14
identity

identity_1

identity_2ЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1496472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs


Д
(__inference_Decoder_layer_call_fn_151784

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
identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_1501722
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
ћ
?__inference_VAE_layer_call_and_return_conditional_losses_150488
input_1
encoder_150423
encoder_150425
encoder_150427
encoder_150429
encoder_150431
encoder_150433
encoder_150435
encoder_150437
encoder_150439
encoder_150441
encoder_150443
encoder_150445
encoder_150447
encoder_150449
encoder_150451
encoder_150453
decoder_150458
decoder_150460
decoder_150462
decoder_150464
decoder_150466
decoder_150468
decoder_150470
decoder_150472
decoder_150474
decoder_150476
decoder_150478
decoder_150480
decoder_150482
decoder_150484
identityЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCallЗ
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_150423encoder_150425encoder_150427encoder_150429encoder_150431encoder_150433encoder_150435encoder_150437encoder_150439encoder_150441encoder_150443encoder_150445encoder_150447encoder_150449encoder_150451encoder_150453*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1496472!
Encoder/StatefulPartitionedCallІ
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_150458decoder_150460decoder_150462decoder_150464decoder_150466decoder_150468decoder_150470decoder_150472decoder_150474decoder_150476decoder_150478decoder_150480decoder_150482decoder_150484*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_1501722!
Decoder/StatefulPartitionedCallк
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ(
!
_user_specified_name	input_1
Щ

*__inference_conv2_dec_layer_call_fn_152102

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_1499282
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs


*__inference_conv1_enc_layer_call_fn_151804

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_1491852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ(::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
еI

C__inference_Encoder_layer_call_and_return_conditional_losses_149647

inputs
conv1_enc_149597
conv1_enc_149599
conv2_enc_149603
conv2_enc_149605
conv3_enc_149609
conv3_enc_149611
conv4_enc_149615
conv4_enc_149617
conv5_enc_149621
conv5_enc_149623
bottleneck_149628
bottleneck_149630
z_mean_149633
z_mean_149635
z_log_var_149638
z_log_var_149640
identity

identity_1

identity_2Ђ"bottleneck/StatefulPartitionedCallЂ!conv1_enc/StatefulPartitionedCallЂ!conv2_enc/StatefulPartitionedCallЂ!conv3_enc/StatefulPartitionedCallЂ!conv4_enc/StatefulPartitionedCallЂ!conv5_enc/StatefulPartitionedCallЂ sampling/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂz_mean/StatefulPartitionedCallЄ
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_149597conv1_enc_149599*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_1491852#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
maxpool1/PartitionedCallП
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_149603conv2_enc_149605*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_1492132#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_1491282
maxpool2/PartitionedCallП
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_149609conv3_enc_149611*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_1492412#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
maxpool3/PartitionedCallП
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_149615conv4_enc_149617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_1492692#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
maxpool4/PartitionedCallР
!conv5_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool4/PartitionedCall:output:0conv5_enc_149621conv5_enc_149623*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_enc_layer_call_and_return_conditional_losses_1492972#
!conv5_enc/StatefulPartitionedCall
maxpool5/PartitionedCallPartitionedCall*conv5_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
maxpool5/PartitionedCall№
flatten/PartitionedCallPartitionedCall!maxpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1493202
flatten/PartitionedCallЛ
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_149628bottleneck_149630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_1493382$
"bottleneck/StatefulPartitionedCallВ
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_149633z_mean_149635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1493642 
z_mean/StatefulPartitionedCallС
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_149638z_log_var_149640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1493902#
!z_log_var/StatefulPartitionedCallН
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1494322"
 sampling/StatefulPartitionedCallМ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityУ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1Т

Identity_2Identity)sampling/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2F
!conv5_enc/StatefulPartitionedCall!conv5_enc/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
Я

о
E__inference_conv3_enc_layer_call_and_return_conditional_losses_151835

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 *
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
:џџџџџџџџџ
 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ
 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
Л
о
E__inference_conv1_dec_layer_call_and_return_conditional_losses_149956

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
_
C__inference_flatten_layer_call_and_return_conditional_losses_151890

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

_
C__inference_upsamp3_layer_call_and_return_conditional_losses_149737

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
strided_slice/stack_2Ю
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
mulе
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(2
resize/ResizeNearestNeighborЄ
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ї
D
(__inference_flatten_layer_call_fn_151895

inputs
identityХ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1493202
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф
Є
?__inference_VAE_layer_call_and_return_conditional_losses_150945

inputs4
0encoder_conv1_enc_conv2d_readvariableop_resource5
1encoder_conv1_enc_biasadd_readvariableop_resource4
0encoder_conv2_enc_conv2d_readvariableop_resource5
1encoder_conv2_enc_biasadd_readvariableop_resource4
0encoder_conv3_enc_conv2d_readvariableop_resource5
1encoder_conv3_enc_biasadd_readvariableop_resource4
0encoder_conv4_enc_conv2d_readvariableop_resource5
1encoder_conv4_enc_biasadd_readvariableop_resource4
0encoder_conv5_enc_conv2d_readvariableop_resource5
1encoder_conv5_enc_biasadd_readvariableop_resource5
1encoder_bottleneck_matmul_readvariableop_resource6
2encoder_bottleneck_biasadd_readvariableop_resource1
-encoder_z_mean_matmul_readvariableop_resource2
.encoder_z_mean_biasadd_readvariableop_resource4
0encoder_z_log_var_matmul_readvariableop_resource5
1encoder_z_log_var_biasadd_readvariableop_resource3
/decoder_decoding_matmul_readvariableop_resource4
0decoder_decoding_biasadd_readvariableop_resource4
0decoder_conv5_dec_conv2d_readvariableop_resource5
1decoder_conv5_dec_biasadd_readvariableop_resource4
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
identityЂ(Decoder/conv1_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv1_dec/Conv2D/ReadVariableOpЂ(Decoder/conv2_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv2_dec/Conv2D/ReadVariableOpЂ(Decoder/conv3_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv3_dec/Conv2D/ReadVariableOpЂ(Decoder/conv4_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv4_dec/Conv2D/ReadVariableOpЂ(Decoder/conv5_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv5_dec/Conv2D/ReadVariableOpЂ'Decoder/decoding/BiasAdd/ReadVariableOpЂ&Decoder/decoding/MatMul/ReadVariableOpЂ%Decoder/output/BiasAdd/ReadVariableOpЂ$Decoder/output/Conv2D/ReadVariableOpЂ)Encoder/bottleneck/BiasAdd/ReadVariableOpЂ(Encoder/bottleneck/MatMul/ReadVariableOpЂ(Encoder/conv1_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv1_enc/Conv2D/ReadVariableOpЂ(Encoder/conv2_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv2_enc/Conv2D/ReadVariableOpЂ(Encoder/conv3_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv3_enc/Conv2D/ReadVariableOpЂ(Encoder/conv4_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv4_enc/Conv2D/ReadVariableOpЂ(Encoder/conv5_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv5_enc/Conv2D/ReadVariableOpЂ(Encoder/z_log_var/BiasAdd/ReadVariableOpЂ'Encoder/z_log_var/MatMul/ReadVariableOpЂ%Encoder/z_mean/BiasAdd/ReadVariableOpЂ$Encoder/z_mean/MatMul/ReadVariableOpЫ
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOpй
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2DТ
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOpа
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Encoder/conv1_enc/BiasAdd
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Encoder/conv1_enc/Reluб
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPoolЫ
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOpє
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2DТ
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOpа
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Encoder/conv2_enc/BiasAdd
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Encoder/conv2_enc/Reluб
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ
*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPoolЫ
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOpє
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 *
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2DТ
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOpа
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
Encoder/conv3_enc/BiasAdd
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
Encoder/conv3_enc/Reluб
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPoolЫ
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOpє
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2DТ
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOpа
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Encoder/conv4_enc/BiasAdd
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Encoder/conv4_enc/Reluб
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool4/MaxPoolЬ
'Encoder/conv5_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Encoder/conv5_enc/Conv2D/ReadVariableOpѕ
Encoder/conv5_enc/Conv2DConv2D!Encoder/maxpool4/MaxPool:output:0/Encoder/conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Encoder/conv5_enc/Conv2DУ
(Encoder/conv5_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Encoder/conv5_enc/BiasAdd/ReadVariableOpб
Encoder/conv5_enc/BiasAddBiasAdd!Encoder/conv5_enc/Conv2D:output:00Encoder/conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Encoder/conv5_enc/BiasAdd
Encoder/conv5_enc/ReluRelu"Encoder/conv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Encoder/conv5_enc/Reluв
Encoder/maxpool5/MaxPoolMaxPool$Encoder/conv5_enc/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool5/MaxPool
Encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Encoder/flatten/ConstГ
Encoder/flatten/ReshapeReshape!Encoder/maxpool5/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Encoder/flatten/ReshapeЧ
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOpЦ
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/bottleneck/MatMulХ
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOpЭ
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/bottleneck/BiasAddК
$Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Encoder/z_mean/MatMul/ReadVariableOpН
Encoder/z_mean/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0,Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/z_mean/MatMulЙ
%Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Encoder/z_mean/BiasAdd/ReadVariableOpН
Encoder/z_mean/BiasAddBiasAddEncoder/z_mean/MatMul:product:0-Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/z_mean/BiasAddУ
'Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Encoder/z_log_var/MatMul/ReadVariableOpЦ
Encoder/z_log_var/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/z_log_var/MatMulТ
(Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/z_log_var/BiasAdd/ReadVariableOpЩ
Encoder/z_log_var/BiasAddBiasAdd"Encoder/z_log_var/MatMul:product:00Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/z_log_var/BiasAdd
Encoder/sampling/ShapeShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling/Shape
$Encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Encoder/sampling/strided_slice/stack
&Encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Encoder/sampling/strided_slice/stack_1
&Encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Encoder/sampling/strided_slice/stack_2Ш
Encoder/sampling/strided_sliceStridedSliceEncoder/sampling/Shape:output:0-Encoder/sampling/strided_slice/stack:output:0/Encoder/sampling/strided_slice/stack_1:output:0/Encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Encoder/sampling/strided_slice
Encoder/sampling/Shape_1ShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling/Shape_1
&Encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&Encoder/sampling/strided_slice_1/stack
(Encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling/strided_slice_1/stack_1
(Encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling/strided_slice_1/stack_2д
 Encoder/sampling/strided_slice_1StridedSlice!Encoder/sampling/Shape_1:output:0/Encoder/sampling/strided_slice_1/stack:output:01Encoder/sampling/strided_slice_1/stack_1:output:01Encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Encoder/sampling/strided_slice_1ж
$Encoder/sampling/random_normal/shapePack'Encoder/sampling/strided_slice:output:0)Encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2&
$Encoder/sampling/random_normal/shape
#Encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#Encoder/sampling/random_normal/mean
%Encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%Encoder/sampling/random_normal/stddev
3Encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal-Encoder/sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2юое25
3Encoder/sampling/random_normal/RandomStandardNormalј
"Encoder/sampling/random_normal/mulMul<Encoder/sampling/random_normal/RandomStandardNormal:output:0.Encoder/sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2$
"Encoder/sampling/random_normal/mulи
Encoder/sampling/random_normalAdd&Encoder/sampling/random_normal/mul:z:0,Encoder/sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2 
Encoder/sampling/random_normalu
Encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Encoder/sampling/mul/xЊ
Encoder/sampling/mulMulEncoder/sampling/mul/x:output:0"Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/sampling/mul
Encoder/sampling/ExpExpEncoder/sampling/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/sampling/ExpЇ
Encoder/sampling/mul_1MulEncoder/sampling/Exp:y:0"Encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/sampling/mul_1Є
Encoder/sampling/addAddV2Encoder/z_mean/BiasAdd:output:0Encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/sampling/addС
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOpЙ
Decoder/decoding/MatMulMatMulEncoder/sampling/add:z:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Decoder/decoding/MatMulР
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOpЦ
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
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
%Decoder/reshape/strided_slice/stack_2Т
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
value	B :2!
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
Decoder/reshape/Reshape/shapeУ
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Decoder/reshape/ReshapeЭ
'Decoder/conv5_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'Decoder/conv5_dec/Conv2D/ReadVariableOpє
Decoder/conv5_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Decoder/conv5_dec/Conv2DУ
(Decoder/conv5_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Decoder/conv5_dec/BiasAdd/ReadVariableOpб
Decoder/conv5_dec/BiasAddBiasAdd!Decoder/conv5_dec/Conv2D:output:00Decoder/conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Decoder/conv5_dec/BiasAdd
Decoder/conv5_dec/ReluRelu"Decoder/conv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Decoder/conv5_dec/Relu
Decoder/upsamp5/ShapeShape$Decoder/conv5_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp5/Shape
#Decoder/upsamp5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp5/strided_slice/stack
%Decoder/upsamp5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp5/strided_slice/stack_1
%Decoder/upsamp5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp5/strided_slice/stack_2Ў
Decoder/upsamp5/strided_sliceStridedSliceDecoder/upsamp5/Shape:output:0,Decoder/upsamp5/strided_slice/stack:output:0.Decoder/upsamp5/strided_slice/stack_1:output:0.Decoder/upsamp5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp5/strided_slice
Decoder/upsamp5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp5/Const
Decoder/upsamp5/mulMul&Decoder/upsamp5/strided_slice:output:0Decoder/upsamp5/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp5/mul
,Decoder/upsamp5/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv5_dec/Relu:activations:0Decoder/upsamp5/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(2.
,Decoder/upsamp5/resize/ResizeNearestNeighborЬ
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOp
Decoder/conv4_dec/Conv2DConv2D=Decoder/upsamp5/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2DТ
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOpа
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Decoder/conv4_dec/BiasAdd
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
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
%Decoder/upsamp4/strided_slice/stack_2Ў
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
Decoder/upsamp4/mul
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighborЫ
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2DТ
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOpа
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Decoder/conv3_dec/BiasAdd
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
%Decoder/upsamp3/strided_slice/stack_2Ў
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
:џџџџџџџџџ *
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighborЫ
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2DТ
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOpа
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Decoder/conv2_dec/BiasAdd
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
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
%Decoder/upsamp2/strided_slice/stack_2Ў
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
:џџџџџџџџџ *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighborЫ
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2DТ
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOpа
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Decoder/conv1_dec/BiasAdd
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
%Decoder/upsamp1/strided_slice/stack_2Ў
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
:џџџџџџџџџ@ *
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighborТ
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides
2
Decoder/output/Conv2DЙ
%Decoder/output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Decoder/output/BiasAdd/ReadVariableOpФ
Decoder/output/BiasAddBiasAddDecoder/output/Conv2D:output:0-Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Decoder/output/BiasAdd
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Decoder/output/Sigmoidх

IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp)^Decoder/conv5_dec/BiasAdd/ReadVariableOp(^Decoder/conv5_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp)^Encoder/conv5_enc/BiasAdd/ReadVariableOp(^Encoder/conv5_enc/Conv2D/ReadVariableOp)^Encoder/z_log_var/BiasAdd/ReadVariableOp(^Encoder/z_log_var/MatMul/ReadVariableOp&^Encoder/z_mean/BiasAdd/ReadVariableOp%^Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ(2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::2T
(Decoder/conv1_dec/BiasAdd/ReadVariableOp(Decoder/conv1_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv1_dec/Conv2D/ReadVariableOp'Decoder/conv1_dec/Conv2D/ReadVariableOp2T
(Decoder/conv2_dec/BiasAdd/ReadVariableOp(Decoder/conv2_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv2_dec/Conv2D/ReadVariableOp'Decoder/conv2_dec/Conv2D/ReadVariableOp2T
(Decoder/conv3_dec/BiasAdd/ReadVariableOp(Decoder/conv3_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv3_dec/Conv2D/ReadVariableOp'Decoder/conv3_dec/Conv2D/ReadVariableOp2T
(Decoder/conv4_dec/BiasAdd/ReadVariableOp(Decoder/conv4_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv4_dec/Conv2D/ReadVariableOp'Decoder/conv4_dec/Conv2D/ReadVariableOp2T
(Decoder/conv5_dec/BiasAdd/ReadVariableOp(Decoder/conv5_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv5_dec/Conv2D/ReadVariableOp'Decoder/conv5_dec/Conv2D/ReadVariableOp2R
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
(Encoder/conv5_enc/BiasAdd/ReadVariableOp(Encoder/conv5_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv5_enc/Conv2D/ReadVariableOp'Encoder/conv5_enc/Conv2D/ReadVariableOp2T
(Encoder/z_log_var/BiasAdd/ReadVariableOp(Encoder/z_log_var/BiasAdd/ReadVariableOp2R
'Encoder/z_log_var/MatMul/ReadVariableOp'Encoder/z_log_var/MatMul/ReadVariableOp2N
%Encoder/z_mean/BiasAdd/ReadVariableOp%Encoder/z_mean/BiasAdd/ReadVariableOp2L
$Encoder/z_mean/MatMul/ReadVariableOp$Encoder/z_mean/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
щ
_
C__inference_reshape_layer_call_and_return_conditional_losses_149825

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
strided_slice/stack_2т
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
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
E
)__inference_maxpool1_layer_call_fn_149122

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
п
F__inference_bottleneck_layer_call_and_return_conditional_losses_149338

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ

*__inference_conv3_dec_layer_call_fn_152082

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_1499002
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

_
C__inference_upsamp4_layer_call_and_return_conditional_losses_149718

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
strided_slice/stack_2Ю
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
mulе
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(2
resize/ResizeNearestNeighborЄ
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
№
є
(__inference_Encoder_layer_call_fn_151473

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

unknown_14
identity

identity_1

identity_2ЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1495532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
Ђ
E
)__inference_maxpool5_layer_call_fn_149170

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъI
І
C__inference_Encoder_layer_call_and_return_conditional_losses_149444
input_encoder
conv1_enc_149196
conv1_enc_149198
conv2_enc_149224
conv2_enc_149226
conv3_enc_149252
conv3_enc_149254
conv4_enc_149280
conv4_enc_149282
conv5_enc_149308
conv5_enc_149310
bottleneck_149349
bottleneck_149351
z_mean_149375
z_mean_149377
z_log_var_149401
z_log_var_149403
identity

identity_1

identity_2Ђ"bottleneck/StatefulPartitionedCallЂ!conv1_enc/StatefulPartitionedCallЂ!conv2_enc/StatefulPartitionedCallЂ!conv3_enc/StatefulPartitionedCallЂ!conv4_enc/StatefulPartitionedCallЂ!conv5_enc/StatefulPartitionedCallЂ sampling/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂz_mean/StatefulPartitionedCallЋ
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_149196conv1_enc_149198*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_1491852#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
maxpool1/PartitionedCallП
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_149224conv2_enc_149226*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_1492132#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_1491282
maxpool2/PartitionedCallП
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_149252conv3_enc_149254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_1492412#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
maxpool3/PartitionedCallП
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_149280conv4_enc_149282*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_1492692#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
maxpool4/PartitionedCallР
!conv5_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool4/PartitionedCall:output:0conv5_enc_149308conv5_enc_149310*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_enc_layer_call_and_return_conditional_losses_1492972#
!conv5_enc/StatefulPartitionedCall
maxpool5/PartitionedCallPartitionedCall*conv5_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
maxpool5/PartitionedCall№
flatten/PartitionedCallPartitionedCall!maxpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1493202
flatten/PartitionedCallЛ
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_149349bottleneck_149351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_1493382$
"bottleneck/StatefulPartitionedCallВ
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_149375z_mean_149377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1493642 
z_mean/StatefulPartitionedCallС
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_149401z_log_var_149403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1493902#
!z_log_var/StatefulPartitionedCallН
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1494322"
 sampling/StatefulPartitionedCallМ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityУ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1Т

Identity_2Identity)sampling/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2F
!conv5_enc/StatefulPartitionedCall!conv5_enc/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:^ Z
/
_output_shapes
:џџџџџџџџџ(
'
_user_specified_nameinput_encoder
е

о
E__inference_conv5_enc_layer_call_and_return_conditional_losses_149297

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

ћ
(__inference_Encoder_layer_call_fn_149686
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

unknown_14
identity

identity_1

identity_2ЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_encoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1496472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:џџџџџџџџџ(
'
_user_specified_nameinput_encoder
Л
о
E__inference_conv2_dec_layer_call_and_return_conditional_losses_152093

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
	
о
E__inference_z_log_var_layer_call_and_return_conditional_losses_151943

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
E
)__inference_maxpool3_layer_call_fn_149146

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
л
B__inference_z_mean_layer_call_and_return_conditional_losses_151924

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я

о
E__inference_conv1_enc_layer_call_and_return_conditional_losses_149185

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
Я

о
E__inference_conv1_enc_layer_call_and_return_conditional_losses_151795

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
е

о
E__inference_conv5_enc_layer_call_and_return_conditional_losses_151875

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ц
Б
$__inference_VAE_layer_call_fn_150622
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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityЂStatefulPartitionedCall
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_VAE_layer_call_and_return_conditional_losses_1505592
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ(
!
_user_specified_name	input_1
Ђ
Ы4
"__inference__traced_restore_152781
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
"assignvariableop_18_conv4_enc_bias(
$assignvariableop_19_conv5_enc_kernel&
"assignvariableop_20_conv5_enc_bias)
%assignvariableop_21_bottleneck_kernel'
#assignvariableop_22_bottleneck_bias%
!assignvariableop_23_z_mean_kernel#
assignvariableop_24_z_mean_bias(
$assignvariableop_25_z_log_var_kernel&
"assignvariableop_26_z_log_var_bias'
#assignvariableop_27_decoding_kernel%
!assignvariableop_28_decoding_bias(
$assignvariableop_29_conv5_dec_kernel&
"assignvariableop_30_conv5_dec_bias(
$assignvariableop_31_conv4_dec_kernel&
"assignvariableop_32_conv4_dec_bias(
$assignvariableop_33_conv3_dec_kernel&
"assignvariableop_34_conv3_dec_bias(
$assignvariableop_35_conv2_dec_kernel&
"assignvariableop_36_conv2_dec_bias(
$assignvariableop_37_conv1_dec_kernel&
"assignvariableop_38_conv1_dec_bias%
!assignvariableop_39_output_kernel#
assignvariableop_40_output_bias/
+assignvariableop_41_adam_conv1_enc_kernel_m-
)assignvariableop_42_adam_conv1_enc_bias_m/
+assignvariableop_43_adam_conv2_enc_kernel_m-
)assignvariableop_44_adam_conv2_enc_bias_m/
+assignvariableop_45_adam_conv3_enc_kernel_m-
)assignvariableop_46_adam_conv3_enc_bias_m/
+assignvariableop_47_adam_conv4_enc_kernel_m-
)assignvariableop_48_adam_conv4_enc_bias_m/
+assignvariableop_49_adam_conv5_enc_kernel_m-
)assignvariableop_50_adam_conv5_enc_bias_m0
,assignvariableop_51_adam_bottleneck_kernel_m.
*assignvariableop_52_adam_bottleneck_bias_m,
(assignvariableop_53_adam_z_mean_kernel_m*
&assignvariableop_54_adam_z_mean_bias_m/
+assignvariableop_55_adam_z_log_var_kernel_m-
)assignvariableop_56_adam_z_log_var_bias_m.
*assignvariableop_57_adam_decoding_kernel_m,
(assignvariableop_58_adam_decoding_bias_m/
+assignvariableop_59_adam_conv5_dec_kernel_m-
)assignvariableop_60_adam_conv5_dec_bias_m/
+assignvariableop_61_adam_conv4_dec_kernel_m-
)assignvariableop_62_adam_conv4_dec_bias_m/
+assignvariableop_63_adam_conv3_dec_kernel_m-
)assignvariableop_64_adam_conv3_dec_bias_m/
+assignvariableop_65_adam_conv2_dec_kernel_m-
)assignvariableop_66_adam_conv2_dec_bias_m/
+assignvariableop_67_adam_conv1_dec_kernel_m-
)assignvariableop_68_adam_conv1_dec_bias_m,
(assignvariableop_69_adam_output_kernel_m*
&assignvariableop_70_adam_output_bias_m/
+assignvariableop_71_adam_conv1_enc_kernel_v-
)assignvariableop_72_adam_conv1_enc_bias_v/
+assignvariableop_73_adam_conv2_enc_kernel_v-
)assignvariableop_74_adam_conv2_enc_bias_v/
+assignvariableop_75_adam_conv3_enc_kernel_v-
)assignvariableop_76_adam_conv3_enc_bias_v/
+assignvariableop_77_adam_conv4_enc_kernel_v-
)assignvariableop_78_adam_conv4_enc_bias_v/
+assignvariableop_79_adam_conv5_enc_kernel_v-
)assignvariableop_80_adam_conv5_enc_bias_v0
,assignvariableop_81_adam_bottleneck_kernel_v.
*assignvariableop_82_adam_bottleneck_bias_v,
(assignvariableop_83_adam_z_mean_kernel_v*
&assignvariableop_84_adam_z_mean_bias_v/
+assignvariableop_85_adam_z_log_var_kernel_v-
)assignvariableop_86_adam_z_log_var_bias_v.
*assignvariableop_87_adam_decoding_kernel_v,
(assignvariableop_88_adam_decoding_bias_v/
+assignvariableop_89_adam_conv5_dec_kernel_v-
)assignvariableop_90_adam_conv5_dec_bias_v/
+assignvariableop_91_adam_conv4_dec_kernel_v-
)assignvariableop_92_adam_conv4_dec_bias_v/
+assignvariableop_93_adam_conv3_dec_kernel_v-
)assignvariableop_94_adam_conv3_dec_bias_v/
+assignvariableop_95_adam_conv2_dec_kernel_v-
)assignvariableop_96_adam_conv2_dec_bias_v/
+assignvariableop_97_adam_conv1_dec_kernel_v-
)assignvariableop_98_adam_conv1_dec_bias_v,
(assignvariableop_99_adam_output_kernel_v+
'assignvariableop_100_adam_output_bias_v
identity_102ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_100ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93ЂAssignVariableOp_94ЂAssignVariableOp_95ЂAssignVariableOp_96ЂAssignVariableOp_97ЂAssignVariableOp_98ЂAssignVariableOp_99Ж6
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*Т5
valueИ5BЕ5fB3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesн
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*с
valueзBдfB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЌ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ў
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*t
dtypesj
h2f	2
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

Identity_6Ё
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ђ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ќ
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv1_enc_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Њ
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv1_enc_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ќ
AssignVariableOp_13AssignVariableOp$assignvariableop_13_conv2_enc_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Њ
AssignVariableOp_14AssignVariableOp"assignvariableop_14_conv2_enc_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ќ
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv3_enc_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Њ
AssignVariableOp_16AssignVariableOp"assignvariableop_16_conv3_enc_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ќ
AssignVariableOp_17AssignVariableOp$assignvariableop_17_conv4_enc_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Њ
AssignVariableOp_18AssignVariableOp"assignvariableop_18_conv4_enc_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ќ
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv5_enc_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Њ
AssignVariableOp_20AssignVariableOp"assignvariableop_20_conv5_enc_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21­
AssignVariableOp_21AssignVariableOp%assignvariableop_21_bottleneck_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ћ
AssignVariableOp_22AssignVariableOp#assignvariableop_22_bottleneck_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Љ
AssignVariableOp_23AssignVariableOp!assignvariableop_23_z_mean_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ї
AssignVariableOp_24AssignVariableOpassignvariableop_24_z_mean_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ќ
AssignVariableOp_25AssignVariableOp$assignvariableop_25_z_log_var_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Њ
AssignVariableOp_26AssignVariableOp"assignvariableop_26_z_log_var_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ћ
AssignVariableOp_27AssignVariableOp#assignvariableop_27_decoding_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Љ
AssignVariableOp_28AssignVariableOp!assignvariableop_28_decoding_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ќ
AssignVariableOp_29AssignVariableOp$assignvariableop_29_conv5_dec_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Њ
AssignVariableOp_30AssignVariableOp"assignvariableop_30_conv5_dec_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ќ
AssignVariableOp_31AssignVariableOp$assignvariableop_31_conv4_dec_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Њ
AssignVariableOp_32AssignVariableOp"assignvariableop_32_conv4_dec_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ќ
AssignVariableOp_33AssignVariableOp$assignvariableop_33_conv3_dec_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Њ
AssignVariableOp_34AssignVariableOp"assignvariableop_34_conv3_dec_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ќ
AssignVariableOp_35AssignVariableOp$assignvariableop_35_conv2_dec_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Њ
AssignVariableOp_36AssignVariableOp"assignvariableop_36_conv2_dec_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ќ
AssignVariableOp_37AssignVariableOp$assignvariableop_37_conv1_dec_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Њ
AssignVariableOp_38AssignVariableOp"assignvariableop_38_conv1_dec_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Љ
AssignVariableOp_39AssignVariableOp!assignvariableop_39_output_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Ї
AssignVariableOp_40AssignVariableOpassignvariableop_40_output_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Г
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv1_enc_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Б
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv1_enc_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Г
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2_enc_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Б
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2_enc_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Г
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv3_enc_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Б
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv3_enc_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Г
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv4_enc_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Б
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv4_enc_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Г
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv5_enc_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Б
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv5_enc_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51Д
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_bottleneck_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52В
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_bottleneck_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53А
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_z_mean_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Ў
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_z_mean_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Г
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_z_log_var_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Б
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_z_log_var_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57В
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_decoding_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58А
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_decoding_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Г
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv5_dec_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Б
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv5_dec_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Г
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv4_dec_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Б
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv4_dec_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Г
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv3_dec_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Б
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv3_dec_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Г
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2_dec_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Б
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2_dec_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Г
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv1_dec_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Б
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv1_dec_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69А
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_output_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ў
AssignVariableOp_70AssignVariableOp&assignvariableop_70_adam_output_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Г
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv1_enc_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Б
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv1_enc_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73Г
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2_enc_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Б
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2_enc_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Г
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv3_enc_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Б
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv3_enc_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Г
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv4_enc_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Б
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv4_enc_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79Г
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_conv5_enc_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80Б
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_conv5_enc_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81Д
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_bottleneck_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82В
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adam_bottleneck_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83А
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_z_mean_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84Ў
AssignVariableOp_84AssignVariableOp&assignvariableop_84_adam_z_mean_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85Г
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_z_log_var_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86Б
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_z_log_var_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87В
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_decoding_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88А
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_decoding_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89Г
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_conv5_dec_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90Б
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_conv5_dec_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91Г
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_conv4_dec_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92Б
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_conv4_dec_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93Г
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_conv3_dec_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94Б
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_conv3_dec_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95Г
AssignVariableOp_95AssignVariableOp+assignvariableop_95_adam_conv2_dec_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96Б
AssignVariableOp_96AssignVariableOp)assignvariableop_96_adam_conv2_dec_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97Г
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_conv1_dec_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98Б
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_conv1_dec_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99А
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_output_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100В
AssignVariableOp_100AssignVariableOp'assignvariableop_100_adam_output_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1009
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_101Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_101
Identity_102IdentityIdentity_101:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_102"%
identity_102Identity_102:output:0*Ћ
_input_shapes
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002*
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
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
и

о
E__inference_conv5_dec_layer_call_and_return_conditional_losses_152033

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
У
|
'__inference_output_layer_call_fn_152142

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Л
о
E__inference_conv3_dec_layer_call_and_return_conditional_losses_149900

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs


*__inference_conv3_enc_layer_call_fn_151844

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_1492412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ
 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs


*__inference_conv4_enc_layer_call_fn_151864

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_1492692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ы

*__inference_conv4_dec_layer_call_fn_152062

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_1498722
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ф
Є
?__inference_VAE_layer_call_and_return_conditional_losses_151128

inputs4
0encoder_conv1_enc_conv2d_readvariableop_resource5
1encoder_conv1_enc_biasadd_readvariableop_resource4
0encoder_conv2_enc_conv2d_readvariableop_resource5
1encoder_conv2_enc_biasadd_readvariableop_resource4
0encoder_conv3_enc_conv2d_readvariableop_resource5
1encoder_conv3_enc_biasadd_readvariableop_resource4
0encoder_conv4_enc_conv2d_readvariableop_resource5
1encoder_conv4_enc_biasadd_readvariableop_resource4
0encoder_conv5_enc_conv2d_readvariableop_resource5
1encoder_conv5_enc_biasadd_readvariableop_resource5
1encoder_bottleneck_matmul_readvariableop_resource6
2encoder_bottleneck_biasadd_readvariableop_resource1
-encoder_z_mean_matmul_readvariableop_resource2
.encoder_z_mean_biasadd_readvariableop_resource4
0encoder_z_log_var_matmul_readvariableop_resource5
1encoder_z_log_var_biasadd_readvariableop_resource3
/decoder_decoding_matmul_readvariableop_resource4
0decoder_decoding_biasadd_readvariableop_resource4
0decoder_conv5_dec_conv2d_readvariableop_resource5
1decoder_conv5_dec_biasadd_readvariableop_resource4
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
identityЂ(Decoder/conv1_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv1_dec/Conv2D/ReadVariableOpЂ(Decoder/conv2_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv2_dec/Conv2D/ReadVariableOpЂ(Decoder/conv3_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv3_dec/Conv2D/ReadVariableOpЂ(Decoder/conv4_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv4_dec/Conv2D/ReadVariableOpЂ(Decoder/conv5_dec/BiasAdd/ReadVariableOpЂ'Decoder/conv5_dec/Conv2D/ReadVariableOpЂ'Decoder/decoding/BiasAdd/ReadVariableOpЂ&Decoder/decoding/MatMul/ReadVariableOpЂ%Decoder/output/BiasAdd/ReadVariableOpЂ$Decoder/output/Conv2D/ReadVariableOpЂ)Encoder/bottleneck/BiasAdd/ReadVariableOpЂ(Encoder/bottleneck/MatMul/ReadVariableOpЂ(Encoder/conv1_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv1_enc/Conv2D/ReadVariableOpЂ(Encoder/conv2_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv2_enc/Conv2D/ReadVariableOpЂ(Encoder/conv3_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv3_enc/Conv2D/ReadVariableOpЂ(Encoder/conv4_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv4_enc/Conv2D/ReadVariableOpЂ(Encoder/conv5_enc/BiasAdd/ReadVariableOpЂ'Encoder/conv5_enc/Conv2D/ReadVariableOpЂ(Encoder/z_log_var/BiasAdd/ReadVariableOpЂ'Encoder/z_log_var/MatMul/ReadVariableOpЂ%Encoder/z_mean/BiasAdd/ReadVariableOpЂ$Encoder/z_mean/MatMul/ReadVariableOpЫ
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOpй
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2DТ
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOpа
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Encoder/conv1_enc/BiasAdd
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Encoder/conv1_enc/Reluб
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPoolЫ
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOpє
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2DТ
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOpа
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Encoder/conv2_enc/BiasAdd
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Encoder/conv2_enc/Reluб
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ
*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPoolЫ
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOpє
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 *
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2DТ
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOpа
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
Encoder/conv3_enc/BiasAdd
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
Encoder/conv3_enc/Reluб
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPoolЫ
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOpє
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2DТ
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOpа
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Encoder/conv4_enc/BiasAdd
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Encoder/conv4_enc/Reluб
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool4/MaxPoolЬ
'Encoder/conv5_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Encoder/conv5_enc/Conv2D/ReadVariableOpѕ
Encoder/conv5_enc/Conv2DConv2D!Encoder/maxpool4/MaxPool:output:0/Encoder/conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Encoder/conv5_enc/Conv2DУ
(Encoder/conv5_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Encoder/conv5_enc/BiasAdd/ReadVariableOpб
Encoder/conv5_enc/BiasAddBiasAdd!Encoder/conv5_enc/Conv2D:output:00Encoder/conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Encoder/conv5_enc/BiasAdd
Encoder/conv5_enc/ReluRelu"Encoder/conv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Encoder/conv5_enc/Reluв
Encoder/maxpool5/MaxPoolMaxPool$Encoder/conv5_enc/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool5/MaxPool
Encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Encoder/flatten/ConstГ
Encoder/flatten/ReshapeReshape!Encoder/maxpool5/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Encoder/flatten/ReshapeЧ
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOpЦ
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/bottleneck/MatMulХ
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOpЭ
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/bottleneck/BiasAddК
$Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Encoder/z_mean/MatMul/ReadVariableOpН
Encoder/z_mean/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0,Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/z_mean/MatMulЙ
%Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Encoder/z_mean/BiasAdd/ReadVariableOpН
Encoder/z_mean/BiasAddBiasAddEncoder/z_mean/MatMul:product:0-Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/z_mean/BiasAddУ
'Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Encoder/z_log_var/MatMul/ReadVariableOpЦ
Encoder/z_log_var/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/z_log_var/MatMulТ
(Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/z_log_var/BiasAdd/ReadVariableOpЩ
Encoder/z_log_var/BiasAddBiasAdd"Encoder/z_log_var/MatMul:product:00Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/z_log_var/BiasAdd
Encoder/sampling/ShapeShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling/Shape
$Encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$Encoder/sampling/strided_slice/stack
&Encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&Encoder/sampling/strided_slice/stack_1
&Encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&Encoder/sampling/strided_slice/stack_2Ш
Encoder/sampling/strided_sliceStridedSliceEncoder/sampling/Shape:output:0-Encoder/sampling/strided_slice/stack:output:0/Encoder/sampling/strided_slice/stack_1:output:0/Encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
Encoder/sampling/strided_slice
Encoder/sampling/Shape_1ShapeEncoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
Encoder/sampling/Shape_1
&Encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&Encoder/sampling/strided_slice_1/stack
(Encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling/strided_slice_1/stack_1
(Encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(Encoder/sampling/strided_slice_1/stack_2д
 Encoder/sampling/strided_slice_1StridedSlice!Encoder/sampling/Shape_1:output:0/Encoder/sampling/strided_slice_1/stack:output:01Encoder/sampling/strided_slice_1/stack_1:output:01Encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Encoder/sampling/strided_slice_1ж
$Encoder/sampling/random_normal/shapePack'Encoder/sampling/strided_slice:output:0)Encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2&
$Encoder/sampling/random_normal/shape
#Encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#Encoder/sampling/random_normal/mean
%Encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%Encoder/sampling/random_normal/stddev
3Encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal-Encoder/sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2ъИЗ25
3Encoder/sampling/random_normal/RandomStandardNormalј
"Encoder/sampling/random_normal/mulMul<Encoder/sampling/random_normal/RandomStandardNormal:output:0.Encoder/sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2$
"Encoder/sampling/random_normal/mulи
Encoder/sampling/random_normalAdd&Encoder/sampling/random_normal/mul:z:0,Encoder/sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2 
Encoder/sampling/random_normalu
Encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Encoder/sampling/mul/xЊ
Encoder/sampling/mulMulEncoder/sampling/mul/x:output:0"Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/sampling/mul
Encoder/sampling/ExpExpEncoder/sampling/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/sampling/ExpЇ
Encoder/sampling/mul_1MulEncoder/sampling/Exp:y:0"Encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/sampling/mul_1Є
Encoder/sampling/addAddV2Encoder/z_mean/BiasAdd:output:0Encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Encoder/sampling/addС
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOpЙ
Decoder/decoding/MatMulMatMulEncoder/sampling/add:z:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Decoder/decoding/MatMulР
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOpЦ
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
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
%Decoder/reshape/strided_slice/stack_2Т
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
value	B :2!
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
Decoder/reshape/Reshape/shapeУ
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Decoder/reshape/ReshapeЭ
'Decoder/conv5_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'Decoder/conv5_dec/Conv2D/ReadVariableOpє
Decoder/conv5_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Decoder/conv5_dec/Conv2DУ
(Decoder/conv5_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Decoder/conv5_dec/BiasAdd/ReadVariableOpб
Decoder/conv5_dec/BiasAddBiasAdd!Decoder/conv5_dec/Conv2D:output:00Decoder/conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Decoder/conv5_dec/BiasAdd
Decoder/conv5_dec/ReluRelu"Decoder/conv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Decoder/conv5_dec/Relu
Decoder/upsamp5/ShapeShape$Decoder/conv5_dec/Relu:activations:0*
T0*
_output_shapes
:2
Decoder/upsamp5/Shape
#Decoder/upsamp5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#Decoder/upsamp5/strided_slice/stack
%Decoder/upsamp5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp5/strided_slice/stack_1
%Decoder/upsamp5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%Decoder/upsamp5/strided_slice/stack_2Ў
Decoder/upsamp5/strided_sliceStridedSliceDecoder/upsamp5/Shape:output:0,Decoder/upsamp5/strided_slice/stack:output:0.Decoder/upsamp5/strided_slice/stack_1:output:0.Decoder/upsamp5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
Decoder/upsamp5/strided_slice
Decoder/upsamp5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Decoder/upsamp5/Const
Decoder/upsamp5/mulMul&Decoder/upsamp5/strided_slice:output:0Decoder/upsamp5/Const:output:0*
T0*
_output_shapes
:2
Decoder/upsamp5/mul
,Decoder/upsamp5/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv5_dec/Relu:activations:0Decoder/upsamp5/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(2.
,Decoder/upsamp5/resize/ResizeNearestNeighborЬ
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOp
Decoder/conv4_dec/Conv2DConv2D=Decoder/upsamp5/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2DТ
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOpа
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Decoder/conv4_dec/BiasAdd
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
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
%Decoder/upsamp4/strided_slice/stack_2Ў
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
Decoder/upsamp4/mul
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighborЫ
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2DТ
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOpа
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Decoder/conv3_dec/BiasAdd
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
%Decoder/upsamp3/strided_slice/stack_2Ў
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
:џџџџџџџџџ *
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighborЫ
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2DТ
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOpа
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Decoder/conv2_dec/BiasAdd
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
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
%Decoder/upsamp2/strided_slice/stack_2Ў
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
:џџџџџџџџџ *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighborЫ
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2DТ
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOpа
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
Decoder/conv1_dec/BiasAdd
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
%Decoder/upsamp1/strided_slice/stack_2Ў
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
:џџџџџџџџџ@ *
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighborТ
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides
2
Decoder/output/Conv2DЙ
%Decoder/output/BiasAdd/ReadVariableOpReadVariableOp.decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Decoder/output/BiasAdd/ReadVariableOpФ
Decoder/output/BiasAddBiasAddDecoder/output/Conv2D:output:0-Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Decoder/output/BiasAdd
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
Decoder/output/Sigmoidх

IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp)^Decoder/conv5_dec/BiasAdd/ReadVariableOp(^Decoder/conv5_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp)^Encoder/conv5_enc/BiasAdd/ReadVariableOp(^Encoder/conv5_enc/Conv2D/ReadVariableOp)^Encoder/z_log_var/BiasAdd/ReadVariableOp(^Encoder/z_log_var/MatMul/ReadVariableOp&^Encoder/z_mean/BiasAdd/ReadVariableOp%^Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ(2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::2T
(Decoder/conv1_dec/BiasAdd/ReadVariableOp(Decoder/conv1_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv1_dec/Conv2D/ReadVariableOp'Decoder/conv1_dec/Conv2D/ReadVariableOp2T
(Decoder/conv2_dec/BiasAdd/ReadVariableOp(Decoder/conv2_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv2_dec/Conv2D/ReadVariableOp'Decoder/conv2_dec/Conv2D/ReadVariableOp2T
(Decoder/conv3_dec/BiasAdd/ReadVariableOp(Decoder/conv3_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv3_dec/Conv2D/ReadVariableOp'Decoder/conv3_dec/Conv2D/ReadVariableOp2T
(Decoder/conv4_dec/BiasAdd/ReadVariableOp(Decoder/conv4_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv4_dec/Conv2D/ReadVariableOp'Decoder/conv4_dec/Conv2D/ReadVariableOp2T
(Decoder/conv5_dec/BiasAdd/ReadVariableOp(Decoder/conv5_dec/BiasAdd/ReadVariableOp2R
'Decoder/conv5_dec/Conv2D/ReadVariableOp'Decoder/conv5_dec/Conv2D/ReadVariableOp2R
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
(Encoder/conv5_enc/BiasAdd/ReadVariableOp(Encoder/conv5_enc/BiasAdd/ReadVariableOp2R
'Encoder/conv5_enc/Conv2D/ReadVariableOp'Encoder/conv5_enc/Conv2D/ReadVariableOp2T
(Encoder/z_log_var/BiasAdd/ReadVariableOp(Encoder/z_log_var/BiasAdd/ReadVariableOp2R
'Encoder/z_log_var/MatMul/ReadVariableOp'Encoder/z_log_var/MatMul/ReadVariableOp2N
%Encoder/z_mean/BiasAdd/ReadVariableOp%Encoder/z_mean/BiasAdd/ReadVariableOp2L
$Encoder/z_mean/MatMul/ReadVariableOp$Encoder/z_mean/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
Б
Р	
C__inference_Decoder_layer_call_and_return_conditional_losses_151616

inputs+
'decoding_matmul_readvariableop_resource,
(decoding_biasadd_readvariableop_resource,
(conv5_dec_conv2d_readvariableop_resource-
)conv5_dec_biasadd_readvariableop_resource,
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
identityЂ conv1_dec/BiasAdd/ReadVariableOpЂconv1_dec/Conv2D/ReadVariableOpЂ conv2_dec/BiasAdd/ReadVariableOpЂconv2_dec/Conv2D/ReadVariableOpЂ conv3_dec/BiasAdd/ReadVariableOpЂconv3_dec/Conv2D/ReadVariableOpЂ conv4_dec/BiasAdd/ReadVariableOpЂconv4_dec/Conv2D/ReadVariableOpЂ conv5_dec/BiasAdd/ReadVariableOpЂconv5_dec/Conv2D/ReadVariableOpЂdecoding/BiasAdd/ReadVariableOpЂdecoding/MatMul/ReadVariableOpЂoutput/BiasAdd/ReadVariableOpЂoutput/Conv2D/ReadVariableOpЉ
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
decoding/MatMul/ReadVariableOp
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoding/MatMulЈ
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
decoding/BiasAdd/ReadVariableOpІ
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
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
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
reshape/Reshape/shape/3ъ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЃ
reshape/ReshapeReshapedecoding/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
reshape/ReshapeЕ
conv5_dec/Conv2D/ReadVariableOpReadVariableOp(conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv5_dec/Conv2D/ReadVariableOpд
conv5_dec/Conv2DConv2Dreshape/Reshape:output:0'conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv5_dec/Conv2DЋ
 conv5_dec/BiasAdd/ReadVariableOpReadVariableOp)conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv5_dec/BiasAdd/ReadVariableOpБ
conv5_dec/BiasAddBiasAddconv5_dec/Conv2D:output:0(conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv5_dec/BiasAdd
conv5_dec/ReluReluconv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv5_dec/Reluj
upsamp5/ShapeShapeconv5_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp5/Shape
upsamp5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp5/strided_slice/stack
upsamp5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp5/strided_slice/stack_1
upsamp5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp5/strided_slice/stack_2ў
upsamp5/strided_sliceStridedSliceupsamp5/Shape:output:0$upsamp5/strided_slice/stack:output:0&upsamp5/strided_slice/stack_1:output:0&upsamp5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp5/strided_sliceo
upsamp5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp5/Const~
upsamp5/mulMulupsamp5/strided_slice:output:0upsamp5/Const:output:0*
T0*
_output_shapes
:2
upsamp5/mulщ
$upsamp5/resize/ResizeNearestNeighborResizeNearestNeighborconv5_dec/Relu:activations:0upsamp5/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(2&
$upsamp5/resize/ResizeNearestNeighborД
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv4_dec/Conv2D/ReadVariableOp№
conv4_dec/Conv2DConv2D5upsamp5/resize/ResizeNearestNeighbor:resized_images:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv4_dec/Conv2DЊ
 conv4_dec/BiasAdd/ReadVariableOpReadVariableOp)conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv4_dec/BiasAdd/ReadVariableOpА
conv4_dec/BiasAddBiasAddconv4_dec/Conv2D:output:0(conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4_dec/BiasAdd~
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
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
upsamp4/strided_slice/stack_2ў
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
upsamp4/mulш
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighborГ
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv3_dec/Conv2D/ReadVariableOp№
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv3_dec/Conv2DЊ
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3_dec/BiasAdd/ReadVariableOpА
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
upsamp3/strided_slice/stack_2ў
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
upsamp3/mulш
$upsamp3/resize/ResizeNearestNeighborResizeNearestNeighborconv3_dec/Relu:activations:0upsamp3/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ *
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighborГ
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_dec/Conv2D/ReadVariableOp№
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2_dec/Conv2DЊ
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2_dec/BiasAdd/ReadVariableOpА
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
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
upsamp2/strided_slice/stack_2ў
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
upsamp2/mulш
$upsamp2/resize/ResizeNearestNeighborResizeNearestNeighborconv2_dec/Relu:activations:0upsamp2/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighborГ
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_dec/Conv2D/ReadVariableOp№
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv1_dec/Conv2DЊ
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOpА
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
upsamp1/strided_slice/stack_2ў
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
upsamp1/mulш
$upsamp1/resize/ResizeNearestNeighborResizeNearestNeighborconv1_dec/Relu:activations:0upsamp1/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighborЊ
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
output/Conv2D/ReadVariableOpш
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides
2
output/Conv2DЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЄ
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
output/SigmoidЩ
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp!^conv5_dec/BiasAdd/ReadVariableOp ^conv5_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ(2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::2D
 conv1_dec/BiasAdd/ReadVariableOp conv1_dec/BiasAdd/ReadVariableOp2B
conv1_dec/Conv2D/ReadVariableOpconv1_dec/Conv2D/ReadVariableOp2D
 conv2_dec/BiasAdd/ReadVariableOp conv2_dec/BiasAdd/ReadVariableOp2B
conv2_dec/Conv2D/ReadVariableOpconv2_dec/Conv2D/ReadVariableOp2D
 conv3_dec/BiasAdd/ReadVariableOp conv3_dec/BiasAdd/ReadVariableOp2B
conv3_dec/Conv2D/ReadVariableOpconv3_dec/Conv2D/ReadVariableOp2D
 conv4_dec/BiasAdd/ReadVariableOp conv4_dec/BiasAdd/ReadVariableOp2B
conv4_dec/Conv2D/ReadVariableOpconv4_dec/Conv2D/ReadVariableOp2D
 conv5_dec/BiasAdd/ReadVariableOp conv5_dec/BiasAdd/ReadVariableOp2B
conv5_dec/Conv2D/ReadVariableOpconv5_dec/Conv2D/ReadVariableOp2B
decoding/BiasAdd/ReadVariableOpdecoding/BiasAdd/ReadVariableOp2@
decoding/MatMul/ReadVariableOpdecoding/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/Conv2D/ReadVariableOpoutput/Conv2D/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ

*__inference_conv1_dec_layer_call_fn_152122

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_1499562
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

q
D__inference_sampling_layer_call_and_return_conditional_losses_149432

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
strided_slice/stack_2т
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
strided_slice_1/stack_2ю
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
random_normal/stddevх
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2ѕщ2$
"random_normal/RandomStandardNormalД
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
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
:џџџџџџџџџ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv2_enc_layer_call_fn_151824

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_1492132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
л
|
'__inference_z_mean_layer_call_fn_151933

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1493642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
о
E__inference_conv1_dec_layer_call_and_return_conditional_losses_152113

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ъI
І
C__inference_Encoder_layer_call_and_return_conditional_losses_149497
input_encoder
conv1_enc_149447
conv1_enc_149449
conv2_enc_149453
conv2_enc_149455
conv3_enc_149459
conv3_enc_149461
conv4_enc_149465
conv4_enc_149467
conv5_enc_149471
conv5_enc_149473
bottleneck_149478
bottleneck_149480
z_mean_149483
z_mean_149485
z_log_var_149488
z_log_var_149490
identity

identity_1

identity_2Ђ"bottleneck/StatefulPartitionedCallЂ!conv1_enc/StatefulPartitionedCallЂ!conv2_enc/StatefulPartitionedCallЂ!conv3_enc/StatefulPartitionedCallЂ!conv4_enc/StatefulPartitionedCallЂ!conv5_enc/StatefulPartitionedCallЂ sampling/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂz_mean/StatefulPartitionedCallЋ
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_149447conv1_enc_149449*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_1491852#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
maxpool1/PartitionedCallП
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_149453conv2_enc_149455*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_1492132#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_1491282
maxpool2/PartitionedCallП
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_149459conv3_enc_149461*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_1492412#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
maxpool3/PartitionedCallП
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_149465conv4_enc_149467*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_1492692#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
maxpool4/PartitionedCallР
!conv5_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool4/PartitionedCall:output:0conv5_enc_149471conv5_enc_149473*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_enc_layer_call_and_return_conditional_losses_1492972#
!conv5_enc/StatefulPartitionedCall
maxpool5/PartitionedCallPartitionedCall*conv5_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
maxpool5/PartitionedCall№
flatten/PartitionedCallPartitionedCall!maxpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1493202
flatten/PartitionedCallЛ
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_149478bottleneck_149480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_1493382$
"bottleneck/StatefulPartitionedCallВ
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_149483z_mean_149485*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1493642 
z_mean/StatefulPartitionedCallС
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_149488z_log_var_149490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1493902#
!z_log_var/StatefulPartitionedCallН
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1494322"
 sampling/StatefulPartitionedCallМ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityУ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1Т

Identity_2Identity)sampling/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2F
!conv5_enc/StatefulPartitionedCall!conv5_enc/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:^ Z
/
_output_shapes
:џџџџџџџџџ(
'
_user_specified_nameinput_encoder
љ
`
D__inference_maxpool3_layer_call_and_return_conditional_losses_149140

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
E
)__inference_maxpool2_layer_call_fn_149134

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_1491282
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ
`
D__inference_maxpool2_layer_call_and_return_conditional_losses_149128

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О
о
E__inference_conv4_dec_layer_call_and_return_conditional_losses_149872

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш9

C__inference_Decoder_layer_call_and_return_conditional_losses_150001
input_decoder
decoding_149806
decoding_149808
conv5_dec_149855
conv5_dec_149857
conv4_dec_149883
conv4_dec_149885
conv3_dec_149911
conv3_dec_149913
conv2_dec_149939
conv2_dec_149941
conv1_dec_149967
conv1_dec_149969
output_149995
output_149997
identityЂ!conv1_dec/StatefulPartitionedCallЂ!conv2_dec/StatefulPartitionedCallЂ!conv3_dec/StatefulPartitionedCallЂ!conv4_dec/StatefulPartitionedCallЂ!conv5_dec/StatefulPartitionedCallЂ decoding/StatefulPartitionedCallЂoutput/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_149806decoding_149808*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_1497952"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1498252
reshape/PartitionedCallП
!conv5_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv5_dec_149855conv5_dec_149857*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_dec_layer_call_and_return_conditional_losses_1498442#
!conv5_dec/StatefulPartitionedCall
upsamp5/PartitionedCallPartitionedCall*conv5_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
upsamp5/PartitionedCallа
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp5/PartitionedCall:output:0conv4_dec_149883conv4_dec_149885*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_1498722#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
upsamp4/PartitionedCallа
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_149911conv3_dec_149913*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_1499002#
!conv3_dec/StatefulPartitionedCall
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
upsamp3/PartitionedCallа
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_149939conv2_dec_149941*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_1499282#
!conv2_dec/StatefulPartitionedCall
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
upsamp2/PartitionedCallа
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_149967conv1_dec_149969*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_1499562#
!conv1_dec/StatefulPartitionedCall
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
upsamp1/PartitionedCallС
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_149995output_149997*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall"^conv5_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2F
!conv5_dec/StatefulPartitionedCall!conv5_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameinput_decoder
Ш9

C__inference_Decoder_layer_call_and_return_conditional_losses_150046
input_decoder
decoding_150004
decoding_150006
conv5_dec_150010
conv5_dec_150012
conv4_dec_150016
conv4_dec_150018
conv3_dec_150022
conv3_dec_150024
conv2_dec_150028
conv2_dec_150030
conv1_dec_150034
conv1_dec_150036
output_150040
output_150042
identityЂ!conv1_dec/StatefulPartitionedCallЂ!conv2_dec/StatefulPartitionedCallЂ!conv3_dec/StatefulPartitionedCallЂ!conv4_dec/StatefulPartitionedCallЂ!conv5_dec/StatefulPartitionedCallЂ decoding/StatefulPartitionedCallЂoutput/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_150004decoding_150006*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_1497952"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1498252
reshape/PartitionedCallП
!conv5_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv5_dec_150010conv5_dec_150012*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_dec_layer_call_and_return_conditional_losses_1498442#
!conv5_dec/StatefulPartitionedCall
upsamp5/PartitionedCallPartitionedCall*conv5_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
upsamp5/PartitionedCallа
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp5/PartitionedCall:output:0conv4_dec_150016conv4_dec_150018*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_1498722#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
upsamp4/PartitionedCallа
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_150022conv3_dec_150024*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_1499002#
!conv3_dec/StatefulPartitionedCall
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
upsamp3/PartitionedCallа
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_150028conv2_dec_150030*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_1499282#
!conv2_dec/StatefulPartitionedCall
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
upsamp2/PartitionedCallа
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_150034conv1_dec_150036*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_1499562#
!conv1_dec/StatefulPartitionedCall
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
upsamp1/PartitionedCallС
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_150040output_150042*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall"^conv5_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2F
!conv5_dec/StatefulPartitionedCall!conv5_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameinput_decoder
щ
_
C__inference_reshape_layer_call_and_return_conditional_losses_152017

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
strided_slice/stack_2т
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
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
Reshape/shape/3К
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
л
B__inference_output_layer_call_and_return_conditional_losses_152133

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
SigmoidЊ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
о
E__inference_z_log_var_layer_call_and_return_conditional_losses_149390

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
л
B__inference_z_mean_layer_call_and_return_conditional_losses_149364

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

_
C__inference_upsamp2_layer_call_and_return_conditional_losses_149756

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
strided_slice/stack_2Ю
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
mulе
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(2
resize/ResizeNearestNeighborЄ
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
н
D__inference_decoding_layer_call_and_return_conditional_losses_149795

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­

Л
(__inference_Decoder_layer_call_fn_150125
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

unknown_10

unknown_11

unknown_12
identityЂStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinput_decoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_1500942
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameinput_decoder
Я

о
E__inference_conv4_enc_layer_call_and_return_conditional_losses_149269

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
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
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Л
л
B__inference_output_layer_call_and_return_conditional_losses_149984

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
SigmoidЊ
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я

о
E__inference_conv2_enc_layer_call_and_return_conditional_losses_151815

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ъ
ћ
?__inference_VAE_layer_call_and_return_conditional_losses_150420
input_1
encoder_150289
encoder_150291
encoder_150293
encoder_150295
encoder_150297
encoder_150299
encoder_150301
encoder_150303
encoder_150305
encoder_150307
encoder_150309
encoder_150311
encoder_150313
encoder_150315
encoder_150317
encoder_150319
decoder_150390
decoder_150392
decoder_150394
decoder_150396
decoder_150398
decoder_150400
decoder_150402
decoder_150404
decoder_150406
decoder_150408
decoder_150410
decoder_150412
decoder_150414
decoder_150416
identityЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCallЗ
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_150289encoder_150291encoder_150293encoder_150295encoder_150297encoder_150299encoder_150301encoder_150303encoder_150305encoder_150307encoder_150309encoder_150311encoder_150313encoder_150315encoder_150317encoder_150319*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1495532!
Encoder/StatefulPartitionedCallІ
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_150390decoder_150392decoder_150394decoder_150396decoder_150398decoder_150400decoder_150402decoder_150404decoder_150406decoder_150408decoder_150410decoder_150412decoder_150414decoder_150416*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_1500942!
Decoder/StatefulPartitionedCallк
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ(
!
_user_specified_name	input_1
­

Л
(__inference_Decoder_layer_call_fn_150203
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

unknown_10

unknown_11

unknown_12
identityЂStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinput_decoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_1501722
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameinput_decoder

s
D__inference_sampling_layer_call_and_return_conditional_losses_151978
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
strided_slice/stack_2т
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
strided_slice_1/stack_2ю
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
random_normal/stddevх
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed22$
"random_normal/RandomStandardNormalД
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
random_normal/mul
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
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
:џџџџџџџџџ2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
 
D
(__inference_upsamp1_layer_call_fn_149781

inputs
identityч
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Н
_
C__inference_flatten_layer_call_and_return_conditional_losses_149320

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


*__inference_conv5_enc_layer_call_fn_151884

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_enc_layer_call_and_return_conditional_losses_1492972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
и

о
E__inference_conv5_dec_layer_call_and_return_conditional_losses_149844

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ТЌ
ї
!__inference__wrapped_model_149110
input_18
4vae_encoder_conv1_enc_conv2d_readvariableop_resource9
5vae_encoder_conv1_enc_biasadd_readvariableop_resource8
4vae_encoder_conv2_enc_conv2d_readvariableop_resource9
5vae_encoder_conv2_enc_biasadd_readvariableop_resource8
4vae_encoder_conv3_enc_conv2d_readvariableop_resource9
5vae_encoder_conv3_enc_biasadd_readvariableop_resource8
4vae_encoder_conv4_enc_conv2d_readvariableop_resource9
5vae_encoder_conv4_enc_biasadd_readvariableop_resource8
4vae_encoder_conv5_enc_conv2d_readvariableop_resource9
5vae_encoder_conv5_enc_biasadd_readvariableop_resource9
5vae_encoder_bottleneck_matmul_readvariableop_resource:
6vae_encoder_bottleneck_biasadd_readvariableop_resource5
1vae_encoder_z_mean_matmul_readvariableop_resource6
2vae_encoder_z_mean_biasadd_readvariableop_resource8
4vae_encoder_z_log_var_matmul_readvariableop_resource9
5vae_encoder_z_log_var_biasadd_readvariableop_resource7
3vae_decoder_decoding_matmul_readvariableop_resource8
4vae_decoder_decoding_biasadd_readvariableop_resource8
4vae_decoder_conv5_dec_conv2d_readvariableop_resource9
5vae_decoder_conv5_dec_biasadd_readvariableop_resource8
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
identityЂ,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpЂ+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOpЂ,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpЂ+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOpЂ,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpЂ+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOpЂ,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpЂ+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOpЂ,VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOpЂ+VAE/Decoder/conv5_dec/Conv2D/ReadVariableOpЂ+VAE/Decoder/decoding/BiasAdd/ReadVariableOpЂ*VAE/Decoder/decoding/MatMul/ReadVariableOpЂ)VAE/Decoder/output/BiasAdd/ReadVariableOpЂ(VAE/Decoder/output/Conv2D/ReadVariableOpЂ-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOpЂ,VAE/Encoder/bottleneck/MatMul/ReadVariableOpЂ,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpЂ+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOpЂ,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpЂ+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOpЂ,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpЂ+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOpЂ,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpЂ+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOpЂ,VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOpЂ+VAE/Encoder/conv5_enc/Conv2D/ReadVariableOpЂ,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOpЂ+VAE/Encoder/z_log_var/MatMul/ReadVariableOpЂ)VAE/Encoder/z_mean/BiasAdd/ReadVariableOpЂ(VAE/Encoder/z_mean/MatMul/ReadVariableOpз
+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOpц
VAE/Encoder/conv1_enc/Conv2DConv2Dinput_13VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingSAME*
strides
2
VAE/Encoder/conv1_enc/Conv2DЮ
,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpр
VAE/Encoder/conv1_enc/BiasAddBiasAdd%VAE/Encoder/conv1_enc/Conv2D:output:04VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
VAE/Encoder/conv1_enc/BiasAddЂ
VAE/Encoder/conv1_enc/ReluRelu&VAE/Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
VAE/Encoder/conv1_enc/Reluн
VAE/Encoder/maxpool1/MaxPoolMaxPool(VAE/Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool1/MaxPoolз
+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv2_enc/Conv2DConv2D%VAE/Encoder/maxpool1/MaxPool:output:03VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
VAE/Encoder/conv2_enc/Conv2DЮ
,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpр
VAE/Encoder/conv2_enc/BiasAddBiasAdd%VAE/Encoder/conv2_enc/Conv2D:output:04VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/conv2_enc/BiasAddЂ
VAE/Encoder/conv2_enc/ReluRelu&VAE/Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/conv2_enc/Reluн
VAE/Encoder/maxpool2/MaxPoolMaxPool(VAE/Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ
*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool2/MaxPoolз
+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv3_enc/Conv2DConv2D%VAE/Encoder/maxpool2/MaxPool:output:03VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 *
paddingSAME*
strides
2
VAE/Encoder/conv3_enc/Conv2DЮ
,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpр
VAE/Encoder/conv3_enc/BiasAddBiasAdd%VAE/Encoder/conv3_enc/Conv2D:output:04VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
VAE/Encoder/conv3_enc/BiasAddЂ
VAE/Encoder/conv3_enc/ReluRelu&VAE/Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
VAE/Encoder/conv3_enc/Reluн
VAE/Encoder/maxpool3/MaxPoolMaxPool(VAE/Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool3/MaxPoolз
+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv4_enc/Conv2DConv2D%VAE/Encoder/maxpool3/MaxPool:output:03VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
VAE/Encoder/conv4_enc/Conv2DЮ
,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpр
VAE/Encoder/conv4_enc/BiasAddBiasAdd%VAE/Encoder/conv4_enc/Conv2D:output:04VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
VAE/Encoder/conv4_enc/BiasAddЂ
VAE/Encoder/conv4_enc/ReluRelu&VAE/Encoder/conv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
VAE/Encoder/conv4_enc/Reluн
VAE/Encoder/maxpool4/MaxPoolMaxPool(VAE/Encoder/conv4_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool4/MaxPoolи
+VAE/Encoder/conv5_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+VAE/Encoder/conv5_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv5_enc/Conv2DConv2D%VAE/Encoder/maxpool4/MaxPool:output:03VAE/Encoder/conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
VAE/Encoder/conv5_enc/Conv2DЯ
,VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOpс
VAE/Encoder/conv5_enc/BiasAddBiasAdd%VAE/Encoder/conv5_enc/Conv2D:output:04VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/conv5_enc/BiasAddЃ
VAE/Encoder/conv5_enc/ReluRelu&VAE/Encoder/conv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/conv5_enc/Reluо
VAE/Encoder/maxpool5/MaxPoolMaxPool(VAE/Encoder/conv5_enc/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool5/MaxPool
VAE/Encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
VAE/Encoder/flatten/ConstУ
VAE/Encoder/flatten/ReshapeReshape%VAE/Encoder/maxpool5/MaxPool:output:0"VAE/Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/flatten/Reshapeг
,VAE/Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp5vae_encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02.
,VAE/Encoder/bottleneck/MatMul/ReadVariableOpж
VAE/Encoder/bottleneck/MatMulMatMul$VAE/Encoder/flatten/Reshape:output:04VAE/Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/bottleneck/MatMulб
-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp6vae_encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOpн
VAE/Encoder/bottleneck/BiasAddBiasAdd'VAE/Encoder/bottleneck/MatMul:product:05VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
VAE/Encoder/bottleneck/BiasAddЦ
(VAE/Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp1vae_encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(VAE/Encoder/z_mean/MatMul/ReadVariableOpЭ
VAE/Encoder/z_mean/MatMulMatMul'VAE/Encoder/bottleneck/BiasAdd:output:00VAE/Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/z_mean/MatMulХ
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp2vae_encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOpЭ
VAE/Encoder/z_mean/BiasAddBiasAdd#VAE/Encoder/z_mean/MatMul:product:01VAE/Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/z_mean/BiasAddЯ
+VAE/Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp4vae_encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+VAE/Encoder/z_log_var/MatMul/ReadVariableOpж
VAE/Encoder/z_log_var/MatMulMatMul'VAE/Encoder/bottleneck/BiasAdd:output:03VAE/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/z_log_var/MatMulЮ
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOpй
VAE/Encoder/z_log_var/BiasAddBiasAdd&VAE/Encoder/z_log_var/MatMul:product:04VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/z_log_var/BiasAdd
VAE/Encoder/sampling/ShapeShape#VAE/Encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
VAE/Encoder/sampling/Shape
(VAE/Encoder/sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(VAE/Encoder/sampling/strided_slice/stackЂ
*VAE/Encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*VAE/Encoder/sampling/strided_slice/stack_1Ђ
*VAE/Encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*VAE/Encoder/sampling/strided_slice/stack_2р
"VAE/Encoder/sampling/strided_sliceStridedSlice#VAE/Encoder/sampling/Shape:output:01VAE/Encoder/sampling/strided_slice/stack:output:03VAE/Encoder/sampling/strided_slice/stack_1:output:03VAE/Encoder/sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"VAE/Encoder/sampling/strided_slice
VAE/Encoder/sampling/Shape_1Shape#VAE/Encoder/z_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
VAE/Encoder/sampling/Shape_1Ђ
*VAE/Encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*VAE/Encoder/sampling/strided_slice_1/stackІ
,VAE/Encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling/strided_slice_1/stack_1І
,VAE/Encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling/strided_slice_1/stack_2ь
$VAE/Encoder/sampling/strided_slice_1StridedSlice%VAE/Encoder/sampling/Shape_1:output:03VAE/Encoder/sampling/strided_slice_1/stack:output:05VAE/Encoder/sampling/strided_slice_1/stack_1:output:05VAE/Encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$VAE/Encoder/sampling/strided_slice_1ц
(VAE/Encoder/sampling/random_normal/shapePack+VAE/Encoder/sampling/strided_slice:output:0-VAE/Encoder/sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2*
(VAE/Encoder/sampling/random_normal/shape
'VAE/Encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'VAE/Encoder/sampling/random_normal/mean
)VAE/Encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)VAE/Encoder/sampling/random_normal/stddevЄ
7VAE/Encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal1VAE/Encoder/sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2аЎъ29
7VAE/Encoder/sampling/random_normal/RandomStandardNormal
&VAE/Encoder/sampling/random_normal/mulMul@VAE/Encoder/sampling/random_normal/RandomStandardNormal:output:02VAE/Encoder/sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2(
&VAE/Encoder/sampling/random_normal/mulш
"VAE/Encoder/sampling/random_normalAdd*VAE/Encoder/sampling/random_normal/mul:z:00VAE/Encoder/sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2$
"VAE/Encoder/sampling/random_normal}
VAE/Encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
VAE/Encoder/sampling/mul/xК
VAE/Encoder/sampling/mulMul#VAE/Encoder/sampling/mul/x:output:0&VAE/Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/sampling/mul
VAE/Encoder/sampling/ExpExpVAE/Encoder/sampling/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/sampling/ExpЗ
VAE/Encoder/sampling/mul_1MulVAE/Encoder/sampling/Exp:y:0&VAE/Encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/sampling/mul_1Д
VAE/Encoder/sampling/addAddV2#VAE/Encoder/z_mean/BiasAdd:output:0VAE/Encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
VAE/Encoder/sampling/addЭ
*VAE/Decoder/decoding/MatMul/ReadVariableOpReadVariableOp3vae_decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*VAE/Decoder/decoding/MatMul/ReadVariableOpЩ
VAE/Decoder/decoding/MatMulMatMulVAE/Encoder/sampling/add:z:02VAE/Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
VAE/Decoder/decoding/MatMulЬ
+VAE/Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp4vae_decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+VAE/Decoder/decoding/BiasAdd/ReadVariableOpж
VAE/Decoder/decoding/BiasAddBiasAdd%VAE/Decoder/decoding/MatMul:product:03VAE/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
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
)VAE/Decoder/reshape/strided_slice/stack_2к
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
value	B :2%
#VAE/Decoder/reshape/Reshape/shape/2
#VAE/Decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2%
#VAE/Decoder/reshape/Reshape/shape/3В
!VAE/Decoder/reshape/Reshape/shapePack*VAE/Decoder/reshape/strided_slice:output:0,VAE/Decoder/reshape/Reshape/shape/1:output:0,VAE/Decoder/reshape/Reshape/shape/2:output:0,VAE/Decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!VAE/Decoder/reshape/Reshape/shapeг
VAE/Decoder/reshape/ReshapeReshape%VAE/Decoder/decoding/BiasAdd:output:0*VAE/Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
VAE/Decoder/reshape/Reshapeй
+VAE/Decoder/conv5_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+VAE/Decoder/conv5_dec/Conv2D/ReadVariableOp
VAE/Decoder/conv5_dec/Conv2DConv2D$VAE/Decoder/reshape/Reshape:output:03VAE/Decoder/conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
VAE/Decoder/conv5_dec/Conv2DЯ
,VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOpс
VAE/Decoder/conv5_dec/BiasAddBiasAdd%VAE/Decoder/conv5_dec/Conv2D:output:04VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
VAE/Decoder/conv5_dec/BiasAddЃ
VAE/Decoder/conv5_dec/ReluRelu&VAE/Decoder/conv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
VAE/Decoder/conv5_dec/Relu
VAE/Decoder/upsamp5/ShapeShape(VAE/Decoder/conv5_dec/Relu:activations:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp5/Shape
'VAE/Decoder/upsamp5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'VAE/Decoder/upsamp5/strided_slice/stack 
)VAE/Decoder/upsamp5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp5/strided_slice/stack_1 
)VAE/Decoder/upsamp5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)VAE/Decoder/upsamp5/strided_slice/stack_2Ц
!VAE/Decoder/upsamp5/strided_sliceStridedSlice"VAE/Decoder/upsamp5/Shape:output:00VAE/Decoder/upsamp5/strided_slice/stack:output:02VAE/Decoder/upsamp5/strided_slice/stack_1:output:02VAE/Decoder/upsamp5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2#
!VAE/Decoder/upsamp5/strided_slice
VAE/Decoder/upsamp5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
VAE/Decoder/upsamp5/ConstЎ
VAE/Decoder/upsamp5/mulMul*VAE/Decoder/upsamp5/strided_slice:output:0"VAE/Decoder/upsamp5/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp5/mul
0VAE/Decoder/upsamp5/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv5_dec/Relu:activations:0VAE/Decoder/upsamp5/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(22
0VAE/Decoder/upsamp5/resize/ResizeNearestNeighborи
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv4_dec/Conv2DConv2DAVAE/Decoder/upsamp5/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
VAE/Decoder/conv4_dec/Conv2DЮ
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpр
VAE/Decoder/conv4_dec/BiasAddBiasAdd%VAE/Decoder/conv4_dec/Conv2D:output:04VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
VAE/Decoder/conv4_dec/BiasAddЂ
VAE/Decoder/conv4_dec/ReluRelu&VAE/Decoder/conv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
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
)VAE/Decoder/upsamp4/strided_slice/stack_2Ц
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
VAE/Decoder/upsamp4/ConstЎ
VAE/Decoder/upsamp4/mulMul*VAE/Decoder/upsamp4/strided_slice:output:0"VAE/Decoder/upsamp4/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp4/mul
0VAE/Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv4_dec/Relu:activations:0VAE/Decoder/upsamp4/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(22
0VAE/Decoder/upsamp4/resize/ResizeNearestNeighborз
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02-
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv3_dec/Conv2DConv2DAVAE/Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
VAE/Decoder/conv3_dec/Conv2DЮ
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpр
VAE/Decoder/conv3_dec/BiasAddBiasAdd%VAE/Decoder/conv3_dec/Conv2D:output:04VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
VAE/Decoder/conv3_dec/BiasAddЂ
VAE/Decoder/conv3_dec/ReluRelu&VAE/Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
)VAE/Decoder/upsamp3/strided_slice/stack_2Ц
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
VAE/Decoder/upsamp3/ConstЎ
VAE/Decoder/upsamp3/mulMul*VAE/Decoder/upsamp3/strided_slice:output:0"VAE/Decoder/upsamp3/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp3/mul
0VAE/Decoder/upsamp3/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv3_dec/Relu:activations:0VAE/Decoder/upsamp3/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ *
half_pixel_centers(22
0VAE/Decoder/upsamp3/resize/ResizeNearestNeighborз
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv2_dec/Conv2DConv2DAVAE/Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
VAE/Decoder/conv2_dec/Conv2DЮ
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpр
VAE/Decoder/conv2_dec/BiasAddBiasAdd%VAE/Decoder/conv2_dec/Conv2D:output:04VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
VAE/Decoder/conv2_dec/BiasAddЂ
VAE/Decoder/conv2_dec/ReluRelu&VAE/Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
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
)VAE/Decoder/upsamp2/strided_slice/stack_2Ц
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
VAE/Decoder/upsamp2/ConstЎ
VAE/Decoder/upsamp2/mulMul*VAE/Decoder/upsamp2/strided_slice:output:0"VAE/Decoder/upsamp2/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp2/mul
0VAE/Decoder/upsamp2/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv2_dec/Relu:activations:0VAE/Decoder/upsamp2/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ *
half_pixel_centers(22
0VAE/Decoder/upsamp2/resize/ResizeNearestNeighborз
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv1_dec/Conv2DConv2DAVAE/Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
VAE/Decoder/conv1_dec/Conv2DЮ
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpр
VAE/Decoder/conv1_dec/BiasAddBiasAdd%VAE/Decoder/conv1_dec/Conv2D:output:04VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
VAE/Decoder/conv1_dec/BiasAddЂ
VAE/Decoder/conv1_dec/ReluRelu&VAE/Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
)VAE/Decoder/upsamp1/strided_slice/stack_2Ц
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
VAE/Decoder/upsamp1/ConstЎ
VAE/Decoder/upsamp1/mulMul*VAE/Decoder/upsamp1/strided_slice:output:0"VAE/Decoder/upsamp1/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp1/mul
0VAE/Decoder/upsamp1/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv1_dec/Relu:activations:0VAE/Decoder/upsamp1/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
half_pixel_centers(22
0VAE/Decoder/upsamp1/resize/ResizeNearestNeighborЮ
(VAE/Decoder/output/Conv2D/ReadVariableOpReadVariableOp1vae_decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(VAE/Decoder/output/Conv2D/ReadVariableOp
VAE/Decoder/output/Conv2DConv2DAVAE/Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:00VAE/Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides
2
VAE/Decoder/output/Conv2DХ
)VAE/Decoder/output/BiasAdd/ReadVariableOpReadVariableOp2vae_decoder_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)VAE/Decoder/output/BiasAdd/ReadVariableOpд
VAE/Decoder/output/BiasAddBiasAdd"VAE/Decoder/output/Conv2D:output:01VAE/Decoder/output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
VAE/Decoder/output/BiasAddЂ
VAE/Decoder/output/SigmoidSigmoid#VAE/Decoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
VAE/Decoder/output/Sigmoidс
IdentityIdentityVAE/Decoder/output/Sigmoid:y:0-^VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv5_dec/Conv2D/ReadVariableOp,^VAE/Decoder/decoding/BiasAdd/ReadVariableOp+^VAE/Decoder/decoding/MatMul/ReadVariableOp*^VAE/Decoder/output/BiasAdd/ReadVariableOp)^VAE/Decoder/output/Conv2D/ReadVariableOp.^VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp-^VAE/Encoder/bottleneck/MatMul/ReadVariableOp-^VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv5_enc/Conv2D/ReadVariableOp-^VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp,^VAE/Encoder/z_log_var/MatMul/ReadVariableOp*^VAE/Encoder/z_mean/BiasAdd/ReadVariableOp)^VAE/Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ(2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::2\
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp2Z
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp2\
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp2Z
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp2\
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp2Z
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp2\
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp2Z
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp2\
,VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOp,VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOp2Z
+VAE/Decoder/conv5_dec/Conv2D/ReadVariableOp+VAE/Decoder/conv5_dec/Conv2D/ReadVariableOp2Z
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
,VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOp,VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOp2Z
+VAE/Encoder/conv5_enc/Conv2D/ReadVariableOp+VAE/Encoder/conv5_enc/Conv2D/ReadVariableOp2\
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp2Z
+VAE/Encoder/z_log_var/MatMul/ReadVariableOp+VAE/Encoder/z_log_var/MatMul/ReadVariableOp2V
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOp)VAE/Encoder/z_mean/BiasAdd/ReadVariableOp2T
(VAE/Encoder/z_mean/MatMul/ReadVariableOp(VAE/Encoder/z_mean/MatMul/ReadVariableOp:X T
/
_output_shapes
:џџџџџџџџџ(
!
_user_specified_name	input_1

Б
$__inference_signature_wrapper_150762
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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityЂStatefulPartitionedCallм
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_1491102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ(2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ(
!
_user_specified_name	input_1
Б
Р	
C__inference_Decoder_layer_call_and_return_conditional_losses_151718

inputs+
'decoding_matmul_readvariableop_resource,
(decoding_biasadd_readvariableop_resource,
(conv5_dec_conv2d_readvariableop_resource-
)conv5_dec_biasadd_readvariableop_resource,
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
identityЂ conv1_dec/BiasAdd/ReadVariableOpЂconv1_dec/Conv2D/ReadVariableOpЂ conv2_dec/BiasAdd/ReadVariableOpЂconv2_dec/Conv2D/ReadVariableOpЂ conv3_dec/BiasAdd/ReadVariableOpЂconv3_dec/Conv2D/ReadVariableOpЂ conv4_dec/BiasAdd/ReadVariableOpЂconv4_dec/Conv2D/ReadVariableOpЂ conv5_dec/BiasAdd/ReadVariableOpЂconv5_dec/Conv2D/ReadVariableOpЂdecoding/BiasAdd/ReadVariableOpЂdecoding/MatMul/ReadVariableOpЂoutput/BiasAdd/ReadVariableOpЂoutput/Conv2D/ReadVariableOpЉ
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
decoding/MatMul/ReadVariableOp
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
decoding/MatMulЈ
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
decoding/BiasAdd/ReadVariableOpІ
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
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
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :2
reshape/Reshape/shape/3ъ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeЃ
reshape/ReshapeReshapedecoding/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
reshape/ReshapeЕ
conv5_dec/Conv2D/ReadVariableOpReadVariableOp(conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv5_dec/Conv2D/ReadVariableOpд
conv5_dec/Conv2DConv2Dreshape/Reshape:output:0'conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv5_dec/Conv2DЋ
 conv5_dec/BiasAdd/ReadVariableOpReadVariableOp)conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv5_dec/BiasAdd/ReadVariableOpБ
conv5_dec/BiasAddBiasAddconv5_dec/Conv2D:output:0(conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv5_dec/BiasAdd
conv5_dec/ReluReluconv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv5_dec/Reluj
upsamp5/ShapeShapeconv5_dec/Relu:activations:0*
T0*
_output_shapes
:2
upsamp5/Shape
upsamp5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
upsamp5/strided_slice/stack
upsamp5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp5/strided_slice/stack_1
upsamp5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
upsamp5/strided_slice/stack_2ў
upsamp5/strided_sliceStridedSliceupsamp5/Shape:output:0$upsamp5/strided_slice/stack:output:0&upsamp5/strided_slice/stack_1:output:0&upsamp5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
upsamp5/strided_sliceo
upsamp5/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
upsamp5/Const~
upsamp5/mulMulupsamp5/strided_slice:output:0upsamp5/Const:output:0*
T0*
_output_shapes
:2
upsamp5/mulщ
$upsamp5/resize/ResizeNearestNeighborResizeNearestNeighborconv5_dec/Relu:activations:0upsamp5/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ*
half_pixel_centers(2&
$upsamp5/resize/ResizeNearestNeighborД
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv4_dec/Conv2D/ReadVariableOp№
conv4_dec/Conv2DConv2D5upsamp5/resize/ResizeNearestNeighbor:resized_images:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv4_dec/Conv2DЊ
 conv4_dec/BiasAdd/ReadVariableOpReadVariableOp)conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv4_dec/BiasAdd/ReadVariableOpА
conv4_dec/BiasAddBiasAddconv4_dec/Conv2D:output:0(conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4_dec/BiasAdd~
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
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
upsamp4/strided_slice/stack_2ў
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
upsamp4/mulш
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighborГ
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv3_dec/Conv2D/ReadVariableOp№
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv3_dec/Conv2DЊ
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3_dec/BiasAdd/ReadVariableOpА
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
upsamp3/strided_slice/stack_2ў
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
upsamp3/mulш
$upsamp3/resize/ResizeNearestNeighborResizeNearestNeighborconv3_dec/Relu:activations:0upsamp3/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ *
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighborГ
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_dec/Conv2D/ReadVariableOp№
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2_dec/Conv2DЊ
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2_dec/BiasAdd/ReadVariableOpА
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
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
upsamp2/strided_slice/stack_2ў
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
upsamp2/mulш
$upsamp2/resize/ResizeNearestNeighborResizeNearestNeighborconv2_dec/Relu:activations:0upsamp2/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighborГ
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_dec/Conv2D/ReadVariableOp№
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv1_dec/Conv2DЊ
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOpА
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ 2
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
upsamp1/strided_slice/stack_2ў
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
upsamp1/mulш
$upsamp1/resize/ResizeNearestNeighborResizeNearestNeighborconv1_dec/Relu:activations:0upsamp1/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ@ *
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighborЊ
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
output/Conv2D/ReadVariableOpш
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingVALID*
strides
2
output/Conv2DЁ
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOpЄ
output/BiasAddBiasAddoutput/Conv2D:output:0%output/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
output/SigmoidЩ
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp!^conv5_dec/BiasAdd/ReadVariableOp ^conv5_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ(2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::2D
 conv1_dec/BiasAdd/ReadVariableOp conv1_dec/BiasAdd/ReadVariableOp2B
conv1_dec/Conv2D/ReadVariableOpconv1_dec/Conv2D/ReadVariableOp2D
 conv2_dec/BiasAdd/ReadVariableOp conv2_dec/BiasAdd/ReadVariableOp2B
conv2_dec/Conv2D/ReadVariableOpconv2_dec/Conv2D/ReadVariableOp2D
 conv3_dec/BiasAdd/ReadVariableOp conv3_dec/BiasAdd/ReadVariableOp2B
conv3_dec/Conv2D/ReadVariableOpconv3_dec/Conv2D/ReadVariableOp2D
 conv4_dec/BiasAdd/ReadVariableOp conv4_dec/BiasAdd/ReadVariableOp2B
conv4_dec/Conv2D/ReadVariableOpconv4_dec/Conv2D/ReadVariableOp2D
 conv5_dec/BiasAdd/ReadVariableOp conv5_dec/BiasAdd/ReadVariableOp2B
conv5_dec/Conv2D/ReadVariableOpconv5_dec/Conv2D/ReadVariableOp2B
decoding/BiasAdd/ReadVariableOpdecoding/BiasAdd/ReadVariableOp2@
decoding/MatMul/ReadVariableOpdecoding/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/Conv2D/ReadVariableOpoutput/Conv2D/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ц

+__inference_bottleneck_layer_call_fn_151914

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_1493382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
 
D
(__inference_upsamp3_layer_call_fn_149743

inputs
identityч
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 
D
(__inference_upsamp5_layer_call_fn_149705

inputs
identityч
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О
о
E__inference_conv4_dec_layer_call_and_return_conditional_losses_152053

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


Д
(__inference_Decoder_layer_call_fn_151751

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
identityЂStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_1500942
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ
E
)__inference_maxpool4_layer_call_fn_149158

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
п
F__inference_bottleneck_layer_call_and_return_conditional_losses_151905

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
І
r
)__inference_sampling_layer_call_fn_151984
inputs_0
inputs_1
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1494322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs/1
Л
о
E__inference_conv3_dec_layer_call_and_return_conditional_losses_152073

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
	
н
D__inference_decoding_layer_call_and_return_conditional_losses_151994

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ћ
(__inference_Encoder_layer_call_fn_149592
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

unknown_14
identity

identity_1

identity_2ЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_encoderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1495532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:џџџџџџџџџ(
'
_user_specified_nameinput_encoder


*__inference_conv5_dec_layer_call_fn_152042

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_dec_layer_call_and_return_conditional_losses_1498442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

_
C__inference_upsamp5_layer_call_and_return_conditional_losses_149699

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
strided_slice/stack_2Ю
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
mulе
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(2
resize/ResizeNearestNeighborЄ
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
 
D
(__inference_upsamp2_layer_call_fn_149762

inputs
identityч
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џН
)
__inference__traced_save_152468
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
)savev2_conv4_enc_bias_read_readvariableop/
+savev2_conv5_enc_kernel_read_readvariableop-
)savev2_conv5_enc_bias_read_readvariableop0
,savev2_bottleneck_kernel_read_readvariableop.
*savev2_bottleneck_bias_read_readvariableop,
(savev2_z_mean_kernel_read_readvariableop*
&savev2_z_mean_bias_read_readvariableop/
+savev2_z_log_var_kernel_read_readvariableop-
)savev2_z_log_var_bias_read_readvariableop.
*savev2_decoding_kernel_read_readvariableop,
(savev2_decoding_bias_read_readvariableop/
+savev2_conv5_dec_kernel_read_readvariableop-
)savev2_conv5_dec_bias_read_readvariableop/
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
0savev2_adam_conv4_enc_bias_m_read_readvariableop6
2savev2_adam_conv5_enc_kernel_m_read_readvariableop4
0savev2_adam_conv5_enc_bias_m_read_readvariableop7
3savev2_adam_bottleneck_kernel_m_read_readvariableop5
1savev2_adam_bottleneck_bias_m_read_readvariableop3
/savev2_adam_z_mean_kernel_m_read_readvariableop1
-savev2_adam_z_mean_bias_m_read_readvariableop6
2savev2_adam_z_log_var_kernel_m_read_readvariableop4
0savev2_adam_z_log_var_bias_m_read_readvariableop5
1savev2_adam_decoding_kernel_m_read_readvariableop3
/savev2_adam_decoding_bias_m_read_readvariableop6
2savev2_adam_conv5_dec_kernel_m_read_readvariableop4
0savev2_adam_conv5_dec_bias_m_read_readvariableop6
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
0savev2_adam_conv4_enc_bias_v_read_readvariableop6
2savev2_adam_conv5_enc_kernel_v_read_readvariableop4
0savev2_adam_conv5_enc_bias_v_read_readvariableop7
3savev2_adam_bottleneck_kernel_v_read_readvariableop5
1savev2_adam_bottleneck_bias_v_read_readvariableop3
/savev2_adam_z_mean_kernel_v_read_readvariableop1
-savev2_adam_z_mean_bias_v_read_readvariableop6
2savev2_adam_z_log_var_kernel_v_read_readvariableop4
0savev2_adam_z_log_var_bias_v_read_readvariableop5
1savev2_adam_decoding_kernel_v_read_readvariableop3
/savev2_adam_decoding_bias_v_read_readvariableop6
2savev2_adam_conv5_dec_kernel_v_read_readvariableop4
0savev2_adam_conv5_dec_bias_v_read_readvariableop6
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

identity_1ЂMergeV2Checkpoints
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameА6
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*Т5
valueИ5BЕ5fB3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesз
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*с
valueзBдfB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЗ'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv1_enc_kernel_read_readvariableop)savev2_conv1_enc_bias_read_readvariableop+savev2_conv2_enc_kernel_read_readvariableop)savev2_conv2_enc_bias_read_readvariableop+savev2_conv3_enc_kernel_read_readvariableop)savev2_conv3_enc_bias_read_readvariableop+savev2_conv4_enc_kernel_read_readvariableop)savev2_conv4_enc_bias_read_readvariableop+savev2_conv5_enc_kernel_read_readvariableop)savev2_conv5_enc_bias_read_readvariableop,savev2_bottleneck_kernel_read_readvariableop*savev2_bottleneck_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableop*savev2_decoding_kernel_read_readvariableop(savev2_decoding_bias_read_readvariableop+savev2_conv5_dec_kernel_read_readvariableop)savev2_conv5_dec_bias_read_readvariableop+savev2_conv4_dec_kernel_read_readvariableop)savev2_conv4_dec_bias_read_readvariableop+savev2_conv3_dec_kernel_read_readvariableop)savev2_conv3_dec_bias_read_readvariableop+savev2_conv2_dec_kernel_read_readvariableop)savev2_conv2_dec_bias_read_readvariableop+savev2_conv1_dec_kernel_read_readvariableop)savev2_conv1_dec_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop2savev2_adam_conv1_enc_kernel_m_read_readvariableop0savev2_adam_conv1_enc_bias_m_read_readvariableop2savev2_adam_conv2_enc_kernel_m_read_readvariableop0savev2_adam_conv2_enc_bias_m_read_readvariableop2savev2_adam_conv3_enc_kernel_m_read_readvariableop0savev2_adam_conv3_enc_bias_m_read_readvariableop2savev2_adam_conv4_enc_kernel_m_read_readvariableop0savev2_adam_conv4_enc_bias_m_read_readvariableop2savev2_adam_conv5_enc_kernel_m_read_readvariableop0savev2_adam_conv5_enc_bias_m_read_readvariableop3savev2_adam_bottleneck_kernel_m_read_readvariableop1savev2_adam_bottleneck_bias_m_read_readvariableop/savev2_adam_z_mean_kernel_m_read_readvariableop-savev2_adam_z_mean_bias_m_read_readvariableop2savev2_adam_z_log_var_kernel_m_read_readvariableop0savev2_adam_z_log_var_bias_m_read_readvariableop1savev2_adam_decoding_kernel_m_read_readvariableop/savev2_adam_decoding_bias_m_read_readvariableop2savev2_adam_conv5_dec_kernel_m_read_readvariableop0savev2_adam_conv5_dec_bias_m_read_readvariableop2savev2_adam_conv4_dec_kernel_m_read_readvariableop0savev2_adam_conv4_dec_bias_m_read_readvariableop2savev2_adam_conv3_dec_kernel_m_read_readvariableop0savev2_adam_conv3_dec_bias_m_read_readvariableop2savev2_adam_conv2_dec_kernel_m_read_readvariableop0savev2_adam_conv2_dec_bias_m_read_readvariableop2savev2_adam_conv1_dec_kernel_m_read_readvariableop0savev2_adam_conv1_dec_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv1_enc_kernel_v_read_readvariableop0savev2_adam_conv1_enc_bias_v_read_readvariableop2savev2_adam_conv2_enc_kernel_v_read_readvariableop0savev2_adam_conv2_enc_bias_v_read_readvariableop2savev2_adam_conv3_enc_kernel_v_read_readvariableop0savev2_adam_conv3_enc_bias_v_read_readvariableop2savev2_adam_conv4_enc_kernel_v_read_readvariableop0savev2_adam_conv4_enc_bias_v_read_readvariableop2savev2_adam_conv5_enc_kernel_v_read_readvariableop0savev2_adam_conv5_enc_bias_v_read_readvariableop3savev2_adam_bottleneck_kernel_v_read_readvariableop1savev2_adam_bottleneck_bias_v_read_readvariableop/savev2_adam_z_mean_kernel_v_read_readvariableop-savev2_adam_z_mean_bias_v_read_readvariableop2savev2_adam_z_log_var_kernel_v_read_readvariableop0savev2_adam_z_log_var_bias_v_read_readvariableop1savev2_adam_decoding_kernel_v_read_readvariableop/savev2_adam_decoding_bias_v_read_readvariableop2savev2_adam_conv5_dec_kernel_v_read_readvariableop0savev2_adam_conv5_dec_bias_v_read_readvariableop2savev2_adam_conv4_dec_kernel_v_read_readvariableop0savev2_adam_conv4_dec_bias_v_read_readvariableop2savev2_adam_conv3_dec_kernel_v_read_readvariableop0savev2_adam_conv3_dec_bias_v_read_readvariableop2savev2_adam_conv2_dec_kernel_v_read_readvariableop0savev2_adam_conv2_dec_bias_v_read_readvariableop2savev2_adam_conv1_dec_kernel_v_read_readvariableop0savev2_adam_conv1_dec_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *t
dtypesj
h2f	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*Ђ
_input_shapes
: : : : : : : : : : : : ::::: : : @:@:@::	::::::	::::@:@:@ : : :::::::::: : : @:@:@::	::::::	::::@:@:@ : : :::::::::: : : @:@:@::	::::::	::::@:@:@ : : :::::: 2(
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::- )
'
_output_shapes
:@: !

_output_shapes
:@:,"(
&
_output_shapes
:@ : #

_output_shapes
: :,$(
&
_output_shapes
: : %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
: : /

_output_shapes
: :,0(
&
_output_shapes
: @: 1

_output_shapes
:@:-2)
'
_output_shapes
:@:!3

_output_shapes	
::%4!

_output_shapes
:	: 5

_output_shapes
::$6 

_output_shapes

:: 7

_output_shapes
::$8 

_output_shapes

:: 9

_output_shapes
::%:!

_output_shapes
:	:!;

_output_shapes	
::.<*
(
_output_shapes
::!=

_output_shapes	
::->)
'
_output_shapes
:@: ?

_output_shapes
:@:,@(
&
_output_shapes
:@ : A

_output_shapes
: :,B(
&
_output_shapes
: : C

_output_shapes
::,D(
&
_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::,L(
&
_output_shapes
: : M

_output_shapes
: :,N(
&
_output_shapes
: @: O

_output_shapes
:@:-P)
'
_output_shapes
:@:!Q

_output_shapes	
::%R!

_output_shapes
:	: S

_output_shapes
::$T 

_output_shapes

:: U

_output_shapes
::$V 

_output_shapes

:: W

_output_shapes
::%X!

_output_shapes
:	:!Y

_output_shapes	
::.Z*
(
_output_shapes
::![

_output_shapes	
::-\)
'
_output_shapes
:@: ]

_output_shapes
:@:,^(
&
_output_shapes
:@ : _

_output_shapes
: :,`(
&
_output_shapes
: : a

_output_shapes
::,b(
&
_output_shapes
:: c

_output_shapes
::,d(
&
_output_shapes
:: e

_output_shapes
::f

_output_shapes
: 
Љ{

C__inference_Encoder_layer_call_and_return_conditional_losses_151432

inputs,
(conv1_enc_conv2d_readvariableop_resource-
)conv1_enc_biasadd_readvariableop_resource,
(conv2_enc_conv2d_readvariableop_resource-
)conv2_enc_biasadd_readvariableop_resource,
(conv3_enc_conv2d_readvariableop_resource-
)conv3_enc_biasadd_readvariableop_resource,
(conv4_enc_conv2d_readvariableop_resource-
)conv4_enc_biasadd_readvariableop_resource,
(conv5_enc_conv2d_readvariableop_resource-
)conv5_enc_biasadd_readvariableop_resource-
)bottleneck_matmul_readvariableop_resource.
*bottleneck_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2Ђ!bottleneck/BiasAdd/ReadVariableOpЂ bottleneck/MatMul/ReadVariableOpЂ conv1_enc/BiasAdd/ReadVariableOpЂconv1_enc/Conv2D/ReadVariableOpЂ conv2_enc/BiasAdd/ReadVariableOpЂconv2_enc/Conv2D/ReadVariableOpЂ conv3_enc/BiasAdd/ReadVariableOpЂconv3_enc/Conv2D/ReadVariableOpЂ conv4_enc/BiasAdd/ReadVariableOpЂconv4_enc/Conv2D/ReadVariableOpЂ conv5_enc/BiasAdd/ReadVariableOpЂconv5_enc/Conv2D/ReadVariableOpЂ z_log_var/BiasAdd/ReadVariableOpЂz_log_var/MatMul/ReadVariableOpЂz_mean/BiasAdd/ReadVariableOpЂz_mean/MatMul/ReadVariableOpГ
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOpС
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingSAME*
strides
2
conv1_enc/Conv2DЊ
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOpА
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
conv1_enc/ReluЙ
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPoolГ
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2_enc/Conv2D/ReadVariableOpд
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2_enc/Conv2DЊ
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2_enc/BiasAdd/ReadVariableOpА
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2_enc/ReluЙ
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ
*
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPoolГ
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv3_enc/Conv2D/ReadVariableOpд
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 *
paddingSAME*
strides
2
conv3_enc/Conv2DЊ
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3_enc/BiasAdd/ReadVariableOpА
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
conv3_enc/ReluЙ
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPoolГ
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv4_enc/Conv2D/ReadVariableOpд
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv4_enc/Conv2DЊ
 conv4_enc/BiasAdd/ReadVariableOpReadVariableOp)conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv4_enc/BiasAdd/ReadVariableOpА
conv4_enc/BiasAddBiasAddconv4_enc/Conv2D:output:0(conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4_enc/BiasAdd~
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4_enc/ReluЙ
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
maxpool4/MaxPoolД
conv5_enc/Conv2D/ReadVariableOpReadVariableOp(conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv5_enc/Conv2D/ReadVariableOpе
conv5_enc/Conv2DConv2Dmaxpool4/MaxPool:output:0'conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv5_enc/Conv2DЋ
 conv5_enc/BiasAdd/ReadVariableOpReadVariableOp)conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv5_enc/BiasAdd/ReadVariableOpБ
conv5_enc/BiasAddBiasAddconv5_enc/Conv2D:output:0(conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv5_enc/BiasAdd
conv5_enc/ReluReluconv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv5_enc/ReluК
maxpool5/MaxPoolMaxPoolconv5_enc/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
maxpool5/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten/Const
flatten/ReshapeReshapemaxpool5/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten/ReshapeЏ
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 bottleneck/MatMul/ReadVariableOpІ
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
bottleneck/MatMul­
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp­
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
bottleneck/BiasAddЂ
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMulbottleneck/BiasAdd:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
z_mean/MatMulЁ
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
z_mean/BiasAddЋ
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
z_log_var/MatMul/ReadVariableOpІ
z_log_var/MatMulMatMulbottleneck/BiasAdd:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
z_log_var/MatMulЊ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOpЉ
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
z_log_var/BiasAddg
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
sampling/strided_slice/stack
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_1
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_2
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slicek
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape_1
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice_1/stack
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_1
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_2Є
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1Ж
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
sampling/random_normal/shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sampling/random_normal/stddev
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2у­Р2-
+sampling/random_normal/RandomStandardNormalи
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
sampling/random_normal/mulИ
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sampling/Exp
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sampling/mul_1
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sampling/add
IdentityIdentityz_mean/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identityz_log_var/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identitysampling/add:z:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::2F
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
 conv5_enc/BiasAdd/ReadVariableOp conv5_enc/BiasAdd/ReadVariableOp2B
conv5_enc/Conv2D/ReadVariableOpconv5_enc/Conv2D/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
ч
њ
?__inference_VAE_layer_call_and_return_conditional_losses_150559

inputs
encoder_150494
encoder_150496
encoder_150498
encoder_150500
encoder_150502
encoder_150504
encoder_150506
encoder_150508
encoder_150510
encoder_150512
encoder_150514
encoder_150516
encoder_150518
encoder_150520
encoder_150522
encoder_150524
decoder_150529
decoder_150531
decoder_150533
decoder_150535
decoder_150537
decoder_150539
decoder_150541
decoder_150543
decoder_150545
decoder_150547
decoder_150549
decoder_150551
decoder_150553
decoder_150555
identityЂDecoder/StatefulPartitionedCallЂEncoder/StatefulPartitionedCallЖ
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_150494encoder_150496encoder_150498encoder_150500encoder_150502encoder_150504encoder_150506encoder_150508encoder_150510encoder_150512encoder_150514encoder_150516encoder_150518encoder_150520encoder_150522encoder_150524*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Encoder_layer_call_and_return_conditional_losses_1496472!
Encoder/StatefulPartitionedCallІ
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_150529decoder_150531decoder_150533decoder_150535decoder_150537decoder_150539decoder_150541decoder_150543decoder_150545decoder_150547decoder_150549decoder_150551decoder_150553decoder_150555*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_Decoder_layer_call_and_return_conditional_losses_1501722!
Decoder/StatefulPartitionedCallк
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
еI

C__inference_Encoder_layer_call_and_return_conditional_losses_149553

inputs
conv1_enc_149503
conv1_enc_149505
conv2_enc_149509
conv2_enc_149511
conv3_enc_149515
conv3_enc_149517
conv4_enc_149521
conv4_enc_149523
conv5_enc_149527
conv5_enc_149529
bottleneck_149534
bottleneck_149536
z_mean_149539
z_mean_149541
z_log_var_149544
z_log_var_149546
identity

identity_1

identity_2Ђ"bottleneck/StatefulPartitionedCallЂ!conv1_enc/StatefulPartitionedCallЂ!conv2_enc/StatefulPartitionedCallЂ!conv3_enc/StatefulPartitionedCallЂ!conv4_enc/StatefulPartitionedCallЂ!conv5_enc/StatefulPartitionedCallЂ sampling/StatefulPartitionedCallЂ!z_log_var/StatefulPartitionedCallЂz_mean/StatefulPartitionedCallЄ
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_149503conv1_enc_149505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ(*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_enc_layer_call_and_return_conditional_losses_1491852#
!conv1_enc/StatefulPartitionedCall
maxpool1/PartitionedCallPartitionedCall*conv1_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
maxpool1/PartitionedCallП
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_149509conv2_enc_149511*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_enc_layer_call_and_return_conditional_losses_1492132#
!conv2_enc/StatefulPartitionedCall
maxpool2/PartitionedCallPartitionedCall*conv2_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool2_layer_call_and_return_conditional_losses_1491282
maxpool2/PartitionedCallП
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_149515conv3_enc_149517*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
 *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_enc_layer_call_and_return_conditional_losses_1492412#
!conv3_enc/StatefulPartitionedCall
maxpool3/PartitionedCallPartitionedCall*conv3_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
maxpool3/PartitionedCallП
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_149521conv4_enc_149523*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_enc_layer_call_and_return_conditional_losses_1492692#
!conv4_enc/StatefulPartitionedCall
maxpool4/PartitionedCallPartitionedCall*conv4_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
maxpool4/PartitionedCallР
!conv5_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool4/PartitionedCall:output:0conv5_enc_149527conv5_enc_149529*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_enc_layer_call_and_return_conditional_losses_1492972#
!conv5_enc/StatefulPartitionedCall
maxpool5/PartitionedCallPartitionedCall*conv5_enc/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
maxpool5/PartitionedCall№
flatten/PartitionedCallPartitionedCall!maxpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1493202
flatten/PartitionedCallЛ
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_149534bottleneck_149536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_1493382$
"bottleneck/StatefulPartitionedCallВ
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_149539z_mean_149541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1493642 
z_mean/StatefulPartitionedCallС
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_149544z_log_var_149546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1493902#
!z_log_var/StatefulPartitionedCallН
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1494322"
 sampling/StatefulPartitionedCallМ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityУ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1Т

Identity_2Identity)sampling/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::2H
"bottleneck/StatefulPartitionedCall"bottleneck/StatefulPartitionedCall2F
!conv1_enc/StatefulPartitionedCall!conv1_enc/StatefulPartitionedCall2F
!conv2_enc/StatefulPartitionedCall!conv2_enc/StatefulPartitionedCall2F
!conv3_enc/StatefulPartitionedCall!conv3_enc/StatefulPartitionedCall2F
!conv4_enc/StatefulPartitionedCall!conv4_enc/StatefulPartitionedCall2F
!conv5_enc/StatefulPartitionedCall!conv5_enc/StatefulPartitionedCall2D
 sampling/StatefulPartitionedCall sampling/StatefulPartitionedCall2F
!z_log_var/StatefulPartitionedCall!z_log_var/StatefulPartitionedCall2@
z_mean/StatefulPartitionedCallz_mean/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
Г9

C__inference_Decoder_layer_call_and_return_conditional_losses_150172

inputs
decoding_150130
decoding_150132
conv5_dec_150136
conv5_dec_150138
conv4_dec_150142
conv4_dec_150144
conv3_dec_150148
conv3_dec_150150
conv2_dec_150154
conv2_dec_150156
conv1_dec_150160
conv1_dec_150162
output_150166
output_150168
identityЂ!conv1_dec/StatefulPartitionedCallЂ!conv2_dec/StatefulPartitionedCallЂ!conv3_dec/StatefulPartitionedCallЂ!conv4_dec/StatefulPartitionedCallЂ!conv5_dec/StatefulPartitionedCallЂ decoding/StatefulPartitionedCallЂoutput/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_150130decoding_150132*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_decoding_layer_call_and_return_conditional_losses_1497952"
 decoding/StatefulPartitionedCall
reshape/PartitionedCallPartitionedCall)decoding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1498252
reshape/PartitionedCallП
!conv5_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv5_dec_150136conv5_dec_150138*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv5_dec_layer_call_and_return_conditional_losses_1498442#
!conv5_dec/StatefulPartitionedCall
upsamp5/PartitionedCallPartitionedCall*conv5_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
upsamp5/PartitionedCallа
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp5/PartitionedCall:output:0conv4_dec_150142conv4_dec_150144*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_1498722#
!conv4_dec/StatefulPartitionedCall
upsamp4/PartitionedCallPartitionedCall*conv4_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
upsamp4/PartitionedCallа
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_150148conv3_dec_150150*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_1499002#
!conv3_dec/StatefulPartitionedCall
upsamp3/PartitionedCallPartitionedCall*conv3_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
upsamp3/PartitionedCallа
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_150154conv2_dec_150156*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_1499282#
!conv2_dec/StatefulPartitionedCall
upsamp2/PartitionedCallPartitionedCall*conv2_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
upsamp2/PartitionedCallа
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_150160conv1_dec_150162*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_1499562#
!conv1_dec/StatefulPartitionedCall
upsamp1/PartitionedCallPartitionedCall*conv1_dec/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
upsamp1/PartitionedCallС
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_150166output_150168*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall"^conv5_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:џџџџџџџџџ::::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2F
!conv5_dec/StatefulPartitionedCall!conv5_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я

о
E__inference_conv2_enc_layer_call_and_return_conditional_losses_149213

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
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
:џџџџџџџџџ2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я

о
E__inference_conv4_enc_layer_call_and_return_conditional_losses_151855

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
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
:џџџџџџџџџ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
У
А
$__inference_VAE_layer_call_fn_151258

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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28
identityЂStatefulPartitionedCall
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*@
_read_only_resource_inputs"
 	
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_VAE_layer_call_and_return_conditional_losses_1505592
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*Ј
_input_shapes
:џџџџџџџџџ(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
Ї
D
(__inference_reshape_layer_call_fn_152022

inputs
identityЭ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1498252
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Љ{

C__inference_Encoder_layer_call_and_return_conditional_losses_151345

inputs,
(conv1_enc_conv2d_readvariableop_resource-
)conv1_enc_biasadd_readvariableop_resource,
(conv2_enc_conv2d_readvariableop_resource-
)conv2_enc_biasadd_readvariableop_resource,
(conv3_enc_conv2d_readvariableop_resource-
)conv3_enc_biasadd_readvariableop_resource,
(conv4_enc_conv2d_readvariableop_resource-
)conv4_enc_biasadd_readvariableop_resource,
(conv5_enc_conv2d_readvariableop_resource-
)conv5_enc_biasadd_readvariableop_resource-
)bottleneck_matmul_readvariableop_resource.
*bottleneck_biasadd_readvariableop_resource)
%z_mean_matmul_readvariableop_resource*
&z_mean_biasadd_readvariableop_resource,
(z_log_var_matmul_readvariableop_resource-
)z_log_var_biasadd_readvariableop_resource
identity

identity_1

identity_2Ђ!bottleneck/BiasAdd/ReadVariableOpЂ bottleneck/MatMul/ReadVariableOpЂ conv1_enc/BiasAdd/ReadVariableOpЂconv1_enc/Conv2D/ReadVariableOpЂ conv2_enc/BiasAdd/ReadVariableOpЂconv2_enc/Conv2D/ReadVariableOpЂ conv3_enc/BiasAdd/ReadVariableOpЂconv3_enc/Conv2D/ReadVariableOpЂ conv4_enc/BiasAdd/ReadVariableOpЂconv4_enc/Conv2D/ReadVariableOpЂ conv5_enc/BiasAdd/ReadVariableOpЂconv5_enc/Conv2D/ReadVariableOpЂ z_log_var/BiasAdd/ReadVariableOpЂz_log_var/MatMul/ReadVariableOpЂz_mean/BiasAdd/ReadVariableOpЂz_mean/MatMul/ReadVariableOpГ
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOpС
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
paddingSAME*
strides
2
conv1_enc/Conv2DЊ
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOpА
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(2
conv1_enc/ReluЙ
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPoolГ
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2_enc/Conv2D/ReadVariableOpд
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2_enc/Conv2DЊ
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2_enc/BiasAdd/ReadVariableOpА
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
conv2_enc/ReluЙ
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ
*
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPoolГ
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv3_enc/Conv2D/ReadVariableOpд
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 *
paddingSAME*
strides
2
conv3_enc/Conv2DЊ
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3_enc/BiasAdd/ReadVariableOpА
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
conv3_enc/ReluЙ
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPoolГ
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv4_enc/Conv2D/ReadVariableOpд
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@*
paddingSAME*
strides
2
conv4_enc/Conv2DЊ
 conv4_enc/BiasAdd/ReadVariableOpReadVariableOp)conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv4_enc/BiasAdd/ReadVariableOpА
conv4_enc/BiasAddBiasAddconv4_enc/Conv2D:output:0(conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4_enc/BiasAdd~
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@2
conv4_enc/ReluЙ
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ@*
ksize
*
paddingSAME*
strides
2
maxpool4/MaxPoolД
conv5_enc/Conv2D/ReadVariableOpReadVariableOp(conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv5_enc/Conv2D/ReadVariableOpе
conv5_enc/Conv2DConv2Dmaxpool4/MaxPool:output:0'conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv5_enc/Conv2DЋ
 conv5_enc/BiasAdd/ReadVariableOpReadVariableOp)conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv5_enc/BiasAdd/ReadVariableOpБ
conv5_enc/BiasAddBiasAddconv5_enc/Conv2D:output:0(conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv5_enc/BiasAdd
conv5_enc/ReluReluconv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv5_enc/ReluК
maxpool5/MaxPoolMaxPoolconv5_enc/Relu:activations:0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
maxpool5/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten/Const
flatten/ReshapeReshapemaxpool5/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten/ReshapeЏ
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 bottleneck/MatMul/ReadVariableOpІ
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
bottleneck/MatMul­
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp­
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
bottleneck/BiasAddЂ
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMulbottleneck/BiasAdd:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
z_mean/MatMulЁ
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
z_mean/BiasAddЋ
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
z_log_var/MatMul/ReadVariableOpІ
z_log_var/MatMulMatMulbottleneck/BiasAdd:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
z_log_var/MatMulЊ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOpЉ
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
z_log_var/BiasAddg
sampling/ShapeShapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
sampling/strided_slice/stack
sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_1
sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice/stack_2
sampling/strided_sliceStridedSlicesampling/Shape:output:0%sampling/strided_slice/stack:output:0'sampling/strided_slice/stack_1:output:0'sampling/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slicek
sampling/Shape_1Shapez_mean/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape_1
sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2 
sampling/strided_slice_1/stack
 sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_1
 sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 sampling/strided_slice_1/stack_2Є
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1Ж
sampling/random_normal/shapePacksampling/strided_slice:output:0!sampling/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2
sampling/random_normal/shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sampling/random_normal/stddev
+sampling/random_normal/RandomStandardNormalRandomStandardNormal%sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*
seedБџх)*
seed2ЩС2-
+sampling/random_normal/RandomStandardNormalи
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
sampling/random_normal/mulИ
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x
sampling/mulMulsampling/mul/x:output:0z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sampling/Exp
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sampling/mul_1
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2
sampling/add
IdentityIdentityz_mean/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identityz_log_var/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identitysampling/add:z:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:џџџџџџџџџ(::::::::::::::::2F
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
 conv5_enc/BiasAdd/ReadVariableOp conv5_enc/BiasAdd/ReadVariableOp2B
conv5_enc/Conv2D/ReadVariableOpconv5_enc/Conv2D/ReadVariableOp2D
 z_log_var/BiasAdd/ReadVariableOp z_log_var/BiasAdd/ReadVariableOp2B
z_log_var/MatMul/ReadVariableOpz_log_var/MatMul/ReadVariableOp2>
z_mean/BiasAdd/ReadVariableOpz_mean/BiasAdd/ReadVariableOp2<
z_mean/MatMul/ReadVariableOpz_mean/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ(
 
_user_specified_nameinputs
Л
о
E__inference_conv2_dec_layer_call_and_return_conditional_losses_149928

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЕ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
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
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
ReluБ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 
_user_specified_nameinputs
с

*__inference_z_log_var_layer_call_fn_151952

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallј
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1493902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я

о
E__inference_conv3_enc_layer_call_and_return_conditional_losses_149241

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
 *
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
:џџџџџџџџџ
 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:џџџџџџџџџ
 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ

 
_user_specified_nameinputs
љ
`
D__inference_maxpool5_layer_call_and_return_conditional_losses_149164

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љ
`
D__inference_maxpool1_layer_call_and_return_conditional_losses_149116

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Л
serving_defaultЇ
C
input_18
serving_default_input_1:0џџџџџџџџџ(D
output_18
StatefulPartitionedCall:0џџџџџџџџџ(tensorflow/serving/predict:Л
§
encoder
decoder
total_loss_tracker
reconstruction_loss_tracker
kl_loss_tracker
	optimizer
loss
trainable_variables
		variables

regularization_losses
	keras_api

signatures
Љ_default_save_signature
Њ__call__
+Ћ&call_and_return_all_conditional_losses"Н
_tf_keras_modelЃ{"class_name": "VAEInternal", "name": "VAE", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "VAEInternal"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
у
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
layer_with_weights-4
layer-9
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
trainable_variables
	variables
regularization_losses
 	keras_api
Ќ__call__
+­&call_and_return_all_conditional_losses"Ќ|
_tf_keras_network|{"class_name": "Functional", "name": "Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_enc", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool5", "inbound_nodes": [[["conv5_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling", "trainable": true, "dtype": "float32"}, "name": "sampling", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 30, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_enc", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool5", "inbound_nodes": [[["conv5_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling", "trainable": true, "dtype": "float32"}, "name": "sampling", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling", 0, 0]]}}}
щt
!layer-0
"layer_with_weights-0
"layer-1
#layer-2
$layer_with_weights-1
$layer-3
%layer-4
&layer_with_weights-2
&layer-5
'layer-6
(layer_with_weights-3
(layer-7
)layer-8
*layer_with_weights-4
*layer-9
+layer-10
,layer_with_weights-5
,layer-11
-layer-12
.layer_with_weights-6
.layer-13
/trainable_variables
0	variables
1regularization_losses
2	keras_api
Ў__call__
+Џ&call_and_return_all_conditional_losses"шp
_tf_keras_networkЬp{"class_name": "Functional", "name": "Decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 1, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp5", "inbound_nodes": [[["conv5_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["upsamp5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [25, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 12]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 1, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp5", "inbound_nodes": [[["conv5_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["upsamp5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [25, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}}}
Ч
	3total
	4count
5	variables
6	keras_api"
_tf_keras_metricv{"class_name": "Mean", "name": "total_loss", "dtype": "float32", "config": {"name": "total_loss", "dtype": "float32"}}
к
	7total
	8count
9	variables
:	keras_api"Ѓ
_tf_keras_metric{"class_name": "Mean", "name": "reconstruction_loss", "dtype": "float32", "config": {"name": "reconstruction_loss", "dtype": "float32"}}
С
	;total
	<count
=	variables
>	keras_api"
_tf_keras_metricp{"class_name": "Mean", "name": "kl_loss", "dtype": "float32", "config": {"name": "kl_loss", "dtype": "float32"}}
Ћ
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rateDmэEmюFmяGm№HmёImђJmѓKmєLmѕMmіNmїOmјPmљQmњRmћSmќTm§UmўVmџWmXmYmZm[m\m]m^m_m`mamDvEvFvGvHvIvJvKvLvMvNvOvPvQvRvSvTvUvVvWvXvYv ZvЁ[vЂ\vЃ]vЄ^vЅ_vІ`vЇavЈ"
	optimizer
 "
trackable_dict_wrapper

D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29"
trackable_list_wrapper
Ж
D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15
T16
U17
V18
W19
X20
Y21
Z22
[23
\24
]25
^26
_27
`28
a29
330
431
732
833
;34
<35"
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
trainable_variables

blayers
		variables

regularization_losses
cnon_trainable_variables
dmetrics
elayer_regularization_losses
flayer_metrics
Њ__call__
Љ_default_save_signature
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
-
Аserving_default"
signature_map
"
_tf_keras_input_layerт{"class_name": "InputLayer", "name": "input_encoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}}
ѓ	

Dkernel
Ebias
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"Ь
_tf_keras_layerВ{"class_name": "Conv2D", "name": "conv1_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 30, 1]}}
ђ
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"с
_tf_keras_layerЧ{"class_name": "MaxPooling2D", "name": "maxpool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
є	

Fkernel
Gbias
otrainable_variables
p	variables
qregularization_losses
r	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Conv2D", "name": "conv2_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 15, 8]}}
ђ
strainable_variables
t	variables
uregularization_losses
v	keras_api
З__call__
+И&call_and_return_all_conditional_losses"с
_tf_keras_layerЧ{"class_name": "MaxPooling2D", "name": "maxpool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ѕ	

Hkernel
Ibias
wtrainable_variables
x	variables
yregularization_losses
z	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"Ю
_tf_keras_layerД{"class_name": "Conv2D", "name": "conv3_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 8, 16]}}
ђ
{trainable_variables
|	variables
}regularization_losses
~	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"с
_tf_keras_layerЧ{"class_name": "MaxPooling2D", "name": "maxpool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ї	

Jkernel
Kbias
trainable_variables
	variables
regularization_losses
	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Conv2D", "name": "conv4_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 4, 32]}}
і
trainable_variables
	variables
regularization_losses
	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"с
_tf_keras_layerЧ{"class_name": "MaxPooling2D", "name": "maxpool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
љ	

Lkernel
Mbias
trainable_variables
	variables
regularization_losses
	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"Ю
_tf_keras_layerД{"class_name": "Conv2D", "name": "conv5_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv5_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 2, 64]}}
і
trainable_variables
	variables
regularization_losses
	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"с
_tf_keras_layerЧ{"class_name": "MaxPooling2D", "name": "maxpool5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш
trainable_variables
	variables
regularization_losses
	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


Nkernel
Obias
trainable_variables
	variables
regularization_losses
	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"е
_tf_keras_layerЛ{"class_name": "Dense", "name": "bottleneck", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
і

Pkernel
Qbias
trainable_variables
	variables
regularization_losses
	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
ќ

Rkernel
Sbias
trainable_variables
	variables
regularization_losses
	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"б
_tf_keras_layerЗ{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
Л
trainable_variables
 	variables
Ёregularization_losses
Ђ	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"І
_tf_keras_layer{"class_name": "Sampling", "name": "sampling", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "sampling", "trainable": true, "dtype": "float32"}}

D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15"
trackable_list_wrapper

D0
E1
F2
G3
H4
I5
J6
K7
L8
M9
N10
O11
P12
Q13
R14
S15"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
trainable_variables
Ѓlayers
	variables
regularization_losses
Єnon_trainable_variables
Ѕmetrics
 Іlayer_regularization_losses
Їlayer_metrics
Ќ__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
ї"є
_tf_keras_input_layerд{"class_name": "InputLayer", "name": "input_decoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}}
ћ

Tkernel
Ubias
Јtrainable_variables
Љ	variables
Њregularization_losses
Ћ	keras_api
Я__call__
+а&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "decoding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
ћ
Ќtrainable_variables
­	variables
Ўregularization_losses
Џ	keras_api
б__call__
+в&call_and_return_all_conditional_losses"ц
_tf_keras_layerЬ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 1, 128]}}}
ћ	

Vkernel
Wbias
Аtrainable_variables
Б	variables
Вregularization_losses
Г	keras_api
г__call__
+д&call_and_return_all_conditional_losses"а
_tf_keras_layerЖ{"class_name": "Conv2D", "name": "conv5_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv5_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 1, 128]}}
П
Дtrainable_variables
Е	variables
Жregularization_losses
З	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"Њ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
њ	

Xkernel
Ybias
Иtrainable_variables
Й	variables
Кregularization_losses
Л	keras_api
з__call__
+и&call_and_return_all_conditional_losses"Я
_tf_keras_layerЕ{"class_name": "Conv2D", "name": "conv4_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 2, 128]}}
П
Мtrainable_variables
Н	variables
Оregularization_losses
П	keras_api
й__call__
+к&call_and_return_all_conditional_losses"Њ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ј	

Zkernel
[bias
Рtrainable_variables
С	variables
Тregularization_losses
У	keras_api
л__call__
+м&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Conv2D", "name": "conv3_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 4, 64]}}
П
Фtrainable_variables
Х	variables
Цregularization_losses
Ч	keras_api
н__call__
+о&call_and_return_all_conditional_losses"Њ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
љ	

\kernel
]bias
Шtrainable_variables
Щ	variables
Ъregularization_losses
Ы	keras_api
п__call__
+р&call_and_return_all_conditional_losses"Ю
_tf_keras_layerД{"class_name": "Conv2D", "name": "conv2_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 8, 32]}}
П
Ьtrainable_variables
Э	variables
Юregularization_losses
Я	keras_api
с__call__
+т&call_and_return_all_conditional_losses"Њ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
љ	

^kernel
_bias
аtrainable_variables
б	variables
вregularization_losses
г	keras_api
у__call__
+ф&call_and_return_all_conditional_losses"Ю
_tf_keras_layerД{"class_name": "Conv2D", "name": "conv1_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 16, 16]}}
П
дtrainable_variables
е	variables
жregularization_losses
з	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"Њ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
і	

`kernel
abias
иtrainable_variables
й	variables
кregularization_losses
л	keras_api
ч__call__
+ш&call_and_return_all_conditional_losses"Ы
_tf_keras_layerБ{"class_name": "Conv2D", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [25, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 32, 8]}}

T0
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11
`12
a13"
trackable_list_wrapper

T0
U1
V2
W3
X4
Y5
Z6
[7
\8
]9
^10
_11
`12
a13"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
/trainable_variables
мlayers
0	variables
1regularization_losses
нnon_trainable_variables
оmetrics
 пlayer_regularization_losses
рlayer_metrics
Ў__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
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
:  (2total
:  (2count
.
;0
<1"
trackable_list_wrapper
-
=	variables"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
*:(2conv1_enc/kernel
:2conv1_enc/bias
*:(2conv2_enc/kernel
:2conv2_enc/bias
*:( 2conv3_enc/kernel
: 2conv3_enc/bias
*:( @2conv4_enc/kernel
:@2conv4_enc/bias
+:)@2conv5_enc/kernel
:2conv5_enc/bias
$:"	2bottleneck/kernel
:2bottleneck/bias
:2z_mean/kernel
:2z_mean/bias
": 2z_log_var/kernel
:2z_log_var/bias
": 	2decoding/kernel
:2decoding/bias
,:*2conv5_dec/kernel
:2conv5_dec/bias
+:)@2conv4_dec/kernel
:@2conv4_dec/bias
*:(@ 2conv3_dec/kernel
: 2conv3_dec/bias
*:( 2conv2_dec/kernel
:2conv2_dec/bias
*:(2conv1_dec/kernel
:2conv1_dec/bias
':%2output/kernel
:2output/bias
.
0
1"
trackable_list_wrapper
J
30
41
72
83
;4
<5"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
V

total_loss
reconstruction_loss
kl_loss"
trackable_dict_wrapper
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
Е
gtrainable_variables
сlayers
h	variables
iregularization_losses
тnon_trainable_variables
уmetrics
 фlayer_regularization_losses
хlayer_metrics
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
ktrainable_variables
цlayers
l	variables
mregularization_losses
чnon_trainable_variables
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
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
Е
otrainable_variables
ыlayers
p	variables
qregularization_losses
ьnon_trainable_variables
эmetrics
 юlayer_regularization_losses
яlayer_metrics
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
strainable_variables
№layers
t	variables
uregularization_losses
ёnon_trainable_variables
ђmetrics
 ѓlayer_regularization_losses
єlayer_metrics
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
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
Е
wtrainable_variables
ѕlayers
x	variables
yregularization_losses
іnon_trainable_variables
їmetrics
 јlayer_regularization_losses
љlayer_metrics
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
{trainable_variables
њlayers
|	variables
}regularization_losses
ћnon_trainable_variables
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
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
З
trainable_variables
џlayers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
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
И
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
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
И
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
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
И
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
  layer_regularization_losses
Ёlayer_metrics
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
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
И
trainable_variables
Ђlayers
	variables
regularization_losses
Ѓnon_trainable_variables
Єmetrics
 Ѕlayer_regularization_losses
Іlayer_metrics
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
trainable_variables
Їlayers
 	variables
Ёregularization_losses
Јnon_trainable_variables
Љmetrics
 Њlayer_regularization_losses
Ћlayer_metrics
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object

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
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
И
Јtrainable_variables
Ќlayers
Љ	variables
Њregularization_losses
­non_trainable_variables
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ќtrainable_variables
Бlayers
­	variables
Ўregularization_losses
Вnon_trainable_variables
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
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
И
Аtrainable_variables
Жlayers
Б	variables
Вregularization_losses
Зnon_trainable_variables
Иmetrics
 Йlayer_regularization_losses
Кlayer_metrics
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Дtrainable_variables
Лlayers
Е	variables
Жregularization_losses
Мnon_trainable_variables
Нmetrics
 Оlayer_regularization_losses
Пlayer_metrics
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
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
И
Иtrainable_variables
Рlayers
Й	variables
Кregularization_losses
Сnon_trainable_variables
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Мtrainable_variables
Хlayers
Н	variables
Оregularization_losses
Цnon_trainable_variables
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Рtrainable_variables
Ъlayers
С	variables
Тregularization_losses
Ыnon_trainable_variables
Ьmetrics
 Эlayer_regularization_losses
Юlayer_metrics
л__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Фtrainable_variables
Яlayers
Х	variables
Цregularization_losses
аnon_trainable_variables
бmetrics
 вlayer_regularization_losses
гlayer_metrics
н__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Шtrainable_variables
дlayers
Щ	variables
Ъregularization_losses
еnon_trainable_variables
жmetrics
 зlayer_regularization_losses
иlayer_metrics
п__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ьtrainable_variables
йlayers
Э	variables
Юregularization_losses
кnon_trainable_variables
лmetrics
 мlayer_regularization_losses
нlayer_metrics
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
аtrainable_variables
оlayers
б	variables
вregularization_losses
пnon_trainable_variables
рmetrics
 сlayer_regularization_losses
тlayer_metrics
у__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
дtrainable_variables
уlayers
е	variables
жregularization_losses
фnon_trainable_variables
хmetrics
 цlayer_regularization_losses
чlayer_metrics
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
иtrainable_variables
шlayers
й	variables
кregularization_losses
щnon_trainable_variables
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
ч__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object

!0
"1
#2
$3
%4
&5
'6
(7
)8
*9
+10
,11
-12
.13"
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
/:-2Adam/conv1_enc/kernel/m
!:2Adam/conv1_enc/bias/m
/:-2Adam/conv2_enc/kernel/m
!:2Adam/conv2_enc/bias/m
/:- 2Adam/conv3_enc/kernel/m
!: 2Adam/conv3_enc/bias/m
/:- @2Adam/conv4_enc/kernel/m
!:@2Adam/conv4_enc/bias/m
0:.@2Adam/conv5_enc/kernel/m
": 2Adam/conv5_enc/bias/m
):'	2Adam/bottleneck/kernel/m
": 2Adam/bottleneck/bias/m
$:"2Adam/z_mean/kernel/m
:2Adam/z_mean/bias/m
':%2Adam/z_log_var/kernel/m
!:2Adam/z_log_var/bias/m
':%	2Adam/decoding/kernel/m
!:2Adam/decoding/bias/m
1:/2Adam/conv5_dec/kernel/m
": 2Adam/conv5_dec/bias/m
0:.@2Adam/conv4_dec/kernel/m
!:@2Adam/conv4_dec/bias/m
/:-@ 2Adam/conv3_dec/kernel/m
!: 2Adam/conv3_dec/bias/m
/:- 2Adam/conv2_dec/kernel/m
!:2Adam/conv2_dec/bias/m
/:-2Adam/conv1_dec/kernel/m
!:2Adam/conv1_dec/bias/m
,:*2Adam/output/kernel/m
:2Adam/output/bias/m
/:-2Adam/conv1_enc/kernel/v
!:2Adam/conv1_enc/bias/v
/:-2Adam/conv2_enc/kernel/v
!:2Adam/conv2_enc/bias/v
/:- 2Adam/conv3_enc/kernel/v
!: 2Adam/conv3_enc/bias/v
/:- @2Adam/conv4_enc/kernel/v
!:@2Adam/conv4_enc/bias/v
0:.@2Adam/conv5_enc/kernel/v
": 2Adam/conv5_enc/bias/v
):'	2Adam/bottleneck/kernel/v
": 2Adam/bottleneck/bias/v
$:"2Adam/z_mean/kernel/v
:2Adam/z_mean/bias/v
':%2Adam/z_log_var/kernel/v
!:2Adam/z_log_var/bias/v
':%	2Adam/decoding/kernel/v
!:2Adam/decoding/bias/v
1:/2Adam/conv5_dec/kernel/v
": 2Adam/conv5_dec/bias/v
0:.@2Adam/conv4_dec/kernel/v
!:@2Adam/conv4_dec/bias/v
/:-@ 2Adam/conv3_dec/kernel/v
!: 2Adam/conv3_dec/bias/v
/:- 2Adam/conv2_dec/kernel/v
!:2Adam/conv2_dec/bias/v
/:-2Adam/conv1_dec/kernel/v
!:2Adam/conv1_dec/bias/v
,:*2Adam/output/kernel/v
:2Adam/output/bias/v
ч2ф
!__inference__wrapped_model_149110О
В
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
annotationsЊ *.Ђ+
)&
input_1џџџџџџџџџ(
б2Ю
$__inference_VAE_layer_call_fn_151193
$__inference_VAE_layer_call_fn_150622
$__inference_VAE_layer_call_fn_150687
$__inference_VAE_layer_call_fn_151258Г
ЊВІ
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
annotationsЊ *
 
Н2К
?__inference_VAE_layer_call_and_return_conditional_losses_151128
?__inference_VAE_layer_call_and_return_conditional_losses_150945
?__inference_VAE_layer_call_and_return_conditional_losses_150420
?__inference_VAE_layer_call_and_return_conditional_losses_150488Г
ЊВІ
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
annotationsЊ *
 
ю2ы
(__inference_Encoder_layer_call_fn_149686
(__inference_Encoder_layer_call_fn_151473
(__inference_Encoder_layer_call_fn_151514
(__inference_Encoder_layer_call_fn_149592Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
C__inference_Encoder_layer_call_and_return_conditional_losses_149444
C__inference_Encoder_layer_call_and_return_conditional_losses_151345
C__inference_Encoder_layer_call_and_return_conditional_losses_151432
C__inference_Encoder_layer_call_and_return_conditional_losses_149497Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
(__inference_Decoder_layer_call_fn_151751
(__inference_Decoder_layer_call_fn_150203
(__inference_Decoder_layer_call_fn_151784
(__inference_Decoder_layer_call_fn_150125Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
к2з
C__inference_Decoder_layer_call_and_return_conditional_losses_151616
C__inference_Decoder_layer_call_and_return_conditional_losses_151718
C__inference_Decoder_layer_call_and_return_conditional_losses_150046
C__inference_Decoder_layer_call_and_return_conditional_losses_150001Р
ЗВГ
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
kwonlydefaultsЊ 
annotationsЊ *
 
ЫBШ
$__inference_signature_wrapper_150762input_1"
В
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
annotationsЊ *
 
д2б
*__inference_conv1_enc_layer_call_fn_151804Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv1_enc_layer_call_and_return_conditional_losses_151795Ђ
В
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
annotationsЊ *
 
2
)__inference_maxpool1_layer_call_fn_149122р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ќ2Љ
D__inference_maxpool1_layer_call_and_return_conditional_losses_149116р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv2_enc_layer_call_fn_151824Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv2_enc_layer_call_and_return_conditional_losses_151815Ђ
В
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
annotationsЊ *
 
2
)__inference_maxpool2_layer_call_fn_149134р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ќ2Љ
D__inference_maxpool2_layer_call_and_return_conditional_losses_149128р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv3_enc_layer_call_fn_151844Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv3_enc_layer_call_and_return_conditional_losses_151835Ђ
В
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
annotationsЊ *
 
2
)__inference_maxpool3_layer_call_fn_149146р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ќ2Љ
D__inference_maxpool3_layer_call_and_return_conditional_losses_149140р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv4_enc_layer_call_fn_151864Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv4_enc_layer_call_and_return_conditional_losses_151855Ђ
В
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
annotationsЊ *
 
2
)__inference_maxpool4_layer_call_fn_149158р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ќ2Љ
D__inference_maxpool4_layer_call_and_return_conditional_losses_149152р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv5_enc_layer_call_fn_151884Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv5_enc_layer_call_and_return_conditional_losses_151875Ђ
В
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
annotationsЊ *
 
2
)__inference_maxpool5_layer_call_fn_149170р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ќ2Љ
D__inference_maxpool5_layer_call_and_return_conditional_losses_149164р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
в2Я
(__inference_flatten_layer_call_fn_151895Ђ
В
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
annotationsЊ *
 
э2ъ
C__inference_flatten_layer_call_and_return_conditional_losses_151890Ђ
В
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
annotationsЊ *
 
е2в
+__inference_bottleneck_layer_call_fn_151914Ђ
В
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
annotationsЊ *
 
№2э
F__inference_bottleneck_layer_call_and_return_conditional_losses_151905Ђ
В
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
annotationsЊ *
 
б2Ю
'__inference_z_mean_layer_call_fn_151933Ђ
В
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
annotationsЊ *
 
ь2щ
B__inference_z_mean_layer_call_and_return_conditional_losses_151924Ђ
В
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
annotationsЊ *
 
д2б
*__inference_z_log_var_layer_call_fn_151952Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_z_log_var_layer_call_and_return_conditional_losses_151943Ђ
В
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
annotationsЊ *
 
г2а
)__inference_sampling_layer_call_fn_151984Ђ
В
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
annotationsЊ *
 
ю2ы
D__inference_sampling_layer_call_and_return_conditional_losses_151978Ђ
В
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
annotationsЊ *
 
г2а
)__inference_decoding_layer_call_fn_152003Ђ
В
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
annotationsЊ *
 
ю2ы
D__inference_decoding_layer_call_and_return_conditional_losses_151994Ђ
В
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
annotationsЊ *
 
в2Я
(__inference_reshape_layer_call_fn_152022Ђ
В
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
annotationsЊ *
 
э2ъ
C__inference_reshape_layer_call_and_return_conditional_losses_152017Ђ
В
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
annotationsЊ *
 
д2б
*__inference_conv5_dec_layer_call_fn_152042Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv5_dec_layer_call_and_return_conditional_losses_152033Ђ
В
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
annotationsЊ *
 
2
(__inference_upsamp5_layer_call_fn_149705р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ћ2Ј
C__inference_upsamp5_layer_call_and_return_conditional_losses_149699р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv4_dec_layer_call_fn_152062Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv4_dec_layer_call_and_return_conditional_losses_152053Ђ
В
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
annotationsЊ *
 
2
(__inference_upsamp4_layer_call_fn_149724р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ћ2Ј
C__inference_upsamp4_layer_call_and_return_conditional_losses_149718р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv3_dec_layer_call_fn_152082Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv3_dec_layer_call_and_return_conditional_losses_152073Ђ
В
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
annotationsЊ *
 
2
(__inference_upsamp3_layer_call_fn_149743р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ћ2Ј
C__inference_upsamp3_layer_call_and_return_conditional_losses_149737р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv2_dec_layer_call_fn_152102Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv2_dec_layer_call_and_return_conditional_losses_152093Ђ
В
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
annotationsЊ *
 
2
(__inference_upsamp2_layer_call_fn_149762р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ћ2Ј
C__inference_upsamp2_layer_call_and_return_conditional_losses_149756р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
д2б
*__inference_conv1_dec_layer_call_fn_152122Ђ
В
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
annotationsЊ *
 
я2ь
E__inference_conv1_dec_layer_call_and_return_conditional_losses_152113Ђ
В
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
annotationsЊ *
 
2
(__inference_upsamp1_layer_call_fn_149781р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Ћ2Ј
C__inference_upsamp1_layer_call_and_return_conditional_losses_149775р
В
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
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
б2Ю
'__inference_output_layer_call_fn_152142Ђ
В
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
annotationsЊ *
 
ь2щ
B__inference_output_layer_call_and_return_conditional_losses_152133Ђ
В
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
annotationsЊ *
 й
C__inference_Decoder_layer_call_and_return_conditional_losses_150001TUVWXYZ[\]^_`a>Ђ;
4Ђ1
'$
input_decoderџџџџџџџџџ
p

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 й
C__inference_Decoder_layer_call_and_return_conditional_losses_150046TUVWXYZ[\]^_`a>Ђ;
4Ђ1
'$
input_decoderџџџџџџџџџ
p 

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
C__inference_Decoder_layer_call_and_return_conditional_losses_151616xTUVWXYZ[\]^_`a7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "-Ђ*
# 
0џџџџџџџџџ(
 П
C__inference_Decoder_layer_call_and_return_conditional_losses_151718xTUVWXYZ[\]^_`a7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "-Ђ*
# 
0џџџџџџџџџ(
 Б
(__inference_Decoder_layer_call_fn_150125TUVWXYZ[\]^_`a>Ђ;
4Ђ1
'$
input_decoderџџџџџџџџџ
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџБ
(__inference_Decoder_layer_call_fn_150203TUVWXYZ[\]^_`a>Ђ;
4Ђ1
'$
input_decoderџџџџџџџџџ
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЉ
(__inference_Decoder_layer_call_fn_151751}TUVWXYZ[\]^_`a7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЉ
(__inference_Decoder_layer_call_fn_151784}TUVWXYZ[\]^_`a7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
C__inference_Encoder_layer_call_and_return_conditional_losses_149444ЦDEFGHIJKLMNOPQRSFЂC
<Ђ9
/,
input_encoderџџџџџџџџџ(
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 
C__inference_Encoder_layer_call_and_return_conditional_losses_149497ЦDEFGHIJKLMNOPQRSFЂC
<Ђ9
/,
input_encoderџџџџџџџџџ(
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 
C__inference_Encoder_layer_call_and_return_conditional_losses_151345ПDEFGHIJKLMNOPQRS?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ(
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 
C__inference_Encoder_layer_call_and_return_conditional_losses_151432ПDEFGHIJKLMNOPQRS?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ(
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 у
(__inference_Encoder_layer_call_fn_149592ЖDEFGHIJKLMNOPQRSFЂC
<Ђ9
/,
input_encoderџџџџџџџџџ(
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџу
(__inference_Encoder_layer_call_fn_149686ЖDEFGHIJKLMNOPQRSFЂC
<Ђ9
/,
input_encoderџџџџџџџџџ(
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџм
(__inference_Encoder_layer_call_fn_151473ЏDEFGHIJKLMNOPQRS?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ(
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџм
(__inference_Encoder_layer_call_fn_151514ЏDEFGHIJKLMNOPQRS?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ(
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџу
?__inference_VAE_layer_call_and_return_conditional_losses_150420DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a<Ђ9
2Ђ/
)&
input_1џџџџџџџџџ(
p
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 у
?__inference_VAE_layer_call_and_return_conditional_losses_150488DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a<Ђ9
2Ђ/
)&
input_1џџџџџџџџџ(
p 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 а
?__inference_VAE_layer_call_and_return_conditional_losses_150945DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ(
p
Њ "-Ђ*
# 
0џџџџџџџџџ(
 а
?__inference_VAE_layer_call_and_return_conditional_losses_151128DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ(
p 
Њ "-Ђ*
# 
0џџџџџџџџџ(
 Л
$__inference_VAE_layer_call_fn_150622DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a<Ђ9
2Ђ/
)&
input_1џџџџџџџџџ(
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЛ
$__inference_VAE_layer_call_fn_150687DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a<Ђ9
2Ђ/
)&
input_1џџџџџџџџџ(
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџК
$__inference_VAE_layer_call_fn_151193DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ(
p
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџК
$__inference_VAE_layer_call_fn_151258DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a;Ђ8
1Ђ.
(%
inputsџџџџџџџџџ(
p 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџН
!__inference__wrapped_model_149110DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a8Ђ5
.Ђ+
)&
input_1џџџџџџџџџ(
Њ ";Њ8
6
output_1*'
output_1џџџџџџџџџ(Ї
F__inference_bottleneck_layer_call_and_return_conditional_losses_151905]NO0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 
+__inference_bottleneck_layer_call_fn_151914PNO0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџк
E__inference_conv1_dec_layer_call_and_return_conditional_losses_152113^_IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 В
*__inference_conv1_dec_layer_call_fn_152122^_IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЕ
E__inference_conv1_enc_layer_call_and_return_conditional_losses_151795lDE7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ(
Њ "-Ђ*
# 
0џџџџџџџџџ(
 
*__inference_conv1_enc_layer_call_fn_151804_DE7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ(
Њ " џџџџџџџџџ(к
E__inference_conv2_dec_layer_call_and_return_conditional_losses_152093\]IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 В
*__inference_conv2_dec_layer_call_fn_152102\]IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЕ
E__inference_conv2_enc_layer_call_and_return_conditional_losses_151815lFG7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ
 
*__inference_conv2_enc_layer_call_fn_151824_FG7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџк
E__inference_conv3_dec_layer_call_and_return_conditional_losses_152073Z[IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ 
 В
*__inference_conv3_dec_layer_call_fn_152082Z[IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ Е
E__inference_conv3_enc_layer_call_and_return_conditional_losses_151835lHI7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ

Њ "-Ђ*
# 
0џџџџџџџџџ
 
 
*__inference_conv3_enc_layer_call_fn_151844_HI7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ

Њ " џџџџџџџџџ
 л
E__inference_conv4_dec_layer_call_and_return_conditional_losses_152053XYJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@
 Г
*__inference_conv4_dec_layer_call_fn_152062XYJЂG
@Ђ=
;8
inputs,џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ@Е
E__inference_conv4_enc_layer_call_and_return_conditional_losses_151855lJK7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ@
 
*__inference_conv4_enc_layer_call_fn_151864_JK7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ@З
E__inference_conv5_dec_layer_call_and_return_conditional_losses_152033nVW8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
*__inference_conv5_dec_layer_call_fn_152042aVW8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЖ
E__inference_conv5_enc_layer_call_and_return_conditional_losses_151875mLM7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ ".Ђ+
$!
0џџџџџџџџџ
 
*__inference_conv5_enc_layer_call_fn_151884`LM7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@
Њ "!џџџџџџџџџЅ
D__inference_decoding_layer_call_and_return_conditional_losses_151994]TU/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 }
)__inference_decoding_layer_call_fn_152003PTU/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЉ
C__inference_flatten_layer_call_and_return_conditional_losses_151890b8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
(__inference_flatten_layer_call_fn_151895U8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "џџџџџџџџџч
D__inference_maxpool1_layer_call_and_return_conditional_losses_149116RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
)__inference_maxpool1_layer_call_fn_149122RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџч
D__inference_maxpool2_layer_call_and_return_conditional_losses_149128RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
)__inference_maxpool2_layer_call_fn_149134RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџч
D__inference_maxpool3_layer_call_and_return_conditional_losses_149140RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
)__inference_maxpool3_layer_call_fn_149146RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџч
D__inference_maxpool4_layer_call_and_return_conditional_losses_149152RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
)__inference_maxpool4_layer_call_fn_149158RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџч
D__inference_maxpool5_layer_call_and_return_conditional_losses_149164RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 П
)__inference_maxpool5_layer_call_fn_149170RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџз
B__inference_output_layer_call_and_return_conditional_losses_152133`aIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Џ
'__inference_output_layer_call_fn_152142`aIЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЉ
C__inference_reshape_layer_call_and_return_conditional_losses_152017b0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
(__inference_reshape_layer_call_fn_152022U0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЬ
D__inference_sampling_layer_call_and_return_conditional_losses_151978ZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 Ѓ
)__inference_sampling_layer_call_fn_151984vZЂW
PЂM
KH
"
inputs/0џџџџџџџџџ
"
inputs/1џџџџџџџџџ
Њ "џџџџџџџџџЫ
$__inference_signature_wrapper_150762ЂDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`aCЂ@
Ђ 
9Њ6
4
input_1)&
input_1џџџџџџџџџ(";Њ8
6
output_1*'
output_1џџџџџџџџџ(ц
C__inference_upsamp1_layer_call_and_return_conditional_losses_149775RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
(__inference_upsamp1_layer_call_fn_149781RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџц
C__inference_upsamp2_layer_call_and_return_conditional_losses_149756RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
(__inference_upsamp2_layer_call_fn_149762RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџц
C__inference_upsamp3_layer_call_and_return_conditional_losses_149737RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
(__inference_upsamp3_layer_call_fn_149743RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџц
C__inference_upsamp4_layer_call_and_return_conditional_losses_149718RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
(__inference_upsamp4_layer_call_fn_149724RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџц
C__inference_upsamp5_layer_call_and_return_conditional_losses_149699RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 О
(__inference_upsamp5_layer_call_fn_149705RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЅ
E__inference_z_log_var_layer_call_and_return_conditional_losses_151943\RS/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 }
*__inference_z_log_var_layer_call_fn_151952ORS/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџЂ
B__inference_z_mean_layer_call_and_return_conditional_losses_151924\PQ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 z
'__inference_z_mean_layer_call_fn_151933OPQ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ