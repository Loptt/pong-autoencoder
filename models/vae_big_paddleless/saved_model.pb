ηΠ%
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
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ψ
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
°Ή
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*κΈ
valueίΈBΫΈ BΣΈ
γ
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
ψ
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
Β
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
Clearning_rateDmνEmξFmοGmπHmρImςJmσKmτLmυMmφNmχOmψPmωQmϊRmϋSmόTmύUmώVm?WmXmYmZm[m\m]m^m_m`mamDvEvFvGvHvIvJvKvLvMvNvOvPvQvRvSvTvUvVvWvXvYv Zv‘[v’\v£]v€^v₯_v¦`v§av¨
 
ζ
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
‘regularization_losses
’	keras_api
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
²
trainable_variables
£layers
	variables
regularization_losses
€non_trainable_variables
₯metrics
 ¦layer_regularization_losses
§layer_metrics
 
l

Tkernel
Ubias
¨trainable_variables
©	variables
ͺregularization_losses
«	keras_api
V
¬trainable_variables
­	variables
?regularization_losses
―	keras_api
l

Vkernel
Wbias
°trainable_variables
±	variables
²regularization_losses
³	keras_api
V
΄trainable_variables
΅	variables
Άregularization_losses
·	keras_api
l

Xkernel
Ybias
Έtrainable_variables
Ή	variables
Ίregularization_losses
»	keras_api
V
Όtrainable_variables
½	variables
Ύregularization_losses
Ώ	keras_api
l

Zkernel
[bias
ΐtrainable_variables
Α	variables
Βregularization_losses
Γ	keras_api
V
Δtrainable_variables
Ε	variables
Ζregularization_losses
Η	keras_api
l

\kernel
]bias
Θtrainable_variables
Ι	variables
Κregularization_losses
Λ	keras_api
V
Μtrainable_variables
Ν	variables
Ξregularization_losses
Ο	keras_api
l

^kernel
_bias
Πtrainable_variables
Ρ	variables
?regularization_losses
Σ	keras_api
V
Τtrainable_variables
Υ	variables
Φregularization_losses
Χ	keras_api
l

`kernel
abias
Ψtrainable_variables
Ω	variables
Ϊregularization_losses
Ϋ	keras_api
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
²
/trainable_variables
άlayers
0	variables
1regularization_losses
έnon_trainable_variables
ήmetrics
 ίlayer_regularization_losses
ΰlayer_metrics
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
²
gtrainable_variables
αlayers
h	variables
iregularization_losses
βnon_trainable_variables
γmetrics
 δlayer_regularization_losses
εlayer_metrics
 
 
 
²
ktrainable_variables
ζlayers
l	variables
mregularization_losses
ηnon_trainable_variables
θmetrics
 ιlayer_regularization_losses
κlayer_metrics

F0
G1

F0
G1
 
²
otrainable_variables
λlayers
p	variables
qregularization_losses
μnon_trainable_variables
νmetrics
 ξlayer_regularization_losses
οlayer_metrics
 
 
 
²
strainable_variables
πlayers
t	variables
uregularization_losses
ρnon_trainable_variables
ςmetrics
 σlayer_regularization_losses
τlayer_metrics

H0
I1

H0
I1
 
²
wtrainable_variables
υlayers
x	variables
yregularization_losses
φnon_trainable_variables
χmetrics
 ψlayer_regularization_losses
ωlayer_metrics
 
 
 
²
{trainable_variables
ϊlayers
|	variables
}regularization_losses
ϋnon_trainable_variables
όmetrics
 ύlayer_regularization_losses
ώlayer_metrics

J0
K1

J0
K1
 
΄
trainable_variables
?layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
 
 
 
΅
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
΅
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
΅
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
΅
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
΅
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
΅
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
  layer_regularization_losses
‘layer_metrics

R0
S1

R0
S1
 
΅
trainable_variables
’layers
	variables
regularization_losses
£non_trainable_variables
€metrics
 ₯layer_regularization_losses
¦layer_metrics
 
 
 
΅
trainable_variables
§layers
 	variables
‘regularization_losses
¨non_trainable_variables
©metrics
 ͺlayer_regularization_losses
«layer_metrics
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
΅
¨trainable_variables
¬layers
©	variables
ͺregularization_losses
­non_trainable_variables
?metrics
 ―layer_regularization_losses
°layer_metrics
 
 
 
΅
¬trainable_variables
±layers
­	variables
?regularization_losses
²non_trainable_variables
³metrics
 ΄layer_regularization_losses
΅layer_metrics

V0
W1

V0
W1
 
΅
°trainable_variables
Άlayers
±	variables
²regularization_losses
·non_trainable_variables
Έmetrics
 Ήlayer_regularization_losses
Ίlayer_metrics
 
 
 
΅
΄trainable_variables
»layers
΅	variables
Άregularization_losses
Όnon_trainable_variables
½metrics
 Ύlayer_regularization_losses
Ώlayer_metrics

X0
Y1

X0
Y1
 
΅
Έtrainable_variables
ΐlayers
Ή	variables
Ίregularization_losses
Αnon_trainable_variables
Βmetrics
 Γlayer_regularization_losses
Δlayer_metrics
 
 
 
΅
Όtrainable_variables
Εlayers
½	variables
Ύregularization_losses
Ζnon_trainable_variables
Ηmetrics
 Θlayer_regularization_losses
Ιlayer_metrics

Z0
[1

Z0
[1
 
΅
ΐtrainable_variables
Κlayers
Α	variables
Βregularization_losses
Λnon_trainable_variables
Μmetrics
 Νlayer_regularization_losses
Ξlayer_metrics
 
 
 
΅
Δtrainable_variables
Οlayers
Ε	variables
Ζregularization_losses
Πnon_trainable_variables
Ρmetrics
 ?layer_regularization_losses
Σlayer_metrics

\0
]1

\0
]1
 
΅
Θtrainable_variables
Τlayers
Ι	variables
Κregularization_losses
Υnon_trainable_variables
Φmetrics
 Χlayer_regularization_losses
Ψlayer_metrics
 
 
 
΅
Μtrainable_variables
Ωlayers
Ν	variables
Ξregularization_losses
Ϊnon_trainable_variables
Ϋmetrics
 άlayer_regularization_losses
έlayer_metrics

^0
_1

^0
_1
 
΅
Πtrainable_variables
ήlayers
Ρ	variables
?regularization_losses
ίnon_trainable_variables
ΰmetrics
 αlayer_regularization_losses
βlayer_metrics
 
 
 
΅
Τtrainable_variables
γlayers
Υ	variables
Φregularization_losses
δnon_trainable_variables
εmetrics
 ζlayer_regularization_losses
ηlayer_metrics

`0
a1

`0
a1
 
΅
Ψtrainable_variables
θlayers
Ω	variables
Ϊregularization_losses
ιnon_trainable_variables
κmetrics
 λlayer_regularization_losses
μlayer_metrics
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
:?????????(*
dtype0*$
shape:?????????(
λ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1_enc/kernelconv1_enc/biasconv2_enc/kernelconv2_enc/biasconv3_enc/kernelconv3_enc/biasconv4_enc/kernelconv4_enc/biasconv5_enc/kernelconv5_enc/biasbottleneck/kernelbottleneck/biasz_mean/kernelz_mean/biasz_log_var/kernelz_log_var/biasdecoding/kerneldecoding/biasconv5_dec/kernelconv5_dec/biasconv4_dec/kernelconv4_dec/biasconv3_dec/kernelconv3_dec/biasconv2_dec/kernelconv2_dec/biasconv1_dec/kernelconv1_dec/biasoutput/kerneloutput/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*@
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
Ψ"
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
ο
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
"__inference__traced_restore_152781Φ
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
Γ
°
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
identity’StatefulPartitionedCall
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
-:+???????????????????????????*@
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
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
α
~
)__inference_decoding_layer_call_fn_152003

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallψ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
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
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
ω
`
D__inference_maxpool4_layer_call_and_return_conditional_losses_149152

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
Ζ
±
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
identity’StatefulPartitionedCall
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
-:+???????????????????????????*@
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
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(
!
_user_specified_name	input_1
 
D
(__inference_upsamp4_layer_call_fn_149724

inputs
identityη
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
GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
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
³9
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
identity’!conv1_dec/StatefulPartitionedCall’!conv2_dec/StatefulPartitionedCall’!conv3_dec/StatefulPartitionedCall’!conv4_dec/StatefulPartitionedCall’!conv5_dec/StatefulPartitionedCall’ decoding/StatefulPartitionedCall’output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_150052decoding_150054*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1498252
reshape/PartitionedCallΏ
!conv5_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv5_dec_150058conv5_dec_150060*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
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
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
upsamp5/PartitionedCallΠ
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp5/PartitionedCall:output:0conv4_dec_150064conv4_dec_150066*
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
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
upsamp4/PartitionedCallΠ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_150070conv3_dec_150072*
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
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
upsamp3/PartitionedCallΠ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_150076conv2_dec_150078*
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
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
upsamp2/PartitionedCallΠ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_150082conv1_dec_150084*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
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
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
upsamp1/PartitionedCallΑ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_150088output_150090*
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
GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall"^conv5_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2F
!conv5_dec/StatefulPartitionedCall!conv5_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
π
τ
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

identity_2’StatefulPartitionedCallΩ
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
9:?????????:?????????:?????????*2
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
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs


΄
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
identity’StatefulPartitionedCall―
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
-:+???????????????????????????*0
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
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
κ
ϋ
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
identity’Decoder/StatefulPartitionedCall’Encoder/StatefulPartitionedCall·
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_150423encoder_150425encoder_150427encoder_150429encoder_150431encoder_150433encoder_150435encoder_150437encoder_150439encoder_150441encoder_150443encoder_150445encoder_150447encoder_150449encoder_150451encoder_150453*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*2
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
Encoder/StatefulPartitionedCall¦
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_150458decoder_150460decoder_150462decoder_150464decoder_150466decoder_150468decoder_150470decoder_150472decoder_150474decoder_150476decoder_150478decoder_150480decoder_150482decoder_150484*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
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
Decoder/StatefulPartitionedCallΪ
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????(
!
_user_specified_name	input_1
Ι

*__inference_conv2_dec_layer_call_fn_152102

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_conv2_dec_layer_call_and_return_conditional_losses_1499282
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


*__inference_conv1_enc_layer_call_fn_151804

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
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
:?????????(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
ΥI
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

identity_2’"bottleneck/StatefulPartitionedCall’!conv1_enc/StatefulPartitionedCall’!conv2_enc/StatefulPartitionedCall’!conv3_enc/StatefulPartitionedCall’!conv4_enc/StatefulPartitionedCall’!conv5_enc/StatefulPartitionedCall’ sampling/StatefulPartitionedCall’!z_log_var/StatefulPartitionedCall’z_mean/StatefulPartitionedCall€
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_149597conv1_enc_149599*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
maxpool1/PartitionedCallΏ
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_149603conv2_enc_149605*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
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
:?????????
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
maxpool2/PartitionedCallΏ
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_149609conv3_enc_149611*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
maxpool3/PartitionedCallΏ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_149615conv4_enc_149617*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
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
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
maxpool4/PartitionedCallΐ
!conv5_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool4/PartitionedCall:output:0conv5_enc_149621conv5_enc_149623*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
maxpool5/PartitionedCallπ
flatten/PartitionedCallPartitionedCall!maxpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1493202
flatten/PartitionedCall»
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_149628bottleneck_149630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_1493382$
"bottleneck/StatefulPartitionedCall²
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_149633z_mean_149635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1493642 
z_mean/StatefulPartitionedCallΑ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_149638z_log_var_149640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1493902#
!z_log_var/StatefulPartitionedCall½
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1494322"
 sampling/StatefulPartitionedCallΌ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

IdentityΓ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1Β

Identity_2Identity)sampling/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::2H
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
:?????????(
 
_user_specified_nameinputs
Ο

ή
E__inference_conv3_enc_layer_call_and_return_conditional_losses_151835

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
:?????????
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
:?????????
 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
»
ή
E__inference_conv1_dec_layer_call_and_return_conditional_losses_149956

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
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
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

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
½
_
C__inference_flatten_layer_call_and_return_conditional_losses_151890

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
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
§
D
(__inference_flatten_layer_call_fn_151895

inputs
identityΕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
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
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
δ
€
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
identity’(Decoder/conv1_dec/BiasAdd/ReadVariableOp’'Decoder/conv1_dec/Conv2D/ReadVariableOp’(Decoder/conv2_dec/BiasAdd/ReadVariableOp’'Decoder/conv2_dec/Conv2D/ReadVariableOp’(Decoder/conv3_dec/BiasAdd/ReadVariableOp’'Decoder/conv3_dec/Conv2D/ReadVariableOp’(Decoder/conv4_dec/BiasAdd/ReadVariableOp’'Decoder/conv4_dec/Conv2D/ReadVariableOp’(Decoder/conv5_dec/BiasAdd/ReadVariableOp’'Decoder/conv5_dec/Conv2D/ReadVariableOp’'Decoder/decoding/BiasAdd/ReadVariableOp’&Decoder/decoding/MatMul/ReadVariableOp’%Decoder/output/BiasAdd/ReadVariableOp’$Decoder/output/Conv2D/ReadVariableOp’)Encoder/bottleneck/BiasAdd/ReadVariableOp’(Encoder/bottleneck/MatMul/ReadVariableOp’(Encoder/conv1_enc/BiasAdd/ReadVariableOp’'Encoder/conv1_enc/Conv2D/ReadVariableOp’(Encoder/conv2_enc/BiasAdd/ReadVariableOp’'Encoder/conv2_enc/Conv2D/ReadVariableOp’(Encoder/conv3_enc/BiasAdd/ReadVariableOp’'Encoder/conv3_enc/Conv2D/ReadVariableOp’(Encoder/conv4_enc/BiasAdd/ReadVariableOp’'Encoder/conv4_enc/Conv2D/ReadVariableOp’(Encoder/conv5_enc/BiasAdd/ReadVariableOp’'Encoder/conv5_enc/Conv2D/ReadVariableOp’(Encoder/z_log_var/BiasAdd/ReadVariableOp’'Encoder/z_log_var/MatMul/ReadVariableOp’%Encoder/z_mean/BiasAdd/ReadVariableOp’$Encoder/z_mean/MatMul/ReadVariableOpΛ
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOpΩ
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2DΒ
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2
Encoder/conv1_enc/BiasAdd
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Encoder/conv1_enc/ReluΡ
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPoolΛ
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOpτ
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2DΒ
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Encoder/conv2_enc/BiasAdd
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Encoder/conv2_enc/ReluΡ
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPoolΛ
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOpτ
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 *
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2DΒ
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 2
Encoder/conv3_enc/BiasAdd
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
 2
Encoder/conv3_enc/ReluΡ
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPoolΛ
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOpτ
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2DΒ
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Encoder/conv4_enc/BiasAdd
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Encoder/conv4_enc/ReluΡ
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool4/MaxPoolΜ
'Encoder/conv5_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Encoder/conv5_enc/Conv2D/ReadVariableOpυ
Encoder/conv5_enc/Conv2DConv2D!Encoder/maxpool4/MaxPool:output:0/Encoder/conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Encoder/conv5_enc/Conv2DΓ
(Encoder/conv5_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Encoder/conv5_enc/BiasAdd/ReadVariableOpΡ
Encoder/conv5_enc/BiasAddBiasAdd!Encoder/conv5_enc/Conv2D:output:00Encoder/conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Encoder/conv5_enc/BiasAdd
Encoder/conv5_enc/ReluRelu"Encoder/conv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Encoder/conv5_enc/Relu?
Encoder/maxpool5/MaxPoolMaxPool$Encoder/conv5_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????   2
Encoder/flatten/Const³
Encoder/flatten/ReshapeReshape!Encoder/maxpool5/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
Encoder/flatten/ReshapeΗ
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOpΖ
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/MatMulΕ
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOpΝ
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/BiasAddΊ
$Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Encoder/z_mean/MatMul/ReadVariableOp½
Encoder/z_mean/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0,Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_mean/MatMulΉ
%Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Encoder/z_mean/BiasAdd/ReadVariableOp½
Encoder/z_mean/BiasAddBiasAddEncoder/z_mean/MatMul:product:0-Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_mean/BiasAddΓ
'Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Encoder/z_log_var/MatMul/ReadVariableOpΖ
Encoder/z_log_var/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_log_var/MatMulΒ
(Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/z_log_var/BiasAdd/ReadVariableOpΙ
Encoder/z_log_var/BiasAddBiasAdd"Encoder/z_log_var/MatMul:product:00Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
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
&Encoder/sampling/strided_slice/stack_2Θ
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
(Encoder/sampling/strided_slice_1/stack_2Τ
 Encoder/sampling/strided_slice_1StridedSlice!Encoder/sampling/Shape_1:output:0/Encoder/sampling/strided_slice_1/stack:output:01Encoder/sampling/strided_slice_1/stack_1:output:01Encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Encoder/sampling/strided_slice_1Φ
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
:??????????????????*
dtype0*
seed±?ε)*
seed2ξήΥ25
3Encoder/sampling/random_normal/RandomStandardNormalψ
"Encoder/sampling/random_normal/mulMul<Encoder/sampling/random_normal/RandomStandardNormal:output:0.Encoder/sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2$
"Encoder/sampling/random_normal/mulΨ
Encoder/sampling/random_normalAdd&Encoder/sampling/random_normal/mul:z:0,Encoder/sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2 
Encoder/sampling/random_normalu
Encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Encoder/sampling/mul/xͺ
Encoder/sampling/mulMulEncoder/sampling/mul/x:output:0"Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling/mul
Encoder/sampling/ExpExpEncoder/sampling/mul:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling/Exp§
Encoder/sampling/mul_1MulEncoder/sampling/Exp:y:0"Encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling/mul_1€
Encoder/sampling/addAddV2Encoder/z_mean/BiasAdd:output:0Encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling/addΑ
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOpΉ
Decoder/decoding/MatMulMatMulEncoder/sampling/add:z:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
Decoder/decoding/MatMulΐ
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOpΖ
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
Decoder/reshape/Reshape/shapeΓ
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????2
Decoder/reshape/ReshapeΝ
'Decoder/conv5_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'Decoder/conv5_dec/Conv2D/ReadVariableOpτ
Decoder/conv5_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv5_dec/Conv2DΓ
(Decoder/conv5_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Decoder/conv5_dec/BiasAdd/ReadVariableOpΡ
Decoder/conv5_dec/BiasAddBiasAdd!Decoder/conv5_dec/Conv2D:output:00Decoder/conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Decoder/conv5_dec/BiasAdd
Decoder/conv5_dec/ReluRelu"Decoder/conv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
%Decoder/upsamp5/strided_slice/stack_2?
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
:?????????*
half_pixel_centers(2.
,Decoder/upsamp5/resize/ResizeNearestNeighborΜ
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOp
Decoder/conv4_dec/Conv2DConv2D=Decoder/upsamp5/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2DΒ
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Decoder/conv4_dec/BiasAdd
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
Decoder/upsamp4/mul
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighborΛ
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2DΒ
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv3_dec/BiasAdd
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:????????? *
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighborΛ
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2DΒ
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Decoder/conv2_dec/BiasAdd
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
:????????? *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighborΛ
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2DΒ
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv1_dec/BiasAdd
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:?????????@ *
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighborΒ
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
Decoder/output/BiasAdd
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Decoder/output/Sigmoidε

IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp)^Decoder/conv5_dec/BiasAdd/ReadVariableOp(^Decoder/conv5_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp)^Encoder/conv5_enc/BiasAdd/ReadVariableOp(^Encoder/conv5_enc/Conv2D/ReadVariableOp)^Encoder/z_log_var/BiasAdd/ReadVariableOp(^Encoder/z_log_var/MatMul/ReadVariableOp&^Encoder/z_mean/BiasAdd/ReadVariableOp%^Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::2T
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
:?????????(
 
_user_specified_nameinputs
ι
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
:?????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
’
E
)__inference_maxpool1_layer_call_fn_149122

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
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
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
	
ί
F__inference_bottleneck_layer_call_and_return_conditional_losses_149338

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ι

*__inference_conv3_dec_layer_call_fn_152082

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_conv3_dec_layer_call_and_return_conditional_losses_1499002
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
π
τ
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

identity_2’StatefulPartitionedCallΩ
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
9:?????????:?????????:?????????*2
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
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
’
E
)__inference_maxpool5_layer_call_fn_149170

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
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
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
κI
¦
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

identity_2’"bottleneck/StatefulPartitionedCall’!conv1_enc/StatefulPartitionedCall’!conv2_enc/StatefulPartitionedCall’!conv3_enc/StatefulPartitionedCall’!conv4_enc/StatefulPartitionedCall’!conv5_enc/StatefulPartitionedCall’ sampling/StatefulPartitionedCall’!z_log_var/StatefulPartitionedCall’z_mean/StatefulPartitionedCall«
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_149196conv1_enc_149198*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
maxpool1/PartitionedCallΏ
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_149224conv2_enc_149226*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
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
:?????????
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
maxpool2/PartitionedCallΏ
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_149252conv3_enc_149254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
maxpool3/PartitionedCallΏ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_149280conv4_enc_149282*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
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
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
maxpool4/PartitionedCallΐ
!conv5_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool4/PartitionedCall:output:0conv5_enc_149308conv5_enc_149310*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
maxpool5/PartitionedCallπ
flatten/PartitionedCallPartitionedCall!maxpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1493202
flatten/PartitionedCall»
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_149349bottleneck_149351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_1493382$
"bottleneck/StatefulPartitionedCall²
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_149375z_mean_149377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1493642 
z_mean/StatefulPartitionedCallΑ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_149401z_log_var_149403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1493902#
!z_log_var/StatefulPartitionedCall½
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1494322"
 sampling/StatefulPartitionedCallΌ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

IdentityΓ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1Β

Identity_2Identity)sampling/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::2H
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
:?????????(
'
_user_specified_nameinput_encoder
Υ

ή
E__inference_conv5_enc_layer_call_and_return_conditional_losses_149297

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
:?????????*
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
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs

ϋ
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

identity_2’StatefulPartitionedCallΰ
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
9:?????????:?????????:?????????*2
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
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????(
'
_user_specified_nameinput_encoder
»
ή
E__inference_conv2_dec_layer_call_and_return_conditional_losses_152093

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
	
ή
E__inference_z_log_var_layer_call_and_return_conditional_losses_151943

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
’
E
)__inference_maxpool3_layer_call_fn_149146

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
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
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
	
Ϋ
B__inference_z_mean_layer_call_and_return_conditional_losses_151924

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ο

ή
E__inference_conv1_enc_layer_call_and_return_conditional_losses_149185

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
Ο

ή
E__inference_conv1_enc_layer_call_and_return_conditional_losses_151795

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????(::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
Υ

ή
E__inference_conv5_enc_layer_call_and_return_conditional_losses_151875

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
:?????????*
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
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ζ
±
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
identity’StatefulPartitionedCall
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
-:+???????????????????????????*@
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
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(
!
_user_specified_name	input_1
’
Λ4
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
identity_102’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_100’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_69’AssignVariableOp_7’AssignVariableOp_70’AssignVariableOp_71’AssignVariableOp_72’AssignVariableOp_73’AssignVariableOp_74’AssignVariableOp_75’AssignVariableOp_76’AssignVariableOp_77’AssignVariableOp_78’AssignVariableOp_79’AssignVariableOp_8’AssignVariableOp_80’AssignVariableOp_81’AssignVariableOp_82’AssignVariableOp_83’AssignVariableOp_84’AssignVariableOp_85’AssignVariableOp_86’AssignVariableOp_87’AssignVariableOp_88’AssignVariableOp_89’AssignVariableOp_9’AssignVariableOp_90’AssignVariableOp_91’AssignVariableOp_92’AssignVariableOp_93’AssignVariableOp_94’AssignVariableOp_95’AssignVariableOp_96’AssignVariableOp_97’AssignVariableOp_98’AssignVariableOp_99Ά6
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*Β5
valueΈ5B΅5fB3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesέ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*α
valueΧBΤfB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
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
Identity_19¬
AssignVariableOp_19AssignVariableOp$assignvariableop_19_conv5_enc_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ͺ
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
Identity_22«
AssignVariableOp_22AssignVariableOp#assignvariableop_22_bottleneck_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23©
AssignVariableOp_23AssignVariableOp!assignvariableop_23_z_mean_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24§
AssignVariableOp_24AssignVariableOpassignvariableop_24_z_mean_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¬
AssignVariableOp_25AssignVariableOp$assignvariableop_25_z_log_var_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ͺ
AssignVariableOp_26AssignVariableOp"assignvariableop_26_z_log_var_biasIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27«
AssignVariableOp_27AssignVariableOp#assignvariableop_27_decoding_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28©
AssignVariableOp_28AssignVariableOp!assignvariableop_28_decoding_biasIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¬
AssignVariableOp_29AssignVariableOp$assignvariableop_29_conv5_dec_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ͺ
AssignVariableOp_30AssignVariableOp"assignvariableop_30_conv5_dec_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¬
AssignVariableOp_31AssignVariableOp$assignvariableop_31_conv4_dec_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ͺ
AssignVariableOp_32AssignVariableOp"assignvariableop_32_conv4_dec_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¬
AssignVariableOp_33AssignVariableOp$assignvariableop_33_conv3_dec_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ͺ
AssignVariableOp_34AssignVariableOp"assignvariableop_34_conv3_dec_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¬
AssignVariableOp_35AssignVariableOp$assignvariableop_35_conv2_dec_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36ͺ
AssignVariableOp_36AssignVariableOp"assignvariableop_36_conv2_dec_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37¬
AssignVariableOp_37AssignVariableOp$assignvariableop_37_conv1_dec_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ͺ
AssignVariableOp_38AssignVariableOp"assignvariableop_38_conv1_dec_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39©
AssignVariableOp_39AssignVariableOp!assignvariableop_39_output_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40§
AssignVariableOp_40AssignVariableOpassignvariableop_40_output_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41³
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv1_enc_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42±
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv1_enc_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43³
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2_enc_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44±
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2_enc_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45³
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv3_enc_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46±
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv3_enc_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47³
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv4_enc_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv4_enc_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49³
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv5_enc_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv5_enc_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51΄
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_bottleneck_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52²
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_bottleneck_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53°
AssignVariableOp_53AssignVariableOp(assignvariableop_53_adam_z_mean_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp&assignvariableop_54_adam_z_mean_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55³
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_z_log_var_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56±
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_z_log_var_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57²
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_decoding_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58°
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_decoding_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59³
AssignVariableOp_59AssignVariableOp+assignvariableop_59_adam_conv5_dec_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60±
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_conv5_dec_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61³
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv4_dec_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62±
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv4_dec_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63³
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv3_dec_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64±
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv3_dec_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65³
AssignVariableOp_65AssignVariableOp+assignvariableop_65_adam_conv2_dec_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66±
AssignVariableOp_66AssignVariableOp)assignvariableop_66_adam_conv2_dec_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67³
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv1_dec_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68±
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv1_dec_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69°
AssignVariableOp_69AssignVariableOp(assignvariableop_69_adam_output_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp&assignvariableop_70_adam_output_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71³
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_conv1_enc_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72±
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adam_conv1_enc_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73³
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2_enc_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74±
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2_enc_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75³
AssignVariableOp_75AssignVariableOp+assignvariableop_75_adam_conv3_enc_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76±
AssignVariableOp_76AssignVariableOp)assignvariableop_76_adam_conv3_enc_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77³
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv4_enc_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78±
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv4_enc_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79³
AssignVariableOp_79AssignVariableOp+assignvariableop_79_adam_conv5_enc_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80±
AssignVariableOp_80AssignVariableOp)assignvariableop_80_adam_conv5_enc_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81΄
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_bottleneck_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82²
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adam_bottleneck_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83°
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adam_z_mean_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp&assignvariableop_84_adam_z_mean_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85³
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_z_log_var_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86±
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_z_log_var_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87²
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_decoding_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88°
AssignVariableOp_88AssignVariableOp(assignvariableop_88_adam_decoding_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89³
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_conv5_dec_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90±
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_conv5_dec_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91³
AssignVariableOp_91AssignVariableOp+assignvariableop_91_adam_conv4_dec_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92±
AssignVariableOp_92AssignVariableOp)assignvariableop_92_adam_conv4_dec_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93³
AssignVariableOp_93AssignVariableOp+assignvariableop_93_adam_conv3_dec_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94±
AssignVariableOp_94AssignVariableOp)assignvariableop_94_adam_conv3_dec_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95³
AssignVariableOp_95AssignVariableOp+assignvariableop_95_adam_conv2_dec_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96±
AssignVariableOp_96AssignVariableOp)assignvariableop_96_adam_conv2_dec_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97³
AssignVariableOp_97AssignVariableOp+assignvariableop_97_adam_conv1_dec_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98±
AssignVariableOp_98AssignVariableOp)assignvariableop_98_adam_conv1_dec_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99°
AssignVariableOp_99AssignVariableOp(assignvariableop_99_adam_output_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100²
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
identity_102Identity_102:output:0*«
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
Ψ

ή
E__inference_conv5_dec_layer_call_and_return_conditional_losses_152033

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
:?????????*
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
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Γ
|
'__inference_output_layer_call_fn_152142

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
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
GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
»
ή
E__inference_conv3_dec_layer_call_and_return_conditional_losses_149900

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


*__inference_conv3_enc_layer_call_fn_151844

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
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
:?????????
 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs


*__inference_conv4_enc_layer_call_fn_151864

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
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
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
Λ

*__inference_conv4_dec_layer_call_fn_152062

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
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
GPU2*0J 8 *N
fIRG
E__inference_conv4_dec_layer_call_and_return_conditional_losses_1498722
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
δ
€
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
identity’(Decoder/conv1_dec/BiasAdd/ReadVariableOp’'Decoder/conv1_dec/Conv2D/ReadVariableOp’(Decoder/conv2_dec/BiasAdd/ReadVariableOp’'Decoder/conv2_dec/Conv2D/ReadVariableOp’(Decoder/conv3_dec/BiasAdd/ReadVariableOp’'Decoder/conv3_dec/Conv2D/ReadVariableOp’(Decoder/conv4_dec/BiasAdd/ReadVariableOp’'Decoder/conv4_dec/Conv2D/ReadVariableOp’(Decoder/conv5_dec/BiasAdd/ReadVariableOp’'Decoder/conv5_dec/Conv2D/ReadVariableOp’'Decoder/decoding/BiasAdd/ReadVariableOp’&Decoder/decoding/MatMul/ReadVariableOp’%Decoder/output/BiasAdd/ReadVariableOp’$Decoder/output/Conv2D/ReadVariableOp’)Encoder/bottleneck/BiasAdd/ReadVariableOp’(Encoder/bottleneck/MatMul/ReadVariableOp’(Encoder/conv1_enc/BiasAdd/ReadVariableOp’'Encoder/conv1_enc/Conv2D/ReadVariableOp’(Encoder/conv2_enc/BiasAdd/ReadVariableOp’'Encoder/conv2_enc/Conv2D/ReadVariableOp’(Encoder/conv3_enc/BiasAdd/ReadVariableOp’'Encoder/conv3_enc/Conv2D/ReadVariableOp’(Encoder/conv4_enc/BiasAdd/ReadVariableOp’'Encoder/conv4_enc/Conv2D/ReadVariableOp’(Encoder/conv5_enc/BiasAdd/ReadVariableOp’'Encoder/conv5_enc/Conv2D/ReadVariableOp’(Encoder/z_log_var/BiasAdd/ReadVariableOp’'Encoder/z_log_var/MatMul/ReadVariableOp’%Encoder/z_mean/BiasAdd/ReadVariableOp’$Encoder/z_mean/MatMul/ReadVariableOpΛ
'Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv1_enc/Conv2D/ReadVariableOpΩ
Encoder/conv1_enc/Conv2DConv2Dinputs/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
2
Encoder/conv1_enc/Conv2DΒ
(Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv1_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv1_enc/BiasAddBiasAdd!Encoder/conv1_enc/Conv2D:output:00Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2
Encoder/conv1_enc/BiasAdd
Encoder/conv1_enc/ReluRelu"Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Encoder/conv1_enc/ReluΡ
Encoder/maxpool1/MaxPoolMaxPool$Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool1/MaxPoolΛ
'Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Encoder/conv2_enc/Conv2D/ReadVariableOpτ
Encoder/conv2_enc/Conv2DConv2D!Encoder/maxpool1/MaxPool:output:0/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Encoder/conv2_enc/Conv2DΒ
(Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/conv2_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv2_enc/BiasAddBiasAdd!Encoder/conv2_enc/Conv2D:output:00Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Encoder/conv2_enc/BiasAdd
Encoder/conv2_enc/ReluRelu"Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Encoder/conv2_enc/ReluΡ
Encoder/maxpool2/MaxPoolMaxPool$Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool2/MaxPoolΛ
'Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Encoder/conv3_enc/Conv2D/ReadVariableOpτ
Encoder/conv3_enc/Conv2DConv2D!Encoder/maxpool2/MaxPool:output:0/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 *
paddingSAME*
strides
2
Encoder/conv3_enc/Conv2DΒ
(Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Encoder/conv3_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv3_enc/BiasAddBiasAdd!Encoder/conv3_enc/Conv2D:output:00Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 2
Encoder/conv3_enc/BiasAdd
Encoder/conv3_enc/ReluRelu"Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
 2
Encoder/conv3_enc/ReluΡ
Encoder/maxpool3/MaxPoolMaxPool$Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
Encoder/maxpool3/MaxPoolΛ
'Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'Encoder/conv4_enc/Conv2D/ReadVariableOpτ
Encoder/conv4_enc/Conv2DConv2D!Encoder/maxpool3/MaxPool:output:0/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Encoder/conv4_enc/Conv2DΒ
(Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Encoder/conv4_enc/BiasAdd/ReadVariableOpΠ
Encoder/conv4_enc/BiasAddBiasAdd!Encoder/conv4_enc/Conv2D:output:00Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Encoder/conv4_enc/BiasAdd
Encoder/conv4_enc/ReluRelu"Encoder/conv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Encoder/conv4_enc/ReluΡ
Encoder/maxpool4/MaxPoolMaxPool$Encoder/conv4_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
Encoder/maxpool4/MaxPoolΜ
'Encoder/conv5_enc/Conv2D/ReadVariableOpReadVariableOp0encoder_conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Encoder/conv5_enc/Conv2D/ReadVariableOpυ
Encoder/conv5_enc/Conv2DConv2D!Encoder/maxpool4/MaxPool:output:0/Encoder/conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Encoder/conv5_enc/Conv2DΓ
(Encoder/conv5_enc/BiasAdd/ReadVariableOpReadVariableOp1encoder_conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Encoder/conv5_enc/BiasAdd/ReadVariableOpΡ
Encoder/conv5_enc/BiasAddBiasAdd!Encoder/conv5_enc/Conv2D:output:00Encoder/conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Encoder/conv5_enc/BiasAdd
Encoder/conv5_enc/ReluRelu"Encoder/conv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Encoder/conv5_enc/Relu?
Encoder/maxpool5/MaxPoolMaxPool$Encoder/conv5_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????   2
Encoder/flatten/Const³
Encoder/flatten/ReshapeReshape!Encoder/maxpool5/MaxPool:output:0Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
Encoder/flatten/ReshapeΗ
(Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp1encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02*
(Encoder/bottleneck/MatMul/ReadVariableOpΖ
Encoder/bottleneck/MatMulMatMul Encoder/flatten/Reshape:output:00Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/MatMulΕ
)Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp2encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)Encoder/bottleneck/BiasAdd/ReadVariableOpΝ
Encoder/bottleneck/BiasAddBiasAdd#Encoder/bottleneck/MatMul:product:01Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/bottleneck/BiasAddΊ
$Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp-encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$Encoder/z_mean/MatMul/ReadVariableOp½
Encoder/z_mean/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0,Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_mean/MatMulΉ
%Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp.encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Encoder/z_mean/BiasAdd/ReadVariableOp½
Encoder/z_mean/BiasAddBiasAddEncoder/z_mean/MatMul:product:0-Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_mean/BiasAddΓ
'Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp0encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'Encoder/z_log_var/MatMul/ReadVariableOpΖ
Encoder/z_log_var/MatMulMatMul#Encoder/bottleneck/BiasAdd:output:0/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Encoder/z_log_var/MatMulΒ
(Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp1encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Encoder/z_log_var/BiasAdd/ReadVariableOpΙ
Encoder/z_log_var/BiasAddBiasAdd"Encoder/z_log_var/MatMul:product:00Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
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
&Encoder/sampling/strided_slice/stack_2Θ
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
(Encoder/sampling/strided_slice_1/stack_2Τ
 Encoder/sampling/strided_slice_1StridedSlice!Encoder/sampling/Shape_1:output:0/Encoder/sampling/strided_slice_1/stack:output:01Encoder/sampling/strided_slice_1/stack_1:output:01Encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 Encoder/sampling/strided_slice_1Φ
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
:??????????????????*
dtype0*
seed±?ε)*
seed2κΈ·25
3Encoder/sampling/random_normal/RandomStandardNormalψ
"Encoder/sampling/random_normal/mulMul<Encoder/sampling/random_normal/RandomStandardNormal:output:0.Encoder/sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2$
"Encoder/sampling/random_normal/mulΨ
Encoder/sampling/random_normalAdd&Encoder/sampling/random_normal/mul:z:0,Encoder/sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2 
Encoder/sampling/random_normalu
Encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Encoder/sampling/mul/xͺ
Encoder/sampling/mulMulEncoder/sampling/mul/x:output:0"Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling/mul
Encoder/sampling/ExpExpEncoder/sampling/mul:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling/Exp§
Encoder/sampling/mul_1MulEncoder/sampling/Exp:y:0"Encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling/mul_1€
Encoder/sampling/addAddV2Encoder/z_mean/BiasAdd:output:0Encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
Encoder/sampling/addΑ
&Decoder/decoding/MatMul/ReadVariableOpReadVariableOp/decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02(
&Decoder/decoding/MatMul/ReadVariableOpΉ
Decoder/decoding/MatMulMatMulEncoder/sampling/add:z:0.Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
Decoder/decoding/MatMulΐ
'Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp0decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'Decoder/decoding/BiasAdd/ReadVariableOpΖ
Decoder/decoding/BiasAddBiasAdd!Decoder/decoding/MatMul:product:0/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
Decoder/reshape/Reshape/shapeΓ
Decoder/reshape/ReshapeReshape!Decoder/decoding/BiasAdd:output:0&Decoder/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:?????????2
Decoder/reshape/ReshapeΝ
'Decoder/conv5_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'Decoder/conv5_dec/Conv2D/ReadVariableOpτ
Decoder/conv5_dec/Conv2DConv2D Decoder/reshape/Reshape:output:0/Decoder/conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv5_dec/Conv2DΓ
(Decoder/conv5_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(Decoder/conv5_dec/BiasAdd/ReadVariableOpΡ
Decoder/conv5_dec/BiasAddBiasAdd!Decoder/conv5_dec/Conv2D:output:00Decoder/conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
Decoder/conv5_dec/BiasAdd
Decoder/conv5_dec/ReluRelu"Decoder/conv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
%Decoder/upsamp5/strided_slice/stack_2?
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
:?????????*
half_pixel_centers(2.
,Decoder/upsamp5/resize/ResizeNearestNeighborΜ
'Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'Decoder/conv4_dec/Conv2D/ReadVariableOp
Decoder/conv4_dec/Conv2DConv2D=Decoder/upsamp5/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Decoder/conv4_dec/Conv2DΒ
(Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(Decoder/conv4_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv4_dec/BiasAddBiasAdd!Decoder/conv4_dec/Conv2D:output:00Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
Decoder/conv4_dec/BiasAdd
Decoder/conv4_dec/ReluRelu"Decoder/conv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
Decoder/upsamp4/mul
,Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor$Decoder/conv4_dec/Relu:activations:0Decoder/upsamp4/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,Decoder/upsamp4/resize/ResizeNearestNeighborΛ
'Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02)
'Decoder/conv3_dec/Conv2D/ReadVariableOp
Decoder/conv3_dec/Conv2DConv2D=Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Decoder/conv3_dec/Conv2DΒ
(Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(Decoder/conv3_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv3_dec/BiasAddBiasAdd!Decoder/conv3_dec/Conv2D:output:00Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv3_dec/BiasAdd
Decoder/conv3_dec/ReluRelu"Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:????????? *
half_pixel_centers(2.
,Decoder/upsamp3/resize/ResizeNearestNeighborΛ
'Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'Decoder/conv2_dec/Conv2D/ReadVariableOp
Decoder/conv2_dec/Conv2DConv2D=Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Decoder/conv2_dec/Conv2DΒ
(Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv2_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv2_dec/BiasAddBiasAdd!Decoder/conv2_dec/Conv2D:output:00Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
Decoder/conv2_dec/BiasAdd
Decoder/conv2_dec/ReluRelu"Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
:????????? *
half_pixel_centers(2.
,Decoder/upsamp2/resize/ResizeNearestNeighborΛ
'Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp0decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'Decoder/conv1_dec/Conv2D/ReadVariableOp
Decoder/conv1_dec/Conv2DConv2D=Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:0/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Decoder/conv1_dec/Conv2DΒ
(Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp1decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(Decoder/conv1_dec/BiasAdd/ReadVariableOpΠ
Decoder/conv1_dec/BiasAddBiasAdd!Decoder/conv1_dec/Conv2D:output:00Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Decoder/conv1_dec/BiasAdd
Decoder/conv1_dec/ReluRelu"Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:?????????@ *
half_pixel_centers(2.
,Decoder/upsamp1/resize/ResizeNearestNeighborΒ
$Decoder/output/Conv2D/ReadVariableOpReadVariableOp-decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$Decoder/output/Conv2D/ReadVariableOp
Decoder/output/Conv2DConv2D=Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:0,Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
Decoder/output/BiasAdd
Decoder/output/SigmoidSigmoidDecoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
Decoder/output/Sigmoidε

IdentityIdentityDecoder/output/Sigmoid:y:0)^Decoder/conv1_dec/BiasAdd/ReadVariableOp(^Decoder/conv1_dec/Conv2D/ReadVariableOp)^Decoder/conv2_dec/BiasAdd/ReadVariableOp(^Decoder/conv2_dec/Conv2D/ReadVariableOp)^Decoder/conv3_dec/BiasAdd/ReadVariableOp(^Decoder/conv3_dec/Conv2D/ReadVariableOp)^Decoder/conv4_dec/BiasAdd/ReadVariableOp(^Decoder/conv4_dec/Conv2D/ReadVariableOp)^Decoder/conv5_dec/BiasAdd/ReadVariableOp(^Decoder/conv5_dec/Conv2D/ReadVariableOp(^Decoder/decoding/BiasAdd/ReadVariableOp'^Decoder/decoding/MatMul/ReadVariableOp&^Decoder/output/BiasAdd/ReadVariableOp%^Decoder/output/Conv2D/ReadVariableOp*^Encoder/bottleneck/BiasAdd/ReadVariableOp)^Encoder/bottleneck/MatMul/ReadVariableOp)^Encoder/conv1_enc/BiasAdd/ReadVariableOp(^Encoder/conv1_enc/Conv2D/ReadVariableOp)^Encoder/conv2_enc/BiasAdd/ReadVariableOp(^Encoder/conv2_enc/Conv2D/ReadVariableOp)^Encoder/conv3_enc/BiasAdd/ReadVariableOp(^Encoder/conv3_enc/Conv2D/ReadVariableOp)^Encoder/conv4_enc/BiasAdd/ReadVariableOp(^Encoder/conv4_enc/Conv2D/ReadVariableOp)^Encoder/conv5_enc/BiasAdd/ReadVariableOp(^Encoder/conv5_enc/Conv2D/ReadVariableOp)^Encoder/z_log_var/BiasAdd/ReadVariableOp(^Encoder/z_log_var/MatMul/ReadVariableOp&^Encoder/z_mean/BiasAdd/ReadVariableOp%^Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::2T
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
:?????????(
 
_user_specified_nameinputs
±
ΐ	
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
identity’ conv1_dec/BiasAdd/ReadVariableOp’conv1_dec/Conv2D/ReadVariableOp’ conv2_dec/BiasAdd/ReadVariableOp’conv2_dec/Conv2D/ReadVariableOp’ conv3_dec/BiasAdd/ReadVariableOp’conv3_dec/Conv2D/ReadVariableOp’ conv4_dec/BiasAdd/ReadVariableOp’conv4_dec/Conv2D/ReadVariableOp’ conv5_dec/BiasAdd/ReadVariableOp’conv5_dec/Conv2D/ReadVariableOp’decoding/BiasAdd/ReadVariableOp’decoding/MatMul/ReadVariableOp’output/BiasAdd/ReadVariableOp’output/Conv2D/ReadVariableOp©
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
decoding/MatMul/ReadVariableOp
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
decoding/MatMul¨
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
decoding/BiasAdd/ReadVariableOp¦
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
:?????????2
reshape/Reshape΅
conv5_dec/Conv2D/ReadVariableOpReadVariableOp(conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv5_dec/Conv2D/ReadVariableOpΤ
conv5_dec/Conv2DConv2Dreshape/Reshape:output:0'conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv5_dec/Conv2D«
 conv5_dec/BiasAdd/ReadVariableOpReadVariableOp)conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv5_dec/BiasAdd/ReadVariableOp±
conv5_dec/BiasAddBiasAddconv5_dec/Conv2D:output:0(conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv5_dec/BiasAdd
conv5_dec/ReluReluconv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
upsamp5/strided_slice/stack_2ώ
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
upsamp5/mulι
$upsamp5/resize/ResizeNearestNeighborResizeNearestNeighborconv5_dec/Relu:activations:0upsamp5/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(2&
$upsamp5/resize/ResizeNearestNeighbor΄
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv4_dec/Conv2D/ReadVariableOpπ
conv4_dec/Conv2DConv2D5upsamp5/resize/ResizeNearestNeighbor:resized_images:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv4_dec/Conv2Dͺ
 conv4_dec/BiasAdd/ReadVariableOpReadVariableOp)conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv4_dec/BiasAdd/ReadVariableOp°
conv4_dec/BiasAddBiasAddconv4_dec/Conv2D:output:0(conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv4_dec/BiasAdd~
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
upsamp4/mulθ
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighbor³
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv3_dec/Conv2D/ReadVariableOpπ
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv3_dec/Conv2Dͺ
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3_dec/BiasAdd/ReadVariableOp°
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:????????? *
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighbor³
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_dec/Conv2D/ReadVariableOpπ
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2_dec/Conv2Dͺ
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2_dec/BiasAdd/ReadVariableOp°
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
:????????? *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighbor³
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_dec/Conv2D/ReadVariableOpπ
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv1_dec/Conv2Dͺ
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOp°
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:?????????@ *
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighborͺ
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
output/Conv2D/ReadVariableOpθ
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
output/SigmoidΙ
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp!^conv5_dec/BiasAdd/ReadVariableOp ^conv5_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::2D
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
:?????????
 
_user_specified_nameinputs
Ι

*__inference_conv1_dec_layer_call_fn_152122

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_conv1_dec_layer_call_and_return_conditional_losses_1499562
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
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
random_normal/stddevε
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed2υι2$
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
:?????????2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mul_1X
addAddV2inputs	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs


*__inference_conv2_enc_layer_call_fn_151824

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
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
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
|
'__inference_z_mean_layer_call_fn_151933

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallυ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
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
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
»
ή
E__inference_conv1_dec_layer_call_and_return_conditional_losses_152113

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp΅
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
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
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu±
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

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
κI
¦
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

identity_2’"bottleneck/StatefulPartitionedCall’!conv1_enc/StatefulPartitionedCall’!conv2_enc/StatefulPartitionedCall’!conv3_enc/StatefulPartitionedCall’!conv4_enc/StatefulPartitionedCall’!conv5_enc/StatefulPartitionedCall’ sampling/StatefulPartitionedCall’!z_log_var/StatefulPartitionedCall’z_mean/StatefulPartitionedCall«
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinput_encoderconv1_enc_149447conv1_enc_149449*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
maxpool1/PartitionedCallΏ
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_149453conv2_enc_149455*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
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
:?????????
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
maxpool2/PartitionedCallΏ
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_149459conv3_enc_149461*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
maxpool3/PartitionedCallΏ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_149465conv4_enc_149467*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
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
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
maxpool4/PartitionedCallΐ
!conv5_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool4/PartitionedCall:output:0conv5_enc_149471conv5_enc_149473*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
maxpool5/PartitionedCallπ
flatten/PartitionedCallPartitionedCall!maxpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1493202
flatten/PartitionedCall»
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_149478bottleneck_149480*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_1493382$
"bottleneck/StatefulPartitionedCall²
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_149483z_mean_149485*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1493642 
z_mean/StatefulPartitionedCallΑ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_149488z_log_var_149490*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1493902#
!z_log_var/StatefulPartitionedCall½
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1494322"
 sampling/StatefulPartitionedCallΌ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

IdentityΓ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1Β

Identity_2Identity)sampling/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::2H
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
:?????????(
'
_user_specified_nameinput_encoder
ω
`
D__inference_maxpool3_layer_call_and_return_conditional_losses_149140

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
’
E
)__inference_maxpool2_layer_call_fn_149134

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
D__inference_maxpool2_layer_call_and_return_conditional_losses_1491282
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
ω
`
D__inference_maxpool2_layer_call_and_return_conditional_losses_149128

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
Ύ
ή
E__inference_conv4_dec_layer_call_and_return_conditional_losses_149872

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
Θ9
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
identity’!conv1_dec/StatefulPartitionedCall’!conv2_dec/StatefulPartitionedCall’!conv3_dec/StatefulPartitionedCall’!conv4_dec/StatefulPartitionedCall’!conv5_dec/StatefulPartitionedCall’ decoding/StatefulPartitionedCall’output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_149806decoding_149808*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1498252
reshape/PartitionedCallΏ
!conv5_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv5_dec_149855conv5_dec_149857*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
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
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
upsamp5/PartitionedCallΠ
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp5/PartitionedCall:output:0conv4_dec_149883conv4_dec_149885*
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
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
upsamp4/PartitionedCallΠ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_149911conv3_dec_149913*
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
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
upsamp3/PartitionedCallΠ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_149939conv2_dec_149941*
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
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
upsamp2/PartitionedCallΠ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_149967conv1_dec_149969*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
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
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
upsamp1/PartitionedCallΑ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_149995output_149997*
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
GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall"^conv5_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2F
!conv5_dec/StatefulPartitionedCall!conv5_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameinput_decoder
Θ9
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
identity’!conv1_dec/StatefulPartitionedCall’!conv2_dec/StatefulPartitionedCall’!conv3_dec/StatefulPartitionedCall’!conv4_dec/StatefulPartitionedCall’!conv5_dec/StatefulPartitionedCall’ decoding/StatefulPartitionedCall’output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinput_decoderdecoding_150004decoding_150006*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1498252
reshape/PartitionedCallΏ
!conv5_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv5_dec_150010conv5_dec_150012*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
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
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
upsamp5/PartitionedCallΠ
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp5/PartitionedCall:output:0conv4_dec_150016conv4_dec_150018*
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
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
upsamp4/PartitionedCallΠ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_150022conv3_dec_150024*
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
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
upsamp3/PartitionedCallΠ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_150028conv2_dec_150030*
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
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
upsamp2/PartitionedCallΠ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_150034conv1_dec_150036*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
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
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
upsamp1/PartitionedCallΑ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_150040output_150042*
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
GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall"^conv5_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2F
!conv5_dec/StatefulPartitionedCall!conv5_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameinput_decoder
ι
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
:?????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
»
Ϋ
B__inference_output_layer_call_and_return_conditional_losses_152133

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
	
ή
E__inference_z_log_var_layer_call_and_return_conditional_losses_149390

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
	
Ϋ
B__inference_z_mean_layer_call_and_return_conditional_losses_149364

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
	
έ
D__inference_decoding_layer_call_and_return_conditional_losses_149795

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
­

»
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
identity’StatefulPartitionedCallΆ
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
-:+???????????????????????????*0
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
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
'
_user_specified_nameinput_decoder
Ο

ή
E__inference_conv4_enc_layer_call_and_return_conditional_losses_149269

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
:?????????@*
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
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
»
Ϋ
B__inference_output_layer_call_and_return_conditional_losses_149984

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
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
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
Ο

ή
E__inference_conv2_enc_layer_call_and_return_conditional_losses_151815

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
κ
ϋ
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
identity’Decoder/StatefulPartitionedCall’Encoder/StatefulPartitionedCall·
Encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_150289encoder_150291encoder_150293encoder_150295encoder_150297encoder_150299encoder_150301encoder_150303encoder_150305encoder_150307encoder_150309encoder_150311encoder_150313encoder_150315encoder_150317encoder_150319*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*2
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
Encoder/StatefulPartitionedCall¦
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_150390decoder_150392decoder_150394decoder_150396decoder_150398decoder_150400decoder_150402decoder_150404decoder_150406decoder_150408decoder_150410decoder_150412decoder_150414decoder_150416*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
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
Decoder/StatefulPartitionedCallΪ
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????(
!
_user_specified_name	input_1
­

»
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
identity’StatefulPartitionedCallΆ
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
-:+???????????????????????????*0
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
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:?????????
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
random_normal/stddevε
"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed22$
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
:?????????2
mulL
ExpExpmul:z:0*
T0*'
_output_shapes
:?????????2
Expc
mul_1MulExp:y:0random_normal:z:0*
T0*'
_output_shapes
:?????????2
mul_1Z
addAddV2inputs_0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add[
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
 
D
(__inference_upsamp1_layer_call_fn_149781

inputs
identityη
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
GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
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
½
_
C__inference_flatten_layer_call_and_return_conditional_losses_149320

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs


*__inference_conv5_enc_layer_call_fn_151884

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
 *0
_output_shapes
:?????????*$
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
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ψ

ή
E__inference_conv5_dec_layer_call_and_return_conditional_losses_149844

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
:?????????*
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
:?????????2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Β¬
χ
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
identity’,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp’+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp’,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp’+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp’,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp’+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp’,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp’+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp’,VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOp’+VAE/Decoder/conv5_dec/Conv2D/ReadVariableOp’+VAE/Decoder/decoding/BiasAdd/ReadVariableOp’*VAE/Decoder/decoding/MatMul/ReadVariableOp’)VAE/Decoder/output/BiasAdd/ReadVariableOp’(VAE/Decoder/output/Conv2D/ReadVariableOp’-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp’,VAE/Encoder/bottleneck/MatMul/ReadVariableOp’,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp’+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp’,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp’+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp’,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp’+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp’,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp’+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp’,VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOp’+VAE/Encoder/conv5_enc/Conv2D/ReadVariableOp’,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp’+VAE/Encoder/z_log_var/MatMul/ReadVariableOp’)VAE/Encoder/z_mean/BiasAdd/ReadVariableOp’(VAE/Encoder/z_mean/MatMul/ReadVariableOpΧ
+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+VAE/Encoder/conv1_enc/Conv2D/ReadVariableOpζ
VAE/Encoder/conv1_enc/Conv2DConv2Dinput_13VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
2
VAE/Encoder/conv1_enc/Conv2DΞ
,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOpΰ
VAE/Encoder/conv1_enc/BiasAddBiasAdd%VAE/Encoder/conv1_enc/Conv2D:output:04VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2
VAE/Encoder/conv1_enc/BiasAdd’
VAE/Encoder/conv1_enc/ReluRelu&VAE/Encoder/conv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
VAE/Encoder/conv1_enc/Reluέ
VAE/Encoder/maxpool1/MaxPoolMaxPool(VAE/Encoder/conv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool1/MaxPoolΧ
+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv2_enc/Conv2DConv2D%VAE/Encoder/maxpool1/MaxPool:output:03VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
VAE/Encoder/conv2_enc/Conv2DΞ
,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOpΰ
VAE/Encoder/conv2_enc/BiasAddBiasAdd%VAE/Encoder/conv2_enc/Conv2D:output:04VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
VAE/Encoder/conv2_enc/BiasAdd’
VAE/Encoder/conv2_enc/ReluRelu&VAE/Encoder/conv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
VAE/Encoder/conv2_enc/Reluέ
VAE/Encoder/maxpool2/MaxPoolMaxPool(VAE/Encoder/conv2_enc/Relu:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool2/MaxPoolΧ
+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv3_enc/Conv2DConv2D%VAE/Encoder/maxpool2/MaxPool:output:03VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 *
paddingSAME*
strides
2
VAE/Encoder/conv3_enc/Conv2DΞ
,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOpΰ
VAE/Encoder/conv3_enc/BiasAddBiasAdd%VAE/Encoder/conv3_enc/Conv2D:output:04VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 2
VAE/Encoder/conv3_enc/BiasAdd’
VAE/Encoder/conv3_enc/ReluRelu&VAE/Encoder/conv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
 2
VAE/Encoder/conv3_enc/Reluέ
VAE/Encoder/maxpool3/MaxPoolMaxPool(VAE/Encoder/conv3_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool3/MaxPoolΧ
+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02-
+VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv4_enc/Conv2DConv2D%VAE/Encoder/maxpool3/MaxPool:output:03VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
VAE/Encoder/conv4_enc/Conv2DΞ
,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOpΰ
VAE/Encoder/conv4_enc/BiasAddBiasAdd%VAE/Encoder/conv4_enc/Conv2D:output:04VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
VAE/Encoder/conv4_enc/BiasAdd’
VAE/Encoder/conv4_enc/ReluRelu&VAE/Encoder/conv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
VAE/Encoder/conv4_enc/Reluέ
VAE/Encoder/maxpool4/MaxPoolMaxPool(VAE/Encoder/conv4_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
VAE/Encoder/maxpool4/MaxPoolΨ
+VAE/Encoder/conv5_enc/Conv2D/ReadVariableOpReadVariableOp4vae_encoder_conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+VAE/Encoder/conv5_enc/Conv2D/ReadVariableOp
VAE/Encoder/conv5_enc/Conv2DConv2D%VAE/Encoder/maxpool4/MaxPool:output:03VAE/Encoder/conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
VAE/Encoder/conv5_enc/Conv2DΟ
,VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOpα
VAE/Encoder/conv5_enc/BiasAddBiasAdd%VAE/Encoder/conv5_enc/Conv2D:output:04VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
VAE/Encoder/conv5_enc/BiasAdd£
VAE/Encoder/conv5_enc/ReluRelu&VAE/Encoder/conv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
VAE/Encoder/conv5_enc/Reluή
VAE/Encoder/maxpool5/MaxPoolMaxPool(VAE/Encoder/conv5_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????   2
VAE/Encoder/flatten/ConstΓ
VAE/Encoder/flatten/ReshapeReshape%VAE/Encoder/maxpool5/MaxPool:output:0"VAE/Encoder/flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
VAE/Encoder/flatten/ReshapeΣ
,VAE/Encoder/bottleneck/MatMul/ReadVariableOpReadVariableOp5vae_encoder_bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02.
,VAE/Encoder/bottleneck/MatMul/ReadVariableOpΦ
VAE/Encoder/bottleneck/MatMulMatMul$VAE/Encoder/flatten/Reshape:output:04VAE/Encoder/bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/bottleneck/MatMulΡ
-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOpReadVariableOp6vae_encoder_bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-VAE/Encoder/bottleneck/BiasAdd/ReadVariableOpέ
VAE/Encoder/bottleneck/BiasAddBiasAdd'VAE/Encoder/bottleneck/MatMul:product:05VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
VAE/Encoder/bottleneck/BiasAddΖ
(VAE/Encoder/z_mean/MatMul/ReadVariableOpReadVariableOp1vae_encoder_z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(VAE/Encoder/z_mean/MatMul/ReadVariableOpΝ
VAE/Encoder/z_mean/MatMulMatMul'VAE/Encoder/bottleneck/BiasAdd:output:00VAE/Encoder/z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/z_mean/MatMulΕ
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOpReadVariableOp2vae_encoder_z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)VAE/Encoder/z_mean/BiasAdd/ReadVariableOpΝ
VAE/Encoder/z_mean/BiasAddBiasAdd#VAE/Encoder/z_mean/MatMul:product:01VAE/Encoder/z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/z_mean/BiasAddΟ
+VAE/Encoder/z_log_var/MatMul/ReadVariableOpReadVariableOp4vae_encoder_z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+VAE/Encoder/z_log_var/MatMul/ReadVariableOpΦ
VAE/Encoder/z_log_var/MatMulMatMul'VAE/Encoder/bottleneck/BiasAdd:output:03VAE/Encoder/z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/z_log_var/MatMulΞ
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOpReadVariableOp5vae_encoder_z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Encoder/z_log_var/BiasAdd/ReadVariableOpΩ
VAE/Encoder/z_log_var/BiasAddBiasAdd&VAE/Encoder/z_log_var/MatMul:product:04VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
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
(VAE/Encoder/sampling/strided_slice/stack’
*VAE/Encoder/sampling/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*VAE/Encoder/sampling/strided_slice/stack_1’
*VAE/Encoder/sampling/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*VAE/Encoder/sampling/strided_slice/stack_2ΰ
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
VAE/Encoder/sampling/Shape_1’
*VAE/Encoder/sampling/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*VAE/Encoder/sampling/strided_slice_1/stack¦
,VAE/Encoder/sampling/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling/strided_slice_1/stack_1¦
,VAE/Encoder/sampling/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,VAE/Encoder/sampling/strided_slice_1/stack_2μ
$VAE/Encoder/sampling/strided_slice_1StridedSlice%VAE/Encoder/sampling/Shape_1:output:03VAE/Encoder/sampling/strided_slice_1/stack:output:05VAE/Encoder/sampling/strided_slice_1/stack_1:output:05VAE/Encoder/sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$VAE/Encoder/sampling/strided_slice_1ζ
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
)VAE/Encoder/sampling/random_normal/stddev€
7VAE/Encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal1VAE/Encoder/sampling/random_normal/shape:output:0*
T0*0
_output_shapes
:??????????????????*
dtype0*
seed±?ε)*
seed2Π?κ29
7VAE/Encoder/sampling/random_normal/RandomStandardNormal
&VAE/Encoder/sampling/random_normal/mulMul@VAE/Encoder/sampling/random_normal/RandomStandardNormal:output:02VAE/Encoder/sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2(
&VAE/Encoder/sampling/random_normal/mulθ
"VAE/Encoder/sampling/random_normalAdd*VAE/Encoder/sampling/random_normal/mul:z:00VAE/Encoder/sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2$
"VAE/Encoder/sampling/random_normal}
VAE/Encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
VAE/Encoder/sampling/mul/xΊ
VAE/Encoder/sampling/mulMul#VAE/Encoder/sampling/mul/x:output:0&VAE/Encoder/z_log_var/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/sampling/mul
VAE/Encoder/sampling/ExpExpVAE/Encoder/sampling/mul:z:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/sampling/Exp·
VAE/Encoder/sampling/mul_1MulVAE/Encoder/sampling/Exp:y:0&VAE/Encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/sampling/mul_1΄
VAE/Encoder/sampling/addAddV2#VAE/Encoder/z_mean/BiasAdd:output:0VAE/Encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
VAE/Encoder/sampling/addΝ
*VAE/Decoder/decoding/MatMul/ReadVariableOpReadVariableOp3vae_decoder_decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02,
*VAE/Decoder/decoding/MatMul/ReadVariableOpΙ
VAE/Decoder/decoding/MatMulMatMulVAE/Encoder/sampling/add:z:02VAE/Decoder/decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
VAE/Decoder/decoding/MatMulΜ
+VAE/Decoder/decoding/BiasAdd/ReadVariableOpReadVariableOp4vae_decoder_decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+VAE/Decoder/decoding/BiasAdd/ReadVariableOpΦ
VAE/Decoder/decoding/BiasAddBiasAdd%VAE/Decoder/decoding/MatMul:product:03VAE/Decoder/decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
:?????????2
VAE/Decoder/reshape/ReshapeΩ
+VAE/Decoder/conv5_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+VAE/Decoder/conv5_dec/Conv2D/ReadVariableOp
VAE/Decoder/conv5_dec/Conv2DConv2D$VAE/Decoder/reshape/Reshape:output:03VAE/Decoder/conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
VAE/Decoder/conv5_dec/Conv2DΟ
,VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOpα
VAE/Decoder/conv5_dec/BiasAddBiasAdd%VAE/Decoder/conv5_dec/Conv2D:output:04VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
VAE/Decoder/conv5_dec/BiasAdd£
VAE/Decoder/conv5_dec/ReluRelu&VAE/Decoder/conv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
)VAE/Decoder/upsamp5/strided_slice/stack_2Ζ
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
VAE/Decoder/upsamp5/Const?
VAE/Decoder/upsamp5/mulMul*VAE/Decoder/upsamp5/strided_slice:output:0"VAE/Decoder/upsamp5/Const:output:0*
T0*
_output_shapes
:2
VAE/Decoder/upsamp5/mul
0VAE/Decoder/upsamp5/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv5_dec/Relu:activations:0VAE/Decoder/upsamp5/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(22
0VAE/Decoder/upsamp5/resize/ResizeNearestNeighborΨ
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv4_dec/Conv2DConv2DAVAE/Decoder/upsamp5/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
VAE/Decoder/conv4_dec/Conv2DΞ
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOpΰ
VAE/Decoder/conv4_dec/BiasAddBiasAdd%VAE/Decoder/conv4_dec/Conv2D:output:04VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
VAE/Decoder/conv4_dec/BiasAdd’
VAE/Decoder/conv4_dec/ReluRelu&VAE/Decoder/conv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
VAE/Decoder/upsamp4/mul
0VAE/Decoder/upsamp4/resize/ResizeNearestNeighborResizeNearestNeighbor(VAE/Decoder/conv4_dec/Relu:activations:0VAE/Decoder/upsamp4/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(22
0VAE/Decoder/upsamp4/resize/ResizeNearestNeighborΧ
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02-
+VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv3_dec/Conv2DConv2DAVAE/Decoder/upsamp4/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
VAE/Decoder/conv3_dec/Conv2DΞ
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOpΰ
VAE/Decoder/conv3_dec/BiasAddBiasAdd%VAE/Decoder/conv3_dec/Conv2D:output:04VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
VAE/Decoder/conv3_dec/BiasAdd’
VAE/Decoder/conv3_dec/ReluRelu&VAE/Decoder/conv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:????????? *
half_pixel_centers(22
0VAE/Decoder/upsamp3/resize/ResizeNearestNeighborΧ
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv2_dec/Conv2DConv2DAVAE/Decoder/upsamp3/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
VAE/Decoder/conv2_dec/Conv2DΞ
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOpΰ
VAE/Decoder/conv2_dec/BiasAddBiasAdd%VAE/Decoder/conv2_dec/Conv2D:output:04VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
VAE/Decoder/conv2_dec/BiasAdd’
VAE/Decoder/conv2_dec/ReluRelu&VAE/Decoder/conv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
:????????? *
half_pixel_centers(22
0VAE/Decoder/upsamp2/resize/ResizeNearestNeighborΧ
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOpReadVariableOp4vae_decoder_conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp 
VAE/Decoder/conv1_dec/Conv2DConv2DAVAE/Decoder/upsamp2/resize/ResizeNearestNeighbor:resized_images:03VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
VAE/Decoder/conv1_dec/Conv2DΞ
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpReadVariableOp5vae_decoder_conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOpΰ
VAE/Decoder/conv1_dec/BiasAddBiasAdd%VAE/Decoder/conv1_dec/Conv2D:output:04VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
VAE/Decoder/conv1_dec/BiasAdd’
VAE/Decoder/conv1_dec/ReluRelu&VAE/Decoder/conv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:?????????@ *
half_pixel_centers(22
0VAE/Decoder/upsamp1/resize/ResizeNearestNeighborΞ
(VAE/Decoder/output/Conv2D/ReadVariableOpReadVariableOp1vae_decoder_output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(VAE/Decoder/output/Conv2D/ReadVariableOp
VAE/Decoder/output/Conv2DConv2DAVAE/Decoder/upsamp1/resize/ResizeNearestNeighbor:resized_images:00VAE/Decoder/output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
VAE/Decoder/output/BiasAdd’
VAE/Decoder/output/SigmoidSigmoid#VAE/Decoder/output/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
VAE/Decoder/output/Sigmoidα
IdentityIdentityVAE/Decoder/output/Sigmoid:y:0-^VAE/Decoder/conv1_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv1_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv2_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv2_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv3_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv3_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv4_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv4_dec/Conv2D/ReadVariableOp-^VAE/Decoder/conv5_dec/BiasAdd/ReadVariableOp,^VAE/Decoder/conv5_dec/Conv2D/ReadVariableOp,^VAE/Decoder/decoding/BiasAdd/ReadVariableOp+^VAE/Decoder/decoding/MatMul/ReadVariableOp*^VAE/Decoder/output/BiasAdd/ReadVariableOp)^VAE/Decoder/output/Conv2D/ReadVariableOp.^VAE/Encoder/bottleneck/BiasAdd/ReadVariableOp-^VAE/Encoder/bottleneck/MatMul/ReadVariableOp-^VAE/Encoder/conv1_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv1_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv2_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv2_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv3_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv3_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv4_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv4_enc/Conv2D/ReadVariableOp-^VAE/Encoder/conv5_enc/BiasAdd/ReadVariableOp,^VAE/Encoder/conv5_enc/Conv2D/ReadVariableOp-^VAE/Encoder/z_log_var/BiasAdd/ReadVariableOp,^VAE/Encoder/z_log_var/MatMul/ReadVariableOp*^VAE/Encoder/z_mean/BiasAdd/ReadVariableOp)^VAE/Encoder/z_mean/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::2\
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
:?????????(
!
_user_specified_name	input_1

±
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
identity’StatefulPartitionedCallά
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
:?????????(*@
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
:?????????(2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(
!
_user_specified_name	input_1
±
ΐ	
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
identity’ conv1_dec/BiasAdd/ReadVariableOp’conv1_dec/Conv2D/ReadVariableOp’ conv2_dec/BiasAdd/ReadVariableOp’conv2_dec/Conv2D/ReadVariableOp’ conv3_dec/BiasAdd/ReadVariableOp’conv3_dec/Conv2D/ReadVariableOp’ conv4_dec/BiasAdd/ReadVariableOp’conv4_dec/Conv2D/ReadVariableOp’ conv5_dec/BiasAdd/ReadVariableOp’conv5_dec/Conv2D/ReadVariableOp’decoding/BiasAdd/ReadVariableOp’decoding/MatMul/ReadVariableOp’output/BiasAdd/ReadVariableOp’output/Conv2D/ReadVariableOp©
decoding/MatMul/ReadVariableOpReadVariableOp'decoding_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02 
decoding/MatMul/ReadVariableOp
decoding/MatMulMatMulinputs&decoding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
decoding/MatMul¨
decoding/BiasAdd/ReadVariableOpReadVariableOp(decoding_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
decoding/BiasAdd/ReadVariableOp¦
decoding/BiasAddBiasAdddecoding/MatMul:product:0'decoding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
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
:?????????2
reshape/Reshape΅
conv5_dec/Conv2D/ReadVariableOpReadVariableOp(conv5_dec_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv5_dec/Conv2D/ReadVariableOpΤ
conv5_dec/Conv2DConv2Dreshape/Reshape:output:0'conv5_dec/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv5_dec/Conv2D«
 conv5_dec/BiasAdd/ReadVariableOpReadVariableOp)conv5_dec_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv5_dec/BiasAdd/ReadVariableOp±
conv5_dec/BiasAddBiasAddconv5_dec/Conv2D:output:0(conv5_dec/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv5_dec/BiasAdd
conv5_dec/ReluReluconv5_dec/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
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
upsamp5/strided_slice/stack_2ώ
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
upsamp5/mulι
$upsamp5/resize/ResizeNearestNeighborResizeNearestNeighborconv5_dec/Relu:activations:0upsamp5/mul:z:0*
T0*0
_output_shapes
:?????????*
half_pixel_centers(2&
$upsamp5/resize/ResizeNearestNeighbor΄
conv4_dec/Conv2D/ReadVariableOpReadVariableOp(conv4_dec_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv4_dec/Conv2D/ReadVariableOpπ
conv4_dec/Conv2DConv2D5upsamp5/resize/ResizeNearestNeighbor:resized_images:0'conv4_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv4_dec/Conv2Dͺ
 conv4_dec/BiasAdd/ReadVariableOpReadVariableOp)conv4_dec_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv4_dec/BiasAdd/ReadVariableOp°
conv4_dec/BiasAddBiasAddconv4_dec/Conv2D:output:0(conv4_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv4_dec/BiasAdd~
conv4_dec/ReluReluconv4_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
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
upsamp4/mulθ
$upsamp4/resize/ResizeNearestNeighborResizeNearestNeighborconv4_dec/Relu:activations:0upsamp4/mul:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2&
$upsamp4/resize/ResizeNearestNeighbor³
conv3_dec/Conv2D/ReadVariableOpReadVariableOp(conv3_dec_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv3_dec/Conv2D/ReadVariableOpπ
conv3_dec/Conv2DConv2D5upsamp4/resize/ResizeNearestNeighbor:resized_images:0'conv3_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv3_dec/Conv2Dͺ
 conv3_dec/BiasAdd/ReadVariableOpReadVariableOp)conv3_dec_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3_dec/BiasAdd/ReadVariableOp°
conv3_dec/BiasAddBiasAddconv3_dec/Conv2D:output:0(conv3_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv3_dec/BiasAdd~
conv3_dec/ReluReluconv3_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:????????? *
half_pixel_centers(2&
$upsamp3/resize/ResizeNearestNeighbor³
conv2_dec/Conv2D/ReadVariableOpReadVariableOp(conv2_dec_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2_dec/Conv2D/ReadVariableOpπ
conv2_dec/Conv2DConv2D5upsamp3/resize/ResizeNearestNeighbor:resized_images:0'conv2_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2_dec/Conv2Dͺ
 conv2_dec/BiasAdd/ReadVariableOpReadVariableOp)conv2_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2_dec/BiasAdd/ReadVariableOp°
conv2_dec/BiasAddBiasAddconv2_dec/Conv2D:output:0(conv2_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2_dec/BiasAdd~
conv2_dec/ReluReluconv2_dec/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
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
:????????? *
half_pixel_centers(2&
$upsamp2/resize/ResizeNearestNeighbor³
conv1_dec/Conv2D/ReadVariableOpReadVariableOp(conv1_dec_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_dec/Conv2D/ReadVariableOpπ
conv1_dec/Conv2DConv2D5upsamp2/resize/ResizeNearestNeighbor:resized_images:0'conv1_dec/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv1_dec/Conv2Dͺ
 conv1_dec/BiasAdd/ReadVariableOpReadVariableOp)conv1_dec_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_dec/BiasAdd/ReadVariableOp°
conv1_dec/BiasAddBiasAddconv1_dec/Conv2D:output:0(conv1_dec/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv1_dec/BiasAdd~
conv1_dec/ReluReluconv1_dec/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
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
:?????????@ *
half_pixel_centers(2&
$upsamp1/resize/ResizeNearestNeighborͺ
output/Conv2D/ReadVariableOpReadVariableOp%output_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
output/Conv2D/ReadVariableOpθ
output/Conv2DConv2D5upsamp1/resize/ResizeNearestNeighbor:resized_images:0$output/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
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
:?????????(2
output/BiasAdd~
output/SigmoidSigmoidoutput/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
output/SigmoidΙ
IdentityIdentityoutput/Sigmoid:y:0!^conv1_dec/BiasAdd/ReadVariableOp ^conv1_dec/Conv2D/ReadVariableOp!^conv2_dec/BiasAdd/ReadVariableOp ^conv2_dec/Conv2D/ReadVariableOp!^conv3_dec/BiasAdd/ReadVariableOp ^conv3_dec/Conv2D/ReadVariableOp!^conv4_dec/BiasAdd/ReadVariableOp ^conv4_dec/Conv2D/ReadVariableOp!^conv5_dec/BiasAdd/ReadVariableOp ^conv5_dec/Conv2D/ReadVariableOp ^decoding/BiasAdd/ReadVariableOp^decoding/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????(2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::2D
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
:?????????
 
_user_specified_nameinputs
ζ

+__inference_bottleneck_layer_call_fn_151914

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
:?????????*$
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
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
 
D
(__inference_upsamp3_layer_call_fn_149743

inputs
identityη
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
GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
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
 
D
(__inference_upsamp5_layer_call_fn_149705

inputs
identityη
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
GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
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
Ύ
ή
E__inference_conv4_dec_layer_call_and_return_conditional_losses_152053

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


΄
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
identity’StatefulPartitionedCall―
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
-:+???????????????????????????*0
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
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
’
E
)__inference_maxpool4_layer_call_fn_149158

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
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
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
	
ί
F__inference_bottleneck_layer_call_and_return_conditional_losses_151905

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
¦
r
)__inference_sampling_layer_call_fn_151984
inputs_0
inputs_1
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
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
:?????????2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
»
ή
E__inference_conv3_dec_layer_call_and_return_conditional_losses_152073

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
	
έ
D__inference_decoding_layer_call_and_return_conditional_losses_151994

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs

ϋ
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

identity_2’StatefulPartitionedCallΰ
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
9:?????????:?????????:?????????*2
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
:?????????2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:?????????(
'
_user_specified_nameinput_encoder


*__inference_conv5_dec_layer_call_fn_152042

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
 *0
_output_shapes
:?????????*$
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
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
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
 
D
(__inference_upsamp2_layer_call_fn_149762

inputs
identityη
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
GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
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
?½
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
ShardedFilename°6
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*Β5
valueΈ5B΅5fB3total_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB3total_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB<reconstruction_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/total/.ATTRIBUTES/VARIABLE_VALUEB0kl_loss_tracker/count/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesΧ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*α
valueΧBΤfB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices·'
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0 savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop+savev2_conv1_enc_kernel_read_readvariableop)savev2_conv1_enc_bias_read_readvariableop+savev2_conv2_enc_kernel_read_readvariableop)savev2_conv2_enc_bias_read_readvariableop+savev2_conv3_enc_kernel_read_readvariableop)savev2_conv3_enc_bias_read_readvariableop+savev2_conv4_enc_kernel_read_readvariableop)savev2_conv4_enc_bias_read_readvariableop+savev2_conv5_enc_kernel_read_readvariableop)savev2_conv5_enc_bias_read_readvariableop,savev2_bottleneck_kernel_read_readvariableop*savev2_bottleneck_bias_read_readvariableop(savev2_z_mean_kernel_read_readvariableop&savev2_z_mean_bias_read_readvariableop+savev2_z_log_var_kernel_read_readvariableop)savev2_z_log_var_bias_read_readvariableop*savev2_decoding_kernel_read_readvariableop(savev2_decoding_bias_read_readvariableop+savev2_conv5_dec_kernel_read_readvariableop)savev2_conv5_dec_bias_read_readvariableop+savev2_conv4_dec_kernel_read_readvariableop)savev2_conv4_dec_bias_read_readvariableop+savev2_conv3_dec_kernel_read_readvariableop)savev2_conv3_dec_bias_read_readvariableop+savev2_conv2_dec_kernel_read_readvariableop)savev2_conv2_dec_bias_read_readvariableop+savev2_conv1_dec_kernel_read_readvariableop)savev2_conv1_dec_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop2savev2_adam_conv1_enc_kernel_m_read_readvariableop0savev2_adam_conv1_enc_bias_m_read_readvariableop2savev2_adam_conv2_enc_kernel_m_read_readvariableop0savev2_adam_conv2_enc_bias_m_read_readvariableop2savev2_adam_conv3_enc_kernel_m_read_readvariableop0savev2_adam_conv3_enc_bias_m_read_readvariableop2savev2_adam_conv4_enc_kernel_m_read_readvariableop0savev2_adam_conv4_enc_bias_m_read_readvariableop2savev2_adam_conv5_enc_kernel_m_read_readvariableop0savev2_adam_conv5_enc_bias_m_read_readvariableop3savev2_adam_bottleneck_kernel_m_read_readvariableop1savev2_adam_bottleneck_bias_m_read_readvariableop/savev2_adam_z_mean_kernel_m_read_readvariableop-savev2_adam_z_mean_bias_m_read_readvariableop2savev2_adam_z_log_var_kernel_m_read_readvariableop0savev2_adam_z_log_var_bias_m_read_readvariableop1savev2_adam_decoding_kernel_m_read_readvariableop/savev2_adam_decoding_bias_m_read_readvariableop2savev2_adam_conv5_dec_kernel_m_read_readvariableop0savev2_adam_conv5_dec_bias_m_read_readvariableop2savev2_adam_conv4_dec_kernel_m_read_readvariableop0savev2_adam_conv4_dec_bias_m_read_readvariableop2savev2_adam_conv3_dec_kernel_m_read_readvariableop0savev2_adam_conv3_dec_bias_m_read_readvariableop2savev2_adam_conv2_dec_kernel_m_read_readvariableop0savev2_adam_conv2_dec_bias_m_read_readvariableop2savev2_adam_conv1_dec_kernel_m_read_readvariableop0savev2_adam_conv1_dec_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop2savev2_adam_conv1_enc_kernel_v_read_readvariableop0savev2_adam_conv1_enc_bias_v_read_readvariableop2savev2_adam_conv2_enc_kernel_v_read_readvariableop0savev2_adam_conv2_enc_bias_v_read_readvariableop2savev2_adam_conv3_enc_kernel_v_read_readvariableop0savev2_adam_conv3_enc_bias_v_read_readvariableop2savev2_adam_conv4_enc_kernel_v_read_readvariableop0savev2_adam_conv4_enc_bias_v_read_readvariableop2savev2_adam_conv5_enc_kernel_v_read_readvariableop0savev2_adam_conv5_enc_bias_v_read_readvariableop3savev2_adam_bottleneck_kernel_v_read_readvariableop1savev2_adam_bottleneck_bias_v_read_readvariableop/savev2_adam_z_mean_kernel_v_read_readvariableop-savev2_adam_z_mean_bias_v_read_readvariableop2savev2_adam_z_log_var_kernel_v_read_readvariableop0savev2_adam_z_log_var_bias_v_read_readvariableop1savev2_adam_decoding_kernel_v_read_readvariableop/savev2_adam_decoding_bias_v_read_readvariableop2savev2_adam_conv5_dec_kernel_v_read_readvariableop0savev2_adam_conv5_dec_bias_v_read_readvariableop2savev2_adam_conv4_dec_kernel_v_read_readvariableop0savev2_adam_conv4_dec_bias_v_read_readvariableop2savev2_adam_conv3_dec_kernel_v_read_readvariableop0savev2_adam_conv3_dec_bias_v_read_readvariableop2savev2_adam_conv2_dec_kernel_v_read_readvariableop0savev2_adam_conv2_dec_bias_v_read_readvariableop2savev2_adam_conv1_dec_kernel_v_read_readvariableop0savev2_adam_conv1_dec_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *t
dtypesj
h2f	2
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

identity_1Identity_1:output:0*’
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
©{
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

identity_2’!bottleneck/BiasAdd/ReadVariableOp’ bottleneck/MatMul/ReadVariableOp’ conv1_enc/BiasAdd/ReadVariableOp’conv1_enc/Conv2D/ReadVariableOp’ conv2_enc/BiasAdd/ReadVariableOp’conv2_enc/Conv2D/ReadVariableOp’ conv3_enc/BiasAdd/ReadVariableOp’conv3_enc/Conv2D/ReadVariableOp’ conv4_enc/BiasAdd/ReadVariableOp’conv4_enc/Conv2D/ReadVariableOp’ conv5_enc/BiasAdd/ReadVariableOp’conv5_enc/Conv2D/ReadVariableOp’ z_log_var/BiasAdd/ReadVariableOp’z_log_var/MatMul/ReadVariableOp’z_mean/BiasAdd/ReadVariableOp’z_mean/MatMul/ReadVariableOp³
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOpΑ
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
2
conv1_enc/Conv2Dͺ
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOp°
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
conv1_enc/ReluΉ
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool³
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2_enc/Conv2D/ReadVariableOpΤ
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2_enc/Conv2Dͺ
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2_enc/BiasAdd/ReadVariableOp°
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2_enc/ReluΉ
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPool³
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv3_enc/Conv2D/ReadVariableOpΤ
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 *
paddingSAME*
strides
2
conv3_enc/Conv2Dͺ
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3_enc/BiasAdd/ReadVariableOp°
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
 2
conv3_enc/ReluΉ
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPool³
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv4_enc/Conv2D/ReadVariableOpΤ
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv4_enc/Conv2Dͺ
 conv4_enc/BiasAdd/ReadVariableOpReadVariableOp)conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv4_enc/BiasAdd/ReadVariableOp°
conv4_enc/BiasAddBiasAddconv4_enc/Conv2D:output:0(conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv4_enc/BiasAdd~
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv4_enc/ReluΉ
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
maxpool4/MaxPool΄
conv5_enc/Conv2D/ReadVariableOpReadVariableOp(conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv5_enc/Conv2D/ReadVariableOpΥ
conv5_enc/Conv2DConv2Dmaxpool4/MaxPool:output:0'conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv5_enc/Conv2D«
 conv5_enc/BiasAdd/ReadVariableOpReadVariableOp)conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv5_enc/BiasAdd/ReadVariableOp±
conv5_enc/BiasAddBiasAddconv5_enc/Conv2D:output:0(conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv5_enc/BiasAdd
conv5_enc/ReluReluconv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
conv5_enc/ReluΊ
maxpool5/MaxPoolMaxPoolconv5_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????   2
flatten/Const
flatten/ReshapeReshapemaxpool5/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten/Reshape―
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 bottleneck/MatMul/ReadVariableOp¦
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/MatMul­
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp­
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/BiasAdd’
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMulbottleneck/BiasAdd:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/MatMul‘
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/BiasAdd«
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
z_log_var/MatMul/ReadVariableOp¦
z_log_var/MatMulMatMulbottleneck/BiasAdd:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/MatMulͺ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp©
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
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
 sampling/strided_slice_1/stack_2€
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1Ά
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
:??????????????????*
dtype0*
seed±?ε)*
seed2γ­ΐ2-
+sampling/random_normal/RandomStandardNormalΨ
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
sampling/random_normal/mulΈ
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
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
:?????????2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling/Exp
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling/mul_1
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling/add
IdentityIdentityz_mean/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identityz_log_var/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identitysampling/add:z:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::2F
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
:?????????(
 
_user_specified_nameinputs
η
ϊ
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
identity’Decoder/StatefulPartitionedCall’Encoder/StatefulPartitionedCallΆ
Encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_150494encoder_150496encoder_150498encoder_150500encoder_150502encoder_150504encoder_150506encoder_150508encoder_150510encoder_150512encoder_150514encoder_150516encoder_150518encoder_150520encoder_150522encoder_150524*
Tin
2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*2
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
Encoder/StatefulPartitionedCall¦
Decoder/StatefulPartitionedCallStatefulPartitionedCall(Encoder/StatefulPartitionedCall:output:2decoder_150529decoder_150531decoder_150533decoder_150535decoder_150537decoder_150539decoder_150541decoder_150543decoder_150545decoder_150547decoder_150549decoder_150551decoder_150553decoder_150555*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
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
Decoder/StatefulPartitionedCallΪ
IdentityIdentity(Decoder/StatefulPartitionedCall:output:0 ^Decoder/StatefulPartitionedCall ^Encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::2B
Decoder/StatefulPartitionedCallDecoder/StatefulPartitionedCall2B
Encoder/StatefulPartitionedCallEncoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
ΥI
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

identity_2’"bottleneck/StatefulPartitionedCall’!conv1_enc/StatefulPartitionedCall’!conv2_enc/StatefulPartitionedCall’!conv3_enc/StatefulPartitionedCall’!conv4_enc/StatefulPartitionedCall’!conv5_enc/StatefulPartitionedCall’ sampling/StatefulPartitionedCall’!z_log_var/StatefulPartitionedCall’z_mean/StatefulPartitionedCall€
!conv1_enc/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_enc_149503conv1_enc_149505*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool1_layer_call_and_return_conditional_losses_1491162
maxpool1/PartitionedCallΏ
!conv2_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0conv2_enc_149509conv2_enc_149511*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
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
:?????????
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
maxpool2/PartitionedCallΏ
!conv3_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0conv3_enc_149515conv3_enc_149517*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool3_layer_call_and_return_conditional_losses_1491402
maxpool3/PartitionedCallΏ
!conv4_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool3/PartitionedCall:output:0conv4_enc_149521conv4_enc_149523*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
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
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool4_layer_call_and_return_conditional_losses_1491522
maxpool4/PartitionedCallΐ
!conv5_enc/StatefulPartitionedCallStatefulPartitionedCall!maxpool4/PartitionedCall:output:0conv5_enc_149527conv5_enc_149529*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_maxpool5_layer_call_and_return_conditional_losses_1491642
maxpool5/PartitionedCallπ
flatten/PartitionedCallPartitionedCall!maxpool5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_1493202
flatten/PartitionedCall»
"bottleneck/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0bottleneck_149534bottleneck_149536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_bottleneck_layer_call_and_return_conditional_losses_1493382$
"bottleneck/StatefulPartitionedCall²
z_mean/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_mean_149539z_mean_149541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_z_mean_layer_call_and_return_conditional_losses_1493642 
z_mean/StatefulPartitionedCallΑ
!z_log_var/StatefulPartitionedCallStatefulPartitionedCall+bottleneck/StatefulPartitionedCall:output:0z_log_var_149544z_log_var_149546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_z_log_var_layer_call_and_return_conditional_losses_1493902#
!z_log_var/StatefulPartitionedCall½
 sampling/StatefulPartitionedCallStatefulPartitionedCall'z_mean/StatefulPartitionedCall:output:0*z_log_var/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sampling_layer_call_and_return_conditional_losses_1494322"
 sampling/StatefulPartitionedCallΌ
IdentityIdentity'z_mean/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

IdentityΓ

Identity_1Identity*z_log_var/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1Β

Identity_2Identity)sampling/StatefulPartitionedCall:output:0#^bottleneck/StatefulPartitionedCall"^conv1_enc/StatefulPartitionedCall"^conv2_enc/StatefulPartitionedCall"^conv3_enc/StatefulPartitionedCall"^conv4_enc/StatefulPartitionedCall"^conv5_enc/StatefulPartitionedCall!^sampling/StatefulPartitionedCall"^z_log_var/StatefulPartitionedCall^z_mean/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::2H
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
:?????????(
 
_user_specified_nameinputs
³9
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
identity’!conv1_dec/StatefulPartitionedCall’!conv2_dec/StatefulPartitionedCall’!conv3_dec/StatefulPartitionedCall’!conv4_dec/StatefulPartitionedCall’!conv5_dec/StatefulPartitionedCall’ decoding/StatefulPartitionedCall’output/StatefulPartitionedCall
 decoding/StatefulPartitionedCallStatefulPartitionedCallinputsdecoding_150130decoding_150132*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
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
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_1498252
reshape/PartitionedCallΏ
!conv5_dec/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv5_dec_150136conv5_dec_150138*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*$
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
.:,???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp5_layer_call_and_return_conditional_losses_1496992
upsamp5/PartitionedCallΠ
!conv4_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp5/PartitionedCall:output:0conv4_dec_150142conv4_dec_150144*
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
-:+???????????????????????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp4_layer_call_and_return_conditional_losses_1497182
upsamp4/PartitionedCallΠ
!conv3_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp4/PartitionedCall:output:0conv3_dec_150148conv3_dec_150150*
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
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp3_layer_call_and_return_conditional_losses_1497372
upsamp3/PartitionedCallΠ
!conv2_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp3/PartitionedCall:output:0conv2_dec_150154conv2_dec_150156*
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
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp2_layer_call_and_return_conditional_losses_1497562
upsamp2/PartitionedCallΠ
!conv1_dec/StatefulPartitionedCallStatefulPartitionedCall upsamp2/PartitionedCall:output:0conv1_dec_150160conv1_dec_150162*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
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
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_upsamp1_layer_call_and_return_conditional_losses_1497752
upsamp1/PartitionedCallΑ
output/StatefulPartitionedCallStatefulPartitionedCall upsamp1/PartitionedCall:output:0output_150166output_150168*
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
GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1499842 
output/StatefulPartitionedCall
IdentityIdentity'output/StatefulPartitionedCall:output:0"^conv1_dec/StatefulPartitionedCall"^conv2_dec/StatefulPartitionedCall"^conv3_dec/StatefulPartitionedCall"^conv4_dec/StatefulPartitionedCall"^conv5_dec/StatefulPartitionedCall!^decoding/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*^
_input_shapesM
K:?????????::::::::::::::2F
!conv1_dec/StatefulPartitionedCall!conv1_dec/StatefulPartitionedCall2F
!conv2_dec/StatefulPartitionedCall!conv2_dec/StatefulPartitionedCall2F
!conv3_dec/StatefulPartitionedCall!conv3_dec/StatefulPartitionedCall2F
!conv4_dec/StatefulPartitionedCall!conv4_dec/StatefulPartitionedCall2F
!conv5_dec/StatefulPartitionedCall!conv5_dec/StatefulPartitionedCall2D
 decoding/StatefulPartitionedCall decoding/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ο

ή
E__inference_conv2_enc_layer_call_and_return_conditional_losses_149213

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
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
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ο

ή
E__inference_conv4_enc_layer_call_and_return_conditional_losses_151855

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
:?????????@*
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
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
Γ
°
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
identity’StatefulPartitionedCall
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
-:+???????????????????????????*@
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
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*¨
_input_shapes
:?????????(::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(
 
_user_specified_nameinputs
§
D
(__inference_reshape_layer_call_fn_152022

inputs
identityΝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
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
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
©{
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

identity_2’!bottleneck/BiasAdd/ReadVariableOp’ bottleneck/MatMul/ReadVariableOp’ conv1_enc/BiasAdd/ReadVariableOp’conv1_enc/Conv2D/ReadVariableOp’ conv2_enc/BiasAdd/ReadVariableOp’conv2_enc/Conv2D/ReadVariableOp’ conv3_enc/BiasAdd/ReadVariableOp’conv3_enc/Conv2D/ReadVariableOp’ conv4_enc/BiasAdd/ReadVariableOp’conv4_enc/Conv2D/ReadVariableOp’ conv5_enc/BiasAdd/ReadVariableOp’conv5_enc/Conv2D/ReadVariableOp’ z_log_var/BiasAdd/ReadVariableOp’z_log_var/MatMul/ReadVariableOp’z_mean/BiasAdd/ReadVariableOp’z_mean/MatMul/ReadVariableOp³
conv1_enc/Conv2D/ReadVariableOpReadVariableOp(conv1_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv1_enc/Conv2D/ReadVariableOpΑ
conv1_enc/Conv2DConv2Dinputs'conv1_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(*
paddingSAME*
strides
2
conv1_enc/Conv2Dͺ
 conv1_enc/BiasAdd/ReadVariableOpReadVariableOp)conv1_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv1_enc/BiasAdd/ReadVariableOp°
conv1_enc/BiasAddBiasAddconv1_enc/Conv2D:output:0(conv1_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????(2
conv1_enc/BiasAdd~
conv1_enc/ReluReluconv1_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????(2
conv1_enc/ReluΉ
maxpool1/MaxPoolMaxPoolconv1_enc/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
maxpool1/MaxPool³
conv2_enc/Conv2D/ReadVariableOpReadVariableOp(conv2_enc_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2_enc/Conv2D/ReadVariableOpΤ
conv2_enc/Conv2DConv2Dmaxpool1/MaxPool:output:0'conv2_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2_enc/Conv2Dͺ
 conv2_enc/BiasAdd/ReadVariableOpReadVariableOp)conv2_enc_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2_enc/BiasAdd/ReadVariableOp°
conv2_enc/BiasAddBiasAddconv2_enc/Conv2D:output:0(conv2_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2_enc/BiasAdd~
conv2_enc/ReluReluconv2_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2_enc/ReluΉ
maxpool2/MaxPoolMaxPoolconv2_enc/Relu:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingSAME*
strides
2
maxpool2/MaxPool³
conv3_enc/Conv2D/ReadVariableOpReadVariableOp(conv3_enc_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv3_enc/Conv2D/ReadVariableOpΤ
conv3_enc/Conv2DConv2Dmaxpool2/MaxPool:output:0'conv3_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 *
paddingSAME*
strides
2
conv3_enc/Conv2Dͺ
 conv3_enc/BiasAdd/ReadVariableOpReadVariableOp)conv3_enc_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv3_enc/BiasAdd/ReadVariableOp°
conv3_enc/BiasAddBiasAddconv3_enc/Conv2D:output:0(conv3_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
 2
conv3_enc/BiasAdd~
conv3_enc/ReluReluconv3_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????
 2
conv3_enc/ReluΉ
maxpool3/MaxPoolMaxPoolconv3_enc/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
maxpool3/MaxPool³
conv4_enc/Conv2D/ReadVariableOpReadVariableOp(conv4_enc_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv4_enc/Conv2D/ReadVariableOpΤ
conv4_enc/Conv2DConv2Dmaxpool3/MaxPool:output:0'conv4_enc/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv4_enc/Conv2Dͺ
 conv4_enc/BiasAdd/ReadVariableOpReadVariableOp)conv4_enc_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv4_enc/BiasAdd/ReadVariableOp°
conv4_enc/BiasAddBiasAddconv4_enc/Conv2D:output:0(conv4_enc/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv4_enc/BiasAdd~
conv4_enc/ReluReluconv4_enc/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv4_enc/ReluΉ
maxpool4/MaxPoolMaxPoolconv4_enc/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingSAME*
strides
2
maxpool4/MaxPool΄
conv5_enc/Conv2D/ReadVariableOpReadVariableOp(conv5_enc_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv5_enc/Conv2D/ReadVariableOpΥ
conv5_enc/Conv2DConv2Dmaxpool4/MaxPool:output:0'conv5_enc/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
2
conv5_enc/Conv2D«
 conv5_enc/BiasAdd/ReadVariableOpReadVariableOp)conv5_enc_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv5_enc/BiasAdd/ReadVariableOp±
conv5_enc/BiasAddBiasAddconv5_enc/Conv2D:output:0(conv5_enc/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????2
conv5_enc/BiasAdd
conv5_enc/ReluReluconv5_enc/BiasAdd:output:0*
T0*0
_output_shapes
:?????????2
conv5_enc/ReluΊ
maxpool5/MaxPoolMaxPoolconv5_enc/Relu:activations:0*0
_output_shapes
:?????????*
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
valueB"????   2
flatten/Const
flatten/ReshapeReshapemaxpool5/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten/Reshape―
 bottleneck/MatMul/ReadVariableOpReadVariableOp)bottleneck_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02"
 bottleneck/MatMul/ReadVariableOp¦
bottleneck/MatMulMatMulflatten/Reshape:output:0(bottleneck/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/MatMul­
!bottleneck/BiasAdd/ReadVariableOpReadVariableOp*bottleneck_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!bottleneck/BiasAdd/ReadVariableOp­
bottleneck/BiasAddBiasAddbottleneck/MatMul:product:0)bottleneck/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
bottleneck/BiasAdd’
z_mean/MatMul/ReadVariableOpReadVariableOp%z_mean_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
z_mean/MatMul/ReadVariableOp
z_mean/MatMulMatMulbottleneck/BiasAdd:output:0$z_mean/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/MatMul‘
z_mean/BiasAdd/ReadVariableOpReadVariableOp&z_mean_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
z_mean/BiasAdd/ReadVariableOp
z_mean/BiasAddBiasAddz_mean/MatMul:product:0%z_mean/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_mean/BiasAdd«
z_log_var/MatMul/ReadVariableOpReadVariableOp(z_log_var_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
z_log_var/MatMul/ReadVariableOp¦
z_log_var/MatMulMatMulbottleneck/BiasAdd:output:0'z_log_var/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
z_log_var/MatMulͺ
 z_log_var/BiasAdd/ReadVariableOpReadVariableOp)z_log_var_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 z_log_var/BiasAdd/ReadVariableOp©
z_log_var/BiasAddBiasAddz_log_var/MatMul:product:0(z_log_var/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
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
 sampling/strided_slice_1/stack_2€
sampling/strided_slice_1StridedSlicesampling/Shape_1:output:0'sampling/strided_slice_1/stack:output:0)sampling/strided_slice_1/stack_1:output:0)sampling/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sampling/strided_slice_1Ά
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
:??????????????????*
dtype0*
seed±?ε)*
seed2ΙΑ2-
+sampling/random_normal/RandomStandardNormalΨ
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*0
_output_shapes
:??????????????????2
sampling/random_normal/mulΈ
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*0
_output_shapes
:??????????????????2
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
:?????????2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling/Exp
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling/mul_1
sampling/addAddV2z_mean/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling/add
IdentityIdentityz_mean/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity

Identity_1Identityz_log_var/BiasAdd:output:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1

Identity_2Identitysampling/add:z:0"^bottleneck/BiasAdd/ReadVariableOp!^bottleneck/MatMul/ReadVariableOp!^conv1_enc/BiasAdd/ReadVariableOp ^conv1_enc/Conv2D/ReadVariableOp!^conv2_enc/BiasAdd/ReadVariableOp ^conv2_enc/Conv2D/ReadVariableOp!^conv3_enc/BiasAdd/ReadVariableOp ^conv3_enc/Conv2D/ReadVariableOp!^conv4_enc/BiasAdd/ReadVariableOp ^conv4_enc/Conv2D/ReadVariableOp!^conv5_enc/BiasAdd/ReadVariableOp ^conv5_enc/Conv2D/ReadVariableOp!^z_log_var/BiasAdd/ReadVariableOp ^z_log_var/MatMul/ReadVariableOp^z_mean/BiasAdd/ReadVariableOp^z_mean/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*n
_input_shapes]
[:?????????(::::::::::::::::2F
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
:?????????(
 
_user_specified_nameinputs
»
ή
E__inference_conv2_dec_layer_call_and_return_conditional_losses_149928

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
α

*__inference_z_log_var_layer_call_fn_151952

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallψ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
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
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
Ο

ή
E__inference_conv3_enc_layer_call_and_return_conditional_losses_149241

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
:?????????
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
:?????????
 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????
 2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????
 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
ω
`
D__inference_maxpool5_layer_call_and_return_conditional_losses_149164

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
ω
`
D__inference_maxpool1_layer_call_and_return_conditional_losses_149116

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
 
_user_specified_nameinputs"±L
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
serving_default_input_1:0?????????(D
output_18
StatefulPartitionedCall:0?????????(tensorflow/serving/predict:»
ύ
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
©_default_save_signature
ͺ__call__
+«&call_and_return_all_conditional_losses"½
_tf_keras_model£{"class_name": "VAEInternal", "name": "VAE", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "VAEInternal"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
γ
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
¬__call__
+­&call_and_return_all_conditional_losses"¬|
_tf_keras_network|{"class_name": "Functional", "name": "Encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_enc", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool5", "inbound_nodes": [[["conv5_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling", "trainable": true, "dtype": "float32"}, "name": "sampling", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 30, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}, "name": "input_encoder", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_enc", "inbound_nodes": [[["input_encoder", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool1", "inbound_nodes": [[["conv1_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_enc", "inbound_nodes": [[["maxpool1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool2", "inbound_nodes": [[["conv2_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_enc", "inbound_nodes": [[["maxpool2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool3", "inbound_nodes": [[["conv3_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_enc", "inbound_nodes": [[["maxpool3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool4", "inbound_nodes": [[["conv4_enc", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_enc", "inbound_nodes": [[["maxpool4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool5", "inbound_nodes": [[["conv5_enc", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["maxpool5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "bottleneck", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_mean", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z_log_var", "inbound_nodes": [[["bottleneck", 0, 0, {}]]]}, {"class_name": "Sampling", "config": {"name": "sampling", "trainable": true, "dtype": "float32"}, "name": "sampling", "inbound_nodes": [[["z_mean", 0, 0, {}], ["z_log_var", 0, 0, {}]]]}], "input_layers": [["input_encoder", 0, 0]], "output_layers": [["z_mean", 0, 0], ["z_log_var", 0, 0], ["sampling", 0, 0]]}}}
ιt
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
?__call__
+―&call_and_return_all_conditional_losses"θp
_tf_keras_networkΜp{"class_name": "Functional", "name": "Decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 1, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp5", "inbound_nodes": [[["conv5_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["upsamp5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [25, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 12]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}, "name": "input_decoder", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "decoding", "inbound_nodes": [[["input_decoder", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 1, 128]}}, "name": "reshape", "inbound_nodes": [[["decoding", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv5_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv5_dec", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp5", "inbound_nodes": [[["conv5_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv4_dec", "inbound_nodes": [[["upsamp5", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp4", "inbound_nodes": [[["conv4_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_dec", "inbound_nodes": [[["upsamp4", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp3", "inbound_nodes": [[["conv3_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_dec", "inbound_nodes": [[["upsamp3", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp2", "inbound_nodes": [[["conv2_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_dec", "inbound_nodes": [[["upsamp2", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "upsamp1", "inbound_nodes": [[["conv1_dec", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [25, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["upsamp1", 0, 0, {}]]]}], "input_layers": [["input_decoder", 0, 0]], "output_layers": [["output", 0, 0]]}}}
Η
	3total
	4count
5	variables
6	keras_api"
_tf_keras_metricv{"class_name": "Mean", "name": "total_loss", "dtype": "float32", "config": {"name": "total_loss", "dtype": "float32"}}
Ϊ
	7total
	8count
9	variables
:	keras_api"£
_tf_keras_metric{"class_name": "Mean", "name": "reconstruction_loss", "dtype": "float32", "config": {"name": "reconstruction_loss", "dtype": "float32"}}
Α
	;total
	<count
=	variables
>	keras_api"
_tf_keras_metricp{"class_name": "Mean", "name": "kl_loss", "dtype": "float32", "config": {"name": "kl_loss", "dtype": "float32"}}
«
?iter

@beta_1

Abeta_2
	Bdecay
Clearning_rateDmνEmξFmοGmπHmρImςJmσKmτLmυMmφNmχOmψPmωQmϊRmϋSmόTmύUmώVm?WmXmYmZm[m\m]m^m_m`mamDvEvFvGvHvIvJvKvLvMvNvOvPvQvRvSvTvUvVvWvXvYv Zv‘[v’\v£]v€^v₯_v¦`v§av¨"
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
Ά
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
Ξ
trainable_variables

blayers
		variables

regularization_losses
cnon_trainable_variables
dmetrics
elayer_regularization_losses
flayer_metrics
ͺ__call__
©_default_save_signature
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
-
°serving_default"
signature_map
"
_tf_keras_input_layerβ{"class_name": "InputLayer", "name": "input_encoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 30, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_encoder"}}
σ	

Dkernel
Ebias
gtrainable_variables
h	variables
iregularization_losses
j	keras_api
±__call__
+²&call_and_return_all_conditional_losses"Μ
_tf_keras_layer²{"class_name": "Conv2D", "name": "conv1_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_enc", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 30, 1]}}
ς
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
³__call__
+΄&call_and_return_all_conditional_losses"α
_tf_keras_layerΗ{"class_name": "MaxPooling2D", "name": "maxpool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
τ	

Fkernel
Gbias
otrainable_variables
p	variables
qregularization_losses
r	keras_api
΅__call__
+Ά&call_and_return_all_conditional_losses"Ν
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv2_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_enc", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 15, 8]}}
ς
strainable_variables
t	variables
uregularization_losses
v	keras_api
·__call__
+Έ&call_and_return_all_conditional_losses"α
_tf_keras_layerΗ{"class_name": "MaxPooling2D", "name": "maxpool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
υ	

Hkernel
Ibias
wtrainable_variables
x	variables
yregularization_losses
z	keras_api
Ή__call__
+Ί&call_and_return_all_conditional_losses"Ξ
_tf_keras_layer΄{"class_name": "Conv2D", "name": "conv3_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_enc", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 8, 16]}}
ς
{trainable_variables
|	variables
}regularization_losses
~	keras_api
»__call__
+Ό&call_and_return_all_conditional_losses"α
_tf_keras_layerΗ{"class_name": "MaxPooling2D", "name": "maxpool3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
χ	

Jkernel
Kbias
trainable_variables
	variables
regularization_losses
	keras_api
½__call__
+Ύ&call_and_return_all_conditional_losses"Ν
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv4_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_enc", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 4, 32]}}
φ
trainable_variables
	variables
regularization_losses
	keras_api
Ώ__call__
+ΐ&call_and_return_all_conditional_losses"α
_tf_keras_layerΗ{"class_name": "MaxPooling2D", "name": "maxpool4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ω	

Lkernel
Mbias
trainable_variables
	variables
regularization_losses
	keras_api
Α__call__
+Β&call_and_return_all_conditional_losses"Ξ
_tf_keras_layer΄{"class_name": "Conv2D", "name": "conv5_enc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv5_enc", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 2, 64]}}
φ
trainable_variables
	variables
regularization_losses
	keras_api
Γ__call__
+Δ&call_and_return_all_conditional_losses"α
_tf_keras_layerΗ{"class_name": "MaxPooling2D", "name": "maxpool5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
θ
trainable_variables
	variables
regularization_losses
	keras_api
Ε__call__
+Ζ&call_and_return_all_conditional_losses"Σ
_tf_keras_layerΉ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}


Nkernel
Obias
trainable_variables
	variables
regularization_losses
	keras_api
Η__call__
+Θ&call_and_return_all_conditional_losses"Υ
_tf_keras_layer»{"class_name": "Dense", "name": "bottleneck", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "bottleneck", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
φ

Pkernel
Qbias
trainable_variables
	variables
regularization_losses
	keras_api
Ι__call__
+Κ&call_and_return_all_conditional_losses"Λ
_tf_keras_layer±{"class_name": "Dense", "name": "z_mean", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_mean", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
ό

Rkernel
Sbias
trainable_variables
	variables
regularization_losses
	keras_api
Λ__call__
+Μ&call_and_return_all_conditional_losses"Ρ
_tf_keras_layer·{"class_name": "Dense", "name": "z_log_var", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "z_log_var", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
»
trainable_variables
 	variables
‘regularization_losses
’	keras_api
Ν__call__
+Ξ&call_and_return_all_conditional_losses"¦
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
΅
trainable_variables
£layers
	variables
regularization_losses
€non_trainable_variables
₯metrics
 ¦layer_regularization_losses
§layer_metrics
¬__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
χ"τ
_tf_keras_input_layerΤ{"class_name": "InputLayer", "name": "input_decoder", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 12]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_decoder"}}
ϋ

Tkernel
Ubias
¨trainable_variables
©	variables
ͺregularization_losses
«	keras_api
Ο__call__
+Π&call_and_return_all_conditional_losses"Π
_tf_keras_layerΆ{"class_name": "Dense", "name": "decoding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoding", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12]}}
ϋ
¬trainable_variables
­	variables
?regularization_losses
―	keras_api
Ρ__call__
+?&call_and_return_all_conditional_losses"ζ
_tf_keras_layerΜ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [2, 1, 128]}}}
ϋ	

Vkernel
Wbias
°trainable_variables
±	variables
²regularization_losses
³	keras_api
Σ__call__
+Τ&call_and_return_all_conditional_losses"Π
_tf_keras_layerΆ{"class_name": "Conv2D", "name": "conv5_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv5_dec", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2, 1, 128]}}
Ώ
΄trainable_variables
΅	variables
Άregularization_losses
·	keras_api
Υ__call__
+Φ&call_and_return_all_conditional_losses"ͺ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp5", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ϊ	

Xkernel
Ybias
Έtrainable_variables
Ή	variables
Ίregularization_losses
»	keras_api
Χ__call__
+Ψ&call_and_return_all_conditional_losses"Ο
_tf_keras_layer΅{"class_name": "Conv2D", "name": "conv4_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4_dec", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 2, 128]}}
Ώ
Όtrainable_variables
½	variables
Ύregularization_losses
Ώ	keras_api
Ω__call__
+Ϊ&call_and_return_all_conditional_losses"ͺ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ψ	

Zkernel
[bias
ΐtrainable_variables
Α	variables
Βregularization_losses
Γ	keras_api
Ϋ__call__
+ά&call_and_return_all_conditional_losses"Ν
_tf_keras_layer³{"class_name": "Conv2D", "name": "conv3_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3_dec", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 4, 64]}}
Ώ
Δtrainable_variables
Ε	variables
Ζregularization_losses
Η	keras_api
έ__call__
+ή&call_and_return_all_conditional_losses"ͺ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ω	

\kernel
]bias
Θtrainable_variables
Ι	variables
Κregularization_losses
Λ	keras_api
ί__call__
+ΰ&call_and_return_all_conditional_losses"Ξ
_tf_keras_layer΄{"class_name": "Conv2D", "name": "conv2_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2_dec", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 8, 32]}}
Ώ
Μtrainable_variables
Ν	variables
Ξregularization_losses
Ο	keras_api
α__call__
+β&call_and_return_all_conditional_losses"ͺ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ω	

^kernel
_bias
Πtrainable_variables
Ρ	variables
?regularization_losses
Σ	keras_api
γ__call__
+δ&call_and_return_all_conditional_losses"Ξ
_tf_keras_layer΄{"class_name": "Conv2D", "name": "conv1_dec", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1_dec", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 16, 16]}}
Ώ
Τtrainable_variables
Υ	variables
Φregularization_losses
Χ	keras_api
ε__call__
+ζ&call_and_return_all_conditional_losses"ͺ
_tf_keras_layer{"class_name": "UpSampling2D", "name": "upsamp1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "upsamp1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
φ	

`kernel
abias
Ψtrainable_variables
Ω	variables
Ϊregularization_losses
Ϋ	keras_api
η__call__
+θ&call_and_return_all_conditional_losses"Λ
_tf_keras_layer±{"class_name": "Conv2D", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [25, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 32, 8]}}
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
΅
/trainable_variables
άlayers
0	variables
1regularization_losses
έnon_trainable_variables
ήmetrics
 ίlayer_regularization_losses
ΰlayer_metrics
?__call__
+―&call_and_return_all_conditional_losses
'―"call_and_return_conditional_losses"
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
΅
gtrainable_variables
αlayers
h	variables
iregularization_losses
βnon_trainable_variables
γmetrics
 δlayer_regularization_losses
εlayer_metrics
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ktrainable_variables
ζlayers
l	variables
mregularization_losses
ηnon_trainable_variables
θmetrics
 ιlayer_regularization_losses
κlayer_metrics
³__call__
+΄&call_and_return_all_conditional_losses
'΄"call_and_return_conditional_losses"
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
otrainable_variables
λlayers
p	variables
qregularization_losses
μnon_trainable_variables
νmetrics
 ξlayer_regularization_losses
οlayer_metrics
΅__call__
+Ά&call_and_return_all_conditional_losses
'Ά"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
strainable_variables
πlayers
t	variables
uregularization_losses
ρnon_trainable_variables
ςmetrics
 σlayer_regularization_losses
τlayer_metrics
·__call__
+Έ&call_and_return_all_conditional_losses
'Έ"call_and_return_conditional_losses"
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
΅
wtrainable_variables
υlayers
x	variables
yregularization_losses
φnon_trainable_variables
χmetrics
 ψlayer_regularization_losses
ωlayer_metrics
Ή__call__
+Ί&call_and_return_all_conditional_losses
'Ί"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
{trainable_variables
ϊlayers
|	variables
}regularization_losses
ϋnon_trainable_variables
όmetrics
 ύlayer_regularization_losses
ώlayer_metrics
»__call__
+Ό&call_and_return_all_conditional_losses
'Ό"call_and_return_conditional_losses"
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
·
trainable_variables
?layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
½__call__
+Ύ&call_and_return_all_conditional_losses
'Ύ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Ώ__call__
+ΐ&call_and_return_all_conditional_losses
'ΐ"call_and_return_conditional_losses"
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
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Α__call__
+Β&call_and_return_all_conditional_losses
'Β"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Γ__call__
+Δ&call_and_return_all_conditional_losses
'Δ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Ε__call__
+Ζ&call_and_return_all_conditional_losses
'Ζ"call_and_return_conditional_losses"
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
Έ
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
 layer_regularization_losses
layer_metrics
Η__call__
+Θ&call_and_return_all_conditional_losses
'Θ"call_and_return_conditional_losses"
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
trainable_variables
layers
	variables
regularization_losses
non_trainable_variables
metrics
  layer_regularization_losses
‘layer_metrics
Ι__call__
+Κ&call_and_return_all_conditional_losses
'Κ"call_and_return_conditional_losses"
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
trainable_variables
’layers
	variables
regularization_losses
£non_trainable_variables
€metrics
 ₯layer_regularization_losses
¦layer_metrics
Λ__call__
+Μ&call_and_return_all_conditional_losses
'Μ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
trainable_variables
§layers
 	variables
‘regularization_losses
¨non_trainable_variables
©metrics
 ͺlayer_regularization_losses
«layer_metrics
Ν__call__
+Ξ&call_and_return_all_conditional_losses
'Ξ"call_and_return_conditional_losses"
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
Έ
¨trainable_variables
¬layers
©	variables
ͺregularization_losses
­non_trainable_variables
?metrics
 ―layer_regularization_losses
°layer_metrics
Ο__call__
+Π&call_and_return_all_conditional_losses
'Π"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
¬trainable_variables
±layers
­	variables
?regularization_losses
²non_trainable_variables
³metrics
 ΄layer_regularization_losses
΅layer_metrics
Ρ__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
°trainable_variables
Άlayers
±	variables
²regularization_losses
·non_trainable_variables
Έmetrics
 Ήlayer_regularization_losses
Ίlayer_metrics
Σ__call__
+Τ&call_and_return_all_conditional_losses
'Τ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
΄trainable_variables
»layers
΅	variables
Άregularization_losses
Όnon_trainable_variables
½metrics
 Ύlayer_regularization_losses
Ώlayer_metrics
Υ__call__
+Φ&call_and_return_all_conditional_losses
'Φ"call_and_return_conditional_losses"
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
Έtrainable_variables
ΐlayers
Ή	variables
Ίregularization_losses
Αnon_trainable_variables
Βmetrics
 Γlayer_regularization_losses
Δlayer_metrics
Χ__call__
+Ψ&call_and_return_all_conditional_losses
'Ψ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Όtrainable_variables
Εlayers
½	variables
Ύregularization_losses
Ζnon_trainable_variables
Ηmetrics
 Θlayer_regularization_losses
Ιlayer_metrics
Ω__call__
+Ϊ&call_and_return_all_conditional_losses
'Ϊ"call_and_return_conditional_losses"
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
Έ
ΐtrainable_variables
Κlayers
Α	variables
Βregularization_losses
Λnon_trainable_variables
Μmetrics
 Νlayer_regularization_losses
Ξlayer_metrics
Ϋ__call__
+ά&call_and_return_all_conditional_losses
'ά"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Δtrainable_variables
Οlayers
Ε	variables
Ζregularization_losses
Πnon_trainable_variables
Ρmetrics
 ?layer_regularization_losses
Σlayer_metrics
έ__call__
+ή&call_and_return_all_conditional_losses
'ή"call_and_return_conditional_losses"
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
Έ
Θtrainable_variables
Τlayers
Ι	variables
Κregularization_losses
Υnon_trainable_variables
Φmetrics
 Χlayer_regularization_losses
Ψlayer_metrics
ί__call__
+ΰ&call_and_return_all_conditional_losses
'ΰ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Μtrainable_variables
Ωlayers
Ν	variables
Ξregularization_losses
Ϊnon_trainable_variables
Ϋmetrics
 άlayer_regularization_losses
έlayer_metrics
α__call__
+β&call_and_return_all_conditional_losses
'β"call_and_return_conditional_losses"
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
Έ
Πtrainable_variables
ήlayers
Ρ	variables
?regularization_losses
ίnon_trainable_variables
ΰmetrics
 αlayer_regularization_losses
βlayer_metrics
γ__call__
+δ&call_and_return_all_conditional_losses
'δ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Τtrainable_variables
γlayers
Υ	variables
Φregularization_losses
δnon_trainable_variables
εmetrics
 ζlayer_regularization_losses
ηlayer_metrics
ε__call__
+ζ&call_and_return_all_conditional_losses
'ζ"call_and_return_conditional_losses"
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
Έ
Ψtrainable_variables
θlayers
Ω	variables
Ϊregularization_losses
ιnon_trainable_variables
κmetrics
 λlayer_regularization_losses
μlayer_metrics
η__call__
+θ&call_and_return_all_conditional_losses
'θ"call_and_return_conditional_losses"
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
η2δ
!__inference__wrapped_model_149110Ύ
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
input_1?????????(
Ρ2Ξ
$__inference_VAE_layer_call_fn_151193
$__inference_VAE_layer_call_fn_150622
$__inference_VAE_layer_call_fn_150687
$__inference_VAE_layer_call_fn_151258³
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
½2Ί
?__inference_VAE_layer_call_and_return_conditional_losses_151128
?__inference_VAE_layer_call_and_return_conditional_losses_150945
?__inference_VAE_layer_call_and_return_conditional_losses_150420
?__inference_VAE_layer_call_and_return_conditional_losses_150488³
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
ξ2λ
(__inference_Encoder_layer_call_fn_149686
(__inference_Encoder_layer_call_fn_151473
(__inference_Encoder_layer_call_fn_151514
(__inference_Encoder_layer_call_fn_149592ΐ
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
Ϊ2Χ
C__inference_Encoder_layer_call_and_return_conditional_losses_149444
C__inference_Encoder_layer_call_and_return_conditional_losses_151345
C__inference_Encoder_layer_call_and_return_conditional_losses_151432
C__inference_Encoder_layer_call_and_return_conditional_losses_149497ΐ
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
ξ2λ
(__inference_Decoder_layer_call_fn_151751
(__inference_Decoder_layer_call_fn_150203
(__inference_Decoder_layer_call_fn_151784
(__inference_Decoder_layer_call_fn_150125ΐ
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
Ϊ2Χ
C__inference_Decoder_layer_call_and_return_conditional_losses_151616
C__inference_Decoder_layer_call_and_return_conditional_losses_151718
C__inference_Decoder_layer_call_and_return_conditional_losses_150046
C__inference_Decoder_layer_call_and_return_conditional_losses_150001ΐ
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
ΛBΘ
$__inference_signature_wrapper_150762input_1"
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
Τ2Ρ
*__inference_conv1_enc_layer_call_fn_151804’
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
E__inference_conv1_enc_layer_call_and_return_conditional_losses_151795’
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
2
)__inference_maxpool1_layer_call_fn_149122ΰ
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
¬2©
D__inference_maxpool1_layer_call_and_return_conditional_losses_149116ΰ
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
Τ2Ρ
*__inference_conv2_enc_layer_call_fn_151824’
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
E__inference_conv2_enc_layer_call_and_return_conditional_losses_151815’
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
2
)__inference_maxpool2_layer_call_fn_149134ΰ
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
¬2©
D__inference_maxpool2_layer_call_and_return_conditional_losses_149128ΰ
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
Τ2Ρ
*__inference_conv3_enc_layer_call_fn_151844’
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
E__inference_conv3_enc_layer_call_and_return_conditional_losses_151835’
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
2
)__inference_maxpool3_layer_call_fn_149146ΰ
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
¬2©
D__inference_maxpool3_layer_call_and_return_conditional_losses_149140ΰ
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
Τ2Ρ
*__inference_conv4_enc_layer_call_fn_151864’
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
E__inference_conv4_enc_layer_call_and_return_conditional_losses_151855’
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
2
)__inference_maxpool4_layer_call_fn_149158ΰ
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
¬2©
D__inference_maxpool4_layer_call_and_return_conditional_losses_149152ΰ
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
Τ2Ρ
*__inference_conv5_enc_layer_call_fn_151884’
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
E__inference_conv5_enc_layer_call_and_return_conditional_losses_151875’
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
2
)__inference_maxpool5_layer_call_fn_149170ΰ
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
¬2©
D__inference_maxpool5_layer_call_and_return_conditional_losses_149164ΰ
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
?2Ο
(__inference_flatten_layer_call_fn_151895’
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
C__inference_flatten_layer_call_and_return_conditional_losses_151890’
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
+__inference_bottleneck_layer_call_fn_151914’
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
F__inference_bottleneck_layer_call_and_return_conditional_losses_151905’
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
Ρ2Ξ
'__inference_z_mean_layer_call_fn_151933’
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
μ2ι
B__inference_z_mean_layer_call_and_return_conditional_losses_151924’
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
*__inference_z_log_var_layer_call_fn_151952’
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
E__inference_z_log_var_layer_call_and_return_conditional_losses_151943’
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
)__inference_sampling_layer_call_fn_151984’
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
D__inference_sampling_layer_call_and_return_conditional_losses_151978’
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
)__inference_decoding_layer_call_fn_152003’
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
D__inference_decoding_layer_call_and_return_conditional_losses_151994’
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
(__inference_reshape_layer_call_fn_152022’
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
C__inference_reshape_layer_call_and_return_conditional_losses_152017’
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
*__inference_conv5_dec_layer_call_fn_152042’
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
E__inference_conv5_dec_layer_call_and_return_conditional_losses_152033’
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
2
(__inference_upsamp5_layer_call_fn_149705ΰ
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
«2¨
C__inference_upsamp5_layer_call_and_return_conditional_losses_149699ΰ
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
Τ2Ρ
*__inference_conv4_dec_layer_call_fn_152062’
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
E__inference_conv4_dec_layer_call_and_return_conditional_losses_152053’
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
2
(__inference_upsamp4_layer_call_fn_149724ΰ
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
«2¨
C__inference_upsamp4_layer_call_and_return_conditional_losses_149718ΰ
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
Τ2Ρ
*__inference_conv3_dec_layer_call_fn_152082’
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
E__inference_conv3_dec_layer_call_and_return_conditional_losses_152073’
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
2
(__inference_upsamp3_layer_call_fn_149743ΰ
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
«2¨
C__inference_upsamp3_layer_call_and_return_conditional_losses_149737ΰ
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
Τ2Ρ
*__inference_conv2_dec_layer_call_fn_152102’
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
E__inference_conv2_dec_layer_call_and_return_conditional_losses_152093’
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
2
(__inference_upsamp2_layer_call_fn_149762ΰ
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
«2¨
C__inference_upsamp2_layer_call_and_return_conditional_losses_149756ΰ
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
Τ2Ρ
*__inference_conv1_dec_layer_call_fn_152122’
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
E__inference_conv1_dec_layer_call_and_return_conditional_losses_152113’
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
2
(__inference_upsamp1_layer_call_fn_149781ΰ
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
«2¨
C__inference_upsamp1_layer_call_and_return_conditional_losses_149775ΰ
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
Ρ2Ξ
'__inference_output_layer_call_fn_152142’
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
μ2ι
B__inference_output_layer_call_and_return_conditional_losses_152133’
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
 Ω
C__inference_Decoder_layer_call_and_return_conditional_losses_150001TUVWXYZ[\]^_`a>’;
4’1
'$
input_decoder?????????
p

 
ͺ "?’<
52
0+???????????????????????????
 Ω
C__inference_Decoder_layer_call_and_return_conditional_losses_150046TUVWXYZ[\]^_`a>’;
4’1
'$
input_decoder?????????
p 

 
ͺ "?’<
52
0+???????????????????????????
 Ώ
C__inference_Decoder_layer_call_and_return_conditional_losses_151616xTUVWXYZ[\]^_`a7’4
-’*
 
inputs?????????
p

 
ͺ "-’*
# 
0?????????(
 Ώ
C__inference_Decoder_layer_call_and_return_conditional_losses_151718xTUVWXYZ[\]^_`a7’4
-’*
 
inputs?????????
p 

 
ͺ "-’*
# 
0?????????(
 ±
(__inference_Decoder_layer_call_fn_150125TUVWXYZ[\]^_`a>’;
4’1
'$
input_decoder?????????
p

 
ͺ "2/+???????????????????????????±
(__inference_Decoder_layer_call_fn_150203TUVWXYZ[\]^_`a>’;
4’1
'$
input_decoder?????????
p 

 
ͺ "2/+???????????????????????????©
(__inference_Decoder_layer_call_fn_151751}TUVWXYZ[\]^_`a7’4
-’*
 
inputs?????????
p

 
ͺ "2/+???????????????????????????©
(__inference_Decoder_layer_call_fn_151784}TUVWXYZ[\]^_`a7’4
-’*
 
inputs?????????
p 

 
ͺ "2/+???????????????????????????
C__inference_Encoder_layer_call_and_return_conditional_losses_149444ΖDEFGHIJKLMNOPQRSF’C
<’9
/,
input_encoder?????????(
p

 
ͺ "j’g
`]

0/0?????????

0/1?????????

0/2?????????
 
C__inference_Encoder_layer_call_and_return_conditional_losses_149497ΖDEFGHIJKLMNOPQRSF’C
<’9
/,
input_encoder?????????(
p 

 
ͺ "j’g
`]

0/0?????????

0/1?????????

0/2?????????
 
C__inference_Encoder_layer_call_and_return_conditional_losses_151345ΏDEFGHIJKLMNOPQRS?’<
5’2
(%
inputs?????????(
p

 
ͺ "j’g
`]

0/0?????????

0/1?????????

0/2?????????
 
C__inference_Encoder_layer_call_and_return_conditional_losses_151432ΏDEFGHIJKLMNOPQRS?’<
5’2
(%
inputs?????????(
p 

 
ͺ "j’g
`]

0/0?????????

0/1?????????

0/2?????????
 γ
(__inference_Encoder_layer_call_fn_149592ΆDEFGHIJKLMNOPQRSF’C
<’9
/,
input_encoder?????????(
p

 
ͺ "ZW

0?????????

1?????????

2?????????γ
(__inference_Encoder_layer_call_fn_149686ΆDEFGHIJKLMNOPQRSF’C
<’9
/,
input_encoder?????????(
p 

 
ͺ "ZW

0?????????

1?????????

2?????????ά
(__inference_Encoder_layer_call_fn_151473―DEFGHIJKLMNOPQRS?’<
5’2
(%
inputs?????????(
p

 
ͺ "ZW

0?????????

1?????????

2?????????ά
(__inference_Encoder_layer_call_fn_151514―DEFGHIJKLMNOPQRS?’<
5’2
(%
inputs?????????(
p 

 
ͺ "ZW

0?????????

1?????????

2?????????γ
?__inference_VAE_layer_call_and_return_conditional_losses_150420DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a<’9
2’/
)&
input_1?????????(
p
ͺ "?’<
52
0+???????????????????????????
 γ
?__inference_VAE_layer_call_and_return_conditional_losses_150488DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a<’9
2’/
)&
input_1?????????(
p 
ͺ "?’<
52
0+???????????????????????????
 Π
?__inference_VAE_layer_call_and_return_conditional_losses_150945DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a;’8
1’.
(%
inputs?????????(
p
ͺ "-’*
# 
0?????????(
 Π
?__inference_VAE_layer_call_and_return_conditional_losses_151128DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a;’8
1’.
(%
inputs?????????(
p 
ͺ "-’*
# 
0?????????(
 »
$__inference_VAE_layer_call_fn_150622DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a<’9
2’/
)&
input_1?????????(
p
ͺ "2/+???????????????????????????»
$__inference_VAE_layer_call_fn_150687DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a<’9
2’/
)&
input_1?????????(
p 
ͺ "2/+???????????????????????????Ί
$__inference_VAE_layer_call_fn_151193DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a;’8
1’.
(%
inputs?????????(
p
ͺ "2/+???????????????????????????Ί
$__inference_VAE_layer_call_fn_151258DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a;’8
1’.
(%
inputs?????????(
p 
ͺ "2/+???????????????????????????½
!__inference__wrapped_model_149110DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`a8’5
.’+
)&
input_1?????????(
ͺ ";ͺ8
6
output_1*'
output_1?????????(§
F__inference_bottleneck_layer_call_and_return_conditional_losses_151905]NO0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????
 
+__inference_bottleneck_layer_call_fn_151914PNO0’-
&’#
!
inputs?????????
ͺ "?????????Ϊ
E__inference_conv1_dec_layer_call_and_return_conditional_losses_152113^_I’F
?’<
:7
inputs+???????????????????????????
ͺ "?’<
52
0+???????????????????????????
 ²
*__inference_conv1_dec_layer_call_fn_152122^_I’F
?’<
:7
inputs+???????????????????????????
ͺ "2/+???????????????????????????΅
E__inference_conv1_enc_layer_call_and_return_conditional_losses_151795lDE7’4
-’*
(%
inputs?????????(
ͺ "-’*
# 
0?????????(
 
*__inference_conv1_enc_layer_call_fn_151804_DE7’4
-’*
(%
inputs?????????(
ͺ " ?????????(Ϊ
E__inference_conv2_dec_layer_call_and_return_conditional_losses_152093\]I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "?’<
52
0+???????????????????????????
 ²
*__inference_conv2_dec_layer_call_fn_152102\]I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "2/+???????????????????????????΅
E__inference_conv2_enc_layer_call_and_return_conditional_losses_151815lFG7’4
-’*
(%
inputs?????????
ͺ "-’*
# 
0?????????
 
*__inference_conv2_enc_layer_call_fn_151824_FG7’4
-’*
(%
inputs?????????
ͺ " ?????????Ϊ
E__inference_conv3_dec_layer_call_and_return_conditional_losses_152073Z[I’F
?’<
:7
inputs+???????????????????????????@
ͺ "?’<
52
0+??????????????????????????? 
 ²
*__inference_conv3_dec_layer_call_fn_152082Z[I’F
?’<
:7
inputs+???????????????????????????@
ͺ "2/+??????????????????????????? ΅
E__inference_conv3_enc_layer_call_and_return_conditional_losses_151835lHI7’4
-’*
(%
inputs?????????

ͺ "-’*
# 
0?????????
 
 
*__inference_conv3_enc_layer_call_fn_151844_HI7’4
-’*
(%
inputs?????????

ͺ " ?????????
 Ϋ
E__inference_conv4_dec_layer_call_and_return_conditional_losses_152053XYJ’G
@’=
;8
inputs,???????????????????????????
ͺ "?’<
52
0+???????????????????????????@
 ³
*__inference_conv4_dec_layer_call_fn_152062XYJ’G
@’=
;8
inputs,???????????????????????????
ͺ "2/+???????????????????????????@΅
E__inference_conv4_enc_layer_call_and_return_conditional_losses_151855lJK7’4
-’*
(%
inputs????????? 
ͺ "-’*
# 
0?????????@
 
*__inference_conv4_enc_layer_call_fn_151864_JK7’4
-’*
(%
inputs????????? 
ͺ " ?????????@·
E__inference_conv5_dec_layer_call_and_return_conditional_losses_152033nVW8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
*__inference_conv5_dec_layer_call_fn_152042aVW8’5
.’+
)&
inputs?????????
ͺ "!?????????Ά
E__inference_conv5_enc_layer_call_and_return_conditional_losses_151875mLM7’4
-’*
(%
inputs?????????@
ͺ ".’+
$!
0?????????
 
*__inference_conv5_enc_layer_call_fn_151884`LM7’4
-’*
(%
inputs?????????@
ͺ "!?????????₯
D__inference_decoding_layer_call_and_return_conditional_losses_151994]TU/’,
%’"
 
inputs?????????
ͺ "&’#

0?????????
 }
)__inference_decoding_layer_call_fn_152003PTU/’,
%’"
 
inputs?????????
ͺ "?????????©
C__inference_flatten_layer_call_and_return_conditional_losses_151890b8’5
.’+
)&
inputs?????????
ͺ "&’#

0?????????
 
(__inference_flatten_layer_call_fn_151895U8’5
.’+
)&
inputs?????????
ͺ "?????????η
D__inference_maxpool1_layer_call_and_return_conditional_losses_149116R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ώ
)__inference_maxpool1_layer_call_fn_149122R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????η
D__inference_maxpool2_layer_call_and_return_conditional_losses_149128R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ώ
)__inference_maxpool2_layer_call_fn_149134R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????η
D__inference_maxpool3_layer_call_and_return_conditional_losses_149140R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ώ
)__inference_maxpool3_layer_call_fn_149146R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????η
D__inference_maxpool4_layer_call_and_return_conditional_losses_149152R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ώ
)__inference_maxpool4_layer_call_fn_149158R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????η
D__inference_maxpool5_layer_call_and_return_conditional_losses_149164R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ώ
)__inference_maxpool5_layer_call_fn_149170R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????Χ
B__inference_output_layer_call_and_return_conditional_losses_152133`aI’F
?’<
:7
inputs+???????????????????????????
ͺ "?’<
52
0+???????????????????????????
 ―
'__inference_output_layer_call_fn_152142`aI’F
?’<
:7
inputs+???????????????????????????
ͺ "2/+???????????????????????????©
C__inference_reshape_layer_call_and_return_conditional_losses_152017b0’-
&’#
!
inputs?????????
ͺ ".’+
$!
0?????????
 
(__inference_reshape_layer_call_fn_152022U0’-
&’#
!
inputs?????????
ͺ "!?????????Μ
D__inference_sampling_layer_call_and_return_conditional_losses_151978Z’W
P’M
KH
"
inputs/0?????????
"
inputs/1?????????
ͺ "%’"

0?????????
 £
)__inference_sampling_layer_call_fn_151984vZ’W
P’M
KH
"
inputs/0?????????
"
inputs/1?????????
ͺ "?????????Λ
$__inference_signature_wrapper_150762’DEFGHIJKLMNOPQRSTUVWXYZ[\]^_`aC’@
’ 
9ͺ6
4
input_1)&
input_1?????????(";ͺ8
6
output_1*'
output_1?????????(ζ
C__inference_upsamp1_layer_call_and_return_conditional_losses_149775R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ύ
(__inference_upsamp1_layer_call_fn_149781R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ζ
C__inference_upsamp2_layer_call_and_return_conditional_losses_149756R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ύ
(__inference_upsamp2_layer_call_fn_149762R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ζ
C__inference_upsamp3_layer_call_and_return_conditional_losses_149737R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ύ
(__inference_upsamp3_layer_call_fn_149743R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ζ
C__inference_upsamp4_layer_call_and_return_conditional_losses_149718R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ύ
(__inference_upsamp4_layer_call_fn_149724R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ζ
C__inference_upsamp5_layer_call_and_return_conditional_losses_149699R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Ύ
(__inference_upsamp5_layer_call_fn_149705R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????₯
E__inference_z_log_var_layer_call_and_return_conditional_losses_151943\RS/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 }
*__inference_z_log_var_layer_call_fn_151952ORS/’,
%’"
 
inputs?????????
ͺ "?????????’
B__inference_z_mean_layer_call_and_return_conditional_losses_151924\PQ/’,
%’"
 
inputs?????????
ͺ "%’"

0?????????
 z
'__inference_z_mean_layer_call_fn_151933OPQ/’,
%’"
 
inputs?????????
ͺ "?????????