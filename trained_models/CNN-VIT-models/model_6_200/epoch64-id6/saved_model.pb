��8
�"�"
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
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
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
*
Erf
x"T
y"T"
Ttype:
2
�
ExtractImagePatches
images"T
patches"T"
ksizes	list(int)(0"
strides	list(int)(0"
rates	list(int)(0"
Ttype:
2	
""
paddingstring:
SAMEVALID
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
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
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
d
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:

2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	�
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
�
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.5.02v2.5.0-0-ga4dfb8d1a718�0
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
:
*
dtype0
n
conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv_1/bias
g
conv_1/bias/Read/ReadVariableOpReadVariableOpconv_1/bias*
_output_shapes
:
*
dtype0
~
conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*
shared_nameconv_2/kernel
w
!conv_2/kernel/Read/ReadVariableOpReadVariableOpconv_2/kernel*&
_output_shapes
:

*
dtype0
n
conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv_2/bias
g
conv_2/bias/Read/ReadVariableOpReadVariableOpconv_2/bias*
_output_shapes
:
*
dtype0
~
conv_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv_3/kernel
w
!conv_3/kernel/Read/ReadVariableOpReadVariableOpconv_3/kernel*&
_output_shapes
:
*
dtype0
n
conv_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_3/bias
g
conv_3/bias/Read/ReadVariableOpReadVariableOpconv_3/bias*
_output_shapes
:*
dtype0
~
conv_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4/kernel
w
!conv_4/kernel/Read/ReadVariableOpReadVariableOpconv_4/kernel*&
_output_shapes
:*
dtype0
n
conv_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_4/bias
g
conv_4/bias/Read/ReadVariableOpReadVariableOpconv_4/bias*
_output_shapes
:*
dtype0
�
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelayer_normalization/gamma
�
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:@*
dtype0
�
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer_normalization/beta
�
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:@*
dtype0
�
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_1/gamma
�
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:@*
dtype0
�
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_1/beta
�
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	@�*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:�*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	�@*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0
�
layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_2/gamma
�
/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
:@*
dtype0
�
layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_2/beta
�
.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
:@*
dtype0
�
layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_3/gamma
�
/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
:@*
dtype0
�
layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_3/beta
�
.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
:@*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	@�*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:�*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	�@*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0
�
layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_4/gamma
�
/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
:@*
dtype0
�
layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_4/beta
�
.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes
:@*
dtype0
s
FC_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�(2*
shared_nameFC_1/kernel
l
FC_1/kernel/Read/ReadVariableOpReadVariableOpFC_1/kernel*
_output_shapes
:	�(2*
dtype0
j
	FC_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_name	FC_1/bias
c
FC_1/bias/Read/ReadVariableOpReadVariableOp	FC_1/bias*
_output_shapes
:2*
dtype0
r
FC_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*
shared_nameFC_2/kernel
k
FC_2/kernel/Read/ReadVariableOpReadVariableOpFC_2/kernel*
_output_shapes

:22*
dtype0
j
	FC_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_name	FC_2/bias
c
FC_2/bias/Read/ReadVariableOpReadVariableOp	FC_2/bias*
_output_shapes
:2*
dtype0
�
output_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*$
shared_nameoutput_layer/kernel
{
'output_layer/kernel/Read/ReadVariableOpReadVariableOpoutput_layer/kernel*
_output_shapes

:2*
dtype0
z
output_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameoutput_layer/bias
s
%output_layer/bias/Read/ReadVariableOpReadVariableOpoutput_layer/bias*
_output_shapes
:*
dtype0
h

AdamW/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
AdamW/iter
a
AdamW/iter/Read/ReadVariableOpReadVariableOp
AdamW/iter*
_output_shapes
: *
dtype0	
l
AdamW/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamW/beta_1
e
 AdamW/beta_1/Read/ReadVariableOpReadVariableOpAdamW/beta_1*
_output_shapes
: *
dtype0
l
AdamW/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamW/beta_2
e
 AdamW/beta_2/Read/ReadVariableOpReadVariableOpAdamW/beta_2*
_output_shapes
: *
dtype0
j
AdamW/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdamW/decay
c
AdamW/decay/Read/ReadVariableOpReadVariableOpAdamW/decay*
_output_shapes
: *
dtype0
z
AdamW/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdamW/learning_rate
s
'AdamW/learning_rate/Read/ReadVariableOpReadVariableOpAdamW/learning_rate*
_output_shapes
: *
dtype0
x
AdamW/weight_decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdamW/weight_decay
q
&AdamW/weight_decay/Read/ReadVariableOpReadVariableOpAdamW/weight_decay*
_output_shapes
: *
dtype0
�
patch_encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*+
shared_namepatch_encoder/dense/kernel
�
.patch_encoder/dense/kernel/Read/ReadVariableOpReadVariableOppatch_encoder/dense/kernel*
_output_shapes
:	�@*
dtype0
�
patch_encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namepatch_encoder/dense/bias
�
,patch_encoder/dense/bias/Read/ReadVariableOpReadVariableOppatch_encoder/dense/bias*
_output_shapes
:@*
dtype0
�
"patch_encoder/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q@*3
shared_name$"patch_encoder/embedding/embeddings
�
6patch_encoder/embedding/embeddings/Read/ReadVariableOpReadVariableOp"patch_encoder/embedding/embeddings*
_output_shapes

:Q@*
dtype0
�
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention/query/kernel
�
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:@@*
dtype0
�
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention/query/bias
�
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:@*
dtype0
�
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!multi_head_attention/key/kernel
�
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:@@*
dtype0
�
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_namemulti_head_attention/key/bias
�
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:@*
dtype0
�
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention/value/kernel
�
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:@@*
dtype0
�
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention/value/bias
�
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:@*
dtype0
�
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*=
shared_name.,multi_head_attention/attention_output/kernel
�
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:@@*
dtype0
�
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*multi_head_attention/attention_output/bias
�
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:@*
dtype0
�
#multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_1/query/kernel
�
7multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/query/kernel*"
_output_shapes
:@@*
dtype0
�
!multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/query/bias
�
5multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/query/bias*
_output_shapes

:@*
dtype0
�
!multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention_1/key/kernel
�
5multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/key/kernel*"
_output_shapes
:@@*
dtype0
�
multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention_1/key/bias
�
3multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_1/key/bias*
_output_shapes

:@*
dtype0
�
#multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_1/value/kernel
�
7multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/value/kernel*"
_output_shapes
:@@*
dtype0
�
!multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/value/bias
�
5multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/value/bias*
_output_shapes

:@*
dtype0
�
.multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.multi_head_attention_1/attention_output/kernel
�
Bmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_1/attention_output/kernel*"
_output_shapes
:@@*
dtype0
�
,multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,multi_head_attention_1/attention_output/bias
�
@multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_1/attention_output/bias*
_output_shapes
:@*
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
�
AdamW/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_1/kernel/m
�
)AdamW/conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/conv_1/kernel/m*&
_output_shapes
:
*
dtype0
~
AdamW/conv_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdamW/conv_1/bias/m
w
'AdamW/conv_1/bias/m/Read/ReadVariableOpReadVariableOpAdamW/conv_1/bias/m*
_output_shapes
:
*
dtype0
�
AdamW/conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*&
shared_nameAdamW/conv_2/kernel/m
�
)AdamW/conv_2/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/conv_2/kernel/m*&
_output_shapes
:

*
dtype0
~
AdamW/conv_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdamW/conv_2/bias/m
w
'AdamW/conv_2/bias/m/Read/ReadVariableOpReadVariableOpAdamW/conv_2/bias/m*
_output_shapes
:
*
dtype0
�
AdamW/conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_3/kernel/m
�
)AdamW/conv_3/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/conv_3/kernel/m*&
_output_shapes
:
*
dtype0
~
AdamW/conv_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamW/conv_3/bias/m
w
'AdamW/conv_3/bias/m/Read/ReadVariableOpReadVariableOpAdamW/conv_3/bias/m*
_output_shapes
:*
dtype0
�
AdamW/conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_4/kernel/m
�
)AdamW/conv_4/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/conv_4/kernel/m*&
_output_shapes
:*
dtype0
~
AdamW/conv_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamW/conv_4/bias/m
w
'AdamW/conv_4/bias/m/Read/ReadVariableOpReadVariableOpAdamW/conv_4/bias/m*
_output_shapes
:*
dtype0
�
!AdamW/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!AdamW/layer_normalization/gamma/m
�
5AdamW/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp!AdamW/layer_normalization/gamma/m*
_output_shapes
:@*
dtype0
�
 AdamW/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/layer_normalization/beta/m
�
4AdamW/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp AdamW/layer_normalization/beta/m*
_output_shapes
:@*
dtype0
�
#AdamW/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_1/gamma/m
�
7AdamW/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
�
"AdamW/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_1/beta/m
�
6AdamW/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_1/beta/m*
_output_shapes
:@*
dtype0
�
AdamW/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*'
shared_nameAdamW/dense_1/kernel/m
�
*AdamW/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_1/kernel/m*
_output_shapes
:	@�*
dtype0
�
AdamW/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdamW/dense_1/bias/m
z
(AdamW/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_1/bias/m*
_output_shapes	
:�*
dtype0
�
AdamW/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdamW/dense_2/kernel/m
�
*AdamW/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_2/kernel/m*
_output_shapes
:	�@*
dtype0
�
AdamW/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdamW/dense_2/bias/m
y
(AdamW/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_2/bias/m*
_output_shapes
:@*
dtype0
�
#AdamW/layer_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_2/gamma/m
�
7AdamW/layer_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
�
"AdamW/layer_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_2/beta/m
�
6AdamW/layer_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_2/beta/m*
_output_shapes
:@*
dtype0
�
#AdamW/layer_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_3/gamma/m
�
7AdamW/layer_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_3/gamma/m*
_output_shapes
:@*
dtype0
�
"AdamW/layer_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_3/beta/m
�
6AdamW/layer_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_3/beta/m*
_output_shapes
:@*
dtype0
�
AdamW/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*'
shared_nameAdamW/dense_3/kernel/m
�
*AdamW/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_3/kernel/m*
_output_shapes
:	@�*
dtype0
�
AdamW/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdamW/dense_3/bias/m
z
(AdamW/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_3/bias/m*
_output_shapes	
:�*
dtype0
�
AdamW/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdamW/dense_4/kernel/m
�
*AdamW/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_4/kernel/m*
_output_shapes
:	�@*
dtype0
�
AdamW/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdamW/dense_4/bias/m
y
(AdamW/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_4/bias/m*
_output_shapes
:@*
dtype0
�
#AdamW/layer_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_4/gamma/m
�
7AdamW/layer_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_4/gamma/m*
_output_shapes
:@*
dtype0
�
"AdamW/layer_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_4/beta/m
�
6AdamW/layer_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_4/beta/m*
_output_shapes
:@*
dtype0
�
AdamW/FC_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�(2*$
shared_nameAdamW/FC_1/kernel/m
|
'AdamW/FC_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/m*
_output_shapes
:	�(2*
dtype0
z
AdamW/FC_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameAdamW/FC_1/bias/m
s
%AdamW/FC_1/bias/m/Read/ReadVariableOpReadVariableOpAdamW/FC_1/bias/m*
_output_shapes
:2*
dtype0
�
AdamW/FC_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*$
shared_nameAdamW/FC_2/kernel/m
{
'AdamW/FC_2/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/FC_2/kernel/m*
_output_shapes

:22*
dtype0
z
AdamW/FC_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameAdamW/FC_2/bias/m
s
%AdamW/FC_2/bias/m/Read/ReadVariableOpReadVariableOpAdamW/FC_2/bias/m*
_output_shapes
:2*
dtype0
�
AdamW/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_nameAdamW/output_layer/kernel/m
�
/AdamW/output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/output_layer/kernel/m*
_output_shapes

:2*
dtype0
�
AdamW/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdamW/output_layer/bias/m
�
-AdamW/output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdamW/output_layer/bias/m*
_output_shapes
:*
dtype0
�
"AdamW/patch_encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*3
shared_name$"AdamW/patch_encoder/dense/kernel/m
�
6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/m*
_output_shapes
:	�@*
dtype0
�
 AdamW/patch_encoder/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/patch_encoder/dense/bias/m
�
4AdamW/patch_encoder/dense/bias/m/Read/ReadVariableOpReadVariableOp AdamW/patch_encoder/dense/bias/m*
_output_shapes
:@*
dtype0
�
*AdamW/patch_encoder/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/m
�
>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/m*
_output_shapes

:Q@*
dtype0
�
)AdamW/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/query/kernel/m
�
=AdamW/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/query/kernel/m*"
_output_shapes
:@@*
dtype0
�
'AdamW/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/query/bias/m
�
;AdamW/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/query/bias/m*
_output_shapes

:@*
dtype0
�
'AdamW/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'AdamW/multi_head_attention/key/kernel/m
�
;AdamW/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/key/kernel/m*"
_output_shapes
:@@*
dtype0
�
%AdamW/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%AdamW/multi_head_attention/key/bias/m
�
9AdamW/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp%AdamW/multi_head_attention/key/bias/m*
_output_shapes

:@*
dtype0
�
)AdamW/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/value/kernel/m
�
=AdamW/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/value/kernel/m*"
_output_shapes
:@@*
dtype0
�
'AdamW/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/value/bias/m
�
;AdamW/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/value/bias/m*
_output_shapes

:@*
dtype0
�
4AdamW/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*E
shared_name64AdamW/multi_head_attention/attention_output/kernel/m
�
HAdamW/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:@@*
dtype0
�
2AdamW/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42AdamW/multi_head_attention/attention_output/bias/m
�
FAdamW/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp2AdamW/multi_head_attention/attention_output/bias/m*
_output_shapes
:@*
dtype0
�
+AdamW/multi_head_attention_1/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/query/kernel/m
�
?AdamW/multi_head_attention_1/query/kernel/m/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/query/kernel/m*"
_output_shapes
:@@*
dtype0
�
)AdamW/multi_head_attention_1/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/query/bias/m
�
=AdamW/multi_head_attention_1/query/bias/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/query/bias/m*
_output_shapes

:@*
dtype0
�
)AdamW/multi_head_attention_1/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention_1/key/kernel/m
�
=AdamW/multi_head_attention_1/key/kernel/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/key/kernel/m*"
_output_shapes
:@@*
dtype0
�
'AdamW/multi_head_attention_1/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention_1/key/bias/m
�
;AdamW/multi_head_attention_1/key/bias/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention_1/key/bias/m*
_output_shapes

:@*
dtype0
�
+AdamW/multi_head_attention_1/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/value/kernel/m
�
?AdamW/multi_head_attention_1/value/kernel/m/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/value/kernel/m*"
_output_shapes
:@@*
dtype0
�
)AdamW/multi_head_attention_1/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/value/bias/m
�
=AdamW/multi_head_attention_1/value/bias/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/value/bias/m*
_output_shapes

:@*
dtype0
�
6AdamW/multi_head_attention_1/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*G
shared_name86AdamW/multi_head_attention_1/attention_output/kernel/m
�
JAdamW/multi_head_attention_1/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp6AdamW/multi_head_attention_1/attention_output/kernel/m*"
_output_shapes
:@@*
dtype0
�
4AdamW/multi_head_attention_1/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64AdamW/multi_head_attention_1/attention_output/bias/m
�
HAdamW/multi_head_attention_1/attention_output/bias/m/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention_1/attention_output/bias/m*
_output_shapes
:@*
dtype0
�
AdamW/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_1/kernel/v
�
)AdamW/conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/conv_1/kernel/v*&
_output_shapes
:
*
dtype0
~
AdamW/conv_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdamW/conv_1/bias/v
w
'AdamW/conv_1/bias/v/Read/ReadVariableOpReadVariableOpAdamW/conv_1/bias/v*
_output_shapes
:
*
dtype0
�
AdamW/conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*&
shared_nameAdamW/conv_2/kernel/v
�
)AdamW/conv_2/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/conv_2/kernel/v*&
_output_shapes
:

*
dtype0
~
AdamW/conv_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdamW/conv_2/bias/v
w
'AdamW/conv_2/bias/v/Read/ReadVariableOpReadVariableOpAdamW/conv_2/bias/v*
_output_shapes
:
*
dtype0
�
AdamW/conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_3/kernel/v
�
)AdamW/conv_3/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/conv_3/kernel/v*&
_output_shapes
:
*
dtype0
~
AdamW/conv_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamW/conv_3/bias/v
w
'AdamW/conv_3/bias/v/Read/ReadVariableOpReadVariableOpAdamW/conv_3/bias/v*
_output_shapes
:*
dtype0
�
AdamW/conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_4/kernel/v
�
)AdamW/conv_4/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/conv_4/kernel/v*&
_output_shapes
:*
dtype0
~
AdamW/conv_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamW/conv_4/bias/v
w
'AdamW/conv_4/bias/v/Read/ReadVariableOpReadVariableOpAdamW/conv_4/bias/v*
_output_shapes
:*
dtype0
�
!AdamW/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!AdamW/layer_normalization/gamma/v
�
5AdamW/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp!AdamW/layer_normalization/gamma/v*
_output_shapes
:@*
dtype0
�
 AdamW/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/layer_normalization/beta/v
�
4AdamW/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp AdamW/layer_normalization/beta/v*
_output_shapes
:@*
dtype0
�
#AdamW/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_1/gamma/v
�
7AdamW/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
�
"AdamW/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_1/beta/v
�
6AdamW/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_1/beta/v*
_output_shapes
:@*
dtype0
�
AdamW/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*'
shared_nameAdamW/dense_1/kernel/v
�
*AdamW/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_1/kernel/v*
_output_shapes
:	@�*
dtype0
�
AdamW/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdamW/dense_1/bias/v
z
(AdamW/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_1/bias/v*
_output_shapes	
:�*
dtype0
�
AdamW/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdamW/dense_2/kernel/v
�
*AdamW/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_2/kernel/v*
_output_shapes
:	�@*
dtype0
�
AdamW/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdamW/dense_2/bias/v
y
(AdamW/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_2/bias/v*
_output_shapes
:@*
dtype0
�
#AdamW/layer_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_2/gamma/v
�
7AdamW/layer_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
�
"AdamW/layer_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_2/beta/v
�
6AdamW/layer_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_2/beta/v*
_output_shapes
:@*
dtype0
�
#AdamW/layer_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_3/gamma/v
�
7AdamW/layer_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_3/gamma/v*
_output_shapes
:@*
dtype0
�
"AdamW/layer_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_3/beta/v
�
6AdamW/layer_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_3/beta/v*
_output_shapes
:@*
dtype0
�
AdamW/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@�*'
shared_nameAdamW/dense_3/kernel/v
�
*AdamW/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_3/kernel/v*
_output_shapes
:	@�*
dtype0
�
AdamW/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdamW/dense_3/bias/v
z
(AdamW/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_3/bias/v*
_output_shapes	
:�*
dtype0
�
AdamW/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*'
shared_nameAdamW/dense_4/kernel/v
�
*AdamW/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_4/kernel/v*
_output_shapes
:	�@*
dtype0
�
AdamW/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdamW/dense_4/bias/v
y
(AdamW/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_4/bias/v*
_output_shapes
:@*
dtype0
�
#AdamW/layer_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_4/gamma/v
�
7AdamW/layer_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_4/gamma/v*
_output_shapes
:@*
dtype0
�
"AdamW/layer_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_4/beta/v
�
6AdamW/layer_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_4/beta/v*
_output_shapes
:@*
dtype0
�
AdamW/FC_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�(2*$
shared_nameAdamW/FC_1/kernel/v
|
'AdamW/FC_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/v*
_output_shapes
:	�(2*
dtype0
z
AdamW/FC_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameAdamW/FC_1/bias/v
s
%AdamW/FC_1/bias/v/Read/ReadVariableOpReadVariableOpAdamW/FC_1/bias/v*
_output_shapes
:2*
dtype0
�
AdamW/FC_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*$
shared_nameAdamW/FC_2/kernel/v
{
'AdamW/FC_2/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/FC_2/kernel/v*
_output_shapes

:22*
dtype0
z
AdamW/FC_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*"
shared_nameAdamW/FC_2/bias/v
s
%AdamW/FC_2/bias/v/Read/ReadVariableOpReadVariableOpAdamW/FC_2/bias/v*
_output_shapes
:2*
dtype0
�
AdamW/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_nameAdamW/output_layer/kernel/v
�
/AdamW/output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/output_layer/kernel/v*
_output_shapes

:2*
dtype0
�
AdamW/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdamW/output_layer/bias/v
�
-AdamW/output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdamW/output_layer/bias/v*
_output_shapes
:*
dtype0
�
"AdamW/patch_encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*3
shared_name$"AdamW/patch_encoder/dense/kernel/v
�
6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/v*
_output_shapes
:	�@*
dtype0
�
 AdamW/patch_encoder/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/patch_encoder/dense/bias/v
�
4AdamW/patch_encoder/dense/bias/v/Read/ReadVariableOpReadVariableOp AdamW/patch_encoder/dense/bias/v*
_output_shapes
:@*
dtype0
�
*AdamW/patch_encoder/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/v
�
>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/v*
_output_shapes

:Q@*
dtype0
�
)AdamW/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/query/kernel/v
�
=AdamW/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/query/kernel/v*"
_output_shapes
:@@*
dtype0
�
'AdamW/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/query/bias/v
�
;AdamW/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/query/bias/v*
_output_shapes

:@*
dtype0
�
'AdamW/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'AdamW/multi_head_attention/key/kernel/v
�
;AdamW/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/key/kernel/v*"
_output_shapes
:@@*
dtype0
�
%AdamW/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%AdamW/multi_head_attention/key/bias/v
�
9AdamW/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp%AdamW/multi_head_attention/key/bias/v*
_output_shapes

:@*
dtype0
�
)AdamW/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/value/kernel/v
�
=AdamW/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/value/kernel/v*"
_output_shapes
:@@*
dtype0
�
'AdamW/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/value/bias/v
�
;AdamW/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/value/bias/v*
_output_shapes

:@*
dtype0
�
4AdamW/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*E
shared_name64AdamW/multi_head_attention/attention_output/kernel/v
�
HAdamW/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:@@*
dtype0
�
2AdamW/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42AdamW/multi_head_attention/attention_output/bias/v
�
FAdamW/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp2AdamW/multi_head_attention/attention_output/bias/v*
_output_shapes
:@*
dtype0
�
+AdamW/multi_head_attention_1/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/query/kernel/v
�
?AdamW/multi_head_attention_1/query/kernel/v/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/query/kernel/v*"
_output_shapes
:@@*
dtype0
�
)AdamW/multi_head_attention_1/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/query/bias/v
�
=AdamW/multi_head_attention_1/query/bias/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/query/bias/v*
_output_shapes

:@*
dtype0
�
)AdamW/multi_head_attention_1/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention_1/key/kernel/v
�
=AdamW/multi_head_attention_1/key/kernel/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/key/kernel/v*"
_output_shapes
:@@*
dtype0
�
'AdamW/multi_head_attention_1/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention_1/key/bias/v
�
;AdamW/multi_head_attention_1/key/bias/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention_1/key/bias/v*
_output_shapes

:@*
dtype0
�
+AdamW/multi_head_attention_1/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/value/kernel/v
�
?AdamW/multi_head_attention_1/value/kernel/v/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/value/kernel/v*"
_output_shapes
:@@*
dtype0
�
)AdamW/multi_head_attention_1/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/value/bias/v
�
=AdamW/multi_head_attention_1/value/bias/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/value/bias/v*
_output_shapes

:@*
dtype0
�
6AdamW/multi_head_attention_1/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*G
shared_name86AdamW/multi_head_attention_1/attention_output/kernel/v
�
JAdamW/multi_head_attention_1/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp6AdamW/multi_head_attention_1/attention_output/kernel/v*"
_output_shapes
:@@*
dtype0
�
4AdamW/multi_head_attention_1/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64AdamW/multi_head_attention_1/attention_output/bias/v
�
HAdamW/multi_head_attention_1/attention_output/bias/v/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention_1/attention_output/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer_with_weights-14
layer-21
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer-26
layer_with_weights-17
layer-27
layer-28
layer_with_weights-18
layer-29
	optimizer
 regularization_losses
!trainable_variables
"	variables
#	keras_api
$
signatures
 
h

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
R
1regularization_losses
2	variables
3trainable_variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
R
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
R
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
z
I
projection
Jposition_embedding
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
q
Oaxis
	Pgamma
Qbeta
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
�
V_query_dense
W
_key_dense
X_value_dense
Y_softmax
Z_dropout_layer
[_output_dense
\regularization_losses
]	variables
^trainable_variables
_	keras_api
R
`regularization_losses
a	variables
btrainable_variables
c	keras_api
q
daxis
	egamma
fbeta
gregularization_losses
h	variables
itrainable_variables
j	keras_api
h

kkernel
lbias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
h

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
R
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
s
{axis
	|gamma
}beta
~regularization_losses
	variables
�trainable_variables
�	keras_api
�
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
x
	�axis

�gamma
	�beta
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
x
	�axis

�gamma
	�beta
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�	
	�iter
�beta_1
�beta_2

�decay
�learning_rate
�weight_decay%m�&m�+m�,m�5m�6m�;m�<m�Pm�Qm�em�fm�km�lm�qm�rm�|m�}m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�%v�&v�+v�,v�5v�6v�;v�<v�Pv�Qv�ev�fv�kv�lv�qv�rv�|v�}v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�
 
�
%0
&1
+2
,3
54
65
;6
<7
�8
�9
�10
P11
Q12
�13
�14
�15
�16
�17
�18
�19
�20
e21
f22
k23
l24
q25
r26
|27
}28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�
%0
&1
+2
,3
54
65
;6
<7
�8
�9
�10
P11
Q12
�13
�14
�15
�16
�17
�18
�19
�20
e21
f22
k23
l24
q25
r26
|27
}28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�
�non_trainable_variables
�layers
 regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
!trainable_variables
"	variables
 
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
�
�non_trainable_variables
�layers
'regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
(	variables
)trainable_variables
YW
VARIABLE_VALUEconv_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
�
�non_trainable_variables
�layers
-regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
.	variables
/trainable_variables
 
 
 
�
�non_trainable_variables
�layers
1regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
2	variables
3trainable_variables
YW
VARIABLE_VALUEconv_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
�
�non_trainable_variables
�layers
7regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
8	variables
9trainable_variables
YW
VARIABLE_VALUEconv_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
�
�non_trainable_variables
�layers
=regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
>	variables
?trainable_variables
 
 
 
�
�non_trainable_variables
�layers
Aregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
B	variables
Ctrainable_variables
 
 
 
�
�non_trainable_variables
�layers
Eregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
F	variables
Gtrainable_variables
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
g
�
embeddings
�regularization_losses
�	variables
�trainable_variables
�	keras_api
 

�0
�1
�2

�0
�1
�2
�
�non_trainable_variables
�layers
Kregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
L	variables
Mtrainable_variables
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1
�
�non_trainable_variables
�layers
Rregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
S	variables
Ttrainable_variables
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
 
@
�0
�1
�2
�3
�4
�5
�6
�7
@
�0
�1
�2
�3
�4
�5
�6
�7
�
�non_trainable_variables
�layers
\regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
]	variables
^trainable_variables
 
 
 
�
�non_trainable_variables
�layers
`regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
a	variables
btrainable_variables
 
fd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
�
�non_trainable_variables
�layers
gregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
h	variables
itrainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

k0
l1

k0
l1
�
�non_trainable_variables
�layers
mregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
n	variables
otrainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
�
�non_trainable_variables
�layers
sregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
t	variables
utrainable_variables
 
 
 
�
�non_trainable_variables
�layers
wregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
x	variables
ytrainable_variables
 
ge
VARIABLE_VALUElayer_normalization_2/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer_normalization_2/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
 

|0
}1

|0
}1
�
�non_trainable_variables
�layers
~regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
	variables
�trainable_variables
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
 
@
�0
�1
�2
�3
�4
�5
�6
�7
@
�0
�1
�2
�3
�4
�5
�6
�7
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
ge
VARIABLE_VALUElayer_normalization_3/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer_normalization_3/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
[Y
VARIABLE_VALUEdense_3/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_3/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
[Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_4/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
ge
VARIABLE_VALUElayer_normalization_4/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer_normalization_4/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
XV
VARIABLE_VALUEFC_1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	FC_1/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
XV
VARIABLE_VALUEFC_2/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	FC_2/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
`^
VARIABLE_VALUEoutput_layer/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEoutput_layer/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
IG
VARIABLE_VALUE
AdamW/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdamW/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdamW/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEAdamW/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEAdamW/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEAdamW/weight_decay1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEpatch_encoder/dense/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEpatch_encoder/dense/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE"patch_encoder/embedding/embeddings1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!multi_head_attention/query/kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmulti_head_attention/query/bias1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmulti_head_attention/key/kernel1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmulti_head_attention/key/bias1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!multi_head_attention/value/kernel1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmulti_head_attention/value/bias1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,multi_head_attention/attention_output/kernel1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*multi_head_attention/attention_output/bias1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#multi_head_attention_1/query/kernel1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!multi_head_attention_1/query/bias1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!multi_head_attention_1/key/kernel1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmulti_head_attention_1/key/bias1trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#multi_head_attention_1/value/kernel1trainable_variables/33/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!multi_head_attention_1/value/bias1trainable_variables/34/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.multi_head_attention_1/attention_output/kernel1trainable_variables/35/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,multi_head_attention_1/attention_output/bias1trainable_variables/36/.ATTRIBUTES/VARIABLE_VALUE
 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 

�0

�0
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 

I0
J1
 
 
 
 
 
 
 
 
 
 
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
*
V0
W1
X2
Y3
Z4
[5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
 
 

�0
�1

�0
�1
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
 
0
�0
�1
�2
�3
�4
�5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
}{
VARIABLE_VALUEAdamW/conv_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!AdamW/layer_normalization/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE AdamW/layer_normalization/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/mRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/mQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_3/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_3/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_3/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_3/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_4/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_4/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_4/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_4/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_1/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_1/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_2/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_2/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdamW/output_layer/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdamW/output_layer/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/mMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/mMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE+AdamW/multi_head_attention_1/query/kernel/mMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention_1/query/bias/mMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention_1/key/kernel/mMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'AdamW/multi_head_attention_1/key/bias/mMtrainable_variables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE+AdamW/multi_head_attention_1/value/kernel/mMtrainable_variables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention_1/value/bias/mMtrainable_variables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE6AdamW/multi_head_attention_1/attention_output/kernel/mMtrainable_variables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4AdamW/multi_head_attention_1/attention_output/bias/mMtrainable_variables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!AdamW/layer_normalization/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE AdamW/layer_normalization/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/vRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/vQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_3/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_3/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_3/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_3/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_4/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_4/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_4/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_4/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_1/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_1/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_2/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_2/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdamW/output_layer/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdamW/output_layer/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/vMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/vMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE+AdamW/multi_head_attention_1/query/kernel/vMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention_1/query/bias/vMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention_1/key/kernel/vMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE'AdamW/multi_head_attention_1/key/bias/vMtrainable_variables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE+AdamW/multi_head_attention_1/value/kernel/vMtrainable_variables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE)AdamW/multi_head_attention_1/value/bias/vMtrainable_variables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE6AdamW/multi_head_attention_1/attention_output/kernel/vMtrainable_variables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4AdamW/multi_head_attention_1/attention_output/bias/vMtrainable_variables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_layerPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasconv_4/kernelconv_4/biaspatch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddingslayer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biaslayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/beta#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biaslayer_normalization_3/gammalayer_normalization_3/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/biaslayer_normalization_4/gammalayer_normalization_4/betaFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*2
config_proto" 

CPU

GPU2 *0J 8� *-
f(R&
$__inference_signature_wrapper_429896
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�B
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp!conv_4/kernel/Read/ReadVariableOpconv_4/bias/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp/layer_normalization_3/gamma/Read/ReadVariableOp.layer_normalization_3/beta/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOpFC_1/kernel/Read/ReadVariableOpFC_1/bias/Read/ReadVariableOpFC_2/kernel/Read/ReadVariableOpFC_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdamW/iter/Read/ReadVariableOp AdamW/beta_1/Read/ReadVariableOp AdamW/beta_2/Read/ReadVariableOpAdamW/decay/Read/ReadVariableOp'AdamW/learning_rate/Read/ReadVariableOp&AdamW/weight_decay/Read/ReadVariableOp.patch_encoder/dense/kernel/Read/ReadVariableOp,patch_encoder/dense/bias/Read/ReadVariableOp6patch_encoder/embedding/embeddings/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOp7multi_head_attention_1/query/kernel/Read/ReadVariableOp5multi_head_attention_1/query/bias/Read/ReadVariableOp5multi_head_attention_1/key/kernel/Read/ReadVariableOp3multi_head_attention_1/key/bias/Read/ReadVariableOp7multi_head_attention_1/value/kernel/Read/ReadVariableOp5multi_head_attention_1/value/bias/Read/ReadVariableOpBmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_1/attention_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)AdamW/conv_1/kernel/m/Read/ReadVariableOp'AdamW/conv_1/bias/m/Read/ReadVariableOp)AdamW/conv_2/kernel/m/Read/ReadVariableOp'AdamW/conv_2/bias/m/Read/ReadVariableOp)AdamW/conv_3/kernel/m/Read/ReadVariableOp'AdamW/conv_3/bias/m/Read/ReadVariableOp)AdamW/conv_4/kernel/m/Read/ReadVariableOp'AdamW/conv_4/bias/m/Read/ReadVariableOp5AdamW/layer_normalization/gamma/m/Read/ReadVariableOp4AdamW/layer_normalization/beta/m/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/m/Read/ReadVariableOp*AdamW/dense_1/kernel/m/Read/ReadVariableOp(AdamW/dense_1/bias/m/Read/ReadVariableOp*AdamW/dense_2/kernel/m/Read/ReadVariableOp(AdamW/dense_2/bias/m/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/m/Read/ReadVariableOp7AdamW/layer_normalization_3/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_3/beta/m/Read/ReadVariableOp*AdamW/dense_3/kernel/m/Read/ReadVariableOp(AdamW/dense_3/bias/m/Read/ReadVariableOp*AdamW/dense_4/kernel/m/Read/ReadVariableOp(AdamW/dense_4/bias/m/Read/ReadVariableOp7AdamW/layer_normalization_4/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_4/beta/m/Read/ReadVariableOp'AdamW/FC_1/kernel/m/Read/ReadVariableOp%AdamW/FC_1/bias/m/Read/ReadVariableOp'AdamW/FC_2/kernel/m/Read/ReadVariableOp%AdamW/FC_2/bias/m/Read/ReadVariableOp/AdamW/output_layer/kernel/m/Read/ReadVariableOp-AdamW/output_layer/bias/m/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/m/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/m/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/m/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/m/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/m/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/m/Read/ReadVariableOp?AdamW/multi_head_attention_1/query/kernel/m/Read/ReadVariableOp=AdamW/multi_head_attention_1/query/bias/m/Read/ReadVariableOp=AdamW/multi_head_attention_1/key/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention_1/key/bias/m/Read/ReadVariableOp?AdamW/multi_head_attention_1/value/kernel/m/Read/ReadVariableOp=AdamW/multi_head_attention_1/value/bias/m/Read/ReadVariableOpJAdamW/multi_head_attention_1/attention_output/kernel/m/Read/ReadVariableOpHAdamW/multi_head_attention_1/attention_output/bias/m/Read/ReadVariableOp)AdamW/conv_1/kernel/v/Read/ReadVariableOp'AdamW/conv_1/bias/v/Read/ReadVariableOp)AdamW/conv_2/kernel/v/Read/ReadVariableOp'AdamW/conv_2/bias/v/Read/ReadVariableOp)AdamW/conv_3/kernel/v/Read/ReadVariableOp'AdamW/conv_3/bias/v/Read/ReadVariableOp)AdamW/conv_4/kernel/v/Read/ReadVariableOp'AdamW/conv_4/bias/v/Read/ReadVariableOp5AdamW/layer_normalization/gamma/v/Read/ReadVariableOp4AdamW/layer_normalization/beta/v/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/v/Read/ReadVariableOp*AdamW/dense_1/kernel/v/Read/ReadVariableOp(AdamW/dense_1/bias/v/Read/ReadVariableOp*AdamW/dense_2/kernel/v/Read/ReadVariableOp(AdamW/dense_2/bias/v/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/v/Read/ReadVariableOp7AdamW/layer_normalization_3/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_3/beta/v/Read/ReadVariableOp*AdamW/dense_3/kernel/v/Read/ReadVariableOp(AdamW/dense_3/bias/v/Read/ReadVariableOp*AdamW/dense_4/kernel/v/Read/ReadVariableOp(AdamW/dense_4/bias/v/Read/ReadVariableOp7AdamW/layer_normalization_4/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_4/beta/v/Read/ReadVariableOp'AdamW/FC_1/kernel/v/Read/ReadVariableOp%AdamW/FC_1/bias/v/Read/ReadVariableOp'AdamW/FC_2/kernel/v/Read/ReadVariableOp%AdamW/FC_2/bias/v/Read/ReadVariableOp/AdamW/output_layer/kernel/v/Read/ReadVariableOp-AdamW/output_layer/bias/v/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/v/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/v/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/v/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/v/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/v/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/v/Read/ReadVariableOp?AdamW/multi_head_attention_1/query/kernel/v/Read/ReadVariableOp=AdamW/multi_head_attention_1/query/bias/v/Read/ReadVariableOp=AdamW/multi_head_attention_1/key/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention_1/key/bias/v/Read/ReadVariableOp?AdamW/multi_head_attention_1/value/kernel/v/Read/ReadVariableOp=AdamW/multi_head_attention_1/value/bias/v/Read/ReadVariableOpJAdamW/multi_head_attention_1/attention_output/kernel/v/Read/ReadVariableOpHAdamW/multi_head_attention_1/attention_output/bias/v/Read/ReadVariableOpConst*�
Tin�
�2�	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *(
f#R!
__inference__traced_save_432282
�)
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasconv_4/kernelconv_4/biaslayer_normalization/gammalayer_normalization/betalayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/betalayer_normalization_3/gammalayer_normalization_3/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/biaslayer_normalization_4/gammalayer_normalization_4/betaFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias
AdamW/iterAdamW/beta_1AdamW/beta_2AdamW/decayAdamW/learning_rateAdamW/weight_decaypatch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddings!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biastotalcounttotal_1count_1AdamW/conv_1/kernel/mAdamW/conv_1/bias/mAdamW/conv_2/kernel/mAdamW/conv_2/bias/mAdamW/conv_3/kernel/mAdamW/conv_3/bias/mAdamW/conv_4/kernel/mAdamW/conv_4/bias/m!AdamW/layer_normalization/gamma/m AdamW/layer_normalization/beta/m#AdamW/layer_normalization_1/gamma/m"AdamW/layer_normalization_1/beta/mAdamW/dense_1/kernel/mAdamW/dense_1/bias/mAdamW/dense_2/kernel/mAdamW/dense_2/bias/m#AdamW/layer_normalization_2/gamma/m"AdamW/layer_normalization_2/beta/m#AdamW/layer_normalization_3/gamma/m"AdamW/layer_normalization_3/beta/mAdamW/dense_3/kernel/mAdamW/dense_3/bias/mAdamW/dense_4/kernel/mAdamW/dense_4/bias/m#AdamW/layer_normalization_4/gamma/m"AdamW/layer_normalization_4/beta/mAdamW/FC_1/kernel/mAdamW/FC_1/bias/mAdamW/FC_2/kernel/mAdamW/FC_2/bias/mAdamW/output_layer/kernel/mAdamW/output_layer/bias/m"AdamW/patch_encoder/dense/kernel/m AdamW/patch_encoder/dense/bias/m*AdamW/patch_encoder/embedding/embeddings/m)AdamW/multi_head_attention/query/kernel/m'AdamW/multi_head_attention/query/bias/m'AdamW/multi_head_attention/key/kernel/m%AdamW/multi_head_attention/key/bias/m)AdamW/multi_head_attention/value/kernel/m'AdamW/multi_head_attention/value/bias/m4AdamW/multi_head_attention/attention_output/kernel/m2AdamW/multi_head_attention/attention_output/bias/m+AdamW/multi_head_attention_1/query/kernel/m)AdamW/multi_head_attention_1/query/bias/m)AdamW/multi_head_attention_1/key/kernel/m'AdamW/multi_head_attention_1/key/bias/m+AdamW/multi_head_attention_1/value/kernel/m)AdamW/multi_head_attention_1/value/bias/m6AdamW/multi_head_attention_1/attention_output/kernel/m4AdamW/multi_head_attention_1/attention_output/bias/mAdamW/conv_1/kernel/vAdamW/conv_1/bias/vAdamW/conv_2/kernel/vAdamW/conv_2/bias/vAdamW/conv_3/kernel/vAdamW/conv_3/bias/vAdamW/conv_4/kernel/vAdamW/conv_4/bias/v!AdamW/layer_normalization/gamma/v AdamW/layer_normalization/beta/v#AdamW/layer_normalization_1/gamma/v"AdamW/layer_normalization_1/beta/vAdamW/dense_1/kernel/vAdamW/dense_1/bias/vAdamW/dense_2/kernel/vAdamW/dense_2/bias/v#AdamW/layer_normalization_2/gamma/v"AdamW/layer_normalization_2/beta/v#AdamW/layer_normalization_3/gamma/v"AdamW/layer_normalization_3/beta/vAdamW/dense_3/kernel/vAdamW/dense_3/bias/vAdamW/dense_4/kernel/vAdamW/dense_4/bias/v#AdamW/layer_normalization_4/gamma/v"AdamW/layer_normalization_4/beta/vAdamW/FC_1/kernel/vAdamW/FC_1/bias/vAdamW/FC_2/kernel/vAdamW/FC_2/bias/vAdamW/output_layer/kernel/vAdamW/output_layer/bias/v"AdamW/patch_encoder/dense/kernel/v AdamW/patch_encoder/dense/bias/v*AdamW/patch_encoder/embedding/embeddings/v)AdamW/multi_head_attention/query/kernel/v'AdamW/multi_head_attention/query/bias/v'AdamW/multi_head_attention/key/kernel/v%AdamW/multi_head_attention/key/bias/v)AdamW/multi_head_attention/value/kernel/v'AdamW/multi_head_attention/value/bias/v4AdamW/multi_head_attention/attention_output/kernel/v2AdamW/multi_head_attention/attention_output/bias/v+AdamW/multi_head_attention_1/query/kernel/v)AdamW/multi_head_attention_1/query/bias/v)AdamW/multi_head_attention_1/key/kernel/v'AdamW/multi_head_attention_1/key/bias/v+AdamW/multi_head_attention_1/value/kernel/v)AdamW/multi_head_attention_1/value/bias/v6AdamW/multi_head_attention_1/attention_output/kernel/v4AdamW/multi_head_attention_1/attention_output/bias/v*�
Tin�
�2�*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *+
f&R$
"__inference__traced_restore_432781��*
�
R
&__inference_add_2_layer_call_fn_431513
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4283242
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:U Q
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/1
�	
�
@__inference_FC_2_layer_call_and_return_conditional_losses_428519

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
a
E__inference_maxpool_1_layer_call_and_return_conditional_losses_427869

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_430795

inputs!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:Q@

unknown_10:@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@ 

unknown_16:@@

unknown_17:@ 

unknown_18:@@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:	@�

unknown_23:	�

unknown_24:	�@

unknown_25:@

unknown_26:@

unknown_27:@ 

unknown_28:@@

unknown_29:@ 

unknown_30:@@

unknown_31:@ 

unknown_32:@@

unknown_33:@ 

unknown_34:@@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:	@�

unknown_39:	�

unknown_40:	�@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:	�(2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identity��StatefulPartitionedCall�
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*2
config_proto" 

CPU

GPU2 *0J 8� *^
fYRW
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_4285502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
k
A__inference_add_2_layer_call_and_return_conditional_losses_428324

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
�
-__inference_output_layer_layer_call_fn_431770

inputs
unknown:2
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_4285432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
4__inference_layer_normalization_layer_call_fn_431079

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_4280462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�

�
B__inference_conv_2_layer_call_and_return_conditional_losses_430931

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������

 
_user_specified_nameinputs
�-
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_428300	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yj
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2
softmax/Softmax�
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������QQ2
dropout/Identity�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�'
�
C__inference_dense_4_layer_call_and_return_conditional_losses_431629

inputs4
!tensordot_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xx
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������Q�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������Q�
 
_user_specified_nameinputs
�
�
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_430902

inputs!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:Q@

unknown_10:@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@ 

unknown_16:@@

unknown_17:@ 

unknown_18:@@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:	@�

unknown_23:	�

unknown_24:	�@

unknown_25:@

unknown_26:@

unknown_27:@ 

unknown_28:@@

unknown_29:@ 

unknown_30:@@

unknown_31:@ 

unknown_32:@@

unknown_33:@ 

unknown_34:@@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:	@�

unknown_39:	�

unknown_40:	�@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:	�(2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identity��StatefulPartitionedCall�
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*2
config_proto" 

CPU

GPU2 *0J 8� *^
fYRW
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_4292972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
I
-__inference_leaky_ReLu_1_layer_call_fn_431721

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_4285072
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
��
�B
!__inference__wrapped_model_427863
input_layerY
?wheatclassifier_cnn_vit_6_conv_1_conv2d_readvariableop_resource:
N
@wheatclassifier_cnn_vit_6_conv_1_biasadd_readvariableop_resource:
Y
?wheatclassifier_cnn_vit_6_conv_2_conv2d_readvariableop_resource:

N
@wheatclassifier_cnn_vit_6_conv_2_biasadd_readvariableop_resource:
Y
?wheatclassifier_cnn_vit_6_conv_3_conv2d_readvariableop_resource:
N
@wheatclassifier_cnn_vit_6_conv_3_biasadd_readvariableop_resource:Y
?wheatclassifier_cnn_vit_6_conv_4_conv2d_readvariableop_resource:N
@wheatclassifier_cnn_vit_6_conv_4_biasadd_readvariableop_resource:b
Owheatclassifier_cnn_vit_6_patch_encoder_dense_tensordot_readvariableop_resource:	�@[
Mwheatclassifier_cnn_vit_6_patch_encoder_dense_biasadd_readvariableop_resource:@[
Iwheatclassifier_cnn_vit_6_patch_encoder_embedding_embedding_lookup_427543:Q@a
Swheatclassifier_cnn_vit_6_layer_normalization_batchnorm_mul_readvariableop_resource:@]
Owheatclassifier_cnn_vit_6_layer_normalization_batchnorm_readvariableop_resource:@p
Zwheatclassifier_cnn_vit_6_multi_head_attention_query_einsum_einsum_readvariableop_resource:@@b
Pwheatclassifier_cnn_vit_6_multi_head_attention_query_add_readvariableop_resource:@n
Xwheatclassifier_cnn_vit_6_multi_head_attention_key_einsum_einsum_readvariableop_resource:@@`
Nwheatclassifier_cnn_vit_6_multi_head_attention_key_add_readvariableop_resource:@p
Zwheatclassifier_cnn_vit_6_multi_head_attention_value_einsum_einsum_readvariableop_resource:@@b
Pwheatclassifier_cnn_vit_6_multi_head_attention_value_add_readvariableop_resource:@{
ewheatclassifier_cnn_vit_6_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:@@i
[wheatclassifier_cnn_vit_6_multi_head_attention_attention_output_add_readvariableop_resource:@c
Uwheatclassifier_cnn_vit_6_layer_normalization_1_batchnorm_mul_readvariableop_resource:@_
Qwheatclassifier_cnn_vit_6_layer_normalization_1_batchnorm_readvariableop_resource:@V
Cwheatclassifier_cnn_vit_6_dense_1_tensordot_readvariableop_resource:	@�P
Awheatclassifier_cnn_vit_6_dense_1_biasadd_readvariableop_resource:	�V
Cwheatclassifier_cnn_vit_6_dense_2_tensordot_readvariableop_resource:	�@O
Awheatclassifier_cnn_vit_6_dense_2_biasadd_readvariableop_resource:@c
Uwheatclassifier_cnn_vit_6_layer_normalization_2_batchnorm_mul_readvariableop_resource:@_
Qwheatclassifier_cnn_vit_6_layer_normalization_2_batchnorm_readvariableop_resource:@r
\wheatclassifier_cnn_vit_6_multi_head_attention_1_query_einsum_einsum_readvariableop_resource:@@d
Rwheatclassifier_cnn_vit_6_multi_head_attention_1_query_add_readvariableop_resource:@p
Zwheatclassifier_cnn_vit_6_multi_head_attention_1_key_einsum_einsum_readvariableop_resource:@@b
Pwheatclassifier_cnn_vit_6_multi_head_attention_1_key_add_readvariableop_resource:@r
\wheatclassifier_cnn_vit_6_multi_head_attention_1_value_einsum_einsum_readvariableop_resource:@@d
Rwheatclassifier_cnn_vit_6_multi_head_attention_1_value_add_readvariableop_resource:@}
gwheatclassifier_cnn_vit_6_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:@@k
]wheatclassifier_cnn_vit_6_multi_head_attention_1_attention_output_add_readvariableop_resource:@c
Uwheatclassifier_cnn_vit_6_layer_normalization_3_batchnorm_mul_readvariableop_resource:@_
Qwheatclassifier_cnn_vit_6_layer_normalization_3_batchnorm_readvariableop_resource:@V
Cwheatclassifier_cnn_vit_6_dense_3_tensordot_readvariableop_resource:	@�P
Awheatclassifier_cnn_vit_6_dense_3_biasadd_readvariableop_resource:	�V
Cwheatclassifier_cnn_vit_6_dense_4_tensordot_readvariableop_resource:	�@O
Awheatclassifier_cnn_vit_6_dense_4_biasadd_readvariableop_resource:@c
Uwheatclassifier_cnn_vit_6_layer_normalization_4_batchnorm_mul_readvariableop_resource:@_
Qwheatclassifier_cnn_vit_6_layer_normalization_4_batchnorm_readvariableop_resource:@P
=wheatclassifier_cnn_vit_6_fc_1_matmul_readvariableop_resource:	�(2L
>wheatclassifier_cnn_vit_6_fc_1_biasadd_readvariableop_resource:2O
=wheatclassifier_cnn_vit_6_fc_2_matmul_readvariableop_resource:22L
>wheatclassifier_cnn_vit_6_fc_2_biasadd_readvariableop_resource:2W
Ewheatclassifier_cnn_vit_6_output_layer_matmul_readvariableop_resource:2T
Fwheatclassifier_cnn_vit_6_output_layer_biasadd_readvariableop_resource:
identity��5WheatClassifier_CNN-VIT_6/FC_1/BiasAdd/ReadVariableOp�4WheatClassifier_CNN-VIT_6/FC_1/MatMul/ReadVariableOp�5WheatClassifier_CNN-VIT_6/FC_2/BiasAdd/ReadVariableOp�4WheatClassifier_CNN-VIT_6/FC_2/MatMul/ReadVariableOp�7WheatClassifier_CNN-VIT_6/conv_1/BiasAdd/ReadVariableOp�6WheatClassifier_CNN-VIT_6/conv_1/Conv2D/ReadVariableOp�7WheatClassifier_CNN-VIT_6/conv_2/BiasAdd/ReadVariableOp�6WheatClassifier_CNN-VIT_6/conv_2/Conv2D/ReadVariableOp�7WheatClassifier_CNN-VIT_6/conv_3/BiasAdd/ReadVariableOp�6WheatClassifier_CNN-VIT_6/conv_3/Conv2D/ReadVariableOp�7WheatClassifier_CNN-VIT_6/conv_4/BiasAdd/ReadVariableOp�6WheatClassifier_CNN-VIT_6/conv_4/Conv2D/ReadVariableOp�8WheatClassifier_CNN-VIT_6/dense_1/BiasAdd/ReadVariableOp�:WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ReadVariableOp�8WheatClassifier_CNN-VIT_6/dense_2/BiasAdd/ReadVariableOp�:WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ReadVariableOp�8WheatClassifier_CNN-VIT_6/dense_3/BiasAdd/ReadVariableOp�:WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ReadVariableOp�8WheatClassifier_CNN-VIT_6/dense_4/BiasAdd/ReadVariableOp�:WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ReadVariableOp�FWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/ReadVariableOp�JWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul/ReadVariableOp�HWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/ReadVariableOp�LWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul/ReadVariableOp�HWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/ReadVariableOp�LWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul/ReadVariableOp�HWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/ReadVariableOp�LWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul/ReadVariableOp�HWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/ReadVariableOp�LWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul/ReadVariableOp�RWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/add/ReadVariableOp�\WheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�EWheatClassifier_CNN-VIT_6/multi_head_attention/key/add/ReadVariableOp�OWheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/Einsum/ReadVariableOp�GWheatClassifier_CNN-VIT_6/multi_head_attention/query/add/ReadVariableOp�QWheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/Einsum/ReadVariableOp�GWheatClassifier_CNN-VIT_6/multi_head_attention/value/add/ReadVariableOp�QWheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/Einsum/ReadVariableOp�TWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/add/ReadVariableOp�^WheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp�GWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/add/ReadVariableOp�QWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp�IWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/add/ReadVariableOp�SWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp�IWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/add/ReadVariableOp�SWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp�=WheatClassifier_CNN-VIT_6/output_layer/BiasAdd/ReadVariableOp�<WheatClassifier_CNN-VIT_6/output_layer/MatMul/ReadVariableOp�DWheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAdd/ReadVariableOp�FWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ReadVariableOp�BWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup�
6WheatClassifier_CNN-VIT_6/conv_1/Conv2D/ReadVariableOpReadVariableOp?wheatclassifier_cnn_vit_6_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype028
6WheatClassifier_CNN-VIT_6/conv_1/Conv2D/ReadVariableOp�
'WheatClassifier_CNN-VIT_6/conv_1/Conv2DConv2Dinput_layer>WheatClassifier_CNN-VIT_6/conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2)
'WheatClassifier_CNN-VIT_6/conv_1/Conv2D�
7WheatClassifier_CNN-VIT_6/conv_1/BiasAdd/ReadVariableOpReadVariableOp@wheatclassifier_cnn_vit_6_conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype029
7WheatClassifier_CNN-VIT_6/conv_1/BiasAdd/ReadVariableOp�
(WheatClassifier_CNN-VIT_6/conv_1/BiasAddBiasAdd0WheatClassifier_CNN-VIT_6/conv_1/Conv2D:output:0?WheatClassifier_CNN-VIT_6/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2*
(WheatClassifier_CNN-VIT_6/conv_1/BiasAdd�
6WheatClassifier_CNN-VIT_6/conv_2/Conv2D/ReadVariableOpReadVariableOp?wheatclassifier_cnn_vit_6_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype028
6WheatClassifier_CNN-VIT_6/conv_2/Conv2D/ReadVariableOp�
'WheatClassifier_CNN-VIT_6/conv_2/Conv2DConv2D1WheatClassifier_CNN-VIT_6/conv_1/BiasAdd:output:0>WheatClassifier_CNN-VIT_6/conv_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2)
'WheatClassifier_CNN-VIT_6/conv_2/Conv2D�
7WheatClassifier_CNN-VIT_6/conv_2/BiasAdd/ReadVariableOpReadVariableOp@wheatclassifier_cnn_vit_6_conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype029
7WheatClassifier_CNN-VIT_6/conv_2/BiasAdd/ReadVariableOp�
(WheatClassifier_CNN-VIT_6/conv_2/BiasAddBiasAdd0WheatClassifier_CNN-VIT_6/conv_2/Conv2D:output:0?WheatClassifier_CNN-VIT_6/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2*
(WheatClassifier_CNN-VIT_6/conv_2/BiasAdd�
+WheatClassifier_CNN-VIT_6/maxpool_1/MaxPoolMaxPool1WheatClassifier_CNN-VIT_6/conv_2/BiasAdd:output:0*/
_output_shapes
:���������bb
*
ksize
*
paddingVALID*
strides
2-
+WheatClassifier_CNN-VIT_6/maxpool_1/MaxPool�
6WheatClassifier_CNN-VIT_6/conv_3/Conv2D/ReadVariableOpReadVariableOp?wheatclassifier_cnn_vit_6_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype028
6WheatClassifier_CNN-VIT_6/conv_3/Conv2D/ReadVariableOp�
'WheatClassifier_CNN-VIT_6/conv_3/Conv2DConv2D4WheatClassifier_CNN-VIT_6/maxpool_1/MaxPool:output:0>WheatClassifier_CNN-VIT_6/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``*
paddingVALID*
strides
2)
'WheatClassifier_CNN-VIT_6/conv_3/Conv2D�
7WheatClassifier_CNN-VIT_6/conv_3/BiasAdd/ReadVariableOpReadVariableOp@wheatclassifier_cnn_vit_6_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7WheatClassifier_CNN-VIT_6/conv_3/BiasAdd/ReadVariableOp�
(WheatClassifier_CNN-VIT_6/conv_3/BiasAddBiasAdd0WheatClassifier_CNN-VIT_6/conv_3/Conv2D:output:0?WheatClassifier_CNN-VIT_6/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``2*
(WheatClassifier_CNN-VIT_6/conv_3/BiasAdd�
6WheatClassifier_CNN-VIT_6/conv_4/Conv2D/ReadVariableOpReadVariableOp?wheatclassifier_cnn_vit_6_conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype028
6WheatClassifier_CNN-VIT_6/conv_4/Conv2D/ReadVariableOp�
'WheatClassifier_CNN-VIT_6/conv_4/Conv2DConv2D1WheatClassifier_CNN-VIT_6/conv_3/BiasAdd:output:0>WheatClassifier_CNN-VIT_6/conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^*
paddingVALID*
strides
2)
'WheatClassifier_CNN-VIT_6/conv_4/Conv2D�
7WheatClassifier_CNN-VIT_6/conv_4/BiasAdd/ReadVariableOpReadVariableOp@wheatclassifier_cnn_vit_6_conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7WheatClassifier_CNN-VIT_6/conv_4/BiasAdd/ReadVariableOp�
(WheatClassifier_CNN-VIT_6/conv_4/BiasAddBiasAdd0WheatClassifier_CNN-VIT_6/conv_4/Conv2D:output:0?WheatClassifier_CNN-VIT_6/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^2*
(WheatClassifier_CNN-VIT_6/conv_4/BiasAdd�
+WheatClassifier_CNN-VIT_6/maxpool_2/MaxPoolMaxPool1WheatClassifier_CNN-VIT_6/conv_4/BiasAdd:output:0*/
_output_shapes
:���������//*
ksize
*
paddingVALID*
strides
2-
+WheatClassifier_CNN-VIT_6/maxpool_2/MaxPool�
'WheatClassifier_CNN-VIT_6/patches/ShapeShape4WheatClassifier_CNN-VIT_6/maxpool_2/MaxPool:output:0*
T0*
_output_shapes
:2)
'WheatClassifier_CNN-VIT_6/patches/Shape�
5WheatClassifier_CNN-VIT_6/patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5WheatClassifier_CNN-VIT_6/patches/strided_slice/stack�
7WheatClassifier_CNN-VIT_6/patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7WheatClassifier_CNN-VIT_6/patches/strided_slice/stack_1�
7WheatClassifier_CNN-VIT_6/patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7WheatClassifier_CNN-VIT_6/patches/strided_slice/stack_2�
/WheatClassifier_CNN-VIT_6/patches/strided_sliceStridedSlice0WheatClassifier_CNN-VIT_6/patches/Shape:output:0>WheatClassifier_CNN-VIT_6/patches/strided_slice/stack:output:0@WheatClassifier_CNN-VIT_6/patches/strided_slice/stack_1:output:0@WheatClassifier_CNN-VIT_6/patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/WheatClassifier_CNN-VIT_6/patches/strided_slice�
5WheatClassifier_CNN-VIT_6/patches/ExtractImagePatchesExtractImagePatches4WheatClassifier_CNN-VIT_6/maxpool_2/MaxPool:output:0*
T0*0
_output_shapes
:���������		�*
ksizes
*
paddingVALID*
rates
*
strides
27
5WheatClassifier_CNN-VIT_6/patches/ExtractImagePatches�
1WheatClassifier_CNN-VIT_6/patches/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������23
1WheatClassifier_CNN-VIT_6/patches/Reshape/shape/1�
1WheatClassifier_CNN-VIT_6/patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�23
1WheatClassifier_CNN-VIT_6/patches/Reshape/shape/2�
/WheatClassifier_CNN-VIT_6/patches/Reshape/shapePack8WheatClassifier_CNN-VIT_6/patches/strided_slice:output:0:WheatClassifier_CNN-VIT_6/patches/Reshape/shape/1:output:0:WheatClassifier_CNN-VIT_6/patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:21
/WheatClassifier_CNN-VIT_6/patches/Reshape/shape�
)WheatClassifier_CNN-VIT_6/patches/ReshapeReshape?WheatClassifier_CNN-VIT_6/patches/ExtractImagePatches:patches:08WheatClassifier_CNN-VIT_6/patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:�������������������2+
)WheatClassifier_CNN-VIT_6/patches/Reshape�
3WheatClassifier_CNN-VIT_6/patch_encoder/range/startConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_CNN-VIT_6/patch_encoder/range/start�
3WheatClassifier_CNN-VIT_6/patch_encoder/range/limitConst*
_output_shapes
: *
dtype0*
value	B :Q25
3WheatClassifier_CNN-VIT_6/patch_encoder/range/limit�
3WheatClassifier_CNN-VIT_6/patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :25
3WheatClassifier_CNN-VIT_6/patch_encoder/range/delta�
-WheatClassifier_CNN-VIT_6/patch_encoder/rangeRange<WheatClassifier_CNN-VIT_6/patch_encoder/range/start:output:0<WheatClassifier_CNN-VIT_6/patch_encoder/range/limit:output:0<WheatClassifier_CNN-VIT_6/patch_encoder/range/delta:output:0*
_output_shapes
:Q2/
-WheatClassifier_CNN-VIT_6/patch_encoder/range�
FWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOpOwheatclassifier_cnn_vit_6_patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02H
FWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ReadVariableOp�
<WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2>
<WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/axes�
<WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2>
<WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/free�
=WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ShapeShape2WheatClassifier_CNN-VIT_6/patches/Reshape:output:0*
T0*
_output_shapes
:2?
=WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Shape�
EWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
EWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2/axis�
@WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2GatherV2FWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Shape:output:0EWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/free:output:0NWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2B
@WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2�
GWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
GWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2_1/axis�
BWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2_1GatherV2FWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Shape:output:0EWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/axes:output:0PWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2D
BWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2_1�
=WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2?
=WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Const�
<WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ProdProdIWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2:output:0FWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2>
<WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Prod�
?WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2A
?WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Const_1�
>WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Prod_1ProdKWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2_1:output:0HWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2@
>WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Prod_1�
CWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
CWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat/axis�
>WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concatConcatV2EWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/free:output:0EWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/axes:output:0LWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2@
>WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat�
=WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/stackPackEWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Prod:output:0GWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2?
=WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/stack�
AWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/transpose	Transpose2WheatClassifier_CNN-VIT_6/patches/Reshape:output:0GWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:�������������������2C
AWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/transpose�
?WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ReshapeReshapeEWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/transpose:y:0FWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2A
?WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Reshape�
>WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/MatMulMatMulHWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Reshape:output:0NWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2@
>WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/MatMul�
?WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2A
?WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Const_2�
EWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2G
EWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat_1/axis�
@WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat_1ConcatV2IWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/GatherV2:output:0HWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/Const_2:output:0NWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2B
@WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat_1�
7WheatClassifier_CNN-VIT_6/patch_encoder/dense/TensordotReshapeHWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/MatMul:product:0IWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@29
7WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot�
DWheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOpMwheatclassifier_cnn_vit_6_patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAdd/ReadVariableOp�
5WheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAddBiasAdd@WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot:output:0LWheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@27
5WheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAdd�
BWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookupResourceGatherIwheatclassifier_cnn_vit_6_patch_encoder_embedding_embedding_lookup_4275436WheatClassifier_CNN-VIT_6/patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*\
_classR
PNloc:@WheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup/427543*
_output_shapes

:Q@*
dtype02D
BWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup�
KWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup/IdentityIdentityKWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*\
_classR
PNloc:@WheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup/427543*
_output_shapes

:Q@2M
KWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup/Identity�
MWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup/Identity_1IdentityTWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@2O
MWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup/Identity_1�
+WheatClassifier_CNN-VIT_6/patch_encoder/addAddV2>WheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAdd:output:0VWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������Q@2-
+WheatClassifier_CNN-VIT_6/patch_encoder/add�
LWheatClassifier_CNN-VIT_6/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
LWheatClassifier_CNN-VIT_6/layer_normalization/moments/mean/reduction_indices�
:WheatClassifier_CNN-VIT_6/layer_normalization/moments/meanMean/WheatClassifier_CNN-VIT_6/patch_encoder/add:z:0UWheatClassifier_CNN-VIT_6/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2<
:WheatClassifier_CNN-VIT_6/layer_normalization/moments/mean�
BWheatClassifier_CNN-VIT_6/layer_normalization/moments/StopGradientStopGradientCWheatClassifier_CNN-VIT_6/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2D
BWheatClassifier_CNN-VIT_6/layer_normalization/moments/StopGradient�
GWheatClassifier_CNN-VIT_6/layer_normalization/moments/SquaredDifferenceSquaredDifference/WheatClassifier_CNN-VIT_6/patch_encoder/add:z:0KWheatClassifier_CNN-VIT_6/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2I
GWheatClassifier_CNN-VIT_6/layer_normalization/moments/SquaredDifference�
PWheatClassifier_CNN-VIT_6/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2R
PWheatClassifier_CNN-VIT_6/layer_normalization/moments/variance/reduction_indices�
>WheatClassifier_CNN-VIT_6/layer_normalization/moments/varianceMeanKWheatClassifier_CNN-VIT_6/layer_normalization/moments/SquaredDifference:z:0YWheatClassifier_CNN-VIT_6/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2@
>WheatClassifier_CNN-VIT_6/layer_normalization/moments/variance�
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52?
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add/y�
;WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/addAddV2GWheatClassifier_CNN-VIT_6/layer_normalization/moments/variance:output:0FWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2=
;WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add�
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/RsqrtRsqrt?WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2?
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/Rsqrt�
JWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpSwheatclassifier_cnn_vit_6_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02L
JWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul/ReadVariableOp�
;WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mulMulAWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/Rsqrt:y:0RWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2=
;WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul�
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul_1Mul/WheatClassifier_CNN-VIT_6/patch_encoder/add:z:0?WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul_1�
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul_2MulCWheatClassifier_CNN-VIT_6/layer_normalization/moments/mean:output:0?WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul_2�
FWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/ReadVariableOpReadVariableOpOwheatclassifier_cnn_vit_6_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02H
FWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/ReadVariableOp�
;WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/subSubNWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/ReadVariableOp:value:0AWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2=
;WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/sub�
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add_1AddV2AWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul_1:z:0?WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add_1�
QWheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpZwheatclassifier_cnn_vit_6_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02S
QWheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/Einsum/ReadVariableOp�
BWheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/EinsumEinsumAWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add_1:z:0YWheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2D
BWheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/Einsum�
GWheatClassifier_CNN-VIT_6/multi_head_attention/query/add/ReadVariableOpReadVariableOpPwheatclassifier_cnn_vit_6_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02I
GWheatClassifier_CNN-VIT_6/multi_head_attention/query/add/ReadVariableOp�
8WheatClassifier_CNN-VIT_6/multi_head_attention/query/addAddV2KWheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/Einsum:output:0OWheatClassifier_CNN-VIT_6/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2:
8WheatClassifier_CNN-VIT_6/multi_head_attention/query/add�
OWheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpXwheatclassifier_cnn_vit_6_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02Q
OWheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/Einsum/ReadVariableOp�
@WheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/EinsumEinsumAWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add_1:z:0WWheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2B
@WheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/Einsum�
EWheatClassifier_CNN-VIT_6/multi_head_attention/key/add/ReadVariableOpReadVariableOpNwheatclassifier_cnn_vit_6_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02G
EWheatClassifier_CNN-VIT_6/multi_head_attention/key/add/ReadVariableOp�
6WheatClassifier_CNN-VIT_6/multi_head_attention/key/addAddV2IWheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/Einsum:output:0MWheatClassifier_CNN-VIT_6/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@28
6WheatClassifier_CNN-VIT_6/multi_head_attention/key/add�
QWheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpZwheatclassifier_cnn_vit_6_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02S
QWheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/Einsum/ReadVariableOp�
BWheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/EinsumEinsumAWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/add_1:z:0YWheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2D
BWheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/Einsum�
GWheatClassifier_CNN-VIT_6/multi_head_attention/value/add/ReadVariableOpReadVariableOpPwheatclassifier_cnn_vit_6_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02I
GWheatClassifier_CNN-VIT_6/multi_head_attention/value/add/ReadVariableOp�
8WheatClassifier_CNN-VIT_6/multi_head_attention/value/addAddV2KWheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/Einsum:output:0OWheatClassifier_CNN-VIT_6/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2:
8WheatClassifier_CNN-VIT_6/multi_head_attention/value/add�
4WheatClassifier_CNN-VIT_6/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >26
4WheatClassifier_CNN-VIT_6/multi_head_attention/Mul/y�
2WheatClassifier_CNN-VIT_6/multi_head_attention/MulMul<WheatClassifier_CNN-VIT_6/multi_head_attention/query/add:z:0=WheatClassifier_CNN-VIT_6/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������Q@24
2WheatClassifier_CNN-VIT_6/multi_head_attention/Mul�
<WheatClassifier_CNN-VIT_6/multi_head_attention/einsum/EinsumEinsum:WheatClassifier_CNN-VIT_6/multi_head_attention/key/add:z:06WheatClassifier_CNN-VIT_6/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2>
<WheatClassifier_CNN-VIT_6/multi_head_attention/einsum/Einsum�
>WheatClassifier_CNN-VIT_6/multi_head_attention/softmax/SoftmaxSoftmaxEWheatClassifier_CNN-VIT_6/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2@
>WheatClassifier_CNN-VIT_6/multi_head_attention/softmax/Softmax�
?WheatClassifier_CNN-VIT_6/multi_head_attention/dropout/IdentityIdentityHWheatClassifier_CNN-VIT_6/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������QQ2A
?WheatClassifier_CNN-VIT_6/multi_head_attention/dropout/Identity�
>WheatClassifier_CNN-VIT_6/multi_head_attention/einsum_1/EinsumEinsumHWheatClassifier_CNN-VIT_6/multi_head_attention/dropout/Identity:output:0<WheatClassifier_CNN-VIT_6/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2@
>WheatClassifier_CNN-VIT_6/multi_head_attention/einsum_1/Einsum�
\WheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpewheatclassifier_cnn_vit_6_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02^
\WheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�
MWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/EinsumEinsumGWheatClassifier_CNN-VIT_6/multi_head_attention/einsum_1/Einsum:output:0dWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2O
MWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/Einsum�
RWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOp[wheatclassifier_cnn_vit_6_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02T
RWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/add/ReadVariableOp�
CWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/addAddV2VWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/Einsum:output:0ZWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2E
CWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/add�
!WheatClassifier_CNN-VIT_6/add/addAddV2GWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/add:z:0/WheatClassifier_CNN-VIT_6/patch_encoder/add:z:0*
T0*+
_output_shapes
:���������Q@2#
!WheatClassifier_CNN-VIT_6/add/add�
NWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/mean/reduction_indices�
<WheatClassifier_CNN-VIT_6/layer_normalization_1/moments/meanMean%WheatClassifier_CNN-VIT_6/add/add:z:0WWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2>
<WheatClassifier_CNN-VIT_6/layer_normalization_1/moments/mean�
DWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/StopGradientStopGradientEWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2F
DWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/StopGradient�
IWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/SquaredDifferenceSquaredDifference%WheatClassifier_CNN-VIT_6/add/add:z:0MWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2K
IWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/SquaredDifference�
RWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
RWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/variance/reduction_indices�
@WheatClassifier_CNN-VIT_6/layer_normalization_1/moments/varianceMeanMWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/SquaredDifference:z:0[WheatClassifier_CNN-VIT_6/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2B
@WheatClassifier_CNN-VIT_6/layer_normalization_1/moments/variance�
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52A
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/add/y�
=WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/addAddV2IWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/variance:output:0HWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2?
=WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/add�
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/RsqrtRsqrtAWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2A
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/Rsqrt�
LWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpUwheatclassifier_cnn_vit_6_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02N
LWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul/ReadVariableOp�
=WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mulMulCWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/Rsqrt:y:0TWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul�
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul_1Mul%WheatClassifier_CNN-VIT_6/add/add:z:0AWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul_1�
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul_2MulEWheatClassifier_CNN-VIT_6/layer_normalization_1/moments/mean:output:0AWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul_2�
HWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpQwheatclassifier_cnn_vit_6_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/ReadVariableOp�
=WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/subSubPWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/ReadVariableOp:value:0CWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/sub�
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/add_1AddV2CWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul_1:z:0AWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/add_1�
:WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ReadVariableOpReadVariableOpCwheatclassifier_cnn_vit_6_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02<
:WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ReadVariableOp�
0WheatClassifier_CNN-VIT_6/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0WheatClassifier_CNN-VIT_6/dense_1/Tensordot/axes�
0WheatClassifier_CNN-VIT_6/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0WheatClassifier_CNN-VIT_6/dense_1/Tensordot/free�
1WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ShapeShapeCWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:23
1WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Shape�
9WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2/axis�
4WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2GatherV2:WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Shape:output:09WheatClassifier_CNN-VIT_6/dense_1/Tensordot/free:output:0BWheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2�
;WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2_1/axis�
6WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2_1GatherV2:WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Shape:output:09WheatClassifier_CNN-VIT_6/dense_1/Tensordot/axes:output:0DWheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2_1�
1WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Const�
0WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ProdProd=WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2:output:0:WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Prod�
3WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Const_1�
2WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Prod_1Prod?WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2_1:output:0<WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Prod_1�
7WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat/axis�
2WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concatConcatV29WheatClassifier_CNN-VIT_6/dense_1/Tensordot/free:output:09WheatClassifier_CNN-VIT_6/dense_1/Tensordot/axes:output:0@WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat�
1WheatClassifier_CNN-VIT_6/dense_1/Tensordot/stackPack9WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Prod:output:0;WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1WheatClassifier_CNN-VIT_6/dense_1/Tensordot/stack�
5WheatClassifier_CNN-VIT_6/dense_1/Tensordot/transpose	TransposeCWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/add_1:z:0;WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@27
5WheatClassifier_CNN-VIT_6/dense_1/Tensordot/transpose�
3WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ReshapeReshape9WheatClassifier_CNN-VIT_6/dense_1/Tensordot/transpose:y:0:WheatClassifier_CNN-VIT_6/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������25
3WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Reshape�
2WheatClassifier_CNN-VIT_6/dense_1/Tensordot/MatMulMatMul<WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Reshape:output:0BWheatClassifier_CNN-VIT_6/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������24
2WheatClassifier_CNN-VIT_6/dense_1/Tensordot/MatMul�
3WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�25
3WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Const_2�
9WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat_1/axis�
4WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat_1ConcatV2=WheatClassifier_CNN-VIT_6/dense_1/Tensordot/GatherV2:output:0<WheatClassifier_CNN-VIT_6/dense_1/Tensordot/Const_2:output:0BWheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat_1�
+WheatClassifier_CNN-VIT_6/dense_1/TensordotReshape<WheatClassifier_CNN-VIT_6/dense_1/Tensordot/MatMul:product:0=WheatClassifier_CNN-VIT_6/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2-
+WheatClassifier_CNN-VIT_6/dense_1/Tensordot�
8WheatClassifier_CNN-VIT_6/dense_1/BiasAdd/ReadVariableOpReadVariableOpAwheatclassifier_cnn_vit_6_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8WheatClassifier_CNN-VIT_6/dense_1/BiasAdd/ReadVariableOp�
)WheatClassifier_CNN-VIT_6/dense_1/BiasAddBiasAdd4WheatClassifier_CNN-VIT_6/dense_1/Tensordot:output:0@WheatClassifier_CNN-VIT_6/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2+
)WheatClassifier_CNN-VIT_6/dense_1/BiasAdd�
,WheatClassifier_CNN-VIT_6/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,WheatClassifier_CNN-VIT_6/dense_1/Gelu/mul/x�
*WheatClassifier_CNN-VIT_6/dense_1/Gelu/mulMul5WheatClassifier_CNN-VIT_6/dense_1/Gelu/mul/x:output:02WheatClassifier_CNN-VIT_6/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2,
*WheatClassifier_CNN-VIT_6/dense_1/Gelu/mul�
-WheatClassifier_CNN-VIT_6/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2/
-WheatClassifier_CNN-VIT_6/dense_1/Gelu/Cast/x�
.WheatClassifier_CNN-VIT_6/dense_1/Gelu/truedivRealDiv2WheatClassifier_CNN-VIT_6/dense_1/BiasAdd:output:06WheatClassifier_CNN-VIT_6/dense_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�20
.WheatClassifier_CNN-VIT_6/dense_1/Gelu/truediv�
*WheatClassifier_CNN-VIT_6/dense_1/Gelu/ErfErf2WheatClassifier_CNN-VIT_6/dense_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2,
*WheatClassifier_CNN-VIT_6/dense_1/Gelu/Erf�
,WheatClassifier_CNN-VIT_6/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,WheatClassifier_CNN-VIT_6/dense_1/Gelu/add/x�
*WheatClassifier_CNN-VIT_6/dense_1/Gelu/addAddV25WheatClassifier_CNN-VIT_6/dense_1/Gelu/add/x:output:0.WheatClassifier_CNN-VIT_6/dense_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2,
*WheatClassifier_CNN-VIT_6/dense_1/Gelu/add�
,WheatClassifier_CNN-VIT_6/dense_1/Gelu/mul_1Mul.WheatClassifier_CNN-VIT_6/dense_1/Gelu/mul:z:0.WheatClassifier_CNN-VIT_6/dense_1/Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2.
,WheatClassifier_CNN-VIT_6/dense_1/Gelu/mul_1�
:WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ReadVariableOpReadVariableOpCwheatclassifier_cnn_vit_6_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02<
:WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ReadVariableOp�
0WheatClassifier_CNN-VIT_6/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0WheatClassifier_CNN-VIT_6/dense_2/Tensordot/axes�
0WheatClassifier_CNN-VIT_6/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0WheatClassifier_CNN-VIT_6/dense_2/Tensordot/free�
1WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ShapeShape0WheatClassifier_CNN-VIT_6/dense_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:23
1WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Shape�
9WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2/axis�
4WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2GatherV2:WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Shape:output:09WheatClassifier_CNN-VIT_6/dense_2/Tensordot/free:output:0BWheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2�
;WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2_1/axis�
6WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2_1GatherV2:WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Shape:output:09WheatClassifier_CNN-VIT_6/dense_2/Tensordot/axes:output:0DWheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2_1�
1WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Const�
0WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ProdProd=WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2:output:0:WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Prod�
3WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Const_1�
2WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Prod_1Prod?WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2_1:output:0<WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Prod_1�
7WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat/axis�
2WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concatConcatV29WheatClassifier_CNN-VIT_6/dense_2/Tensordot/free:output:09WheatClassifier_CNN-VIT_6/dense_2/Tensordot/axes:output:0@WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat�
1WheatClassifier_CNN-VIT_6/dense_2/Tensordot/stackPack9WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Prod:output:0;WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1WheatClassifier_CNN-VIT_6/dense_2/Tensordot/stack�
5WheatClassifier_CNN-VIT_6/dense_2/Tensordot/transpose	Transpose0WheatClassifier_CNN-VIT_6/dense_1/Gelu/mul_1:z:0;WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�27
5WheatClassifier_CNN-VIT_6/dense_2/Tensordot/transpose�
3WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ReshapeReshape9WheatClassifier_CNN-VIT_6/dense_2/Tensordot/transpose:y:0:WheatClassifier_CNN-VIT_6/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������25
3WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Reshape�
2WheatClassifier_CNN-VIT_6/dense_2/Tensordot/MatMulMatMul<WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Reshape:output:0BWheatClassifier_CNN-VIT_6/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@24
2WheatClassifier_CNN-VIT_6/dense_2/Tensordot/MatMul�
3WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@25
3WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Const_2�
9WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat_1/axis�
4WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat_1ConcatV2=WheatClassifier_CNN-VIT_6/dense_2/Tensordot/GatherV2:output:0<WheatClassifier_CNN-VIT_6/dense_2/Tensordot/Const_2:output:0BWheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat_1�
+WheatClassifier_CNN-VIT_6/dense_2/TensordotReshape<WheatClassifier_CNN-VIT_6/dense_2/Tensordot/MatMul:product:0=WheatClassifier_CNN-VIT_6/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2-
+WheatClassifier_CNN-VIT_6/dense_2/Tensordot�
8WheatClassifier_CNN-VIT_6/dense_2/BiasAdd/ReadVariableOpReadVariableOpAwheatclassifier_cnn_vit_6_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8WheatClassifier_CNN-VIT_6/dense_2/BiasAdd/ReadVariableOp�
)WheatClassifier_CNN-VIT_6/dense_2/BiasAddBiasAdd4WheatClassifier_CNN-VIT_6/dense_2/Tensordot:output:0@WheatClassifier_CNN-VIT_6/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2+
)WheatClassifier_CNN-VIT_6/dense_2/BiasAdd�
,WheatClassifier_CNN-VIT_6/dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,WheatClassifier_CNN-VIT_6/dense_2/Gelu/mul/x�
*WheatClassifier_CNN-VIT_6/dense_2/Gelu/mulMul5WheatClassifier_CNN-VIT_6/dense_2/Gelu/mul/x:output:02WheatClassifier_CNN-VIT_6/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2,
*WheatClassifier_CNN-VIT_6/dense_2/Gelu/mul�
-WheatClassifier_CNN-VIT_6/dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2/
-WheatClassifier_CNN-VIT_6/dense_2/Gelu/Cast/x�
.WheatClassifier_CNN-VIT_6/dense_2/Gelu/truedivRealDiv2WheatClassifier_CNN-VIT_6/dense_2/BiasAdd:output:06WheatClassifier_CNN-VIT_6/dense_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@20
.WheatClassifier_CNN-VIT_6/dense_2/Gelu/truediv�
*WheatClassifier_CNN-VIT_6/dense_2/Gelu/ErfErf2WheatClassifier_CNN-VIT_6/dense_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2,
*WheatClassifier_CNN-VIT_6/dense_2/Gelu/Erf�
,WheatClassifier_CNN-VIT_6/dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,WheatClassifier_CNN-VIT_6/dense_2/Gelu/add/x�
*WheatClassifier_CNN-VIT_6/dense_2/Gelu/addAddV25WheatClassifier_CNN-VIT_6/dense_2/Gelu/add/x:output:0.WheatClassifier_CNN-VIT_6/dense_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2,
*WheatClassifier_CNN-VIT_6/dense_2/Gelu/add�
,WheatClassifier_CNN-VIT_6/dense_2/Gelu/mul_1Mul.WheatClassifier_CNN-VIT_6/dense_2/Gelu/mul:z:0.WheatClassifier_CNN-VIT_6/dense_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2.
,WheatClassifier_CNN-VIT_6/dense_2/Gelu/mul_1�
#WheatClassifier_CNN-VIT_6/add_1/addAddV20WheatClassifier_CNN-VIT_6/dense_2/Gelu/mul_1:z:0%WheatClassifier_CNN-VIT_6/add/add:z:0*
T0*+
_output_shapes
:���������Q@2%
#WheatClassifier_CNN-VIT_6/add_1/add�
NWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/mean/reduction_indices�
<WheatClassifier_CNN-VIT_6/layer_normalization_2/moments/meanMean'WheatClassifier_CNN-VIT_6/add_1/add:z:0WWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2>
<WheatClassifier_CNN-VIT_6/layer_normalization_2/moments/mean�
DWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/StopGradientStopGradientEWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2F
DWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/StopGradient�
IWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/SquaredDifferenceSquaredDifference'WheatClassifier_CNN-VIT_6/add_1/add:z:0MWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2K
IWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/SquaredDifference�
RWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
RWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/variance/reduction_indices�
@WheatClassifier_CNN-VIT_6/layer_normalization_2/moments/varianceMeanMWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/SquaredDifference:z:0[WheatClassifier_CNN-VIT_6/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2B
@WheatClassifier_CNN-VIT_6/layer_normalization_2/moments/variance�
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52A
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add/y�
=WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/addAddV2IWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/variance:output:0HWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2?
=WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add�
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/RsqrtRsqrtAWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2A
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/Rsqrt�
LWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpUwheatclassifier_cnn_vit_6_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02N
LWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul/ReadVariableOp�
=WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mulMulCWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/Rsqrt:y:0TWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul�
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul_1Mul'WheatClassifier_CNN-VIT_6/add_1/add:z:0AWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul_1�
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul_2MulEWheatClassifier_CNN-VIT_6/layer_normalization_2/moments/mean:output:0AWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul_2�
HWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpQwheatclassifier_cnn_vit_6_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/ReadVariableOp�
=WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/subSubPWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/ReadVariableOp:value:0CWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/sub�
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add_1AddV2CWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul_1:z:0AWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add_1�
SWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOp\wheatclassifier_cnn_vit_6_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02U
SWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp�
DWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/EinsumEinsumCWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add_1:z:0[WheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2F
DWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/Einsum�
IWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpRwheatclassifier_cnn_vit_6_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02K
IWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/add/ReadVariableOp�
:WheatClassifier_CNN-VIT_6/multi_head_attention_1/query/addAddV2MWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/Einsum:output:0QWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2<
:WheatClassifier_CNN-VIT_6/multi_head_attention_1/query/add�
QWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpZwheatclassifier_cnn_vit_6_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02S
QWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp�
BWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/EinsumEinsumCWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add_1:z:0YWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2D
BWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/Einsum�
GWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpPwheatclassifier_cnn_vit_6_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02I
GWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/add/ReadVariableOp�
8WheatClassifier_CNN-VIT_6/multi_head_attention_1/key/addAddV2KWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/Einsum:output:0OWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2:
8WheatClassifier_CNN-VIT_6/multi_head_attention_1/key/add�
SWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOp\wheatclassifier_cnn_vit_6_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02U
SWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp�
DWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/EinsumEinsumCWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/add_1:z:0[WheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2F
DWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/Einsum�
IWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpRwheatclassifier_cnn_vit_6_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02K
IWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/add/ReadVariableOp�
:WheatClassifier_CNN-VIT_6/multi_head_attention_1/value/addAddV2MWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/Einsum:output:0QWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2<
:WheatClassifier_CNN-VIT_6/multi_head_attention_1/value/add�
6WheatClassifier_CNN-VIT_6/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >28
6WheatClassifier_CNN-VIT_6/multi_head_attention_1/Mul/y�
4WheatClassifier_CNN-VIT_6/multi_head_attention_1/MulMul>WheatClassifier_CNN-VIT_6/multi_head_attention_1/query/add:z:0?WheatClassifier_CNN-VIT_6/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:���������Q@26
4WheatClassifier_CNN-VIT_6/multi_head_attention_1/Mul�
>WheatClassifier_CNN-VIT_6/multi_head_attention_1/einsum/EinsumEinsum<WheatClassifier_CNN-VIT_6/multi_head_attention_1/key/add:z:08WheatClassifier_CNN-VIT_6/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2@
>WheatClassifier_CNN-VIT_6/multi_head_attention_1/einsum/Einsum�
@WheatClassifier_CNN-VIT_6/multi_head_attention_1/softmax/SoftmaxSoftmaxGWheatClassifier_CNN-VIT_6/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2B
@WheatClassifier_CNN-VIT_6/multi_head_attention_1/softmax/Softmax�
AWheatClassifier_CNN-VIT_6/multi_head_attention_1/dropout/IdentityIdentityJWheatClassifier_CNN-VIT_6/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������QQ2C
AWheatClassifier_CNN-VIT_6/multi_head_attention_1/dropout/Identity�
@WheatClassifier_CNN-VIT_6/multi_head_attention_1/einsum_1/EinsumEinsumJWheatClassifier_CNN-VIT_6/multi_head_attention_1/dropout/Identity:output:0>WheatClassifier_CNN-VIT_6/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2B
@WheatClassifier_CNN-VIT_6/multi_head_attention_1/einsum_1/Einsum�
^WheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpgwheatclassifier_cnn_vit_6_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02`
^WheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp�
OWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/EinsumEinsumIWheatClassifier_CNN-VIT_6/multi_head_attention_1/einsum_1/Einsum:output:0fWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2Q
OWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/Einsum�
TWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOp]wheatclassifier_cnn_vit_6_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02V
TWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/add/ReadVariableOp�
EWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/addAddV2XWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/Einsum:output:0\WheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2G
EWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/add�
#WheatClassifier_CNN-VIT_6/add_2/addAddV2IWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/add:z:0'WheatClassifier_CNN-VIT_6/add_1/add:z:0*
T0*+
_output_shapes
:���������Q@2%
#WheatClassifier_CNN-VIT_6/add_2/add�
NWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/mean/reduction_indices�
<WheatClassifier_CNN-VIT_6/layer_normalization_3/moments/meanMean'WheatClassifier_CNN-VIT_6/add_2/add:z:0WWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2>
<WheatClassifier_CNN-VIT_6/layer_normalization_3/moments/mean�
DWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/StopGradientStopGradientEWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2F
DWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/StopGradient�
IWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/SquaredDifferenceSquaredDifference'WheatClassifier_CNN-VIT_6/add_2/add:z:0MWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2K
IWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/SquaredDifference�
RWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
RWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/variance/reduction_indices�
@WheatClassifier_CNN-VIT_6/layer_normalization_3/moments/varianceMeanMWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/SquaredDifference:z:0[WheatClassifier_CNN-VIT_6/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2B
@WheatClassifier_CNN-VIT_6/layer_normalization_3/moments/variance�
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52A
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/add/y�
=WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/addAddV2IWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/variance:output:0HWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2?
=WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/add�
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/RsqrtRsqrtAWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2A
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/Rsqrt�
LWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpUwheatclassifier_cnn_vit_6_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02N
LWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul/ReadVariableOp�
=WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mulMulCWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/Rsqrt:y:0TWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul�
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul_1Mul'WheatClassifier_CNN-VIT_6/add_2/add:z:0AWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul_1�
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul_2MulEWheatClassifier_CNN-VIT_6/layer_normalization_3/moments/mean:output:0AWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul_2�
HWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpQwheatclassifier_cnn_vit_6_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/ReadVariableOp�
=WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/subSubPWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/ReadVariableOp:value:0CWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/sub�
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/add_1AddV2CWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul_1:z:0AWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/add_1�
:WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ReadVariableOpReadVariableOpCwheatclassifier_cnn_vit_6_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02<
:WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ReadVariableOp�
0WheatClassifier_CNN-VIT_6/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0WheatClassifier_CNN-VIT_6/dense_3/Tensordot/axes�
0WheatClassifier_CNN-VIT_6/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0WheatClassifier_CNN-VIT_6/dense_3/Tensordot/free�
1WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ShapeShapeCWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:23
1WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Shape�
9WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2/axis�
4WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2GatherV2:WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Shape:output:09WheatClassifier_CNN-VIT_6/dense_3/Tensordot/free:output:0BWheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2�
;WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2_1/axis�
6WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2_1GatherV2:WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Shape:output:09WheatClassifier_CNN-VIT_6/dense_3/Tensordot/axes:output:0DWheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2_1�
1WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Const�
0WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ProdProd=WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2:output:0:WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Prod�
3WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Const_1�
2WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Prod_1Prod?WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2_1:output:0<WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Prod_1�
7WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat/axis�
2WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concatConcatV29WheatClassifier_CNN-VIT_6/dense_3/Tensordot/free:output:09WheatClassifier_CNN-VIT_6/dense_3/Tensordot/axes:output:0@WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat�
1WheatClassifier_CNN-VIT_6/dense_3/Tensordot/stackPack9WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Prod:output:0;WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1WheatClassifier_CNN-VIT_6/dense_3/Tensordot/stack�
5WheatClassifier_CNN-VIT_6/dense_3/Tensordot/transpose	TransposeCWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/add_1:z:0;WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@27
5WheatClassifier_CNN-VIT_6/dense_3/Tensordot/transpose�
3WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ReshapeReshape9WheatClassifier_CNN-VIT_6/dense_3/Tensordot/transpose:y:0:WheatClassifier_CNN-VIT_6/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������25
3WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Reshape�
2WheatClassifier_CNN-VIT_6/dense_3/Tensordot/MatMulMatMul<WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Reshape:output:0BWheatClassifier_CNN-VIT_6/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������24
2WheatClassifier_CNN-VIT_6/dense_3/Tensordot/MatMul�
3WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�25
3WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Const_2�
9WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat_1/axis�
4WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat_1ConcatV2=WheatClassifier_CNN-VIT_6/dense_3/Tensordot/GatherV2:output:0<WheatClassifier_CNN-VIT_6/dense_3/Tensordot/Const_2:output:0BWheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat_1�
+WheatClassifier_CNN-VIT_6/dense_3/TensordotReshape<WheatClassifier_CNN-VIT_6/dense_3/Tensordot/MatMul:product:0=WheatClassifier_CNN-VIT_6/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2-
+WheatClassifier_CNN-VIT_6/dense_3/Tensordot�
8WheatClassifier_CNN-VIT_6/dense_3/BiasAdd/ReadVariableOpReadVariableOpAwheatclassifier_cnn_vit_6_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02:
8WheatClassifier_CNN-VIT_6/dense_3/BiasAdd/ReadVariableOp�
)WheatClassifier_CNN-VIT_6/dense_3/BiasAddBiasAdd4WheatClassifier_CNN-VIT_6/dense_3/Tensordot:output:0@WheatClassifier_CNN-VIT_6/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2+
)WheatClassifier_CNN-VIT_6/dense_3/BiasAdd�
,WheatClassifier_CNN-VIT_6/dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,WheatClassifier_CNN-VIT_6/dense_3/Gelu/mul/x�
*WheatClassifier_CNN-VIT_6/dense_3/Gelu/mulMul5WheatClassifier_CNN-VIT_6/dense_3/Gelu/mul/x:output:02WheatClassifier_CNN-VIT_6/dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2,
*WheatClassifier_CNN-VIT_6/dense_3/Gelu/mul�
-WheatClassifier_CNN-VIT_6/dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2/
-WheatClassifier_CNN-VIT_6/dense_3/Gelu/Cast/x�
.WheatClassifier_CNN-VIT_6/dense_3/Gelu/truedivRealDiv2WheatClassifier_CNN-VIT_6/dense_3/BiasAdd:output:06WheatClassifier_CNN-VIT_6/dense_3/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�20
.WheatClassifier_CNN-VIT_6/dense_3/Gelu/truediv�
*WheatClassifier_CNN-VIT_6/dense_3/Gelu/ErfErf2WheatClassifier_CNN-VIT_6/dense_3/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2,
*WheatClassifier_CNN-VIT_6/dense_3/Gelu/Erf�
,WheatClassifier_CNN-VIT_6/dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,WheatClassifier_CNN-VIT_6/dense_3/Gelu/add/x�
*WheatClassifier_CNN-VIT_6/dense_3/Gelu/addAddV25WheatClassifier_CNN-VIT_6/dense_3/Gelu/add/x:output:0.WheatClassifier_CNN-VIT_6/dense_3/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2,
*WheatClassifier_CNN-VIT_6/dense_3/Gelu/add�
,WheatClassifier_CNN-VIT_6/dense_3/Gelu/mul_1Mul.WheatClassifier_CNN-VIT_6/dense_3/Gelu/mul:z:0.WheatClassifier_CNN-VIT_6/dense_3/Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2.
,WheatClassifier_CNN-VIT_6/dense_3/Gelu/mul_1�
:WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ReadVariableOpReadVariableOpCwheatclassifier_cnn_vit_6_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02<
:WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ReadVariableOp�
0WheatClassifier_CNN-VIT_6/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0WheatClassifier_CNN-VIT_6/dense_4/Tensordot/axes�
0WheatClassifier_CNN-VIT_6/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0WheatClassifier_CNN-VIT_6/dense_4/Tensordot/free�
1WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ShapeShape0WheatClassifier_CNN-VIT_6/dense_3/Gelu/mul_1:z:0*
T0*
_output_shapes
:23
1WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Shape�
9WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2/axis�
4WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2GatherV2:WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Shape:output:09WheatClassifier_CNN-VIT_6/dense_4/Tensordot/free:output:0BWheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2�
;WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2_1/axis�
6WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2_1GatherV2:WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Shape:output:09WheatClassifier_CNN-VIT_6/dense_4/Tensordot/axes:output:0DWheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2_1�
1WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Const�
0WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ProdProd=WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2:output:0:WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Prod�
3WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Const_1�
2WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Prod_1Prod?WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2_1:output:0<WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Prod_1�
7WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat/axis�
2WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concatConcatV29WheatClassifier_CNN-VIT_6/dense_4/Tensordot/free:output:09WheatClassifier_CNN-VIT_6/dense_4/Tensordot/axes:output:0@WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat�
1WheatClassifier_CNN-VIT_6/dense_4/Tensordot/stackPack9WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Prod:output:0;WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1WheatClassifier_CNN-VIT_6/dense_4/Tensordot/stack�
5WheatClassifier_CNN-VIT_6/dense_4/Tensordot/transpose	Transpose0WheatClassifier_CNN-VIT_6/dense_3/Gelu/mul_1:z:0;WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�27
5WheatClassifier_CNN-VIT_6/dense_4/Tensordot/transpose�
3WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ReshapeReshape9WheatClassifier_CNN-VIT_6/dense_4/Tensordot/transpose:y:0:WheatClassifier_CNN-VIT_6/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������25
3WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Reshape�
2WheatClassifier_CNN-VIT_6/dense_4/Tensordot/MatMulMatMul<WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Reshape:output:0BWheatClassifier_CNN-VIT_6/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@24
2WheatClassifier_CNN-VIT_6/dense_4/Tensordot/MatMul�
3WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@25
3WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Const_2�
9WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat_1/axis�
4WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat_1ConcatV2=WheatClassifier_CNN-VIT_6/dense_4/Tensordot/GatherV2:output:0<WheatClassifier_CNN-VIT_6/dense_4/Tensordot/Const_2:output:0BWheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat_1�
+WheatClassifier_CNN-VIT_6/dense_4/TensordotReshape<WheatClassifier_CNN-VIT_6/dense_4/Tensordot/MatMul:product:0=WheatClassifier_CNN-VIT_6/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2-
+WheatClassifier_CNN-VIT_6/dense_4/Tensordot�
8WheatClassifier_CNN-VIT_6/dense_4/BiasAdd/ReadVariableOpReadVariableOpAwheatclassifier_cnn_vit_6_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8WheatClassifier_CNN-VIT_6/dense_4/BiasAdd/ReadVariableOp�
)WheatClassifier_CNN-VIT_6/dense_4/BiasAddBiasAdd4WheatClassifier_CNN-VIT_6/dense_4/Tensordot:output:0@WheatClassifier_CNN-VIT_6/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2+
)WheatClassifier_CNN-VIT_6/dense_4/BiasAdd�
,WheatClassifier_CNN-VIT_6/dense_4/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,WheatClassifier_CNN-VIT_6/dense_4/Gelu/mul/x�
*WheatClassifier_CNN-VIT_6/dense_4/Gelu/mulMul5WheatClassifier_CNN-VIT_6/dense_4/Gelu/mul/x:output:02WheatClassifier_CNN-VIT_6/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2,
*WheatClassifier_CNN-VIT_6/dense_4/Gelu/mul�
-WheatClassifier_CNN-VIT_6/dense_4/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2/
-WheatClassifier_CNN-VIT_6/dense_4/Gelu/Cast/x�
.WheatClassifier_CNN-VIT_6/dense_4/Gelu/truedivRealDiv2WheatClassifier_CNN-VIT_6/dense_4/BiasAdd:output:06WheatClassifier_CNN-VIT_6/dense_4/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@20
.WheatClassifier_CNN-VIT_6/dense_4/Gelu/truediv�
*WheatClassifier_CNN-VIT_6/dense_4/Gelu/ErfErf2WheatClassifier_CNN-VIT_6/dense_4/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2,
*WheatClassifier_CNN-VIT_6/dense_4/Gelu/Erf�
,WheatClassifier_CNN-VIT_6/dense_4/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2.
,WheatClassifier_CNN-VIT_6/dense_4/Gelu/add/x�
*WheatClassifier_CNN-VIT_6/dense_4/Gelu/addAddV25WheatClassifier_CNN-VIT_6/dense_4/Gelu/add/x:output:0.WheatClassifier_CNN-VIT_6/dense_4/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2,
*WheatClassifier_CNN-VIT_6/dense_4/Gelu/add�
,WheatClassifier_CNN-VIT_6/dense_4/Gelu/mul_1Mul.WheatClassifier_CNN-VIT_6/dense_4/Gelu/mul:z:0.WheatClassifier_CNN-VIT_6/dense_4/Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2.
,WheatClassifier_CNN-VIT_6/dense_4/Gelu/mul_1�
#WheatClassifier_CNN-VIT_6/add_3/addAddV20WheatClassifier_CNN-VIT_6/dense_4/Gelu/mul_1:z:0'WheatClassifier_CNN-VIT_6/add_2/add:z:0*
T0*+
_output_shapes
:���������Q@2%
#WheatClassifier_CNN-VIT_6/add_3/add�
NWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/mean/reduction_indices�
<WheatClassifier_CNN-VIT_6/layer_normalization_4/moments/meanMean'WheatClassifier_CNN-VIT_6/add_3/add:z:0WWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2>
<WheatClassifier_CNN-VIT_6/layer_normalization_4/moments/mean�
DWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/StopGradientStopGradientEWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2F
DWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/StopGradient�
IWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/SquaredDifferenceSquaredDifference'WheatClassifier_CNN-VIT_6/add_3/add:z:0MWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2K
IWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/SquaredDifference�
RWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2T
RWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/variance/reduction_indices�
@WheatClassifier_CNN-VIT_6/layer_normalization_4/moments/varianceMeanMWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/SquaredDifference:z:0[WheatClassifier_CNN-VIT_6/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2B
@WheatClassifier_CNN-VIT_6/layer_normalization_4/moments/variance�
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52A
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/add/y�
=WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/addAddV2IWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/variance:output:0HWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2?
=WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/add�
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/RsqrtRsqrtAWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2A
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/Rsqrt�
LWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpUwheatclassifier_cnn_vit_6_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02N
LWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul/ReadVariableOp�
=WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mulMulCWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/Rsqrt:y:0TWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul�
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul_1Mul'WheatClassifier_CNN-VIT_6/add_3/add:z:0AWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul_1�
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul_2MulEWheatClassifier_CNN-VIT_6/layer_normalization_4/moments/mean:output:0AWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul_2�
HWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpQwheatclassifier_cnn_vit_6_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/ReadVariableOp�
=WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/subSubPWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/ReadVariableOp:value:0CWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2?
=WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/sub�
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/add_1AddV2CWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul_1:z:0AWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2A
?WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/add_1�
-WheatClassifier_CNN-VIT_6/flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2/
-WheatClassifier_CNN-VIT_6/flatten_layer/Const�
/WheatClassifier_CNN-VIT_6/flatten_layer/ReshapeReshapeCWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/add_1:z:06WheatClassifier_CNN-VIT_6/flatten_layer/Const:output:0*
T0*(
_output_shapes
:����������(21
/WheatClassifier_CNN-VIT_6/flatten_layer/Reshape�
4WheatClassifier_CNN-VIT_6/FC_1/MatMul/ReadVariableOpReadVariableOp=wheatclassifier_cnn_vit_6_fc_1_matmul_readvariableop_resource*
_output_shapes
:	�(2*
dtype026
4WheatClassifier_CNN-VIT_6/FC_1/MatMul/ReadVariableOp�
%WheatClassifier_CNN-VIT_6/FC_1/MatMulMatMul8WheatClassifier_CNN-VIT_6/flatten_layer/Reshape:output:0<WheatClassifier_CNN-VIT_6/FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22'
%WheatClassifier_CNN-VIT_6/FC_1/MatMul�
5WheatClassifier_CNN-VIT_6/FC_1/BiasAdd/ReadVariableOpReadVariableOp>wheatclassifier_cnn_vit_6_fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype027
5WheatClassifier_CNN-VIT_6/FC_1/BiasAdd/ReadVariableOp�
&WheatClassifier_CNN-VIT_6/FC_1/BiasAddBiasAdd/WheatClassifier_CNN-VIT_6/FC_1/MatMul:product:0=WheatClassifier_CNN-VIT_6/FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22(
&WheatClassifier_CNN-VIT_6/FC_1/BiasAdd�
0WheatClassifier_CNN-VIT_6/leaky_ReLu_1/LeakyRelu	LeakyRelu/WheatClassifier_CNN-VIT_6/FC_1/BiasAdd:output:0*'
_output_shapes
:���������2*
alpha%���>22
0WheatClassifier_CNN-VIT_6/leaky_ReLu_1/LeakyRelu�
4WheatClassifier_CNN-VIT_6/FC_2/MatMul/ReadVariableOpReadVariableOp=wheatclassifier_cnn_vit_6_fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype026
4WheatClassifier_CNN-VIT_6/FC_2/MatMul/ReadVariableOp�
%WheatClassifier_CNN-VIT_6/FC_2/MatMulMatMul>WheatClassifier_CNN-VIT_6/leaky_ReLu_1/LeakyRelu:activations:0<WheatClassifier_CNN-VIT_6/FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22'
%WheatClassifier_CNN-VIT_6/FC_2/MatMul�
5WheatClassifier_CNN-VIT_6/FC_2/BiasAdd/ReadVariableOpReadVariableOp>wheatclassifier_cnn_vit_6_fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype027
5WheatClassifier_CNN-VIT_6/FC_2/BiasAdd/ReadVariableOp�
&WheatClassifier_CNN-VIT_6/FC_2/BiasAddBiasAdd/WheatClassifier_CNN-VIT_6/FC_2/MatMul:product:0=WheatClassifier_CNN-VIT_6/FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22(
&WheatClassifier_CNN-VIT_6/FC_2/BiasAdd�
0WheatClassifier_CNN-VIT_6/leaky_ReLu_2/LeakyRelu	LeakyRelu/WheatClassifier_CNN-VIT_6/FC_2/BiasAdd:output:0*'
_output_shapes
:���������2*
alpha%���>22
0WheatClassifier_CNN-VIT_6/leaky_ReLu_2/LeakyRelu�
<WheatClassifier_CNN-VIT_6/output_layer/MatMul/ReadVariableOpReadVariableOpEwheatclassifier_cnn_vit_6_output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02>
<WheatClassifier_CNN-VIT_6/output_layer/MatMul/ReadVariableOp�
-WheatClassifier_CNN-VIT_6/output_layer/MatMulMatMul>WheatClassifier_CNN-VIT_6/leaky_ReLu_2/LeakyRelu:activations:0DWheatClassifier_CNN-VIT_6/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2/
-WheatClassifier_CNN-VIT_6/output_layer/MatMul�
=WheatClassifier_CNN-VIT_6/output_layer/BiasAdd/ReadVariableOpReadVariableOpFwheatclassifier_cnn_vit_6_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=WheatClassifier_CNN-VIT_6/output_layer/BiasAdd/ReadVariableOp�
.WheatClassifier_CNN-VIT_6/output_layer/BiasAddBiasAdd7WheatClassifier_CNN-VIT_6/output_layer/MatMul:product:0EWheatClassifier_CNN-VIT_6/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������20
.WheatClassifier_CNN-VIT_6/output_layer/BiasAdd�
.WheatClassifier_CNN-VIT_6/output_layer/SoftmaxSoftmax7WheatClassifier_CNN-VIT_6/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������20
.WheatClassifier_CNN-VIT_6/output_layer/Softmax�
IdentityIdentity8WheatClassifier_CNN-VIT_6/output_layer/Softmax:softmax:06^WheatClassifier_CNN-VIT_6/FC_1/BiasAdd/ReadVariableOp5^WheatClassifier_CNN-VIT_6/FC_1/MatMul/ReadVariableOp6^WheatClassifier_CNN-VIT_6/FC_2/BiasAdd/ReadVariableOp5^WheatClassifier_CNN-VIT_6/FC_2/MatMul/ReadVariableOp8^WheatClassifier_CNN-VIT_6/conv_1/BiasAdd/ReadVariableOp7^WheatClassifier_CNN-VIT_6/conv_1/Conv2D/ReadVariableOp8^WheatClassifier_CNN-VIT_6/conv_2/BiasAdd/ReadVariableOp7^WheatClassifier_CNN-VIT_6/conv_2/Conv2D/ReadVariableOp8^WheatClassifier_CNN-VIT_6/conv_3/BiasAdd/ReadVariableOp7^WheatClassifier_CNN-VIT_6/conv_3/Conv2D/ReadVariableOp8^WheatClassifier_CNN-VIT_6/conv_4/BiasAdd/ReadVariableOp7^WheatClassifier_CNN-VIT_6/conv_4/Conv2D/ReadVariableOp9^WheatClassifier_CNN-VIT_6/dense_1/BiasAdd/ReadVariableOp;^WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ReadVariableOp9^WheatClassifier_CNN-VIT_6/dense_2/BiasAdd/ReadVariableOp;^WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ReadVariableOp9^WheatClassifier_CNN-VIT_6/dense_3/BiasAdd/ReadVariableOp;^WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ReadVariableOp9^WheatClassifier_CNN-VIT_6/dense_4/BiasAdd/ReadVariableOp;^WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ReadVariableOpG^WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/ReadVariableOpK^WheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul/ReadVariableOpI^WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/ReadVariableOpM^WheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul/ReadVariableOpI^WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/ReadVariableOpM^WheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul/ReadVariableOpI^WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/ReadVariableOpM^WheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul/ReadVariableOpI^WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/ReadVariableOpM^WheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul/ReadVariableOpS^WheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/add/ReadVariableOp]^WheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpF^WheatClassifier_CNN-VIT_6/multi_head_attention/key/add/ReadVariableOpP^WheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/Einsum/ReadVariableOpH^WheatClassifier_CNN-VIT_6/multi_head_attention/query/add/ReadVariableOpR^WheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/Einsum/ReadVariableOpH^WheatClassifier_CNN-VIT_6/multi_head_attention/value/add/ReadVariableOpR^WheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/Einsum/ReadVariableOpU^WheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/add/ReadVariableOp_^WheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpH^WheatClassifier_CNN-VIT_6/multi_head_attention_1/key/add/ReadVariableOpR^WheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpJ^WheatClassifier_CNN-VIT_6/multi_head_attention_1/query/add/ReadVariableOpT^WheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpJ^WheatClassifier_CNN-VIT_6/multi_head_attention_1/value/add/ReadVariableOpT^WheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp>^WheatClassifier_CNN-VIT_6/output_layer/BiasAdd/ReadVariableOp=^WheatClassifier_CNN-VIT_6/output_layer/MatMul/ReadVariableOpE^WheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAdd/ReadVariableOpG^WheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ReadVariableOpC^WheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5WheatClassifier_CNN-VIT_6/FC_1/BiasAdd/ReadVariableOp5WheatClassifier_CNN-VIT_6/FC_1/BiasAdd/ReadVariableOp2l
4WheatClassifier_CNN-VIT_6/FC_1/MatMul/ReadVariableOp4WheatClassifier_CNN-VIT_6/FC_1/MatMul/ReadVariableOp2n
5WheatClassifier_CNN-VIT_6/FC_2/BiasAdd/ReadVariableOp5WheatClassifier_CNN-VIT_6/FC_2/BiasAdd/ReadVariableOp2l
4WheatClassifier_CNN-VIT_6/FC_2/MatMul/ReadVariableOp4WheatClassifier_CNN-VIT_6/FC_2/MatMul/ReadVariableOp2r
7WheatClassifier_CNN-VIT_6/conv_1/BiasAdd/ReadVariableOp7WheatClassifier_CNN-VIT_6/conv_1/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN-VIT_6/conv_1/Conv2D/ReadVariableOp6WheatClassifier_CNN-VIT_6/conv_1/Conv2D/ReadVariableOp2r
7WheatClassifier_CNN-VIT_6/conv_2/BiasAdd/ReadVariableOp7WheatClassifier_CNN-VIT_6/conv_2/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN-VIT_6/conv_2/Conv2D/ReadVariableOp6WheatClassifier_CNN-VIT_6/conv_2/Conv2D/ReadVariableOp2r
7WheatClassifier_CNN-VIT_6/conv_3/BiasAdd/ReadVariableOp7WheatClassifier_CNN-VIT_6/conv_3/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN-VIT_6/conv_3/Conv2D/ReadVariableOp6WheatClassifier_CNN-VIT_6/conv_3/Conv2D/ReadVariableOp2r
7WheatClassifier_CNN-VIT_6/conv_4/BiasAdd/ReadVariableOp7WheatClassifier_CNN-VIT_6/conv_4/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN-VIT_6/conv_4/Conv2D/ReadVariableOp6WheatClassifier_CNN-VIT_6/conv_4/Conv2D/ReadVariableOp2t
8WheatClassifier_CNN-VIT_6/dense_1/BiasAdd/ReadVariableOp8WheatClassifier_CNN-VIT_6/dense_1/BiasAdd/ReadVariableOp2x
:WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ReadVariableOp:WheatClassifier_CNN-VIT_6/dense_1/Tensordot/ReadVariableOp2t
8WheatClassifier_CNN-VIT_6/dense_2/BiasAdd/ReadVariableOp8WheatClassifier_CNN-VIT_6/dense_2/BiasAdd/ReadVariableOp2x
:WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ReadVariableOp:WheatClassifier_CNN-VIT_6/dense_2/Tensordot/ReadVariableOp2t
8WheatClassifier_CNN-VIT_6/dense_3/BiasAdd/ReadVariableOp8WheatClassifier_CNN-VIT_6/dense_3/BiasAdd/ReadVariableOp2x
:WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ReadVariableOp:WheatClassifier_CNN-VIT_6/dense_3/Tensordot/ReadVariableOp2t
8WheatClassifier_CNN-VIT_6/dense_4/BiasAdd/ReadVariableOp8WheatClassifier_CNN-VIT_6/dense_4/BiasAdd/ReadVariableOp2x
:WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ReadVariableOp:WheatClassifier_CNN-VIT_6/dense_4/Tensordot/ReadVariableOp2�
FWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/ReadVariableOpFWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/ReadVariableOp2�
JWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul/ReadVariableOpJWheatClassifier_CNN-VIT_6/layer_normalization/batchnorm/mul/ReadVariableOp2�
HWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/ReadVariableOpHWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/ReadVariableOp2�
LWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul/ReadVariableOpLWheatClassifier_CNN-VIT_6/layer_normalization_1/batchnorm/mul/ReadVariableOp2�
HWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/ReadVariableOpHWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/ReadVariableOp2�
LWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul/ReadVariableOpLWheatClassifier_CNN-VIT_6/layer_normalization_2/batchnorm/mul/ReadVariableOp2�
HWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/ReadVariableOpHWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/ReadVariableOp2�
LWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul/ReadVariableOpLWheatClassifier_CNN-VIT_6/layer_normalization_3/batchnorm/mul/ReadVariableOp2�
HWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/ReadVariableOpHWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/ReadVariableOp2�
LWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul/ReadVariableOpLWheatClassifier_CNN-VIT_6/layer_normalization_4/batchnorm/mul/ReadVariableOp2�
RWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/add/ReadVariableOpRWheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/add/ReadVariableOp2�
\WheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp\WheatClassifier_CNN-VIT_6/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2�
EWheatClassifier_CNN-VIT_6/multi_head_attention/key/add/ReadVariableOpEWheatClassifier_CNN-VIT_6/multi_head_attention/key/add/ReadVariableOp2�
OWheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/Einsum/ReadVariableOpOWheatClassifier_CNN-VIT_6/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
GWheatClassifier_CNN-VIT_6/multi_head_attention/query/add/ReadVariableOpGWheatClassifier_CNN-VIT_6/multi_head_attention/query/add/ReadVariableOp2�
QWheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/Einsum/ReadVariableOpQWheatClassifier_CNN-VIT_6/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
GWheatClassifier_CNN-VIT_6/multi_head_attention/value/add/ReadVariableOpGWheatClassifier_CNN-VIT_6/multi_head_attention/value/add/ReadVariableOp2�
QWheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/Einsum/ReadVariableOpQWheatClassifier_CNN-VIT_6/multi_head_attention/value/einsum/Einsum/ReadVariableOp2�
TWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/add/ReadVariableOpTWheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/add/ReadVariableOp2�
^WheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp^WheatClassifier_CNN-VIT_6/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2�
GWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/add/ReadVariableOpGWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/add/ReadVariableOp2�
QWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpQWheatClassifier_CNN-VIT_6/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2�
IWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/add/ReadVariableOpIWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/add/ReadVariableOp2�
SWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpSWheatClassifier_CNN-VIT_6/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2�
IWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/add/ReadVariableOpIWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/add/ReadVariableOp2�
SWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpSWheatClassifier_CNN-VIT_6/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2~
=WheatClassifier_CNN-VIT_6/output_layer/BiasAdd/ReadVariableOp=WheatClassifier_CNN-VIT_6/output_layer/BiasAdd/ReadVariableOp2|
<WheatClassifier_CNN-VIT_6/output_layer/MatMul/ReadVariableOp<WheatClassifier_CNN-VIT_6/output_layer/MatMul/ReadVariableOp2�
DWheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAdd/ReadVariableOpDWheatClassifier_CNN-VIT_6/patch_encoder/dense/BiasAdd/ReadVariableOp2�
FWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ReadVariableOpFWheatClassifier_CNN-VIT_6/patch_encoder/dense/Tensordot/ReadVariableOp2�
BWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookupBWheatClassifier_CNN-VIT_6/patch_encoder/embedding/embedding_lookup:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�'
�
C__inference_dense_4_layer_call_and_return_conditional_losses_428436

inputs4
!tensordot_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xx
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������Q�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������Q�
 
_user_specified_nameinputs
�/
�
I__inference_patch_encoder_layer_call_and_return_conditional_losses_431037	
patch:
'dense_tensordot_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@3
!embedding_embedding_lookup_431030:Q@
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�embedding/embedding_lookup\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/limitConst*
_output_shapes
: *
dtype0*
value	B :Q2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltau
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:Q2
range�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freec
dense/Tensordot/ShapeShapepatch*
T0*
_output_shapes
:2
dense/Tensordot/Shape�
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis�
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2�
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis�
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const�
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1�
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis�
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack�
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:�������������������2
dense/Tensordot/transpose�
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense/Tensordot/Reshape�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense/Tensordot/Const_2�
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis�
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@2
dense/Tensordot�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2
dense/BiasAdd�
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_431030range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/431030*
_output_shapes

:Q@*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/431030*
_output_shapes

:Q@2%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@2'
%embedding/embedding_lookup/Identity_1�
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������Q@2
add�
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:�������������������

_user_specified_namepatch
��
�
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_428550

inputs'
conv_1_427905:

conv_1_427907:
'
conv_2_427921:


conv_2_427923:
'
conv_3_427938:

conv_3_427940:'
conv_4_427954:
conv_4_427956:'
patch_encoder_428017:	�@"
patch_encoder_428019:@&
patch_encoder_428021:Q@(
layer_normalization_428047:@(
layer_normalization_428049:@1
multi_head_attention_428088:@@-
multi_head_attention_428090:@1
multi_head_attention_428092:@@-
multi_head_attention_428094:@1
multi_head_attention_428096:@@-
multi_head_attention_428098:@1
multi_head_attention_428100:@@)
multi_head_attention_428102:@*
layer_normalization_1_428136:@*
layer_normalization_1_428138:@!
dense_1_428180:	@�
dense_1_428182:	�!
dense_2_428224:	�@
dense_2_428226:@*
layer_normalization_2_428260:@*
layer_normalization_2_428262:@3
multi_head_attention_1_428301:@@/
multi_head_attention_1_428303:@3
multi_head_attention_1_428305:@@/
multi_head_attention_1_428307:@3
multi_head_attention_1_428309:@@/
multi_head_attention_1_428311:@3
multi_head_attention_1_428313:@@+
multi_head_attention_1_428315:@*
layer_normalization_3_428349:@*
layer_normalization_3_428351:@!
dense_3_428393:	@�
dense_3_428395:	�!
dense_4_428437:	�@
dense_4_428439:@*
layer_normalization_4_428473:@*
layer_normalization_4_428475:@
fc_1_428497:	�(2
fc_1_428499:2
fc_2_428520:22
fc_2_428522:2%
output_layer_428544:2!
output_layer_428546:
identity��FC_1/StatefulPartitionedCall�FC_2/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�conv_4/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�.multi_head_attention_1/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�%patch_encoder/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_427905conv_1_427907*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_4279042 
conv_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_427921conv_2_427923*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_4279202 
conv_2/StatefulPartitionedCall�
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������bb
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_4278692
maxpool_1/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_427938conv_3_427940*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������``*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_4279372 
conv_3/StatefulPartitionedCall�
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_427954conv_4_427956*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������^^*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_4279532 
conv_4/StatefulPartitionedCall�
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������//* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_2_layer_call_and_return_conditional_losses_4278812
maxpool_2/PartitionedCall�
patches/PartitionedCallPartitionedCall"maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_4279742
patches/PartitionedCall�
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_428017patch_encoder_428019patch_encoder_428021*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_4280162'
%patch_encoder/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_428047layer_normalization_428049*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_4280462-
+layer_normalization/StatefulPartitionedCall�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_428088multi_head_attention_428090multi_head_attention_428092multi_head_attention_428094multi_head_attention_428096multi_head_attention_428098multi_head_attention_428100multi_head_attention_428102*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_4280872.
,multi_head_attention/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4281112
add/PartitionedCall�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_428136layer_normalization_1_428138*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_4281352/
-layer_normalization_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_428180dense_1_428182*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4281792!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_428224dense_2_428226*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4282232!
dense_2/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4282352
add_1/PartitionedCall�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_428260layer_normalization_2_428262*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_4282592/
-layer_normalization_2/StatefulPartitionedCall�
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_428301multi_head_attention_1_428303multi_head_attention_1_428305multi_head_attention_1_428307multi_head_attention_1_428309multi_head_attention_1_428311multi_head_attention_1_428313multi_head_attention_1_428315*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_42830020
.multi_head_attention_1/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4283242
add_2/PartitionedCall�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_428349layer_normalization_3_428351*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_4283482/
-layer_normalization_3/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_428393dense_3_428395*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4283922!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_428437dense_4_428439*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4284362!
dense_4/StatefulPartitionedCall�
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4284482
add_3/PartitionedCall�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_428473layer_normalization_4_428475*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_4284722/
-layer_normalization_4/StatefulPartitionedCall�
flatten_layer/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_4284842
flatten_layer/PartitionedCall�
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_428497fc_1_428499*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_4284962
FC_1/StatefulPartitionedCall�
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_4285072
leaky_ReLu_1/PartitionedCall�
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_428520fc_2_428522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_4285192
FC_2/StatefulPartitionedCall�
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_4285302
leaky_ReLu_2/PartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_428544output_layer_428546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_4285432&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_429509
input_layer!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:Q@

unknown_10:@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@ 

unknown_16:@@

unknown_17:@ 

unknown_18:@@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:	@�

unknown_23:	�

unknown_24:	�@

unknown_25:@

unknown_26:@

unknown_27:@ 

unknown_28:@@

unknown_29:@ 

unknown_30:@@

unknown_31:@ 

unknown_32:@@

unknown_33:@ 

unknown_34:@@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:	@�

unknown_39:	�

unknown_40:	�@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:	�(2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*2
config_proto" 

CPU

GPU2 *0J 8� *^
fYRW
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_4292972
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�	
�
@__inference_FC_1_layer_call_and_return_conditional_losses_428496

inputs1
matmul_readvariableop_resource:	�(2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�(2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������(
 
_user_specified_nameinputs
�

�
B__inference_conv_1_layer_call_and_return_conditional_losses_427904

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
_
C__inference_patches_layer_call_and_return_conditional_losses_427974

images
identityD
ShapeShapeimages*
T0*
_output_shapes
:2
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
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
ExtractImagePatchesExtractImagePatchesimages*
T0*0
_output_shapes
:���������		�*
ksizes
*
paddingVALID*
rates
*
strides
2
ExtractImagePatchesm
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape�
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:�������������������2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������//:W S
/
_output_shapes
:���������//
 
_user_specified_nameimages
�

�
B__inference_conv_1_layer_call_and_return_conditional_losses_430912

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�'
�
C__inference_dense_2_layer_call_and_return_conditional_losses_428223

inputs4
!tensordot_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xx
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������Q�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������Q�
 
_user_specified_nameinputs
�
R
&__inference_add_3_layer_call_fn_431650
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4284482
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:U Q
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/1
�
�
.__inference_patch_encoder_layer_call_fn_431048	
patch
unknown:	�@
	unknown_0:@
	unknown_1:Q@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpatchunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_4280162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:�������������������

_user_specified_namepatch
�
e
I__inference_flatten_layer_layer_call_and_return_conditional_losses_428484

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Q@:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�-
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_431415	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yj
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2
softmax/Softmax�
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������QQ2
dropout/Identity�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�

�
5__inference_multi_head_attention_layer_call_fn_431200	
query	
value
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_4289662
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�
�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_428046

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_431234

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
��
�
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_429644
input_layer'
conv_1_429512:

conv_1_429514:
'
conv_2_429517:


conv_2_429519:
'
conv_3_429523:

conv_3_429525:'
conv_4_429528:
conv_4_429530:'
patch_encoder_429535:	�@"
patch_encoder_429537:@&
patch_encoder_429539:Q@(
layer_normalization_429542:@(
layer_normalization_429544:@1
multi_head_attention_429547:@@-
multi_head_attention_429549:@1
multi_head_attention_429551:@@-
multi_head_attention_429553:@1
multi_head_attention_429555:@@-
multi_head_attention_429557:@1
multi_head_attention_429559:@@)
multi_head_attention_429561:@*
layer_normalization_1_429565:@*
layer_normalization_1_429567:@!
dense_1_429570:	@�
dense_1_429572:	�!
dense_2_429575:	�@
dense_2_429577:@*
layer_normalization_2_429581:@*
layer_normalization_2_429583:@3
multi_head_attention_1_429586:@@/
multi_head_attention_1_429588:@3
multi_head_attention_1_429590:@@/
multi_head_attention_1_429592:@3
multi_head_attention_1_429594:@@/
multi_head_attention_1_429596:@3
multi_head_attention_1_429598:@@+
multi_head_attention_1_429600:@*
layer_normalization_3_429604:@*
layer_normalization_3_429606:@!
dense_3_429609:	@�
dense_3_429611:	�!
dense_4_429614:	�@
dense_4_429616:@*
layer_normalization_4_429620:@*
layer_normalization_4_429622:@
fc_1_429626:	�(2
fc_1_429628:2
fc_2_429632:22
fc_2_429634:2%
output_layer_429638:2!
output_layer_429640:
identity��FC_1/StatefulPartitionedCall�FC_2/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�conv_4/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�.multi_head_attention_1/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�%patch_encoder/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_429512conv_1_429514*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_4279042 
conv_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_429517conv_2_429519*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_4279202 
conv_2/StatefulPartitionedCall�
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������bb
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_4278692
maxpool_1/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_429523conv_3_429525*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������``*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_4279372 
conv_3/StatefulPartitionedCall�
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_429528conv_4_429530*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������^^*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_4279532 
conv_4/StatefulPartitionedCall�
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������//* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_2_layer_call_and_return_conditional_losses_4278812
maxpool_2/PartitionedCall�
patches/PartitionedCallPartitionedCall"maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_4279742
patches/PartitionedCall�
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_429535patch_encoder_429537patch_encoder_429539*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_4280162'
%patch_encoder/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_429542layer_normalization_429544*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_4280462-
+layer_normalization/StatefulPartitionedCall�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_429547multi_head_attention_429549multi_head_attention_429551multi_head_attention_429553multi_head_attention_429555multi_head_attention_429557multi_head_attention_429559multi_head_attention_429561*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_4280872.
,multi_head_attention/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4281112
add/PartitionedCall�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_429565layer_normalization_1_429567*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_4281352/
-layer_normalization_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_429570dense_1_429572*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4281792!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_429575dense_2_429577*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4282232!
dense_2/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4282352
add_1/PartitionedCall�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_429581layer_normalization_2_429583*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_4282592/
-layer_normalization_2/StatefulPartitionedCall�
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_429586multi_head_attention_1_429588multi_head_attention_1_429590multi_head_attention_1_429592multi_head_attention_1_429594multi_head_attention_1_429596multi_head_attention_1_429598multi_head_attention_1_429600*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_42830020
.multi_head_attention_1/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4283242
add_2/PartitionedCall�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_429604layer_normalization_3_429606*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_4283482/
-layer_normalization_3/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_429609dense_3_429611*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4283922!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_429614dense_4_429616*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4284362!
dense_4/StatefulPartitionedCall�
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4284482
add_3/PartitionedCall�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_429620layer_normalization_4_429622*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_4284722/
-layer_normalization_4/StatefulPartitionedCall�
flatten_layer/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_4284842
flatten_layer/PartitionedCall�
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_429626fc_1_429628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_4284962
FC_1/StatefulPartitionedCall�
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_4285072
leaky_ReLu_1/PartitionedCall�
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_429632fc_2_429634*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_4285192
FC_2/StatefulPartitionedCall�
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_4285302
leaky_ReLu_2/PartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_429638output_layer_429640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_4285432&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�
d
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_431716

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������2*
alpha%���>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
��
�.
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_430688

inputs?
%conv_1_conv2d_readvariableop_resource:
4
&conv_1_biasadd_readvariableop_resource:
?
%conv_2_conv2d_readvariableop_resource:

4
&conv_2_biasadd_readvariableop_resource:
?
%conv_3_conv2d_readvariableop_resource:
4
&conv_3_biasadd_readvariableop_resource:?
%conv_4_conv2d_readvariableop_resource:4
&conv_4_biasadd_readvariableop_resource:H
5patch_encoder_dense_tensordot_readvariableop_resource:	�@A
3patch_encoder_dense_biasadd_readvariableop_resource:@A
/patch_encoder_embedding_embedding_lookup_430354:Q@G
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@V
@multi_head_attention_query_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_query_add_readvariableop_resource:@T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:@@F
4multi_head_attention_key_add_readvariableop_resource:@V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_value_add_readvariableop_resource:@a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:@@O
Amulti_head_attention_attention_output_add_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@<
)dense_1_tensordot_readvariableop_resource:	@�6
'dense_1_biasadd_readvariableop_resource:	�<
)dense_2_tensordot_readvariableop_resource:	�@5
'dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@X
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_1_query_add_readvariableop_resource:@V
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_1_key_add_readvariableop_resource:@X
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_1_value_add_readvariableop_resource:@c
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:@@Q
Cmulti_head_attention_1_attention_output_add_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@<
)dense_3_tensordot_readvariableop_resource:	@�6
'dense_3_biasadd_readvariableop_resource:	�<
)dense_4_tensordot_readvariableop_resource:	�@5
'dense_4_biasadd_readvariableop_resource:@I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_4_batchnorm_readvariableop_resource:@6
#fc_1_matmul_readvariableop_resource:	�(22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identity��FC_1/BiasAdd/ReadVariableOp�FC_1/MatMul/ReadVariableOp�FC_2/BiasAdd/ReadVariableOp�FC_2/MatMul/ReadVariableOp�conv_1/BiasAdd/ReadVariableOp�conv_1/Conv2D/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�conv_3/Conv2D/ReadVariableOp�conv_4/BiasAdd/ReadVariableOp�conv_4/Conv2D/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp� dense_2/Tensordot/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp� dense_4/Tensordot/ReadVariableOp�,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�.layer_normalization_2/batchnorm/ReadVariableOp�2layer_normalization_2/batchnorm/mul/ReadVariableOp�.layer_normalization_3/batchnorm/ReadVariableOp�2layer_normalization_3/batchnorm/mul/ReadVariableOp�.layer_normalization_4/batchnorm/ReadVariableOp�2layer_normalization_4/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�:multi_head_attention_1/attention_output/add/ReadVariableOp�Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_1/key/add/ReadVariableOp�7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_1/query/add/ReadVariableOp�9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_1/value/add/ReadVariableOp�9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�*patch_encoder/dense/BiasAdd/ReadVariableOp�,patch_encoder/dense/Tensordot/ReadVariableOp�(patch_encoder/embedding/embedding_lookup�
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOp�
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2
conv_1/Conv2D�
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_1/BiasAdd/ReadVariableOp�
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2
conv_1/BiasAdd�
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
conv_2/Conv2D/ReadVariableOp�
conv_2/Conv2DConv2Dconv_1/BiasAdd:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2
conv_2/Conv2D�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_2/BiasAdd/ReadVariableOp�
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2
conv_2/BiasAdd�
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:���������bb
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPool�
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_3/Conv2D/ReadVariableOp�
conv_3/Conv2DConv2Dmaxpool_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``*
paddingVALID*
strides
2
conv_3/Conv2D�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``2
conv_3/BiasAdd�
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOp�
conv_4/Conv2DConv2Dconv_3/BiasAdd:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^*
paddingVALID*
strides
2
conv_4/Conv2D�
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_4/BiasAdd/ReadVariableOp�
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^2
conv_4/BiasAdd�
maxpool_2/MaxPoolMaxPoolconv_4/BiasAdd:output:0*/
_output_shapes
:���������//*
ksize
*
paddingVALID*
strides
2
maxpool_2/MaxPoolh
patches/ShapeShapemaxpool_2/MaxPool:output:0*
T0*
_output_shapes
:2
patches/Shape�
patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
patches/strided_slice/stack�
patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_1�
patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_2�
patches/strided_sliceStridedSlicepatches/Shape:output:0$patches/strided_slice/stack:output:0&patches/strided_slice/stack_1:output:0&patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
patches/strided_slice�
patches/ExtractImagePatchesExtractImagePatchesmaxpool_2/MaxPool:output:0*
T0*0
_output_shapes
:���������		�*
ksizes
*
paddingVALID*
rates
*
strides
2
patches/ExtractImagePatches}
patches/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������2
patches/Reshape/shape/1u
patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�2
patches/Reshape/shape/2�
patches/Reshape/shapePackpatches/strided_slice:output:0 patches/Reshape/shape/1:output:0 patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
patches/Reshape/shape�
patches/ReshapeReshape%patches/ExtractImagePatches:patches:0patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:�������������������2
patches/Reshapex
patch_encoder/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
patch_encoder/range/startx
patch_encoder/range/limitConst*
_output_shapes
: *
dtype0*
value	B :Q2
patch_encoder/range/limitx
patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder/range/delta�
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes
:Q2
patch_encoder/range�
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,patch_encoder/dense/Tensordot/ReadVariableOp�
"patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"patch_encoder/dense/Tensordot/axes�
"patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"patch_encoder/dense/Tensordot/free�
#patch_encoder/dense/Tensordot/ShapeShapepatches/Reshape:output:0*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/Shape�
+patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/GatherV2/axis�
&patch_encoder/dense/Tensordot/GatherV2GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/free:output:04patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/GatherV2�
-patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-patch_encoder/dense/Tensordot/GatherV2_1/axis�
(patch_encoder/dense/Tensordot/GatherV2_1GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/axes:output:06patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(patch_encoder/dense/Tensordot/GatherV2_1�
#patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#patch_encoder/dense/Tensordot/Const�
"patch_encoder/dense/Tensordot/ProdProd/patch_encoder/dense/Tensordot/GatherV2:output:0,patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"patch_encoder/dense/Tensordot/Prod�
%patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%patch_encoder/dense/Tensordot/Const_1�
$patch_encoder/dense/Tensordot/Prod_1Prod1patch_encoder/dense/Tensordot/GatherV2_1:output:0.patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$patch_encoder/dense/Tensordot/Prod_1�
)patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)patch_encoder/dense/Tensordot/concat/axis�
$patch_encoder/dense/Tensordot/concatConcatV2+patch_encoder/dense/Tensordot/free:output:0+patch_encoder/dense/Tensordot/axes:output:02patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$patch_encoder/dense/Tensordot/concat�
#patch_encoder/dense/Tensordot/stackPack+patch_encoder/dense/Tensordot/Prod:output:0-patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/stack�
'patch_encoder/dense/Tensordot/transpose	Transposepatches/Reshape:output:0-patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:�������������������2)
'patch_encoder/dense/Tensordot/transpose�
%patch_encoder/dense/Tensordot/ReshapeReshape+patch_encoder/dense/Tensordot/transpose:y:0,patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2'
%patch_encoder/dense/Tensordot/Reshape�
$patch_encoder/dense/Tensordot/MatMulMatMul.patch_encoder/dense/Tensordot/Reshape:output:04patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2&
$patch_encoder/dense/Tensordot/MatMul�
%patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2'
%patch_encoder/dense/Tensordot/Const_2�
+patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/concat_1/axis�
&patch_encoder/dense/Tensordot/concat_1ConcatV2/patch_encoder/dense/Tensordot/GatherV2:output:0.patch_encoder/dense/Tensordot/Const_2:output:04patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/concat_1�
patch_encoder/dense/TensordotReshape.patch_encoder/dense/Tensordot/MatMul:product:0/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@2
patch_encoder/dense/Tensordot�
*patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp3patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*patch_encoder/dense/BiasAdd/ReadVariableOp�
patch_encoder/dense/BiasAddBiasAdd&patch_encoder/dense/Tensordot:output:02patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2
patch_encoder/dense/BiasAdd�
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_430354patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/430354*
_output_shapes

:Q@*
dtype02*
(patch_encoder/embedding/embedding_lookup�
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/430354*
_output_shapes

:Q@23
1patch_encoder/embedding/embedding_lookup/Identity�
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@25
3patch_encoder/embedding/embedding_lookup/Identity_1�
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������Q@2
patch_encoder/add�
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices�
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2"
 layer_normalization/moments/mean�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2*
(layer_normalization/moments/StopGradient�
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2/
-layer_normalization/moments/SquaredDifference�
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2&
$layer_normalization/moments/variance�
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52%
#layer_normalization/batchnorm/add/y�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2#
!layer_normalization/batchnorm/add�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization/batchnorm/Rsqrt�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2#
!layer_normalization/batchnorm/mul�
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization/batchnorm/mul_1�
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization/batchnorm/mul_2�
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOp�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2#
!layer_normalization/batchnorm/sub�
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization/batchnorm/add_1�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp�
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOp�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2 
multi_head_attention/query/add�
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp�
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOp�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
multi_head_attention/key/add�
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp�
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOp�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/y�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
multi_head_attention/Mul�
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsum�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2&
$multi_head_attention/softmax/Softmax�
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2,
*multi_head_attention/dropout/dropout/Const�
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������QQ2*
(multi_head_attention/dropout/dropout/Mul�
*multi_head_attention/dropout/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2,
*multi_head_attention/dropout/dropout/Shape�
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������QQ*
dtype02C
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniform�
3multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=25
3multi_head_attention/dropout/dropout/GreaterEqual/y�
1multi_head_attention/dropout/dropout/GreaterEqualGreaterEqualJmulti_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0<multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������QQ23
1multi_head_attention/dropout/dropout/GreaterEqual�
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������QQ2+
)multi_head_attention/dropout/dropout/Cast�
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������QQ2,
*multi_head_attention/dropout/dropout/Mul_1�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsum�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsum�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOp�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2+
)multi_head_attention/attention_output/add�
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*+
_output_shapes
:���������Q@2	
add/add�
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices�
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2$
"layer_normalization_1/moments/mean�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2,
*layer_normalization_1/moments/StopGradient�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@21
/layer_normalization_1/moments/SquaredDifference�
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2(
&layer_normalization_1/moments/variance�
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52'
%layer_normalization_1/batchnorm/add/y�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization_1/batchnorm/add�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2'
%layer_normalization_1/batchnorm/Rsqrt�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_1/batchnorm/mul�
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_1/batchnorm/mul_1�
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_1/batchnorm/mul_2�
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_1/batchnorm/sub�
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_1/batchnorm/add_1�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes�
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/free�
dense_1/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape�
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis�
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2�
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis�
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const�
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod�
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1�
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1�
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis�
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack�
dense_1/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@2
dense_1/Tensordot/transpose�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_1/Tensordot/Reshape�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/Tensordot/MatMul�
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_1/Tensordot/Const_2�
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis�
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Tensordot�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/x�
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_1/Gelu/Cast/x�
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/truediv|
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1/Gelu/add/x�
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/add�
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/mul_1�
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes�
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/freex
dense_2/Tensordot/ShapeShapedense_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape�
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis�
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2�
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis�
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const�
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod�
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1�
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1�
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis�
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat�
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack�
dense_2/Tensordot/transpose	Transposedense_1/Gelu/mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�2
dense_2/Tensordot/transpose�
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_2/Tensordot/Reshape�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/Tensordot/MatMul�
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_2/Tensordot/Const_2�
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis�
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1�
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Tensordot�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/x�
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_2/Gelu/Cast/x�
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/truediv{
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_2/Gelu/add/x�
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/add�
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/mul_1z
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*+
_output_shapes
:���������Q@2
	add_1/add�
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indices�
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2$
"layer_normalization_2/moments/mean�
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2,
*layer_normalization_2/moments/StopGradient�
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@21
/layer_normalization_2/moments/SquaredDifference�
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices�
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2(
&layer_normalization_2/moments/variance�
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52'
%layer_normalization_2/batchnorm/add/y�
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization_2/batchnorm/add�
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2'
%layer_normalization_2/batchnorm/Rsqrt�
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOp�
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_2/batchnorm/mul�
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_2/batchnorm/mul_1�
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_2/batchnorm/mul_2�
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOp�
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_2/batchnorm/sub�
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_2/batchnorm/add_1�
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp�
*multi_head_attention_1/query/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsum�
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/query/add/ReadVariableOp�
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2"
 multi_head_attention_1/query/add�
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp�
(multi_head_attention_1/key/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum�
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_1/key/add/ReadVariableOp�
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2 
multi_head_attention_1/key/add�
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp�
*multi_head_attention_1/value/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsum�
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/value/add/ReadVariableOp�
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2"
 multi_head_attention_1/value/add�
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_1/Mul/y�
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
multi_head_attention_1/Mul�
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum�
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2(
&multi_head_attention_1/softmax/Softmax�
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2.
,multi_head_attention_1/dropout/dropout/Const�
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������QQ2,
*multi_head_attention_1/dropout/dropout/Mul�
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_1/dropout/dropout/Shape�
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������QQ*
dtype02E
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform�
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=27
5multi_head_attention_1/dropout/dropout/GreaterEqual/y�
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������QQ25
3multi_head_attention_1/dropout/dropout/GreaterEqual�
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������QQ2-
+multi_head_attention_1/dropout/dropout/Cast�
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������QQ2.
,multi_head_attention_1/dropout/dropout/Mul_1�
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum�
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp�
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsum�
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp�
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2-
+multi_head_attention_1/attention_output/add�
	add_2/addAddV2/multi_head_attention_1/attention_output/add:z:0add_1/add:z:0*
T0*+
_output_shapes
:���������Q@2
	add_2/add�
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indices�
"layer_normalization_3/moments/meanMeanadd_2/add:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2$
"layer_normalization_3/moments/mean�
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2,
*layer_normalization_3/moments/StopGradient�
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd_2/add:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@21
/layer_normalization_3/moments/SquaredDifference�
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices�
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2(
&layer_normalization_3/moments/variance�
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52'
%layer_normalization_3/batchnorm/add/y�
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization_3/batchnorm/add�
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2'
%layer_normalization_3/batchnorm/Rsqrt�
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOp�
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_3/batchnorm/mul�
%layer_normalization_3/batchnorm/mul_1Muladd_2/add:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_3/batchnorm/mul_1�
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_3/batchnorm/mul_2�
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_3/batchnorm/ReadVariableOp�
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_3/batchnorm/sub�
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_3/batchnorm/add_1�
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes�
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/free�
dense_3/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape�
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis�
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2�
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis�
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const�
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod�
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1�
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1�
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis�
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat�
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack�
dense_3/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@2
dense_3/Tensordot/transpose�
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_3/Tensordot/Reshape�
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_3/Tensordot/MatMul�
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_3/Tensordot/Const_2�
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis�
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1�
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Tensordot�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2
dense_3/BiasAddm
dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_3/Gelu/mul/x�
dense_3/Gelu/mulMuldense_3/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/mulo
dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_3/Gelu/Cast/x�
dense_3/Gelu/truedivRealDivdense_3/BiasAdd:output:0dense_3/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/truediv|
dense_3/Gelu/ErfErfdense_3/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/Erfm
dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_3/Gelu/add/x�
dense_3/Gelu/addAddV2dense_3/Gelu/add/x:output:0dense_3/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/add�
dense_3/Gelu/mul_1Muldense_3/Gelu/mul:z:0dense_3/Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/mul_1�
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes�
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/freex
dense_4/Tensordot/ShapeShapedense_3/Gelu/mul_1:z:0*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape�
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axis�
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2�
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis�
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const�
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod�
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1�
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1�
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axis�
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat�
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack�
dense_4/Tensordot/transpose	Transposedense_3/Gelu/mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�2
dense_4/Tensordot/transpose�
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_4/Tensordot/Reshape�
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_4/Tensordot/MatMul�
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_4/Tensordot/Const_2�
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axis�
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1�
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Tensordot�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
dense_4/BiasAddm
dense_4/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_4/Gelu/mul/x�
dense_4/Gelu/mulMuldense_4/Gelu/mul/x:output:0dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/mulo
dense_4/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_4/Gelu/Cast/x�
dense_4/Gelu/truedivRealDivdense_4/BiasAdd:output:0dense_4/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/truediv{
dense_4/Gelu/ErfErfdense_4/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/Erfm
dense_4/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_4/Gelu/add/x�
dense_4/Gelu/addAddV2dense_4/Gelu/add/x:output:0dense_4/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/add�
dense_4/Gelu/mul_1Muldense_4/Gelu/mul:z:0dense_4/Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/mul_1|
	add_3/addAddV2dense_4/Gelu/mul_1:z:0add_2/add:z:0*
T0*+
_output_shapes
:���������Q@2
	add_3/add�
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indices�
"layer_normalization_4/moments/meanMeanadd_3/add:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2$
"layer_normalization_4/moments/mean�
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2,
*layer_normalization_4/moments/StopGradient�
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd_3/add:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@21
/layer_normalization_4/moments/SquaredDifference�
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices�
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2(
&layer_normalization_4/moments/variance�
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52'
%layer_normalization_4/batchnorm/add/y�
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization_4/batchnorm/add�
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2'
%layer_normalization_4/batchnorm/Rsqrt�
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOp�
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_4/batchnorm/mul�
%layer_normalization_4/batchnorm/mul_1Muladd_3/add:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_4/batchnorm/mul_1�
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_4/batchnorm/mul_2�
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_4/batchnorm/ReadVariableOp�
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_4/batchnorm/sub�
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_4/batchnorm/add_1{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten_layer/Const�
flatten_layer/ReshapeReshape)layer_normalization_4/batchnorm/add_1:z:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:����������(2
flatten_layer/Reshape�
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	�(2*
dtype02
FC_1/MatMul/ReadVariableOp�
FC_1/MatMulMatMulflatten_layer/Reshape:output:0"FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
FC_1/MatMul�
FC_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_1/BiasAdd/ReadVariableOp�
FC_1/BiasAddBiasAddFC_1/MatMul:product:0#FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
FC_1/BiasAdd�
leaky_ReLu_1/LeakyRelu	LeakyReluFC_1/BiasAdd:output:0*'
_output_shapes
:���������2*
alpha%���>2
leaky_ReLu_1/LeakyRelu�
FC_2/MatMul/ReadVariableOpReadVariableOp#fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
FC_2/MatMul/ReadVariableOp�
FC_2/MatMulMatMul$leaky_ReLu_1/LeakyRelu:activations:0"FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
FC_2/MatMul�
FC_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_2/BiasAdd/ReadVariableOp�
FC_2/BiasAddBiasAddFC_2/MatMul:product:0#FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
FC_2/BiasAdd�
leaky_ReLu_2/LeakyRelu	LeakyReluFC_2/BiasAdd:output:0*'
_output_shapes
:���������2*
alpha%���>2
leaky_ReLu_2/LeakyRelu�
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02$
"output_layer/MatMul/ReadVariableOp�
output_layer/MatMulMatMul$leaky_ReLu_2/LeakyRelu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output_layer/MatMul�
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOp�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output_layer/BiasAdd�
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
output_layer/Softmax�
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
FC_1/BiasAdd/ReadVariableOpFC_1/BiasAdd/ReadVariableOp28
FC_1/MatMul/ReadVariableOpFC_1/MatMul/ReadVariableOp2:
FC_2/BiasAdd/ReadVariableOpFC_2/BiasAdd/ReadVariableOp28
FC_2/MatMul/ReadVariableOpFC_2/MatMul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
conv_4/BiasAdd/ReadVariableOpconv_4/BiasAdd/ReadVariableOp2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2X
*patch_encoder/dense/BiasAdd/ReadVariableOp*patch_encoder/dense/BiasAdd/ReadVariableOp2\
,patch_encoder/dense/Tensordot/ReadVariableOp,patch_encoder/dense/Tensordot/ReadVariableOp2T
(patch_encoder/embedding/embedding_lookup(patch_encoder/embedding/embedding_lookup:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
e
I__inference_flatten_layer_layer_call_and_return_conditional_losses_431687

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Q@:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�'
�
C__inference_dense_3_layer_call_and_return_conditional_losses_431582

inputs4
!tensordot_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xy
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:���������Q�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_428259

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�

�
7__inference_multi_head_attention_1_layer_call_fn_431501	
query	
value
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_4288252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�
m
A__inference_add_3_layer_call_and_return_conditional_losses_431644
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:U Q
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/1
�
_
C__inference_patches_layer_call_and_return_conditional_losses_430992

images
identityD
ShapeShapeimages*
T0*
_output_shapes
:2
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
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice�
ExtractImagePatchesExtractImagePatchesimages*
T0*0
_output_shapes
:���������		�*
ksizes
*
paddingVALID*
rates
*
strides
2
ExtractImagePatchesm
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape�
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:�������������������2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������//:W S
/
_output_shapes
:���������//
 
_user_specified_nameimages
�
J
.__inference_flatten_layer_layer_call_fn_431692

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_4284842
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������Q@:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�

�
H__inference_output_layer_layer_call_and_return_conditional_losses_431761

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
B__inference_conv_4_layer_call_and_return_conditional_losses_427953

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������^^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������``: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������``
 
_user_specified_nameinputs
�
i
?__inference_add_layer_call_and_return_conditional_losses_428111

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
d
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_428507

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������2*
alpha%���>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
��
�
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_429779
input_layer'
conv_1_429647:

conv_1_429649:
'
conv_2_429652:


conv_2_429654:
'
conv_3_429658:

conv_3_429660:'
conv_4_429663:
conv_4_429665:'
patch_encoder_429670:	�@"
patch_encoder_429672:@&
patch_encoder_429674:Q@(
layer_normalization_429677:@(
layer_normalization_429679:@1
multi_head_attention_429682:@@-
multi_head_attention_429684:@1
multi_head_attention_429686:@@-
multi_head_attention_429688:@1
multi_head_attention_429690:@@-
multi_head_attention_429692:@1
multi_head_attention_429694:@@)
multi_head_attention_429696:@*
layer_normalization_1_429700:@*
layer_normalization_1_429702:@!
dense_1_429705:	@�
dense_1_429707:	�!
dense_2_429710:	�@
dense_2_429712:@*
layer_normalization_2_429716:@*
layer_normalization_2_429718:@3
multi_head_attention_1_429721:@@/
multi_head_attention_1_429723:@3
multi_head_attention_1_429725:@@/
multi_head_attention_1_429727:@3
multi_head_attention_1_429729:@@/
multi_head_attention_1_429731:@3
multi_head_attention_1_429733:@@+
multi_head_attention_1_429735:@*
layer_normalization_3_429739:@*
layer_normalization_3_429741:@!
dense_3_429744:	@�
dense_3_429746:	�!
dense_4_429749:	�@
dense_4_429751:@*
layer_normalization_4_429755:@*
layer_normalization_4_429757:@
fc_1_429761:	�(2
fc_1_429763:2
fc_2_429767:22
fc_2_429769:2%
output_layer_429773:2!
output_layer_429775:
identity��FC_1/StatefulPartitionedCall�FC_2/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�conv_4/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�.multi_head_attention_1/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�%patch_encoder/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_429647conv_1_429649*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_4279042 
conv_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_429652conv_2_429654*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_4279202 
conv_2/StatefulPartitionedCall�
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������bb
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_4278692
maxpool_1/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_429658conv_3_429660*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������``*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_4279372 
conv_3/StatefulPartitionedCall�
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_429663conv_4_429665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������^^*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_4279532 
conv_4/StatefulPartitionedCall�
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������//* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_2_layer_call_and_return_conditional_losses_4278812
maxpool_2/PartitionedCall�
patches/PartitionedCallPartitionedCall"maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_4279742
patches/PartitionedCall�
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_429670patch_encoder_429672patch_encoder_429674*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_4280162'
%patch_encoder/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_429677layer_normalization_429679*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_4280462-
+layer_normalization/StatefulPartitionedCall�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_429682multi_head_attention_429684multi_head_attention_429686multi_head_attention_429688multi_head_attention_429690multi_head_attention_429692multi_head_attention_429694multi_head_attention_429696*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_4289662.
,multi_head_attention/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4281112
add/PartitionedCall�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_429700layer_normalization_1_429702*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_4281352/
-layer_normalization_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_429705dense_1_429707*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4281792!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_429710dense_2_429712*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4282232!
dense_2/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4282352
add_1/PartitionedCall�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_429716layer_normalization_2_429718*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_4282592/
-layer_normalization_2/StatefulPartitionedCall�
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_429721multi_head_attention_1_429723multi_head_attention_1_429725multi_head_attention_1_429727multi_head_attention_1_429729multi_head_attention_1_429731multi_head_attention_1_429733multi_head_attention_1_429735*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_42882520
.multi_head_attention_1/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4283242
add_2/PartitionedCall�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_429739layer_normalization_3_429741*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_4283482/
-layer_normalization_3/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_429744dense_3_429746*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4283922!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_429749dense_4_429751*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4284362!
dense_4/StatefulPartitionedCall�
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4284482
add_3/PartitionedCall�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_429755layer_normalization_4_429757*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_4284722/
-layer_normalization_4/StatefulPartitionedCall�
flatten_layer/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_4284842
flatten_layer/PartitionedCall�
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_429761fc_1_429763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_4284962
FC_1/StatefulPartitionedCall�
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_4285072
leaky_ReLu_1/PartitionedCall�
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_429767fc_2_429769*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_4285192
FC_2/StatefulPartitionedCall�
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_4285302
leaky_ReLu_2/PartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_429773output_layer_429775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_4285432&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�
m
A__inference_add_2_layer_call_and_return_conditional_losses_431507
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:U Q
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/1
�
�
6__inference_layer_normalization_2_layer_call_fn_431380

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_4282592
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
F
*__inference_maxpool_1_layer_call_fn_427875

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_4278692
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_431290

inputs
unknown:	@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4281792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������Q�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_431371

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�7
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_428825	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yj
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/dropout/Const�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������QQ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������QQ2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/Mul_1�
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�

�
B__inference_conv_2_layer_call_and_return_conditional_losses_427920

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:�����������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:�����������

 
_user_specified_nameinputs
�/
�
I__inference_patch_encoder_layer_call_and_return_conditional_losses_428016	
patch:
'dense_tensordot_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@3
!embedding_embedding_lookup_428009:Q@
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�embedding/embedding_lookup\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/limitConst*
_output_shapes
: *
dtype0*
value	B :Q2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltau
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:Q2
range�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freec
dense/Tensordot/ShapeShapepatch*
T0*
_output_shapes
:2
dense/Tensordot/Shape�
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis�
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2�
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis�
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const�
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1�
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis�
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat�
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack�
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:�������������������2
dense/Tensordot/transpose�
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense/Tensordot/Reshape�
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense/Tensordot/Const_2�
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis�
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1�
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@2
dense/Tensordot�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2
dense/BiasAdd�
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_428009range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/428009*
_output_shapes

:Q@*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/428009*
_output_shapes

:Q@2%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@2'
%embedding/embedding_lookup/Identity_1�
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������Q@2
add�
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:�������������������

_user_specified_namepatch
�'
�
C__inference_dense_1_layer_call_and_return_conditional_losses_428179

inputs4
!tensordot_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xy
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:���������Q�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
�
'__inference_conv_1_layer_call_fn_430921

inputs!
unknown:

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_4279042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
B__inference_conv_4_layer_call_and_return_conditional_losses_430969

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������^^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������``: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������``
 
_user_specified_nameinputs
�
�
'__inference_conv_4_layer_call_fn_430978

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������^^*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_4279532
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������^^2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������``: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������``
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_429896
input_layer!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:Q@

unknown_10:@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@ 

unknown_16:@@

unknown_17:@ 

unknown_18:@@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:	@�

unknown_23:	�

unknown_24:	�@

unknown_25:@

unknown_26:@

unknown_27:@ 

unknown_28:@@

unknown_29:@ 

unknown_30:@@

unknown_31:@ 

unknown_32:@@

unknown_33:@ 

unknown_34:@@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:	@�

unknown_39:	�

unknown_40:	�@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:	�(2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*2
config_proto" 

CPU

GPU2 *0J 8� **
f%R#
!__inference__wrapped_model_4278632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�
P
$__inference_add_layer_call_fn_431212
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4281112
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:U Q
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/1
�'
�
C__inference_dense_3_layer_call_and_return_conditional_losses_428392

inputs4
!tensordot_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xy
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:���������Q�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
k
A__inference_add_3_layer_call_and_return_conditional_losses_428448

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
d
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_428530

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������2*
alpha%���>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
H__inference_output_layer_layer_call_and_return_conditional_losses_428543

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_431672

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
�
'__inference_conv_2_layer_call_fn_430940

inputs!
unknown:


	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_4279202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:�����������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������

 
_user_specified_nameinputs
�	
�
@__inference_FC_2_layer_call_and_return_conditional_losses_431731

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_428472

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
�
%__inference_FC_1_layer_call_fn_431711

inputs
unknown:	�(2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_4284962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������(: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������(
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_3_layer_call_fn_431544

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_4283482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�-
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_431114	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yj
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2
softmax/Softmax�
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������QQ2
dropout/Identity�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�
d
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_431745

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������2*
alpha%���>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�'
�
C__inference_dense_2_layer_call_and_return_conditional_losses_431328

inputs4
!tensordot_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xx
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������Q�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:���������Q�
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_4_layer_call_fn_431681

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_4284722
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�

�
B__inference_conv_3_layer_call_and_return_conditional_losses_427937

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������bb
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������bb

 
_user_specified_nameinputs
�
a
E__inference_maxpool_2_layer_call_and_return_conditional_losses_427881

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
I
-__inference_leaky_ReLu_2_layer_call_fn_431750

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_4285302
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������2:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�7
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_431156	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yj
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/dropout/Const�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������QQ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������QQ2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/Mul_1�
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�
�
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_428655
input_layer!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:
#
	unknown_3:

	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:	�@
	unknown_8:@
	unknown_9:Q@

unknown_10:@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@ 

unknown_16:@@

unknown_17:@ 

unknown_18:@@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:	@�

unknown_23:	�

unknown_24:	�@

unknown_25:@

unknown_26:@

unknown_27:@ 

unknown_28:@@

unknown_29:@ 

unknown_30:@@

unknown_31:@ 

unknown_32:@@

unknown_33:@ 

unknown_34:@@

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:	@�

unknown_39:	�

unknown_40:	�@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:	�(2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*2
config_proto" 

CPU

GPU2 *0J 8� *^
fYRW
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_4285502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�
m
A__inference_add_1_layer_call_and_return_conditional_losses_431343
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:U Q
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/1
��
�.
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_430285

inputs?
%conv_1_conv2d_readvariableop_resource:
4
&conv_1_biasadd_readvariableop_resource:
?
%conv_2_conv2d_readvariableop_resource:

4
&conv_2_biasadd_readvariableop_resource:
?
%conv_3_conv2d_readvariableop_resource:
4
&conv_3_biasadd_readvariableop_resource:?
%conv_4_conv2d_readvariableop_resource:4
&conv_4_biasadd_readvariableop_resource:H
5patch_encoder_dense_tensordot_readvariableop_resource:	�@A
3patch_encoder_dense_biasadd_readvariableop_resource:@A
/patch_encoder_embedding_embedding_lookup_429965:Q@G
9layer_normalization_batchnorm_mul_readvariableop_resource:@C
5layer_normalization_batchnorm_readvariableop_resource:@V
@multi_head_attention_query_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_query_add_readvariableop_resource:@T
>multi_head_attention_key_einsum_einsum_readvariableop_resource:@@F
4multi_head_attention_key_add_readvariableop_resource:@V
@multi_head_attention_value_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_value_add_readvariableop_resource:@a
Kmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource:@@O
Amulti_head_attention_attention_output_add_readvariableop_resource:@I
;layer_normalization_1_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_1_batchnorm_readvariableop_resource:@<
)dense_1_tensordot_readvariableop_resource:	@�6
'dense_1_biasadd_readvariableop_resource:	�<
)dense_2_tensordot_readvariableop_resource:	�@5
'dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@X
Bmulti_head_attention_1_query_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_1_query_add_readvariableop_resource:@V
@multi_head_attention_1_key_einsum_einsum_readvariableop_resource:@@H
6multi_head_attention_1_key_add_readvariableop_resource:@X
Bmulti_head_attention_1_value_einsum_einsum_readvariableop_resource:@@J
8multi_head_attention_1_value_add_readvariableop_resource:@c
Mmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:@@Q
Cmulti_head_attention_1_attention_output_add_readvariableop_resource:@I
;layer_normalization_3_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_3_batchnorm_readvariableop_resource:@<
)dense_3_tensordot_readvariableop_resource:	@�6
'dense_3_biasadd_readvariableop_resource:	�<
)dense_4_tensordot_readvariableop_resource:	�@5
'dense_4_biasadd_readvariableop_resource:@I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_4_batchnorm_readvariableop_resource:@6
#fc_1_matmul_readvariableop_resource:	�(22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identity��FC_1/BiasAdd/ReadVariableOp�FC_1/MatMul/ReadVariableOp�FC_2/BiasAdd/ReadVariableOp�FC_2/MatMul/ReadVariableOp�conv_1/BiasAdd/ReadVariableOp�conv_1/Conv2D/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�conv_3/BiasAdd/ReadVariableOp�conv_3/Conv2D/ReadVariableOp�conv_4/BiasAdd/ReadVariableOp�conv_4/Conv2D/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp� dense_2/Tensordot/ReadVariableOp�dense_3/BiasAdd/ReadVariableOp� dense_3/Tensordot/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp� dense_4/Tensordot/ReadVariableOp�,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�.layer_normalization_2/batchnorm/ReadVariableOp�2layer_normalization_2/batchnorm/mul/ReadVariableOp�.layer_normalization_3/batchnorm/ReadVariableOp�2layer_normalization_3/batchnorm/mul/ReadVariableOp�.layer_normalization_4/batchnorm/ReadVariableOp�2layer_normalization_4/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�:multi_head_attention_1/attention_output/add/ReadVariableOp�Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp�-multi_head_attention_1/key/add/ReadVariableOp�7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp�/multi_head_attention_1/query/add/ReadVariableOp�9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp�/multi_head_attention_1/value/add/ReadVariableOp�9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�*patch_encoder/dense/BiasAdd/ReadVariableOp�,patch_encoder/dense/Tensordot/ReadVariableOp�(patch_encoder/embedding/embedding_lookup�
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOp�
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2
conv_1/Conv2D�
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_1/BiasAdd/ReadVariableOp�
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2
conv_1/BiasAdd�
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
conv_2/Conv2D/ReadVariableOp�
conv_2/Conv2DConv2Dconv_1/BiasAdd:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
*
paddingVALID*
strides
2
conv_2/Conv2D�
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_2/BiasAdd/ReadVariableOp�
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������
2
conv_2/BiasAdd�
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:���������bb
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPool�
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_3/Conv2D/ReadVariableOp�
conv_3/Conv2DConv2Dmaxpool_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``*
paddingVALID*
strides
2
conv_3/Conv2D�
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOp�
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``2
conv_3/BiasAdd�
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOp�
conv_4/Conv2DConv2Dconv_3/BiasAdd:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^*
paddingVALID*
strides
2
conv_4/Conv2D�
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_4/BiasAdd/ReadVariableOp�
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������^^2
conv_4/BiasAdd�
maxpool_2/MaxPoolMaxPoolconv_4/BiasAdd:output:0*/
_output_shapes
:���������//*
ksize
*
paddingVALID*
strides
2
maxpool_2/MaxPoolh
patches/ShapeShapemaxpool_2/MaxPool:output:0*
T0*
_output_shapes
:2
patches/Shape�
patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
patches/strided_slice/stack�
patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_1�
patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_2�
patches/strided_sliceStridedSlicepatches/Shape:output:0$patches/strided_slice/stack:output:0&patches/strided_slice/stack_1:output:0&patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
patches/strided_slice�
patches/ExtractImagePatchesExtractImagePatchesmaxpool_2/MaxPool:output:0*
T0*0
_output_shapes
:���������		�*
ksizes
*
paddingVALID*
rates
*
strides
2
patches/ExtractImagePatches}
patches/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������2
patches/Reshape/shape/1u
patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�2
patches/Reshape/shape/2�
patches/Reshape/shapePackpatches/strided_slice:output:0 patches/Reshape/shape/1:output:0 patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
patches/Reshape/shape�
patches/ReshapeReshape%patches/ExtractImagePatches:patches:0patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:�������������������2
patches/Reshapex
patch_encoder/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
patch_encoder/range/startx
patch_encoder/range/limitConst*
_output_shapes
: *
dtype0*
value	B :Q2
patch_encoder/range/limitx
patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder/range/delta�
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes
:Q2
patch_encoder/range�
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02.
,patch_encoder/dense/Tensordot/ReadVariableOp�
"patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"patch_encoder/dense/Tensordot/axes�
"patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"patch_encoder/dense/Tensordot/free�
#patch_encoder/dense/Tensordot/ShapeShapepatches/Reshape:output:0*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/Shape�
+patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/GatherV2/axis�
&patch_encoder/dense/Tensordot/GatherV2GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/free:output:04patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/GatherV2�
-patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-patch_encoder/dense/Tensordot/GatherV2_1/axis�
(patch_encoder/dense/Tensordot/GatherV2_1GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/axes:output:06patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(patch_encoder/dense/Tensordot/GatherV2_1�
#patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#patch_encoder/dense/Tensordot/Const�
"patch_encoder/dense/Tensordot/ProdProd/patch_encoder/dense/Tensordot/GatherV2:output:0,patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"patch_encoder/dense/Tensordot/Prod�
%patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%patch_encoder/dense/Tensordot/Const_1�
$patch_encoder/dense/Tensordot/Prod_1Prod1patch_encoder/dense/Tensordot/GatherV2_1:output:0.patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$patch_encoder/dense/Tensordot/Prod_1�
)patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)patch_encoder/dense/Tensordot/concat/axis�
$patch_encoder/dense/Tensordot/concatConcatV2+patch_encoder/dense/Tensordot/free:output:0+patch_encoder/dense/Tensordot/axes:output:02patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$patch_encoder/dense/Tensordot/concat�
#patch_encoder/dense/Tensordot/stackPack+patch_encoder/dense/Tensordot/Prod:output:0-patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/stack�
'patch_encoder/dense/Tensordot/transpose	Transposepatches/Reshape:output:0-patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:�������������������2)
'patch_encoder/dense/Tensordot/transpose�
%patch_encoder/dense/Tensordot/ReshapeReshape+patch_encoder/dense/Tensordot/transpose:y:0,patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2'
%patch_encoder/dense/Tensordot/Reshape�
$patch_encoder/dense/Tensordot/MatMulMatMul.patch_encoder/dense/Tensordot/Reshape:output:04patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2&
$patch_encoder/dense/Tensordot/MatMul�
%patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2'
%patch_encoder/dense/Tensordot/Const_2�
+patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/concat_1/axis�
&patch_encoder/dense/Tensordot/concat_1ConcatV2/patch_encoder/dense/Tensordot/GatherV2:output:0.patch_encoder/dense/Tensordot/Const_2:output:04patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/concat_1�
patch_encoder/dense/TensordotReshape.patch_encoder/dense/Tensordot/MatMul:product:0/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@2
patch_encoder/dense/Tensordot�
*patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp3patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*patch_encoder/dense/BiasAdd/ReadVariableOp�
patch_encoder/dense/BiasAddBiasAdd&patch_encoder/dense/Tensordot:output:02patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2
patch_encoder/dense/BiasAdd�
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_429965patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/429965*
_output_shapes

:Q@*
dtype02*
(patch_encoder/embedding/embedding_lookup�
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/429965*
_output_shapes

:Q@23
1patch_encoder/embedding/embedding_lookup/Identity�
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@25
3patch_encoder/embedding/embedding_lookup/Identity_1�
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:���������Q@2
patch_encoder/add�
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices�
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2"
 layer_normalization/moments/mean�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2*
(layer_normalization/moments/StopGradient�
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2/
-layer_normalization/moments/SquaredDifference�
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2&
$layer_normalization/moments/variance�
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52%
#layer_normalization/batchnorm/add/y�
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2#
!layer_normalization/batchnorm/add�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization/batchnorm/Rsqrt�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2#
!layer_normalization/batchnorm/mul�
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization/batchnorm/mul_1�
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization/batchnorm/mul_2�
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOp�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2#
!layer_normalization/batchnorm/sub�
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization/batchnorm/add_1�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp�
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOp�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2 
multi_head_attention/query/add�
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp�
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOp�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
multi_head_attention/key/add�
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp�
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOp�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/y�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
multi_head_attention/Mul�
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsum�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2&
$multi_head_attention/softmax/Softmax�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������QQ2'
%multi_head_attention/dropout/Identity�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsum�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsum�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOp�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2+
)multi_head_attention/attention_output/add�
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*+
_output_shapes
:���������Q@2	
add/add�
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices�
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2$
"layer_normalization_1/moments/mean�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2,
*layer_normalization_1/moments/StopGradient�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@21
/layer_normalization_1/moments/SquaredDifference�
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2(
&layer_normalization_1/moments/variance�
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52'
%layer_normalization_1/batchnorm/add/y�
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization_1/batchnorm/add�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2'
%layer_normalization_1/batchnorm/Rsqrt�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_1/batchnorm/mul�
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_1/batchnorm/mul_1�
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_1/batchnorm/mul_2�
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_1/batchnorm/sub�
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_1/batchnorm/add_1�
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes�
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/free�
dense_1/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape�
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axis�
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2�
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis�
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const�
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod�
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1�
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1�
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axis�
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat�
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack�
dense_1/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@2
dense_1/Tensordot/transpose�
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_1/Tensordot/Reshape�
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/Tensordot/MatMul�
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_1/Tensordot/Const_2�
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axis�
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1�
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Tensordot�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/x�
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_1/Gelu/Cast/x�
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/truediv|
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1/Gelu/add/x�
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/add�
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2
dense_1/Gelu/mul_1�
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes�
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/freex
dense_2/Tensordot/ShapeShapedense_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape�
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis�
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2�
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis�
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const�
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod�
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1�
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1�
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis�
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat�
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack�
dense_2/Tensordot/transpose	Transposedense_1/Gelu/mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�2
dense_2/Tensordot/transpose�
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_2/Tensordot/Reshape�
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/Tensordot/MatMul�
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_2/Tensordot/Const_2�
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis�
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1�
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Tensordot�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/x�
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_2/Gelu/Cast/x�
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/truediv{
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_2/Gelu/add/x�
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/add�
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2
dense_2/Gelu/mul_1z
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*+
_output_shapes
:���������Q@2
	add_1/add�
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indices�
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2$
"layer_normalization_2/moments/mean�
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2,
*layer_normalization_2/moments/StopGradient�
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@21
/layer_normalization_2/moments/SquaredDifference�
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices�
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2(
&layer_normalization_2/moments/variance�
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52'
%layer_normalization_2/batchnorm/add/y�
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization_2/batchnorm/add�
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2'
%layer_normalization_2/batchnorm/Rsqrt�
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOp�
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_2/batchnorm/mul�
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_2/batchnorm/mul_1�
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_2/batchnorm/mul_2�
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOp�
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_2/batchnorm/sub�
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_2/batchnorm/add_1�
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp�
*multi_head_attention_1/query/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsum�
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/query/add/ReadVariableOp�
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2"
 multi_head_attention_1/query/add�
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp�
(multi_head_attention_1/key/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum�
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_1/key/add/ReadVariableOp�
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2 
multi_head_attention_1/key/add�
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp�
*multi_head_attention_1/value/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsum�
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/value/add/ReadVariableOp�
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2"
 multi_head_attention_1/value/add�
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_1/Mul/y�
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
multi_head_attention_1/Mul�
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum�
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2(
&multi_head_attention_1/softmax/Softmax�
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������QQ2)
'multi_head_attention_1/dropout/Identity�
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsum�
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp�
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsum�
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOp�
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2-
+multi_head_attention_1/attention_output/add�
	add_2/addAddV2/multi_head_attention_1/attention_output/add:z:0add_1/add:z:0*
T0*+
_output_shapes
:���������Q@2
	add_2/add�
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indices�
"layer_normalization_3/moments/meanMeanadd_2/add:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2$
"layer_normalization_3/moments/mean�
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2,
*layer_normalization_3/moments/StopGradient�
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd_2/add:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@21
/layer_normalization_3/moments/SquaredDifference�
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indices�
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2(
&layer_normalization_3/moments/variance�
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52'
%layer_normalization_3/batchnorm/add/y�
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization_3/batchnorm/add�
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2'
%layer_normalization_3/batchnorm/Rsqrt�
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOp�
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_3/batchnorm/mul�
%layer_normalization_3/batchnorm/mul_1Muladd_2/add:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_3/batchnorm/mul_1�
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_3/batchnorm/mul_2�
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_3/batchnorm/ReadVariableOp�
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_3/batchnorm/sub�
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_3/batchnorm/add_1�
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axes�
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/free�
dense_3/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shape�
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axis�
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2�
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis�
dense_3/Tensordot/GatherV2_1GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/axes:output:0*dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2_1|
dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const�
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod�
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1�
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1�
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axis�
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat�
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack�
dense_3/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@2
dense_3/Tensordot/transpose�
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_3/Tensordot/Reshape�
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_3/Tensordot/MatMul�
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
dense_3/Tensordot/Const_2�
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axis�
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1�
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Tensordot�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2
dense_3/BiasAddm
dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_3/Gelu/mul/x�
dense_3/Gelu/mulMuldense_3/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/mulo
dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_3/Gelu/Cast/x�
dense_3/Gelu/truedivRealDivdense_3/BiasAdd:output:0dense_3/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/truediv|
dense_3/Gelu/ErfErfdense_3/Gelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/Erfm
dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_3/Gelu/add/x�
dense_3/Gelu/addAddV2dense_3/Gelu/add/x:output:0dense_3/Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/add�
dense_3/Gelu/mul_1Muldense_3/Gelu/mul:z:0dense_3/Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2
dense_3/Gelu/mul_1�
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes�
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/freex
dense_4/Tensordot/ShapeShapedense_3/Gelu/mul_1:z:0*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape�
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axis�
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2�
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis�
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const�
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod�
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1�
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1�
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axis�
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat�
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack�
dense_4/Tensordot/transpose	Transposedense_3/Gelu/mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:���������Q�2
dense_4/Tensordot/transpose�
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
dense_4/Tensordot/Reshape�
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_4/Tensordot/MatMul�
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_4/Tensordot/Const_2�
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axis�
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1�
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Tensordot�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
dense_4/BiasAddm
dense_4/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_4/Gelu/mul/x�
dense_4/Gelu/mulMuldense_4/Gelu/mul/x:output:0dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/mulo
dense_4/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_4/Gelu/Cast/x�
dense_4/Gelu/truedivRealDivdense_4/BiasAdd:output:0dense_4/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/truediv{
dense_4/Gelu/ErfErfdense_4/Gelu/truediv:z:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/Erfm
dense_4/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_4/Gelu/add/x�
dense_4/Gelu/addAddV2dense_4/Gelu/add/x:output:0dense_4/Gelu/Erf:y:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/add�
dense_4/Gelu/mul_1Muldense_4/Gelu/mul:z:0dense_4/Gelu/add:z:0*
T0*+
_output_shapes
:���������Q@2
dense_4/Gelu/mul_1|
	add_3/addAddV2dense_4/Gelu/mul_1:z:0add_2/add:z:0*
T0*+
_output_shapes
:���������Q@2
	add_3/add�
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indices�
"layer_normalization_4/moments/meanMeanadd_3/add:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2$
"layer_normalization_4/moments/mean�
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:���������Q2,
*layer_normalization_4/moments/StopGradient�
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd_3/add:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@21
/layer_normalization_4/moments/SquaredDifference�
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indices�
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2(
&layer_normalization_4/moments/variance�
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52'
%layer_normalization_4/batchnorm/add/y�
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2%
#layer_normalization_4/batchnorm/add�
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2'
%layer_normalization_4/batchnorm/Rsqrt�
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOp�
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_4/batchnorm/mul�
%layer_normalization_4/batchnorm/mul_1Muladd_3/add:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_4/batchnorm/mul_1�
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_4/batchnorm/mul_2�
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_4/batchnorm/ReadVariableOp�
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2%
#layer_normalization_4/batchnorm/sub�
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2'
%layer_normalization_4/batchnorm/add_1{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
flatten_layer/Const�
flatten_layer/ReshapeReshape)layer_normalization_4/batchnorm/add_1:z:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:����������(2
flatten_layer/Reshape�
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	�(2*
dtype02
FC_1/MatMul/ReadVariableOp�
FC_1/MatMulMatMulflatten_layer/Reshape:output:0"FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
FC_1/MatMul�
FC_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_1/BiasAdd/ReadVariableOp�
FC_1/BiasAddBiasAddFC_1/MatMul:product:0#FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
FC_1/BiasAdd�
leaky_ReLu_1/LeakyRelu	LeakyReluFC_1/BiasAdd:output:0*'
_output_shapes
:���������2*
alpha%���>2
leaky_ReLu_1/LeakyRelu�
FC_2/MatMul/ReadVariableOpReadVariableOp#fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
FC_2/MatMul/ReadVariableOp�
FC_2/MatMulMatMul$leaky_ReLu_1/LeakyRelu:activations:0"FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
FC_2/MatMul�
FC_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_2/BiasAdd/ReadVariableOp�
FC_2/BiasAddBiasAddFC_2/MatMul:product:0#FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
FC_2/BiasAdd�
leaky_ReLu_2/LeakyRelu	LeakyReluFC_2/BiasAdd:output:0*'
_output_shapes
:���������2*
alpha%���>2
leaky_ReLu_2/LeakyRelu�
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02$
"output_layer/MatMul/ReadVariableOp�
output_layer/MatMulMatMul$leaky_ReLu_2/LeakyRelu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output_layer/MatMul�
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOp�
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
output_layer/BiasAdd�
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
output_layer/Softmax�
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
FC_1/BiasAdd/ReadVariableOpFC_1/BiasAdd/ReadVariableOp28
FC_1/MatMul/ReadVariableOpFC_1/MatMul/ReadVariableOp2:
FC_2/BiasAdd/ReadVariableOpFC_2/BiasAdd/ReadVariableOp28
FC_2/MatMul/ReadVariableOpFC_2/MatMul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2>
conv_3/BiasAdd/ReadVariableOpconv_3/BiasAdd/ReadVariableOp2<
conv_3/Conv2D/ReadVariableOpconv_3/Conv2D/ReadVariableOp2>
conv_4/BiasAdd/ReadVariableOpconv_4/BiasAdd/ReadVariableOp2<
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2`
.layer_normalization_3/batchnorm/ReadVariableOp.layer_normalization_3/batchnorm/ReadVariableOp2h
2layer_normalization_3/batchnorm/mul/ReadVariableOp2layer_normalization_3/batchnorm/mul/ReadVariableOp2`
.layer_normalization_4/batchnorm/ReadVariableOp.layer_normalization_4/batchnorm/ReadVariableOp2h
2layer_normalization_4/batchnorm/mul/ReadVariableOp2layer_normalization_4/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2�
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2X
*patch_encoder/dense/BiasAdd/ReadVariableOp*patch_encoder/dense/BiasAdd/ReadVariableOp2\
,patch_encoder/dense/Tensordot/ReadVariableOp,patch_encoder/dense/Tensordot/ReadVariableOp2T
(patch_encoder/embedding/embedding_lookup(patch_encoder/embedding/embedding_lookup:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
k
A__inference_add_1_layer_call_and_return_conditional_losses_428235

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs:SO
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
�
6__inference_layer_normalization_1_layer_call_fn_431243

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_4281352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
��
�M
__inference__traced_save_432282
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop,
(savev2_conv_4_kernel_read_readvariableop*
&savev2_conv_4_bias_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop:
6savev2_layer_normalization_3_gamma_read_readvariableop9
5savev2_layer_normalization_3_beta_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop:
6savev2_layer_normalization_4_gamma_read_readvariableop9
5savev2_layer_normalization_4_beta_read_readvariableop*
&savev2_fc_1_kernel_read_readvariableop(
$savev2_fc_1_bias_read_readvariableop*
&savev2_fc_2_kernel_read_readvariableop(
$savev2_fc_2_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop)
%savev2_adamw_iter_read_readvariableop	+
'savev2_adamw_beta_1_read_readvariableop+
'savev2_adamw_beta_2_read_readvariableop*
&savev2_adamw_decay_read_readvariableop2
.savev2_adamw_learning_rate_read_readvariableop1
-savev2_adamw_weight_decay_read_readvariableop9
5savev2_patch_encoder_dense_kernel_read_readvariableop7
3savev2_patch_encoder_dense_bias_read_readvariableopA
=savev2_patch_encoder_embedding_embeddings_read_readvariableop@
<savev2_multi_head_attention_query_kernel_read_readvariableop>
:savev2_multi_head_attention_query_bias_read_readvariableop>
:savev2_multi_head_attention_key_kernel_read_readvariableop<
8savev2_multi_head_attention_key_bias_read_readvariableop@
<savev2_multi_head_attention_value_kernel_read_readvariableop>
:savev2_multi_head_attention_value_bias_read_readvariableopK
Gsavev2_multi_head_attention_attention_output_kernel_read_readvariableopI
Esavev2_multi_head_attention_attention_output_bias_read_readvariableopB
>savev2_multi_head_attention_1_query_kernel_read_readvariableop@
<savev2_multi_head_attention_1_query_bias_read_readvariableop@
<savev2_multi_head_attention_1_key_kernel_read_readvariableop>
:savev2_multi_head_attention_1_key_bias_read_readvariableopB
>savev2_multi_head_attention_1_value_kernel_read_readvariableop@
<savev2_multi_head_attention_1_value_bias_read_readvariableopM
Isavev2_multi_head_attention_1_attention_output_kernel_read_readvariableopK
Gsavev2_multi_head_attention_1_attention_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adamw_conv_1_kernel_m_read_readvariableop2
.savev2_adamw_conv_1_bias_m_read_readvariableop4
0savev2_adamw_conv_2_kernel_m_read_readvariableop2
.savev2_adamw_conv_2_bias_m_read_readvariableop4
0savev2_adamw_conv_3_kernel_m_read_readvariableop2
.savev2_adamw_conv_3_bias_m_read_readvariableop4
0savev2_adamw_conv_4_kernel_m_read_readvariableop2
.savev2_adamw_conv_4_bias_m_read_readvariableop@
<savev2_adamw_layer_normalization_gamma_m_read_readvariableop?
;savev2_adamw_layer_normalization_beta_m_read_readvariableopB
>savev2_adamw_layer_normalization_1_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_1_beta_m_read_readvariableop5
1savev2_adamw_dense_1_kernel_m_read_readvariableop3
/savev2_adamw_dense_1_bias_m_read_readvariableop5
1savev2_adamw_dense_2_kernel_m_read_readvariableop3
/savev2_adamw_dense_2_bias_m_read_readvariableopB
>savev2_adamw_layer_normalization_2_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_2_beta_m_read_readvariableopB
>savev2_adamw_layer_normalization_3_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_3_beta_m_read_readvariableop5
1savev2_adamw_dense_3_kernel_m_read_readvariableop3
/savev2_adamw_dense_3_bias_m_read_readvariableop5
1savev2_adamw_dense_4_kernel_m_read_readvariableop3
/savev2_adamw_dense_4_bias_m_read_readvariableopB
>savev2_adamw_layer_normalization_4_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_4_beta_m_read_readvariableop2
.savev2_adamw_fc_1_kernel_m_read_readvariableop0
,savev2_adamw_fc_1_bias_m_read_readvariableop2
.savev2_adamw_fc_2_kernel_m_read_readvariableop0
,savev2_adamw_fc_2_bias_m_read_readvariableop:
6savev2_adamw_output_layer_kernel_m_read_readvariableop8
4savev2_adamw_output_layer_bias_m_read_readvariableopA
=savev2_adamw_patch_encoder_dense_kernel_m_read_readvariableop?
;savev2_adamw_patch_encoder_dense_bias_m_read_readvariableopI
Esavev2_adamw_patch_encoder_embedding_embeddings_m_read_readvariableopH
Dsavev2_adamw_multi_head_attention_query_kernel_m_read_readvariableopF
Bsavev2_adamw_multi_head_attention_query_bias_m_read_readvariableopF
Bsavev2_adamw_multi_head_attention_key_kernel_m_read_readvariableopD
@savev2_adamw_multi_head_attention_key_bias_m_read_readvariableopH
Dsavev2_adamw_multi_head_attention_value_kernel_m_read_readvariableopF
Bsavev2_adamw_multi_head_attention_value_bias_m_read_readvariableopS
Osavev2_adamw_multi_head_attention_attention_output_kernel_m_read_readvariableopQ
Msavev2_adamw_multi_head_attention_attention_output_bias_m_read_readvariableopJ
Fsavev2_adamw_multi_head_attention_1_query_kernel_m_read_readvariableopH
Dsavev2_adamw_multi_head_attention_1_query_bias_m_read_readvariableopH
Dsavev2_adamw_multi_head_attention_1_key_kernel_m_read_readvariableopF
Bsavev2_adamw_multi_head_attention_1_key_bias_m_read_readvariableopJ
Fsavev2_adamw_multi_head_attention_1_value_kernel_m_read_readvariableopH
Dsavev2_adamw_multi_head_attention_1_value_bias_m_read_readvariableopU
Qsavev2_adamw_multi_head_attention_1_attention_output_kernel_m_read_readvariableopS
Osavev2_adamw_multi_head_attention_1_attention_output_bias_m_read_readvariableop4
0savev2_adamw_conv_1_kernel_v_read_readvariableop2
.savev2_adamw_conv_1_bias_v_read_readvariableop4
0savev2_adamw_conv_2_kernel_v_read_readvariableop2
.savev2_adamw_conv_2_bias_v_read_readvariableop4
0savev2_adamw_conv_3_kernel_v_read_readvariableop2
.savev2_adamw_conv_3_bias_v_read_readvariableop4
0savev2_adamw_conv_4_kernel_v_read_readvariableop2
.savev2_adamw_conv_4_bias_v_read_readvariableop@
<savev2_adamw_layer_normalization_gamma_v_read_readvariableop?
;savev2_adamw_layer_normalization_beta_v_read_readvariableopB
>savev2_adamw_layer_normalization_1_gamma_v_read_readvariableopA
=savev2_adamw_layer_normalization_1_beta_v_read_readvariableop5
1savev2_adamw_dense_1_kernel_v_read_readvariableop3
/savev2_adamw_dense_1_bias_v_read_readvariableop5
1savev2_adamw_dense_2_kernel_v_read_readvariableop3
/savev2_adamw_dense_2_bias_v_read_readvariableopB
>savev2_adamw_layer_normalization_2_gamma_v_read_readvariableopA
=savev2_adamw_layer_normalization_2_beta_v_read_readvariableopB
>savev2_adamw_layer_normalization_3_gamma_v_read_readvariableopA
=savev2_adamw_layer_normalization_3_beta_v_read_readvariableop5
1savev2_adamw_dense_3_kernel_v_read_readvariableop3
/savev2_adamw_dense_3_bias_v_read_readvariableop5
1savev2_adamw_dense_4_kernel_v_read_readvariableop3
/savev2_adamw_dense_4_bias_v_read_readvariableopB
>savev2_adamw_layer_normalization_4_gamma_v_read_readvariableopA
=savev2_adamw_layer_normalization_4_beta_v_read_readvariableop2
.savev2_adamw_fc_1_kernel_v_read_readvariableop0
,savev2_adamw_fc_1_bias_v_read_readvariableop2
.savev2_adamw_fc_2_kernel_v_read_readvariableop0
,savev2_adamw_fc_2_bias_v_read_readvariableop:
6savev2_adamw_output_layer_kernel_v_read_readvariableop8
4savev2_adamw_output_layer_bias_v_read_readvariableopA
=savev2_adamw_patch_encoder_dense_kernel_v_read_readvariableop?
;savev2_adamw_patch_encoder_dense_bias_v_read_readvariableopI
Esavev2_adamw_patch_encoder_embedding_embeddings_v_read_readvariableopH
Dsavev2_adamw_multi_head_attention_query_kernel_v_read_readvariableopF
Bsavev2_adamw_multi_head_attention_query_bias_v_read_readvariableopF
Bsavev2_adamw_multi_head_attention_key_kernel_v_read_readvariableopD
@savev2_adamw_multi_head_attention_key_bias_v_read_readvariableopH
Dsavev2_adamw_multi_head_attention_value_kernel_v_read_readvariableopF
Bsavev2_adamw_multi_head_attention_value_bias_v_read_readvariableopS
Osavev2_adamw_multi_head_attention_attention_output_kernel_v_read_readvariableopQ
Msavev2_adamw_multi_head_attention_attention_output_bias_v_read_readvariableopJ
Fsavev2_adamw_multi_head_attention_1_query_kernel_v_read_readvariableopH
Dsavev2_adamw_multi_head_attention_1_query_bias_v_read_readvariableopH
Dsavev2_adamw_multi_head_attention_1_key_kernel_v_read_readvariableopF
Bsavev2_adamw_multi_head_attention_1_key_bias_v_read_readvariableopJ
Fsavev2_adamw_multi_head_attention_1_value_kernel_v_read_readvariableopH
Dsavev2_adamw_multi_head_attention_1_value_bias_v_read_readvariableopU
Qsavev2_adamw_multi_head_attention_1_attention_output_kernel_v_read_readvariableopS
Osavev2_adamw_multi_head_attention_1_attention_output_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�[
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�Z
value�ZB�Z�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/33/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/34/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/35/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�J
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop6savev2_layer_normalization_3_gamma_read_readvariableop5savev2_layer_normalization_3_beta_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_2_kernel_read_readvariableop$savev2_fc_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop%savev2_adamw_iter_read_readvariableop'savev2_adamw_beta_1_read_readvariableop'savev2_adamw_beta_2_read_readvariableop&savev2_adamw_decay_read_readvariableop.savev2_adamw_learning_rate_read_readvariableop-savev2_adamw_weight_decay_read_readvariableop5savev2_patch_encoder_dense_kernel_read_readvariableop3savev2_patch_encoder_dense_bias_read_readvariableop=savev2_patch_encoder_embedding_embeddings_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop>savev2_multi_head_attention_1_query_kernel_read_readvariableop<savev2_multi_head_attention_1_query_bias_read_readvariableop<savev2_multi_head_attention_1_key_kernel_read_readvariableop:savev2_multi_head_attention_1_key_bias_read_readvariableop>savev2_multi_head_attention_1_value_kernel_read_readvariableop<savev2_multi_head_attention_1_value_bias_read_readvariableopIsavev2_multi_head_attention_1_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_1_attention_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adamw_conv_1_kernel_m_read_readvariableop.savev2_adamw_conv_1_bias_m_read_readvariableop0savev2_adamw_conv_2_kernel_m_read_readvariableop.savev2_adamw_conv_2_bias_m_read_readvariableop0savev2_adamw_conv_3_kernel_m_read_readvariableop.savev2_adamw_conv_3_bias_m_read_readvariableop0savev2_adamw_conv_4_kernel_m_read_readvariableop.savev2_adamw_conv_4_bias_m_read_readvariableop<savev2_adamw_layer_normalization_gamma_m_read_readvariableop;savev2_adamw_layer_normalization_beta_m_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_1_beta_m_read_readvariableop1savev2_adamw_dense_1_kernel_m_read_readvariableop/savev2_adamw_dense_1_bias_m_read_readvariableop1savev2_adamw_dense_2_kernel_m_read_readvariableop/savev2_adamw_dense_2_bias_m_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_2_beta_m_read_readvariableop>savev2_adamw_layer_normalization_3_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_3_beta_m_read_readvariableop1savev2_adamw_dense_3_kernel_m_read_readvariableop/savev2_adamw_dense_3_bias_m_read_readvariableop1savev2_adamw_dense_4_kernel_m_read_readvariableop/savev2_adamw_dense_4_bias_m_read_readvariableop>savev2_adamw_layer_normalization_4_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_4_beta_m_read_readvariableop.savev2_adamw_fc_1_kernel_m_read_readvariableop,savev2_adamw_fc_1_bias_m_read_readvariableop.savev2_adamw_fc_2_kernel_m_read_readvariableop,savev2_adamw_fc_2_bias_m_read_readvariableop6savev2_adamw_output_layer_kernel_m_read_readvariableop4savev2_adamw_output_layer_bias_m_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_m_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_m_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_m_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_m_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_m_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_m_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_m_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_m_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_m_read_readvariableopFsavev2_adamw_multi_head_attention_1_query_kernel_m_read_readvariableopDsavev2_adamw_multi_head_attention_1_query_bias_m_read_readvariableopDsavev2_adamw_multi_head_attention_1_key_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_1_key_bias_m_read_readvariableopFsavev2_adamw_multi_head_attention_1_value_kernel_m_read_readvariableopDsavev2_adamw_multi_head_attention_1_value_bias_m_read_readvariableopQsavev2_adamw_multi_head_attention_1_attention_output_kernel_m_read_readvariableopOsavev2_adamw_multi_head_attention_1_attention_output_bias_m_read_readvariableop0savev2_adamw_conv_1_kernel_v_read_readvariableop.savev2_adamw_conv_1_bias_v_read_readvariableop0savev2_adamw_conv_2_kernel_v_read_readvariableop.savev2_adamw_conv_2_bias_v_read_readvariableop0savev2_adamw_conv_3_kernel_v_read_readvariableop.savev2_adamw_conv_3_bias_v_read_readvariableop0savev2_adamw_conv_4_kernel_v_read_readvariableop.savev2_adamw_conv_4_bias_v_read_readvariableop<savev2_adamw_layer_normalization_gamma_v_read_readvariableop;savev2_adamw_layer_normalization_beta_v_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_1_beta_v_read_readvariableop1savev2_adamw_dense_1_kernel_v_read_readvariableop/savev2_adamw_dense_1_bias_v_read_readvariableop1savev2_adamw_dense_2_kernel_v_read_readvariableop/savev2_adamw_dense_2_bias_v_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_2_beta_v_read_readvariableop>savev2_adamw_layer_normalization_3_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_3_beta_v_read_readvariableop1savev2_adamw_dense_3_kernel_v_read_readvariableop/savev2_adamw_dense_3_bias_v_read_readvariableop1savev2_adamw_dense_4_kernel_v_read_readvariableop/savev2_adamw_dense_4_bias_v_read_readvariableop>savev2_adamw_layer_normalization_4_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_4_beta_v_read_readvariableop.savev2_adamw_fc_1_kernel_v_read_readvariableop,savev2_adamw_fc_1_bias_v_read_readvariableop.savev2_adamw_fc_2_kernel_v_read_readvariableop,savev2_adamw_fc_2_bias_v_read_readvariableop6savev2_adamw_output_layer_kernel_v_read_readvariableop4savev2_adamw_output_layer_bias_v_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_v_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_v_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_v_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_v_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_v_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_v_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_v_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_v_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_v_read_readvariableopFsavev2_adamw_multi_head_attention_1_query_kernel_v_read_readvariableopDsavev2_adamw_multi_head_attention_1_query_bias_v_read_readvariableopDsavev2_adamw_multi_head_attention_1_key_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_1_key_bias_v_read_readvariableopFsavev2_adamw_multi_head_attention_1_value_kernel_v_read_readvariableopDsavev2_adamw_multi_head_attention_1_value_bias_v_read_readvariableopQsavev2_adamw_multi_head_attention_1_attention_output_kernel_v_read_readvariableopOsavev2_adamw_multi_head_attention_1_attention_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
�2�	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :
:
:

:
:
::::@:@:@:@:	@�:�:	�@:@:@:@:@:@:	@�:�:	�@:@:@:@:	�(2:2:22:2:2:: : : : : : :	�@:@:Q@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@: : : : :
:
:

:
:
::::@:@:@:@:	@�:�:	�@:@:@:@:@:@:	@�:�:	�@:@:@:@:	�(2:2:22:2:2::	�@:@:Q@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:
:
:

:
:
::::@:@:@:@:	@�:�:	�@:@:@:@:@:@:	@�:�:	�@:@:@:@:	�(2:2:22:2:2::	�@:@:Q@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@�:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@�:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	�(2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2:  

_output_shapes
::!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :%'!

_output_shapes
:	�@: (

_output_shapes
:@:$) 

_output_shapes

:Q@:(*$
"
_output_shapes
:@@:$+ 

_output_shapes

:@:(,$
"
_output_shapes
:@@:$- 

_output_shapes

:@:(.$
"
_output_shapes
:@@:$/ 

_output_shapes

:@:(0$
"
_output_shapes
:@@: 1

_output_shapes
:@:(2$
"
_output_shapes
:@@:$3 

_output_shapes

:@:(4$
"
_output_shapes
:@@:$5 

_output_shapes

:@:(6$
"
_output_shapes
:@@:$7 

_output_shapes

:@:(8$
"
_output_shapes
:@@: 9

_output_shapes
:@::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :,>(
&
_output_shapes
:
: ?

_output_shapes
:
:,@(
&
_output_shapes
:

: A

_output_shapes
:
:,B(
&
_output_shapes
:
: C

_output_shapes
::,D(
&
_output_shapes
:: E

_output_shapes
:: F

_output_shapes
:@: G

_output_shapes
:@: H

_output_shapes
:@: I

_output_shapes
:@:%J!

_output_shapes
:	@�:!K

_output_shapes	
:�:%L!

_output_shapes
:	�@: M

_output_shapes
:@: N

_output_shapes
:@: O

_output_shapes
:@: P

_output_shapes
:@: Q

_output_shapes
:@:%R!

_output_shapes
:	@�:!S

_output_shapes	
:�:%T!

_output_shapes
:	�@: U

_output_shapes
:@: V

_output_shapes
:@: W

_output_shapes
:@:%X!

_output_shapes
:	�(2: Y

_output_shapes
:2:$Z 

_output_shapes

:22: [

_output_shapes
:2:$\ 

_output_shapes

:2: ]

_output_shapes
::%^!

_output_shapes
:	�@: _

_output_shapes
:@:$` 

_output_shapes

:Q@:(a$
"
_output_shapes
:@@:$b 

_output_shapes

:@:(c$
"
_output_shapes
:@@:$d 

_output_shapes

:@:(e$
"
_output_shapes
:@@:$f 

_output_shapes

:@:(g$
"
_output_shapes
:@@: h

_output_shapes
:@:(i$
"
_output_shapes
:@@:$j 

_output_shapes

:@:(k$
"
_output_shapes
:@@:$l 

_output_shapes

:@:(m$
"
_output_shapes
:@@:$n 

_output_shapes

:@:(o$
"
_output_shapes
:@@: p

_output_shapes
:@:,q(
&
_output_shapes
:
: r

_output_shapes
:
:,s(
&
_output_shapes
:

: t

_output_shapes
:
:,u(
&
_output_shapes
:
: v

_output_shapes
::,w(
&
_output_shapes
:: x

_output_shapes
:: y

_output_shapes
:@: z

_output_shapes
:@: {

_output_shapes
:@: |

_output_shapes
:@:%}!

_output_shapes
:	@�:!~

_output_shapes	
:�:%!

_output_shapes
:	�@:!�

_output_shapes
:@:!�

_output_shapes
:@:!�

_output_shapes
:@:!�

_output_shapes
:@:!�

_output_shapes
:@:&�!

_output_shapes
:	@�:"�

_output_shapes	
:�:&�!

_output_shapes
:	�@:!�

_output_shapes
:@:!�

_output_shapes
:@:!�

_output_shapes
:@:&�!

_output_shapes
:	�(2:!�

_output_shapes
:2:%� 

_output_shapes

:22:!�

_output_shapes
:2:%� 

_output_shapes

:2:!�

_output_shapes
::&�!

_output_shapes
:	�@:!�

_output_shapes
:@:%� 

_output_shapes

:Q@:)�$
"
_output_shapes
:@@:%� 

_output_shapes

:@:)�$
"
_output_shapes
:@@:%� 

_output_shapes

:@:)�$
"
_output_shapes
:@@:%� 

_output_shapes

:@:)�$
"
_output_shapes
:@@:!�

_output_shapes
:@:)�$
"
_output_shapes
:@@:%� 

_output_shapes

:@:)�$
"
_output_shapes
:@@:%� 

_output_shapes

:@:)�$
"
_output_shapes
:@@:%� 

_output_shapes

:@:)�$
"
_output_shapes
:@@:!�

_output_shapes
:@:�

_output_shapes
: 
�
�
(__inference_dense_2_layer_call_fn_431337

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4282232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������Q�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������Q�
 
_user_specified_nameinputs
�
�
(__inference_dense_3_layer_call_fn_431591

inputs
unknown:	@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4283922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:���������Q�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
��
�
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_429297

inputs'
conv_1_429165:

conv_1_429167:
'
conv_2_429170:


conv_2_429172:
'
conv_3_429176:

conv_3_429178:'
conv_4_429181:
conv_4_429183:'
patch_encoder_429188:	�@"
patch_encoder_429190:@&
patch_encoder_429192:Q@(
layer_normalization_429195:@(
layer_normalization_429197:@1
multi_head_attention_429200:@@-
multi_head_attention_429202:@1
multi_head_attention_429204:@@-
multi_head_attention_429206:@1
multi_head_attention_429208:@@-
multi_head_attention_429210:@1
multi_head_attention_429212:@@)
multi_head_attention_429214:@*
layer_normalization_1_429218:@*
layer_normalization_1_429220:@!
dense_1_429223:	@�
dense_1_429225:	�!
dense_2_429228:	�@
dense_2_429230:@*
layer_normalization_2_429234:@*
layer_normalization_2_429236:@3
multi_head_attention_1_429239:@@/
multi_head_attention_1_429241:@3
multi_head_attention_1_429243:@@/
multi_head_attention_1_429245:@3
multi_head_attention_1_429247:@@/
multi_head_attention_1_429249:@3
multi_head_attention_1_429251:@@+
multi_head_attention_1_429253:@*
layer_normalization_3_429257:@*
layer_normalization_3_429259:@!
dense_3_429262:	@�
dense_3_429264:	�!
dense_4_429267:	�@
dense_4_429269:@*
layer_normalization_4_429273:@*
layer_normalization_4_429275:@
fc_1_429279:	�(2
fc_1_429281:2
fc_2_429285:22
fc_2_429287:2%
output_layer_429291:2!
output_layer_429293:
identity��FC_1/StatefulPartitionedCall�FC_2/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�conv_3/StatefulPartitionedCall�conv_4/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�-layer_normalization_3/StatefulPartitionedCall�-layer_normalization_4/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�.multi_head_attention_1/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�%patch_encoder/StatefulPartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_429165conv_1_429167*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_4279042 
conv_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_429170conv_2_429172*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_4279202 
conv_2/StatefulPartitionedCall�
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������bb
* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_4278692
maxpool_1/PartitionedCall�
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_429176conv_3_429178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������``*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_4279372 
conv_3/StatefulPartitionedCall�
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_429181conv_4_429183*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������^^*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_4279532 
conv_4/StatefulPartitionedCall�
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������//* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_2_layer_call_and_return_conditional_losses_4278812
maxpool_2/PartitionedCall�
patches/PartitionedCallPartitionedCall"maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_4279742
patches/PartitionedCall�
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_429188patch_encoder_429190patch_encoder_429192*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_4280162'
%patch_encoder/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_429195layer_normalization_429197*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_4280462-
+layer_normalization/StatefulPartitionedCall�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_429200multi_head_attention_429202multi_head_attention_429204multi_head_attention_429206multi_head_attention_429208multi_head_attention_429210multi_head_attention_429212multi_head_attention_429214*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_4289662.
,multi_head_attention/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_4281112
add/PartitionedCall�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_429218layer_normalization_1_429220*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_4281352/
-layer_normalization_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_429223dense_1_429225*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4281792!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_429228dense_2_429230*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4282232!
dense_2/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4282352
add_1/PartitionedCall�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_429234layer_normalization_2_429236*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_4282592/
-layer_normalization_2/StatefulPartitionedCall�
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_429239multi_head_attention_1_429241multi_head_attention_1_429243multi_head_attention_1_429245multi_head_attention_1_429247multi_head_attention_1_429249multi_head_attention_1_429251multi_head_attention_1_429253*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_42882520
.multi_head_attention_1/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_4283242
add_2/PartitionedCall�
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_429257layer_normalization_3_429259*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_4283482/
-layer_normalization_3/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_429262dense_3_429264*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:���������Q�*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_4283922!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_429267dense_4_429269*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4284362!
dense_4/StatefulPartitionedCall�
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_4284482
add_3/PartitionedCall�
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_429273layer_normalization_4_429275*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_4284722/
-layer_normalization_4/StatefulPartitionedCall�
flatten_layer/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������(* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_4284842
flatten_layer/PartitionedCall�
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_429279fc_1_429281*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_4284962
FC_1/StatefulPartitionedCall�
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_4285072
leaky_ReLu_1/PartitionedCall�
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_429285fc_2_429287*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_4285192
FC_2/StatefulPartitionedCall�
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_4285302
leaky_ReLu_2/PartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_429291output_layer_429293*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_4285432&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
(__inference_dense_4_layer_call_fn_431638

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_4284362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������Q�: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:���������Q�
 
_user_specified_nameinputs
�'
�
C__inference_dense_1_layer_call_and_return_conditional_losses_431281

inputs4
!tensordot_readvariableop_resource:	@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������Q@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:���������Q�2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:���������Q�2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xy
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*,
_output_shapes
:���������Q�2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:���������Q�2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:���������Q�2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:���������Q�2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:���������Q�2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:���������Q�2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
�
'__inference_conv_3_layer_call_fn_430959

inputs!
unknown:

	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������``*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_4279372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������bb
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������bb

 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_431535

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�
D
(__inference_patches_layer_call_fn_430997

images
identity�
PartitionedCallPartitionedCallimages*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_4279742
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������//:W S
/
_output_shapes
:���������//
 
_user_specified_nameimages
�

�
7__inference_multi_head_attention_1_layer_call_fn_431479	
query	
value
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_4283002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�
�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_431070

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�7
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_428966	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yj
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/dropout/Const�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������QQ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������QQ2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/Mul_1�
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�
F
*__inference_maxpool_2_layer_call_fn_427887

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *N
fIRG
E__inference_maxpool_2_layer_call_and_return_conditional_losses_4278812
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_428348

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�

�
5__inference_multi_head_attention_layer_call_fn_431178	
query	
value
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@**
_read_only_resource_inputs

	*2
config_proto" 

CPU

GPU2 *0J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_4280872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�
R
&__inference_add_1_layer_call_fn_431349
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������Q@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_4282352
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:U Q
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/1
�
�
%__inference_FC_2_layer_call_fn_431740

inputs
unknown:22
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_4285192
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������2
 
_user_specified_nameinputs
�

�
B__inference_conv_3_layer_call_and_return_conditional_losses_430950

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������``2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������``2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������bb
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������bb

 
_user_specified_nameinputs
�-
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_428087	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yj
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2
softmax/Softmax�
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:���������QQ2
dropout/Identity�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�7
�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_431457	
query	
valueA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������Q@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������Q@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yj
MulMulquery/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������Q@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:���������QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:���������QQ2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/dropout/Const�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:���������QQ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:���������QQ2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:���������QQ2
dropout/dropout/Mul_1�
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:���������Q@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:���������Q@:���������Q@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:R N
+
_output_shapes
:���������Q@

_user_specified_namequery:RN
+
_output_shapes
:���������Q@

_user_specified_namevalue
�
�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_428135

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOp�
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indices�
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:���������Q2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:���������Q@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:���������Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52
batchnorm/add/y�
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:���������Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:���������Q2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:���������Q@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:���������Q@
 
_user_specified_nameinputs
�	
�
@__inference_FC_1_layer_call_and_return_conditional_losses_431702

inputs1
matmul_readvariableop_resource:	�(2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�(2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������(
 
_user_specified_nameinputs
�
k
?__inference_add_layer_call_and_return_conditional_losses_431206
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������Q@:���������Q@:U Q
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:���������Q@
"
_user_specified_name
inputs/1
��
�r
"__inference__traced_restore_432781
file_prefix8
assignvariableop_conv_1_kernel:
,
assignvariableop_1_conv_1_bias:
:
 assignvariableop_2_conv_2_kernel:

,
assignvariableop_3_conv_2_bias:
:
 assignvariableop_4_conv_3_kernel:
,
assignvariableop_5_conv_3_bias::
 assignvariableop_6_conv_4_kernel:,
assignvariableop_7_conv_4_bias::
,assignvariableop_8_layer_normalization_gamma:@9
+assignvariableop_9_layer_normalization_beta:@=
/assignvariableop_10_layer_normalization_1_gamma:@<
.assignvariableop_11_layer_normalization_1_beta:@5
"assignvariableop_12_dense_1_kernel:	@�/
 assignvariableop_13_dense_1_bias:	�5
"assignvariableop_14_dense_2_kernel:	�@.
 assignvariableop_15_dense_2_bias:@=
/assignvariableop_16_layer_normalization_2_gamma:@<
.assignvariableop_17_layer_normalization_2_beta:@=
/assignvariableop_18_layer_normalization_3_gamma:@<
.assignvariableop_19_layer_normalization_3_beta:@5
"assignvariableop_20_dense_3_kernel:	@�/
 assignvariableop_21_dense_3_bias:	�5
"assignvariableop_22_dense_4_kernel:	�@.
 assignvariableop_23_dense_4_bias:@=
/assignvariableop_24_layer_normalization_4_gamma:@<
.assignvariableop_25_layer_normalization_4_beta:@2
assignvariableop_26_fc_1_kernel:	�(2+
assignvariableop_27_fc_1_bias:21
assignvariableop_28_fc_2_kernel:22+
assignvariableop_29_fc_2_bias:29
'assignvariableop_30_output_layer_kernel:23
%assignvariableop_31_output_layer_bias:(
assignvariableop_32_adamw_iter:	 *
 assignvariableop_33_adamw_beta_1: *
 assignvariableop_34_adamw_beta_2: )
assignvariableop_35_adamw_decay: 1
'assignvariableop_36_adamw_learning_rate: 0
&assignvariableop_37_adamw_weight_decay: A
.assignvariableop_38_patch_encoder_dense_kernel:	�@:
,assignvariableop_39_patch_encoder_dense_bias:@H
6assignvariableop_40_patch_encoder_embedding_embeddings:Q@K
5assignvariableop_41_multi_head_attention_query_kernel:@@E
3assignvariableop_42_multi_head_attention_query_bias:@I
3assignvariableop_43_multi_head_attention_key_kernel:@@C
1assignvariableop_44_multi_head_attention_key_bias:@K
5assignvariableop_45_multi_head_attention_value_kernel:@@E
3assignvariableop_46_multi_head_attention_value_bias:@V
@assignvariableop_47_multi_head_attention_attention_output_kernel:@@L
>assignvariableop_48_multi_head_attention_attention_output_bias:@M
7assignvariableop_49_multi_head_attention_1_query_kernel:@@G
5assignvariableop_50_multi_head_attention_1_query_bias:@K
5assignvariableop_51_multi_head_attention_1_key_kernel:@@E
3assignvariableop_52_multi_head_attention_1_key_bias:@M
7assignvariableop_53_multi_head_attention_1_value_kernel:@@G
5assignvariableop_54_multi_head_attention_1_value_bias:@X
Bassignvariableop_55_multi_head_attention_1_attention_output_kernel:@@N
@assignvariableop_56_multi_head_attention_1_attention_output_bias:@#
assignvariableop_57_total: #
assignvariableop_58_count: %
assignvariableop_59_total_1: %
assignvariableop_60_count_1: C
)assignvariableop_61_adamw_conv_1_kernel_m:
5
'assignvariableop_62_adamw_conv_1_bias_m:
C
)assignvariableop_63_adamw_conv_2_kernel_m:

5
'assignvariableop_64_adamw_conv_2_bias_m:
C
)assignvariableop_65_adamw_conv_3_kernel_m:
5
'assignvariableop_66_adamw_conv_3_bias_m:C
)assignvariableop_67_adamw_conv_4_kernel_m:5
'assignvariableop_68_adamw_conv_4_bias_m:C
5assignvariableop_69_adamw_layer_normalization_gamma_m:@B
4assignvariableop_70_adamw_layer_normalization_beta_m:@E
7assignvariableop_71_adamw_layer_normalization_1_gamma_m:@D
6assignvariableop_72_adamw_layer_normalization_1_beta_m:@=
*assignvariableop_73_adamw_dense_1_kernel_m:	@�7
(assignvariableop_74_adamw_dense_1_bias_m:	�=
*assignvariableop_75_adamw_dense_2_kernel_m:	�@6
(assignvariableop_76_adamw_dense_2_bias_m:@E
7assignvariableop_77_adamw_layer_normalization_2_gamma_m:@D
6assignvariableop_78_adamw_layer_normalization_2_beta_m:@E
7assignvariableop_79_adamw_layer_normalization_3_gamma_m:@D
6assignvariableop_80_adamw_layer_normalization_3_beta_m:@=
*assignvariableop_81_adamw_dense_3_kernel_m:	@�7
(assignvariableop_82_adamw_dense_3_bias_m:	�=
*assignvariableop_83_adamw_dense_4_kernel_m:	�@6
(assignvariableop_84_adamw_dense_4_bias_m:@E
7assignvariableop_85_adamw_layer_normalization_4_gamma_m:@D
6assignvariableop_86_adamw_layer_normalization_4_beta_m:@:
'assignvariableop_87_adamw_fc_1_kernel_m:	�(23
%assignvariableop_88_adamw_fc_1_bias_m:29
'assignvariableop_89_adamw_fc_2_kernel_m:223
%assignvariableop_90_adamw_fc_2_bias_m:2A
/assignvariableop_91_adamw_output_layer_kernel_m:2;
-assignvariableop_92_adamw_output_layer_bias_m:I
6assignvariableop_93_adamw_patch_encoder_dense_kernel_m:	�@B
4assignvariableop_94_adamw_patch_encoder_dense_bias_m:@P
>assignvariableop_95_adamw_patch_encoder_embedding_embeddings_m:Q@S
=assignvariableop_96_adamw_multi_head_attention_query_kernel_m:@@M
;assignvariableop_97_adamw_multi_head_attention_query_bias_m:@Q
;assignvariableop_98_adamw_multi_head_attention_key_kernel_m:@@K
9assignvariableop_99_adamw_multi_head_attention_key_bias_m:@T
>assignvariableop_100_adamw_multi_head_attention_value_kernel_m:@@N
<assignvariableop_101_adamw_multi_head_attention_value_bias_m:@_
Iassignvariableop_102_adamw_multi_head_attention_attention_output_kernel_m:@@U
Gassignvariableop_103_adamw_multi_head_attention_attention_output_bias_m:@V
@assignvariableop_104_adamw_multi_head_attention_1_query_kernel_m:@@P
>assignvariableop_105_adamw_multi_head_attention_1_query_bias_m:@T
>assignvariableop_106_adamw_multi_head_attention_1_key_kernel_m:@@N
<assignvariableop_107_adamw_multi_head_attention_1_key_bias_m:@V
@assignvariableop_108_adamw_multi_head_attention_1_value_kernel_m:@@P
>assignvariableop_109_adamw_multi_head_attention_1_value_bias_m:@a
Kassignvariableop_110_adamw_multi_head_attention_1_attention_output_kernel_m:@@W
Iassignvariableop_111_adamw_multi_head_attention_1_attention_output_bias_m:@D
*assignvariableop_112_adamw_conv_1_kernel_v:
6
(assignvariableop_113_adamw_conv_1_bias_v:
D
*assignvariableop_114_adamw_conv_2_kernel_v:

6
(assignvariableop_115_adamw_conv_2_bias_v:
D
*assignvariableop_116_adamw_conv_3_kernel_v:
6
(assignvariableop_117_adamw_conv_3_bias_v:D
*assignvariableop_118_adamw_conv_4_kernel_v:6
(assignvariableop_119_adamw_conv_4_bias_v:D
6assignvariableop_120_adamw_layer_normalization_gamma_v:@C
5assignvariableop_121_adamw_layer_normalization_beta_v:@F
8assignvariableop_122_adamw_layer_normalization_1_gamma_v:@E
7assignvariableop_123_adamw_layer_normalization_1_beta_v:@>
+assignvariableop_124_adamw_dense_1_kernel_v:	@�8
)assignvariableop_125_adamw_dense_1_bias_v:	�>
+assignvariableop_126_adamw_dense_2_kernel_v:	�@7
)assignvariableop_127_adamw_dense_2_bias_v:@F
8assignvariableop_128_adamw_layer_normalization_2_gamma_v:@E
7assignvariableop_129_adamw_layer_normalization_2_beta_v:@F
8assignvariableop_130_adamw_layer_normalization_3_gamma_v:@E
7assignvariableop_131_adamw_layer_normalization_3_beta_v:@>
+assignvariableop_132_adamw_dense_3_kernel_v:	@�8
)assignvariableop_133_adamw_dense_3_bias_v:	�>
+assignvariableop_134_adamw_dense_4_kernel_v:	�@7
)assignvariableop_135_adamw_dense_4_bias_v:@F
8assignvariableop_136_adamw_layer_normalization_4_gamma_v:@E
7assignvariableop_137_adamw_layer_normalization_4_beta_v:@;
(assignvariableop_138_adamw_fc_1_kernel_v:	�(24
&assignvariableop_139_adamw_fc_1_bias_v:2:
(assignvariableop_140_adamw_fc_2_kernel_v:224
&assignvariableop_141_adamw_fc_2_bias_v:2B
0assignvariableop_142_adamw_output_layer_kernel_v:2<
.assignvariableop_143_adamw_output_layer_bias_v:J
7assignvariableop_144_adamw_patch_encoder_dense_kernel_v:	�@C
5assignvariableop_145_adamw_patch_encoder_dense_bias_v:@Q
?assignvariableop_146_adamw_patch_encoder_embedding_embeddings_v:Q@T
>assignvariableop_147_adamw_multi_head_attention_query_kernel_v:@@N
<assignvariableop_148_adamw_multi_head_attention_query_bias_v:@R
<assignvariableop_149_adamw_multi_head_attention_key_kernel_v:@@L
:assignvariableop_150_adamw_multi_head_attention_key_bias_v:@T
>assignvariableop_151_adamw_multi_head_attention_value_kernel_v:@@N
<assignvariableop_152_adamw_multi_head_attention_value_bias_v:@_
Iassignvariableop_153_adamw_multi_head_attention_attention_output_kernel_v:@@U
Gassignvariableop_154_adamw_multi_head_attention_attention_output_bias_v:@V
@assignvariableop_155_adamw_multi_head_attention_1_query_kernel_v:@@P
>assignvariableop_156_adamw_multi_head_attention_1_query_bias_v:@T
>assignvariableop_157_adamw_multi_head_attention_1_key_kernel_v:@@N
<assignvariableop_158_adamw_multi_head_attention_1_key_bias_v:@V
@assignvariableop_159_adamw_multi_head_attention_1_value_kernel_v:@@P
>assignvariableop_160_adamw_multi_head_attention_1_value_bias_v:@a
Kassignvariableop_161_adamw_multi_head_attention_1_attention_output_kernel_v:@@W
Iassignvariableop_162_adamw_multi_head_attention_1_attention_output_bias_v:@
identity_164��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_141�AssignVariableOp_142�AssignVariableOp_143�AssignVariableOp_144�AssignVariableOp_145�AssignVariableOp_146�AssignVariableOp_147�AssignVariableOp_148�AssignVariableOp_149�AssignVariableOp_15�AssignVariableOp_150�AssignVariableOp_151�AssignVariableOp_152�AssignVariableOp_153�AssignVariableOp_154�AssignVariableOp_155�AssignVariableOp_156�AssignVariableOp_157�AssignVariableOp_158�AssignVariableOp_159�AssignVariableOp_16�AssignVariableOp_160�AssignVariableOp_161�AssignVariableOp_162�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�[
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�Z
value�ZB�Z�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/29/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/30/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/31/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/32/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/33/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/34/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/35/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/36/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/34/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/35/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-10/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/29/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/34/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/35/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp,assignvariableop_8_layer_normalization_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp+assignvariableop_9_layer_normalization_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_layer_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_layer_normalization_2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_layer_normalization_2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp/assignvariableop_18_layer_normalization_3_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_layer_normalization_3_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_3_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_3_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_4_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_4_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp/assignvariableop_24_layer_normalization_4_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_layer_normalization_4_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_fc_1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_fc_1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOpassignvariableop_28_fc_2_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOpassignvariableop_29_fc_2_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp'assignvariableop_30_output_layer_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp%assignvariableop_31_output_layer_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOpassignvariableop_32_adamw_iterIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp assignvariableop_33_adamw_beta_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp assignvariableop_34_adamw_beta_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOpassignvariableop_35_adamw_decayIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adamw_learning_rateIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp&assignvariableop_37_adamw_weight_decayIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp.assignvariableop_38_patch_encoder_dense_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp,assignvariableop_39_patch_encoder_dense_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp6assignvariableop_40_patch_encoder_embedding_embeddingsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp5assignvariableop_41_multi_head_attention_query_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp3assignvariableop_42_multi_head_attention_query_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp3assignvariableop_43_multi_head_attention_key_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp1assignvariableop_44_multi_head_attention_key_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp5assignvariableop_45_multi_head_attention_value_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp3assignvariableop_46_multi_head_attention_value_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp@assignvariableop_47_multi_head_attention_attention_output_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp>assignvariableop_48_multi_head_attention_attention_output_biasIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp7assignvariableop_49_multi_head_attention_1_query_kernelIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp5assignvariableop_50_multi_head_attention_1_query_biasIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp5assignvariableop_51_multi_head_attention_1_key_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp3assignvariableop_52_multi_head_attention_1_key_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp7assignvariableop_53_multi_head_attention_1_value_kernelIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp5assignvariableop_54_multi_head_attention_1_value_biasIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOpBassignvariableop_55_multi_head_attention_1_attention_output_kernelIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp@assignvariableop_56_multi_head_attention_1_attention_output_biasIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOpassignvariableop_57_totalIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOpassignvariableop_58_countIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOpassignvariableop_59_total_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOpassignvariableop_60_count_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp)assignvariableop_61_adamw_conv_1_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp'assignvariableop_62_adamw_conv_1_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp)assignvariableop_63_adamw_conv_2_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp'assignvariableop_64_adamw_conv_2_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp)assignvariableop_65_adamw_conv_3_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp'assignvariableop_66_adamw_conv_3_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp)assignvariableop_67_adamw_conv_4_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp'assignvariableop_68_adamw_conv_4_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp5assignvariableop_69_adamw_layer_normalization_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOp4assignvariableop_70_adamw_layer_normalization_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adamw_layer_normalization_1_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adamw_layer_normalization_1_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adamw_dense_1_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adamw_dense_1_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adamw_dense_2_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adamw_dense_2_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adamw_layer_normalization_2_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adamw_layer_normalization_2_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adamw_layer_normalization_3_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adamw_layer_normalization_3_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adamw_dense_3_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adamw_dense_3_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83�
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adamw_dense_4_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84�
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adamw_dense_4_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85�
AssignVariableOp_85AssignVariableOp7assignvariableop_85_adamw_layer_normalization_4_gamma_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86�
AssignVariableOp_86AssignVariableOp6assignvariableop_86_adamw_layer_normalization_4_beta_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87�
AssignVariableOp_87AssignVariableOp'assignvariableop_87_adamw_fc_1_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88�
AssignVariableOp_88AssignVariableOp%assignvariableop_88_adamw_fc_1_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89�
AssignVariableOp_89AssignVariableOp'assignvariableop_89_adamw_fc_2_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90�
AssignVariableOp_90AssignVariableOp%assignvariableop_90_adamw_fc_2_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91�
AssignVariableOp_91AssignVariableOp/assignvariableop_91_adamw_output_layer_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92�
AssignVariableOp_92AssignVariableOp-assignvariableop_92_adamw_output_layer_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93�
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adamw_patch_encoder_dense_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94�
AssignVariableOp_94AssignVariableOp4assignvariableop_94_adamw_patch_encoder_dense_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95�
AssignVariableOp_95AssignVariableOp>assignvariableop_95_adamw_patch_encoder_embedding_embeddings_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96�
AssignVariableOp_96AssignVariableOp=assignvariableop_96_adamw_multi_head_attention_query_kernel_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97�
AssignVariableOp_97AssignVariableOp;assignvariableop_97_adamw_multi_head_attention_query_bias_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98�
AssignVariableOp_98AssignVariableOp;assignvariableop_98_adamw_multi_head_attention_key_kernel_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99�
AssignVariableOp_99AssignVariableOp9assignvariableop_99_adamw_multi_head_attention_key_bias_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100�
AssignVariableOp_100AssignVariableOp>assignvariableop_100_adamw_multi_head_attention_value_kernel_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101�
AssignVariableOp_101AssignVariableOp<assignvariableop_101_adamw_multi_head_attention_value_bias_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102�
AssignVariableOp_102AssignVariableOpIassignvariableop_102_adamw_multi_head_attention_attention_output_kernel_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103�
AssignVariableOp_103AssignVariableOpGassignvariableop_103_adamw_multi_head_attention_attention_output_bias_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104�
AssignVariableOp_104AssignVariableOp@assignvariableop_104_adamw_multi_head_attention_1_query_kernel_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105�
AssignVariableOp_105AssignVariableOp>assignvariableop_105_adamw_multi_head_attention_1_query_bias_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106�
AssignVariableOp_106AssignVariableOp>assignvariableop_106_adamw_multi_head_attention_1_key_kernel_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107�
AssignVariableOp_107AssignVariableOp<assignvariableop_107_adamw_multi_head_attention_1_key_bias_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108�
AssignVariableOp_108AssignVariableOp@assignvariableop_108_adamw_multi_head_attention_1_value_kernel_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109�
AssignVariableOp_109AssignVariableOp>assignvariableop_109_adamw_multi_head_attention_1_value_bias_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110�
AssignVariableOp_110AssignVariableOpKassignvariableop_110_adamw_multi_head_attention_1_attention_output_kernel_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111�
AssignVariableOp_111AssignVariableOpIassignvariableop_111_adamw_multi_head_attention_1_attention_output_bias_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112�
AssignVariableOp_112AssignVariableOp*assignvariableop_112_adamw_conv_1_kernel_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113�
AssignVariableOp_113AssignVariableOp(assignvariableop_113_adamw_conv_1_bias_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114�
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adamw_conv_2_kernel_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115�
AssignVariableOp_115AssignVariableOp(assignvariableop_115_adamw_conv_2_bias_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116�
AssignVariableOp_116AssignVariableOp*assignvariableop_116_adamw_conv_3_kernel_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117�
AssignVariableOp_117AssignVariableOp(assignvariableop_117_adamw_conv_3_bias_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118�
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adamw_conv_4_kernel_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119�
AssignVariableOp_119AssignVariableOp(assignvariableop_119_adamw_conv_4_bias_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120�
AssignVariableOp_120AssignVariableOp6assignvariableop_120_adamw_layer_normalization_gamma_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121�
AssignVariableOp_121AssignVariableOp5assignvariableop_121_adamw_layer_normalization_beta_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122�
AssignVariableOp_122AssignVariableOp8assignvariableop_122_adamw_layer_normalization_1_gamma_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123�
AssignVariableOp_123AssignVariableOp7assignvariableop_123_adamw_layer_normalization_1_beta_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124�
AssignVariableOp_124AssignVariableOp+assignvariableop_124_adamw_dense_1_kernel_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125�
AssignVariableOp_125AssignVariableOp)assignvariableop_125_adamw_dense_1_bias_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126�
AssignVariableOp_126AssignVariableOp+assignvariableop_126_adamw_dense_2_kernel_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127�
AssignVariableOp_127AssignVariableOp)assignvariableop_127_adamw_dense_2_bias_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128�
AssignVariableOp_128AssignVariableOp8assignvariableop_128_adamw_layer_normalization_2_gamma_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129�
AssignVariableOp_129AssignVariableOp7assignvariableop_129_adamw_layer_normalization_2_beta_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130�
AssignVariableOp_130AssignVariableOp8assignvariableop_130_adamw_layer_normalization_3_gamma_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131�
AssignVariableOp_131AssignVariableOp7assignvariableop_131_adamw_layer_normalization_3_beta_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132�
AssignVariableOp_132AssignVariableOp+assignvariableop_132_adamw_dense_3_kernel_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133�
AssignVariableOp_133AssignVariableOp)assignvariableop_133_adamw_dense_3_bias_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134�
AssignVariableOp_134AssignVariableOp+assignvariableop_134_adamw_dense_4_kernel_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135�
AssignVariableOp_135AssignVariableOp)assignvariableop_135_adamw_dense_4_bias_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136�
AssignVariableOp_136AssignVariableOp8assignvariableop_136_adamw_layer_normalization_4_gamma_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137�
AssignVariableOp_137AssignVariableOp7assignvariableop_137_adamw_layer_normalization_4_beta_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138�
AssignVariableOp_138AssignVariableOp(assignvariableop_138_adamw_fc_1_kernel_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139�
AssignVariableOp_139AssignVariableOp&assignvariableop_139_adamw_fc_1_bias_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140�
AssignVariableOp_140AssignVariableOp(assignvariableop_140_adamw_fc_2_kernel_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141�
AssignVariableOp_141AssignVariableOp&assignvariableop_141_adamw_fc_2_bias_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142�
AssignVariableOp_142AssignVariableOp0assignvariableop_142_adamw_output_layer_kernel_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143�
AssignVariableOp_143AssignVariableOp.assignvariableop_143_adamw_output_layer_bias_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144�
AssignVariableOp_144AssignVariableOp7assignvariableop_144_adamw_patch_encoder_dense_kernel_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145�
AssignVariableOp_145AssignVariableOp5assignvariableop_145_adamw_patch_encoder_dense_bias_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146�
AssignVariableOp_146AssignVariableOp?assignvariableop_146_adamw_patch_encoder_embedding_embeddings_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_146q
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:2
Identity_147�
AssignVariableOp_147AssignVariableOp>assignvariableop_147_adamw_multi_head_attention_query_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_147q
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:2
Identity_148�
AssignVariableOp_148AssignVariableOp<assignvariableop_148_adamw_multi_head_attention_query_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_148q
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:2
Identity_149�
AssignVariableOp_149AssignVariableOp<assignvariableop_149_adamw_multi_head_attention_key_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149q
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:2
Identity_150�
AssignVariableOp_150AssignVariableOp:assignvariableop_150_adamw_multi_head_attention_key_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_150q
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:2
Identity_151�
AssignVariableOp_151AssignVariableOp>assignvariableop_151_adamw_multi_head_attention_value_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_151q
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:2
Identity_152�
AssignVariableOp_152AssignVariableOp<assignvariableop_152_adamw_multi_head_attention_value_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_152q
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:2
Identity_153�
AssignVariableOp_153AssignVariableOpIassignvariableop_153_adamw_multi_head_attention_attention_output_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_153q
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:2
Identity_154�
AssignVariableOp_154AssignVariableOpGassignvariableop_154_adamw_multi_head_attention_attention_output_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_154q
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:2
Identity_155�
AssignVariableOp_155AssignVariableOp@assignvariableop_155_adamw_multi_head_attention_1_query_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_155q
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:2
Identity_156�
AssignVariableOp_156AssignVariableOp>assignvariableop_156_adamw_multi_head_attention_1_query_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_156q
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:2
Identity_157�
AssignVariableOp_157AssignVariableOp>assignvariableop_157_adamw_multi_head_attention_1_key_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_157q
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:2
Identity_158�
AssignVariableOp_158AssignVariableOp<assignvariableop_158_adamw_multi_head_attention_1_key_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_158q
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:2
Identity_159�
AssignVariableOp_159AssignVariableOp@assignvariableop_159_adamw_multi_head_attention_1_value_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159q
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:2
Identity_160�
AssignVariableOp_160AssignVariableOp>assignvariableop_160_adamw_multi_head_attention_1_value_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_160q
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:2
Identity_161�
AssignVariableOp_161AssignVariableOpKassignvariableop_161_adamw_multi_head_attention_1_attention_output_kernel_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_161q
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:2
Identity_162�
AssignVariableOp_162AssignVariableOpIassignvariableop_162_adamw_multi_head_attention_1_attention_output_bias_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_163Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_163�
Identity_164IdentityIdentity_163:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_164"%
identity_164Identity_164:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622*
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
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
input_layer>
serving_default_input_layer:0�����������@
output_layer0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�{
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer-15
layer_with_weights-10
layer-16
layer_with_weights-11
layer-17
layer-18
layer_with_weights-12
layer-19
layer_with_weights-13
layer-20
layer_with_weights-14
layer-21
layer-22
layer_with_weights-15
layer-23
layer-24
layer_with_weights-16
layer-25
layer-26
layer_with_weights-17
layer-27
layer-28
layer_with_weights-18
layer-29
	optimizer
 regularization_losses
!trainable_variables
"	variables
#	keras_api
$
signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�r
_tf_keras_network�r{"name": "WheatClassifier_CNN-VIT_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "WheatClassifier_CNN-VIT_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["maxpool_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_2", "inbound_nodes": [[["conv_4", 0, 0, {}]]]}, {"class_name": "Patches", "config": {"layer was saved without config": true}, "name": "patches", "inbound_nodes": [[["maxpool_2", 0, 0, {}]]]}, {"class_name": "PatchEncoder", "config": {"layer was saved without config": true}, "name": "patch_encoder", "inbound_nodes": [[["patches", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization", "inbound_nodes": [[["patch_encoder", 0, 0, {}]]]}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}, "name": "multi_head_attention", "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_2", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}, "name": "multi_head_attention_1", "inbound_nodes": [[["layer_normalization_2", 0, 0, {"value": ["layer_normalization_2", 0, 0]}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["multi_head_attention_1", 0, 0, {}], ["add_1", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_3", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["layer_normalization_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["dense_4", 0, 0, {}], ["add_2", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_4", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_layer", "inbound_nodes": [[["layer_normalization_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1", "inbound_nodes": [[["flatten_layer", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_1", "inbound_nodes": [[["FC_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_2", "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_2", "inbound_nodes": [[["FC_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "shared_object_id": 64, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 200, 200, 3]}, "float32", "input_layer"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 66}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Addons>AdamW", "config": {"name": "AdamW", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false, "weight_decay": 9.999999747378752e-05}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
�

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_layer", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 3]}}
�


+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 198, 198, 10]}}
�
1regularization_losses
2	variables
3trainable_variables
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "maxpool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 69}}
�


5kernel
6bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["maxpool_1", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 98, 98, 10]}}
�


;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 96, 96, 16]}}
�
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "maxpool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_4", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 72}}
�
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "patches", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Patches", "config": {"layer was saved without config": true}}
�
I
projection
Jposition_embedding
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "patch_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PatchEncoder", "config": {"layer was saved without config": true}}
�
Oaxis
	Pgamma
Qbeta
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["patch_encoder", 0, 0, {}]]], "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�

V_query_dense
W
_key_dense
X_value_dense
Y_softmax
Z_dropout_layer
[_output_dense
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "multi_head_attention", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}, "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]], "shared_object_id": 20}
�
`regularization_losses
a	variables
btrainable_variables
c	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]], "shared_object_id": 21, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 81, 64]}, {"class_name": "TensorShape", "items": [null, 81, 64]}]}
�
daxis
	egamma
fbeta
gregularization_losses
h	variables
itrainable_variables
j	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 23}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add", 0, 0, {}]]], "shared_object_id": 24, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�	

kkernel
lbias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�	

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 74}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 128]}}
�
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]], "shared_object_id": 31, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 81, 64]}, {"class_name": "TensorShape", "items": [null, 81, 64]}]}
�
{axis
	|gamma
}beta
~regularization_losses
	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 33}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_1", 0, 0, {}]]], "shared_object_id": 34, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�
�_query_dense
�
_key_dense
�_value_dense
�_softmax
�_dropout_layer
�_output_dense
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�{"name": "multi_head_attention_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}, "inbound_nodes": [[["layer_normalization_2", 0, 0, {"value": ["layer_normalization_2", 0, 0]}]]], "shared_object_id": 37}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["multi_head_attention_1", 0, 0, {}], ["add_1", 0, 0, {}]]], "shared_object_id": 38, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 81, 64]}, {"class_name": "TensorShape", "items": [null, 81, 64]}]}
�
	�axis

�gamma
	�beta
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "layer_normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 40}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_2", 0, 0, {}]]], "shared_object_id": 41, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 43}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["layer_normalization_3", 0, 0, {}]]], "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_3", 0, 0, {}]]], "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 128]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["dense_4", 0, 0, {}], ["add_2", 0, 0, {}]]], "shared_object_id": 48, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 81, 64]}, {"class_name": "TensorShape", "items": [null, 81, 64]}]}
�
	�axis

�gamma
	�beta
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "layer_normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 50}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_3", 0, 0, {}]]], "shared_object_id": 51, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["layer_normalization_4", 0, 0, {}]]], "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 77}}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "FC_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_layer", 0, 0, {}]]], "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5184}}, "shared_object_id": 78}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5184]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_ReLu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_1", 0, 0, {}]]], "shared_object_id": 56}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "FC_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]], "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 79}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_ReLu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_2", 0, 0, {}]]], "shared_object_id": 60}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]], "shared_object_id": 63, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
�	
	�iter
�beta_1
�beta_2

�decay
�learning_rate
�weight_decay%m�&m�+m�,m�5m�6m�;m�<m�Pm�Qm�em�fm�km�lm�qm�rm�|m�}m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�%v�&v�+v�,v�5v�6v�;v�<v�Pv�Qv�ev�fv�kv�lv�qv�rv�|v�}v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
 "
trackable_list_wrapper
�
%0
&1
+2
,3
54
65
;6
<7
�8
�9
�10
P11
Q12
�13
�14
�15
�16
�17
�18
�19
�20
e21
f22
k23
l24
q25
r26
|27
}28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50"
trackable_list_wrapper
�
%0
&1
+2
,3
54
65
;6
<7
�8
�9
�10
P11
Q12
�13
�14
�15
�16
�17
�18
�19
�20
e21
f22
k23
l24
q25
r26
|27
}28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50"
trackable_list_wrapper
�
�non_trainable_variables
�layers
 regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
!trainable_variables
"	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
':%
2conv_1/kernel
:
2conv_1/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
'regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
(	variables
)trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%

2conv_2/kernel
:
2conv_2/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
-regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
.	variables
/trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
1regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
2	variables
3trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%
2conv_3/kernel
:2conv_3/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
�
�non_trainable_variables
�layers
7regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
8	variables
9trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%2conv_4/kernel
:2conv_4/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
=regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
>	variables
?trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
Aregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
B	variables
Ctrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
Eregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
F	variables
Gtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 81}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 82}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 83, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}, "shared_object_id": 84}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 400]}}
�
�
embeddings
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 81, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 85}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "shared_object_id": 86, "build_input_shape": {"class_name": "TensorShape", "items": [81]}}
 "
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
�
�non_trainable_variables
�layers
Kregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
L	variables
Mtrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@2layer_normalization/gamma
&:$@2layer_normalization/beta
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
Rregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
S	variables
Ttrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 87, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 88, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 89, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "shared_object_id": 90}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 91}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 64], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 92, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 4, 64]}}
 "
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
�
�non_trainable_variables
�layers
\regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
]	variables
^trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
`regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
a	variables
btrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_1/gamma
(:&@2layer_normalization_1/beta
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
gregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
h	variables
itrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	@�2dense_1/kernel
:�2dense_1/bias
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
mregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
n	variables
otrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�@2dense_2/kernel
:@2dense_2/bias
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
sregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
t	variables
utrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
wregularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
x	variables
ytrainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_2/gamma
(:&@2layer_normalization_2/beta
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
~regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 93, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 94, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 95, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "shared_object_id": 96}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 97}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 64], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 98, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 4, 64]}}
 "
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_3/gamma
(:&@2layer_normalization_3/beta
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	@�2dense_3/kernel
:�2dense_3/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�@2dense_4/kernel
:@2dense_4/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_4/gamma
(:&@2layer_normalization_4/beta
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�(22FC_1/kernel
:22	FC_1/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:222FC_2/kernel
:22	FC_2/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
%:#22output_layer/kernel
:2output_layer/bias
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2
AdamW/iter
: (2AdamW/beta_1
: (2AdamW/beta_2
: (2AdamW/decay
: (2AdamW/learning_rate
: (2AdamW/weight_decay
-:+	�@2patch_encoder/dense/kernel
&:$@2patch_encoder/dense/bias
4:2Q@2"patch_encoder/embedding/embeddings
7:5@@2!multi_head_attention/query/kernel
1:/@2multi_head_attention/query/bias
5:3@@2multi_head_attention/key/kernel
/:-@2multi_head_attention/key/bias
7:5@@2!multi_head_attention/value/kernel
1:/@2multi_head_attention/value/bias
B:@@@2,multi_head_attention/attention_output/kernel
8:6@2*multi_head_attention/attention_output/bias
9:7@@2#multi_head_attention_1/query/kernel
3:1@2!multi_head_attention_1/query/bias
7:5@@2!multi_head_attention_1/key/kernel
1:/@2multi_head_attention_1/key/bias
9:7@@2#multi_head_attention_1/value/kernel
3:1@2!multi_head_attention_1/value/bias
D:B@@2.multi_head_attention_1/attention_output/kernel
::8@2,multi_head_attention_1/attention_output/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
�0
�1"
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
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
I0
J1"
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
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
V0
W1
X2
Y3
Z4
[5"
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
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�regularization_losses
�layer_metrics
 �layer_regularization_losses
�metrics
�	variables
�trainable_variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
P
�0
�1
�2
�3
�4
�5"
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
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 99}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 66}
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
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
-:+
2AdamW/conv_1/kernel/m
:
2AdamW/conv_1/bias/m
-:+

2AdamW/conv_2/kernel/m
:
2AdamW/conv_2/bias/m
-:+
2AdamW/conv_3/kernel/m
:2AdamW/conv_3/bias/m
-:+2AdamW/conv_4/kernel/m
:2AdamW/conv_4/bias/m
-:+@2!AdamW/layer_normalization/gamma/m
,:*@2 AdamW/layer_normalization/beta/m
/:-@2#AdamW/layer_normalization_1/gamma/m
.:,@2"AdamW/layer_normalization_1/beta/m
':%	@�2AdamW/dense_1/kernel/m
!:�2AdamW/dense_1/bias/m
':%	�@2AdamW/dense_2/kernel/m
 :@2AdamW/dense_2/bias/m
/:-@2#AdamW/layer_normalization_2/gamma/m
.:,@2"AdamW/layer_normalization_2/beta/m
/:-@2#AdamW/layer_normalization_3/gamma/m
.:,@2"AdamW/layer_normalization_3/beta/m
':%	@�2AdamW/dense_3/kernel/m
!:�2AdamW/dense_3/bias/m
':%	�@2AdamW/dense_4/kernel/m
 :@2AdamW/dense_4/bias/m
/:-@2#AdamW/layer_normalization_4/gamma/m
.:,@2"AdamW/layer_normalization_4/beta/m
$:"	�(22AdamW/FC_1/kernel/m
:22AdamW/FC_1/bias/m
#:!222AdamW/FC_2/kernel/m
:22AdamW/FC_2/bias/m
+:)22AdamW/output_layer/kernel/m
%:#2AdamW/output_layer/bias/m
3:1	�@2"AdamW/patch_encoder/dense/kernel/m
,:*@2 AdamW/patch_encoder/dense/bias/m
::8Q@2*AdamW/patch_encoder/embedding/embeddings/m
=:;@@2)AdamW/multi_head_attention/query/kernel/m
7:5@2'AdamW/multi_head_attention/query/bias/m
;:9@@2'AdamW/multi_head_attention/key/kernel/m
5:3@2%AdamW/multi_head_attention/key/bias/m
=:;@@2)AdamW/multi_head_attention/value/kernel/m
7:5@2'AdamW/multi_head_attention/value/bias/m
H:F@@24AdamW/multi_head_attention/attention_output/kernel/m
>:<@22AdamW/multi_head_attention/attention_output/bias/m
?:=@@2+AdamW/multi_head_attention_1/query/kernel/m
9:7@2)AdamW/multi_head_attention_1/query/bias/m
=:;@@2)AdamW/multi_head_attention_1/key/kernel/m
7:5@2'AdamW/multi_head_attention_1/key/bias/m
?:=@@2+AdamW/multi_head_attention_1/value/kernel/m
9:7@2)AdamW/multi_head_attention_1/value/bias/m
J:H@@26AdamW/multi_head_attention_1/attention_output/kernel/m
@:>@24AdamW/multi_head_attention_1/attention_output/bias/m
-:+
2AdamW/conv_1/kernel/v
:
2AdamW/conv_1/bias/v
-:+

2AdamW/conv_2/kernel/v
:
2AdamW/conv_2/bias/v
-:+
2AdamW/conv_3/kernel/v
:2AdamW/conv_3/bias/v
-:+2AdamW/conv_4/kernel/v
:2AdamW/conv_4/bias/v
-:+@2!AdamW/layer_normalization/gamma/v
,:*@2 AdamW/layer_normalization/beta/v
/:-@2#AdamW/layer_normalization_1/gamma/v
.:,@2"AdamW/layer_normalization_1/beta/v
':%	@�2AdamW/dense_1/kernel/v
!:�2AdamW/dense_1/bias/v
':%	�@2AdamW/dense_2/kernel/v
 :@2AdamW/dense_2/bias/v
/:-@2#AdamW/layer_normalization_2/gamma/v
.:,@2"AdamW/layer_normalization_2/beta/v
/:-@2#AdamW/layer_normalization_3/gamma/v
.:,@2"AdamW/layer_normalization_3/beta/v
':%	@�2AdamW/dense_3/kernel/v
!:�2AdamW/dense_3/bias/v
':%	�@2AdamW/dense_4/kernel/v
 :@2AdamW/dense_4/bias/v
/:-@2#AdamW/layer_normalization_4/gamma/v
.:,@2"AdamW/layer_normalization_4/beta/v
$:"	�(22AdamW/FC_1/kernel/v
:22AdamW/FC_1/bias/v
#:!222AdamW/FC_2/kernel/v
:22AdamW/FC_2/bias/v
+:)22AdamW/output_layer/kernel/v
%:#2AdamW/output_layer/bias/v
3:1	�@2"AdamW/patch_encoder/dense/kernel/v
,:*@2 AdamW/patch_encoder/dense/bias/v
::8Q@2*AdamW/patch_encoder/embedding/embeddings/v
=:;@@2)AdamW/multi_head_attention/query/kernel/v
7:5@2'AdamW/multi_head_attention/query/bias/v
;:9@@2'AdamW/multi_head_attention/key/kernel/v
5:3@2%AdamW/multi_head_attention/key/bias/v
=:;@@2)AdamW/multi_head_attention/value/kernel/v
7:5@2'AdamW/multi_head_attention/value/bias/v
H:F@@24AdamW/multi_head_attention/attention_output/kernel/v
>:<@22AdamW/multi_head_attention/attention_output/bias/v
?:=@@2+AdamW/multi_head_attention_1/query/kernel/v
9:7@2)AdamW/multi_head_attention_1/query/bias/v
=:;@@2)AdamW/multi_head_attention_1/key/kernel/v
7:5@2'AdamW/multi_head_attention_1/key/bias/v
?:=@@2+AdamW/multi_head_attention_1/value/kernel/v
9:7@2)AdamW/multi_head_attention_1/value/bias/v
J:H@@26AdamW/multi_head_attention_1/attention_output/kernel/v
@:>@24AdamW/multi_head_attention_1/attention_output/bias/v
�2�
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_430285
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_430688
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_429644
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_429779�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
!__inference__wrapped_model_427863�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *4�1
/�,
input_layer�����������
�2�
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_428655
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_430795
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_430902
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_429509�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_conv_1_layer_call_and_return_conditional_losses_430912�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv_1_layer_call_fn_430921�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv_2_layer_call_and_return_conditional_losses_430931�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv_2_layer_call_fn_430940�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_maxpool_1_layer_call_and_return_conditional_losses_427869�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
*__inference_maxpool_1_layer_call_fn_427875�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
B__inference_conv_3_layer_call_and_return_conditional_losses_430950�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv_3_layer_call_fn_430959�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_conv_4_layer_call_and_return_conditional_losses_430969�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_conv_4_layer_call_fn_430978�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_maxpool_2_layer_call_and_return_conditional_losses_427881�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
*__inference_maxpool_2_layer_call_fn_427887�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
C__inference_patches_layer_call_and_return_conditional_losses_430992�
���
FullArgSpec
args�
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_patches_layer_call_fn_430997�
���
FullArgSpec
args�
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_patch_encoder_layer_call_and_return_conditional_losses_431037�
���
FullArgSpec
args�
jself
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_patch_encoder_layer_call_fn_431048�
���
FullArgSpec
args�
jself
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_431070�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_layer_normalization_layer_call_fn_431079�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_431114
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_431156�
���
FullArgSpece
args]�Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
5__inference_multi_head_attention_layer_call_fn_431178
5__inference_multi_head_attention_layer_call_fn_431200�
���
FullArgSpece
args]�Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
?__inference_add_layer_call_and_return_conditional_losses_431206�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
$__inference_add_layer_call_fn_431212�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_431234�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
6__inference_layer_normalization_1_layer_call_fn_431243�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_431281�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_431290�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_431328�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_2_layer_call_fn_431337�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_add_1_layer_call_and_return_conditional_losses_431343�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_add_1_layer_call_fn_431349�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_431371�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
6__inference_layer_normalization_2_layer_call_fn_431380�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_431415
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_431457�
���
FullArgSpece
args]�Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
7__inference_multi_head_attention_1_layer_call_fn_431479
7__inference_multi_head_attention_1_layer_call_fn_431501�
���
FullArgSpece
args]�Z
jself
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
varargs
 
varkw
 
defaults�

 

 
p 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_add_2_layer_call_and_return_conditional_losses_431507�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_add_2_layer_call_fn_431513�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_431535�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
6__inference_layer_normalization_3_layer_call_fn_431544�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_3_layer_call_and_return_conditional_losses_431582�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_3_layer_call_fn_431591�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_4_layer_call_and_return_conditional_losses_431629�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_4_layer_call_fn_431638�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_add_3_layer_call_and_return_conditional_losses_431644�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_add_3_layer_call_fn_431650�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_431672�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
6__inference_layer_normalization_4_layer_call_fn_431681�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
I__inference_flatten_layer_layer_call_and_return_conditional_losses_431687�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
.__inference_flatten_layer_layer_call_fn_431692�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_FC_1_layer_call_and_return_conditional_losses_431702�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_FC_1_layer_call_fn_431711�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_431716�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_leaky_ReLu_1_layer_call_fn_431721�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_FC_2_layer_call_and_return_conditional_losses_431731�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_FC_2_layer_call_fn_431740�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_431745�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_leaky_ReLu_2_layer_call_fn_431750�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_output_layer_layer_call_and_return_conditional_losses_431761�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_output_layer_layer_call_fn_431770�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_signature_wrapper_429896input_layer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec%
args�
jself
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
@__inference_FC_1_layer_call_and_return_conditional_losses_431702_��0�-
&�#
!�
inputs����������(
� "%�"
�
0���������2
� {
%__inference_FC_1_layer_call_fn_431711R��0�-
&�#
!�
inputs����������(
� "����������2�
@__inference_FC_2_layer_call_and_return_conditional_losses_431731^��/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� z
%__inference_FC_2_layer_call_fn_431740Q��/�,
%�"
 �
inputs���������2
� "����������2�
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_429644�T%&+,56;<���PQ��������efklqr|}����������������������F�C
<�9
/�,
input_layer�����������
p 

 
� "%�"
�
0���������
� �
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_429779�T%&+,56;<���PQ��������efklqr|}����������������������F�C
<�9
/�,
input_layer�����������
p

 
� "%�"
�
0���������
� �
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_430285�T%&+,56;<���PQ��������efklqr|}����������������������A�>
7�4
*�'
inputs�����������
p 

 
� "%�"
�
0���������
� �
U__inference_WheatClassifier_CNN-VIT_6_layer_call_and_return_conditional_losses_430688�T%&+,56;<���PQ��������efklqr|}����������������������A�>
7�4
*�'
inputs�����������
p

 
� "%�"
�
0���������
� �
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_428655�T%&+,56;<���PQ��������efklqr|}����������������������F�C
<�9
/�,
input_layer�����������
p 

 
� "�����������
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_429509�T%&+,56;<���PQ��������efklqr|}����������������������F�C
<�9
/�,
input_layer�����������
p

 
� "�����������
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_430795�T%&+,56;<���PQ��������efklqr|}����������������������A�>
7�4
*�'
inputs�����������
p 

 
� "�����������
:__inference_WheatClassifier_CNN-VIT_6_layer_call_fn_430902�T%&+,56;<���PQ��������efklqr|}����������������������A�>
7�4
*�'
inputs�����������
p

 
� "�����������
!__inference__wrapped_model_427863�T%&+,56;<���PQ��������efklqr|}����������������������>�;
4�1
/�,
input_layer�����������
� ";�8
6
output_layer&�#
output_layer����������
A__inference_add_1_layer_call_and_return_conditional_losses_431343�b�_
X�U
S�P
&�#
inputs/0���������Q@
&�#
inputs/1���������Q@
� ")�&
�
0���������Q@
� �
&__inference_add_1_layer_call_fn_431349�b�_
X�U
S�P
&�#
inputs/0���������Q@
&�#
inputs/1���������Q@
� "����������Q@�
A__inference_add_2_layer_call_and_return_conditional_losses_431507�b�_
X�U
S�P
&�#
inputs/0���������Q@
&�#
inputs/1���������Q@
� ")�&
�
0���������Q@
� �
&__inference_add_2_layer_call_fn_431513�b�_
X�U
S�P
&�#
inputs/0���������Q@
&�#
inputs/1���������Q@
� "����������Q@�
A__inference_add_3_layer_call_and_return_conditional_losses_431644�b�_
X�U
S�P
&�#
inputs/0���������Q@
&�#
inputs/1���������Q@
� ")�&
�
0���������Q@
� �
&__inference_add_3_layer_call_fn_431650�b�_
X�U
S�P
&�#
inputs/0���������Q@
&�#
inputs/1���������Q@
� "����������Q@�
?__inference_add_layer_call_and_return_conditional_losses_431206�b�_
X�U
S�P
&�#
inputs/0���������Q@
&�#
inputs/1���������Q@
� ")�&
�
0���������Q@
� �
$__inference_add_layer_call_fn_431212�b�_
X�U
S�P
&�#
inputs/0���������Q@
&�#
inputs/1���������Q@
� "����������Q@�
B__inference_conv_1_layer_call_and_return_conditional_losses_430912p%&9�6
/�,
*�'
inputs�����������
� "/�,
%�"
0�����������

� �
'__inference_conv_1_layer_call_fn_430921c%&9�6
/�,
*�'
inputs�����������
� ""������������
�
B__inference_conv_2_layer_call_and_return_conditional_losses_430931p+,9�6
/�,
*�'
inputs�����������

� "/�,
%�"
0�����������

� �
'__inference_conv_2_layer_call_fn_430940c+,9�6
/�,
*�'
inputs�����������

� ""������������
�
B__inference_conv_3_layer_call_and_return_conditional_losses_430950l567�4
-�*
(�%
inputs���������bb

� "-�*
#� 
0���������``
� �
'__inference_conv_3_layer_call_fn_430959_567�4
-�*
(�%
inputs���������bb

� " ����������``�
B__inference_conv_4_layer_call_and_return_conditional_losses_430969l;<7�4
-�*
(�%
inputs���������``
� "-�*
#� 
0���������^^
� �
'__inference_conv_4_layer_call_fn_430978_;<7�4
-�*
(�%
inputs���������``
� " ����������^^�
C__inference_dense_1_layer_call_and_return_conditional_losses_431281ekl3�0
)�&
$�!
inputs���������Q@
� "*�'
 �
0���������Q�
� �
(__inference_dense_1_layer_call_fn_431290Xkl3�0
)�&
$�!
inputs���������Q@
� "����������Q��
C__inference_dense_2_layer_call_and_return_conditional_losses_431328eqr4�1
*�'
%�"
inputs���������Q�
� ")�&
�
0���������Q@
� �
(__inference_dense_2_layer_call_fn_431337Xqr4�1
*�'
%�"
inputs���������Q�
� "����������Q@�
C__inference_dense_3_layer_call_and_return_conditional_losses_431582g��3�0
)�&
$�!
inputs���������Q@
� "*�'
 �
0���������Q�
� �
(__inference_dense_3_layer_call_fn_431591Z��3�0
)�&
$�!
inputs���������Q@
� "����������Q��
C__inference_dense_4_layer_call_and_return_conditional_losses_431629g��4�1
*�'
%�"
inputs���������Q�
� ")�&
�
0���������Q@
� �
(__inference_dense_4_layer_call_fn_431638Z��4�1
*�'
%�"
inputs���������Q�
� "����������Q@�
I__inference_flatten_layer_layer_call_and_return_conditional_losses_431687]3�0
)�&
$�!
inputs���������Q@
� "&�#
�
0����������(
� �
.__inference_flatten_layer_layer_call_fn_431692P3�0
)�&
$�!
inputs���������Q@
� "�����������(�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_431234def3�0
)�&
$�!
inputs���������Q@
� ")�&
�
0���������Q@
� �
6__inference_layer_normalization_1_layer_call_fn_431243Wef3�0
)�&
$�!
inputs���������Q@
� "����������Q@�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_431371d|}3�0
)�&
$�!
inputs���������Q@
� ")�&
�
0���������Q@
� �
6__inference_layer_normalization_2_layer_call_fn_431380W|}3�0
)�&
$�!
inputs���������Q@
� "����������Q@�
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_431535f��3�0
)�&
$�!
inputs���������Q@
� ")�&
�
0���������Q@
� �
6__inference_layer_normalization_3_layer_call_fn_431544Y��3�0
)�&
$�!
inputs���������Q@
� "����������Q@�
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_431672f��3�0
)�&
$�!
inputs���������Q@
� ")�&
�
0���������Q@
� �
6__inference_layer_normalization_4_layer_call_fn_431681Y��3�0
)�&
$�!
inputs���������Q@
� "����������Q@�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_431070dPQ3�0
)�&
$�!
inputs���������Q@
� ")�&
�
0���������Q@
� �
4__inference_layer_normalization_layer_call_fn_431079WPQ3�0
)�&
$�!
inputs���������Q@
� "����������Q@�
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_431716X/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� |
-__inference_leaky_ReLu_1_layer_call_fn_431721K/�,
%�"
 �
inputs���������2
� "����������2�
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_431745X/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� |
-__inference_leaky_ReLu_2_layer_call_fn_431750K/�,
%�"
 �
inputs���������2
� "����������2�
E__inference_maxpool_1_layer_call_and_return_conditional_losses_427869�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_maxpool_1_layer_call_fn_427875�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
E__inference_maxpool_2_layer_call_and_return_conditional_losses_427881�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_maxpool_2_layer_call_fn_427887�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_431415���������g�d
]�Z
#� 
query���������Q@
#� 
value���������Q@

 

 
p 
p 
� ")�&
�
0���������Q@
� �
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_431457���������g�d
]�Z
#� 
query���������Q@
#� 
value���������Q@

 

 
p 
p
� ")�&
�
0���������Q@
� �
7__inference_multi_head_attention_1_layer_call_fn_431479���������g�d
]�Z
#� 
query���������Q@
#� 
value���������Q@

 

 
p 
p 
� "����������Q@�
7__inference_multi_head_attention_1_layer_call_fn_431501���������g�d
]�Z
#� 
query���������Q@
#� 
value���������Q@

 

 
p 
p
� "����������Q@�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_431114���������g�d
]�Z
#� 
query���������Q@
#� 
value���������Q@

 

 
p 
p 
� ")�&
�
0���������Q@
� �
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_431156���������g�d
]�Z
#� 
query���������Q@
#� 
value���������Q@

 

 
p 
p
� ")�&
�
0���������Q@
� �
5__inference_multi_head_attention_layer_call_fn_431178���������g�d
]�Z
#� 
query���������Q@
#� 
value���������Q@

 

 
p 
p 
� "����������Q@�
5__inference_multi_head_attention_layer_call_fn_431200���������g�d
]�Z
#� 
query���������Q@
#� 
value���������Q@

 

 
p 
p
� "����������Q@�
H__inference_output_layer_layer_call_and_return_conditional_losses_431761^��/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� �
-__inference_output_layer_layer_call_fn_431770Q��/�,
%�"
 �
inputs���������2
� "�����������
I__inference_patch_encoder_layer_call_and_return_conditional_losses_431037q���<�9
2�/
-�*
patch�������������������
� ")�&
�
0���������Q@
� �
.__inference_patch_encoder_layer_call_fn_431048d���<�9
2�/
-�*
patch�������������������
� "����������Q@�
C__inference_patches_layer_call_and_return_conditional_losses_430992n7�4
-�*
(�%
images���������//
� "3�0
)�&
0�������������������
� �
(__inference_patches_layer_call_fn_430997a7�4
-�*
(�%
images���������//
� "&�#��������������������
$__inference_signature_wrapper_429896�T%&+,56;<���PQ��������efklqr|}����������������������M�J
� 
C�@
>
input_layer/�,
input_layer�����������";�8
6
output_layer&�#
output_layer���������