��#
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
 �"serve*2.5.02v2.5.0-0-ga4dfb8d1a718��
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
~
conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@
*
shared_nameconv_1/kernel
w
!conv_1/kernel/Read/ReadVariableOpReadVariableOpconv_1/kernel*&
_output_shapes
:@
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
s
FC_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*
shared_nameFC_1/kernel
l
FC_1/kernel/Read/ReadVariableOpReadVariableOpFC_1/kernel*
_output_shapes
:	�2*
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
shape:	�@*+
shared_namepatch_encoder/dense/kernel
�
.patch_encoder/dense/kernel/Read/ReadVariableOpReadVariableOppatch_encoder/dense/kernel*
_output_shapes
:	�@*
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
dtype0*
shape:	�@*3
shared_name$"patch_encoder/embedding/embeddings
�
6patch_encoder/embedding/embeddings/Read/ReadVariableOpReadVariableOp"patch_encoder/embedding/embeddings*
_output_shapes
:	�@*
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
AdamW/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@
*&
shared_nameAdamW/conv_1/kernel/m
�
)AdamW/conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/conv_1/kernel/m*&
_output_shapes
:@
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
AdamW/FC_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*$
shared_nameAdamW/FC_1/kernel/m
|
'AdamW/FC_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/m*
_output_shapes
:	�2*
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
shape:	�@*3
shared_name$"AdamW/patch_encoder/dense/kernel/m
�
6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/m*
_output_shapes
:	�@*
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
dtype0*
shape:	�@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/m
�
>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/m*
_output_shapes
:	�@*
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
AdamW/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@
*&
shared_nameAdamW/conv_1/kernel/v
�
)AdamW/conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/conv_1/kernel/v*&
_output_shapes
:@
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
AdamW/FC_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�2*$
shared_nameAdamW/FC_1/kernel/v
|
'AdamW/FC_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/v*
_output_shapes
:	�2*
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
shape:	�@*3
shared_name$"AdamW/patch_encoder/dense/kernel/v
�
6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/v*
_output_shapes
:	�@*
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
dtype0*
shape:	�@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/v
�
>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/v*
_output_shapes
:	�@*
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

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value�B� B�
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
z
 
projection
!position_embedding
"regularization_losses
#	variables
$trainable_variables
%	keras_api
q
&axis
	'gamma
(beta
)regularization_losses
*	variables
+trainable_variables
,	keras_api
�
-_query_dense
.
_key_dense
/_value_dense
0_softmax
1_dropout_layer
2_output_dense
3regularization_losses
4	variables
5trainable_variables
6	keras_api
R
7regularization_losses
8	variables
9trainable_variables
:	keras_api
q
;axis
	<gamma
=beta
>regularization_losses
?	variables
@trainable_variables
A	keras_api
h

Bkernel
Cbias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
h

Hkernel
Ibias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
q
Raxis
	Sgamma
Tbeta
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
R
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
h

]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
h

ckernel
dbias
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
R
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
R
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
i

{kernel
|bias
}regularization_losses
~	variables
trainable_variables
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate
�weight_decay'm�(m�<m�=m�Bm�Cm�Hm�Im�Sm�Tm�]m�^m�cm�dm�qm�rm�{m�|m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�'v�(v�<v�=v�Bv�Cv�Hv�Iv�Sv�Tv�]v�^v�cv�dv�qv�rv�{v�|v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�
 
�
�0
�1
�2
'3
(4
�5
�6
�7
�8
�9
�10
�11
�12
<13
=14
B15
C16
H17
I18
S19
T20
]21
^22
c23
d24
q25
r26
{27
|28
�29
�30
�
�0
�1
�2
'3
(4
�5
�6
�7
�8
�9
�10
�11
�12
<13
=14
B15
C16
H17
I18
S19
T20
]21
^22
c23
d24
q25
r26
{27
|28
�29
�30
�
�metrics
regularization_losses
	variables
�non_trainable_variables
�layer_metrics
�layers
trainable_variables
 �layer_regularization_losses
 
 
 
 
�
�metrics
regularization_losses
	variables
�non_trainable_variables
�layer_metrics
�layers
trainable_variables
 �layer_regularization_losses
n
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
g
�
embeddings
�regularization_losses
�	variables
�trainable_variables
�	keras_api
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
�metrics
"regularization_losses
#	variables
�non_trainable_variables
�layer_metrics
�layers
$trainable_variables
 �layer_regularization_losses
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
�
�metrics
)regularization_losses
*	variables
�non_trainable_variables
�layer_metrics
�layers
+trainable_variables
 �layer_regularization_losses
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
�partial_output_shape
�full_output_shape
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
V
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
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
�metrics
3regularization_losses
4	variables
�non_trainable_variables
�layer_metrics
�layers
5trainable_variables
 �layer_regularization_losses
 
 
 
�
�metrics
7regularization_losses
8	variables
�non_trainable_variables
�layer_metrics
�layers
9trainable_variables
 �layer_regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
 

<0
=1

<0
=1
�
�metrics
>regularization_losses
?	variables
�non_trainable_variables
�layer_metrics
�layers
@trainable_variables
 �layer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
�
�metrics
Dregularization_losses
E	variables
�non_trainable_variables
�layer_metrics
�layers
Ftrainable_variables
 �layer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

H0
I1

H0
I1
�
�metrics
Jregularization_losses
K	variables
�non_trainable_variables
�layer_metrics
�layers
Ltrainable_variables
 �layer_regularization_losses
 
 
 
�
�metrics
Nregularization_losses
O	variables
�non_trainable_variables
�layer_metrics
�layers
Ptrainable_variables
 �layer_regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
�
�metrics
Uregularization_losses
V	variables
�non_trainable_variables
�layer_metrics
�layers
Wtrainable_variables
 �layer_regularization_losses
 
 
 
�
�metrics
Yregularization_losses
Z	variables
�non_trainable_variables
�layer_metrics
�layers
[trainable_variables
 �layer_regularization_losses
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

]0
^1

]0
^1
�
�metrics
_regularization_losses
`	variables
�non_trainable_variables
�layer_metrics
�layers
atrainable_variables
 �layer_regularization_losses
YW
VARIABLE_VALUEconv_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

c0
d1

c0
d1
�
�metrics
eregularization_losses
f	variables
�non_trainable_variables
�layer_metrics
�layers
gtrainable_variables
 �layer_regularization_losses
 
 
 
�
�metrics
iregularization_losses
j	variables
�non_trainable_variables
�layer_metrics
�layers
ktrainable_variables
 �layer_regularization_losses
 
 
 
�
�metrics
mregularization_losses
n	variables
�non_trainable_variables
�layer_metrics
�layers
otrainable_variables
 �layer_regularization_losses
WU
VARIABLE_VALUEFC_1/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	FC_1/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
�
�metrics
sregularization_losses
t	variables
�non_trainable_variables
�layer_metrics
�layers
utrainable_variables
 �layer_regularization_losses
 
 
 
�
�metrics
wregularization_losses
x	variables
�non_trainable_variables
�layer_metrics
�layers
ytrainable_variables
 �layer_regularization_losses
XV
VARIABLE_VALUEFC_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	FC_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

{0
|1

{0
|1
�
�metrics
}regularization_losses
~	variables
�non_trainable_variables
�layer_metrics
�layers
trainable_variables
 �layer_regularization_losses
 
 
 
�
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
`^
VARIABLE_VALUEoutput_layer/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEoutput_layer/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�0
�1
�
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
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
VT
VARIABLE_VALUEpatch_encoder/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEpatch_encoder/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE"patch_encoder/embedding/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!multi_head_attention/query/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmulti_head_attention/query/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmulti_head_attention/key/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEmulti_head_attention/key/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE!multi_head_attention/value/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEmulti_head_attention/value/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,multi_head_attention/attention_output/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*multi_head_attention/attention_output/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
 

�0

�0
�
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 

 0
!1
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
�
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
�
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
 
 
 
*
-0
.1
/2
03
14
25
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
��
VARIABLE_VALUE!AdamW/layer_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE AdamW/layer_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/FC_1/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamW/FC_1/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_2/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_2/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdamW/output_layer/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdamW/output_layer/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!AdamW/layer_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE AdamW/layer_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/FC_1/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamW/FC_1/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_2/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_2/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdamW/output_layer/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUEAdamW/output_layer/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�~
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_layerPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerpatch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddingslayer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biaslayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/betaconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*A
_read_only_resource_inputs#
!	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_301235
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�)
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOpFC_1/kernel/Read/ReadVariableOpFC_1/bias/Read/ReadVariableOpFC_2/kernel/Read/ReadVariableOpFC_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdamW/iter/Read/ReadVariableOp AdamW/beta_1/Read/ReadVariableOp AdamW/beta_2/Read/ReadVariableOpAdamW/decay/Read/ReadVariableOp'AdamW/learning_rate/Read/ReadVariableOp&AdamW/weight_decay/Read/ReadVariableOp.patch_encoder/dense/kernel/Read/ReadVariableOp,patch_encoder/dense/bias/Read/ReadVariableOp6patch_encoder/embedding/embeddings/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp5AdamW/layer_normalization/gamma/m/Read/ReadVariableOp4AdamW/layer_normalization/beta/m/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/m/Read/ReadVariableOp*AdamW/dense_1/kernel/m/Read/ReadVariableOp(AdamW/dense_1/bias/m/Read/ReadVariableOp*AdamW/dense_2/kernel/m/Read/ReadVariableOp(AdamW/dense_2/bias/m/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/m/Read/ReadVariableOp)AdamW/conv_1/kernel/m/Read/ReadVariableOp'AdamW/conv_1/bias/m/Read/ReadVariableOp)AdamW/conv_2/kernel/m/Read/ReadVariableOp'AdamW/conv_2/bias/m/Read/ReadVariableOp'AdamW/FC_1/kernel/m/Read/ReadVariableOp%AdamW/FC_1/bias/m/Read/ReadVariableOp'AdamW/FC_2/kernel/m/Read/ReadVariableOp%AdamW/FC_2/bias/m/Read/ReadVariableOp/AdamW/output_layer/kernel/m/Read/ReadVariableOp-AdamW/output_layer/bias/m/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/m/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/m/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/m/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/m/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/m/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/m/Read/ReadVariableOp5AdamW/layer_normalization/gamma/v/Read/ReadVariableOp4AdamW/layer_normalization/beta/v/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/v/Read/ReadVariableOp*AdamW/dense_1/kernel/v/Read/ReadVariableOp(AdamW/dense_1/bias/v/Read/ReadVariableOp*AdamW/dense_2/kernel/v/Read/ReadVariableOp(AdamW/dense_2/bias/v/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/v/Read/ReadVariableOp)AdamW/conv_1/kernel/v/Read/ReadVariableOp'AdamW/conv_1/bias/v/Read/ReadVariableOp)AdamW/conv_2/kernel/v/Read/ReadVariableOp'AdamW/conv_2/bias/v/Read/ReadVariableOp'AdamW/FC_1/kernel/v/Read/ReadVariableOp%AdamW/FC_1/bias/v/Read/ReadVariableOp'AdamW/FC_2/kernel/v/Read/ReadVariableOp%AdamW/FC_2/bias/v/Read/ReadVariableOp/AdamW/output_layer/kernel/v/Read/ReadVariableOp-AdamW/output_layer/bias/v/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/v/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/v/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/v/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/v/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/v/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpConst*t
Tinm
k2i	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_302756
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betalayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/betaconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias
AdamW/iterAdamW/beta_1AdamW/beta_2AdamW/decayAdamW/learning_rateAdamW/weight_decaypatch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddings!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biastotalcounttotal_1count_1!AdamW/layer_normalization/gamma/m AdamW/layer_normalization/beta/m#AdamW/layer_normalization_1/gamma/m"AdamW/layer_normalization_1/beta/mAdamW/dense_1/kernel/mAdamW/dense_1/bias/mAdamW/dense_2/kernel/mAdamW/dense_2/bias/m#AdamW/layer_normalization_2/gamma/m"AdamW/layer_normalization_2/beta/mAdamW/conv_1/kernel/mAdamW/conv_1/bias/mAdamW/conv_2/kernel/mAdamW/conv_2/bias/mAdamW/FC_1/kernel/mAdamW/FC_1/bias/mAdamW/FC_2/kernel/mAdamW/FC_2/bias/mAdamW/output_layer/kernel/mAdamW/output_layer/bias/m"AdamW/patch_encoder/dense/kernel/m AdamW/patch_encoder/dense/bias/m*AdamW/patch_encoder/embedding/embeddings/m)AdamW/multi_head_attention/query/kernel/m'AdamW/multi_head_attention/query/bias/m'AdamW/multi_head_attention/key/kernel/m%AdamW/multi_head_attention/key/bias/m)AdamW/multi_head_attention/value/kernel/m'AdamW/multi_head_attention/value/bias/m4AdamW/multi_head_attention/attention_output/kernel/m2AdamW/multi_head_attention/attention_output/bias/m!AdamW/layer_normalization/gamma/v AdamW/layer_normalization/beta/v#AdamW/layer_normalization_1/gamma/v"AdamW/layer_normalization_1/beta/vAdamW/dense_1/kernel/vAdamW/dense_1/bias/vAdamW/dense_2/kernel/vAdamW/dense_2/bias/v#AdamW/layer_normalization_2/gamma/v"AdamW/layer_normalization_2/beta/vAdamW/conv_1/kernel/vAdamW/conv_1/bias/vAdamW/conv_2/kernel/vAdamW/conv_2/bias/vAdamW/FC_1/kernel/vAdamW/FC_1/bias/vAdamW/FC_2/kernel/vAdamW/FC_2/bias/vAdamW/output_layer/kernel/vAdamW/output_layer/bias/v"AdamW/patch_encoder/dense/kernel/v AdamW/patch_encoder/dense/bias/v*AdamW/patch_encoder/embedding/embeddings/v)AdamW/multi_head_attention/query/kernel/v'AdamW/multi_head_attention/query/bias/v'AdamW/multi_head_attention/key/kernel/v%AdamW/multi_head_attention/key/bias/v)AdamW/multi_head_attention/value/kernel/v'AdamW/multi_head_attention/value/bias/v4AdamW/multi_head_attention/attention_output/kernel/v2AdamW/multi_head_attention/attention_output/bias/v*s
Tinl
j2h*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_303075��
�b
�
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301072
input_layer'
patch_encoder_300990:	�@"
patch_encoder_300992:@'
patch_encoder_300994:	�@(
layer_normalization_300997:@(
layer_normalization_300999:@1
multi_head_attention_301002:@@-
multi_head_attention_301004:@1
multi_head_attention_301006:@@-
multi_head_attention_301008:@1
multi_head_attention_301010:@@-
multi_head_attention_301012:@1
multi_head_attention_301014:@@)
multi_head_attention_301016:@*
layer_normalization_1_301020:@*
layer_normalization_1_301022:@!
dense_1_301025:	@�
dense_1_301027:	�!
dense_2_301030:	�@
dense_2_301032:@*
layer_normalization_2_301036:@*
layer_normalization_2_301038:@'
conv_1_301042:@

conv_1_301044:
'
conv_2_301047:


conv_2_301049:

fc_1_301054:	�2
fc_1_301056:2
fc_2_301060:22
fc_2_301062:2%
output_layer_301066:2!
output_layer_301068:
identity��FC_1/StatefulPartitionedCall�FC_2/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�%patch_encoder/StatefulPartitionedCall�
patches/PartitionedCallPartitionedCallinput_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_2999792
patches/PartitionedCall�
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_300990patch_encoder_300992patch_encoder_300994*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3000212'
%patch_encoder/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_300997layer_normalization_300999*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3000512-
+layer_normalization/StatefulPartitionedCall�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_301002multi_head_attention_301004multi_head_attention_301006multi_head_attention_301008multi_head_attention_301010multi_head_attention_301012multi_head_attention_301014multi_head_attention_301016*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3000922.
,multi_head_attention/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3001162
add/PartitionedCall�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_301020layer_normalization_1_301022*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3001402/
-layer_normalization_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_301025dense_1_301027*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3001842!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_301030dense_2_301032*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3002282!
dense_2/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3002402
add_1/PartitionedCall�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_301036layer_normalization_2_301038*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3002642/
-layer_normalization_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3002842
reshape/PartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv_1_301042conv_1_301044*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3002962 
conv_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_301047conv_2_301049*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3003122 
conv_2/StatefulPartitionedCall�
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_2999522
maxpool_1/PartitionedCall�
flatten_layer/PartitionedCallPartitionedCall"maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3003252
flatten_layer/PartitionedCall�
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_301054fc_1_301056*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3003372
FC_1/StatefulPartitionedCall�
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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3003482
leaky_ReLu_1/PartitionedCall�
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_301060fc_2_301062*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3003602
FC_2/StatefulPartitionedCall�
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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3003712
leaky_ReLu_2/PartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_301066output_layer_301068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3003842&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�
�
6__inference_WheatClassifier_CNN_1_layer_call_fn_301876

inputs
unknown:	�@
	unknown_0:@
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:@@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:	@�

unknown_15:	�

unknown_16:	�@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@


unknown_21:
$

unknown_22:



unknown_23:


unknown_24:	�2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identity��StatefulPartitionedCall�
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
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*A
_read_only_resource_inputs#
!	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_3008542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�	
�
@__inference_FC_1_layer_call_and_return_conditional_losses_302356

inputs1
matmul_readvariableop_resource:	�2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�2*
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_WheatClassifier_CNN_1_layer_call_fn_300986
input_layer
unknown:	�@
	unknown_0:@
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:@@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:	@�

unknown_15:	�

unknown_16:	�@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@


unknown_21:
$

unknown_22:



unknown_23:


unknown_24:	�2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identity��StatefulPartitionedCall�
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
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*A
_read_only_resource_inputs#
!	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_3008542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�
R
&__inference_add_1_layer_call_fn_302247
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3002402
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������@:����������@:V R
,
_output_shapes
:����������@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������@
"
_user_specified_name
inputs/1
��
�
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301742

inputsH
5patch_encoder_dense_tensordot_readvariableop_resource:	�@A
3patch_encoder_dense_biasadd_readvariableop_resource:@B
/patch_encoder_embedding_embedding_lookup_301528:	�@G
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
7layer_normalization_2_batchnorm_readvariableop_resource:@?
%conv_1_conv2d_readvariableop_resource:@
4
&conv_1_biasadd_readvariableop_resource:
?
%conv_2_conv2d_readvariableop_resource:

4
&conv_2_biasadd_readvariableop_resource:
6
#fc_1_matmul_readvariableop_resource:	�22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identity��FC_1/BiasAdd/ReadVariableOp�FC_1/MatMul/ReadVariableOp�FC_2/BiasAdd/ReadVariableOp�FC_2/MatMul/ReadVariableOp�conv_1/BiasAdd/ReadVariableOp�conv_1/Conv2D/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp� dense_2/Tensordot/ReadVariableOp�,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�.layer_normalization_2/batchnorm/ReadVariableOp�2layer_normalization_2/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�*patch_encoder/dense/BiasAdd/ReadVariableOp�,patch_encoder/dense/Tensordot/ReadVariableOp�(patch_encoder/embedding/embedding_lookupT
patches/ShapeShapeinputs*
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
patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*0
_output_shapes
:����������*
ksizes


*
paddingVALID*
rates
*
strides


2
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
B :�2
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
!:�������������������2
patches/Reshapex
patch_encoder/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
patch_encoder/range/starty
patch_encoder/range/limitConst*
_output_shapes
: *
dtype0*
value
B :�2
patch_encoder/range/limitx
patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder/range/delta�
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes	
:�2
patch_encoder/range�
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
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
!:�������������������2)
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
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_301528patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/301528*
_output_shapes
:	�@*
dtype02*
(patch_encoder/embedding/embedding_lookup�
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/301528*
_output_shapes
:	�@23
1patch_encoder/embedding/embedding_lookup/Identity�
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	�@25
3patch_encoder/embedding/embedding_lookup/Identity_1�
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������@2
patch_encoder/add�
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices�
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2"
 layer_normalization/moments/mean�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:����������2*
(layer_normalization/moments/StopGradient�
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2/
-layer_normalization/moments/SquaredDifference�
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2#
!layer_normalization/batchnorm/add�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:����������2%
#layer_normalization/batchnorm/Rsqrt�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!layer_normalization/batchnorm/mul�
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization/batchnorm/mul_1�
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization/batchnorm/mul_2�
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOp�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2#
!layer_normalization/batchnorm/sub�
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization/batchnorm/add_1�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp�
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOp�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2 
multi_head_attention/query/add�
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp�
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOp�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
multi_head_attention/key/add�
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp�
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOp�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/y�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:����������@2
multi_head_attention/Mul�
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:�����������*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsum�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:�����������2&
$multi_head_attention/softmax/Softmax�
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2,
*multi_head_attention/dropout/dropout/Const�
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:�����������2*
(multi_head_attention/dropout/dropout/Mul�
*multi_head_attention/dropout/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2,
*multi_head_attention/dropout/dropout/Shape�
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:�����������*
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
T0*1
_output_shapes
:�����������23
1multi_head_attention/dropout/dropout/GreaterEqual�
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:�����������2+
)multi_head_attention/dropout/dropout/Cast�
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:�����������2,
*multi_head_attention/dropout/dropout/Mul_1�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:����������@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsum�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsum�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOp�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2+
)multi_head_attention/attention_output/add�
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*,
_output_shapes
:����������@2	
add/add�
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices�
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2$
"layer_normalization_1/moments/mean�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:����������2,
*layer_normalization_1/moments/StopGradient�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@21
/layer_normalization_1/moments/SquaredDifference�
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2%
#layer_normalization_1/batchnorm/add�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:����������2'
%layer_normalization_1/batchnorm/Rsqrt�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization_1/batchnorm/mul�
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_1/batchnorm/mul_1�
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_1/batchnorm/mul_2�
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization_1/batchnorm/sub�
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2'
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
T0*,
_output_shapes
:����������@2
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
T0*-
_output_shapes
:�����������2
dense_1/Tensordot�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/x�
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*-
_output_shapes
:�����������2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_1/Gelu/Cast/x�
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:�����������2
dense_1/Gelu/truediv}
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:�����������2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1/Gelu/add/x�
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:�����������2
dense_1/Gelu/add�
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*-
_output_shapes
:�����������2
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
T0*-
_output_shapes
:�����������2
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
T0*,
_output_shapes
:����������@2
dense_2/Tensordot�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/x�
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_2/Gelu/Cast/x�
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/truediv|
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_2/Gelu/add/x�
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/add�
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/mul_1{
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*,
_output_shapes
:����������@2
	add_1/add�
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indices�
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2$
"layer_normalization_2/moments/mean�
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:����������2,
*layer_normalization_2/moments/StopGradient�
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@21
/layer_normalization_2/moments/SquaredDifference�
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices�
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2%
#layer_normalization_2/batchnorm/add�
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:����������2'
%layer_normalization_2/batchnorm/Rsqrt�
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOp�
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization_2/batchnorm/mul�
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_2/batchnorm/mul_1�
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_2/batchnorm/mul_2�
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOp�
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization_2/batchnorm/sub�
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_2/batchnorm/add_1w
reshape/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
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
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/3�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
reshape/Reshape�
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype02
conv_1/Conv2D/ReadVariableOp�
conv_1/Conv2DConv2Dreshape/Reshape:output:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
2
conv_2/BiasAdd�
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPool{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_layer/Const�
flatten_layer/ReshapeReshapemaxpool_1/MaxPool:output:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_layer/Reshape�
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	�2*
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
output_layer/Softmax�
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
FC_1/BiasAdd/ReadVariableOpFC_1/BiasAdd/ReadVariableOp28
FC_1/MatMul/ReadVariableOpFC_1/MatMul/ReadVariableOp2:
FC_2/BiasAdd/ReadVariableOpFC_2/BiasAdd/ReadVariableOp28
FC_2/MatMul/ReadVariableOpFC_2/MatMul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2X
*patch_encoder/dense/BiasAdd/ReadVariableOp*patch_encoder/dense/BiasAdd/ReadVariableOp2\
,patch_encoder/dense/Tensordot/ReadVariableOp,patch_encoder/dense/Tensordot/ReadVariableOp2T
(patch_encoder/embedding/embedding_lookup(patch_encoder/embedding/embedding_lookup:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�/
�
I__inference_patch_encoder_layer_call_and_return_conditional_losses_301935	
patch:
'dense_tensordot_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@4
!embedding_embedding_lookup_301928:	�@
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�embedding/embedding_lookup\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start]
range/limitConst*
_output_shapes
: *
dtype0*
value
B :�2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:�2
range�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
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
!:�������������������2
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
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_301928range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/301928*
_output_shapes
:	�@*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/301928*
_output_shapes
:	�@2%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	�@2'
%embedding/embedding_lookup/Identity_1�
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������@2
add�
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:�������������������

_user_specified_namepatch
��
�&
!__inference__wrapped_model_299946
input_layer^
Kwheatclassifier_cnn_1_patch_encoder_dense_tensordot_readvariableop_resource:	�@W
Iwheatclassifier_cnn_1_patch_encoder_dense_biasadd_readvariableop_resource:@X
Ewheatclassifier_cnn_1_patch_encoder_embedding_embedding_lookup_299739:	�@]
Owheatclassifier_cnn_1_layer_normalization_batchnorm_mul_readvariableop_resource:@Y
Kwheatclassifier_cnn_1_layer_normalization_batchnorm_readvariableop_resource:@l
Vwheatclassifier_cnn_1_multi_head_attention_query_einsum_einsum_readvariableop_resource:@@^
Lwheatclassifier_cnn_1_multi_head_attention_query_add_readvariableop_resource:@j
Twheatclassifier_cnn_1_multi_head_attention_key_einsum_einsum_readvariableop_resource:@@\
Jwheatclassifier_cnn_1_multi_head_attention_key_add_readvariableop_resource:@l
Vwheatclassifier_cnn_1_multi_head_attention_value_einsum_einsum_readvariableop_resource:@@^
Lwheatclassifier_cnn_1_multi_head_attention_value_add_readvariableop_resource:@w
awheatclassifier_cnn_1_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:@@e
Wwheatclassifier_cnn_1_multi_head_attention_attention_output_add_readvariableop_resource:@_
Qwheatclassifier_cnn_1_layer_normalization_1_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_cnn_1_layer_normalization_1_batchnorm_readvariableop_resource:@R
?wheatclassifier_cnn_1_dense_1_tensordot_readvariableop_resource:	@�L
=wheatclassifier_cnn_1_dense_1_biasadd_readvariableop_resource:	�R
?wheatclassifier_cnn_1_dense_2_tensordot_readvariableop_resource:	�@K
=wheatclassifier_cnn_1_dense_2_biasadd_readvariableop_resource:@_
Qwheatclassifier_cnn_1_layer_normalization_2_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_cnn_1_layer_normalization_2_batchnorm_readvariableop_resource:@U
;wheatclassifier_cnn_1_conv_1_conv2d_readvariableop_resource:@
J
<wheatclassifier_cnn_1_conv_1_biasadd_readvariableop_resource:
U
;wheatclassifier_cnn_1_conv_2_conv2d_readvariableop_resource:

J
<wheatclassifier_cnn_1_conv_2_biasadd_readvariableop_resource:
L
9wheatclassifier_cnn_1_fc_1_matmul_readvariableop_resource:	�2H
:wheatclassifier_cnn_1_fc_1_biasadd_readvariableop_resource:2K
9wheatclassifier_cnn_1_fc_2_matmul_readvariableop_resource:22H
:wheatclassifier_cnn_1_fc_2_biasadd_readvariableop_resource:2S
Awheatclassifier_cnn_1_output_layer_matmul_readvariableop_resource:2P
Bwheatclassifier_cnn_1_output_layer_biasadd_readvariableop_resource:
identity��1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp�0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp�1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp�0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp�3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp�2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp�3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp�2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp�4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp�6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp�4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp�6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOp�BWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOp�FWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOp�DWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOp�HWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOp�DWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOp�HWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOp�NWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOp�XWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�AWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOp�KWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOp�CWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOp�MWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOp�CWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOp�MWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOp�9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp�8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOp�@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOp�BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOp�>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup�
#WheatClassifier_CNN_1/patches/ShapeShapeinput_layer*
T0*
_output_shapes
:2%
#WheatClassifier_CNN_1/patches/Shape�
1WheatClassifier_CNN_1/patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_CNN_1/patches/strided_slice/stack�
3WheatClassifier_CNN_1/patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_1/patches/strided_slice/stack_1�
3WheatClassifier_CNN_1/patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_1/patches/strided_slice/stack_2�
+WheatClassifier_CNN_1/patches/strided_sliceStridedSlice,WheatClassifier_CNN_1/patches/Shape:output:0:WheatClassifier_CNN_1/patches/strided_slice/stack:output:0<WheatClassifier_CNN_1/patches/strided_slice/stack_1:output:0<WheatClassifier_CNN_1/patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+WheatClassifier_CNN_1/patches/strided_slice�
1WheatClassifier_CNN_1/patches/ExtractImagePatchesExtractImagePatchesinput_layer*
T0*0
_output_shapes
:����������*
ksizes


*
paddingVALID*
rates
*
strides


23
1WheatClassifier_CNN_1/patches/ExtractImagePatches�
-WheatClassifier_CNN_1/patches/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
���������2/
-WheatClassifier_CNN_1/patches/Reshape/shape/1�
-WheatClassifier_CNN_1/patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :�2/
-WheatClassifier_CNN_1/patches/Reshape/shape/2�
+WheatClassifier_CNN_1/patches/Reshape/shapePack4WheatClassifier_CNN_1/patches/strided_slice:output:06WheatClassifier_CNN_1/patches/Reshape/shape/1:output:06WheatClassifier_CNN_1/patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+WheatClassifier_CNN_1/patches/Reshape/shape�
%WheatClassifier_CNN_1/patches/ReshapeReshape;WheatClassifier_CNN_1/patches/ExtractImagePatches:patches:04WheatClassifier_CNN_1/patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:�������������������2'
%WheatClassifier_CNN_1/patches/Reshape�
/WheatClassifier_CNN_1/patch_encoder/range/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/WheatClassifier_CNN_1/patch_encoder/range/start�
/WheatClassifier_CNN_1/patch_encoder/range/limitConst*
_output_shapes
: *
dtype0*
value
B :�21
/WheatClassifier_CNN_1/patch_encoder/range/limit�
/WheatClassifier_CNN_1/patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/WheatClassifier_CNN_1/patch_encoder/range/delta�
)WheatClassifier_CNN_1/patch_encoder/rangeRange8WheatClassifier_CNN_1/patch_encoder/range/start:output:08WheatClassifier_CNN_1/patch_encoder/range/limit:output:08WheatClassifier_CNN_1/patch_encoder/range/delta:output:0*
_output_shapes	
:�2+
)WheatClassifier_CNN_1/patch_encoder/range�
BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOpKwheatclassifier_cnn_1_patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype02D
BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOp�
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/axes�
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/free�
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ShapeShape.WheatClassifier_CNN_1/patches/Reshape:output:0*
T0*
_output_shapes
:2;
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Shape�
AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2/axis�
<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2GatherV2BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Shape:output:0AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/free:output:0JWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2�
CWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
CWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1/axis�
>WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1GatherV2BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Shape:output:0AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/axes:output:0LWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1�
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const�
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ProdProdEWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2:output:0BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod�
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_1�
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod_1ProdGWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1:output:0DWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod_1�
?WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat/axis�
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concatConcatV2AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/free:output:0AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/axes:output:0HWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat�
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/stackPackAWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod:output:0CWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/stack�
=WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/transpose	Transpose.WheatClassifier_CNN_1/patches/Reshape:output:0CWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:�������������������2?
=WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/transpose�
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReshapeReshapeAWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/transpose:y:0BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2=
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Reshape�
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/MatMulMatMulDWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Reshape:output:0JWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2<
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/MatMul�
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_2�
AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1/axis�
<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1ConcatV2EWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2:output:0DWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_2:output:0JWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1�
3WheatClassifier_CNN_1/patch_encoder/dense/TensordotReshapeDWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/MatMul:product:0EWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@25
3WheatClassifier_CNN_1/patch_encoder/dense/Tensordot�
@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOpIwheatclassifier_cnn_1_patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOp�
1WheatClassifier_CNN_1/patch_encoder/dense/BiasAddBiasAdd<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot:output:0HWheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@23
1WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd�
>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookupResourceGatherEwheatclassifier_cnn_1_patch_encoder_embedding_embedding_lookup_2997392WheatClassifier_CNN_1/patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/299739*
_output_shapes
:	�@*
dtype02@
>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup�
GWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/IdentityIdentityGWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/299739*
_output_shapes
:	�@2I
GWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/Identity�
IWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/Identity_1IdentityPWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	�@2K
IWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/Identity_1�
'WheatClassifier_CNN_1/patch_encoder/addAddV2:WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd:output:0RWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������@2)
'WheatClassifier_CNN_1/patch_encoder/add�
HWheatClassifier_CNN_1/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
HWheatClassifier_CNN_1/layer_normalization/moments/mean/reduction_indices�
6WheatClassifier_CNN_1/layer_normalization/moments/meanMean+WheatClassifier_CNN_1/patch_encoder/add:z:0QWheatClassifier_CNN_1/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(28
6WheatClassifier_CNN_1/layer_normalization/moments/mean�
>WheatClassifier_CNN_1/layer_normalization/moments/StopGradientStopGradient?WheatClassifier_CNN_1/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:����������2@
>WheatClassifier_CNN_1/layer_normalization/moments/StopGradient�
CWheatClassifier_CNN_1/layer_normalization/moments/SquaredDifferenceSquaredDifference+WheatClassifier_CNN_1/patch_encoder/add:z:0GWheatClassifier_CNN_1/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2E
CWheatClassifier_CNN_1/layer_normalization/moments/SquaredDifference�
LWheatClassifier_CNN_1/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
LWheatClassifier_CNN_1/layer_normalization/moments/variance/reduction_indices�
:WheatClassifier_CNN_1/layer_normalization/moments/varianceMeanGWheatClassifier_CNN_1/layer_normalization/moments/SquaredDifference:z:0UWheatClassifier_CNN_1/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2<
:WheatClassifier_CNN_1/layer_normalization/moments/variance�
9WheatClassifier_CNN_1/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/add/y�
7WheatClassifier_CNN_1/layer_normalization/batchnorm/addAddV2CWheatClassifier_CNN_1/layer_normalization/moments/variance:output:0BWheatClassifier_CNN_1/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:����������29
7WheatClassifier_CNN_1/layer_normalization/batchnorm/add�
9WheatClassifier_CNN_1/layer_normalization/batchnorm/RsqrtRsqrt;WheatClassifier_CNN_1/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:����������2;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/Rsqrt�
FWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpOwheatclassifier_cnn_1_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02H
FWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOp�
7WheatClassifier_CNN_1/layer_normalization/batchnorm/mulMul=WheatClassifier_CNN_1/layer_normalization/batchnorm/Rsqrt:y:0NWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@29
7WheatClassifier_CNN_1/layer_normalization/batchnorm/mul�
9WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_1Mul+WheatClassifier_CNN_1/patch_encoder/add:z:0;WheatClassifier_CNN_1/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_1�
9WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_2Mul?WheatClassifier_CNN_1/layer_normalization/moments/mean:output:0;WheatClassifier_CNN_1/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_2�
BWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOpReadVariableOpKwheatclassifier_cnn_1_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02D
BWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOp�
7WheatClassifier_CNN_1/layer_normalization/batchnorm/subSubJWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOp:value:0=WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@29
7WheatClassifier_CNN_1/layer_normalization/batchnorm/sub�
9WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1AddV2=WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_1:z:0;WheatClassifier_CNN_1/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1�
MWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_cnn_1_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOp�
>WheatClassifier_CNN_1/multi_head_attention/query/einsum/EinsumEinsum=WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1:z:0UWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2@
>WheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum�
CWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOpReadVariableOpLwheatclassifier_cnn_1_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOp�
4WheatClassifier_CNN_1/multi_head_attention/query/addAddV2GWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum:output:0KWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@26
4WheatClassifier_CNN_1/multi_head_attention/query/add�
KWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpTwheatclassifier_cnn_1_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02M
KWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOp�
<WheatClassifier_CNN_1/multi_head_attention/key/einsum/EinsumEinsum=WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1:z:0SWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2>
<WheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum�
AWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOpReadVariableOpJwheatclassifier_cnn_1_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02C
AWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOp�
2WheatClassifier_CNN_1/multi_head_attention/key/addAddV2EWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum:output:0IWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@24
2WheatClassifier_CNN_1/multi_head_attention/key/add�
MWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_cnn_1_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOp�
>WheatClassifier_CNN_1/multi_head_attention/value/einsum/EinsumEinsum=WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1:z:0UWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2@
>WheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum�
CWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOpReadVariableOpLwheatclassifier_cnn_1_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOp�
4WheatClassifier_CNN_1/multi_head_attention/value/addAddV2GWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum:output:0KWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@26
4WheatClassifier_CNN_1/multi_head_attention/value/add�
0WheatClassifier_CNN_1/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >22
0WheatClassifier_CNN_1/multi_head_attention/Mul/y�
.WheatClassifier_CNN_1/multi_head_attention/MulMul8WheatClassifier_CNN_1/multi_head_attention/query/add:z:09WheatClassifier_CNN_1/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:����������@20
.WheatClassifier_CNN_1/multi_head_attention/Mul�
8WheatClassifier_CNN_1/multi_head_attention/einsum/EinsumEinsum6WheatClassifier_CNN_1/multi_head_attention/key/add:z:02WheatClassifier_CNN_1/multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:�����������*
equationaecd,abcd->acbe2:
8WheatClassifier_CNN_1/multi_head_attention/einsum/Einsum�
:WheatClassifier_CNN_1/multi_head_attention/softmax/SoftmaxSoftmaxAWheatClassifier_CNN_1/multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:�����������2<
:WheatClassifier_CNN_1/multi_head_attention/softmax/Softmax�
;WheatClassifier_CNN_1/multi_head_attention/dropout/IdentityIdentityDWheatClassifier_CNN_1/multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:�����������2=
;WheatClassifier_CNN_1/multi_head_attention/dropout/Identity�
:WheatClassifier_CNN_1/multi_head_attention/einsum_1/EinsumEinsumDWheatClassifier_CNN_1/multi_head_attention/dropout/Identity:output:08WheatClassifier_CNN_1/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:����������@*
equationacbe,aecd->abcd2<
:WheatClassifier_CNN_1/multi_head_attention/einsum_1/Einsum�
XWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpawheatclassifier_cnn_1_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02Z
XWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�
IWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/EinsumEinsumCWheatClassifier_CNN_1/multi_head_attention/einsum_1/Einsum:output:0`WheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������@*
equationabcd,cde->abe2K
IWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum�
NWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpWwheatclassifier_cnn_1_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02P
NWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOp�
?WheatClassifier_CNN_1/multi_head_attention/attention_output/addAddV2RWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum:output:0VWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2A
?WheatClassifier_CNN_1/multi_head_attention/attention_output/add�
WheatClassifier_CNN_1/add/addAddV2CWheatClassifier_CNN_1/multi_head_attention/attention_output/add:z:0+WheatClassifier_CNN_1/patch_encoder/add:z:0*
T0*,
_output_shapes
:����������@2
WheatClassifier_CNN_1/add/add�
JWheatClassifier_CNN_1/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_CNN_1/layer_normalization_1/moments/mean/reduction_indices�
8WheatClassifier_CNN_1/layer_normalization_1/moments/meanMean!WheatClassifier_CNN_1/add/add:z:0SWheatClassifier_CNN_1/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2:
8WheatClassifier_CNN_1/layer_normalization_1/moments/mean�
@WheatClassifier_CNN_1/layer_normalization_1/moments/StopGradientStopGradientAWheatClassifier_CNN_1/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:����������2B
@WheatClassifier_CNN_1/layer_normalization_1/moments/StopGradient�
EWheatClassifier_CNN_1/layer_normalization_1/moments/SquaredDifferenceSquaredDifference!WheatClassifier_CNN_1/add/add:z:0IWheatClassifier_CNN_1/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2G
EWheatClassifier_CNN_1/layer_normalization_1/moments/SquaredDifference�
NWheatClassifier_CNN_1/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN_1/layer_normalization_1/moments/variance/reduction_indices�
<WheatClassifier_CNN_1/layer_normalization_1/moments/varianceMeanIWheatClassifier_CNN_1/layer_normalization_1/moments/SquaredDifference:z:0WWheatClassifier_CNN_1/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2>
<WheatClassifier_CNN_1/layer_normalization_1/moments/variance�
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add/y�
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/addAddV2EWheatClassifier_CNN_1/layer_normalization_1/moments/variance:output:0DWheatClassifier_CNN_1/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:����������2;
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add�
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/RsqrtRsqrt=WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:����������2=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/Rsqrt�
HWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_cnn_1_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOp�
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mulMul?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/Rsqrt:y:0PWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2;
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul�
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_1Mul!WheatClassifier_CNN_1/add/add:z:0=WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_1�
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_2MulAWheatClassifier_CNN_1/layer_normalization_1/moments/mean:output:0=WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_2�
DWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_cnn_1_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOp�
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/subSubLWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOp:value:0?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2;
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/sub�
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add_1AddV2?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_1:z:0=WheatClassifier_CNN_1/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add_1�
6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_cnn_1_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@�*
dtype028
6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp�
,WheatClassifier_CNN_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_CNN_1/dense_1/Tensordot/axes�
,WheatClassifier_CNN_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_CNN_1/dense_1/Tensordot/free�
-WheatClassifier_CNN_1/dense_1/Tensordot/ShapeShape?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_1/Tensordot/Shape�
5WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2/axis�
0WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2GatherV26WheatClassifier_CNN_1/dense_1/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_1/Tensordot/free:output:0>WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2�
7WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1/axis�
2WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1GatherV26WheatClassifier_CNN_1/dense_1/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_1/Tensordot/axes:output:0@WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1�
-WheatClassifier_CNN_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_CNN_1/dense_1/Tensordot/Const�
,WheatClassifier_CNN_1/dense_1/Tensordot/ProdProd9WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2:output:06WheatClassifier_CNN_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_CNN_1/dense_1/Tensordot/Prod�
/WheatClassifier_CNN_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_CNN_1/dense_1/Tensordot/Const_1�
.WheatClassifier_CNN_1/dense_1/Tensordot/Prod_1Prod;WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1:output:08WheatClassifier_CNN_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_CNN_1/dense_1/Tensordot/Prod_1�
3WheatClassifier_CNN_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_CNN_1/dense_1/Tensordot/concat/axis�
.WheatClassifier_CNN_1/dense_1/Tensordot/concatConcatV25WheatClassifier_CNN_1/dense_1/Tensordot/free:output:05WheatClassifier_CNN_1/dense_1/Tensordot/axes:output:0<WheatClassifier_CNN_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_CNN_1/dense_1/Tensordot/concat�
-WheatClassifier_CNN_1/dense_1/Tensordot/stackPack5WheatClassifier_CNN_1/dense_1/Tensordot/Prod:output:07WheatClassifier_CNN_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_1/Tensordot/stack�
1WheatClassifier_CNN_1/dense_1/Tensordot/transpose	Transpose?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add_1:z:07WheatClassifier_CNN_1/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������@23
1WheatClassifier_CNN_1/dense_1/Tensordot/transpose�
/WheatClassifier_CNN_1/dense_1/Tensordot/ReshapeReshape5WheatClassifier_CNN_1/dense_1/Tensordot/transpose:y:06WheatClassifier_CNN_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������21
/WheatClassifier_CNN_1/dense_1/Tensordot/Reshape�
.WheatClassifier_CNN_1/dense_1/Tensordot/MatMulMatMul8WheatClassifier_CNN_1/dense_1/Tensordot/Reshape:output:0>WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������20
.WheatClassifier_CNN_1/dense_1/Tensordot/MatMul�
/WheatClassifier_CNN_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:�21
/WheatClassifier_CNN_1/dense_1/Tensordot/Const_2�
5WheatClassifier_CNN_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_1/Tensordot/concat_1/axis�
0WheatClassifier_CNN_1/dense_1/Tensordot/concat_1ConcatV29WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2:output:08WheatClassifier_CNN_1/dense_1/Tensordot/Const_2:output:0>WheatClassifier_CNN_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_1/Tensordot/concat_1�
'WheatClassifier_CNN_1/dense_1/TensordotReshape8WheatClassifier_CNN_1/dense_1/Tensordot/MatMul:product:09WheatClassifier_CNN_1/dense_1/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:�����������2)
'WheatClassifier_CNN_1/dense_1/Tensordot�
4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_cnn_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype026
4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp�
%WheatClassifier_CNN_1/dense_1/BiasAddBiasAdd0WheatClassifier_CNN_1/dense_1/Tensordot:output:0<WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������2'
%WheatClassifier_CNN_1/dense_1/BiasAdd�
(WheatClassifier_CNN_1/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_CNN_1/dense_1/Gelu/mul/x�
&WheatClassifier_CNN_1/dense_1/Gelu/mulMul1WheatClassifier_CNN_1/dense_1/Gelu/mul/x:output:0.WheatClassifier_CNN_1/dense_1/BiasAdd:output:0*
T0*-
_output_shapes
:�����������2(
&WheatClassifier_CNN_1/dense_1/Gelu/mul�
)WheatClassifier_CNN_1/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2+
)WheatClassifier_CNN_1/dense_1/Gelu/Cast/x�
*WheatClassifier_CNN_1/dense_1/Gelu/truedivRealDiv.WheatClassifier_CNN_1/dense_1/BiasAdd:output:02WheatClassifier_CNN_1/dense_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:�����������2,
*WheatClassifier_CNN_1/dense_1/Gelu/truediv�
&WheatClassifier_CNN_1/dense_1/Gelu/ErfErf.WheatClassifier_CNN_1/dense_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:�����������2(
&WheatClassifier_CNN_1/dense_1/Gelu/Erf�
(WheatClassifier_CNN_1/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2*
(WheatClassifier_CNN_1/dense_1/Gelu/add/x�
&WheatClassifier_CNN_1/dense_1/Gelu/addAddV21WheatClassifier_CNN_1/dense_1/Gelu/add/x:output:0*WheatClassifier_CNN_1/dense_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:�����������2(
&WheatClassifier_CNN_1/dense_1/Gelu/add�
(WheatClassifier_CNN_1/dense_1/Gelu/mul_1Mul*WheatClassifier_CNN_1/dense_1/Gelu/mul:z:0*WheatClassifier_CNN_1/dense_1/Gelu/add:z:0*
T0*-
_output_shapes
:�����������2*
(WheatClassifier_CNN_1/dense_1/Gelu/mul_1�
6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_cnn_1_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	�@*
dtype028
6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOp�
,WheatClassifier_CNN_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_CNN_1/dense_2/Tensordot/axes�
,WheatClassifier_CNN_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_CNN_1/dense_2/Tensordot/free�
-WheatClassifier_CNN_1/dense_2/Tensordot/ShapeShape,WheatClassifier_CNN_1/dense_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_2/Tensordot/Shape�
5WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2/axis�
0WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2GatherV26WheatClassifier_CNN_1/dense_2/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_2/Tensordot/free:output:0>WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2�
7WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1/axis�
2WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1GatherV26WheatClassifier_CNN_1/dense_2/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_2/Tensordot/axes:output:0@WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1�
-WheatClassifier_CNN_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_CNN_1/dense_2/Tensordot/Const�
,WheatClassifier_CNN_1/dense_2/Tensordot/ProdProd9WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2:output:06WheatClassifier_CNN_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_CNN_1/dense_2/Tensordot/Prod�
/WheatClassifier_CNN_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_CNN_1/dense_2/Tensordot/Const_1�
.WheatClassifier_CNN_1/dense_2/Tensordot/Prod_1Prod;WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1:output:08WheatClassifier_CNN_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_CNN_1/dense_2/Tensordot/Prod_1�
3WheatClassifier_CNN_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_CNN_1/dense_2/Tensordot/concat/axis�
.WheatClassifier_CNN_1/dense_2/Tensordot/concatConcatV25WheatClassifier_CNN_1/dense_2/Tensordot/free:output:05WheatClassifier_CNN_1/dense_2/Tensordot/axes:output:0<WheatClassifier_CNN_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_CNN_1/dense_2/Tensordot/concat�
-WheatClassifier_CNN_1/dense_2/Tensordot/stackPack5WheatClassifier_CNN_1/dense_2/Tensordot/Prod:output:07WheatClassifier_CNN_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_2/Tensordot/stack�
1WheatClassifier_CNN_1/dense_2/Tensordot/transpose	Transpose,WheatClassifier_CNN_1/dense_1/Gelu/mul_1:z:07WheatClassifier_CNN_1/dense_2/Tensordot/concat:output:0*
T0*-
_output_shapes
:�����������23
1WheatClassifier_CNN_1/dense_2/Tensordot/transpose�
/WheatClassifier_CNN_1/dense_2/Tensordot/ReshapeReshape5WheatClassifier_CNN_1/dense_2/Tensordot/transpose:y:06WheatClassifier_CNN_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������21
/WheatClassifier_CNN_1/dense_2/Tensordot/Reshape�
.WheatClassifier_CNN_1/dense_2/Tensordot/MatMulMatMul8WheatClassifier_CNN_1/dense_2/Tensordot/Reshape:output:0>WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@20
.WheatClassifier_CNN_1/dense_2/Tensordot/MatMul�
/WheatClassifier_CNN_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@21
/WheatClassifier_CNN_1/dense_2/Tensordot/Const_2�
5WheatClassifier_CNN_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_2/Tensordot/concat_1/axis�
0WheatClassifier_CNN_1/dense_2/Tensordot/concat_1ConcatV29WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2:output:08WheatClassifier_CNN_1/dense_2/Tensordot/Const_2:output:0>WheatClassifier_CNN_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_2/Tensordot/concat_1�
'WheatClassifier_CNN_1/dense_2/TensordotReshape8WheatClassifier_CNN_1/dense_2/Tensordot/MatMul:product:09WheatClassifier_CNN_1/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������@2)
'WheatClassifier_CNN_1/dense_2/Tensordot�
4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_cnn_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp�
%WheatClassifier_CNN_1/dense_2/BiasAddBiasAdd0WheatClassifier_CNN_1/dense_2/Tensordot:output:0<WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2'
%WheatClassifier_CNN_1/dense_2/BiasAdd�
(WheatClassifier_CNN_1/dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_CNN_1/dense_2/Gelu/mul/x�
&WheatClassifier_CNN_1/dense_2/Gelu/mulMul1WheatClassifier_CNN_1/dense_2/Gelu/mul/x:output:0.WheatClassifier_CNN_1/dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2(
&WheatClassifier_CNN_1/dense_2/Gelu/mul�
)WheatClassifier_CNN_1/dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2+
)WheatClassifier_CNN_1/dense_2/Gelu/Cast/x�
*WheatClassifier_CNN_1/dense_2/Gelu/truedivRealDiv.WheatClassifier_CNN_1/dense_2/BiasAdd:output:02WheatClassifier_CNN_1/dense_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:����������@2,
*WheatClassifier_CNN_1/dense_2/Gelu/truediv�
&WheatClassifier_CNN_1/dense_2/Gelu/ErfErf.WheatClassifier_CNN_1/dense_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:����������@2(
&WheatClassifier_CNN_1/dense_2/Gelu/Erf�
(WheatClassifier_CNN_1/dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2*
(WheatClassifier_CNN_1/dense_2/Gelu/add/x�
&WheatClassifier_CNN_1/dense_2/Gelu/addAddV21WheatClassifier_CNN_1/dense_2/Gelu/add/x:output:0*WheatClassifier_CNN_1/dense_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:����������@2(
&WheatClassifier_CNN_1/dense_2/Gelu/add�
(WheatClassifier_CNN_1/dense_2/Gelu/mul_1Mul*WheatClassifier_CNN_1/dense_2/Gelu/mul:z:0*WheatClassifier_CNN_1/dense_2/Gelu/add:z:0*
T0*,
_output_shapes
:����������@2*
(WheatClassifier_CNN_1/dense_2/Gelu/mul_1�
WheatClassifier_CNN_1/add_1/addAddV2,WheatClassifier_CNN_1/dense_2/Gelu/mul_1:z:0!WheatClassifier_CNN_1/add/add:z:0*
T0*,
_output_shapes
:����������@2!
WheatClassifier_CNN_1/add_1/add�
JWheatClassifier_CNN_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_CNN_1/layer_normalization_2/moments/mean/reduction_indices�
8WheatClassifier_CNN_1/layer_normalization_2/moments/meanMean#WheatClassifier_CNN_1/add_1/add:z:0SWheatClassifier_CNN_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2:
8WheatClassifier_CNN_1/layer_normalization_2/moments/mean�
@WheatClassifier_CNN_1/layer_normalization_2/moments/StopGradientStopGradientAWheatClassifier_CNN_1/layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:����������2B
@WheatClassifier_CNN_1/layer_normalization_2/moments/StopGradient�
EWheatClassifier_CNN_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifference#WheatClassifier_CNN_1/add_1/add:z:0IWheatClassifier_CNN_1/layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2G
EWheatClassifier_CNN_1/layer_normalization_2/moments/SquaredDifference�
NWheatClassifier_CNN_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN_1/layer_normalization_2/moments/variance/reduction_indices�
<WheatClassifier_CNN_1/layer_normalization_2/moments/varianceMeanIWheatClassifier_CNN_1/layer_normalization_2/moments/SquaredDifference:z:0WWheatClassifier_CNN_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2>
<WheatClassifier_CNN_1/layer_normalization_2/moments/variance�
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *�7�52=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add/y�
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/addAddV2EWheatClassifier_CNN_1/layer_normalization_2/moments/variance:output:0DWheatClassifier_CNN_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:����������2;
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add�
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/RsqrtRsqrt=WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:����������2=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/Rsqrt�
HWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_cnn_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOp�
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mulMul?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/Rsqrt:y:0PWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2;
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul�
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_1Mul#WheatClassifier_CNN_1/add_1/add:z:0=WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_1�
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_2MulAWheatClassifier_CNN_1/layer_normalization_2/moments/mean:output:0=WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_2�
DWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_cnn_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOp�
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/subSubLWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2;
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/sub�
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add_1AddV2?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_1:z:0=WheatClassifier_CNN_1/layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add_1�
#WheatClassifier_CNN_1/reshape/ShapeShape?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2%
#WheatClassifier_CNN_1/reshape/Shape�
1WheatClassifier_CNN_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_CNN_1/reshape/strided_slice/stack�
3WheatClassifier_CNN_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_1/reshape/strided_slice/stack_1�
3WheatClassifier_CNN_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_1/reshape/strided_slice/stack_2�
+WheatClassifier_CNN_1/reshape/strided_sliceStridedSlice,WheatClassifier_CNN_1/reshape/Shape:output:0:WheatClassifier_CNN_1/reshape/strided_slice/stack:output:0<WheatClassifier_CNN_1/reshape/strided_slice/stack_1:output:0<WheatClassifier_CNN_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+WheatClassifier_CNN_1/reshape/strided_slice�
-WheatClassifier_CNN_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-WheatClassifier_CNN_1/reshape/Reshape/shape/1�
-WheatClassifier_CNN_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-WheatClassifier_CNN_1/reshape/Reshape/shape/2�
-WheatClassifier_CNN_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2/
-WheatClassifier_CNN_1/reshape/Reshape/shape/3�
+WheatClassifier_CNN_1/reshape/Reshape/shapePack4WheatClassifier_CNN_1/reshape/strided_slice:output:06WheatClassifier_CNN_1/reshape/Reshape/shape/1:output:06WheatClassifier_CNN_1/reshape/Reshape/shape/2:output:06WheatClassifier_CNN_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+WheatClassifier_CNN_1/reshape/Reshape/shape�
%WheatClassifier_CNN_1/reshape/ReshapeReshape?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add_1:z:04WheatClassifier_CNN_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2'
%WheatClassifier_CNN_1/reshape/Reshape�
2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_1_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype024
2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp�
#WheatClassifier_CNN_1/conv_1/Conv2DConv2D.WheatClassifier_CNN_1/reshape/Reshape:output:0:WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_1/conv_1/Conv2D�
3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp�
$WheatClassifier_CNN_1/conv_1/BiasAddBiasAdd,WheatClassifier_CNN_1/conv_1/Conv2D:output:0;WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
2&
$WheatClassifier_CNN_1/conv_1/BiasAdd�
2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_1_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype024
2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp�
#WheatClassifier_CNN_1/conv_2/Conv2DConv2D-WheatClassifier_CNN_1/conv_1/BiasAdd:output:0:WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_1/conv_2/Conv2D�
3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_1_conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp�
$WheatClassifier_CNN_1/conv_2/BiasAddBiasAdd,WheatClassifier_CNN_1/conv_2/Conv2D:output:0;WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
2&
$WheatClassifier_CNN_1/conv_2/BiasAdd�
'WheatClassifier_CNN_1/maxpool_1/MaxPoolMaxPool-WheatClassifier_CNN_1/conv_2/BiasAdd:output:0*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
2)
'WheatClassifier_CNN_1/maxpool_1/MaxPool�
)WheatClassifier_CNN_1/flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2+
)WheatClassifier_CNN_1/flatten_layer/Const�
+WheatClassifier_CNN_1/flatten_layer/ReshapeReshape0WheatClassifier_CNN_1/maxpool_1/MaxPool:output:02WheatClassifier_CNN_1/flatten_layer/Const:output:0*
T0*(
_output_shapes
:����������2-
+WheatClassifier_CNN_1/flatten_layer/Reshape�
0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_1_fc_1_matmul_readvariableop_resource*
_output_shapes
:	�2*
dtype022
0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp�
!WheatClassifier_CNN_1/FC_1/MatMulMatMul4WheatClassifier_CNN_1/flatten_layer/Reshape:output:08WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22#
!WheatClassifier_CNN_1/FC_1/MatMul�
1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_1_fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp�
"WheatClassifier_CNN_1/FC_1/BiasAddBiasAdd+WheatClassifier_CNN_1/FC_1/MatMul:product:09WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22$
"WheatClassifier_CNN_1/FC_1/BiasAdd�
,WheatClassifier_CNN_1/leaky_ReLu_1/LeakyRelu	LeakyRelu+WheatClassifier_CNN_1/FC_1/BiasAdd:output:0*'
_output_shapes
:���������2*
alpha%���>2.
,WheatClassifier_CNN_1/leaky_ReLu_1/LeakyRelu�
0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_1_fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype022
0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp�
!WheatClassifier_CNN_1/FC_2/MatMulMatMul:WheatClassifier_CNN_1/leaky_ReLu_1/LeakyRelu:activations:08WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22#
!WheatClassifier_CNN_1/FC_2/MatMul�
1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_1_fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp�
"WheatClassifier_CNN_1/FC_2/BiasAddBiasAdd+WheatClassifier_CNN_1/FC_2/MatMul:product:09WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������22$
"WheatClassifier_CNN_1/FC_2/BiasAdd�
,WheatClassifier_CNN_1/leaky_ReLu_2/LeakyRelu	LeakyRelu+WheatClassifier_CNN_1/FC_2/BiasAdd:output:0*'
_output_shapes
:���������2*
alpha%���>2.
,WheatClassifier_CNN_1/leaky_ReLu_2/LeakyRelu�
8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOpReadVariableOpAwheatclassifier_cnn_1_output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02:
8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOp�
)WheatClassifier_CNN_1/output_layer/MatMulMatMul:WheatClassifier_CNN_1/leaky_ReLu_2/LeakyRelu:activations:0@WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2+
)WheatClassifier_CNN_1/output_layer/MatMul�
9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOpReadVariableOpBwheatclassifier_cnn_1_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp�
*WheatClassifier_CNN_1/output_layer/BiasAddBiasAdd3WheatClassifier_CNN_1/output_layer/MatMul:product:0AWheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2,
*WheatClassifier_CNN_1/output_layer/BiasAdd�
*WheatClassifier_CNN_1/output_layer/SoftmaxSoftmax3WheatClassifier_CNN_1/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2,
*WheatClassifier_CNN_1/output_layer/Softmax�
IdentityIdentity4WheatClassifier_CNN_1/output_layer/Softmax:softmax:02^WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp2^WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp4^WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp4^WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp5^WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp7^WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp5^WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp7^WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOpC^WheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOpG^WheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOpE^WheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOpI^WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOpE^WheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOpI^WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOpO^WheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOpY^WheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpB^WheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOpL^WheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOpD^WheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOpN^WheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOpD^WheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOpN^WheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOp:^WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp9^WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOpA^WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOpC^WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOp?^WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp2d
0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp2f
1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp2d
0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp2j
3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp2l
4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp2l
4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOp6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOp2�
BWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOpBWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOp2�
FWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOpFWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOp2�
DWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOpDWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOp2�
HWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOpHWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOp2�
DWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOpDWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOp2�
HWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOpHWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2�
NWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOpNWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOp2�
XWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpXWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2�
AWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOpAWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOp2�
KWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOpKWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOp2�
CWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOpCWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOp2�
MWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOpMWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOp2�
CWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOpCWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOp2�
MWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOpMWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOp2v
9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp2t
8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOp8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOp2�
@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOp@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOp2�
BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOpBWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOp2�
>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�
P
$__inference_add_layer_call_fn_302110
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3001162
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������@:����������@:V R
,
_output_shapes
:����������@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������@
"
_user_specified_name
inputs/1
�	
�
@__inference_FC_2_layer_call_and_return_conditional_losses_300360

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
�

�
5__inference_multi_head_attention_layer_call_fn_302098	
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
 *,
_output_shapes
:����������@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3006522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������@:����������@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:����������@

_user_specified_namequery:SO
,
_output_shapes
:����������@

_user_specified_namevalue
�.
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_300092	
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
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:����������@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:�����������*
equationaecd,abcd->acbe2
einsum/Einsum�
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:�����������2
softmax/Softmax�
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:�����������2
dropout/Identity�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:����������@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������@:����������@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:����������@

_user_specified_namequery:SO
,
_output_shapes
:����������@

_user_specified_namevalue
��
�
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301485

inputsH
5patch_encoder_dense_tensordot_readvariableop_resource:	�@A
3patch_encoder_dense_biasadd_readvariableop_resource:@B
/patch_encoder_embedding_embedding_lookup_301278:	�@G
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
7layer_normalization_2_batchnorm_readvariableop_resource:@?
%conv_1_conv2d_readvariableop_resource:@
4
&conv_1_biasadd_readvariableop_resource:
?
%conv_2_conv2d_readvariableop_resource:

4
&conv_2_biasadd_readvariableop_resource:
6
#fc_1_matmul_readvariableop_resource:	�22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identity��FC_1/BiasAdd/ReadVariableOp�FC_1/MatMul/ReadVariableOp�FC_2/BiasAdd/ReadVariableOp�FC_2/MatMul/ReadVariableOp�conv_1/BiasAdd/ReadVariableOp�conv_1/Conv2D/ReadVariableOp�conv_2/BiasAdd/ReadVariableOp�conv_2/Conv2D/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp� dense_1/Tensordot/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp� dense_2/Tensordot/ReadVariableOp�,layer_normalization/batchnorm/ReadVariableOp�0layer_normalization/batchnorm/mul/ReadVariableOp�.layer_normalization_1/batchnorm/ReadVariableOp�2layer_normalization_1/batchnorm/mul/ReadVariableOp�.layer_normalization_2/batchnorm/ReadVariableOp�2layer_normalization_2/batchnorm/mul/ReadVariableOp�8multi_head_attention/attention_output/add/ReadVariableOp�Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�+multi_head_attention/key/add/ReadVariableOp�5multi_head_attention/key/einsum/Einsum/ReadVariableOp�-multi_head_attention/query/add/ReadVariableOp�7multi_head_attention/query/einsum/Einsum/ReadVariableOp�-multi_head_attention/value/add/ReadVariableOp�7multi_head_attention/value/einsum/Einsum/ReadVariableOp�#output_layer/BiasAdd/ReadVariableOp�"output_layer/MatMul/ReadVariableOp�*patch_encoder/dense/BiasAdd/ReadVariableOp�,patch_encoder/dense/Tensordot/ReadVariableOp�(patch_encoder/embedding/embedding_lookupT
patches/ShapeShapeinputs*
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
patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*0
_output_shapes
:����������*
ksizes


*
paddingVALID*
rates
*
strides


2
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
B :�2
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
!:�������������������2
patches/Reshapex
patch_encoder/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
patch_encoder/range/starty
patch_encoder/range/limitConst*
_output_shapes
: *
dtype0*
value
B :�2
patch_encoder/range/limitx
patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder/range/delta�
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes	
:�2
patch_encoder/range�
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
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
!:�������������������2)
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
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_301278patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/301278*
_output_shapes
:	�@*
dtype02*
(patch_encoder/embedding/embedding_lookup�
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/301278*
_output_shapes
:	�@23
1patch_encoder/embedding/embedding_lookup/Identity�
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	�@25
3patch_encoder/embedding/embedding_lookup/Identity_1�
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������@2
patch_encoder/add�
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indices�
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2"
 layer_normalization/moments/mean�
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:����������2*
(layer_normalization/moments/StopGradient�
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2/
-layer_normalization/moments/SquaredDifference�
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indices�
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2#
!layer_normalization/batchnorm/add�
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:����������2%
#layer_normalization/batchnorm/Rsqrt�
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOp�
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2#
!layer_normalization/batchnorm/mul�
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization/batchnorm/mul_1�
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization/batchnorm/mul_2�
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOp�
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2#
!layer_normalization/batchnorm/sub�
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization/batchnorm/add_1�
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp�
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum�
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOp�
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2 
multi_head_attention/query/add�
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp�
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum�
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOp�
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
multi_head_attention/key/add�
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp�
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum�
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOp�
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/y�
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:����������@2
multi_head_attention/Mul�
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:�����������*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsum�
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:�����������2&
$multi_head_attention/softmax/Softmax�
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:�����������2'
%multi_head_attention/dropout/Identity�
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:����������@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsum�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp�
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsum�
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOp�
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2+
)multi_head_attention/attention_output/add�
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*,
_output_shapes
:����������@2	
add/add�
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indices�
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2$
"layer_normalization_1/moments/mean�
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:����������2,
*layer_normalization_1/moments/StopGradient�
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@21
/layer_normalization_1/moments/SquaredDifference�
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indices�
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2%
#layer_normalization_1/batchnorm/add�
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:����������2'
%layer_normalization_1/batchnorm/Rsqrt�
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOp�
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization_1/batchnorm/mul�
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_1/batchnorm/mul_1�
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_1/batchnorm/mul_2�
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOp�
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization_1/batchnorm/sub�
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2'
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
T0*,
_output_shapes
:����������@2
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
T0*-
_output_shapes
:�����������2
dense_1/Tensordot�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/x�
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*-
_output_shapes
:�����������2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_1/Gelu/Cast/x�
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:�����������2
dense_1/Gelu/truediv}
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:�����������2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_1/Gelu/add/x�
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:�����������2
dense_1/Gelu/add�
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*-
_output_shapes
:�����������2
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
T0*-
_output_shapes
:�����������2
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
T0*,
_output_shapes
:����������@2
dense_2/Tensordot�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/x�
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
dense_2/Gelu/Cast/x�
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/truediv|
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dense_2/Gelu/add/x�
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/add�
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*,
_output_shapes
:����������@2
dense_2/Gelu/mul_1{
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*,
_output_shapes
:����������@2
	add_1/add�
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indices�
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
	keep_dims(2$
"layer_normalization_2/moments/mean�
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:����������2,
*layer_normalization_2/moments/StopGradient�
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:����������@21
/layer_normalization_2/moments/SquaredDifference�
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indices�
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2%
#layer_normalization_2/batchnorm/add�
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:����������2'
%layer_normalization_2/batchnorm/Rsqrt�
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOp�
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization_2/batchnorm/mul�
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_2/batchnorm/mul_1�
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_2/batchnorm/mul_2�
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOp�
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2%
#layer_normalization_2/batchnorm/sub�
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2'
%layer_normalization_2/batchnorm/add_1w
reshape/ShapeShape)layer_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
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
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/3�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:���������@2
reshape/Reshape�
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype02
conv_1/Conv2D/ReadVariableOp�
conv_1/Conv2DConv2Dreshape/Reshape:output:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
2
conv_2/BiasAdd�
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPool{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
flatten_layer/Const�
flatten_layer/ReshapeReshapemaxpool_1/MaxPool:output:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_layer/Reshape�
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	�2*
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
output_layer/Softmax�
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
FC_1/BiasAdd/ReadVariableOpFC_1/BiasAdd/ReadVariableOp28
FC_1/MatMul/ReadVariableOpFC_1/MatMul/ReadVariableOp2:
FC_2/BiasAdd/ReadVariableOpFC_2/BiasAdd/ReadVariableOp28
FC_2/MatMul/ReadVariableOpFC_2/MatMul/ReadVariableOp2>
conv_1/BiasAdd/ReadVariableOpconv_1/BiasAdd/ReadVariableOp2<
conv_1/Conv2D/ReadVariableOpconv_1/Conv2D/ReadVariableOp2>
conv_2/BiasAdd/ReadVariableOpconv_2/BiasAdd/ReadVariableOp2<
conv_2/Conv2D/ReadVariableOpconv_2/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2\
,layer_normalization/batchnorm/ReadVariableOp,layer_normalization/batchnorm/ReadVariableOp2d
0layer_normalization/batchnorm/mul/ReadVariableOp0layer_normalization/batchnorm/mul/ReadVariableOp2`
.layer_normalization_1/batchnorm/ReadVariableOp.layer_normalization_1/batchnorm/ReadVariableOp2h
2layer_normalization_1/batchnorm/mul/ReadVariableOp2layer_normalization_1/batchnorm/mul/ReadVariableOp2`
.layer_normalization_2/batchnorm/ReadVariableOp.layer_normalization_2/batchnorm/ReadVariableOp2h
2layer_normalization_2/batchnorm/mul/ReadVariableOp2layer_normalization_2/batchnorm/mul/ReadVariableOp2t
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2�
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp2X
*patch_encoder/dense/BiasAdd/ReadVariableOp*patch_encoder/dense/BiasAdd/ReadVariableOp2\
,patch_encoder/dense/Tensordot/ReadVariableOp,patch_encoder/dense/Tensordot/ReadVariableOp2T
(patch_encoder/embedding/embedding_lookup(patch_encoder/embedding/embedding_lookup:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
d
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_302399

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
�
�
%__inference_FC_2_layer_call_fn_302394

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
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3003602
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
�
�
-__inference_output_layer_layer_call_fn_302424

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
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3003842
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
��
�F
"__inference__traced_restore_303075
file_prefix8
*assignvariableop_layer_normalization_gamma:@9
+assignvariableop_1_layer_normalization_beta:@<
.assignvariableop_2_layer_normalization_1_gamma:@;
-assignvariableop_3_layer_normalization_1_beta:@4
!assignvariableop_4_dense_1_kernel:	@�.
assignvariableop_5_dense_1_bias:	�4
!assignvariableop_6_dense_2_kernel:	�@-
assignvariableop_7_dense_2_bias:@<
.assignvariableop_8_layer_normalization_2_gamma:@;
-assignvariableop_9_layer_normalization_2_beta:@;
!assignvariableop_10_conv_1_kernel:@
-
assignvariableop_11_conv_1_bias:
;
!assignvariableop_12_conv_2_kernel:

-
assignvariableop_13_conv_2_bias:
2
assignvariableop_14_fc_1_kernel:	�2+
assignvariableop_15_fc_1_bias:21
assignvariableop_16_fc_2_kernel:22+
assignvariableop_17_fc_2_bias:29
'assignvariableop_18_output_layer_kernel:23
%assignvariableop_19_output_layer_bias:(
assignvariableop_20_adamw_iter:	 *
 assignvariableop_21_adamw_beta_1: *
 assignvariableop_22_adamw_beta_2: )
assignvariableop_23_adamw_decay: 1
'assignvariableop_24_adamw_learning_rate: 0
&assignvariableop_25_adamw_weight_decay: A
.assignvariableop_26_patch_encoder_dense_kernel:	�@:
,assignvariableop_27_patch_encoder_dense_bias:@I
6assignvariableop_28_patch_encoder_embedding_embeddings:	�@K
5assignvariableop_29_multi_head_attention_query_kernel:@@E
3assignvariableop_30_multi_head_attention_query_bias:@I
3assignvariableop_31_multi_head_attention_key_kernel:@@C
1assignvariableop_32_multi_head_attention_key_bias:@K
5assignvariableop_33_multi_head_attention_value_kernel:@@E
3assignvariableop_34_multi_head_attention_value_bias:@V
@assignvariableop_35_multi_head_attention_attention_output_kernel:@@L
>assignvariableop_36_multi_head_attention_attention_output_bias:@#
assignvariableop_37_total: #
assignvariableop_38_count: %
assignvariableop_39_total_1: %
assignvariableop_40_count_1: C
5assignvariableop_41_adamw_layer_normalization_gamma_m:@B
4assignvariableop_42_adamw_layer_normalization_beta_m:@E
7assignvariableop_43_adamw_layer_normalization_1_gamma_m:@D
6assignvariableop_44_adamw_layer_normalization_1_beta_m:@=
*assignvariableop_45_adamw_dense_1_kernel_m:	@�7
(assignvariableop_46_adamw_dense_1_bias_m:	�=
*assignvariableop_47_adamw_dense_2_kernel_m:	�@6
(assignvariableop_48_adamw_dense_2_bias_m:@E
7assignvariableop_49_adamw_layer_normalization_2_gamma_m:@D
6assignvariableop_50_adamw_layer_normalization_2_beta_m:@C
)assignvariableop_51_adamw_conv_1_kernel_m:@
5
'assignvariableop_52_adamw_conv_1_bias_m:
C
)assignvariableop_53_adamw_conv_2_kernel_m:

5
'assignvariableop_54_adamw_conv_2_bias_m:
:
'assignvariableop_55_adamw_fc_1_kernel_m:	�23
%assignvariableop_56_adamw_fc_1_bias_m:29
'assignvariableop_57_adamw_fc_2_kernel_m:223
%assignvariableop_58_adamw_fc_2_bias_m:2A
/assignvariableop_59_adamw_output_layer_kernel_m:2;
-assignvariableop_60_adamw_output_layer_bias_m:I
6assignvariableop_61_adamw_patch_encoder_dense_kernel_m:	�@B
4assignvariableop_62_adamw_patch_encoder_dense_bias_m:@Q
>assignvariableop_63_adamw_patch_encoder_embedding_embeddings_m:	�@S
=assignvariableop_64_adamw_multi_head_attention_query_kernel_m:@@M
;assignvariableop_65_adamw_multi_head_attention_query_bias_m:@Q
;assignvariableop_66_adamw_multi_head_attention_key_kernel_m:@@K
9assignvariableop_67_adamw_multi_head_attention_key_bias_m:@S
=assignvariableop_68_adamw_multi_head_attention_value_kernel_m:@@M
;assignvariableop_69_adamw_multi_head_attention_value_bias_m:@^
Hassignvariableop_70_adamw_multi_head_attention_attention_output_kernel_m:@@T
Fassignvariableop_71_adamw_multi_head_attention_attention_output_bias_m:@C
5assignvariableop_72_adamw_layer_normalization_gamma_v:@B
4assignvariableop_73_adamw_layer_normalization_beta_v:@E
7assignvariableop_74_adamw_layer_normalization_1_gamma_v:@D
6assignvariableop_75_adamw_layer_normalization_1_beta_v:@=
*assignvariableop_76_adamw_dense_1_kernel_v:	@�7
(assignvariableop_77_adamw_dense_1_bias_v:	�=
*assignvariableop_78_adamw_dense_2_kernel_v:	�@6
(assignvariableop_79_adamw_dense_2_bias_v:@E
7assignvariableop_80_adamw_layer_normalization_2_gamma_v:@D
6assignvariableop_81_adamw_layer_normalization_2_beta_v:@C
)assignvariableop_82_adamw_conv_1_kernel_v:@
5
'assignvariableop_83_adamw_conv_1_bias_v:
C
)assignvariableop_84_adamw_conv_2_kernel_v:

5
'assignvariableop_85_adamw_conv_2_bias_v:
:
'assignvariableop_86_adamw_fc_1_kernel_v:	�23
%assignvariableop_87_adamw_fc_1_bias_v:29
'assignvariableop_88_adamw_fc_2_kernel_v:223
%assignvariableop_89_adamw_fc_2_bias_v:2A
/assignvariableop_90_adamw_output_layer_kernel_v:2;
-assignvariableop_91_adamw_output_layer_bias_v:I
6assignvariableop_92_adamw_patch_encoder_dense_kernel_v:	�@B
4assignvariableop_93_adamw_patch_encoder_dense_bias_v:@Q
>assignvariableop_94_adamw_patch_encoder_embedding_embeddings_v:	�@S
=assignvariableop_95_adamw_multi_head_attention_query_kernel_v:@@M
;assignvariableop_96_adamw_multi_head_attention_query_bias_v:@Q
;assignvariableop_97_adamw_multi_head_attention_key_kernel_v:@@K
9assignvariableop_98_adamw_multi_head_attention_key_bias_v:@S
=assignvariableop_99_adamw_multi_head_attention_value_kernel_v:@@N
<assignvariableop_100_adamw_multi_head_attention_value_bias_v:@_
Iassignvariableop_101_adamw_multi_head_attention_attention_output_kernel_v:@@U
Gassignvariableop_102_adamw_multi_head_attention_attention_output_bias_v:@
identity_104��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�6
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*�6
value�5B�5hB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*�
value�B�hB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*v
dtypesl
j2h	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_layer_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp-assignvariableop_3_layer_normalization_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp.assignvariableop_8_layer_normalization_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp-assignvariableop_9_layer_normalization_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_conv_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp!assignvariableop_12_conv_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOpassignvariableop_13_conv_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_fc_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_fc_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_fc_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_fc_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_output_layer_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp%assignvariableop_19_output_layer_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_adamw_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp assignvariableop_21_adamw_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp assignvariableop_22_adamw_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_adamw_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adamw_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adamw_weight_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp.assignvariableop_26_patch_encoder_dense_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp,assignvariableop_27_patch_encoder_dense_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp6assignvariableop_28_patch_encoder_embedding_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp5assignvariableop_29_multi_head_attention_query_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp3assignvariableop_30_multi_head_attention_query_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp3assignvariableop_31_multi_head_attention_key_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp1assignvariableop_32_multi_head_attention_key_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp5assignvariableop_33_multi_head_attention_value_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp3assignvariableop_34_multi_head_attention_value_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp@assignvariableop_35_multi_head_attention_attention_output_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp>assignvariableop_36_multi_head_attention_attention_output_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp5assignvariableop_41_adamw_layer_normalization_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adamw_layer_normalization_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adamw_layer_normalization_1_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adamw_layer_normalization_1_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adamw_dense_1_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adamw_dense_1_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adamw_dense_2_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adamw_dense_2_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adamw_layer_normalization_2_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adamw_layer_normalization_2_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp)assignvariableop_51_adamw_conv_1_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp'assignvariableop_52_adamw_conv_1_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adamw_conv_2_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adamw_conv_2_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adamw_fc_1_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp%assignvariableop_56_adamw_fc_1_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adamw_fc_2_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp%assignvariableop_58_adamw_fc_2_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp/assignvariableop_59_adamw_output_layer_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp-assignvariableop_60_adamw_output_layer_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adamw_patch_encoder_dense_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adamw_patch_encoder_dense_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp>assignvariableop_63_adamw_patch_encoder_embedding_embeddings_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64�
AssignVariableOp_64AssignVariableOp=assignvariableop_64_adamw_multi_head_attention_query_kernel_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65�
AssignVariableOp_65AssignVariableOp;assignvariableop_65_adamw_multi_head_attention_query_bias_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66�
AssignVariableOp_66AssignVariableOp;assignvariableop_66_adamw_multi_head_attention_key_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67�
AssignVariableOp_67AssignVariableOp9assignvariableop_67_adamw_multi_head_attention_key_bias_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68�
AssignVariableOp_68AssignVariableOp=assignvariableop_68_adamw_multi_head_attention_value_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69�
AssignVariableOp_69AssignVariableOp;assignvariableop_69_adamw_multi_head_attention_value_bias_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70�
AssignVariableOp_70AssignVariableOpHassignvariableop_70_adamw_multi_head_attention_attention_output_kernel_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71�
AssignVariableOp_71AssignVariableOpFassignvariableop_71_adamw_multi_head_attention_attention_output_bias_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72�
AssignVariableOp_72AssignVariableOp5assignvariableop_72_adamw_layer_normalization_gamma_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73�
AssignVariableOp_73AssignVariableOp4assignvariableop_73_adamw_layer_normalization_beta_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74�
AssignVariableOp_74AssignVariableOp7assignvariableop_74_adamw_layer_normalization_1_gamma_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75�
AssignVariableOp_75AssignVariableOp6assignvariableop_75_adamw_layer_normalization_1_beta_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76�
AssignVariableOp_76AssignVariableOp*assignvariableop_76_adamw_dense_1_kernel_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77�
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adamw_dense_1_bias_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78�
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adamw_dense_2_kernel_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79�
AssignVariableOp_79AssignVariableOp(assignvariableop_79_adamw_dense_2_bias_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80�
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adamw_layer_normalization_2_gamma_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81�
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adamw_layer_normalization_2_beta_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82�
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adamw_conv_1_kernel_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83�
AssignVariableOp_83AssignVariableOp'assignvariableop_83_adamw_conv_1_bias_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84�
AssignVariableOp_84AssignVariableOp)assignvariableop_84_adamw_conv_2_kernel_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85�
AssignVariableOp_85AssignVariableOp'assignvariableop_85_adamw_conv_2_bias_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86�
AssignVariableOp_86AssignVariableOp'assignvariableop_86_adamw_fc_1_kernel_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87�
AssignVariableOp_87AssignVariableOp%assignvariableop_87_adamw_fc_1_bias_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88�
AssignVariableOp_88AssignVariableOp'assignvariableop_88_adamw_fc_2_kernel_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89�
AssignVariableOp_89AssignVariableOp%assignvariableop_89_adamw_fc_2_bias_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90�
AssignVariableOp_90AssignVariableOp/assignvariableop_90_adamw_output_layer_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91�
AssignVariableOp_91AssignVariableOp-assignvariableop_91_adamw_output_layer_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92�
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adamw_patch_encoder_dense_kernel_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93�
AssignVariableOp_93AssignVariableOp4assignvariableop_93_adamw_patch_encoder_dense_bias_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94�
AssignVariableOp_94AssignVariableOp>assignvariableop_94_adamw_patch_encoder_embedding_embeddings_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95�
AssignVariableOp_95AssignVariableOp=assignvariableop_95_adamw_multi_head_attention_query_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96�
AssignVariableOp_96AssignVariableOp;assignvariableop_96_adamw_multi_head_attention_query_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97�
AssignVariableOp_97AssignVariableOp;assignvariableop_97_adamw_multi_head_attention_key_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98�
AssignVariableOp_98AssignVariableOp9assignvariableop_98_adamw_multi_head_attention_key_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99�
AssignVariableOp_99AssignVariableOp=assignvariableop_99_adamw_multi_head_attention_value_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100�
AssignVariableOp_100AssignVariableOp<assignvariableop_100_adamw_multi_head_attention_value_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101�
AssignVariableOp_101AssignVariableOpIassignvariableop_101_adamw_multi_head_attention_attention_output_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102�
AssignVariableOp_102AssignVariableOpGassignvariableop_102_adamw_multi_head_attention_attention_output_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1029
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_103Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_103�
Identity_104IdentityIdentity_103:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_104"%
identity_104Identity_104:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022*
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
�b
�
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301158
input_layer'
patch_encoder_301076:	�@"
patch_encoder_301078:@'
patch_encoder_301080:	�@(
layer_normalization_301083:@(
layer_normalization_301085:@1
multi_head_attention_301088:@@-
multi_head_attention_301090:@1
multi_head_attention_301092:@@-
multi_head_attention_301094:@1
multi_head_attention_301096:@@-
multi_head_attention_301098:@1
multi_head_attention_301100:@@)
multi_head_attention_301102:@*
layer_normalization_1_301106:@*
layer_normalization_1_301108:@!
dense_1_301111:	@�
dense_1_301113:	�!
dense_2_301116:	�@
dense_2_301118:@*
layer_normalization_2_301122:@*
layer_normalization_2_301124:@'
conv_1_301128:@

conv_1_301130:
'
conv_2_301133:


conv_2_301135:

fc_1_301140:	�2
fc_1_301142:2
fc_2_301146:22
fc_2_301148:2%
output_layer_301152:2!
output_layer_301154:
identity��FC_1/StatefulPartitionedCall�FC_2/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�%patch_encoder/StatefulPartitionedCall�
patches/PartitionedCallPartitionedCallinput_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_2999792
patches/PartitionedCall�
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_301076patch_encoder_301078patch_encoder_301080*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3000212'
%patch_encoder/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_301083layer_normalization_301085*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3000512-
+layer_normalization/StatefulPartitionedCall�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_301088multi_head_attention_301090multi_head_attention_301092multi_head_attention_301094multi_head_attention_301096multi_head_attention_301098multi_head_attention_301100multi_head_attention_301102*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3006522.
,multi_head_attention/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3001162
add/PartitionedCall�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_301106layer_normalization_1_301108*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3001402/
-layer_normalization_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_301111dense_1_301113*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3001842!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_301116dense_2_301118*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3002282!
dense_2/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3002402
add_1/PartitionedCall�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_301122layer_normalization_2_301124*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3002642/
-layer_normalization_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3002842
reshape/PartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv_1_301128conv_1_301130*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3002962 
conv_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_301133conv_2_301135*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3003122 
conv_2/StatefulPartitionedCall�
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_2999522
maxpool_1/PartitionedCall�
flatten_layer/PartitionedCallPartitionedCall"maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3003252
flatten_layer/PartitionedCall�
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_301140fc_1_301142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3003372
FC_1/StatefulPartitionedCall�
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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3003482
leaky_ReLu_1/PartitionedCall�
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_301146fc_2_301148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3003602
FC_2/StatefulPartitionedCall�
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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3003712
leaky_ReLu_2/PartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_301152output_layer_301154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3003842&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�7
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_302054	
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
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:����������@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:�����������*
equationaecd,abcd->acbe2
einsum/Einsum�
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:�����������2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/dropout/Const�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:�����������2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:�����������*
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
T0*1
_output_shapes
:�����������2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:�����������2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:�����������2
dropout/dropout/Mul_1�
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:����������@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������@:����������@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:����������@

_user_specified_namequery:SO
,
_output_shapes
:����������@

_user_specified_namevalue
�
e
I__inference_flatten_layer_layer_call_and_return_conditional_losses_302341

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_302235

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
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3002282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
B__inference_conv_2_layer_call_and_return_conditional_losses_300312

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
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
@__inference_FC_1_layer_call_and_return_conditional_losses_300337

inputs1
matmul_readvariableop_resource:	�2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�2*
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
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�b
�
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_300854

inputs'
patch_encoder_300772:	�@"
patch_encoder_300774:@'
patch_encoder_300776:	�@(
layer_normalization_300779:@(
layer_normalization_300781:@1
multi_head_attention_300784:@@-
multi_head_attention_300786:@1
multi_head_attention_300788:@@-
multi_head_attention_300790:@1
multi_head_attention_300792:@@-
multi_head_attention_300794:@1
multi_head_attention_300796:@@)
multi_head_attention_300798:@*
layer_normalization_1_300802:@*
layer_normalization_1_300804:@!
dense_1_300807:	@�
dense_1_300809:	�!
dense_2_300812:	�@
dense_2_300814:@*
layer_normalization_2_300818:@*
layer_normalization_2_300820:@'
conv_1_300824:@

conv_1_300826:
'
conv_2_300829:


conv_2_300831:

fc_1_300836:	�2
fc_1_300838:2
fc_2_300842:22
fc_2_300844:2%
output_layer_300848:2!
output_layer_300850:
identity��FC_1/StatefulPartitionedCall�FC_2/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�%patch_encoder/StatefulPartitionedCall�
patches/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_2999792
patches/PartitionedCall�
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_300772patch_encoder_300774patch_encoder_300776*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3000212'
%patch_encoder/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_300779layer_normalization_300781*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3000512-
+layer_normalization/StatefulPartitionedCall�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_300784multi_head_attention_300786multi_head_attention_300788multi_head_attention_300790multi_head_attention_300792multi_head_attention_300794multi_head_attention_300796multi_head_attention_300798*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3006522.
,multi_head_attention/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3001162
add/PartitionedCall�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_300802layer_normalization_1_300804*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3001402/
-layer_normalization_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_300807dense_1_300809*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3001842!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_300812dense_2_300814*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3002282!
dense_2/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3002402
add_1/PartitionedCall�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_300818layer_normalization_2_300820*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3002642/
-layer_normalization_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3002842
reshape/PartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv_1_300824conv_1_300826*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3002962 
conv_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_300829conv_2_300831*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3003122 
conv_2/StatefulPartitionedCall�
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_2999522
maxpool_1/PartitionedCall�
flatten_layer/PartitionedCallPartitionedCall"maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3003252
flatten_layer/PartitionedCall�
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_300836fc_1_300838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3003372
FC_1/StatefulPartitionedCall�
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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3003482
leaky_ReLu_1/PartitionedCall�
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_300842fc_2_300844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3003602
FC_2/StatefulPartitionedCall�
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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3003712
leaky_ReLu_2/PartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_300848output_layer_300850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3003842&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_300140

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
T0*,
_output_shapes
:����������*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:����������2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:����������2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
_
C__inference_patches_layer_call_and_return_conditional_losses_299979

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
:����������*
ksizes


*
paddingVALID*
rates
*
strides


2
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
B :�2
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
!:�������������������2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameimages
�
_
C__inference_reshape_layer_call_and_return_conditional_losses_300284

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
_
C__inference_reshape_layer_call_and_return_conditional_losses_302292

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
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
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�7
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_300652	
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
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:����������@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:�����������*
equationaecd,abcd->acbe2
einsum/Einsum�
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:�����������2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�8�?2
dropout/dropout/Const�
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:�����������2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:�����������*
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
T0*1
_output_shapes
:�����������2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:�����������2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:�����������2
dropout/dropout/Mul_1�
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:����������@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������@:����������@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:����������@

_user_specified_namequery:SO
,
_output_shapes
:����������@

_user_specified_namevalue
�
m
A__inference_add_1_layer_call_and_return_conditional_losses_302241
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:����������@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������@:����������@:V R
,
_output_shapes
:����������@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������@
"
_user_specified_name
inputs/1
�
�
'__inference_conv_2_layer_call_fn_302335

inputs!
unknown:


	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3003122
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�
F
*__inference_maxpool_1_layer_call_fn_299958

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
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_2999522
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
%__inference_FC_1_layer_call_fn_302365

inputs
unknown:	�2
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
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3003372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
_
C__inference_patches_layer_call_and_return_conditional_losses_301890

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
:����������*
ksizes


*
paddingVALID*
rates
*
strides


2
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
B :�2
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
!:�������������������2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameimages
�
�
4__inference_layer_normalization_layer_call_fn_301977

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
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3000512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
'__inference_conv_1_layer_call_fn_302316

inputs!
unknown:@

	unknown_0:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3002962
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�.
�
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_302012	
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
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
query/einsum/Einsum�
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOp�
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
	query/add�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp�
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
key/einsum/Einsum�
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOp�
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2	
key/add�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:����������@*
equationabc,cde->abde2
value/einsum/Einsum�
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOp�
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������@2
	value/addS
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
Mul/yk
MulMulquery/add:z:0Mul/y:output:0*
T0*0
_output_shapes
:����������@2
Mul�
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:�����������*
equationaecd,abcd->acbe2
einsum/Einsum�
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:�����������2
softmax/Softmax�
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:�����������2
dropout/Identity�
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:����������@*
equationacbe,aecd->abcd2
einsum_1/Einsum�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOp�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:����������@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp�
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
attention_output/add�
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������@:����������@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:S O
,
_output_shapes
:����������@

_user_specified_namequery:SO
,
_output_shapes
:����������@

_user_specified_namevalue
�

�
5__inference_multi_head_attention_layer_call_fn_302076	
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
 *,
_output_shapes
:����������@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3000922
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������@:����������@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:����������@

_user_specified_namequery:SO
,
_output_shapes
:����������@

_user_specified_namevalue
�
I
-__inference_leaky_ReLu_2_layer_call_fn_302404

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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3003712
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
�
d
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_302370

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
�
d
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_300348

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
�
d
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_300371

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
�
�
6__inference_WheatClassifier_CNN_1_layer_call_fn_301809

inputs
unknown:	�@
	unknown_0:@
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:@@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:	@�

unknown_15:	�

unknown_16:	�@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@


unknown_21:
$

unknown_22:



unknown_23:


unknown_24:	�2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identity��StatefulPartitionedCall�
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
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*A
_read_only_resource_inputs#
!	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_3003912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
I
-__inference_leaky_ReLu_1_layer_call_fn_302375

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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3003482
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
�
D
(__inference_reshape_layer_call_fn_302297

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3002842
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_300264

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
T0*,
_output_shapes
:����������*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:����������2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:����������2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�b
�
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_300391

inputs'
patch_encoder_300022:	�@"
patch_encoder_300024:@'
patch_encoder_300026:	�@(
layer_normalization_300052:@(
layer_normalization_300054:@1
multi_head_attention_300093:@@-
multi_head_attention_300095:@1
multi_head_attention_300097:@@-
multi_head_attention_300099:@1
multi_head_attention_300101:@@-
multi_head_attention_300103:@1
multi_head_attention_300105:@@)
multi_head_attention_300107:@*
layer_normalization_1_300141:@*
layer_normalization_1_300143:@!
dense_1_300185:	@�
dense_1_300187:	�!
dense_2_300229:	�@
dense_2_300231:@*
layer_normalization_2_300265:@*
layer_normalization_2_300267:@'
conv_1_300297:@

conv_1_300299:
'
conv_2_300313:


conv_2_300315:

fc_1_300338:	�2
fc_1_300340:2
fc_2_300361:22
fc_2_300363:2%
output_layer_300385:2!
output_layer_300387:
identity��FC_1/StatefulPartitionedCall�FC_2/StatefulPartitionedCall�conv_1/StatefulPartitionedCall�conv_2/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�+layer_normalization/StatefulPartitionedCall�-layer_normalization_1/StatefulPartitionedCall�-layer_normalization_2/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall�$output_layer/StatefulPartitionedCall�%patch_encoder/StatefulPartitionedCall�
patches/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_2999792
patches/PartitionedCall�
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_300022patch_encoder_300024patch_encoder_300026*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3000212'
%patch_encoder/StatefulPartitionedCall�
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_300052layer_normalization_300054*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3000512-
+layer_normalization/StatefulPartitionedCall�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_300093multi_head_attention_300095multi_head_attention_300097multi_head_attention_300099multi_head_attention_300101multi_head_attention_300103multi_head_attention_300105multi_head_attention_300107*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3000922.
,multi_head_attention/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3001162
add/PartitionedCall�
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_300141layer_normalization_1_300143*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3001402/
-layer_normalization_1/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_300185dense_1_300187*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3001842!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_300229dense_2_300231*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3002282!
dense_2/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3002402
add_1/PartitionedCall�
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_300265layer_normalization_2_300267*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3002642/
-layer_normalization_2/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_3002842
reshape/PartitionedCall�
conv_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv_1_300297conv_1_300299*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3002962 
conv_1/StatefulPartitionedCall�
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_300313conv_2_300315*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3003122 
conv_2/StatefulPartitionedCall�
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_2999522
maxpool_1/PartitionedCall�
flatten_layer/PartitionedCallPartitionedCall"maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3003252
flatten_layer/PartitionedCall�
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_300338fc_1_300340*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3003372
FC_1/StatefulPartitionedCall�
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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3003482
leaky_ReLu_1/PartitionedCall�
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_300361fc_2_300363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3003602
FC_2/StatefulPartitionedCall�
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
 *-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3003712
leaky_ReLu_2/PartitionedCall�
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_300385output_layer_300387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3003842&
$output_layer/StatefulPartitionedCall�
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
i
?__inference_add_layer_call_and_return_conditional_losses_300116

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:����������@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������@:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�'
�
C__inference_dense_2_layer_call_and_return_conditional_losses_302226

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
T0*-
_output_shapes
:�����������2
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
T0*,
_output_shapes
:����������@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
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
:����������@2

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
:����������@2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:����������@2

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
:����������@2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:����������@2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�'
�
C__inference_dense_2_layer_call_and_return_conditional_losses_300228

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
T0*-
_output_shapes
:�����������2
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
T0*,
_output_shapes
:����������@2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2	
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
:����������@2

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
:����������@2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:����������@2

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
:����������@2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:����������@2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:�����������
 
_user_specified_nameinputs
�
k
A__inference_add_1_layer_call_and_return_conditional_losses_300240

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:����������@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������@:����������@:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs:TP
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_302132

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
T0*,
_output_shapes
:����������*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:����������2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:����������2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
.__inference_patch_encoder_layer_call_fn_301946	
patch
unknown:	�@
	unknown_0:@
	unknown_1:	�@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallpatchunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3000212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:�������������������

_user_specified_namepatch
�
k
?__inference_add_layer_call_and_return_conditional_losses_302104
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:����������@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:����������@:����������@:V R
,
_output_shapes
:����������@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:����������@
"
_user_specified_name
inputs/1
�/
�
I__inference_patch_encoder_layer_call_and_return_conditional_losses_300021	
patch:
'dense_tensordot_readvariableop_resource:	�@3
%dense_biasadd_readvariableop_resource:@4
!embedding_embedding_lookup_300014:	�@
identity��dense/BiasAdd/ReadVariableOp�dense/Tensordot/ReadVariableOp�embedding/embedding_lookup\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start]
range/limitConst*
_output_shapes
: *
dtype0*
value
B :�2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:�2
range�
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	�@*
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
!:�������������������2
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
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_300014range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/300014*
_output_shapes
:	�@*
dtype02
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/300014*
_output_shapes
:	�@2%
#embedding/embedding_lookup/Identity�
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	�@2'
%embedding/embedding_lookup/Identity_1�
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:����������@2
add�
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':�������������������: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:�������������������

_user_specified_namepatch
�

�
H__inference_output_layer_layer_call_and_return_conditional_losses_302415

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
�
D
(__inference_patches_layer_call_fn_301895

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
!:�������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_2999792
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:�������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:�����������:Y U
1
_output_shapes
:�����������
 
_user_specified_nameimages
�
�
6__inference_layer_normalization_1_layer_call_fn_302141

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
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3001402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�

�
B__inference_conv_1_layer_call_and_return_conditional_losses_300296

inputs8
conv2d_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
B__inference_conv_1_layer_call_and_return_conditional_losses_302307

inputs8
conv2d_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_dense_1_layer_call_fn_302188

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
 *-
_output_shapes
:�����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3001842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
6__inference_WheatClassifier_CNN_1_layer_call_fn_300456
input_layer
unknown:	�@
	unknown_0:@
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:@@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:	@�

unknown_15:	�

unknown_16:	�@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@


unknown_21:
$

unknown_22:



unknown_23:


unknown_24:	�2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identity��StatefulPartitionedCall�
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
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*A
_read_only_resource_inputs#
!	
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_3003912
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�
�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_302269

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
T0*,
_output_shapes
:����������*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:����������2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:����������2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�

�
H__inference_output_layer_layer_call_and_return_conditional_losses_300384

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
�'
�
C__inference_dense_1_layer_call_and_return_conditional_losses_302179

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
T0*,
_output_shapes
:����������@2
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
T0*-
_output_shapes
:�����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xz
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*-
_output_shapes
:�����������2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*-
_output_shapes
:�����������2
Gelu/truedive
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:�����������2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xx
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:�����������2

Gelu/adds

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:�����������2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*-
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�	
�
@__inference_FC_2_layer_call_and_return_conditional_losses_302385

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
��
�0
__inference__traced_save_302756
file_prefix8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop*
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
Esavev2_multi_head_attention_attention_output_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop@
<savev2_adamw_layer_normalization_gamma_m_read_readvariableop?
;savev2_adamw_layer_normalization_beta_m_read_readvariableopB
>savev2_adamw_layer_normalization_1_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_1_beta_m_read_readvariableop5
1savev2_adamw_dense_1_kernel_m_read_readvariableop3
/savev2_adamw_dense_1_bias_m_read_readvariableop5
1savev2_adamw_dense_2_kernel_m_read_readvariableop3
/savev2_adamw_dense_2_bias_m_read_readvariableopB
>savev2_adamw_layer_normalization_2_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_2_beta_m_read_readvariableop4
0savev2_adamw_conv_1_kernel_m_read_readvariableop2
.savev2_adamw_conv_1_bias_m_read_readvariableop4
0savev2_adamw_conv_2_kernel_m_read_readvariableop2
.savev2_adamw_conv_2_bias_m_read_readvariableop2
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
Msavev2_adamw_multi_head_attention_attention_output_bias_m_read_readvariableop@
<savev2_adamw_layer_normalization_gamma_v_read_readvariableop?
;savev2_adamw_layer_normalization_beta_v_read_readvariableopB
>savev2_adamw_layer_normalization_1_gamma_v_read_readvariableopA
=savev2_adamw_layer_normalization_1_beta_v_read_readvariableop5
1savev2_adamw_dense_1_kernel_v_read_readvariableop3
/savev2_adamw_dense_1_bias_v_read_readvariableop5
1savev2_adamw_dense_2_kernel_v_read_readvariableop3
/savev2_adamw_dense_2_bias_v_read_readvariableopB
>savev2_adamw_layer_normalization_2_gamma_v_read_readvariableopA
=savev2_adamw_layer_normalization_2_beta_v_read_readvariableop4
0savev2_adamw_conv_1_kernel_v_read_readvariableop2
.savev2_adamw_conv_1_bias_v_read_readvariableop4
0savev2_adamw_conv_2_kernel_v_read_readvariableop2
.savev2_adamw_conv_2_bias_v_read_readvariableop2
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
Msavev2_adamw_multi_head_attention_attention_output_bias_v_read_readvariableop
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
ShardedFilename�6
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*�6
value�5B�5hB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*�
value�B�hB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�.
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_2_kernel_read_readvariableop$savev2_fc_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop%savev2_adamw_iter_read_readvariableop'savev2_adamw_beta_1_read_readvariableop'savev2_adamw_beta_2_read_readvariableop&savev2_adamw_decay_read_readvariableop.savev2_adamw_learning_rate_read_readvariableop-savev2_adamw_weight_decay_read_readvariableop5savev2_patch_encoder_dense_kernel_read_readvariableop3savev2_patch_encoder_dense_bias_read_readvariableop=savev2_patch_encoder_embedding_embeddings_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop<savev2_adamw_layer_normalization_gamma_m_read_readvariableop;savev2_adamw_layer_normalization_beta_m_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_1_beta_m_read_readvariableop1savev2_adamw_dense_1_kernel_m_read_readvariableop/savev2_adamw_dense_1_bias_m_read_readvariableop1savev2_adamw_dense_2_kernel_m_read_readvariableop/savev2_adamw_dense_2_bias_m_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_2_beta_m_read_readvariableop0savev2_adamw_conv_1_kernel_m_read_readvariableop.savev2_adamw_conv_1_bias_m_read_readvariableop0savev2_adamw_conv_2_kernel_m_read_readvariableop.savev2_adamw_conv_2_bias_m_read_readvariableop.savev2_adamw_fc_1_kernel_m_read_readvariableop,savev2_adamw_fc_1_bias_m_read_readvariableop.savev2_adamw_fc_2_kernel_m_read_readvariableop,savev2_adamw_fc_2_bias_m_read_readvariableop6savev2_adamw_output_layer_kernel_m_read_readvariableop4savev2_adamw_output_layer_bias_m_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_m_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_m_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_m_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_m_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_m_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_m_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_m_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_m_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_m_read_readvariableop<savev2_adamw_layer_normalization_gamma_v_read_readvariableop;savev2_adamw_layer_normalization_beta_v_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_1_beta_v_read_readvariableop1savev2_adamw_dense_1_kernel_v_read_readvariableop/savev2_adamw_dense_1_bias_v_read_readvariableop1savev2_adamw_dense_2_kernel_v_read_readvariableop/savev2_adamw_dense_2_bias_v_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_2_beta_v_read_readvariableop0savev2_adamw_conv_1_kernel_v_read_readvariableop.savev2_adamw_conv_1_bias_v_read_readvariableop0savev2_adamw_conv_2_kernel_v_read_readvariableop.savev2_adamw_conv_2_bias_v_read_readvariableop.savev2_adamw_fc_1_kernel_v_read_readvariableop,savev2_adamw_fc_1_bias_v_read_readvariableop.savev2_adamw_fc_2_kernel_v_read_readvariableop,savev2_adamw_fc_2_bias_v_read_readvariableop6savev2_adamw_output_layer_kernel_v_read_readvariableop4savev2_adamw_output_layer_bias_v_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_v_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_v_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_v_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_v_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_v_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_v_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_v_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_v_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *v
dtypesl
j2h	2
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:@:@:@:	@�:�:	�@:@:@:@:@
:
:

:
:	�2:2:22:2:2:: : : : : : :	�@:@:	�@:@@:@:@@:@:@@:@:@@:@: : : : :@:@:@:@:	@�:�:	�@:@:@:@:@
:
:

:
:	�2:2:22:2:2::	�@:@:	�@:@@:@:@@:@:@@:@:@@:@:@:@:@:@:	@�:�:	�@:@:@:@:@
:
:

:
:	�2:2:22:2:2::	�@:@:	�@:@@:@:@@:@:@@:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@�:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@:,(
&
_output_shapes
:@
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:%!

_output_shapes
:	�2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	�@: 

_output_shapes
:@:%!

_output_shapes
:	�@:($
"
_output_shapes
:@@:$ 

_output_shapes

:@:( $
"
_output_shapes
:@@:$! 

_output_shapes

:@:("$
"
_output_shapes
:@@:$# 

_output_shapes

:@:($$
"
_output_shapes
:@@: %

_output_shapes
:@:&

_output_shapes
: :'

_output_shapes
: :(

_output_shapes
: :)

_output_shapes
: : *

_output_shapes
:@: +

_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@:%.!

_output_shapes
:	@�:!/

_output_shapes	
:�:%0!

_output_shapes
:	�@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:,4(
&
_output_shapes
:@
: 5

_output_shapes
:
:,6(
&
_output_shapes
:

: 7

_output_shapes
:
:%8!

_output_shapes
:	�2: 9

_output_shapes
:2:$: 

_output_shapes

:22: ;

_output_shapes
:2:$< 

_output_shapes

:2: =

_output_shapes
::%>!

_output_shapes
:	�@: ?

_output_shapes
:@:%@!

_output_shapes
:	�@:(A$
"
_output_shapes
:@@:$B 

_output_shapes

:@:(C$
"
_output_shapes
:@@:$D 

_output_shapes

:@:(E$
"
_output_shapes
:@@:$F 

_output_shapes

:@:(G$
"
_output_shapes
:@@: H

_output_shapes
:@: I

_output_shapes
:@: J

_output_shapes
:@: K

_output_shapes
:@: L

_output_shapes
:@:%M!

_output_shapes
:	@�:!N

_output_shapes	
:�:%O!

_output_shapes
:	�@: P

_output_shapes
:@: Q

_output_shapes
:@: R

_output_shapes
:@:,S(
&
_output_shapes
:@
: T

_output_shapes
:
:,U(
&
_output_shapes
:

: V

_output_shapes
:
:%W!

_output_shapes
:	�2: X

_output_shapes
:2:$Y 

_output_shapes

:22: Z

_output_shapes
:2:$[ 

_output_shapes

:2: \

_output_shapes
::%]!

_output_shapes
:	�@: ^

_output_shapes
:@:%_!

_output_shapes
:	�@:(`$
"
_output_shapes
:@@:$a 

_output_shapes

:@:(b$
"
_output_shapes
:@@:$c 

_output_shapes

:@:(d$
"
_output_shapes
:@@:$e 

_output_shapes

:@:(f$
"
_output_shapes
:@@: g

_output_shapes
:@:h

_output_shapes
: 
�
�
6__inference_layer_normalization_2_layer_call_fn_302278

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
 *,
_output_shapes
:����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3002642
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_300051

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
T0*,
_output_shapes
:����������*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:����������2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:����������2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_301235
input_layer
unknown:	�@
	unknown_0:@
	unknown_1:	�@
	unknown_2:@
	unknown_3:@
	unknown_4:@@
	unknown_5:@
	unknown_6:@@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@

unknown_12:@

unknown_13:@

unknown_14:	@�

unknown_15:	�

unknown_16:	�@

unknown_17:@

unknown_18:@

unknown_19:@$

unknown_20:@


unknown_21:
$

unknown_22:



unknown_23:


unknown_24:	�2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identity��StatefulPartitionedCall�
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
unknown_29*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*A
_read_only_resource_inputs#
!	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_2999462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:�����������
%
_user_specified_nameinput_layer
�'
�
C__inference_dense_1_layer_call_and_return_conditional_losses_300184

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
T0*,
_output_shapes
:����������@2
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
T0*-
_output_shapes
:�����������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:�����������2	
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xz
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*-
_output_shapes
:�����������2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *��?2
Gelu/Cast/x�
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*-
_output_shapes
:�����������2
Gelu/truedive
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:�����������2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2

Gelu/add/xx
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:�����������2

Gelu/adds

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:�����������2

Gelu/mul_1�
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*-
_output_shapes
:�����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�
a
E__inference_maxpool_1_layer_call_and_return_conditional_losses_299952

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
J
.__inference_flatten_layer_layer_call_fn_302346

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
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3003252
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_301968

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
T0*,
_output_shapes
:����������*
	keep_dims(2
moments/mean�
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:����������2
moments/StopGradient�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:����������@2
moments/SquaredDifference�
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices�
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:����������2
batchnorm/Rsqrt�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOp�
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_1�
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/mul_2�
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp�
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/sub�
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:����������@2
batchnorm/add_1�
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:����������@
 
_user_specified_nameinputs
�

�
B__inference_conv_2_layer_call_and_return_conditional_losses_302326

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
T0*/
_output_shapes
:���������
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
T0*/
_output_shapes
:���������
2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs
�
e
I__inference_flatten_layer_layer_call_and_return_conditional_losses_300325

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"�����  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������
:W S
/
_output_shapes
:���������

 
_user_specified_nameinputs"�L
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
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�S
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer_with_weights-6
layer-10
layer-11
layer_with_weights-7
layer-12
layer_with_weights-8
layer-13
layer-14
layer-15
layer_with_weights-9
layer-16
layer-17
layer_with_weights-10
layer-18
layer-19
layer_with_weights-11
layer-20
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"�M
_tf_keras_network�L{"name": "WheatClassifier_CNN_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "WheatClassifier_CNN_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Patches", "config": {"layer was saved without config": true}, "name": "patches", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "PatchEncoder", "config": {"layer was saved without config": true}, "name": "patch_encoder", "inbound_nodes": [[["patches", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization", "inbound_nodes": [[["patch_encoder", 0, 0, {}]]]}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}, "name": "multi_head_attention", "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_2", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [20, 20, 64]}}, "name": "reshape", "inbound_nodes": [[["layer_normalization_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_layer", "inbound_nodes": [[["maxpool_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1", "inbound_nodes": [[["flatten_layer", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_1", "inbound_nodes": [[["FC_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_2", "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_2", "inbound_nodes": [[["FC_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "shared_object_id": 41, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 200, 200, 3]}, "float32", "input_layer"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 43}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Addons>AdamW", "config": {"name": "AdamW", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false, "weight_decay": 9.999999747378752e-05}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
�
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "patches", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Patches", "config": {"layer was saved without config": true}}
�
 
projection
!position_embedding
"regularization_losses
#	variables
$trainable_variables
%	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "patch_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PatchEncoder", "config": {"layer was saved without config": true}}
�
&axis
	'gamma
(beta
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 2}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["patch_encoder", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
�

-_query_dense
.
_key_dense
/_value_dense
0_softmax
1_dropout_layer
2_output_dense
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "multi_head_attention", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}, "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]], "shared_object_id": 6}
�
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 400, 64]}, {"class_name": "TensorShape", "items": [null, 400, 64]}]}
�
;axis
	<gamma
=beta
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
�	

Bkernel
Cbias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
�	

Hkernel
Ibias
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 45}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 128]}}
�
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]], "shared_object_id": 17, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 400, 64]}, {"class_name": "TensorShape", "items": [null, 400, 64]}]}
�
Raxis
	Sgamma
Tbeta
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_1", 0, 0, {}]]], "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
�
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [20, 20, 64]}}, "inbound_nodes": [[["layer_normalization_2", 0, 0, {}]]], "shared_object_id": 21}
�


]kernel
^bias
_regularization_losses
`	variables
atrainable_variables
b	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["reshape", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 64]}}
�


ckernel
dbias
eregularization_losses
f	variables
gtrainable_variables
h	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�	
_tf_keras_layer�	{"name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 18, 18, 10]}}
�
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "maxpool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 48}}
�
mregularization_losses
n	variables
otrainable_variables
p	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "flatten_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["maxpool_1", 0, 0, {}]]], "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 49}}
�	

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "FC_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_layer", 0, 0, {}]]], "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 640}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 640]}}
�
wregularization_losses
x	variables
ytrainable_variables
z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_ReLu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_1", 0, 0, {}]]], "shared_object_id": 33}
�	

{kernel
|bias
}regularization_losses
~	variables
trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "FC_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]], "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "leaky_ReLu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_2", 0, 0, {}]]], "shared_object_id": 37}
�	
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 38}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 39}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]], "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate
�weight_decay'm�(m�<m�=m�Bm�Cm�Hm�Im�Sm�Tm�]m�^m�cm�dm�qm�rm�{m�|m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�'v�(v�<v�=v�Bv�Cv�Hv�Iv�Sv�Tv�]v�^v�cv�dv�qv�rv�{v�|v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
 "
trackable_list_wrapper
�
�0
�1
�2
'3
(4
�5
�6
�7
�8
�9
�10
�11
�12
<13
=14
B15
C16
H17
I18
S19
T20
]21
^22
c23
d24
q25
r26
{27
|28
�29
�30"
trackable_list_wrapper
�
�0
�1
�2
'3
(4
�5
�6
�7
�8
�9
�10
�11
�12
<13
=14
B15
C16
H17
I18
S19
T20
]21
^22
c23
d24
q25
r26
{27
|28
�29
�30"
trackable_list_wrapper
�
�metrics
regularization_losses
	variables
�non_trainable_variables
�layer_metrics
�layers
trainable_variables
 �layer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
regularization_losses
	variables
�non_trainable_variables
�layer_metrics
�layers
trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 300]}}
�
�
embeddings
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 400, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 57}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "shared_object_id": 58, "build_input_shape": {"class_name": "TensorShape", "items": [400]}}
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
�metrics
"regularization_losses
#	variables
�non_trainable_variables
�layer_metrics
�layers
$trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@2layer_normalization/gamma
&:$@2layer_normalization/beta
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
�
�metrics
)regularization_losses
*	variables
�non_trainable_variables
�layer_metrics
�layers
+trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 59, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 61, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "shared_object_id": 62}
�
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 63}
�
�partial_output_shape
�full_output_shape
�kernel
	�bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 64], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 64, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 4, 64]}}
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
�metrics
3regularization_losses
4	variables
�non_trainable_variables
�layer_metrics
�layers
5trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
7regularization_losses
8	variables
�non_trainable_variables
�layer_metrics
�layers
9trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_1/gamma
(:&@2layer_normalization_1/beta
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
�
�metrics
>regularization_losses
?	variables
�non_trainable_variables
�layer_metrics
�layers
@trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	@�2dense_1/kernel
:�2dense_1/bias
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
�
�metrics
Dregularization_losses
E	variables
�non_trainable_variables
�layer_metrics
�layers
Ftrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�@2dense_2/kernel
:@2dense_2/bias
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
�
�metrics
Jregularization_losses
K	variables
�non_trainable_variables
�layer_metrics
�layers
Ltrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
Nregularization_losses
O	variables
�non_trainable_variables
�layer_metrics
�layers
Ptrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_2/gamma
(:&@2layer_normalization_2/beta
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
�
�metrics
Uregularization_losses
V	variables
�non_trainable_variables
�layer_metrics
�layers
Wtrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
Yregularization_losses
Z	variables
�non_trainable_variables
�layer_metrics
�layers
[trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%@
2conv_1/kernel
:
2conv_1/bias
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
�
�metrics
_regularization_losses
`	variables
�non_trainable_variables
�layer_metrics
�layers
atrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':%

2conv_2/kernel
:
2conv_2/bias
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
�
�metrics
eregularization_losses
f	variables
�non_trainable_variables
�layer_metrics
�layers
gtrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
iregularization_losses
j	variables
�non_trainable_variables
�layer_metrics
�layers
ktrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
mregularization_losses
n	variables
�non_trainable_variables
�layer_metrics
�layers
otrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	�22FC_1/kernel
:22	FC_1/bias
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
�metrics
sregularization_losses
t	variables
�non_trainable_variables
�layer_metrics
�layers
utrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
wregularization_losses
x	variables
�non_trainable_variables
�layer_metrics
�layers
ytrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:222FC_2/kernel
:22	FC_2/bias
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
�
�metrics
}regularization_losses
~	variables
�non_trainable_variables
�layer_metrics
�layers
trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2
AdamW/iter
: (2AdamW/beta_1
: (2AdamW/beta_2
: (2AdamW/decay
: (2AdamW/learning_rate
: (2AdamW/weight_decay
-:+	�@2patch_encoder/dense/kernel
&:$@2patch_encoder/dense/bias
5:3	�@2"patch_encoder/embedding/embeddings
7:5@@2!multi_head_attention/query/kernel
1:/@2multi_head_attention/query/bias
5:3@@2multi_head_attention/key/kernel
/:-@2multi_head_attention/key/bias
7:5@@2!multi_head_attention/value/kernel
1:/@2multi_head_attention/value/bias
B:@@@2,multi_head_attention/attention_output/kernel
8:6@2*multi_head_attention/attention_output/bias
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
20"
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
 0
!1"
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�metrics
�regularization_losses
�	variables
�non_trainable_variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
-0
.1
/2
03
14
25"
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
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 65}
�

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 43}
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
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
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
-:+@
2AdamW/conv_1/kernel/m
:
2AdamW/conv_1/bias/m
-:+

2AdamW/conv_2/kernel/m
:
2AdamW/conv_2/bias/m
$:"	�22AdamW/FC_1/kernel/m
:22AdamW/FC_1/bias/m
#:!222AdamW/FC_2/kernel/m
:22AdamW/FC_2/bias/m
+:)22AdamW/output_layer/kernel/m
%:#2AdamW/output_layer/bias/m
3:1	�@2"AdamW/patch_encoder/dense/kernel/m
,:*@2 AdamW/patch_encoder/dense/bias/m
;:9	�@2*AdamW/patch_encoder/embedding/embeddings/m
=:;@@2)AdamW/multi_head_attention/query/kernel/m
7:5@2'AdamW/multi_head_attention/query/bias/m
;:9@@2'AdamW/multi_head_attention/key/kernel/m
5:3@2%AdamW/multi_head_attention/key/bias/m
=:;@@2)AdamW/multi_head_attention/value/kernel/m
7:5@2'AdamW/multi_head_attention/value/bias/m
H:F@@24AdamW/multi_head_attention/attention_output/kernel/m
>:<@22AdamW/multi_head_attention/attention_output/bias/m
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
-:+@
2AdamW/conv_1/kernel/v
:
2AdamW/conv_1/bias/v
-:+

2AdamW/conv_2/kernel/v
:
2AdamW/conv_2/bias/v
$:"	�22AdamW/FC_1/kernel/v
:22AdamW/FC_1/bias/v
#:!222AdamW/FC_2/kernel/v
:22AdamW/FC_2/bias/v
+:)22AdamW/output_layer/kernel/v
%:#2AdamW/output_layer/bias/v
3:1	�@2"AdamW/patch_encoder/dense/kernel/v
,:*@2 AdamW/patch_encoder/dense/bias/v
;:9	�@2*AdamW/patch_encoder/embedding/embeddings/v
=:;@@2)AdamW/multi_head_attention/query/kernel/v
7:5@2'AdamW/multi_head_attention/query/bias/v
;:9@@2'AdamW/multi_head_attention/key/kernel/v
5:3@2%AdamW/multi_head_attention/key/bias/v
=:;@@2)AdamW/multi_head_attention/value/kernel/v
7:5@2'AdamW/multi_head_attention/value/bias/v
H:F@@24AdamW/multi_head_attention/attention_output/kernel/v
>:<@22AdamW/multi_head_attention/attention_output/bias/v
�2�
!__inference__wrapped_model_299946�
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
�2�
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301485
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301742
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301072
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301158�
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
�2�
6__inference_WheatClassifier_CNN_1_layer_call_fn_300456
6__inference_WheatClassifier_CNN_1_layer_call_fn_301809
6__inference_WheatClassifier_CNN_1_layer_call_fn_301876
6__inference_WheatClassifier_CNN_1_layer_call_fn_300986�
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
C__inference_patches_layer_call_and_return_conditional_losses_301890�
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
(__inference_patches_layer_call_fn_301895�
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
I__inference_patch_encoder_layer_call_and_return_conditional_losses_301935�
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
.__inference_patch_encoder_layer_call_fn_301946�
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
O__inference_layer_normalization_layer_call_and_return_conditional_losses_301968�
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
4__inference_layer_normalization_layer_call_fn_301977�
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
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_302012
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_302054�
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
5__inference_multi_head_attention_layer_call_fn_302076
5__inference_multi_head_attention_layer_call_fn_302098�
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
?__inference_add_layer_call_and_return_conditional_losses_302104�
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
$__inference_add_layer_call_fn_302110�
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
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_302132�
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
6__inference_layer_normalization_1_layer_call_fn_302141�
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
C__inference_dense_1_layer_call_and_return_conditional_losses_302179�
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
(__inference_dense_1_layer_call_fn_302188�
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
C__inference_dense_2_layer_call_and_return_conditional_losses_302226�
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
(__inference_dense_2_layer_call_fn_302235�
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
A__inference_add_1_layer_call_and_return_conditional_losses_302241�
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
&__inference_add_1_layer_call_fn_302247�
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
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_302269�
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
6__inference_layer_normalization_2_layer_call_fn_302278�
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
C__inference_reshape_layer_call_and_return_conditional_losses_302292�
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
(__inference_reshape_layer_call_fn_302297�
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
B__inference_conv_1_layer_call_and_return_conditional_losses_302307�
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
'__inference_conv_1_layer_call_fn_302316�
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
B__inference_conv_2_layer_call_and_return_conditional_losses_302326�
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
'__inference_conv_2_layer_call_fn_302335�
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
E__inference_maxpool_1_layer_call_and_return_conditional_losses_299952�
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
*__inference_maxpool_1_layer_call_fn_299958�
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
I__inference_flatten_layer_layer_call_and_return_conditional_losses_302341�
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
.__inference_flatten_layer_layer_call_fn_302346�
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
@__inference_FC_1_layer_call_and_return_conditional_losses_302356�
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
%__inference_FC_1_layer_call_fn_302365�
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
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_302370�
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
-__inference_leaky_ReLu_1_layer_call_fn_302375�
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
@__inference_FC_2_layer_call_and_return_conditional_losses_302385�
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
%__inference_FC_2_layer_call_fn_302394�
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
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_302399�
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
-__inference_leaky_ReLu_2_layer_call_fn_302404�
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
H__inference_output_layer_layer_call_and_return_conditional_losses_302415�
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
-__inference_output_layer_layer_call_fn_302424�
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
$__inference_signature_wrapper_301235input_layer"�
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
 �
@__inference_FC_1_layer_call_and_return_conditional_losses_302356]qr0�-
&�#
!�
inputs����������
� "%�"
�
0���������2
� y
%__inference_FC_1_layer_call_fn_302365Pqr0�-
&�#
!�
inputs����������
� "����������2�
@__inference_FC_2_layer_call_and_return_conditional_losses_302385\{|/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� x
%__inference_FC_2_layer_call_fn_302394O{|/�,
%�"
 �
inputs���������2
� "����������2�
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301072�,���'(��������<=BCHIST]^cdqr{|��F�C
<�9
/�,
input_layer�����������
p 

 
� "%�"
�
0���������
� �
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301158�,���'(��������<=BCHIST]^cdqr{|��F�C
<�9
/�,
input_layer�����������
p

 
� "%�"
�
0���������
� �
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301485�,���'(��������<=BCHIST]^cdqr{|��A�>
7�4
*�'
inputs�����������
p 

 
� "%�"
�
0���������
� �
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_301742�,���'(��������<=BCHIST]^cdqr{|��A�>
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
6__inference_WheatClassifier_CNN_1_layer_call_fn_300456�,���'(��������<=BCHIST]^cdqr{|��F�C
<�9
/�,
input_layer�����������
p 

 
� "�����������
6__inference_WheatClassifier_CNN_1_layer_call_fn_300986�,���'(��������<=BCHIST]^cdqr{|��F�C
<�9
/�,
input_layer�����������
p

 
� "�����������
6__inference_WheatClassifier_CNN_1_layer_call_fn_301809�,���'(��������<=BCHIST]^cdqr{|��A�>
7�4
*�'
inputs�����������
p 

 
� "�����������
6__inference_WheatClassifier_CNN_1_layer_call_fn_301876�,���'(��������<=BCHIST]^cdqr{|��A�>
7�4
*�'
inputs�����������
p

 
� "�����������
!__inference__wrapped_model_299946�,���'(��������<=BCHIST]^cdqr{|��>�;
4�1
/�,
input_layer�����������
� ";�8
6
output_layer&�#
output_layer����������
A__inference_add_1_layer_call_and_return_conditional_losses_302241�d�a
Z�W
U�R
'�$
inputs/0����������@
'�$
inputs/1����������@
� "*�'
 �
0����������@
� �
&__inference_add_1_layer_call_fn_302247�d�a
Z�W
U�R
'�$
inputs/0����������@
'�$
inputs/1����������@
� "�����������@�
?__inference_add_layer_call_and_return_conditional_losses_302104�d�a
Z�W
U�R
'�$
inputs/0����������@
'�$
inputs/1����������@
� "*�'
 �
0����������@
� �
$__inference_add_layer_call_fn_302110�d�a
Z�W
U�R
'�$
inputs/0����������@
'�$
inputs/1����������@
� "�����������@�
B__inference_conv_1_layer_call_and_return_conditional_losses_302307l]^7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������

� �
'__inference_conv_1_layer_call_fn_302316_]^7�4
-�*
(�%
inputs���������@
� " ����������
�
B__inference_conv_2_layer_call_and_return_conditional_losses_302326lcd7�4
-�*
(�%
inputs���������

� "-�*
#� 
0���������

� �
'__inference_conv_2_layer_call_fn_302335_cd7�4
-�*
(�%
inputs���������

� " ����������
�
C__inference_dense_1_layer_call_and_return_conditional_losses_302179gBC4�1
*�'
%�"
inputs����������@
� "+�(
!�
0�����������
� �
(__inference_dense_1_layer_call_fn_302188ZBC4�1
*�'
%�"
inputs����������@
� "�������������
C__inference_dense_2_layer_call_and_return_conditional_losses_302226gHI5�2
+�(
&�#
inputs�����������
� "*�'
 �
0����������@
� �
(__inference_dense_2_layer_call_fn_302235ZHI5�2
+�(
&�#
inputs�����������
� "�����������@�
I__inference_flatten_layer_layer_call_and_return_conditional_losses_302341a7�4
-�*
(�%
inputs���������

� "&�#
�
0����������
� �
.__inference_flatten_layer_layer_call_fn_302346T7�4
-�*
(�%
inputs���������

� "������������
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_302132f<=4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
6__inference_layer_normalization_1_layer_call_fn_302141Y<=4�1
*�'
%�"
inputs����������@
� "�����������@�
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_302269fST4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
6__inference_layer_normalization_2_layer_call_fn_302278YST4�1
*�'
%�"
inputs����������@
� "�����������@�
O__inference_layer_normalization_layer_call_and_return_conditional_losses_301968f'(4�1
*�'
%�"
inputs����������@
� "*�'
 �
0����������@
� �
4__inference_layer_normalization_layer_call_fn_301977Y'(4�1
*�'
%�"
inputs����������@
� "�����������@�
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_302370X/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� |
-__inference_leaky_ReLu_1_layer_call_fn_302375K/�,
%�"
 �
inputs���������2
� "����������2�
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_302399X/�,
%�"
 �
inputs���������2
� "%�"
�
0���������2
� |
-__inference_leaky_ReLu_2_layer_call_fn_302404K/�,
%�"
 �
inputs���������2
� "����������2�
E__inference_maxpool_1_layer_call_and_return_conditional_losses_299952�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_maxpool_1_layer_call_fn_299958�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_302012���������i�f
_�\
$�!
query����������@
$�!
value����������@

 

 
p 
p 
� "*�'
 �
0����������@
� �
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_302054���������i�f
_�\
$�!
query����������@
$�!
value����������@

 

 
p 
p
� "*�'
 �
0����������@
� �
5__inference_multi_head_attention_layer_call_fn_302076���������i�f
_�\
$�!
query����������@
$�!
value����������@

 

 
p 
p 
� "�����������@�
5__inference_multi_head_attention_layer_call_fn_302098���������i�f
_�\
$�!
query����������@
$�!
value����������@

 

 
p 
p
� "�����������@�
H__inference_output_layer_layer_call_and_return_conditional_losses_302415^��/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� �
-__inference_output_layer_layer_call_fn_302424Q��/�,
%�"
 �
inputs���������2
� "�����������
I__inference_patch_encoder_layer_call_and_return_conditional_losses_301935r���<�9
2�/
-�*
patch�������������������
� "*�'
 �
0����������@
� �
.__inference_patch_encoder_layer_call_fn_301946e���<�9
2�/
-�*
patch�������������������
� "�����������@�
C__inference_patches_layer_call_and_return_conditional_losses_301890p9�6
/�,
*�'
images�����������
� "3�0
)�&
0�������������������
� �
(__inference_patches_layer_call_fn_301895c9�6
/�,
*�'
images�����������
� "&�#��������������������
C__inference_reshape_layer_call_and_return_conditional_losses_302292e4�1
*�'
%�"
inputs����������@
� "-�*
#� 
0���������@
� �
(__inference_reshape_layer_call_fn_302297X4�1
*�'
%�"
inputs����������@
� " ����������@�
$__inference_signature_wrapper_301235�,���'(��������<=BCHIST]^cdqr{|��M�J
� 
C�@
>
input_layer/�,
input_layer�����������";�8
6
output_layer&�#
output_layer���������