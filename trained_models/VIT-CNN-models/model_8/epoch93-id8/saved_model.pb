яЌ8
Н Ф 
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
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
Џ
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
┐
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
Г
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
alphafloat%═╠L>"
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
Ї
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
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
Ї
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
dtypetypeѕ
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
Ц
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	ѕ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
list(type)(0ѕ
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

2	љ
Й
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
executor_typestring ѕ
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
Ш
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.5.02v2.5.0-0-ga4dfb8d1a718дЎ0
і
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelayer_normalization/gamma
Ѓ
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:@*
dtype0
ѕ
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer_normalization/beta
Ђ
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:@*
dtype0
ј
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_1/gamma
Є
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:@*
dtype0
ї
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_1/beta
Ё
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	@ђ*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:ђ*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	ђ@*
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
ј
layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_2/gamma
Є
/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
:@*
dtype0
ї
layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_2/beta
Ё
.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
:@*
dtype0
ј
layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_3/gamma
Є
/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
:@*
dtype0
ї
layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_3/beta
Ё
.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
:@*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	@ђ*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:ђ*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	ђ@*
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
ј
layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_4/gamma
Є
/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
:@*
dtype0
ї
layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_4/beta
Ё
.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
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
r
FC_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@2*
shared_nameFC_1/kernel
k
FC_1/kernel/Read/ReadVariableOpReadVariableOpFC_1/kernel*
_output_shapes

:@2*
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
ѓ
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
Љ
patch_encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	г@*+
shared_namepatch_encoder/dense/kernel
і
.patch_encoder/dense/kernel/Read/ReadVariableOpReadVariableOppatch_encoder/dense/kernel*
_output_shapes
:	г@*
dtype0
ѕ
patch_encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namepatch_encoder/dense/bias
Ђ
,patch_encoder/dense/bias/Read/ReadVariableOpReadVariableOppatch_encoder/dense/bias*
_output_shapes
:@*
dtype0
а
"patch_encoder/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*3
shared_name$"patch_encoder/embedding/embeddings
Ў
6patch_encoder/embedding/embeddings/Read/ReadVariableOpReadVariableOp"patch_encoder/embedding/embeddings*
_output_shapes

:d@*
dtype0
б
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention/query/kernel
Џ
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:@@*
dtype0
џ
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention/query/bias
Њ
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:@*
dtype0
ъ
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!multi_head_attention/key/kernel
Ќ
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:@@*
dtype0
ќ
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_namemulti_head_attention/key/bias
Ј
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:@*
dtype0
б
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention/value/kernel
Џ
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:@@*
dtype0
џ
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention/value/bias
Њ
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:@*
dtype0
И
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*=
shared_name.,multi_head_attention/attention_output/kernel
▒
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:@@*
dtype0
г
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*multi_head_attention/attention_output/bias
Ц
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:@*
dtype0
д
#multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_1/query/kernel
Ъ
7multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/query/kernel*"
_output_shapes
:@@*
dtype0
ъ
!multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/query/bias
Ќ
5multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/query/bias*
_output_shapes

:@*
dtype0
б
!multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention_1/key/kernel
Џ
5multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/key/kernel*"
_output_shapes
:@@*
dtype0
џ
multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention_1/key/bias
Њ
3multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_1/key/bias*
_output_shapes

:@*
dtype0
д
#multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_1/value/kernel
Ъ
7multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/value/kernel*"
_output_shapes
:@@*
dtype0
ъ
!multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/value/bias
Ќ
5multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/value/bias*
_output_shapes

:@*
dtype0
╝
.multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.multi_head_attention_1/attention_output/kernel
х
Bmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_1/attention_output/kernel*"
_output_shapes
:@@*
dtype0
░
,multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,multi_head_attention_1/attention_output/bias
Е
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
џ
!AdamW/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!AdamW/layer_normalization/gamma/m
Њ
5AdamW/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp!AdamW/layer_normalization/gamma/m*
_output_shapes
:@*
dtype0
ў
 AdamW/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/layer_normalization/beta/m
Љ
4AdamW/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp AdamW/layer_normalization/beta/m*
_output_shapes
:@*
dtype0
ъ
#AdamW/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_1/gamma/m
Ќ
7AdamW/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
ю
"AdamW/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_1/beta/m
Ћ
6AdamW/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_1/beta/m*
_output_shapes
:@*
dtype0
Ѕ
AdamW/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*'
shared_nameAdamW/dense_1/kernel/m
ѓ
*AdamW/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_1/kernel/m*
_output_shapes
:	@ђ*
dtype0
Ђ
AdamW/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdamW/dense_1/bias/m
z
(AdamW/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_1/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
AdamW/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*'
shared_nameAdamW/dense_2/kernel/m
ѓ
*AdamW/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_2/kernel/m*
_output_shapes
:	ђ@*
dtype0
ђ
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
ъ
#AdamW/layer_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_2/gamma/m
Ќ
7AdamW/layer_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
ю
"AdamW/layer_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_2/beta/m
Ћ
6AdamW/layer_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_2/beta/m*
_output_shapes
:@*
dtype0
ъ
#AdamW/layer_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_3/gamma/m
Ќ
7AdamW/layer_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_3/gamma/m*
_output_shapes
:@*
dtype0
ю
"AdamW/layer_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_3/beta/m
Ћ
6AdamW/layer_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_3/beta/m*
_output_shapes
:@*
dtype0
Ѕ
AdamW/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*'
shared_nameAdamW/dense_3/kernel/m
ѓ
*AdamW/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_3/kernel/m*
_output_shapes
:	@ђ*
dtype0
Ђ
AdamW/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdamW/dense_3/bias/m
z
(AdamW/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_3/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
AdamW/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*'
shared_nameAdamW/dense_4/kernel/m
ѓ
*AdamW/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_4/kernel/m*
_output_shapes
:	ђ@*
dtype0
ђ
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
ъ
#AdamW/layer_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_4/gamma/m
Ќ
7AdamW/layer_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_4/gamma/m*
_output_shapes
:@*
dtype0
ю
"AdamW/layer_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_4/beta/m
Ћ
6AdamW/layer_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_4/beta/m*
_output_shapes
:@*
dtype0
ј
AdamW/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@
*&
shared_nameAdamW/conv_1/kernel/m
Є
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
ј
AdamW/conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*&
shared_nameAdamW/conv_2/kernel/m
Є
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
ј
AdamW/conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_3/kernel/m
Є
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
ј
AdamW/conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_4/kernel/m
Є
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
ѓ
AdamW/FC_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@2*$
shared_nameAdamW/FC_1/kernel/m
{
'AdamW/FC_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/m*
_output_shapes

:@2*
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
ѓ
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
њ
AdamW/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_nameAdamW/output_layer/kernel/m
І
/AdamW/output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/output_layer/kernel/m*
_output_shapes

:2*
dtype0
і
AdamW/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdamW/output_layer/bias/m
Ѓ
-AdamW/output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdamW/output_layer/bias/m*
_output_shapes
:*
dtype0
А
"AdamW/patch_encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	г@*3
shared_name$"AdamW/patch_encoder/dense/kernel/m
џ
6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/m*
_output_shapes
:	г@*
dtype0
ў
 AdamW/patch_encoder/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/patch_encoder/dense/bias/m
Љ
4AdamW/patch_encoder/dense/bias/m/Read/ReadVariableOpReadVariableOp AdamW/patch_encoder/dense/bias/m*
_output_shapes
:@*
dtype0
░
*AdamW/patch_encoder/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/m
Е
>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/m*
_output_shapes

:d@*
dtype0
▓
)AdamW/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/query/kernel/m
Ф
=AdamW/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/query/kernel/m*"
_output_shapes
:@@*
dtype0
ф
'AdamW/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/query/bias/m
Б
;AdamW/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/query/bias/m*
_output_shapes

:@*
dtype0
«
'AdamW/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'AdamW/multi_head_attention/key/kernel/m
Д
;AdamW/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/key/kernel/m*"
_output_shapes
:@@*
dtype0
д
%AdamW/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%AdamW/multi_head_attention/key/bias/m
Ъ
9AdamW/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp%AdamW/multi_head_attention/key/bias/m*
_output_shapes

:@*
dtype0
▓
)AdamW/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/value/kernel/m
Ф
=AdamW/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/value/kernel/m*"
_output_shapes
:@@*
dtype0
ф
'AdamW/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/value/bias/m
Б
;AdamW/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/value/bias/m*
_output_shapes

:@*
dtype0
╚
4AdamW/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*E
shared_name64AdamW/multi_head_attention/attention_output/kernel/m
┴
HAdamW/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:@@*
dtype0
╝
2AdamW/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42AdamW/multi_head_attention/attention_output/bias/m
х
FAdamW/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp2AdamW/multi_head_attention/attention_output/bias/m*
_output_shapes
:@*
dtype0
Х
+AdamW/multi_head_attention_1/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/query/kernel/m
»
?AdamW/multi_head_attention_1/query/kernel/m/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/query/kernel/m*"
_output_shapes
:@@*
dtype0
«
)AdamW/multi_head_attention_1/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/query/bias/m
Д
=AdamW/multi_head_attention_1/query/bias/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/query/bias/m*
_output_shapes

:@*
dtype0
▓
)AdamW/multi_head_attention_1/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention_1/key/kernel/m
Ф
=AdamW/multi_head_attention_1/key/kernel/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/key/kernel/m*"
_output_shapes
:@@*
dtype0
ф
'AdamW/multi_head_attention_1/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention_1/key/bias/m
Б
;AdamW/multi_head_attention_1/key/bias/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention_1/key/bias/m*
_output_shapes

:@*
dtype0
Х
+AdamW/multi_head_attention_1/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/value/kernel/m
»
?AdamW/multi_head_attention_1/value/kernel/m/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/value/kernel/m*"
_output_shapes
:@@*
dtype0
«
)AdamW/multi_head_attention_1/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/value/bias/m
Д
=AdamW/multi_head_attention_1/value/bias/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/value/bias/m*
_output_shapes

:@*
dtype0
╠
6AdamW/multi_head_attention_1/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*G
shared_name86AdamW/multi_head_attention_1/attention_output/kernel/m
┼
JAdamW/multi_head_attention_1/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp6AdamW/multi_head_attention_1/attention_output/kernel/m*"
_output_shapes
:@@*
dtype0
└
4AdamW/multi_head_attention_1/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64AdamW/multi_head_attention_1/attention_output/bias/m
╣
HAdamW/multi_head_attention_1/attention_output/bias/m/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention_1/attention_output/bias/m*
_output_shapes
:@*
dtype0
џ
!AdamW/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!AdamW/layer_normalization/gamma/v
Њ
5AdamW/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp!AdamW/layer_normalization/gamma/v*
_output_shapes
:@*
dtype0
ў
 AdamW/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/layer_normalization/beta/v
Љ
4AdamW/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp AdamW/layer_normalization/beta/v*
_output_shapes
:@*
dtype0
ъ
#AdamW/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_1/gamma/v
Ќ
7AdamW/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
ю
"AdamW/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_1/beta/v
Ћ
6AdamW/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_1/beta/v*
_output_shapes
:@*
dtype0
Ѕ
AdamW/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*'
shared_nameAdamW/dense_1/kernel/v
ѓ
*AdamW/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_1/kernel/v*
_output_shapes
:	@ђ*
dtype0
Ђ
AdamW/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdamW/dense_1/bias/v
z
(AdamW/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_1/bias/v*
_output_shapes	
:ђ*
dtype0
Ѕ
AdamW/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*'
shared_nameAdamW/dense_2/kernel/v
ѓ
*AdamW/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_2/kernel/v*
_output_shapes
:	ђ@*
dtype0
ђ
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
ъ
#AdamW/layer_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_2/gamma/v
Ќ
7AdamW/layer_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
ю
"AdamW/layer_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_2/beta/v
Ћ
6AdamW/layer_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_2/beta/v*
_output_shapes
:@*
dtype0
ъ
#AdamW/layer_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_3/gamma/v
Ќ
7AdamW/layer_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_3/gamma/v*
_output_shapes
:@*
dtype0
ю
"AdamW/layer_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_3/beta/v
Ћ
6AdamW/layer_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_3/beta/v*
_output_shapes
:@*
dtype0
Ѕ
AdamW/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*'
shared_nameAdamW/dense_3/kernel/v
ѓ
*AdamW/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_3/kernel/v*
_output_shapes
:	@ђ*
dtype0
Ђ
AdamW/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*%
shared_nameAdamW/dense_3/bias/v
z
(AdamW/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_3/bias/v*
_output_shapes	
:ђ*
dtype0
Ѕ
AdamW/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*'
shared_nameAdamW/dense_4/kernel/v
ѓ
*AdamW/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_4/kernel/v*
_output_shapes
:	ђ@*
dtype0
ђ
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
ъ
#AdamW/layer_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_4/gamma/v
Ќ
7AdamW/layer_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_4/gamma/v*
_output_shapes
:@*
dtype0
ю
"AdamW/layer_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_4/beta/v
Ћ
6AdamW/layer_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_4/beta/v*
_output_shapes
:@*
dtype0
ј
AdamW/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@
*&
shared_nameAdamW/conv_1/kernel/v
Є
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
ј
AdamW/conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*&
shared_nameAdamW/conv_2/kernel/v
Є
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
ј
AdamW/conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_3/kernel/v
Є
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
ј
AdamW/conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_4/kernel/v
Є
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
ѓ
AdamW/FC_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@2*$
shared_nameAdamW/FC_1/kernel/v
{
'AdamW/FC_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/v*
_output_shapes

:@2*
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
ѓ
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
њ
AdamW/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_nameAdamW/output_layer/kernel/v
І
/AdamW/output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/output_layer/kernel/v*
_output_shapes

:2*
dtype0
і
AdamW/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdamW/output_layer/bias/v
Ѓ
-AdamW/output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdamW/output_layer/bias/v*
_output_shapes
:*
dtype0
А
"AdamW/patch_encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	г@*3
shared_name$"AdamW/patch_encoder/dense/kernel/v
џ
6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/v*
_output_shapes
:	г@*
dtype0
ў
 AdamW/patch_encoder/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/patch_encoder/dense/bias/v
Љ
4AdamW/patch_encoder/dense/bias/v/Read/ReadVariableOpReadVariableOp AdamW/patch_encoder/dense/bias/v*
_output_shapes
:@*
dtype0
░
*AdamW/patch_encoder/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/v
Е
>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/v*
_output_shapes

:d@*
dtype0
▓
)AdamW/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/query/kernel/v
Ф
=AdamW/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/query/kernel/v*"
_output_shapes
:@@*
dtype0
ф
'AdamW/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/query/bias/v
Б
;AdamW/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/query/bias/v*
_output_shapes

:@*
dtype0
«
'AdamW/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'AdamW/multi_head_attention/key/kernel/v
Д
;AdamW/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/key/kernel/v*"
_output_shapes
:@@*
dtype0
д
%AdamW/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%AdamW/multi_head_attention/key/bias/v
Ъ
9AdamW/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp%AdamW/multi_head_attention/key/bias/v*
_output_shapes

:@*
dtype0
▓
)AdamW/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/value/kernel/v
Ф
=AdamW/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/value/kernel/v*"
_output_shapes
:@@*
dtype0
ф
'AdamW/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/value/bias/v
Б
;AdamW/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/value/bias/v*
_output_shapes

:@*
dtype0
╚
4AdamW/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*E
shared_name64AdamW/multi_head_attention/attention_output/kernel/v
┴
HAdamW/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:@@*
dtype0
╝
2AdamW/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42AdamW/multi_head_attention/attention_output/bias/v
х
FAdamW/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp2AdamW/multi_head_attention/attention_output/bias/v*
_output_shapes
:@*
dtype0
Х
+AdamW/multi_head_attention_1/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/query/kernel/v
»
?AdamW/multi_head_attention_1/query/kernel/v/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/query/kernel/v*"
_output_shapes
:@@*
dtype0
«
)AdamW/multi_head_attention_1/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/query/bias/v
Д
=AdamW/multi_head_attention_1/query/bias/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/query/bias/v*
_output_shapes

:@*
dtype0
▓
)AdamW/multi_head_attention_1/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention_1/key/kernel/v
Ф
=AdamW/multi_head_attention_1/key/kernel/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/key/kernel/v*"
_output_shapes
:@@*
dtype0
ф
'AdamW/multi_head_attention_1/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention_1/key/bias/v
Б
;AdamW/multi_head_attention_1/key/bias/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention_1/key/bias/v*
_output_shapes

:@*
dtype0
Х
+AdamW/multi_head_attention_1/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/value/kernel/v
»
?AdamW/multi_head_attention_1/value/kernel/v/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/value/kernel/v*"
_output_shapes
:@@*
dtype0
«
)AdamW/multi_head_attention_1/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/value/bias/v
Д
=AdamW/multi_head_attention_1/value/bias/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/value/bias/v*
_output_shapes

:@*
dtype0
╠
6AdamW/multi_head_attention_1/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*G
shared_name86AdamW/multi_head_attention_1/attention_output/kernel/v
┼
JAdamW/multi_head_attention_1/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp6AdamW/multi_head_attention_1/attention_output/kernel/v*"
_output_shapes
:@@*
dtype0
└
4AdamW/multi_head_attention_1/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64AdamW/multi_head_attention_1/attention_output/bias/v
╣
HAdamW/multi_head_attention_1/attention_output/bias/v/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention_1/attention_output/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
└Е
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ще
value№еBве Bсе
З
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
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
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
layer_with_weights-15
layer-22
layer-23
layer_with_weights-16
layer-24
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
	optimizer
trainable_variables
 regularization_losses
!	variables
"	keras_api
#
signatures
 
R
$trainable_variables
%regularization_losses
&	variables
'	keras_api
z
(
projection
)position_embedding
*trainable_variables
+regularization_losses
,	variables
-	keras_api
q
.axis
	/gamma
0beta
1trainable_variables
2regularization_losses
3	variables
4	keras_api
╗
5_query_dense
6
_key_dense
7_value_dense
8_softmax
9_dropout_layer
:_output_dense
;trainable_variables
<regularization_losses
=	variables
>	keras_api
R
?trainable_variables
@regularization_losses
A	variables
B	keras_api
q
Caxis
	Dgamma
Ebeta
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
h

Jkernel
Kbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
h

Pkernel
Qbias
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
R
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
q
Zaxis
	[gamma
\beta
]trainable_variables
^regularization_losses
_	variables
`	keras_api
╗
a_query_dense
b
_key_dense
c_value_dense
d_softmax
e_dropout_layer
f_output_dense
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
R
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
q
oaxis
	pgamma
qbeta
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
h

vkernel
wbias
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
j

|kernel
}bias
~trainable_variables
regularization_losses
ђ	variables
Ђ	keras_api
V
ѓtrainable_variables
Ѓregularization_losses
ё	variables
Ё	keras_api
x
	єaxis

Єgamma
	ѕbeta
Ѕtrainable_variables
іregularization_losses
І	variables
ї	keras_api
V
Їtrainable_variables
јregularization_losses
Ј	variables
љ	keras_api
n
Љkernel
	њbias
Њtrainable_variables
ћregularization_losses
Ћ	variables
ќ	keras_api
n
Ќkernel
	ўbias
Ўtrainable_variables
џregularization_losses
Џ	variables
ю	keras_api
n
Юkernel
	ъbias
Ъtrainable_variables
аregularization_losses
А	variables
б	keras_api
n
Бkernel
	цbias
Цtrainable_variables
дregularization_losses
Д	variables
е	keras_api
V
Еtrainable_variables
фregularization_losses
Ф	variables
г	keras_api
n
Гkernel
	«bias
»trainable_variables
░regularization_losses
▒	variables
▓	keras_api
V
│trainable_variables
┤regularization_losses
х	variables
Х	keras_api
n
иkernel
	Иbias
╣trainable_variables
║regularization_losses
╗	variables
╝	keras_api
V
йtrainable_variables
Йregularization_losses
┐	variables
└	keras_api
n
┴kernel
	┬bias
├trainable_variables
─regularization_losses
┼	variables
к	keras_api
џ	
	Кiter
╚beta_1
╔beta_2

╩decay
╦learning_rate
╠weight_decay/mі0mІDmїEmЇJmјKmЈPmљQmЉ[mњ\mЊpmћqmЋvmќwmЌ|mў}mЎ	Єmџ	ѕmЏ	Љmю	њmЮ	Ќmъ	ўmЪ	Юmа	ъmА	Бmб	цmБ	Гmц	«mЦ	иmд	ИmД	┴mе	┬mЕ	═mф	╬mФ	¤mг	лmГ	Лm«	мm»	Мm░	нm▒	Нm▓	оm│	Оm┤	пmх	┘mХ	┌mи	█mИ	▄m╣	Пm║	яm╗	▀m╝/vй0vЙDv┐Ev└Jv┴Kv┬Pv├Qv─[v┼\vкpvКqv╚vv╔wv╩|v╦}v╠	Єv═	ѕv╬	Љv¤	њvл	ЌvЛ	ўvм	ЮvМ	ъvн	БvН	цvо	ГvО	«vп	иv┘	Иv┌	┴v█	┬v▄	═vП	╬vя	¤v▀	лvЯ	Лvр	мvР	Мvс	нvС	Нvт	оvТ	Оvу	пvУ	┘vж	┌vЖ	█vв	▄vВ	Пvь	яvЬ	▀v№
▒
═0
╬1
¤2
/3
04
л5
Л6
м7
М8
н9
Н10
о11
О12
D13
E14
J15
K16
P17
Q18
[19
\20
п21
┘22
┌23
█24
▄25
П26
я27
▀28
p29
q30
v31
w32
|33
}34
Є35
ѕ36
Љ37
њ38
Ќ39
ў40
Ю41
ъ42
Б43
ц44
Г45
«46
и47
И48
┴49
┬50
 
▒
═0
╬1
¤2
/3
04
л5
Л6
м7
М8
н9
Н10
о11
О12
D13
E14
J15
K16
P17
Q18
[19
\20
п21
┘22
┌23
█24
▄25
П26
я27
▀28
p29
q30
v31
w32
|33
}34
Є35
ѕ36
Љ37
њ38
Ќ39
ў40
Ю41
ъ42
Б43
ц44
Г45
«46
и47
И48
┴49
┬50
▓
trainable_variables
Яmetrics
рlayers
 regularization_losses
 Рlayer_regularization_losses
!	variables
сlayer_metrics
Сnon_trainable_variables
 
 
 
 
▓
$trainable_variables
тmetrics
Тlayers
 уlayer_regularization_losses
%regularization_losses
&	variables
Уlayer_metrics
жnon_trainable_variables
n
═kernel
	╬bias
Жtrainable_variables
вregularization_losses
В	variables
ь	keras_api
g
¤
embeddings
Ьtrainable_variables
№regularization_losses
­	variables
ы	keras_api

═0
╬1
¤2
 

═0
╬1
¤2
▓
*trainable_variables
Ыmetrics
зlayers
 Зlayer_regularization_losses
+regularization_losses
,	variables
шlayer_metrics
Шnon_trainable_variables
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

/0
01
 

/0
01
▓
1trainable_variables
эmetrics
Эlayers
 щlayer_regularization_losses
2regularization_losses
3	variables
Щlayer_metrics
чnon_trainable_variables
А
Чpartial_output_shape
§full_output_shape
лkernel
	Лbias
■trainable_variables
 regularization_losses
ђ	variables
Ђ	keras_api
А
ѓpartial_output_shape
Ѓfull_output_shape
мkernel
	Мbias
ёtrainable_variables
Ёregularization_losses
є	variables
Є	keras_api
А
ѕpartial_output_shape
Ѕfull_output_shape
нkernel
	Нbias
іtrainable_variables
Іregularization_losses
ї	variables
Ї	keras_api
V
јtrainable_variables
Јregularization_losses
љ	variables
Љ	keras_api
V
њtrainable_variables
Њregularization_losses
ћ	variables
Ћ	keras_api
А
ќpartial_output_shape
Ќfull_output_shape
оkernel
	Оbias
ўtrainable_variables
Ўregularization_losses
џ	variables
Џ	keras_api
@
л0
Л1
м2
М3
н4
Н5
о6
О7
 
@
л0
Л1
м2
М3
н4
Н5
о6
О7
▓
;trainable_variables
юmetrics
Юlayers
 ъlayer_regularization_losses
<regularization_losses
=	variables
Ъlayer_metrics
аnon_trainable_variables
 
 
 
▓
?trainable_variables
Аmetrics
бlayers
 Бlayer_regularization_losses
@regularization_losses
A	variables
цlayer_metrics
Цnon_trainable_variables
 
fd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
▓
Ftrainable_variables
дmetrics
Дlayers
 еlayer_regularization_losses
Gregularization_losses
H	variables
Еlayer_metrics
фnon_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
▓
Ltrainable_variables
Фmetrics
гlayers
 Гlayer_regularization_losses
Mregularization_losses
N	variables
«layer_metrics
»non_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1
 

P0
Q1
▓
Rtrainable_variables
░metrics
▒layers
 ▓layer_regularization_losses
Sregularization_losses
T	variables
│layer_metrics
┤non_trainable_variables
 
 
 
▓
Vtrainable_variables
хmetrics
Хlayers
 иlayer_regularization_losses
Wregularization_losses
X	variables
Иlayer_metrics
╣non_trainable_variables
 
fd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
 

[0
\1
▓
]trainable_variables
║metrics
╗layers
 ╝layer_regularization_losses
^regularization_losses
_	variables
йlayer_metrics
Йnon_trainable_variables
А
┐partial_output_shape
└full_output_shape
пkernel
	┘bias
┴trainable_variables
┬regularization_losses
├	variables
─	keras_api
А
┼partial_output_shape
кfull_output_shape
┌kernel
	█bias
Кtrainable_variables
╚regularization_losses
╔	variables
╩	keras_api
А
╦partial_output_shape
╠full_output_shape
▄kernel
	Пbias
═trainable_variables
╬regularization_losses
¤	variables
л	keras_api
V
Лtrainable_variables
мregularization_losses
М	variables
н	keras_api
V
Нtrainable_variables
оregularization_losses
О	variables
п	keras_api
А
┘partial_output_shape
┌full_output_shape
яkernel
	▀bias
█trainable_variables
▄regularization_losses
П	variables
я	keras_api
@
п0
┘1
┌2
█3
▄4
П5
я6
▀7
 
@
п0
┘1
┌2
█3
▄4
П5
я6
▀7
▓
gtrainable_variables
▀metrics
Яlayers
 рlayer_regularization_losses
hregularization_losses
i	variables
Рlayer_metrics
сnon_trainable_variables
 
 
 
▓
ktrainable_variables
Сmetrics
тlayers
 Тlayer_regularization_losses
lregularization_losses
m	variables
уlayer_metrics
Уnon_trainable_variables
 
fd
VARIABLE_VALUElayer_normalization_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
 

p0
q1
▓
rtrainable_variables
жmetrics
Жlayers
 вlayer_regularization_losses
sregularization_losses
t	variables
Вlayer_metrics
ьnon_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
 

v0
w1
▓
xtrainable_variables
Ьmetrics
№layers
 ­layer_regularization_losses
yregularization_losses
z	variables
ыlayer_metrics
Ыnon_trainable_variables
[Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_4/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

|0
}1
 

|0
}1
│
~trainable_variables
зmetrics
Зlayers
 шlayer_regularization_losses
regularization_losses
ђ	variables
Шlayer_metrics
эnon_trainable_variables
 
 
 
х
ѓtrainable_variables
Эmetrics
щlayers
 Щlayer_regularization_losses
Ѓregularization_losses
ё	variables
чlayer_metrics
Чnon_trainable_variables
 
ge
VARIABLE_VALUElayer_normalization_4/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer_normalization_4/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE

Є0
ѕ1
 

Є0
ѕ1
х
Ѕtrainable_variables
§metrics
■layers
  layer_regularization_losses
іregularization_losses
І	variables
ђlayer_metrics
Ђnon_trainable_variables
 
 
 
х
Їtrainable_variables
ѓmetrics
Ѓlayers
 ёlayer_regularization_losses
јregularization_losses
Ј	variables
Ёlayer_metrics
єnon_trainable_variables
ZX
VARIABLE_VALUEconv_1/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_1/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

Љ0
њ1
 

Љ0
њ1
х
Њtrainable_variables
Єmetrics
ѕlayers
 Ѕlayer_regularization_losses
ћregularization_losses
Ћ	variables
іlayer_metrics
Іnon_trainable_variables
ZX
VARIABLE_VALUEconv_2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

Ќ0
ў1
 

Ќ0
ў1
х
Ўtrainable_variables
їmetrics
Їlayers
 јlayer_regularization_losses
џregularization_losses
Џ	variables
Јlayer_metrics
љnon_trainable_variables
ZX
VARIABLE_VALUEconv_3/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_3/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

Ю0
ъ1
 

Ю0
ъ1
х
Ъtrainable_variables
Љmetrics
њlayers
 Њlayer_regularization_losses
аregularization_losses
А	variables
ћlayer_metrics
Ћnon_trainable_variables
ZX
VARIABLE_VALUEconv_4/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv_4/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

Б0
ц1
 

Б0
ц1
х
Цtrainable_variables
ќmetrics
Ќlayers
 ўlayer_regularization_losses
дregularization_losses
Д	variables
Ўlayer_metrics
џnon_trainable_variables
 
 
 
х
Еtrainable_variables
Џmetrics
юlayers
 Юlayer_regularization_losses
фregularization_losses
Ф	variables
ъlayer_metrics
Ъnon_trainable_variables
XV
VARIABLE_VALUEFC_1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	FC_1/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

Г0
«1
 

Г0
«1
х
»trainable_variables
аmetrics
Аlayers
 бlayer_regularization_losses
░regularization_losses
▒	variables
Бlayer_metrics
цnon_trainable_variables
 
 
 
х
│trainable_variables
Цmetrics
дlayers
 Дlayer_regularization_losses
┤regularization_losses
х	variables
еlayer_metrics
Еnon_trainable_variables
XV
VARIABLE_VALUEFC_2/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	FC_2/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE

и0
И1
 

и0
И1
х
╣trainable_variables
фmetrics
Фlayers
 гlayer_regularization_losses
║regularization_losses
╗	variables
Гlayer_metrics
«non_trainable_variables
 
 
 
х
йtrainable_variables
»metrics
░layers
 ▒layer_regularization_losses
Йregularization_losses
┐	variables
▓layer_metrics
│non_trainable_variables
`^
VARIABLE_VALUEoutput_layer/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEoutput_layer/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

┴0
┬1
 

┴0
┬1
х
├trainable_variables
┤metrics
хlayers
 Хlayer_regularization_losses
─regularization_losses
┼	variables
иlayer_metrics
Иnon_trainable_variables
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
VARIABLE_VALUEpatch_encoder/dense/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEpatch_encoder/dense/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE"patch_encoder/embedding/embeddings0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!multi_head_attention/query/kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEmulti_head_attention/query/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEmulti_head_attention/key/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEmulti_head_attention/key/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!multi_head_attention/value/kernel0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmulti_head_attention/value/bias1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,multi_head_attention/attention_output/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*multi_head_attention/attention_output/bias1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#multi_head_attention_1/query/kernel1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!multi_head_attention_1/query/bias1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!multi_head_attention_1/key/kernel1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmulti_head_attention_1/key/bias1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#multi_head_attention_1/value/kernel1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!multi_head_attention_1/value/bias1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.multi_head_attention_1/attention_output/kernel1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,multi_head_attention_1/attention_output/bias1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUE

╣0
║1
я
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
 
 
 
 
 
 
 
 

═0
╬1
 

═0
╬1
х
Жtrainable_variables
╗metrics
╝layers
 йlayer_regularization_losses
вregularization_losses
В	variables
Йlayer_metrics
┐non_trainable_variables

¤0
 

¤0
х
Ьtrainable_variables
└metrics
┴layers
 ┬layer_regularization_losses
№regularization_losses
­	variables
├layer_metrics
─non_trainable_variables
 

(0
)1
 
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
л0
Л1
 

л0
Л1
х
■trainable_variables
┼metrics
кlayers
 Кlayer_regularization_losses
 regularization_losses
ђ	variables
╚layer_metrics
╔non_trainable_variables
 
 

м0
М1
 

м0
М1
х
ёtrainable_variables
╩metrics
╦layers
 ╠layer_regularization_losses
Ёregularization_losses
є	variables
═layer_metrics
╬non_trainable_variables
 
 

н0
Н1
 

н0
Н1
х
іtrainable_variables
¤metrics
лlayers
 Лlayer_regularization_losses
Іregularization_losses
ї	variables
мlayer_metrics
Мnon_trainable_variables
 
 
 
х
јtrainable_variables
нmetrics
Нlayers
 оlayer_regularization_losses
Јregularization_losses
љ	variables
Оlayer_metrics
пnon_trainable_variables
 
 
 
х
њtrainable_variables
┘metrics
┌layers
 █layer_regularization_losses
Њregularization_losses
ћ	variables
▄layer_metrics
Пnon_trainable_variables
 
 

о0
О1
 

о0
О1
х
ўtrainable_variables
яmetrics
▀layers
 Яlayer_regularization_losses
Ўregularization_losses
џ	variables
рlayer_metrics
Рnon_trainable_variables
 
*
50
61
72
83
94
:5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
п0
┘1
 

п0
┘1
х
┴trainable_variables
сmetrics
Сlayers
 тlayer_regularization_losses
┬regularization_losses
├	variables
Тlayer_metrics
уnon_trainable_variables
 
 

┌0
█1
 

┌0
█1
х
Кtrainable_variables
Уmetrics
жlayers
 Жlayer_regularization_losses
╚regularization_losses
╔	variables
вlayer_metrics
Вnon_trainable_variables
 
 

▄0
П1
 

▄0
П1
х
═trainable_variables
ьmetrics
Ьlayers
 №layer_regularization_losses
╬regularization_losses
¤	variables
­layer_metrics
ыnon_trainable_variables
 
 
 
х
Лtrainable_variables
Ыmetrics
зlayers
 Зlayer_regularization_losses
мregularization_losses
М	variables
шlayer_metrics
Шnon_trainable_variables
 
 
 
х
Нtrainable_variables
эmetrics
Эlayers
 щlayer_regularization_losses
оregularization_losses
О	variables
Щlayer_metrics
чnon_trainable_variables
 
 

я0
▀1
 

я0
▀1
х
█trainable_variables
Чmetrics
§layers
 ■layer_regularization_losses
▄regularization_losses
П	variables
 layer_metrics
ђnon_trainable_variables
 
*
a0
b1
c2
d3
e4
f5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

Ђtotal

ѓcount
Ѓ	variables
ё	keras_api
I

Ёtotal

єcount
Є
_fn_kwargs
ѕ	variables
Ѕ	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
Ђ0
ѓ1

Ѓ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ё0
є1

ѕ	variables
Ѕє
VARIABLE_VALUE!AdamW/layer_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Єё
VARIABLE_VALUE AdamW/layer_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#AdamW/layer_normalization_3/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"AdamW/layer_normalization_3/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_3/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_3/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_4/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_4/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#AdamW/layer_normalization_4/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"AdamW/layer_normalization_4/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/conv_1/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/conv_1/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/conv_2/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/conv_2/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/conv_3/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/conv_3/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/conv_4/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/conv_4/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_1/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_1/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_2/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_2/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUEAdamW/output_layer/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdamW/output_layer/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ўЋ
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ќЊ
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Јї
VARIABLE_VALUE+AdamW/multi_head_attention_1/query/kernel/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE)AdamW/multi_head_attention_1/query/bias/mMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE)AdamW/multi_head_attention_1/key/kernel/mMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE'AdamW/multi_head_attention_1/key/bias/mMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Јї
VARIABLE_VALUE+AdamW/multi_head_attention_1/value/kernel/mMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE)AdamW/multi_head_attention_1/value/bias/mMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
џЌ
VARIABLE_VALUE6AdamW/multi_head_attention_1/attention_output/kernel/mMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ўЋ
VARIABLE_VALUE4AdamW/multi_head_attention_1/attention_output/bias/mMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE!AdamW/layer_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Єё
VARIABLE_VALUE AdamW/layer_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE#AdamW/layer_normalization_3/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѕє
VARIABLE_VALUE"AdamW/layer_normalization_3/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_3/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_3/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_4/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_4/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#AdamW/layer_normalization_4/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE"AdamW/layer_normalization_4/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/conv_1/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/conv_1/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/conv_2/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/conv_2/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/conv_3/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/conv_3/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/conv_4/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/conv_4/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_1/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_1/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_2/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_2/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUEAdamW/output_layer/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUEAdamW/output_layer/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ѓђ
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
іЄ
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ўЋ
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ќЊ
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Јї
VARIABLE_VALUE+AdamW/multi_head_attention_1/query/kernel/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE)AdamW/multi_head_attention_1/query/bias/vMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE)AdamW/multi_head_attention_1/key/kernel/vMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Іѕ
VARIABLE_VALUE'AdamW/multi_head_attention_1/key/bias/vMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Јї
VARIABLE_VALUE+AdamW/multi_head_attention_1/value/kernel/vMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Їі
VARIABLE_VALUE)AdamW/multi_head_attention_1/value/bias/vMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
џЌ
VARIABLE_VALUE6AdamW/multi_head_attention_1/attention_output/kernel/vMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ўЋ
VARIABLE_VALUE4AdamW/multi_head_attention_1/attention_output/bias/vMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ј
serving_default_input_layerPlaceholder*/
_output_shapes
:         dd*
dtype0*$
shape:         dd
Б
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerpatch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddingslayer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biaslayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/beta#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biaslayer_normalization_3/gammalayer_normalization_3/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/biaslayer_normalization_4/gammalayer_normalization_4/betaconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasconv_4/kernelconv_4/biasFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_623807
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┴B
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp/layer_normalization_3/gamma/Read/ReadVariableOp.layer_normalization_3/beta/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOp!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp!conv_4/kernel/Read/ReadVariableOpconv_4/bias/Read/ReadVariableOpFC_1/kernel/Read/ReadVariableOpFC_1/bias/Read/ReadVariableOpFC_2/kernel/Read/ReadVariableOpFC_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdamW/iter/Read/ReadVariableOp AdamW/beta_1/Read/ReadVariableOp AdamW/beta_2/Read/ReadVariableOpAdamW/decay/Read/ReadVariableOp'AdamW/learning_rate/Read/ReadVariableOp&AdamW/weight_decay/Read/ReadVariableOp.patch_encoder/dense/kernel/Read/ReadVariableOp,patch_encoder/dense/bias/Read/ReadVariableOp6patch_encoder/embedding/embeddings/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOp7multi_head_attention_1/query/kernel/Read/ReadVariableOp5multi_head_attention_1/query/bias/Read/ReadVariableOp5multi_head_attention_1/key/kernel/Read/ReadVariableOp3multi_head_attention_1/key/bias/Read/ReadVariableOp7multi_head_attention_1/value/kernel/Read/ReadVariableOp5multi_head_attention_1/value/bias/Read/ReadVariableOpBmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_1/attention_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp5AdamW/layer_normalization/gamma/m/Read/ReadVariableOp4AdamW/layer_normalization/beta/m/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/m/Read/ReadVariableOp*AdamW/dense_1/kernel/m/Read/ReadVariableOp(AdamW/dense_1/bias/m/Read/ReadVariableOp*AdamW/dense_2/kernel/m/Read/ReadVariableOp(AdamW/dense_2/bias/m/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/m/Read/ReadVariableOp7AdamW/layer_normalization_3/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_3/beta/m/Read/ReadVariableOp*AdamW/dense_3/kernel/m/Read/ReadVariableOp(AdamW/dense_3/bias/m/Read/ReadVariableOp*AdamW/dense_4/kernel/m/Read/ReadVariableOp(AdamW/dense_4/bias/m/Read/ReadVariableOp7AdamW/layer_normalization_4/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_4/beta/m/Read/ReadVariableOp)AdamW/conv_1/kernel/m/Read/ReadVariableOp'AdamW/conv_1/bias/m/Read/ReadVariableOp)AdamW/conv_2/kernel/m/Read/ReadVariableOp'AdamW/conv_2/bias/m/Read/ReadVariableOp)AdamW/conv_3/kernel/m/Read/ReadVariableOp'AdamW/conv_3/bias/m/Read/ReadVariableOp)AdamW/conv_4/kernel/m/Read/ReadVariableOp'AdamW/conv_4/bias/m/Read/ReadVariableOp'AdamW/FC_1/kernel/m/Read/ReadVariableOp%AdamW/FC_1/bias/m/Read/ReadVariableOp'AdamW/FC_2/kernel/m/Read/ReadVariableOp%AdamW/FC_2/bias/m/Read/ReadVariableOp/AdamW/output_layer/kernel/m/Read/ReadVariableOp-AdamW/output_layer/bias/m/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/m/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/m/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/m/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/m/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/m/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/m/Read/ReadVariableOp?AdamW/multi_head_attention_1/query/kernel/m/Read/ReadVariableOp=AdamW/multi_head_attention_1/query/bias/m/Read/ReadVariableOp=AdamW/multi_head_attention_1/key/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention_1/key/bias/m/Read/ReadVariableOp?AdamW/multi_head_attention_1/value/kernel/m/Read/ReadVariableOp=AdamW/multi_head_attention_1/value/bias/m/Read/ReadVariableOpJAdamW/multi_head_attention_1/attention_output/kernel/m/Read/ReadVariableOpHAdamW/multi_head_attention_1/attention_output/bias/m/Read/ReadVariableOp5AdamW/layer_normalization/gamma/v/Read/ReadVariableOp4AdamW/layer_normalization/beta/v/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/v/Read/ReadVariableOp*AdamW/dense_1/kernel/v/Read/ReadVariableOp(AdamW/dense_1/bias/v/Read/ReadVariableOp*AdamW/dense_2/kernel/v/Read/ReadVariableOp(AdamW/dense_2/bias/v/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/v/Read/ReadVariableOp7AdamW/layer_normalization_3/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_3/beta/v/Read/ReadVariableOp*AdamW/dense_3/kernel/v/Read/ReadVariableOp(AdamW/dense_3/bias/v/Read/ReadVariableOp*AdamW/dense_4/kernel/v/Read/ReadVariableOp(AdamW/dense_4/bias/v/Read/ReadVariableOp7AdamW/layer_normalization_4/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_4/beta/v/Read/ReadVariableOp)AdamW/conv_1/kernel/v/Read/ReadVariableOp'AdamW/conv_1/bias/v/Read/ReadVariableOp)AdamW/conv_2/kernel/v/Read/ReadVariableOp'AdamW/conv_2/bias/v/Read/ReadVariableOp)AdamW/conv_3/kernel/v/Read/ReadVariableOp'AdamW/conv_3/bias/v/Read/ReadVariableOp)AdamW/conv_4/kernel/v/Read/ReadVariableOp'AdamW/conv_4/bias/v/Read/ReadVariableOp'AdamW/FC_1/kernel/v/Read/ReadVariableOp%AdamW/FC_1/bias/v/Read/ReadVariableOp'AdamW/FC_2/kernel/v/Read/ReadVariableOp%AdamW/FC_2/bias/v/Read/ReadVariableOp/AdamW/output_layer/kernel/v/Read/ReadVariableOp-AdamW/output_layer/bias/v/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/v/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/v/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/v/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/v/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/v/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/v/Read/ReadVariableOp?AdamW/multi_head_attention_1/query/kernel/v/Read/ReadVariableOp=AdamW/multi_head_attention_1/query/bias/v/Read/ReadVariableOp=AdamW/multi_head_attention_1/key/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention_1/key/bias/v/Read/ReadVariableOp?AdamW/multi_head_attention_1/value/kernel/v/Read/ReadVariableOp=AdamW/multi_head_attention_1/value/bias/v/Read/ReadVariableOpJAdamW/multi_head_attention_1/attention_output/kernel/v/Read/ReadVariableOpHAdamW/multi_head_attention_1/attention_output/bias/v/Read/ReadVariableOpConst*│
TinФ
е2Ц	*
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
GPU 2J 8ѓ *(
f#R!
__inference__traced_save_626228
ђ)
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betalayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/betalayer_normalization_3/gammalayer_normalization_3/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/biaslayer_normalization_4/gammalayer_normalization_4/betaconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasconv_4/kernelconv_4/biasFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias
AdamW/iterAdamW/beta_1AdamW/beta_2AdamW/decayAdamW/learning_rateAdamW/weight_decaypatch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddings!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biastotalcounttotal_1count_1!AdamW/layer_normalization/gamma/m AdamW/layer_normalization/beta/m#AdamW/layer_normalization_1/gamma/m"AdamW/layer_normalization_1/beta/mAdamW/dense_1/kernel/mAdamW/dense_1/bias/mAdamW/dense_2/kernel/mAdamW/dense_2/bias/m#AdamW/layer_normalization_2/gamma/m"AdamW/layer_normalization_2/beta/m#AdamW/layer_normalization_3/gamma/m"AdamW/layer_normalization_3/beta/mAdamW/dense_3/kernel/mAdamW/dense_3/bias/mAdamW/dense_4/kernel/mAdamW/dense_4/bias/m#AdamW/layer_normalization_4/gamma/m"AdamW/layer_normalization_4/beta/mAdamW/conv_1/kernel/mAdamW/conv_1/bias/mAdamW/conv_2/kernel/mAdamW/conv_2/bias/mAdamW/conv_3/kernel/mAdamW/conv_3/bias/mAdamW/conv_4/kernel/mAdamW/conv_4/bias/mAdamW/FC_1/kernel/mAdamW/FC_1/bias/mAdamW/FC_2/kernel/mAdamW/FC_2/bias/mAdamW/output_layer/kernel/mAdamW/output_layer/bias/m"AdamW/patch_encoder/dense/kernel/m AdamW/patch_encoder/dense/bias/m*AdamW/patch_encoder/embedding/embeddings/m)AdamW/multi_head_attention/query/kernel/m'AdamW/multi_head_attention/query/bias/m'AdamW/multi_head_attention/key/kernel/m%AdamW/multi_head_attention/key/bias/m)AdamW/multi_head_attention/value/kernel/m'AdamW/multi_head_attention/value/bias/m4AdamW/multi_head_attention/attention_output/kernel/m2AdamW/multi_head_attention/attention_output/bias/m+AdamW/multi_head_attention_1/query/kernel/m)AdamW/multi_head_attention_1/query/bias/m)AdamW/multi_head_attention_1/key/kernel/m'AdamW/multi_head_attention_1/key/bias/m+AdamW/multi_head_attention_1/value/kernel/m)AdamW/multi_head_attention_1/value/bias/m6AdamW/multi_head_attention_1/attention_output/kernel/m4AdamW/multi_head_attention_1/attention_output/bias/m!AdamW/layer_normalization/gamma/v AdamW/layer_normalization/beta/v#AdamW/layer_normalization_1/gamma/v"AdamW/layer_normalization_1/beta/vAdamW/dense_1/kernel/vAdamW/dense_1/bias/vAdamW/dense_2/kernel/vAdamW/dense_2/bias/v#AdamW/layer_normalization_2/gamma/v"AdamW/layer_normalization_2/beta/v#AdamW/layer_normalization_3/gamma/v"AdamW/layer_normalization_3/beta/vAdamW/dense_3/kernel/vAdamW/dense_3/bias/vAdamW/dense_4/kernel/vAdamW/dense_4/bias/v#AdamW/layer_normalization_4/gamma/v"AdamW/layer_normalization_4/beta/vAdamW/conv_1/kernel/vAdamW/conv_1/bias/vAdamW/conv_2/kernel/vAdamW/conv_2/bias/vAdamW/conv_3/kernel/vAdamW/conv_3/bias/vAdamW/conv_4/kernel/vAdamW/conv_4/bias/vAdamW/FC_1/kernel/vAdamW/FC_1/bias/vAdamW/FC_2/kernel/vAdamW/FC_2/bias/vAdamW/output_layer/kernel/vAdamW/output_layer/bias/v"AdamW/patch_encoder/dense/kernel/v AdamW/patch_encoder/dense/bias/v*AdamW/patch_encoder/embedding/embeddings/v)AdamW/multi_head_attention/query/kernel/v'AdamW/multi_head_attention/query/bias/v'AdamW/multi_head_attention/key/kernel/v%AdamW/multi_head_attention/key/bias/v)AdamW/multi_head_attention/value/kernel/v'AdamW/multi_head_attention/value/bias/v4AdamW/multi_head_attention/attention_output/kernel/v2AdamW/multi_head_attention/attention_output/bias/v+AdamW/multi_head_attention_1/query/kernel/v)AdamW/multi_head_attention_1/query/bias/v)AdamW/multi_head_attention_1/key/kernel/v'AdamW/multi_head_attention_1/key/bias/v+AdamW/multi_head_attention_1/value/kernel/v)AdamW/multi_head_attention_1/value/bias/v6AdamW/multi_head_attention_1/attention_output/kernel/v4AdamW/multi_head_attention_1/attention_output/bias/v*▓
Tinф
Д2ц*
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
GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_626727џЃ*
ќр
Д.
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_624418

inputsH
5patch_encoder_dense_tensordot_readvariableop_resource:	г@A
3patch_encoder_dense_biasadd_readvariableop_resource:@A
/patch_encoder_embedding_embedding_lookup_624064:d@G
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
)dense_1_tensordot_readvariableop_resource:	@ђ6
'dense_1_biasadd_readvariableop_resource:	ђ<
)dense_2_tensordot_readvariableop_resource:	ђ@5
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
)dense_3_tensordot_readvariableop_resource:	@ђ6
'dense_3_biasadd_readvariableop_resource:	ђ<
)dense_4_tensordot_readvariableop_resource:	ђ@5
'dense_4_biasadd_readvariableop_resource:@I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_4_batchnorm_readvariableop_resource:@?
%conv_1_conv2d_readvariableop_resource:@
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
&conv_4_biasadd_readvariableop_resource:5
#fc_1_matmul_readvariableop_resource:@22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identityѕбFC_1/BiasAdd/ReadVariableOpбFC_1/MatMul/ReadVariableOpбFC_2/BiasAdd/ReadVariableOpбFC_2/MatMul/ReadVariableOpбconv_1/BiasAdd/ReadVariableOpбconv_1/Conv2D/ReadVariableOpбconv_2/BiasAdd/ReadVariableOpбconv_2/Conv2D/ReadVariableOpбconv_3/BiasAdd/ReadVariableOpбconv_3/Conv2D/ReadVariableOpбconv_4/BiasAdd/ReadVariableOpбconv_4/Conv2D/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpб dense_1/Tensordot/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpб dense_2/Tensordot/ReadVariableOpбdense_3/BiasAdd/ReadVariableOpб dense_3/Tensordot/ReadVariableOpбdense_4/BiasAdd/ReadVariableOpб dense_4/Tensordot/ReadVariableOpб,layer_normalization/batchnorm/ReadVariableOpб0layer_normalization/batchnorm/mul/ReadVariableOpб.layer_normalization_1/batchnorm/ReadVariableOpб2layer_normalization_1/batchnorm/mul/ReadVariableOpб.layer_normalization_2/batchnorm/ReadVariableOpб2layer_normalization_2/batchnorm/mul/ReadVariableOpб.layer_normalization_3/batchnorm/ReadVariableOpб2layer_normalization_3/batchnorm/mul/ReadVariableOpб.layer_normalization_4/batchnorm/ReadVariableOpб2layer_normalization_4/batchnorm/mul/ReadVariableOpб8multi_head_attention/attention_output/add/ReadVariableOpбBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpб+multi_head_attention/key/add/ReadVariableOpб5multi_head_attention/key/einsum/Einsum/ReadVariableOpб-multi_head_attention/query/add/ReadVariableOpб7multi_head_attention/query/einsum/Einsum/ReadVariableOpб-multi_head_attention/value/add/ReadVariableOpб7multi_head_attention/value/einsum/Einsum/ReadVariableOpб:multi_head_attention_1/attention_output/add/ReadVariableOpбDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpб-multi_head_attention_1/key/add/ReadVariableOpб7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpб/multi_head_attention_1/query/add/ReadVariableOpб9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpб/multi_head_attention_1/value/add/ReadVariableOpб9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpб#output_layer/BiasAdd/ReadVariableOpб"output_layer/MatMul/ReadVariableOpб*patch_encoder/dense/BiasAdd/ReadVariableOpб,patch_encoder/dense/Tensordot/ReadVariableOpб(patch_encoder/embedding/embedding_lookupT
patches/ShapeShapeinputs*
T0*
_output_shapes
:2
patches/Shapeё
patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
patches/strided_slice/stackѕ
patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_1ѕ
patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_2њ
patches/strided_sliceStridedSlicepatches/Shape:output:0$patches/strided_slice/stack:output:0&patches/strided_slice/stack_1:output:0&patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
patches/strided_sliceС
patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*0
_output_shapes
:         

г*
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
         2
patches/Reshape/shape/1u
patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :г2
patches/Reshape/shape/2╚
patches/Reshape/shapePackpatches/strided_slice:output:0 patches/Reshape/shape/1:output:0 patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
patches/Reshape/shape┤
patches/ReshapeReshape%patches/ExtractImagePatches:patches:0patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  г2
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
value	B :d2
patch_encoder/range/limitx
patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder/range/delta╗
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes
:d2
patch_encoder/rangeМ
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	г@*
dtype02.
,patch_encoder/dense/Tensordot/ReadVariableOpњ
"patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"patch_encoder/dense/Tensordot/axesЎ
"patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"patch_encoder/dense/Tensordot/freeњ
#patch_encoder/dense/Tensordot/ShapeShapepatches/Reshape:output:0*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/Shapeю
+patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/GatherV2/axisх
&patch_encoder/dense/Tensordot/GatherV2GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/free:output:04patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/GatherV2а
-patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-patch_encoder/dense/Tensordot/GatherV2_1/axis╗
(patch_encoder/dense/Tensordot/GatherV2_1GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/axes:output:06patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(patch_encoder/dense/Tensordot/GatherV2_1ћ
#patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#patch_encoder/dense/Tensordot/Constл
"patch_encoder/dense/Tensordot/ProdProd/patch_encoder/dense/Tensordot/GatherV2:output:0,patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"patch_encoder/dense/Tensordot/Prodў
%patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%patch_encoder/dense/Tensordot/Const_1п
$patch_encoder/dense/Tensordot/Prod_1Prod1patch_encoder/dense/Tensordot/GatherV2_1:output:0.patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$patch_encoder/dense/Tensordot/Prod_1ў
)patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)patch_encoder/dense/Tensordot/concat/axisћ
$patch_encoder/dense/Tensordot/concatConcatV2+patch_encoder/dense/Tensordot/free:output:0+patch_encoder/dense/Tensordot/axes:output:02patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$patch_encoder/dense/Tensordot/concat▄
#patch_encoder/dense/Tensordot/stackPack+patch_encoder/dense/Tensordot/Prod:output:0-patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/stackУ
'patch_encoder/dense/Tensordot/transpose	Transposepatches/Reshape:output:0-patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  г2)
'patch_encoder/dense/Tensordot/transpose№
%patch_encoder/dense/Tensordot/ReshapeReshape+patch_encoder/dense/Tensordot/transpose:y:0,patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2'
%patch_encoder/dense/Tensordot/ReshapeЬ
$patch_encoder/dense/Tensordot/MatMulMatMul.patch_encoder/dense/Tensordot/Reshape:output:04patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2&
$patch_encoder/dense/Tensordot/MatMulў
%patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2'
%patch_encoder/dense/Tensordot/Const_2ю
+patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/concat_1/axisА
&patch_encoder/dense/Tensordot/concat_1ConcatV2/patch_encoder/dense/Tensordot/GatherV2:output:0.patch_encoder/dense/Tensordot/Const_2:output:04patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/concat_1ж
patch_encoder/dense/TensordotReshape.patch_encoder/dense/Tensordot/MatMul:product:0/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @2
patch_encoder/dense/Tensordot╚
*patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp3patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*patch_encoder/dense/BiasAdd/ReadVariableOpЯ
patch_encoder/dense/BiasAddBiasAdd&patch_encoder/dense/Tensordot:output:02patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @2
patch_encoder/dense/BiasAddС
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_624064patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/624064*
_output_shapes

:d@*
dtype02*
(patch_encoder/embedding/embedding_lookup└
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/624064*
_output_shapes

:d@23
1patch_encoder/embedding/embedding_lookup/Identity█
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:d@25
3patch_encoder/embedding/embedding_lookup/Identity_1╔
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         d@2
patch_encoder/add▓
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indicesу
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2"
 layer_normalization/moments/mean┼
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         d2*
(layer_normalization/moments/StopGradientз
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2/
-layer_normalization/moments/SquaredDifference║
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indicesЈ
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2&
$layer_normalization/moments/varianceЈ
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52%
#layer_normalization/batchnorm/add/yР
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2#
!layer_normalization/batchnorm/add░
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2%
#layer_normalization/batchnorm/Rsqrt┌
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOpТ
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2#
!layer_normalization/batchnorm/mul┼
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization/batchnorm/mul_1┘
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization/batchnorm/mul_2╬
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOpР
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2#
!layer_normalization/batchnorm/sub┘
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization/batchnorm/add_1э
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOpе
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/EinsumН
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOpь
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2 
multi_head_attention/query/addы
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOpб
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum¤
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOpт
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
multi_head_attention/key/addэ
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOpе
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/EinsumН
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOpь
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/yЙ
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:         d@2
multi_head_attention/MulЗ
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/EinsumЙ
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2&
$multi_head_attention/softmax/Softmax─
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:         dd2'
%multi_head_attention/dropout/Identityї
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsumў
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp╦
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/EinsumЫ
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOpЋ
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2+
)multi_head_attention/attention_output/addЌ
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*+
_output_shapes
:         d@2	
add/addХ
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indicesс
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2$
"layer_normalization_1/moments/mean╦
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         d2,
*layer_normalization_1/moments/StopGradient№
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@21
/layer_normalization_1/moments/SquaredDifferenceЙ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indicesЌ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2(
&layer_normalization_1/moments/varianceЊ
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52'
%layer_normalization_1/batchnorm/add/yЖ
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2%
#layer_normalization_1/batchnorm/addХ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2'
%layer_normalization_1/batchnorm/RsqrtЯ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOpЬ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_1/batchnorm/mul┴
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_1/batchnorm/mul_1р
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_1/batchnorm/mul_2н
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOpЖ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_1/batchnorm/subр
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_1/batchnorm/add_1»
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axesЂ
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/freeІ
dense_1/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shapeё
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axisщ
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2ѕ
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis 
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
dense_1/Tensordot/Constа
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prodђ
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1е
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1ђ
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axisп
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concatг
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack╦
dense_1/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:         d@2
dense_1/Tensordot/transpose┐
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_1/Tensordot/Reshape┐
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/Tensordot/MatMulЂ
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ2
dense_1/Tensordot/Const_2ё
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axisт
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1▒
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2
dense_1/TensordotЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_1/BiasAdd/ReadVariableOpе
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/xЎ
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
dense_1/Gelu/Cast/xд
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/truediv|
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dense_1/Gelu/add/xЌ
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/addњ
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/mul_1»
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axesЂ
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
dense_2/Tensordot/Shapeё
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axisщ
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2ѕ
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis 
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
dense_2/Tensordot/Constа
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prodђ
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1е
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1ђ
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axisп
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concatг
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack╣
dense_2/Tensordot/transpose	Transposedense_1/Gelu/mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:         dђ2
dense_2/Tensordot/transpose┐
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_2/Tensordot/ReshapeЙ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_2/Tensordot/MatMulђ
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_2/Tensordot/Const_2ё
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axisт
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1░
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2
dense_2/Tensordotц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOpД
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/xў
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
dense_2/Gelu/Cast/xЦ
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/truediv{
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dense_2/Gelu/add/xќ
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/addЉ
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/mul_1z
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*+
_output_shapes
:         d@2
	add_1/addХ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesт
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2$
"layer_normalization_2/moments/mean╦
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:         d2,
*layer_normalization_2/moments/StopGradientы
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@21
/layer_normalization_2/moments/SquaredDifferenceЙ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesЌ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2(
&layer_normalization_2/moments/varianceЊ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52'
%layer_normalization_2/batchnorm/add/yЖ
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2%
#layer_normalization_2/batchnorm/addХ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2'
%layer_normalization_2/batchnorm/RsqrtЯ
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpЬ
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_2/batchnorm/mul├
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_2/batchnorm/mul_1р
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_2/batchnorm/mul_2н
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpЖ
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_2/batchnorm/subр
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_2/batchnorm/add_1§
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp░
*multi_head_attention_1/query/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsum█
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/query/add/ReadVariableOpш
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2"
 multi_head_attention_1/query/addэ
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpф
(multi_head_attention_1/key/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/EinsumН
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpь
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2 
multi_head_attention_1/key/add§
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp░
*multi_head_attention_1/value/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsum█
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/value/add/ReadVariableOpш
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2"
 multi_head_attention_1/value/addЂ
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_1/Mul/yк
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:         d@2
multi_head_attention_1/MulЧ
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum─
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2(
&multi_head_attention_1/softmax/Softmax╩
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:         dd2)
'multi_head_attention_1/dropout/Identityћ
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsumъ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpМ
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/EinsumЭ
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOpЮ
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2-
+multi_head_attention_1/attention_output/addЋ
	add_2/addAddV2/multi_head_attention_1/attention_output/add:z:0add_1/add:z:0*
T0*+
_output_shapes
:         d@2
	add_2/addХ
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesт
"layer_normalization_3/moments/meanMeanadd_2/add:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2$
"layer_normalization_3/moments/mean╦
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:         d2,
*layer_normalization_3/moments/StopGradientы
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd_2/add:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@21
/layer_normalization_3/moments/SquaredDifferenceЙ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indicesЌ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2(
&layer_normalization_3/moments/varianceЊ
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52'
%layer_normalization_3/batchnorm/add/yЖ
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2%
#layer_normalization_3/batchnorm/addХ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2'
%layer_normalization_3/batchnorm/RsqrtЯ
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpЬ
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_3/batchnorm/mul├
%layer_normalization_3/batchnorm/mul_1Muladd_2/add:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_3/batchnorm/mul_1р
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_3/batchnorm/mul_2н
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpЖ
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_3/batchnorm/subр
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_3/batchnorm/add_1»
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axesЂ
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/freeІ
dense_3/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shapeё
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axisщ
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2ѕ
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis 
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
dense_3/Tensordot/Constа
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prodђ
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1е
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1ђ
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axisп
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concatг
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack╦
dense_3/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:         d@2
dense_3/Tensordot/transpose┐
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_3/Tensordot/Reshape┐
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_3/Tensordot/MatMulЂ
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ2
dense_3/Tensordot/Const_2ё
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axisт
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1▒
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2
dense_3/TensordotЦ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_3/BiasAdd/ReadVariableOpе
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2
dense_3/BiasAddm
dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_3/Gelu/mul/xЎ
dense_3/Gelu/mulMuldense_3/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/mulo
dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
dense_3/Gelu/Cast/xд
dense_3/Gelu/truedivRealDivdense_3/BiasAdd:output:0dense_3/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/truediv|
dense_3/Gelu/ErfErfdense_3/Gelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/Erfm
dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dense_3/Gelu/add/xЌ
dense_3/Gelu/addAddV2dense_3/Gelu/add/x:output:0dense_3/Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/addњ
dense_3/Gelu/mul_1Muldense_3/Gelu/mul:z:0dense_3/Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/mul_1»
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axesЂ
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
dense_4/Tensordot/Shapeё
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axisщ
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2ѕ
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis 
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
dense_4/Tensordot/Constа
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prodђ
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1е
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1ђ
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axisп
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concatг
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack╣
dense_4/Tensordot/transpose	Transposedense_3/Gelu/mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:         dђ2
dense_4/Tensordot/transpose┐
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_4/Tensordot/ReshapeЙ
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_4/Tensordot/MatMulђ
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_4/Tensordot/Const_2ё
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axisт
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1░
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2
dense_4/Tensordotц
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOpД
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
dense_4/BiasAddm
dense_4/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_4/Gelu/mul/xў
dense_4/Gelu/mulMuldense_4/Gelu/mul/x:output:0dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/mulo
dense_4/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
dense_4/Gelu/Cast/xЦ
dense_4/Gelu/truedivRealDivdense_4/BiasAdd:output:0dense_4/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/truediv{
dense_4/Gelu/ErfErfdense_4/Gelu/truediv:z:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/Erfm
dense_4/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dense_4/Gelu/add/xќ
dense_4/Gelu/addAddV2dense_4/Gelu/add/x:output:0dense_4/Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/addЉ
dense_4/Gelu/mul_1Muldense_4/Gelu/mul:z:0dense_4/Gelu/add:z:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/mul_1|
	add_3/addAddV2dense_4/Gelu/mul_1:z:0add_2/add:z:0*
T0*+
_output_shapes
:         d@2
	add_3/addХ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesт
"layer_normalization_4/moments/meanMeanadd_3/add:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2$
"layer_normalization_4/moments/mean╦
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:         d2,
*layer_normalization_4/moments/StopGradientы
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd_3/add:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@21
/layer_normalization_4/moments/SquaredDifferenceЙ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indicesЌ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2(
&layer_normalization_4/moments/varianceЊ
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52'
%layer_normalization_4/batchnorm/add/yЖ
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2%
#layer_normalization_4/batchnorm/addХ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2'
%layer_normalization_4/batchnorm/RsqrtЯ
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpЬ
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_4/batchnorm/mul├
%layer_normalization_4/batchnorm/mul_1Muladd_3/add:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_4/batchnorm/mul_1р
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_4/batchnorm/mul_2н
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpЖ
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_4/batchnorm/subр
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_4/batchnorm/add_1w
reshape/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
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
value	B :
2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape▓
reshape/ReshapeReshape)layer_normalization_4/batchnorm/add_1:z:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         

@2
reshape/Reshapeф
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype02
conv_1/Conv2D/ReadVariableOp╦
conv_1/Conv2DConv2Dreshape/Reshape:output:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
conv_1/Conv2DА
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_1/BiasAdd/ReadVariableOpц
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
conv_1/BiasAddф
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
conv_2/Conv2D/ReadVariableOp╩
conv_2/Conv2DConv2Dconv_1/BiasAdd:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
conv_2/Conv2DА
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_2/BiasAdd/ReadVariableOpц
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
conv_2/BiasAddф
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_3/Conv2D/ReadVariableOp╩
conv_3/Conv2DConv2Dconv_2/BiasAdd:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv_3/Conv2DА
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOpц
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv_3/BiasAddф
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOp╩
conv_4/Conv2DConv2Dconv_3/BiasAdd:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv_4/Conv2DА
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_4/BiasAdd/ReadVariableOpц
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv_4/BiasAdd{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
flatten_layer/Constб
flatten_layer/ReshapeReshapeconv_4/BiasAdd:output:0flatten_layer/Const:output:0*
T0*'
_output_shapes
:         @2
flatten_layer/Reshapeю
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes

:@2*
dtype02
FC_1/MatMul/ReadVariableOpџ
FC_1/MatMulMatMulflatten_layer/Reshape:output:0"FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
FC_1/MatMulЏ
FC_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_1/BiasAdd/ReadVariableOpЋ
FC_1/BiasAddBiasAddFC_1/MatMul:product:0#FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
FC_1/BiasAddЇ
leaky_ReLu_1/LeakyRelu	LeakyReluFC_1/BiasAdd:output:0*'
_output_shapes
:         2*
alpha%џЎЎ>2
leaky_ReLu_1/LeakyReluю
FC_2/MatMul/ReadVariableOpReadVariableOp#fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
FC_2/MatMul/ReadVariableOpа
FC_2/MatMulMatMul$leaky_ReLu_1/LeakyRelu:activations:0"FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
FC_2/MatMulЏ
FC_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_2/BiasAdd/ReadVariableOpЋ
FC_2/BiasAddBiasAddFC_2/MatMul:product:0#FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
FC_2/BiasAddЇ
leaky_ReLu_2/LeakyRelu	LeakyReluFC_2/BiasAdd:output:0*'
_output_shapes
:         2*
alpha%џЎЎ>2
leaky_ReLu_2/LeakyRelu┤
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02$
"output_layer/MatMul/ReadVariableOpИ
output_layer/MatMulMatMul$leaky_ReLu_2/LeakyRelu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output_layer/MatMul│
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpх
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output_layer/BiasAddѕ
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output_layer/Softmax─
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
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
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2ѕ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2ї
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
(patch_encoder/embedding/embedding_lookup(patch_encoder/embedding/embedding_lookup:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
о
J
.__inference_flatten_layer_layer_call_fn_625638

inputs
identityК
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_6223922
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Б
љ
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_625523

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
┌

В
7__inference_multi_head_attention_1_layer_call_fn_625352	
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
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_6227792
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
Б
љ
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_625222

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
р
k
A__inference_add_2_layer_call_and_return_conditional_losses_622152

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:         d@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs:SO
+
_output_shapes
:         d@
 
_user_specified_nameinputs
о

Ж
5__inference_multi_head_attention_layer_call_fn_625029	
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
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_6219152
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
ж
m
A__inference_add_3_layer_call_and_return_conditional_losses_625495
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:         d@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:U Q
+
_output_shapes
:         d@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         d@
"
_user_specified_name
inputs/1
Ћ7
ч
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_625308	
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
identityѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpК
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЎ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┴
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpЉ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpК
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЎ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
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
:         d@2
Mulа
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/dropout/Constд
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeн
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         dd*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2 
dropout/dropout/GreaterEqual/yТ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/GreaterEqualЪ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         dd2
dropout/dropout/Castб
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/Mul_1И
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpэ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┴
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
attention_output/addѓ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 2J
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
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
ж
e
I__inference_flatten_layer_layer_call_and_return_conditional_losses_625633

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         @2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Б
љ
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_622300

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
ѓ'
ч
C__inference_dense_4_layer_call_and_return_conditional_losses_622264

inputs4
!tensordot_readvariableop_resource:	ђ@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpЌ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackЉ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         dђ2
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/Reshapeъ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
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
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2
	Tensordotї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЄ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2	
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
:         d@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
Gelu/Cast/xЁ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:         d@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:         d@2

Gelu/mul_1џ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         dђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         dђ
 
_user_specified_nameinputs
Б
љ
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_622087

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
┐
ю
'__inference_conv_2_layer_call_fn_625589

inputs!
unknown:


	unknown_0:

identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_6223482
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
р
k
A__inference_add_3_layer_call_and_return_conditional_losses_622276

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:         d@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs:SO
+
_output_shapes
:         d@
 
_user_specified_nameinputs
■
╚
6__inference_WheatClassifier_CNN_1_layer_call_fn_622563
input_layer
unknown:	г@
	unknown_0:@
	unknown_1:d@
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

unknown_14:	@ђ

unknown_15:	ђ

unknown_16:	ђ@

unknown_17:@

unknown_18:@

unknown_19:@ 

unknown_20:@@

unknown_21:@ 

unknown_22:@@

unknown_23:@ 

unknown_24:@@

unknown_25:@ 

unknown_26:@@

unknown_27:@

unknown_28:@

unknown_29:@

unknown_30:	@ђ

unknown_31:	ђ

unknown_32:	ђ@

unknown_33:@

unknown_34:@

unknown_35:@$

unknown_36:@


unknown_37:
$

unknown_38:



unknown_39:
$

unknown_40:


unknown_41:$

unknown_42:

unknown_43:

unknown_44:@2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identityѕбStatefulPartitionedCallФ
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
:         *U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_6224582
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameinput_layer
А
ј
O__inference_layer_normalization_layer_call_and_return_conditional_losses_621874

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
»

ч
B__inference_conv_1_layer_call_and_return_conditional_losses_622332

inputs8
conv2d_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         

@
 
_user_specified_nameinputs
§љ
щ
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_623690
input_layer'
patch_encoder_623560:	г@"
patch_encoder_623562:@&
patch_encoder_623564:d@(
layer_normalization_623567:@(
layer_normalization_623569:@1
multi_head_attention_623572:@@-
multi_head_attention_623574:@1
multi_head_attention_623576:@@-
multi_head_attention_623578:@1
multi_head_attention_623580:@@-
multi_head_attention_623582:@1
multi_head_attention_623584:@@)
multi_head_attention_623586:@*
layer_normalization_1_623590:@*
layer_normalization_1_623592:@!
dense_1_623595:	@ђ
dense_1_623597:	ђ!
dense_2_623600:	ђ@
dense_2_623602:@*
layer_normalization_2_623606:@*
layer_normalization_2_623608:@3
multi_head_attention_1_623611:@@/
multi_head_attention_1_623613:@3
multi_head_attention_1_623615:@@/
multi_head_attention_1_623617:@3
multi_head_attention_1_623619:@@/
multi_head_attention_1_623621:@3
multi_head_attention_1_623623:@@+
multi_head_attention_1_623625:@*
layer_normalization_3_623629:@*
layer_normalization_3_623631:@!
dense_3_623634:	@ђ
dense_3_623636:	ђ!
dense_4_623639:	ђ@
dense_4_623641:@*
layer_normalization_4_623645:@*
layer_normalization_4_623647:@'
conv_1_623651:@

conv_1_623653:
'
conv_2_623656:


conv_2_623658:
'
conv_3_623661:

conv_3_623663:'
conv_4_623666:
conv_4_623668:
fc_1_623672:@2
fc_1_623674:2
fc_2_623678:22
fc_2_623680:2%
output_layer_623684:2!
output_layer_623686:
identityѕбFC_1/StatefulPartitionedCallбFC_2/StatefulPartitionedCallбconv_1/StatefulPartitionedCallбconv_2/StatefulPartitionedCallбconv_3/StatefulPartitionedCallбconv_4/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallб+layer_normalization/StatefulPartitionedCallб-layer_normalization_1/StatefulPartitionedCallб-layer_normalization_2/StatefulPartitionedCallб-layer_normalization_3/StatefulPartitionedCallб-layer_normalization_4/StatefulPartitionedCallб,multi_head_attention/StatefulPartitionedCallб.multi_head_attention_1/StatefulPartitionedCallб$output_layer/StatefulPartitionedCallб%patch_encoder/StatefulPartitionedCallС
patches/PartitionedCallPartitionedCallinput_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  г* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_6218022
patches/PartitionedCallс
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_623560patch_encoder_623562patch_encoder_623564*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_6218442'
%patch_encoder/StatefulPartitionedCallэ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_623567layer_normalization_623569*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_6218742-
+layer_normalization/StatefulPartitionedCallз
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_623572multi_head_attention_623574multi_head_attention_623576multi_head_attention_623578multi_head_attention_623580multi_head_attention_623582multi_head_attention_623584multi_head_attention_623586*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_6229202.
,multi_head_attention/StatefulPartitionedCallЕ
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_6219392
add/PartitionedCall№
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_623590layer_normalization_1_623592*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_6219632/
-layer_normalization_1/StatefulPartitionedCall─
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_623595dense_1_623597*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6220072!
dense_1/StatefulPartitionedCallх
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_623600dense_2_623602*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6220512!
dense_2/StatefulPartitionedCallљ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_6220632
add_1/PartitionedCallы
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_623606layer_normalization_2_623608*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_6220872/
-layer_normalization_2/StatefulPartitionedCallЇ
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_623611multi_head_attention_1_623613multi_head_attention_1_623615multi_head_attention_1_623617multi_head_attention_1_623619multi_head_attention_1_623621multi_head_attention_1_623623multi_head_attention_1_623625*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_62277920
.multi_head_attention_1/StatefulPartitionedCallА
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_6221522
add_2/PartitionedCallы
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_623629layer_normalization_3_623631*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_6221762/
-layer_normalization_3/StatefulPartitionedCall─
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_623634dense_3_623636*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6222202!
dense_3/StatefulPartitionedCallх
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_623639dense_4_623641*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6222642!
dense_4/StatefulPartitionedCallњ
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_6222762
add_3/PartitionedCallы
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_623645layer_normalization_4_623647*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_6223002/
-layer_normalization_4/StatefulPartitionedCallЅ
reshape/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6223202
reshape/PartitionedCallг
conv_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv_1_623651conv_1_623653*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_6223322 
conv_1/StatefulPartitionedCall│
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_623656conv_2_623658*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_6223482 
conv_2/StatefulPartitionedCall│
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_623661conv_3_623663*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_6223642 
conv_3/StatefulPartitionedCall│
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_623666conv_4_623668*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_6223802 
conv_4/StatefulPartitionedCallё
flatten_layer/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_6223922
flatten_layer/PartitionedCallа
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_623672fc_1_623674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_6224042
FC_1/StatefulPartitionedCall 
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_6224152
leaky_ReLu_1/PartitionedCallЪ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_623678fc_2_623680*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_6224272
FC_2/StatefulPartitionedCall 
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_6224382
leaky_ReLu_2/PartitionedCallК
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_623684output_layer_623686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_6224512&
$output_layer/StatefulPartitionedCallУ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
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
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameinput_layer
Ьљ
З
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_622458

inputs'
patch_encoder_621845:	г@"
patch_encoder_621847:@&
patch_encoder_621849:d@(
layer_normalization_621875:@(
layer_normalization_621877:@1
multi_head_attention_621916:@@-
multi_head_attention_621918:@1
multi_head_attention_621920:@@-
multi_head_attention_621922:@1
multi_head_attention_621924:@@-
multi_head_attention_621926:@1
multi_head_attention_621928:@@)
multi_head_attention_621930:@*
layer_normalization_1_621964:@*
layer_normalization_1_621966:@!
dense_1_622008:	@ђ
dense_1_622010:	ђ!
dense_2_622052:	ђ@
dense_2_622054:@*
layer_normalization_2_622088:@*
layer_normalization_2_622090:@3
multi_head_attention_1_622129:@@/
multi_head_attention_1_622131:@3
multi_head_attention_1_622133:@@/
multi_head_attention_1_622135:@3
multi_head_attention_1_622137:@@/
multi_head_attention_1_622139:@3
multi_head_attention_1_622141:@@+
multi_head_attention_1_622143:@*
layer_normalization_3_622177:@*
layer_normalization_3_622179:@!
dense_3_622221:	@ђ
dense_3_622223:	ђ!
dense_4_622265:	ђ@
dense_4_622267:@*
layer_normalization_4_622301:@*
layer_normalization_4_622303:@'
conv_1_622333:@

conv_1_622335:
'
conv_2_622349:


conv_2_622351:
'
conv_3_622365:

conv_3_622367:'
conv_4_622381:
conv_4_622383:
fc_1_622405:@2
fc_1_622407:2
fc_2_622428:22
fc_2_622430:2%
output_layer_622452:2!
output_layer_622454:
identityѕбFC_1/StatefulPartitionedCallбFC_2/StatefulPartitionedCallбconv_1/StatefulPartitionedCallбconv_2/StatefulPartitionedCallбconv_3/StatefulPartitionedCallбconv_4/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallб+layer_normalization/StatefulPartitionedCallб-layer_normalization_1/StatefulPartitionedCallб-layer_normalization_2/StatefulPartitionedCallб-layer_normalization_3/StatefulPartitionedCallб-layer_normalization_4/StatefulPartitionedCallб,multi_head_attention/StatefulPartitionedCallб.multi_head_attention_1/StatefulPartitionedCallб$output_layer/StatefulPartitionedCallб%patch_encoder/StatefulPartitionedCall▀
patches/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  г* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_6218022
patches/PartitionedCallс
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_621845patch_encoder_621847patch_encoder_621849*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_6218442'
%patch_encoder/StatefulPartitionedCallэ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_621875layer_normalization_621877*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_6218742-
+layer_normalization/StatefulPartitionedCallз
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_621916multi_head_attention_621918multi_head_attention_621920multi_head_attention_621922multi_head_attention_621924multi_head_attention_621926multi_head_attention_621928multi_head_attention_621930*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_6219152.
,multi_head_attention/StatefulPartitionedCallЕ
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_6219392
add/PartitionedCall№
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_621964layer_normalization_1_621966*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_6219632/
-layer_normalization_1/StatefulPartitionedCall─
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_622008dense_1_622010*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6220072!
dense_1/StatefulPartitionedCallх
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_622052dense_2_622054*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6220512!
dense_2/StatefulPartitionedCallљ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_6220632
add_1/PartitionedCallы
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_622088layer_normalization_2_622090*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_6220872/
-layer_normalization_2/StatefulPartitionedCallЇ
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_622129multi_head_attention_1_622131multi_head_attention_1_622133multi_head_attention_1_622135multi_head_attention_1_622137multi_head_attention_1_622139multi_head_attention_1_622141multi_head_attention_1_622143*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_62212820
.multi_head_attention_1/StatefulPartitionedCallА
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_6221522
add_2/PartitionedCallы
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_622177layer_normalization_3_622179*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_6221762/
-layer_normalization_3/StatefulPartitionedCall─
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_622221dense_3_622223*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6222202!
dense_3/StatefulPartitionedCallх
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_622265dense_4_622267*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6222642!
dense_4/StatefulPartitionedCallњ
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_6222762
add_3/PartitionedCallы
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_622301layer_normalization_4_622303*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_6223002/
-layer_normalization_4/StatefulPartitionedCallЅ
reshape/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6223202
reshape/PartitionedCallг
conv_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv_1_622333conv_1_622335*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_6223322 
conv_1/StatefulPartitionedCall│
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_622349conv_2_622351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_6223482 
conv_2/StatefulPartitionedCall│
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_622365conv_3_622367*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_6223642 
conv_3/StatefulPartitionedCall│
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_622381conv_4_622383*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_6223802 
conv_4/StatefulPartitionedCallё
flatten_layer/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_6223922
flatten_layer/PartitionedCallа
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_622405fc_1_622407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_6224042
FC_1/StatefulPartitionedCall 
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_6224152
leaky_ReLu_1/PartitionedCallЪ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_622428fc_2_622430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_6224272
FC_2/StatefulPartitionedCall 
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_6224382
leaky_ReLu_2/PartitionedCallК
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_622452output_layer_622454*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_6224512&
$output_layer/StatefulPartitionedCallУ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
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
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
╠	
ы
@__inference_FC_1_layer_call_and_return_conditional_losses_622404

inputs0
matmul_readvariableop_resource:@2-
biasadd_readvariableop_resource:2
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
у
k
?__inference_add_layer_call_and_return_conditional_losses_625057
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:         d@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:U Q
+
_output_shapes
:         d@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         d@
"
_user_specified_name
inputs/1
І'
Ч
C__inference_dense_3_layer_call_and_return_conditional_losses_625433

inputs4
!tensordot_readvariableop_resource:	@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpЌ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackљ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         d@2
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/ReshapeЪ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2
	TensordotЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2	
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
:         dђ2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
Gelu/Cast/xє
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2

Gelu/mul_1Џ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:         dђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
╝
Х
$__inference_signature_wrapper_623807
input_layer
unknown:	г@
	unknown_0:@
	unknown_1:d@
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

unknown_14:	@ђ

unknown_15:	ђ

unknown_16:	ђ@

unknown_17:@

unknown_18:@

unknown_19:@ 

unknown_20:@@

unknown_21:@ 

unknown_22:@@

unknown_23:@ 

unknown_24:@@

unknown_25:@ 

unknown_26:@@

unknown_27:@

unknown_28:@

unknown_29:@

unknown_30:	@ђ

unknown_31:	ђ

unknown_32:	ђ@

unknown_33:@

unknown_34:@

unknown_35:@$

unknown_36:@


unknown_37:
$

unknown_38:



unknown_39:
$

unknown_40:


unknown_41:$

unknown_42:

unknown_43:

unknown_44:@2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identityѕбStatefulPartitionedCallч
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
:         *U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_6217812
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameinput_layer
І'
Ч
C__inference_dense_1_layer_call_and_return_conditional_losses_625132

inputs4
!tensordot_readvariableop_resource:	@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpЌ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackљ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         d@2
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/ReshapeЪ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2
	TensordotЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2	
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
:         dђ2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
Gelu/Cast/xє
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2

Gelu/mul_1Џ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:         dђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
ч
d
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_625691

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         2*
alpha%џЎЎ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
■
╚
6__inference_WheatClassifier_CNN_1_layer_call_fn_623422
input_layer
unknown:	г@
	unknown_0:@
	unknown_1:d@
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

unknown_14:	@ђ

unknown_15:	ђ

unknown_16:	ђ@

unknown_17:@

unknown_18:@

unknown_19:@ 

unknown_20:@@

unknown_21:@ 

unknown_22:@@

unknown_23:@ 

unknown_24:@@

unknown_25:@ 

unknown_26:@@

unknown_27:@

unknown_28:@

unknown_29:@

unknown_30:	@ђ

unknown_31:	ђ

unknown_32:	ђ@

unknown_33:@

unknown_34:@

unknown_35:@$

unknown_36:@


unknown_37:
$

unknown_38:



unknown_39:
$

unknown_40:


unknown_41:$

unknown_42:

unknown_43:

unknown_44:@2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identityѕбStatefulPartitionedCallФ
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
:         *U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_6232102
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameinput_layer
╠	
ы
@__inference_FC_2_layer_call_and_return_conditional_losses_622427

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
┴
Ъ
6__inference_layer_normalization_2_layer_call_fn_625231

inputs
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_6220872
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
─
I
-__inference_leaky_ReLu_2_layer_call_fn_625696

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_6224382
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
М
R
&__inference_add_3_layer_call_fn_625501
inputs_0
inputs_1
identityл
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_6222762
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:U Q
+
_output_shapes
:         d@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         d@
"
_user_specified_name
inputs/1
ч
d
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_622415

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         2*
alpha%џЎЎ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
м
D
(__inference_reshape_layer_call_fn_625551

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6223202
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         

@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d@:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
Џл
└?
!__inference__wrapped_model_621781
input_layer^
Kwheatclassifier_cnn_1_patch_encoder_dense_tensordot_readvariableop_resource:	г@W
Iwheatclassifier_cnn_1_patch_encoder_dense_biasadd_readvariableop_resource:@W
Ewheatclassifier_cnn_1_patch_encoder_embedding_embedding_lookup_621427:d@]
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
?wheatclassifier_cnn_1_dense_1_tensordot_readvariableop_resource:	@ђL
=wheatclassifier_cnn_1_dense_1_biasadd_readvariableop_resource:	ђR
?wheatclassifier_cnn_1_dense_2_tensordot_readvariableop_resource:	ђ@K
=wheatclassifier_cnn_1_dense_2_biasadd_readvariableop_resource:@_
Qwheatclassifier_cnn_1_layer_normalization_2_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_cnn_1_layer_normalization_2_batchnorm_readvariableop_resource:@n
Xwheatclassifier_cnn_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource:@@`
Nwheatclassifier_cnn_1_multi_head_attention_1_query_add_readvariableop_resource:@l
Vwheatclassifier_cnn_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource:@@^
Lwheatclassifier_cnn_1_multi_head_attention_1_key_add_readvariableop_resource:@n
Xwheatclassifier_cnn_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource:@@`
Nwheatclassifier_cnn_1_multi_head_attention_1_value_add_readvariableop_resource:@y
cwheatclassifier_cnn_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:@@g
Ywheatclassifier_cnn_1_multi_head_attention_1_attention_output_add_readvariableop_resource:@_
Qwheatclassifier_cnn_1_layer_normalization_3_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_cnn_1_layer_normalization_3_batchnorm_readvariableop_resource:@R
?wheatclassifier_cnn_1_dense_3_tensordot_readvariableop_resource:	@ђL
=wheatclassifier_cnn_1_dense_3_biasadd_readvariableop_resource:	ђR
?wheatclassifier_cnn_1_dense_4_tensordot_readvariableop_resource:	ђ@K
=wheatclassifier_cnn_1_dense_4_biasadd_readvariableop_resource:@_
Qwheatclassifier_cnn_1_layer_normalization_4_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_cnn_1_layer_normalization_4_batchnorm_readvariableop_resource:@U
;wheatclassifier_cnn_1_conv_1_conv2d_readvariableop_resource:@
J
<wheatclassifier_cnn_1_conv_1_biasadd_readvariableop_resource:
U
;wheatclassifier_cnn_1_conv_2_conv2d_readvariableop_resource:

J
<wheatclassifier_cnn_1_conv_2_biasadd_readvariableop_resource:
U
;wheatclassifier_cnn_1_conv_3_conv2d_readvariableop_resource:
J
<wheatclassifier_cnn_1_conv_3_biasadd_readvariableop_resource:U
;wheatclassifier_cnn_1_conv_4_conv2d_readvariableop_resource:J
<wheatclassifier_cnn_1_conv_4_biasadd_readvariableop_resource:K
9wheatclassifier_cnn_1_fc_1_matmul_readvariableop_resource:@2H
:wheatclassifier_cnn_1_fc_1_biasadd_readvariableop_resource:2K
9wheatclassifier_cnn_1_fc_2_matmul_readvariableop_resource:22H
:wheatclassifier_cnn_1_fc_2_biasadd_readvariableop_resource:2S
Awheatclassifier_cnn_1_output_layer_matmul_readvariableop_resource:2P
Bwheatclassifier_cnn_1_output_layer_biasadd_readvariableop_resource:
identityѕб1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOpб0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOpб1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOpб0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOpб3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOpб3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOpб3WheatClassifier_CNN_1/conv_3/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_1/conv_3/Conv2D/ReadVariableOpб3WheatClassifier_CNN_1/conv_4/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_1/conv_4/Conv2D/ReadVariableOpб4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOpб6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOpб4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOpб6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOpб4WheatClassifier_CNN_1/dense_3/BiasAdd/ReadVariableOpб6WheatClassifier_CNN_1/dense_3/Tensordot/ReadVariableOpб4WheatClassifier_CNN_1/dense_4/BiasAdd/ReadVariableOpб6WheatClassifier_CNN_1/dense_4/Tensordot/ReadVariableOpбBWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOpбFWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOpбDWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOpбHWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOpбDWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOpбHWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOpбDWheatClassifier_CNN_1/layer_normalization_3/batchnorm/ReadVariableOpбHWheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul/ReadVariableOpбDWheatClassifier_CNN_1/layer_normalization_4/batchnorm/ReadVariableOpбHWheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul/ReadVariableOpбNWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOpбXWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpбAWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOpбKWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOpбCWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOpбMWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOpбCWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOpбMWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOpбPWheatClassifier_CNN_1/multi_head_attention_1/attention_output/add/ReadVariableOpбZWheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpбCWheatClassifier_CNN_1/multi_head_attention_1/key/add/ReadVariableOpбMWheatClassifier_CNN_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpбEWheatClassifier_CNN_1/multi_head_attention_1/query/add/ReadVariableOpбOWheatClassifier_CNN_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpбEWheatClassifier_CNN_1/multi_head_attention_1/value/add/ReadVariableOpбOWheatClassifier_CNN_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpб9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOpб8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOpб@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOpбBWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOpб>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookupЁ
#WheatClassifier_CNN_1/patches/ShapeShapeinput_layer*
T0*
_output_shapes
:2%
#WheatClassifier_CNN_1/patches/Shape░
1WheatClassifier_CNN_1/patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_CNN_1/patches/strided_slice/stack┤
3WheatClassifier_CNN_1/patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_1/patches/strided_slice/stack_1┤
3WheatClassifier_CNN_1/patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_1/patches/strided_slice/stack_2ќ
+WheatClassifier_CNN_1/patches/strided_sliceStridedSlice,WheatClassifier_CNN_1/patches/Shape:output:0:WheatClassifier_CNN_1/patches/strided_slice/stack:output:0<WheatClassifier_CNN_1/patches/strided_slice/stack_1:output:0<WheatClassifier_CNN_1/patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+WheatClassifier_CNN_1/patches/strided_sliceЋ
1WheatClassifier_CNN_1/patches/ExtractImagePatchesExtractImagePatchesinput_layer*
T0*0
_output_shapes
:         

г*
ksizes


*
paddingVALID*
rates
*
strides


23
1WheatClassifier_CNN_1/patches/ExtractImagePatchesЕ
-WheatClassifier_CNN_1/patches/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2/
-WheatClassifier_CNN_1/patches/Reshape/shape/1А
-WheatClassifier_CNN_1/patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :г2/
-WheatClassifier_CNN_1/patches/Reshape/shape/2Х
+WheatClassifier_CNN_1/patches/Reshape/shapePack4WheatClassifier_CNN_1/patches/strided_slice:output:06WheatClassifier_CNN_1/patches/Reshape/shape/1:output:06WheatClassifier_CNN_1/patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+WheatClassifier_CNN_1/patches/Reshape/shapeї
%WheatClassifier_CNN_1/patches/ReshapeReshape;WheatClassifier_CNN_1/patches/ExtractImagePatches:patches:04WheatClassifier_CNN_1/patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  г2'
%WheatClassifier_CNN_1/patches/Reshapeц
/WheatClassifier_CNN_1/patch_encoder/range/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/WheatClassifier_CNN_1/patch_encoder/range/startц
/WheatClassifier_CNN_1/patch_encoder/range/limitConst*
_output_shapes
: *
dtype0*
value	B :d21
/WheatClassifier_CNN_1/patch_encoder/range/limitц
/WheatClassifier_CNN_1/patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/WheatClassifier_CNN_1/patch_encoder/range/deltaЕ
)WheatClassifier_CNN_1/patch_encoder/rangeRange8WheatClassifier_CNN_1/patch_encoder/range/start:output:08WheatClassifier_CNN_1/patch_encoder/range/limit:output:08WheatClassifier_CNN_1/patch_encoder/range/delta:output:0*
_output_shapes
:d2+
)WheatClassifier_CNN_1/patch_encoder/rangeЋ
BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOpKwheatclassifier_cnn_1_patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	г@*
dtype02D
BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOpЙ
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/axes┼
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/freeн
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ShapeShape.WheatClassifier_CNN_1/patches/Reshape:output:0*
T0*
_output_shapes
:2;
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Shape╚
AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2/axisБ
<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2GatherV2BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Shape:output:0AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/free:output:0JWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2╠
CWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
CWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1/axisЕ
>WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1GatherV2BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Shape:output:0AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/axes:output:0LWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1└
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Constе
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ProdProdEWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2:output:0BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod─
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_1░
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod_1ProdGWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2_1:output:0DWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod_1─
?WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat/axisѓ
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concatConcatV2AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/free:output:0AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/axes:output:0HWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat┤
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/stackPackAWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod:output:0CWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/stack└
=WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/transpose	Transpose.WheatClassifier_CNN_1/patches/Reshape:output:0CWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  г2?
=WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/transposeК
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReshapeReshapeAWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/transpose:y:0BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2=
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Reshapeк
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/MatMulMatMulDWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Reshape:output:0JWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2<
:WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/MatMul─
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_2╚
AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1/axisЈ
<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1ConcatV2EWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/GatherV2:output:0DWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/Const_2:output:0JWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1┴
3WheatClassifier_CNN_1/patch_encoder/dense/TensordotReshapeDWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/MatMul:product:0EWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @25
3WheatClassifier_CNN_1/patch_encoder/dense/Tensordotі
@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOpIwheatclassifier_cnn_1_patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOpИ
1WheatClassifier_CNN_1/patch_encoder/dense/BiasAddBiasAdd<WheatClassifier_CNN_1/patch_encoder/dense/Tensordot:output:0HWheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @23
1WheatClassifier_CNN_1/patch_encoder/dense/BiasAddм
>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookupResourceGatherEwheatclassifier_cnn_1_patch_encoder_embedding_embedding_lookup_6214272WheatClassifier_CNN_1/patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/621427*
_output_shapes

:d@*
dtype02@
>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookupў
GWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/IdentityIdentityGWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/621427*
_output_shapes

:d@2I
GWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/IdentityЮ
IWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/Identity_1IdentityPWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:d@2K
IWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/Identity_1А
'WheatClassifier_CNN_1/patch_encoder/addAddV2:WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd:output:0RWheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         d@2)
'WheatClassifier_CNN_1/patch_encoder/addя
HWheatClassifier_CNN_1/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
HWheatClassifier_CNN_1/layer_normalization/moments/mean/reduction_indices┐
6WheatClassifier_CNN_1/layer_normalization/moments/meanMean+WheatClassifier_CNN_1/patch_encoder/add:z:0QWheatClassifier_CNN_1/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(28
6WheatClassifier_CNN_1/layer_normalization/moments/meanЄ
>WheatClassifier_CNN_1/layer_normalization/moments/StopGradientStopGradient?WheatClassifier_CNN_1/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         d2@
>WheatClassifier_CNN_1/layer_normalization/moments/StopGradient╦
CWheatClassifier_CNN_1/layer_normalization/moments/SquaredDifferenceSquaredDifference+WheatClassifier_CNN_1/patch_encoder/add:z:0GWheatClassifier_CNN_1/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2E
CWheatClassifier_CNN_1/layer_normalization/moments/SquaredDifferenceТ
LWheatClassifier_CNN_1/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
LWheatClassifier_CNN_1/layer_normalization/moments/variance/reduction_indicesу
:WheatClassifier_CNN_1/layer_normalization/moments/varianceMeanGWheatClassifier_CNN_1/layer_normalization/moments/SquaredDifference:z:0UWheatClassifier_CNN_1/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2<
:WheatClassifier_CNN_1/layer_normalization/moments/variance╗
9WheatClassifier_CNN_1/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/add/y║
7WheatClassifier_CNN_1/layer_normalization/batchnorm/addAddV2CWheatClassifier_CNN_1/layer_normalization/moments/variance:output:0BWheatClassifier_CNN_1/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d29
7WheatClassifier_CNN_1/layer_normalization/batchnorm/addЫ
9WheatClassifier_CNN_1/layer_normalization/batchnorm/RsqrtRsqrt;WheatClassifier_CNN_1/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/Rsqrtю
FWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpOwheatclassifier_cnn_1_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02H
FWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOpЙ
7WheatClassifier_CNN_1/layer_normalization/batchnorm/mulMul=WheatClassifier_CNN_1/layer_normalization/batchnorm/Rsqrt:y:0NWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@29
7WheatClassifier_CNN_1/layer_normalization/batchnorm/mulЮ
9WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_1Mul+WheatClassifier_CNN_1/patch_encoder/add:z:0;WheatClassifier_CNN_1/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_1▒
9WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_2Mul?WheatClassifier_CNN_1/layer_normalization/moments/mean:output:0;WheatClassifier_CNN_1/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_2љ
BWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOpReadVariableOpKwheatclassifier_cnn_1_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02D
BWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOp║
7WheatClassifier_CNN_1/layer_normalization/batchnorm/subSubJWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOp:value:0=WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@29
7WheatClassifier_CNN_1/layer_normalization/batchnorm/sub▒
9WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1AddV2=WheatClassifier_CNN_1/layer_normalization/batchnorm/mul_1:z:0;WheatClassifier_CNN_1/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1╣
MWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_cnn_1_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOpђ
>WheatClassifier_CNN_1/multi_head_attention/query/einsum/EinsumEinsum=WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1:z:0UWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2@
>WheatClassifier_CNN_1/multi_head_attention/query/einsum/EinsumЌ
CWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOpReadVariableOpLwheatclassifier_cnn_1_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOp┼
4WheatClassifier_CNN_1/multi_head_attention/query/addAddV2GWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum:output:0KWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@26
4WheatClassifier_CNN_1/multi_head_attention/query/add│
KWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpTwheatclassifier_cnn_1_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02M
KWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOpЩ
<WheatClassifier_CNN_1/multi_head_attention/key/einsum/EinsumEinsum=WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1:z:0SWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2>
<WheatClassifier_CNN_1/multi_head_attention/key/einsum/EinsumЉ
AWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOpReadVariableOpJwheatclassifier_cnn_1_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02C
AWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOpй
2WheatClassifier_CNN_1/multi_head_attention/key/addAddV2EWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum:output:0IWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@24
2WheatClassifier_CNN_1/multi_head_attention/key/add╣
MWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_cnn_1_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOpђ
>WheatClassifier_CNN_1/multi_head_attention/value/einsum/EinsumEinsum=WheatClassifier_CNN_1/layer_normalization/batchnorm/add_1:z:0UWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2@
>WheatClassifier_CNN_1/multi_head_attention/value/einsum/EinsumЌ
CWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOpReadVariableOpLwheatclassifier_cnn_1_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOp┼
4WheatClassifier_CNN_1/multi_head_attention/value/addAddV2GWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum:output:0KWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@26
4WheatClassifier_CNN_1/multi_head_attention/value/addЕ
0WheatClassifier_CNN_1/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >22
0WheatClassifier_CNN_1/multi_head_attention/Mul/yќ
.WheatClassifier_CNN_1/multi_head_attention/MulMul8WheatClassifier_CNN_1/multi_head_attention/query/add:z:09WheatClassifier_CNN_1/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:         d@20
.WheatClassifier_CNN_1/multi_head_attention/Mul╠
8WheatClassifier_CNN_1/multi_head_attention/einsum/EinsumEinsum6WheatClassifier_CNN_1/multi_head_attention/key/add:z:02WheatClassifier_CNN_1/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2:
8WheatClassifier_CNN_1/multi_head_attention/einsum/Einsumђ
:WheatClassifier_CNN_1/multi_head_attention/softmax/SoftmaxSoftmaxAWheatClassifier_CNN_1/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2<
:WheatClassifier_CNN_1/multi_head_attention/softmax/Softmaxє
;WheatClassifier_CNN_1/multi_head_attention/dropout/IdentityIdentityDWheatClassifier_CNN_1/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:         dd2=
;WheatClassifier_CNN_1/multi_head_attention/dropout/IdentityС
:WheatClassifier_CNN_1/multi_head_attention/einsum_1/EinsumEinsumDWheatClassifier_CNN_1/multi_head_attention/dropout/Identity:output:08WheatClassifier_CNN_1/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2<
:WheatClassifier_CNN_1/multi_head_attention/einsum_1/Einsum┌
XWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpawheatclassifier_cnn_1_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02Z
XWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpБ
IWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/EinsumEinsumCWheatClassifier_CNN_1/multi_head_attention/einsum_1/Einsum:output:0`WheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2K
IWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum┤
NWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpWwheatclassifier_cnn_1_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02P
NWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOpь
?WheatClassifier_CNN_1/multi_head_attention/attention_output/addAddV2RWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum:output:0VWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2A
?WheatClassifier_CNN_1/multi_head_attention/attention_output/add№
WheatClassifier_CNN_1/add/addAddV2CWheatClassifier_CNN_1/multi_head_attention/attention_output/add:z:0+WheatClassifier_CNN_1/patch_encoder/add:z:0*
T0*+
_output_shapes
:         d@2
WheatClassifier_CNN_1/add/addР
JWheatClassifier_CNN_1/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_CNN_1/layer_normalization_1/moments/mean/reduction_indices╗
8WheatClassifier_CNN_1/layer_normalization_1/moments/meanMean!WheatClassifier_CNN_1/add/add:z:0SWheatClassifier_CNN_1/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2:
8WheatClassifier_CNN_1/layer_normalization_1/moments/meanЇ
@WheatClassifier_CNN_1/layer_normalization_1/moments/StopGradientStopGradientAWheatClassifier_CNN_1/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         d2B
@WheatClassifier_CNN_1/layer_normalization_1/moments/StopGradientК
EWheatClassifier_CNN_1/layer_normalization_1/moments/SquaredDifferenceSquaredDifference!WheatClassifier_CNN_1/add/add:z:0IWheatClassifier_CNN_1/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2G
EWheatClassifier_CNN_1/layer_normalization_1/moments/SquaredDifferenceЖ
NWheatClassifier_CNN_1/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN_1/layer_normalization_1/moments/variance/reduction_indices№
<WheatClassifier_CNN_1/layer_normalization_1/moments/varianceMeanIWheatClassifier_CNN_1/layer_normalization_1/moments/SquaredDifference:z:0WWheatClassifier_CNN_1/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2>
<WheatClassifier_CNN_1/layer_normalization_1/moments/variance┐
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add/y┬
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/addAddV2EWheatClassifier_CNN_1/layer_normalization_1/moments/variance:output:0DWheatClassifier_CNN_1/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2;
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/addЭ
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/RsqrtRsqrt=WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/Rsqrtб
HWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_cnn_1_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOpк
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mulMul?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/Rsqrt:y:0PWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mulЎ
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_1Mul!WheatClassifier_CNN_1/add/add:z:0=WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_1╣
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_2MulAWheatClassifier_CNN_1/layer_normalization_1/moments/mean:output:0=WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_2ќ
DWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_cnn_1_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOp┬
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/subSubLWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOp:value:0?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization_1/batchnorm/sub╣
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add_1AddV2?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul_1:z:0=WheatClassifier_CNN_1/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add_1ы
6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_cnn_1_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
dtype028
6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOpд
,WheatClassifier_CNN_1/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_CNN_1/dense_1/Tensordot/axesГ
,WheatClassifier_CNN_1/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_CNN_1/dense_1/Tensordot/free═
-WheatClassifier_CNN_1/dense_1/Tensordot/ShapeShape?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_1/Tensordot/Shape░
5WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2/axisу
0WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2GatherV26WheatClassifier_CNN_1/dense_1/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_1/Tensordot/free:output:0>WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2┤
7WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1/axisь
2WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1GatherV26WheatClassifier_CNN_1/dense_1/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_1/Tensordot/axes:output:0@WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1е
-WheatClassifier_CNN_1/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_CNN_1/dense_1/Tensordot/ConstЭ
,WheatClassifier_CNN_1/dense_1/Tensordot/ProdProd9WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2:output:06WheatClassifier_CNN_1/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_CNN_1/dense_1/Tensordot/Prodг
/WheatClassifier_CNN_1/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_CNN_1/dense_1/Tensordot/Const_1ђ
.WheatClassifier_CNN_1/dense_1/Tensordot/Prod_1Prod;WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2_1:output:08WheatClassifier_CNN_1/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_CNN_1/dense_1/Tensordot/Prod_1г
3WheatClassifier_CNN_1/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_CNN_1/dense_1/Tensordot/concat/axisк
.WheatClassifier_CNN_1/dense_1/Tensordot/concatConcatV25WheatClassifier_CNN_1/dense_1/Tensordot/free:output:05WheatClassifier_CNN_1/dense_1/Tensordot/axes:output:0<WheatClassifier_CNN_1/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_CNN_1/dense_1/Tensordot/concatё
-WheatClassifier_CNN_1/dense_1/Tensordot/stackPack5WheatClassifier_CNN_1/dense_1/Tensordot/Prod:output:07WheatClassifier_CNN_1/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_1/Tensordot/stackБ
1WheatClassifier_CNN_1/dense_1/Tensordot/transpose	Transpose?WheatClassifier_CNN_1/layer_normalization_1/batchnorm/add_1:z:07WheatClassifier_CNN_1/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:         d@23
1WheatClassifier_CNN_1/dense_1/Tensordot/transposeЌ
/WheatClassifier_CNN_1/dense_1/Tensordot/ReshapeReshape5WheatClassifier_CNN_1/dense_1/Tensordot/transpose:y:06WheatClassifier_CNN_1/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  21
/WheatClassifier_CNN_1/dense_1/Tensordot/ReshapeЌ
.WheatClassifier_CNN_1/dense_1/Tensordot/MatMulMatMul8WheatClassifier_CNN_1/dense_1/Tensordot/Reshape:output:0>WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ20
.WheatClassifier_CNN_1/dense_1/Tensordot/MatMulГ
/WheatClassifier_CNN_1/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ21
/WheatClassifier_CNN_1/dense_1/Tensordot/Const_2░
5WheatClassifier_CNN_1/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_1/Tensordot/concat_1/axisМ
0WheatClassifier_CNN_1/dense_1/Tensordot/concat_1ConcatV29WheatClassifier_CNN_1/dense_1/Tensordot/GatherV2:output:08WheatClassifier_CNN_1/dense_1/Tensordot/Const_2:output:0>WheatClassifier_CNN_1/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_1/Tensordot/concat_1Ѕ
'WheatClassifier_CNN_1/dense_1/TensordotReshape8WheatClassifier_CNN_1/dense_1/Tensordot/MatMul:product:09WheatClassifier_CNN_1/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2)
'WheatClassifier_CNN_1/dense_1/Tensordotу
4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_cnn_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOpђ
%WheatClassifier_CNN_1/dense_1/BiasAddBiasAdd0WheatClassifier_CNN_1/dense_1/Tensordot:output:0<WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2'
%WheatClassifier_CNN_1/dense_1/BiasAddЎ
(WheatClassifier_CNN_1/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_CNN_1/dense_1/Gelu/mul/xы
&WheatClassifier_CNN_1/dense_1/Gelu/mulMul1WheatClassifier_CNN_1/dense_1/Gelu/mul/x:output:0.WheatClassifier_CNN_1/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:         dђ2(
&WheatClassifier_CNN_1/dense_1/Gelu/mulЏ
)WheatClassifier_CNN_1/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2+
)WheatClassifier_CNN_1/dense_1/Gelu/Cast/x■
*WheatClassifier_CNN_1/dense_1/Gelu/truedivRealDiv.WheatClassifier_CNN_1/dense_1/BiasAdd:output:02WheatClassifier_CNN_1/dense_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2,
*WheatClassifier_CNN_1/dense_1/Gelu/truedivЙ
&WheatClassifier_CNN_1/dense_1/Gelu/ErfErf.WheatClassifier_CNN_1/dense_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2(
&WheatClassifier_CNN_1/dense_1/Gelu/ErfЎ
(WheatClassifier_CNN_1/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(WheatClassifier_CNN_1/dense_1/Gelu/add/x№
&WheatClassifier_CNN_1/dense_1/Gelu/addAddV21WheatClassifier_CNN_1/dense_1/Gelu/add/x:output:0*WheatClassifier_CNN_1/dense_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2(
&WheatClassifier_CNN_1/dense_1/Gelu/addЖ
(WheatClassifier_CNN_1/dense_1/Gelu/mul_1Mul*WheatClassifier_CNN_1/dense_1/Gelu/mul:z:0*WheatClassifier_CNN_1/dense_1/Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2*
(WheatClassifier_CNN_1/dense_1/Gelu/mul_1ы
6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_cnn_1_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
dtype028
6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOpд
,WheatClassifier_CNN_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_CNN_1/dense_2/Tensordot/axesГ
,WheatClassifier_CNN_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_CNN_1/dense_2/Tensordot/free║
-WheatClassifier_CNN_1/dense_2/Tensordot/ShapeShape,WheatClassifier_CNN_1/dense_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_2/Tensordot/Shape░
5WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2/axisу
0WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2GatherV26WheatClassifier_CNN_1/dense_2/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_2/Tensordot/free:output:0>WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2┤
7WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1/axisь
2WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1GatherV26WheatClassifier_CNN_1/dense_2/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_2/Tensordot/axes:output:0@WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1е
-WheatClassifier_CNN_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_CNN_1/dense_2/Tensordot/ConstЭ
,WheatClassifier_CNN_1/dense_2/Tensordot/ProdProd9WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2:output:06WheatClassifier_CNN_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_CNN_1/dense_2/Tensordot/Prodг
/WheatClassifier_CNN_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_CNN_1/dense_2/Tensordot/Const_1ђ
.WheatClassifier_CNN_1/dense_2/Tensordot/Prod_1Prod;WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2_1:output:08WheatClassifier_CNN_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_CNN_1/dense_2/Tensordot/Prod_1г
3WheatClassifier_CNN_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_CNN_1/dense_2/Tensordot/concat/axisк
.WheatClassifier_CNN_1/dense_2/Tensordot/concatConcatV25WheatClassifier_CNN_1/dense_2/Tensordot/free:output:05WheatClassifier_CNN_1/dense_2/Tensordot/axes:output:0<WheatClassifier_CNN_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_CNN_1/dense_2/Tensordot/concatё
-WheatClassifier_CNN_1/dense_2/Tensordot/stackPack5WheatClassifier_CNN_1/dense_2/Tensordot/Prod:output:07WheatClassifier_CNN_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_2/Tensordot/stackЉ
1WheatClassifier_CNN_1/dense_2/Tensordot/transpose	Transpose,WheatClassifier_CNN_1/dense_1/Gelu/mul_1:z:07WheatClassifier_CNN_1/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:         dђ23
1WheatClassifier_CNN_1/dense_2/Tensordot/transposeЌ
/WheatClassifier_CNN_1/dense_2/Tensordot/ReshapeReshape5WheatClassifier_CNN_1/dense_2/Tensordot/transpose:y:06WheatClassifier_CNN_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  21
/WheatClassifier_CNN_1/dense_2/Tensordot/Reshapeќ
.WheatClassifier_CNN_1/dense_2/Tensordot/MatMulMatMul8WheatClassifier_CNN_1/dense_2/Tensordot/Reshape:output:0>WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @20
.WheatClassifier_CNN_1/dense_2/Tensordot/MatMulг
/WheatClassifier_CNN_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@21
/WheatClassifier_CNN_1/dense_2/Tensordot/Const_2░
5WheatClassifier_CNN_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_2/Tensordot/concat_1/axisМ
0WheatClassifier_CNN_1/dense_2/Tensordot/concat_1ConcatV29WheatClassifier_CNN_1/dense_2/Tensordot/GatherV2:output:08WheatClassifier_CNN_1/dense_2/Tensordot/Const_2:output:0>WheatClassifier_CNN_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_2/Tensordot/concat_1ѕ
'WheatClassifier_CNN_1/dense_2/TensordotReshape8WheatClassifier_CNN_1/dense_2/Tensordot/MatMul:product:09WheatClassifier_CNN_1/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2)
'WheatClassifier_CNN_1/dense_2/TensordotТ
4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_cnn_1_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp 
%WheatClassifier_CNN_1/dense_2/BiasAddBiasAdd0WheatClassifier_CNN_1/dense_2/Tensordot:output:0<WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2'
%WheatClassifier_CNN_1/dense_2/BiasAddЎ
(WheatClassifier_CNN_1/dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_CNN_1/dense_2/Gelu/mul/x­
&WheatClassifier_CNN_1/dense_2/Gelu/mulMul1WheatClassifier_CNN_1/dense_2/Gelu/mul/x:output:0.WheatClassifier_CNN_1/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:         d@2(
&WheatClassifier_CNN_1/dense_2/Gelu/mulЏ
)WheatClassifier_CNN_1/dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2+
)WheatClassifier_CNN_1/dense_2/Gelu/Cast/x§
*WheatClassifier_CNN_1/dense_2/Gelu/truedivRealDiv.WheatClassifier_CNN_1/dense_2/BiasAdd:output:02WheatClassifier_CNN_1/dense_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2,
*WheatClassifier_CNN_1/dense_2/Gelu/truedivй
&WheatClassifier_CNN_1/dense_2/Gelu/ErfErf.WheatClassifier_CNN_1/dense_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:         d@2(
&WheatClassifier_CNN_1/dense_2/Gelu/ErfЎ
(WheatClassifier_CNN_1/dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(WheatClassifier_CNN_1/dense_2/Gelu/add/xЬ
&WheatClassifier_CNN_1/dense_2/Gelu/addAddV21WheatClassifier_CNN_1/dense_2/Gelu/add/x:output:0*WheatClassifier_CNN_1/dense_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2(
&WheatClassifier_CNN_1/dense_2/Gelu/addж
(WheatClassifier_CNN_1/dense_2/Gelu/mul_1Mul*WheatClassifier_CNN_1/dense_2/Gelu/mul:z:0*WheatClassifier_CNN_1/dense_2/Gelu/add:z:0*
T0*+
_output_shapes
:         d@2*
(WheatClassifier_CNN_1/dense_2/Gelu/mul_1м
WheatClassifier_CNN_1/add_1/addAddV2,WheatClassifier_CNN_1/dense_2/Gelu/mul_1:z:0!WheatClassifier_CNN_1/add/add:z:0*
T0*+
_output_shapes
:         d@2!
WheatClassifier_CNN_1/add_1/addР
JWheatClassifier_CNN_1/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_CNN_1/layer_normalization_2/moments/mean/reduction_indicesй
8WheatClassifier_CNN_1/layer_normalization_2/moments/meanMean#WheatClassifier_CNN_1/add_1/add:z:0SWheatClassifier_CNN_1/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2:
8WheatClassifier_CNN_1/layer_normalization_2/moments/meanЇ
@WheatClassifier_CNN_1/layer_normalization_2/moments/StopGradientStopGradientAWheatClassifier_CNN_1/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:         d2B
@WheatClassifier_CNN_1/layer_normalization_2/moments/StopGradient╔
EWheatClassifier_CNN_1/layer_normalization_2/moments/SquaredDifferenceSquaredDifference#WheatClassifier_CNN_1/add_1/add:z:0IWheatClassifier_CNN_1/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2G
EWheatClassifier_CNN_1/layer_normalization_2/moments/SquaredDifferenceЖ
NWheatClassifier_CNN_1/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN_1/layer_normalization_2/moments/variance/reduction_indices№
<WheatClassifier_CNN_1/layer_normalization_2/moments/varianceMeanIWheatClassifier_CNN_1/layer_normalization_2/moments/SquaredDifference:z:0WWheatClassifier_CNN_1/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2>
<WheatClassifier_CNN_1/layer_normalization_2/moments/variance┐
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add/y┬
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/addAddV2EWheatClassifier_CNN_1/layer_normalization_2/moments/variance:output:0DWheatClassifier_CNN_1/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2;
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/addЭ
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/RsqrtRsqrt=WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/Rsqrtб
HWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_cnn_1_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOpк
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mulMul?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/Rsqrt:y:0PWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mulЏ
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_1Mul#WheatClassifier_CNN_1/add_1/add:z:0=WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_1╣
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_2MulAWheatClassifier_CNN_1/layer_normalization_2/moments/mean:output:0=WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_2ќ
DWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_cnn_1_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOp┬
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/subSubLWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOp:value:0?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization_2/batchnorm/sub╣
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add_1AddV2?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul_1:z:0=WheatClassifier_CNN_1/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add_1┐
OWheatClassifier_CNN_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpXwheatclassifier_cnn_1_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02Q
OWheatClassifier_CNN_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpѕ
@WheatClassifier_CNN_1/multi_head_attention_1/query/einsum/EinsumEinsum?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add_1:z:0WWheatClassifier_CNN_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2B
@WheatClassifier_CNN_1/multi_head_attention_1/query/einsum/EinsumЮ
EWheatClassifier_CNN_1/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpNwheatclassifier_cnn_1_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02G
EWheatClassifier_CNN_1/multi_head_attention_1/query/add/ReadVariableOp═
6WheatClassifier_CNN_1/multi_head_attention_1/query/addAddV2IWheatClassifier_CNN_1/multi_head_attention_1/query/einsum/Einsum:output:0MWheatClassifier_CNN_1/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@28
6WheatClassifier_CNN_1/multi_head_attention_1/query/add╣
MWheatClassifier_CNN_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_cnn_1_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_CNN_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpѓ
>WheatClassifier_CNN_1/multi_head_attention_1/key/einsum/EinsumEinsum?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add_1:z:0UWheatClassifier_CNN_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2@
>WheatClassifier_CNN_1/multi_head_attention_1/key/einsum/EinsumЌ
CWheatClassifier_CNN_1/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpLwheatclassifier_cnn_1_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_CNN_1/multi_head_attention_1/key/add/ReadVariableOp┼
4WheatClassifier_CNN_1/multi_head_attention_1/key/addAddV2GWheatClassifier_CNN_1/multi_head_attention_1/key/einsum/Einsum:output:0KWheatClassifier_CNN_1/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@26
4WheatClassifier_CNN_1/multi_head_attention_1/key/add┐
OWheatClassifier_CNN_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpXwheatclassifier_cnn_1_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02Q
OWheatClassifier_CNN_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpѕ
@WheatClassifier_CNN_1/multi_head_attention_1/value/einsum/EinsumEinsum?WheatClassifier_CNN_1/layer_normalization_2/batchnorm/add_1:z:0WWheatClassifier_CNN_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2B
@WheatClassifier_CNN_1/multi_head_attention_1/value/einsum/EinsumЮ
EWheatClassifier_CNN_1/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpNwheatclassifier_cnn_1_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02G
EWheatClassifier_CNN_1/multi_head_attention_1/value/add/ReadVariableOp═
6WheatClassifier_CNN_1/multi_head_attention_1/value/addAddV2IWheatClassifier_CNN_1/multi_head_attention_1/value/einsum/Einsum:output:0MWheatClassifier_CNN_1/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@28
6WheatClassifier_CNN_1/multi_head_attention_1/value/addГ
2WheatClassifier_CNN_1/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >24
2WheatClassifier_CNN_1/multi_head_attention_1/Mul/yъ
0WheatClassifier_CNN_1/multi_head_attention_1/MulMul:WheatClassifier_CNN_1/multi_head_attention_1/query/add:z:0;WheatClassifier_CNN_1/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:         d@22
0WheatClassifier_CNN_1/multi_head_attention_1/Mulн
:WheatClassifier_CNN_1/multi_head_attention_1/einsum/EinsumEinsum8WheatClassifier_CNN_1/multi_head_attention_1/key/add:z:04WheatClassifier_CNN_1/multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2<
:WheatClassifier_CNN_1/multi_head_attention_1/einsum/Einsumє
<WheatClassifier_CNN_1/multi_head_attention_1/softmax/SoftmaxSoftmaxCWheatClassifier_CNN_1/multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2>
<WheatClassifier_CNN_1/multi_head_attention_1/softmax/Softmaxї
=WheatClassifier_CNN_1/multi_head_attention_1/dropout/IdentityIdentityFWheatClassifier_CNN_1/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:         dd2?
=WheatClassifier_CNN_1/multi_head_attention_1/dropout/IdentityВ
<WheatClassifier_CNN_1/multi_head_attention_1/einsum_1/EinsumEinsumFWheatClassifier_CNN_1/multi_head_attention_1/dropout/Identity:output:0:WheatClassifier_CNN_1/multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2>
<WheatClassifier_CNN_1/multi_head_attention_1/einsum_1/EinsumЯ
ZWheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpcwheatclassifier_cnn_1_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02\
ZWheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpФ
KWheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/EinsumEinsumEWheatClassifier_CNN_1/multi_head_attention_1/einsum_1/Einsum:output:0bWheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2M
KWheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/Einsum║
PWheatClassifier_CNN_1/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpYwheatclassifier_cnn_1_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02R
PWheatClassifier_CNN_1/multi_head_attention_1/attention_output/add/ReadVariableOpш
AWheatClassifier_CNN_1/multi_head_attention_1/attention_output/addAddV2TWheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/Einsum:output:0XWheatClassifier_CNN_1/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2C
AWheatClassifier_CNN_1/multi_head_attention_1/attention_output/addь
WheatClassifier_CNN_1/add_2/addAddV2EWheatClassifier_CNN_1/multi_head_attention_1/attention_output/add:z:0#WheatClassifier_CNN_1/add_1/add:z:0*
T0*+
_output_shapes
:         d@2!
WheatClassifier_CNN_1/add_2/addР
JWheatClassifier_CNN_1/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_CNN_1/layer_normalization_3/moments/mean/reduction_indicesй
8WheatClassifier_CNN_1/layer_normalization_3/moments/meanMean#WheatClassifier_CNN_1/add_2/add:z:0SWheatClassifier_CNN_1/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2:
8WheatClassifier_CNN_1/layer_normalization_3/moments/meanЇ
@WheatClassifier_CNN_1/layer_normalization_3/moments/StopGradientStopGradientAWheatClassifier_CNN_1/layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:         d2B
@WheatClassifier_CNN_1/layer_normalization_3/moments/StopGradient╔
EWheatClassifier_CNN_1/layer_normalization_3/moments/SquaredDifferenceSquaredDifference#WheatClassifier_CNN_1/add_2/add:z:0IWheatClassifier_CNN_1/layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2G
EWheatClassifier_CNN_1/layer_normalization_3/moments/SquaredDifferenceЖ
NWheatClassifier_CNN_1/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN_1/layer_normalization_3/moments/variance/reduction_indices№
<WheatClassifier_CNN_1/layer_normalization_3/moments/varianceMeanIWheatClassifier_CNN_1/layer_normalization_3/moments/SquaredDifference:z:0WWheatClassifier_CNN_1/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2>
<WheatClassifier_CNN_1/layer_normalization_3/moments/variance┐
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52=
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/add/y┬
9WheatClassifier_CNN_1/layer_normalization_3/batchnorm/addAddV2EWheatClassifier_CNN_1/layer_normalization_3/moments/variance:output:0DWheatClassifier_CNN_1/layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2;
9WheatClassifier_CNN_1/layer_normalization_3/batchnorm/addЭ
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/RsqrtRsqrt=WheatClassifier_CNN_1/layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2=
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/Rsqrtб
HWheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_cnn_1_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul/ReadVariableOpк
9WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mulMul?WheatClassifier_CNN_1/layer_normalization_3/batchnorm/Rsqrt:y:0PWheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mulЏ
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul_1Mul#WheatClassifier_CNN_1/add_2/add:z:0=WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul_1╣
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul_2MulAWheatClassifier_CNN_1/layer_normalization_3/moments/mean:output:0=WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul_2ќ
DWheatClassifier_CNN_1/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_cnn_1_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_CNN_1/layer_normalization_3/batchnorm/ReadVariableOp┬
9WheatClassifier_CNN_1/layer_normalization_3/batchnorm/subSubLWheatClassifier_CNN_1/layer_normalization_3/batchnorm/ReadVariableOp:value:0?WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization_3/batchnorm/sub╣
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/add_1AddV2?WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul_1:z:0=WheatClassifier_CNN_1/layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_3/batchnorm/add_1ы
6WheatClassifier_CNN_1/dense_3/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_cnn_1_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
dtype028
6WheatClassifier_CNN_1/dense_3/Tensordot/ReadVariableOpд
,WheatClassifier_CNN_1/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_CNN_1/dense_3/Tensordot/axesГ
,WheatClassifier_CNN_1/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_CNN_1/dense_3/Tensordot/free═
-WheatClassifier_CNN_1/dense_3/Tensordot/ShapeShape?WheatClassifier_CNN_1/layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_3/Tensordot/Shape░
5WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2/axisу
0WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2GatherV26WheatClassifier_CNN_1/dense_3/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_3/Tensordot/free:output:0>WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2┤
7WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2_1/axisь
2WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2_1GatherV26WheatClassifier_CNN_1/dense_3/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_3/Tensordot/axes:output:0@WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2_1е
-WheatClassifier_CNN_1/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_CNN_1/dense_3/Tensordot/ConstЭ
,WheatClassifier_CNN_1/dense_3/Tensordot/ProdProd9WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2:output:06WheatClassifier_CNN_1/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_CNN_1/dense_3/Tensordot/Prodг
/WheatClassifier_CNN_1/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_CNN_1/dense_3/Tensordot/Const_1ђ
.WheatClassifier_CNN_1/dense_3/Tensordot/Prod_1Prod;WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2_1:output:08WheatClassifier_CNN_1/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_CNN_1/dense_3/Tensordot/Prod_1г
3WheatClassifier_CNN_1/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_CNN_1/dense_3/Tensordot/concat/axisк
.WheatClassifier_CNN_1/dense_3/Tensordot/concatConcatV25WheatClassifier_CNN_1/dense_3/Tensordot/free:output:05WheatClassifier_CNN_1/dense_3/Tensordot/axes:output:0<WheatClassifier_CNN_1/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_CNN_1/dense_3/Tensordot/concatё
-WheatClassifier_CNN_1/dense_3/Tensordot/stackPack5WheatClassifier_CNN_1/dense_3/Tensordot/Prod:output:07WheatClassifier_CNN_1/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_3/Tensordot/stackБ
1WheatClassifier_CNN_1/dense_3/Tensordot/transpose	Transpose?WheatClassifier_CNN_1/layer_normalization_3/batchnorm/add_1:z:07WheatClassifier_CNN_1/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:         d@23
1WheatClassifier_CNN_1/dense_3/Tensordot/transposeЌ
/WheatClassifier_CNN_1/dense_3/Tensordot/ReshapeReshape5WheatClassifier_CNN_1/dense_3/Tensordot/transpose:y:06WheatClassifier_CNN_1/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  21
/WheatClassifier_CNN_1/dense_3/Tensordot/ReshapeЌ
.WheatClassifier_CNN_1/dense_3/Tensordot/MatMulMatMul8WheatClassifier_CNN_1/dense_3/Tensordot/Reshape:output:0>WheatClassifier_CNN_1/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ20
.WheatClassifier_CNN_1/dense_3/Tensordot/MatMulГ
/WheatClassifier_CNN_1/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ21
/WheatClassifier_CNN_1/dense_3/Tensordot/Const_2░
5WheatClassifier_CNN_1/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_3/Tensordot/concat_1/axisМ
0WheatClassifier_CNN_1/dense_3/Tensordot/concat_1ConcatV29WheatClassifier_CNN_1/dense_3/Tensordot/GatherV2:output:08WheatClassifier_CNN_1/dense_3/Tensordot/Const_2:output:0>WheatClassifier_CNN_1/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_3/Tensordot/concat_1Ѕ
'WheatClassifier_CNN_1/dense_3/TensordotReshape8WheatClassifier_CNN_1/dense_3/Tensordot/MatMul:product:09WheatClassifier_CNN_1/dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2)
'WheatClassifier_CNN_1/dense_3/Tensordotу
4WheatClassifier_CNN_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_cnn_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype026
4WheatClassifier_CNN_1/dense_3/BiasAdd/ReadVariableOpђ
%WheatClassifier_CNN_1/dense_3/BiasAddBiasAdd0WheatClassifier_CNN_1/dense_3/Tensordot:output:0<WheatClassifier_CNN_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2'
%WheatClassifier_CNN_1/dense_3/BiasAddЎ
(WheatClassifier_CNN_1/dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_CNN_1/dense_3/Gelu/mul/xы
&WheatClassifier_CNN_1/dense_3/Gelu/mulMul1WheatClassifier_CNN_1/dense_3/Gelu/mul/x:output:0.WheatClassifier_CNN_1/dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:         dђ2(
&WheatClassifier_CNN_1/dense_3/Gelu/mulЏ
)WheatClassifier_CNN_1/dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2+
)WheatClassifier_CNN_1/dense_3/Gelu/Cast/x■
*WheatClassifier_CNN_1/dense_3/Gelu/truedivRealDiv.WheatClassifier_CNN_1/dense_3/BiasAdd:output:02WheatClassifier_CNN_1/dense_3/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2,
*WheatClassifier_CNN_1/dense_3/Gelu/truedivЙ
&WheatClassifier_CNN_1/dense_3/Gelu/ErfErf.WheatClassifier_CNN_1/dense_3/Gelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2(
&WheatClassifier_CNN_1/dense_3/Gelu/ErfЎ
(WheatClassifier_CNN_1/dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(WheatClassifier_CNN_1/dense_3/Gelu/add/x№
&WheatClassifier_CNN_1/dense_3/Gelu/addAddV21WheatClassifier_CNN_1/dense_3/Gelu/add/x:output:0*WheatClassifier_CNN_1/dense_3/Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2(
&WheatClassifier_CNN_1/dense_3/Gelu/addЖ
(WheatClassifier_CNN_1/dense_3/Gelu/mul_1Mul*WheatClassifier_CNN_1/dense_3/Gelu/mul:z:0*WheatClassifier_CNN_1/dense_3/Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2*
(WheatClassifier_CNN_1/dense_3/Gelu/mul_1ы
6WheatClassifier_CNN_1/dense_4/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_cnn_1_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
dtype028
6WheatClassifier_CNN_1/dense_4/Tensordot/ReadVariableOpд
,WheatClassifier_CNN_1/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_CNN_1/dense_4/Tensordot/axesГ
,WheatClassifier_CNN_1/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_CNN_1/dense_4/Tensordot/free║
-WheatClassifier_CNN_1/dense_4/Tensordot/ShapeShape,WheatClassifier_CNN_1/dense_3/Gelu/mul_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_4/Tensordot/Shape░
5WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2/axisу
0WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2GatherV26WheatClassifier_CNN_1/dense_4/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_4/Tensordot/free:output:0>WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2┤
7WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2_1/axisь
2WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2_1GatherV26WheatClassifier_CNN_1/dense_4/Tensordot/Shape:output:05WheatClassifier_CNN_1/dense_4/Tensordot/axes:output:0@WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2_1е
-WheatClassifier_CNN_1/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_CNN_1/dense_4/Tensordot/ConstЭ
,WheatClassifier_CNN_1/dense_4/Tensordot/ProdProd9WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2:output:06WheatClassifier_CNN_1/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_CNN_1/dense_4/Tensordot/Prodг
/WheatClassifier_CNN_1/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_CNN_1/dense_4/Tensordot/Const_1ђ
.WheatClassifier_CNN_1/dense_4/Tensordot/Prod_1Prod;WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2_1:output:08WheatClassifier_CNN_1/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_CNN_1/dense_4/Tensordot/Prod_1г
3WheatClassifier_CNN_1/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_CNN_1/dense_4/Tensordot/concat/axisк
.WheatClassifier_CNN_1/dense_4/Tensordot/concatConcatV25WheatClassifier_CNN_1/dense_4/Tensordot/free:output:05WheatClassifier_CNN_1/dense_4/Tensordot/axes:output:0<WheatClassifier_CNN_1/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_CNN_1/dense_4/Tensordot/concatё
-WheatClassifier_CNN_1/dense_4/Tensordot/stackPack5WheatClassifier_CNN_1/dense_4/Tensordot/Prod:output:07WheatClassifier_CNN_1/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_1/dense_4/Tensordot/stackЉ
1WheatClassifier_CNN_1/dense_4/Tensordot/transpose	Transpose,WheatClassifier_CNN_1/dense_3/Gelu/mul_1:z:07WheatClassifier_CNN_1/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:         dђ23
1WheatClassifier_CNN_1/dense_4/Tensordot/transposeЌ
/WheatClassifier_CNN_1/dense_4/Tensordot/ReshapeReshape5WheatClassifier_CNN_1/dense_4/Tensordot/transpose:y:06WheatClassifier_CNN_1/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  21
/WheatClassifier_CNN_1/dense_4/Tensordot/Reshapeќ
.WheatClassifier_CNN_1/dense_4/Tensordot/MatMulMatMul8WheatClassifier_CNN_1/dense_4/Tensordot/Reshape:output:0>WheatClassifier_CNN_1/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @20
.WheatClassifier_CNN_1/dense_4/Tensordot/MatMulг
/WheatClassifier_CNN_1/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@21
/WheatClassifier_CNN_1/dense_4/Tensordot/Const_2░
5WheatClassifier_CNN_1/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_1/dense_4/Tensordot/concat_1/axisМ
0WheatClassifier_CNN_1/dense_4/Tensordot/concat_1ConcatV29WheatClassifier_CNN_1/dense_4/Tensordot/GatherV2:output:08WheatClassifier_CNN_1/dense_4/Tensordot/Const_2:output:0>WheatClassifier_CNN_1/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_CNN_1/dense_4/Tensordot/concat_1ѕ
'WheatClassifier_CNN_1/dense_4/TensordotReshape8WheatClassifier_CNN_1/dense_4/Tensordot/MatMul:product:09WheatClassifier_CNN_1/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2)
'WheatClassifier_CNN_1/dense_4/TensordotТ
4WheatClassifier_CNN_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_cnn_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4WheatClassifier_CNN_1/dense_4/BiasAdd/ReadVariableOp 
%WheatClassifier_CNN_1/dense_4/BiasAddBiasAdd0WheatClassifier_CNN_1/dense_4/Tensordot:output:0<WheatClassifier_CNN_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2'
%WheatClassifier_CNN_1/dense_4/BiasAddЎ
(WheatClassifier_CNN_1/dense_4/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_CNN_1/dense_4/Gelu/mul/x­
&WheatClassifier_CNN_1/dense_4/Gelu/mulMul1WheatClassifier_CNN_1/dense_4/Gelu/mul/x:output:0.WheatClassifier_CNN_1/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:         d@2(
&WheatClassifier_CNN_1/dense_4/Gelu/mulЏ
)WheatClassifier_CNN_1/dense_4/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2+
)WheatClassifier_CNN_1/dense_4/Gelu/Cast/x§
*WheatClassifier_CNN_1/dense_4/Gelu/truedivRealDiv.WheatClassifier_CNN_1/dense_4/BiasAdd:output:02WheatClassifier_CNN_1/dense_4/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2,
*WheatClassifier_CNN_1/dense_4/Gelu/truedivй
&WheatClassifier_CNN_1/dense_4/Gelu/ErfErf.WheatClassifier_CNN_1/dense_4/Gelu/truediv:z:0*
T0*+
_output_shapes
:         d@2(
&WheatClassifier_CNN_1/dense_4/Gelu/ErfЎ
(WheatClassifier_CNN_1/dense_4/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2*
(WheatClassifier_CNN_1/dense_4/Gelu/add/xЬ
&WheatClassifier_CNN_1/dense_4/Gelu/addAddV21WheatClassifier_CNN_1/dense_4/Gelu/add/x:output:0*WheatClassifier_CNN_1/dense_4/Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2(
&WheatClassifier_CNN_1/dense_4/Gelu/addж
(WheatClassifier_CNN_1/dense_4/Gelu/mul_1Mul*WheatClassifier_CNN_1/dense_4/Gelu/mul:z:0*WheatClassifier_CNN_1/dense_4/Gelu/add:z:0*
T0*+
_output_shapes
:         d@2*
(WheatClassifier_CNN_1/dense_4/Gelu/mul_1н
WheatClassifier_CNN_1/add_3/addAddV2,WheatClassifier_CNN_1/dense_4/Gelu/mul_1:z:0#WheatClassifier_CNN_1/add_2/add:z:0*
T0*+
_output_shapes
:         d@2!
WheatClassifier_CNN_1/add_3/addР
JWheatClassifier_CNN_1/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_CNN_1/layer_normalization_4/moments/mean/reduction_indicesй
8WheatClassifier_CNN_1/layer_normalization_4/moments/meanMean#WheatClassifier_CNN_1/add_3/add:z:0SWheatClassifier_CNN_1/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2:
8WheatClassifier_CNN_1/layer_normalization_4/moments/meanЇ
@WheatClassifier_CNN_1/layer_normalization_4/moments/StopGradientStopGradientAWheatClassifier_CNN_1/layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:         d2B
@WheatClassifier_CNN_1/layer_normalization_4/moments/StopGradient╔
EWheatClassifier_CNN_1/layer_normalization_4/moments/SquaredDifferenceSquaredDifference#WheatClassifier_CNN_1/add_3/add:z:0IWheatClassifier_CNN_1/layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2G
EWheatClassifier_CNN_1/layer_normalization_4/moments/SquaredDifferenceЖ
NWheatClassifier_CNN_1/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN_1/layer_normalization_4/moments/variance/reduction_indices№
<WheatClassifier_CNN_1/layer_normalization_4/moments/varianceMeanIWheatClassifier_CNN_1/layer_normalization_4/moments/SquaredDifference:z:0WWheatClassifier_CNN_1/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2>
<WheatClassifier_CNN_1/layer_normalization_4/moments/variance┐
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52=
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/add/y┬
9WheatClassifier_CNN_1/layer_normalization_4/batchnorm/addAddV2EWheatClassifier_CNN_1/layer_normalization_4/moments/variance:output:0DWheatClassifier_CNN_1/layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2;
9WheatClassifier_CNN_1/layer_normalization_4/batchnorm/addЭ
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/RsqrtRsqrt=WheatClassifier_CNN_1/layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2=
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/Rsqrtб
HWheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_cnn_1_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul/ReadVariableOpк
9WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mulMul?WheatClassifier_CNN_1/layer_normalization_4/batchnorm/Rsqrt:y:0PWheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mulЏ
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul_1Mul#WheatClassifier_CNN_1/add_3/add:z:0=WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul_1╣
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul_2MulAWheatClassifier_CNN_1/layer_normalization_4/moments/mean:output:0=WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul_2ќ
DWheatClassifier_CNN_1/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_cnn_1_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_CNN_1/layer_normalization_4/batchnorm/ReadVariableOp┬
9WheatClassifier_CNN_1/layer_normalization_4/batchnorm/subSubLWheatClassifier_CNN_1/layer_normalization_4/batchnorm/ReadVariableOp:value:0?WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2;
9WheatClassifier_CNN_1/layer_normalization_4/batchnorm/sub╣
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/add_1AddV2?WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul_1:z:0=WheatClassifier_CNN_1/layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2=
;WheatClassifier_CNN_1/layer_normalization_4/batchnorm/add_1╣
#WheatClassifier_CNN_1/reshape/ShapeShape?WheatClassifier_CNN_1/layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2%
#WheatClassifier_CNN_1/reshape/Shape░
1WheatClassifier_CNN_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_CNN_1/reshape/strided_slice/stack┤
3WheatClassifier_CNN_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_1/reshape/strided_slice/stack_1┤
3WheatClassifier_CNN_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_1/reshape/strided_slice/stack_2ќ
+WheatClassifier_CNN_1/reshape/strided_sliceStridedSlice,WheatClassifier_CNN_1/reshape/Shape:output:0:WheatClassifier_CNN_1/reshape/strided_slice/stack:output:0<WheatClassifier_CNN_1/reshape/strided_slice/stack_1:output:0<WheatClassifier_CNN_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+WheatClassifier_CNN_1/reshape/strided_sliceа
-WheatClassifier_CNN_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
2/
-WheatClassifier_CNN_1/reshape/Reshape/shape/1а
-WheatClassifier_CNN_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2/
-WheatClassifier_CNN_1/reshape/Reshape/shape/2а
-WheatClassifier_CNN_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2/
-WheatClassifier_CNN_1/reshape/Reshape/shape/3Ь
+WheatClassifier_CNN_1/reshape/Reshape/shapePack4WheatClassifier_CNN_1/reshape/strided_slice:output:06WheatClassifier_CNN_1/reshape/Reshape/shape/1:output:06WheatClassifier_CNN_1/reshape/Reshape/shape/2:output:06WheatClassifier_CNN_1/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+WheatClassifier_CNN_1/reshape/Reshape/shapeі
%WheatClassifier_CNN_1/reshape/ReshapeReshape?WheatClassifier_CNN_1/layer_normalization_4/batchnorm/add_1:z:04WheatClassifier_CNN_1/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         

@2'
%WheatClassifier_CNN_1/reshape/ReshapeВ
2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_1_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype024
2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOpБ
#WheatClassifier_CNN_1/conv_1/Conv2DConv2D.WheatClassifier_CNN_1/reshape/Reshape:output:0:WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_1/conv_1/Conv2Dс
3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_1_conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_1/conv_1/BiasAddBiasAdd,WheatClassifier_CNN_1/conv_1/Conv2D:output:0;WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2&
$WheatClassifier_CNN_1/conv_1/BiasAddВ
2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_1_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype024
2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOpб
#WheatClassifier_CNN_1/conv_2/Conv2DConv2D-WheatClassifier_CNN_1/conv_1/BiasAdd:output:0:WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_1/conv_2/Conv2Dс
3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_1_conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_1/conv_2/BiasAddBiasAdd,WheatClassifier_CNN_1/conv_2/Conv2D:output:0;WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2&
$WheatClassifier_CNN_1/conv_2/BiasAddВ
2WheatClassifier_CNN_1/conv_3/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_1_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype024
2WheatClassifier_CNN_1/conv_3/Conv2D/ReadVariableOpб
#WheatClassifier_CNN_1/conv_3/Conv2DConv2D-WheatClassifier_CNN_1/conv_2/BiasAdd:output:0:WheatClassifier_CNN_1/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2%
#WheatClassifier_CNN_1/conv_3/Conv2Dс
3WheatClassifier_CNN_1/conv_3/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_1_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_1/conv_3/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_1/conv_3/BiasAddBiasAdd,WheatClassifier_CNN_1/conv_3/Conv2D:output:0;WheatClassifier_CNN_1/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2&
$WheatClassifier_CNN_1/conv_3/BiasAddВ
2WheatClassifier_CNN_1/conv_4/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_1_conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2WheatClassifier_CNN_1/conv_4/Conv2D/ReadVariableOpб
#WheatClassifier_CNN_1/conv_4/Conv2DConv2D-WheatClassifier_CNN_1/conv_3/BiasAdd:output:0:WheatClassifier_CNN_1/conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2%
#WheatClassifier_CNN_1/conv_4/Conv2Dс
3WheatClassifier_CNN_1/conv_4/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_1_conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_1/conv_4/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_1/conv_4/BiasAddBiasAdd,WheatClassifier_CNN_1/conv_4/Conv2D:output:0;WheatClassifier_CNN_1/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2&
$WheatClassifier_CNN_1/conv_4/BiasAddД
)WheatClassifier_CNN_1/flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2+
)WheatClassifier_CNN_1/flatten_layer/ConstЩ
+WheatClassifier_CNN_1/flatten_layer/ReshapeReshape-WheatClassifier_CNN_1/conv_4/BiasAdd:output:02WheatClassifier_CNN_1/flatten_layer/Const:output:0*
T0*'
_output_shapes
:         @2-
+WheatClassifier_CNN_1/flatten_layer/Reshapeя
0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_1_fc_1_matmul_readvariableop_resource*
_output_shapes

:@2*
dtype022
0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOpЫ
!WheatClassifier_CNN_1/FC_1/MatMulMatMul4WheatClassifier_CNN_1/flatten_layer/Reshape:output:08WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22#
!WheatClassifier_CNN_1/FC_1/MatMulП
1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_1_fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOpь
"WheatClassifier_CNN_1/FC_1/BiasAddBiasAdd+WheatClassifier_CNN_1/FC_1/MatMul:product:09WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22$
"WheatClassifier_CNN_1/FC_1/BiasAdd¤
,WheatClassifier_CNN_1/leaky_ReLu_1/LeakyRelu	LeakyRelu+WheatClassifier_CNN_1/FC_1/BiasAdd:output:0*'
_output_shapes
:         2*
alpha%џЎЎ>2.
,WheatClassifier_CNN_1/leaky_ReLu_1/LeakyReluя
0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_1_fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype022
0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOpЭ
!WheatClassifier_CNN_1/FC_2/MatMulMatMul:WheatClassifier_CNN_1/leaky_ReLu_1/LeakyRelu:activations:08WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22#
!WheatClassifier_CNN_1/FC_2/MatMulП
1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_1_fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOpь
"WheatClassifier_CNN_1/FC_2/BiasAddBiasAdd+WheatClassifier_CNN_1/FC_2/MatMul:product:09WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22$
"WheatClassifier_CNN_1/FC_2/BiasAdd¤
,WheatClassifier_CNN_1/leaky_ReLu_2/LeakyRelu	LeakyRelu+WheatClassifier_CNN_1/FC_2/BiasAdd:output:0*'
_output_shapes
:         2*
alpha%џЎЎ>2.
,WheatClassifier_CNN_1/leaky_ReLu_2/LeakyReluШ
8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOpReadVariableOpAwheatclassifier_cnn_1_output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02:
8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOpљ
)WheatClassifier_CNN_1/output_layer/MatMulMatMul:WheatClassifier_CNN_1/leaky_ReLu_2/LeakyRelu:activations:0@WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2+
)WheatClassifier_CNN_1/output_layer/MatMulш
9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOpReadVariableOpBwheatclassifier_cnn_1_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOpЇ
*WheatClassifier_CNN_1/output_layer/BiasAddBiasAdd3WheatClassifier_CNN_1/output_layer/MatMul:product:0AWheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2,
*WheatClassifier_CNN_1/output_layer/BiasAdd╩
*WheatClassifier_CNN_1/output_layer/SoftmaxSoftmax3WheatClassifier_CNN_1/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:         2,
*WheatClassifier_CNN_1/output_layer/Softmax╝
IdentityIdentity4WheatClassifier_CNN_1/output_layer/Softmax:softmax:02^WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp2^WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp4^WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp4^WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp4^WheatClassifier_CNN_1/conv_3/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_1/conv_3/Conv2D/ReadVariableOp4^WheatClassifier_CNN_1/conv_4/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_1/conv_4/Conv2D/ReadVariableOp5^WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp7^WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp5^WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp7^WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOp5^WheatClassifier_CNN_1/dense_3/BiasAdd/ReadVariableOp7^WheatClassifier_CNN_1/dense_3/Tensordot/ReadVariableOp5^WheatClassifier_CNN_1/dense_4/BiasAdd/ReadVariableOp7^WheatClassifier_CNN_1/dense_4/Tensordot/ReadVariableOpC^WheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOpG^WheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOpE^WheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOpI^WheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOpE^WheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOpI^WheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOpE^WheatClassifier_CNN_1/layer_normalization_3/batchnorm/ReadVariableOpI^WheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul/ReadVariableOpE^WheatClassifier_CNN_1/layer_normalization_4/batchnorm/ReadVariableOpI^WheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul/ReadVariableOpO^WheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOpY^WheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpB^WheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOpL^WheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOpD^WheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOpN^WheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOpD^WheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOpN^WheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOpQ^WheatClassifier_CNN_1/multi_head_attention_1/attention_output/add/ReadVariableOp[^WheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpD^WheatClassifier_CNN_1/multi_head_attention_1/key/add/ReadVariableOpN^WheatClassifier_CNN_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpF^WheatClassifier_CNN_1/multi_head_attention_1/query/add/ReadVariableOpP^WheatClassifier_CNN_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpF^WheatClassifier_CNN_1/multi_head_attention_1/value/add/ReadVariableOpP^WheatClassifier_CNN_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:^WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp9^WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOpA^WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOpC^WheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOp?^WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp1WheatClassifier_CNN_1/FC_1/BiasAdd/ReadVariableOp2d
0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp0WheatClassifier_CNN_1/FC_1/MatMul/ReadVariableOp2f
1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp1WheatClassifier_CNN_1/FC_2/BiasAdd/ReadVariableOp2d
0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp0WheatClassifier_CNN_1/FC_2/MatMul/ReadVariableOp2j
3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp3WheatClassifier_CNN_1/conv_1/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp2WheatClassifier_CNN_1/conv_1/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp3WheatClassifier_CNN_1/conv_2/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp2WheatClassifier_CNN_1/conv_2/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_1/conv_3/BiasAdd/ReadVariableOp3WheatClassifier_CNN_1/conv_3/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_1/conv_3/Conv2D/ReadVariableOp2WheatClassifier_CNN_1/conv_3/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_1/conv_4/BiasAdd/ReadVariableOp3WheatClassifier_CNN_1/conv_4/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_1/conv_4/Conv2D/ReadVariableOp2WheatClassifier_CNN_1/conv_4/Conv2D/ReadVariableOp2l
4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp4WheatClassifier_CNN_1/dense_1/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp6WheatClassifier_CNN_1/dense_1/Tensordot/ReadVariableOp2l
4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp4WheatClassifier_CNN_1/dense_2/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOp6WheatClassifier_CNN_1/dense_2/Tensordot/ReadVariableOp2l
4WheatClassifier_CNN_1/dense_3/BiasAdd/ReadVariableOp4WheatClassifier_CNN_1/dense_3/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN_1/dense_3/Tensordot/ReadVariableOp6WheatClassifier_CNN_1/dense_3/Tensordot/ReadVariableOp2l
4WheatClassifier_CNN_1/dense_4/BiasAdd/ReadVariableOp4WheatClassifier_CNN_1/dense_4/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN_1/dense_4/Tensordot/ReadVariableOp6WheatClassifier_CNN_1/dense_4/Tensordot/ReadVariableOp2ѕ
BWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOpBWheatClassifier_CNN_1/layer_normalization/batchnorm/ReadVariableOp2љ
FWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOpFWheatClassifier_CNN_1/layer_normalization/batchnorm/mul/ReadVariableOp2ї
DWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOpDWheatClassifier_CNN_1/layer_normalization_1/batchnorm/ReadVariableOp2ћ
HWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOpHWheatClassifier_CNN_1/layer_normalization_1/batchnorm/mul/ReadVariableOp2ї
DWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOpDWheatClassifier_CNN_1/layer_normalization_2/batchnorm/ReadVariableOp2ћ
HWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOpHWheatClassifier_CNN_1/layer_normalization_2/batchnorm/mul/ReadVariableOp2ї
DWheatClassifier_CNN_1/layer_normalization_3/batchnorm/ReadVariableOpDWheatClassifier_CNN_1/layer_normalization_3/batchnorm/ReadVariableOp2ћ
HWheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul/ReadVariableOpHWheatClassifier_CNN_1/layer_normalization_3/batchnorm/mul/ReadVariableOp2ї
DWheatClassifier_CNN_1/layer_normalization_4/batchnorm/ReadVariableOpDWheatClassifier_CNN_1/layer_normalization_4/batchnorm/ReadVariableOp2ћ
HWheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul/ReadVariableOpHWheatClassifier_CNN_1/layer_normalization_4/batchnorm/mul/ReadVariableOp2а
NWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOpNWheatClassifier_CNN_1/multi_head_attention/attention_output/add/ReadVariableOp2┤
XWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpXWheatClassifier_CNN_1/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2є
AWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOpAWheatClassifier_CNN_1/multi_head_attention/key/add/ReadVariableOp2џ
KWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOpKWheatClassifier_CNN_1/multi_head_attention/key/einsum/Einsum/ReadVariableOp2і
CWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOpCWheatClassifier_CNN_1/multi_head_attention/query/add/ReadVariableOp2ъ
MWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOpMWheatClassifier_CNN_1/multi_head_attention/query/einsum/Einsum/ReadVariableOp2і
CWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOpCWheatClassifier_CNN_1/multi_head_attention/value/add/ReadVariableOp2ъ
MWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOpMWheatClassifier_CNN_1/multi_head_attention/value/einsum/Einsum/ReadVariableOp2ц
PWheatClassifier_CNN_1/multi_head_attention_1/attention_output/add/ReadVariableOpPWheatClassifier_CNN_1/multi_head_attention_1/attention_output/add/ReadVariableOp2И
ZWheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpZWheatClassifier_CNN_1/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2і
CWheatClassifier_CNN_1/multi_head_attention_1/key/add/ReadVariableOpCWheatClassifier_CNN_1/multi_head_attention_1/key/add/ReadVariableOp2ъ
MWheatClassifier_CNN_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpMWheatClassifier_CNN_1/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2ј
EWheatClassifier_CNN_1/multi_head_attention_1/query/add/ReadVariableOpEWheatClassifier_CNN_1/multi_head_attention_1/query/add/ReadVariableOp2б
OWheatClassifier_CNN_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpOWheatClassifier_CNN_1/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2ј
EWheatClassifier_CNN_1/multi_head_attention_1/value/add/ReadVariableOpEWheatClassifier_CNN_1/multi_head_attention_1/value/add/ReadVariableOp2б
OWheatClassifier_CNN_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpOWheatClassifier_CNN_1/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2v
9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp9WheatClassifier_CNN_1/output_layer/BiasAdd/ReadVariableOp2t
8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOp8WheatClassifier_CNN_1/output_layer/MatMul/ReadVariableOp2ё
@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOp@WheatClassifier_CNN_1/patch_encoder/dense/BiasAdd/ReadVariableOp2ѕ
BWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOpBWheatClassifier_CNN_1/patch_encoder/dense/Tensordot/ReadVariableOp2ђ
>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup>WheatClassifier_CNN_1/patch_encoder/embedding/embedding_lookup:\ X
/
_output_shapes
:         dd
%
_user_specified_nameinput_layer
┌

В
7__inference_multi_head_attention_1_layer_call_fn_625330	
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
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_6221282
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
о

Ж
5__inference_multi_head_attention_layer_call_fn_625051	
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
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_6229202
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
з
И
.__inference_patch_encoder_layer_call_fn_624899	
patch
unknown:	г@
	unknown_0:@
	unknown_1:d@
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallpatchunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_6218442
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  г: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:                  г

_user_specified_namepatch
й
Ю
4__inference_layer_normalization_layer_call_fn_624930

inputs
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_6218742
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
ш-
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_621915	
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
identityѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpК
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЎ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┴
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpЉ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpК
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЎ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
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
:         d@2
Mulа
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2
softmax/SoftmaxЁ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:         dd2
dropout/IdentityИ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpэ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┴
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
attention_output/addѓ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 2J
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
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
г
ќ
(__inference_dense_2_layer_call_fn_625188

inputs
unknown:	ђ@
	unknown_0:@
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6220512
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         dђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         dђ
 
_user_specified_nameinputs
Б
љ
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_625085

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
Ш/
Ж
I__inference_patch_encoder_layer_call_and_return_conditional_losses_624888	
patch:
'dense_tensordot_readvariableop_resource:	г@3
%dense_biasadd_readvariableop_resource:@3
!embedding_embedding_lookup_624881:d@
identityѕбdense/BiasAdd/ReadVariableOpбdense/Tensordot/ReadVariableOpбembedding/embedding_lookup\
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
value	B :d2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltau
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:d2
rangeЕ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	г@*
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
dense/Tensordot/Shapeђ
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis№
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2ё
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisш
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
dense/Tensordot/Constў
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
dense/Tensordot/Const_1а
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
dense/Tensordot/concat/axis╬
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concatц
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stackФ
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  г2
dense/Tensordot/transposeи
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense/Tensordot/ReshapeХ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense/Tensordot/Const_2ђ
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis█
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1▒
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @2
dense/Tensordotъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpе
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @2
dense/BiasAddъ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_624881range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/624881*
_output_shapes

:d@*
dtype02
embedding/embedding_lookupѕ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/624881*
_output_shapes

:d@2%
#embedding/embedding_lookup/Identity▒
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:d@2'
%embedding/embedding_lookup/Identity_1Љ
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         d@2
add╝
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  г: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:                  г

_user_specified_namepatch
ъч
Д.
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_624829

inputsH
5patch_encoder_dense_tensordot_readvariableop_resource:	г@A
3patch_encoder_dense_biasadd_readvariableop_resource:@A
/patch_encoder_embedding_embedding_lookup_624461:d@G
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
)dense_1_tensordot_readvariableop_resource:	@ђ6
'dense_1_biasadd_readvariableop_resource:	ђ<
)dense_2_tensordot_readvariableop_resource:	ђ@5
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
)dense_3_tensordot_readvariableop_resource:	@ђ6
'dense_3_biasadd_readvariableop_resource:	ђ<
)dense_4_tensordot_readvariableop_resource:	ђ@5
'dense_4_biasadd_readvariableop_resource:@I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_4_batchnorm_readvariableop_resource:@?
%conv_1_conv2d_readvariableop_resource:@
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
&conv_4_biasadd_readvariableop_resource:5
#fc_1_matmul_readvariableop_resource:@22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identityѕбFC_1/BiasAdd/ReadVariableOpбFC_1/MatMul/ReadVariableOpбFC_2/BiasAdd/ReadVariableOpбFC_2/MatMul/ReadVariableOpбconv_1/BiasAdd/ReadVariableOpбconv_1/Conv2D/ReadVariableOpбconv_2/BiasAdd/ReadVariableOpбconv_2/Conv2D/ReadVariableOpбconv_3/BiasAdd/ReadVariableOpбconv_3/Conv2D/ReadVariableOpбconv_4/BiasAdd/ReadVariableOpбconv_4/Conv2D/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpб dense_1/Tensordot/ReadVariableOpбdense_2/BiasAdd/ReadVariableOpб dense_2/Tensordot/ReadVariableOpбdense_3/BiasAdd/ReadVariableOpб dense_3/Tensordot/ReadVariableOpбdense_4/BiasAdd/ReadVariableOpб dense_4/Tensordot/ReadVariableOpб,layer_normalization/batchnorm/ReadVariableOpб0layer_normalization/batchnorm/mul/ReadVariableOpб.layer_normalization_1/batchnorm/ReadVariableOpб2layer_normalization_1/batchnorm/mul/ReadVariableOpб.layer_normalization_2/batchnorm/ReadVariableOpб2layer_normalization_2/batchnorm/mul/ReadVariableOpб.layer_normalization_3/batchnorm/ReadVariableOpб2layer_normalization_3/batchnorm/mul/ReadVariableOpб.layer_normalization_4/batchnorm/ReadVariableOpб2layer_normalization_4/batchnorm/mul/ReadVariableOpб8multi_head_attention/attention_output/add/ReadVariableOpбBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpб+multi_head_attention/key/add/ReadVariableOpб5multi_head_attention/key/einsum/Einsum/ReadVariableOpб-multi_head_attention/query/add/ReadVariableOpб7multi_head_attention/query/einsum/Einsum/ReadVariableOpб-multi_head_attention/value/add/ReadVariableOpб7multi_head_attention/value/einsum/Einsum/ReadVariableOpб:multi_head_attention_1/attention_output/add/ReadVariableOpбDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpб-multi_head_attention_1/key/add/ReadVariableOpб7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpб/multi_head_attention_1/query/add/ReadVariableOpб9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpб/multi_head_attention_1/value/add/ReadVariableOpб9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpб#output_layer/BiasAdd/ReadVariableOpб"output_layer/MatMul/ReadVariableOpб*patch_encoder/dense/BiasAdd/ReadVariableOpб,patch_encoder/dense/Tensordot/ReadVariableOpб(patch_encoder/embedding/embedding_lookupT
patches/ShapeShapeinputs*
T0*
_output_shapes
:2
patches/Shapeё
patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
patches/strided_slice/stackѕ
patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_1ѕ
patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_2њ
patches/strided_sliceStridedSlicepatches/Shape:output:0$patches/strided_slice/stack:output:0&patches/strided_slice/stack_1:output:0&patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
patches/strided_sliceС
patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*0
_output_shapes
:         

г*
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
         2
patches/Reshape/shape/1u
patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :г2
patches/Reshape/shape/2╚
patches/Reshape/shapePackpatches/strided_slice:output:0 patches/Reshape/shape/1:output:0 patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
patches/Reshape/shape┤
patches/ReshapeReshape%patches/ExtractImagePatches:patches:0patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  г2
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
value	B :d2
patch_encoder/range/limitx
patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder/range/delta╗
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes
:d2
patch_encoder/rangeМ
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	г@*
dtype02.
,patch_encoder/dense/Tensordot/ReadVariableOpњ
"patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"patch_encoder/dense/Tensordot/axesЎ
"patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"patch_encoder/dense/Tensordot/freeњ
#patch_encoder/dense/Tensordot/ShapeShapepatches/Reshape:output:0*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/Shapeю
+patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/GatherV2/axisх
&patch_encoder/dense/Tensordot/GatherV2GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/free:output:04patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/GatherV2а
-patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-patch_encoder/dense/Tensordot/GatherV2_1/axis╗
(patch_encoder/dense/Tensordot/GatherV2_1GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/axes:output:06patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(patch_encoder/dense/Tensordot/GatherV2_1ћ
#patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#patch_encoder/dense/Tensordot/Constл
"patch_encoder/dense/Tensordot/ProdProd/patch_encoder/dense/Tensordot/GatherV2:output:0,patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"patch_encoder/dense/Tensordot/Prodў
%patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%patch_encoder/dense/Tensordot/Const_1п
$patch_encoder/dense/Tensordot/Prod_1Prod1patch_encoder/dense/Tensordot/GatherV2_1:output:0.patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$patch_encoder/dense/Tensordot/Prod_1ў
)patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)patch_encoder/dense/Tensordot/concat/axisћ
$patch_encoder/dense/Tensordot/concatConcatV2+patch_encoder/dense/Tensordot/free:output:0+patch_encoder/dense/Tensordot/axes:output:02patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$patch_encoder/dense/Tensordot/concat▄
#patch_encoder/dense/Tensordot/stackPack+patch_encoder/dense/Tensordot/Prod:output:0-patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/stackУ
'patch_encoder/dense/Tensordot/transpose	Transposepatches/Reshape:output:0-patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  г2)
'patch_encoder/dense/Tensordot/transpose№
%patch_encoder/dense/Tensordot/ReshapeReshape+patch_encoder/dense/Tensordot/transpose:y:0,patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2'
%patch_encoder/dense/Tensordot/ReshapeЬ
$patch_encoder/dense/Tensordot/MatMulMatMul.patch_encoder/dense/Tensordot/Reshape:output:04patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2&
$patch_encoder/dense/Tensordot/MatMulў
%patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2'
%patch_encoder/dense/Tensordot/Const_2ю
+patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/concat_1/axisА
&patch_encoder/dense/Tensordot/concat_1ConcatV2/patch_encoder/dense/Tensordot/GatherV2:output:0.patch_encoder/dense/Tensordot/Const_2:output:04patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/concat_1ж
patch_encoder/dense/TensordotReshape.patch_encoder/dense/Tensordot/MatMul:product:0/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @2
patch_encoder/dense/Tensordot╚
*patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp3patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*patch_encoder/dense/BiasAdd/ReadVariableOpЯ
patch_encoder/dense/BiasAddBiasAdd&patch_encoder/dense/Tensordot:output:02patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @2
patch_encoder/dense/BiasAddС
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_624461patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/624461*
_output_shapes

:d@*
dtype02*
(patch_encoder/embedding/embedding_lookup└
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/624461*
_output_shapes

:d@23
1patch_encoder/embedding/embedding_lookup/Identity█
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:d@25
3patch_encoder/embedding/embedding_lookup/Identity_1╔
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         d@2
patch_encoder/add▓
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indicesу
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2"
 layer_normalization/moments/mean┼
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:         d2*
(layer_normalization/moments/StopGradientз
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2/
-layer_normalization/moments/SquaredDifference║
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indicesЈ
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2&
$layer_normalization/moments/varianceЈ
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52%
#layer_normalization/batchnorm/add/yР
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2#
!layer_normalization/batchnorm/add░
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2%
#layer_normalization/batchnorm/Rsqrt┌
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOpТ
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2#
!layer_normalization/batchnorm/mul┼
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization/batchnorm/mul_1┘
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization/batchnorm/mul_2╬
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOpР
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2#
!layer_normalization/batchnorm/sub┘
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization/batchnorm/add_1э
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOpе
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/EinsumН
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOpь
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2 
multi_head_attention/query/addы
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOpб
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsum¤
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOpт
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
multi_head_attention/key/addэ
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOpе
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/EinsumН
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOpь
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/yЙ
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:         d@2
multi_head_attention/MulЗ
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/EinsumЙ
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2&
$multi_head_attention/softmax/SoftmaxЮ
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2,
*multi_head_attention/dropout/dropout/ConstЩ
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         dd2*
(multi_head_attention/dropout/dropout/MulХ
*multi_head_attention/dropout/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2,
*multi_head_attention/dropout/dropout/ShapeЊ
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         dd*
dtype02C
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniform»
3multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=25
3multi_head_attention/dropout/dropout/GreaterEqual/y║
1multi_head_attention/dropout/dropout/GreaterEqualGreaterEqualJmulti_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0<multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         dd23
1multi_head_attention/dropout/dropout/GreaterEqualя
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         dd2+
)multi_head_attention/dropout/dropout/CastШ
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         dd2,
*multi_head_attention/dropout/dropout/Mul_1ї
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/Einsumў
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp╦
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/EinsumЫ
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOpЋ
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2+
)multi_head_attention/attention_output/addЌ
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*+
_output_shapes
:         d@2	
add/addХ
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indicesс
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2$
"layer_normalization_1/moments/mean╦
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:         d2,
*layer_normalization_1/moments/StopGradient№
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@21
/layer_normalization_1/moments/SquaredDifferenceЙ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indicesЌ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2(
&layer_normalization_1/moments/varianceЊ
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52'
%layer_normalization_1/batchnorm/add/yЖ
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2%
#layer_normalization_1/batchnorm/addХ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2'
%layer_normalization_1/batchnorm/RsqrtЯ
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOpЬ
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_1/batchnorm/mul┴
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_1/batchnorm/mul_1р
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_1/batchnorm/mul_2н
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOpЖ
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_1/batchnorm/subр
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_1/batchnorm/add_1»
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axesЂ
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/freeІ
dense_1/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shapeё
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axisщ
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2ѕ
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis 
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
dense_1/Tensordot/Constа
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prodђ
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1е
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1ђ
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axisп
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concatг
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stack╦
dense_1/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:         d@2
dense_1/Tensordot/transpose┐
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_1/Tensordot/Reshape┐
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_1/Tensordot/MatMulЂ
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ2
dense_1/Tensordot/Const_2ё
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axisт
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1▒
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2
dense_1/TensordotЦ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_1/BiasAdd/ReadVariableOpе
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/xЎ
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
dense_1/Gelu/Cast/xд
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/truediv|
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dense_1/Gelu/add/xЌ
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/addњ
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2
dense_1/Gelu/mul_1»
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axesЂ
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
dense_2/Tensordot/Shapeё
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axisщ
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2ѕ
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis 
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
dense_2/Tensordot/Constа
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prodђ
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1е
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1ђ
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axisп
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concatг
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack╣
dense_2/Tensordot/transpose	Transposedense_1/Gelu/mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:         dђ2
dense_2/Tensordot/transpose┐
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_2/Tensordot/ReshapeЙ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_2/Tensordot/MatMulђ
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_2/Tensordot/Const_2ё
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axisт
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1░
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2
dense_2/Tensordotц
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOpД
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/xў
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
dense_2/Gelu/Cast/xЦ
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/truediv{
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dense_2/Gelu/add/xќ
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/addЉ
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*+
_output_shapes
:         d@2
dense_2/Gelu/mul_1z
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*+
_output_shapes
:         d@2
	add_1/addХ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesт
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2$
"layer_normalization_2/moments/mean╦
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:         d2,
*layer_normalization_2/moments/StopGradientы
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@21
/layer_normalization_2/moments/SquaredDifferenceЙ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesЌ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2(
&layer_normalization_2/moments/varianceЊ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52'
%layer_normalization_2/batchnorm/add/yЖ
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2%
#layer_normalization_2/batchnorm/addХ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2'
%layer_normalization_2/batchnorm/RsqrtЯ
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpЬ
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_2/batchnorm/mul├
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_2/batchnorm/mul_1р
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_2/batchnorm/mul_2н
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpЖ
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_2/batchnorm/subр
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_2/batchnorm/add_1§
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp░
*multi_head_attention_1/query/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsum█
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/query/add/ReadVariableOpш
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2"
 multi_head_attention_1/query/addэ
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpф
(multi_head_attention_1/key/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/EinsumН
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpь
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2 
multi_head_attention_1/key/add§
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp░
*multi_head_attention_1/value/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsum█
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/value/add/ReadVariableOpш
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2"
 multi_head_attention_1/value/addЂ
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_1/Mul/yк
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:         d@2
multi_head_attention_1/MulЧ
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum─
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2(
&multi_head_attention_1/softmax/SoftmaxА
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2.
,multi_head_attention_1/dropout/dropout/Constѓ
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         dd2,
*multi_head_attention_1/dropout/dropout/Mul╝
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_1/dropout/dropout/ShapeЎ
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         dd*
dtype02E
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform│
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=27
5multi_head_attention_1/dropout/dropout/GreaterEqual/y┬
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         dd25
3multi_head_attention_1/dropout/dropout/GreaterEqualС
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         dd2-
+multi_head_attention_1/dropout/dropout/Cast■
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         dd2.
,multi_head_attention_1/dropout/dropout/Mul_1ћ
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/Einsumъ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpМ
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/EinsumЭ
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOpЮ
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2-
+multi_head_attention_1/attention_output/addЋ
	add_2/addAddV2/multi_head_attention_1/attention_output/add:z:0add_1/add:z:0*
T0*+
_output_shapes
:         d@2
	add_2/addХ
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesт
"layer_normalization_3/moments/meanMeanadd_2/add:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2$
"layer_normalization_3/moments/mean╦
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*+
_output_shapes
:         d2,
*layer_normalization_3/moments/StopGradientы
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd_2/add:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@21
/layer_normalization_3/moments/SquaredDifferenceЙ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indicesЌ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2(
&layer_normalization_3/moments/varianceЊ
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52'
%layer_normalization_3/batchnorm/add/yЖ
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2%
#layer_normalization_3/batchnorm/addХ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2'
%layer_normalization_3/batchnorm/RsqrtЯ
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpЬ
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_3/batchnorm/mul├
%layer_normalization_3/batchnorm/mul_1Muladd_2/add:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_3/batchnorm/mul_1р
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_3/batchnorm/mul_2н
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpЖ
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_3/batchnorm/subр
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_3/batchnorm/add_1»
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axesЂ
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/freeІ
dense_3/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_3/Tensordot/Shapeё
dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/GatherV2/axisщ
dense_3/Tensordot/GatherV2GatherV2 dense_3/Tensordot/Shape:output:0dense_3/Tensordot/free:output:0(dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_3/Tensordot/GatherV2ѕ
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis 
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
dense_3/Tensordot/Constа
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prodђ
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1е
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1ђ
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axisп
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concatг
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stack╦
dense_3/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0!dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:         d@2
dense_3/Tensordot/transpose┐
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_3/Tensordot/Reshape┐
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
dense_3/Tensordot/MatMulЂ
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ2
dense_3/Tensordot/Const_2ё
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axisт
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1▒
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2
dense_3/TensordotЦ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02 
dense_3/BiasAdd/ReadVariableOpе
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2
dense_3/BiasAddm
dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_3/Gelu/mul/xЎ
dense_3/Gelu/mulMuldense_3/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/mulo
dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
dense_3/Gelu/Cast/xд
dense_3/Gelu/truedivRealDivdense_3/BiasAdd:output:0dense_3/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/truediv|
dense_3/Gelu/ErfErfdense_3/Gelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/Erfm
dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dense_3/Gelu/add/xЌ
dense_3/Gelu/addAddV2dense_3/Gelu/add/x:output:0dense_3/Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/addњ
dense_3/Gelu/mul_1Muldense_3/Gelu/mul:z:0dense_3/Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2
dense_3/Gelu/mul_1»
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axesЂ
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
dense_4/Tensordot/Shapeё
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axisщ
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2ѕ
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis 
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
dense_4/Tensordot/Constа
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prodђ
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1е
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1ђ
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axisп
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concatг
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack╣
dense_4/Tensordot/transpose	Transposedense_3/Gelu/mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:         dђ2
dense_4/Tensordot/transpose┐
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_4/Tensordot/ReshapeЙ
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense_4/Tensordot/MatMulђ
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_4/Tensordot/Const_2ё
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axisт
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1░
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2
dense_4/Tensordotц
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOpД
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
dense_4/BiasAddm
dense_4/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_4/Gelu/mul/xў
dense_4/Gelu/mulMuldense_4/Gelu/mul/x:output:0dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/mulo
dense_4/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
dense_4/Gelu/Cast/xЦ
dense_4/Gelu/truedivRealDivdense_4/BiasAdd:output:0dense_4/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/truediv{
dense_4/Gelu/ErfErfdense_4/Gelu/truediv:z:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/Erfm
dense_4/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2
dense_4/Gelu/add/xќ
dense_4/Gelu/addAddV2dense_4/Gelu/add/x:output:0dense_4/Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/addЉ
dense_4/Gelu/mul_1Muldense_4/Gelu/mul:z:0dense_4/Gelu/add:z:0*
T0*+
_output_shapes
:         d@2
dense_4/Gelu/mul_1|
	add_3/addAddV2dense_4/Gelu/mul_1:z:0add_2/add:z:0*
T0*+
_output_shapes
:         d@2
	add_3/addХ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesт
"layer_normalization_4/moments/meanMeanadd_3/add:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2$
"layer_normalization_4/moments/mean╦
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*+
_output_shapes
:         d2,
*layer_normalization_4/moments/StopGradientы
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd_3/add:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*+
_output_shapes
:         d@21
/layer_normalization_4/moments/SquaredDifferenceЙ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indicesЌ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2(
&layer_normalization_4/moments/varianceЊ
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52'
%layer_normalization_4/batchnorm/add/yЖ
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2%
#layer_normalization_4/batchnorm/addХ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*+
_output_shapes
:         d2'
%layer_normalization_4/batchnorm/RsqrtЯ
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpЬ
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_4/batchnorm/mul├
%layer_normalization_4/batchnorm/mul_1Muladd_3/add:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_4/batchnorm/mul_1р
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_4/batchnorm/mul_2н
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpЖ
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2%
#layer_normalization_4/batchnorm/subр
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2'
%layer_normalization_4/batchnorm/add_1w
reshape/ShapeShape)layer_normalization_4/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
reshape/Shapeё
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stackѕ
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1ѕ
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2њ
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
value	B :
2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
reshape/Reshape/shape/3Ж
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape▓
reshape/ReshapeReshape)layer_normalization_4/batchnorm/add_1:z:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:         

@2
reshape/Reshapeф
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype02
conv_1/Conv2D/ReadVariableOp╦
conv_1/Conv2DConv2Dreshape/Reshape:output:0$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
conv_1/Conv2DА
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_1/BiasAdd/ReadVariableOpц
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
conv_1/BiasAddф
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
conv_2/Conv2D/ReadVariableOp╩
conv_2/Conv2DConv2Dconv_1/BiasAdd:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
conv_2/Conv2DА
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_2/BiasAdd/ReadVariableOpц
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
conv_2/BiasAddф
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_3/Conv2D/ReadVariableOp╩
conv_3/Conv2DConv2Dconv_2/BiasAdd:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv_3/Conv2DА
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOpц
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv_3/BiasAddф
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOp╩
conv_4/Conv2DConv2Dconv_3/BiasAdd:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv_4/Conv2DА
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_4/BiasAdd/ReadVariableOpц
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv_4/BiasAdd{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
flatten_layer/Constб
flatten_layer/ReshapeReshapeconv_4/BiasAdd:output:0flatten_layer/Const:output:0*
T0*'
_output_shapes
:         @2
flatten_layer/Reshapeю
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes

:@2*
dtype02
FC_1/MatMul/ReadVariableOpџ
FC_1/MatMulMatMulflatten_layer/Reshape:output:0"FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
FC_1/MatMulЏ
FC_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_1/BiasAdd/ReadVariableOpЋ
FC_1/BiasAddBiasAddFC_1/MatMul:product:0#FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
FC_1/BiasAddЇ
leaky_ReLu_1/LeakyRelu	LeakyReluFC_1/BiasAdd:output:0*'
_output_shapes
:         2*
alpha%џЎЎ>2
leaky_ReLu_1/LeakyReluю
FC_2/MatMul/ReadVariableOpReadVariableOp#fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
FC_2/MatMul/ReadVariableOpа
FC_2/MatMulMatMul$leaky_ReLu_1/LeakyRelu:activations:0"FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
FC_2/MatMulЏ
FC_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_2/BiasAdd/ReadVariableOpЋ
FC_2/BiasAddBiasAddFC_2/MatMul:product:0#FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
FC_2/BiasAddЇ
leaky_ReLu_2/LeakyRelu	LeakyReluFC_2/BiasAdd:output:0*'
_output_shapes
:         2*
alpha%џЎЎ>2
leaky_ReLu_2/LeakyRelu┤
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02$
"output_layer/MatMul/ReadVariableOpИ
output_layer/MatMulMatMul$leaky_ReLu_2/LeakyRelu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output_layer/MatMul│
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpх
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
output_layer/BiasAddѕ
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:         2
output_layer/Softmax─
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
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
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2ѕ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2ї
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
(patch_encoder/embedding/embedding_lookup(patch_encoder/embedding/embedding_lookup:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Њ7
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_622920	
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
identityѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpК
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЎ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┴
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpЉ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpК
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЎ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
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
:         d@2
Mulа
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/dropout/Constд
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeн
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         dd*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2 
dropout/dropout/GreaterEqual/yТ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/GreaterEqualЪ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         dd2
dropout/dropout/Castб
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/Mul_1И
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpэ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┴
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
attention_output/addѓ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 2J
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
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
т└
ВM
__inference__traced_save_626228
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
5savev2_layer_normalization_2_beta_read_readvariableop:
6savev2_layer_normalization_3_gamma_read_readvariableop9
5savev2_layer_normalization_3_beta_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop:
6savev2_layer_normalization_4_gamma_read_readvariableop9
5savev2_layer_normalization_4_beta_read_readvariableop,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop,
(savev2_conv_4_kernel_read_readvariableop*
&savev2_conv_4_bias_read_readvariableop*
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
=savev2_adamw_layer_normalization_2_beta_m_read_readvariableopB
>savev2_adamw_layer_normalization_3_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_3_beta_m_read_readvariableop5
1savev2_adamw_dense_3_kernel_m_read_readvariableop3
/savev2_adamw_dense_3_bias_m_read_readvariableop5
1savev2_adamw_dense_4_kernel_m_read_readvariableop3
/savev2_adamw_dense_4_bias_m_read_readvariableopB
>savev2_adamw_layer_normalization_4_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_4_beta_m_read_readvariableop4
0savev2_adamw_conv_1_kernel_m_read_readvariableop2
.savev2_adamw_conv_1_bias_m_read_readvariableop4
0savev2_adamw_conv_2_kernel_m_read_readvariableop2
.savev2_adamw_conv_2_bias_m_read_readvariableop4
0savev2_adamw_conv_3_kernel_m_read_readvariableop2
.savev2_adamw_conv_3_bias_m_read_readvariableop4
0savev2_adamw_conv_4_kernel_m_read_readvariableop2
.savev2_adamw_conv_4_bias_m_read_readvariableop2
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
Osavev2_adamw_multi_head_attention_1_attention_output_bias_m_read_readvariableop@
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
=savev2_adamw_layer_normalization_4_beta_v_read_readvariableop4
0savev2_adamw_conv_1_kernel_v_read_readvariableop2
.savev2_adamw_conv_1_bias_v_read_readvariableop4
0savev2_adamw_conv_2_kernel_v_read_readvariableop2
.savev2_adamw_conv_2_bias_v_read_readvariableop4
0savev2_adamw_conv_3_kernel_v_read_readvariableop2
.savev2_adamw_conv_3_bias_v_read_readvariableop4
0savev2_adamw_conv_4_kernel_v_read_readvariableop2
.savev2_adamw_conv_4_bias_v_read_readvariableop2
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

identity_1ѕбMergeV2CheckpointsЈ
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
Const_1І
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
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename┼[
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:ц*
dtype0*оZ
value╠ZB╔ZцB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesН
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:ц*
dtype0*я
valueнBЛцB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesНJ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop6savev2_layer_normalization_3_gamma_read_readvariableop5savev2_layer_normalization_3_beta_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_2_kernel_read_readvariableop$savev2_fc_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop%savev2_adamw_iter_read_readvariableop'savev2_adamw_beta_1_read_readvariableop'savev2_adamw_beta_2_read_readvariableop&savev2_adamw_decay_read_readvariableop.savev2_adamw_learning_rate_read_readvariableop-savev2_adamw_weight_decay_read_readvariableop5savev2_patch_encoder_dense_kernel_read_readvariableop3savev2_patch_encoder_dense_bias_read_readvariableop=savev2_patch_encoder_embedding_embeddings_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop>savev2_multi_head_attention_1_query_kernel_read_readvariableop<savev2_multi_head_attention_1_query_bias_read_readvariableop<savev2_multi_head_attention_1_key_kernel_read_readvariableop:savev2_multi_head_attention_1_key_bias_read_readvariableop>savev2_multi_head_attention_1_value_kernel_read_readvariableop<savev2_multi_head_attention_1_value_bias_read_readvariableopIsavev2_multi_head_attention_1_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_1_attention_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop<savev2_adamw_layer_normalization_gamma_m_read_readvariableop;savev2_adamw_layer_normalization_beta_m_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_1_beta_m_read_readvariableop1savev2_adamw_dense_1_kernel_m_read_readvariableop/savev2_adamw_dense_1_bias_m_read_readvariableop1savev2_adamw_dense_2_kernel_m_read_readvariableop/savev2_adamw_dense_2_bias_m_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_2_beta_m_read_readvariableop>savev2_adamw_layer_normalization_3_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_3_beta_m_read_readvariableop1savev2_adamw_dense_3_kernel_m_read_readvariableop/savev2_adamw_dense_3_bias_m_read_readvariableop1savev2_adamw_dense_4_kernel_m_read_readvariableop/savev2_adamw_dense_4_bias_m_read_readvariableop>savev2_adamw_layer_normalization_4_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_4_beta_m_read_readvariableop0savev2_adamw_conv_1_kernel_m_read_readvariableop.savev2_adamw_conv_1_bias_m_read_readvariableop0savev2_adamw_conv_2_kernel_m_read_readvariableop.savev2_adamw_conv_2_bias_m_read_readvariableop0savev2_adamw_conv_3_kernel_m_read_readvariableop.savev2_adamw_conv_3_bias_m_read_readvariableop0savev2_adamw_conv_4_kernel_m_read_readvariableop.savev2_adamw_conv_4_bias_m_read_readvariableop.savev2_adamw_fc_1_kernel_m_read_readvariableop,savev2_adamw_fc_1_bias_m_read_readvariableop.savev2_adamw_fc_2_kernel_m_read_readvariableop,savev2_adamw_fc_2_bias_m_read_readvariableop6savev2_adamw_output_layer_kernel_m_read_readvariableop4savev2_adamw_output_layer_bias_m_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_m_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_m_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_m_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_m_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_m_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_m_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_m_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_m_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_m_read_readvariableopFsavev2_adamw_multi_head_attention_1_query_kernel_m_read_readvariableopDsavev2_adamw_multi_head_attention_1_query_bias_m_read_readvariableopDsavev2_adamw_multi_head_attention_1_key_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_1_key_bias_m_read_readvariableopFsavev2_adamw_multi_head_attention_1_value_kernel_m_read_readvariableopDsavev2_adamw_multi_head_attention_1_value_bias_m_read_readvariableopQsavev2_adamw_multi_head_attention_1_attention_output_kernel_m_read_readvariableopOsavev2_adamw_multi_head_attention_1_attention_output_bias_m_read_readvariableop<savev2_adamw_layer_normalization_gamma_v_read_readvariableop;savev2_adamw_layer_normalization_beta_v_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_1_beta_v_read_readvariableop1savev2_adamw_dense_1_kernel_v_read_readvariableop/savev2_adamw_dense_1_bias_v_read_readvariableop1savev2_adamw_dense_2_kernel_v_read_readvariableop/savev2_adamw_dense_2_bias_v_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_2_beta_v_read_readvariableop>savev2_adamw_layer_normalization_3_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_3_beta_v_read_readvariableop1savev2_adamw_dense_3_kernel_v_read_readvariableop/savev2_adamw_dense_3_bias_v_read_readvariableop1savev2_adamw_dense_4_kernel_v_read_readvariableop/savev2_adamw_dense_4_bias_v_read_readvariableop>savev2_adamw_layer_normalization_4_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_4_beta_v_read_readvariableop0savev2_adamw_conv_1_kernel_v_read_readvariableop.savev2_adamw_conv_1_bias_v_read_readvariableop0savev2_adamw_conv_2_kernel_v_read_readvariableop.savev2_adamw_conv_2_bias_v_read_readvariableop0savev2_adamw_conv_3_kernel_v_read_readvariableop.savev2_adamw_conv_3_bias_v_read_readvariableop0savev2_adamw_conv_4_kernel_v_read_readvariableop.savev2_adamw_conv_4_bias_v_read_readvariableop.savev2_adamw_fc_1_kernel_v_read_readvariableop,savev2_adamw_fc_1_bias_v_read_readvariableop.savev2_adamw_fc_2_kernel_v_read_readvariableop,savev2_adamw_fc_2_bias_v_read_readvariableop6savev2_adamw_output_layer_kernel_v_read_readvariableop4savev2_adamw_output_layer_bias_v_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_v_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_v_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_v_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_v_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_v_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_v_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_v_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_v_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_v_read_readvariableopFsavev2_adamw_multi_head_attention_1_query_kernel_v_read_readvariableopDsavev2_adamw_multi_head_attention_1_query_bias_v_read_readvariableopDsavev2_adamw_multi_head_attention_1_key_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_1_key_bias_v_read_readvariableopFsavev2_adamw_multi_head_attention_1_value_kernel_v_read_readvariableopDsavev2_adamw_multi_head_attention_1_value_bias_v_read_readvariableopQsavev2_adamw_multi_head_attention_1_attention_output_kernel_v_read_readvariableopOsavev2_adamw_multi_head_attention_1_attention_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *х
dtypesф
Д2ц	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
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

identity_1Identity_1:output:0*▄
_input_shapes╩
К: :@:@:@:@:	@ђ:ђ:	ђ@:@:@:@:@:@:	@ђ:ђ:	ђ@:@:@:@:@
:
:

:
:
::::@2:2:22:2:2:: : : : : : :	г@:@:d@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@: : : : :@:@:@:@:	@ђ:ђ:	ђ@:@:@:@:@:@:	@ђ:ђ:	ђ@:@:@:@:@
:
:

:
:
::::@2:2:22:2:2::	г@:@:d@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@:@:@:@:	@ђ:ђ:	ђ@:@:@:@:@:@:	@ђ:ђ:	ђ@:@:@:@:@
:
:

:
:
::::@2:2:22:2:2::	г@:@:d@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@: 2(
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
:	@ђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ@: 

_output_shapes
:@: 	
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
:	@ђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@
: 

_output_shapes
:
:,(
&
_output_shapes
:

: 

_output_shapes
:
:,(
&
_output_shapes
:
: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:@2: 
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
:	г@: (

_output_shapes
:@:$) 

_output_shapes

:d@:(*$
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
: : >

_output_shapes
:@: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@:%B!

_output_shapes
:	@ђ:!C

_output_shapes	
:ђ:%D!

_output_shapes
:	ђ@: E

_output_shapes
:@: F
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
:	@ђ:!K

_output_shapes	
:ђ:%L!

_output_shapes
:	ђ@: M

_output_shapes
:@: N

_output_shapes
:@: O

_output_shapes
:@:,P(
&
_output_shapes
:@
: Q

_output_shapes
:
:,R(
&
_output_shapes
:

: S

_output_shapes
:
:,T(
&
_output_shapes
:
: U

_output_shapes
::,V(
&
_output_shapes
:: W

_output_shapes
::$X 

_output_shapes

:@2: Y
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
:	г@: _

_output_shapes
:@:$` 

_output_shapes

:d@:(a$
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
:@: q

_output_shapes
:@: r

_output_shapes
:@: s

_output_shapes
:@: t

_output_shapes
:@:%u!

_output_shapes
:	@ђ:!v

_output_shapes	
:ђ:%w!

_output_shapes
:	ђ@: x

_output_shapes
:@: y
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
:	@ђ:!~

_output_shapes	
:ђ:%!

_output_shapes
:	ђ@:!ђ

_output_shapes
:@:!Ђ

_output_shapes
:@:!ѓ

_output_shapes
:@:-Ѓ(
&
_output_shapes
:@
:!ё

_output_shapes
:
:-Ё(
&
_output_shapes
:

:!є

_output_shapes
:
:-Є(
&
_output_shapes
:
:!ѕ

_output_shapes
::-Ѕ(
&
_output_shapes
::!і

_output_shapes
::%І 

_output_shapes

:@2:!ї

_output_shapes
:2:%Ї 

_output_shapes

:22:!ј

_output_shapes
:2:%Ј 

_output_shapes

:2:!љ

_output_shapes
::&Љ!

_output_shapes
:	г@:!њ

_output_shapes
:@:%Њ 

_output_shapes

:d@:)ћ$
"
_output_shapes
:@@:%Ћ 

_output_shapes

:@:)ќ$
"
_output_shapes
:@@:%Ќ 

_output_shapes

:@:)ў$
"
_output_shapes
:@@:%Ў 

_output_shapes

:@:)џ$
"
_output_shapes
:@@:!Џ

_output_shapes
:@:)ю$
"
_output_shapes
:@@:%Ю 

_output_shapes

:@:)ъ$
"
_output_shapes
:@@:%Ъ 

_output_shapes

:@:)а$
"
_output_shapes
:@@:%А 

_output_shapes

:@:)б$
"
_output_shapes
:@@:!Б

_output_shapes
:@:ц

_output_shapes
: 
э-
ч
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_625266	
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
identityѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpК
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЎ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┴
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpЉ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpК
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЎ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
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
:         d@2
Mulа
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2
softmax/SoftmaxЁ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:         dd2
dropout/IdentityИ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpэ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┴
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
attention_output/addѓ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 2J
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
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
А
ј
O__inference_layer_normalization_layer_call_and_return_conditional_losses_624921

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
Њ7
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_625007	
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
identityѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpК
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЎ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┴
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpЉ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpК
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЎ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
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
:         d@2
Mulа
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/dropout/Constд
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeн
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         dd*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2 
dropout/dropout/GreaterEqual/yТ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/GreaterEqualЪ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         dd2
dropout/dropout/Castб
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/Mul_1И
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpэ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┴
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
attention_output/addѓ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 2J
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
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
Ћ7
ч
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_622779	
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
identityѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpК
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЎ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┴
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpЉ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpК
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЎ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
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
:         d@2
Mulа
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?2
dropout/dropout/Constд
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeн
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:         dd*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЁ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=2 
dropout/dropout/GreaterEqual/yТ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/GreaterEqualЪ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         dd2
dropout/dropout/Castб
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         dd2
dropout/dropout/Mul_1И
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpэ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┴
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
attention_output/addѓ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 2J
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
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
№
├
6__inference_WheatClassifier_CNN_1_layer_call_fn_623914

inputs
unknown:	г@
	unknown_0:@
	unknown_1:d@
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

unknown_14:	@ђ

unknown_15:	ђ

unknown_16:	ђ@

unknown_17:@

unknown_18:@

unknown_19:@ 

unknown_20:@@

unknown_21:@ 

unknown_22:@@

unknown_23:@ 

unknown_24:@@

unknown_25:@ 

unknown_26:@@

unknown_27:@

unknown_28:@

unknown_29:@

unknown_30:	@ђ

unknown_31:	ђ

unknown_32:	ђ@

unknown_33:@

unknown_34:@

unknown_35:@$

unknown_36:@


unknown_37:
$

unknown_38:



unknown_39:
$

unknown_40:


unknown_41:$

unknown_42:

unknown_43:

unknown_44:@2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identityѕбStatefulPartitionedCallд
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
:         *U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_6224582
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
ќ
_
C__inference_reshape_layer_call_and_return_conditional_losses_625546

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
strided_slice/stack_2Р
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
value	B :
2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         

@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         

@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d@:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
Б
љ
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_621963

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
І'
Ч
C__inference_dense_1_layer_call_and_return_conditional_losses_622007

inputs4
!tensordot_readvariableop_resource:	@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpЌ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackљ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         d@2
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/ReshapeЪ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2
	TensordotЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2	
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
:         dђ2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
Gelu/Cast/xє
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2

Gelu/mul_1Џ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:         dђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
Ш/
Ж
I__inference_patch_encoder_layer_call_and_return_conditional_losses_621844	
patch:
'dense_tensordot_readvariableop_resource:	г@3
%dense_biasadd_readvariableop_resource:@3
!embedding_embedding_lookup_621837:d@
identityѕбdense/BiasAdd/ReadVariableOpбdense/Tensordot/ReadVariableOpбembedding/embedding_lookup\
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
value	B :d2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltau
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes
:d2
rangeЕ
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	г@*
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
dense/Tensordot/Shapeђ
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis№
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2ё
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisш
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
dense/Tensordot/Constў
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
dense/Tensordot/Const_1а
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
dense/Tensordot/concat/axis╬
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concatц
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stackФ
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:                  г2
dense/Tensordot/transposeи
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense/Tensordot/ReshapeХ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense/Tensordot/Const_2ђ
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis█
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1▒
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @2
dense/Tensordotъ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOpе
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @2
dense/BiasAddъ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_621837range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/621837*
_output_shapes

:d@*
dtype02
embedding/embedding_lookupѕ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/621837*
_output_shapes

:d@2%
#embedding/embedding_lookup/Identity▒
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:d@2'
%embedding/embedding_lookup/Identity_1Љ
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:         d@2
add╝
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  г: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:                  г

_user_specified_namepatch
М
R
&__inference_add_1_layer_call_fn_625200
inputs_0
inputs_1
identityл
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_6220632
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:U Q
+
_output_shapes
:         d@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         d@
"
_user_specified_name
inputs/1
Ьљ
З
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_623210

inputs'
patch_encoder_623080:	г@"
patch_encoder_623082:@&
patch_encoder_623084:d@(
layer_normalization_623087:@(
layer_normalization_623089:@1
multi_head_attention_623092:@@-
multi_head_attention_623094:@1
multi_head_attention_623096:@@-
multi_head_attention_623098:@1
multi_head_attention_623100:@@-
multi_head_attention_623102:@1
multi_head_attention_623104:@@)
multi_head_attention_623106:@*
layer_normalization_1_623110:@*
layer_normalization_1_623112:@!
dense_1_623115:	@ђ
dense_1_623117:	ђ!
dense_2_623120:	ђ@
dense_2_623122:@*
layer_normalization_2_623126:@*
layer_normalization_2_623128:@3
multi_head_attention_1_623131:@@/
multi_head_attention_1_623133:@3
multi_head_attention_1_623135:@@/
multi_head_attention_1_623137:@3
multi_head_attention_1_623139:@@/
multi_head_attention_1_623141:@3
multi_head_attention_1_623143:@@+
multi_head_attention_1_623145:@*
layer_normalization_3_623149:@*
layer_normalization_3_623151:@!
dense_3_623154:	@ђ
dense_3_623156:	ђ!
dense_4_623159:	ђ@
dense_4_623161:@*
layer_normalization_4_623165:@*
layer_normalization_4_623167:@'
conv_1_623171:@

conv_1_623173:
'
conv_2_623176:


conv_2_623178:
'
conv_3_623181:

conv_3_623183:'
conv_4_623186:
conv_4_623188:
fc_1_623192:@2
fc_1_623194:2
fc_2_623198:22
fc_2_623200:2%
output_layer_623204:2!
output_layer_623206:
identityѕбFC_1/StatefulPartitionedCallбFC_2/StatefulPartitionedCallбconv_1/StatefulPartitionedCallбconv_2/StatefulPartitionedCallбconv_3/StatefulPartitionedCallбconv_4/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallб+layer_normalization/StatefulPartitionedCallб-layer_normalization_1/StatefulPartitionedCallб-layer_normalization_2/StatefulPartitionedCallб-layer_normalization_3/StatefulPartitionedCallб-layer_normalization_4/StatefulPartitionedCallб,multi_head_attention/StatefulPartitionedCallб.multi_head_attention_1/StatefulPartitionedCallб$output_layer/StatefulPartitionedCallб%patch_encoder/StatefulPartitionedCall▀
patches/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  г* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_6218022
patches/PartitionedCallс
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_623080patch_encoder_623082patch_encoder_623084*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_6218442'
%patch_encoder/StatefulPartitionedCallэ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_623087layer_normalization_623089*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_6218742-
+layer_normalization/StatefulPartitionedCallз
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_623092multi_head_attention_623094multi_head_attention_623096multi_head_attention_623098multi_head_attention_623100multi_head_attention_623102multi_head_attention_623104multi_head_attention_623106*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_6229202.
,multi_head_attention/StatefulPartitionedCallЕ
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_6219392
add/PartitionedCall№
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_623110layer_normalization_1_623112*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_6219632/
-layer_normalization_1/StatefulPartitionedCall─
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_623115dense_1_623117*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6220072!
dense_1/StatefulPartitionedCallх
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_623120dense_2_623122*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6220512!
dense_2/StatefulPartitionedCallљ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_6220632
add_1/PartitionedCallы
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_623126layer_normalization_2_623128*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_6220872/
-layer_normalization_2/StatefulPartitionedCallЇ
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_623131multi_head_attention_1_623133multi_head_attention_1_623135multi_head_attention_1_623137multi_head_attention_1_623139multi_head_attention_1_623141multi_head_attention_1_623143multi_head_attention_1_623145*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_62277920
.multi_head_attention_1/StatefulPartitionedCallА
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_6221522
add_2/PartitionedCallы
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_623149layer_normalization_3_623151*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_6221762/
-layer_normalization_3/StatefulPartitionedCall─
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_623154dense_3_623156*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6222202!
dense_3/StatefulPartitionedCallх
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_623159dense_4_623161*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6222642!
dense_4/StatefulPartitionedCallњ
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_6222762
add_3/PartitionedCallы
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_623165layer_normalization_4_623167*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_6223002/
-layer_normalization_4/StatefulPartitionedCallЅ
reshape/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6223202
reshape/PartitionedCallг
conv_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv_1_623171conv_1_623173*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_6223322 
conv_1/StatefulPartitionedCall│
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_623176conv_2_623178*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_6223482 
conv_2/StatefulPartitionedCall│
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_623181conv_3_623183*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_6223642 
conv_3/StatefulPartitionedCall│
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_623186conv_4_623188*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_6223802 
conv_4/StatefulPartitionedCallё
flatten_layer/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_6223922
flatten_layer/PartitionedCallа
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_623192fc_1_623194*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_6224042
FC_1/StatefulPartitionedCall 
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_6224152
leaky_ReLu_1/PartitionedCallЪ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_623198fc_2_623200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_6224272
FC_2/StatefulPartitionedCall 
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_6224382
leaky_ReLu_2/PartitionedCallК
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_623204output_layer_623206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_6224512&
$output_layer/StatefulPartitionedCallУ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
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
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
ч
d
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_625662

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         2*
alpha%џЎЎ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
┐
ю
'__inference_conv_3_layer_call_fn_625608

inputs!
unknown:

	unknown_0:
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_6223642
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
г
ќ
(__inference_dense_4_layer_call_fn_625489

inputs
unknown:	ђ@
	unknown_0:@
identityѕбStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6222642
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         dђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         dђ
 
_user_specified_nameinputs
ж
e
I__inference_flatten_layer_layer_call_and_return_conditional_losses_622392

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         @2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
»

ч
B__inference_conv_4_layer_call_and_return_conditional_losses_625618

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ж
m
A__inference_add_1_layer_call_and_return_conditional_losses_625194
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:         d@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:U Q
+
_output_shapes
:         d@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         d@
"
_user_specified_name
inputs/1
»

ч
B__inference_conv_3_layer_call_and_return_conditional_losses_622364

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
Г
Ќ
(__inference_dense_1_layer_call_fn_625141

inputs
unknown:	@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6220072
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         dђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
И

щ
H__inference_output_layer_layer_call_and_return_conditional_losses_622451

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
Т
D
(__inference_patches_layer_call_fn_624848

images
identity¤
PartitionedCallPartitionedCallimages*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  г* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_6218022
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:                  г2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd:W S
/
_output_shapes
:         dd
 
_user_specified_nameimages
┴
Ъ
6__inference_layer_normalization_3_layer_call_fn_625395

inputs
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_6221762
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
ж
m
A__inference_add_2_layer_call_and_return_conditional_losses_625358
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:         d@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:U Q
+
_output_shapes
:         d@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         d@
"
_user_specified_name
inputs/1
╠	
ы
@__inference_FC_2_layer_call_and_return_conditional_losses_625677

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
И

щ
H__inference_output_layer_layer_call_and_return_conditional_losses_625707

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Softmaxќ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
Б
љ
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_625386

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
Њ
њ
%__inference_FC_2_layer_call_fn_625686

inputs
unknown:22
	unknown_0:2
identityѕбStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_6224272
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
р
k
A__inference_add_1_layer_call_and_return_conditional_losses_622063

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:         d@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs:SO
+
_output_shapes
:         d@
 
_user_specified_nameinputs
ќ
_
C__inference_reshape_layer_call_and_return_conditional_losses_622320

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
strided_slice/stack_2Р
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
value	B :
2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/3║
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:         

@2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:         

@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         d@:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
┐
ю
'__inference_conv_1_layer_call_fn_625570

inputs!
unknown:@

	unknown_0:

identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_6223322
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         

@
 
_user_specified_nameinputs
»

ч
B__inference_conv_2_layer_call_and_return_conditional_losses_622348

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
Б
_
C__inference_patches_layer_call_and_return_conditional_losses_624843

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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceн
ExtractImagePatchesExtractImagePatchesimages*
T0*0
_output_shapes
:         

г*
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
         2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :г2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeћ
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  г2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:                  г2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd:W S
/
_output_shapes
:         dd
 
_user_specified_nameimages
»

ч
B__inference_conv_3_layer_call_and_return_conditional_losses_625599

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
Б
_
C__inference_patches_layer_call_and_return_conditional_losses_621802

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
strided_slice/stack_2Р
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceн
ExtractImagePatchesExtractImagePatchesimages*
T0*0
_output_shapes
:         

г*
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
         2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :г2
Reshape/shape/2а
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeћ
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:                  г2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:                  г2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         dd:W S
/
_output_shapes
:         dd
 
_user_specified_nameimages
»

ч
B__inference_conv_4_layer_call_and_return_conditional_losses_622380

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
ч
d
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_622438

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:         2*
alpha%џЎЎ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
┴
Ъ
6__inference_layer_normalization_1_layer_call_fn_625094

inputs
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_6219632
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
┴
Ъ
6__inference_layer_normalization_4_layer_call_fn_625532

inputs
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_6223002
StatefulPartitionedCallњ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
─
I
-__inference_leaky_ReLu_1_layer_call_fn_625667

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_6224152
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         2:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
┬¤
їr
"__inference__traced_restore_626727
file_prefix8
*assignvariableop_layer_normalization_gamma:@9
+assignvariableop_1_layer_normalization_beta:@<
.assignvariableop_2_layer_normalization_1_gamma:@;
-assignvariableop_3_layer_normalization_1_beta:@4
!assignvariableop_4_dense_1_kernel:	@ђ.
assignvariableop_5_dense_1_bias:	ђ4
!assignvariableop_6_dense_2_kernel:	ђ@-
assignvariableop_7_dense_2_bias:@<
.assignvariableop_8_layer_normalization_2_gamma:@;
-assignvariableop_9_layer_normalization_2_beta:@=
/assignvariableop_10_layer_normalization_3_gamma:@<
.assignvariableop_11_layer_normalization_3_beta:@5
"assignvariableop_12_dense_3_kernel:	@ђ/
 assignvariableop_13_dense_3_bias:	ђ5
"assignvariableop_14_dense_4_kernel:	ђ@.
 assignvariableop_15_dense_4_bias:@=
/assignvariableop_16_layer_normalization_4_gamma:@<
.assignvariableop_17_layer_normalization_4_beta:@;
!assignvariableop_18_conv_1_kernel:@
-
assignvariableop_19_conv_1_bias:
;
!assignvariableop_20_conv_2_kernel:

-
assignvariableop_21_conv_2_bias:
;
!assignvariableop_22_conv_3_kernel:
-
assignvariableop_23_conv_3_bias:;
!assignvariableop_24_conv_4_kernel:-
assignvariableop_25_conv_4_bias:1
assignvariableop_26_fc_1_kernel:@2+
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
.assignvariableop_38_patch_encoder_dense_kernel:	г@:
,assignvariableop_39_patch_encoder_dense_bias:@H
6assignvariableop_40_patch_encoder_embedding_embeddings:d@K
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
5assignvariableop_61_adamw_layer_normalization_gamma_m:@B
4assignvariableop_62_adamw_layer_normalization_beta_m:@E
7assignvariableop_63_adamw_layer_normalization_1_gamma_m:@D
6assignvariableop_64_adamw_layer_normalization_1_beta_m:@=
*assignvariableop_65_adamw_dense_1_kernel_m:	@ђ7
(assignvariableop_66_adamw_dense_1_bias_m:	ђ=
*assignvariableop_67_adamw_dense_2_kernel_m:	ђ@6
(assignvariableop_68_adamw_dense_2_bias_m:@E
7assignvariableop_69_adamw_layer_normalization_2_gamma_m:@D
6assignvariableop_70_adamw_layer_normalization_2_beta_m:@E
7assignvariableop_71_adamw_layer_normalization_3_gamma_m:@D
6assignvariableop_72_adamw_layer_normalization_3_beta_m:@=
*assignvariableop_73_adamw_dense_3_kernel_m:	@ђ7
(assignvariableop_74_adamw_dense_3_bias_m:	ђ=
*assignvariableop_75_adamw_dense_4_kernel_m:	ђ@6
(assignvariableop_76_adamw_dense_4_bias_m:@E
7assignvariableop_77_adamw_layer_normalization_4_gamma_m:@D
6assignvariableop_78_adamw_layer_normalization_4_beta_m:@C
)assignvariableop_79_adamw_conv_1_kernel_m:@
5
'assignvariableop_80_adamw_conv_1_bias_m:
C
)assignvariableop_81_adamw_conv_2_kernel_m:

5
'assignvariableop_82_adamw_conv_2_bias_m:
C
)assignvariableop_83_adamw_conv_3_kernel_m:
5
'assignvariableop_84_adamw_conv_3_bias_m:C
)assignvariableop_85_adamw_conv_4_kernel_m:5
'assignvariableop_86_adamw_conv_4_bias_m:9
'assignvariableop_87_adamw_fc_1_kernel_m:@23
%assignvariableop_88_adamw_fc_1_bias_m:29
'assignvariableop_89_adamw_fc_2_kernel_m:223
%assignvariableop_90_adamw_fc_2_bias_m:2A
/assignvariableop_91_adamw_output_layer_kernel_m:2;
-assignvariableop_92_adamw_output_layer_bias_m:I
6assignvariableop_93_adamw_patch_encoder_dense_kernel_m:	г@B
4assignvariableop_94_adamw_patch_encoder_dense_bias_m:@P
>assignvariableop_95_adamw_patch_encoder_embedding_embeddings_m:d@S
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
6assignvariableop_112_adamw_layer_normalization_gamma_v:@C
5assignvariableop_113_adamw_layer_normalization_beta_v:@F
8assignvariableop_114_adamw_layer_normalization_1_gamma_v:@E
7assignvariableop_115_adamw_layer_normalization_1_beta_v:@>
+assignvariableop_116_adamw_dense_1_kernel_v:	@ђ8
)assignvariableop_117_adamw_dense_1_bias_v:	ђ>
+assignvariableop_118_adamw_dense_2_kernel_v:	ђ@7
)assignvariableop_119_adamw_dense_2_bias_v:@F
8assignvariableop_120_adamw_layer_normalization_2_gamma_v:@E
7assignvariableop_121_adamw_layer_normalization_2_beta_v:@F
8assignvariableop_122_adamw_layer_normalization_3_gamma_v:@E
7assignvariableop_123_adamw_layer_normalization_3_beta_v:@>
+assignvariableop_124_adamw_dense_3_kernel_v:	@ђ8
)assignvariableop_125_adamw_dense_3_bias_v:	ђ>
+assignvariableop_126_adamw_dense_4_kernel_v:	ђ@7
)assignvariableop_127_adamw_dense_4_bias_v:@F
8assignvariableop_128_adamw_layer_normalization_4_gamma_v:@E
7assignvariableop_129_adamw_layer_normalization_4_beta_v:@D
*assignvariableop_130_adamw_conv_1_kernel_v:@
6
(assignvariableop_131_adamw_conv_1_bias_v:
D
*assignvariableop_132_adamw_conv_2_kernel_v:

6
(assignvariableop_133_adamw_conv_2_bias_v:
D
*assignvariableop_134_adamw_conv_3_kernel_v:
6
(assignvariableop_135_adamw_conv_3_bias_v:D
*assignvariableop_136_adamw_conv_4_kernel_v:6
(assignvariableop_137_adamw_conv_4_bias_v::
(assignvariableop_138_adamw_fc_1_kernel_v:@24
&assignvariableop_139_adamw_fc_1_bias_v:2:
(assignvariableop_140_adamw_fc_2_kernel_v:224
&assignvariableop_141_adamw_fc_2_bias_v:2B
0assignvariableop_142_adamw_output_layer_kernel_v:2<
.assignvariableop_143_adamw_output_layer_bias_v:J
7assignvariableop_144_adamw_patch_encoder_dense_kernel_v:	г@C
5assignvariableop_145_adamw_patch_encoder_dense_bias_v:@Q
?assignvariableop_146_adamw_patch_encoder_embedding_embeddings_v:d@T
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
identity_164ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_100бAssignVariableOp_101бAssignVariableOp_102бAssignVariableOp_103бAssignVariableOp_104бAssignVariableOp_105бAssignVariableOp_106бAssignVariableOp_107бAssignVariableOp_108бAssignVariableOp_109бAssignVariableOp_11бAssignVariableOp_110бAssignVariableOp_111бAssignVariableOp_112бAssignVariableOp_113бAssignVariableOp_114бAssignVariableOp_115бAssignVariableOp_116бAssignVariableOp_117бAssignVariableOp_118бAssignVariableOp_119бAssignVariableOp_12бAssignVariableOp_120бAssignVariableOp_121бAssignVariableOp_122бAssignVariableOp_123бAssignVariableOp_124бAssignVariableOp_125бAssignVariableOp_126бAssignVariableOp_127бAssignVariableOp_128бAssignVariableOp_129бAssignVariableOp_13бAssignVariableOp_130бAssignVariableOp_131бAssignVariableOp_132бAssignVariableOp_133бAssignVariableOp_134бAssignVariableOp_135бAssignVariableOp_136бAssignVariableOp_137бAssignVariableOp_138бAssignVariableOp_139бAssignVariableOp_14бAssignVariableOp_140бAssignVariableOp_141бAssignVariableOp_142бAssignVariableOp_143бAssignVariableOp_144бAssignVariableOp_145бAssignVariableOp_146бAssignVariableOp_147бAssignVariableOp_148бAssignVariableOp_149бAssignVariableOp_15бAssignVariableOp_150бAssignVariableOp_151бAssignVariableOp_152бAssignVariableOp_153бAssignVariableOp_154бAssignVariableOp_155бAssignVariableOp_156бAssignVariableOp_157бAssignVariableOp_158бAssignVariableOp_159бAssignVariableOp_16бAssignVariableOp_160бAssignVariableOp_161бAssignVariableOp_162бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_75бAssignVariableOp_76бAssignVariableOp_77бAssignVariableOp_78бAssignVariableOp_79бAssignVariableOp_8бAssignVariableOp_80бAssignVariableOp_81бAssignVariableOp_82бAssignVariableOp_83бAssignVariableOp_84бAssignVariableOp_85бAssignVariableOp_86бAssignVariableOp_87бAssignVariableOp_88бAssignVariableOp_89бAssignVariableOp_9бAssignVariableOp_90бAssignVariableOp_91бAssignVariableOp_92бAssignVariableOp_93бAssignVariableOp_94бAssignVariableOp_95бAssignVariableOp_96бAssignVariableOp_97бAssignVariableOp_98бAssignVariableOp_99╦[
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:ц*
dtype0*оZ
value╠ZB╔ZцB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names█
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:ц*
dtype0*я
valueнBЛцB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesТ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*д
_output_shapesЊ
љ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*х
dtypesф
Д2ц	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЕ
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1░
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2│
AssignVariableOp_2AssignVariableOp.assignvariableop_2_layer_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3▓
AssignVariableOp_3AssignVariableOp-assignvariableop_3_layer_normalization_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4д
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ц
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6д
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ц
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8│
AssignVariableOp_8AssignVariableOp.assignvariableop_8_layer_normalization_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9▓
AssignVariableOp_9AssignVariableOp-assignvariableop_9_layer_normalization_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10и
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_normalization_3_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Х
AssignVariableOp_11AssignVariableOp.assignvariableop_11_layer_normalization_3_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ф
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13е
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ф
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15е
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16и
AssignVariableOp_16AssignVariableOp/assignvariableop_16_layer_normalization_4_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Х
AssignVariableOp_17AssignVariableOp.assignvariableop_17_layer_normalization_4_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Е
AssignVariableOp_18AssignVariableOp!assignvariableop_18_conv_1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Д
AssignVariableOp_19AssignVariableOpassignvariableop_19_conv_1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Е
AssignVariableOp_20AssignVariableOp!assignvariableop_20_conv_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Д
AssignVariableOp_21AssignVariableOpassignvariableop_21_conv_2_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Е
AssignVariableOp_22AssignVariableOp!assignvariableop_22_conv_3_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Д
AssignVariableOp_23AssignVariableOpassignvariableop_23_conv_3_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Е
AssignVariableOp_24AssignVariableOp!assignvariableop_24_conv_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Д
AssignVariableOp_25AssignVariableOpassignvariableop_25_conv_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Д
AssignVariableOp_26AssignVariableOpassignvariableop_26_fc_1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ц
AssignVariableOp_27AssignVariableOpassignvariableop_27_fc_1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Д
AssignVariableOp_28AssignVariableOpassignvariableop_28_fc_2_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ц
AssignVariableOp_29AssignVariableOpassignvariableop_29_fc_2_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30»
AssignVariableOp_30AssignVariableOp'assignvariableop_30_output_layer_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Г
AssignVariableOp_31AssignVariableOp%assignvariableop_31_output_layer_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_32д
AssignVariableOp_32AssignVariableOpassignvariableop_32_adamw_iterIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33е
AssignVariableOp_33AssignVariableOp assignvariableop_33_adamw_beta_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34е
AssignVariableOp_34AssignVariableOp assignvariableop_34_adamw_beta_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Д
AssignVariableOp_35AssignVariableOpassignvariableop_35_adamw_decayIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36»
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adamw_learning_rateIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37«
AssignVariableOp_37AssignVariableOp&assignvariableop_37_adamw_weight_decayIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Х
AssignVariableOp_38AssignVariableOp.assignvariableop_38_patch_encoder_dense_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39┤
AssignVariableOp_39AssignVariableOp,assignvariableop_39_patch_encoder_dense_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Й
AssignVariableOp_40AssignVariableOp6assignvariableop_40_patch_encoder_embedding_embeddingsIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41й
AssignVariableOp_41AssignVariableOp5assignvariableop_41_multi_head_attention_query_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42╗
AssignVariableOp_42AssignVariableOp3assignvariableop_42_multi_head_attention_query_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43╗
AssignVariableOp_43AssignVariableOp3assignvariableop_43_multi_head_attention_key_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44╣
AssignVariableOp_44AssignVariableOp1assignvariableop_44_multi_head_attention_key_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45й
AssignVariableOp_45AssignVariableOp5assignvariableop_45_multi_head_attention_value_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46╗
AssignVariableOp_46AssignVariableOp3assignvariableop_46_multi_head_attention_value_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47╚
AssignVariableOp_47AssignVariableOp@assignvariableop_47_multi_head_attention_attention_output_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48к
AssignVariableOp_48AssignVariableOp>assignvariableop_48_multi_head_attention_attention_output_biasIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49┐
AssignVariableOp_49AssignVariableOp7assignvariableop_49_multi_head_attention_1_query_kernelIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50й
AssignVariableOp_50AssignVariableOp5assignvariableop_50_multi_head_attention_1_query_biasIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51й
AssignVariableOp_51AssignVariableOp5assignvariableop_51_multi_head_attention_1_key_kernelIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52╗
AssignVariableOp_52AssignVariableOp3assignvariableop_52_multi_head_attention_1_key_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53┐
AssignVariableOp_53AssignVariableOp7assignvariableop_53_multi_head_attention_1_value_kernelIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54й
AssignVariableOp_54AssignVariableOp5assignvariableop_54_multi_head_attention_1_value_biasIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55╩
AssignVariableOp_55AssignVariableOpBassignvariableop_55_multi_head_attention_1_attention_output_kernelIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56╚
AssignVariableOp_56AssignVariableOp@assignvariableop_56_multi_head_attention_1_attention_output_biasIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57А
AssignVariableOp_57AssignVariableOpassignvariableop_57_totalIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58А
AssignVariableOp_58AssignVariableOpassignvariableop_58_countIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Б
AssignVariableOp_59AssignVariableOpassignvariableop_59_total_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Б
AssignVariableOp_60AssignVariableOpassignvariableop_60_count_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61й
AssignVariableOp_61AssignVariableOp5assignvariableop_61_adamw_layer_normalization_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62╝
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adamw_layer_normalization_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63┐
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adamw_layer_normalization_1_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Й
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adamw_layer_normalization_1_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65▓
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adamw_dense_1_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66░
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adamw_dense_1_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67▓
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adamw_dense_2_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68░
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adamw_dense_2_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69┐
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adamw_layer_normalization_2_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Й
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adamw_layer_normalization_2_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71┐
AssignVariableOp_71AssignVariableOp7assignvariableop_71_adamw_layer_normalization_3_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Й
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adamw_layer_normalization_3_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73▓
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adamw_dense_3_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74░
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adamw_dense_3_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75▓
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adamw_dense_4_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76░
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adamw_dense_4_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77┐
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adamw_layer_normalization_4_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Й
AssignVariableOp_78AssignVariableOp6assignvariableop_78_adamw_layer_normalization_4_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79▒
AssignVariableOp_79AssignVariableOp)assignvariableop_79_adamw_conv_1_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80»
AssignVariableOp_80AssignVariableOp'assignvariableop_80_adamw_conv_1_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81▒
AssignVariableOp_81AssignVariableOp)assignvariableop_81_adamw_conv_2_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82»
AssignVariableOp_82AssignVariableOp'assignvariableop_82_adamw_conv_2_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83▒
AssignVariableOp_83AssignVariableOp)assignvariableop_83_adamw_conv_3_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84»
AssignVariableOp_84AssignVariableOp'assignvariableop_84_adamw_conv_3_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85▒
AssignVariableOp_85AssignVariableOp)assignvariableop_85_adamw_conv_4_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86»
AssignVariableOp_86AssignVariableOp'assignvariableop_86_adamw_conv_4_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87»
AssignVariableOp_87AssignVariableOp'assignvariableop_87_adamw_fc_1_kernel_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88Г
AssignVariableOp_88AssignVariableOp%assignvariableop_88_adamw_fc_1_bias_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89»
AssignVariableOp_89AssignVariableOp'assignvariableop_89_adamw_fc_2_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90Г
AssignVariableOp_90AssignVariableOp%assignvariableop_90_adamw_fc_2_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91и
AssignVariableOp_91AssignVariableOp/assignvariableop_91_adamw_output_layer_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92х
AssignVariableOp_92AssignVariableOp-assignvariableop_92_adamw_output_layer_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93Й
AssignVariableOp_93AssignVariableOp6assignvariableop_93_adamw_patch_encoder_dense_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94╝
AssignVariableOp_94AssignVariableOp4assignvariableop_94_adamw_patch_encoder_dense_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95к
AssignVariableOp_95AssignVariableOp>assignvariableop_95_adamw_patch_encoder_embedding_embeddings_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96┼
AssignVariableOp_96AssignVariableOp=assignvariableop_96_adamw_multi_head_attention_query_kernel_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97├
AssignVariableOp_97AssignVariableOp;assignvariableop_97_adamw_multi_head_attention_query_bias_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98├
AssignVariableOp_98AssignVariableOp;assignvariableop_98_adamw_multi_head_attention_key_kernel_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99┴
AssignVariableOp_99AssignVariableOp9assignvariableop_99_adamw_multi_head_attention_key_bias_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100╔
AssignVariableOp_100AssignVariableOp>assignvariableop_100_adamw_multi_head_attention_value_kernel_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101К
AssignVariableOp_101AssignVariableOp<assignvariableop_101_adamw_multi_head_attention_value_bias_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102н
AssignVariableOp_102AssignVariableOpIassignvariableop_102_adamw_multi_head_attention_attention_output_kernel_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103м
AssignVariableOp_103AssignVariableOpGassignvariableop_103_adamw_multi_head_attention_attention_output_bias_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104╦
AssignVariableOp_104AssignVariableOp@assignvariableop_104_adamw_multi_head_attention_1_query_kernel_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105╔
AssignVariableOp_105AssignVariableOp>assignvariableop_105_adamw_multi_head_attention_1_query_bias_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106╔
AssignVariableOp_106AssignVariableOp>assignvariableop_106_adamw_multi_head_attention_1_key_kernel_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107К
AssignVariableOp_107AssignVariableOp<assignvariableop_107_adamw_multi_head_attention_1_key_bias_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108╦
AssignVariableOp_108AssignVariableOp@assignvariableop_108_adamw_multi_head_attention_1_value_kernel_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109╔
AssignVariableOp_109AssignVariableOp>assignvariableop_109_adamw_multi_head_attention_1_value_bias_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110о
AssignVariableOp_110AssignVariableOpKassignvariableop_110_adamw_multi_head_attention_1_attention_output_kernel_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111н
AssignVariableOp_111AssignVariableOpIassignvariableop_111_adamw_multi_head_attention_1_attention_output_bias_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112┴
AssignVariableOp_112AssignVariableOp6assignvariableop_112_adamw_layer_normalization_gamma_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113└
AssignVariableOp_113AssignVariableOp5assignvariableop_113_adamw_layer_normalization_beta_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114├
AssignVariableOp_114AssignVariableOp8assignvariableop_114_adamw_layer_normalization_1_gamma_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115┬
AssignVariableOp_115AssignVariableOp7assignvariableop_115_adamw_layer_normalization_1_beta_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116Х
AssignVariableOp_116AssignVariableOp+assignvariableop_116_adamw_dense_1_kernel_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117┤
AssignVariableOp_117AssignVariableOp)assignvariableop_117_adamw_dense_1_bias_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118Х
AssignVariableOp_118AssignVariableOp+assignvariableop_118_adamw_dense_2_kernel_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119┤
AssignVariableOp_119AssignVariableOp)assignvariableop_119_adamw_dense_2_bias_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120├
AssignVariableOp_120AssignVariableOp8assignvariableop_120_adamw_layer_normalization_2_gamma_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121┬
AssignVariableOp_121AssignVariableOp7assignvariableop_121_adamw_layer_normalization_2_beta_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122├
AssignVariableOp_122AssignVariableOp8assignvariableop_122_adamw_layer_normalization_3_gamma_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123┬
AssignVariableOp_123AssignVariableOp7assignvariableop_123_adamw_layer_normalization_3_beta_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124Х
AssignVariableOp_124AssignVariableOp+assignvariableop_124_adamw_dense_3_kernel_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125┤
AssignVariableOp_125AssignVariableOp)assignvariableop_125_adamw_dense_3_bias_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126Х
AssignVariableOp_126AssignVariableOp+assignvariableop_126_adamw_dense_4_kernel_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127┤
AssignVariableOp_127AssignVariableOp)assignvariableop_127_adamw_dense_4_bias_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128├
AssignVariableOp_128AssignVariableOp8assignvariableop_128_adamw_layer_normalization_4_gamma_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129┬
AssignVariableOp_129AssignVariableOp7assignvariableop_129_adamw_layer_normalization_4_beta_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130х
AssignVariableOp_130AssignVariableOp*assignvariableop_130_adamw_conv_1_kernel_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131│
AssignVariableOp_131AssignVariableOp(assignvariableop_131_adamw_conv_1_bias_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132х
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adamw_conv_2_kernel_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133│
AssignVariableOp_133AssignVariableOp(assignvariableop_133_adamw_conv_2_bias_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134х
AssignVariableOp_134AssignVariableOp*assignvariableop_134_adamw_conv_3_kernel_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135│
AssignVariableOp_135AssignVariableOp(assignvariableop_135_adamw_conv_3_bias_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136х
AssignVariableOp_136AssignVariableOp*assignvariableop_136_adamw_conv_4_kernel_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137│
AssignVariableOp_137AssignVariableOp(assignvariableop_137_adamw_conv_4_bias_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138│
AssignVariableOp_138AssignVariableOp(assignvariableop_138_adamw_fc_1_kernel_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139▒
AssignVariableOp_139AssignVariableOp&assignvariableop_139_adamw_fc_1_bias_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140│
AssignVariableOp_140AssignVariableOp(assignvariableop_140_adamw_fc_2_kernel_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_140q
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:2
Identity_141▒
AssignVariableOp_141AssignVariableOp&assignvariableop_141_adamw_fc_2_bias_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_141q
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:2
Identity_142╗
AssignVariableOp_142AssignVariableOp0assignvariableop_142_adamw_output_layer_kernel_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_142q
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:2
Identity_143╣
AssignVariableOp_143AssignVariableOp.assignvariableop_143_adamw_output_layer_bias_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_143q
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:2
Identity_144┬
AssignVariableOp_144AssignVariableOp7assignvariableop_144_adamw_patch_encoder_dense_kernel_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_144q
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:2
Identity_145└
AssignVariableOp_145AssignVariableOp5assignvariableop_145_adamw_patch_encoder_dense_bias_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_145q
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:2
Identity_146╩
AssignVariableOp_146AssignVariableOp?assignvariableop_146_adamw_patch_encoder_embedding_embeddings_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_146q
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:2
Identity_147╔
AssignVariableOp_147AssignVariableOp>assignvariableop_147_adamw_multi_head_attention_query_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_147q
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:2
Identity_148К
AssignVariableOp_148AssignVariableOp<assignvariableop_148_adamw_multi_head_attention_query_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_148q
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:2
Identity_149К
AssignVariableOp_149AssignVariableOp<assignvariableop_149_adamw_multi_head_attention_key_kernel_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149q
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:2
Identity_150┼
AssignVariableOp_150AssignVariableOp:assignvariableop_150_adamw_multi_head_attention_key_bias_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_150q
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:2
Identity_151╔
AssignVariableOp_151AssignVariableOp>assignvariableop_151_adamw_multi_head_attention_value_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_151q
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:2
Identity_152К
AssignVariableOp_152AssignVariableOp<assignvariableop_152_adamw_multi_head_attention_value_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_152q
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:2
Identity_153н
AssignVariableOp_153AssignVariableOpIassignvariableop_153_adamw_multi_head_attention_attention_output_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_153q
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:2
Identity_154м
AssignVariableOp_154AssignVariableOpGassignvariableop_154_adamw_multi_head_attention_attention_output_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_154q
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:2
Identity_155╦
AssignVariableOp_155AssignVariableOp@assignvariableop_155_adamw_multi_head_attention_1_query_kernel_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_155q
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:2
Identity_156╔
AssignVariableOp_156AssignVariableOp>assignvariableop_156_adamw_multi_head_attention_1_query_bias_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_156q
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:2
Identity_157╔
AssignVariableOp_157AssignVariableOp>assignvariableop_157_adamw_multi_head_attention_1_key_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_157q
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:2
Identity_158К
AssignVariableOp_158AssignVariableOp<assignvariableop_158_adamw_multi_head_attention_1_key_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_158q
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:2
Identity_159╦
AssignVariableOp_159AssignVariableOp@assignvariableop_159_adamw_multi_head_attention_1_value_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159q
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:2
Identity_160╔
AssignVariableOp_160AssignVariableOp>assignvariableop_160_adamw_multi_head_attention_1_value_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_160q
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:2
Identity_161о
AssignVariableOp_161AssignVariableOpKassignvariableop_161_adamw_multi_head_attention_1_attention_output_kernel_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_161q
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:2
Identity_162н
AssignVariableOp_162AssignVariableOpIassignvariableop_162_adamw_multi_head_attention_1_attention_output_bias_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1629
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpА
Identity_163Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_163Ћ
Identity_164IdentityIdentity_163:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_164"%
identity_164Identity_164:output:0*П
_input_shapes╦
╚: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
_user_specified_namefile_prefix
Њ
њ
%__inference_FC_1_layer_call_fn_625657

inputs
unknown:@2
	unknown_0:2
identityѕбStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_6224042
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Б
џ
-__inference_output_layer_layer_call_fn_625716

inputs
unknown:2
	unknown_0:
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_6224512
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         2
 
_user_specified_nameinputs
¤
P
$__inference_add_layer_call_fn_625063
inputs_0
inputs_1
identity╬
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_6219392
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:U Q
+
_output_shapes
:         d@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         d@
"
_user_specified_name
inputs/1
І'
Ч
C__inference_dense_3_layer_call_and_return_conditional_losses_622220

inputs4
!tensordot_readvariableop_resource:	@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpЌ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@ђ*
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackљ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         d@2
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/ReshapeЪ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:ђ2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:         dђ2
	TensordotЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         dђ2	
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
:         dђ2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
Gelu/Cast/xє
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:         dђ2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:         dђ2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:         dђ2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:         dђ2

Gelu/mul_1Џ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:         dђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
ѓ'
ч
C__inference_dense_4_layer_call_and_return_conditional_losses_625480

inputs4
!tensordot_readvariableop_resource:	ђ@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpЌ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackЉ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         dђ2
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/Reshapeъ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
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
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2
	Tensordotї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЄ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2	
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
:         d@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
Gelu/Cast/xЁ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:         d@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:         d@2

Gelu/mul_1џ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         dђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         dђ
 
_user_specified_nameinputs
ѓ'
ч
C__inference_dense_2_layer_call_and_return_conditional_losses_622051

inputs4
!tensordot_readvariableop_resource:	ђ@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpЌ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackЉ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         dђ2
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/Reshapeъ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
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
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2
	Tensordotї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЄ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2	
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
:         d@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
Gelu/Cast/xЁ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:         d@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:         d@2

Gelu/mul_1џ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         dђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         dђ
 
_user_specified_nameinputs
Г
Ќ
(__inference_dense_3_layer_call_fn_625442

inputs
unknown:	@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6222202
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         dђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
э-
ч
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_622128	
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
identityѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpК
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЎ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┴
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpЉ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpК
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЎ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
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
:         d@2
Mulа
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2
softmax/SoftmaxЁ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:         dd2
dropout/IdentityИ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpэ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┴
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
attention_output/addѓ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 2J
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
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
ѓ'
ч
C__inference_dense_2_layer_call_and_return_conditional_losses_625179

inputs4
!tensordot_readvariableop_resource:	ђ@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpЌ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
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
Tensordot/GatherV2/axisЛ
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
Tensordot/GatherV2_1/axisО
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
Tensordot/Constђ
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
Tensordot/Const_1ѕ
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
Tensordot/concat/axis░
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatї
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackЉ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         dђ2
Tensordot/transposeЪ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
Tensordot/Reshapeъ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @2
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
Tensordot/concat_1/axisй
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1љ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         d@2
	Tensordotї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЄ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2	
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
:         d@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *зх?2
Gelu/Cast/xЁ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:         d@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:         d@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:         d@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:         d@2

Gelu/mul_1џ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         dђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         dђ
 
_user_specified_nameinputs
┐
ю
'__inference_conv_4_layer_call_fn_625627

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_6223802
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
§љ
щ
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_623556
input_layer'
patch_encoder_623426:	г@"
patch_encoder_623428:@&
patch_encoder_623430:d@(
layer_normalization_623433:@(
layer_normalization_623435:@1
multi_head_attention_623438:@@-
multi_head_attention_623440:@1
multi_head_attention_623442:@@-
multi_head_attention_623444:@1
multi_head_attention_623446:@@-
multi_head_attention_623448:@1
multi_head_attention_623450:@@)
multi_head_attention_623452:@*
layer_normalization_1_623456:@*
layer_normalization_1_623458:@!
dense_1_623461:	@ђ
dense_1_623463:	ђ!
dense_2_623466:	ђ@
dense_2_623468:@*
layer_normalization_2_623472:@*
layer_normalization_2_623474:@3
multi_head_attention_1_623477:@@/
multi_head_attention_1_623479:@3
multi_head_attention_1_623481:@@/
multi_head_attention_1_623483:@3
multi_head_attention_1_623485:@@/
multi_head_attention_1_623487:@3
multi_head_attention_1_623489:@@+
multi_head_attention_1_623491:@*
layer_normalization_3_623495:@*
layer_normalization_3_623497:@!
dense_3_623500:	@ђ
dense_3_623502:	ђ!
dense_4_623505:	ђ@
dense_4_623507:@*
layer_normalization_4_623511:@*
layer_normalization_4_623513:@'
conv_1_623517:@

conv_1_623519:
'
conv_2_623522:


conv_2_623524:
'
conv_3_623527:

conv_3_623529:'
conv_4_623532:
conv_4_623534:
fc_1_623538:@2
fc_1_623540:2
fc_2_623544:22
fc_2_623546:2%
output_layer_623550:2!
output_layer_623552:
identityѕбFC_1/StatefulPartitionedCallбFC_2/StatefulPartitionedCallбconv_1/StatefulPartitionedCallбconv_2/StatefulPartitionedCallбconv_3/StatefulPartitionedCallбconv_4/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdense_2/StatefulPartitionedCallбdense_3/StatefulPartitionedCallбdense_4/StatefulPartitionedCallб+layer_normalization/StatefulPartitionedCallб-layer_normalization_1/StatefulPartitionedCallб-layer_normalization_2/StatefulPartitionedCallб-layer_normalization_3/StatefulPartitionedCallб-layer_normalization_4/StatefulPartitionedCallб,multi_head_attention/StatefulPartitionedCallб.multi_head_attention_1/StatefulPartitionedCallб$output_layer/StatefulPartitionedCallб%patch_encoder/StatefulPartitionedCallС
patches/PartitionedCallPartitionedCallinput_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  г* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_6218022
patches/PartitionedCallс
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_623426patch_encoder_623428patch_encoder_623430*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_6218442'
%patch_encoder/StatefulPartitionedCallэ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_623433layer_normalization_623435*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_6218742-
+layer_normalization/StatefulPartitionedCallз
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_623438multi_head_attention_623440multi_head_attention_623442multi_head_attention_623444multi_head_attention_623446multi_head_attention_623448multi_head_attention_623450multi_head_attention_623452*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_6219152.
,multi_head_attention/StatefulPartitionedCallЕ
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_6219392
add/PartitionedCall№
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_623456layer_normalization_1_623458*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_6219632/
-layer_normalization_1/StatefulPartitionedCall─
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_623461dense_1_623463*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_6220072!
dense_1/StatefulPartitionedCallх
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_623466dense_2_623468*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_6220512!
dense_2/StatefulPartitionedCallљ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_6220632
add_1/PartitionedCallы
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_623472layer_normalization_2_623474*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_6220872/
-layer_normalization_2/StatefulPartitionedCallЇ
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_623477multi_head_attention_1_623479multi_head_attention_1_623481multi_head_attention_1_623483multi_head_attention_1_623485multi_head_attention_1_623487multi_head_attention_1_623489multi_head_attention_1_623491*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8ѓ *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_62212820
.multi_head_attention_1/StatefulPartitionedCallА
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_6221522
add_2/PartitionedCallы
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_623495layer_normalization_3_623497*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_6221762/
-layer_normalization_3/StatefulPartitionedCall─
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_623500dense_3_623502*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         dђ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_6222202!
dense_3/StatefulPartitionedCallх
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_623505dense_4_623507*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_6222642!
dense_4/StatefulPartitionedCallњ
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_6222762
add_3/PartitionedCallы
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_623511layer_normalization_4_623513*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_6223002/
-layer_normalization_4/StatefulPartitionedCallЅ
reshape/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_reshape_layer_call_and_return_conditional_losses_6223202
reshape/PartitionedCallг
conv_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv_1_623517conv_1_623519*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_6223322 
conv_1/StatefulPartitionedCall│
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_623522conv_2_623524*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_6223482 
conv_2/StatefulPartitionedCall│
conv_3/StatefulPartitionedCallStatefulPartitionedCall'conv_2/StatefulPartitionedCall:output:0conv_3_623527conv_3_623529*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_3_layer_call_and_return_conditional_losses_6223642 
conv_3/StatefulPartitionedCall│
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_623532conv_4_623534*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_conv_4_layer_call_and_return_conditional_losses_6223802 
conv_4/StatefulPartitionedCallё
flatten_layer/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_6223922
flatten_layer/PartitionedCallа
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_623538fc_1_623540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_6224042
FC_1/StatefulPartitionedCall 
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_6224152
leaky_ReLu_1/PartitionedCallЪ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_623544fc_2_623546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_6224272
FC_2/StatefulPartitionedCall 
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_6224382
leaky_ReLu_2/PartitionedCallК
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_623550output_layer_623552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_6224512&
$output_layer/StatefulPartitionedCallУ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
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
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameinput_layer
»

ч
B__inference_conv_2_layer_call_and_return_conditional_losses_625580

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         

 
_user_specified_nameinputs
№
├
6__inference_WheatClassifier_CNN_1_layer_call_fn_624021

inputs
unknown:	г@
	unknown_0:@
	unknown_1:d@
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

unknown_14:	@ђ

unknown_15:	ђ

unknown_16:	ђ@

unknown_17:@

unknown_18:@

unknown_19:@ 

unknown_20:@@

unknown_21:@ 

unknown_22:@@

unknown_23:@ 

unknown_24:@@

unknown_25:@ 

unknown_26:@@

unknown_27:@

unknown_28:@

unknown_29:@

unknown_30:	@ђ

unknown_31:	ђ

unknown_32:	ђ@

unknown_33:@

unknown_34:@

unknown_35:@$

unknown_36:@


unknown_37:
$

unknown_38:



unknown_39:
$

unknown_40:


unknown_41:$

unknown_42:

unknown_43:

unknown_44:@2

unknown_45:2

unknown_46:22

unknown_47:2

unknown_48:2

unknown_49:
identityѕбStatefulPartitionedCallд
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
:         *U
_read_only_resource_inputs7
53	
 !"#$%&'()*+,-./0123*-
config_proto

CPU

GPU 2J 8ѓ *Z
fURS
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_6232102
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ќ
_input_shapesё
Ђ:         dd: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Б
љ
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_622176

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityѕбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOpі
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesю
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/meanЅ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:         d2
moments/StopGradientе
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:         d@2
moments/SquaredDifferenceњ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indices┐
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:         d*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *й7є52
batchnorm/add/yњ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:         d2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:         d2
batchnorm/Rsqrtъ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpќ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_1Ѕ
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/mul_2њ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpњ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/subЅ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:         d@2
batchnorm/add_1Ц
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         d@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs
╠	
ы
@__inference_FC_1_layer_call_and_return_conditional_losses_625648

inputs0
matmul_readvariableop_resource:@2-
biasadd_readvariableop_resource:2
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЇ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22	
BiasAddЋ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ш-
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_624965	
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
identityѕб#attention_output/add/ReadVariableOpб-attention_output/einsum/Einsum/ReadVariableOpбkey/add/ReadVariableOpб key/einsum/Einsum/ReadVariableOpбquery/add/ReadVariableOpб"query/einsum/Einsum/ReadVariableOpбvalue/add/ReadVariableOpб"value/einsum/Einsum/ReadVariableOpИ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOpК
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
query/einsum/Einsumќ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЎ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
	query/add▓
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp┴
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
key/einsum/Einsumљ
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpЉ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2	
key/addИ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOpК
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         d@*
equationabc,cde->abde2
value/einsum/Einsumќ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЎ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         d@2
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
:         d@2
Mulа
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:         dd*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:         dd2
softmax/SoftmaxЁ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:         dd2
dropout/IdentityИ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:         d@*
equationacbe,aecd->abcd2
einsum_1/Einsum┘
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpэ
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         d@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum│
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp┴
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d@2
attention_output/addѓ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:         d@:         d@: : : : : : : : 2J
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
:         d@

_user_specified_namequery:RN
+
_output_shapes
:         d@

_user_specified_namevalue
»

ч
B__inference_conv_1_layer_call_and_return_conditional_losses_625561

inputs8
conv2d_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@
*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
*
paddingVALID*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         

@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         

@
 
_user_specified_nameinputs
▀
i
?__inference_add_layer_call_and_return_conditional_losses_621939

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:         d@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:S O
+
_output_shapes
:         d@
 
_user_specified_nameinputs:SO
+
_output_shapes
:         d@
 
_user_specified_nameinputs
М
R
&__inference_add_2_layer_call_fn_625364
inputs_0
inputs_1
identityл
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         d@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_6221522
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         d@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         d@:         d@:U Q
+
_output_shapes
:         d@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:         d@
"
_user_specified_name
inputs/1"╠L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┐
serving_defaultФ
K
input_layer<
serving_default_input_layer:0         dd@
output_layer0
StatefulPartitionedCall:0         tensorflow/serving/predict:у┌
┌w
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
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
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
layer_with_weights-15
layer-22
layer-23
layer_with_weights-16
layer-24
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
	optimizer
trainable_variables
 regularization_losses
!	variables
"	keras_api
#
signatures
­__call__
+ы&call_and_return_all_conditional_losses
Ы_default_save_signature"Ѕo
_tf_keras_networkьn{"name": "WheatClassifier_CNN_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "WheatClassifier_CNN_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Patches", "config": {"layer was saved without config": true}, "name": "patches", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "PatchEncoder", "config": {"layer was saved without config": true}, "name": "patch_encoder", "inbound_nodes": [[["patches", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization", "inbound_nodes": [[["patch_encoder", 0, 0, {}]]]}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}, "name": "multi_head_attention", "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_2", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}, "name": "multi_head_attention_1", "inbound_nodes": [[["layer_normalization_2", 0, 0, {"value": ["layer_normalization_2", 0, 0]}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["multi_head_attention_1", 0, 0, {}], ["add_1", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_3", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["layer_normalization_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["dense_4", 0, 0, {}], ["add_2", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_4", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [10, 10, 64]}}, "name": "reshape", "inbound_nodes": [[["layer_normalization_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_layer", "inbound_nodes": [[["conv_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1", "inbound_nodes": [[["flatten_layer", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_1", "inbound_nodes": [[["FC_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_2", "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_2", "inbound_nodes": [[["FC_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "shared_object_id": 63, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "float32", "input_layer"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 65}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Addons>AdamW", "config": {"name": "AdamW", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false, "weight_decay": 9.999999747378752e-05}}}}
Ё"ѓ
_tf_keras_input_layerР{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
б
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+з&call_and_return_all_conditional_losses
З__call__"Љ
_tf_keras_layerэ{"name": "patches", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Patches", "config": {"layer was saved without config": true}}
Н
(
projection
)position_embedding
*trainable_variables
+regularization_losses
,	variables
-	keras_api
+ш&call_and_return_all_conditional_losses
Ш__call__"ю
_tf_keras_layerѓ{"name": "patch_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PatchEncoder", "config": {"layer was saved without config": true}}
О
.axis
	/gamma
0beta
1trainable_variables
2regularization_losses
3	variables
4	keras_api
+э&call_and_return_all_conditional_losses
Э__call__"Д
_tf_keras_layerЇ{"name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 2}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["patch_encoder", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
з

5_query_dense
6
_key_dense
7_value_dense
8_softmax
9_dropout_layer
:_output_dense
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+щ&call_and_return_all_conditional_losses
Щ__call__"щ
_tf_keras_layer▀{"name": "multi_head_attention", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}, "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]], "shared_object_id": 6}
ъ
?trainable_variables
@regularization_losses
A	variables
B	keras_api
+ч&call_and_return_all_conditional_losses
Ч__call__"Ї
_tf_keras_layerз{"name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 100, 64]}, {"class_name": "TensorShape", "items": [null, 100, 64]}]}
м
Caxis
	Dgamma
Ebeta
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
+§&call_and_return_all_conditional_losses
■__call__"б
_tf_keras_layerѕ{"name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
њ	

Jkernel
Kbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
+ &call_and_return_all_conditional_losses
ђ__call__"в
_tf_keras_layerЛ{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
Ё	

Pkernel
Qbias
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
+Ђ&call_and_return_all_conditional_losses
ѓ__call__"я
_tf_keras_layer─{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 128]}}
ї
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+Ѓ&call_and_return_all_conditional_losses
ё__call__"ч
_tf_keras_layerр{"name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]], "shared_object_id": 17, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 100, 64]}, {"class_name": "TensorShape", "items": [null, 100, 64]}]}
о
Zaxis
	[gamma
\beta
]trainable_variables
^regularization_losses
_	variables
`	keras_api
+Ё&call_and_return_all_conditional_losses
є__call__"д
_tf_keras_layerї{"name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_1", 0, 0, {}]]], "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
■

a_query_dense
b
_key_dense
c_value_dense
d_softmax
e_dropout_layer
f_output_dense
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
+Є&call_and_return_all_conditional_losses
ѕ__call__"ё	
_tf_keras_layerЖ{"name": "multi_head_attention_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}, "inbound_nodes": [[["layer_normalization_2", 0, 0, {"value": ["layer_normalization_2", 0, 0]}]]], "shared_object_id": 23}
Ю
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
+Ѕ&call_and_return_all_conditional_losses
і__call__"ї
_tf_keras_layerЫ{"name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["multi_head_attention_1", 0, 0, {}], ["add_1", 0, 0, {}]]], "shared_object_id": 24, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 100, 64]}, {"class_name": "TensorShape", "items": [null, 100, 64]}]}
о
oaxis
	pgamma
qbeta
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
+І&call_and_return_all_conditional_losses
ї__call__"д
_tf_keras_layerї{"name": "layer_normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_2", 0, 0, {}]]], "shared_object_id": 27, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
њ	

vkernel
wbias
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
+Ї&call_and_return_all_conditional_losses
ј__call__"в
_tf_keras_layerЛ{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["layer_normalization_3", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
Є	

|kernel
}bias
~trainable_variables
regularization_losses
ђ	variables
Ђ	keras_api
+Ј&call_and_return_all_conditional_losses
љ__call__"я
_tf_keras_layer─{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_3", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 128]}}
њ
ѓtrainable_variables
Ѓregularization_losses
ё	variables
Ё	keras_api
+Љ&call_and_return_all_conditional_losses
њ__call__"§
_tf_keras_layerс{"name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["dense_4", 0, 0, {}], ["add_2", 0, 0, {}]]], "shared_object_id": 34, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 100, 64]}, {"class_name": "TensorShape", "items": [null, 100, 64]}]}
П
	єaxis

Єgamma
	ѕbeta
Ѕtrainable_variables
іregularization_losses
І	variables
ї	keras_api
+Њ&call_and_return_all_conditional_losses
ћ__call__"д
_tf_keras_layerї{"name": "layer_normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 36}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_3", 0, 0, {}]]], "shared_object_id": 37, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
╬
Їtrainable_variables
јregularization_losses
Ј	variables
љ	keras_api
+Ћ&call_and_return_all_conditional_losses
ќ__call__"╣
_tf_keras_layerЪ{"name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [10, 10, 64]}}, "inbound_nodes": [[["layer_normalization_4", 0, 0, {}]]], "shared_object_id": 38}
Ё
Љkernel
	њbias
Њtrainable_variables
ћregularization_losses
Ћ	variables
ќ	keras_api
+Ќ&call_and_return_all_conditional_losses
ў__call__"п	
_tf_keras_layerЙ	{"name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["reshape", 0, 0, {}]]], "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10, 10, 64]}}
ѓ
Ќkernel
	ўbias
Ўtrainable_variables
џregularization_losses
Џ	variables
ю	keras_api
+Ў&call_and_return_all_conditional_losses
џ__call__"Н	
_tf_keras_layer╗	{"name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 43}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 10]}}
ѓ
Юkernel
	ъbias
Ъtrainable_variables
аregularization_losses
А	variables
б	keras_api
+Џ&call_and_return_all_conditional_losses
ю__call__"Н	
_tf_keras_layer╗	{"name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6, 6, 10]}}
ѓ
Бkernel
	цbias
Цtrainable_variables
дregularization_losses
Д	variables
е	keras_api
+Ю&call_and_return_all_conditional_losses
ъ__call__"Н	
_tf_keras_layer╗	{"name": "conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 48}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 49}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 50, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 16]}}
¤
Еtrainable_variables
фregularization_losses
Ф	variables
г	keras_api
+Ъ&call_and_return_all_conditional_losses
а__call__"║
_tf_keras_layerа{"name": "flatten_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["conv_4", 0, 0, {}]]], "shared_object_id": 51, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 74}}
є	
Гkernel
	«bias
»trainable_variables
░regularization_losses
▒	variables
▓	keras_api
+А&call_and_return_all_conditional_losses
б__call__"┘
_tf_keras_layer┐{"name": "FC_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_layer", 0, 0, {}]]], "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
Б
│trainable_variables
┤regularization_losses
х	variables
Х	keras_api
+Б&call_and_return_all_conditional_losses
ц__call__"ј
_tf_keras_layerЗ{"name": "leaky_ReLu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_1", 0, 0, {}]]], "shared_object_id": 55}
Ё	
иkernel
	Иbias
╣trainable_variables
║regularization_losses
╗	variables
╝	keras_api
+Ц&call_and_return_all_conditional_losses
д__call__"п
_tf_keras_layerЙ{"name": "FC_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 56}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 57}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]], "shared_object_id": 58, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Б
йtrainable_variables
Йregularization_losses
┐	variables
└	keras_api
+Д&call_and_return_all_conditional_losses
е__call__"ј
_tf_keras_layerЗ{"name": "leaky_ReLu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_2", 0, 0, {}]]], "shared_object_id": 59}
Ћ	
┴kernel
	┬bias
├trainable_variables
─regularization_losses
┼	variables
к	keras_api
+Е&call_and_return_all_conditional_losses
ф__call__"У
_tf_keras_layer╬{"name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 60}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 61}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]], "shared_object_id": 62, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Г	
	Кiter
╚beta_1
╔beta_2

╩decay
╦learning_rate
╠weight_decay/mі0mІDmїEmЇJmјKmЈPmљQmЉ[mњ\mЊpmћqmЋvmќwmЌ|mў}mЎ	Єmџ	ѕmЏ	Љmю	њmЮ	Ќmъ	ўmЪ	Юmа	ъmА	Бmб	цmБ	Гmц	«mЦ	иmд	ИmД	┴mе	┬mЕ	═mф	╬mФ	¤mг	лmГ	Лm«	мm»	Мm░	нm▒	Нm▓	оm│	Оm┤	пmх	┘mХ	┌mи	█mИ	▄m╣	Пm║	яm╗	▀m╝/vй0vЙDv┐Ev└Jv┴Kv┬Pv├Qv─[v┼\vкpvКqv╚vv╔wv╩|v╦}v╠	Єv═	ѕv╬	Љv¤	њvл	ЌvЛ	ўvм	ЮvМ	ъvн	БvН	цvо	ГvО	«vп	иv┘	Иv┌	┴v█	┬v▄	═vП	╬vя	¤v▀	лvЯ	Лvр	мvР	Мvс	нvС	Нvт	оvТ	Оvу	пvУ	┘vж	┌vЖ	█vв	▄vВ	Пvь	яvЬ	▀v№"
	optimizer
Л
═0
╬1
¤2
/3
04
л5
Л6
м7
М8
н9
Н10
о11
О12
D13
E14
J15
K16
P17
Q18
[19
\20
п21
┘22
┌23
█24
▄25
П26
я27
▀28
p29
q30
v31
w32
|33
}34
Є35
ѕ36
Љ37
њ38
Ќ39
ў40
Ю41
ъ42
Б43
ц44
Г45
«46
и47
И48
┴49
┬50"
trackable_list_wrapper
 "
trackable_list_wrapper
Л
═0
╬1
¤2
/3
04
л5
Л6
м7
М8
н9
Н10
о11
О12
D13
E14
J15
K16
P17
Q18
[19
\20
п21
┘22
┌23
█24
▄25
П26
я27
▀28
p29
q30
v31
w32
|33
}34
Є35
ѕ36
Љ37
њ38
Ќ39
ў40
Ю41
ъ42
Б43
ц44
Г45
«46
и47
И48
┴49
┬50"
trackable_list_wrapper
М
trainable_variables
Яmetrics
рlayers
 regularization_losses
 Рlayer_regularization_losses
!	variables
сlayer_metrics
Сnon_trainable_variables
­__call__
Ы_default_save_signature
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
-
Фserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
$trainable_variables
тmetrics
Тlayers
 уlayer_regularization_losses
%regularization_losses
&	variables
Уlayer_metrics
жnon_trainable_variables
З__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
я
═kernel
	╬bias
Жtrainable_variables
вregularization_losses
В	variables
ь	keras_api
+г&call_and_return_all_conditional_losses
Г__call__"▒
_tf_keras_layerЌ{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 78}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 79}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 80, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 300]}}
█
¤
embeddings
Ьtrainable_variables
№regularization_losses
­	variables
ы	keras_api
+«&call_and_return_all_conditional_losses
»__call__"х
_tf_keras_layerЏ{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 100, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 82}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "shared_object_id": 83, "build_input_shape": {"class_name": "TensorShape", "items": [100]}}
8
═0
╬1
¤2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
═0
╬1
¤2"
trackable_list_wrapper
х
*trainable_variables
Ыmetrics
зlayers
 Зlayer_regularization_losses
+regularization_losses
,	variables
шlayer_metrics
Шnon_trainable_variables
Ш__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@2layer_normalization/gamma
&:$@2layer_normalization/beta
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
х
1trainable_variables
эmetrics
Эlayers
 щlayer_regularization_losses
2regularization_losses
3	variables
Щlayer_metrics
чnon_trainable_variables
Э__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
њ
Чpartial_output_shape
§full_output_shape
лkernel
	Лbias
■trainable_variables
 regularization_losses
ђ	variables
Ђ	keras_api
+░&call_and_return_all_conditional_losses
▒__call__"▓
_tf_keras_layerў{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 84, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
ј
ѓpartial_output_shape
Ѓfull_output_shape
мkernel
	Мbias
ёtrainable_variables
Ёregularization_losses
є	variables
Є	keras_api
+▓&call_and_return_all_conditional_losses
│__call__"«
_tf_keras_layerћ{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 85, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
њ
ѕpartial_output_shape
Ѕfull_output_shape
нkernel
	Нbias
іtrainable_variables
Іregularization_losses
ї	variables
Ї	keras_api
+┤&call_and_return_all_conditional_losses
х__call__"▓
_tf_keras_layerў{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 86, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
Ѓ
јtrainable_variables
Јregularization_losses
љ	variables
Љ	keras_api
+Х&call_and_return_all_conditional_losses
и__call__"Ь
_tf_keras_layerн{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "shared_object_id": 87}
 
њtrainable_variables
Њregularization_losses
ћ	variables
Ћ	keras_api
+И&call_and_return_all_conditional_losses
╣__call__"Ж
_tf_keras_layerл{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 88}
Д
ќpartial_output_shape
Ќfull_output_shape
оkernel
	Оbias
ўtrainable_variables
Ўregularization_losses
џ	variables
Џ	keras_api
+║&call_and_return_all_conditional_losses
╗__call__"К
_tf_keras_layerГ{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 64], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 89, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 4, 64]}}
`
л0
Л1
м2
М3
н4
Н5
о6
О7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
л0
Л1
м2
М3
н4
Н5
о6
О7"
trackable_list_wrapper
х
;trainable_variables
юmetrics
Юlayers
 ъlayer_regularization_losses
<regularization_losses
=	variables
Ъlayer_metrics
аnon_trainable_variables
Щ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
?trainable_variables
Аmetrics
бlayers
 Бlayer_regularization_losses
@regularization_losses
A	variables
цlayer_metrics
Цnon_trainable_variables
Ч__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_1/gamma
(:&@2layer_normalization_1/beta
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
х
Ftrainable_variables
дmetrics
Дlayers
 еlayer_regularization_losses
Gregularization_losses
H	variables
Еlayer_metrics
фnon_trainable_variables
■__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
!:	@ђ2dense_1/kernel
:ђ2dense_1/bias
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
х
Ltrainable_variables
Фmetrics
гlayers
 Гlayer_regularization_losses
Mregularization_losses
N	variables
«layer_metrics
»non_trainable_variables
ђ__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
!:	ђ@2dense_2/kernel
:@2dense_2/bias
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
х
Rtrainable_variables
░metrics
▒layers
 ▓layer_regularization_losses
Sregularization_losses
T	variables
│layer_metrics
┤non_trainable_variables
ѓ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Vtrainable_variables
хmetrics
Хlayers
 иlayer_regularization_losses
Wregularization_losses
X	variables
Иlayer_metrics
╣non_trainable_variables
ё__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_2/gamma
(:&@2layer_normalization_2/beta
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
х
]trainable_variables
║metrics
╗layers
 ╝layer_regularization_losses
^regularization_losses
_	variables
йlayer_metrics
Йnon_trainable_variables
є__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
ћ
┐partial_output_shape
└full_output_shape
пkernel
	┘bias
┴trainable_variables
┬regularization_losses
├	variables
─	keras_api
+╝&call_and_return_all_conditional_losses
й__call__"┤
_tf_keras_layerџ{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 90, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
љ
┼partial_output_shape
кfull_output_shape
┌kernel
	█bias
Кtrainable_variables
╚regularization_losses
╔	variables
╩	keras_api
+Й&call_and_return_all_conditional_losses
┐__call__"░
_tf_keras_layerќ{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 91, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
ћ
╦partial_output_shape
╠full_output_shape
▄kernel
	Пbias
═trainable_variables
╬regularization_losses
¤	variables
л	keras_api
+└&call_and_return_all_conditional_losses
┴__call__"┤
_tf_keras_layerџ{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 92, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 64]}}
Ѓ
Лtrainable_variables
мregularization_losses
М	variables
н	keras_api
+┬&call_and_return_all_conditional_losses
├__call__"Ь
_tf_keras_layerн{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "shared_object_id": 93}
 
Нtrainable_variables
оregularization_losses
О	variables
п	keras_api
+─&call_and_return_all_conditional_losses
┼__call__"Ж
_tf_keras_layerл{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 94}
Е
┘partial_output_shape
┌full_output_shape
яkernel
	▀bias
█trainable_variables
▄regularization_losses
П	variables
я	keras_api
+к&call_and_return_all_conditional_losses
К__call__"╔
_tf_keras_layer»{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 64], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 95, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 4, 64]}}
`
п0
┘1
┌2
█3
▄4
П5
я6
▀7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
п0
┘1
┌2
█3
▄4
П5
я6
▀7"
trackable_list_wrapper
х
gtrainable_variables
▀metrics
Яlayers
 рlayer_regularization_losses
hregularization_losses
i	variables
Рlayer_metrics
сnon_trainable_variables
ѕ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
ktrainable_variables
Сmetrics
тlayers
 Тlayer_regularization_losses
lregularization_losses
m	variables
уlayer_metrics
Уnon_trainable_variables
і__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_3/gamma
(:&@2layer_normalization_3/beta
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
х
rtrainable_variables
жmetrics
Жlayers
 вlayer_regularization_losses
sregularization_losses
t	variables
Вlayer_metrics
ьnon_trainable_variables
ї__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
!:	@ђ2dense_3/kernel
:ђ2dense_3/bias
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
х
xtrainable_variables
Ьmetrics
№layers
 ­layer_regularization_losses
yregularization_losses
z	variables
ыlayer_metrics
Ыnon_trainable_variables
ј__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
!:	ђ@2dense_4/kernel
:@2dense_4/bias
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
Х
~trainable_variables
зmetrics
Зlayers
 шlayer_regularization_losses
regularization_losses
ђ	variables
Шlayer_metrics
эnon_trainable_variables
љ__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓtrainable_variables
Эmetrics
щlayers
 Щlayer_regularization_losses
Ѓregularization_losses
ё	variables
чlayer_metrics
Чnon_trainable_variables
њ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_4/gamma
(:&@2layer_normalization_4/beta
0
Є0
ѕ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Є0
ѕ1"
trackable_list_wrapper
И
Ѕtrainable_variables
§metrics
■layers
  layer_regularization_losses
іregularization_losses
І	variables
ђlayer_metrics
Ђnon_trainable_variables
ћ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Їtrainable_variables
ѓmetrics
Ѓlayers
 ёlayer_regularization_losses
јregularization_losses
Ј	variables
Ёlayer_metrics
єnon_trainable_variables
ќ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
':%@
2conv_1/kernel
:
2conv_1/bias
0
Љ0
њ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Љ0
њ1"
trackable_list_wrapper
И
Њtrainable_variables
Єmetrics
ѕlayers
 Ѕlayer_regularization_losses
ћregularization_losses
Ћ	variables
іlayer_metrics
Іnon_trainable_variables
ў__call__
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
':%

2conv_2/kernel
:
2conv_2/bias
0
Ќ0
ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ќ0
ў1"
trackable_list_wrapper
И
Ўtrainable_variables
їmetrics
Їlayers
 јlayer_regularization_losses
џregularization_losses
Џ	variables
Јlayer_metrics
љnon_trainable_variables
џ__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
':%
2conv_3/kernel
:2conv_3/bias
0
Ю0
ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ю0
ъ1"
trackable_list_wrapper
И
Ъtrainable_variables
Љmetrics
њlayers
 Њlayer_regularization_losses
аregularization_losses
А	variables
ћlayer_metrics
Ћnon_trainable_variables
ю__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
':%2conv_4/kernel
:2conv_4/bias
0
Б0
ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Б0
ц1"
trackable_list_wrapper
И
Цtrainable_variables
ќmetrics
Ќlayers
 ўlayer_regularization_losses
дregularization_losses
Д	variables
Ўlayer_metrics
џnon_trainable_variables
ъ__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Еtrainable_variables
Џmetrics
юlayers
 Юlayer_regularization_losses
фregularization_losses
Ф	variables
ъlayer_metrics
Ъnon_trainable_variables
а__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
:@22FC_1/kernel
:22	FC_1/bias
0
Г0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Г0
«1"
trackable_list_wrapper
И
»trainable_variables
аmetrics
Аlayers
 бlayer_regularization_losses
░regularization_losses
▒	variables
Бlayer_metrics
цnon_trainable_variables
б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
│trainable_variables
Цmetrics
дlayers
 Дlayer_regularization_losses
┤regularization_losses
х	variables
еlayer_metrics
Еnon_trainable_variables
ц__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
:222FC_2/kernel
:22	FC_2/bias
0
и0
И1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
и0
И1"
trackable_list_wrapper
И
╣trainable_variables
фmetrics
Фlayers
 гlayer_regularization_losses
║regularization_losses
╗	variables
Гlayer_metrics
«non_trainable_variables
д__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
йtrainable_variables
»metrics
░layers
 ▒layer_regularization_losses
Йregularization_losses
┐	variables
▓layer_metrics
│non_trainable_variables
е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
%:#22output_layer/kernel
:2output_layer/bias
0
┴0
┬1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
┴0
┬1"
trackable_list_wrapper
И
├trainable_variables
┤metrics
хlayers
 Хlayer_regularization_losses
─regularization_losses
┼	variables
иlayer_metrics
Иnon_trainable_variables
ф__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
:	 (2
AdamW/iter
: (2AdamW/beta_1
: (2AdamW/beta_2
: (2AdamW/decay
: (2AdamW/learning_rate
: (2AdamW/weight_decay
-:+	г@2patch_encoder/dense/kernel
&:$@2patch_encoder/dense/bias
4:2d@2"patch_encoder/embedding/embeddings
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
0
╣0
║1"
trackable_list_wrapper
■
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
28"
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
0
═0
╬1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
═0
╬1"
trackable_list_wrapper
И
Жtrainable_variables
╗metrics
╝layers
 йlayer_regularization_losses
вregularization_losses
В	variables
Йlayer_metrics
┐non_trainable_variables
Г__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
(
¤0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
¤0"
trackable_list_wrapper
И
Ьtrainable_variables
└metrics
┴layers
 ┬layer_regularization_losses
№regularization_losses
­	variables
├layer_metrics
─non_trainable_variables
»__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
(0
)1"
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
л0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
л0
Л1"
trackable_list_wrapper
И
■trainable_variables
┼metrics
кlayers
 Кlayer_regularization_losses
 regularization_losses
ђ	variables
╚layer_metrics
╔non_trainable_variables
▒__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
м0
М1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
м0
М1"
trackable_list_wrapper
И
ёtrainable_variables
╩metrics
╦layers
 ╠layer_regularization_losses
Ёregularization_losses
є	variables
═layer_metrics
╬non_trainable_variables
│__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
н0
Н1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
н0
Н1"
trackable_list_wrapper
И
іtrainable_variables
¤metrics
лlayers
 Лlayer_regularization_losses
Іregularization_losses
ї	variables
мlayer_metrics
Мnon_trainable_variables
х__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
јtrainable_variables
нmetrics
Нlayers
 оlayer_regularization_losses
Јregularization_losses
љ	variables
Оlayer_metrics
пnon_trainable_variables
и__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
њtrainable_variables
┘metrics
┌layers
 █layer_regularization_losses
Њregularization_losses
ћ	variables
▄layer_metrics
Пnon_trainable_variables
╣__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
о0
О1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
о0
О1"
trackable_list_wrapper
И
ўtrainable_variables
яmetrics
▀layers
 Яlayer_regularization_losses
Ўregularization_losses
џ	variables
рlayer_metrics
Рnon_trainable_variables
╗__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
50
61
72
83
94
:5"
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
п0
┘1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
п0
┘1"
trackable_list_wrapper
И
┴trainable_variables
сmetrics
Сlayers
 тlayer_regularization_losses
┬regularization_losses
├	variables
Тlayer_metrics
уnon_trainable_variables
й__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
┌0
█1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
┌0
█1"
trackable_list_wrapper
И
Кtrainable_variables
Уmetrics
жlayers
 Жlayer_regularization_losses
╚regularization_losses
╔	variables
вlayer_metrics
Вnon_trainable_variables
┐__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
▄0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
▄0
П1"
trackable_list_wrapper
И
═trainable_variables
ьmetrics
Ьlayers
 №layer_regularization_losses
╬regularization_losses
¤	variables
­layer_metrics
ыnon_trainable_variables
┴__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Лtrainable_variables
Ыmetrics
зlayers
 Зlayer_regularization_losses
мregularization_losses
М	variables
шlayer_metrics
Шnon_trainable_variables
├__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Нtrainable_variables
эmetrics
Эlayers
 щlayer_regularization_losses
оregularization_losses
О	variables
Щlayer_metrics
чnon_trainable_variables
┼__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
я0
▀1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
я0
▀1"
trackable_list_wrapper
И
█trainable_variables
Чmetrics
§layers
 ■layer_regularization_losses
▄regularization_losses
П	variables
 layer_metrics
ђnon_trainable_variables
К__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
a0
b1
c2
d3
e4
f5"
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
п

Ђtotal

ѓcount
Ѓ	variables
ё	keras_api"Ю
_tf_keras_metricѓ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 96}
њ

Ёtotal

єcount
Є
_fn_kwargs
ѕ	variables
Ѕ	keras_api"к
_tf_keras_metricФ{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 65}
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
:  (2total
:  (2count
0
Ђ0
ѓ1"
trackable_list_wrapper
.
Ѓ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ё0
є1"
trackable_list_wrapper
.
ѕ	variables"
_generic_user_object
-:+@2!AdamW/layer_normalization/gamma/m
,:*@2 AdamW/layer_normalization/beta/m
/:-@2#AdamW/layer_normalization_1/gamma/m
.:,@2"AdamW/layer_normalization_1/beta/m
':%	@ђ2AdamW/dense_1/kernel/m
!:ђ2AdamW/dense_1/bias/m
':%	ђ@2AdamW/dense_2/kernel/m
 :@2AdamW/dense_2/bias/m
/:-@2#AdamW/layer_normalization_2/gamma/m
.:,@2"AdamW/layer_normalization_2/beta/m
/:-@2#AdamW/layer_normalization_3/gamma/m
.:,@2"AdamW/layer_normalization_3/beta/m
':%	@ђ2AdamW/dense_3/kernel/m
!:ђ2AdamW/dense_3/bias/m
':%	ђ@2AdamW/dense_4/kernel/m
 :@2AdamW/dense_4/bias/m
/:-@2#AdamW/layer_normalization_4/gamma/m
.:,@2"AdamW/layer_normalization_4/beta/m
-:+@
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
#:!@22AdamW/FC_1/kernel/m
:22AdamW/FC_1/bias/m
#:!222AdamW/FC_2/kernel/m
:22AdamW/FC_2/bias/m
+:)22AdamW/output_layer/kernel/m
%:#2AdamW/output_layer/bias/m
3:1	г@2"AdamW/patch_encoder/dense/kernel/m
,:*@2 AdamW/patch_encoder/dense/bias/m
::8d@2*AdamW/patch_encoder/embedding/embeddings/m
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
-:+@2!AdamW/layer_normalization/gamma/v
,:*@2 AdamW/layer_normalization/beta/v
/:-@2#AdamW/layer_normalization_1/gamma/v
.:,@2"AdamW/layer_normalization_1/beta/v
':%	@ђ2AdamW/dense_1/kernel/v
!:ђ2AdamW/dense_1/bias/v
':%	ђ@2AdamW/dense_2/kernel/v
 :@2AdamW/dense_2/bias/v
/:-@2#AdamW/layer_normalization_2/gamma/v
.:,@2"AdamW/layer_normalization_2/beta/v
/:-@2#AdamW/layer_normalization_3/gamma/v
.:,@2"AdamW/layer_normalization_3/beta/v
':%	@ђ2AdamW/dense_3/kernel/v
!:ђ2AdamW/dense_3/bias/v
':%	ђ@2AdamW/dense_4/kernel/v
 :@2AdamW/dense_4/bias/v
/:-@2#AdamW/layer_normalization_4/gamma/v
.:,@2"AdamW/layer_normalization_4/beta/v
-:+@
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
#:!@22AdamW/FC_1/kernel/v
:22AdamW/FC_1/bias/v
#:!222AdamW/FC_2/kernel/v
:22AdamW/FC_2/bias/v
+:)22AdamW/output_layer/kernel/v
%:#2AdamW/output_layer/bias/v
3:1	г@2"AdamW/patch_encoder/dense/kernel/v
,:*@2 AdamW/patch_encoder/dense/bias/v
::8d@2*AdamW/patch_encoder/embedding/embeddings/v
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
д2Б
6__inference_WheatClassifier_CNN_1_layer_call_fn_622563
6__inference_WheatClassifier_CNN_1_layer_call_fn_623914
6__inference_WheatClassifier_CNN_1_layer_call_fn_624021
6__inference_WheatClassifier_CNN_1_layer_call_fn_623422└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_624418
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_624829
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_623556
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_623690└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
в2У
!__inference__wrapped_model_621781┬
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *2б/
-і*
input_layer         dd
ь2Ж
C__inference_patches_layer_call_and_return_conditional_losses_624843б
Ў▓Ћ
FullArgSpec
argsџ
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_patches_layer_call_fn_624848б
Ў▓Ћ
FullArgSpec
argsџ
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
I__inference_patch_encoder_layer_call_and_return_conditional_losses_624888А
ў▓ћ
FullArgSpec
argsџ
jself
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
.__inference_patch_encoder_layer_call_fn_624899А
ў▓ћ
FullArgSpec
argsџ
jself
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
щ2Ш
O__inference_layer_normalization_layer_call_and_return_conditional_losses_624921б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
я2█
4__inference_layer_normalization_layer_call_fn_624930б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
д2Б
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_624965
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_625007Ч
з▓№
FullArgSpece
args]џZ
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
defaultsџ

 

 
p 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
­2ь
5__inference_multi_head_attention_layer_call_fn_625029
5__inference_multi_head_attention_layer_call_fn_625051Ч
з▓№
FullArgSpece
args]џZ
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
defaultsџ

 

 
p 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ж2Т
?__inference_add_layer_call_and_return_conditional_losses_625057б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╬2╦
$__inference_add_layer_call_fn_625063б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ч2Э
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_625085б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Я2П
6__inference_layer_normalization_1_layer_call_fn_625094б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_1_layer_call_and_return_conditional_losses_625132б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_1_layer_call_fn_625141б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_2_layer_call_and_return_conditional_losses_625179б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_2_layer_call_fn_625188б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_add_1_layer_call_and_return_conditional_losses_625194б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_add_1_layer_call_fn_625200б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ч2Э
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_625222б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Я2П
6__inference_layer_normalization_2_layer_call_fn_625231б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ф2Д
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_625266
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_625308Ч
з▓№
FullArgSpece
args]џZ
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
defaultsџ

 

 
p 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
З2ы
7__inference_multi_head_attention_1_layer_call_fn_625330
7__inference_multi_head_attention_1_layer_call_fn_625352Ч
з▓№
FullArgSpece
args]џZ
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
defaultsџ

 

 
p 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
в2У
A__inference_add_2_layer_call_and_return_conditional_losses_625358б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_add_2_layer_call_fn_625364б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ч2Э
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_625386б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Я2П
6__inference_layer_normalization_3_layer_call_fn_625395б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_3_layer_call_and_return_conditional_losses_625433б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_3_layer_call_fn_625442б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_4_layer_call_and_return_conditional_losses_625480б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_4_layer_call_fn_625489б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_add_3_layer_call_and_return_conditional_losses_625495б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_add_3_layer_call_fn_625501б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ч2Э
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_625523б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Я2П
6__inference_layer_normalization_4_layer_call_fn_625532б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_reshape_layer_call_and_return_conditional_losses_625546б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_reshape_layer_call_fn_625551б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_conv_1_layer_call_and_return_conditional_losses_625561б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_conv_1_layer_call_fn_625570б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_conv_2_layer_call_and_return_conditional_losses_625580б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_conv_2_layer_call_fn_625589б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_conv_3_layer_call_and_return_conditional_losses_625599б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_conv_3_layer_call_fn_625608б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_conv_4_layer_call_and_return_conditional_losses_625618б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_conv_4_layer_call_fn_625627б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
з2­
I__inference_flatten_layer_layer_call_and_return_conditional_losses_625633б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
.__inference_flatten_layer_layer_call_fn_625638б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_FC_1_layer_call_and_return_conditional_losses_625648б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤2╠
%__inference_FC_1_layer_call_fn_625657б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_625662б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_leaky_ReLu_1_layer_call_fn_625667б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_FC_2_layer_call_and_return_conditional_losses_625677б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤2╠
%__inference_FC_2_layer_call_fn_625686б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_625691б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_leaky_ReLu_2_layer_call_fn_625696б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ы2№
H__inference_output_layer_layer_call_and_return_conditional_losses_625707б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
О2н
-__inference_output_layer_layer_call_fn_625716б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤B╠
$__inference_signature_wrapper_623807input_layer"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
х2▓»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
х2▓»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
х2▓»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
х2▓»
д▓б
FullArgSpec%
argsџ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
║2и┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 б
@__inference_FC_1_layer_call_and_return_conditional_losses_625648^Г«/б,
%б"
 і
inputs         @
ф "%б"
і
0         2
џ z
%__inference_FC_1_layer_call_fn_625657QГ«/б,
%б"
 і
inputs         @
ф "і         2б
@__inference_FC_2_layer_call_and_return_conditional_losses_625677^иИ/б,
%б"
 і
inputs         2
ф "%б"
і
0         2
џ z
%__inference_FC_2_layer_call_fn_625686QиИ/б,
%б"
 і
inputs         2
ф "і         2Џ
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_623556┼V═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬DбA
:б7
-і*
input_layer         dd
p 

 
ф "%б"
і
0         
џ Џ
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_623690┼V═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬DбA
:б7
-і*
input_layer         dd
p

 
ф "%б"
і
0         
џ ќ
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_624418└V═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬?б<
5б2
(і%
inputs         dd
p 

 
ф "%б"
і
0         
џ ќ
Q__inference_WheatClassifier_CNN_1_layer_call_and_return_conditional_losses_624829└V═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬?б<
5б2
(і%
inputs         dd
p

 
ф "%б"
і
0         
џ з
6__inference_WheatClassifier_CNN_1_layer_call_fn_622563ИV═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬DбA
:б7
-і*
input_layer         dd
p 

 
ф "і         з
6__inference_WheatClassifier_CNN_1_layer_call_fn_623422ИV═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬DбA
:б7
-і*
input_layer         dd
p

 
ф "і         Ь
6__inference_WheatClassifier_CNN_1_layer_call_fn_623914│V═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬?б<
5б2
(і%
inputs         dd
p 

 
ф "і         Ь
6__inference_WheatClassifier_CNN_1_layer_call_fn_624021│V═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬?б<
5б2
(і%
inputs         dd
p

 
ф "і         щ
!__inference__wrapped_model_621781МV═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬<б9
2б/
-і*
input_layer         dd
ф ";ф8
6
output_layer&і#
output_layer         Н
A__inference_add_1_layer_call_and_return_conditional_losses_625194Јbб_
XбU
SџP
&і#
inputs/0         d@
&і#
inputs/1         d@
ф ")б&
і
0         d@
џ Г
&__inference_add_1_layer_call_fn_625200ѓbб_
XбU
SџP
&і#
inputs/0         d@
&і#
inputs/1         d@
ф "і         d@Н
A__inference_add_2_layer_call_and_return_conditional_losses_625358Јbб_
XбU
SџP
&і#
inputs/0         d@
&і#
inputs/1         d@
ф ")б&
і
0         d@
џ Г
&__inference_add_2_layer_call_fn_625364ѓbб_
XбU
SџP
&і#
inputs/0         d@
&і#
inputs/1         d@
ф "і         d@Н
A__inference_add_3_layer_call_and_return_conditional_losses_625495Јbб_
XбU
SџP
&і#
inputs/0         d@
&і#
inputs/1         d@
ф ")б&
і
0         d@
џ Г
&__inference_add_3_layer_call_fn_625501ѓbб_
XбU
SџP
&і#
inputs/0         d@
&і#
inputs/1         d@
ф "і         d@М
?__inference_add_layer_call_and_return_conditional_losses_625057Јbб_
XбU
SџP
&і#
inputs/0         d@
&і#
inputs/1         d@
ф ")б&
і
0         d@
џ Ф
$__inference_add_layer_call_fn_625063ѓbб_
XбU
SџP
&і#
inputs/0         d@
&і#
inputs/1         d@
ф "і         d@┤
B__inference_conv_1_layer_call_and_return_conditional_losses_625561nЉњ7б4
-б*
(і%
inputs         

@
ф "-б*
#і 
0         

џ ї
'__inference_conv_1_layer_call_fn_625570aЉњ7б4
-б*
(і%
inputs         

@
ф " і         
┤
B__inference_conv_2_layer_call_and_return_conditional_losses_625580nЌў7б4
-б*
(і%
inputs         

ф "-б*
#і 
0         

џ ї
'__inference_conv_2_layer_call_fn_625589aЌў7б4
-б*
(і%
inputs         

ф " і         
┤
B__inference_conv_3_layer_call_and_return_conditional_losses_625599nЮъ7б4
-б*
(і%
inputs         

ф "-б*
#і 
0         
џ ї
'__inference_conv_3_layer_call_fn_625608aЮъ7б4
-б*
(і%
inputs         

ф " і         ┤
B__inference_conv_4_layer_call_and_return_conditional_losses_625618nБц7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ ї
'__inference_conv_4_layer_call_fn_625627aБц7б4
-б*
(і%
inputs         
ф " і         г
C__inference_dense_1_layer_call_and_return_conditional_losses_625132eJK3б0
)б&
$і!
inputs         d@
ф "*б'
 і
0         dђ
џ ё
(__inference_dense_1_layer_call_fn_625141XJK3б0
)б&
$і!
inputs         d@
ф "і         dђг
C__inference_dense_2_layer_call_and_return_conditional_losses_625179ePQ4б1
*б'
%і"
inputs         dђ
ф ")б&
і
0         d@
џ ё
(__inference_dense_2_layer_call_fn_625188XPQ4б1
*б'
%і"
inputs         dђ
ф "і         d@г
C__inference_dense_3_layer_call_and_return_conditional_losses_625433evw3б0
)б&
$і!
inputs         d@
ф "*б'
 і
0         dђ
џ ё
(__inference_dense_3_layer_call_fn_625442Xvw3б0
)б&
$і!
inputs         d@
ф "і         dђг
C__inference_dense_4_layer_call_and_return_conditional_losses_625480e|}4б1
*б'
%і"
inputs         dђ
ф ")б&
і
0         d@
џ ё
(__inference_dense_4_layer_call_fn_625489X|}4б1
*б'
%і"
inputs         dђ
ф "і         d@Г
I__inference_flatten_layer_layer_call_and_return_conditional_losses_625633`7б4
-б*
(і%
inputs         
ф "%б"
і
0         @
џ Ё
.__inference_flatten_layer_layer_call_fn_625638S7б4
-б*
(і%
inputs         
ф "і         @╣
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_625085dDE3б0
)б&
$і!
inputs         d@
ф ")б&
і
0         d@
џ Љ
6__inference_layer_normalization_1_layer_call_fn_625094WDE3б0
)б&
$і!
inputs         d@
ф "і         d@╣
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_625222d[\3б0
)б&
$і!
inputs         d@
ф ")б&
і
0         d@
џ Љ
6__inference_layer_normalization_2_layer_call_fn_625231W[\3б0
)б&
$і!
inputs         d@
ф "і         d@╣
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_625386dpq3б0
)б&
$і!
inputs         d@
ф ")б&
і
0         d@
џ Љ
6__inference_layer_normalization_3_layer_call_fn_625395Wpq3б0
)б&
$і!
inputs         d@
ф "і         d@╗
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_625523fЄѕ3б0
)б&
$і!
inputs         d@
ф ")б&
і
0         d@
џ Њ
6__inference_layer_normalization_4_layer_call_fn_625532YЄѕ3б0
)б&
$і!
inputs         d@
ф "і         d@и
O__inference_layer_normalization_layer_call_and_return_conditional_losses_624921d/03б0
)б&
$і!
inputs         d@
ф ")б&
і
0         d@
џ Ј
4__inference_layer_normalization_layer_call_fn_624930W/03б0
)б&
$і!
inputs         d@
ф "і         d@ц
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_625662X/б,
%б"
 і
inputs         2
ф "%б"
і
0         2
џ |
-__inference_leaky_ReLu_1_layer_call_fn_625667K/б,
%б"
 і
inputs         2
ф "і         2ц
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_625691X/б,
%б"
 і
inputs         2
ф "%б"
і
0         2
џ |
-__inference_leaky_ReLu_2_layer_call_fn_625696K/б,
%б"
 і
inputs         2
ф "і         2§
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_625266дп┘┌█▄Пя▀gбd
]бZ
#і 
query         d@
#і 
value         d@

 

 
p 
p 
ф ")б&
і
0         d@
џ §
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_625308дп┘┌█▄Пя▀gбd
]бZ
#і 
query         d@
#і 
value         d@

 

 
p 
p
ф ")б&
і
0         d@
џ Н
7__inference_multi_head_attention_1_layer_call_fn_625330Ўп┘┌█▄Пя▀gбd
]бZ
#і 
query         d@
#і 
value         d@

 

 
p 
p 
ф "і         d@Н
7__inference_multi_head_attention_1_layer_call_fn_625352Ўп┘┌█▄Пя▀gбd
]бZ
#і 
query         d@
#і 
value         d@

 

 
p 
p
ф "і         d@ч
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_624965длЛмМнНоОgбd
]бZ
#і 
query         d@
#і 
value         d@

 

 
p 
p 
ф ")б&
і
0         d@
џ ч
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_625007длЛмМнНоОgбd
]бZ
#і 
query         d@
#і 
value         d@

 

 
p 
p
ф ")б&
і
0         d@
џ М
5__inference_multi_head_attention_layer_call_fn_625029ЎлЛмМнНоОgбd
]бZ
#і 
query         d@
#і 
value         d@

 

 
p 
p 
ф "і         d@М
5__inference_multi_head_attention_layer_call_fn_625051ЎлЛмМнНоОgбd
]бZ
#і 
query         d@
#і 
value         d@

 

 
p 
p
ф "і         d@ф
H__inference_output_layer_layer_call_and_return_conditional_losses_625707^┴┬/б,
%б"
 і
inputs         2
ф "%б"
і
0         
џ ѓ
-__inference_output_layer_layer_call_fn_625716Q┴┬/б,
%б"
 і
inputs         2
ф "і         Й
I__inference_patch_encoder_layer_call_and_return_conditional_losses_624888q═╬¤<б9
2б/
-і*
patch                  г
ф ")б&
і
0         d@
џ ќ
.__inference_patch_encoder_layer_call_fn_624899d═╬¤<б9
2б/
-і*
patch                  г
ф "і         d@х
C__inference_patches_layer_call_and_return_conditional_losses_624843n7б4
-б*
(і%
images         dd
ф "3б0
)і&
0                  г
џ Ї
(__inference_patches_layer_call_fn_624848a7б4
-б*
(і%
images         dd
ф "&і#                  гФ
C__inference_reshape_layer_call_and_return_conditional_losses_625546d3б0
)б&
$і!
inputs         d@
ф "-б*
#і 
0         

@
џ Ѓ
(__inference_reshape_layer_call_fn_625551W3б0
)б&
$і!
inputs         d@
ф " і         

@І
$__inference_signature_wrapper_623807РV═╬¤/0лЛмМнНоОDEJKPQ[\п┘┌█▄Пя▀pqvw|}ЄѕЉњЌўЮъБцГ«иИ┴┬KбH
б 
Aф>
<
input_layer-і*
input_layer         dd";ф8
6
output_layer&і#
output_layer         