ўЕ#
Џ"∞"
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
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
Ы
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
њ
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
≠
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
alphafloat%ЌћL>"
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
В
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
Н
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
Н
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
dtypetypeИ
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
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
list(type)(0И
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

2	Р
Њ
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
executor_typestring И
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
ц
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-0-ga4dfb8d1a718ТЏ
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
К
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelayer_normalization/gamma
Г
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:@*
dtype0
И
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer_normalization/beta
Б
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:@*
dtype0
О
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_1/gamma
З
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:@*
dtype0
М
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_1/beta
Е
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	@А*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:А*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	А@*
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
О
layer_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_2/gamma
З
/layer_normalization_2/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_2/gamma*
_output_shapes
:@*
dtype0
М
layer_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_2/beta
Е
.layer_normalization_2/beta/Read/ReadVariableOpReadVariableOplayer_normalization_2/beta*
_output_shapes
:@*
dtype0
s
FC_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј(2*
shared_nameFC_1/kernel
l
FC_1/kernel/Read/ReadVariableOpReadVariableOpFC_1/kernel*
_output_shapes
:	ј(2*
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
В
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
С
patch_encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и@*+
shared_namepatch_encoder/dense/kernel
К
.patch_encoder/dense/kernel/Read/ReadVariableOpReadVariableOppatch_encoder/dense/kernel*
_output_shapes
:	и@*
dtype0
И
patch_encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namepatch_encoder/dense/bias
Б
,patch_encoder/dense/bias/Read/ReadVariableOpReadVariableOppatch_encoder/dense/bias*
_output_shapes
:@*
dtype0
†
"patch_encoder/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q@*3
shared_name$"patch_encoder/embedding/embeddings
Щ
6patch_encoder/embedding/embeddings/Read/ReadVariableOpReadVariableOp"patch_encoder/embedding/embeddings*
_output_shapes

:Q@*
dtype0
Ґ
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention/query/kernel
Ы
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:@@*
dtype0
Ъ
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention/query/bias
У
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:@*
dtype0
Ю
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*0
shared_name!multi_head_attention/key/kernel
Ч
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:@@*
dtype0
Ц
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_namemulti_head_attention/key/bias
П
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:@*
dtype0
Ґ
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention/value/kernel
Ы
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:@@*
dtype0
Ъ
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention/value/bias
У
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:@*
dtype0
Є
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*=
shared_name.,multi_head_attention/attention_output/kernel
±
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:@@*
dtype0
ђ
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*multi_head_attention/attention_output/bias
•
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
О
AdamW/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_1/kernel/m
З
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
О
AdamW/conv_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*&
shared_nameAdamW/conv_2/kernel/m
З
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
Ъ
!AdamW/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!AdamW/layer_normalization/gamma/m
У
5AdamW/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp!AdamW/layer_normalization/gamma/m*
_output_shapes
:@*
dtype0
Ш
 AdamW/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/layer_normalization/beta/m
С
4AdamW/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp AdamW/layer_normalization/beta/m*
_output_shapes
:@*
dtype0
Ю
#AdamW/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_1/gamma/m
Ч
7AdamW/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_1/gamma/m*
_output_shapes
:@*
dtype0
Ь
"AdamW/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_1/beta/m
Х
6AdamW/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_1/beta/m*
_output_shapes
:@*
dtype0
Й
AdamW/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*'
shared_nameAdamW/dense_1/kernel/m
В
*AdamW/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_1/kernel/m*
_output_shapes
:	@А*
dtype0
Б
AdamW/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdamW/dense_1/bias/m
z
(AdamW/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_1/bias/m*
_output_shapes	
:А*
dtype0
Й
AdamW/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*'
shared_nameAdamW/dense_2/kernel/m
В
*AdamW/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_2/kernel/m*
_output_shapes
:	А@*
dtype0
А
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
Ю
#AdamW/layer_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_2/gamma/m
Ч
7AdamW/layer_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
Ь
"AdamW/layer_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_2/beta/m
Х
6AdamW/layer_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_2/beta/m*
_output_shapes
:@*
dtype0
Г
AdamW/FC_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј(2*$
shared_nameAdamW/FC_1/kernel/m
|
'AdamW/FC_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/m*
_output_shapes
:	ј(2*
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
В
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
Т
AdamW/output_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_nameAdamW/output_layer/kernel/m
Л
/AdamW/output_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/output_layer/kernel/m*
_output_shapes

:2*
dtype0
К
AdamW/output_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdamW/output_layer/bias/m
Г
-AdamW/output_layer/bias/m/Read/ReadVariableOpReadVariableOpAdamW/output_layer/bias/m*
_output_shapes
:*
dtype0
°
"AdamW/patch_encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и@*3
shared_name$"AdamW/patch_encoder/dense/kernel/m
Ъ
6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/m*
_output_shapes
:	и@*
dtype0
Ш
 AdamW/patch_encoder/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/patch_encoder/dense/bias/m
С
4AdamW/patch_encoder/dense/bias/m/Read/ReadVariableOpReadVariableOp AdamW/patch_encoder/dense/bias/m*
_output_shapes
:@*
dtype0
∞
*AdamW/patch_encoder/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/m
©
>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/m*
_output_shapes

:Q@*
dtype0
≤
)AdamW/multi_head_attention/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/query/kernel/m
Ђ
=AdamW/multi_head_attention/query/kernel/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/query/kernel/m*"
_output_shapes
:@@*
dtype0
™
'AdamW/multi_head_attention/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/query/bias/m
£
;AdamW/multi_head_attention/query/bias/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/query/bias/m*
_output_shapes

:@*
dtype0
Ѓ
'AdamW/multi_head_attention/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'AdamW/multi_head_attention/key/kernel/m
І
;AdamW/multi_head_attention/key/kernel/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/key/kernel/m*"
_output_shapes
:@@*
dtype0
¶
%AdamW/multi_head_attention/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%AdamW/multi_head_attention/key/bias/m
Я
9AdamW/multi_head_attention/key/bias/m/Read/ReadVariableOpReadVariableOp%AdamW/multi_head_attention/key/bias/m*
_output_shapes

:@*
dtype0
≤
)AdamW/multi_head_attention/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/value/kernel/m
Ђ
=AdamW/multi_head_attention/value/kernel/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/value/kernel/m*"
_output_shapes
:@@*
dtype0
™
'AdamW/multi_head_attention/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/value/bias/m
£
;AdamW/multi_head_attention/value/bias/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/value/bias/m*
_output_shapes

:@*
dtype0
»
4AdamW/multi_head_attention/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*E
shared_name64AdamW/multi_head_attention/attention_output/kernel/m
Ѕ
HAdamW/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention/attention_output/kernel/m*"
_output_shapes
:@@*
dtype0
Љ
2AdamW/multi_head_attention/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42AdamW/multi_head_attention/attention_output/bias/m
µ
FAdamW/multi_head_attention/attention_output/bias/m/Read/ReadVariableOpReadVariableOp2AdamW/multi_head_attention/attention_output/bias/m*
_output_shapes
:@*
dtype0
О
AdamW/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_1/kernel/v
З
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
О
AdamW/conv_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:

*&
shared_nameAdamW/conv_2/kernel/v
З
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
Ъ
!AdamW/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!AdamW/layer_normalization/gamma/v
У
5AdamW/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp!AdamW/layer_normalization/gamma/v*
_output_shapes
:@*
dtype0
Ш
 AdamW/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/layer_normalization/beta/v
С
4AdamW/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp AdamW/layer_normalization/beta/v*
_output_shapes
:@*
dtype0
Ю
#AdamW/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_1/gamma/v
Ч
7AdamW/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_1/gamma/v*
_output_shapes
:@*
dtype0
Ь
"AdamW/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_1/beta/v
Х
6AdamW/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_1/beta/v*
_output_shapes
:@*
dtype0
Й
AdamW/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*'
shared_nameAdamW/dense_1/kernel/v
В
*AdamW/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_1/kernel/v*
_output_shapes
:	@А*
dtype0
Б
AdamW/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdamW/dense_1/bias/v
z
(AdamW/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_1/bias/v*
_output_shapes	
:А*
dtype0
Й
AdamW/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*'
shared_nameAdamW/dense_2/kernel/v
В
*AdamW/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_2/kernel/v*
_output_shapes
:	А@*
dtype0
А
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
Ю
#AdamW/layer_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_2/gamma/v
Ч
7AdamW/layer_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
Ь
"AdamW/layer_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_2/beta/v
Х
6AdamW/layer_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_2/beta/v*
_output_shapes
:@*
dtype0
Г
AdamW/FC_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ј(2*$
shared_nameAdamW/FC_1/kernel/v
|
'AdamW/FC_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/v*
_output_shapes
:	ј(2*
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
В
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
Т
AdamW/output_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*,
shared_nameAdamW/output_layer/kernel/v
Л
/AdamW/output_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/output_layer/kernel/v*
_output_shapes

:2*
dtype0
К
AdamW/output_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdamW/output_layer/bias/v
Г
-AdamW/output_layer/bias/v/Read/ReadVariableOpReadVariableOpAdamW/output_layer/bias/v*
_output_shapes
:*
dtype0
°
"AdamW/patch_encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	и@*3
shared_name$"AdamW/patch_encoder/dense/kernel/v
Ъ
6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/v*
_output_shapes
:	и@*
dtype0
Ш
 AdamW/patch_encoder/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*1
shared_name" AdamW/patch_encoder/dense/bias/v
С
4AdamW/patch_encoder/dense/bias/v/Read/ReadVariableOpReadVariableOp AdamW/patch_encoder/dense/bias/v*
_output_shapes
:@*
dtype0
∞
*AdamW/patch_encoder/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Q@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/v
©
>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/v*
_output_shapes

:Q@*
dtype0
≤
)AdamW/multi_head_attention/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/query/kernel/v
Ђ
=AdamW/multi_head_attention/query/kernel/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/query/kernel/v*"
_output_shapes
:@@*
dtype0
™
'AdamW/multi_head_attention/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/query/bias/v
£
;AdamW/multi_head_attention/query/bias/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/query/bias/v*
_output_shapes

:@*
dtype0
Ѓ
'AdamW/multi_head_attention/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'AdamW/multi_head_attention/key/kernel/v
І
;AdamW/multi_head_attention/key/kernel/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/key/kernel/v*"
_output_shapes
:@@*
dtype0
¶
%AdamW/multi_head_attention/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*6
shared_name'%AdamW/multi_head_attention/key/bias/v
Я
9AdamW/multi_head_attention/key/bias/v/Read/ReadVariableOpReadVariableOp%AdamW/multi_head_attention/key/bias/v*
_output_shapes

:@*
dtype0
≤
)AdamW/multi_head_attention/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention/value/kernel/v
Ђ
=AdamW/multi_head_attention/value/kernel/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention/value/kernel/v*"
_output_shapes
:@@*
dtype0
™
'AdamW/multi_head_attention/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention/value/bias/v
£
;AdamW/multi_head_attention/value/bias/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention/value/bias/v*
_output_shapes

:@*
dtype0
»
4AdamW/multi_head_attention/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*E
shared_name64AdamW/multi_head_attention/attention_output/kernel/v
Ѕ
HAdamW/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention/attention_output/kernel/v*"
_output_shapes
:@@*
dtype0
Љ
2AdamW/multi_head_attention/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42AdamW/multi_head_attention/attention_output/bias/v
µ
FAdamW/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpReadVariableOp2AdamW/multi_head_attention/attention_output/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
шЇ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*≤Ї
valueІЇB£Ї BЫЇ
є
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
h

!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
R
'trainable_variables
(regularization_losses
)	variables
*	keras_api
R
+trainable_variables
,regularization_losses
-	variables
.	keras_api
z
/
projection
0position_embedding
1trainable_variables
2regularization_losses
3	variables
4	keras_api
q
5axis
	6gamma
7beta
8trainable_variables
9regularization_losses
:	variables
;	keras_api
ї
<_query_dense
=
_key_dense
>_value_dense
?_softmax
@_dropout_layer
A_output_dense
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
R
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
q
Jaxis
	Kgamma
Lbeta
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
h

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
h

Wkernel
Xbias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
R
]trainable_variables
^regularization_losses
_	variables
`	keras_api
q
aaxis
	bgamma
cbeta
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
R
htrainable_variables
iregularization_losses
j	variables
k	keras_api
h

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
R
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
R
|trainable_variables
}regularization_losses
~	variables
	keras_api
n
Аkernel
	Бbias
Вtrainable_variables
Гregularization_losses
Д	variables
Е	keras_api
ё
	Жiter
Зbeta_1
Иbeta_2

Йdecay
Кlearning_rate
Лweight_decaym÷m„!mЎ"mў6mЏ7mџKm№LmЁQmёRmяWmаXmбbmвcmгlmдmmеvmжwmз	Аmи	Бmй	Мmк	Нmл	Оmм	Пmн	Рmо	Сmп	Тmр	Уmс	Фmт	Хmу	Цmфvхvц!vч"vш6vщ7vъKvыLvьQvэRvюWv€XvАbvБcvВlvГmvДvvЕwvЖ	АvЗ	БvИ	МvЙ	НvК	ОvЛ	ПvМ	РvН	СvО	ТvП	УvР	ФvС	ХvТ	ЦvУ
ы
0
1
!2
"3
М4
Н5
О6
67
78
П9
Р10
С11
Т12
У13
Ф14
Х15
Ц16
K17
L18
Q19
R20
W21
X22
b23
c24
l25
m26
v27
w28
А29
Б30
 
ы
0
1
!2
"3
М4
Н5
О6
67
78
П9
Р10
С11
Т12
У13
Ф14
Х15
Ц16
K17
L18
Q19
R20
W21
X22
b23
c24
l25
m26
v27
w28
А29
Б30
≤
trainable_variables
regularization_losses
	variables
Чnon_trainable_variables
Шlayer_metrics
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
 
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
≤
trainable_variables
regularization_losses
	variables
Ьnon_trainable_variables
Эlayer_metrics
Юlayers
Яmetrics
 †layer_regularization_losses
YW
VARIABLE_VALUEconv_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
≤
#trainable_variables
$regularization_losses
%	variables
°non_trainable_variables
Ґlayer_metrics
£layers
§metrics
 •layer_regularization_losses
 
 
 
≤
'trainable_variables
(regularization_losses
)	variables
¶non_trainable_variables
Іlayer_metrics
®layers
©metrics
 ™layer_regularization_losses
 
 
 
≤
+trainable_variables
,regularization_losses
-	variables
Ђnon_trainable_variables
ђlayer_metrics
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
n
Мkernel
	Нbias
∞trainable_variables
±regularization_losses
≤	variables
≥	keras_api
g
О
embeddings
іtrainable_variables
µregularization_losses
ґ	variables
Ј	keras_api

М0
Н1
О2
 

М0
Н1
О2
≤
1trainable_variables
2regularization_losses
3	variables
Єnon_trainable_variables
єlayer_metrics
Їlayers
їmetrics
 Љlayer_regularization_losses
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
≤
8trainable_variables
9regularization_losses
:	variables
љnon_trainable_variables
Њlayer_metrics
њlayers
јmetrics
 Ѕlayer_regularization_losses
°
¬partial_output_shape
√full_output_shape
Пkernel
	Рbias
ƒtrainable_variables
≈regularization_losses
∆	variables
«	keras_api
°
»partial_output_shape
…full_output_shape
Сkernel
	Тbias
 trainable_variables
Ћregularization_losses
ћ	variables
Ќ	keras_api
°
ќpartial_output_shape
ѕfull_output_shape
Уkernel
	Фbias
–trainable_variables
—regularization_losses
“	variables
”	keras_api
V
‘trainable_variables
’regularization_losses
÷	variables
„	keras_api
V
Ўtrainable_variables
ўregularization_losses
Џ	variables
џ	keras_api
°
№partial_output_shape
Ёfull_output_shape
Хkernel
	Цbias
ёtrainable_variables
яregularization_losses
а	variables
б	keras_api
@
П0
Р1
С2
Т3
У4
Ф5
Х6
Ц7
 
@
П0
Р1
С2
Т3
У4
Ф5
Х6
Ц7
≤
Btrainable_variables
Cregularization_losses
D	variables
вnon_trainable_variables
гlayer_metrics
дlayers
еmetrics
 жlayer_regularization_losses
 
 
 
≤
Ftrainable_variables
Gregularization_losses
H	variables
зnon_trainable_variables
иlayer_metrics
йlayers
кmetrics
 лlayer_regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE

K0
L1
 

K0
L1
≤
Mtrainable_variables
Nregularization_losses
O	variables
мnon_trainable_variables
нlayer_metrics
оlayers
пmetrics
 рlayer_regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
 

Q0
R1
≤
Strainable_variables
Tregularization_losses
U	variables
сnon_trainable_variables
тlayer_metrics
уlayers
фmetrics
 хlayer_regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

W0
X1
 

W0
X1
≤
Ytrainable_variables
Zregularization_losses
[	variables
цnon_trainable_variables
чlayer_metrics
шlayers
щmetrics
 ъlayer_regularization_losses
 
 
 
≤
]trainable_variables
^regularization_losses
_	variables
ыnon_trainable_variables
ьlayer_metrics
эlayers
юmetrics
 €layer_regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE

b0
c1
 

b0
c1
≤
dtrainable_variables
eregularization_losses
f	variables
Аnon_trainable_variables
Бlayer_metrics
Вlayers
Гmetrics
 Дlayer_regularization_losses
 
 
 
≤
htrainable_variables
iregularization_losses
j	variables
Еnon_trainable_variables
Жlayer_metrics
Зlayers
Иmetrics
 Йlayer_regularization_losses
WU
VARIABLE_VALUEFC_1/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	FC_1/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1
 

l0
m1
≤
ntrainable_variables
oregularization_losses
p	variables
Кnon_trainable_variables
Лlayer_metrics
Мlayers
Нmetrics
 Оlayer_regularization_losses
 
 
 
≤
rtrainable_variables
sregularization_losses
t	variables
Пnon_trainable_variables
Рlayer_metrics
Сlayers
Тmetrics
 Уlayer_regularization_losses
XV
VARIABLE_VALUEFC_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	FC_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
 

v0
w1
≤
xtrainable_variables
yregularization_losses
z	variables
Фnon_trainable_variables
Хlayer_metrics
Цlayers
Чmetrics
 Шlayer_regularization_losses
 
 
 
≤
|trainable_variables
}regularization_losses
~	variables
Щnon_trainable_variables
Ъlayer_metrics
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
`^
VARIABLE_VALUEoutput_layer/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEoutput_layer/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

А0
Б1
 

А0
Б1
µ
Вtrainable_variables
Гregularization_losses
Д	variables
Юnon_trainable_variables
Яlayer_metrics
†layers
°metrics
 Ґlayer_regularization_losses
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
VARIABLE_VALUEpatch_encoder/dense/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEpatch_encoder/dense/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE"patch_encoder/embedding/embeddings0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!multi_head_attention/query/kernel0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmulti_head_attention/query/bias1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmulti_head_attention/key/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEmulti_head_attention/key/bias1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!multi_head_attention/value/kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEmulti_head_attention/value/bias1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE,multi_head_attention/attention_output/kernel1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE*multi_head_attention/attention_output/bias1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
 
 
Ц
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

£0
§1
 
 
 
 
 
 
 
 
 
 
 
 
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
М0
Н1
 

М0
Н1
µ
∞trainable_variables
±regularization_losses
≤	variables
•non_trainable_variables
¶layer_metrics
Іlayers
®metrics
 ©layer_regularization_losses

О0
 

О0
µ
іtrainable_variables
µregularization_losses
ґ	variables
™non_trainable_variables
Ђlayer_metrics
ђlayers
≠metrics
 Ѓlayer_regularization_losses
 
 

/0
01
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
П0
Р1
 

П0
Р1
µ
ƒtrainable_variables
≈regularization_losses
∆	variables
ѓnon_trainable_variables
∞layer_metrics
±layers
≤metrics
 ≥layer_regularization_losses
 
 

С0
Т1
 

С0
Т1
µ
 trainable_variables
Ћregularization_losses
ћ	variables
іnon_trainable_variables
µlayer_metrics
ґlayers
Јmetrics
 Єlayer_regularization_losses
 
 

У0
Ф1
 

У0
Ф1
µ
–trainable_variables
—regularization_losses
“	variables
єnon_trainable_variables
Їlayer_metrics
їlayers
Љmetrics
 љlayer_regularization_losses
 
 
 
µ
‘trainable_variables
’regularization_losses
÷	variables
Њnon_trainable_variables
њlayer_metrics
јlayers
Ѕmetrics
 ¬layer_regularization_losses
 
 
 
µ
Ўtrainable_variables
ўregularization_losses
Џ	variables
√non_trainable_variables
ƒlayer_metrics
≈layers
∆metrics
 «layer_regularization_losses
 
 

Х0
Ц1
 

Х0
Ц1
µ
ёtrainable_variables
яregularization_losses
а	variables
»non_trainable_variables
…layer_metrics
 layers
Ћmetrics
 ћlayer_regularization_losses
 
 
*
<0
=1
>2
?3
@4
A5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

Ќtotal

ќcount
ѕ	variables
–	keras_api
I

—total

“count
”
_fn_kwargs
‘	variables
’	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
Ќ0
ќ1

ѕ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

—0
“1

‘	variables
}{
VARIABLE_VALUEAdamW/conv_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE!AdamW/layer_normalization/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE AdamW/layer_normalization/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/FC_1/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamW/FC_1/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_2/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_2/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdamW/output_layer/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdamW/output_layer/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE!AdamW/layer_normalization/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE AdamW/layer_normalization/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/FC_1/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamW/FC_1/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdamW/FC_2/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdamW/FC_2/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEAdamW/output_layer/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEAdamW/output_layer/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Т
serving_default_input_layerPlaceholder*1
_output_shapes
:€€€€€€€€€»»*
dtype0*&
shape:€€€€€€€€€»»
ж
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerconv_1/kernelconv_1/biasconv_2/kernelconv_2/biaspatch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddingslayer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biaslayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/betaFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_391816
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ћ)
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOpFC_1/kernel/Read/ReadVariableOpFC_1/bias/Read/ReadVariableOpFC_2/kernel/Read/ReadVariableOpFC_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdamW/iter/Read/ReadVariableOp AdamW/beta_1/Read/ReadVariableOp AdamW/beta_2/Read/ReadVariableOpAdamW/decay/Read/ReadVariableOp'AdamW/learning_rate/Read/ReadVariableOp&AdamW/weight_decay/Read/ReadVariableOp.patch_encoder/dense/kernel/Read/ReadVariableOp,patch_encoder/dense/bias/Read/ReadVariableOp6patch_encoder/embedding/embeddings/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)AdamW/conv_1/kernel/m/Read/ReadVariableOp'AdamW/conv_1/bias/m/Read/ReadVariableOp)AdamW/conv_2/kernel/m/Read/ReadVariableOp'AdamW/conv_2/bias/m/Read/ReadVariableOp5AdamW/layer_normalization/gamma/m/Read/ReadVariableOp4AdamW/layer_normalization/beta/m/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/m/Read/ReadVariableOp*AdamW/dense_1/kernel/m/Read/ReadVariableOp(AdamW/dense_1/bias/m/Read/ReadVariableOp*AdamW/dense_2/kernel/m/Read/ReadVariableOp(AdamW/dense_2/bias/m/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/m/Read/ReadVariableOp'AdamW/FC_1/kernel/m/Read/ReadVariableOp%AdamW/FC_1/bias/m/Read/ReadVariableOp'AdamW/FC_2/kernel/m/Read/ReadVariableOp%AdamW/FC_2/bias/m/Read/ReadVariableOp/AdamW/output_layer/kernel/m/Read/ReadVariableOp-AdamW/output_layer/bias/m/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/m/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/m/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/m/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/m/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/m/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/m/Read/ReadVariableOp)AdamW/conv_1/kernel/v/Read/ReadVariableOp'AdamW/conv_1/bias/v/Read/ReadVariableOp)AdamW/conv_2/kernel/v/Read/ReadVariableOp'AdamW/conv_2/bias/v/Read/ReadVariableOp5AdamW/layer_normalization/gamma/v/Read/ReadVariableOp4AdamW/layer_normalization/beta/v/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/v/Read/ReadVariableOp*AdamW/dense_1/kernel/v/Read/ReadVariableOp(AdamW/dense_1/bias/v/Read/ReadVariableOp*AdamW/dense_2/kernel/v/Read/ReadVariableOp(AdamW/dense_2/bias/v/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/v/Read/ReadVariableOp'AdamW/FC_1/kernel/v/Read/ReadVariableOp%AdamW/FC_1/bias/v/Read/ReadVariableOp'AdamW/FC_2/kernel/v/Read/ReadVariableOp%AdamW/FC_2/bias/v/Read/ReadVariableOp/AdamW/output_layer/kernel/v/Read/ReadVariableOp-AdamW/output_layer/bias/v/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/v/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/v/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/v/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/v/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/v/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/v/Read/ReadVariableOpConst*t
Tinm
k2i	*
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_393298
ї
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasconv_2/kernelconv_2/biaslayer_normalization/gammalayer_normalization/betalayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/betaFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias
AdamW/iterAdamW/beta_1AdamW/beta_2AdamW/decayAdamW/learning_rateAdamW/weight_decaypatch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddings!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biastotalcounttotal_1count_1AdamW/conv_1/kernel/mAdamW/conv_1/bias/mAdamW/conv_2/kernel/mAdamW/conv_2/bias/m!AdamW/layer_normalization/gamma/m AdamW/layer_normalization/beta/m#AdamW/layer_normalization_1/gamma/m"AdamW/layer_normalization_1/beta/mAdamW/dense_1/kernel/mAdamW/dense_1/bias/mAdamW/dense_2/kernel/mAdamW/dense_2/bias/m#AdamW/layer_normalization_2/gamma/m"AdamW/layer_normalization_2/beta/mAdamW/FC_1/kernel/mAdamW/FC_1/bias/mAdamW/FC_2/kernel/mAdamW/FC_2/bias/mAdamW/output_layer/kernel/mAdamW/output_layer/bias/m"AdamW/patch_encoder/dense/kernel/m AdamW/patch_encoder/dense/bias/m*AdamW/patch_encoder/embedding/embeddings/m)AdamW/multi_head_attention/query/kernel/m'AdamW/multi_head_attention/query/bias/m'AdamW/multi_head_attention/key/kernel/m%AdamW/multi_head_attention/key/bias/m)AdamW/multi_head_attention/value/kernel/m'AdamW/multi_head_attention/value/bias/m4AdamW/multi_head_attention/attention_output/kernel/m2AdamW/multi_head_attention/attention_output/bias/mAdamW/conv_1/kernel/vAdamW/conv_1/bias/vAdamW/conv_2/kernel/vAdamW/conv_2/bias/v!AdamW/layer_normalization/gamma/v AdamW/layer_normalization/beta/v#AdamW/layer_normalization_1/gamma/v"AdamW/layer_normalization_1/beta/vAdamW/dense_1/kernel/vAdamW/dense_1/bias/vAdamW/dense_2/kernel/vAdamW/dense_2/bias/v#AdamW/layer_normalization_2/gamma/v"AdamW/layer_normalization_2/beta/vAdamW/FC_1/kernel/vAdamW/FC_1/bias/vAdamW/FC_2/kernel/vAdamW/FC_2/bias/vAdamW/output_layer/kernel/vAdamW/output_layer/bias/v"AdamW/patch_encoder/dense/kernel/v AdamW/patch_encoder/dense/bias/v*AdamW/patch_encoder/embedding/embeddings/v)AdamW/multi_head_attention/query/kernel/v'AdamW/multi_head_attention/query/bias/v'AdamW/multi_head_attention/key/kernel/v%AdamW/multi_head_attention/key/bias/v)AdamW/multi_head_attention/value/kernel/v'AdamW/multi_head_attention/value/bias/v4AdamW/multi_head_attention/attention_output/kernel/v2AdamW/multi_head_attention/attention_output/bias/v*s
Tinl
j2h*
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_393617љн
ы
d
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_390961

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€2:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Л'
ь
C__inference_dense_1_layer_call_and_return_conditional_losses_392778

inputs4
!tensordot_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@А*
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЯ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€QА2	
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
:€€€€€€€€€QА2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xЖ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€QА2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2

Gelu/mul_1Ы
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€QА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
х-
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_392611	
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
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpЄ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp«
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЩ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpЅ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpС
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp«
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЩ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
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
:€€€€€€€€€Q@2
Mul†
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
softmax/SoftmaxЕ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
dropout/IdentityЄ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpч
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpЅ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
attention_output/addВ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€Q@:€€€€€€€€€Q@: : : : : : : : 2J
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
:€€€€€€€€€Q@

_user_specified_namequery:RN
+
_output_shapes
:€€€€€€€€€Q@

_user_specified_namevalue
я
i
?__inference_add_layer_call_and_return_conditional_losses_390755

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:€€€€€€€€€Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€Q@:€€€€€€€€€Q@:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
ы
d
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_392912

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€2:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
В'
ы
C__inference_dense_2_layer_call_and_return_conditional_losses_392825

inputs4
!tensordot_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	А@*
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2	
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
:€€€€€€€€€Q@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xЕ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Gelu/mul_1Ъ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€QА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€QА
 
_user_specified_nameinputs
•
a
E__inference_maxpool_1_layer_call_and_return_conditional_losses_390558

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
 
Ь
'__inference_conv_1_layer_call_fn_392456

inputs!
unknown:

	unknown_0:

identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€∆∆
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3905812
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€∆∆
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€»»: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
÷
R
&__inference_add_1_layer_call_fn_392846
inputs_0
inputs_1
identity”
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3908792
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€Q@:€€€€€€€€€Q@:U Q
+
_output_shapes
:€€€€€€€€€Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€Q@
"
_user_specified_name
inputs/1
й
m
A__inference_add_1_layer_call_and_return_conditional_losses_392840
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:€€€€€€€€€Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€Q@:€€€€€€€€€Q@:U Q
+
_output_shapes
:€€€€€€€€€Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€Q@
"
_user_specified_name
inputs/1
£
Р
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_390903

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЬ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/meanЙ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesњ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yТ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЦ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_1Й
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpТ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/add_1•
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
–Ѕ
ВF
"__inference__traced_restore_393617
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
,assignvariableop_4_layer_normalization_gamma:@9
+assignvariableop_5_layer_normalization_beta:@<
.assignvariableop_6_layer_normalization_1_gamma:@;
-assignvariableop_7_layer_normalization_1_beta:@4
!assignvariableop_8_dense_1_kernel:	@А.
assignvariableop_9_dense_1_bias:	А5
"assignvariableop_10_dense_2_kernel:	А@.
 assignvariableop_11_dense_2_bias:@=
/assignvariableop_12_layer_normalization_2_gamma:@<
.assignvariableop_13_layer_normalization_2_beta:@2
assignvariableop_14_fc_1_kernel:	ј(2+
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
.assignvariableop_26_patch_encoder_dense_kernel:	и@:
,assignvariableop_27_patch_encoder_dense_bias:@H
6assignvariableop_28_patch_encoder_embedding_embeddings:Q@K
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
)assignvariableop_41_adamw_conv_1_kernel_m:
5
'assignvariableop_42_adamw_conv_1_bias_m:
C
)assignvariableop_43_adamw_conv_2_kernel_m:

5
'assignvariableop_44_adamw_conv_2_bias_m:
C
5assignvariableop_45_adamw_layer_normalization_gamma_m:@B
4assignvariableop_46_adamw_layer_normalization_beta_m:@E
7assignvariableop_47_adamw_layer_normalization_1_gamma_m:@D
6assignvariableop_48_adamw_layer_normalization_1_beta_m:@=
*assignvariableop_49_adamw_dense_1_kernel_m:	@А7
(assignvariableop_50_adamw_dense_1_bias_m:	А=
*assignvariableop_51_adamw_dense_2_kernel_m:	А@6
(assignvariableop_52_adamw_dense_2_bias_m:@E
7assignvariableop_53_adamw_layer_normalization_2_gamma_m:@D
6assignvariableop_54_adamw_layer_normalization_2_beta_m:@:
'assignvariableop_55_adamw_fc_1_kernel_m:	ј(23
%assignvariableop_56_adamw_fc_1_bias_m:29
'assignvariableop_57_adamw_fc_2_kernel_m:223
%assignvariableop_58_adamw_fc_2_bias_m:2A
/assignvariableop_59_adamw_output_layer_kernel_m:2;
-assignvariableop_60_adamw_output_layer_bias_m:I
6assignvariableop_61_adamw_patch_encoder_dense_kernel_m:	и@B
4assignvariableop_62_adamw_patch_encoder_dense_bias_m:@P
>assignvariableop_63_adamw_patch_encoder_embedding_embeddings_m:Q@S
=assignvariableop_64_adamw_multi_head_attention_query_kernel_m:@@M
;assignvariableop_65_adamw_multi_head_attention_query_bias_m:@Q
;assignvariableop_66_adamw_multi_head_attention_key_kernel_m:@@K
9assignvariableop_67_adamw_multi_head_attention_key_bias_m:@S
=assignvariableop_68_adamw_multi_head_attention_value_kernel_m:@@M
;assignvariableop_69_adamw_multi_head_attention_value_bias_m:@^
Hassignvariableop_70_adamw_multi_head_attention_attention_output_kernel_m:@@T
Fassignvariableop_71_adamw_multi_head_attention_attention_output_bias_m:@C
)assignvariableop_72_adamw_conv_1_kernel_v:
5
'assignvariableop_73_adamw_conv_1_bias_v:
C
)assignvariableop_74_adamw_conv_2_kernel_v:

5
'assignvariableop_75_adamw_conv_2_bias_v:
C
5assignvariableop_76_adamw_layer_normalization_gamma_v:@B
4assignvariableop_77_adamw_layer_normalization_beta_v:@E
7assignvariableop_78_adamw_layer_normalization_1_gamma_v:@D
6assignvariableop_79_adamw_layer_normalization_1_beta_v:@=
*assignvariableop_80_adamw_dense_1_kernel_v:	@А7
(assignvariableop_81_adamw_dense_1_bias_v:	А=
*assignvariableop_82_adamw_dense_2_kernel_v:	А@6
(assignvariableop_83_adamw_dense_2_bias_v:@E
7assignvariableop_84_adamw_layer_normalization_2_gamma_v:@D
6assignvariableop_85_adamw_layer_normalization_2_beta_v:@:
'assignvariableop_86_adamw_fc_1_kernel_v:	ј(23
%assignvariableop_87_adamw_fc_1_bias_v:29
'assignvariableop_88_adamw_fc_2_kernel_v:223
%assignvariableop_89_adamw_fc_2_bias_v:2A
/assignvariableop_90_adamw_output_layer_kernel_v:2;
-assignvariableop_91_adamw_output_layer_bias_v:I
6assignvariableop_92_adamw_patch_encoder_dense_kernel_v:	и@B
4assignvariableop_93_adamw_patch_encoder_dense_bias_v:@P
>assignvariableop_94_adamw_patch_encoder_embedding_embeddings_v:Q@S
=assignvariableop_95_adamw_multi_head_attention_query_kernel_v:@@M
;assignvariableop_96_adamw_multi_head_attention_query_bias_v:@Q
;assignvariableop_97_adamw_multi_head_attention_key_kernel_v:@@K
9assignvariableop_98_adamw_multi_head_attention_key_bias_v:@S
=assignvariableop_99_adamw_multi_head_attention_value_kernel_v:@@N
<assignvariableop_100_adamw_multi_head_attention_value_bias_v:@_
Iassignvariableop_101_adamw_multi_head_attention_attention_output_kernel_v:@@U
Gassignvariableop_102_adamw_multi_head_attention_attention_output_bias_v:@
identity_104ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_100ҐAssignVariableOp_101ҐAssignVariableOp_102ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ҐAssignVariableOp_99Ќ9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*ў8
valueѕ8Bћ8hB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesб
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*е
valueџBЎhB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesґ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ґ
_output_shapes£
†::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*v
dtypesl
j2h	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЭ
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2•
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3£
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_layer_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5∞
AssignVariableOp_5AssignVariableOp+assignvariableop_5_layer_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6≥
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7≤
AssignVariableOp_7AssignVariableOp-assignvariableop_7_layer_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¶
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9§
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10™
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11®
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ј
AssignVariableOp_12AssignVariableOp/assignvariableop_12_layer_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ґ
AssignVariableOp_13AssignVariableOp.assignvariableop_13_layer_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14І
AssignVariableOp_14AssignVariableOpassignvariableop_14_fc_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15•
AssignVariableOp_15AssignVariableOpassignvariableop_15_fc_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16І
AssignVariableOp_16AssignVariableOpassignvariableop_16_fc_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17•
AssignVariableOp_17AssignVariableOpassignvariableop_17_fc_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ѓ
AssignVariableOp_18AssignVariableOp'assignvariableop_18_output_layer_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19≠
AssignVariableOp_19AssignVariableOp%assignvariableop_19_output_layer_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20¶
AssignVariableOp_20AssignVariableOpassignvariableop_20_adamw_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp assignvariableop_21_adamw_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22®
AssignVariableOp_22AssignVariableOp assignvariableop_22_adamw_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23І
AssignVariableOp_23AssignVariableOpassignvariableop_23_adamw_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24ѓ
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adamw_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ѓ
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adamw_weight_decayIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ґ
AssignVariableOp_26AssignVariableOp.assignvariableop_26_patch_encoder_dense_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27і
AssignVariableOp_27AssignVariableOp,assignvariableop_27_patch_encoder_dense_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Њ
AssignVariableOp_28AssignVariableOp6assignvariableop_28_patch_encoder_embedding_embeddingsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29љ
AssignVariableOp_29AssignVariableOp5assignvariableop_29_multi_head_attention_query_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ї
AssignVariableOp_30AssignVariableOp3assignvariableop_30_multi_head_attention_query_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ї
AssignVariableOp_31AssignVariableOp3assignvariableop_31_multi_head_attention_key_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32є
AssignVariableOp_32AssignVariableOp1assignvariableop_32_multi_head_attention_key_biasIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33љ
AssignVariableOp_33AssignVariableOp5assignvariableop_33_multi_head_attention_value_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ї
AssignVariableOp_34AssignVariableOp3assignvariableop_34_multi_head_attention_value_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35»
AssignVariableOp_35AssignVariableOp@assignvariableop_35_multi_head_attention_attention_output_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36∆
AssignVariableOp_36AssignVariableOp>assignvariableop_36_multi_head_attention_attention_output_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37°
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38°
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39£
AssignVariableOp_39AssignVariableOpassignvariableop_39_total_1Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40£
AssignVariableOp_40AssignVariableOpassignvariableop_40_count_1Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41±
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adamw_conv_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ѓ
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adamw_conv_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43±
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adamw_conv_2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44ѓ
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adamw_conv_2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45љ
AssignVariableOp_45AssignVariableOp5assignvariableop_45_adamw_layer_normalization_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Љ
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adamw_layer_normalization_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47њ
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adamw_layer_normalization_1_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Њ
AssignVariableOp_48AssignVariableOp6assignvariableop_48_adamw_layer_normalization_1_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49≤
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adamw_dense_1_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50∞
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adamw_dense_1_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51≤
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adamw_dense_2_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52∞
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adamw_dense_2_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53њ
AssignVariableOp_53AssignVariableOp7assignvariableop_53_adamw_layer_normalization_2_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Њ
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adamw_layer_normalization_2_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55ѓ
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adamw_fc_1_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56≠
AssignVariableOp_56AssignVariableOp%assignvariableop_56_adamw_fc_1_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57ѓ
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adamw_fc_2_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58≠
AssignVariableOp_58AssignVariableOp%assignvariableop_58_adamw_fc_2_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ј
AssignVariableOp_59AssignVariableOp/assignvariableop_59_adamw_output_layer_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60µ
AssignVariableOp_60AssignVariableOp-assignvariableop_60_adamw_output_layer_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Њ
AssignVariableOp_61AssignVariableOp6assignvariableop_61_adamw_patch_encoder_dense_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Љ
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adamw_patch_encoder_dense_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63∆
AssignVariableOp_63AssignVariableOp>assignvariableop_63_adamw_patch_encoder_embedding_embeddings_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64≈
AssignVariableOp_64AssignVariableOp=assignvariableop_64_adamw_multi_head_attention_query_kernel_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65√
AssignVariableOp_65AssignVariableOp;assignvariableop_65_adamw_multi_head_attention_query_bias_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66√
AssignVariableOp_66AssignVariableOp;assignvariableop_66_adamw_multi_head_attention_key_kernel_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Ѕ
AssignVariableOp_67AssignVariableOp9assignvariableop_67_adamw_multi_head_attention_key_bias_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68≈
AssignVariableOp_68AssignVariableOp=assignvariableop_68_adamw_multi_head_attention_value_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69√
AssignVariableOp_69AssignVariableOp;assignvariableop_69_adamw_multi_head_attention_value_bias_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70–
AssignVariableOp_70AssignVariableOpHassignvariableop_70_adamw_multi_head_attention_attention_output_kernel_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71ќ
AssignVariableOp_71AssignVariableOpFassignvariableop_71_adamw_multi_head_attention_attention_output_bias_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72±
AssignVariableOp_72AssignVariableOp)assignvariableop_72_adamw_conv_1_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73ѓ
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adamw_conv_1_bias_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74±
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adamw_conv_2_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75ѓ
AssignVariableOp_75AssignVariableOp'assignvariableop_75_adamw_conv_2_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76љ
AssignVariableOp_76AssignVariableOp5assignvariableop_76_adamw_layer_normalization_gamma_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Љ
AssignVariableOp_77AssignVariableOp4assignvariableop_77_adamw_layer_normalization_beta_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78њ
AssignVariableOp_78AssignVariableOp7assignvariableop_78_adamw_layer_normalization_1_gamma_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79Њ
AssignVariableOp_79AssignVariableOp6assignvariableop_79_adamw_layer_normalization_1_beta_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80≤
AssignVariableOp_80AssignVariableOp*assignvariableop_80_adamw_dense_1_kernel_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81∞
AssignVariableOp_81AssignVariableOp(assignvariableop_81_adamw_dense_1_bias_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82≤
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adamw_dense_2_kernel_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83∞
AssignVariableOp_83AssignVariableOp(assignvariableop_83_adamw_dense_2_bias_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84њ
AssignVariableOp_84AssignVariableOp7assignvariableop_84_adamw_layer_normalization_2_gamma_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85Њ
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adamw_layer_normalization_2_beta_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86ѓ
AssignVariableOp_86AssignVariableOp'assignvariableop_86_adamw_fc_1_kernel_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87≠
AssignVariableOp_87AssignVariableOp%assignvariableop_87_adamw_fc_1_bias_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88ѓ
AssignVariableOp_88AssignVariableOp'assignvariableop_88_adamw_fc_2_kernel_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89≠
AssignVariableOp_89AssignVariableOp%assignvariableop_89_adamw_fc_2_bias_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90Ј
AssignVariableOp_90AssignVariableOp/assignvariableop_90_adamw_output_layer_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91µ
AssignVariableOp_91AssignVariableOp-assignvariableop_91_adamw_output_layer_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92Њ
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adamw_patch_encoder_dense_kernel_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93Љ
AssignVariableOp_93AssignVariableOp4assignvariableop_93_adamw_patch_encoder_dense_bias_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94∆
AssignVariableOp_94AssignVariableOp>assignvariableop_94_adamw_patch_encoder_embedding_embeddings_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95≈
AssignVariableOp_95AssignVariableOp=assignvariableop_95_adamw_multi_head_attention_query_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96√
AssignVariableOp_96AssignVariableOp;assignvariableop_96_adamw_multi_head_attention_query_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97√
AssignVariableOp_97AssignVariableOp;assignvariableop_97_adamw_multi_head_attention_key_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98Ѕ
AssignVariableOp_98AssignVariableOp9assignvariableop_98_adamw_multi_head_attention_key_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99≈
AssignVariableOp_99AssignVariableOp=assignvariableop_99_adamw_multi_head_attention_value_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100«
AssignVariableOp_100AssignVariableOp<assignvariableop_100_adamw_multi_head_attention_value_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101‘
AssignVariableOp_101AssignVariableOpIassignvariableop_101_adamw_multi_head_attention_attention_output_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102“
AssignVariableOp_102AssignVariableOpGassignvariableop_102_adamw_multi_head_attention_attention_output_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1029
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpљ
Identity_103Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_103±
Identity_104IdentityIdentity_103:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_104"%
identity_104Identity_104:output:0*е
_input_shapes”
–: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
¶
Ъ
-__inference_output_layer_layer_call_fn_392966

inputs
unknown:2
	unknown_0:
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3909742
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Ё
н
6__inference_WheatClassifier_CNN_5_layer_call_fn_392437

inputs!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:

	unknown_3:	и@
	unknown_4:@
	unknown_5:Q@
	unknown_6:@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@А

unknown_19:	А

unknown_20:	А@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:	ј(2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identityИҐStatefulPartitionedCallС
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
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_3914372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
Юю
Т&
!__inference__wrapped_model_390552
input_layerU
;wheatclassifier_cnn_5_conv_1_conv2d_readvariableop_resource:
J
<wheatclassifier_cnn_5_conv_1_biasadd_readvariableop_resource:
U
;wheatclassifier_cnn_5_conv_2_conv2d_readvariableop_resource:

J
<wheatclassifier_cnn_5_conv_2_biasadd_readvariableop_resource:
^
Kwheatclassifier_cnn_5_patch_encoder_dense_tensordot_readvariableop_resource:	и@W
Iwheatclassifier_cnn_5_patch_encoder_dense_biasadd_readvariableop_resource:@W
Ewheatclassifier_cnn_5_patch_encoder_embedding_embedding_lookup_390368:Q@]
Owheatclassifier_cnn_5_layer_normalization_batchnorm_mul_readvariableop_resource:@Y
Kwheatclassifier_cnn_5_layer_normalization_batchnorm_readvariableop_resource:@l
Vwheatclassifier_cnn_5_multi_head_attention_query_einsum_einsum_readvariableop_resource:@@^
Lwheatclassifier_cnn_5_multi_head_attention_query_add_readvariableop_resource:@j
Twheatclassifier_cnn_5_multi_head_attention_key_einsum_einsum_readvariableop_resource:@@\
Jwheatclassifier_cnn_5_multi_head_attention_key_add_readvariableop_resource:@l
Vwheatclassifier_cnn_5_multi_head_attention_value_einsum_einsum_readvariableop_resource:@@^
Lwheatclassifier_cnn_5_multi_head_attention_value_add_readvariableop_resource:@w
awheatclassifier_cnn_5_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:@@e
Wwheatclassifier_cnn_5_multi_head_attention_attention_output_add_readvariableop_resource:@_
Qwheatclassifier_cnn_5_layer_normalization_1_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_cnn_5_layer_normalization_1_batchnorm_readvariableop_resource:@R
?wheatclassifier_cnn_5_dense_1_tensordot_readvariableop_resource:	@АL
=wheatclassifier_cnn_5_dense_1_biasadd_readvariableop_resource:	АR
?wheatclassifier_cnn_5_dense_2_tensordot_readvariableop_resource:	А@K
=wheatclassifier_cnn_5_dense_2_biasadd_readvariableop_resource:@_
Qwheatclassifier_cnn_5_layer_normalization_2_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_cnn_5_layer_normalization_2_batchnorm_readvariableop_resource:@L
9wheatclassifier_cnn_5_fc_1_matmul_readvariableop_resource:	ј(2H
:wheatclassifier_cnn_5_fc_1_biasadd_readvariableop_resource:2K
9wheatclassifier_cnn_5_fc_2_matmul_readvariableop_resource:22H
:wheatclassifier_cnn_5_fc_2_biasadd_readvariableop_resource:2S
Awheatclassifier_cnn_5_output_layer_matmul_readvariableop_resource:2P
Bwheatclassifier_cnn_5_output_layer_biasadd_readvariableop_resource:
identityИҐ1WheatClassifier_CNN_5/FC_1/BiasAdd/ReadVariableOpҐ0WheatClassifier_CNN_5/FC_1/MatMul/ReadVariableOpҐ1WheatClassifier_CNN_5/FC_2/BiasAdd/ReadVariableOpҐ0WheatClassifier_CNN_5/FC_2/MatMul/ReadVariableOpҐ3WheatClassifier_CNN_5/conv_1/BiasAdd/ReadVariableOpҐ2WheatClassifier_CNN_5/conv_1/Conv2D/ReadVariableOpҐ3WheatClassifier_CNN_5/conv_2/BiasAdd/ReadVariableOpҐ2WheatClassifier_CNN_5/conv_2/Conv2D/ReadVariableOpҐ4WheatClassifier_CNN_5/dense_1/BiasAdd/ReadVariableOpҐ6WheatClassifier_CNN_5/dense_1/Tensordot/ReadVariableOpҐ4WheatClassifier_CNN_5/dense_2/BiasAdd/ReadVariableOpҐ6WheatClassifier_CNN_5/dense_2/Tensordot/ReadVariableOpҐBWheatClassifier_CNN_5/layer_normalization/batchnorm/ReadVariableOpҐFWheatClassifier_CNN_5/layer_normalization/batchnorm/mul/ReadVariableOpҐDWheatClassifier_CNN_5/layer_normalization_1/batchnorm/ReadVariableOpҐHWheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul/ReadVariableOpҐDWheatClassifier_CNN_5/layer_normalization_2/batchnorm/ReadVariableOpҐHWheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul/ReadVariableOpҐNWheatClassifier_CNN_5/multi_head_attention/attention_output/add/ReadVariableOpҐXWheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpҐAWheatClassifier_CNN_5/multi_head_attention/key/add/ReadVariableOpҐKWheatClassifier_CNN_5/multi_head_attention/key/einsum/Einsum/ReadVariableOpҐCWheatClassifier_CNN_5/multi_head_attention/query/add/ReadVariableOpҐMWheatClassifier_CNN_5/multi_head_attention/query/einsum/Einsum/ReadVariableOpҐCWheatClassifier_CNN_5/multi_head_attention/value/add/ReadVariableOpҐMWheatClassifier_CNN_5/multi_head_attention/value/einsum/Einsum/ReadVariableOpҐ9WheatClassifier_CNN_5/output_layer/BiasAdd/ReadVariableOpҐ8WheatClassifier_CNN_5/output_layer/MatMul/ReadVariableOpҐ@WheatClassifier_CNN_5/patch_encoder/dense/BiasAdd/ReadVariableOpҐBWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ReadVariableOpҐ>WheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookupм
2WheatClassifier_CNN_5/conv_1/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_5_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype024
2WheatClassifier_CNN_5/conv_1/Conv2D/ReadVariableOpВ
#WheatClassifier_CNN_5/conv_1/Conv2DConv2Dinput_layer:WheatClassifier_CNN_5/conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_5/conv_1/Conv2Dг
3WheatClassifier_CNN_5/conv_1/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_5_conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_5/conv_1/BiasAdd/ReadVariableOpю
$WheatClassifier_CNN_5/conv_1/BiasAddBiasAdd,WheatClassifier_CNN_5/conv_1/Conv2D:output:0;WheatClassifier_CNN_5/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
2&
$WheatClassifier_CNN_5/conv_1/BiasAddм
2WheatClassifier_CNN_5/conv_2/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_5_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype024
2WheatClassifier_CNN_5/conv_2/Conv2D/ReadVariableOp§
#WheatClassifier_CNN_5/conv_2/Conv2DConv2D-WheatClassifier_CNN_5/conv_1/BiasAdd:output:0:WheatClassifier_CNN_5/conv_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_5/conv_2/Conv2Dг
3WheatClassifier_CNN_5/conv_2/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_5_conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_5/conv_2/BiasAdd/ReadVariableOpю
$WheatClassifier_CNN_5/conv_2/BiasAddBiasAdd,WheatClassifier_CNN_5/conv_2/Conv2D:output:0;WheatClassifier_CNN_5/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
2&
$WheatClassifier_CNN_5/conv_2/BiasAddщ
'WheatClassifier_CNN_5/maxpool_1/MaxPoolMaxPool-WheatClassifier_CNN_5/conv_2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€bb
*
ksize
*
paddingVALID*
strides
2)
'WheatClassifier_CNN_5/maxpool_1/MaxPool™
#WheatClassifier_CNN_5/patches/ShapeShape0WheatClassifier_CNN_5/maxpool_1/MaxPool:output:0*
T0*
_output_shapes
:2%
#WheatClassifier_CNN_5/patches/Shape∞
1WheatClassifier_CNN_5/patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_CNN_5/patches/strided_slice/stackі
3WheatClassifier_CNN_5/patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_5/patches/strided_slice/stack_1і
3WheatClassifier_CNN_5/patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_CNN_5/patches/strided_slice/stack_2Ц
+WheatClassifier_CNN_5/patches/strided_sliceStridedSlice,WheatClassifier_CNN_5/patches/Shape:output:0:WheatClassifier_CNN_5/patches/strided_slice/stack:output:0<WheatClassifier_CNN_5/patches/strided_slice/stack_1:output:0<WheatClassifier_CNN_5/patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+WheatClassifier_CNN_5/patches/strided_sliceЇ
1WheatClassifier_CNN_5/patches/ExtractImagePatchesExtractImagePatches0WheatClassifier_CNN_5/maxpool_1/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€		и*
ksizes


*
paddingVALID*
rates
*
strides


23
1WheatClassifier_CNN_5/patches/ExtractImagePatches©
-WheatClassifier_CNN_5/patches/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-WheatClassifier_CNN_5/patches/Reshape/shape/1°
-WheatClassifier_CNN_5/patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :и2/
-WheatClassifier_CNN_5/patches/Reshape/shape/2ґ
+WheatClassifier_CNN_5/patches/Reshape/shapePack4WheatClassifier_CNN_5/patches/strided_slice:output:06WheatClassifier_CNN_5/patches/Reshape/shape/1:output:06WheatClassifier_CNN_5/patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+WheatClassifier_CNN_5/patches/Reshape/shapeМ
%WheatClassifier_CNN_5/patches/ReshapeReshape;WheatClassifier_CNN_5/patches/ExtractImagePatches:patches:04WheatClassifier_CNN_5/patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2'
%WheatClassifier_CNN_5/patches/Reshape§
/WheatClassifier_CNN_5/patch_encoder/range/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/WheatClassifier_CNN_5/patch_encoder/range/start§
/WheatClassifier_CNN_5/patch_encoder/range/limitConst*
_output_shapes
: *
dtype0*
value	B :Q21
/WheatClassifier_CNN_5/patch_encoder/range/limit§
/WheatClassifier_CNN_5/patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/WheatClassifier_CNN_5/patch_encoder/range/delta©
)WheatClassifier_CNN_5/patch_encoder/rangeRange8WheatClassifier_CNN_5/patch_encoder/range/start:output:08WheatClassifier_CNN_5/patch_encoder/range/limit:output:08WheatClassifier_CNN_5/patch_encoder/range/delta:output:0*
_output_shapes
:Q2+
)WheatClassifier_CNN_5/patch_encoder/rangeХ
BWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOpKwheatclassifier_cnn_5_patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	и@*
dtype02D
BWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ReadVariableOpЊ
8WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/axes≈
8WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/free‘
9WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ShapeShape.WheatClassifier_CNN_5/patches/Reshape:output:0*
T0*
_output_shapes
:2;
9WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Shape»
AWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2/axis£
<WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2GatherV2BWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Shape:output:0AWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/free:output:0JWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2ћ
CWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
CWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2_1/axis©
>WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2_1GatherV2BWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Shape:output:0AWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/axes:output:0LWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2_1ј
9WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Const®
8WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ProdProdEWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2:output:0BWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Prodƒ
;WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Const_1∞
:WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Prod_1ProdGWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2_1:output:0DWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Prod_1ƒ
?WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat/axisВ
:WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concatConcatV2AWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/free:output:0AWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/axes:output:0HWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concatі
9WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/stackPackAWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Prod:output:0CWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/stackј
=WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/transpose	Transpose.WheatClassifier_CNN_5/patches/Reshape:output:0CWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2?
=WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/transpose«
;WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ReshapeReshapeAWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/transpose:y:0BWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Reshape∆
:WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/MatMulMatMulDWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Reshape:output:0JWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2<
:WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/MatMulƒ
;WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Const_2»
AWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat_1/axisП
<WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat_1ConcatV2EWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/GatherV2:output:0DWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/Const_2:output:0JWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat_1Ѕ
3WheatClassifier_CNN_5/patch_encoder/dense/TensordotReshapeDWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/MatMul:product:0EWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@25
3WheatClassifier_CNN_5/patch_encoder/dense/TensordotК
@WheatClassifier_CNN_5/patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOpIwheatclassifier_cnn_5_patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@WheatClassifier_CNN_5/patch_encoder/dense/BiasAdd/ReadVariableOpЄ
1WheatClassifier_CNN_5/patch_encoder/dense/BiasAddBiasAdd<WheatClassifier_CNN_5/patch_encoder/dense/Tensordot:output:0HWheatClassifier_CNN_5/patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@23
1WheatClassifier_CNN_5/patch_encoder/dense/BiasAdd“
>WheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookupResourceGatherEwheatclassifier_cnn_5_patch_encoder_embedding_embedding_lookup_3903682WheatClassifier_CNN_5/patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@WheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup/390368*
_output_shapes

:Q@*
dtype02@
>WheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookupШ
GWheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup/IdentityIdentityGWheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@WheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup/390368*
_output_shapes

:Q@2I
GWheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup/IdentityЭ
IWheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup/Identity_1IdentityPWheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@2K
IWheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup/Identity_1°
'WheatClassifier_CNN_5/patch_encoder/addAddV2:WheatClassifier_CNN_5/patch_encoder/dense/BiasAdd:output:0RWheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2)
'WheatClassifier_CNN_5/patch_encoder/addё
HWheatClassifier_CNN_5/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
HWheatClassifier_CNN_5/layer_normalization/moments/mean/reduction_indicesњ
6WheatClassifier_CNN_5/layer_normalization/moments/meanMean+WheatClassifier_CNN_5/patch_encoder/add:z:0QWheatClassifier_CNN_5/layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(28
6WheatClassifier_CNN_5/layer_normalization/moments/meanЗ
>WheatClassifier_CNN_5/layer_normalization/moments/StopGradientStopGradient?WheatClassifier_CNN_5/layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2@
>WheatClassifier_CNN_5/layer_normalization/moments/StopGradientЋ
CWheatClassifier_CNN_5/layer_normalization/moments/SquaredDifferenceSquaredDifference+WheatClassifier_CNN_5/patch_encoder/add:z:0GWheatClassifier_CNN_5/layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2E
CWheatClassifier_CNN_5/layer_normalization/moments/SquaredDifferenceж
LWheatClassifier_CNN_5/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
LWheatClassifier_CNN_5/layer_normalization/moments/variance/reduction_indicesз
:WheatClassifier_CNN_5/layer_normalization/moments/varianceMeanGWheatClassifier_CNN_5/layer_normalization/moments/SquaredDifference:z:0UWheatClassifier_CNN_5/layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2<
:WheatClassifier_CNN_5/layer_normalization/moments/varianceї
9WheatClassifier_CNN_5/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9WheatClassifier_CNN_5/layer_normalization/batchnorm/add/yЇ
7WheatClassifier_CNN_5/layer_normalization/batchnorm/addAddV2CWheatClassifier_CNN_5/layer_normalization/moments/variance:output:0BWheatClassifier_CNN_5/layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q29
7WheatClassifier_CNN_5/layer_normalization/batchnorm/addт
9WheatClassifier_CNN_5/layer_normalization/batchnorm/RsqrtRsqrt;WheatClassifier_CNN_5/layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2;
9WheatClassifier_CNN_5/layer_normalization/batchnorm/RsqrtЬ
FWheatClassifier_CNN_5/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpOwheatclassifier_cnn_5_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02H
FWheatClassifier_CNN_5/layer_normalization/batchnorm/mul/ReadVariableOpЊ
7WheatClassifier_CNN_5/layer_normalization/batchnorm/mulMul=WheatClassifier_CNN_5/layer_normalization/batchnorm/Rsqrt:y:0NWheatClassifier_CNN_5/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@29
7WheatClassifier_CNN_5/layer_normalization/batchnorm/mulЭ
9WheatClassifier_CNN_5/layer_normalization/batchnorm/mul_1Mul+WheatClassifier_CNN_5/patch_encoder/add:z:0;WheatClassifier_CNN_5/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2;
9WheatClassifier_CNN_5/layer_normalization/batchnorm/mul_1±
9WheatClassifier_CNN_5/layer_normalization/batchnorm/mul_2Mul?WheatClassifier_CNN_5/layer_normalization/moments/mean:output:0;WheatClassifier_CNN_5/layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2;
9WheatClassifier_CNN_5/layer_normalization/batchnorm/mul_2Р
BWheatClassifier_CNN_5/layer_normalization/batchnorm/ReadVariableOpReadVariableOpKwheatclassifier_cnn_5_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02D
BWheatClassifier_CNN_5/layer_normalization/batchnorm/ReadVariableOpЇ
7WheatClassifier_CNN_5/layer_normalization/batchnorm/subSubJWheatClassifier_CNN_5/layer_normalization/batchnorm/ReadVariableOp:value:0=WheatClassifier_CNN_5/layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@29
7WheatClassifier_CNN_5/layer_normalization/batchnorm/sub±
9WheatClassifier_CNN_5/layer_normalization/batchnorm/add_1AddV2=WheatClassifier_CNN_5/layer_normalization/batchnorm/mul_1:z:0;WheatClassifier_CNN_5/layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2;
9WheatClassifier_CNN_5/layer_normalization/batchnorm/add_1є
MWheatClassifier_CNN_5/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_cnn_5_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_CNN_5/multi_head_attention/query/einsum/Einsum/ReadVariableOpА
>WheatClassifier_CNN_5/multi_head_attention/query/einsum/EinsumEinsum=WheatClassifier_CNN_5/layer_normalization/batchnorm/add_1:z:0UWheatClassifier_CNN_5/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2@
>WheatClassifier_CNN_5/multi_head_attention/query/einsum/EinsumЧ
CWheatClassifier_CNN_5/multi_head_attention/query/add/ReadVariableOpReadVariableOpLwheatclassifier_cnn_5_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_CNN_5/multi_head_attention/query/add/ReadVariableOp≈
4WheatClassifier_CNN_5/multi_head_attention/query/addAddV2GWheatClassifier_CNN_5/multi_head_attention/query/einsum/Einsum:output:0KWheatClassifier_CNN_5/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@26
4WheatClassifier_CNN_5/multi_head_attention/query/add≥
KWheatClassifier_CNN_5/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpTwheatclassifier_cnn_5_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02M
KWheatClassifier_CNN_5/multi_head_attention/key/einsum/Einsum/ReadVariableOpъ
<WheatClassifier_CNN_5/multi_head_attention/key/einsum/EinsumEinsum=WheatClassifier_CNN_5/layer_normalization/batchnorm/add_1:z:0SWheatClassifier_CNN_5/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2>
<WheatClassifier_CNN_5/multi_head_attention/key/einsum/EinsumС
AWheatClassifier_CNN_5/multi_head_attention/key/add/ReadVariableOpReadVariableOpJwheatclassifier_cnn_5_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02C
AWheatClassifier_CNN_5/multi_head_attention/key/add/ReadVariableOpљ
2WheatClassifier_CNN_5/multi_head_attention/key/addAddV2EWheatClassifier_CNN_5/multi_head_attention/key/einsum/Einsum:output:0IWheatClassifier_CNN_5/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@24
2WheatClassifier_CNN_5/multi_head_attention/key/addє
MWheatClassifier_CNN_5/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_cnn_5_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_CNN_5/multi_head_attention/value/einsum/Einsum/ReadVariableOpА
>WheatClassifier_CNN_5/multi_head_attention/value/einsum/EinsumEinsum=WheatClassifier_CNN_5/layer_normalization/batchnorm/add_1:z:0UWheatClassifier_CNN_5/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2@
>WheatClassifier_CNN_5/multi_head_attention/value/einsum/EinsumЧ
CWheatClassifier_CNN_5/multi_head_attention/value/add/ReadVariableOpReadVariableOpLwheatclassifier_cnn_5_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_CNN_5/multi_head_attention/value/add/ReadVariableOp≈
4WheatClassifier_CNN_5/multi_head_attention/value/addAddV2GWheatClassifier_CNN_5/multi_head_attention/value/einsum/Einsum:output:0KWheatClassifier_CNN_5/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@26
4WheatClassifier_CNN_5/multi_head_attention/value/add©
0WheatClassifier_CNN_5/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >22
0WheatClassifier_CNN_5/multi_head_attention/Mul/yЦ
.WheatClassifier_CNN_5/multi_head_attention/MulMul8WheatClassifier_CNN_5/multi_head_attention/query/add:z:09WheatClassifier_CNN_5/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€Q@20
.WheatClassifier_CNN_5/multi_head_attention/Mulћ
8WheatClassifier_CNN_5/multi_head_attention/einsum/EinsumEinsum6WheatClassifier_CNN_5/multi_head_attention/key/add:z:02WheatClassifier_CNN_5/multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€QQ*
equationaecd,abcd->acbe2:
8WheatClassifier_CNN_5/multi_head_attention/einsum/EinsumА
:WheatClassifier_CNN_5/multi_head_attention/softmax/SoftmaxSoftmaxAWheatClassifier_CNN_5/multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2<
:WheatClassifier_CNN_5/multi_head_attention/softmax/SoftmaxЖ
;WheatClassifier_CNN_5/multi_head_attention/dropout/IdentityIdentityDWheatClassifier_CNN_5/multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2=
;WheatClassifier_CNN_5/multi_head_attention/dropout/Identityд
:WheatClassifier_CNN_5/multi_head_attention/einsum_1/EinsumEinsumDWheatClassifier_CNN_5/multi_head_attention/dropout/Identity:output:08WheatClassifier_CNN_5/multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationacbe,aecd->abcd2<
:WheatClassifier_CNN_5/multi_head_attention/einsum_1/EinsumЏ
XWheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpawheatclassifier_cnn_5_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02Z
XWheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp£
IWheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/EinsumEinsumCWheatClassifier_CNN_5/multi_head_attention/einsum_1/Einsum:output:0`WheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€Q@*
equationabcd,cde->abe2K
IWheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/Einsumі
NWheatClassifier_CNN_5/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpWwheatclassifier_cnn_5_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02P
NWheatClassifier_CNN_5/multi_head_attention/attention_output/add/ReadVariableOpн
?WheatClassifier_CNN_5/multi_head_attention/attention_output/addAddV2RWheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/Einsum:output:0VWheatClassifier_CNN_5/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2A
?WheatClassifier_CNN_5/multi_head_attention/attention_output/addп
WheatClassifier_CNN_5/add/addAddV2CWheatClassifier_CNN_5/multi_head_attention/attention_output/add:z:0+WheatClassifier_CNN_5/patch_encoder/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
WheatClassifier_CNN_5/add/addв
JWheatClassifier_CNN_5/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_CNN_5/layer_normalization_1/moments/mean/reduction_indicesї
8WheatClassifier_CNN_5/layer_normalization_1/moments/meanMean!WheatClassifier_CNN_5/add/add:z:0SWheatClassifier_CNN_5/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2:
8WheatClassifier_CNN_5/layer_normalization_1/moments/meanН
@WheatClassifier_CNN_5/layer_normalization_1/moments/StopGradientStopGradientAWheatClassifier_CNN_5/layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2B
@WheatClassifier_CNN_5/layer_normalization_1/moments/StopGradient«
EWheatClassifier_CNN_5/layer_normalization_1/moments/SquaredDifferenceSquaredDifference!WheatClassifier_CNN_5/add/add:z:0IWheatClassifier_CNN_5/layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2G
EWheatClassifier_CNN_5/layer_normalization_1/moments/SquaredDifferenceк
NWheatClassifier_CNN_5/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN_5/layer_normalization_1/moments/variance/reduction_indicesп
<WheatClassifier_CNN_5/layer_normalization_1/moments/varianceMeanIWheatClassifier_CNN_5/layer_normalization_1/moments/SquaredDifference:z:0WWheatClassifier_CNN_5/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2>
<WheatClassifier_CNN_5/layer_normalization_1/moments/varianceњ
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52=
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/add/y¬
9WheatClassifier_CNN_5/layer_normalization_1/batchnorm/addAddV2EWheatClassifier_CNN_5/layer_normalization_1/moments/variance:output:0DWheatClassifier_CNN_5/layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2;
9WheatClassifier_CNN_5/layer_normalization_1/batchnorm/addш
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/RsqrtRsqrt=WheatClassifier_CNN_5/layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2=
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/RsqrtҐ
HWheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_cnn_5_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul/ReadVariableOp∆
9WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mulMul?WheatClassifier_CNN_5/layer_normalization_1/batchnorm/Rsqrt:y:0PWheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2;
9WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mulЩ
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul_1Mul!WheatClassifier_CNN_5/add/add:z:0=WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2=
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul_1є
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul_2MulAWheatClassifier_CNN_5/layer_normalization_1/moments/mean:output:0=WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2=
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul_2Ц
DWheatClassifier_CNN_5/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_cnn_5_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_CNN_5/layer_normalization_1/batchnorm/ReadVariableOp¬
9WheatClassifier_CNN_5/layer_normalization_1/batchnorm/subSubLWheatClassifier_CNN_5/layer_normalization_1/batchnorm/ReadVariableOp:value:0?WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2;
9WheatClassifier_CNN_5/layer_normalization_1/batchnorm/subє
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/add_1AddV2?WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul_1:z:0=WheatClassifier_CNN_5/layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2=
;WheatClassifier_CNN_5/layer_normalization_1/batchnorm/add_1с
6WheatClassifier_CNN_5/dense_1/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_cnn_5_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@А*
dtype028
6WheatClassifier_CNN_5/dense_1/Tensordot/ReadVariableOp¶
,WheatClassifier_CNN_5/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_CNN_5/dense_1/Tensordot/axes≠
,WheatClassifier_CNN_5/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_CNN_5/dense_1/Tensordot/freeЌ
-WheatClassifier_CNN_5/dense_1/Tensordot/ShapeShape?WheatClassifier_CNN_5/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_5/dense_1/Tensordot/Shape∞
5WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2/axisз
0WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2GatherV26WheatClassifier_CNN_5/dense_1/Tensordot/Shape:output:05WheatClassifier_CNN_5/dense_1/Tensordot/free:output:0>WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2і
7WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2_1/axisн
2WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2_1GatherV26WheatClassifier_CNN_5/dense_1/Tensordot/Shape:output:05WheatClassifier_CNN_5/dense_1/Tensordot/axes:output:0@WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2_1®
-WheatClassifier_CNN_5/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_CNN_5/dense_1/Tensordot/Constш
,WheatClassifier_CNN_5/dense_1/Tensordot/ProdProd9WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2:output:06WheatClassifier_CNN_5/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_CNN_5/dense_1/Tensordot/Prodђ
/WheatClassifier_CNN_5/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_CNN_5/dense_1/Tensordot/Const_1А
.WheatClassifier_CNN_5/dense_1/Tensordot/Prod_1Prod;WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2_1:output:08WheatClassifier_CNN_5/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_CNN_5/dense_1/Tensordot/Prod_1ђ
3WheatClassifier_CNN_5/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_CNN_5/dense_1/Tensordot/concat/axis∆
.WheatClassifier_CNN_5/dense_1/Tensordot/concatConcatV25WheatClassifier_CNN_5/dense_1/Tensordot/free:output:05WheatClassifier_CNN_5/dense_1/Tensordot/axes:output:0<WheatClassifier_CNN_5/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_CNN_5/dense_1/Tensordot/concatД
-WheatClassifier_CNN_5/dense_1/Tensordot/stackPack5WheatClassifier_CNN_5/dense_1/Tensordot/Prod:output:07WheatClassifier_CNN_5/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_5/dense_1/Tensordot/stack£
1WheatClassifier_CNN_5/dense_1/Tensordot/transpose	Transpose?WheatClassifier_CNN_5/layer_normalization_1/batchnorm/add_1:z:07WheatClassifier_CNN_5/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@23
1WheatClassifier_CNN_5/dense_1/Tensordot/transposeЧ
/WheatClassifier_CNN_5/dense_1/Tensordot/ReshapeReshape5WheatClassifier_CNN_5/dense_1/Tensordot/transpose:y:06WheatClassifier_CNN_5/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€21
/WheatClassifier_CNN_5/dense_1/Tensordot/ReshapeЧ
.WheatClassifier_CNN_5/dense_1/Tensordot/MatMulMatMul8WheatClassifier_CNN_5/dense_1/Tensordot/Reshape:output:0>WheatClassifier_CNN_5/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А20
.WheatClassifier_CNN_5/dense_1/Tensordot/MatMul≠
/WheatClassifier_CNN_5/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А21
/WheatClassifier_CNN_5/dense_1/Tensordot/Const_2∞
5WheatClassifier_CNN_5/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_5/dense_1/Tensordot/concat_1/axis”
0WheatClassifier_CNN_5/dense_1/Tensordot/concat_1ConcatV29WheatClassifier_CNN_5/dense_1/Tensordot/GatherV2:output:08WheatClassifier_CNN_5/dense_1/Tensordot/Const_2:output:0>WheatClassifier_CNN_5/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_CNN_5/dense_1/Tensordot/concat_1Й
'WheatClassifier_CNN_5/dense_1/TensordotReshape8WheatClassifier_CNN_5/dense_1/Tensordot/MatMul:product:09WheatClassifier_CNN_5/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2)
'WheatClassifier_CNN_5/dense_1/Tensordotз
4WheatClassifier_CNN_5/dense_1/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_cnn_5_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype026
4WheatClassifier_CNN_5/dense_1/BiasAdd/ReadVariableOpА
%WheatClassifier_CNN_5/dense_1/BiasAddBiasAdd0WheatClassifier_CNN_5/dense_1/Tensordot:output:0<WheatClassifier_CNN_5/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€QА2'
%WheatClassifier_CNN_5/dense_1/BiasAddЩ
(WheatClassifier_CNN_5/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_CNN_5/dense_1/Gelu/mul/xс
&WheatClassifier_CNN_5/dense_1/Gelu/mulMul1WheatClassifier_CNN_5/dense_1/Gelu/mul/x:output:0.WheatClassifier_CNN_5/dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2(
&WheatClassifier_CNN_5/dense_1/Gelu/mulЫ
)WheatClassifier_CNN_5/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2+
)WheatClassifier_CNN_5/dense_1/Gelu/Cast/xю
*WheatClassifier_CNN_5/dense_1/Gelu/truedivRealDiv.WheatClassifier_CNN_5/dense_1/BiasAdd:output:02WheatClassifier_CNN_5/dense_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2,
*WheatClassifier_CNN_5/dense_1/Gelu/truedivЊ
&WheatClassifier_CNN_5/dense_1/Gelu/ErfErf.WheatClassifier_CNN_5/dense_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2(
&WheatClassifier_CNN_5/dense_1/Gelu/ErfЩ
(WheatClassifier_CNN_5/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(WheatClassifier_CNN_5/dense_1/Gelu/add/xп
&WheatClassifier_CNN_5/dense_1/Gelu/addAddV21WheatClassifier_CNN_5/dense_1/Gelu/add/x:output:0*WheatClassifier_CNN_5/dense_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€QА2(
&WheatClassifier_CNN_5/dense_1/Gelu/addк
(WheatClassifier_CNN_5/dense_1/Gelu/mul_1Mul*WheatClassifier_CNN_5/dense_1/Gelu/mul:z:0*WheatClassifier_CNN_5/dense_1/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2*
(WheatClassifier_CNN_5/dense_1/Gelu/mul_1с
6WheatClassifier_CNN_5/dense_2/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_cnn_5_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	А@*
dtype028
6WheatClassifier_CNN_5/dense_2/Tensordot/ReadVariableOp¶
,WheatClassifier_CNN_5/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_CNN_5/dense_2/Tensordot/axes≠
,WheatClassifier_CNN_5/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_CNN_5/dense_2/Tensordot/freeЇ
-WheatClassifier_CNN_5/dense_2/Tensordot/ShapeShape,WheatClassifier_CNN_5/dense_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_5/dense_2/Tensordot/Shape∞
5WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2/axisз
0WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2GatherV26WheatClassifier_CNN_5/dense_2/Tensordot/Shape:output:05WheatClassifier_CNN_5/dense_2/Tensordot/free:output:0>WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2і
7WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2_1/axisн
2WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2_1GatherV26WheatClassifier_CNN_5/dense_2/Tensordot/Shape:output:05WheatClassifier_CNN_5/dense_2/Tensordot/axes:output:0@WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2_1®
-WheatClassifier_CNN_5/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_CNN_5/dense_2/Tensordot/Constш
,WheatClassifier_CNN_5/dense_2/Tensordot/ProdProd9WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2:output:06WheatClassifier_CNN_5/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_CNN_5/dense_2/Tensordot/Prodђ
/WheatClassifier_CNN_5/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_CNN_5/dense_2/Tensordot/Const_1А
.WheatClassifier_CNN_5/dense_2/Tensordot/Prod_1Prod;WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2_1:output:08WheatClassifier_CNN_5/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_CNN_5/dense_2/Tensordot/Prod_1ђ
3WheatClassifier_CNN_5/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_CNN_5/dense_2/Tensordot/concat/axis∆
.WheatClassifier_CNN_5/dense_2/Tensordot/concatConcatV25WheatClassifier_CNN_5/dense_2/Tensordot/free:output:05WheatClassifier_CNN_5/dense_2/Tensordot/axes:output:0<WheatClassifier_CNN_5/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_CNN_5/dense_2/Tensordot/concatД
-WheatClassifier_CNN_5/dense_2/Tensordot/stackPack5WheatClassifier_CNN_5/dense_2/Tensordot/Prod:output:07WheatClassifier_CNN_5/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_CNN_5/dense_2/Tensordot/stackС
1WheatClassifier_CNN_5/dense_2/Tensordot/transpose	Transpose,WheatClassifier_CNN_5/dense_1/Gelu/mul_1:z:07WheatClassifier_CNN_5/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА23
1WheatClassifier_CNN_5/dense_2/Tensordot/transposeЧ
/WheatClassifier_CNN_5/dense_2/Tensordot/ReshapeReshape5WheatClassifier_CNN_5/dense_2/Tensordot/transpose:y:06WheatClassifier_CNN_5/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€21
/WheatClassifier_CNN_5/dense_2/Tensordot/ReshapeЦ
.WheatClassifier_CNN_5/dense_2/Tensordot/MatMulMatMul8WheatClassifier_CNN_5/dense_2/Tensordot/Reshape:output:0>WheatClassifier_CNN_5/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@20
.WheatClassifier_CNN_5/dense_2/Tensordot/MatMulђ
/WheatClassifier_CNN_5/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@21
/WheatClassifier_CNN_5/dense_2/Tensordot/Const_2∞
5WheatClassifier_CNN_5/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_CNN_5/dense_2/Tensordot/concat_1/axis”
0WheatClassifier_CNN_5/dense_2/Tensordot/concat_1ConcatV29WheatClassifier_CNN_5/dense_2/Tensordot/GatherV2:output:08WheatClassifier_CNN_5/dense_2/Tensordot/Const_2:output:0>WheatClassifier_CNN_5/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_CNN_5/dense_2/Tensordot/concat_1И
'WheatClassifier_CNN_5/dense_2/TensordotReshape8WheatClassifier_CNN_5/dense_2/Tensordot/MatMul:product:09WheatClassifier_CNN_5/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2)
'WheatClassifier_CNN_5/dense_2/Tensordotж
4WheatClassifier_CNN_5/dense_2/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_cnn_5_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4WheatClassifier_CNN_5/dense_2/BiasAdd/ReadVariableOp€
%WheatClassifier_CNN_5/dense_2/BiasAddBiasAdd0WheatClassifier_CNN_5/dense_2/Tensordot:output:0<WheatClassifier_CNN_5/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%WheatClassifier_CNN_5/dense_2/BiasAddЩ
(WheatClassifier_CNN_5/dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_CNN_5/dense_2/Gelu/mul/xр
&WheatClassifier_CNN_5/dense_2/Gelu/mulMul1WheatClassifier_CNN_5/dense_2/Gelu/mul/x:output:0.WheatClassifier_CNN_5/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2(
&WheatClassifier_CNN_5/dense_2/Gelu/mulЫ
)WheatClassifier_CNN_5/dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2+
)WheatClassifier_CNN_5/dense_2/Gelu/Cast/xэ
*WheatClassifier_CNN_5/dense_2/Gelu/truedivRealDiv.WheatClassifier_CNN_5/dense_2/BiasAdd:output:02WheatClassifier_CNN_5/dense_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2,
*WheatClassifier_CNN_5/dense_2/Gelu/truedivљ
&WheatClassifier_CNN_5/dense_2/Gelu/ErfErf.WheatClassifier_CNN_5/dense_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2(
&WheatClassifier_CNN_5/dense_2/Gelu/ErfЩ
(WheatClassifier_CNN_5/dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(WheatClassifier_CNN_5/dense_2/Gelu/add/xо
&WheatClassifier_CNN_5/dense_2/Gelu/addAddV21WheatClassifier_CNN_5/dense_2/Gelu/add/x:output:0*WheatClassifier_CNN_5/dense_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2(
&WheatClassifier_CNN_5/dense_2/Gelu/addй
(WheatClassifier_CNN_5/dense_2/Gelu/mul_1Mul*WheatClassifier_CNN_5/dense_2/Gelu/mul:z:0*WheatClassifier_CNN_5/dense_2/Gelu/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2*
(WheatClassifier_CNN_5/dense_2/Gelu/mul_1“
WheatClassifier_CNN_5/add_1/addAddV2,WheatClassifier_CNN_5/dense_2/Gelu/mul_1:z:0!WheatClassifier_CNN_5/add/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2!
WheatClassifier_CNN_5/add_1/addв
JWheatClassifier_CNN_5/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_CNN_5/layer_normalization_2/moments/mean/reduction_indicesљ
8WheatClassifier_CNN_5/layer_normalization_2/moments/meanMean#WheatClassifier_CNN_5/add_1/add:z:0SWheatClassifier_CNN_5/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2:
8WheatClassifier_CNN_5/layer_normalization_2/moments/meanН
@WheatClassifier_CNN_5/layer_normalization_2/moments/StopGradientStopGradientAWheatClassifier_CNN_5/layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2B
@WheatClassifier_CNN_5/layer_normalization_2/moments/StopGradient…
EWheatClassifier_CNN_5/layer_normalization_2/moments/SquaredDifferenceSquaredDifference#WheatClassifier_CNN_5/add_1/add:z:0IWheatClassifier_CNN_5/layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2G
EWheatClassifier_CNN_5/layer_normalization_2/moments/SquaredDifferenceк
NWheatClassifier_CNN_5/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_CNN_5/layer_normalization_2/moments/variance/reduction_indicesп
<WheatClassifier_CNN_5/layer_normalization_2/moments/varianceMeanIWheatClassifier_CNN_5/layer_normalization_2/moments/SquaredDifference:z:0WWheatClassifier_CNN_5/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2>
<WheatClassifier_CNN_5/layer_normalization_2/moments/varianceњ
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52=
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/add/y¬
9WheatClassifier_CNN_5/layer_normalization_2/batchnorm/addAddV2EWheatClassifier_CNN_5/layer_normalization_2/moments/variance:output:0DWheatClassifier_CNN_5/layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2;
9WheatClassifier_CNN_5/layer_normalization_2/batchnorm/addш
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/RsqrtRsqrt=WheatClassifier_CNN_5/layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2=
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/RsqrtҐ
HWheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_cnn_5_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul/ReadVariableOp∆
9WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mulMul?WheatClassifier_CNN_5/layer_normalization_2/batchnorm/Rsqrt:y:0PWheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2;
9WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mulЫ
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul_1Mul#WheatClassifier_CNN_5/add_1/add:z:0=WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2=
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul_1є
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul_2MulAWheatClassifier_CNN_5/layer_normalization_2/moments/mean:output:0=WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2=
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul_2Ц
DWheatClassifier_CNN_5/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_cnn_5_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_CNN_5/layer_normalization_2/batchnorm/ReadVariableOp¬
9WheatClassifier_CNN_5/layer_normalization_2/batchnorm/subSubLWheatClassifier_CNN_5/layer_normalization_2/batchnorm/ReadVariableOp:value:0?WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2;
9WheatClassifier_CNN_5/layer_normalization_2/batchnorm/subє
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/add_1AddV2?WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul_1:z:0=WheatClassifier_CNN_5/layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2=
;WheatClassifier_CNN_5/layer_normalization_2/batchnorm/add_1І
)WheatClassifier_CNN_5/flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2+
)WheatClassifier_CNN_5/flatten_layer/ConstН
+WheatClassifier_CNN_5/flatten_layer/ReshapeReshape?WheatClassifier_CNN_5/layer_normalization_2/batchnorm/add_1:z:02WheatClassifier_CNN_5/flatten_layer/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(2-
+WheatClassifier_CNN_5/flatten_layer/Reshapeя
0WheatClassifier_CNN_5/FC_1/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_5_fc_1_matmul_readvariableop_resource*
_output_shapes
:	ј(2*
dtype022
0WheatClassifier_CNN_5/FC_1/MatMul/ReadVariableOpт
!WheatClassifier_CNN_5/FC_1/MatMulMatMul4WheatClassifier_CNN_5/flatten_layer/Reshape:output:08WheatClassifier_CNN_5/FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22#
!WheatClassifier_CNN_5/FC_1/MatMulЁ
1WheatClassifier_CNN_5/FC_1/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_5_fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_5/FC_1/BiasAdd/ReadVariableOpн
"WheatClassifier_CNN_5/FC_1/BiasAddBiasAdd+WheatClassifier_CNN_5/FC_1/MatMul:product:09WheatClassifier_CNN_5/FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22$
"WheatClassifier_CNN_5/FC_1/BiasAddѕ
,WheatClassifier_CNN_5/leaky_ReLu_1/LeakyRelu	LeakyRelu+WheatClassifier_CNN_5/FC_1/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2.
,WheatClassifier_CNN_5/leaky_ReLu_1/LeakyReluё
0WheatClassifier_CNN_5/FC_2/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_5_fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype022
0WheatClassifier_CNN_5/FC_2/MatMul/ReadVariableOpш
!WheatClassifier_CNN_5/FC_2/MatMulMatMul:WheatClassifier_CNN_5/leaky_ReLu_1/LeakyRelu:activations:08WheatClassifier_CNN_5/FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22#
!WheatClassifier_CNN_5/FC_2/MatMulЁ
1WheatClassifier_CNN_5/FC_2/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_5_fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_5/FC_2/BiasAdd/ReadVariableOpн
"WheatClassifier_CNN_5/FC_2/BiasAddBiasAdd+WheatClassifier_CNN_5/FC_2/MatMul:product:09WheatClassifier_CNN_5/FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22$
"WheatClassifier_CNN_5/FC_2/BiasAddѕ
,WheatClassifier_CNN_5/leaky_ReLu_2/LeakyRelu	LeakyRelu+WheatClassifier_CNN_5/FC_2/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2.
,WheatClassifier_CNN_5/leaky_ReLu_2/LeakyReluц
8WheatClassifier_CNN_5/output_layer/MatMul/ReadVariableOpReadVariableOpAwheatclassifier_cnn_5_output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02:
8WheatClassifier_CNN_5/output_layer/MatMul/ReadVariableOpР
)WheatClassifier_CNN_5/output_layer/MatMulMatMul:WheatClassifier_CNN_5/leaky_ReLu_2/LeakyRelu:activations:0@WheatClassifier_CNN_5/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2+
)WheatClassifier_CNN_5/output_layer/MatMulх
9WheatClassifier_CNN_5/output_layer/BiasAdd/ReadVariableOpReadVariableOpBwheatclassifier_cnn_5_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9WheatClassifier_CNN_5/output_layer/BiasAdd/ReadVariableOpН
*WheatClassifier_CNN_5/output_layer/BiasAddBiasAdd3WheatClassifier_CNN_5/output_layer/MatMul:product:0AWheatClassifier_CNN_5/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2,
*WheatClassifier_CNN_5/output_layer/BiasAdd 
*WheatClassifier_CNN_5/output_layer/SoftmaxSoftmax3WheatClassifier_CNN_5/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2,
*WheatClassifier_CNN_5/output_layer/Softmaxи
IdentityIdentity4WheatClassifier_CNN_5/output_layer/Softmax:softmax:02^WheatClassifier_CNN_5/FC_1/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_5/FC_1/MatMul/ReadVariableOp2^WheatClassifier_CNN_5/FC_2/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_5/FC_2/MatMul/ReadVariableOp4^WheatClassifier_CNN_5/conv_1/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_5/conv_1/Conv2D/ReadVariableOp4^WheatClassifier_CNN_5/conv_2/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_5/conv_2/Conv2D/ReadVariableOp5^WheatClassifier_CNN_5/dense_1/BiasAdd/ReadVariableOp7^WheatClassifier_CNN_5/dense_1/Tensordot/ReadVariableOp5^WheatClassifier_CNN_5/dense_2/BiasAdd/ReadVariableOp7^WheatClassifier_CNN_5/dense_2/Tensordot/ReadVariableOpC^WheatClassifier_CNN_5/layer_normalization/batchnorm/ReadVariableOpG^WheatClassifier_CNN_5/layer_normalization/batchnorm/mul/ReadVariableOpE^WheatClassifier_CNN_5/layer_normalization_1/batchnorm/ReadVariableOpI^WheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul/ReadVariableOpE^WheatClassifier_CNN_5/layer_normalization_2/batchnorm/ReadVariableOpI^WheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul/ReadVariableOpO^WheatClassifier_CNN_5/multi_head_attention/attention_output/add/ReadVariableOpY^WheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpB^WheatClassifier_CNN_5/multi_head_attention/key/add/ReadVariableOpL^WheatClassifier_CNN_5/multi_head_attention/key/einsum/Einsum/ReadVariableOpD^WheatClassifier_CNN_5/multi_head_attention/query/add/ReadVariableOpN^WheatClassifier_CNN_5/multi_head_attention/query/einsum/Einsum/ReadVariableOpD^WheatClassifier_CNN_5/multi_head_attention/value/add/ReadVariableOpN^WheatClassifier_CNN_5/multi_head_attention/value/einsum/Einsum/ReadVariableOp:^WheatClassifier_CNN_5/output_layer/BiasAdd/ReadVariableOp9^WheatClassifier_CNN_5/output_layer/MatMul/ReadVariableOpA^WheatClassifier_CNN_5/patch_encoder/dense/BiasAdd/ReadVariableOpC^WheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ReadVariableOp?^WheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1WheatClassifier_CNN_5/FC_1/BiasAdd/ReadVariableOp1WheatClassifier_CNN_5/FC_1/BiasAdd/ReadVariableOp2d
0WheatClassifier_CNN_5/FC_1/MatMul/ReadVariableOp0WheatClassifier_CNN_5/FC_1/MatMul/ReadVariableOp2f
1WheatClassifier_CNN_5/FC_2/BiasAdd/ReadVariableOp1WheatClassifier_CNN_5/FC_2/BiasAdd/ReadVariableOp2d
0WheatClassifier_CNN_5/FC_2/MatMul/ReadVariableOp0WheatClassifier_CNN_5/FC_2/MatMul/ReadVariableOp2j
3WheatClassifier_CNN_5/conv_1/BiasAdd/ReadVariableOp3WheatClassifier_CNN_5/conv_1/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_5/conv_1/Conv2D/ReadVariableOp2WheatClassifier_CNN_5/conv_1/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_5/conv_2/BiasAdd/ReadVariableOp3WheatClassifier_CNN_5/conv_2/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_5/conv_2/Conv2D/ReadVariableOp2WheatClassifier_CNN_5/conv_2/Conv2D/ReadVariableOp2l
4WheatClassifier_CNN_5/dense_1/BiasAdd/ReadVariableOp4WheatClassifier_CNN_5/dense_1/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN_5/dense_1/Tensordot/ReadVariableOp6WheatClassifier_CNN_5/dense_1/Tensordot/ReadVariableOp2l
4WheatClassifier_CNN_5/dense_2/BiasAdd/ReadVariableOp4WheatClassifier_CNN_5/dense_2/BiasAdd/ReadVariableOp2p
6WheatClassifier_CNN_5/dense_2/Tensordot/ReadVariableOp6WheatClassifier_CNN_5/dense_2/Tensordot/ReadVariableOp2И
BWheatClassifier_CNN_5/layer_normalization/batchnorm/ReadVariableOpBWheatClassifier_CNN_5/layer_normalization/batchnorm/ReadVariableOp2Р
FWheatClassifier_CNN_5/layer_normalization/batchnorm/mul/ReadVariableOpFWheatClassifier_CNN_5/layer_normalization/batchnorm/mul/ReadVariableOp2М
DWheatClassifier_CNN_5/layer_normalization_1/batchnorm/ReadVariableOpDWheatClassifier_CNN_5/layer_normalization_1/batchnorm/ReadVariableOp2Ф
HWheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul/ReadVariableOpHWheatClassifier_CNN_5/layer_normalization_1/batchnorm/mul/ReadVariableOp2М
DWheatClassifier_CNN_5/layer_normalization_2/batchnorm/ReadVariableOpDWheatClassifier_CNN_5/layer_normalization_2/batchnorm/ReadVariableOp2Ф
HWheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul/ReadVariableOpHWheatClassifier_CNN_5/layer_normalization_2/batchnorm/mul/ReadVariableOp2†
NWheatClassifier_CNN_5/multi_head_attention/attention_output/add/ReadVariableOpNWheatClassifier_CNN_5/multi_head_attention/attention_output/add/ReadVariableOp2і
XWheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpXWheatClassifier_CNN_5/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Ж
AWheatClassifier_CNN_5/multi_head_attention/key/add/ReadVariableOpAWheatClassifier_CNN_5/multi_head_attention/key/add/ReadVariableOp2Ъ
KWheatClassifier_CNN_5/multi_head_attention/key/einsum/Einsum/ReadVariableOpKWheatClassifier_CNN_5/multi_head_attention/key/einsum/Einsum/ReadVariableOp2К
CWheatClassifier_CNN_5/multi_head_attention/query/add/ReadVariableOpCWheatClassifier_CNN_5/multi_head_attention/query/add/ReadVariableOp2Ю
MWheatClassifier_CNN_5/multi_head_attention/query/einsum/Einsum/ReadVariableOpMWheatClassifier_CNN_5/multi_head_attention/query/einsum/Einsum/ReadVariableOp2К
CWheatClassifier_CNN_5/multi_head_attention/value/add/ReadVariableOpCWheatClassifier_CNN_5/multi_head_attention/value/add/ReadVariableOp2Ю
MWheatClassifier_CNN_5/multi_head_attention/value/einsum/Einsum/ReadVariableOpMWheatClassifier_CNN_5/multi_head_attention/value/einsum/Einsum/ReadVariableOp2v
9WheatClassifier_CNN_5/output_layer/BiasAdd/ReadVariableOp9WheatClassifier_CNN_5/output_layer/BiasAdd/ReadVariableOp2t
8WheatClassifier_CNN_5/output_layer/MatMul/ReadVariableOp8WheatClassifier_CNN_5/output_layer/MatMul/ReadVariableOp2Д
@WheatClassifier_CNN_5/patch_encoder/dense/BiasAdd/ReadVariableOp@WheatClassifier_CNN_5/patch_encoder/dense/BiasAdd/ReadVariableOp2И
BWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ReadVariableOpBWheatClassifier_CNN_5/patch_encoder/dense/Tensordot/ReadVariableOp2А
>WheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup>WheatClassifier_CNN_5/patch_encoder/embedding/embedding_lookup:^ Z
1
_output_shapes
:€€€€€€€€€»»
%
_user_specified_nameinput_layer
 
Ь
'__inference_conv_2_layer_call_fn_392475

inputs!
unknown:


	unknown_0:

identityИҐStatefulPartitionedCall€
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ƒƒ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3905972
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€∆∆
: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€∆∆

 
_user_specified_nameinputs
£
_
C__inference_patches_layer_call_and_return_conditional_losses_390618

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
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice‘
ExtractImagePatchesExtractImagePatchesimages*
T0*0
_output_shapes
:€€€€€€€€€		и*
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
€€€€€€€€€2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :и2
Reshape/shape/2†
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeФ
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€bb
:W S
/
_output_shapes
:€€€€€€€€€bb

 
_user_specified_nameimages
є

ы
B__inference_conv_1_layer_call_and_return_conditional_losses_390581

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€∆∆
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€»»: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
є

ы
B__inference_conv_2_layer_call_and_return_conditional_losses_390597

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€∆∆
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€∆∆

 
_user_specified_nameinputs
“_
Р
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_391654
input_layer'
conv_1_391572:

conv_1_391574:
'
conv_2_391577:


conv_2_391579:
'
patch_encoder_391584:	и@"
patch_encoder_391586:@&
patch_encoder_391588:Q@(
layer_normalization_391591:@(
layer_normalization_391593:@1
multi_head_attention_391596:@@-
multi_head_attention_391598:@1
multi_head_attention_391600:@@-
multi_head_attention_391602:@1
multi_head_attention_391604:@@-
multi_head_attention_391606:@1
multi_head_attention_391608:@@)
multi_head_attention_391610:@*
layer_normalization_1_391614:@*
layer_normalization_1_391616:@!
dense_1_391619:	@А
dense_1_391621:	А!
dense_2_391624:	А@
dense_2_391626:@*
layer_normalization_2_391630:@*
layer_normalization_2_391632:@
fc_1_391636:	ј(2
fc_1_391638:2
fc_2_391642:22
fc_2_391644:2%
output_layer_391648:2!
output_layer_391650:
identityИҐFC_1/StatefulPartitionedCallҐFC_2/StatefulPartitionedCallҐconv_1/StatefulPartitionedCallҐconv_2/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ+layer_normalization/StatefulPartitionedCallҐ-layer_normalization_1/StatefulPartitionedCallҐ-layer_normalization_2/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallҐ%patch_encoder/StatefulPartitionedCallЬ
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_391572conv_1_391574*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€∆∆
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3905812 
conv_1/StatefulPartitionedCallЄ
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_391577conv_2_391579*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ƒƒ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3905972 
conv_2/StatefulPartitionedCallГ
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€bb
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_3905582
maxpool_1/PartitionedCallю
patches/PartitionedCallPartitionedCall"maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_3906182
patches/PartitionedCallж
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_391584patch_encoder_391586patch_encoder_391588*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3906602'
%patch_encoder/StatefulPartitionedCallъ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_391591layer_normalization_391593*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3906902-
+layer_normalization/StatefulPartitionedCallц
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_391596multi_head_attention_391598multi_head_attention_391600multi_head_attention_391602multi_head_attention_391604multi_head_attention_391606multi_head_attention_391608multi_head_attention_391610*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3907312.
,multi_head_attention/StatefulPartitionedCallђ
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3907552
add/PartitionedCallт
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_391614layer_normalization_1_391616*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3907792/
-layer_normalization_1/StatefulPartitionedCall«
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_391619dense_1_391621*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€QА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3908232!
dense_1/StatefulPartitionedCallЄ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_391624dense_2_391626*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3908672!
dense_2/StatefulPartitionedCallУ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3908792
add_1/PartitionedCallф
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_391630layer_normalization_2_391632*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3909032/
-layer_normalization_2/StatefulPartitionedCallЧ
flatten_layer/PartitionedCallPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3909152
flatten_layer/PartitionedCall£
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_391636fc_1_391638*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3909272
FC_1/StatefulPartitionedCallВ
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3909382
leaky_ReLu_1/PartitionedCallҐ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_391642fc_2_391644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3909502
FC_2/StatefulPartitionedCallВ
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3909612
leaky_ReLu_2/PartitionedCall 
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_391648output_layer_391650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3909742&
$output_layer/StatefulPartitionedCall—
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
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
:€€€€€€€€€»»
%
_user_specified_nameinput_layer
ђЌ
†0
__inference__traced_save_393298
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop:
6savev2_layer_normalization_2_gamma_read_readvariableop9
5savev2_layer_normalization_2_beta_read_readvariableop*
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
"savev2_count_1_read_readvariableop4
0savev2_adamw_conv_1_kernel_m_read_readvariableop2
.savev2_adamw_conv_1_bias_m_read_readvariableop4
0savev2_adamw_conv_2_kernel_m_read_readvariableop2
.savev2_adamw_conv_2_bias_m_read_readvariableop@
<savev2_adamw_layer_normalization_gamma_m_read_readvariableop?
;savev2_adamw_layer_normalization_beta_m_read_readvariableopB
>savev2_adamw_layer_normalization_1_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_1_beta_m_read_readvariableop5
1savev2_adamw_dense_1_kernel_m_read_readvariableop3
/savev2_adamw_dense_1_bias_m_read_readvariableop5
1savev2_adamw_dense_2_kernel_m_read_readvariableop3
/savev2_adamw_dense_2_bias_m_read_readvariableopB
>savev2_adamw_layer_normalization_2_gamma_m_read_readvariableopA
=savev2_adamw_layer_normalization_2_beta_m_read_readvariableop2
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
Msavev2_adamw_multi_head_attention_attention_output_bias_m_read_readvariableop4
0savev2_adamw_conv_1_kernel_v_read_readvariableop2
.savev2_adamw_conv_1_bias_v_read_readvariableop4
0savev2_adamw_conv_2_kernel_v_read_readvariableop2
.savev2_adamw_conv_2_bias_v_read_readvariableop@
<savev2_adamw_layer_normalization_gamma_v_read_readvariableop?
;savev2_adamw_layer_normalization_beta_v_read_readvariableopB
>savev2_adamw_layer_normalization_1_gamma_v_read_readvariableopA
=savev2_adamw_layer_normalization_1_beta_v_read_readvariableop5
1savev2_adamw_dense_1_kernel_v_read_readvariableop3
/savev2_adamw_dense_1_bias_v_read_readvariableop5
1savev2_adamw_dense_2_kernel_v_read_readvariableop3
/savev2_adamw_dense_2_bias_v_read_readvariableopB
>savev2_adamw_layer_normalization_2_gamma_v_read_readvariableopA
=savev2_adamw_layer_normalization_2_beta_v_read_readvariableop2
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

identity_1ИҐMergeV2CheckpointsП
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
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename«9
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*ў8
valueѕ8Bћ8hB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesџ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*е
valueџBЎhB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesє.
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_2_kernel_read_readvariableop$savev2_fc_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop%savev2_adamw_iter_read_readvariableop'savev2_adamw_beta_1_read_readvariableop'savev2_adamw_beta_2_read_readvariableop&savev2_adamw_decay_read_readvariableop.savev2_adamw_learning_rate_read_readvariableop-savev2_adamw_weight_decay_read_readvariableop5savev2_patch_encoder_dense_kernel_read_readvariableop3savev2_patch_encoder_dense_bias_read_readvariableop=savev2_patch_encoder_embedding_embeddings_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adamw_conv_1_kernel_m_read_readvariableop.savev2_adamw_conv_1_bias_m_read_readvariableop0savev2_adamw_conv_2_kernel_m_read_readvariableop.savev2_adamw_conv_2_bias_m_read_readvariableop<savev2_adamw_layer_normalization_gamma_m_read_readvariableop;savev2_adamw_layer_normalization_beta_m_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_1_beta_m_read_readvariableop1savev2_adamw_dense_1_kernel_m_read_readvariableop/savev2_adamw_dense_1_bias_m_read_readvariableop1savev2_adamw_dense_2_kernel_m_read_readvariableop/savev2_adamw_dense_2_bias_m_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_2_beta_m_read_readvariableop.savev2_adamw_fc_1_kernel_m_read_readvariableop,savev2_adamw_fc_1_bias_m_read_readvariableop.savev2_adamw_fc_2_kernel_m_read_readvariableop,savev2_adamw_fc_2_bias_m_read_readvariableop6savev2_adamw_output_layer_kernel_m_read_readvariableop4savev2_adamw_output_layer_bias_m_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_m_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_m_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_m_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_m_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_m_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_m_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_m_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_m_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_m_read_readvariableop0savev2_adamw_conv_1_kernel_v_read_readvariableop.savev2_adamw_conv_1_bias_v_read_readvariableop0savev2_adamw_conv_2_kernel_v_read_readvariableop.savev2_adamw_conv_2_bias_v_read_readvariableop<savev2_adamw_layer_normalization_gamma_v_read_readvariableop;savev2_adamw_layer_normalization_beta_v_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_1_beta_v_read_readvariableop1savev2_adamw_dense_1_kernel_v_read_readvariableop/savev2_adamw_dense_1_bias_v_read_readvariableop1savev2_adamw_dense_2_kernel_v_read_readvariableop/savev2_adamw_dense_2_bias_v_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_2_beta_v_read_readvariableop.savev2_adamw_fc_1_kernel_v_read_readvariableop,savev2_adamw_fc_1_bias_v_read_readvariableop.savev2_adamw_fc_2_kernel_v_read_readvariableop,savev2_adamw_fc_2_bias_v_read_readvariableop6savev2_adamw_output_layer_kernel_v_read_readvariableop4savev2_adamw_output_layer_bias_v_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_v_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_v_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_v_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_v_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_v_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_v_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_v_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_v_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *v
dtypesl
j2h	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*К
_input_shapesш
х: :
:
:

:
:@:@:@:@:	@А:А:	А@:@:@:@:	ј(2:2:22:2:2:: : : : : : :	и@:@:Q@:@@:@:@@:@:@@:@:@@:@: : : : :
:
:

:
:@:@:@:@:	@А:А:	А@:@:@:@:	ј(2:2:22:2:2::	и@:@:Q@:@@:@:@@:@:@@:@:@@:@:
:
:

:
:@:@:@:@:	@А:А:	А@:@:@:@:	ј(2:2:22:2:2::	и@:@:Q@:@@:@:@@:@:@@:@:@@:@: 2(
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
: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%	!

_output_shapes
:	@А:!


_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	ј(2: 
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
:	и@: 

_output_shapes
:@:$ 

_output_shapes

:Q@:($
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
: :,*(
&
_output_shapes
:
: +

_output_shapes
:
:,,(
&
_output_shapes
:

: -

_output_shapes
:
: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@: 1

_output_shapes
:@:%2!

_output_shapes
:	@А:!3

_output_shapes	
:А:%4!

_output_shapes
:	А@: 5

_output_shapes
:@: 6

_output_shapes
:@: 7

_output_shapes
:@:%8!

_output_shapes
:	ј(2: 9
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
:	и@: ?

_output_shapes
:@:$@ 

_output_shapes

:Q@:(A$
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
:@:,I(
&
_output_shapes
:
: J

_output_shapes
:
:,K(
&
_output_shapes
:

: L

_output_shapes
:
: M
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
:@:%Q!

_output_shapes
:	@А:!R

_output_shapes	
:А:%S!

_output_shapes
:	А@: T

_output_shapes
:@: U

_output_shapes
:@: V

_output_shapes
:@:%W!

_output_shapes
:	ј(2: X
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
:	и@: ^

_output_shapes
:@:$_ 

_output_shapes

:Q@:(`$
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
ћ	
с
@__inference_FC_2_layer_call_and_return_conditional_losses_390950

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
ы
d
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_392941

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€2:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
б
k
A__inference_add_1_layer_call_and_return_conditional_losses_390879

inputs
inputs_1
identity[
addAddV2inputsinputs_1*
T0*+
_output_shapes
:€€€€€€€€€Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€Q@:€€€€€€€€€Q@:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs:SO
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
Л'
ь
C__inference_dense_1_layer_call_and_return_conditional_losses_390823

inputs4
!tensordot_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	@А*
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackР
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЯ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€QА2	
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
:€€€€€€€€€QА2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xЖ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xw
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€QА2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2

Gelu/mul_1Ы
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€QА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
Є

щ
H__inference_output_layer_layer_call_and_return_conditional_losses_390974

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
”
J
.__inference_flatten_layer_layer_call_fn_392888

inputs
identityЋ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3909152
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€Q@:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
ц/
к
I__inference_patch_encoder_layer_call_and_return_conditional_losses_390660	
patch:
'dense_tensordot_readvariableop_resource:	и@3
%dense_biasadd_readvariableop_resource:@3
!embedding_embedding_lookup_390653:Q@
identityИҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOpҐembedding/embedding_lookup\
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
range©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	и@*
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
dense/Tensordot/ShapeА
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisп
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2Д
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisх
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
dense/Tensordot/ConstШ
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
dense/Tensordot/Const_1†
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
dense/Tensordot/concat/axisќ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat§
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stackЂ
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2
dense/Tensordot/transposeЈ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense/Tensordot/Reshapeґ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense/Tensordot/Const_2А
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisџ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1±
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
dense/TensordotЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp®
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
dense/BiasAddЮ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_390653range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/390653*
_output_shapes

:Q@*
dtype02
embedding/embedding_lookupИ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/390653*
_output_shapes

:Q@2%
#embedding/embedding_lookup/Identity±
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@2'
%embedding/embedding_lookup/Identity_1С
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
addЉ
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€и: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и

_user_specified_namepatch
–	
т
@__inference_FC_1_layer_call_and_return_conditional_losses_390927

inputs1
matmul_readvariableop_resource:	ј(2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј(2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј(
 
_user_specified_nameinputs
–	
т
@__inference_FC_1_layer_call_and_return_conditional_losses_392898

inputs1
matmul_readvariableop_resource:	ј(2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ј(2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј(: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€ј(
 
_user_specified_nameinputs
Щ
У
%__inference_FC_1_layer_call_fn_392907

inputs
unknown:	ј(2
	unknown_0:2
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3909272
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€ј(: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€ј(
 
_user_specified_nameinputs
™
а
$__inference_signature_wrapper_391816
input_layer!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:

	unknown_3:	и@
	unknown_4:@
	unknown_5:Q@
	unknown_6:@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@А

unknown_19:	А

unknown_20:	А@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:	ј(2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identityИҐStatefulPartitionedCallж
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
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_3905522
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:€€€€€€€€€»»
%
_user_specified_nameinput_layer
∞
Ч
(__inference_dense_1_layer_call_fn_392787

inputs
unknown:	@А
	unknown_0:	А
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€QА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3908232
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€QА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
г
e
I__inference_flatten_layer_layer_call_and_return_conditional_losses_392883

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€Q@:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
Ц
Т
%__inference_FC_2_layer_call_fn_392936

inputs
unknown:22
	unknown_0:2
identityИҐStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3909502
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
ц
Є
.__inference_patch_encoder_layer_call_fn_392545	
patch
unknown:	и@
	unknown_0:@
	unknown_1:Q@
identityИҐStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCallpatchunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3906602
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€и: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и

_user_specified_namepatch
°
О
O__inference_layer_normalization_layer_call_and_return_conditional_losses_392567

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЬ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/meanЙ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesњ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yТ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЦ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_1Й
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpТ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/add_1•
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
ћ	
с
@__inference_FC_2_layer_call_and_return_conditional_losses_392927

inputs0
matmul_readvariableop_resource:22-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:22*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
г
e
I__inference_flatten_layer_layer_call_and_return_conditional_losses_390915

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€Q@:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
ы
d
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_390938

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€2:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
х-
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_390731	
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
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpЄ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp«
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЩ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpЅ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpС
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp«
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЩ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
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
:€€€€€€€€€Q@2
Mul†
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
softmax/SoftmaxЕ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
dropout/IdentityЄ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpч
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpЅ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
attention_output/addВ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€Q@:€€€€€€€€€Q@: : : : : : : : 2J
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
:€€€€€€€€€Q@

_user_specified_namequery:RN
+
_output_shapes
:€€€€€€€€€Q@

_user_specified_namevalue
ј
Э
4__inference_layer_normalization_layer_call_fn_392576

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3906902
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
У7
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_392653	
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
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpЄ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp«
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЩ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpЅ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpС
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp«
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЩ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
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
:€€€€€€€€€Q@2
Mul†
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/dropout/Const¶
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape‘
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2 
dropout/dropout/GreaterEqual/yж
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
dropout/dropout/GreaterEqualЯ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€QQ2
dropout/dropout/CastҐ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
dropout/dropout/Mul_1Є
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpч
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpЅ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
attention_output/addВ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€Q@:€€€€€€€€€Q@: : : : : : : : 2J
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
:€€€€€€€€€Q@

_user_specified_namequery:RN
+
_output_shapes
:€€€€€€€€€Q@

_user_specified_namevalue
ƒ
Я
6__inference_layer_normalization_2_layer_call_fn_392877

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3909032
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
џы
й
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_392303

inputs?
%conv_1_conv2d_readvariableop_resource:
4
&conv_1_biasadd_readvariableop_resource:
?
%conv_2_conv2d_readvariableop_resource:

4
&conv_2_biasadd_readvariableop_resource:
H
5patch_encoder_dense_tensordot_readvariableop_resource:	и@A
3patch_encoder_dense_biasadd_readvariableop_resource:@A
/patch_encoder_embedding_embedding_lookup_392112:Q@G
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
)dense_1_tensordot_readvariableop_resource:	@А6
'dense_1_biasadd_readvariableop_resource:	А<
)dense_2_tensordot_readvariableop_resource:	А@5
'dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@6
#fc_1_matmul_readvariableop_resource:	ј(22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identityИҐFC_1/BiasAdd/ReadVariableOpҐFC_1/MatMul/ReadVariableOpҐFC_2/BiasAdd/ReadVariableOpҐFC_2/MatMul/ReadVariableOpҐconv_1/BiasAdd/ReadVariableOpҐconv_1/Conv2D/ReadVariableOpҐconv_2/BiasAdd/ReadVariableOpҐconv_2/Conv2D/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐ dense_1/Tensordot/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐ dense_2/Tensordot/ReadVariableOpҐ,layer_normalization/batchnorm/ReadVariableOpҐ0layer_normalization/batchnorm/mul/ReadVariableOpҐ.layer_normalization_1/batchnorm/ReadVariableOpҐ2layer_normalization_1/batchnorm/mul/ReadVariableOpҐ.layer_normalization_2/batchnorm/ReadVariableOpҐ2layer_normalization_2/batchnorm/mul/ReadVariableOpҐ8multi_head_attention/attention_output/add/ReadVariableOpҐBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpҐ+multi_head_attention/key/add/ReadVariableOpҐ5multi_head_attention/key/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/query/add/ReadVariableOpҐ7multi_head_attention/query/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/value/add/ReadVariableOpҐ7multi_head_attention/value/einsum/Einsum/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOpҐ*patch_encoder/dense/BiasAdd/ReadVariableOpҐ,patch_encoder/dense/Tensordot/ReadVariableOpҐ(patch_encoder/embedding/embedding_lookup™
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOpї
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
*
paddingVALID*
strides
2
conv_1/Conv2D°
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_1/BiasAdd/ReadVariableOp¶
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
2
conv_1/BiasAdd™
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
conv_2/Conv2D/ReadVariableOpћ
conv_2/Conv2DConv2Dconv_1/BiasAdd:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
*
paddingVALID*
strides
2
conv_2/Conv2D°
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_2/BiasAdd/ReadVariableOp¶
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
2
conv_2/BiasAddЈ
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€bb
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPoolh
patches/ShapeShapemaxpool_1/MaxPool:output:0*
T0*
_output_shapes
:2
patches/ShapeД
patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
patches/strided_slice/stackИ
patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_1И
patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_2Т
patches/strided_sliceStridedSlicepatches/Shape:output:0$patches/strided_slice/stack:output:0&patches/strided_slice/stack_1:output:0&patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
patches/strided_sliceш
patches/ExtractImagePatchesExtractImagePatchesmaxpool_1/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€		и*
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
€€€€€€€€€2
patches/Reshape/shape/1u
patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :и2
patches/Reshape/shape/2»
patches/Reshape/shapePackpatches/strided_slice:output:0 patches/Reshape/shape/1:output:0 patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
patches/Reshape/shapeі
patches/ReshapeReshape%patches/ExtractImagePatches:patches:0patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2
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
patch_encoder/range/deltaї
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes
:Q2
patch_encoder/range”
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	и@*
dtype02.
,patch_encoder/dense/Tensordot/ReadVariableOpТ
"patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"patch_encoder/dense/Tensordot/axesЩ
"patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"patch_encoder/dense/Tensordot/freeТ
#patch_encoder/dense/Tensordot/ShapeShapepatches/Reshape:output:0*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/ShapeЬ
+patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/GatherV2/axisµ
&patch_encoder/dense/Tensordot/GatherV2GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/free:output:04patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/GatherV2†
-patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-patch_encoder/dense/Tensordot/GatherV2_1/axisї
(patch_encoder/dense/Tensordot/GatherV2_1GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/axes:output:06patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(patch_encoder/dense/Tensordot/GatherV2_1Ф
#patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#patch_encoder/dense/Tensordot/Const–
"patch_encoder/dense/Tensordot/ProdProd/patch_encoder/dense/Tensordot/GatherV2:output:0,patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"patch_encoder/dense/Tensordot/ProdШ
%patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%patch_encoder/dense/Tensordot/Const_1Ў
$patch_encoder/dense/Tensordot/Prod_1Prod1patch_encoder/dense/Tensordot/GatherV2_1:output:0.patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$patch_encoder/dense/Tensordot/Prod_1Ш
)patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)patch_encoder/dense/Tensordot/concat/axisФ
$patch_encoder/dense/Tensordot/concatConcatV2+patch_encoder/dense/Tensordot/free:output:0+patch_encoder/dense/Tensordot/axes:output:02patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$patch_encoder/dense/Tensordot/concat№
#patch_encoder/dense/Tensordot/stackPack+patch_encoder/dense/Tensordot/Prod:output:0-patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/stackи
'patch_encoder/dense/Tensordot/transpose	Transposepatches/Reshape:output:0-patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2)
'patch_encoder/dense/Tensordot/transposeп
%patch_encoder/dense/Tensordot/ReshapeReshape+patch_encoder/dense/Tensordot/transpose:y:0,patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2'
%patch_encoder/dense/Tensordot/Reshapeо
$patch_encoder/dense/Tensordot/MatMulMatMul.patch_encoder/dense/Tensordot/Reshape:output:04patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2&
$patch_encoder/dense/Tensordot/MatMulШ
%patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2'
%patch_encoder/dense/Tensordot/Const_2Ь
+patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/concat_1/axis°
&patch_encoder/dense/Tensordot/concat_1ConcatV2/patch_encoder/dense/Tensordot/GatherV2:output:0.patch_encoder/dense/Tensordot/Const_2:output:04patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/concat_1й
patch_encoder/dense/TensordotReshape.patch_encoder/dense/Tensordot/MatMul:product:0/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
patch_encoder/dense/Tensordot»
*patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp3patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*patch_encoder/dense/BiasAdd/ReadVariableOpа
patch_encoder/dense/BiasAddBiasAdd&patch_encoder/dense/Tensordot:output:02patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
patch_encoder/dense/BiasAddд
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_392112patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/392112*
_output_shapes

:Q@*
dtype02*
(patch_encoder/embedding/embedding_lookupј
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/392112*
_output_shapes

:Q@23
1patch_encoder/embedding/embedding_lookup/Identityџ
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@25
3patch_encoder/embedding/embedding_lookup/Identity_1…
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
patch_encoder/add≤
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indicesз
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2"
 layer_normalization/moments/mean≈
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2*
(layer_normalization/moments/StopGradientу
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2/
-layer_normalization/moments/SquaredDifferenceЇ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indicesП
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2&
$layer_normalization/moments/varianceП
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52%
#layer_normalization/batchnorm/add/yв
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2#
!layer_normalization/batchnorm/add∞
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2%
#layer_normalization/batchnorm/RsqrtЏ
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOpж
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2#
!layer_normalization/batchnorm/mul≈
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization/batchnorm/mul_1ў
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization/batchnorm/mul_2ќ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOpв
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2#
!layer_normalization/batchnorm/subў
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization/batchnorm/add_1ч
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp®
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum’
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOpн
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2 
multi_head_attention/query/addс
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOpҐ
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsumѕ
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOpе
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
multi_head_attention/key/addч
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp®
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum’
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOpн
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/yЊ
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
multi_head_attention/Mulф
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€QQ*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/EinsumЊ
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2&
$multi_head_attention/softmax/SoftmaxЭ
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2,
*multi_head_attention/dropout/dropout/Constъ
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2*
(multi_head_attention/dropout/dropout/Mulґ
*multi_head_attention/dropout/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2,
*multi_head_attention/dropout/dropout/ShapeУ
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ*
dtype02C
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformѓ
3multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=25
3multi_head_attention/dropout/dropout/GreaterEqual/yЇ
1multi_head_attention/dropout/dropout/GreaterEqualGreaterEqualJmulti_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0<multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ23
1multi_head_attention/dropout/dropout/GreaterEqualё
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€QQ2+
)multi_head_attention/dropout/dropout/Castц
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2,
*multi_head_attention/dropout/dropout/Mul_1М
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/EinsumШ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpЋ
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€Q@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsumт
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOpХ
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2+
)multi_head_attention/attention_output/addЧ
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2	
add/addґ
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indicesг
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2$
"layer_normalization_1/moments/meanЋ
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2,
*layer_normalization_1/moments/StopGradientп
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@21
/layer_normalization_1/moments/SquaredDifferenceЊ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indicesЧ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2(
&layer_normalization_1/moments/varianceУ
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_1/batchnorm/add/yк
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2%
#layer_normalization_1/batchnorm/addґ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2'
%layer_normalization_1/batchnorm/Rsqrtа
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOpо
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization_1/batchnorm/mulЅ
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_1/batchnorm/mul_1б
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_1/batchnorm/mul_2‘
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOpк
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization_1/batchnorm/subб
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_1/batchnorm/add_1ѓ
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@А*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axesБ
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/freeЛ
dense_1/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_1/Tensordot/ShapeД
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
dense_1/Tensordot/GatherV2И
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis€
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
dense_1/Tensordot/Const†
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/ProdА
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1®
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1А
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axisЎ
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concatђ
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stackЋ
dense_1/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_1/Tensordot/transposeњ
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_1/Tensordot/Reshapeњ
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/Tensordot/MatMulБ
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_1/Tensordot/Const_2Д
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axisе
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1±
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Tensordot•
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_1/BiasAdd/ReadVariableOp®
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/xЩ
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_1/Gelu/Cast/x¶
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/truediv|
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_1/Gelu/add/xЧ
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/addТ
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/mul_1ѓ
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes
:	А@*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axesБ
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
dense_2/Tensordot/ShapeД
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
dense_2/Tensordot/GatherV2И
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis€
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
dense_2/Tensordot/Const†
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/ProdА
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1®
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1А
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axisЎ
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concatђ
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stackє
dense_2/Tensordot/transpose	Transposedense_1/Gelu/mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_2/Tensordot/transposeњ
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_2/Tensordot/ReshapeЊ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_2/Tensordot/MatMulА
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_2/Tensordot/Const_2Д
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axisе
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1∞
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Tensordot§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOpІ
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/xШ
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_2/Gelu/Cast/x•
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/truediv{
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_2/Gelu/add/xЦ
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/addС
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/mul_1z
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
	add_1/addґ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesе
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2$
"layer_normalization_2/moments/meanЋ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2,
*layer_normalization_2/moments/StopGradientс
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@21
/layer_normalization_2/moments/SquaredDifferenceЊ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesЧ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2(
&layer_normalization_2/moments/varianceУ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_2/batchnorm/add/yк
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2%
#layer_normalization_2/batchnorm/addґ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2'
%layer_normalization_2/batchnorm/Rsqrtа
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpо
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization_2/batchnorm/mul√
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_2/batchnorm/mul_1б
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_2/batchnorm/mul_2‘
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpк
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization_2/batchnorm/subб
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_2/batchnorm/add_1{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
flatten_layer/Constµ
flatten_layer/ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(2
flatten_layer/ReshapeЭ
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	ј(2*
dtype02
FC_1/MatMul/ReadVariableOpЪ
FC_1/MatMulMatMulflatten_layer/Reshape:output:0"FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
FC_1/MatMulЫ
FC_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_1/BiasAdd/ReadVariableOpХ
FC_1/BiasAddBiasAddFC_1/MatMul:product:0#FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
FC_1/BiasAddН
leaky_ReLu_1/LeakyRelu	LeakyReluFC_1/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2
leaky_ReLu_1/LeakyReluЬ
FC_2/MatMul/ReadVariableOpReadVariableOp#fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
FC_2/MatMul/ReadVariableOp†
FC_2/MatMulMatMul$leaky_ReLu_1/LeakyRelu:activations:0"FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
FC_2/MatMulЫ
FC_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_2/BiasAdd/ReadVariableOpХ
FC_2/BiasAddBiasAddFC_2/MatMul:product:0#FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
FC_2/BiasAddН
leaky_ReLu_2/LeakyRelu	LeakyReluFC_2/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2
leaky_ReLu_2/LeakyReluі
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02$
"output_layer/MatMul/ReadVariableOpЄ
output_layer/MatMulMatMul$leaky_ReLu_2/LeakyRelu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/MatMul≥
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpµ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/BiasAddИ
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/Softmax®
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
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
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2И
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
:€€€€€€€€€»»
 
_user_specified_nameinputs
√_
Л
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_390981

inputs'
conv_1_390582:

conv_1_390584:
'
conv_2_390598:


conv_2_390600:
'
patch_encoder_390661:	и@"
patch_encoder_390663:@&
patch_encoder_390665:Q@(
layer_normalization_390691:@(
layer_normalization_390693:@1
multi_head_attention_390732:@@-
multi_head_attention_390734:@1
multi_head_attention_390736:@@-
multi_head_attention_390738:@1
multi_head_attention_390740:@@-
multi_head_attention_390742:@1
multi_head_attention_390744:@@)
multi_head_attention_390746:@*
layer_normalization_1_390780:@*
layer_normalization_1_390782:@!
dense_1_390824:	@А
dense_1_390826:	А!
dense_2_390868:	А@
dense_2_390870:@*
layer_normalization_2_390904:@*
layer_normalization_2_390906:@
fc_1_390928:	ј(2
fc_1_390930:2
fc_2_390951:22
fc_2_390953:2%
output_layer_390975:2!
output_layer_390977:
identityИҐFC_1/StatefulPartitionedCallҐFC_2/StatefulPartitionedCallҐconv_1/StatefulPartitionedCallҐconv_2/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ+layer_normalization/StatefulPartitionedCallҐ-layer_normalization_1/StatefulPartitionedCallҐ-layer_normalization_2/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallҐ%patch_encoder/StatefulPartitionedCallЧ
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_390582conv_1_390584*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€∆∆
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3905812 
conv_1/StatefulPartitionedCallЄ
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_390598conv_2_390600*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ƒƒ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3905972 
conv_2/StatefulPartitionedCallГ
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€bb
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_3905582
maxpool_1/PartitionedCallю
patches/PartitionedCallPartitionedCall"maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_3906182
patches/PartitionedCallж
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_390661patch_encoder_390663patch_encoder_390665*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3906602'
%patch_encoder/StatefulPartitionedCallъ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_390691layer_normalization_390693*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3906902-
+layer_normalization/StatefulPartitionedCallц
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_390732multi_head_attention_390734multi_head_attention_390736multi_head_attention_390738multi_head_attention_390740multi_head_attention_390742multi_head_attention_390744multi_head_attention_390746*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3907312.
,multi_head_attention/StatefulPartitionedCallђ
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3907552
add/PartitionedCallт
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_390780layer_normalization_1_390782*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3907792/
-layer_normalization_1/StatefulPartitionedCall«
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_390824dense_1_390826*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€QА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3908232!
dense_1/StatefulPartitionedCallЄ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_390868dense_2_390870*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3908672!
dense_2/StatefulPartitionedCallУ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3908792
add_1/PartitionedCallф
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_390904layer_normalization_2_390906*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3909032/
-layer_normalization_2/StatefulPartitionedCallЧ
flatten_layer/PartitionedCallPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3909152
flatten_layer/PartitionedCall£
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_390928fc_1_390930*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3909272
FC_1/StatefulPartitionedCallВ
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3909382
leaky_ReLu_1/PartitionedCallҐ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_390951fc_2_390953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3909502
FC_2/StatefulPartitionedCallВ
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3909612
leaky_ReLu_2/PartitionedCall 
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_390975output_layer_390977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3909742&
$output_layer/StatefulPartitionedCall—
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
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
:€€€€€€€€€»»
 
_user_specified_nameinputs
Є

щ
H__inference_output_layer_layer_call_and_return_conditional_losses_392957

inputs0
matmul_readvariableop_resource:2-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:2*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
Ё
н
6__inference_WheatClassifier_CNN_5_layer_call_fn_392370

inputs!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:

	unknown_3:	и@
	unknown_4:@
	unknown_5:Q@
	unknown_6:@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@А

unknown_19:	А

unknown_20:	А@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:	ј(2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identityИҐStatefulPartitionedCallС
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
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_3909812
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
з
k
?__inference_add_layer_call_and_return_conditional_losses_392703
inputs_0
inputs_1
identity]
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:€€€€€€€€€Q@2
add_
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€Q@:€€€€€€€€€Q@:U Q
+
_output_shapes
:€€€€€€€€€Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€Q@
"
_user_specified_name
inputs/1
й
D
(__inference_patches_layer_call_fn_392494

images
identity“
PartitionedCallPartitionedCallimages*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_3906182
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€bb
:W S
/
_output_shapes
:€€€€€€€€€bb

 
_user_specified_nameimages
ў

к
5__inference_multi_head_attention_layer_call_fn_392697	
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
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3912162
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€Q@:€€€€€€€€€Q@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:€€€€€€€€€Q@

_user_specified_namequery:RN
+
_output_shapes
:€€€€€€€€€Q@

_user_specified_namevalue
ў

к
5__inference_multi_head_attention_layer_call_fn_392675	
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
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3907312
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€Q@:€€€€€€€€€Q@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
+
_output_shapes
:€€€€€€€€€Q@

_user_specified_namequery:RN
+
_output_shapes
:€€€€€€€€€Q@

_user_specified_namevalue
£
_
C__inference_patches_layer_call_and_return_conditional_losses_392489

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
strided_slice/stack_2в
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice‘
ExtractImagePatchesExtractImagePatchesimages*
T0*0
_output_shapes
:€€€€€€€€€		и*
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
€€€€€€€€€2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :и2
Reshape/shape/2†
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeФ
ReshapeReshapeExtractImagePatches:patches:0Reshape/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€bb
:W S
/
_output_shapes
:€€€€€€€€€bb

 
_user_specified_nameimages
ц/
к
I__inference_patch_encoder_layer_call_and_return_conditional_losses_392534	
patch:
'dense_tensordot_readvariableop_resource:	и@3
%dense_biasadd_readvariableop_resource:@3
!embedding_embedding_lookup_392527:Q@
identityИҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOpҐembedding/embedding_lookup\
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
range©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	и@*
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
dense/Tensordot/ShapeА
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisп
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2Д
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisх
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
dense/Tensordot/ConstШ
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
dense/Tensordot/Const_1†
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
dense/Tensordot/concat/axisќ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat§
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stackЂ
dense/Tensordot/transpose	Transposepatchdense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2
dense/Tensordot/transposeЈ
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense/Tensordot/Reshapeґ
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense/Tensordot/Const_2А
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisџ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1±
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
dense/TensordotЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp®
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
dense/BiasAddЮ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_392527range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/392527*
_output_shapes

:Q@*
dtype02
embedding/embedding_lookupИ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/392527*
_output_shapes

:Q@2%
#embedding/embedding_lookup/Identity±
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@2'
%embedding/embedding_lookup/Identity_1С
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
addЉ
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€и: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и

_user_specified_namepatch
У7
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_391216	
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
identityИҐ#attention_output/add/ReadVariableOpҐ-attention_output/einsum/Einsum/ReadVariableOpҐkey/add/ReadVariableOpҐ key/einsum/Einsum/ReadVariableOpҐquery/add/ReadVariableOpҐ"query/einsum/Einsum/ReadVariableOpҐvalue/add/ReadVariableOpҐ"value/einsum/Einsum/ReadVariableOpЄ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"query/einsum/Einsum/ReadVariableOp«
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЩ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOpЅ
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpС
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp«
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЩ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
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
:€€€€€€€€€Q@2
Mul†
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€QQ*
equationaecd,abcd->acbe2
einsum/Einsum
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/dropout/Const¶
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape‘
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2 
dropout/dropout/GreaterEqual/yж
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
dropout/dropout/GreaterEqualЯ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€QQ2
dropout/dropout/CastҐ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2
dropout/dropout/Mul_1Є
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpч
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€Q@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOpЅ
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
attention_output/addВ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:€€€€€€€€€Q@:€€€€€€€€€Q@: : : : : : : : 2J
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
:€€€€€€€€€Q@

_user_specified_namequery:RN
+
_output_shapes
:€€€€€€€€€Q@

_user_specified_namevalue
£
Р
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_392731

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЬ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/meanЙ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesњ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yТ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЦ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_1Й
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpТ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/add_1•
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
м
т
6__inference_WheatClassifier_CNN_5_layer_call_fn_391569
input_layer!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:

	unknown_3:	и@
	unknown_4:@
	unknown_5:Q@
	unknown_6:@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@А

unknown_19:	А

unknown_20:	А@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:	ј(2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identityИҐStatefulPartitionedCallЦ
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
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_3914372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:€€€€€€€€€»»
%
_user_specified_nameinput_layer
“
P
$__inference_add_layer_call_fn_392709
inputs_0
inputs_1
identity—
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3907552
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:€€€€€€€€€Q@:€€€€€€€€€Q@:U Q
+
_output_shapes
:€€€€€€€€€Q@
"
_user_specified_name
inputs/0:UQ
+
_output_shapes
:€€€€€€€€€Q@
"
_user_specified_name
inputs/1
В'
ы
C__inference_dense_2_layer_call_and_return_conditional_losses_390867

inputs4
!tensordot_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	А@*
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
Tensordot/GatherV2/axis—
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
Tensordot/GatherV2_1/axis„
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
Tensordot/ConstА
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
Tensordot/Const_1И
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
Tensordot/concat/axis∞
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concatМ
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
Tensordot/transposeЯ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
Tensordot/ReshapeЮ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
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
Tensordot/concat_1/axisљ
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1Р
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЗ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2	
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
:€€€€€€€€€Q@2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xЕ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
Gelu/truedivc
Gelu/ErfErfGelu/truediv:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xv
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Gelu/addq

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Gelu/mul_1Ъ
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€QА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€QА
 
_user_specified_nameinputs
“_
Р
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_391739
input_layer'
conv_1_391657:

conv_1_391659:
'
conv_2_391662:


conv_2_391664:
'
patch_encoder_391669:	и@"
patch_encoder_391671:@&
patch_encoder_391673:Q@(
layer_normalization_391676:@(
layer_normalization_391678:@1
multi_head_attention_391681:@@-
multi_head_attention_391683:@1
multi_head_attention_391685:@@-
multi_head_attention_391687:@1
multi_head_attention_391689:@@-
multi_head_attention_391691:@1
multi_head_attention_391693:@@)
multi_head_attention_391695:@*
layer_normalization_1_391699:@*
layer_normalization_1_391701:@!
dense_1_391704:	@А
dense_1_391706:	А!
dense_2_391709:	А@
dense_2_391711:@*
layer_normalization_2_391715:@*
layer_normalization_2_391717:@
fc_1_391721:	ј(2
fc_1_391723:2
fc_2_391727:22
fc_2_391729:2%
output_layer_391733:2!
output_layer_391735:
identityИҐFC_1/StatefulPartitionedCallҐFC_2/StatefulPartitionedCallҐconv_1/StatefulPartitionedCallҐconv_2/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ+layer_normalization/StatefulPartitionedCallҐ-layer_normalization_1/StatefulPartitionedCallҐ-layer_normalization_2/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallҐ%patch_encoder/StatefulPartitionedCallЬ
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_391657conv_1_391659*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€∆∆
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3905812 
conv_1/StatefulPartitionedCallЄ
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_391662conv_2_391664*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ƒƒ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3905972 
conv_2/StatefulPartitionedCallГ
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€bb
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_3905582
maxpool_1/PartitionedCallю
patches/PartitionedCallPartitionedCall"maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_3906182
patches/PartitionedCallж
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_391669patch_encoder_391671patch_encoder_391673*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3906602'
%patch_encoder/StatefulPartitionedCallъ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_391676layer_normalization_391678*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3906902-
+layer_normalization/StatefulPartitionedCallц
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_391681multi_head_attention_391683multi_head_attention_391685multi_head_attention_391687multi_head_attention_391689multi_head_attention_391691multi_head_attention_391693multi_head_attention_391695*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3912162.
,multi_head_attention/StatefulPartitionedCallђ
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3907552
add/PartitionedCallт
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_391699layer_normalization_1_391701*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3907792/
-layer_normalization_1/StatefulPartitionedCall«
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_391704dense_1_391706*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€QА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3908232!
dense_1/StatefulPartitionedCallЄ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_391709dense_2_391711*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3908672!
dense_2/StatefulPartitionedCallУ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3908792
add_1/PartitionedCallф
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_391715layer_normalization_2_391717*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3909032/
-layer_normalization_2/StatefulPartitionedCallЧ
flatten_layer/PartitionedCallPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3909152
flatten_layer/PartitionedCall£
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_391721fc_1_391723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3909272
FC_1/StatefulPartitionedCallВ
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3909382
leaky_ReLu_1/PartitionedCallҐ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_391727fc_2_391729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3909502
FC_2/StatefulPartitionedCallВ
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3909612
leaky_ReLu_2/PartitionedCall 
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_391733output_layer_391735*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3909742&
$output_layer/StatefulPartitionedCall—
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
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
:€€€€€€€€€»»
%
_user_specified_nameinput_layer
√_
Л
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_391437

inputs'
conv_1_391355:

conv_1_391357:
'
conv_2_391360:


conv_2_391362:
'
patch_encoder_391367:	и@"
patch_encoder_391369:@&
patch_encoder_391371:Q@(
layer_normalization_391374:@(
layer_normalization_391376:@1
multi_head_attention_391379:@@-
multi_head_attention_391381:@1
multi_head_attention_391383:@@-
multi_head_attention_391385:@1
multi_head_attention_391387:@@-
multi_head_attention_391389:@1
multi_head_attention_391391:@@)
multi_head_attention_391393:@*
layer_normalization_1_391397:@*
layer_normalization_1_391399:@!
dense_1_391402:	@А
dense_1_391404:	А!
dense_2_391407:	А@
dense_2_391409:@*
layer_normalization_2_391413:@*
layer_normalization_2_391415:@
fc_1_391419:	ј(2
fc_1_391421:2
fc_2_391425:22
fc_2_391427:2%
output_layer_391431:2!
output_layer_391433:
identityИҐFC_1/StatefulPartitionedCallҐFC_2/StatefulPartitionedCallҐconv_1/StatefulPartitionedCallҐconv_2/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐ+layer_normalization/StatefulPartitionedCallҐ-layer_normalization_1/StatefulPartitionedCallҐ-layer_normalization_2/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallҐ%patch_encoder/StatefulPartitionedCallЧ
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_391355conv_1_391357*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€∆∆
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_1_layer_call_and_return_conditional_losses_3905812 
conv_1/StatefulPartitionedCallЄ
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_391360conv_2_391362*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€ƒƒ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv_2_layer_call_and_return_conditional_losses_3905972 
conv_2/StatefulPartitionedCallГ
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€bb
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_3905582
maxpool_1/PartitionedCallю
patches/PartitionedCallPartitionedCall"maxpool_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_3906182
patches/PartitionedCallж
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_391367patch_encoder_391369patch_encoder_391371*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_3906602'
%patch_encoder/StatefulPartitionedCallъ
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_391374layer_normalization_391376*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_3906902-
+layer_normalization/StatefulPartitionedCallц
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_391379multi_head_attention_391381multi_head_attention_391383multi_head_attention_391385multi_head_attention_391387multi_head_attention_391389multi_head_attention_391391multi_head_attention_391393*
Tin
2
*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_3912162.
,multi_head_attention/StatefulPartitionedCallђ
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_3907552
add/PartitionedCallт
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_391397layer_normalization_1_391399*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3907792/
-layer_normalization_1/StatefulPartitionedCall«
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_391402dense_1_391404*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€QА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3908232!
dense_1/StatefulPartitionedCallЄ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_391407dense_2_391409*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3908672!
dense_2/StatefulPartitionedCallУ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_3908792
add_1/PartitionedCallф
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_391413layer_normalization_2_391415*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_3909032/
-layer_normalization_2/StatefulPartitionedCallЧ
flatten_layer/PartitionedCallPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ј(* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_flatten_layer_layer_call_and_return_conditional_losses_3909152
flatten_layer/PartitionedCall£
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0fc_1_391419fc_1_391421*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_1_layer_call_and_return_conditional_losses_3909272
FC_1/StatefulPartitionedCallВ
leaky_ReLu_1/PartitionedCallPartitionedCall%FC_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3909382
leaky_ReLu_1/PartitionedCallҐ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0fc_2_391425fc_2_391427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_FC_2_layer_call_and_return_conditional_losses_3909502
FC_2/StatefulPartitionedCallВ
leaky_ReLu_2/PartitionedCallPartitionedCall%FC_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3909612
leaky_ReLu_2/PartitionedCall 
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_391431output_layer_391433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_output_layer_layer_call_and_return_conditional_losses_3909742&
$output_layer/StatefulPartitionedCall—
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
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
:€€€€€€€€€»»
 
_user_specified_nameinputs
ѓ
Ц
(__inference_dense_2_layer_call_fn_392834

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_3908672
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€QА: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€QА
 
_user_specified_nameinputs
ƒ
Я
6__inference_layer_normalization_1_layer_call_fn_392740

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€Q@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_3907792
StatefulPartitionedCallТ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
«
I
-__inference_leaky_ReLu_1_layer_call_fn_392917

inputs
identity…
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_3909382
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€2:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
«
I
-__inference_leaky_ReLu_2_layer_call_fn_392946

inputs
identity…
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_3909612
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€2:O K
'
_output_shapes
:€€€€€€€€€2
 
_user_specified_nameinputs
£
Р
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_390779

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЬ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/meanЙ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesњ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yТ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЦ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_1Й
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpТ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/add_1•
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
ќ
F
*__inference_maxpool_1_layer_call_fn_390564

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_maxpool_1_layer_call_and_return_conditional_losses_3905582
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м
т
6__inference_WheatClassifier_CNN_5_layer_call_fn_391046
input_layer!
unknown:

	unknown_0:
#
	unknown_1:


	unknown_2:

	unknown_3:	и@
	unknown_4:@
	unknown_5:Q@
	unknown_6:@
	unknown_7:@
	unknown_8:@@
	unknown_9:@ 

unknown_10:@@

unknown_11:@ 

unknown_12:@@

unknown_13:@ 

unknown_14:@@

unknown_15:@

unknown_16:@

unknown_17:@

unknown_18:	@А

unknown_19:	А

unknown_20:	А@

unknown_21:@

unknown_22:@

unknown_23:@

unknown_24:	ј(2

unknown_25:2

unknown_26:22

unknown_27:2

unknown_28:2

unknown_29:
identityИҐStatefulPartitionedCallЦ
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
:€€€€€€€€€*A
_read_only_resource_inputs#
!	
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_3909812
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
1
_output_shapes
:€€€€€€€€€»»
%
_user_specified_nameinput_layer
є

ы
B__inference_conv_2_layer_call_and_return_conditional_losses_392466

inputs8
conv2d_readvariableop_resource:

-
biasadd_readvariableop_resource:

identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€∆∆
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€∆∆

 
_user_specified_nameinputs
но
й
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_392056

inputs?
%conv_1_conv2d_readvariableop_resource:
4
&conv_1_biasadd_readvariableop_resource:
?
%conv_2_conv2d_readvariableop_resource:

4
&conv_2_biasadd_readvariableop_resource:
H
5patch_encoder_dense_tensordot_readvariableop_resource:	и@A
3patch_encoder_dense_biasadd_readvariableop_resource:@A
/patch_encoder_embedding_embedding_lookup_391872:Q@G
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
)dense_1_tensordot_readvariableop_resource:	@А6
'dense_1_biasadd_readvariableop_resource:	А<
)dense_2_tensordot_readvariableop_resource:	А@5
'dense_2_biasadd_readvariableop_resource:@I
;layer_normalization_2_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_2_batchnorm_readvariableop_resource:@6
#fc_1_matmul_readvariableop_resource:	ј(22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identityИҐFC_1/BiasAdd/ReadVariableOpҐFC_1/MatMul/ReadVariableOpҐFC_2/BiasAdd/ReadVariableOpҐFC_2/MatMul/ReadVariableOpҐconv_1/BiasAdd/ReadVariableOpҐconv_1/Conv2D/ReadVariableOpҐconv_2/BiasAdd/ReadVariableOpҐconv_2/Conv2D/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐ dense_1/Tensordot/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐ dense_2/Tensordot/ReadVariableOpҐ,layer_normalization/batchnorm/ReadVariableOpҐ0layer_normalization/batchnorm/mul/ReadVariableOpҐ.layer_normalization_1/batchnorm/ReadVariableOpҐ2layer_normalization_1/batchnorm/mul/ReadVariableOpҐ.layer_normalization_2/batchnorm/ReadVariableOpҐ2layer_normalization_2/batchnorm/mul/ReadVariableOpҐ8multi_head_attention/attention_output/add/ReadVariableOpҐBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpҐ+multi_head_attention/key/add/ReadVariableOpҐ5multi_head_attention/key/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/query/add/ReadVariableOpҐ7multi_head_attention/query/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/value/add/ReadVariableOpҐ7multi_head_attention/value/einsum/Einsum/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOpҐ*patch_encoder/dense/BiasAdd/ReadVariableOpҐ,patch_encoder/dense/Tensordot/ReadVariableOpҐ(patch_encoder/embedding/embedding_lookup™
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOpї
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
*
paddingVALID*
strides
2
conv_1/Conv2D°
conv_1/BiasAdd/ReadVariableOpReadVariableOp&conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_1/BiasAdd/ReadVariableOp¶
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
2
conv_1/BiasAdd™
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
conv_2/Conv2D/ReadVariableOpћ
conv_2/Conv2DConv2Dconv_1/BiasAdd:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
*
paddingVALID*
strides
2
conv_2/Conv2D°
conv_2/BiasAdd/ReadVariableOpReadVariableOp&conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
conv_2/BiasAdd/ReadVariableOp¶
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ƒƒ
2
conv_2/BiasAddЈ
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€bb
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPoolh
patches/ShapeShapemaxpool_1/MaxPool:output:0*
T0*
_output_shapes
:2
patches/ShapeД
patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
patches/strided_slice/stackИ
patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_1И
patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
patches/strided_slice/stack_2Т
patches/strided_sliceStridedSlicepatches/Shape:output:0$patches/strided_slice/stack:output:0&patches/strided_slice/stack_1:output:0&patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
patches/strided_sliceш
patches/ExtractImagePatchesExtractImagePatchesmaxpool_1/MaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€		и*
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
€€€€€€€€€2
patches/Reshape/shape/1u
patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :и2
patches/Reshape/shape/2»
patches/Reshape/shapePackpatches/strided_slice:output:0 patches/Reshape/shape/1:output:0 patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
patches/Reshape/shapeі
patches/ReshapeReshape%patches/ExtractImagePatches:patches:0patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2
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
patch_encoder/range/deltaї
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes
:Q2
patch_encoder/range”
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	и@*
dtype02.
,patch_encoder/dense/Tensordot/ReadVariableOpТ
"patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"patch_encoder/dense/Tensordot/axesЩ
"patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"patch_encoder/dense/Tensordot/freeТ
#patch_encoder/dense/Tensordot/ShapeShapepatches/Reshape:output:0*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/ShapeЬ
+patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/GatherV2/axisµ
&patch_encoder/dense/Tensordot/GatherV2GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/free:output:04patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/GatherV2†
-patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-patch_encoder/dense/Tensordot/GatherV2_1/axisї
(patch_encoder/dense/Tensordot/GatherV2_1GatherV2,patch_encoder/dense/Tensordot/Shape:output:0+patch_encoder/dense/Tensordot/axes:output:06patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(patch_encoder/dense/Tensordot/GatherV2_1Ф
#patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#patch_encoder/dense/Tensordot/Const–
"patch_encoder/dense/Tensordot/ProdProd/patch_encoder/dense/Tensordot/GatherV2:output:0,patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"patch_encoder/dense/Tensordot/ProdШ
%patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%patch_encoder/dense/Tensordot/Const_1Ў
$patch_encoder/dense/Tensordot/Prod_1Prod1patch_encoder/dense/Tensordot/GatherV2_1:output:0.patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$patch_encoder/dense/Tensordot/Prod_1Ш
)patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)patch_encoder/dense/Tensordot/concat/axisФ
$patch_encoder/dense/Tensordot/concatConcatV2+patch_encoder/dense/Tensordot/free:output:0+patch_encoder/dense/Tensordot/axes:output:02patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$patch_encoder/dense/Tensordot/concat№
#patch_encoder/dense/Tensordot/stackPack+patch_encoder/dense/Tensordot/Prod:output:0-patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#patch_encoder/dense/Tensordot/stackи
'patch_encoder/dense/Tensordot/transpose	Transposepatches/Reshape:output:0-patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€и2)
'patch_encoder/dense/Tensordot/transposeп
%patch_encoder/dense/Tensordot/ReshapeReshape+patch_encoder/dense/Tensordot/transpose:y:0,patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2'
%patch_encoder/dense/Tensordot/Reshapeо
$patch_encoder/dense/Tensordot/MatMulMatMul.patch_encoder/dense/Tensordot/Reshape:output:04patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2&
$patch_encoder/dense/Tensordot/MatMulШ
%patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2'
%patch_encoder/dense/Tensordot/Const_2Ь
+patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+patch_encoder/dense/Tensordot/concat_1/axis°
&patch_encoder/dense/Tensordot/concat_1ConcatV2/patch_encoder/dense/Tensordot/GatherV2:output:0.patch_encoder/dense/Tensordot/Const_2:output:04patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&patch_encoder/dense/Tensordot/concat_1й
patch_encoder/dense/TensordotReshape.patch_encoder/dense/Tensordot/MatMul:product:0/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
patch_encoder/dense/Tensordot»
*patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp3patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*patch_encoder/dense/BiasAdd/ReadVariableOpа
patch_encoder/dense/BiasAddBiasAdd&patch_encoder/dense/Tensordot:output:02patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@2
patch_encoder/dense/BiasAddд
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_391872patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/391872*
_output_shapes

:Q@*
dtype02*
(patch_encoder/embedding/embedding_lookupј
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/391872*
_output_shapes

:Q@23
1patch_encoder/embedding/embedding_lookup/Identityџ
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes

:Q@25
3patch_encoder/embedding/embedding_lookup/Identity_1…
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
patch_encoder/add≤
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indicesз
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2"
 layer_normalization/moments/mean≈
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2*
(layer_normalization/moments/StopGradientу
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2/
-layer_normalization/moments/SquaredDifferenceЇ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indicesП
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2&
$layer_normalization/moments/varianceП
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52%
#layer_normalization/batchnorm/add/yв
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2#
!layer_normalization/batchnorm/add∞
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2%
#layer_normalization/batchnorm/RsqrtЏ
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOpж
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2#
!layer_normalization/batchnorm/mul≈
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization/batchnorm/mul_1ў
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization/batchnorm/mul_2ќ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOpв
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2#
!layer_normalization/batchnorm/subў
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization/batchnorm/add_1ч
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp®
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum’
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOpн
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2 
multi_head_attention/query/addс
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOpҐ
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsumѕ
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOpе
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
multi_head_attention/key/addч
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp®
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum’
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOpн
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/yЊ
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€Q@2
multi_head_attention/Mulф
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€QQ*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/EinsumЊ
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2&
$multi_head_attention/softmax/Softmaxƒ
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*/
_output_shapes
:€€€€€€€€€QQ2'
%multi_head_attention/dropout/IdentityМ
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*/
_output_shapes
:€€€€€€€€€Q@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/EinsumШ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpЋ
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:€€€€€€€€€Q@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsumт
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOpХ
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2+
)multi_head_attention/attention_output/addЧ
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2	
add/addґ
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indicesг
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2$
"layer_normalization_1/moments/meanЋ
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2,
*layer_normalization_1/moments/StopGradientп
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@21
/layer_normalization_1/moments/SquaredDifferenceЊ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indicesЧ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2(
&layer_normalization_1/moments/varianceУ
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_1/batchnorm/add/yк
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2%
#layer_normalization_1/batchnorm/addґ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2'
%layer_normalization_1/batchnorm/Rsqrtа
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOpо
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization_1/batchnorm/mulЅ
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_1/batchnorm/mul_1б
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_1/batchnorm/mul_2‘
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOpк
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization_1/batchnorm/subб
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_1/batchnorm/add_1ѓ
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@А*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axesБ
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/freeЛ
dense_1/Tensordot/ShapeShape)layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_1/Tensordot/ShapeД
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
dense_1/Tensordot/GatherV2И
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axis€
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
dense_1/Tensordot/Const†
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/ProdА
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1®
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1А
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axisЎ
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concatђ
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stackЋ
dense_1/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_1/Tensordot/transposeњ
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_1/Tensordot/Reshapeњ
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_1/Tensordot/MatMulБ
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_1/Tensordot/Const_2Д
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axisе
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1±
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Tensordot•
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_1/BiasAdd/ReadVariableOp®
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/xЩ
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_1/Gelu/Cast/x¶
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/truediv|
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_1/Gelu/add/xЧ
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/addТ
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_1/Gelu/mul_1ѓ
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource*
_output_shapes
:	А@*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axesБ
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
dense_2/Tensordot/ShapeД
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
dense_2/Tensordot/GatherV2И
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis€
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
dense_2/Tensordot/Const†
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/ProdА
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1®
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1А
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axisЎ
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concatђ
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stackє
dense_2/Tensordot/transpose	Transposedense_1/Gelu/mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€QА2
dense_2/Tensordot/transposeњ
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_2/Tensordot/ReshapeЊ
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_2/Tensordot/MatMulА
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_2/Tensordot/Const_2Д
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axisе
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1∞
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Tensordot§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOpІ
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/xШ
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_2/Gelu/Cast/x•
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/truediv{
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_2/Gelu/add/xЦ
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/addС
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
dense_2/Gelu/mul_1z
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
	add_1/addґ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesе
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2$
"layer_normalization_2/moments/meanЋ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2,
*layer_normalization_2/moments/StopGradientс
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@21
/layer_normalization_2/moments/SquaredDifferenceЊ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesЧ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2(
&layer_normalization_2/moments/varianceУ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_2/batchnorm/add/yк
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2%
#layer_normalization_2/batchnorm/addґ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2'
%layer_normalization_2/batchnorm/Rsqrtа
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpо
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization_2/batchnorm/mul√
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_2/batchnorm/mul_1б
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_2/batchnorm/mul_2‘
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpк
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2%
#layer_normalization_2/batchnorm/subб
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2'
%layer_normalization_2/batchnorm/add_1{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€@  2
flatten_layer/Constµ
flatten_layer/ReshapeReshape)layer_normalization_2/batchnorm/add_1:z:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€ј(2
flatten_layer/ReshapeЭ
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	ј(2*
dtype02
FC_1/MatMul/ReadVariableOpЪ
FC_1/MatMulMatMulflatten_layer/Reshape:output:0"FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
FC_1/MatMulЫ
FC_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_1/BiasAdd/ReadVariableOpХ
FC_1/BiasAddBiasAddFC_1/MatMul:product:0#FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
FC_1/BiasAddН
leaky_ReLu_1/LeakyRelu	LeakyReluFC_1/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2
leaky_ReLu_1/LeakyReluЬ
FC_2/MatMul/ReadVariableOpReadVariableOp#fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
FC_2/MatMul/ReadVariableOp†
FC_2/MatMulMatMul$leaky_ReLu_1/LeakyRelu:activations:0"FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
FC_2/MatMulЫ
FC_2/BiasAdd/ReadVariableOpReadVariableOp$fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02
FC_2/BiasAdd/ReadVariableOpХ
FC_2/BiasAddBiasAddFC_2/MatMul:product:0#FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
FC_2/BiasAddН
leaky_ReLu_2/LeakyRelu	LeakyReluFC_2/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2
leaky_ReLu_2/LeakyReluі
"output_layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02$
"output_layer/MatMul/ReadVariableOpЄ
output_layer/MatMulMatMul$leaky_ReLu_2/LeakyRelu:activations:0*output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/MatMul≥
#output_layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#output_layer/BiasAdd/ReadVariableOpµ
output_layer/BiasAddBiasAddoutput_layer/MatMul:product:0+output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/BiasAddИ
output_layer/SoftmaxSoftmaxoutput_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
output_layer/Softmax®
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
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
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2И
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
:€€€€€€€€€»»
 
_user_specified_nameinputs
°
О
O__inference_layer_normalization_layer_call_and_return_conditional_losses_390690

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЬ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/meanЙ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesњ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yТ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЦ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_1Й
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpТ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/add_1•
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs
є

ы
B__inference_conv_1_layer_call_and_return_conditional_losses_392447

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€∆∆
2	
BiasAddЯ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€∆∆
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€»»: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
£
Р
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_392868

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЬ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/meanЙ
moments/StopGradientStopGradientmoments/mean:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
moments/StopGradient®
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesњ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yТ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/addt
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЦ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mulz
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_1Й
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpТ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/subЙ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*+
_output_shapes
:€€€€€€€€€Q@2
batchnorm/add_1•
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*+
_output_shapes
:€€€€€€€€€Q@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€Q@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:S O
+
_output_shapes
:€€€€€€€€€Q@
 
_user_specified_nameinputs"ћL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ѕ
serving_default≠
M
input_layer>
serving_default_input_layer:0€€€€€€€€€»»@
output_layer0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:пЕ
ЈQ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer_with_weights-7
layer-11
layer-12
layer_with_weights-8
layer-13
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer-18
layer_with_weights-11
layer-19
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+Ф&call_and_return_all_conditional_losses
Х__call__
Ц_default_save_signature"°K
_tf_keras_networkЕK{"name": "WheatClassifier_CNN_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "WheatClassifier_CNN_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Patches", "config": {"layer was saved without config": true}, "name": "patches", "inbound_nodes": [[["maxpool_1", 0, 0, {}]]]}, {"class_name": "PatchEncoder", "config": {"layer was saved without config": true}, "name": "patch_encoder", "inbound_nodes": [[["patches", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization", "inbound_nodes": [[["patch_encoder", 0, 0, {}]]]}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}, "name": "multi_head_attention", "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_2", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_layer", "inbound_nodes": [[["layer_normalization_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1", "inbound_nodes": [[["flatten_layer", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_1", "inbound_nodes": [[["FC_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_2", "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_2", "inbound_nodes": [[["FC_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "shared_object_id": 40, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 200, 200, 3]}, "float32", "input_layer"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 42}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Addons>AdamW", "config": {"name": "AdamW", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false, "weight_decay": 9.999999747378752e-05}}}}
Е"В
_tf_keras_input_layerв{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
А

kernel
bias
trainable_variables
regularization_losses
	variables
 	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"ў	
_tf_keras_layerњ	{"name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_layer", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 3]}}
э


!kernel
"bias
#trainable_variables
$regularization_losses
%	variables
&	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"÷	
_tf_keras_layerЉ	{"name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 198, 198, 10]}}
ѕ
'trainable_variables
(regularization_losses
)	variables
*	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"Њ
_tf_keras_layer§{"name": "maxpool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
Ґ
+trainable_variables
,regularization_losses
-	variables
.	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"С
_tf_keras_layerч{"name": "patches", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Patches", "config": {"layer was saved without config": true}}
’
/
projection
0position_embedding
1trainable_variables
2regularization_losses
3	variables
4	keras_api
+Я&call_and_return_all_conditional_losses
†__call__"Ь
_tf_keras_layerВ{"name": "patch_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PatchEncoder", "config": {"layer was saved without config": true}}
„
5axis
	6gamma
7beta
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+°&call_and_return_all_conditional_losses
Ґ__call__"І
_tf_keras_layerН{"name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["patch_encoder", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
у

<_query_dense
=
_key_dense
>_value_dense
?_softmax
@_dropout_layer
A_output_dense
Btrainable_variables
Cregularization_losses
D	variables
E	keras_api
+£&call_and_return_all_conditional_losses
§__call__"щ
_tf_keras_layerя{"name": "multi_head_attention", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}, "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]], "shared_object_id": 13}
Э
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
+•&call_and_return_all_conditional_losses
¶__call__"М
_tf_keras_layerт{"name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]], "shared_object_id": 14, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 81, 64]}, {"class_name": "TensorShape", "items": [null, 81, 64]}]}
”
Jaxis
	Kgamma
Lbeta
Mtrainable_variables
Nregularization_losses
O	variables
P	keras_api
+І&call_and_return_all_conditional_losses
®__call__"£
_tf_keras_layerЙ{"name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add", 0, 0, {}]]], "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
С	

Qkernel
Rbias
Strainable_variables
Tregularization_losses
U	variables
V	keras_api
+©&call_and_return_all_conditional_losses
™__call__"к
_tf_keras_layer–{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
Д	

Wkernel
Xbias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
+Ђ&call_and_return_all_conditional_losses
ђ__call__"Ё
_tf_keras_layer√{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 128]}}
К
]trainable_variables
^regularization_losses
_	variables
`	keras_api
+≠&call_and_return_all_conditional_losses
Ѓ__call__"щ
_tf_keras_layerя{"name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]], "shared_object_id": 24, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 81, 64]}, {"class_name": "TensorShape", "items": [null, 81, 64]}]}
’
aaxis
	bgamma
cbeta
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
+ѓ&call_and_return_all_conditional_losses
∞__call__"•
_tf_keras_layerЛ{"name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_1", 0, 0, {}]]], "shared_object_id": 27, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
Џ
htrainable_variables
iregularization_losses
j	variables
k	keras_api
+±&call_and_return_all_conditional_losses
≤__call__"…
_tf_keras_layerѓ{"name": "flatten_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["layer_normalization_2", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 48}}
Д	

lkernel
mbias
ntrainable_variables
oregularization_losses
p	variables
q	keras_api
+≥&call_and_return_all_conditional_losses
і__call__"Ё
_tf_keras_layer√{"name": "FC_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_layer", 0, 0, {}]]], "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5184}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5184]}}
Я
rtrainable_variables
sregularization_losses
t	variables
u	keras_api
+µ&call_and_return_all_conditional_losses
ґ__call__"О
_tf_keras_layerф{"name": "leaky_ReLu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_1", 0, 0, {}]]], "shared_object_id": 32}
€

vkernel
wbias
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
+Ј&call_and_return_all_conditional_losses
Є__call__"Ў
_tf_keras_layerЊ{"name": "FC_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]], "shared_object_id": 35, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Я
|trainable_variables
}regularization_losses
~	variables
	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"О
_tf_keras_layerф{"name": "leaky_ReLu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_2", 0, 0, {}]]], "shared_object_id": 36}
Х	
Аkernel
	Бbias
Вtrainable_variables
Гregularization_losses
Д	variables
Е	keras_api
+ї&call_and_return_all_conditional_losses
Љ__call__"и
_tf_keras_layerќ{"name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
с
	Жiter
Зbeta_1
Иbeta_2

Йdecay
Кlearning_rate
Лweight_decaym÷m„!mЎ"mў6mЏ7mџKm№LmЁQmёRmяWmаXmбbmвcmгlmдmmеvmжwmз	Аmи	Бmй	Мmк	Нmл	Оmм	Пmн	Рmо	Сmп	Тmр	Уmс	Фmт	Хmу	Цmфvхvц!vч"vш6vщ7vъKvыLvьQvэRvюWv€XvАbvБcvВlvГmvДvvЕwvЖ	АvЗ	БvИ	МvЙ	НvК	ОvЛ	ПvМ	РvН	СvО	ТvП	УvР	ФvС	ХvТ	ЦvУ"
	optimizer
Ы
0
1
!2
"3
М4
Н5
О6
67
78
П9
Р10
С11
Т12
У13
Ф14
Х15
Ц16
K17
L18
Q19
R20
W21
X22
b23
c24
l25
m26
v27
w28
А29
Б30"
trackable_list_wrapper
 "
trackable_list_wrapper
Ы
0
1
!2
"3
М4
Н5
О6
67
78
П9
Р10
С11
Т12
У13
Ф14
Х15
Ц16
K17
L18
Q19
R20
W21
X22
b23
c24
l25
m26
v27
w28
А29
Б30"
trackable_list_wrapper
”
trainable_variables
regularization_losses
	variables
Чnon_trainable_variables
Шlayer_metrics
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Х__call__
Ц_default_save_signature
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
-
љserving_default"
signature_map
':%
2conv_1/kernel
:
2conv_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
µ
trainable_variables
regularization_losses
	variables
Ьnon_trainable_variables
Эlayer_metrics
Юlayers
Яmetrics
 †layer_regularization_losses
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
':%

2conv_2/kernel
:
2conv_2/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
µ
#trainable_variables
$regularization_losses
%	variables
°non_trainable_variables
Ґlayer_metrics
£layers
§metrics
 •layer_regularization_losses
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
'trainable_variables
(regularization_losses
)	variables
¶non_trainable_variables
Іlayer_metrics
®layers
©metrics
 ™layer_regularization_losses
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
+trainable_variables
,regularization_losses
-	variables
Ђnon_trainable_variables
ђlayer_metrics
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
а
Мkernel
	Нbias
∞trainable_variables
±regularization_losses
≤	variables
≥	keras_api
+Њ&call_and_return_all_conditional_losses
њ__call__"≥
_tf_keras_layerЩ{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 52}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 53}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 54, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}, "shared_object_id": 55}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 1000]}}
ў
О
embeddings
іtrainable_variables
µregularization_losses
ґ	variables
Ј	keras_api
+ј&call_and_return_all_conditional_losses
Ѕ__call__"≥
_tf_keras_layerЩ{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 81, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 56}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "shared_object_id": 57, "build_input_shape": {"class_name": "TensorShape", "items": [81]}}
8
М0
Н1
О2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
М0
Н1
О2"
trackable_list_wrapper
µ
1trainable_variables
2regularization_losses
3	variables
Єnon_trainable_variables
єlayer_metrics
Їlayers
їmetrics
 Љlayer_regularization_losses
†__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@2layer_normalization/gamma
&:$@2layer_normalization/beta
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
µ
8trainable_variables
9regularization_losses
:	variables
љnon_trainable_variables
Њlayer_metrics
њlayers
јmetrics
 Ѕlayer_regularization_losses
Ґ__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
У
¬partial_output_shape
√full_output_shape
Пkernel
	Рbias
ƒtrainable_variables
≈regularization_losses
∆	variables
«	keras_api
+¬&call_and_return_all_conditional_losses
√__call__"≥
_tf_keras_layerЩ{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 58, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
П
»partial_output_shape
…full_output_shape
Сkernel
	Тbias
 trainable_variables
Ћregularization_losses
ћ	variables
Ќ	keras_api
+ƒ&call_and_return_all_conditional_losses
≈__call__"ѓ
_tf_keras_layerХ{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 59, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
У
ќpartial_output_shape
ѕfull_output_shape
Уkernel
	Фbias
–trainable_variables
—regularization_losses
“	variables
”	keras_api
+∆&call_and_return_all_conditional_losses
«__call__"≥
_tf_keras_layerЩ{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 64]}}
Г
‘trainable_variables
’regularization_losses
÷	variables
„	keras_api
+»&call_and_return_all_conditional_losses
…__call__"о
_tf_keras_layer‘{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "shared_object_id": 61}
€
Ўtrainable_variables
ўregularization_losses
Џ	variables
џ	keras_api
+ &call_and_return_all_conditional_losses
Ћ__call__"к
_tf_keras_layer–{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 62}
®
№partial_output_shape
Ёfull_output_shape
Хkernel
	Цbias
ёtrainable_variables
яregularization_losses
а	variables
б	keras_api
+ћ&call_and_return_all_conditional_losses
Ќ__call__"»
_tf_keras_layerЃ{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 64], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 63, "build_input_shape": {"class_name": "TensorShape", "items": [null, 81, 4, 64]}}
`
П0
Р1
С2
Т3
У4
Ф5
Х6
Ц7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
П0
Р1
С2
Т3
У4
Ф5
Х6
Ц7"
trackable_list_wrapper
µ
Btrainable_variables
Cregularization_losses
D	variables
вnon_trainable_variables
гlayer_metrics
дlayers
еmetrics
 жlayer_regularization_losses
§__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ftrainable_variables
Gregularization_losses
H	variables
зnon_trainable_variables
иlayer_metrics
йlayers
кmetrics
 лlayer_regularization_losses
¶__call__
+•&call_and_return_all_conditional_losses
'•"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_1/gamma
(:&@2layer_normalization_1/beta
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
µ
Mtrainable_variables
Nregularization_losses
O	variables
мnon_trainable_variables
нlayer_metrics
оlayers
пmetrics
 рlayer_regularization_losses
®__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
!:	@А2dense_1/kernel
:А2dense_1/bias
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
µ
Strainable_variables
Tregularization_losses
U	variables
сnon_trainable_variables
тlayer_metrics
уlayers
фmetrics
 хlayer_regularization_losses
™__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
!:	А@2dense_2/kernel
:@2dense_2/bias
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
µ
Ytrainable_variables
Zregularization_losses
[	variables
цnon_trainable_variables
чlayer_metrics
шlayers
щmetrics
 ъlayer_regularization_losses
ђ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
]trainable_variables
^regularization_losses
_	variables
ыnon_trainable_variables
ьlayer_metrics
эlayers
юmetrics
 €layer_regularization_losses
Ѓ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_2/gamma
(:&@2layer_normalization_2/beta
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
µ
dtrainable_variables
eregularization_losses
f	variables
Аnon_trainable_variables
Бlayer_metrics
Вlayers
Гmetrics
 Дlayer_regularization_losses
∞__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
htrainable_variables
iregularization_losses
j	variables
Еnon_trainable_variables
Жlayer_metrics
Зlayers
Иmetrics
 Йlayer_regularization_losses
≤__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
:	ј(22FC_1/kernel
:22	FC_1/bias
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
µ
ntrainable_variables
oregularization_losses
p	variables
Кnon_trainable_variables
Лlayer_metrics
Мlayers
Нmetrics
 Оlayer_regularization_losses
і__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
rtrainable_variables
sregularization_losses
t	variables
Пnon_trainable_variables
Рlayer_metrics
Сlayers
Тmetrics
 Уlayer_regularization_losses
ґ__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
:222FC_2/kernel
:22	FC_2/bias
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
µ
xtrainable_variables
yregularization_losses
z	variables
Фnon_trainable_variables
Хlayer_metrics
Цlayers
Чmetrics
 Шlayer_regularization_losses
Є__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
|trainable_variables
}regularization_losses
~	variables
Щnon_trainable_variables
Ъlayer_metrics
Ыlayers
Ьmetrics
 Эlayer_regularization_losses
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
%:#22output_layer/kernel
:2output_layer/bias
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
Є
Вtrainable_variables
Гregularization_losses
Д	variables
Юnon_trainable_variables
Яlayer_metrics
†layers
°metrics
 Ґlayer_regularization_losses
Љ__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
:	 (2
AdamW/iter
: (2AdamW/beta_1
: (2AdamW/beta_2
: (2AdamW/decay
: (2AdamW/learning_rate
: (2AdamW/weight_decay
-:+	и@2patch_encoder/dense/kernel
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ґ
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
19"
trackable_list_wrapper
0
£0
§1"
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
М0
Н1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
М0
Н1"
trackable_list_wrapper
Є
∞trainable_variables
±regularization_losses
≤	variables
•non_trainable_variables
¶layer_metrics
Іlayers
®metrics
 ©layer_regularization_losses
њ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
(
О0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
О0"
trackable_list_wrapper
Є
іtrainable_variables
µregularization_losses
ґ	variables
™non_trainable_variables
Ђlayer_metrics
ђlayers
≠metrics
 Ѓlayer_regularization_losses
Ѕ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
/0
01"
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
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
Є
ƒtrainable_variables
≈regularization_losses
∆	variables
ѓnon_trainable_variables
∞layer_metrics
±layers
≤metrics
 ≥layer_regularization_losses
√__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
Є
 trainable_variables
Ћregularization_losses
ћ	variables
іnon_trainable_variables
µlayer_metrics
ґlayers
Јmetrics
 Єlayer_regularization_losses
≈__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
Є
–trainable_variables
—regularization_losses
“	variables
єnon_trainable_variables
Їlayer_metrics
їlayers
Љmetrics
 љlayer_regularization_losses
«__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
‘trainable_variables
’regularization_losses
÷	variables
Њnon_trainable_variables
њlayer_metrics
јlayers
Ѕmetrics
 ¬layer_regularization_losses
…__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ўtrainable_variables
ўregularization_losses
Џ	variables
√non_trainable_variables
ƒlayer_metrics
≈layers
∆metrics
 «layer_regularization_losses
Ћ__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Х0
Ц1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Х0
Ц1"
trackable_list_wrapper
Є
ёtrainable_variables
яregularization_losses
а	variables
»non_trainable_variables
…layer_metrics
 layers
Ћmetrics
 ћlayer_regularization_losses
Ќ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
<0
=1
>2
?3
@4
A5"
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
Ў

Ќtotal

ќcount
ѕ	variables
–	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 64}
Т

—total

“count
”
_fn_kwargs
‘	variables
’	keras_api"∆
_tf_keras_metricЂ{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 42}
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
:  (2total
:  (2count
0
Ќ0
ќ1"
trackable_list_wrapper
.
ѕ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
—0
“1"
trackable_list_wrapper
.
‘	variables"
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
-:+@2!AdamW/layer_normalization/gamma/m
,:*@2 AdamW/layer_normalization/beta/m
/:-@2#AdamW/layer_normalization_1/gamma/m
.:,@2"AdamW/layer_normalization_1/beta/m
':%	@А2AdamW/dense_1/kernel/m
!:А2AdamW/dense_1/bias/m
':%	А@2AdamW/dense_2/kernel/m
 :@2AdamW/dense_2/bias/m
/:-@2#AdamW/layer_normalization_2/gamma/m
.:,@2"AdamW/layer_normalization_2/beta/m
$:"	ј(22AdamW/FC_1/kernel/m
:22AdamW/FC_1/bias/m
#:!222AdamW/FC_2/kernel/m
:22AdamW/FC_2/bias/m
+:)22AdamW/output_layer/kernel/m
%:#2AdamW/output_layer/bias/m
3:1	и@2"AdamW/patch_encoder/dense/kernel/m
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
-:+
2AdamW/conv_1/kernel/v
:
2AdamW/conv_1/bias/v
-:+

2AdamW/conv_2/kernel/v
:
2AdamW/conv_2/bias/v
-:+@2!AdamW/layer_normalization/gamma/v
,:*@2 AdamW/layer_normalization/beta/v
/:-@2#AdamW/layer_normalization_1/gamma/v
.:,@2"AdamW/layer_normalization_1/beta/v
':%	@А2AdamW/dense_1/kernel/v
!:А2AdamW/dense_1/bias/v
':%	А@2AdamW/dense_2/kernel/v
 :@2AdamW/dense_2/bias/v
/:-@2#AdamW/layer_normalization_2/gamma/v
.:,@2"AdamW/layer_normalization_2/beta/v
$:"	ј(22AdamW/FC_1/kernel/v
:22AdamW/FC_1/bias/v
#:!222AdamW/FC_2/kernel/v
:22AdamW/FC_2/bias/v
+:)22AdamW/output_layer/kernel/v
%:#2AdamW/output_layer/bias/v
3:1	и@2"AdamW/patch_encoder/dense/kernel/v
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
Т2П
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_392056
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_392303
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_391654
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_391739ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
¶2£
6__inference_WheatClassifier_CNN_5_layer_call_fn_391046
6__inference_WheatClassifier_CNN_5_layer_call_fn_392370
6__inference_WheatClassifier_CNN_5_layer_call_fn_392437
6__inference_WheatClassifier_CNN_5_layer_call_fn_391569ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
н2к
!__inference__wrapped_model_390552ƒ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *4Ґ1
/К,
input_layer€€€€€€€€€»»
м2й
B__inference_conv_1_layer_call_and_return_conditional_losses_392447Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv_1_layer_call_fn_392456Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv_2_layer_call_and_return_conditional_losses_392466Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv_2_layer_call_fn_392475Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≠2™
E__inference_maxpool_1_layer_call_and_return_conditional_losses_390558а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Т2П
*__inference_maxpool_1_layer_call_fn_390564а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
н2к
C__inference_patches_layer_call_and_return_conditional_losses_392489Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_patches_layer_call_fn_392494Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jimages
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
I__inference_patch_encoder_layer_call_and_return_conditional_losses_392534°
Ш≤Ф
FullArgSpec
argsЪ
jself
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
.__inference_patch_encoder_layer_call_fn_392545°
Ш≤Ф
FullArgSpec
argsЪ
jself
jpatch
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щ2ц
O__inference_layer_normalization_layer_call_and_return_conditional_losses_392567Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ё2џ
4__inference_layer_normalization_layer_call_fn_392576Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¶2£
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_392611
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_392653ь
у≤п
FullArgSpece
args]ЪZ
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
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
р2н
5__inference_multi_head_attention_layer_call_fn_392675
5__inference_multi_head_attention_layer_call_fn_392697ь
у≤п
FullArgSpece
args]ЪZ
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
defaultsЪ

 

 
p 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
й2ж
?__inference_add_layer_call_and_return_conditional_losses_392703Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ2Ћ
$__inference_add_layer_call_fn_392709Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ы2ш
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_392731Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
а2Ё
6__inference_layer_normalization_1_layer_call_fn_392740Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_1_layer_call_and_return_conditional_losses_392778Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_1_layer_call_fn_392787Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_dense_2_layer_call_and_return_conditional_losses_392825Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
“2ѕ
(__inference_dense_2_layer_call_fn_392834Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_add_1_layer_call_and_return_conditional_losses_392840Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_add_1_layer_call_fn_392846Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ы2ш
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_392868Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
а2Ё
6__inference_layer_normalization_2_layer_call_fn_392877Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
у2р
I__inference_flatten_layer_layer_call_and_return_conditional_losses_392883Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ў2’
.__inference_flatten_layer_layer_call_fn_392888Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_FC_1_layer_call_and_return_conditional_losses_392898Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_FC_1_layer_call_fn_392907Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_392912Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_leaky_ReLu_1_layer_call_fn_392917Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_FC_2_layer_call_and_return_conditional_losses_392927Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_FC_2_layer_call_fn_392936Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_392941Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_leaky_ReLu_2_layer_call_fn_392946Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
т2п
H__inference_output_layer_layer_call_and_return_conditional_losses_392957Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
„2‘
-__inference_output_layer_layer_call_fn_392966Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕBћ
$__inference_signature_wrapper_391816input_layer"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
µ2≤ѓ
¶≤Ґ
FullArgSpec%
argsЪ
jself
jinputs
jmask
varargs
 
varkw
 
defaultsҐ

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ї2Јі
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 °
@__inference_FC_1_layer_call_and_return_conditional_losses_392898]lm0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј(
™ "%Ґ"
К
0€€€€€€€€€2
Ъ y
%__inference_FC_1_layer_call_fn_392907Plm0Ґ-
&Ґ#
!К
inputs€€€€€€€€€ј(
™ "К€€€€€€€€€2†
@__inference_FC_2_layer_call_and_return_conditional_losses_392927\vw/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€2
Ъ x
%__inference_FC_2_layer_call_fn_392936Ovw/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€2у
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_391654Э,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБFҐC
<Ґ9
/К,
input_layer€€€€€€€€€»»
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ у
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_391739Э,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБFҐC
<Ґ9
/К,
input_layer€€€€€€€€€»»
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ о
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_392056Ш,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБAҐ>
7Ґ4
*К'
inputs€€€€€€€€€»»
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ о
Q__inference_WheatClassifier_CNN_5_layer_call_and_return_conditional_losses_392303Ш,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБAҐ>
7Ґ4
*К'
inputs€€€€€€€€€»»
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Ћ
6__inference_WheatClassifier_CNN_5_layer_call_fn_391046Р,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБFҐC
<Ґ9
/К,
input_layer€€€€€€€€€»»
p 

 
™ "К€€€€€€€€€Ћ
6__inference_WheatClassifier_CNN_5_layer_call_fn_391569Р,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБFҐC
<Ґ9
/К,
input_layer€€€€€€€€€»»
p

 
™ "К€€€€€€€€€∆
6__inference_WheatClassifier_CNN_5_layer_call_fn_392370Л,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБAҐ>
7Ґ4
*К'
inputs€€€€€€€€€»»
p 

 
™ "К€€€€€€€€€∆
6__inference_WheatClassifier_CNN_5_layer_call_fn_392437Л,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБAҐ>
7Ґ4
*К'
inputs€€€€€€€€€»»
p

 
™ "К€€€€€€€€€—
!__inference__wrapped_model_390552Ђ,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБ>Ґ;
4Ґ1
/К,
input_layer€€€€€€€€€»»
™ ";™8
6
output_layer&К#
output_layer€€€€€€€€€’
A__inference_add_1_layer_call_and_return_conditional_losses_392840ПbҐ_
XҐU
SЪP
&К#
inputs/0€€€€€€€€€Q@
&К#
inputs/1€€€€€€€€€Q@
™ ")Ґ&
К
0€€€€€€€€€Q@
Ъ ≠
&__inference_add_1_layer_call_fn_392846ВbҐ_
XҐU
SЪP
&К#
inputs/0€€€€€€€€€Q@
&К#
inputs/1€€€€€€€€€Q@
™ "К€€€€€€€€€Q@”
?__inference_add_layer_call_and_return_conditional_losses_392703ПbҐ_
XҐU
SЪP
&К#
inputs/0€€€€€€€€€Q@
&К#
inputs/1€€€€€€€€€Q@
™ ")Ґ&
К
0€€€€€€€€€Q@
Ъ Ђ
$__inference_add_layer_call_fn_392709ВbҐ_
XҐU
SЪP
&К#
inputs/0€€€€€€€€€Q@
&К#
inputs/1€€€€€€€€€Q@
™ "К€€€€€€€€€Q@ґ
B__inference_conv_1_layer_call_and_return_conditional_losses_392447p9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€»»
™ "/Ґ,
%К"
0€€€€€€€€€∆∆

Ъ О
'__inference_conv_1_layer_call_fn_392456c9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€»»
™ ""К€€€€€€€€€∆∆
ґ
B__inference_conv_2_layer_call_and_return_conditional_losses_392466p!"9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€∆∆

™ "/Ґ,
%К"
0€€€€€€€€€ƒƒ

Ъ О
'__inference_conv_2_layer_call_fn_392475c!"9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€∆∆

™ ""К€€€€€€€€€ƒƒ
ђ
C__inference_dense_1_layer_call_and_return_conditional_losses_392778eQR3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ "*Ґ'
 К
0€€€€€€€€€QА
Ъ Д
(__inference_dense_1_layer_call_fn_392787XQR3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ "К€€€€€€€€€QАђ
C__inference_dense_2_layer_call_and_return_conditional_losses_392825eWX4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€QА
™ ")Ґ&
К
0€€€€€€€€€Q@
Ъ Д
(__inference_dense_2_layer_call_fn_392834XWX4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€QА
™ "К€€€€€€€€€Q@™
I__inference_flatten_layer_layer_call_and_return_conditional_losses_392883]3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ "&Ґ#
К
0€€€€€€€€€ј(
Ъ В
.__inference_flatten_layer_layer_call_fn_392888P3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ "К€€€€€€€€€ј(є
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_392731dKL3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ ")Ґ&
К
0€€€€€€€€€Q@
Ъ С
6__inference_layer_normalization_1_layer_call_fn_392740WKL3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ "К€€€€€€€€€Q@є
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_392868dbc3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ ")Ґ&
К
0€€€€€€€€€Q@
Ъ С
6__inference_layer_normalization_2_layer_call_fn_392877Wbc3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ "К€€€€€€€€€Q@Ј
O__inference_layer_normalization_layer_call_and_return_conditional_losses_392567d673Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ ")Ґ&
К
0€€€€€€€€€Q@
Ъ П
4__inference_layer_normalization_layer_call_fn_392576W673Ґ0
)Ґ&
$К!
inputs€€€€€€€€€Q@
™ "К€€€€€€€€€Q@§
H__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_392912X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€2
Ъ |
-__inference_leaky_ReLu_1_layer_call_fn_392917K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€2§
H__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_392941X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€2
Ъ |
-__inference_leaky_ReLu_2_layer_call_fn_392946K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€2и
E__inference_maxpool_1_layer_call_and_return_conditional_losses_390558ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ј
*__inference_maxpool_1_layer_call_fn_390564СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€ы
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_392611¶ПРСТУФХЦgҐd
]ҐZ
#К 
query€€€€€€€€€Q@
#К 
value€€€€€€€€€Q@

 

 
p 
p 
™ ")Ґ&
К
0€€€€€€€€€Q@
Ъ ы
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_392653¶ПРСТУФХЦgҐd
]ҐZ
#К 
query€€€€€€€€€Q@
#К 
value€€€€€€€€€Q@

 

 
p 
p
™ ")Ґ&
К
0€€€€€€€€€Q@
Ъ ”
5__inference_multi_head_attention_layer_call_fn_392675ЩПРСТУФХЦgҐd
]ҐZ
#К 
query€€€€€€€€€Q@
#К 
value€€€€€€€€€Q@

 

 
p 
p 
™ "К€€€€€€€€€Q@”
5__inference_multi_head_attention_layer_call_fn_392697ЩПРСТУФХЦgҐd
]ҐZ
#К 
query€€€€€€€€€Q@
#К 
value€€€€€€€€€Q@

 

 
p 
p
™ "К€€€€€€€€€Q@™
H__inference_output_layer_layer_call_and_return_conditional_losses_392957^АБ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€
Ъ В
-__inference_output_layer_layer_call_fn_392966QАБ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€Њ
I__inference_patch_encoder_layer_call_and_return_conditional_losses_392534qМНО<Ґ9
2Ґ/
-К*
patch€€€€€€€€€€€€€€€€€€и
™ ")Ґ&
К
0€€€€€€€€€Q@
Ъ Ц
.__inference_patch_encoder_layer_call_fn_392545dМНО<Ґ9
2Ґ/
-К*
patch€€€€€€€€€€€€€€€€€€и
™ "К€€€€€€€€€Q@µ
C__inference_patches_layer_call_and_return_conditional_losses_392489n7Ґ4
-Ґ*
(К%
images€€€€€€€€€bb

™ "3Ґ0
)К&
0€€€€€€€€€€€€€€€€€€и
Ъ Н
(__inference_patches_layer_call_fn_392494a7Ґ4
-Ґ*
(К%
images€€€€€€€€€bb

™ "&К#€€€€€€€€€€€€€€€€€€иг
$__inference_signature_wrapper_391816Ї,!"МНО67ПРСТУФХЦKLQRWXbclmvwАБMҐJ
Ґ 
C™@
>
input_layer/К,
input_layer€€€€€€€€€»»";™8
6
output_layer&К#
output_layer€€€€€€€€€