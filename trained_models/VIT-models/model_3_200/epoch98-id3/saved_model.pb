’ф1
ўѓ
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
 И"serve*2.5.02v2.5.0-0-ga4dfb8d1a718о¶+
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
О
layer_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_3/gamma
З
/layer_normalization_3/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_3/gamma*
_output_shapes
:@*
dtype0
М
layer_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_3/beta
Е
.layer_normalization_3/beta/Read/ReadVariableOpReadVariableOplayer_normalization_3/beta*
_output_shapes
:@*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	@А*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:А*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	А@*
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
О
layer_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_4/gamma
З
/layer_normalization_4/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_4/gamma*
_output_shapes
:@*
dtype0
М
layer_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_4/beta
Е
.layer_normalization_4/beta/Read/ReadVariableOpReadVariableOplayer_normalization_4/beta*
_output_shapes
:@*
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А»2*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
А»2*
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:2*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:22*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:2*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:2*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
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
shape:	ђ@*+
shared_namepatch_encoder/dense/kernel
К
.patch_encoder/dense/kernel/Read/ReadVariableOpReadVariableOppatch_encoder/dense/kernel*
_output_shapes
:	ђ@*
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
°
"patch_encoder/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*3
shared_name$"patch_encoder/embedding/embeddings
Ъ
6patch_encoder/embedding/embeddings/Read/ReadVariableOpReadVariableOp"patch_encoder/embedding/embeddings*
_output_shapes
:	Р@*
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
¶
#multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_1/query/kernel
Я
7multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/query/kernel*"
_output_shapes
:@@*
dtype0
Ю
!multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/query/bias
Ч
5multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/query/bias*
_output_shapes

:@*
dtype0
Ґ
!multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*2
shared_name#!multi_head_attention_1/key/kernel
Ы
5multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/key/kernel*"
_output_shapes
:@@*
dtype0
Ъ
multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!multi_head_attention_1/key/bias
У
3multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_1/key/bias*
_output_shapes

:@*
dtype0
¶
#multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#multi_head_attention_1/value/kernel
Я
7multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/value/kernel*"
_output_shapes
:@@*
dtype0
Ю
!multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/value/bias
Ч
5multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/value/bias*
_output_shapes

:@*
dtype0
Љ
.multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*?
shared_name0.multi_head_attention_1/attention_output/kernel
µ
Bmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_1/attention_output/kernel*"
_output_shapes
:@@*
dtype0
∞
,multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*=
shared_name.,multi_head_attention_1/attention_output/bias
©
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
Ю
#AdamW/layer_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_3/gamma/m
Ч
7AdamW/layer_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_3/gamma/m*
_output_shapes
:@*
dtype0
Ь
"AdamW/layer_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_3/beta/m
Х
6AdamW/layer_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_3/beta/m*
_output_shapes
:@*
dtype0
Й
AdamW/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*'
shared_nameAdamW/dense_3/kernel/m
В
*AdamW/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_3/kernel/m*
_output_shapes
:	@А*
dtype0
Б
AdamW/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdamW/dense_3/bias/m
z
(AdamW/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_3/bias/m*
_output_shapes	
:А*
dtype0
Й
AdamW/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*'
shared_nameAdamW/dense_4/kernel/m
В
*AdamW/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_4/kernel/m*
_output_shapes
:	А@*
dtype0
А
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
Ю
#AdamW/layer_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_4/gamma/m
Ч
7AdamW/layer_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_4/gamma/m*
_output_shapes
:@*
dtype0
Ь
"AdamW/layer_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_4/beta/m
Х
6AdamW/layer_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_4/beta/m*
_output_shapes
:@*
dtype0
К
AdamW/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А»2*'
shared_nameAdamW/dense_5/kernel/m
Г
*AdamW/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_5/kernel/m* 
_output_shapes
:
А»2*
dtype0
А
AdamW/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdamW/dense_5/bias/m
y
(AdamW/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_5/bias/m*
_output_shapes
:2*
dtype0
И
AdamW/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*'
shared_nameAdamW/dense_6/kernel/m
Б
*AdamW/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_6/kernel/m*
_output_shapes

:22*
dtype0
А
AdamW/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdamW/dense_6/bias/m
y
(AdamW/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_6/bias/m*
_output_shapes
:2*
dtype0
И
AdamW/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdamW/dense_7/kernel/m
Б
*AdamW/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/dense_7/kernel/m*
_output_shapes

:2*
dtype0
А
AdamW/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdamW/dense_7/bias/m
y
(AdamW/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdamW/dense_7/bias/m*
_output_shapes
:*
dtype0
°
"AdamW/patch_encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*3
shared_name$"AdamW/patch_encoder/dense/kernel/m
Ъ
6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/m*
_output_shapes
:	ђ@*
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
±
*AdamW/patch_encoder/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/m
™
>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/m*
_output_shapes
:	Р@*
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
ґ
+AdamW/multi_head_attention_1/query/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/query/kernel/m
ѓ
?AdamW/multi_head_attention_1/query/kernel/m/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/query/kernel/m*"
_output_shapes
:@@*
dtype0
Ѓ
)AdamW/multi_head_attention_1/query/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/query/bias/m
І
=AdamW/multi_head_attention_1/query/bias/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/query/bias/m*
_output_shapes

:@*
dtype0
≤
)AdamW/multi_head_attention_1/key/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention_1/key/kernel/m
Ђ
=AdamW/multi_head_attention_1/key/kernel/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/key/kernel/m*"
_output_shapes
:@@*
dtype0
™
'AdamW/multi_head_attention_1/key/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention_1/key/bias/m
£
;AdamW/multi_head_attention_1/key/bias/m/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention_1/key/bias/m*
_output_shapes

:@*
dtype0
ґ
+AdamW/multi_head_attention_1/value/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/value/kernel/m
ѓ
?AdamW/multi_head_attention_1/value/kernel/m/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/value/kernel/m*"
_output_shapes
:@@*
dtype0
Ѓ
)AdamW/multi_head_attention_1/value/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/value/bias/m
І
=AdamW/multi_head_attention_1/value/bias/m/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/value/bias/m*
_output_shapes

:@*
dtype0
ћ
6AdamW/multi_head_attention_1/attention_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*G
shared_name86AdamW/multi_head_attention_1/attention_output/kernel/m
≈
JAdamW/multi_head_attention_1/attention_output/kernel/m/Read/ReadVariableOpReadVariableOp6AdamW/multi_head_attention_1/attention_output/kernel/m*"
_output_shapes
:@@*
dtype0
ј
4AdamW/multi_head_attention_1/attention_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64AdamW/multi_head_attention_1/attention_output/bias/m
є
HAdamW/multi_head_attention_1/attention_output/bias/m/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention_1/attention_output/bias/m*
_output_shapes
:@*
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
Ю
#AdamW/layer_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_3/gamma/v
Ч
7AdamW/layer_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_3/gamma/v*
_output_shapes
:@*
dtype0
Ь
"AdamW/layer_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_3/beta/v
Х
6AdamW/layer_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_3/beta/v*
_output_shapes
:@*
dtype0
Й
AdamW/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*'
shared_nameAdamW/dense_3/kernel/v
В
*AdamW/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_3/kernel/v*
_output_shapes
:	@А*
dtype0
Б
AdamW/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*%
shared_nameAdamW/dense_3/bias/v
z
(AdamW/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_3/bias/v*
_output_shapes	
:А*
dtype0
Й
AdamW/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*'
shared_nameAdamW/dense_4/kernel/v
В
*AdamW/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_4/kernel/v*
_output_shapes
:	А@*
dtype0
А
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
Ю
#AdamW/layer_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#AdamW/layer_normalization_4/gamma/v
Ч
7AdamW/layer_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp#AdamW/layer_normalization_4/gamma/v*
_output_shapes
:@*
dtype0
Ь
"AdamW/layer_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"AdamW/layer_normalization_4/beta/v
Х
6AdamW/layer_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp"AdamW/layer_normalization_4/beta/v*
_output_shapes
:@*
dtype0
К
AdamW/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А»2*'
shared_nameAdamW/dense_5/kernel/v
Г
*AdamW/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_5/kernel/v* 
_output_shapes
:
А»2*
dtype0
А
AdamW/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdamW/dense_5/bias/v
y
(AdamW/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_5/bias/v*
_output_shapes
:2*
dtype0
И
AdamW/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:22*'
shared_nameAdamW/dense_6/kernel/v
Б
*AdamW/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_6/kernel/v*
_output_shapes

:22*
dtype0
А
AdamW/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*%
shared_nameAdamW/dense_6/bias/v
y
(AdamW/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_6/bias/v*
_output_shapes
:2*
dtype0
И
AdamW/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*'
shared_nameAdamW/dense_7/kernel/v
Б
*AdamW/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/dense_7/kernel/v*
_output_shapes

:2*
dtype0
А
AdamW/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdamW/dense_7/bias/v
y
(AdamW/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdamW/dense_7/bias/v*
_output_shapes
:*
dtype0
°
"AdamW/patch_encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ@*3
shared_name$"AdamW/patch_encoder/dense/kernel/v
Ъ
6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp"AdamW/patch_encoder/dense/kernel/v*
_output_shapes
:	ђ@*
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
±
*AdamW/patch_encoder/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р@*;
shared_name,*AdamW/patch_encoder/embedding/embeddings/v
™
>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp*AdamW/patch_encoder/embedding/embeddings/v*
_output_shapes
:	Р@*
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
ґ
+AdamW/multi_head_attention_1/query/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/query/kernel/v
ѓ
?AdamW/multi_head_attention_1/query/kernel/v/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/query/kernel/v*"
_output_shapes
:@@*
dtype0
Ѓ
)AdamW/multi_head_attention_1/query/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/query/bias/v
І
=AdamW/multi_head_attention_1/query/bias/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/query/bias/v*
_output_shapes

:@*
dtype0
≤
)AdamW/multi_head_attention_1/key/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)AdamW/multi_head_attention_1/key/kernel/v
Ђ
=AdamW/multi_head_attention_1/key/kernel/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/key/kernel/v*"
_output_shapes
:@@*
dtype0
™
'AdamW/multi_head_attention_1/key/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*8
shared_name)'AdamW/multi_head_attention_1/key/bias/v
£
;AdamW/multi_head_attention_1/key/bias/v/Read/ReadVariableOpReadVariableOp'AdamW/multi_head_attention_1/key/bias/v*
_output_shapes

:@*
dtype0
ґ
+AdamW/multi_head_attention_1/value/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+AdamW/multi_head_attention_1/value/kernel/v
ѓ
?AdamW/multi_head_attention_1/value/kernel/v/Read/ReadVariableOpReadVariableOp+AdamW/multi_head_attention_1/value/kernel/v*"
_output_shapes
:@@*
dtype0
Ѓ
)AdamW/multi_head_attention_1/value/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*:
shared_name+)AdamW/multi_head_attention_1/value/bias/v
І
=AdamW/multi_head_attention_1/value/bias/v/Read/ReadVariableOpReadVariableOp)AdamW/multi_head_attention_1/value/bias/v*
_output_shapes

:@*
dtype0
ћ
6AdamW/multi_head_attention_1/attention_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*G
shared_name86AdamW/multi_head_attention_1/attention_output/kernel/v
≈
JAdamW/multi_head_attention_1/attention_output/kernel/v/Read/ReadVariableOpReadVariableOp6AdamW/multi_head_attention_1/attention_output/kernel/v*"
_output_shapes
:@@*
dtype0
ј
4AdamW/multi_head_attention_1/attention_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*E
shared_name64AdamW/multi_head_attention_1/attention_output/bias/v
є
HAdamW/multi_head_attention_1/attention_output/bias/v/Read/ReadVariableOpReadVariableOp4AdamW/multi_head_attention_1/attention_output/bias/v*
_output_shapes
:@*
dtype0

NoOpNoOp
€э
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*єэ
valueЃэB™э BҐэ
¶
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
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
 	keras_api
z
!
projection
"position_embedding
#trainable_variables
$regularization_losses
%	variables
&	keras_api
q
'axis
	(gamma
)beta
*trainable_variables
+regularization_losses
,	variables
-	keras_api
ї
._query_dense
/
_key_dense
0_value_dense
1_softmax
2_dropout_layer
3_output_dense
4trainable_variables
5regularization_losses
6	variables
7	keras_api
R
8trainable_variables
9regularization_losses
:	variables
;	keras_api
q
<axis
	=gamma
>beta
?trainable_variables
@regularization_losses
A	variables
B	keras_api
h

Ckernel
Dbias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
h

Ikernel
Jbias
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
R
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
q
Saxis
	Tgamma
Ubeta
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
ї
Z_query_dense
[
_key_dense
\_value_dense
]_softmax
^_dropout_layer
__output_dense
`trainable_variables
aregularization_losses
b	variables
c	keras_api
R
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
q
haxis
	igamma
jbeta
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
h

okernel
pbias
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
h

ukernel
vbias
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
R
{trainable_variables
|regularization_losses
}	variables
~	keras_api
w
axis

Аgamma
	Бbeta
Вtrainable_variables
Гregularization_losses
Д	variables
Е	keras_api
V
Жtrainable_variables
Зregularization_losses
И	variables
Й	keras_api
n
Кkernel
	Лbias
Мtrainable_variables
Нregularization_losses
О	variables
П	keras_api
n
Рkernel
	Сbias
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
n
Цkernel
	Чbias
Шtrainable_variables
Щregularization_losses
Ъ	variables
Ы	keras_api
к
	Ьiter
Эbeta_1
Юbeta_2

Яdecay
†learning_rate
°weight_decay(mЉ)mљ=mЊ>mњCmјDmЅIm¬Jm√TmƒUm≈im∆jm«om»pm…um vmЋ	Аmћ	БmЌ	Кmќ	Лmѕ	Рm–	Сm—	Цm“	Чm”	Ґm‘	£m’	§m÷	•m„	¶mЎ	Іmў	®mЏ	©mџ	™m№	ЂmЁ	ђmё	≠mя	Ѓmа	ѓmб	∞mв	±mг	≤mд	≥mе	іmж(vз)vи=vй>vкCvлDvмIvнJvоTvпUvрivсjvтovуpvфuvхvvц	Аvч	Бvш	Кvщ	Лvъ	Рvы	Сvь	Цvэ	Чvю	Ґv€	£vА	§vБ	•vВ	¶vГ	ІvД	®vЕ	©vЖ	™vЗ	ЂvИ	ђvЙ	≠vК	ЃvЛ	ѓvМ	∞vН	±vО	≤vП	≥vР	іvС
й
Ґ0
£1
§2
(3
)4
•5
¶6
І7
®8
©9
™10
Ђ11
ђ12
=13
>14
C15
D16
I17
J18
T19
U20
≠21
Ѓ22
ѓ23
∞24
±25
≤26
≥27
і28
i29
j30
o31
p32
u33
v34
А35
Б36
К37
Л38
Р39
С40
Ц41
Ч42
 
й
Ґ0
£1
§2
(3
)4
•5
¶6
І7
®8
©9
™10
Ђ11
ђ12
=13
>14
C15
D16
I17
J18
T19
U20
≠21
Ѓ22
ѓ23
∞24
±25
≤26
≥27
і28
i29
j30
o31
p32
u33
v34
А35
Б36
К37
Л38
Р39
С40
Ц41
Ч42
≤
trainable_variables
regularization_losses
µmetrics
 ґlayer_regularization_losses
Јnon_trainable_variables
Єlayer_metrics
	variables
єlayers
 
 
 
 
≤
trainable_variables
regularization_losses
Їmetrics
 їlayer_regularization_losses
Љnon_trainable_variables
љlayer_metrics
	variables
Њlayers
n
Ґkernel
	£bias
њtrainable_variables
јregularization_losses
Ѕ	variables
¬	keras_api
g
§
embeddings
√trainable_variables
ƒregularization_losses
≈	variables
∆	keras_api

Ґ0
£1
§2
 

Ґ0
£1
§2
≤
#trainable_variables
$regularization_losses
«metrics
 »layer_regularization_losses
…non_trainable_variables
 layer_metrics
%	variables
Ћlayers
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
≤
*trainable_variables
+regularization_losses
ћmetrics
 Ќlayer_regularization_losses
ќnon_trainable_variables
ѕlayer_metrics
,	variables
–layers
°
—partial_output_shape
“full_output_shape
•kernel
	¶bias
”trainable_variables
‘regularization_losses
’	variables
÷	keras_api
°
„partial_output_shape
Ўfull_output_shape
Іkernel
	®bias
ўtrainable_variables
Џregularization_losses
џ	variables
№	keras_api
°
Ёpartial_output_shape
ёfull_output_shape
©kernel
	™bias
яtrainable_variables
аregularization_losses
б	variables
в	keras_api
V
гtrainable_variables
дregularization_losses
е	variables
ж	keras_api
V
зtrainable_variables
иregularization_losses
й	variables
к	keras_api
°
лpartial_output_shape
мfull_output_shape
Ђkernel
	ђbias
нtrainable_variables
оregularization_losses
п	variables
р	keras_api
@
•0
¶1
І2
®3
©4
™5
Ђ6
ђ7
 
@
•0
¶1
І2
®3
©4
™5
Ђ6
ђ7
≤
4trainable_variables
5regularization_losses
сmetrics
 тlayer_regularization_losses
уnon_trainable_variables
фlayer_metrics
6	variables
хlayers
 
 
 
≤
8trainable_variables
9regularization_losses
цmetrics
 чlayer_regularization_losses
шnon_trainable_variables
щlayer_metrics
:	variables
ъlayers
 
fd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
 

=0
>1
≤
?trainable_variables
@regularization_losses
ыmetrics
 ьlayer_regularization_losses
эnon_trainable_variables
юlayer_metrics
A	variables
€layers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
 

C0
D1
≤
Etrainable_variables
Fregularization_losses
Аmetrics
 Бlayer_regularization_losses
Вnon_trainable_variables
Гlayer_metrics
G	variables
Дlayers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 

I0
J1
≤
Ktrainable_variables
Lregularization_losses
Еmetrics
 Жlayer_regularization_losses
Зnon_trainable_variables
Иlayer_metrics
M	variables
Йlayers
 
 
 
≤
Otrainable_variables
Pregularization_losses
Кmetrics
 Лlayer_regularization_losses
Мnon_trainable_variables
Нlayer_metrics
Q	variables
Оlayers
 
fd
VARIABLE_VALUElayer_normalization_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE

T0
U1
 

T0
U1
≤
Vtrainable_variables
Wregularization_losses
Пmetrics
 Рlayer_regularization_losses
Сnon_trainable_variables
Тlayer_metrics
X	variables
Уlayers
°
Фpartial_output_shape
Хfull_output_shape
≠kernel
	Ѓbias
Цtrainable_variables
Чregularization_losses
Ш	variables
Щ	keras_api
°
Ъpartial_output_shape
Ыfull_output_shape
ѓkernel
	∞bias
Ьtrainable_variables
Эregularization_losses
Ю	variables
Я	keras_api
°
†partial_output_shape
°full_output_shape
±kernel
	≤bias
Ґtrainable_variables
£regularization_losses
§	variables
•	keras_api
V
¶trainable_variables
Іregularization_losses
®	variables
©	keras_api
V
™trainable_variables
Ђregularization_losses
ђ	variables
≠	keras_api
°
Ѓpartial_output_shape
ѓfull_output_shape
≥kernel
	іbias
∞trainable_variables
±regularization_losses
≤	variables
≥	keras_api
@
≠0
Ѓ1
ѓ2
∞3
±4
≤5
≥6
і7
 
@
≠0
Ѓ1
ѓ2
∞3
±4
≤5
≥6
і7
≤
`trainable_variables
aregularization_losses
іmetrics
 µlayer_regularization_losses
ґnon_trainable_variables
Јlayer_metrics
b	variables
Єlayers
 
 
 
≤
dtrainable_variables
eregularization_losses
єmetrics
 Їlayer_regularization_losses
їnon_trainable_variables
Љlayer_metrics
f	variables
љlayers
 
fd
VARIABLE_VALUElayer_normalization_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE

i0
j1
 

i0
j1
≤
ktrainable_variables
lregularization_losses
Њmetrics
 њlayer_regularization_losses
јnon_trainable_variables
Ѕlayer_metrics
m	variables
¬layers
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

o0
p1
 

o0
p1
≤
qtrainable_variables
rregularization_losses
√metrics
 ƒlayer_regularization_losses
≈non_trainable_variables
∆layer_metrics
s	variables
«layers
[Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_4/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

u0
v1
 

u0
v1
≤
wtrainable_variables
xregularization_losses
»metrics
 …layer_regularization_losses
 non_trainable_variables
Ћlayer_metrics
y	variables
ћlayers
 
 
 
≤
{trainable_variables
|regularization_losses
Ќmetrics
 ќlayer_regularization_losses
ѕnon_trainable_variables
–layer_metrics
}	variables
—layers
 
ge
VARIABLE_VALUElayer_normalization_4/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElayer_normalization_4/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
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
“metrics
 ”layer_regularization_losses
‘non_trainable_variables
’layer_metrics
Д	variables
÷layers
 
 
 
µ
Жtrainable_variables
Зregularization_losses
„metrics
 Ўlayer_regularization_losses
ўnon_trainable_variables
Џlayer_metrics
И	variables
џlayers
[Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_5/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

К0
Л1
 

К0
Л1
µ
Мtrainable_variables
Нregularization_losses
№metrics
 Ёlayer_regularization_losses
ёnon_trainable_variables
яlayer_metrics
О	variables
аlayers
[Y
VARIABLE_VALUEdense_6/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_6/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

Р0
С1
 

Р0
С1
µ
Тtrainable_variables
Уregularization_losses
бmetrics
 вlayer_regularization_losses
гnon_trainable_variables
дlayer_metrics
Ф	variables
еlayers
[Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

Ц0
Ч1
 

Ц0
Ч1
µ
Шtrainable_variables
Щregularization_losses
жmetrics
 зlayer_regularization_losses
иnon_trainable_variables
йlayer_metrics
Ъ	variables
кlayers
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
л0
м1
 
 
 
¶
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
 
 
 
 
 

Ґ0
£1
 

Ґ0
£1
µ
њtrainable_variables
јregularization_losses
нmetrics
 оlayer_regularization_losses
пnon_trainable_variables
рlayer_metrics
Ѕ	variables
сlayers

§0
 

§0
µ
√trainable_variables
ƒregularization_losses
тmetrics
 уlayer_regularization_losses
фnon_trainable_variables
хlayer_metrics
≈	variables
цlayers
 
 
 
 

!0
"1
 
 
 
 
 
 
 

•0
¶1
 

•0
¶1
µ
”trainable_variables
‘regularization_losses
чmetrics
 шlayer_regularization_losses
щnon_trainable_variables
ъlayer_metrics
’	variables
ыlayers
 
 

І0
®1
 

І0
®1
µ
ўtrainable_variables
Џregularization_losses
ьmetrics
 эlayer_regularization_losses
юnon_trainable_variables
€layer_metrics
џ	variables
Аlayers
 
 

©0
™1
 

©0
™1
µ
яtrainable_variables
аregularization_losses
Бmetrics
 Вlayer_regularization_losses
Гnon_trainable_variables
Дlayer_metrics
б	variables
Еlayers
 
 
 
µ
гtrainable_variables
дregularization_losses
Жmetrics
 Зlayer_regularization_losses
Иnon_trainable_variables
Йlayer_metrics
е	variables
Кlayers
 
 
 
µ
зtrainable_variables
иregularization_losses
Лmetrics
 Мlayer_regularization_losses
Нnon_trainable_variables
Оlayer_metrics
й	variables
Пlayers
 
 

Ђ0
ђ1
 

Ђ0
ђ1
µ
нtrainable_variables
оregularization_losses
Рmetrics
 Сlayer_regularization_losses
Тnon_trainable_variables
Уlayer_metrics
п	variables
Фlayers
 
 
 
 
*
.0
/1
02
13
24
35
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
≠0
Ѓ1
 

≠0
Ѓ1
µ
Цtrainable_variables
Чregularization_losses
Хmetrics
 Цlayer_regularization_losses
Чnon_trainable_variables
Шlayer_metrics
Ш	variables
Щlayers
 
 

ѓ0
∞1
 

ѓ0
∞1
µ
Ьtrainable_variables
Эregularization_losses
Ъmetrics
 Ыlayer_regularization_losses
Ьnon_trainable_variables
Эlayer_metrics
Ю	variables
Юlayers
 
 

±0
≤1
 

±0
≤1
µ
Ґtrainable_variables
£regularization_losses
Яmetrics
 †layer_regularization_losses
°non_trainable_variables
Ґlayer_metrics
§	variables
£layers
 
 
 
µ
¶trainable_variables
Іregularization_losses
§metrics
 •layer_regularization_losses
¶non_trainable_variables
Іlayer_metrics
®	variables
®layers
 
 
 
µ
™trainable_variables
Ђregularization_losses
©metrics
 ™layer_regularization_losses
Ђnon_trainable_variables
ђlayer_metrics
ђ	variables
≠layers
 
 

≥0
і1
 

≥0
і1
µ
∞trainable_variables
±regularization_losses
Ѓmetrics
 ѓlayer_regularization_losses
∞non_trainable_variables
±layer_metrics
≤	variables
≤layers
 
 
 
 
*
Z0
[1
\2
]3
^4
_5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

≥total

іcount
µ	variables
ґ	keras_api
I

Јtotal

Єcount
є
_fn_kwargs
Ї	variables
ї	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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
≥0
і1

µ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ј0
Є1

Ї	variables
ЙЖ
VARIABLE_VALUE!AdamW/layer_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE AdamW/layer_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_3/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_3/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_3/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_3/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_4/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_4/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#AdamW/layer_normalization_4/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"AdamW/layer_normalization_4/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_5/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_5/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_6/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_6/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_7/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_7/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE+AdamW/multi_head_attention_1/query/kernel/mMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)AdamW/multi_head_attention_1/query/bias/mMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)AdamW/multi_head_attention_1/key/kernel/mMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention_1/key/bias/mMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE+AdamW/multi_head_attention_1/value/kernel/mMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)AdamW/multi_head_attention_1/value/bias/mMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE6AdamW/multi_head_attention_1/attention_output/kernel/mMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE4AdamW/multi_head_attention_1/attention_output/bias/mMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE!AdamW/layer_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЗД
VARIABLE_VALUE AdamW/layer_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_2/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_2/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE#AdamW/layer_normalization_3/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE"AdamW/layer_normalization_3/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdamW/dense_3/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdamW/dense_3/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_4/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_4/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE#AdamW/layer_normalization_4/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE"AdamW/layer_normalization_4/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_5/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_5/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_6/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_6/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/dense_7/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/dense_7/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"AdamW/patch_encoder/dense/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE AdamW/patch_encoder/dense/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE*AdamW/patch_encoder/embedding/embeddings/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)AdamW/multi_head_attention/query/kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'AdamW/multi_head_attention/query/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'AdamW/multi_head_attention/key/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%AdamW/multi_head_attention/key/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE)AdamW/multi_head_attention/value/kernel/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention/value/bias/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE4AdamW/multi_head_attention/attention_output/kernel/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЦУ
VARIABLE_VALUE2AdamW/multi_head_attention/attention_output/bias/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE+AdamW/multi_head_attention_1/query/kernel/vMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)AdamW/multi_head_attention_1/query/bias/vMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)AdamW/multi_head_attention_1/key/kernel/vMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'AdamW/multi_head_attention_1/key/bias/vMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE+AdamW/multi_head_attention_1/value/kernel/vMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
НК
VARIABLE_VALUE)AdamW/multi_head_attention_1/value/bias/vMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE6AdamW/multi_head_attention_1/attention_output/kernel/vMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ШХ
VARIABLE_VALUE4AdamW/multi_head_attention_1/attention_output/bias/vMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
О
serving_default_input_1Placeholder*1
_output_shapes
:€€€€€€€€€»»*
dtype0*&
shape:€€€€€€€€€»»
§
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1patch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddingslayer_normalization/gammalayer_normalization/beta!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biaslayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/beta#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biaslayer_normalization_3/gammalayer_normalization_3/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/biaslayer_normalization_4/gammalayer_normalization_4/betadense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_586250
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
В;
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp/layer_normalization_2/gamma/Read/ReadVariableOp.layer_normalization_2/beta/Read/ReadVariableOp/layer_normalization_3/gamma/Read/ReadVariableOp.layer_normalization_3/beta/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp/layer_normalization_4/gamma/Read/ReadVariableOp.layer_normalization_4/beta/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdamW/iter/Read/ReadVariableOp AdamW/beta_1/Read/ReadVariableOp AdamW/beta_2/Read/ReadVariableOpAdamW/decay/Read/ReadVariableOp'AdamW/learning_rate/Read/ReadVariableOp&AdamW/weight_decay/Read/ReadVariableOp.patch_encoder/dense/kernel/Read/ReadVariableOp,patch_encoder/dense/bias/Read/ReadVariableOp6patch_encoder/embedding/embeddings/Read/ReadVariableOp5multi_head_attention/query/kernel/Read/ReadVariableOp3multi_head_attention/query/bias/Read/ReadVariableOp3multi_head_attention/key/kernel/Read/ReadVariableOp1multi_head_attention/key/bias/Read/ReadVariableOp5multi_head_attention/value/kernel/Read/ReadVariableOp3multi_head_attention/value/bias/Read/ReadVariableOp@multi_head_attention/attention_output/kernel/Read/ReadVariableOp>multi_head_attention/attention_output/bias/Read/ReadVariableOp7multi_head_attention_1/query/kernel/Read/ReadVariableOp5multi_head_attention_1/query/bias/Read/ReadVariableOp5multi_head_attention_1/key/kernel/Read/ReadVariableOp3multi_head_attention_1/key/bias/Read/ReadVariableOp7multi_head_attention_1/value/kernel/Read/ReadVariableOp5multi_head_attention_1/value/bias/Read/ReadVariableOpBmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOp@multi_head_attention_1/attention_output/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp5AdamW/layer_normalization/gamma/m/Read/ReadVariableOp4AdamW/layer_normalization/beta/m/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/m/Read/ReadVariableOp*AdamW/dense_1/kernel/m/Read/ReadVariableOp(AdamW/dense_1/bias/m/Read/ReadVariableOp*AdamW/dense_2/kernel/m/Read/ReadVariableOp(AdamW/dense_2/bias/m/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/m/Read/ReadVariableOp7AdamW/layer_normalization_3/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_3/beta/m/Read/ReadVariableOp*AdamW/dense_3/kernel/m/Read/ReadVariableOp(AdamW/dense_3/bias/m/Read/ReadVariableOp*AdamW/dense_4/kernel/m/Read/ReadVariableOp(AdamW/dense_4/bias/m/Read/ReadVariableOp7AdamW/layer_normalization_4/gamma/m/Read/ReadVariableOp6AdamW/layer_normalization_4/beta/m/Read/ReadVariableOp*AdamW/dense_5/kernel/m/Read/ReadVariableOp(AdamW/dense_5/bias/m/Read/ReadVariableOp*AdamW/dense_6/kernel/m/Read/ReadVariableOp(AdamW/dense_6/bias/m/Read/ReadVariableOp*AdamW/dense_7/kernel/m/Read/ReadVariableOp(AdamW/dense_7/bias/m/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/m/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/m/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/m/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/m/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/m/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/m/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/m/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/m/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/m/Read/ReadVariableOp?AdamW/multi_head_attention_1/query/kernel/m/Read/ReadVariableOp=AdamW/multi_head_attention_1/query/bias/m/Read/ReadVariableOp=AdamW/multi_head_attention_1/key/kernel/m/Read/ReadVariableOp;AdamW/multi_head_attention_1/key/bias/m/Read/ReadVariableOp?AdamW/multi_head_attention_1/value/kernel/m/Read/ReadVariableOp=AdamW/multi_head_attention_1/value/bias/m/Read/ReadVariableOpJAdamW/multi_head_attention_1/attention_output/kernel/m/Read/ReadVariableOpHAdamW/multi_head_attention_1/attention_output/bias/m/Read/ReadVariableOp5AdamW/layer_normalization/gamma/v/Read/ReadVariableOp4AdamW/layer_normalization/beta/v/Read/ReadVariableOp7AdamW/layer_normalization_1/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_1/beta/v/Read/ReadVariableOp*AdamW/dense_1/kernel/v/Read/ReadVariableOp(AdamW/dense_1/bias/v/Read/ReadVariableOp*AdamW/dense_2/kernel/v/Read/ReadVariableOp(AdamW/dense_2/bias/v/Read/ReadVariableOp7AdamW/layer_normalization_2/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_2/beta/v/Read/ReadVariableOp7AdamW/layer_normalization_3/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_3/beta/v/Read/ReadVariableOp*AdamW/dense_3/kernel/v/Read/ReadVariableOp(AdamW/dense_3/bias/v/Read/ReadVariableOp*AdamW/dense_4/kernel/v/Read/ReadVariableOp(AdamW/dense_4/bias/v/Read/ReadVariableOp7AdamW/layer_normalization_4/gamma/v/Read/ReadVariableOp6AdamW/layer_normalization_4/beta/v/Read/ReadVariableOp*AdamW/dense_5/kernel/v/Read/ReadVariableOp(AdamW/dense_5/bias/v/Read/ReadVariableOp*AdamW/dense_6/kernel/v/Read/ReadVariableOp(AdamW/dense_6/bias/v/Read/ReadVariableOp*AdamW/dense_7/kernel/v/Read/ReadVariableOp(AdamW/dense_7/bias/v/Read/ReadVariableOp6AdamW/patch_encoder/dense/kernel/v/Read/ReadVariableOp4AdamW/patch_encoder/dense/bias/v/Read/ReadVariableOp>AdamW/patch_encoder/embedding/embeddings/v/Read/ReadVariableOp=AdamW/multi_head_attention/query/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/query/bias/v/Read/ReadVariableOp;AdamW/multi_head_attention/key/kernel/v/Read/ReadVariableOp9AdamW/multi_head_attention/key/bias/v/Read/ReadVariableOp=AdamW/multi_head_attention/value/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention/value/bias/v/Read/ReadVariableOpHAdamW/multi_head_attention/attention_output/kernel/v/Read/ReadVariableOpFAdamW/multi_head_attention/attention_output/bias/v/Read/ReadVariableOp?AdamW/multi_head_attention_1/query/kernel/v/Read/ReadVariableOp=AdamW/multi_head_attention_1/query/bias/v/Read/ReadVariableOp=AdamW/multi_head_attention_1/key/kernel/v/Read/ReadVariableOp;AdamW/multi_head_attention_1/key/bias/v/Read/ReadVariableOp?AdamW/multi_head_attention_1/value/kernel/v/Read/ReadVariableOp=AdamW/multi_head_attention_1/value/bias/v/Read/ReadVariableOpJAdamW/multi_head_attention_1/attention_output/kernel/v/Read/ReadVariableOpHAdamW/multi_head_attention_1/attention_output/bias/v/Read/ReadVariableOpConst*Ы
TinУ
Р2Н	*
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
__inference__traced_save_588428
°%
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_normalization/gammalayer_normalization/betalayer_normalization_1/gammalayer_normalization_1/betadense_1/kerneldense_1/biasdense_2/kerneldense_2/biaslayer_normalization_2/gammalayer_normalization_2/betalayer_normalization_3/gammalayer_normalization_3/betadense_3/kerneldense_3/biasdense_4/kerneldense_4/biaslayer_normalization_4/gammalayer_normalization_4/betadense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias
AdamW/iterAdamW/beta_1AdamW/beta_2AdamW/decayAdamW/learning_rateAdamW/weight_decaypatch_encoder/dense/kernelpatch_encoder/dense/bias"patch_encoder/embedding/embeddings!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biastotalcounttotal_1count_1!AdamW/layer_normalization/gamma/m AdamW/layer_normalization/beta/m#AdamW/layer_normalization_1/gamma/m"AdamW/layer_normalization_1/beta/mAdamW/dense_1/kernel/mAdamW/dense_1/bias/mAdamW/dense_2/kernel/mAdamW/dense_2/bias/m#AdamW/layer_normalization_2/gamma/m"AdamW/layer_normalization_2/beta/m#AdamW/layer_normalization_3/gamma/m"AdamW/layer_normalization_3/beta/mAdamW/dense_3/kernel/mAdamW/dense_3/bias/mAdamW/dense_4/kernel/mAdamW/dense_4/bias/m#AdamW/layer_normalization_4/gamma/m"AdamW/layer_normalization_4/beta/mAdamW/dense_5/kernel/mAdamW/dense_5/bias/mAdamW/dense_6/kernel/mAdamW/dense_6/bias/mAdamW/dense_7/kernel/mAdamW/dense_7/bias/m"AdamW/patch_encoder/dense/kernel/m AdamW/patch_encoder/dense/bias/m*AdamW/patch_encoder/embedding/embeddings/m)AdamW/multi_head_attention/query/kernel/m'AdamW/multi_head_attention/query/bias/m'AdamW/multi_head_attention/key/kernel/m%AdamW/multi_head_attention/key/bias/m)AdamW/multi_head_attention/value/kernel/m'AdamW/multi_head_attention/value/bias/m4AdamW/multi_head_attention/attention_output/kernel/m2AdamW/multi_head_attention/attention_output/bias/m+AdamW/multi_head_attention_1/query/kernel/m)AdamW/multi_head_attention_1/query/bias/m)AdamW/multi_head_attention_1/key/kernel/m'AdamW/multi_head_attention_1/key/bias/m+AdamW/multi_head_attention_1/value/kernel/m)AdamW/multi_head_attention_1/value/bias/m6AdamW/multi_head_attention_1/attention_output/kernel/m4AdamW/multi_head_attention_1/attention_output/bias/m!AdamW/layer_normalization/gamma/v AdamW/layer_normalization/beta/v#AdamW/layer_normalization_1/gamma/v"AdamW/layer_normalization_1/beta/vAdamW/dense_1/kernel/vAdamW/dense_1/bias/vAdamW/dense_2/kernel/vAdamW/dense_2/bias/v#AdamW/layer_normalization_2/gamma/v"AdamW/layer_normalization_2/beta/v#AdamW/layer_normalization_3/gamma/v"AdamW/layer_normalization_3/beta/vAdamW/dense_3/kernel/vAdamW/dense_3/bias/vAdamW/dense_4/kernel/vAdamW/dense_4/bias/v#AdamW/layer_normalization_4/gamma/v"AdamW/layer_normalization_4/beta/vAdamW/dense_5/kernel/vAdamW/dense_5/bias/vAdamW/dense_6/kernel/vAdamW/dense_6/bias/vAdamW/dense_7/kernel/vAdamW/dense_7/bias/v"AdamW/patch_encoder/dense/kernel/v AdamW/patch_encoder/dense/bias/v*AdamW/patch_encoder/embedding/embeddings/v)AdamW/multi_head_attention/query/kernel/v'AdamW/multi_head_attention/query/bias/v'AdamW/multi_head_attention/key/kernel/v%AdamW/multi_head_attention/key/bias/v)AdamW/multi_head_attention/value/kernel/v'AdamW/multi_head_attention/value/bias/v4AdamW/multi_head_attention/attention_output/kernel/v2AdamW/multi_head_attention/attention_output/bias/v+AdamW/multi_head_attention_1/query/kernel/v)AdamW/multi_head_attention_1/query/bias/v)AdamW/multi_head_attention_1/key/kernel/v'AdamW/multi_head_attention_1/key/bias/v+AdamW/multi_head_attention_1/value/kernel/v)AdamW/multi_head_attention_1/value/bias/v6AdamW/multi_head_attention_1/attention_output/kernel/v4AdamW/multi_head_attention_1/attention_output/bias/v*Ъ
TinТ
П2М*
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
"__inference__traced_restore_588855щч%
±7
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_587378	
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
"query/einsum/Einsum/ReadVariableOp»
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЪ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp¬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpТ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp»
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЪ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
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
:€€€€€€€€€Р@2
MulҐ
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2
einsum/EinsumБ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/dropout/Const®
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Mul_1є
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpш
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp¬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
attention_output/addГ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 2J
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
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
≥7
ы
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_585355	
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
"query/einsum/Einsum/ReadVariableOp»
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЪ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp¬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpТ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp»
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЪ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
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
:€€€€€€€€€Р@2
MulҐ
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2
einsum/EinsumБ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/dropout/Const®
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Mul_1є
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpш
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp¬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
attention_output/addГ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 2J
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
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
Н.
ы
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_584856	
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
"query/einsum/Einsum/ReadVariableOp»
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЪ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp¬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpТ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp»
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЪ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
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
:€€€€€€€€€Р@2
MulҐ
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2
einsum/EinsumБ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
softmax/SoftmaxЗ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/Identityє
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpш
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp¬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
attention_output/addГ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 2J
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
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
щ
є
.__inference_patch_encoder_layer_call_fn_587270	
patch
unknown:	ђ@
	unknown_0:@
	unknown_1:	Р@
identityИҐStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallpatchunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_5845722
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ

_user_specified_namepatch
ѓ
О
O__inference_layer_normalization_layer_call_and_return_conditional_losses_587292

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
ѓ
О
O__inference_layer_normalization_layer_call_and_return_conditional_losses_584602

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
Н'
ы
C__inference_dense_2_layer_call_and_return_conditional_losses_587550

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
Tensordot/stackТ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
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
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2	
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
:€€€€€€€€€Р@2

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
:€€€€€€€€€Р@2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

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
:€€€€€€€€€Р@2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Gelu/mul_1Ы
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€РА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€РА
 
_user_specified_nameinputs
Є
µ

$__inference_signature_wrapper_586250
input_1
unknown:	ђ@
	unknown_0:@
	unknown_1:	Р@
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

unknown_14:	@А

unknown_15:	А

unknown_16:	А@

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

unknown_30:	@А

unknown_31:	А

unknown_32:	А@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:
А»2

unknown_37:2

unknown_38:22

unknown_39:2

unknown_40:2

unknown_41:
identityИҐStatefulPartitionedCallК
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
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_5845092
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€»»
!
_user_specified_name	input_1
г

м
7__inference_multi_head_attention_1_layer_call_fn_587701	
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
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_5848562
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
п
m
A__inference_add_2_layer_call_and_return_conditional_losses_587729
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:€€€€€€€€€Р@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:V R
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/1
Ь
Х
(__inference_dense_6_layer_call_fn_587968

inputs
unknown:22
	unknown_0:2
identityИҐStatefulPartitionedCallц
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
GPU2*0J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_5850842
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
г

м
7__inference_multi_head_attention_1_layer_call_fn_587723	
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
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_5853552
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
≥
Ц
(__inference_dense_2_layer_call_fn_587559

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_5847792
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€РА: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€РА
 
_user_specified_nameinputs
Ћ
D
(__inference_flatten_layer_call_fn_587914

inputs
identity∆
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€А»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5850402
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:€€€€€€€€€А»2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Р@:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
≥7
ы
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_587679	
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
"query/einsum/Einsum/ReadVariableOp»
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЪ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp¬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpТ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp»
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЪ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
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
:€€€€€€€€€Р@2
MulҐ
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2
einsum/EinsumБ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/dropout/Const®
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Mul_1є
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpш
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp¬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
attention_output/addГ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 2J
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
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
е
i
?__inference_add_layer_call_and_return_conditional_losses_584667

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:€€€€€€€€€Р@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
±7
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_585496	
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
"query/einsum/Einsum/ReadVariableOp»
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЪ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp¬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpТ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp»
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЪ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
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
:€€€€€€€€€Р@2
MulҐ
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2
einsum/EinsumБ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
softmax/Softmaxs
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/dropout/Const®
dropout/dropout/MulMulsoftmax/Softmax:softmax:0dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Mulw
dropout/dropout/ShapeShapesoftmax/Softmax:softmax:0*
T0*
_output_shapes
:2
dropout/dropout/Shape÷
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР*
dtype02.
,dropout/dropout/random_uniform/RandomUniformЕ
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2 
dropout/dropout/GreaterEqual/yи
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/GreaterEqual°
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Cast§
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/dropout/Mul_1є
einsum_1/EinsumEinsumdropout/dropout/Mul_1:z:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpш
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp¬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
attention_output/addГ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 2J
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
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
з
k
A__inference_add_3_layer_call_and_return_conditional_losses_585004

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:€€€€€€€€€Р@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
я

к
5__inference_multi_head_attention_layer_call_fn_587400	
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
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_5846432
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
Ѓt
ё
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_585108

inputs'
patch_encoder_584573:	ђ@"
patch_encoder_584575:@'
patch_encoder_584577:	Р@(
layer_normalization_584603:@(
layer_normalization_584605:@1
multi_head_attention_584644:@@-
multi_head_attention_584646:@1
multi_head_attention_584648:@@-
multi_head_attention_584650:@1
multi_head_attention_584652:@@-
multi_head_attention_584654:@1
multi_head_attention_584656:@@)
multi_head_attention_584658:@*
layer_normalization_1_584692:@*
layer_normalization_1_584694:@!
dense_1_584736:	@А
dense_1_584738:	А!
dense_2_584780:	А@
dense_2_584782:@*
layer_normalization_2_584816:@*
layer_normalization_2_584818:@3
multi_head_attention_1_584857:@@/
multi_head_attention_1_584859:@3
multi_head_attention_1_584861:@@/
multi_head_attention_1_584863:@3
multi_head_attention_1_584865:@@/
multi_head_attention_1_584867:@3
multi_head_attention_1_584869:@@+
multi_head_attention_1_584871:@*
layer_normalization_3_584905:@*
layer_normalization_3_584907:@!
dense_3_584949:	@А
dense_3_584951:	А!
dense_4_584993:	А@
dense_4_584995:@*
layer_normalization_4_585029:@*
layer_normalization_4_585031:@"
dense_5_585061:
А»2
dense_5_585063:2 
dense_6_585085:22
dense_6_585087:2 
dense_7_585102:2
dense_7_585104:
identityИҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐ+layer_normalization/StatefulPartitionedCallҐ-layer_normalization_1/StatefulPartitionedCallҐ-layer_normalization_2/StatefulPartitionedCallҐ-layer_normalization_3/StatefulPartitionedCallҐ-layer_normalization_4/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ.multi_head_attention_1/StatefulPartitionedCallҐ%patch_encoder/StatefulPartitionedCallв
patches/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_5845302
patches/PartitionedCallз
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_584573patch_encoder_584575patch_encoder_584577*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_5845722'
%patch_encoder/StatefulPartitionedCallы
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_584603layer_normalization_584605*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5846022-
+layer_normalization/StatefulPartitionedCallч
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_584644multi_head_attention_584646multi_head_attention_584648multi_head_attention_584650multi_head_attention_584652multi_head_attention_584654multi_head_attention_584656multi_head_attention_584658*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_5846432.
,multi_head_attention/StatefulPartitionedCall≠
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_5846672
add/PartitionedCallу
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_584692layer_normalization_1_584694*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5846912/
-layer_normalization_1/StatefulPartitionedCall»
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_584736dense_1_584738*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5847352!
dense_1/StatefulPartitionedCallє
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_584780dense_2_584782*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_5847792!
dense_2/StatefulPartitionedCallФ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5847912
add_1/PartitionedCallх
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_584816layer_normalization_2_584818*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5848152/
-layer_normalization_2/StatefulPartitionedCallС
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_584857multi_head_attention_1_584859multi_head_attention_1_584861multi_head_attention_1_584863multi_head_attention_1_584865multi_head_attention_1_584867multi_head_attention_1_584869multi_head_attention_1_584871*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_58485620
.multi_head_attention_1/StatefulPartitionedCall•
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_5848802
add_2/PartitionedCallх
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_584905layer_normalization_3_584907*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_5849042/
-layer_normalization_3/StatefulPartitionedCall»
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_584949dense_3_584951*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_5849482!
dense_3/StatefulPartitionedCallє
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_584993dense_4_584995*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_5849922!
dense_4/StatefulPartitionedCallЦ
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_5850042
add_3/PartitionedCallх
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_585029layer_normalization_4_585031*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_5850282/
-layer_normalization_4/StatefulPartitionedCallЖ
flatten/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€А»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5850402
flatten/PartitionedCallђ
dense_5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5_585061dense_5_585063*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_5850602!
dense_5/StatefulPartitionedCallі
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_585085dense_6_585087*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_5850842!
dense_6/StatefulPartitionedCallі
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_585102dense_7_585104*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_5851012!
dense_7/StatefulPartitionedCallа
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
п
m
A__inference_add_3_layer_call_and_return_conditional_losses_587866
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:€€€€€€€€€Р@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:V R
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/1
±
Р
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_587894

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
б
_
C__inference_flatten_layer_call_and_return_conditional_losses_587909

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ d  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:€€€€€€€€€А»2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:€€€€€€€€€А»2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Р@:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
Ґ
Ч
(__inference_dense_5_layer_call_fn_587941

inputs
unknown:
А»2
	unknown_0:2
identityИҐStatefulPartitionedCallц
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
GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_5850602
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:€€€€€€€€€А»: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:€€€€€€€€€А»
 
_user_specified_nameinputs
»
Я
6__inference_layer_normalization_2_layer_call_fn_587602

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5848152
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
±
Р
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_587757

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
І
_
C__inference_patches_layer_call_and_return_conditional_losses_584530

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
:€€€€€€€€€ђ*
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
B :ђ2
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
!:€€€€€€€€€€€€€€€€€€ђ2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€»»:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameimages
б
_
C__inference_flatten_layer_call_and_return_conditional_losses_585040

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ d  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:€€€€€€€€€А»2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:€€€€€€€€€А»2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Р@:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
≥

ф
C__inference_dense_7_layer_call_and_return_conditional_losses_587979

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
Н.
ы
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_587637	
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
"query/einsum/Einsum/ReadVariableOp»
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЪ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp¬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpТ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp»
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЪ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
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
:€€€€€€€€€Р@2
MulҐ
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2
einsum/EinsumБ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
softmax/SoftmaxЗ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/Identityє
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpш
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp¬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
attention_output/addГ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 2J
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
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
№Ґ
Л7
!__inference__wrapped_model_584509
input_1^
Kwheatclassifier_vit_3_patch_encoder_dense_tensordot_readvariableop_resource:	ђ@W
Iwheatclassifier_vit_3_patch_encoder_dense_biasadd_readvariableop_resource:@X
Ewheatclassifier_vit_3_patch_encoder_embedding_embedding_lookup_584175:	Р@]
Owheatclassifier_vit_3_layer_normalization_batchnorm_mul_readvariableop_resource:@Y
Kwheatclassifier_vit_3_layer_normalization_batchnorm_readvariableop_resource:@l
Vwheatclassifier_vit_3_multi_head_attention_query_einsum_einsum_readvariableop_resource:@@^
Lwheatclassifier_vit_3_multi_head_attention_query_add_readvariableop_resource:@j
Twheatclassifier_vit_3_multi_head_attention_key_einsum_einsum_readvariableop_resource:@@\
Jwheatclassifier_vit_3_multi_head_attention_key_add_readvariableop_resource:@l
Vwheatclassifier_vit_3_multi_head_attention_value_einsum_einsum_readvariableop_resource:@@^
Lwheatclassifier_vit_3_multi_head_attention_value_add_readvariableop_resource:@w
awheatclassifier_vit_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:@@e
Wwheatclassifier_vit_3_multi_head_attention_attention_output_add_readvariableop_resource:@_
Qwheatclassifier_vit_3_layer_normalization_1_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_vit_3_layer_normalization_1_batchnorm_readvariableop_resource:@R
?wheatclassifier_vit_3_dense_1_tensordot_readvariableop_resource:	@АL
=wheatclassifier_vit_3_dense_1_biasadd_readvariableop_resource:	АR
?wheatclassifier_vit_3_dense_2_tensordot_readvariableop_resource:	А@K
=wheatclassifier_vit_3_dense_2_biasadd_readvariableop_resource:@_
Qwheatclassifier_vit_3_layer_normalization_2_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_vit_3_layer_normalization_2_batchnorm_readvariableop_resource:@n
Xwheatclassifier_vit_3_multi_head_attention_1_query_einsum_einsum_readvariableop_resource:@@`
Nwheatclassifier_vit_3_multi_head_attention_1_query_add_readvariableop_resource:@l
Vwheatclassifier_vit_3_multi_head_attention_1_key_einsum_einsum_readvariableop_resource:@@^
Lwheatclassifier_vit_3_multi_head_attention_1_key_add_readvariableop_resource:@n
Xwheatclassifier_vit_3_multi_head_attention_1_value_einsum_einsum_readvariableop_resource:@@`
Nwheatclassifier_vit_3_multi_head_attention_1_value_add_readvariableop_resource:@y
cwheatclassifier_vit_3_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:@@g
Ywheatclassifier_vit_3_multi_head_attention_1_attention_output_add_readvariableop_resource:@_
Qwheatclassifier_vit_3_layer_normalization_3_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_vit_3_layer_normalization_3_batchnorm_readvariableop_resource:@R
?wheatclassifier_vit_3_dense_3_tensordot_readvariableop_resource:	@АL
=wheatclassifier_vit_3_dense_3_biasadd_readvariableop_resource:	АR
?wheatclassifier_vit_3_dense_4_tensordot_readvariableop_resource:	А@K
=wheatclassifier_vit_3_dense_4_biasadd_readvariableop_resource:@_
Qwheatclassifier_vit_3_layer_normalization_4_batchnorm_mul_readvariableop_resource:@[
Mwheatclassifier_vit_3_layer_normalization_4_batchnorm_readvariableop_resource:@P
<wheatclassifier_vit_3_dense_5_matmul_readvariableop_resource:
А»2K
=wheatclassifier_vit_3_dense_5_biasadd_readvariableop_resource:2N
<wheatclassifier_vit_3_dense_6_matmul_readvariableop_resource:22K
=wheatclassifier_vit_3_dense_6_biasadd_readvariableop_resource:2N
<wheatclassifier_vit_3_dense_7_matmul_readvariableop_resource:2K
=wheatclassifier_vit_3_dense_7_biasadd_readvariableop_resource:
identityИҐ4WheatClassifier_VIT_3/dense_1/BiasAdd/ReadVariableOpҐ6WheatClassifier_VIT_3/dense_1/Tensordot/ReadVariableOpҐ4WheatClassifier_VIT_3/dense_2/BiasAdd/ReadVariableOpҐ6WheatClassifier_VIT_3/dense_2/Tensordot/ReadVariableOpҐ4WheatClassifier_VIT_3/dense_3/BiasAdd/ReadVariableOpҐ6WheatClassifier_VIT_3/dense_3/Tensordot/ReadVariableOpҐ4WheatClassifier_VIT_3/dense_4/BiasAdd/ReadVariableOpҐ6WheatClassifier_VIT_3/dense_4/Tensordot/ReadVariableOpҐ4WheatClassifier_VIT_3/dense_5/BiasAdd/ReadVariableOpҐ3WheatClassifier_VIT_3/dense_5/MatMul/ReadVariableOpҐ4WheatClassifier_VIT_3/dense_6/BiasAdd/ReadVariableOpҐ3WheatClassifier_VIT_3/dense_6/MatMul/ReadVariableOpҐ4WheatClassifier_VIT_3/dense_7/BiasAdd/ReadVariableOpҐ3WheatClassifier_VIT_3/dense_7/MatMul/ReadVariableOpҐBWheatClassifier_VIT_3/layer_normalization/batchnorm/ReadVariableOpҐFWheatClassifier_VIT_3/layer_normalization/batchnorm/mul/ReadVariableOpҐDWheatClassifier_VIT_3/layer_normalization_1/batchnorm/ReadVariableOpҐHWheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul/ReadVariableOpҐDWheatClassifier_VIT_3/layer_normalization_2/batchnorm/ReadVariableOpҐHWheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul/ReadVariableOpҐDWheatClassifier_VIT_3/layer_normalization_3/batchnorm/ReadVariableOpҐHWheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul/ReadVariableOpҐDWheatClassifier_VIT_3/layer_normalization_4/batchnorm/ReadVariableOpҐHWheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul/ReadVariableOpҐNWheatClassifier_VIT_3/multi_head_attention/attention_output/add/ReadVariableOpҐXWheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpҐAWheatClassifier_VIT_3/multi_head_attention/key/add/ReadVariableOpҐKWheatClassifier_VIT_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpҐCWheatClassifier_VIT_3/multi_head_attention/query/add/ReadVariableOpҐMWheatClassifier_VIT_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpҐCWheatClassifier_VIT_3/multi_head_attention/value/add/ReadVariableOpҐMWheatClassifier_VIT_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpҐPWheatClassifier_VIT_3/multi_head_attention_1/attention_output/add/ReadVariableOpҐZWheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐCWheatClassifier_VIT_3/multi_head_attention_1/key/add/ReadVariableOpҐMWheatClassifier_VIT_3/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐEWheatClassifier_VIT_3/multi_head_attention_1/query/add/ReadVariableOpҐOWheatClassifier_VIT_3/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐEWheatClassifier_VIT_3/multi_head_attention_1/value/add/ReadVariableOpҐOWheatClassifier_VIT_3/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐ@WheatClassifier_VIT_3/patch_encoder/dense/BiasAdd/ReadVariableOpҐBWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ReadVariableOpҐ>WheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookupБ
#WheatClassifier_VIT_3/patches/ShapeShapeinput_1*
T0*
_output_shapes
:2%
#WheatClassifier_VIT_3/patches/Shape∞
1WheatClassifier_VIT_3/patches/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1WheatClassifier_VIT_3/patches/strided_slice/stackі
3WheatClassifier_VIT_3/patches/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_VIT_3/patches/strided_slice/stack_1і
3WheatClassifier_VIT_3/patches/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3WheatClassifier_VIT_3/patches/strided_slice/stack_2Ц
+WheatClassifier_VIT_3/patches/strided_sliceStridedSlice,WheatClassifier_VIT_3/patches/Shape:output:0:WheatClassifier_VIT_3/patches/strided_slice/stack:output:0<WheatClassifier_VIT_3/patches/strided_slice/stack_1:output:0<WheatClassifier_VIT_3/patches/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+WheatClassifier_VIT_3/patches/strided_sliceС
1WheatClassifier_VIT_3/patches/ExtractImagePatchesExtractImagePatchesinput_1*
T0*0
_output_shapes
:€€€€€€€€€ђ*
ksizes


*
paddingVALID*
rates
*
strides


23
1WheatClassifier_VIT_3/patches/ExtractImagePatches©
-WheatClassifier_VIT_3/patches/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2/
-WheatClassifier_VIT_3/patches/Reshape/shape/1°
-WheatClassifier_VIT_3/patches/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ђ2/
-WheatClassifier_VIT_3/patches/Reshape/shape/2ґ
+WheatClassifier_VIT_3/patches/Reshape/shapePack4WheatClassifier_VIT_3/patches/strided_slice:output:06WheatClassifier_VIT_3/patches/Reshape/shape/1:output:06WheatClassifier_VIT_3/patches/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2-
+WheatClassifier_VIT_3/patches/Reshape/shapeМ
%WheatClassifier_VIT_3/patches/ReshapeReshape;WheatClassifier_VIT_3/patches/ExtractImagePatches:patches:04WheatClassifier_VIT_3/patches/Reshape/shape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2'
%WheatClassifier_VIT_3/patches/Reshape§
/WheatClassifier_VIT_3/patch_encoder/range/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/WheatClassifier_VIT_3/patch_encoder/range/start•
/WheatClassifier_VIT_3/patch_encoder/range/limitConst*
_output_shapes
: *
dtype0*
value
B :Р21
/WheatClassifier_VIT_3/patch_encoder/range/limit§
/WheatClassifier_VIT_3/patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/WheatClassifier_VIT_3/patch_encoder/range/delta™
)WheatClassifier_VIT_3/patch_encoder/rangeRange8WheatClassifier_VIT_3/patch_encoder/range/start:output:08WheatClassifier_VIT_3/patch_encoder/range/limit:output:08WheatClassifier_VIT_3/patch_encoder/range/delta:output:0*
_output_shapes	
:Р2+
)WheatClassifier_VIT_3/patch_encoder/rangeХ
BWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOpKwheatclassifier_vit_3_patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
dtype02D
BWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ReadVariableOpЊ
8WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2:
8WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/axes≈
8WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2:
8WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/free‘
9WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ShapeShape.WheatClassifier_VIT_3/patches/Reshape:output:0*
T0*
_output_shapes
:2;
9WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Shape»
AWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2/axis£
<WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2GatherV2BWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Shape:output:0AWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/free:output:0JWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2>
<WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2ћ
CWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
CWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2_1/axis©
>WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2_1GatherV2BWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Shape:output:0AWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/axes:output:0LWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2@
>WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2_1ј
9WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2;
9WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Const®
8WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ProdProdEWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2:output:0BWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2:
8WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Prodƒ
;WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2=
;WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Const_1∞
:WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Prod_1ProdGWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2_1:output:0DWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2<
:WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Prod_1ƒ
?WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2A
?WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat/axisВ
:WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concatConcatV2AWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/free:output:0AWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/axes:output:0HWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2<
:WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concatі
9WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/stackPackAWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Prod:output:0CWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2;
9WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/stackј
=WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/transpose	Transpose.WheatClassifier_VIT_3/patches/Reshape:output:0CWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2?
=WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/transpose«
;WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ReshapeReshapeAWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/transpose:y:0BWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2=
;WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Reshape∆
:WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/MatMulMatMulDWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Reshape:output:0JWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2<
:WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/MatMulƒ
;WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2=
;WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Const_2»
AWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
AWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat_1/axisП
<WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat_1ConcatV2EWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/GatherV2:output:0DWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/Const_2:output:0JWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2>
<WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat_1Ѕ
3WheatClassifier_VIT_3/patch_encoder/dense/TensordotReshapeDWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/MatMul:product:0EWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@25
3WheatClassifier_VIT_3/patch_encoder/dense/TensordotК
@WheatClassifier_VIT_3/patch_encoder/dense/BiasAdd/ReadVariableOpReadVariableOpIwheatclassifier_vit_3_patch_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@WheatClassifier_VIT_3/patch_encoder/dense/BiasAdd/ReadVariableOpЄ
1WheatClassifier_VIT_3/patch_encoder/dense/BiasAddBiasAdd<WheatClassifier_VIT_3/patch_encoder/dense/Tensordot:output:0HWheatClassifier_VIT_3/patch_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€@23
1WheatClassifier_VIT_3/patch_encoder/dense/BiasAdd”
>WheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookupResourceGatherEwheatclassifier_vit_3_patch_encoder_embedding_embedding_lookup_5841752WheatClassifier_VIT_3/patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*X
_classN
LJloc:@WheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup/584175*
_output_shapes
:	Р@*
dtype02@
>WheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookupЩ
GWheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup/IdentityIdentityGWheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*X
_classN
LJloc:@WheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup/584175*
_output_shapes
:	Р@2I
GWheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup/IdentityЮ
IWheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup/Identity_1IdentityPWheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Р@2K
IWheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup/Identity_1Ґ
'WheatClassifier_VIT_3/patch_encoder/addAddV2:WheatClassifier_VIT_3/patch_encoder/dense/BiasAdd:output:0RWheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2)
'WheatClassifier_VIT_3/patch_encoder/addё
HWheatClassifier_VIT_3/layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2J
HWheatClassifier_VIT_3/layer_normalization/moments/mean/reduction_indicesј
6WheatClassifier_VIT_3/layer_normalization/moments/meanMean+WheatClassifier_VIT_3/patch_encoder/add:z:0QWheatClassifier_VIT_3/layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(28
6WheatClassifier_VIT_3/layer_normalization/moments/meanИ
>WheatClassifier_VIT_3/layer_normalization/moments/StopGradientStopGradient?WheatClassifier_VIT_3/layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2@
>WheatClassifier_VIT_3/layer_normalization/moments/StopGradientћ
CWheatClassifier_VIT_3/layer_normalization/moments/SquaredDifferenceSquaredDifference+WheatClassifier_VIT_3/patch_encoder/add:z:0GWheatClassifier_VIT_3/layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2E
CWheatClassifier_VIT_3/layer_normalization/moments/SquaredDifferenceж
LWheatClassifier_VIT_3/layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2N
LWheatClassifier_VIT_3/layer_normalization/moments/variance/reduction_indicesи
:WheatClassifier_VIT_3/layer_normalization/moments/varianceMeanGWheatClassifier_VIT_3/layer_normalization/moments/SquaredDifference:z:0UWheatClassifier_VIT_3/layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2<
:WheatClassifier_VIT_3/layer_normalization/moments/varianceї
9WheatClassifier_VIT_3/layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52;
9WheatClassifier_VIT_3/layer_normalization/batchnorm/add/yї
7WheatClassifier_VIT_3/layer_normalization/batchnorm/addAddV2CWheatClassifier_VIT_3/layer_normalization/moments/variance:output:0BWheatClassifier_VIT_3/layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р29
7WheatClassifier_VIT_3/layer_normalization/batchnorm/addу
9WheatClassifier_VIT_3/layer_normalization/batchnorm/RsqrtRsqrt;WheatClassifier_VIT_3/layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2;
9WheatClassifier_VIT_3/layer_normalization/batchnorm/RsqrtЬ
FWheatClassifier_VIT_3/layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOpOwheatclassifier_vit_3_layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02H
FWheatClassifier_VIT_3/layer_normalization/batchnorm/mul/ReadVariableOpњ
7WheatClassifier_VIT_3/layer_normalization/batchnorm/mulMul=WheatClassifier_VIT_3/layer_normalization/batchnorm/Rsqrt:y:0NWheatClassifier_VIT_3/layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@29
7WheatClassifier_VIT_3/layer_normalization/batchnorm/mulЮ
9WheatClassifier_VIT_3/layer_normalization/batchnorm/mul_1Mul+WheatClassifier_VIT_3/patch_encoder/add:z:0;WheatClassifier_VIT_3/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization/batchnorm/mul_1≤
9WheatClassifier_VIT_3/layer_normalization/batchnorm/mul_2Mul?WheatClassifier_VIT_3/layer_normalization/moments/mean:output:0;WheatClassifier_VIT_3/layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization/batchnorm/mul_2Р
BWheatClassifier_VIT_3/layer_normalization/batchnorm/ReadVariableOpReadVariableOpKwheatclassifier_vit_3_layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02D
BWheatClassifier_VIT_3/layer_normalization/batchnorm/ReadVariableOpї
7WheatClassifier_VIT_3/layer_normalization/batchnorm/subSubJWheatClassifier_VIT_3/layer_normalization/batchnorm/ReadVariableOp:value:0=WheatClassifier_VIT_3/layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@29
7WheatClassifier_VIT_3/layer_normalization/batchnorm/sub≤
9WheatClassifier_VIT_3/layer_normalization/batchnorm/add_1AddV2=WheatClassifier_VIT_3/layer_normalization/batchnorm/mul_1:z:0;WheatClassifier_VIT_3/layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization/batchnorm/add_1є
MWheatClassifier_VIT_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_vit_3_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_VIT_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpБ
>WheatClassifier_VIT_3/multi_head_attention/query/einsum/EinsumEinsum=WheatClassifier_VIT_3/layer_normalization/batchnorm/add_1:z:0UWheatClassifier_VIT_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2@
>WheatClassifier_VIT_3/multi_head_attention/query/einsum/EinsumЧ
CWheatClassifier_VIT_3/multi_head_attention/query/add/ReadVariableOpReadVariableOpLwheatclassifier_vit_3_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_VIT_3/multi_head_attention/query/add/ReadVariableOp∆
4WheatClassifier_VIT_3/multi_head_attention/query/addAddV2GWheatClassifier_VIT_3/multi_head_attention/query/einsum/Einsum:output:0KWheatClassifier_VIT_3/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@26
4WheatClassifier_VIT_3/multi_head_attention/query/add≥
KWheatClassifier_VIT_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpTwheatclassifier_vit_3_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02M
KWheatClassifier_VIT_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpы
<WheatClassifier_VIT_3/multi_head_attention/key/einsum/EinsumEinsum=WheatClassifier_VIT_3/layer_normalization/batchnorm/add_1:z:0SWheatClassifier_VIT_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2>
<WheatClassifier_VIT_3/multi_head_attention/key/einsum/EinsumС
AWheatClassifier_VIT_3/multi_head_attention/key/add/ReadVariableOpReadVariableOpJwheatclassifier_vit_3_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02C
AWheatClassifier_VIT_3/multi_head_attention/key/add/ReadVariableOpЊ
2WheatClassifier_VIT_3/multi_head_attention/key/addAddV2EWheatClassifier_VIT_3/multi_head_attention/key/einsum/Einsum:output:0IWheatClassifier_VIT_3/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@24
2WheatClassifier_VIT_3/multi_head_attention/key/addє
MWheatClassifier_VIT_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_vit_3_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_VIT_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpБ
>WheatClassifier_VIT_3/multi_head_attention/value/einsum/EinsumEinsum=WheatClassifier_VIT_3/layer_normalization/batchnorm/add_1:z:0UWheatClassifier_VIT_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2@
>WheatClassifier_VIT_3/multi_head_attention/value/einsum/EinsumЧ
CWheatClassifier_VIT_3/multi_head_attention/value/add/ReadVariableOpReadVariableOpLwheatclassifier_vit_3_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_VIT_3/multi_head_attention/value/add/ReadVariableOp∆
4WheatClassifier_VIT_3/multi_head_attention/value/addAddV2GWheatClassifier_VIT_3/multi_head_attention/value/einsum/Einsum:output:0KWheatClassifier_VIT_3/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@26
4WheatClassifier_VIT_3/multi_head_attention/value/add©
0WheatClassifier_VIT_3/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >22
0WheatClassifier_VIT_3/multi_head_attention/Mul/yЧ
.WheatClassifier_VIT_3/multi_head_attention/MulMul8WheatClassifier_VIT_3/multi_head_attention/query/add:z:09WheatClassifier_VIT_3/multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€Р@20
.WheatClassifier_VIT_3/multi_head_attention/Mulќ
8WheatClassifier_VIT_3/multi_head_attention/einsum/EinsumEinsum6WheatClassifier_VIT_3/multi_head_attention/key/add:z:02WheatClassifier_VIT_3/multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2:
8WheatClassifier_VIT_3/multi_head_attention/einsum/EinsumВ
:WheatClassifier_VIT_3/multi_head_attention/softmax/SoftmaxSoftmaxAWheatClassifier_VIT_3/multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2<
:WheatClassifier_VIT_3/multi_head_attention/softmax/SoftmaxИ
;WheatClassifier_VIT_3/multi_head_attention/dropout/IdentityIdentityDWheatClassifier_VIT_3/multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€РР2=
;WheatClassifier_VIT_3/multi_head_attention/dropout/Identityе
:WheatClassifier_VIT_3/multi_head_attention/einsum_1/EinsumEinsumDWheatClassifier_VIT_3/multi_head_attention/dropout/Identity:output:08WheatClassifier_VIT_3/multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2<
:WheatClassifier_VIT_3/multi_head_attention/einsum_1/EinsumЏ
XWheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpawheatclassifier_vit_3_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02Z
XWheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp§
IWheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/EinsumEinsumCWheatClassifier_VIT_3/multi_head_attention/einsum_1/Einsum:output:0`WheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2K
IWheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/Einsumі
NWheatClassifier_VIT_3/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpWwheatclassifier_vit_3_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02P
NWheatClassifier_VIT_3/multi_head_attention/attention_output/add/ReadVariableOpо
?WheatClassifier_VIT_3/multi_head_attention/attention_output/addAddV2RWheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/Einsum:output:0VWheatClassifier_VIT_3/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2A
?WheatClassifier_VIT_3/multi_head_attention/attention_output/addр
WheatClassifier_VIT_3/add/addAddV2CWheatClassifier_VIT_3/multi_head_attention/attention_output/add:z:0+WheatClassifier_VIT_3/patch_encoder/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
WheatClassifier_VIT_3/add/addв
JWheatClassifier_VIT_3/layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_VIT_3/layer_normalization_1/moments/mean/reduction_indicesЉ
8WheatClassifier_VIT_3/layer_normalization_1/moments/meanMean!WheatClassifier_VIT_3/add/add:z:0SWheatClassifier_VIT_3/layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2:
8WheatClassifier_VIT_3/layer_normalization_1/moments/meanО
@WheatClassifier_VIT_3/layer_normalization_1/moments/StopGradientStopGradientAWheatClassifier_VIT_3/layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2B
@WheatClassifier_VIT_3/layer_normalization_1/moments/StopGradient»
EWheatClassifier_VIT_3/layer_normalization_1/moments/SquaredDifferenceSquaredDifference!WheatClassifier_VIT_3/add/add:z:0IWheatClassifier_VIT_3/layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2G
EWheatClassifier_VIT_3/layer_normalization_1/moments/SquaredDifferenceк
NWheatClassifier_VIT_3/layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_VIT_3/layer_normalization_1/moments/variance/reduction_indicesр
<WheatClassifier_VIT_3/layer_normalization_1/moments/varianceMeanIWheatClassifier_VIT_3/layer_normalization_1/moments/SquaredDifference:z:0WWheatClassifier_VIT_3/layer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2>
<WheatClassifier_VIT_3/layer_normalization_1/moments/varianceњ
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52=
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/add/y√
9WheatClassifier_VIT_3/layer_normalization_1/batchnorm/addAddV2EWheatClassifier_VIT_3/layer_normalization_1/moments/variance:output:0DWheatClassifier_VIT_3/layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2;
9WheatClassifier_VIT_3/layer_normalization_1/batchnorm/addщ
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/RsqrtRsqrt=WheatClassifier_VIT_3/layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2=
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/RsqrtҐ
HWheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_vit_3_layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul/ReadVariableOp«
9WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mulMul?WheatClassifier_VIT_3/layer_normalization_1/batchnorm/Rsqrt:y:0PWheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mulЪ
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul_1Mul!WheatClassifier_VIT_3/add/add:z:0=WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul_1Ї
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul_2MulAWheatClassifier_VIT_3/layer_normalization_1/moments/mean:output:0=WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul_2Ц
DWheatClassifier_VIT_3/layer_normalization_1/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_vit_3_layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_VIT_3/layer_normalization_1/batchnorm/ReadVariableOp√
9WheatClassifier_VIT_3/layer_normalization_1/batchnorm/subSubLWheatClassifier_VIT_3/layer_normalization_1/batchnorm/ReadVariableOp:value:0?WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization_1/batchnorm/subЇ
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/add_1AddV2?WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul_1:z:0=WheatClassifier_VIT_3/layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_1/batchnorm/add_1с
6WheatClassifier_VIT_3/dense_1/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_vit_3_dense_1_tensordot_readvariableop_resource*
_output_shapes
:	@А*
dtype028
6WheatClassifier_VIT_3/dense_1/Tensordot/ReadVariableOp¶
,WheatClassifier_VIT_3/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_VIT_3/dense_1/Tensordot/axes≠
,WheatClassifier_VIT_3/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_VIT_3/dense_1/Tensordot/freeЌ
-WheatClassifier_VIT_3/dense_1/Tensordot/ShapeShape?WheatClassifier_VIT_3/layer_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_VIT_3/dense_1/Tensordot/Shape∞
5WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2/axisз
0WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2GatherV26WheatClassifier_VIT_3/dense_1/Tensordot/Shape:output:05WheatClassifier_VIT_3/dense_1/Tensordot/free:output:0>WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2і
7WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2_1/axisн
2WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2_1GatherV26WheatClassifier_VIT_3/dense_1/Tensordot/Shape:output:05WheatClassifier_VIT_3/dense_1/Tensordot/axes:output:0@WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2_1®
-WheatClassifier_VIT_3/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_VIT_3/dense_1/Tensordot/Constш
,WheatClassifier_VIT_3/dense_1/Tensordot/ProdProd9WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2:output:06WheatClassifier_VIT_3/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_VIT_3/dense_1/Tensordot/Prodђ
/WheatClassifier_VIT_3/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_VIT_3/dense_1/Tensordot/Const_1А
.WheatClassifier_VIT_3/dense_1/Tensordot/Prod_1Prod;WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2_1:output:08WheatClassifier_VIT_3/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_VIT_3/dense_1/Tensordot/Prod_1ђ
3WheatClassifier_VIT_3/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_VIT_3/dense_1/Tensordot/concat/axis∆
.WheatClassifier_VIT_3/dense_1/Tensordot/concatConcatV25WheatClassifier_VIT_3/dense_1/Tensordot/free:output:05WheatClassifier_VIT_3/dense_1/Tensordot/axes:output:0<WheatClassifier_VIT_3/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_VIT_3/dense_1/Tensordot/concatД
-WheatClassifier_VIT_3/dense_1/Tensordot/stackPack5WheatClassifier_VIT_3/dense_1/Tensordot/Prod:output:07WheatClassifier_VIT_3/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_VIT_3/dense_1/Tensordot/stack§
1WheatClassifier_VIT_3/dense_1/Tensordot/transpose	Transpose?WheatClassifier_VIT_3/layer_normalization_1/batchnorm/add_1:z:07WheatClassifier_VIT_3/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@23
1WheatClassifier_VIT_3/dense_1/Tensordot/transposeЧ
/WheatClassifier_VIT_3/dense_1/Tensordot/ReshapeReshape5WheatClassifier_VIT_3/dense_1/Tensordot/transpose:y:06WheatClassifier_VIT_3/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€21
/WheatClassifier_VIT_3/dense_1/Tensordot/ReshapeЧ
.WheatClassifier_VIT_3/dense_1/Tensordot/MatMulMatMul8WheatClassifier_VIT_3/dense_1/Tensordot/Reshape:output:0>WheatClassifier_VIT_3/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А20
.WheatClassifier_VIT_3/dense_1/Tensordot/MatMul≠
/WheatClassifier_VIT_3/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А21
/WheatClassifier_VIT_3/dense_1/Tensordot/Const_2∞
5WheatClassifier_VIT_3/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_VIT_3/dense_1/Tensordot/concat_1/axis”
0WheatClassifier_VIT_3/dense_1/Tensordot/concat_1ConcatV29WheatClassifier_VIT_3/dense_1/Tensordot/GatherV2:output:08WheatClassifier_VIT_3/dense_1/Tensordot/Const_2:output:0>WheatClassifier_VIT_3/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_VIT_3/dense_1/Tensordot/concat_1К
'WheatClassifier_VIT_3/dense_1/TensordotReshape8WheatClassifier_VIT_3/dense_1/Tensordot/MatMul:product:09WheatClassifier_VIT_3/dense_1/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2)
'WheatClassifier_VIT_3/dense_1/Tensordotз
4WheatClassifier_VIT_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_vit_3_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype026
4WheatClassifier_VIT_3/dense_1/BiasAdd/ReadVariableOpБ
%WheatClassifier_VIT_3/dense_1/BiasAddBiasAdd0WheatClassifier_VIT_3/dense_1/Tensordot:output:0<WheatClassifier_VIT_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2'
%WheatClassifier_VIT_3/dense_1/BiasAddЩ
(WheatClassifier_VIT_3/dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_VIT_3/dense_1/Gelu/mul/xт
&WheatClassifier_VIT_3/dense_1/Gelu/mulMul1WheatClassifier_VIT_3/dense_1/Gelu/mul/x:output:0.WheatClassifier_VIT_3/dense_1/BiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2(
&WheatClassifier_VIT_3/dense_1/Gelu/mulЫ
)WheatClassifier_VIT_3/dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2+
)WheatClassifier_VIT_3/dense_1/Gelu/Cast/x€
*WheatClassifier_VIT_3/dense_1/Gelu/truedivRealDiv.WheatClassifier_VIT_3/dense_1/BiasAdd:output:02WheatClassifier_VIT_3/dense_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2,
*WheatClassifier_VIT_3/dense_1/Gelu/truedivњ
&WheatClassifier_VIT_3/dense_1/Gelu/ErfErf.WheatClassifier_VIT_3/dense_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2(
&WheatClassifier_VIT_3/dense_1/Gelu/ErfЩ
(WheatClassifier_VIT_3/dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(WheatClassifier_VIT_3/dense_1/Gelu/add/xр
&WheatClassifier_VIT_3/dense_1/Gelu/addAddV21WheatClassifier_VIT_3/dense_1/Gelu/add/x:output:0*WheatClassifier_VIT_3/dense_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2(
&WheatClassifier_VIT_3/dense_1/Gelu/addл
(WheatClassifier_VIT_3/dense_1/Gelu/mul_1Mul*WheatClassifier_VIT_3/dense_1/Gelu/mul:z:0*WheatClassifier_VIT_3/dense_1/Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2*
(WheatClassifier_VIT_3/dense_1/Gelu/mul_1с
6WheatClassifier_VIT_3/dense_2/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_vit_3_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	А@*
dtype028
6WheatClassifier_VIT_3/dense_2/Tensordot/ReadVariableOp¶
,WheatClassifier_VIT_3/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_VIT_3/dense_2/Tensordot/axes≠
,WheatClassifier_VIT_3/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_VIT_3/dense_2/Tensordot/freeЇ
-WheatClassifier_VIT_3/dense_2/Tensordot/ShapeShape,WheatClassifier_VIT_3/dense_1/Gelu/mul_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_VIT_3/dense_2/Tensordot/Shape∞
5WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2/axisз
0WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2GatherV26WheatClassifier_VIT_3/dense_2/Tensordot/Shape:output:05WheatClassifier_VIT_3/dense_2/Tensordot/free:output:0>WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2і
7WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2_1/axisн
2WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2_1GatherV26WheatClassifier_VIT_3/dense_2/Tensordot/Shape:output:05WheatClassifier_VIT_3/dense_2/Tensordot/axes:output:0@WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2_1®
-WheatClassifier_VIT_3/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_VIT_3/dense_2/Tensordot/Constш
,WheatClassifier_VIT_3/dense_2/Tensordot/ProdProd9WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2:output:06WheatClassifier_VIT_3/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_VIT_3/dense_2/Tensordot/Prodђ
/WheatClassifier_VIT_3/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_VIT_3/dense_2/Tensordot/Const_1А
.WheatClassifier_VIT_3/dense_2/Tensordot/Prod_1Prod;WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2_1:output:08WheatClassifier_VIT_3/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_VIT_3/dense_2/Tensordot/Prod_1ђ
3WheatClassifier_VIT_3/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_VIT_3/dense_2/Tensordot/concat/axis∆
.WheatClassifier_VIT_3/dense_2/Tensordot/concatConcatV25WheatClassifier_VIT_3/dense_2/Tensordot/free:output:05WheatClassifier_VIT_3/dense_2/Tensordot/axes:output:0<WheatClassifier_VIT_3/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_VIT_3/dense_2/Tensordot/concatД
-WheatClassifier_VIT_3/dense_2/Tensordot/stackPack5WheatClassifier_VIT_3/dense_2/Tensordot/Prod:output:07WheatClassifier_VIT_3/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_VIT_3/dense_2/Tensordot/stackТ
1WheatClassifier_VIT_3/dense_2/Tensordot/transpose	Transpose,WheatClassifier_VIT_3/dense_1/Gelu/mul_1:z:07WheatClassifier_VIT_3/dense_2/Tensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА23
1WheatClassifier_VIT_3/dense_2/Tensordot/transposeЧ
/WheatClassifier_VIT_3/dense_2/Tensordot/ReshapeReshape5WheatClassifier_VIT_3/dense_2/Tensordot/transpose:y:06WheatClassifier_VIT_3/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€21
/WheatClassifier_VIT_3/dense_2/Tensordot/ReshapeЦ
.WheatClassifier_VIT_3/dense_2/Tensordot/MatMulMatMul8WheatClassifier_VIT_3/dense_2/Tensordot/Reshape:output:0>WheatClassifier_VIT_3/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@20
.WheatClassifier_VIT_3/dense_2/Tensordot/MatMulђ
/WheatClassifier_VIT_3/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@21
/WheatClassifier_VIT_3/dense_2/Tensordot/Const_2∞
5WheatClassifier_VIT_3/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_VIT_3/dense_2/Tensordot/concat_1/axis”
0WheatClassifier_VIT_3/dense_2/Tensordot/concat_1ConcatV29WheatClassifier_VIT_3/dense_2/Tensordot/GatherV2:output:08WheatClassifier_VIT_3/dense_2/Tensordot/Const_2:output:0>WheatClassifier_VIT_3/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_VIT_3/dense_2/Tensordot/concat_1Й
'WheatClassifier_VIT_3/dense_2/TensordotReshape8WheatClassifier_VIT_3/dense_2/Tensordot/MatMul:product:09WheatClassifier_VIT_3/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2)
'WheatClassifier_VIT_3/dense_2/Tensordotж
4WheatClassifier_VIT_3/dense_2/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_vit_3_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4WheatClassifier_VIT_3/dense_2/BiasAdd/ReadVariableOpА
%WheatClassifier_VIT_3/dense_2/BiasAddBiasAdd0WheatClassifier_VIT_3/dense_2/Tensordot:output:0<WheatClassifier_VIT_3/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%WheatClassifier_VIT_3/dense_2/BiasAddЩ
(WheatClassifier_VIT_3/dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_VIT_3/dense_2/Gelu/mul/xс
&WheatClassifier_VIT_3/dense_2/Gelu/mulMul1WheatClassifier_VIT_3/dense_2/Gelu/mul/x:output:0.WheatClassifier_VIT_3/dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2(
&WheatClassifier_VIT_3/dense_2/Gelu/mulЫ
)WheatClassifier_VIT_3/dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2+
)WheatClassifier_VIT_3/dense_2/Gelu/Cast/xю
*WheatClassifier_VIT_3/dense_2/Gelu/truedivRealDiv.WheatClassifier_VIT_3/dense_2/BiasAdd:output:02WheatClassifier_VIT_3/dense_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2,
*WheatClassifier_VIT_3/dense_2/Gelu/truedivЊ
&WheatClassifier_VIT_3/dense_2/Gelu/ErfErf.WheatClassifier_VIT_3/dense_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2(
&WheatClassifier_VIT_3/dense_2/Gelu/ErfЩ
(WheatClassifier_VIT_3/dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(WheatClassifier_VIT_3/dense_2/Gelu/add/xп
&WheatClassifier_VIT_3/dense_2/Gelu/addAddV21WheatClassifier_VIT_3/dense_2/Gelu/add/x:output:0*WheatClassifier_VIT_3/dense_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2(
&WheatClassifier_VIT_3/dense_2/Gelu/addк
(WheatClassifier_VIT_3/dense_2/Gelu/mul_1Mul*WheatClassifier_VIT_3/dense_2/Gelu/mul:z:0*WheatClassifier_VIT_3/dense_2/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2*
(WheatClassifier_VIT_3/dense_2/Gelu/mul_1”
WheatClassifier_VIT_3/add_1/addAddV2,WheatClassifier_VIT_3/dense_2/Gelu/mul_1:z:0!WheatClassifier_VIT_3/add/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2!
WheatClassifier_VIT_3/add_1/addв
JWheatClassifier_VIT_3/layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_VIT_3/layer_normalization_2/moments/mean/reduction_indicesЊ
8WheatClassifier_VIT_3/layer_normalization_2/moments/meanMean#WheatClassifier_VIT_3/add_1/add:z:0SWheatClassifier_VIT_3/layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2:
8WheatClassifier_VIT_3/layer_normalization_2/moments/meanО
@WheatClassifier_VIT_3/layer_normalization_2/moments/StopGradientStopGradientAWheatClassifier_VIT_3/layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2B
@WheatClassifier_VIT_3/layer_normalization_2/moments/StopGradient 
EWheatClassifier_VIT_3/layer_normalization_2/moments/SquaredDifferenceSquaredDifference#WheatClassifier_VIT_3/add_1/add:z:0IWheatClassifier_VIT_3/layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2G
EWheatClassifier_VIT_3/layer_normalization_2/moments/SquaredDifferenceк
NWheatClassifier_VIT_3/layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_VIT_3/layer_normalization_2/moments/variance/reduction_indicesр
<WheatClassifier_VIT_3/layer_normalization_2/moments/varianceMeanIWheatClassifier_VIT_3/layer_normalization_2/moments/SquaredDifference:z:0WWheatClassifier_VIT_3/layer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2>
<WheatClassifier_VIT_3/layer_normalization_2/moments/varianceњ
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52=
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/add/y√
9WheatClassifier_VIT_3/layer_normalization_2/batchnorm/addAddV2EWheatClassifier_VIT_3/layer_normalization_2/moments/variance:output:0DWheatClassifier_VIT_3/layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2;
9WheatClassifier_VIT_3/layer_normalization_2/batchnorm/addщ
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/RsqrtRsqrt=WheatClassifier_VIT_3/layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2=
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/RsqrtҐ
HWheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_vit_3_layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul/ReadVariableOp«
9WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mulMul?WheatClassifier_VIT_3/layer_normalization_2/batchnorm/Rsqrt:y:0PWheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mulЬ
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul_1Mul#WheatClassifier_VIT_3/add_1/add:z:0=WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul_1Ї
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul_2MulAWheatClassifier_VIT_3/layer_normalization_2/moments/mean:output:0=WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul_2Ц
DWheatClassifier_VIT_3/layer_normalization_2/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_vit_3_layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_VIT_3/layer_normalization_2/batchnorm/ReadVariableOp√
9WheatClassifier_VIT_3/layer_normalization_2/batchnorm/subSubLWheatClassifier_VIT_3/layer_normalization_2/batchnorm/ReadVariableOp:value:0?WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization_2/batchnorm/subЇ
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/add_1AddV2?WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul_1:z:0=WheatClassifier_VIT_3/layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_2/batchnorm/add_1њ
OWheatClassifier_VIT_3/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpXwheatclassifier_vit_3_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02Q
OWheatClassifier_VIT_3/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpЙ
@WheatClassifier_VIT_3/multi_head_attention_1/query/einsum/EinsumEinsum?WheatClassifier_VIT_3/layer_normalization_2/batchnorm/add_1:z:0WWheatClassifier_VIT_3/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2B
@WheatClassifier_VIT_3/multi_head_attention_1/query/einsum/EinsumЭ
EWheatClassifier_VIT_3/multi_head_attention_1/query/add/ReadVariableOpReadVariableOpNwheatclassifier_vit_3_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02G
EWheatClassifier_VIT_3/multi_head_attention_1/query/add/ReadVariableOpќ
6WheatClassifier_VIT_3/multi_head_attention_1/query/addAddV2IWheatClassifier_VIT_3/multi_head_attention_1/query/einsum/Einsum:output:0MWheatClassifier_VIT_3/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@28
6WheatClassifier_VIT_3/multi_head_attention_1/query/addє
MWheatClassifier_VIT_3/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpVwheatclassifier_vit_3_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02O
MWheatClassifier_VIT_3/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpГ
>WheatClassifier_VIT_3/multi_head_attention_1/key/einsum/EinsumEinsum?WheatClassifier_VIT_3/layer_normalization_2/batchnorm/add_1:z:0UWheatClassifier_VIT_3/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2@
>WheatClassifier_VIT_3/multi_head_attention_1/key/einsum/EinsumЧ
CWheatClassifier_VIT_3/multi_head_attention_1/key/add/ReadVariableOpReadVariableOpLwheatclassifier_vit_3_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02E
CWheatClassifier_VIT_3/multi_head_attention_1/key/add/ReadVariableOp∆
4WheatClassifier_VIT_3/multi_head_attention_1/key/addAddV2GWheatClassifier_VIT_3/multi_head_attention_1/key/einsum/Einsum:output:0KWheatClassifier_VIT_3/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@26
4WheatClassifier_VIT_3/multi_head_attention_1/key/addњ
OWheatClassifier_VIT_3/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpXwheatclassifier_vit_3_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02Q
OWheatClassifier_VIT_3/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpЙ
@WheatClassifier_VIT_3/multi_head_attention_1/value/einsum/EinsumEinsum?WheatClassifier_VIT_3/layer_normalization_2/batchnorm/add_1:z:0WWheatClassifier_VIT_3/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2B
@WheatClassifier_VIT_3/multi_head_attention_1/value/einsum/EinsumЭ
EWheatClassifier_VIT_3/multi_head_attention_1/value/add/ReadVariableOpReadVariableOpNwheatclassifier_vit_3_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02G
EWheatClassifier_VIT_3/multi_head_attention_1/value/add/ReadVariableOpќ
6WheatClassifier_VIT_3/multi_head_attention_1/value/addAddV2IWheatClassifier_VIT_3/multi_head_attention_1/value/einsum/Einsum:output:0MWheatClassifier_VIT_3/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@28
6WheatClassifier_VIT_3/multi_head_attention_1/value/add≠
2WheatClassifier_VIT_3/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >24
2WheatClassifier_VIT_3/multi_head_attention_1/Mul/yЯ
0WheatClassifier_VIT_3/multi_head_attention_1/MulMul:WheatClassifier_VIT_3/multi_head_attention_1/query/add:z:0;WheatClassifier_VIT_3/multi_head_attention_1/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€Р@22
0WheatClassifier_VIT_3/multi_head_attention_1/Mul÷
:WheatClassifier_VIT_3/multi_head_attention_1/einsum/EinsumEinsum8WheatClassifier_VIT_3/multi_head_attention_1/key/add:z:04WheatClassifier_VIT_3/multi_head_attention_1/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2<
:WheatClassifier_VIT_3/multi_head_attention_1/einsum/EinsumИ
<WheatClassifier_VIT_3/multi_head_attention_1/softmax/SoftmaxSoftmaxCWheatClassifier_VIT_3/multi_head_attention_1/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2>
<WheatClassifier_VIT_3/multi_head_attention_1/softmax/SoftmaxО
=WheatClassifier_VIT_3/multi_head_attention_1/dropout/IdentityIdentityFWheatClassifier_VIT_3/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€РР2?
=WheatClassifier_VIT_3/multi_head_attention_1/dropout/Identityн
<WheatClassifier_VIT_3/multi_head_attention_1/einsum_1/EinsumEinsumFWheatClassifier_VIT_3/multi_head_attention_1/dropout/Identity:output:0:WheatClassifier_VIT_3/multi_head_attention_1/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2>
<WheatClassifier_VIT_3/multi_head_attention_1/einsum_1/Einsumа
ZWheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpcwheatclassifier_vit_3_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02\
ZWheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpђ
KWheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/EinsumEinsumEWheatClassifier_VIT_3/multi_head_attention_1/einsum_1/Einsum:output:0bWheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2M
KWheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/EinsumЇ
PWheatClassifier_VIT_3/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpYwheatclassifier_vit_3_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02R
PWheatClassifier_VIT_3/multi_head_attention_1/attention_output/add/ReadVariableOpц
AWheatClassifier_VIT_3/multi_head_attention_1/attention_output/addAddV2TWheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/Einsum:output:0XWheatClassifier_VIT_3/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2C
AWheatClassifier_VIT_3/multi_head_attention_1/attention_output/addо
WheatClassifier_VIT_3/add_2/addAddV2EWheatClassifier_VIT_3/multi_head_attention_1/attention_output/add:z:0#WheatClassifier_VIT_3/add_1/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2!
WheatClassifier_VIT_3/add_2/addв
JWheatClassifier_VIT_3/layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_VIT_3/layer_normalization_3/moments/mean/reduction_indicesЊ
8WheatClassifier_VIT_3/layer_normalization_3/moments/meanMean#WheatClassifier_VIT_3/add_2/add:z:0SWheatClassifier_VIT_3/layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2:
8WheatClassifier_VIT_3/layer_normalization_3/moments/meanО
@WheatClassifier_VIT_3/layer_normalization_3/moments/StopGradientStopGradientAWheatClassifier_VIT_3/layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2B
@WheatClassifier_VIT_3/layer_normalization_3/moments/StopGradient 
EWheatClassifier_VIT_3/layer_normalization_3/moments/SquaredDifferenceSquaredDifference#WheatClassifier_VIT_3/add_2/add:z:0IWheatClassifier_VIT_3/layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2G
EWheatClassifier_VIT_3/layer_normalization_3/moments/SquaredDifferenceк
NWheatClassifier_VIT_3/layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_VIT_3/layer_normalization_3/moments/variance/reduction_indicesр
<WheatClassifier_VIT_3/layer_normalization_3/moments/varianceMeanIWheatClassifier_VIT_3/layer_normalization_3/moments/SquaredDifference:z:0WWheatClassifier_VIT_3/layer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2>
<WheatClassifier_VIT_3/layer_normalization_3/moments/varianceњ
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52=
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/add/y√
9WheatClassifier_VIT_3/layer_normalization_3/batchnorm/addAddV2EWheatClassifier_VIT_3/layer_normalization_3/moments/variance:output:0DWheatClassifier_VIT_3/layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2;
9WheatClassifier_VIT_3/layer_normalization_3/batchnorm/addщ
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/RsqrtRsqrt=WheatClassifier_VIT_3/layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2=
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/RsqrtҐ
HWheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_vit_3_layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul/ReadVariableOp«
9WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mulMul?WheatClassifier_VIT_3/layer_normalization_3/batchnorm/Rsqrt:y:0PWheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mulЬ
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul_1Mul#WheatClassifier_VIT_3/add_2/add:z:0=WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul_1Ї
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul_2MulAWheatClassifier_VIT_3/layer_normalization_3/moments/mean:output:0=WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul_2Ц
DWheatClassifier_VIT_3/layer_normalization_3/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_vit_3_layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_VIT_3/layer_normalization_3/batchnorm/ReadVariableOp√
9WheatClassifier_VIT_3/layer_normalization_3/batchnorm/subSubLWheatClassifier_VIT_3/layer_normalization_3/batchnorm/ReadVariableOp:value:0?WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization_3/batchnorm/subЇ
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/add_1AddV2?WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul_1:z:0=WheatClassifier_VIT_3/layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_3/batchnorm/add_1с
6WheatClassifier_VIT_3/dense_3/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_vit_3_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@А*
dtype028
6WheatClassifier_VIT_3/dense_3/Tensordot/ReadVariableOp¶
,WheatClassifier_VIT_3/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_VIT_3/dense_3/Tensordot/axes≠
,WheatClassifier_VIT_3/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_VIT_3/dense_3/Tensordot/freeЌ
-WheatClassifier_VIT_3/dense_3/Tensordot/ShapeShape?WheatClassifier_VIT_3/layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_VIT_3/dense_3/Tensordot/Shape∞
5WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2/axisз
0WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2GatherV26WheatClassifier_VIT_3/dense_3/Tensordot/Shape:output:05WheatClassifier_VIT_3/dense_3/Tensordot/free:output:0>WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2і
7WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2_1/axisн
2WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2_1GatherV26WheatClassifier_VIT_3/dense_3/Tensordot/Shape:output:05WheatClassifier_VIT_3/dense_3/Tensordot/axes:output:0@WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2_1®
-WheatClassifier_VIT_3/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_VIT_3/dense_3/Tensordot/Constш
,WheatClassifier_VIT_3/dense_3/Tensordot/ProdProd9WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2:output:06WheatClassifier_VIT_3/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_VIT_3/dense_3/Tensordot/Prodђ
/WheatClassifier_VIT_3/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_VIT_3/dense_3/Tensordot/Const_1А
.WheatClassifier_VIT_3/dense_3/Tensordot/Prod_1Prod;WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2_1:output:08WheatClassifier_VIT_3/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_VIT_3/dense_3/Tensordot/Prod_1ђ
3WheatClassifier_VIT_3/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_VIT_3/dense_3/Tensordot/concat/axis∆
.WheatClassifier_VIT_3/dense_3/Tensordot/concatConcatV25WheatClassifier_VIT_3/dense_3/Tensordot/free:output:05WheatClassifier_VIT_3/dense_3/Tensordot/axes:output:0<WheatClassifier_VIT_3/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_VIT_3/dense_3/Tensordot/concatД
-WheatClassifier_VIT_3/dense_3/Tensordot/stackPack5WheatClassifier_VIT_3/dense_3/Tensordot/Prod:output:07WheatClassifier_VIT_3/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_VIT_3/dense_3/Tensordot/stack§
1WheatClassifier_VIT_3/dense_3/Tensordot/transpose	Transpose?WheatClassifier_VIT_3/layer_normalization_3/batchnorm/add_1:z:07WheatClassifier_VIT_3/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@23
1WheatClassifier_VIT_3/dense_3/Tensordot/transposeЧ
/WheatClassifier_VIT_3/dense_3/Tensordot/ReshapeReshape5WheatClassifier_VIT_3/dense_3/Tensordot/transpose:y:06WheatClassifier_VIT_3/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€21
/WheatClassifier_VIT_3/dense_3/Tensordot/ReshapeЧ
.WheatClassifier_VIT_3/dense_3/Tensordot/MatMulMatMul8WheatClassifier_VIT_3/dense_3/Tensordot/Reshape:output:0>WheatClassifier_VIT_3/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А20
.WheatClassifier_VIT_3/dense_3/Tensordot/MatMul≠
/WheatClassifier_VIT_3/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А21
/WheatClassifier_VIT_3/dense_3/Tensordot/Const_2∞
5WheatClassifier_VIT_3/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_VIT_3/dense_3/Tensordot/concat_1/axis”
0WheatClassifier_VIT_3/dense_3/Tensordot/concat_1ConcatV29WheatClassifier_VIT_3/dense_3/Tensordot/GatherV2:output:08WheatClassifier_VIT_3/dense_3/Tensordot/Const_2:output:0>WheatClassifier_VIT_3/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_VIT_3/dense_3/Tensordot/concat_1К
'WheatClassifier_VIT_3/dense_3/TensordotReshape8WheatClassifier_VIT_3/dense_3/Tensordot/MatMul:product:09WheatClassifier_VIT_3/dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2)
'WheatClassifier_VIT_3/dense_3/Tensordotз
4WheatClassifier_VIT_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_vit_3_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype026
4WheatClassifier_VIT_3/dense_3/BiasAdd/ReadVariableOpБ
%WheatClassifier_VIT_3/dense_3/BiasAddBiasAdd0WheatClassifier_VIT_3/dense_3/Tensordot:output:0<WheatClassifier_VIT_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2'
%WheatClassifier_VIT_3/dense_3/BiasAddЩ
(WheatClassifier_VIT_3/dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_VIT_3/dense_3/Gelu/mul/xт
&WheatClassifier_VIT_3/dense_3/Gelu/mulMul1WheatClassifier_VIT_3/dense_3/Gelu/mul/x:output:0.WheatClassifier_VIT_3/dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2(
&WheatClassifier_VIT_3/dense_3/Gelu/mulЫ
)WheatClassifier_VIT_3/dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2+
)WheatClassifier_VIT_3/dense_3/Gelu/Cast/x€
*WheatClassifier_VIT_3/dense_3/Gelu/truedivRealDiv.WheatClassifier_VIT_3/dense_3/BiasAdd:output:02WheatClassifier_VIT_3/dense_3/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2,
*WheatClassifier_VIT_3/dense_3/Gelu/truedivњ
&WheatClassifier_VIT_3/dense_3/Gelu/ErfErf.WheatClassifier_VIT_3/dense_3/Gelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2(
&WheatClassifier_VIT_3/dense_3/Gelu/ErfЩ
(WheatClassifier_VIT_3/dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(WheatClassifier_VIT_3/dense_3/Gelu/add/xр
&WheatClassifier_VIT_3/dense_3/Gelu/addAddV21WheatClassifier_VIT_3/dense_3/Gelu/add/x:output:0*WheatClassifier_VIT_3/dense_3/Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2(
&WheatClassifier_VIT_3/dense_3/Gelu/addл
(WheatClassifier_VIT_3/dense_3/Gelu/mul_1Mul*WheatClassifier_VIT_3/dense_3/Gelu/mul:z:0*WheatClassifier_VIT_3/dense_3/Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2*
(WheatClassifier_VIT_3/dense_3/Gelu/mul_1с
6WheatClassifier_VIT_3/dense_4/Tensordot/ReadVariableOpReadVariableOp?wheatclassifier_vit_3_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	А@*
dtype028
6WheatClassifier_VIT_3/dense_4/Tensordot/ReadVariableOp¶
,WheatClassifier_VIT_3/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2.
,WheatClassifier_VIT_3/dense_4/Tensordot/axes≠
,WheatClassifier_VIT_3/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2.
,WheatClassifier_VIT_3/dense_4/Tensordot/freeЇ
-WheatClassifier_VIT_3/dense_4/Tensordot/ShapeShape,WheatClassifier_VIT_3/dense_3/Gelu/mul_1:z:0*
T0*
_output_shapes
:2/
-WheatClassifier_VIT_3/dense_4/Tensordot/Shape∞
5WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2/axisз
0WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2GatherV26WheatClassifier_VIT_3/dense_4/Tensordot/Shape:output:05WheatClassifier_VIT_3/dense_4/Tensordot/free:output:0>WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2і
7WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2_1/axisн
2WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2_1GatherV26WheatClassifier_VIT_3/dense_4/Tensordot/Shape:output:05WheatClassifier_VIT_3/dense_4/Tensordot/axes:output:0@WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2_1®
-WheatClassifier_VIT_3/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-WheatClassifier_VIT_3/dense_4/Tensordot/Constш
,WheatClassifier_VIT_3/dense_4/Tensordot/ProdProd9WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2:output:06WheatClassifier_VIT_3/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2.
,WheatClassifier_VIT_3/dense_4/Tensordot/Prodђ
/WheatClassifier_VIT_3/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/WheatClassifier_VIT_3/dense_4/Tensordot/Const_1А
.WheatClassifier_VIT_3/dense_4/Tensordot/Prod_1Prod;WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2_1:output:08WheatClassifier_VIT_3/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 20
.WheatClassifier_VIT_3/dense_4/Tensordot/Prod_1ђ
3WheatClassifier_VIT_3/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3WheatClassifier_VIT_3/dense_4/Tensordot/concat/axis∆
.WheatClassifier_VIT_3/dense_4/Tensordot/concatConcatV25WheatClassifier_VIT_3/dense_4/Tensordot/free:output:05WheatClassifier_VIT_3/dense_4/Tensordot/axes:output:0<WheatClassifier_VIT_3/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:20
.WheatClassifier_VIT_3/dense_4/Tensordot/concatД
-WheatClassifier_VIT_3/dense_4/Tensordot/stackPack5WheatClassifier_VIT_3/dense_4/Tensordot/Prod:output:07WheatClassifier_VIT_3/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2/
-WheatClassifier_VIT_3/dense_4/Tensordot/stackТ
1WheatClassifier_VIT_3/dense_4/Tensordot/transpose	Transpose,WheatClassifier_VIT_3/dense_3/Gelu/mul_1:z:07WheatClassifier_VIT_3/dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА23
1WheatClassifier_VIT_3/dense_4/Tensordot/transposeЧ
/WheatClassifier_VIT_3/dense_4/Tensordot/ReshapeReshape5WheatClassifier_VIT_3/dense_4/Tensordot/transpose:y:06WheatClassifier_VIT_3/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€21
/WheatClassifier_VIT_3/dense_4/Tensordot/ReshapeЦ
.WheatClassifier_VIT_3/dense_4/Tensordot/MatMulMatMul8WheatClassifier_VIT_3/dense_4/Tensordot/Reshape:output:0>WheatClassifier_VIT_3/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@20
.WheatClassifier_VIT_3/dense_4/Tensordot/MatMulђ
/WheatClassifier_VIT_3/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@21
/WheatClassifier_VIT_3/dense_4/Tensordot/Const_2∞
5WheatClassifier_VIT_3/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5WheatClassifier_VIT_3/dense_4/Tensordot/concat_1/axis”
0WheatClassifier_VIT_3/dense_4/Tensordot/concat_1ConcatV29WheatClassifier_VIT_3/dense_4/Tensordot/GatherV2:output:08WheatClassifier_VIT_3/dense_4/Tensordot/Const_2:output:0>WheatClassifier_VIT_3/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:22
0WheatClassifier_VIT_3/dense_4/Tensordot/concat_1Й
'WheatClassifier_VIT_3/dense_4/TensordotReshape8WheatClassifier_VIT_3/dense_4/Tensordot/MatMul:product:09WheatClassifier_VIT_3/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2)
'WheatClassifier_VIT_3/dense_4/Tensordotж
4WheatClassifier_VIT_3/dense_4/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_vit_3_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4WheatClassifier_VIT_3/dense_4/BiasAdd/ReadVariableOpА
%WheatClassifier_VIT_3/dense_4/BiasAddBiasAdd0WheatClassifier_VIT_3/dense_4/Tensordot:output:0<WheatClassifier_VIT_3/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%WheatClassifier_VIT_3/dense_4/BiasAddЩ
(WheatClassifier_VIT_3/dense_4/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_VIT_3/dense_4/Gelu/mul/xс
&WheatClassifier_VIT_3/dense_4/Gelu/mulMul1WheatClassifier_VIT_3/dense_4/Gelu/mul/x:output:0.WheatClassifier_VIT_3/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2(
&WheatClassifier_VIT_3/dense_4/Gelu/mulЫ
)WheatClassifier_VIT_3/dense_4/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2+
)WheatClassifier_VIT_3/dense_4/Gelu/Cast/xю
*WheatClassifier_VIT_3/dense_4/Gelu/truedivRealDiv.WheatClassifier_VIT_3/dense_4/BiasAdd:output:02WheatClassifier_VIT_3/dense_4/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2,
*WheatClassifier_VIT_3/dense_4/Gelu/truedivЊ
&WheatClassifier_VIT_3/dense_4/Gelu/ErfErf.WheatClassifier_VIT_3/dense_4/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2(
&WheatClassifier_VIT_3/dense_4/Gelu/ErfЩ
(WheatClassifier_VIT_3/dense_4/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(WheatClassifier_VIT_3/dense_4/Gelu/add/xп
&WheatClassifier_VIT_3/dense_4/Gelu/addAddV21WheatClassifier_VIT_3/dense_4/Gelu/add/x:output:0*WheatClassifier_VIT_3/dense_4/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2(
&WheatClassifier_VIT_3/dense_4/Gelu/addк
(WheatClassifier_VIT_3/dense_4/Gelu/mul_1Mul*WheatClassifier_VIT_3/dense_4/Gelu/mul:z:0*WheatClassifier_VIT_3/dense_4/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2*
(WheatClassifier_VIT_3/dense_4/Gelu/mul_1’
WheatClassifier_VIT_3/add_3/addAddV2,WheatClassifier_VIT_3/dense_4/Gelu/mul_1:z:0#WheatClassifier_VIT_3/add_2/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2!
WheatClassifier_VIT_3/add_3/addв
JWheatClassifier_VIT_3/layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2L
JWheatClassifier_VIT_3/layer_normalization_4/moments/mean/reduction_indicesЊ
8WheatClassifier_VIT_3/layer_normalization_4/moments/meanMean#WheatClassifier_VIT_3/add_3/add:z:0SWheatClassifier_VIT_3/layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2:
8WheatClassifier_VIT_3/layer_normalization_4/moments/meanО
@WheatClassifier_VIT_3/layer_normalization_4/moments/StopGradientStopGradientAWheatClassifier_VIT_3/layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2B
@WheatClassifier_VIT_3/layer_normalization_4/moments/StopGradient 
EWheatClassifier_VIT_3/layer_normalization_4/moments/SquaredDifferenceSquaredDifference#WheatClassifier_VIT_3/add_3/add:z:0IWheatClassifier_VIT_3/layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2G
EWheatClassifier_VIT_3/layer_normalization_4/moments/SquaredDifferenceк
NWheatClassifier_VIT_3/layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2P
NWheatClassifier_VIT_3/layer_normalization_4/moments/variance/reduction_indicesр
<WheatClassifier_VIT_3/layer_normalization_4/moments/varianceMeanIWheatClassifier_VIT_3/layer_normalization_4/moments/SquaredDifference:z:0WWheatClassifier_VIT_3/layer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2>
<WheatClassifier_VIT_3/layer_normalization_4/moments/varianceњ
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52=
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/add/y√
9WheatClassifier_VIT_3/layer_normalization_4/batchnorm/addAddV2EWheatClassifier_VIT_3/layer_normalization_4/moments/variance:output:0DWheatClassifier_VIT_3/layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2;
9WheatClassifier_VIT_3/layer_normalization_4/batchnorm/addщ
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/RsqrtRsqrt=WheatClassifier_VIT_3/layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2=
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/RsqrtҐ
HWheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpQwheatclassifier_vit_3_layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02J
HWheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul/ReadVariableOp«
9WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mulMul?WheatClassifier_VIT_3/layer_normalization_4/batchnorm/Rsqrt:y:0PWheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mulЬ
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul_1Mul#WheatClassifier_VIT_3/add_3/add:z:0=WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul_1Ї
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul_2MulAWheatClassifier_VIT_3/layer_normalization_4/moments/mean:output:0=WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul_2Ц
DWheatClassifier_VIT_3/layer_normalization_4/batchnorm/ReadVariableOpReadVariableOpMwheatclassifier_vit_3_layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02F
DWheatClassifier_VIT_3/layer_normalization_4/batchnorm/ReadVariableOp√
9WheatClassifier_VIT_3/layer_normalization_4/batchnorm/subSubLWheatClassifier_VIT_3/layer_normalization_4/batchnorm/ReadVariableOp:value:0?WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2;
9WheatClassifier_VIT_3/layer_normalization_4/batchnorm/subЇ
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/add_1AddV2?WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul_1:z:0=WheatClassifier_VIT_3/layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2=
;WheatClassifier_VIT_3/layer_normalization_4/batchnorm/add_1Ы
#WheatClassifier_VIT_3/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ d  2%
#WheatClassifier_VIT_3/flatten/Constь
%WheatClassifier_VIT_3/flatten/ReshapeReshape?WheatClassifier_VIT_3/layer_normalization_4/batchnorm/add_1:z:0,WheatClassifier_VIT_3/flatten/Const:output:0*
T0*)
_output_shapes
:€€€€€€€€€А»2'
%WheatClassifier_VIT_3/flatten/Reshapeй
3WheatClassifier_VIT_3/dense_5/MatMul/ReadVariableOpReadVariableOp<wheatclassifier_vit_3_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
А»2*
dtype025
3WheatClassifier_VIT_3/dense_5/MatMul/ReadVariableOpх
$WheatClassifier_VIT_3/dense_5/MatMulMatMul.WheatClassifier_VIT_3/flatten/Reshape:output:0;WheatClassifier_VIT_3/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$WheatClassifier_VIT_3/dense_5/MatMulж
4WheatClassifier_VIT_3/dense_5/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_vit_3_dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype026
4WheatClassifier_VIT_3/dense_5/BiasAdd/ReadVariableOpщ
%WheatClassifier_VIT_3/dense_5/BiasAddBiasAdd.WheatClassifier_VIT_3/dense_5/MatMul:product:0<WheatClassifier_VIT_3/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%WheatClassifier_VIT_3/dense_5/BiasAddЩ
(WheatClassifier_VIT_3/dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_VIT_3/dense_5/Gelu/mul/xм
&WheatClassifier_VIT_3/dense_5/Gelu/mulMul1WheatClassifier_VIT_3/dense_5/Gelu/mul/x:output:0.WheatClassifier_VIT_3/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&WheatClassifier_VIT_3/dense_5/Gelu/mulЫ
)WheatClassifier_VIT_3/dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2+
)WheatClassifier_VIT_3/dense_5/Gelu/Cast/xщ
*WheatClassifier_VIT_3/dense_5/Gelu/truedivRealDiv.WheatClassifier_VIT_3/dense_5/BiasAdd:output:02WheatClassifier_VIT_3/dense_5/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22,
*WheatClassifier_VIT_3/dense_5/Gelu/truedivє
&WheatClassifier_VIT_3/dense_5/Gelu/ErfErf.WheatClassifier_VIT_3/dense_5/Gelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&WheatClassifier_VIT_3/dense_5/Gelu/ErfЩ
(WheatClassifier_VIT_3/dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(WheatClassifier_VIT_3/dense_5/Gelu/add/xк
&WheatClassifier_VIT_3/dense_5/Gelu/addAddV21WheatClassifier_VIT_3/dense_5/Gelu/add/x:output:0*WheatClassifier_VIT_3/dense_5/Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&WheatClassifier_VIT_3/dense_5/Gelu/addе
(WheatClassifier_VIT_3/dense_5/Gelu/mul_1Mul*WheatClassifier_VIT_3/dense_5/Gelu/mul:z:0*WheatClassifier_VIT_3/dense_5/Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(WheatClassifier_VIT_3/dense_5/Gelu/mul_1з
3WheatClassifier_VIT_3/dense_6/MatMul/ReadVariableOpReadVariableOp<wheatclassifier_vit_3_dense_6_matmul_readvariableop_resource*
_output_shapes

:22*
dtype025
3WheatClassifier_VIT_3/dense_6/MatMul/ReadVariableOpу
$WheatClassifier_VIT_3/dense_6/MatMulMatMul,WheatClassifier_VIT_3/dense_5/Gelu/mul_1:z:0;WheatClassifier_VIT_3/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22&
$WheatClassifier_VIT_3/dense_6/MatMulж
4WheatClassifier_VIT_3/dense_6/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_vit_3_dense_6_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype026
4WheatClassifier_VIT_3/dense_6/BiasAdd/ReadVariableOpщ
%WheatClassifier_VIT_3/dense_6/BiasAddBiasAdd.WheatClassifier_VIT_3/dense_6/MatMul:product:0<WheatClassifier_VIT_3/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22'
%WheatClassifier_VIT_3/dense_6/BiasAddЩ
(WheatClassifier_VIT_3/dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2*
(WheatClassifier_VIT_3/dense_6/Gelu/mul/xм
&WheatClassifier_VIT_3/dense_6/Gelu/mulMul1WheatClassifier_VIT_3/dense_6/Gelu/mul/x:output:0.WheatClassifier_VIT_3/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&WheatClassifier_VIT_3/dense_6/Gelu/mulЫ
)WheatClassifier_VIT_3/dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2+
)WheatClassifier_VIT_3/dense_6/Gelu/Cast/xщ
*WheatClassifier_VIT_3/dense_6/Gelu/truedivRealDiv.WheatClassifier_VIT_3/dense_6/BiasAdd:output:02WheatClassifier_VIT_3/dense_6/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22,
*WheatClassifier_VIT_3/dense_6/Gelu/truedivє
&WheatClassifier_VIT_3/dense_6/Gelu/ErfErf.WheatClassifier_VIT_3/dense_6/Gelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&WheatClassifier_VIT_3/dense_6/Gelu/ErfЩ
(WheatClassifier_VIT_3/dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2*
(WheatClassifier_VIT_3/dense_6/Gelu/add/xк
&WheatClassifier_VIT_3/dense_6/Gelu/addAddV21WheatClassifier_VIT_3/dense_6/Gelu/add/x:output:0*WheatClassifier_VIT_3/dense_6/Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22(
&WheatClassifier_VIT_3/dense_6/Gelu/addе
(WheatClassifier_VIT_3/dense_6/Gelu/mul_1Mul*WheatClassifier_VIT_3/dense_6/Gelu/mul:z:0*WheatClassifier_VIT_3/dense_6/Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22*
(WheatClassifier_VIT_3/dense_6/Gelu/mul_1з
3WheatClassifier_VIT_3/dense_7/MatMul/ReadVariableOpReadVariableOp<wheatclassifier_vit_3_dense_7_matmul_readvariableop_resource*
_output_shapes

:2*
dtype025
3WheatClassifier_VIT_3/dense_7/MatMul/ReadVariableOpу
$WheatClassifier_VIT_3/dense_7/MatMulMatMul,WheatClassifier_VIT_3/dense_6/Gelu/mul_1:z:0;WheatClassifier_VIT_3/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2&
$WheatClassifier_VIT_3/dense_7/MatMulж
4WheatClassifier_VIT_3/dense_7/BiasAdd/ReadVariableOpReadVariableOp=wheatclassifier_vit_3_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4WheatClassifier_VIT_3/dense_7/BiasAdd/ReadVariableOpщ
%WheatClassifier_VIT_3/dense_7/BiasAddBiasAdd.WheatClassifier_VIT_3/dense_7/MatMul:product:0<WheatClassifier_VIT_3/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%WheatClassifier_VIT_3/dense_7/BiasAddї
%WheatClassifier_VIT_3/dense_7/SoftmaxSoftmax.WheatClassifier_VIT_3/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%WheatClassifier_VIT_3/dense_7/SoftmaxН
IdentityIdentity/WheatClassifier_VIT_3/dense_7/Softmax:softmax:05^WheatClassifier_VIT_3/dense_1/BiasAdd/ReadVariableOp7^WheatClassifier_VIT_3/dense_1/Tensordot/ReadVariableOp5^WheatClassifier_VIT_3/dense_2/BiasAdd/ReadVariableOp7^WheatClassifier_VIT_3/dense_2/Tensordot/ReadVariableOp5^WheatClassifier_VIT_3/dense_3/BiasAdd/ReadVariableOp7^WheatClassifier_VIT_3/dense_3/Tensordot/ReadVariableOp5^WheatClassifier_VIT_3/dense_4/BiasAdd/ReadVariableOp7^WheatClassifier_VIT_3/dense_4/Tensordot/ReadVariableOp5^WheatClassifier_VIT_3/dense_5/BiasAdd/ReadVariableOp4^WheatClassifier_VIT_3/dense_5/MatMul/ReadVariableOp5^WheatClassifier_VIT_3/dense_6/BiasAdd/ReadVariableOp4^WheatClassifier_VIT_3/dense_6/MatMul/ReadVariableOp5^WheatClassifier_VIT_3/dense_7/BiasAdd/ReadVariableOp4^WheatClassifier_VIT_3/dense_7/MatMul/ReadVariableOpC^WheatClassifier_VIT_3/layer_normalization/batchnorm/ReadVariableOpG^WheatClassifier_VIT_3/layer_normalization/batchnorm/mul/ReadVariableOpE^WheatClassifier_VIT_3/layer_normalization_1/batchnorm/ReadVariableOpI^WheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul/ReadVariableOpE^WheatClassifier_VIT_3/layer_normalization_2/batchnorm/ReadVariableOpI^WheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul/ReadVariableOpE^WheatClassifier_VIT_3/layer_normalization_3/batchnorm/ReadVariableOpI^WheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul/ReadVariableOpE^WheatClassifier_VIT_3/layer_normalization_4/batchnorm/ReadVariableOpI^WheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul/ReadVariableOpO^WheatClassifier_VIT_3/multi_head_attention/attention_output/add/ReadVariableOpY^WheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpB^WheatClassifier_VIT_3/multi_head_attention/key/add/ReadVariableOpL^WheatClassifier_VIT_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpD^WheatClassifier_VIT_3/multi_head_attention/query/add/ReadVariableOpN^WheatClassifier_VIT_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpD^WheatClassifier_VIT_3/multi_head_attention/value/add/ReadVariableOpN^WheatClassifier_VIT_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpQ^WheatClassifier_VIT_3/multi_head_attention_1/attention_output/add/ReadVariableOp[^WheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpD^WheatClassifier_VIT_3/multi_head_attention_1/key/add/ReadVariableOpN^WheatClassifier_VIT_3/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpF^WheatClassifier_VIT_3/multi_head_attention_1/query/add/ReadVariableOpP^WheatClassifier_VIT_3/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpF^WheatClassifier_VIT_3/multi_head_attention_1/value/add/ReadVariableOpP^WheatClassifier_VIT_3/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpA^WheatClassifier_VIT_3/patch_encoder/dense/BiasAdd/ReadVariableOpC^WheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ReadVariableOp?^WheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2l
4WheatClassifier_VIT_3/dense_1/BiasAdd/ReadVariableOp4WheatClassifier_VIT_3/dense_1/BiasAdd/ReadVariableOp2p
6WheatClassifier_VIT_3/dense_1/Tensordot/ReadVariableOp6WheatClassifier_VIT_3/dense_1/Tensordot/ReadVariableOp2l
4WheatClassifier_VIT_3/dense_2/BiasAdd/ReadVariableOp4WheatClassifier_VIT_3/dense_2/BiasAdd/ReadVariableOp2p
6WheatClassifier_VIT_3/dense_2/Tensordot/ReadVariableOp6WheatClassifier_VIT_3/dense_2/Tensordot/ReadVariableOp2l
4WheatClassifier_VIT_3/dense_3/BiasAdd/ReadVariableOp4WheatClassifier_VIT_3/dense_3/BiasAdd/ReadVariableOp2p
6WheatClassifier_VIT_3/dense_3/Tensordot/ReadVariableOp6WheatClassifier_VIT_3/dense_3/Tensordot/ReadVariableOp2l
4WheatClassifier_VIT_3/dense_4/BiasAdd/ReadVariableOp4WheatClassifier_VIT_3/dense_4/BiasAdd/ReadVariableOp2p
6WheatClassifier_VIT_3/dense_4/Tensordot/ReadVariableOp6WheatClassifier_VIT_3/dense_4/Tensordot/ReadVariableOp2l
4WheatClassifier_VIT_3/dense_5/BiasAdd/ReadVariableOp4WheatClassifier_VIT_3/dense_5/BiasAdd/ReadVariableOp2j
3WheatClassifier_VIT_3/dense_5/MatMul/ReadVariableOp3WheatClassifier_VIT_3/dense_5/MatMul/ReadVariableOp2l
4WheatClassifier_VIT_3/dense_6/BiasAdd/ReadVariableOp4WheatClassifier_VIT_3/dense_6/BiasAdd/ReadVariableOp2j
3WheatClassifier_VIT_3/dense_6/MatMul/ReadVariableOp3WheatClassifier_VIT_3/dense_6/MatMul/ReadVariableOp2l
4WheatClassifier_VIT_3/dense_7/BiasAdd/ReadVariableOp4WheatClassifier_VIT_3/dense_7/BiasAdd/ReadVariableOp2j
3WheatClassifier_VIT_3/dense_7/MatMul/ReadVariableOp3WheatClassifier_VIT_3/dense_7/MatMul/ReadVariableOp2И
BWheatClassifier_VIT_3/layer_normalization/batchnorm/ReadVariableOpBWheatClassifier_VIT_3/layer_normalization/batchnorm/ReadVariableOp2Р
FWheatClassifier_VIT_3/layer_normalization/batchnorm/mul/ReadVariableOpFWheatClassifier_VIT_3/layer_normalization/batchnorm/mul/ReadVariableOp2М
DWheatClassifier_VIT_3/layer_normalization_1/batchnorm/ReadVariableOpDWheatClassifier_VIT_3/layer_normalization_1/batchnorm/ReadVariableOp2Ф
HWheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul/ReadVariableOpHWheatClassifier_VIT_3/layer_normalization_1/batchnorm/mul/ReadVariableOp2М
DWheatClassifier_VIT_3/layer_normalization_2/batchnorm/ReadVariableOpDWheatClassifier_VIT_3/layer_normalization_2/batchnorm/ReadVariableOp2Ф
HWheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul/ReadVariableOpHWheatClassifier_VIT_3/layer_normalization_2/batchnorm/mul/ReadVariableOp2М
DWheatClassifier_VIT_3/layer_normalization_3/batchnorm/ReadVariableOpDWheatClassifier_VIT_3/layer_normalization_3/batchnorm/ReadVariableOp2Ф
HWheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul/ReadVariableOpHWheatClassifier_VIT_3/layer_normalization_3/batchnorm/mul/ReadVariableOp2М
DWheatClassifier_VIT_3/layer_normalization_4/batchnorm/ReadVariableOpDWheatClassifier_VIT_3/layer_normalization_4/batchnorm/ReadVariableOp2Ф
HWheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul/ReadVariableOpHWheatClassifier_VIT_3/layer_normalization_4/batchnorm/mul/ReadVariableOp2†
NWheatClassifier_VIT_3/multi_head_attention/attention_output/add/ReadVariableOpNWheatClassifier_VIT_3/multi_head_attention/attention_output/add/ReadVariableOp2і
XWheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpXWheatClassifier_VIT_3/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Ж
AWheatClassifier_VIT_3/multi_head_attention/key/add/ReadVariableOpAWheatClassifier_VIT_3/multi_head_attention/key/add/ReadVariableOp2Ъ
KWheatClassifier_VIT_3/multi_head_attention/key/einsum/Einsum/ReadVariableOpKWheatClassifier_VIT_3/multi_head_attention/key/einsum/Einsum/ReadVariableOp2К
CWheatClassifier_VIT_3/multi_head_attention/query/add/ReadVariableOpCWheatClassifier_VIT_3/multi_head_attention/query/add/ReadVariableOp2Ю
MWheatClassifier_VIT_3/multi_head_attention/query/einsum/Einsum/ReadVariableOpMWheatClassifier_VIT_3/multi_head_attention/query/einsum/Einsum/ReadVariableOp2К
CWheatClassifier_VIT_3/multi_head_attention/value/add/ReadVariableOpCWheatClassifier_VIT_3/multi_head_attention/value/add/ReadVariableOp2Ю
MWheatClassifier_VIT_3/multi_head_attention/value/einsum/Einsum/ReadVariableOpMWheatClassifier_VIT_3/multi_head_attention/value/einsum/Einsum/ReadVariableOp2§
PWheatClassifier_VIT_3/multi_head_attention_1/attention_output/add/ReadVariableOpPWheatClassifier_VIT_3/multi_head_attention_1/attention_output/add/ReadVariableOp2Є
ZWheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpZWheatClassifier_VIT_3/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2К
CWheatClassifier_VIT_3/multi_head_attention_1/key/add/ReadVariableOpCWheatClassifier_VIT_3/multi_head_attention_1/key/add/ReadVariableOp2Ю
MWheatClassifier_VIT_3/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpMWheatClassifier_VIT_3/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2О
EWheatClassifier_VIT_3/multi_head_attention_1/query/add/ReadVariableOpEWheatClassifier_VIT_3/multi_head_attention_1/query/add/ReadVariableOp2Ґ
OWheatClassifier_VIT_3/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpOWheatClassifier_VIT_3/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2О
EWheatClassifier_VIT_3/multi_head_attention_1/value/add/ReadVariableOpEWheatClassifier_VIT_3/multi_head_attention_1/value/add/ReadVariableOp2Ґ
OWheatClassifier_VIT_3/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpOWheatClassifier_VIT_3/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2Д
@WheatClassifier_VIT_3/patch_encoder/dense/BiasAdd/ReadVariableOp@WheatClassifier_VIT_3/patch_encoder/dense/BiasAdd/ReadVariableOp2И
BWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ReadVariableOpBWheatClassifier_VIT_3/patch_encoder/dense/Tensordot/ReadVariableOp2А
>WheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup>WheatClassifier_VIT_3/patch_encoder/embedding/embedding_lookup:Z V
1
_output_shapes
:€€€€€€€€€»»
!
_user_specified_name	input_1
ъ
«

6__inference_WheatClassifier_VIT_3_layer_call_fn_585927
input_1
unknown:	ђ@
	unknown_0:@
	unknown_1:	Р@
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

unknown_14:	@А

unknown_15:	А

unknown_16:	А@

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

unknown_30:	@А

unknown_31:	А

unknown_32:	А@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:
А»2

unknown_37:2

unknown_38:22

unknown_39:2

unknown_40:2

unknown_41:
identityИҐStatefulPartitionedCallЇ
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
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_5857472
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€»»
!
_user_specified_name	input_1
ƒ
Э
4__inference_layer_normalization_layer_call_fn_587301

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5846022
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
≥
Ц
(__inference_dense_4_layer_call_fn_587860

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_5849922
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€РА: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
-
_output_shapes
:€€€€€€€€€РА
 
_user_specified_nameinputs
»
Я
6__inference_layer_normalization_1_layer_call_fn_587465

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5846912
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
≤
ц
C__inference_dense_5_layer_call_and_return_conditional_losses_585060

inputs2
matmul_readvariableop_resource:
А»2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А»2*
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
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xt
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xБ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
Gelu/truediv_
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xr
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/addm

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/mul_1У
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:€€€€€€€€€А»: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:€€€€€€€€€А»
 
_user_specified_nameinputs
і
Ч
(__inference_dense_3_layer_call_fn_587813

inputs
unknown:	@А
	unknown_0:	А
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_5849482
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:€€€€€€€€€РА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
№
R
&__inference_add_1_layer_call_fn_587571
inputs_0
inputs_1
identity‘
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5847912
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:V R
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/1
Ь
Х
(__inference_dense_7_layer_call_fn_587988

inputs
unknown:2
	unknown_0:
identityИҐStatefulPartitionedCallц
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
GPU2*0J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_5851012
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
±t
я
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_586038
input_1'
patch_encoder_585931:	ђ@"
patch_encoder_585933:@'
patch_encoder_585935:	Р@(
layer_normalization_585938:@(
layer_normalization_585940:@1
multi_head_attention_585943:@@-
multi_head_attention_585945:@1
multi_head_attention_585947:@@-
multi_head_attention_585949:@1
multi_head_attention_585951:@@-
multi_head_attention_585953:@1
multi_head_attention_585955:@@)
multi_head_attention_585957:@*
layer_normalization_1_585961:@*
layer_normalization_1_585963:@!
dense_1_585966:	@А
dense_1_585968:	А!
dense_2_585971:	А@
dense_2_585973:@*
layer_normalization_2_585977:@*
layer_normalization_2_585979:@3
multi_head_attention_1_585982:@@/
multi_head_attention_1_585984:@3
multi_head_attention_1_585986:@@/
multi_head_attention_1_585988:@3
multi_head_attention_1_585990:@@/
multi_head_attention_1_585992:@3
multi_head_attention_1_585994:@@+
multi_head_attention_1_585996:@*
layer_normalization_3_586000:@*
layer_normalization_3_586002:@!
dense_3_586005:	@А
dense_3_586007:	А!
dense_4_586010:	А@
dense_4_586012:@*
layer_normalization_4_586016:@*
layer_normalization_4_586018:@"
dense_5_586022:
А»2
dense_5_586024:2 
dense_6_586027:22
dense_6_586029:2 
dense_7_586032:2
dense_7_586034:
identityИҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐ+layer_normalization/StatefulPartitionedCallҐ-layer_normalization_1/StatefulPartitionedCallҐ-layer_normalization_2/StatefulPartitionedCallҐ-layer_normalization_3/StatefulPartitionedCallҐ-layer_normalization_4/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ.multi_head_attention_1/StatefulPartitionedCallҐ%patch_encoder/StatefulPartitionedCallг
patches/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_5845302
patches/PartitionedCallз
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_585931patch_encoder_585933patch_encoder_585935*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_5845722'
%patch_encoder/StatefulPartitionedCallы
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_585938layer_normalization_585940*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5846022-
+layer_normalization/StatefulPartitionedCallч
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_585943multi_head_attention_585945multi_head_attention_585947multi_head_attention_585949multi_head_attention_585951multi_head_attention_585953multi_head_attention_585955multi_head_attention_585957*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_5846432.
,multi_head_attention/StatefulPartitionedCall≠
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_5846672
add/PartitionedCallу
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_585961layer_normalization_1_585963*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5846912/
-layer_normalization_1/StatefulPartitionedCall»
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_585966dense_1_585968*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5847352!
dense_1/StatefulPartitionedCallє
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_585971dense_2_585973*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_5847792!
dense_2/StatefulPartitionedCallФ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5847912
add_1/PartitionedCallх
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_585977layer_normalization_2_585979*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5848152/
-layer_normalization_2/StatefulPartitionedCallС
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_585982multi_head_attention_1_585984multi_head_attention_1_585986multi_head_attention_1_585988multi_head_attention_1_585990multi_head_attention_1_585992multi_head_attention_1_585994multi_head_attention_1_585996*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_58485620
.multi_head_attention_1/StatefulPartitionedCall•
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_5848802
add_2/PartitionedCallх
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_586000layer_normalization_3_586002*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_5849042/
-layer_normalization_3/StatefulPartitionedCall»
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_586005dense_3_586007*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_5849482!
dense_3/StatefulPartitionedCallє
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_586010dense_4_586012*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_5849922!
dense_4/StatefulPartitionedCallЦ
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_5850042
add_3/PartitionedCallх
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_586016layer_normalization_4_586018*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_5850282/
-layer_normalization_4/StatefulPartitionedCallЖ
flatten/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€А»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5850402
flatten/PartitionedCallђ
dense_5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5_586022dense_5_586024*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_5850602!
dense_5/StatefulPartitionedCallі
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_586027dense_6_586029*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_5850842!
dense_6/StatefulPartitionedCallі
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_586032dense_7_586034*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_5851012!
dense_7/StatefulPartitionedCallа
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€»»
!
_user_specified_name	input_1
Ь№
÷(
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_587018

inputsH
5patch_encoder_dense_tensordot_readvariableop_resource:	ђ@A
3patch_encoder_dense_biasadd_readvariableop_resource:@B
/patch_encoder_embedding_embedding_lookup_586670:	Р@G
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
)dense_3_tensordot_readvariableop_resource:	@А6
'dense_3_biasadd_readvariableop_resource:	А<
)dense_4_tensordot_readvariableop_resource:	А@5
'dense_4_biasadd_readvariableop_resource:@I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_4_batchnorm_readvariableop_resource:@:
&dense_5_matmul_readvariableop_resource:
А»25
'dense_5_biasadd_readvariableop_resource:28
&dense_6_matmul_readvariableop_resource:225
'dense_6_biasadd_readvariableop_resource:28
&dense_7_matmul_readvariableop_resource:25
'dense_7_biasadd_readvariableop_resource:
identityИҐdense_1/BiasAdd/ReadVariableOpҐ dense_1/Tensordot/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐ dense_2/Tensordot/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐ dense_3/Tensordot/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐ dense_4/Tensordot/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ,layer_normalization/batchnorm/ReadVariableOpҐ0layer_normalization/batchnorm/mul/ReadVariableOpҐ.layer_normalization_1/batchnorm/ReadVariableOpҐ2layer_normalization_1/batchnorm/mul/ReadVariableOpҐ.layer_normalization_2/batchnorm/ReadVariableOpҐ2layer_normalization_2/batchnorm/mul/ReadVariableOpҐ.layer_normalization_3/batchnorm/ReadVariableOpҐ2layer_normalization_3/batchnorm/mul/ReadVariableOpҐ.layer_normalization_4/batchnorm/ReadVariableOpҐ2layer_normalization_4/batchnorm/mul/ReadVariableOpҐ8multi_head_attention/attention_output/add/ReadVariableOpҐBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpҐ+multi_head_attention/key/add/ReadVariableOpҐ5multi_head_attention/key/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/query/add/ReadVariableOpҐ7multi_head_attention/query/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/value/add/ReadVariableOpҐ7multi_head_attention/value/einsum/Einsum/ReadVariableOpҐ:multi_head_attention_1/attention_output/add/ReadVariableOpҐDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_1/key/add/ReadVariableOpҐ7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/query/add/ReadVariableOpҐ9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/value/add/ReadVariableOpҐ9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐ*patch_encoder/dense/BiasAdd/ReadVariableOpҐ,patch_encoder/dense/Tensordot/ReadVariableOpҐ(patch_encoder/embedding/embedding_lookupT
patches/ShapeShapeinputs*
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
patches/strided_sliceд
patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*0
_output_shapes
:€€€€€€€€€ђ*
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
B :ђ2
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
!:€€€€€€€€€€€€€€€€€€ђ2
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
B :Р2
patch_encoder/range/limitx
patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder/range/deltaЉ
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes	
:Р2
patch_encoder/range”
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
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
!:€€€€€€€€€€€€€€€€€€ђ2)
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
patch_encoder/dense/BiasAddе
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_586670patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/586670*
_output_shapes
:	Р@*
dtype02*
(patch_encoder/embedding/embedding_lookupЅ
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/586670*
_output_shapes
:	Р@23
1patch_encoder/embedding/embedding_lookup/Identity№
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Р@25
3patch_encoder/embedding/embedding_lookup/Identity_1 
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
patch_encoder/add≤
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indicesи
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2"
 layer_normalization/moments/mean∆
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2*
(layer_normalization/moments/StopGradientф
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2/
-layer_normalization/moments/SquaredDifferenceЇ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indicesР
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2&
$layer_normalization/moments/varianceП
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52%
#layer_normalization/batchnorm/add/yг
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2#
!layer_normalization/batchnorm/add±
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization/batchnorm/RsqrtЏ
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOpз
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2#
!layer_normalization/batchnorm/mul∆
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization/batchnorm/mul_1Џ
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization/batchnorm/mul_2ќ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOpг
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2#
!layer_normalization/batchnorm/subЏ
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization/batchnorm/add_1ч
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp©
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum’
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOpо
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2 
multi_head_attention/query/addс
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp£
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsumѕ
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOpж
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
multi_head_attention/key/addч
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp©
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum’
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOpо
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/yњ
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
multi_head_attention/Mulц
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsumј
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2&
$multi_head_attention/softmax/SoftmaxЭ
*multi_head_attention/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2,
*multi_head_attention/dropout/dropout/Constь
(multi_head_attention/dropout/dropout/MulMul.multi_head_attention/softmax/Softmax:softmax:03multi_head_attention/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2*
(multi_head_attention/dropout/dropout/Mulґ
*multi_head_attention/dropout/dropout/ShapeShape.multi_head_attention/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2,
*multi_head_attention/dropout/dropout/ShapeХ
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformRandomUniform3multi_head_attention/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР*
dtype02C
Amulti_head_attention/dropout/dropout/random_uniform/RandomUniformѓ
3multi_head_attention/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=25
3multi_head_attention/dropout/dropout/GreaterEqual/yЉ
1multi_head_attention/dropout/dropout/GreaterEqualGreaterEqualJmulti_head_attention/dropout/dropout/random_uniform/RandomUniform:output:0<multi_head_attention/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР23
1multi_head_attention/dropout/dropout/GreaterEqualа
)multi_head_attention/dropout/dropout/CastCast5multi_head_attention/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€РР2+
)multi_head_attention/dropout/dropout/Castш
*multi_head_attention/dropout/dropout/Mul_1Mul,multi_head_attention/dropout/dropout/Mul:z:0-multi_head_attention/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€РР2,
*multi_head_attention/dropout/dropout/Mul_1Н
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/dropout/Mul_1:z:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/EinsumШ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpћ
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsumт
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOpЦ
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2+
)multi_head_attention/attention_output/addШ
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2	
add/addґ
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indicesд
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2$
"layer_normalization_1/moments/meanћ
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2,
*layer_normalization_1/moments/StopGradientр
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@21
/layer_normalization_1/moments/SquaredDifferenceЊ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indicesШ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2(
&layer_normalization_1/moments/varianceУ
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_1/batchnorm/add/yл
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization_1/batchnorm/addЈ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2'
%layer_normalization_1/batchnorm/Rsqrtа
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOpп
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_1/batchnorm/mul¬
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_1/batchnorm/mul_1в
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_1/batchnorm/mul_2‘
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOpл
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_1/batchnorm/subв
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
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
dense_1/Tensordot/stackћ
dense_1/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
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
dense_1/Tensordot/concat_1≤
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Tensordot•
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_1/BiasAdd/ReadVariableOp©
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/xЪ
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_1/Gelu/Cast/xІ
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Gelu/truediv}
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_1/Gelu/add/xШ
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Gelu/addУ
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
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
dense_2/Tensordot/stackЇ
dense_2/Tensordot/transpose	Transposedense_1/Gelu/mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
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
dense_2/Tensordot/concat_1±
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Tensordot§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp®
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/xЩ
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_2/Gelu/Cast/x¶
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/truediv|
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_2/Gelu/add/xЧ
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/addТ
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/mul_1{
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	add_1/addґ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesж
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2$
"layer_normalization_2/moments/meanћ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2,
*layer_normalization_2/moments/StopGradientт
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@21
/layer_normalization_2/moments/SquaredDifferenceЊ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesШ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2(
&layer_normalization_2/moments/varianceУ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_2/batchnorm/add/yл
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization_2/batchnorm/addЈ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2'
%layer_normalization_2/batchnorm/Rsqrtа
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpп
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_2/batchnorm/mulƒ
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_2/batchnorm/mul_1в
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_2/batchnorm/mul_2‘
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpл
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_2/batchnorm/subв
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_2/batchnorm/add_1э
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp±
*multi_head_attention_1/query/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsumџ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/query/add/ReadVariableOpц
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2"
 multi_head_attention_1/query/addч
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpЂ
(multi_head_attention_1/key/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum’
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpо
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2 
multi_head_attention_1/key/addэ
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp±
*multi_head_attention_1/value/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsumџ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/value/add/ReadVariableOpц
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2"
 multi_head_attention_1/value/addБ
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_1/Mul/y«
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
multi_head_attention_1/Mulю
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum∆
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2(
&multi_head_attention_1/softmax/Softmax°
,multi_head_attention_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2.
,multi_head_attention_1/dropout/dropout/ConstД
*multi_head_attention_1/dropout/dropout/MulMul0multi_head_attention_1/softmax/Softmax:softmax:05multi_head_attention_1/dropout/dropout/Const:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2,
*multi_head_attention_1/dropout/dropout/MulЉ
,multi_head_attention_1/dropout/dropout/ShapeShape0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*
_output_shapes
:2.
,multi_head_attention_1/dropout/dropout/ShapeЫ
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniformRandomUniform5multi_head_attention_1/dropout/dropout/Shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР*
dtype02E
Cmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform≥
5multi_head_attention_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=27
5multi_head_attention_1/dropout/dropout/GreaterEqual/yƒ
3multi_head_attention_1/dropout/dropout/GreaterEqualGreaterEqualLmulti_head_attention_1/dropout/dropout/random_uniform/RandomUniform:output:0>multi_head_attention_1/dropout/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР25
3multi_head_attention_1/dropout/dropout/GreaterEqualж
+multi_head_attention_1/dropout/dropout/CastCast7multi_head_attention_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:€€€€€€€€€РР2-
+multi_head_attention_1/dropout/dropout/CastА
,multi_head_attention_1/dropout/dropout/Mul_1Mul.multi_head_attention_1/dropout/dropout/Mul:z:0/multi_head_attention_1/dropout/dropout/Cast:y:0*
T0*1
_output_shapes
:€€€€€€€€€РР2.
,multi_head_attention_1/dropout/dropout/Mul_1Х
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/dropout/Mul_1:z:0$multi_head_attention_1/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/EinsumЮ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumш
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOpЮ
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2-
+multi_head_attention_1/attention_output/addЦ
	add_2/addAddV2/multi_head_attention_1/attention_output/add:z:0add_1/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	add_2/addґ
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesж
"layer_normalization_3/moments/meanMeanadd_2/add:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2$
"layer_normalization_3/moments/meanћ
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2,
*layer_normalization_3/moments/StopGradientт
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd_2/add:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@21
/layer_normalization_3/moments/SquaredDifferenceЊ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indicesШ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2(
&layer_normalization_3/moments/varianceУ
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_3/batchnorm/add/yл
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization_3/batchnorm/addЈ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2'
%layer_normalization_3/batchnorm/Rsqrtа
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpп
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_3/batchnorm/mulƒ
%layer_normalization_3/batchnorm/mul_1Muladd_2/add:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_3/batchnorm/mul_1в
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_3/batchnorm/mul_2‘
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpл
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_3/batchnorm/subв
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_3/batchnorm/add_1ѓ
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@А*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axesБ
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/freeЛ
dense_3/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_3/Tensordot/ShapeД
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
dense_3/Tensordot/GatherV2И
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis€
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
dense_3/Tensordot/Const†
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/ProdА
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1®
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1А
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axisЎ
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concatђ
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stackћ
dense_3/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_3/Tensordot/transposeњ
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_3/Tensordot/Reshapeњ
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_3/Tensordot/MatMulБ
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_3/Tensordot/Const_2Д
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axisе
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1≤
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Tensordot•
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_3/BiasAdd/ReadVariableOp©
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/BiasAddm
dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_3/Gelu/mul/xЪ
dense_3/Gelu/mulMuldense_3/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/mulo
dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_3/Gelu/Cast/xІ
dense_3/Gelu/truedivRealDivdense_3/BiasAdd:output:0dense_3/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/truediv}
dense_3/Gelu/ErfErfdense_3/Gelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/Erfm
dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_3/Gelu/add/xШ
dense_3/Gelu/addAddV2dense_3/Gelu/add/x:output:0dense_3/Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/addУ
dense_3/Gelu/mul_1Muldense_3/Gelu/mul:z:0dense_3/Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/mul_1ѓ
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	А@*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axesБ
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
dense_4/Tensordot/ShapeД
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
dense_4/Tensordot/GatherV2И
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis€
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
dense_4/Tensordot/Const†
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/ProdА
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1®
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1А
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axisЎ
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concatђ
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stackЇ
dense_4/Tensordot/transpose	Transposedense_3/Gelu/mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_4/Tensordot/transposeњ
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_4/Tensordot/ReshapeЊ
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_4/Tensordot/MatMulА
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_4/Tensordot/Const_2Д
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axisе
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1±
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Tensordot§
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp®
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/BiasAddm
dense_4/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_4/Gelu/mul/xЩ
dense_4/Gelu/mulMuldense_4/Gelu/mul/x:output:0dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/mulo
dense_4/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_4/Gelu/Cast/x¶
dense_4/Gelu/truedivRealDivdense_4/BiasAdd:output:0dense_4/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/truediv|
dense_4/Gelu/ErfErfdense_4/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/Erfm
dense_4/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_4/Gelu/add/xЧ
dense_4/Gelu/addAddV2dense_4/Gelu/add/x:output:0dense_4/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/addТ
dense_4/Gelu/mul_1Muldense_4/Gelu/mul:z:0dense_4/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/mul_1}
	add_3/addAddV2dense_4/Gelu/mul_1:z:0add_2/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	add_3/addґ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesж
"layer_normalization_4/moments/meanMeanadd_3/add:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2$
"layer_normalization_4/moments/meanћ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2,
*layer_normalization_4/moments/StopGradientт
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd_3/add:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@21
/layer_normalization_4/moments/SquaredDifferenceЊ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indicesШ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2(
&layer_normalization_4/moments/varianceУ
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_4/batchnorm/add/yл
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization_4/batchnorm/addЈ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2'
%layer_normalization_4/batchnorm/Rsqrtа
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpп
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_4/batchnorm/mulƒ
%layer_normalization_4/batchnorm/mul_1Muladd_3/add:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_4/batchnorm/mul_1в
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_4/batchnorm/mul_2‘
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpл
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_4/batchnorm/subв
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_4/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ d  2
flatten/Const§
flatten/ReshapeReshape)layer_normalization_4/batchnorm/add_1:z:0flatten/Const:output:0*
T0*)
_output_shapes
:€€€€€€€€€А»2
flatten/ReshapeІ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
А»2*
dtype02
dense_5/MatMul/ReadVariableOpЭ
dense_5/MatMulMatMulflatten/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/MatMul§
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_5/BiasAdd/ReadVariableOp°
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/BiasAddm
dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_5/Gelu/mul/xФ
dense_5/Gelu/mulMuldense_5/Gelu/mul/x:output:0dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/mulo
dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_5/Gelu/Cast/x°
dense_5/Gelu/truedivRealDivdense_5/BiasAdd:output:0dense_5/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/truedivw
dense_5/Gelu/ErfErfdense_5/Gelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/Erfm
dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_5/Gelu/add/xТ
dense_5/Gelu/addAddV2dense_5/Gelu/add/x:output:0dense_5/Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/addН
dense_5/Gelu/mul_1Muldense_5/Gelu/mul:z:0dense_5/Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/mul_1•
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
dense_6/MatMul/ReadVariableOpЫ
dense_6/MatMulMatMuldense_5/Gelu/mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/BiasAddm
dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_6/Gelu/mul/xФ
dense_6/Gelu/mulMuldense_6/Gelu/mul/x:output:0dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/mulo
dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_6/Gelu/Cast/x°
dense_6/Gelu/truedivRealDivdense_6/BiasAdd:output:0dense_6/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/truedivw
dense_6/Gelu/ErfErfdense_6/Gelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/Erfm
dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_6/Gelu/add/xТ
dense_6/Gelu/addAddV2dense_6/Gelu/add/x:output:0dense_6/Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/addН
dense_6/Gelu/mul_1Muldense_6/Gelu/mul:z:0dense_6/Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/mul_1•
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
dense_7/MatMul/ReadVariableOpЫ
dense_7/MatMulMatMuldense_6/Gelu/mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/MatMul§
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp°
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/Softmax≈
IdentityIdentitydense_7/Softmax:softmax:0^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2\
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
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2И
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2X
*patch_encoder/dense/BiasAdd/ReadVariableOp*patch_encoder/dense/BiasAdd/ReadVariableOp2\
,patch_encoder/dense/Tensordot/ReadVariableOp,patch_encoder/dense/Tensordot/ReadVariableOp2T
(patch_encoder/embedding/embedding_lookup(patch_encoder/embedding/embedding_lookup:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
Н'
ы
C__inference_dense_4_layer_call_and_return_conditional_losses_584992

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
Tensordot/stackТ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
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
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2	
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
:€€€€€€€€€Р@2

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
:€€€€€€€€€Р@2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

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
:€€€€€€€€€Р@2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Gelu/mul_1Ы
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€РА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€РА
 
_user_specified_nameinputs
Л.
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_584643	
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
"query/einsum/Einsum/ReadVariableOp»
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЪ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp¬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpТ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp»
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЪ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
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
:€€€€€€€€€Р@2
MulҐ
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2
einsum/EinsumБ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
softmax/SoftmaxЗ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/Identityє
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpш
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp¬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
attention_output/addГ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 2J
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
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
±
Р
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_587593

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
±
Р
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_584904

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
ю/
л
I__inference_patch_encoder_layer_call_and_return_conditional_losses_587259	
patch:
'dense_tensordot_readvariableop_resource:	ђ@3
%dense_biasadd_readvariableop_resource:@4
!embedding_embedding_lookup_587252:	Р@
identityИҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOpҐembedding/embedding_lookup\
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
B :Р2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:Р2
range©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
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
!:€€€€€€€€€€€€€€€€€€ђ2
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
dense/BiasAddЯ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_587252range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/587252*
_output_shapes
:	Р@*
dtype02
embedding/embedding_lookupЙ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/587252*
_output_shapes
:	Р@2%
#embedding/embedding_lookup/Identity≤
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Р@2'
%embedding/embedding_lookup/Identity_1Т
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
addљ
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ

_user_specified_namepatch
Н'
ы
C__inference_dense_2_layer_call_and_return_conditional_losses_584779

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
Tensordot/stackТ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
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
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2	
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
:€€€€€€€€€Р@2

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
:€€€€€€€€€Р@2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

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
:€€€€€€€€€Р@2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Gelu/mul_1Ы
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€РА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€РА
 
_user_specified_nameinputs
з
k
A__inference_add_2_layer_call_and_return_conditional_losses_584880

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:€€€€€€€€€Р@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
±
Р
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_584815

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
Н'
ы
C__inference_dense_4_layer_call_and_return_conditional_losses_587851

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
Tensordot/stackТ
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
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
Tensordot/concat_1С
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	TensordotМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2	
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
:€€€€€€€€€Р@2

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
:€€€€€€€€€Р@2
Gelu/truedivd
Gelu/ErfErfGelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

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
:€€€€€€€€€Р@2

Gelu/addr

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Gelu/mul_1Ы
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€РА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:U Q
-
_output_shapes
:€€€€€€€€€РА
 
_user_specified_nameinputs
Ц'
ь
C__inference_dense_3_layer_call_and_return_conditional_losses_584948

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
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
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
Tensordot/concat_1Т
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2	
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
:€€€€€€€€€РА2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xЗ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
Gelu/truedive
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xx
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/adds

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/mul_1Ь
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€РА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
з
k
A__inference_add_1_layer_call_and_return_conditional_losses_584791

inputs
inputs_1
identity\
addAddV2inputsinputs_1*
T0*,
_output_shapes
:€€€€€€€€€Р@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs:TP
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
яз
Ћb
"__inference__traced_restore_588855
file_prefix8
*assignvariableop_layer_normalization_gamma:@9
+assignvariableop_1_layer_normalization_beta:@<
.assignvariableop_2_layer_normalization_1_gamma:@;
-assignvariableop_3_layer_normalization_1_beta:@4
!assignvariableop_4_dense_1_kernel:	@А.
assignvariableop_5_dense_1_bias:	А4
!assignvariableop_6_dense_2_kernel:	А@-
assignvariableop_7_dense_2_bias:@<
.assignvariableop_8_layer_normalization_2_gamma:@;
-assignvariableop_9_layer_normalization_2_beta:@=
/assignvariableop_10_layer_normalization_3_gamma:@<
.assignvariableop_11_layer_normalization_3_beta:@5
"assignvariableop_12_dense_3_kernel:	@А/
 assignvariableop_13_dense_3_bias:	А5
"assignvariableop_14_dense_4_kernel:	А@.
 assignvariableop_15_dense_4_bias:@=
/assignvariableop_16_layer_normalization_4_gamma:@<
.assignvariableop_17_layer_normalization_4_beta:@6
"assignvariableop_18_dense_5_kernel:
А»2.
 assignvariableop_19_dense_5_bias:24
"assignvariableop_20_dense_6_kernel:22.
 assignvariableop_21_dense_6_bias:24
"assignvariableop_22_dense_7_kernel:2.
 assignvariableop_23_dense_7_bias:(
assignvariableop_24_adamw_iter:	 *
 assignvariableop_25_adamw_beta_1: *
 assignvariableop_26_adamw_beta_2: )
assignvariableop_27_adamw_decay: 1
'assignvariableop_28_adamw_learning_rate: 0
&assignvariableop_29_adamw_weight_decay: A
.assignvariableop_30_patch_encoder_dense_kernel:	ђ@:
,assignvariableop_31_patch_encoder_dense_bias:@I
6assignvariableop_32_patch_encoder_embedding_embeddings:	Р@K
5assignvariableop_33_multi_head_attention_query_kernel:@@E
3assignvariableop_34_multi_head_attention_query_bias:@I
3assignvariableop_35_multi_head_attention_key_kernel:@@C
1assignvariableop_36_multi_head_attention_key_bias:@K
5assignvariableop_37_multi_head_attention_value_kernel:@@E
3assignvariableop_38_multi_head_attention_value_bias:@V
@assignvariableop_39_multi_head_attention_attention_output_kernel:@@L
>assignvariableop_40_multi_head_attention_attention_output_bias:@M
7assignvariableop_41_multi_head_attention_1_query_kernel:@@G
5assignvariableop_42_multi_head_attention_1_query_bias:@K
5assignvariableop_43_multi_head_attention_1_key_kernel:@@E
3assignvariableop_44_multi_head_attention_1_key_bias:@M
7assignvariableop_45_multi_head_attention_1_value_kernel:@@G
5assignvariableop_46_multi_head_attention_1_value_bias:@X
Bassignvariableop_47_multi_head_attention_1_attention_output_kernel:@@N
@assignvariableop_48_multi_head_attention_1_attention_output_bias:@#
assignvariableop_49_total: #
assignvariableop_50_count: %
assignvariableop_51_total_1: %
assignvariableop_52_count_1: C
5assignvariableop_53_adamw_layer_normalization_gamma_m:@B
4assignvariableop_54_adamw_layer_normalization_beta_m:@E
7assignvariableop_55_adamw_layer_normalization_1_gamma_m:@D
6assignvariableop_56_adamw_layer_normalization_1_beta_m:@=
*assignvariableop_57_adamw_dense_1_kernel_m:	@А7
(assignvariableop_58_adamw_dense_1_bias_m:	А=
*assignvariableop_59_adamw_dense_2_kernel_m:	А@6
(assignvariableop_60_adamw_dense_2_bias_m:@E
7assignvariableop_61_adamw_layer_normalization_2_gamma_m:@D
6assignvariableop_62_adamw_layer_normalization_2_beta_m:@E
7assignvariableop_63_adamw_layer_normalization_3_gamma_m:@D
6assignvariableop_64_adamw_layer_normalization_3_beta_m:@=
*assignvariableop_65_adamw_dense_3_kernel_m:	@А7
(assignvariableop_66_adamw_dense_3_bias_m:	А=
*assignvariableop_67_adamw_dense_4_kernel_m:	А@6
(assignvariableop_68_adamw_dense_4_bias_m:@E
7assignvariableop_69_adamw_layer_normalization_4_gamma_m:@D
6assignvariableop_70_adamw_layer_normalization_4_beta_m:@>
*assignvariableop_71_adamw_dense_5_kernel_m:
А»26
(assignvariableop_72_adamw_dense_5_bias_m:2<
*assignvariableop_73_adamw_dense_6_kernel_m:226
(assignvariableop_74_adamw_dense_6_bias_m:2<
*assignvariableop_75_adamw_dense_7_kernel_m:26
(assignvariableop_76_adamw_dense_7_bias_m:I
6assignvariableop_77_adamw_patch_encoder_dense_kernel_m:	ђ@B
4assignvariableop_78_adamw_patch_encoder_dense_bias_m:@Q
>assignvariableop_79_adamw_patch_encoder_embedding_embeddings_m:	Р@S
=assignvariableop_80_adamw_multi_head_attention_query_kernel_m:@@M
;assignvariableop_81_adamw_multi_head_attention_query_bias_m:@Q
;assignvariableop_82_adamw_multi_head_attention_key_kernel_m:@@K
9assignvariableop_83_adamw_multi_head_attention_key_bias_m:@S
=assignvariableop_84_adamw_multi_head_attention_value_kernel_m:@@M
;assignvariableop_85_adamw_multi_head_attention_value_bias_m:@^
Hassignvariableop_86_adamw_multi_head_attention_attention_output_kernel_m:@@T
Fassignvariableop_87_adamw_multi_head_attention_attention_output_bias_m:@U
?assignvariableop_88_adamw_multi_head_attention_1_query_kernel_m:@@O
=assignvariableop_89_adamw_multi_head_attention_1_query_bias_m:@S
=assignvariableop_90_adamw_multi_head_attention_1_key_kernel_m:@@M
;assignvariableop_91_adamw_multi_head_attention_1_key_bias_m:@U
?assignvariableop_92_adamw_multi_head_attention_1_value_kernel_m:@@O
=assignvariableop_93_adamw_multi_head_attention_1_value_bias_m:@`
Jassignvariableop_94_adamw_multi_head_attention_1_attention_output_kernel_m:@@V
Hassignvariableop_95_adamw_multi_head_attention_1_attention_output_bias_m:@C
5assignvariableop_96_adamw_layer_normalization_gamma_v:@B
4assignvariableop_97_adamw_layer_normalization_beta_v:@E
7assignvariableop_98_adamw_layer_normalization_1_gamma_v:@D
6assignvariableop_99_adamw_layer_normalization_1_beta_v:@>
+assignvariableop_100_adamw_dense_1_kernel_v:	@А8
)assignvariableop_101_adamw_dense_1_bias_v:	А>
+assignvariableop_102_adamw_dense_2_kernel_v:	А@7
)assignvariableop_103_adamw_dense_2_bias_v:@F
8assignvariableop_104_adamw_layer_normalization_2_gamma_v:@E
7assignvariableop_105_adamw_layer_normalization_2_beta_v:@F
8assignvariableop_106_adamw_layer_normalization_3_gamma_v:@E
7assignvariableop_107_adamw_layer_normalization_3_beta_v:@>
+assignvariableop_108_adamw_dense_3_kernel_v:	@А8
)assignvariableop_109_adamw_dense_3_bias_v:	А>
+assignvariableop_110_adamw_dense_4_kernel_v:	А@7
)assignvariableop_111_adamw_dense_4_bias_v:@F
8assignvariableop_112_adamw_layer_normalization_4_gamma_v:@E
7assignvariableop_113_adamw_layer_normalization_4_beta_v:@?
+assignvariableop_114_adamw_dense_5_kernel_v:
А»27
)assignvariableop_115_adamw_dense_5_bias_v:2=
+assignvariableop_116_adamw_dense_6_kernel_v:227
)assignvariableop_117_adamw_dense_6_bias_v:2=
+assignvariableop_118_adamw_dense_7_kernel_v:27
)assignvariableop_119_adamw_dense_7_bias_v:J
7assignvariableop_120_adamw_patch_encoder_dense_kernel_v:	ђ@C
5assignvariableop_121_adamw_patch_encoder_dense_bias_v:@R
?assignvariableop_122_adamw_patch_encoder_embedding_embeddings_v:	Р@T
>assignvariableop_123_adamw_multi_head_attention_query_kernel_v:@@N
<assignvariableop_124_adamw_multi_head_attention_query_bias_v:@R
<assignvariableop_125_adamw_multi_head_attention_key_kernel_v:@@L
:assignvariableop_126_adamw_multi_head_attention_key_bias_v:@T
>assignvariableop_127_adamw_multi_head_attention_value_kernel_v:@@N
<assignvariableop_128_adamw_multi_head_attention_value_bias_v:@_
Iassignvariableop_129_adamw_multi_head_attention_attention_output_kernel_v:@@U
Gassignvariableop_130_adamw_multi_head_attention_attention_output_bias_v:@V
@assignvariableop_131_adamw_multi_head_attention_1_query_kernel_v:@@P
>assignvariableop_132_adamw_multi_head_attention_1_query_bias_v:@T
>assignvariableop_133_adamw_multi_head_attention_1_key_kernel_v:@@N
<assignvariableop_134_adamw_multi_head_attention_1_key_bias_v:@V
@assignvariableop_135_adamw_multi_head_attention_1_value_kernel_v:@@P
>assignvariableop_136_adamw_multi_head_attention_1_value_bias_v:@a
Kassignvariableop_137_adamw_multi_head_attention_1_attention_output_kernel_v:@@W
Iassignvariableop_138_adamw_multi_head_attention_1_attention_output_bias_v:@
identity_140ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_100ҐAssignVariableOp_101ҐAssignVariableOp_102ҐAssignVariableOp_103ҐAssignVariableOp_104ҐAssignVariableOp_105ҐAssignVariableOp_106ҐAssignVariableOp_107ҐAssignVariableOp_108ҐAssignVariableOp_109ҐAssignVariableOp_11ҐAssignVariableOp_110ҐAssignVariableOp_111ҐAssignVariableOp_112ҐAssignVariableOp_113ҐAssignVariableOp_114ҐAssignVariableOp_115ҐAssignVariableOp_116ҐAssignVariableOp_117ҐAssignVariableOp_118ҐAssignVariableOp_119ҐAssignVariableOp_12ҐAssignVariableOp_120ҐAssignVariableOp_121ҐAssignVariableOp_122ҐAssignVariableOp_123ҐAssignVariableOp_124ҐAssignVariableOp_125ҐAssignVariableOp_126ҐAssignVariableOp_127ҐAssignVariableOp_128ҐAssignVariableOp_129ҐAssignVariableOp_13ҐAssignVariableOp_130ҐAssignVariableOp_131ҐAssignVariableOp_132ҐAssignVariableOp_133ҐAssignVariableOp_134ҐAssignVariableOp_135ҐAssignVariableOp_136ҐAssignVariableOp_137ҐAssignVariableOp_138ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ҐAssignVariableOp_99ЋM
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:М*
dtype0*÷L
valueћLB…LМB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:М*
dtype0*Ѓ
value§B°МB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesо
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*∆
_output_shapes≥
∞::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Э
dtypesТ
П2М	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity©
AssignVariableOpAssignVariableOp*assignvariableop_layer_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1∞
AssignVariableOp_1AssignVariableOp+assignvariableop_1_layer_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2≥
AssignVariableOp_2AssignVariableOp.assignvariableop_2_layer_normalization_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3≤
AssignVariableOp_3AssignVariableOp-assignvariableop_3_layer_normalization_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¶
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¶
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8≥
AssignVariableOp_8AssignVariableOp.assignvariableop_8_layer_normalization_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9≤
AssignVariableOp_9AssignVariableOp-assignvariableop_9_layer_normalization_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ј
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_normalization_3_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ґ
AssignVariableOp_11AssignVariableOp.assignvariableop_11_layer_normalization_3_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12™
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13®
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14™
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15®
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ј
AssignVariableOp_16AssignVariableOp/assignvariableop_16_layer_normalization_4_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ґ
AssignVariableOp_17AssignVariableOp.assignvariableop_17_layer_normalization_4_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18™
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_5_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19®
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_5_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20™
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_6_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21®
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_6_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22™
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_7_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23®
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_7_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_24¶
AssignVariableOp_24AssignVariableOpassignvariableop_24_adamw_iterIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25®
AssignVariableOp_25AssignVariableOp assignvariableop_25_adamw_beta_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26®
AssignVariableOp_26AssignVariableOp assignvariableop_26_adamw_beta_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27І
AssignVariableOp_27AssignVariableOpassignvariableop_27_adamw_decayIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28ѓ
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adamw_learning_rateIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ѓ
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adamw_weight_decayIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30ґ
AssignVariableOp_30AssignVariableOp.assignvariableop_30_patch_encoder_dense_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31і
AssignVariableOp_31AssignVariableOp,assignvariableop_31_patch_encoder_dense_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Њ
AssignVariableOp_32AssignVariableOp6assignvariableop_32_patch_encoder_embedding_embeddingsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33љ
AssignVariableOp_33AssignVariableOp5assignvariableop_33_multi_head_attention_query_kernelIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34ї
AssignVariableOp_34AssignVariableOp3assignvariableop_34_multi_head_attention_query_biasIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ї
AssignVariableOp_35AssignVariableOp3assignvariableop_35_multi_head_attention_key_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36є
AssignVariableOp_36AssignVariableOp1assignvariableop_36_multi_head_attention_key_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37љ
AssignVariableOp_37AssignVariableOp5assignvariableop_37_multi_head_attention_value_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38ї
AssignVariableOp_38AssignVariableOp3assignvariableop_38_multi_head_attention_value_biasIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39»
AssignVariableOp_39AssignVariableOp@assignvariableop_39_multi_head_attention_attention_output_kernelIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40∆
AssignVariableOp_40AssignVariableOp>assignvariableop_40_multi_head_attention_attention_output_biasIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41њ
AssignVariableOp_41AssignVariableOp7assignvariableop_41_multi_head_attention_1_query_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42љ
AssignVariableOp_42AssignVariableOp5assignvariableop_42_multi_head_attention_1_query_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43љ
AssignVariableOp_43AssignVariableOp5assignvariableop_43_multi_head_attention_1_key_kernelIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44ї
AssignVariableOp_44AssignVariableOp3assignvariableop_44_multi_head_attention_1_key_biasIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45њ
AssignVariableOp_45AssignVariableOp7assignvariableop_45_multi_head_attention_1_value_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46љ
AssignVariableOp_46AssignVariableOp5assignvariableop_46_multi_head_attention_1_value_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47 
AssignVariableOp_47AssignVariableOpBassignvariableop_47_multi_head_attention_1_attention_output_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48»
AssignVariableOp_48AssignVariableOp@assignvariableop_48_multi_head_attention_1_attention_output_biasIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49°
AssignVariableOp_49AssignVariableOpassignvariableop_49_totalIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50°
AssignVariableOp_50AssignVariableOpassignvariableop_50_countIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51£
AssignVariableOp_51AssignVariableOpassignvariableop_51_total_1Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52£
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_1Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53љ
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adamw_layer_normalization_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Љ
AssignVariableOp_54AssignVariableOp4assignvariableop_54_adamw_layer_normalization_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55њ
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adamw_layer_normalization_1_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Њ
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adamw_layer_normalization_1_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57≤
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adamw_dense_1_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58∞
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adamw_dense_1_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59≤
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adamw_dense_2_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60∞
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adamw_dense_2_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61њ
AssignVariableOp_61AssignVariableOp7assignvariableop_61_adamw_layer_normalization_2_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Њ
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adamw_layer_normalization_2_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63њ
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adamw_layer_normalization_3_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Њ
AssignVariableOp_64AssignVariableOp6assignvariableop_64_adamw_layer_normalization_3_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65≤
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adamw_dense_3_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66∞
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adamw_dense_3_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67≤
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adamw_dense_4_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68∞
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adamw_dense_4_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69њ
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adamw_layer_normalization_4_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Њ
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adamw_layer_normalization_4_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71≤
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adamw_dense_5_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72∞
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adamw_dense_5_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73≤
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adamw_dense_6_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74∞
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adamw_dense_6_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75≤
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adamw_dense_7_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76∞
AssignVariableOp_76AssignVariableOp(assignvariableop_76_adamw_dense_7_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77Њ
AssignVariableOp_77AssignVariableOp6assignvariableop_77_adamw_patch_encoder_dense_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78Љ
AssignVariableOp_78AssignVariableOp4assignvariableop_78_adamw_patch_encoder_dense_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79∆
AssignVariableOp_79AssignVariableOp>assignvariableop_79_adamw_patch_encoder_embedding_embeddings_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80≈
AssignVariableOp_80AssignVariableOp=assignvariableop_80_adamw_multi_head_attention_query_kernel_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81√
AssignVariableOp_81AssignVariableOp;assignvariableop_81_adamw_multi_head_attention_query_bias_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82√
AssignVariableOp_82AssignVariableOp;assignvariableop_82_adamw_multi_head_attention_key_kernel_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83Ѕ
AssignVariableOp_83AssignVariableOp9assignvariableop_83_adamw_multi_head_attention_key_bias_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84≈
AssignVariableOp_84AssignVariableOp=assignvariableop_84_adamw_multi_head_attention_value_kernel_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85√
AssignVariableOp_85AssignVariableOp;assignvariableop_85_adamw_multi_head_attention_value_bias_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86–
AssignVariableOp_86AssignVariableOpHassignvariableop_86_adamw_multi_head_attention_attention_output_kernel_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87ќ
AssignVariableOp_87AssignVariableOpFassignvariableop_87_adamw_multi_head_attention_attention_output_bias_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88«
AssignVariableOp_88AssignVariableOp?assignvariableop_88_adamw_multi_head_attention_1_query_kernel_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89≈
AssignVariableOp_89AssignVariableOp=assignvariableop_89_adamw_multi_head_attention_1_query_bias_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90≈
AssignVariableOp_90AssignVariableOp=assignvariableop_90_adamw_multi_head_attention_1_key_kernel_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91√
AssignVariableOp_91AssignVariableOp;assignvariableop_91_adamw_multi_head_attention_1_key_bias_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92«
AssignVariableOp_92AssignVariableOp?assignvariableop_92_adamw_multi_head_attention_1_value_kernel_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93≈
AssignVariableOp_93AssignVariableOp=assignvariableop_93_adamw_multi_head_attention_1_value_bias_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94“
AssignVariableOp_94AssignVariableOpJassignvariableop_94_adamw_multi_head_attention_1_attention_output_kernel_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95–
AssignVariableOp_95AssignVariableOpHassignvariableop_95_adamw_multi_head_attention_1_attention_output_bias_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96љ
AssignVariableOp_96AssignVariableOp5assignvariableop_96_adamw_layer_normalization_gamma_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97Љ
AssignVariableOp_97AssignVariableOp4assignvariableop_97_adamw_layer_normalization_beta_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98њ
AssignVariableOp_98AssignVariableOp7assignvariableop_98_adamw_layer_normalization_1_gamma_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99Њ
AssignVariableOp_99AssignVariableOp6assignvariableop_99_adamw_layer_normalization_1_beta_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100ґ
AssignVariableOp_100AssignVariableOp+assignvariableop_100_adamw_dense_1_kernel_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101і
AssignVariableOp_101AssignVariableOp)assignvariableop_101_adamw_dense_1_bias_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102ґ
AssignVariableOp_102AssignVariableOp+assignvariableop_102_adamw_dense_2_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103і
AssignVariableOp_103AssignVariableOp)assignvariableop_103_adamw_dense_2_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104√
AssignVariableOp_104AssignVariableOp8assignvariableop_104_adamw_layer_normalization_2_gamma_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105¬
AssignVariableOp_105AssignVariableOp7assignvariableop_105_adamw_layer_normalization_2_beta_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106√
AssignVariableOp_106AssignVariableOp8assignvariableop_106_adamw_layer_normalization_3_gamma_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107¬
AssignVariableOp_107AssignVariableOp7assignvariableop_107_adamw_layer_normalization_3_beta_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108ґ
AssignVariableOp_108AssignVariableOp+assignvariableop_108_adamw_dense_3_kernel_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109і
AssignVariableOp_109AssignVariableOp)assignvariableop_109_adamw_dense_3_bias_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110ґ
AssignVariableOp_110AssignVariableOp+assignvariableop_110_adamw_dense_4_kernel_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111і
AssignVariableOp_111AssignVariableOp)assignvariableop_111_adamw_dense_4_bias_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112√
AssignVariableOp_112AssignVariableOp8assignvariableop_112_adamw_layer_normalization_4_gamma_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113¬
AssignVariableOp_113AssignVariableOp7assignvariableop_113_adamw_layer_normalization_4_beta_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114ґ
AssignVariableOp_114AssignVariableOp+assignvariableop_114_adamw_dense_5_kernel_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115і
AssignVariableOp_115AssignVariableOp)assignvariableop_115_adamw_dense_5_bias_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116ґ
AssignVariableOp_116AssignVariableOp+assignvariableop_116_adamw_dense_6_kernel_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117і
AssignVariableOp_117AssignVariableOp)assignvariableop_117_adamw_dense_6_bias_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118ґ
AssignVariableOp_118AssignVariableOp+assignvariableop_118_adamw_dense_7_kernel_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119і
AssignVariableOp_119AssignVariableOp)assignvariableop_119_adamw_dense_7_bias_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120¬
AssignVariableOp_120AssignVariableOp7assignvariableop_120_adamw_patch_encoder_dense_kernel_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121ј
AssignVariableOp_121AssignVariableOp5assignvariableop_121_adamw_patch_encoder_dense_bias_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122 
AssignVariableOp_122AssignVariableOp?assignvariableop_122_adamw_patch_encoder_embedding_embeddings_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123…
AssignVariableOp_123AssignVariableOp>assignvariableop_123_adamw_multi_head_attention_query_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124«
AssignVariableOp_124AssignVariableOp<assignvariableop_124_adamw_multi_head_attention_query_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125«
AssignVariableOp_125AssignVariableOp<assignvariableop_125_adamw_multi_head_attention_key_kernel_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126≈
AssignVariableOp_126AssignVariableOp:assignvariableop_126_adamw_multi_head_attention_key_bias_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127…
AssignVariableOp_127AssignVariableOp>assignvariableop_127_adamw_multi_head_attention_value_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128«
AssignVariableOp_128AssignVariableOp<assignvariableop_128_adamw_multi_head_attention_value_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129‘
AssignVariableOp_129AssignVariableOpIassignvariableop_129_adamw_multi_head_attention_attention_output_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130“
AssignVariableOp_130AssignVariableOpGassignvariableop_130_adamw_multi_head_attention_attention_output_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131Ћ
AssignVariableOp_131AssignVariableOp@assignvariableop_131_adamw_multi_head_attention_1_query_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132…
AssignVariableOp_132AssignVariableOp>assignvariableop_132_adamw_multi_head_attention_1_query_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133…
AssignVariableOp_133AssignVariableOp>assignvariableop_133_adamw_multi_head_attention_1_key_kernel_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134«
AssignVariableOp_134AssignVariableOp<assignvariableop_134_adamw_multi_head_attention_1_key_bias_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135Ћ
AssignVariableOp_135AssignVariableOp@assignvariableop_135_adamw_multi_head_attention_1_value_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136…
AssignVariableOp_136AssignVariableOp>assignvariableop_136_adamw_multi_head_attention_1_value_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137÷
AssignVariableOp_137AssignVariableOpKassignvariableop_137_adamw_multi_head_attention_1_attention_output_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138‘
AssignVariableOp_138AssignVariableOpIassignvariableop_138_adamw_multi_head_attention_1_attention_output_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpщ
Identity_139Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_139н
Identity_140IdentityIdentity_139:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_140"%
identity_140Identity_140:output:0*≠
_input_shapesЫ
Ш: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_138AssignVariableOp_1382*
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
≤
ц
C__inference_dense_5_layer_call_and_return_conditional_losses_587932

inputs2
matmul_readvariableop_resource:
А»2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А»2*
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
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xt
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xБ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
Gelu/truediv_
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xr
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/addm

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/mul_1У
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:€€€€€€€€€А»: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:€€€€€€€€€А»
 
_user_specified_nameinputs
н
k
?__inference_add_layer_call_and_return_conditional_losses_587428
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:€€€€€€€€€Р@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:V R
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/1
№
R
&__inference_add_2_layer_call_fn_587735
inputs_0
inputs_1
identity‘
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_5848802
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:V R
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/1
»
Я
6__inference_layer_normalization_3_layer_call_fn_587766

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_5849042
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
»
Я
6__inference_layer_normalization_4_layer_call_fn_587903

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_5850282
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
Л.
щ
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_587336	
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
"query/einsum/Einsum/ReadVariableOp»
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
query/einsum/EinsumЦ
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype02
query/add/ReadVariableOpЪ
	query/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
	query/add≤
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02"
 key/einsum/Einsum/ReadVariableOp¬
key/einsum/EinsumEinsumvalue(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
key/einsum/EinsumР
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype02
key/add/ReadVariableOpТ
key/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2	
key/addЄ
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"value/einsum/Einsum/ReadVariableOp»
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2
value/einsum/EinsumЦ
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype02
value/add/ReadVariableOpЪ
	value/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
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
:€€€€€€€€€Р@2
MulҐ
einsum/EinsumEinsumkey/add:z:0Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2
einsum/EinsumБ
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
softmax/SoftmaxЗ
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€РР2
dropout/Identityє
einsum_1/EinsumEinsumdropout/Identity:output:0value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2
einsum_1/Einsumў
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02/
-attention_output/einsum/Einsum/ReadVariableOpш
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe2 
attention_output/einsum/Einsum≥
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02%
#attention_output/add/ReadVariableOp¬
attention_output/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
attention_output/addГ
IdentityIdentityattention_output/add:z:0$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 2J
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
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
п
m
A__inference_add_1_layer_call_and_return_conditional_losses_587565
inputs_0
inputs_1
identity^
addAddV2inputs_0inputs_1*
T0*,
_output_shapes
:€€€€€€€€€Р@2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:V R
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/1
Ц'
ь
C__inference_dense_3_layer_call_and_return_conditional_losses_587804

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
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
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
Tensordot/concat_1Т
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2	
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
:€€€€€€€€€РА2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xЗ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
Gelu/truedive
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xx
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/adds

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/mul_1Ь
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€РА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
І
_
C__inference_patches_layer_call_and_return_conditional_losses_587214

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
:€€€€€€€€€ђ*
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
B :ђ2
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
!:€€€€€€€€€€€€€€€€€€ђ2	
Reshaper
IdentityIdentityReshape:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€»»:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameimages
Ѓt
ё
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_585747

inputs'
patch_encoder_585640:	ђ@"
patch_encoder_585642:@'
patch_encoder_585644:	Р@(
layer_normalization_585647:@(
layer_normalization_585649:@1
multi_head_attention_585652:@@-
multi_head_attention_585654:@1
multi_head_attention_585656:@@-
multi_head_attention_585658:@1
multi_head_attention_585660:@@-
multi_head_attention_585662:@1
multi_head_attention_585664:@@)
multi_head_attention_585666:@*
layer_normalization_1_585670:@*
layer_normalization_1_585672:@!
dense_1_585675:	@А
dense_1_585677:	А!
dense_2_585680:	А@
dense_2_585682:@*
layer_normalization_2_585686:@*
layer_normalization_2_585688:@3
multi_head_attention_1_585691:@@/
multi_head_attention_1_585693:@3
multi_head_attention_1_585695:@@/
multi_head_attention_1_585697:@3
multi_head_attention_1_585699:@@/
multi_head_attention_1_585701:@3
multi_head_attention_1_585703:@@+
multi_head_attention_1_585705:@*
layer_normalization_3_585709:@*
layer_normalization_3_585711:@!
dense_3_585714:	@А
dense_3_585716:	А!
dense_4_585719:	А@
dense_4_585721:@*
layer_normalization_4_585725:@*
layer_normalization_4_585727:@"
dense_5_585731:
А»2
dense_5_585733:2 
dense_6_585736:22
dense_6_585738:2 
dense_7_585741:2
dense_7_585743:
identityИҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐ+layer_normalization/StatefulPartitionedCallҐ-layer_normalization_1/StatefulPartitionedCallҐ-layer_normalization_2/StatefulPartitionedCallҐ-layer_normalization_3/StatefulPartitionedCallҐ-layer_normalization_4/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ.multi_head_attention_1/StatefulPartitionedCallҐ%patch_encoder/StatefulPartitionedCallв
patches/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_5845302
patches/PartitionedCallз
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_585640patch_encoder_585642patch_encoder_585644*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_5845722'
%patch_encoder/StatefulPartitionedCallы
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_585647layer_normalization_585649*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5846022-
+layer_normalization/StatefulPartitionedCallч
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_585652multi_head_attention_585654multi_head_attention_585656multi_head_attention_585658multi_head_attention_585660multi_head_attention_585662multi_head_attention_585664multi_head_attention_585666*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_5854962.
,multi_head_attention/StatefulPartitionedCall≠
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_5846672
add/PartitionedCallу
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_585670layer_normalization_1_585672*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5846912/
-layer_normalization_1/StatefulPartitionedCall»
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_585675dense_1_585677*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5847352!
dense_1/StatefulPartitionedCallє
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_585680dense_2_585682*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_5847792!
dense_2/StatefulPartitionedCallФ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5847912
add_1/PartitionedCallх
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_585686layer_normalization_2_585688*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5848152/
-layer_normalization_2/StatefulPartitionedCallС
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_585691multi_head_attention_1_585693multi_head_attention_1_585695multi_head_attention_1_585697multi_head_attention_1_585699multi_head_attention_1_585701multi_head_attention_1_585703multi_head_attention_1_585705*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_58535520
.multi_head_attention_1/StatefulPartitionedCall•
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_5848802
add_2/PartitionedCallх
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_585709layer_normalization_3_585711*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_5849042/
-layer_normalization_3/StatefulPartitionedCall»
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_585714dense_3_585716*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_5849482!
dense_3/StatefulPartitionedCallє
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_585719dense_4_585721*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_5849922!
dense_4/StatefulPartitionedCallЦ
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_5850042
add_3/PartitionedCallх
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_585725layer_normalization_4_585727*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_5850282/
-layer_normalization_4/StatefulPartitionedCallЖ
flatten/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€А»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5850402
flatten/PartitionedCallђ
dense_5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5_585731dense_5_585733*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_5850602!
dense_5/StatefulPartitionedCallі
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_585736dense_6_585738*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_5850842!
dense_6/StatefulPartitionedCallі
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_585741dense_7_585743*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_5851012!
dense_7/StatefulPartitionedCallа
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
±
Р
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_585028

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
ч
∆

6__inference_WheatClassifier_VIT_3_layer_call_fn_587200

inputs
unknown:	ђ@
	unknown_0:@
	unknown_1:	Р@
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

unknown_14:	@А

unknown_15:	А

unknown_16:	А@

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

unknown_30:	@А

unknown_31:	А

unknown_32:	А@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:
А»2

unknown_37:2

unknown_38:22

unknown_39:2

unknown_40:2

unknown_41:
identityИҐStatefulPartitionedCallє
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
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_5857472
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
№
R
&__inference_add_3_layer_call_fn_587872
inputs_0
inputs_1
identity‘
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_5850042
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:V R
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/1
≥Ц
ЇD
__inference__traced_save_588428
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
5savev2_layer_normalization_4_beta_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop)
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
=savev2_adamw_layer_normalization_4_beta_m_read_readvariableop5
1savev2_adamw_dense_5_kernel_m_read_readvariableop3
/savev2_adamw_dense_5_bias_m_read_readvariableop5
1savev2_adamw_dense_6_kernel_m_read_readvariableop3
/savev2_adamw_dense_6_bias_m_read_readvariableop5
1savev2_adamw_dense_7_kernel_m_read_readvariableop3
/savev2_adamw_dense_7_bias_m_read_readvariableopA
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
=savev2_adamw_layer_normalization_4_beta_v_read_readvariableop5
1savev2_adamw_dense_5_kernel_v_read_readvariableop3
/savev2_adamw_dense_5_bias_v_read_readvariableop5
1savev2_adamw_dense_6_kernel_v_read_readvariableop3
/savev2_adamw_dense_6_bias_v_read_readvariableop5
1savev2_adamw_dense_7_kernel_v_read_readvariableop3
/savev2_adamw_dense_7_bias_v_read_readvariableopA
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
ShardedFilename≈M
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:М*
dtype0*÷L
valueћLB…LМB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/28/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names•
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:М*
dtype0*Ѓ
value§B°МB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesлA
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop6savev2_layer_normalization_2_gamma_read_readvariableop5savev2_layer_normalization_2_beta_read_readvariableop6savev2_layer_normalization_3_gamma_read_readvariableop5savev2_layer_normalization_3_beta_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop6savev2_layer_normalization_4_gamma_read_readvariableop5savev2_layer_normalization_4_beta_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop%savev2_adamw_iter_read_readvariableop'savev2_adamw_beta_1_read_readvariableop'savev2_adamw_beta_2_read_readvariableop&savev2_adamw_decay_read_readvariableop.savev2_adamw_learning_rate_read_readvariableop-savev2_adamw_weight_decay_read_readvariableop5savev2_patch_encoder_dense_kernel_read_readvariableop3savev2_patch_encoder_dense_bias_read_readvariableop=savev2_patch_encoder_embedding_embeddings_read_readvariableop<savev2_multi_head_attention_query_kernel_read_readvariableop:savev2_multi_head_attention_query_bias_read_readvariableop:savev2_multi_head_attention_key_kernel_read_readvariableop8savev2_multi_head_attention_key_bias_read_readvariableop<savev2_multi_head_attention_value_kernel_read_readvariableop:savev2_multi_head_attention_value_bias_read_readvariableopGsavev2_multi_head_attention_attention_output_kernel_read_readvariableopEsavev2_multi_head_attention_attention_output_bias_read_readvariableop>savev2_multi_head_attention_1_query_kernel_read_readvariableop<savev2_multi_head_attention_1_query_bias_read_readvariableop<savev2_multi_head_attention_1_key_kernel_read_readvariableop:savev2_multi_head_attention_1_key_bias_read_readvariableop>savev2_multi_head_attention_1_value_kernel_read_readvariableop<savev2_multi_head_attention_1_value_bias_read_readvariableopIsavev2_multi_head_attention_1_attention_output_kernel_read_readvariableopGsavev2_multi_head_attention_1_attention_output_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop<savev2_adamw_layer_normalization_gamma_m_read_readvariableop;savev2_adamw_layer_normalization_beta_m_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_1_beta_m_read_readvariableop1savev2_adamw_dense_1_kernel_m_read_readvariableop/savev2_adamw_dense_1_bias_m_read_readvariableop1savev2_adamw_dense_2_kernel_m_read_readvariableop/savev2_adamw_dense_2_bias_m_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_2_beta_m_read_readvariableop>savev2_adamw_layer_normalization_3_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_3_beta_m_read_readvariableop1savev2_adamw_dense_3_kernel_m_read_readvariableop/savev2_adamw_dense_3_bias_m_read_readvariableop1savev2_adamw_dense_4_kernel_m_read_readvariableop/savev2_adamw_dense_4_bias_m_read_readvariableop>savev2_adamw_layer_normalization_4_gamma_m_read_readvariableop=savev2_adamw_layer_normalization_4_beta_m_read_readvariableop1savev2_adamw_dense_5_kernel_m_read_readvariableop/savev2_adamw_dense_5_bias_m_read_readvariableop1savev2_adamw_dense_6_kernel_m_read_readvariableop/savev2_adamw_dense_6_bias_m_read_readvariableop1savev2_adamw_dense_7_kernel_m_read_readvariableop/savev2_adamw_dense_7_bias_m_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_m_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_m_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_m_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_m_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_m_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_m_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_m_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_m_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_m_read_readvariableopFsavev2_adamw_multi_head_attention_1_query_kernel_m_read_readvariableopDsavev2_adamw_multi_head_attention_1_query_bias_m_read_readvariableopDsavev2_adamw_multi_head_attention_1_key_kernel_m_read_readvariableopBsavev2_adamw_multi_head_attention_1_key_bias_m_read_readvariableopFsavev2_adamw_multi_head_attention_1_value_kernel_m_read_readvariableopDsavev2_adamw_multi_head_attention_1_value_bias_m_read_readvariableopQsavev2_adamw_multi_head_attention_1_attention_output_kernel_m_read_readvariableopOsavev2_adamw_multi_head_attention_1_attention_output_bias_m_read_readvariableop<savev2_adamw_layer_normalization_gamma_v_read_readvariableop;savev2_adamw_layer_normalization_beta_v_read_readvariableop>savev2_adamw_layer_normalization_1_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_1_beta_v_read_readvariableop1savev2_adamw_dense_1_kernel_v_read_readvariableop/savev2_adamw_dense_1_bias_v_read_readvariableop1savev2_adamw_dense_2_kernel_v_read_readvariableop/savev2_adamw_dense_2_bias_v_read_readvariableop>savev2_adamw_layer_normalization_2_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_2_beta_v_read_readvariableop>savev2_adamw_layer_normalization_3_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_3_beta_v_read_readvariableop1savev2_adamw_dense_3_kernel_v_read_readvariableop/savev2_adamw_dense_3_bias_v_read_readvariableop1savev2_adamw_dense_4_kernel_v_read_readvariableop/savev2_adamw_dense_4_bias_v_read_readvariableop>savev2_adamw_layer_normalization_4_gamma_v_read_readvariableop=savev2_adamw_layer_normalization_4_beta_v_read_readvariableop1savev2_adamw_dense_5_kernel_v_read_readvariableop/savev2_adamw_dense_5_bias_v_read_readvariableop1savev2_adamw_dense_6_kernel_v_read_readvariableop/savev2_adamw_dense_6_bias_v_read_readvariableop1savev2_adamw_dense_7_kernel_v_read_readvariableop/savev2_adamw_dense_7_bias_v_read_readvariableop=savev2_adamw_patch_encoder_dense_kernel_v_read_readvariableop;savev2_adamw_patch_encoder_dense_bias_v_read_readvariableopEsavev2_adamw_patch_encoder_embedding_embeddings_v_read_readvariableopDsavev2_adamw_multi_head_attention_query_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_query_bias_v_read_readvariableopBsavev2_adamw_multi_head_attention_key_kernel_v_read_readvariableop@savev2_adamw_multi_head_attention_key_bias_v_read_readvariableopDsavev2_adamw_multi_head_attention_value_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_value_bias_v_read_readvariableopOsavev2_adamw_multi_head_attention_attention_output_kernel_v_read_readvariableopMsavev2_adamw_multi_head_attention_attention_output_bias_v_read_readvariableopFsavev2_adamw_multi_head_attention_1_query_kernel_v_read_readvariableopDsavev2_adamw_multi_head_attention_1_query_bias_v_read_readvariableopDsavev2_adamw_multi_head_attention_1_key_kernel_v_read_readvariableopBsavev2_adamw_multi_head_attention_1_key_bias_v_read_readvariableopFsavev2_adamw_multi_head_attention_1_value_kernel_v_read_readvariableopDsavev2_adamw_multi_head_attention_1_value_bias_v_read_readvariableopQsavev2_adamw_multi_head_attention_1_attention_output_kernel_v_read_readvariableopOsavev2_adamw_multi_head_attention_1_attention_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Э
dtypesТ
П2М	2
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

identity_1Identity_1:output:0*≈	
_input_shapes≥	
∞	: :@:@:@:@:	@А:А:	А@:@:@:@:@:@:	@А:А:	А@:@:@:@:
А»2:2:22:2:2:: : : : : : :	ђ@:@:	Р@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@: : : : :@:@:@:@:	@А:А:	А@:@:@:@:@:@:	@А:А:	А@:@:@:@:
А»2:2:22:2:2::	ђ@:@:	Р@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@:@:@:@:	@А:А:	А@:@:@:@:@:@:	@А:А:	А@:@:@:@:
А»2:2:22:2:2::	ђ@:@:	Р@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@@:@: 2(
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
:	@А:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 
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
:	@А:!

_output_shapes	
:А:%!

_output_shapes
:	А@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
А»2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	ђ@:  

_output_shapes
:@:%!!

_output_shapes
:	Р@:("$
"
_output_shapes
:@@:$# 

_output_shapes

:@:($$
"
_output_shapes
:@@:$% 

_output_shapes

:@:(&$
"
_output_shapes
:@@:$' 

_output_shapes

:@:(($
"
_output_shapes
:@@: )

_output_shapes
:@:(*$
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
:@:2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: : 6

_output_shapes
:@: 7

_output_shapes
:@: 8

_output_shapes
:@: 9

_output_shapes
:@:%:!

_output_shapes
:	@А:!;

_output_shapes	
:А:%<!

_output_shapes
:	А@: =

_output_shapes
:@: >
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
:	@А:!C

_output_shapes	
:А:%D!

_output_shapes
:	А@: E

_output_shapes
:@: F

_output_shapes
:@: G

_output_shapes
:@:&H"
 
_output_shapes
:
А»2: I

_output_shapes
:2:$J 

_output_shapes

:22: K

_output_shapes
:2:$L 

_output_shapes

:2: M

_output_shapes
::%N!

_output_shapes
:	ђ@: O

_output_shapes
:@:%P!

_output_shapes
:	Р@:(Q$
"
_output_shapes
:@@:$R 

_output_shapes

:@:(S$
"
_output_shapes
:@@:$T 

_output_shapes

:@:(U$
"
_output_shapes
:@@:$V 

_output_shapes

:@:(W$
"
_output_shapes
:@@: X

_output_shapes
:@:(Y$
"
_output_shapes
:@@:$Z 

_output_shapes

:@:([$
"
_output_shapes
:@@:$\ 

_output_shapes

:@:(]$
"
_output_shapes
:@@:$^ 

_output_shapes

:@:(_$
"
_output_shapes
:@@: `

_output_shapes
:@: a

_output_shapes
:@: b

_output_shapes
:@: c

_output_shapes
:@: d

_output_shapes
:@:%e!

_output_shapes
:	@А:!f

_output_shapes	
:А:%g!

_output_shapes
:	А@: h

_output_shapes
:@: i

_output_shapes
:@: j

_output_shapes
:@: k

_output_shapes
:@: l

_output_shapes
:@:%m!

_output_shapes
:	@А:!n

_output_shapes	
:А:%o!

_output_shapes
:	А@: p

_output_shapes
:@: q

_output_shapes
:@: r

_output_shapes
:@:&s"
 
_output_shapes
:
А»2: t

_output_shapes
:2:$u 

_output_shapes

:22: v

_output_shapes
:2:$w 

_output_shapes

:2: x

_output_shapes
::%y!

_output_shapes
:	ђ@: z

_output_shapes
:@:%{!

_output_shapes
:	Р@:(|$
"
_output_shapes
:@@:$} 

_output_shapes

:@:(~$
"
_output_shapes
:@@:$ 

_output_shapes

:@:)А$
"
_output_shapes
:@@:%Б 

_output_shapes

:@:)В$
"
_output_shapes
:@@:!Г

_output_shapes
:@:)Д$
"
_output_shapes
:@@:%Е 

_output_shapes

:@:)Ж$
"
_output_shapes
:@@:%З 

_output_shapes

:@:)И$
"
_output_shapes
:@@:%Й 

_output_shapes

:@:)К$
"
_output_shapes
:@@:!Л

_output_shapes
:@:М

_output_shapes
: 
±t
я
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_586149
input_1'
patch_encoder_586042:	ђ@"
patch_encoder_586044:@'
patch_encoder_586046:	Р@(
layer_normalization_586049:@(
layer_normalization_586051:@1
multi_head_attention_586054:@@-
multi_head_attention_586056:@1
multi_head_attention_586058:@@-
multi_head_attention_586060:@1
multi_head_attention_586062:@@-
multi_head_attention_586064:@1
multi_head_attention_586066:@@)
multi_head_attention_586068:@*
layer_normalization_1_586072:@*
layer_normalization_1_586074:@!
dense_1_586077:	@А
dense_1_586079:	А!
dense_2_586082:	А@
dense_2_586084:@*
layer_normalization_2_586088:@*
layer_normalization_2_586090:@3
multi_head_attention_1_586093:@@/
multi_head_attention_1_586095:@3
multi_head_attention_1_586097:@@/
multi_head_attention_1_586099:@3
multi_head_attention_1_586101:@@/
multi_head_attention_1_586103:@3
multi_head_attention_1_586105:@@+
multi_head_attention_1_586107:@*
layer_normalization_3_586111:@*
layer_normalization_3_586113:@!
dense_3_586116:	@А
dense_3_586118:	А!
dense_4_586121:	А@
dense_4_586123:@*
layer_normalization_4_586127:@*
layer_normalization_4_586129:@"
dense_5_586133:
А»2
dense_5_586135:2 
dense_6_586138:22
dense_6_586140:2 
dense_7_586143:2
dense_7_586145:
identityИҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐ+layer_normalization/StatefulPartitionedCallҐ-layer_normalization_1/StatefulPartitionedCallҐ-layer_normalization_2/StatefulPartitionedCallҐ-layer_normalization_3/StatefulPartitionedCallҐ-layer_normalization_4/StatefulPartitionedCallҐ,multi_head_attention/StatefulPartitionedCallҐ.multi_head_attention_1/StatefulPartitionedCallҐ%patch_encoder/StatefulPartitionedCallг
patches/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_5845302
patches/PartitionedCallз
%patch_encoder/StatefulPartitionedCallStatefulPartitionedCall patches/PartitionedCall:output:0patch_encoder_586042patch_encoder_586044patch_encoder_586046*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_patch_encoder_layer_call_and_return_conditional_losses_5845722'
%patch_encoder/StatefulPartitionedCallы
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall.patch_encoder/StatefulPartitionedCall:output:0layer_normalization_586049layer_normalization_586051*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_layer_normalization_layer_call_and_return_conditional_losses_5846022-
+layer_normalization/StatefulPartitionedCallч
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:04layer_normalization/StatefulPartitionedCall:output:0multi_head_attention_586054multi_head_attention_586056multi_head_attention_586058multi_head_attention_586060multi_head_attention_586062multi_head_attention_586064multi_head_attention_586066multi_head_attention_586068*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_5854962.
,multi_head_attention/StatefulPartitionedCall≠
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:0.patch_encoder/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_5846672
add/PartitionedCallу
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0layer_normalization_1_586072layer_normalization_1_586074*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_5846912/
-layer_normalization_1/StatefulPartitionedCall»
dense_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0dense_1_586077dense_1_586079*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5847352!
dense_1/StatefulPartitionedCallє
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_586082dense_2_586084*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_5847792!
dense_2/StatefulPartitionedCallФ
add_1/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0add/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_1_layer_call_and_return_conditional_losses_5847912
add_1/PartitionedCallх
-layer_normalization_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0layer_normalization_2_586088layer_normalization_2_586090*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_5848152/
-layer_normalization_2/StatefulPartitionedCallС
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_2/StatefulPartitionedCall:output:06layer_normalization_2/StatefulPartitionedCall:output:0multi_head_attention_1_586093multi_head_attention_1_586095multi_head_attention_1_586097multi_head_attention_1_586099multi_head_attention_1_586101multi_head_attention_1_586103multi_head_attention_1_586105multi_head_attention_1_586107*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *[
fVRT
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_58535520
.multi_head_attention_1/StatefulPartitionedCall•
add_2/PartitionedCallPartitionedCall7multi_head_attention_1/StatefulPartitionedCall:output:0add_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_2_layer_call_and_return_conditional_losses_5848802
add_2/PartitionedCallх
-layer_normalization_3/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0layer_normalization_3_586111layer_normalization_3_586113*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_5849042/
-layer_normalization_3/StatefulPartitionedCall»
dense_3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_3/StatefulPartitionedCall:output:0dense_3_586116dense_3_586118*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_5849482!
dense_3/StatefulPartitionedCallє
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_586121dense_4_586123*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_5849922!
dense_4/StatefulPartitionedCallЦ
add_3/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0add_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_add_3_layer_call_and_return_conditional_losses_5850042
add_3/PartitionedCallх
-layer_normalization_4/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0layer_normalization_4_586127layer_normalization_4_586129*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_5850282/
-layer_normalization_4/StatefulPartitionedCallЖ
flatten/PartitionedCallPartitionedCall6layer_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:€€€€€€€€€А»* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_5850402
flatten/PartitionedCallђ
dense_5/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_5_586133dense_5_586135*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_5850602!
dense_5/StatefulPartitionedCallі
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_586138dense_6_586140*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_5850842!
dense_6/StatefulPartitionedCallі
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_586143dense_7_586145*
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
GPU2*0J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_5851012!
dense_7/StatefulPartitionedCallа
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall.^layer_normalization_2/StatefulPartitionedCall.^layer_normalization_3/StatefulPartitionedCall.^layer_normalization_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall&^patch_encoder/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2^
-layer_normalization_2/StatefulPartitionedCall-layer_normalization_2/StatefulPartitionedCall2^
-layer_normalization_3/StatefulPartitionedCall-layer_normalization_3/StatefulPartitionedCall2^
-layer_normalization_4/StatefulPartitionedCall-layer_normalization_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2N
%patch_encoder/StatefulPartitionedCall%patch_encoder/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€»»
!
_user_specified_name	input_1
і
Ч
(__inference_dense_1_layer_call_fn_587512

inputs
unknown:	@А
	unknown_0:	А
identityИҐStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *-
_output_shapes
:€€€€€€€€€РА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5847352
StatefulPartitionedCallФ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*-
_output_shapes
:€€€€€€€€€РА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
Д¬
÷(
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_586627

inputsH
5patch_encoder_dense_tensordot_readvariableop_resource:	ђ@A
3patch_encoder_dense_biasadd_readvariableop_resource:@B
/patch_encoder_embedding_embedding_lookup_586293:	Р@G
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
)dense_3_tensordot_readvariableop_resource:	@А6
'dense_3_biasadd_readvariableop_resource:	А<
)dense_4_tensordot_readvariableop_resource:	А@5
'dense_4_biasadd_readvariableop_resource:@I
;layer_normalization_4_batchnorm_mul_readvariableop_resource:@E
7layer_normalization_4_batchnorm_readvariableop_resource:@:
&dense_5_matmul_readvariableop_resource:
А»25
'dense_5_biasadd_readvariableop_resource:28
&dense_6_matmul_readvariableop_resource:225
'dense_6_biasadd_readvariableop_resource:28
&dense_7_matmul_readvariableop_resource:25
'dense_7_biasadd_readvariableop_resource:
identityИҐdense_1/BiasAdd/ReadVariableOpҐ dense_1/Tensordot/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐ dense_2/Tensordot/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐ dense_3/Tensordot/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐ dense_4/Tensordot/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ,layer_normalization/batchnorm/ReadVariableOpҐ0layer_normalization/batchnorm/mul/ReadVariableOpҐ.layer_normalization_1/batchnorm/ReadVariableOpҐ2layer_normalization_1/batchnorm/mul/ReadVariableOpҐ.layer_normalization_2/batchnorm/ReadVariableOpҐ2layer_normalization_2/batchnorm/mul/ReadVariableOpҐ.layer_normalization_3/batchnorm/ReadVariableOpҐ2layer_normalization_3/batchnorm/mul/ReadVariableOpҐ.layer_normalization_4/batchnorm/ReadVariableOpҐ2layer_normalization_4/batchnorm/mul/ReadVariableOpҐ8multi_head_attention/attention_output/add/ReadVariableOpҐBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpҐ+multi_head_attention/key/add/ReadVariableOpҐ5multi_head_attention/key/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/query/add/ReadVariableOpҐ7multi_head_attention/query/einsum/Einsum/ReadVariableOpҐ-multi_head_attention/value/add/ReadVariableOpҐ7multi_head_attention/value/einsum/Einsum/ReadVariableOpҐ:multi_head_attention_1/attention_output/add/ReadVariableOpҐDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpҐ-multi_head_attention_1/key/add/ReadVariableOpҐ7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/query/add/ReadVariableOpҐ9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpҐ/multi_head_attention_1/value/add/ReadVariableOpҐ9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpҐ*patch_encoder/dense/BiasAdd/ReadVariableOpҐ,patch_encoder/dense/Tensordot/ReadVariableOpҐ(patch_encoder/embedding/embedding_lookupT
patches/ShapeShapeinputs*
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
patches/strided_sliceд
patches/ExtractImagePatchesExtractImagePatchesinputs*
T0*0
_output_shapes
:€€€€€€€€€ђ*
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
B :ђ2
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
!:€€€€€€€€€€€€€€€€€€ђ2
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
B :Р2
patch_encoder/range/limitx
patch_encoder/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
patch_encoder/range/deltaЉ
patch_encoder/rangeRange"patch_encoder/range/start:output:0"patch_encoder/range/limit:output:0"patch_encoder/range/delta:output:0*
_output_shapes	
:Р2
patch_encoder/range”
,patch_encoder/dense/Tensordot/ReadVariableOpReadVariableOp5patch_encoder_dense_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
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
!:€€€€€€€€€€€€€€€€€€ђ2)
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
patch_encoder/dense/BiasAddе
(patch_encoder/embedding/embedding_lookupResourceGather/patch_encoder_embedding_embedding_lookup_586293patch_encoder/range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/586293*
_output_shapes
:	Р@*
dtype02*
(patch_encoder/embedding/embedding_lookupЅ
1patch_encoder/embedding/embedding_lookup/IdentityIdentity1patch_encoder/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*B
_class8
64loc:@patch_encoder/embedding/embedding_lookup/586293*
_output_shapes
:	Р@23
1patch_encoder/embedding/embedding_lookup/Identity№
3patch_encoder/embedding/embedding_lookup/Identity_1Identity:patch_encoder/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Р@25
3patch_encoder/embedding/embedding_lookup/Identity_1 
patch_encoder/addAddV2$patch_encoder/dense/BiasAdd:output:0<patch_encoder/embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
patch_encoder/add≤
2layer_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:24
2layer_normalization/moments/mean/reduction_indicesи
 layer_normalization/moments/meanMeanpatch_encoder/add:z:0;layer_normalization/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2"
 layer_normalization/moments/mean∆
(layer_normalization/moments/StopGradientStopGradient)layer_normalization/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2*
(layer_normalization/moments/StopGradientф
-layer_normalization/moments/SquaredDifferenceSquaredDifferencepatch_encoder/add:z:01layer_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2/
-layer_normalization/moments/SquaredDifferenceЇ
6layer_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:28
6layer_normalization/moments/variance/reduction_indicesР
$layer_normalization/moments/varianceMean1layer_normalization/moments/SquaredDifference:z:0?layer_normalization/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2&
$layer_normalization/moments/varianceП
#layer_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52%
#layer_normalization/batchnorm/add/yг
!layer_normalization/batchnorm/addAddV2-layer_normalization/moments/variance:output:0,layer_normalization/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2#
!layer_normalization/batchnorm/add±
#layer_normalization/batchnorm/RsqrtRsqrt%layer_normalization/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization/batchnorm/RsqrtЏ
0layer_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9layer_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype022
0layer_normalization/batchnorm/mul/ReadVariableOpз
!layer_normalization/batchnorm/mulMul'layer_normalization/batchnorm/Rsqrt:y:08layer_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2#
!layer_normalization/batchnorm/mul∆
#layer_normalization/batchnorm/mul_1Mulpatch_encoder/add:z:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization/batchnorm/mul_1Џ
#layer_normalization/batchnorm/mul_2Mul)layer_normalization/moments/mean:output:0%layer_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization/batchnorm/mul_2ќ
,layer_normalization/batchnorm/ReadVariableOpReadVariableOp5layer_normalization_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02.
,layer_normalization/batchnorm/ReadVariableOpг
!layer_normalization/batchnorm/subSub4layer_normalization/batchnorm/ReadVariableOp:value:0'layer_normalization/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2#
!layer_normalization/batchnorm/subЏ
#layer_normalization/batchnorm/add_1AddV2'layer_normalization/batchnorm/mul_1:z:0%layer_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization/batchnorm/add_1ч
7multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/query/einsum/Einsum/ReadVariableOp©
(multi_head_attention/query/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2*
(multi_head_attention/query/einsum/Einsum’
-multi_head_attention/query/add/ReadVariableOpReadVariableOp6multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/query/add/ReadVariableOpо
multi_head_attention/query/addAddV21multi_head_attention/query/einsum/Einsum:output:05multi_head_attention/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2 
multi_head_attention/query/addс
5multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOp>multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype027
5multi_head_attention/key/einsum/Einsum/ReadVariableOp£
&multi_head_attention/key/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0=multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2(
&multi_head_attention/key/einsum/Einsumѕ
+multi_head_attention/key/add/ReadVariableOpReadVariableOp4multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02-
+multi_head_attention/key/add/ReadVariableOpж
multi_head_attention/key/addAddV2/multi_head_attention/key/einsum/Einsum:output:03multi_head_attention/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
multi_head_attention/key/addч
7multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention/value/einsum/Einsum/ReadVariableOp©
(multi_head_attention/value/einsum/EinsumEinsum'layer_normalization/batchnorm/add_1:z:0?multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2*
(multi_head_attention/value/einsum/Einsum’
-multi_head_attention/value/add/ReadVariableOpReadVariableOp6multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention/value/add/ReadVariableOpо
multi_head_attention/value/addAddV21multi_head_attention/value/einsum/Einsum:output:05multi_head_attention/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2 
multi_head_attention/value/add}
multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention/Mul/yњ
multi_head_attention/MulMul"multi_head_attention/query/add:z:0#multi_head_attention/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
multi_head_attention/Mulц
"multi_head_attention/einsum/EinsumEinsum multi_head_attention/key/add:z:0multi_head_attention/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2$
"multi_head_attention/einsum/Einsumј
$multi_head_attention/softmax/SoftmaxSoftmax+multi_head_attention/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2&
$multi_head_attention/softmax/Softmax∆
%multi_head_attention/dropout/IdentityIdentity.multi_head_attention/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€РР2'
%multi_head_attention/dropout/IdentityН
$multi_head_attention/einsum_1/EinsumEinsum.multi_head_attention/dropout/Identity:output:0"multi_head_attention/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2&
$multi_head_attention/einsum_1/EinsumШ
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpKmulti_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02D
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpћ
3multi_head_attention/attention_output/einsum/EinsumEinsum-multi_head_attention/einsum_1/Einsum:output:0Jmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe25
3multi_head_attention/attention_output/einsum/Einsumт
8multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpAmulti_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02:
8multi_head_attention/attention_output/add/ReadVariableOpЦ
)multi_head_attention/attention_output/addAddV2<multi_head_attention/attention_output/einsum/Einsum:output:0@multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2+
)multi_head_attention/attention_output/addШ
add/addAddV2-multi_head_attention/attention_output/add:z:0patch_encoder/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2	
add/addґ
4layer_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_1/moments/mean/reduction_indicesд
"layer_normalization_1/moments/meanMeanadd/add:z:0=layer_normalization_1/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2$
"layer_normalization_1/moments/meanћ
*layer_normalization_1/moments/StopGradientStopGradient+layer_normalization_1/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2,
*layer_normalization_1/moments/StopGradientр
/layer_normalization_1/moments/SquaredDifferenceSquaredDifferenceadd/add:z:03layer_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@21
/layer_normalization_1/moments/SquaredDifferenceЊ
8layer_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_1/moments/variance/reduction_indicesШ
&layer_normalization_1/moments/varianceMean3layer_normalization_1/moments/SquaredDifference:z:0Alayer_normalization_1/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2(
&layer_normalization_1/moments/varianceУ
%layer_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_1/batchnorm/add/yл
#layer_normalization_1/batchnorm/addAddV2/layer_normalization_1/moments/variance:output:0.layer_normalization_1/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization_1/batchnorm/addЈ
%layer_normalization_1/batchnorm/RsqrtRsqrt'layer_normalization_1/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2'
%layer_normalization_1/batchnorm/Rsqrtа
2layer_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_1/batchnorm/mul/ReadVariableOpп
#layer_normalization_1/batchnorm/mulMul)layer_normalization_1/batchnorm/Rsqrt:y:0:layer_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_1/batchnorm/mul¬
%layer_normalization_1/batchnorm/mul_1Muladd/add:z:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_1/batchnorm/mul_1в
%layer_normalization_1/batchnorm/mul_2Mul+layer_normalization_1/moments/mean:output:0'layer_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_1/batchnorm/mul_2‘
.layer_normalization_1/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_1/batchnorm/ReadVariableOpл
#layer_normalization_1/batchnorm/subSub6layer_normalization_1/batchnorm/ReadVariableOp:value:0)layer_normalization_1/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_1/batchnorm/subв
%layer_normalization_1/batchnorm/add_1AddV2)layer_normalization_1/batchnorm/mul_1:z:0'layer_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
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
dense_1/Tensordot/stackћ
dense_1/Tensordot/transpose	Transpose)layer_normalization_1/batchnorm/add_1:z:0!dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
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
dense_1/Tensordot/concat_1≤
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Tensordot•
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_1/BiasAdd/ReadVariableOp©
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/BiasAddm
dense_1/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_1/Gelu/mul/xЪ
dense_1/Gelu/mulMuldense_1/Gelu/mul/x:output:0dense_1/BiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Gelu/mulo
dense_1/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_1/Gelu/Cast/xІ
dense_1/Gelu/truedivRealDivdense_1/BiasAdd:output:0dense_1/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Gelu/truediv}
dense_1/Gelu/ErfErfdense_1/Gelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Gelu/Erfm
dense_1/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_1/Gelu/add/xШ
dense_1/Gelu/addAddV2dense_1/Gelu/add/x:output:0dense_1/Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_1/Gelu/addУ
dense_1/Gelu/mul_1Muldense_1/Gelu/mul:z:0dense_1/Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
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
dense_2/Tensordot/stackЇ
dense_2/Tensordot/transpose	Transposedense_1/Gelu/mul_1:z:0!dense_2/Tensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
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
dense_2/Tensordot/concat_1±
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Tensordot§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp®
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/BiasAddm
dense_2/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_2/Gelu/mul/xЩ
dense_2/Gelu/mulMuldense_2/Gelu/mul/x:output:0dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/mulo
dense_2/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_2/Gelu/Cast/x¶
dense_2/Gelu/truedivRealDivdense_2/BiasAdd:output:0dense_2/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/truediv|
dense_2/Gelu/ErfErfdense_2/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/Erfm
dense_2/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_2/Gelu/add/xЧ
dense_2/Gelu/addAddV2dense_2/Gelu/add/x:output:0dense_2/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/addТ
dense_2/Gelu/mul_1Muldense_2/Gelu/mul:z:0dense_2/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_2/Gelu/mul_1{
	add_1/addAddV2dense_2/Gelu/mul_1:z:0add/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	add_1/addґ
4layer_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_2/moments/mean/reduction_indicesж
"layer_normalization_2/moments/meanMeanadd_1/add:z:0=layer_normalization_2/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2$
"layer_normalization_2/moments/meanћ
*layer_normalization_2/moments/StopGradientStopGradient+layer_normalization_2/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2,
*layer_normalization_2/moments/StopGradientт
/layer_normalization_2/moments/SquaredDifferenceSquaredDifferenceadd_1/add:z:03layer_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@21
/layer_normalization_2/moments/SquaredDifferenceЊ
8layer_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_2/moments/variance/reduction_indicesШ
&layer_normalization_2/moments/varianceMean3layer_normalization_2/moments/SquaredDifference:z:0Alayer_normalization_2/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2(
&layer_normalization_2/moments/varianceУ
%layer_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_2/batchnorm/add/yл
#layer_normalization_2/batchnorm/addAddV2/layer_normalization_2/moments/variance:output:0.layer_normalization_2/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization_2/batchnorm/addЈ
%layer_normalization_2/batchnorm/RsqrtRsqrt'layer_normalization_2/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2'
%layer_normalization_2/batchnorm/Rsqrtа
2layer_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_2/batchnorm/mul/ReadVariableOpп
#layer_normalization_2/batchnorm/mulMul)layer_normalization_2/batchnorm/Rsqrt:y:0:layer_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_2/batchnorm/mulƒ
%layer_normalization_2/batchnorm/mul_1Muladd_1/add:z:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_2/batchnorm/mul_1в
%layer_normalization_2/batchnorm/mul_2Mul+layer_normalization_2/moments/mean:output:0'layer_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_2/batchnorm/mul_2‘
.layer_normalization_2/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_2/batchnorm/ReadVariableOpл
#layer_normalization_2/batchnorm/subSub6layer_normalization_2/batchnorm/ReadVariableOp:value:0)layer_normalization_2/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_2/batchnorm/subв
%layer_normalization_2/batchnorm/add_1AddV2)layer_normalization_2/batchnorm/mul_1:z:0'layer_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_2/batchnorm/add_1э
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp±
*multi_head_attention_1/query/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2,
*multi_head_attention_1/query/einsum/Einsumџ
/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp8multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/query/add/ReadVariableOpц
 multi_head_attention_1/query/addAddV23multi_head_attention_1/query/einsum/Einsum:output:07multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2"
 multi_head_attention_1/query/addч
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOp@multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype029
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOpЂ
(multi_head_attention_1/key/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0?multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2*
(multi_head_attention_1/key/einsum/Einsum’
-multi_head_attention_1/key/add/ReadVariableOpReadVariableOp6multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype02/
-multi_head_attention_1/key/add/ReadVariableOpо
multi_head_attention_1/key/addAddV21multi_head_attention_1/key/einsum/Einsum:output:05multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2 
multi_head_attention_1/key/addэ
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpBmulti_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02;
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp±
*multi_head_attention_1/value/einsum/EinsumEinsum)layer_normalization_2/batchnorm/add_1:z:0Amulti_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationabc,cde->abde2,
*multi_head_attention_1/value/einsum/Einsumџ
/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp8multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype021
/multi_head_attention_1/value/add/ReadVariableOpц
 multi_head_attention_1/value/addAddV23multi_head_attention_1/value/einsum/Einsum:output:07multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2"
 multi_head_attention_1/value/addБ
multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >2
multi_head_attention_1/Mul/y«
multi_head_attention_1/MulMul$multi_head_attention_1/query/add:z:0%multi_head_attention_1/Mul/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€Р@2
multi_head_attention_1/Mulю
$multi_head_attention_1/einsum/EinsumEinsum"multi_head_attention_1/key/add:z:0multi_head_attention_1/Mul:z:0*
N*
T0*1
_output_shapes
:€€€€€€€€€РР*
equationaecd,abcd->acbe2&
$multi_head_attention_1/einsum/Einsum∆
&multi_head_attention_1/softmax/SoftmaxSoftmax-multi_head_attention_1/einsum/Einsum:output:0*
T0*1
_output_shapes
:€€€€€€€€€РР2(
&multi_head_attention_1/softmax/Softmaxћ
'multi_head_attention_1/dropout/IdentityIdentity0multi_head_attention_1/softmax/Softmax:softmax:0*
T0*1
_output_shapes
:€€€€€€€€€РР2)
'multi_head_attention_1/dropout/IdentityХ
&multi_head_attention_1/einsum_1/EinsumEinsum0multi_head_attention_1/dropout/Identity:output:0$multi_head_attention_1/value/add:z:0*
N*
T0*0
_output_shapes
:€€€€€€€€€Р@*
equationacbe,aecd->abcd2(
&multi_head_attention_1/einsum_1/EinsumЮ
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpMmulti_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype02F
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp‘
5multi_head_attention_1/attention_output/einsum/EinsumEinsum/multi_head_attention_1/einsum_1/Einsum:output:0Lmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*,
_output_shapes
:€€€€€€€€€Р@*
equationabcd,cde->abe27
5multi_head_attention_1/attention_output/einsum/Einsumш
:multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpCmulti_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype02<
:multi_head_attention_1/attention_output/add/ReadVariableOpЮ
+multi_head_attention_1/attention_output/addAddV2>multi_head_attention_1/attention_output/einsum/Einsum:output:0Bmulti_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2-
+multi_head_attention_1/attention_output/addЦ
	add_2/addAddV2/multi_head_attention_1/attention_output/add:z:0add_1/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	add_2/addґ
4layer_normalization_3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_3/moments/mean/reduction_indicesж
"layer_normalization_3/moments/meanMeanadd_2/add:z:0=layer_normalization_3/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2$
"layer_normalization_3/moments/meanћ
*layer_normalization_3/moments/StopGradientStopGradient+layer_normalization_3/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2,
*layer_normalization_3/moments/StopGradientт
/layer_normalization_3/moments/SquaredDifferenceSquaredDifferenceadd_2/add:z:03layer_normalization_3/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@21
/layer_normalization_3/moments/SquaredDifferenceЊ
8layer_normalization_3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_3/moments/variance/reduction_indicesШ
&layer_normalization_3/moments/varianceMean3layer_normalization_3/moments/SquaredDifference:z:0Alayer_normalization_3/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2(
&layer_normalization_3/moments/varianceУ
%layer_normalization_3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_3/batchnorm/add/yл
#layer_normalization_3/batchnorm/addAddV2/layer_normalization_3/moments/variance:output:0.layer_normalization_3/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization_3/batchnorm/addЈ
%layer_normalization_3/batchnorm/RsqrtRsqrt'layer_normalization_3/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2'
%layer_normalization_3/batchnorm/Rsqrtа
2layer_normalization_3/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_3/batchnorm/mul/ReadVariableOpп
#layer_normalization_3/batchnorm/mulMul)layer_normalization_3/batchnorm/Rsqrt:y:0:layer_normalization_3/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_3/batchnorm/mulƒ
%layer_normalization_3/batchnorm/mul_1Muladd_2/add:z:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_3/batchnorm/mul_1в
%layer_normalization_3/batchnorm/mul_2Mul+layer_normalization_3/moments/mean:output:0'layer_normalization_3/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_3/batchnorm/mul_2‘
.layer_normalization_3/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_3/batchnorm/ReadVariableOpл
#layer_normalization_3/batchnorm/subSub6layer_normalization_3/batchnorm/ReadVariableOp:value:0)layer_normalization_3/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_3/batchnorm/subв
%layer_normalization_3/batchnorm/add_1AddV2)layer_normalization_3/batchnorm/mul_1:z:0'layer_normalization_3/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_3/batchnorm/add_1ѓ
 dense_3/Tensordot/ReadVariableOpReadVariableOp)dense_3_tensordot_readvariableop_resource*
_output_shapes
:	@А*
dtype02"
 dense_3/Tensordot/ReadVariableOpz
dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_3/Tensordot/axesБ
dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_3/Tensordot/freeЛ
dense_3/Tensordot/ShapeShape)layer_normalization_3/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dense_3/Tensordot/ShapeД
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
dense_3/Tensordot/GatherV2И
!dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_3/Tensordot/GatherV2_1/axis€
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
dense_3/Tensordot/Const†
dense_3/Tensordot/ProdProd#dense_3/Tensordot/GatherV2:output:0 dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/ProdА
dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_3/Tensordot/Const_1®
dense_3/Tensordot/Prod_1Prod%dense_3/Tensordot/GatherV2_1:output:0"dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_3/Tensordot/Prod_1А
dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_3/Tensordot/concat/axisЎ
dense_3/Tensordot/concatConcatV2dense_3/Tensordot/free:output:0dense_3/Tensordot/axes:output:0&dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concatђ
dense_3/Tensordot/stackPackdense_3/Tensordot/Prod:output:0!dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/stackћ
dense_3/Tensordot/transpose	Transpose)layer_normalization_3/batchnorm/add_1:z:0!dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_3/Tensordot/transposeњ
dense_3/Tensordot/ReshapeReshapedense_3/Tensordot/transpose:y:0 dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_3/Tensordot/Reshapeњ
dense_3/Tensordot/MatMulMatMul"dense_3/Tensordot/Reshape:output:0(dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€А2
dense_3/Tensordot/MatMulБ
dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:А2
dense_3/Tensordot/Const_2Д
dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_3/Tensordot/concat_1/axisе
dense_3/Tensordot/concat_1ConcatV2#dense_3/Tensordot/GatherV2:output:0"dense_3/Tensordot/Const_2:output:0(dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_3/Tensordot/concat_1≤
dense_3/TensordotReshape"dense_3/Tensordot/MatMul:product:0#dense_3/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Tensordot•
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02 
dense_3/BiasAdd/ReadVariableOp©
dense_3/BiasAddBiasAdddense_3/Tensordot:output:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/BiasAddm
dense_3/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_3/Gelu/mul/xЪ
dense_3/Gelu/mulMuldense_3/Gelu/mul/x:output:0dense_3/BiasAdd:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/mulo
dense_3/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_3/Gelu/Cast/xІ
dense_3/Gelu/truedivRealDivdense_3/BiasAdd:output:0dense_3/Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/truediv}
dense_3/Gelu/ErfErfdense_3/Gelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/Erfm
dense_3/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_3/Gelu/add/xШ
dense_3/Gelu/addAddV2dense_3/Gelu/add/x:output:0dense_3/Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/addУ
dense_3/Gelu/mul_1Muldense_3/Gelu/mul:z:0dense_3/Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_3/Gelu/mul_1ѓ
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	А@*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axesБ
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
dense_4/Tensordot/ShapeД
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
dense_4/Tensordot/GatherV2И
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axis€
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
dense_4/Tensordot/Const†
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/ProdА
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1®
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1А
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axisЎ
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concatђ
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stackЇ
dense_4/Tensordot/transpose	Transposedense_3/Gelu/mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
dense_4/Tensordot/transposeњ
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:€€€€€€€€€€€€€€€€€€2
dense_4/Tensordot/ReshapeЊ
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_4/Tensordot/MatMulА
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_4/Tensordot/Const_2Д
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axisе
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1±
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Tensordot§
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_4/BiasAdd/ReadVariableOp®
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/BiasAddm
dense_4/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_4/Gelu/mul/xЩ
dense_4/Gelu/mulMuldense_4/Gelu/mul/x:output:0dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/mulo
dense_4/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_4/Gelu/Cast/x¶
dense_4/Gelu/truedivRealDivdense_4/BiasAdd:output:0dense_4/Gelu/Cast/x:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/truediv|
dense_4/Gelu/ErfErfdense_4/Gelu/truediv:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/Erfm
dense_4/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_4/Gelu/add/xЧ
dense_4/Gelu/addAddV2dense_4/Gelu/add/x:output:0dense_4/Gelu/Erf:y:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/addТ
dense_4/Gelu/mul_1Muldense_4/Gelu/mul:z:0dense_4/Gelu/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
dense_4/Gelu/mul_1}
	add_3/addAddV2dense_4/Gelu/mul_1:z:0add_2/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
	add_3/addґ
4layer_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:26
4layer_normalization_4/moments/mean/reduction_indicesж
"layer_normalization_4/moments/meanMeanadd_3/add:z:0=layer_normalization_4/moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2$
"layer_normalization_4/moments/meanћ
*layer_normalization_4/moments/StopGradientStopGradient+layer_normalization_4/moments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2,
*layer_normalization_4/moments/StopGradientт
/layer_normalization_4/moments/SquaredDifferenceSquaredDifferenceadd_3/add:z:03layer_normalization_4/moments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@21
/layer_normalization_4/moments/SquaredDifferenceЊ
8layer_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2:
8layer_normalization_4/moments/variance/reduction_indicesШ
&layer_normalization_4/moments/varianceMean3layer_normalization_4/moments/SquaredDifference:z:0Alayer_normalization_4/moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2(
&layer_normalization_4/moments/varianceУ
%layer_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52'
%layer_normalization_4/batchnorm/add/yл
#layer_normalization_4/batchnorm/addAddV2/layer_normalization_4/moments/variance:output:0.layer_normalization_4/batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2%
#layer_normalization_4/batchnorm/addЈ
%layer_normalization_4/batchnorm/RsqrtRsqrt'layer_normalization_4/batchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2'
%layer_normalization_4/batchnorm/Rsqrtа
2layer_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;layer_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2layer_normalization_4/batchnorm/mul/ReadVariableOpп
#layer_normalization_4/batchnorm/mulMul)layer_normalization_4/batchnorm/Rsqrt:y:0:layer_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_4/batchnorm/mulƒ
%layer_normalization_4/batchnorm/mul_1Muladd_3/add:z:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_4/batchnorm/mul_1в
%layer_normalization_4/batchnorm/mul_2Mul+layer_normalization_4/moments/mean:output:0'layer_normalization_4/batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_4/batchnorm/mul_2‘
.layer_normalization_4/batchnorm/ReadVariableOpReadVariableOp7layer_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.layer_normalization_4/batchnorm/ReadVariableOpл
#layer_normalization_4/batchnorm/subSub6layer_normalization_4/batchnorm/ReadVariableOp:value:0)layer_normalization_4/batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2%
#layer_normalization_4/batchnorm/subв
%layer_normalization_4/batchnorm/add_1AddV2)layer_normalization_4/batchnorm/mul_1:z:0'layer_normalization_4/batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2'
%layer_normalization_4/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€ d  2
flatten/Const§
flatten/ReshapeReshape)layer_normalization_4/batchnorm/add_1:z:0flatten/Const:output:0*
T0*)
_output_shapes
:€€€€€€€€€А»2
flatten/ReshapeІ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
А»2*
dtype02
dense_5/MatMul/ReadVariableOpЭ
dense_5/MatMulMatMulflatten/Reshape:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/MatMul§
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_5/BiasAdd/ReadVariableOp°
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/BiasAddm
dense_5/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_5/Gelu/mul/xФ
dense_5/Gelu/mulMuldense_5/Gelu/mul/x:output:0dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/mulo
dense_5/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_5/Gelu/Cast/x°
dense_5/Gelu/truedivRealDivdense_5/BiasAdd:output:0dense_5/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/truedivw
dense_5/Gelu/ErfErfdense_5/Gelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/Erfm
dense_5/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_5/Gelu/add/xТ
dense_5/Gelu/addAddV2dense_5/Gelu/add/x:output:0dense_5/Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/addН
dense_5/Gelu/mul_1Muldense_5/Gelu/mul:z:0dense_5/Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_5/Gelu/mul_1•
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:22*
dtype02
dense_6/MatMul/ReadVariableOpЫ
dense_6/MatMulMatMuldense_5/Gelu/mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/BiasAddm
dense_6/Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dense_6/Gelu/mul/xФ
dense_6/Gelu/mulMuldense_6/Gelu/mul/x:output:0dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/mulo
dense_6/Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
dense_6/Gelu/Cast/x°
dense_6/Gelu/truedivRealDivdense_6/BiasAdd:output:0dense_6/Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/truedivw
dense_6/Gelu/ErfErfdense_6/Gelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/Erfm
dense_6/Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
dense_6/Gelu/add/xТ
dense_6/Gelu/addAddV2dense_6/Gelu/add/x:output:0dense_6/Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/addН
dense_6/Gelu/mul_1Muldense_6/Gelu/mul:z:0dense_6/Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22
dense_6/Gelu/mul_1•
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02
dense_7/MatMul/ReadVariableOpЫ
dense_7/MatMulMatMuldense_6/Gelu/mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/MatMul§
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp°
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/Softmax≈
IdentityIdentitydense_7/Softmax:softmax:0^dense_1/BiasAdd/ReadVariableOp!^dense_1/Tensordot/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp!^dense_3/Tensordot/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp-^layer_normalization/batchnorm/ReadVariableOp1^layer_normalization/batchnorm/mul/ReadVariableOp/^layer_normalization_1/batchnorm/ReadVariableOp3^layer_normalization_1/batchnorm/mul/ReadVariableOp/^layer_normalization_2/batchnorm/ReadVariableOp3^layer_normalization_2/batchnorm/mul/ReadVariableOp/^layer_normalization_3/batchnorm/ReadVariableOp3^layer_normalization_3/batchnorm/mul/ReadVariableOp/^layer_normalization_4/batchnorm/ReadVariableOp3^layer_normalization_4/batchnorm/mul/ReadVariableOp9^multi_head_attention/attention_output/add/ReadVariableOpC^multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp,^multi_head_attention/key/add/ReadVariableOp6^multi_head_attention/key/einsum/Einsum/ReadVariableOp.^multi_head_attention/query/add/ReadVariableOp8^multi_head_attention/query/einsum/Einsum/ReadVariableOp.^multi_head_attention/value/add/ReadVariableOp8^multi_head_attention/value/einsum/Einsum/ReadVariableOp;^multi_head_attention_1/attention_output/add/ReadVariableOpE^multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp.^multi_head_attention_1/key/add/ReadVariableOp8^multi_head_attention_1/key/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/query/add/ReadVariableOp:^multi_head_attention_1/query/einsum/Einsum/ReadVariableOp0^multi_head_attention_1/value/add/ReadVariableOp:^multi_head_attention_1/value/einsum/Einsum/ReadVariableOp+^patch_encoder/dense/BiasAdd/ReadVariableOp-^patch_encoder/dense/Tensordot/ReadVariableOp)^patch_encoder/embedding/embedding_lookup*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/Tensordot/ReadVariableOp dense_1/Tensordot/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2D
 dense_3/Tensordot/ReadVariableOp dense_3/Tensordot/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2\
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
8multi_head_attention/attention_output/add/ReadVariableOp8multi_head_attention/attention_output/add/ReadVariableOp2И
Bmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOpBmulti_head_attention/attention_output/einsum/Einsum/ReadVariableOp2Z
+multi_head_attention/key/add/ReadVariableOp+multi_head_attention/key/add/ReadVariableOp2n
5multi_head_attention/key/einsum/Einsum/ReadVariableOp5multi_head_attention/key/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/query/add/ReadVariableOp-multi_head_attention/query/add/ReadVariableOp2r
7multi_head_attention/query/einsum/Einsum/ReadVariableOp7multi_head_attention/query/einsum/Einsum/ReadVariableOp2^
-multi_head_attention/value/add/ReadVariableOp-multi_head_attention/value/add/ReadVariableOp2r
7multi_head_attention/value/einsum/Einsum/ReadVariableOp7multi_head_attention/value/einsum/Einsum/ReadVariableOp2x
:multi_head_attention_1/attention_output/add/ReadVariableOp:multi_head_attention_1/attention_output/add/ReadVariableOp2М
Dmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpDmulti_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2^
-multi_head_attention_1/key/add/ReadVariableOp-multi_head_attention_1/key/add/ReadVariableOp2r
7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp7multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/query/add/ReadVariableOp/multi_head_attention_1/query/add/ReadVariableOp2v
9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp9multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2b
/multi_head_attention_1/value/add/ReadVariableOp/multi_head_attention_1/value/add/ReadVariableOp2v
9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp9multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2X
*patch_encoder/dense/BiasAdd/ReadVariableOp*patch_encoder/dense/BiasAdd/ReadVariableOp2\
,patch_encoder/dense/Tensordot/ReadVariableOp,patch_encoder/dense/Tensordot/ReadVariableOp2T
(patch_encoder/embedding/embedding_lookup(patch_encoder/embedding/embedding_lookup:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
™
ф
C__inference_dense_6_layer_call_and_return_conditional_losses_587959

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
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xt
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xБ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
Gelu/truediv_
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xr
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/addm

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/mul_1У
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
Ц'
ь
C__inference_dense_1_layer_call_and_return_conditional_losses_587503

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
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
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
Tensordot/concat_1Т
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2	
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
:€€€€€€€€€РА2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xЗ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
Gelu/truedive
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xx
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/adds

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/mul_1Ь
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€РА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
ю/
л
I__inference_patch_encoder_layer_call_and_return_conditional_losses_584572	
patch:
'dense_tensordot_readvariableop_resource:	ђ@3
%dense_biasadd_readvariableop_resource:@4
!embedding_embedding_lookup_584565:	Р@
identityИҐdense/BiasAdd/ReadVariableOpҐdense/Tensordot/ReadVariableOpҐembedding/embedding_lookup\
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
B :Р2
range/limit\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltav
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*
_output_shapes	
:Р2
range©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	ђ@*
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
!:€€€€€€€€€€€€€€€€€€ђ2
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
dense/BiasAddЯ
embedding/embedding_lookupResourceGather!embedding_embedding_lookup_584565range:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*4
_class*
(&loc:@embedding/embedding_lookup/584565*
_output_shapes
:	Р@*
dtype02
embedding/embedding_lookupЙ
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*4
_class*
(&loc:@embedding/embedding_lookup/584565*
_output_shapes
:	Р@2%
#embedding/embedding_lookup/Identity≤
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:	Р@2'
%embedding/embedding_lookup/Identity_1Т
addAddV2dense/BiasAdd:output:0.embedding/embedding_lookup/Identity_1:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
addљ
IdentityIdentityadd:z:0^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^embedding/embedding_lookup*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':€€€€€€€€€€€€€€€€€€ђ: : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:\ X
5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ

_user_specified_namepatch
±
Р
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_587456

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
Ў
P
$__inference_add_layer_call_fn_587434
inputs_0
inputs_1
identity“
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_add_layer_call_and_return_conditional_losses_5846672
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:€€€€€€€€€Р@:€€€€€€€€€Р@:V R
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/0:VR
,
_output_shapes
:€€€€€€€€€Р@
"
_user_specified_name
inputs/1
™
ф
C__inference_dense_6_layer_call_and_return_conditional_losses_585084

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
BiasAdd]

Gelu/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

Gelu/mul/xt
Gelu/mulMulGelu/mul/x:output:0BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xБ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*'
_output_shapes
:€€€€€€€€€22
Gelu/truediv_
Gelu/ErfErfGelu/truediv:z:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xr
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/addm

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*'
_output_shapes
:€€€€€€€€€22

Gelu/mul_1У
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
Ц'
ь
C__inference_dense_1_layer_call_and_return_conditional_losses_584735

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
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
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
Tensordot/concat_1Т
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
	TensordotН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:€€€€€€€€€РА2	
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
:€€€€€€€€€РА2

Gelu/mul_
Gelu/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *уµ?2
Gelu/Cast/xЗ
Gelu/truedivRealDivBiasAdd:output:0Gelu/Cast/x:output:0*
T0*-
_output_shapes
:€€€€€€€€€РА2
Gelu/truedive
Gelu/ErfErfGelu/truediv:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/Erf]

Gelu/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2

Gelu/add/xx
Gelu/addAddV2Gelu/add/x:output:0Gelu/Erf:y:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/adds

Gelu/mul_1MulGelu/mul:z:0Gelu/add:z:0*
T0*-
_output_shapes
:€€€€€€€€€РА2

Gelu/mul_1Ь
IdentityIdentityGelu/mul_1:z:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*-
_output_shapes
:€€€€€€€€€РА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
±
Р
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_584691

inputs3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOpК
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2 
moments/mean/reduction_indicesЭ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/meanК
moments/StopGradientStopGradientmoments/mean:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
moments/StopGradient©
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
moments/SquaredDifferenceТ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2$
"moments/variance/reduction_indicesј
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р*
	keep_dims(2
moments/varianceg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *љ7Ж52
batchnorm/add/yУ
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/addu
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЧ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_1К
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpУ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:€€€€€€€€€Р@2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€Р@: : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:€€€€€€€€€Р@
 
_user_specified_nameinputs
ч
∆

6__inference_WheatClassifier_VIT_3_layer_call_fn_587109

inputs
unknown:	ђ@
	unknown_0:@
	unknown_1:	Р@
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

unknown_14:	@А

unknown_15:	А

unknown_16:	А@

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

unknown_30:	@А

unknown_31:	А

unknown_32:	А@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:
А»2

unknown_37:2

unknown_38:22

unknown_39:2

unknown_40:2

unknown_41:
identityИҐStatefulPartitionedCallє
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
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_5851082
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameinputs
ъ
«

6__inference_WheatClassifier_VIT_3_layer_call_fn_585197
input_1
unknown:	ђ@
	unknown_0:@
	unknown_1:	Р@
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

unknown_14:	@А

unknown_15:	А

unknown_16:	А@

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

unknown_30:	@А

unknown_31:	А

unknown_32:	А@

unknown_33:@

unknown_34:@

unknown_35:@

unknown_36:
А»2

unknown_37:2

unknown_38:22

unknown_39:2

unknown_40:2

unknown_41:
identityИҐStatefulPartitionedCallЇ
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
unknown_41*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*M
_read_only_resource_inputs/
-+	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_5851082
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ж
_input_shapesu
s:€€€€€€€€€»»: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€»»
!
_user_specified_name	input_1
≥

ф
C__inference_dense_7_layer_call_and_return_conditional_losses_585101

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
я

к
5__inference_multi_head_attention_layer_call_fn_587422	
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
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallqueryvalueunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:€€€€€€€€€Р@**
_read_only_resource_inputs

	*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_5854962
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:€€€€€€€€€Р@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:€€€€€€€€€Р@:€€€€€€€€€Р@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namequery:SO
,
_output_shapes
:€€€€€€€€€Р@

_user_specified_namevalue
н
D
(__inference_patches_layer_call_fn_587219

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
!:€€€€€€€€€€€€€€€€€€ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_patches_layer_call_and_return_conditional_losses_5845302
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:€€€€€€€€€€€€€€€€€€ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€»»:Y U
1
_output_shapes
:€€€€€€€€€»»
 
_user_specified_nameimages"ћL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*і
serving_default†
E
input_1:
serving_default_input_1:0€€€€€€€€€»»;
dense_70
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Э≠
зY
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
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+Т&call_and_return_all_conditional_losses
У_default_save_signature
Ф__call__"дR
_tf_keras_network»R{"name": "WheatClassifier_VIT_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "WheatClassifier_VIT_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Patches", "config": {"layer was saved without config": true}, "name": "patches", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "PatchEncoder", "config": {"layer was saved without config": true}, "name": "patch_encoder", "inbound_nodes": [[["patches", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization", "inbound_nodes": [[["patch_encoder", 0, 0, {}]]]}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}, "name": "multi_head_attention", "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_2", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}, "name": "multi_head_attention_1", "inbound_nodes": [[["layer_normalization_2", 0, 0, {"value": ["layer_normalization_2", 0, 0]}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["multi_head_attention_1", 0, 0, {}], ["add_1", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_3", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["layer_normalization_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["dense_4", 0, 0, {}], ["add_2", 0, 0, {}]]]}, {"class_name": "LayerNormalization", "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "layer_normalization_4", "inbound_nodes": [[["add_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["layer_normalization_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 50, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "shared_object_id": 48, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 200, 200, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 200, 200, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 50}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Addons>AdamW", "config": {"name": "AdamW", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false, "weight_decay": 9.999999747378752e-05}}}}
э"ъ
_tf_keras_input_layerЏ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 200, 200, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
Ґ
trainable_variables
regularization_losses
	variables
 	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"С
_tf_keras_layerч{"name": "patches", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Patches", "config": {"layer was saved without config": true}}
’
!
projection
"position_embedding
#trainable_variables
$regularization_losses
%	variables
&	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"Ь
_tf_keras_layerВ{"name": "patch_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PatchEncoder", "config": {"layer was saved without config": true}}
„
'axis
	(gamma
)beta
*trainable_variables
+regularization_losses
,	variables
-	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"І
_tf_keras_layerН{"name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 2}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["patch_encoder", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
у

._query_dense
/
_key_dense
0_value_dense
1_softmax
2_dropout_layer
3_output_dense
4trainable_variables
5regularization_losses
6	variables
7	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"щ
_tf_keras_layerя{"name": "multi_head_attention", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}, "inbound_nodes": [[["layer_normalization", 0, 0, {"value": ["layer_normalization", 0, 0]}]]], "shared_object_id": 6}
Ю
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"Н
_tf_keras_layerу{"name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["multi_head_attention", 0, 0, {}], ["patch_encoder", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 400, 64]}, {"class_name": "TensorShape", "items": [null, 400, 64]}]}
“
<axis
	=gamma
>beta
?trainable_variables
@regularization_losses
A	variables
B	keras_api
+Я&call_and_return_all_conditional_losses
†__call__"Ґ
_tf_keras_layerИ{"name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
Т	

Ckernel
Dbias
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
+°&call_and_return_all_conditional_losses
Ґ__call__"л
_tf_keras_layer—{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["layer_normalization_1", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
Е	

Ikernel
Jbias
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+£&call_and_return_all_conditional_losses
§__call__"ё
_tf_keras_layerƒ{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 128]}}
М
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
+•&call_and_return_all_conditional_losses
¶__call__"ы
_tf_keras_layerб{"name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["dense_2", 0, 0, {}], ["add", 0, 0, {}]]], "shared_object_id": 17, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 400, 64]}, {"class_name": "TensorShape", "items": [null, 400, 64]}]}
÷
Saxis
	Tgamma
Ubeta
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+І&call_and_return_all_conditional_losses
®__call__"¶
_tf_keras_layerМ{"name": "layer_normalization_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_1", 0, 0, {}]]], "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
ю

Z_query_dense
[
_key_dense
\_value_dense
]_softmax
^_dropout_layer
__output_dense
`trainable_variables
aregularization_losses
b	variables
c	keras_api
+©&call_and_return_all_conditional_losses
™__call__"Д	
_tf_keras_layerк{"name": "multi_head_attention_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MultiHeadAttention", "config": {"name": "multi_head_attention_1", "trainable": true, "dtype": "float32", "num_heads": 4, "key_dim": 64, "value_dim": 64, "dropout": 0.1, "use_bias": true, "output_shape": null, "attention_axes": {"class_name": "__tuple__", "items": [1]}, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "query_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "key_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}, "value_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}, "inbound_nodes": [[["layer_normalization_2", 0, 0, {"value": ["layer_normalization_2", 0, 0]}]]], "shared_object_id": 23}
Э
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
+Ђ&call_and_return_all_conditional_losses
ђ__call__"М
_tf_keras_layerт{"name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["multi_head_attention_1", 0, 0, {}], ["add_1", 0, 0, {}]]], "shared_object_id": 24, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 400, 64]}, {"class_name": "TensorShape", "items": [null, 400, 64]}]}
÷
haxis
	igamma
jbeta
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
+≠&call_and_return_all_conditional_losses
Ѓ__call__"¶
_tf_keras_layerМ{"name": "layer_normalization_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_3", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_2", 0, 0, {}]]], "shared_object_id": 27, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
Т	

okernel
pbias
qtrainable_variables
rregularization_losses
s	variables
t	keras_api
+ѓ&call_and_return_all_conditional_losses
∞__call__"л
_tf_keras_layer—{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["layer_normalization_3", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 53}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
Е	

ukernel
vbias
wtrainable_variables
xregularization_losses
y	variables
z	keras_api
+±&call_and_return_all_conditional_losses
≤__call__"ё
_tf_keras_layerƒ{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_3", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 128]}}
О
{trainable_variables
|regularization_losses
}	variables
~	keras_api
+≥&call_and_return_all_conditional_losses
і__call__"э
_tf_keras_layerг{"name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["dense_4", 0, 0, {}], ["add_2", 0, 0, {}]]], "shared_object_id": 34, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 400, 64]}, {"class_name": "TensorShape", "items": [null, 400, 64]}]}
№
axis

Аgamma
	Бbeta
Вtrainable_variables
Гregularization_losses
Д	variables
Е	keras_api
+µ&call_and_return_all_conditional_losses
ґ__call__"¶
_tf_keras_layerМ{"name": "layer_normalization_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LayerNormalization", "config": {"name": "layer_normalization_4", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 1e-06, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 36}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["add_3", 0, 0, {}]]], "shared_object_id": 37, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
“
Жtrainable_variables
Зregularization_losses
И	variables
Й	keras_api
+Ј&call_and_return_all_conditional_losses
Є__call__"љ
_tf_keras_layer£{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["layer_normalization_4", 0, 0, {}]]], "shared_object_id": 38, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 55}}
К	
Кkernel
	Лbias
Мtrainable_variables
Нregularization_losses
О	variables
П	keras_api
+є&call_and_return_all_conditional_losses
Ї__call__"Ё
_tf_keras_layer√{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 50, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 25600}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 25600]}}
Д	
Рkernel
	Сbias
Тtrainable_variables
Уregularization_losses
Ф	variables
Х	keras_api
+ї&call_and_return_all_conditional_losses
Љ__call__"„
_tf_keras_layerљ{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 50, "activation": "gelu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 42}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 43}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_5", 0, 0, {}]]], "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 57}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Ж	
Цkernel
	Чbias
Шtrainable_variables
Щregularization_losses
Ъ	variables
Ы	keras_api
+љ&call_and_return_all_conditional_losses
Њ__call__"ў
_tf_keras_layerњ{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_6", 0, 0, {}]]], "shared_object_id": 47, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 58}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
э
	Ьiter
Эbeta_1
Юbeta_2

Яdecay
†learning_rate
°weight_decay(mЉ)mљ=mЊ>mњCmјDmЅIm¬Jm√TmƒUm≈im∆jm«om»pm…um vmЋ	Аmћ	БmЌ	Кmќ	Лmѕ	Рm–	Сm—	Цm“	Чm”	Ґm‘	£m’	§m÷	•m„	¶mЎ	Іmў	®mЏ	©mџ	™m№	ЂmЁ	ђmё	≠mя	Ѓmа	ѓmб	∞mв	±mг	≤mд	≥mе	іmж(vз)vи=vй>vкCvлDvмIvнJvоTvпUvрivсjvтovуpvфuvхvvц	Аvч	Бvш	Кvщ	Лvъ	Рvы	Сvь	Цvэ	Чvю	Ґv€	£vА	§vБ	•vВ	¶vГ	ІvД	®vЕ	©vЖ	™vЗ	ЂvИ	ђvЙ	≠vК	ЃvЛ	ѓvМ	∞vН	±vО	≤vП	≥vР	іvС"
	optimizer
Й
Ґ0
£1
§2
(3
)4
•5
¶6
І7
®8
©9
™10
Ђ11
ђ12
=13
>14
C15
D16
I17
J18
T19
U20
≠21
Ѓ22
ѓ23
∞24
±25
≤26
≥27
і28
i29
j30
o31
p32
u33
v34
А35
Б36
К37
Л38
Р39
С40
Ц41
Ч42"
trackable_list_wrapper
 "
trackable_list_wrapper
Й
Ґ0
£1
§2
(3
)4
•5
¶6
І7
®8
©9
™10
Ђ11
ђ12
=13
>14
C15
D16
I17
J18
T19
U20
≠21
Ѓ22
ѓ23
∞24
±25
≤26
≥27
і28
i29
j30
o31
p32
u33
v34
А35
Б36
К37
Л38
Р39
С40
Ц41
Ч42"
trackable_list_wrapper
”
trainable_variables
regularization_losses
µmetrics
 ґlayer_regularization_losses
Јnon_trainable_variables
Єlayer_metrics
	variables
єlayers
Ф__call__
У_default_save_signature
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
-
њserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
trainable_variables
regularization_losses
Їmetrics
 їlayer_regularization_losses
Љnon_trainable_variables
љlayer_metrics
	variables
Њlayers
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
ё
Ґkernel
	£bias
њtrainable_variables
јregularization_losses
Ѕ	variables
¬	keras_api
+ј&call_and_return_all_conditional_losses
Ѕ__call__"±
_tf_keras_layerЧ{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 59}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 60}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 61, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 300]}}
џ
§
embeddings
√trainable_variables
ƒregularization_losses
≈	variables
∆	keras_api
+¬&call_and_return_all_conditional_losses
√__call__"µ
_tf_keras_layerЫ{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 400, "output_dim": 64, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 63}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "shared_object_id": 64, "build_input_shape": {"class_name": "TensorShape", "items": [400]}}
8
Ґ0
£1
§2"
trackable_list_wrapper
 "
trackable_list_wrapper
8
Ґ0
£1
§2"
trackable_list_wrapper
µ
#trainable_variables
$regularization_losses
«metrics
 »layer_regularization_losses
…non_trainable_variables
 layer_metrics
%	variables
Ћlayers
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@2layer_normalization/gamma
&:$@2layer_normalization/beta
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
µ
*trainable_variables
+regularization_losses
ћmetrics
 Ќlayer_regularization_losses
ќnon_trainable_variables
ѕlayer_metrics
,	variables
–layers
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
Т
—partial_output_shape
“full_output_shape
•kernel
	¶bias
”trainable_variables
‘regularization_losses
’	variables
÷	keras_api
+ƒ&call_and_return_all_conditional_losses
≈__call__"≤
_tf_keras_layerШ{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 65, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
О
„partial_output_shape
Ўfull_output_shape
Іkernel
	®bias
ўtrainable_variables
Џregularization_losses
џ	variables
№	keras_api
+∆&call_and_return_all_conditional_losses
«__call__"Ѓ
_tf_keras_layerФ{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 66, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
Т
Ёpartial_output_shape
ёfull_output_shape
©kernel
	™bias
яtrainable_variables
аregularization_losses
б	variables
в	keras_api
+»&call_and_return_all_conditional_losses
…__call__"≤
_tf_keras_layerШ{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 67, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
Г
гtrainable_variables
дregularization_losses
е	variables
ж	keras_api
+ &call_and_return_all_conditional_losses
Ћ__call__"о
_tf_keras_layer‘{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "shared_object_id": 68}
€
зtrainable_variables
иregularization_losses
й	variables
к	keras_api
+ћ&call_and_return_all_conditional_losses
Ќ__call__"к
_tf_keras_layer–{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 69}
І
лpartial_output_shape
мfull_output_shape
Ђkernel
	ђbias
нtrainable_variables
оregularization_losses
п	variables
р	keras_api
+ќ&call_and_return_all_conditional_losses
ѕ__call__"«
_tf_keras_layer≠{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 64], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 70, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 4, 64]}}
`
•0
¶1
І2
®3
©4
™5
Ђ6
ђ7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
•0
¶1
І2
®3
©4
™5
Ђ6
ђ7"
trackable_list_wrapper
µ
4trainable_variables
5regularization_losses
сmetrics
 тlayer_regularization_losses
уnon_trainable_variables
фlayer_metrics
6	variables
хlayers
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
8trainable_variables
9regularization_losses
цmetrics
 чlayer_regularization_losses
шnon_trainable_variables
щlayer_metrics
:	variables
ъlayers
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_1/gamma
(:&@2layer_normalization_1/beta
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
µ
?trainable_variables
@regularization_losses
ыmetrics
 ьlayer_regularization_losses
эnon_trainable_variables
юlayer_metrics
A	variables
€layers
†__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
!:	@А2dense_1/kernel
:А2dense_1/bias
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
µ
Etrainable_variables
Fregularization_losses
Аmetrics
 Бlayer_regularization_losses
Вnon_trainable_variables
Гlayer_metrics
G	variables
Дlayers
Ґ__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
!:	А@2dense_2/kernel
:@2dense_2/bias
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
µ
Ktrainable_variables
Lregularization_losses
Еmetrics
 Жlayer_regularization_losses
Зnon_trainable_variables
Иlayer_metrics
M	variables
Йlayers
§__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Otrainable_variables
Pregularization_losses
Кmetrics
 Лlayer_regularization_losses
Мnon_trainable_variables
Нlayer_metrics
Q	variables
Оlayers
¶__call__
+•&call_and_return_all_conditional_losses
'•"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_2/gamma
(:&@2layer_normalization_2/beta
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
µ
Vtrainable_variables
Wregularization_losses
Пmetrics
 Рlayer_regularization_losses
Сnon_trainable_variables
Тlayer_metrics
X	variables
Уlayers
®__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
Ф
Фpartial_output_shape
Хfull_output_shape
≠kernel
	Ѓbias
Цtrainable_variables
Чregularization_losses
Ш	variables
Щ	keras_api
+–&call_and_return_all_conditional_losses
—__call__"і
_tf_keras_layerЪ{"name": "query", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "query", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 71, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
Р
Ъpartial_output_shape
Ыfull_output_shape
ѓkernel
	∞bias
Ьtrainable_variables
Эregularization_losses
Ю	variables
Я	keras_api
+“&call_and_return_all_conditional_losses
”__call__"∞
_tf_keras_layerЦ{"name": "key", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "key", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 72, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
Ф
†partial_output_shape
°full_output_shape
±kernel
	≤bias
Ґtrainable_variables
£regularization_losses
§	variables
•	keras_api
+‘&call_and_return_all_conditional_losses
’__call__"і
_tf_keras_layerЪ{"name": "value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "value", "trainable": true, "dtype": "float32", "output_shape": [null, 4, 64], "equation": "abc,cde->abde", "activation": "linear", "bias_axes": "de", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 73, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 64]}}
Г
¶trainable_variables
Іregularization_losses
®	variables
©	keras_api
+÷&call_and_return_all_conditional_losses
„__call__"о
_tf_keras_layer‘{"name": "softmax", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [3]}}, "shared_object_id": 74}
€
™trainable_variables
Ђregularization_losses
ђ	variables
≠	keras_api
+Ў&call_and_return_all_conditional_losses
ў__call__"к
_tf_keras_layer–{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "shared_object_id": 75}
©
Ѓpartial_output_shape
ѓfull_output_shape
≥kernel
	іbias
∞trainable_variables
±regularization_losses
≤	variables
≥	keras_api
+Џ&call_and_return_all_conditional_losses
џ__call__"…
_tf_keras_layerѓ{"name": "attention_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EinsumDense", "config": {"name": "attention_output", "trainable": true, "dtype": "float32", "output_shape": [null, 64], "equation": "abcd,cde->abe", "activation": "linear", "bias_axes": "e", "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 76, "build_input_shape": {"class_name": "TensorShape", "items": [null, 400, 4, 64]}}
`
≠0
Ѓ1
ѓ2
∞3
±4
≤5
≥6
і7"
trackable_list_wrapper
 "
trackable_list_wrapper
`
≠0
Ѓ1
ѓ2
∞3
±4
≤5
≥6
і7"
trackable_list_wrapper
µ
`trainable_variables
aregularization_losses
іmetrics
 µlayer_regularization_losses
ґnon_trainable_variables
Јlayer_metrics
b	variables
Єlayers
™__call__
+©&call_and_return_all_conditional_losses
'©"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
dtrainable_variables
eregularization_losses
єmetrics
 Їlayer_regularization_losses
їnon_trainable_variables
Љlayer_metrics
f	variables
љlayers
ђ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_3/gamma
(:&@2layer_normalization_3/beta
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
µ
ktrainable_variables
lregularization_losses
Њmetrics
 њlayer_regularization_losses
јnon_trainable_variables
Ѕlayer_metrics
m	variables
¬layers
Ѓ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
!:	@А2dense_3/kernel
:А2dense_3/bias
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
µ
qtrainable_variables
rregularization_losses
√metrics
 ƒlayer_regularization_losses
≈non_trainable_variables
∆layer_metrics
s	variables
«layers
∞__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
!:	А@2dense_4/kernel
:@2dense_4/bias
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
µ
wtrainable_variables
xregularization_losses
»metrics
 …layer_regularization_losses
 non_trainable_variables
Ћlayer_metrics
y	variables
ћlayers
≤__call__
+±&call_and_return_all_conditional_losses
'±"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
{trainable_variables
|regularization_losses
Ќmetrics
 ќlayer_regularization_losses
ѕnon_trainable_variables
–layer_metrics
}	variables
—layers
і__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_4/gamma
(:&@2layer_normalization_4/beta
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
“metrics
 ”layer_regularization_losses
‘non_trainable_variables
’layer_metrics
Д	variables
÷layers
ґ__call__
+µ&call_and_return_all_conditional_losses
'µ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Жtrainable_variables
Зregularization_losses
„metrics
 Ўlayer_regularization_losses
ўnon_trainable_variables
Џlayer_metrics
И	variables
џlayers
Є__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
": 
А»22dense_5/kernel
:22dense_5/bias
0
К0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
К0
Л1"
trackable_list_wrapper
Є
Мtrainable_variables
Нregularization_losses
№metrics
 Ёlayer_regularization_losses
ёnon_trainable_variables
яlayer_metrics
О	variables
аlayers
Ї__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
 :222dense_6/kernel
:22dense_6/bias
0
Р0
С1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Р0
С1"
trackable_list_wrapper
Є
Тtrainable_variables
Уregularization_losses
бmetrics
 вlayer_regularization_losses
гnon_trainable_variables
дlayer_metrics
Ф	variables
еlayers
Љ__call__
+ї&call_and_return_all_conditional_losses
'ї"call_and_return_conditional_losses"
_generic_user_object
 :22dense_7/kernel
:2dense_7/bias
0
Ц0
Ч1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ц0
Ч1"
trackable_list_wrapper
Є
Шtrainable_variables
Щregularization_losses
жmetrics
 зlayer_regularization_losses
иnon_trainable_variables
йlayer_metrics
Ъ	variables
кlayers
Њ__call__
+љ&call_and_return_all_conditional_losses
'љ"call_and_return_conditional_losses"
_generic_user_object
:	 (2
AdamW/iter
: (2AdamW/beta_1
: (2AdamW/beta_2
: (2AdamW/decay
: (2AdamW/learning_rate
: (2AdamW/weight_decay
-:+	ђ@2patch_encoder/dense/kernel
&:$@2patch_encoder/dense/bias
5:3	Р@2"patch_encoder/embedding/embeddings
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
л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
∆
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
21"
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
Ґ0
£1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ґ0
£1"
trackable_list_wrapper
Є
њtrainable_variables
јregularization_losses
нmetrics
 оlayer_regularization_losses
пnon_trainable_variables
рlayer_metrics
Ѕ	variables
сlayers
Ѕ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
(
§0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
§0"
trackable_list_wrapper
Є
√trainable_variables
ƒregularization_losses
тmetrics
 уlayer_regularization_losses
фnon_trainable_variables
хlayer_metrics
≈	variables
цlayers
√__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
!0
"1"
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
•0
¶1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
•0
¶1"
trackable_list_wrapper
Є
”trainable_variables
‘regularization_losses
чmetrics
 шlayer_regularization_losses
щnon_trainable_variables
ъlayer_metrics
’	variables
ыlayers
≈__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
І0
®1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
І0
®1"
trackable_list_wrapper
Є
ўtrainable_variables
Џregularization_losses
ьmetrics
 эlayer_regularization_losses
юnon_trainable_variables
€layer_metrics
џ	variables
Аlayers
«__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
©0
™1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
©0
™1"
trackable_list_wrapper
Є
яtrainable_variables
аregularization_losses
Бmetrics
 Вlayer_regularization_losses
Гnon_trainable_variables
Дlayer_metrics
б	variables
Еlayers
…__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
гtrainable_variables
дregularization_losses
Жmetrics
 Зlayer_regularization_losses
Иnon_trainable_variables
Йlayer_metrics
е	variables
Кlayers
Ћ__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
зtrainable_variables
иregularization_losses
Лmetrics
 Мlayer_regularization_losses
Нnon_trainable_variables
Оlayer_metrics
й	variables
Пlayers
Ќ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ђ0
ђ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ђ0
ђ1"
trackable_list_wrapper
Є
нtrainable_variables
оregularization_losses
Рmetrics
 Сlayer_regularization_losses
Тnon_trainable_variables
Уlayer_metrics
п	variables
Фlayers
ѕ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
.0
/1
02
13
24
35"
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
≠0
Ѓ1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
≠0
Ѓ1"
trackable_list_wrapper
Є
Цtrainable_variables
Чregularization_losses
Хmetrics
 Цlayer_regularization_losses
Чnon_trainable_variables
Шlayer_metrics
Ш	variables
Щlayers
—__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ѓ0
∞1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
ѓ0
∞1"
trackable_list_wrapper
Є
Ьtrainable_variables
Эregularization_losses
Ъmetrics
 Ыlayer_regularization_losses
Ьnon_trainable_variables
Эlayer_metrics
Ю	variables
Юlayers
”__call__
+“&call_and_return_all_conditional_losses
'“"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
±0
≤1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
±0
≤1"
trackable_list_wrapper
Є
Ґtrainable_variables
£regularization_losses
Яmetrics
 †layer_regularization_losses
°non_trainable_variables
Ґlayer_metrics
§	variables
£layers
’__call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
¶trainable_variables
Іregularization_losses
§metrics
 •layer_regularization_losses
¶non_trainable_variables
Іlayer_metrics
®	variables
®layers
„__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
™trainable_variables
Ђregularization_losses
©metrics
 ™layer_regularization_losses
Ђnon_trainable_variables
ђlayer_metrics
ђ	variables
≠layers
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
≥0
і1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
≥0
і1"
trackable_list_wrapper
Є
∞trainable_variables
±regularization_losses
Ѓmetrics
 ѓlayer_regularization_losses
∞non_trainable_variables
±layer_metrics
≤	variables
≤layers
џ__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
Z0
[1
\2
]3
^4
_5"
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
Ў

≥total

іcount
µ	variables
ґ	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 77}
Т

Јtotal

Єcount
є
_fn_kwargs
Ї	variables
ї	keras_api"∆
_tf_keras_metricЂ{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 50}
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
≥0
і1"
trackable_list_wrapper
.
µ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ј0
Є1"
trackable_list_wrapper
.
Ї	variables"
_generic_user_object
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
/:-@2#AdamW/layer_normalization_3/gamma/m
.:,@2"AdamW/layer_normalization_3/beta/m
':%	@А2AdamW/dense_3/kernel/m
!:А2AdamW/dense_3/bias/m
':%	А@2AdamW/dense_4/kernel/m
 :@2AdamW/dense_4/bias/m
/:-@2#AdamW/layer_normalization_4/gamma/m
.:,@2"AdamW/layer_normalization_4/beta/m
(:&
А»22AdamW/dense_5/kernel/m
 :22AdamW/dense_5/bias/m
&:$222AdamW/dense_6/kernel/m
 :22AdamW/dense_6/bias/m
&:$22AdamW/dense_7/kernel/m
 :2AdamW/dense_7/bias/m
3:1	ђ@2"AdamW/patch_encoder/dense/kernel/m
,:*@2 AdamW/patch_encoder/dense/bias/m
;:9	Р@2*AdamW/patch_encoder/embedding/embeddings/m
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
':%	@А2AdamW/dense_1/kernel/v
!:А2AdamW/dense_1/bias/v
':%	А@2AdamW/dense_2/kernel/v
 :@2AdamW/dense_2/bias/v
/:-@2#AdamW/layer_normalization_2/gamma/v
.:,@2"AdamW/layer_normalization_2/beta/v
/:-@2#AdamW/layer_normalization_3/gamma/v
.:,@2"AdamW/layer_normalization_3/beta/v
':%	@А2AdamW/dense_3/kernel/v
!:А2AdamW/dense_3/bias/v
':%	А@2AdamW/dense_4/kernel/v
 :@2AdamW/dense_4/bias/v
/:-@2#AdamW/layer_normalization_4/gamma/v
.:,@2"AdamW/layer_normalization_4/beta/v
(:&
А»22AdamW/dense_5/kernel/v
 :22AdamW/dense_5/bias/v
&:$222AdamW/dense_6/kernel/v
 :22AdamW/dense_6/bias/v
&:$22AdamW/dense_7/kernel/v
 :2AdamW/dense_7/bias/v
3:1	ђ@2"AdamW/patch_encoder/dense/kernel/v
,:*@2 AdamW/patch_encoder/dense/bias/v
;:9	Р@2*AdamW/patch_encoder/embedding/embeddings/v
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
Т2П
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_586627
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_587018
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_586038
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_586149ј
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
й2ж
!__inference__wrapped_model_584509ј
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
annotations™ *0Ґ-
+К(
input_1€€€€€€€€€»»
¶2£
6__inference_WheatClassifier_VIT_3_layer_call_fn_585197
6__inference_WheatClassifier_VIT_3_layer_call_fn_587109
6__inference_WheatClassifier_VIT_3_layer_call_fn_587200
6__inference_WheatClassifier_VIT_3_layer_call_fn_585927ј
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
C__inference_patches_layer_call_and_return_conditional_losses_587214Ґ
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
(__inference_patches_layer_call_fn_587219Ґ
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
I__inference_patch_encoder_layer_call_and_return_conditional_losses_587259°
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
.__inference_patch_encoder_layer_call_fn_587270°
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
O__inference_layer_normalization_layer_call_and_return_conditional_losses_587292Ґ
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
4__inference_layer_normalization_layer_call_fn_587301Ґ
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
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_587336
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_587378ь
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
5__inference_multi_head_attention_layer_call_fn_587400
5__inference_multi_head_attention_layer_call_fn_587422ь
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
?__inference_add_layer_call_and_return_conditional_losses_587428Ґ
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
$__inference_add_layer_call_fn_587434Ґ
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
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_587456Ґ
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
6__inference_layer_normalization_1_layer_call_fn_587465Ґ
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
C__inference_dense_1_layer_call_and_return_conditional_losses_587503Ґ
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
(__inference_dense_1_layer_call_fn_587512Ґ
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
C__inference_dense_2_layer_call_and_return_conditional_losses_587550Ґ
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
(__inference_dense_2_layer_call_fn_587559Ґ
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
A__inference_add_1_layer_call_and_return_conditional_losses_587565Ґ
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
&__inference_add_1_layer_call_fn_587571Ґ
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
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_587593Ґ
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
6__inference_layer_normalization_2_layer_call_fn_587602Ґ
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
™2І
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_587637
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_587679ь
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
ф2с
7__inference_multi_head_attention_1_layer_call_fn_587701
7__inference_multi_head_attention_1_layer_call_fn_587723ь
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
л2и
A__inference_add_2_layer_call_and_return_conditional_losses_587729Ґ
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
&__inference_add_2_layer_call_fn_587735Ґ
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
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_587757Ґ
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
6__inference_layer_normalization_3_layer_call_fn_587766Ґ
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
C__inference_dense_3_layer_call_and_return_conditional_losses_587804Ґ
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
(__inference_dense_3_layer_call_fn_587813Ґ
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
C__inference_dense_4_layer_call_and_return_conditional_losses_587851Ґ
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
(__inference_dense_4_layer_call_fn_587860Ґ
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
A__inference_add_3_layer_call_and_return_conditional_losses_587866Ґ
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
&__inference_add_3_layer_call_fn_587872Ґ
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
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_587894Ґ
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
6__inference_layer_normalization_4_layer_call_fn_587903Ґ
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
C__inference_flatten_layer_call_and_return_conditional_losses_587909Ґ
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
(__inference_flatten_layer_call_fn_587914Ґ
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
C__inference_dense_5_layer_call_and_return_conditional_losses_587932Ґ
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
(__inference_dense_5_layer_call_fn_587941Ґ
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
C__inference_dense_6_layer_call_and_return_conditional_losses_587959Ґ
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
(__inference_dense_6_layer_call_fn_587968Ґ
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
C__inference_dense_7_layer_call_and_return_conditional_losses_587979Ґ
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
(__inference_dense_7_layer_call_fn_587988Ґ
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
ЋB»
$__inference_signature_wrapper_586250input_1"Ф
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
 Й
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_586038≥FҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧBҐ?
8Ґ5
+К(
input_1€€€€€€€€€»»
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ Й
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_586149≥FҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧBҐ?
8Ґ5
+К(
input_1€€€€€€€€€»»
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ И
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_586627≤FҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧAҐ>
7Ґ4
*К'
inputs€€€€€€€€€»»
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ И
Q__inference_WheatClassifier_VIT_3_layer_call_and_return_conditional_losses_587018≤FҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧAҐ>
7Ґ4
*К'
inputs€€€€€€€€€»»
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ б
6__inference_WheatClassifier_VIT_3_layer_call_fn_585197¶FҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧBҐ?
8Ґ5
+К(
input_1€€€€€€€€€»»
p 

 
™ "К€€€€€€€€€б
6__inference_WheatClassifier_VIT_3_layer_call_fn_585927¶FҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧBҐ?
8Ґ5
+К(
input_1€€€€€€€€€»»
p

 
™ "К€€€€€€€€€а
6__inference_WheatClassifier_VIT_3_layer_call_fn_587109•FҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧAҐ>
7Ґ4
*К'
inputs€€€€€€€€€»»
p 

 
™ "К€€€€€€€€€а
6__inference_WheatClassifier_VIT_3_layer_call_fn_587200•FҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧAҐ>
7Ґ4
*К'
inputs€€€€€€€€€»»
p

 
™ "К€€€€€€€€€Ё
!__inference__wrapped_model_584509ЈFҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧ:Ґ7
0Ґ-
+К(
input_1€€€€€€€€€»»
™ "1™.
,
dense_7!К
dense_7€€€€€€€€€Ў
A__inference_add_1_layer_call_and_return_conditional_losses_587565ТdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€Р@
'К$
inputs/1€€€€€€€€€Р@
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ ∞
&__inference_add_1_layer_call_fn_587571ЕdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€Р@
'К$
inputs/1€€€€€€€€€Р@
™ "К€€€€€€€€€Р@Ў
A__inference_add_2_layer_call_and_return_conditional_losses_587729ТdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€Р@
'К$
inputs/1€€€€€€€€€Р@
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ ∞
&__inference_add_2_layer_call_fn_587735ЕdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€Р@
'К$
inputs/1€€€€€€€€€Р@
™ "К€€€€€€€€€Р@Ў
A__inference_add_3_layer_call_and_return_conditional_losses_587866ТdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€Р@
'К$
inputs/1€€€€€€€€€Р@
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ ∞
&__inference_add_3_layer_call_fn_587872ЕdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€Р@
'К$
inputs/1€€€€€€€€€Р@
™ "К€€€€€€€€€Р@÷
?__inference_add_layer_call_and_return_conditional_losses_587428ТdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€Р@
'К$
inputs/1€€€€€€€€€Р@
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ Ѓ
$__inference_add_layer_call_fn_587434ЕdҐa
ZҐW
UЪR
'К$
inputs/0€€€€€€€€€Р@
'К$
inputs/1€€€€€€€€€Р@
™ "К€€€€€€€€€Р@Ѓ
C__inference_dense_1_layer_call_and_return_conditional_losses_587503gCD4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "+Ґ(
!К
0€€€€€€€€€РА
Ъ Ж
(__inference_dense_1_layer_call_fn_587512ZCD4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "К€€€€€€€€€РАЃ
C__inference_dense_2_layer_call_and_return_conditional_losses_587550gIJ5Ґ2
+Ґ(
&К#
inputs€€€€€€€€€РА
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ Ж
(__inference_dense_2_layer_call_fn_587559ZIJ5Ґ2
+Ґ(
&К#
inputs€€€€€€€€€РА
™ "К€€€€€€€€€Р@Ѓ
C__inference_dense_3_layer_call_and_return_conditional_losses_587804gop4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "+Ґ(
!К
0€€€€€€€€€РА
Ъ Ж
(__inference_dense_3_layer_call_fn_587813Zop4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "К€€€€€€€€€РАЃ
C__inference_dense_4_layer_call_and_return_conditional_losses_587851guv5Ґ2
+Ґ(
&К#
inputs€€€€€€€€€РА
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ Ж
(__inference_dense_4_layer_call_fn_587860Zuv5Ґ2
+Ґ(
&К#
inputs€€€€€€€€€РА
™ "К€€€€€€€€€Р@І
C__inference_dense_5_layer_call_and_return_conditional_losses_587932`КЛ1Ґ.
'Ґ$
"К
inputs€€€€€€€€€А»
™ "%Ґ"
К
0€€€€€€€€€2
Ъ 
(__inference_dense_5_layer_call_fn_587941SКЛ1Ґ.
'Ґ$
"К
inputs€€€€€€€€€А»
™ "К€€€€€€€€€2•
C__inference_dense_6_layer_call_and_return_conditional_losses_587959^РС/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€2
Ъ }
(__inference_dense_6_layer_call_fn_587968QРС/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€2•
C__inference_dense_7_layer_call_and_return_conditional_losses_587979^ЦЧ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€
Ъ }
(__inference_dense_7_layer_call_fn_587988QЦЧ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€¶
C__inference_flatten_layer_call_and_return_conditional_losses_587909_4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "'Ґ$
К
0€€€€€€€€€А»
Ъ ~
(__inference_flatten_layer_call_fn_587914R4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "К€€€€€€€€€А»ї
Q__inference_layer_normalization_1_layer_call_and_return_conditional_losses_587456f=>4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ У
6__inference_layer_normalization_1_layer_call_fn_587465Y=>4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "К€€€€€€€€€Р@ї
Q__inference_layer_normalization_2_layer_call_and_return_conditional_losses_587593fTU4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ У
6__inference_layer_normalization_2_layer_call_fn_587602YTU4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "К€€€€€€€€€Р@ї
Q__inference_layer_normalization_3_layer_call_and_return_conditional_losses_587757fij4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ У
6__inference_layer_normalization_3_layer_call_fn_587766Yij4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "К€€€€€€€€€Р@љ
Q__inference_layer_normalization_4_layer_call_and_return_conditional_losses_587894hАБ4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ Х
6__inference_layer_normalization_4_layer_call_fn_587903[АБ4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "К€€€€€€€€€Р@є
O__inference_layer_normalization_layer_call_and_return_conditional_losses_587292f()4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ С
4__inference_layer_normalization_layer_call_fn_587301Y()4Ґ1
*Ґ'
%К"
inputs€€€€€€€€€Р@
™ "К€€€€€€€€€Р@А
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_587637©≠Ѓѓ∞±≤≥іiҐf
_Ґ\
$К!
query€€€€€€€€€Р@
$К!
value€€€€€€€€€Р@

 

 
p 
p 
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ А
R__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_587679©≠Ѓѓ∞±≤≥іiҐf
_Ґ\
$К!
query€€€€€€€€€Р@
$К!
value€€€€€€€€€Р@

 

 
p 
p
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ Ў
7__inference_multi_head_attention_1_layer_call_fn_587701Ь≠Ѓѓ∞±≤≥іiҐf
_Ґ\
$К!
query€€€€€€€€€Р@
$К!
value€€€€€€€€€Р@

 

 
p 
p 
™ "К€€€€€€€€€Р@Ў
7__inference_multi_head_attention_1_layer_call_fn_587723Ь≠Ѓѓ∞±≤≥іiҐf
_Ґ\
$К!
query€€€€€€€€€Р@
$К!
value€€€€€€€€€Р@

 

 
p 
p
™ "К€€€€€€€€€Р@ю
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_587336©•¶І®©™ЂђiҐf
_Ґ\
$К!
query€€€€€€€€€Р@
$К!
value€€€€€€€€€Р@

 

 
p 
p 
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ ю
P__inference_multi_head_attention_layer_call_and_return_conditional_losses_587378©•¶І®©™ЂђiҐf
_Ґ\
$К!
query€€€€€€€€€Р@
$К!
value€€€€€€€€€Р@

 

 
p 
p
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ ÷
5__inference_multi_head_attention_layer_call_fn_587400Ь•¶І®©™ЂђiҐf
_Ґ\
$К!
query€€€€€€€€€Р@
$К!
value€€€€€€€€€Р@

 

 
p 
p 
™ "К€€€€€€€€€Р@÷
5__inference_multi_head_attention_layer_call_fn_587422Ь•¶І®©™ЂђiҐf
_Ґ\
$К!
query€€€€€€€€€Р@
$К!
value€€€€€€€€€Р@

 

 
p 
p
™ "К€€€€€€€€€Р@њ
I__inference_patch_encoder_layer_call_and_return_conditional_losses_587259rҐ£§<Ґ9
2Ґ/
-К*
patch€€€€€€€€€€€€€€€€€€ђ
™ "*Ґ'
 К
0€€€€€€€€€Р@
Ъ Ч
.__inference_patch_encoder_layer_call_fn_587270eҐ£§<Ґ9
2Ґ/
-К*
patch€€€€€€€€€€€€€€€€€€ђ
™ "К€€€€€€€€€Р@Ј
C__inference_patches_layer_call_and_return_conditional_losses_587214p9Ґ6
/Ґ,
*К'
images€€€€€€€€€»»
™ "3Ґ0
)К&
0€€€€€€€€€€€€€€€€€€ђ
Ъ П
(__inference_patches_layer_call_fn_587219c9Ґ6
/Ґ,
*К'
images€€€€€€€€€»»
™ "&К#€€€€€€€€€€€€€€€€€€ђл
$__inference_signature_wrapper_586250¬FҐ£§()•¶І®©™Ђђ=>CDIJTU≠Ѓѓ∞±≤≥іijopuvАБКЛРСЦЧEҐB
Ґ 
;™8
6
input_1+К(
input_1€€€€€€€€€»»"1™.
,
dense_7!К
dense_7€€€€€€€€€