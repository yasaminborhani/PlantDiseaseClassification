Їн
€’
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
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
dtypetypeИ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-0-ga4dfb8d1a718ћб
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
~
conv_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_5/kernel
w
!conv_5/kernel/Read/ReadVariableOpReadVariableOpconv_5/kernel*&
_output_shapes
:*
dtype0
n
conv_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_5/bias
g
conv_5/bias/Read/ReadVariableOpReadVariableOpconv_5/bias*
_output_shapes
:*
dtype0
~
conv_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_6/kernel
w
!conv_6/kernel/Read/ReadVariableOpReadVariableOpconv_6/kernel*&
_output_shapes
:*
dtype0
n
conv_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv_6/bias
g
conv_6/bias/Read/ReadVariableOpReadVariableOpconv_6/bias*
_output_shapes
:*
dtype0
s
FC_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р
2*
shared_nameFC_1/kernel
l
FC_1/kernel/Read/ReadVariableOpReadVariableOpFC_1/kernel*
_output_shapes
:	Р
2*
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
О
AdamW/conv_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_3/kernel/m
З
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
О
AdamW/conv_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_4/kernel/m
З
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
О
AdamW/conv_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_5/kernel/m
З
)AdamW/conv_5/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/conv_5/kernel/m*&
_output_shapes
:*
dtype0
~
AdamW/conv_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamW/conv_5/bias/m
w
'AdamW/conv_5/bias/m/Read/ReadVariableOpReadVariableOpAdamW/conv_5/bias/m*
_output_shapes
:*
dtype0
О
AdamW/conv_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_6/kernel/m
З
)AdamW/conv_6/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/conv_6/kernel/m*&
_output_shapes
:*
dtype0
~
AdamW/conv_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamW/conv_6/bias/m
w
'AdamW/conv_6/bias/m/Read/ReadVariableOpReadVariableOpAdamW/conv_6/bias/m*
_output_shapes
:*
dtype0
Г
AdamW/FC_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р
2*$
shared_nameAdamW/FC_1/kernel/m
|
'AdamW/FC_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/m*
_output_shapes
:	Р
2*
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
О
AdamW/conv_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_3/kernel/v
З
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
О
AdamW/conv_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_4/kernel/v
З
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
О
AdamW/conv_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_5/kernel/v
З
)AdamW/conv_5/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/conv_5/kernel/v*&
_output_shapes
:*
dtype0
~
AdamW/conv_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamW/conv_5/bias/v
w
'AdamW/conv_5/bias/v/Read/ReadVariableOpReadVariableOpAdamW/conv_5/bias/v*
_output_shapes
:*
dtype0
О
AdamW/conv_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_6/kernel/v
З
)AdamW/conv_6/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/conv_6/kernel/v*&
_output_shapes
:*
dtype0
~
AdamW/conv_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdamW/conv_6/bias/v
w
'AdamW/conv_6/bias/v/Read/ReadVariableOpReadVariableOpAdamW/conv_6/bias/v*
_output_shapes
:*
dtype0
Г
AdamW/FC_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Р
2*$
shared_nameAdamW/FC_1/kernel/v
|
'AdamW/FC_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/v*
_output_shapes
:	Р
2*
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

NoOpNoOp
ґi
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*сh
valueзhBдh BЁh
±
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
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
R
#regularization_losses
$	variables
%trainable_variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
h

-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
R
3regularization_losses
4	variables
5trainable_variables
6	keras_api
h

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
R
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
R
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
h

Kkernel
Lbias
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
R
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
h

Ukernel
Vbias
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
R
[regularization_losses
\	variables
]trainable_variables
^	keras_api
h

_kernel
`bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
Ї
eiter

fbeta_1

gbeta_2
	hdecay
ilearning_rate
jweight_decaym∆m«m»m…'m (mЋ-mћ.mЌ7mќ8mѕ=m–>m—Km“Lm”Um‘Vm’_m÷`m„vЎvўvЏvџ'v№(vЁ-vё.vя7vа8vб=vв>vгKvдLvеUvжVvз_vи`vй
 
Ж
0
1
2
3
'4
(5
-6
.7
78
89
=10
>11
K12
L13
U14
V15
_16
`17
Ж
0
1
2
3
'4
(5
-6
.7
78
89
=10
>11
K12
L13
U14
V15
_16
`17
≠
regularization_losses
knon_trainable_variables
llayer_metrics
mlayer_regularization_losses
	variables

nlayers
trainable_variables
ometrics
 
YW
VARIABLE_VALUEconv_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
regularization_losses
pnon_trainable_variables
qlayer_metrics
rlayer_regularization_losses
	variables

slayers
trainable_variables
tmetrics
YW
VARIABLE_VALUEconv_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
regularization_losses
unon_trainable_variables
vlayer_metrics
wlayer_regularization_losses
 	variables

xlayers
!trainable_variables
ymetrics
 
 
 
≠
#regularization_losses
znon_trainable_variables
{layer_metrics
|layer_regularization_losses
$	variables

}layers
%trainable_variables
~metrics
YW
VARIABLE_VALUEconv_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
±
)regularization_losses
non_trainable_variables
Аlayer_metrics
 Бlayer_regularization_losses
*	variables
Вlayers
+trainable_variables
Гmetrics
YW
VARIABLE_VALUEconv_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

-0
.1

-0
.1
≤
/regularization_losses
Дnon_trainable_variables
Еlayer_metrics
 Жlayer_regularization_losses
0	variables
Зlayers
1trainable_variables
Иmetrics
 
 
 
≤
3regularization_losses
Йnon_trainable_variables
Кlayer_metrics
 Лlayer_regularization_losses
4	variables
Мlayers
5trainable_variables
Нmetrics
YW
VARIABLE_VALUEconv_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

70
81

70
81
≤
9regularization_losses
Оnon_trainable_variables
Пlayer_metrics
 Рlayer_regularization_losses
:	variables
Сlayers
;trainable_variables
Тmetrics
YW
VARIABLE_VALUEconv_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
≤
?regularization_losses
Уnon_trainable_variables
Фlayer_metrics
 Хlayer_regularization_losses
@	variables
Цlayers
Atrainable_variables
Чmetrics
 
 
 
≤
Cregularization_losses
Шnon_trainable_variables
Щlayer_metrics
 Ъlayer_regularization_losses
D	variables
Ыlayers
Etrainable_variables
Ьmetrics
 
 
 
≤
Gregularization_losses
Эnon_trainable_variables
Юlayer_metrics
 Яlayer_regularization_losses
H	variables
†layers
Itrainable_variables
°metrics
WU
VARIABLE_VALUEFC_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	FC_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

K0
L1

K0
L1
≤
Mregularization_losses
Ґnon_trainable_variables
£layer_metrics
 §layer_regularization_losses
N	variables
•layers
Otrainable_variables
¶metrics
 
 
 
≤
Qregularization_losses
Іnon_trainable_variables
®layer_metrics
 ©layer_regularization_losses
R	variables
™layers
Strainable_variables
Ђmetrics
WU
VARIABLE_VALUEFC_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	FC_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1

U0
V1
≤
Wregularization_losses
ђnon_trainable_variables
≠layer_metrics
 Ѓlayer_regularization_losses
X	variables
ѓlayers
Ytrainable_variables
∞metrics
 
 
 
≤
[regularization_losses
±non_trainable_variables
≤layer_metrics
 ≥layer_regularization_losses
\	variables
іlayers
]trainable_variables
µmetrics
_]
VARIABLE_VALUEoutput_layer/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEoutput_layer/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

_0
`1

_0
`1
≤
aregularization_losses
ґnon_trainable_variables
Јlayer_metrics
 Єlayer_regularization_losses
b	variables
єlayers
ctrainable_variables
Їmetrics
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
 
 
 
v
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

ї0
Љ1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

љtotal

Њcount
њ	variables
ј	keras_api
I

Ѕtotal

¬count
√
_fn_kwargs
ƒ	variables
≈	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

љ0
Њ1

њ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ѕ0
¬1

ƒ	variables
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
}{
VARIABLE_VALUEAdamW/conv_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_6/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_6/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/FC_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamW/FC_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/FC_2/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamW/FC_2/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEAdamW/output_layer/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/output_layer/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
}{
VARIABLE_VALUEAdamW/conv_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdamW/conv_6/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdamW/conv_6/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/FC_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamW/FC_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdamW/FC_2/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdamW/FC_2/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEAdamW/output_layer/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/output_layer/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
О
serving_default_input_layerPlaceholder*/
_output_shapes
:€€€€€€€€€dd*
dtype0*$
shape:€€€€€€€€€dd
ў
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasconv_4/kernelconv_4/biasconv_5/kernelconv_5/biasconv_6/kernelconv_6/biasFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_57182
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ш
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv_1/kernel/Read/ReadVariableOpconv_1/bias/Read/ReadVariableOp!conv_2/kernel/Read/ReadVariableOpconv_2/bias/Read/ReadVariableOp!conv_3/kernel/Read/ReadVariableOpconv_3/bias/Read/ReadVariableOp!conv_4/kernel/Read/ReadVariableOpconv_4/bias/Read/ReadVariableOp!conv_5/kernel/Read/ReadVariableOpconv_5/bias/Read/ReadVariableOp!conv_6/kernel/Read/ReadVariableOpconv_6/bias/Read/ReadVariableOpFC_1/kernel/Read/ReadVariableOpFC_1/bias/Read/ReadVariableOpFC_2/kernel/Read/ReadVariableOpFC_2/bias/Read/ReadVariableOp'output_layer/kernel/Read/ReadVariableOp%output_layer/bias/Read/ReadVariableOpAdamW/iter/Read/ReadVariableOp AdamW/beta_1/Read/ReadVariableOp AdamW/beta_2/Read/ReadVariableOpAdamW/decay/Read/ReadVariableOp'AdamW/learning_rate/Read/ReadVariableOp&AdamW/weight_decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)AdamW/conv_1/kernel/m/Read/ReadVariableOp'AdamW/conv_1/bias/m/Read/ReadVariableOp)AdamW/conv_2/kernel/m/Read/ReadVariableOp'AdamW/conv_2/bias/m/Read/ReadVariableOp)AdamW/conv_3/kernel/m/Read/ReadVariableOp'AdamW/conv_3/bias/m/Read/ReadVariableOp)AdamW/conv_4/kernel/m/Read/ReadVariableOp'AdamW/conv_4/bias/m/Read/ReadVariableOp)AdamW/conv_5/kernel/m/Read/ReadVariableOp'AdamW/conv_5/bias/m/Read/ReadVariableOp)AdamW/conv_6/kernel/m/Read/ReadVariableOp'AdamW/conv_6/bias/m/Read/ReadVariableOp'AdamW/FC_1/kernel/m/Read/ReadVariableOp%AdamW/FC_1/bias/m/Read/ReadVariableOp'AdamW/FC_2/kernel/m/Read/ReadVariableOp%AdamW/FC_2/bias/m/Read/ReadVariableOp/AdamW/output_layer/kernel/m/Read/ReadVariableOp-AdamW/output_layer/bias/m/Read/ReadVariableOp)AdamW/conv_1/kernel/v/Read/ReadVariableOp'AdamW/conv_1/bias/v/Read/ReadVariableOp)AdamW/conv_2/kernel/v/Read/ReadVariableOp'AdamW/conv_2/bias/v/Read/ReadVariableOp)AdamW/conv_3/kernel/v/Read/ReadVariableOp'AdamW/conv_3/bias/v/Read/ReadVariableOp)AdamW/conv_4/kernel/v/Read/ReadVariableOp'AdamW/conv_4/bias/v/Read/ReadVariableOp)AdamW/conv_5/kernel/v/Read/ReadVariableOp'AdamW/conv_5/bias/v/Read/ReadVariableOp)AdamW/conv_6/kernel/v/Read/ReadVariableOp'AdamW/conv_6/bias/v/Read/ReadVariableOp'AdamW/FC_1/kernel/v/Read/ReadVariableOp%AdamW/FC_1/bias/v/Read/ReadVariableOp'AdamW/FC_2/kernel/v/Read/ReadVariableOp%AdamW/FC_2/bias/v/Read/ReadVariableOp/AdamW/output_layer/kernel/v/Read/ReadVariableOp-AdamW/output_layer/bias/v/Read/ReadVariableOpConst*M
TinF
D2B	*
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
GPU 2J 8В *'
f"R 
__inference__traced_save_57814
у
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasconv_4/kernelconv_4/biasconv_5/kernelconv_5/biasconv_6/kernelconv_6/biasFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias
AdamW/iterAdamW/beta_1AdamW/beta_2AdamW/decayAdamW/learning_rateAdamW/weight_decaytotalcounttotal_1count_1AdamW/conv_1/kernel/mAdamW/conv_1/bias/mAdamW/conv_2/kernel/mAdamW/conv_2/bias/mAdamW/conv_3/kernel/mAdamW/conv_3/bias/mAdamW/conv_4/kernel/mAdamW/conv_4/bias/mAdamW/conv_5/kernel/mAdamW/conv_5/bias/mAdamW/conv_6/kernel/mAdamW/conv_6/bias/mAdamW/FC_1/kernel/mAdamW/FC_1/bias/mAdamW/FC_2/kernel/mAdamW/FC_2/bias/mAdamW/output_layer/kernel/mAdamW/output_layer/bias/mAdamW/conv_1/kernel/vAdamW/conv_1/bias/vAdamW/conv_2/kernel/vAdamW/conv_2/bias/vAdamW/conv_3/kernel/vAdamW/conv_3/bias/vAdamW/conv_4/kernel/vAdamW/conv_4/bias/vAdamW/conv_5/kernel/vAdamW/conv_5/bias/vAdamW/conv_6/kernel/vAdamW/conv_6/bias/vAdamW/FC_1/kernel/vAdamW/FC_1/bias/vAdamW/FC_2/kernel/vAdamW/FC_2/bias/vAdamW/output_layer/kernel/vAdamW/output_layer/bias/v*L
TinE
C2A*
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_58016ј‘

п
э
#__inference_signature_wrapper_57182
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
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:	Р
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityИҐStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_564822
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:€€€€€€€€€dd
%
_user_specified_nameinput_layer
Ґ
К
5__inference_WheatClassifier_CNN_2_layer_call_fn_57396

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
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:	Р
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_569412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€dd
 
_user_specified_nameinputs
љ
Ы
&__inference_conv_6_layer_call_fn_57510

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_6_layer_call_and_return_conditional_losses_566172
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
АB
л
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_56696

inputs&
conv_1_56536:

conv_1_56538:
&
conv_2_56552:


conv_2_56554:
&
conv_3_56569:

conv_3_56571:&
conv_4_56585:
conv_4_56587:&
conv_5_56602:
conv_5_56604:&
conv_6_56618:
conv_6_56620:

fc_1_56643:	Р
2

fc_1_56645:2

fc_2_56666:22

fc_2_56668:2$
output_layer_56690:2 
output_layer_56692:
identityИҐFC_1/StatefulPartitionedCallҐFC_2/StatefulPartitionedCallҐconv_1/StatefulPartitionedCallҐconv_2/StatefulPartitionedCallҐconv_3/StatefulPartitionedCallҐconv_4/StatefulPartitionedCallҐconv_5/StatefulPartitionedCallҐconv_6/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallП
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_56536conv_1_56538*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_565352 
conv_1/StatefulPartitionedCall∞
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_56552conv_2_56554*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_565512 
conv_2/StatefulPartitionedCall€
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€00
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_1_layer_call_and_return_conditional_losses_564882
maxpool_1/PartitionedCallЂ
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_56569conv_3_56571*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_565682 
conv_3/StatefulPartitionedCall∞
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_56585conv_4_56587*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_565842 
conv_4/StatefulPartitionedCall€
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_2_layer_call_and_return_conditional_losses_565002
maxpool_2/PartitionedCallЂ
conv_5/StatefulPartitionedCallStatefulPartitionedCall"maxpool_2/PartitionedCall:output:0conv_5_56602conv_5_56604*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_566012 
conv_5/StatefulPartitionedCall∞
conv_6/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0conv_6_56618conv_6_56620*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_6_layer_call_and_return_conditional_losses_566172 
conv_6/StatefulPartitionedCall€
maxpool_3/PartitionedCallPartitionedCall'conv_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_3_layer_call_and_return_conditional_losses_565122
maxpool_3/PartitionedCall€
flatten_layer/PartitionedCallPartitionedCall"maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_flatten_layer_layer_call_and_return_conditional_losses_566302
flatten_layer/PartitionedCallЭ
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0
fc_1_56643
fc_1_56645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_1_layer_call_and_return_conditional_losses_566422
FC_1/StatefulPartitionedCallю
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_566532
leaky_ReLu_1/PartitionedCallЬ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0
fc_2_56666
fc_2_56668*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_2_layer_call_and_return_conditional_losses_566652
FC_2/StatefulPartitionedCallю
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_566762
leaky_ReLu_2/PartitionedCallƒ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_56690output_layer_56692*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_566892&
$output_layer/StatefulPartitionedCallђ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€dd
 
_user_specified_nameinputs
љ
Ы
&__inference_conv_4_layer_call_fn_57472

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_565842
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€,,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€..: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€..
 
_user_specified_nameinputs
ъ
c
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_56676

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
љ
Ы
&__inference_conv_5_layer_call_fn_57491

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_566012
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ПB
р
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57076
input_layer&
conv_1_57024:

conv_1_57026:
&
conv_2_57029:


conv_2_57031:
&
conv_3_57035:

conv_3_57037:&
conv_4_57040:
conv_4_57042:&
conv_5_57046:
conv_5_57048:&
conv_6_57051:
conv_6_57053:

fc_1_57058:	Р
2

fc_1_57060:2

fc_2_57064:22

fc_2_57066:2$
output_layer_57070:2 
output_layer_57072:
identityИҐFC_1/StatefulPartitionedCallҐFC_2/StatefulPartitionedCallҐconv_1/StatefulPartitionedCallҐconv_2/StatefulPartitionedCallҐconv_3/StatefulPartitionedCallҐconv_4/StatefulPartitionedCallҐconv_5/StatefulPartitionedCallҐconv_6/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallФ
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_57024conv_1_57026*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_565352 
conv_1/StatefulPartitionedCall∞
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_57029conv_2_57031*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_565512 
conv_2/StatefulPartitionedCall€
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€00
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_1_layer_call_and_return_conditional_losses_564882
maxpool_1/PartitionedCallЂ
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_57035conv_3_57037*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_565682 
conv_3/StatefulPartitionedCall∞
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_57040conv_4_57042*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_565842 
conv_4/StatefulPartitionedCall€
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_2_layer_call_and_return_conditional_losses_565002
maxpool_2/PartitionedCallЂ
conv_5/StatefulPartitionedCallStatefulPartitionedCall"maxpool_2/PartitionedCall:output:0conv_5_57046conv_5_57048*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_566012 
conv_5/StatefulPartitionedCall∞
conv_6/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0conv_6_57051conv_6_57053*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_6_layer_call_and_return_conditional_losses_566172 
conv_6/StatefulPartitionedCall€
maxpool_3/PartitionedCallPartitionedCall'conv_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_3_layer_call_and_return_conditional_losses_565122
maxpool_3/PartitionedCall€
flatten_layer/PartitionedCallPartitionedCall"maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_flatten_layer_layer_call_and_return_conditional_losses_566302
flatten_layer/PartitionedCallЭ
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0
fc_1_57058
fc_1_57060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_1_layer_call_and_return_conditional_losses_566422
FC_1/StatefulPartitionedCallю
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_566532
leaky_ReLu_1/PartitionedCallЬ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0
fc_2_57064
fc_2_57066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_2_layer_call_and_return_conditional_losses_566652
FC_2/StatefulPartitionedCallю
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_566762
leaky_ReLu_2/PartitionedCallƒ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_57070output_layer_57072*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_566892&
$output_layer/StatefulPartitionedCallђ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:\ X
/
_output_shapes
:€€€€€€€€€dd
%
_user_specified_nameinput_layer
Ј

ш
G__inference_output_layer_layer_call_and_return_conditional_losses_56689

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
ПB
р
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57131
input_layer&
conv_1_57079:

conv_1_57081:
&
conv_2_57084:


conv_2_57086:
&
conv_3_57090:

conv_3_57092:&
conv_4_57095:
conv_4_57097:&
conv_5_57101:
conv_5_57103:&
conv_6_57106:
conv_6_57108:

fc_1_57113:	Р
2

fc_1_57115:2

fc_2_57119:22

fc_2_57121:2$
output_layer_57125:2 
output_layer_57127:
identityИҐFC_1/StatefulPartitionedCallҐFC_2/StatefulPartitionedCallҐconv_1/StatefulPartitionedCallҐconv_2/StatefulPartitionedCallҐconv_3/StatefulPartitionedCallҐconv_4/StatefulPartitionedCallҐconv_5/StatefulPartitionedCallҐconv_6/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallФ
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_57079conv_1_57081*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_565352 
conv_1/StatefulPartitionedCall∞
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_57084conv_2_57086*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_565512 
conv_2/StatefulPartitionedCall€
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€00
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_1_layer_call_and_return_conditional_losses_564882
maxpool_1/PartitionedCallЂ
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_57090conv_3_57092*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_565682 
conv_3/StatefulPartitionedCall∞
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_57095conv_4_57097*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_565842 
conv_4/StatefulPartitionedCall€
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_2_layer_call_and_return_conditional_losses_565002
maxpool_2/PartitionedCallЂ
conv_5/StatefulPartitionedCallStatefulPartitionedCall"maxpool_2/PartitionedCall:output:0conv_5_57101conv_5_57103*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_566012 
conv_5/StatefulPartitionedCall∞
conv_6/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0conv_6_57106conv_6_57108*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_6_layer_call_and_return_conditional_losses_566172 
conv_6/StatefulPartitionedCall€
maxpool_3/PartitionedCallPartitionedCall'conv_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_3_layer_call_and_return_conditional_losses_565122
maxpool_3/PartitionedCall€
flatten_layer/PartitionedCallPartitionedCall"maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_flatten_layer_layer_call_and_return_conditional_losses_566302
flatten_layer/PartitionedCallЭ
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0
fc_1_57113
fc_1_57115*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_1_layer_call_and_return_conditional_losses_566422
FC_1/StatefulPartitionedCallю
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_566532
leaky_ReLu_1/PartitionedCallЬ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0
fc_2_57119
fc_2_57121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_2_layer_call_and_return_conditional_losses_566652
FC_2/StatefulPartitionedCallю
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_566762
leaky_ReLu_2/PartitionedCallƒ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_57125output_layer_57127*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_566892&
$output_layer/StatefulPartitionedCallђ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:\ X
/
_output_shapes
:€€€€€€€€€dd
%
_user_specified_nameinput_layer
С
С
$__inference_FC_2_layer_call_fn_57569

inputs
unknown:22
	unknown_0:2
identityИҐStatefulPartitionedCallп
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
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_2_layer_call_and_return_conditional_losses_566652
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
Ѓ

ъ
A__inference_conv_4_layer_call_and_return_conditional_losses_57463

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€,,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€..: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€..
 
_user_specified_nameinputs
ъ
c
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_57545

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
±
П
5__inference_WheatClassifier_CNN_2_layer_call_fn_56735
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
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:	Р
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_566962
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:€€€€€€€€€dd
%
_user_specified_nameinput_layer
§
`
D__inference_maxpool_1_layer_call_and_return_conditional_losses_56488

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
ъ
c
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_56653

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
Ѓ

ъ
A__inference_conv_3_layer_call_and_return_conditional_losses_57444

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€..2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€00
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€00

 
_user_specified_nameinputs
Ћ	
р
?__inference_FC_2_layer_call_and_return_conditional_losses_57560

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
ъ
c
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_57574

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
к
d
H__inference_flatten_layer_layer_call_and_return_conditional_losses_57516

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€		:W S
/
_output_shapes
:€€€€€€€€€		
 
_user_specified_nameinputs
¬
H
,__inference_leaky_ReLu_1_layer_call_fn_57550

inputs
identity≈
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_566532
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
Ѓ

ъ
A__inference_conv_6_layer_call_and_return_conditional_losses_57501

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѓ

ъ
A__inference_conv_5_layer_call_and_return_conditional_losses_56601

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
В~
…
__inference__traced_save_57814
file_prefix,
(savev2_conv_1_kernel_read_readvariableop*
&savev2_conv_1_bias_read_readvariableop,
(savev2_conv_2_kernel_read_readvariableop*
&savev2_conv_2_bias_read_readvariableop,
(savev2_conv_3_kernel_read_readvariableop*
&savev2_conv_3_bias_read_readvariableop,
(savev2_conv_4_kernel_read_readvariableop*
&savev2_conv_4_bias_read_readvariableop,
(savev2_conv_5_kernel_read_readvariableop*
&savev2_conv_5_bias_read_readvariableop,
(savev2_conv_6_kernel_read_readvariableop*
&savev2_conv_6_bias_read_readvariableop*
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
-savev2_adamw_weight_decay_read_readvariableop$
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
.savev2_adamw_conv_4_bias_m_read_readvariableop4
0savev2_adamw_conv_5_kernel_m_read_readvariableop2
.savev2_adamw_conv_5_bias_m_read_readvariableop4
0savev2_adamw_conv_6_kernel_m_read_readvariableop2
.savev2_adamw_conv_6_bias_m_read_readvariableop2
.savev2_adamw_fc_1_kernel_m_read_readvariableop0
,savev2_adamw_fc_1_bias_m_read_readvariableop2
.savev2_adamw_fc_2_kernel_m_read_readvariableop0
,savev2_adamw_fc_2_bias_m_read_readvariableop:
6savev2_adamw_output_layer_kernel_m_read_readvariableop8
4savev2_adamw_output_layer_bias_m_read_readvariableop4
0savev2_adamw_conv_1_kernel_v_read_readvariableop2
.savev2_adamw_conv_1_bias_v_read_readvariableop4
0savev2_adamw_conv_2_kernel_v_read_readvariableop2
.savev2_adamw_conv_2_bias_v_read_readvariableop4
0savev2_adamw_conv_3_kernel_v_read_readvariableop2
.savev2_adamw_conv_3_bias_v_read_readvariableop4
0savev2_adamw_conv_4_kernel_v_read_readvariableop2
.savev2_adamw_conv_4_bias_v_read_readvariableop4
0savev2_adamw_conv_5_kernel_v_read_readvariableop2
.savev2_adamw_conv_5_bias_v_read_readvariableop4
0savev2_adamw_conv_6_kernel_v_read_readvariableop2
.savev2_adamw_conv_6_bias_v_read_readvariableop2
.savev2_adamw_fc_1_kernel_v_read_readvariableop0
,savev2_adamw_fc_1_bias_v_read_readvariableop2
.savev2_adamw_fc_2_kernel_v_read_readvariableop0
,savev2_adamw_fc_2_bias_v_read_readvariableop:
6savev2_adamw_output_layer_kernel_v_read_readvariableop8
4savev2_adamw_output_layer_bias_v_read_readvariableop
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
ShardedFilenameЫ$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*≠#
value£#B†#AB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesН
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*Ч
valueНBКAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЎ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop(savev2_conv_5_kernel_read_readvariableop&savev2_conv_5_bias_read_readvariableop(savev2_conv_6_kernel_read_readvariableop&savev2_conv_6_bias_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_2_kernel_read_readvariableop$savev2_fc_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop%savev2_adamw_iter_read_readvariableop'savev2_adamw_beta_1_read_readvariableop'savev2_adamw_beta_2_read_readvariableop&savev2_adamw_decay_read_readvariableop.savev2_adamw_learning_rate_read_readvariableop-savev2_adamw_weight_decay_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adamw_conv_1_kernel_m_read_readvariableop.savev2_adamw_conv_1_bias_m_read_readvariableop0savev2_adamw_conv_2_kernel_m_read_readvariableop.savev2_adamw_conv_2_bias_m_read_readvariableop0savev2_adamw_conv_3_kernel_m_read_readvariableop.savev2_adamw_conv_3_bias_m_read_readvariableop0savev2_adamw_conv_4_kernel_m_read_readvariableop.savev2_adamw_conv_4_bias_m_read_readvariableop0savev2_adamw_conv_5_kernel_m_read_readvariableop.savev2_adamw_conv_5_bias_m_read_readvariableop0savev2_adamw_conv_6_kernel_m_read_readvariableop.savev2_adamw_conv_6_bias_m_read_readvariableop.savev2_adamw_fc_1_kernel_m_read_readvariableop,savev2_adamw_fc_1_bias_m_read_readvariableop.savev2_adamw_fc_2_kernel_m_read_readvariableop,savev2_adamw_fc_2_bias_m_read_readvariableop6savev2_adamw_output_layer_kernel_m_read_readvariableop4savev2_adamw_output_layer_bias_m_read_readvariableop0savev2_adamw_conv_1_kernel_v_read_readvariableop.savev2_adamw_conv_1_bias_v_read_readvariableop0savev2_adamw_conv_2_kernel_v_read_readvariableop.savev2_adamw_conv_2_bias_v_read_readvariableop0savev2_adamw_conv_3_kernel_v_read_readvariableop.savev2_adamw_conv_3_bias_v_read_readvariableop0savev2_adamw_conv_4_kernel_v_read_readvariableop.savev2_adamw_conv_4_bias_v_read_readvariableop0savev2_adamw_conv_5_kernel_v_read_readvariableop.savev2_adamw_conv_5_bias_v_read_readvariableop0savev2_adamw_conv_6_kernel_v_read_readvariableop.savev2_adamw_conv_6_bias_v_read_readvariableop.savev2_adamw_fc_1_kernel_v_read_readvariableop,savev2_adamw_fc_1_bias_v_read_readvariableop.savev2_adamw_fc_2_kernel_v_read_readvariableop,savev2_adamw_fc_2_bias_v_read_readvariableop6savev2_adamw_output_layer_kernel_v_read_readvariableop4savev2_adamw_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *O
dtypesE
C2A	2
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

identity_1Identity_1:output:0*р
_input_shapesё
џ: :
:
:

:
:
::::::::	Р
2:2:22:2:2:: : : : : : : : : : :
:
:

:
:
::::::::	Р
2:2:22:2:2::
:
:

:
:
::::::::	Р
2:2:22:2:2:: 2(
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
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	Р
2: 

_output_shapes
:2:$ 

_output_shapes

:22: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:
: 

_output_shapes
:
:,(
&
_output_shapes
:

:  

_output_shapes
:
:,!(
&
_output_shapes
:
: "

_output_shapes
::,#(
&
_output_shapes
:: $

_output_shapes
::,%(
&
_output_shapes
:: &

_output_shapes
::,'(
&
_output_shapes
:: (

_output_shapes
::%)!

_output_shapes
:	Р
2: *

_output_shapes
:2:$+ 

_output_shapes

:22: ,

_output_shapes
:2:$- 

_output_shapes

:2: .

_output_shapes
::,/(
&
_output_shapes
:
: 0

_output_shapes
:
:,1(
&
_output_shapes
:

: 2

_output_shapes
:
:,3(
&
_output_shapes
:
: 4

_output_shapes
::,5(
&
_output_shapes
:: 6

_output_shapes
::,7(
&
_output_shapes
:: 8

_output_shapes
::,9(
&
_output_shapes
:: :

_output_shapes
::%;!

_output_shapes
:	Р
2: <

_output_shapes
:2:$= 

_output_shapes

:22: >

_output_shapes
:2:$? 

_output_shapes

:2: @

_output_shapes
::A

_output_shapes
: 
§
`
D__inference_maxpool_3_layer_call_and_return_conditional_losses_56512

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
љ
Ы
&__inference_conv_3_layer_call_fn_57453

inputs!
unknown:

	unknown_0:
identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_565682
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€..2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€00
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€00

 
_user_specified_nameinputs
÷
I
-__inference_flatten_layer_layer_call_fn_57521

inputs
identity«
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_flatten_layer_layer_call_and_return_conditional_losses_566302
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€		:W S
/
_output_shapes
:€€€€€€€€€		
 
_user_specified_nameinputs
¬Б
≥
 __inference__wrapped_model_56482
input_layerU
;wheatclassifier_cnn_2_conv_1_conv2d_readvariableop_resource:
J
<wheatclassifier_cnn_2_conv_1_biasadd_readvariableop_resource:
U
;wheatclassifier_cnn_2_conv_2_conv2d_readvariableop_resource:

J
<wheatclassifier_cnn_2_conv_2_biasadd_readvariableop_resource:
U
;wheatclassifier_cnn_2_conv_3_conv2d_readvariableop_resource:
J
<wheatclassifier_cnn_2_conv_3_biasadd_readvariableop_resource:U
;wheatclassifier_cnn_2_conv_4_conv2d_readvariableop_resource:J
<wheatclassifier_cnn_2_conv_4_biasadd_readvariableop_resource:U
;wheatclassifier_cnn_2_conv_5_conv2d_readvariableop_resource:J
<wheatclassifier_cnn_2_conv_5_biasadd_readvariableop_resource:U
;wheatclassifier_cnn_2_conv_6_conv2d_readvariableop_resource:J
<wheatclassifier_cnn_2_conv_6_biasadd_readvariableop_resource:L
9wheatclassifier_cnn_2_fc_1_matmul_readvariableop_resource:	Р
2H
:wheatclassifier_cnn_2_fc_1_biasadd_readvariableop_resource:2K
9wheatclassifier_cnn_2_fc_2_matmul_readvariableop_resource:22H
:wheatclassifier_cnn_2_fc_2_biasadd_readvariableop_resource:2S
Awheatclassifier_cnn_2_output_layer_matmul_readvariableop_resource:2P
Bwheatclassifier_cnn_2_output_layer_biasadd_readvariableop_resource:
identityИҐ1WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOpҐ0WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOpҐ1WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOpҐ0WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOpҐ3WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOpҐ2WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOpҐ3WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOpҐ2WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOpҐ3WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOpҐ2WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOpҐ3WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOpҐ2WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOpҐ3WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOpҐ2WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOpҐ3WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOpҐ2WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOpҐ9WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOpҐ8WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOpм
2WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype024
2WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOpА
#WheatClassifier_CNN_2/conv_1/Conv2DConv2Dinput_layer:WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_1/Conv2Dг
3WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOpь
$WheatClassifier_CNN_2/conv_1/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_1/Conv2D:output:0;WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
2&
$WheatClassifier_CNN_2/conv_1/BiasAddм
2WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype024
2WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOpҐ
#WheatClassifier_CNN_2/conv_2/Conv2DConv2D-WheatClassifier_CNN_2/conv_1/BiasAdd:output:0:WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_2/Conv2Dг
3WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOpь
$WheatClassifier_CNN_2/conv_2/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_2/Conv2D:output:0;WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
2&
$WheatClassifier_CNN_2/conv_2/BiasAddщ
'WheatClassifier_CNN_2/maxpool_1/MaxPoolMaxPool-WheatClassifier_CNN_2/conv_2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€00
*
ksize
*
paddingVALID*
strides
2)
'WheatClassifier_CNN_2/maxpool_1/MaxPoolм
2WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype024
2WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOp•
#WheatClassifier_CNN_2/conv_3/Conv2DConv2D0WheatClassifier_CNN_2/maxpool_1/MaxPool:output:0:WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_3/Conv2Dг
3WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOpь
$WheatClassifier_CNN_2/conv_3/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_3/Conv2D:output:0;WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..2&
$WheatClassifier_CNN_2/conv_3/BiasAddм
2WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOpҐ
#WheatClassifier_CNN_2/conv_4/Conv2DConv2D-WheatClassifier_CNN_2/conv_3/BiasAdd:output:0:WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_4/Conv2Dг
3WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOpь
$WheatClassifier_CNN_2/conv_4/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_4/Conv2D:output:0;WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,2&
$WheatClassifier_CNN_2/conv_4/BiasAddщ
'WheatClassifier_CNN_2/maxpool_2/MaxPoolMaxPool-WheatClassifier_CNN_2/conv_4/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2)
'WheatClassifier_CNN_2/maxpool_2/MaxPoolм
2WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOp•
#WheatClassifier_CNN_2/conv_5/Conv2DConv2D0WheatClassifier_CNN_2/maxpool_2/MaxPool:output:0:WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_5/Conv2Dг
3WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOpь
$WheatClassifier_CNN_2/conv_5/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_5/Conv2D:output:0;WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2&
$WheatClassifier_CNN_2/conv_5/BiasAddм
2WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOpҐ
#WheatClassifier_CNN_2/conv_6/Conv2DConv2D-WheatClassifier_CNN_2/conv_5/BiasAdd:output:0:WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_6/Conv2Dг
3WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOpь
$WheatClassifier_CNN_2/conv_6/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_6/Conv2D:output:0;WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2&
$WheatClassifier_CNN_2/conv_6/BiasAddщ
'WheatClassifier_CNN_2/maxpool_3/MaxPoolMaxPool-WheatClassifier_CNN_2/conv_6/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€		*
ksize
*
paddingVALID*
strides
2)
'WheatClassifier_CNN_2/maxpool_3/MaxPoolІ
)WheatClassifier_CNN_2/flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2+
)WheatClassifier_CNN_2/flatten_layer/Constю
+WheatClassifier_CNN_2/flatten_layer/ReshapeReshape0WheatClassifier_CNN_2/maxpool_3/MaxPool:output:02WheatClassifier_CNN_2/flatten_layer/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р
2-
+WheatClassifier_CNN_2/flatten_layer/Reshapeя
0WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_2_fc_1_matmul_readvariableop_resource*
_output_shapes
:	Р
2*
dtype022
0WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOpт
!WheatClassifier_CNN_2/FC_1/MatMulMatMul4WheatClassifier_CNN_2/flatten_layer/Reshape:output:08WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22#
!WheatClassifier_CNN_2/FC_1/MatMulЁ
1WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_2_fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOpн
"WheatClassifier_CNN_2/FC_1/BiasAddBiasAdd+WheatClassifier_CNN_2/FC_1/MatMul:product:09WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22$
"WheatClassifier_CNN_2/FC_1/BiasAddѕ
,WheatClassifier_CNN_2/leaky_ReLu_1/LeakyRelu	LeakyRelu+WheatClassifier_CNN_2/FC_1/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2.
,WheatClassifier_CNN_2/leaky_ReLu_1/LeakyReluё
0WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_2_fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype022
0WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOpш
!WheatClassifier_CNN_2/FC_2/MatMulMatMul:WheatClassifier_CNN_2/leaky_ReLu_1/LeakyRelu:activations:08WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22#
!WheatClassifier_CNN_2/FC_2/MatMulЁ
1WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_2_fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOpн
"WheatClassifier_CNN_2/FC_2/BiasAddBiasAdd+WheatClassifier_CNN_2/FC_2/MatMul:product:09WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€22$
"WheatClassifier_CNN_2/FC_2/BiasAddѕ
,WheatClassifier_CNN_2/leaky_ReLu_2/LeakyRelu	LeakyRelu+WheatClassifier_CNN_2/FC_2/BiasAdd:output:0*'
_output_shapes
:€€€€€€€€€2*
alpha%ЪЩЩ>2.
,WheatClassifier_CNN_2/leaky_ReLu_2/LeakyReluц
8WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOpReadVariableOpAwheatclassifier_cnn_2_output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02:
8WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOpР
)WheatClassifier_CNN_2/output_layer/MatMulMatMul:WheatClassifier_CNN_2/leaky_ReLu_2/LeakyRelu:activations:0@WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2+
)WheatClassifier_CNN_2/output_layer/MatMulх
9WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOpReadVariableOpBwheatclassifier_cnn_2_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOpН
*WheatClassifier_CNN_2/output_layer/BiasAddBiasAdd3WheatClassifier_CNN_2/output_layer/MatMul:product:0AWheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2,
*WheatClassifier_CNN_2/output_layer/BiasAdd 
*WheatClassifier_CNN_2/output_layer/SoftmaxSoftmax3WheatClassifier_CNN_2/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2,
*WheatClassifier_CNN_2/output_layer/Softmaxѕ
IdentityIdentity4WheatClassifier_CNN_2/output_layer/Softmax:softmax:02^WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOp2^WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOp4^WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOp:^WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOp9^WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 2f
1WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOp1WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOp2d
0WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOp0WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOp2f
1WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOp1WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOp2d
0WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOp0WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOp2j
3WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOp3WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOp2WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOp3WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOp2WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOp3WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOp2WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOp3WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOp2WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOp3WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOp2WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOp2j
3WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOp3WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOp2h
2WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOp2WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOp2v
9WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOp9WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOp2t
8WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOp8WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOp:\ X
/
_output_shapes
:€€€€€€€€€dd
%
_user_specified_nameinput_layer
±
П
5__inference_WheatClassifier_CNN_2_layer_call_fn_57021
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
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:	Р
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinput_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_569412
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:€€€€€€€€€dd
%
_user_specified_nameinput_layer
…
E
)__inference_maxpool_3_layer_call_fn_56518

inputs
identityе
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
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_3_layer_call_and_return_conditional_losses_565122
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
к
d
H__inference_flatten_layer_layer_call_and_return_conditional_losses_56630

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€		:W S
/
_output_shapes
:€€€€€€€€€		
 
_user_specified_nameinputs
Ј

ш
G__inference_output_layer_layer_call_and_return_conditional_losses_57590

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
Ѓ

ъ
A__inference_conv_1_layer_call_and_return_conditional_losses_57406

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
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
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
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€bb
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€dd
 
_user_specified_nameinputs
љ
Ы
&__inference_conv_1_layer_call_fn_57415

inputs!
unknown:

	unknown_0:

identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_565352
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€bb
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€dd
 
_user_specified_nameinputs
Ѓ

ъ
A__inference_conv_1_layer_call_and_return_conditional_losses_56535

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
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
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
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€bb
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€dd
 
_user_specified_nameinputs
¬X
∆
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57248

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
&conv_4_biasadd_readvariableop_resource:?
%conv_5_conv2d_readvariableop_resource:4
&conv_5_biasadd_readvariableop_resource:?
%conv_6_conv2d_readvariableop_resource:4
&conv_6_biasadd_readvariableop_resource:6
#fc_1_matmul_readvariableop_resource:	Р
22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identityИҐFC_1/BiasAdd/ReadVariableOpҐFC_1/MatMul/ReadVariableOpҐFC_2/BiasAdd/ReadVariableOpҐFC_2/MatMul/ReadVariableOpҐconv_1/BiasAdd/ReadVariableOpҐconv_1/Conv2D/ReadVariableOpҐconv_2/BiasAdd/ReadVariableOpҐconv_2/Conv2D/ReadVariableOpҐconv_3/BiasAdd/ReadVariableOpҐconv_3/Conv2D/ReadVariableOpҐconv_4/BiasAdd/ReadVariableOpҐconv_4/Conv2D/ReadVariableOpҐconv_5/BiasAdd/ReadVariableOpҐconv_5/Conv2D/ReadVariableOpҐconv_6/BiasAdd/ReadVariableOpҐconv_6/Conv2D/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOp™
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOpє
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
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
conv_1/BiasAdd/ReadVariableOp§
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
2
conv_1/BiasAdd™
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
conv_2/Conv2D/ReadVariableOp 
conv_2/Conv2DConv2Dconv_1/BiasAdd:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
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
conv_2/BiasAdd/ReadVariableOp§
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
2
conv_2/BiasAddЈ
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€00
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPool™
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_3/Conv2D/ReadVariableOpЌ
conv_3/Conv2DConv2Dmaxpool_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..*
paddingVALID*
strides
2
conv_3/Conv2D°
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOp§
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..2
conv_3/BiasAdd™
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOp 
conv_4/Conv2DConv2Dconv_3/BiasAdd:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,*
paddingVALID*
strides
2
conv_4/Conv2D°
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_4/BiasAdd/ReadVariableOp§
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,2
conv_4/BiasAddЈ
maxpool_2/MaxPoolMaxPoolconv_4/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
maxpool_2/MaxPool™
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_5/Conv2D/ReadVariableOpЌ
conv_5/Conv2DConv2Dmaxpool_2/MaxPool:output:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv_5/Conv2D°
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_5/BiasAdd/ReadVariableOp§
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv_5/BiasAdd™
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_6/Conv2D/ReadVariableOp 
conv_6/Conv2DConv2Dconv_5/BiasAdd:output:0$conv_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv_6/Conv2D°
conv_6/BiasAdd/ReadVariableOpReadVariableOp&conv_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_6/BiasAdd/ReadVariableOp§
conv_6/BiasAddBiasAddconv_6/Conv2D:output:0%conv_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv_6/BiasAddЈ
maxpool_3/MaxPoolMaxPoolconv_6/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€		*
ksize
*
paddingVALID*
strides
2
maxpool_3/MaxPool{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
flatten_layer/Const¶
flatten_layer/ReshapeReshapemaxpool_3/MaxPool:output:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р
2
flatten_layer/ReshapeЭ
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	Р
2*
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
output_layer/Softmax≠
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^conv_5/BiasAdd/ReadVariableOp^conv_5/Conv2D/ReadVariableOp^conv_6/BiasAdd/ReadVariableOp^conv_6/Conv2D/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 2:
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
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2>
conv_5/BiasAdd/ReadVariableOpconv_5/BiasAdd/ReadVariableOp2<
conv_5/Conv2D/ReadVariableOpconv_5/Conv2D/ReadVariableOp2>
conv_6/BiasAdd/ReadVariableOpconv_6/BiasAdd/ReadVariableOp2<
conv_6/Conv2D/ReadVariableOpconv_6/Conv2D/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€dd
 
_user_specified_nameinputs
јР
•'
!__inference__traced_restore_58016
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
 assignvariableop_8_conv_5_kernel:,
assignvariableop_9_conv_5_bias:;
!assignvariableop_10_conv_6_kernel:-
assignvariableop_11_conv_6_bias:2
assignvariableop_12_fc_1_kernel:	Р
2+
assignvariableop_13_fc_1_bias:21
assignvariableop_14_fc_2_kernel:22+
assignvariableop_15_fc_2_bias:29
'assignvariableop_16_output_layer_kernel:23
%assignvariableop_17_output_layer_bias:(
assignvariableop_18_adamw_iter:	 *
 assignvariableop_19_adamw_beta_1: *
 assignvariableop_20_adamw_beta_2: )
assignvariableop_21_adamw_decay: 1
'assignvariableop_22_adamw_learning_rate: 0
&assignvariableop_23_adamw_weight_decay: #
assignvariableop_24_total: #
assignvariableop_25_count: %
assignvariableop_26_total_1: %
assignvariableop_27_count_1: C
)assignvariableop_28_adamw_conv_1_kernel_m:
5
'assignvariableop_29_adamw_conv_1_bias_m:
C
)assignvariableop_30_adamw_conv_2_kernel_m:

5
'assignvariableop_31_adamw_conv_2_bias_m:
C
)assignvariableop_32_adamw_conv_3_kernel_m:
5
'assignvariableop_33_adamw_conv_3_bias_m:C
)assignvariableop_34_adamw_conv_4_kernel_m:5
'assignvariableop_35_adamw_conv_4_bias_m:C
)assignvariableop_36_adamw_conv_5_kernel_m:5
'assignvariableop_37_adamw_conv_5_bias_m:C
)assignvariableop_38_adamw_conv_6_kernel_m:5
'assignvariableop_39_adamw_conv_6_bias_m::
'assignvariableop_40_adamw_fc_1_kernel_m:	Р
23
%assignvariableop_41_adamw_fc_1_bias_m:29
'assignvariableop_42_adamw_fc_2_kernel_m:223
%assignvariableop_43_adamw_fc_2_bias_m:2A
/assignvariableop_44_adamw_output_layer_kernel_m:2;
-assignvariableop_45_adamw_output_layer_bias_m:C
)assignvariableop_46_adamw_conv_1_kernel_v:
5
'assignvariableop_47_adamw_conv_1_bias_v:
C
)assignvariableop_48_adamw_conv_2_kernel_v:

5
'assignvariableop_49_adamw_conv_2_bias_v:
C
)assignvariableop_50_adamw_conv_3_kernel_v:
5
'assignvariableop_51_adamw_conv_3_bias_v:C
)assignvariableop_52_adamw_conv_4_kernel_v:5
'assignvariableop_53_adamw_conv_4_bias_v:C
)assignvariableop_54_adamw_conv_5_kernel_v:5
'assignvariableop_55_adamw_conv_5_bias_v:C
)assignvariableop_56_adamw_conv_6_kernel_v:5
'assignvariableop_57_adamw_conv_6_bias_v::
'assignvariableop_58_adamw_fc_1_kernel_v:	Р
23
%assignvariableop_59_adamw_fc_1_bias_v:29
'assignvariableop_60_adamw_fc_2_kernel_v:223
%assignvariableop_61_adamw_fc_2_bias_v:2A
/assignvariableop_62_adamw_output_layer_kernel_v:2;
-assignvariableop_63_adamw_output_layer_bias_v:
identity_65ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9°$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*≠#
value£#B†#AB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesУ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*Ч
valueНBКAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesу
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ъ
_output_shapesЗ
Д:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A	2
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

Identity_4•
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6•
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8•
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9£
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10©
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11І
AssignVariableOp_11AssignVariableOpassignvariableop_11_conv_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12І
AssignVariableOp_12AssignVariableOpassignvariableop_12_fc_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13•
AssignVariableOp_13AssignVariableOpassignvariableop_13_fc_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14І
AssignVariableOp_14AssignVariableOpassignvariableop_14_fc_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15•
AssignVariableOp_15AssignVariableOpassignvariableop_15_fc_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ѓ
AssignVariableOp_16AssignVariableOp'assignvariableop_16_output_layer_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17≠
AssignVariableOp_17AssignVariableOp%assignvariableop_17_output_layer_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18¶
AssignVariableOp_18AssignVariableOpassignvariableop_18_adamw_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19®
AssignVariableOp_19AssignVariableOp assignvariableop_19_adamw_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp assignvariableop_20_adamw_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21І
AssignVariableOp_21AssignVariableOpassignvariableop_21_adamw_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ѓ
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adamw_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ѓ
AssignVariableOp_23AssignVariableOp&assignvariableop_23_adamw_weight_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24°
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25°
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27£
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adamw_conv_1_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29ѓ
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adamw_conv_1_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adamw_conv_2_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31ѓ
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adamw_conv_2_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adamw_conv_3_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33ѓ
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adamw_conv_3_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adamw_conv_4_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35ѓ
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adamw_conv_4_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36±
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adamw_conv_5_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37ѓ
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adamw_conv_5_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adamw_conv_6_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39ѓ
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adamw_conv_6_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ѓ
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adamw_fc_1_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41≠
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adamw_fc_1_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ѓ
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adamw_fc_2_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43≠
AssignVariableOp_43AssignVariableOp%assignvariableop_43_adamw_fc_2_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ј
AssignVariableOp_44AssignVariableOp/assignvariableop_44_adamw_output_layer_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45µ
AssignVariableOp_45AssignVariableOp-assignvariableop_45_adamw_output_layer_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46±
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adamw_conv_1_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47ѓ
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adamw_conv_1_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adamw_conv_2_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49ѓ
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adamw_conv_2_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adamw_conv_3_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51ѓ
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adamw_conv_3_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52±
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adamw_conv_4_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53ѓ
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adamw_conv_4_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54±
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adamw_conv_5_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55ѓ
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adamw_conv_5_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56±
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adamw_conv_6_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57ѓ
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adamw_conv_6_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58ѓ
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adamw_fc_1_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59≠
AssignVariableOp_59AssignVariableOp%assignvariableop_59_adamw_fc_1_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60ѓ
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adamw_fc_2_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61≠
AssignVariableOp_61AssignVariableOp%assignvariableop_61_adamw_fc_2_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Ј
AssignVariableOp_62AssignVariableOp/assignvariableop_62_adamw_output_layer_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63µ
AssignVariableOp_63AssignVariableOp-assignvariableop_63_adamw_output_layer_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_639
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpё
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_64—
Identity_65IdentityIdentity_64:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_65"#
identity_65Identity_65:output:0*Ч
_input_shapesЕ
В: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_63AssignVariableOp_632(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¬X
∆
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57314

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
&conv_4_biasadd_readvariableop_resource:?
%conv_5_conv2d_readvariableop_resource:4
&conv_5_biasadd_readvariableop_resource:?
%conv_6_conv2d_readvariableop_resource:4
&conv_6_biasadd_readvariableop_resource:6
#fc_1_matmul_readvariableop_resource:	Р
22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identityИҐFC_1/BiasAdd/ReadVariableOpҐFC_1/MatMul/ReadVariableOpҐFC_2/BiasAdd/ReadVariableOpҐFC_2/MatMul/ReadVariableOpҐconv_1/BiasAdd/ReadVariableOpҐconv_1/Conv2D/ReadVariableOpҐconv_2/BiasAdd/ReadVariableOpҐconv_2/Conv2D/ReadVariableOpҐconv_3/BiasAdd/ReadVariableOpҐconv_3/Conv2D/ReadVariableOpҐconv_4/BiasAdd/ReadVariableOpҐconv_4/Conv2D/ReadVariableOpҐconv_5/BiasAdd/ReadVariableOpҐconv_5/Conv2D/ReadVariableOpҐconv_6/BiasAdd/ReadVariableOpҐconv_6/Conv2D/ReadVariableOpҐ#output_layer/BiasAdd/ReadVariableOpҐ"output_layer/MatMul/ReadVariableOp™
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOpє
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
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
conv_1/BiasAdd/ReadVariableOp§
conv_1/BiasAddBiasAddconv_1/Conv2D:output:0%conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€bb
2
conv_1/BiasAdd™
conv_2/Conv2D/ReadVariableOpReadVariableOp%conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype02
conv_2/Conv2D/ReadVariableOp 
conv_2/Conv2DConv2Dconv_1/BiasAdd:output:0$conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
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
conv_2/BiasAdd/ReadVariableOp§
conv_2/BiasAddBiasAddconv_2/Conv2D:output:0%conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
2
conv_2/BiasAddЈ
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€00
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPool™
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_3/Conv2D/ReadVariableOpЌ
conv_3/Conv2DConv2Dmaxpool_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..*
paddingVALID*
strides
2
conv_3/Conv2D°
conv_3/BiasAdd/ReadVariableOpReadVariableOp&conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_3/BiasAdd/ReadVariableOp§
conv_3/BiasAddBiasAddconv_3/Conv2D:output:0%conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..2
conv_3/BiasAdd™
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOp 
conv_4/Conv2DConv2Dconv_3/BiasAdd:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,*
paddingVALID*
strides
2
conv_4/Conv2D°
conv_4/BiasAdd/ReadVariableOpReadVariableOp&conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_4/BiasAdd/ReadVariableOp§
conv_4/BiasAddBiasAddconv_4/Conv2D:output:0%conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,2
conv_4/BiasAddЈ
maxpool_2/MaxPoolMaxPoolconv_4/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
maxpool_2/MaxPool™
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_5/Conv2D/ReadVariableOpЌ
conv_5/Conv2DConv2Dmaxpool_2/MaxPool:output:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv_5/Conv2D°
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_5/BiasAdd/ReadVariableOp§
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv_5/BiasAdd™
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_6/Conv2D/ReadVariableOp 
conv_6/Conv2DConv2Dconv_5/BiasAdd:output:0$conv_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv_6/Conv2D°
conv_6/BiasAdd/ReadVariableOpReadVariableOp&conv_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_6/BiasAdd/ReadVariableOp§
conv_6/BiasAddBiasAddconv_6/Conv2D:output:0%conv_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv_6/BiasAddЈ
maxpool_3/MaxPoolMaxPoolconv_6/BiasAdd:output:0*/
_output_shapes
:€€€€€€€€€		*
ksize
*
paddingVALID*
strides
2
maxpool_3/MaxPool{
flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2
flatten_layer/Const¶
flatten_layer/ReshapeReshapemaxpool_3/MaxPool:output:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Р
2
flatten_layer/ReshapeЭ
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	Р
2*
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
output_layer/Softmax≠
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^conv_5/BiasAdd/ReadVariableOp^conv_5/Conv2D/ReadVariableOp^conv_6/BiasAdd/ReadVariableOp^conv_6/Conv2D/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 2:
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
conv_4/Conv2D/ReadVariableOpconv_4/Conv2D/ReadVariableOp2>
conv_5/BiasAdd/ReadVariableOpconv_5/BiasAdd/ReadVariableOp2<
conv_5/Conv2D/ReadVariableOpconv_5/Conv2D/ReadVariableOp2>
conv_6/BiasAdd/ReadVariableOpconv_6/BiasAdd/ReadVariableOp2<
conv_6/Conv2D/ReadVariableOpconv_6/Conv2D/ReadVariableOp2J
#output_layer/BiasAdd/ReadVariableOp#output_layer/BiasAdd/ReadVariableOp2H
"output_layer/MatMul/ReadVariableOp"output_layer/MatMul/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€dd
 
_user_specified_nameinputs
ѕ	
с
?__inference_FC_1_layer_call_and_return_conditional_losses_56642

inputs1
matmul_readvariableop_resource:	Р
2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р
2*
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
:€€€€€€€€€Р
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р

 
_user_specified_nameinputs
…
E
)__inference_maxpool_2_layer_call_fn_56506

inputs
identityе
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
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_2_layer_call_and_return_conditional_losses_565002
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
Ф
Т
$__inference_FC_1_layer_call_fn_57540

inputs
unknown:	Р
2
	unknown_0:2
identityИҐStatefulPartitionedCallп
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
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_1_layer_call_and_return_conditional_losses_566422
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Р
: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Р

 
_user_specified_nameinputs
љ
Ы
&__inference_conv_2_layer_call_fn_57434

inputs!
unknown:


	unknown_0:

identityИҐStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_565512
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€``
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€bb
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€bb

 
_user_specified_nameinputs
Ѓ

ъ
A__inference_conv_5_layer_call_and_return_conditional_losses_57482

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ґ
К
5__inference_WheatClassifier_CNN_2_layer_call_fn_57355

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
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:

unknown_11:	Р
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityИҐStatefulPartitionedCall„
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_566962
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€dd
 
_user_specified_nameinputs
…
E
)__inference_maxpool_1_layer_call_fn_56494

inputs
identityе
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
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_1_layer_call_and_return_conditional_losses_564882
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
АB
л
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_56941

inputs&
conv_1_56889:

conv_1_56891:
&
conv_2_56894:


conv_2_56896:
&
conv_3_56900:

conv_3_56902:&
conv_4_56905:
conv_4_56907:&
conv_5_56911:
conv_5_56913:&
conv_6_56916:
conv_6_56918:

fc_1_56923:	Р
2

fc_1_56925:2

fc_2_56929:22

fc_2_56931:2$
output_layer_56935:2 
output_layer_56937:
identityИҐFC_1/StatefulPartitionedCallҐFC_2/StatefulPartitionedCallҐconv_1/StatefulPartitionedCallҐconv_2/StatefulPartitionedCallҐconv_3/StatefulPartitionedCallҐconv_4/StatefulPartitionedCallҐconv_5/StatefulPartitionedCallҐconv_6/StatefulPartitionedCallҐ$output_layer/StatefulPartitionedCallП
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_56889conv_1_56891*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_1_layer_call_and_return_conditional_losses_565352 
conv_1/StatefulPartitionedCall∞
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_56894conv_2_56896*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_2_layer_call_and_return_conditional_losses_565512 
conv_2/StatefulPartitionedCall€
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€00
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_1_layer_call_and_return_conditional_losses_564882
maxpool_1/PartitionedCallЂ
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_56900conv_3_56902*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_3_layer_call_and_return_conditional_losses_565682 
conv_3/StatefulPartitionedCall∞
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_56905conv_4_56907*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_4_layer_call_and_return_conditional_losses_565842 
conv_4/StatefulPartitionedCall€
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_2_layer_call_and_return_conditional_losses_565002
maxpool_2/PartitionedCallЂ
conv_5/StatefulPartitionedCallStatefulPartitionedCall"maxpool_2/PartitionedCall:output:0conv_5_56911conv_5_56913*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_5_layer_call_and_return_conditional_losses_566012 
conv_5/StatefulPartitionedCall∞
conv_6/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0conv_6_56916conv_6_56918*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_conv_6_layer_call_and_return_conditional_losses_566172 
conv_6/StatefulPartitionedCall€
maxpool_3/PartitionedCallPartitionedCall'conv_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_maxpool_3_layer_call_and_return_conditional_losses_565122
maxpool_3/PartitionedCall€
flatten_layer/PartitionedCallPartitionedCall"maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Р
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_flatten_layer_layer_call_and_return_conditional_losses_566302
flatten_layer/PartitionedCallЭ
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0
fc_1_56923
fc_1_56925*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_1_layer_call_and_return_conditional_losses_566422
FC_1/StatefulPartitionedCallю
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_566532
leaky_ReLu_1/PartitionedCallЬ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0
fc_2_56929
fc_2_56931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *H
fCRA
?__inference_FC_2_layer_call_and_return_conditional_losses_566652
FC_2/StatefulPartitionedCallю
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_566762
leaky_ReLu_2/PartitionedCallƒ
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_56935output_layer_56937*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_566892&
$output_layer/StatefulPartitionedCallђ
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€dd: : : : : : : : : : : : : : : : : : 2<
FC_1/StatefulPartitionedCallFC_1/StatefulPartitionedCall2<
FC_2/StatefulPartitionedCallFC_2/StatefulPartitionedCall2@
conv_1/StatefulPartitionedCallconv_1/StatefulPartitionedCall2@
conv_2/StatefulPartitionedCallconv_2/StatefulPartitionedCall2@
conv_3/StatefulPartitionedCallconv_3/StatefulPartitionedCall2@
conv_4/StatefulPartitionedCallconv_4/StatefulPartitionedCall2@
conv_5/StatefulPartitionedCallconv_5/StatefulPartitionedCall2@
conv_6/StatefulPartitionedCallconv_6/StatefulPartitionedCall2L
$output_layer/StatefulPartitionedCall$output_layer/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€dd
 
_user_specified_nameinputs
Ѓ

ъ
A__inference_conv_6_layer_call_and_return_conditional_losses_56617

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѓ

ъ
A__inference_conv_2_layer_call_and_return_conditional_losses_57425

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
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
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
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€``
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€bb
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€bb

 
_user_specified_nameinputs
Ѓ

ъ
A__inference_conv_2_layer_call_and_return_conditional_losses_56551

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
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
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
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€``
2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€``
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€bb
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€bb

 
_user_specified_nameinputs
°
Щ
,__inference_output_layer_layer_call_fn_57599

inputs
unknown:2
	unknown_0:
identityИҐStatefulPartitionedCallч
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
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_output_layer_layer_call_and_return_conditional_losses_566892
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
Ћ	
р
?__inference_FC_2_layer_call_and_return_conditional_losses_56665

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
¬
H
,__inference_leaky_ReLu_2_layer_call_fn_57579

inputs
identity≈
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
 *-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_566762
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
Ѓ

ъ
A__inference_conv_4_layer_call_and_return_conditional_losses_56584

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€,,2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€,,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€..: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€..
 
_user_specified_nameinputs
ѕ	
с
?__inference_FC_1_layer_call_and_return_conditional_losses_57531

inputs1
matmul_readvariableop_resource:	Р
2-
biasadd_readvariableop_resource:2
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Р
2*
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
:€€€€€€€€€Р
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Р

 
_user_specified_nameinputs
Ѓ

ъ
A__inference_conv_3_layer_call_and_return_conditional_losses_56568

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€..2	
BiasAddЭ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€..2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€00
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€00

 
_user_specified_nameinputs
§
`
D__inference_maxpool_2_layer_call_and_return_conditional_losses_56500

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
 
_user_specified_nameinputs"ћL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*њ
serving_defaultЂ
K
input_layer<
serving_default_input_layer:0€€€€€€€€€dd@
output_layer0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:ух
∞С
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
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
к_default_save_signature
+л&call_and_return_all_conditional_losses
м__call__"°М
_tf_keras_networkДМ{"name": "WheatClassifier_CNN_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "WheatClassifier_CNN_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["maxpool_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_2", "inbound_nodes": [[["conv_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["maxpool_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_6", "inbound_nodes": [[["conv_5", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_3", "inbound_nodes": [[["conv_6", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_layer", "inbound_nodes": [[["maxpool_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1", "inbound_nodes": [[["flatten_layer", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_1", "inbound_nodes": [[["FC_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_2", "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_2", "inbound_nodes": [[["FC_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "shared_object_id": 34, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "float32", "input_layer"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "WheatClassifier_CNN_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_layer", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["maxpool_1", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_2", "inbound_nodes": [[["conv_4", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["maxpool_2", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_6", "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_3", "inbound_nodes": [[["conv_6", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_layer", "inbound_nodes": [[["maxpool_3", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1", "inbound_nodes": [[["flatten_layer", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_1", "inbound_nodes": [[["FC_1", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_2", "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]], "shared_object_id": 29}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_2", "inbound_nodes": [[["FC_2", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]], "shared_object_id": 33}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 36}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Addons>AdamW", "config": {"name": "AdamW", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false, "weight_decay": 9.999999747378752e-05}}}}
Е"В
_tf_keras_input_layerв{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
А

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+н&call_and_return_all_conditional_losses
о__call__"ў	
_tf_keras_layerњ	{"name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_layer", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}}
ы


kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
+п&call_and_return_all_conditional_losses
р__call__"‘	
_tf_keras_layerЇ	{"name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 98, 98, 10]}}
ѕ
#regularization_losses
$	variables
%trainable_variables
&	keras_api
+с&call_and_return_all_conditional_losses
т__call__"Њ
_tf_keras_layer§{"name": "maxpool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 39}}
€


'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+у&call_and_return_all_conditional_losses
ф__call__"Ў	
_tf_keras_layerЊ	{"name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["maxpool_1", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 10]}}
ю


-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+х&call_and_return_all_conditional_losses
ц__call__"„	
_tf_keras_layerљ	{"name": "conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 16]}}
–
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+ч&call_and_return_all_conditional_losses
ш__call__"њ
_tf_keras_layer•{"name": "maxpool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_4", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 42}}
Б

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+щ&call_and_return_all_conditional_losses
ъ__call__"Џ	
_tf_keras_layerј	{"name": "conv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["maxpool_2", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 16]}}
ю


=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
+ы&call_and_return_all_conditional_losses
ь__call__"„	
_tf_keras_layerљ	{"name": "conv_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 16]}}
–
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
+э&call_and_return_all_conditional_losses
ю__call__"њ
_tf_keras_layer•{"name": "maxpool_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_6", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
ќ
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+€&call_and_return_all_conditional_losses
А__call__"љ
_tf_keras_layer£{"name": "flatten_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["maxpool_3", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
Д	

Kkernel
Lbias
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"Ё
_tf_keras_layer√{"name": "FC_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_layer", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1296}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1296]}}
Я
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"О
_tf_keras_layerф{"name": "leaky_ReLu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_1", 0, 0, {}]]], "shared_object_id": 26}
€

Ukernel
Vbias
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"Ў
_tf_keras_layerЊ{"name": "FC_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]], "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Я
[regularization_losses
\	variables
]trainable_variables
^	keras_api
+З&call_and_return_all_conditional_losses
И__call__"О
_tf_keras_layerф{"name": "leaky_ReLu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_2", 0, 0, {}]]], "shared_object_id": 30}
П	

_kernel
`bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"и
_tf_keras_layerќ{"name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Ќ
eiter

fbeta_1

gbeta_2
	hdecay
ilearning_rate
jweight_decaym∆m«m»m…'m (mЋ-mћ.mЌ7mќ8mѕ=m–>m—Km“Lm”Um‘Vm’_m÷`m„vЎvўvЏvџ'v№(vЁ-vё.vя7vа8vб=vв>vгKvдLvеUvжVvз_vи`vй"
	optimizer
 "
trackable_list_wrapper
¶
0
1
2
3
'4
(5
-6
.7
78
89
=10
>11
K12
L13
U14
V15
_16
`17"
trackable_list_wrapper
¶
0
1
2
3
'4
(5
-6
.7
78
89
=10
>11
K12
L13
U14
V15
_16
`17"
trackable_list_wrapper
ќ
regularization_losses
knon_trainable_variables
llayer_metrics
mlayer_regularization_losses
	variables

nlayers
trainable_variables
ometrics
м__call__
к_default_save_signature
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
-
Лserving_default"
signature_map
':%
2conv_1/kernel
:
2conv_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
regularization_losses
pnon_trainable_variables
qlayer_metrics
rlayer_regularization_losses
	variables

slayers
trainable_variables
tmetrics
о__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
':%

2conv_2/kernel
:
2conv_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
regularization_losses
unon_trainable_variables
vlayer_metrics
wlayer_regularization_losses
 	variables

xlayers
!trainable_variables
ymetrics
р__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
#regularization_losses
znon_trainable_variables
{layer_metrics
|layer_regularization_losses
$	variables

}layers
%trainable_variables
~metrics
т__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
':%
2conv_3/kernel
:2conv_3/bias
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
і
)regularization_losses
non_trainable_variables
Аlayer_metrics
 Бlayer_regularization_losses
*	variables
Вlayers
+trainable_variables
Гmetrics
ф__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
':%2conv_4/kernel
:2conv_4/bias
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
µ
/regularization_losses
Дnon_trainable_variables
Еlayer_metrics
 Жlayer_regularization_losses
0	variables
Зlayers
1trainable_variables
Иmetrics
ц__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
3regularization_losses
Йnon_trainable_variables
Кlayer_metrics
 Лlayer_regularization_losses
4	variables
Мlayers
5trainable_variables
Нmetrics
ш__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
':%2conv_5/kernel
:2conv_5/bias
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
µ
9regularization_losses
Оnon_trainable_variables
Пlayer_metrics
 Рlayer_regularization_losses
:	variables
Сlayers
;trainable_variables
Тmetrics
ъ__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
':%2conv_6/kernel
:2conv_6/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
µ
?regularization_losses
Уnon_trainable_variables
Фlayer_metrics
 Хlayer_regularization_losses
@	variables
Цlayers
Atrainable_variables
Чmetrics
ь__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Cregularization_losses
Шnon_trainable_variables
Щlayer_metrics
 Ъlayer_regularization_losses
D	variables
Ыlayers
Etrainable_variables
Ьmetrics
ю__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Gregularization_losses
Эnon_trainable_variables
Юlayer_metrics
 Яlayer_regularization_losses
H	variables
†layers
Itrainable_variables
°metrics
А__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
:	Р
22FC_1/kernel
:22	FC_1/bias
 "
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
µ
Mregularization_losses
Ґnon_trainable_variables
£layer_metrics
 §layer_regularization_losses
N	variables
•layers
Otrainable_variables
¶metrics
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Qregularization_losses
Іnon_trainable_variables
®layer_metrics
 ©layer_regularization_losses
R	variables
™layers
Strainable_variables
Ђmetrics
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
:222FC_2/kernel
:22	FC_2/bias
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
µ
Wregularization_losses
ђnon_trainable_variables
≠layer_metrics
 Ѓlayer_regularization_losses
X	variables
ѓlayers
Ytrainable_variables
∞metrics
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
[regularization_losses
±non_trainable_variables
≤layer_metrics
 ≥layer_regularization_losses
\	variables
іlayers
]trainable_variables
µmetrics
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
%:#22output_layer/kernel
:2output_layer/bias
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
µ
aregularization_losses
ґnon_trainable_variables
Јlayer_metrics
 Єlayer_regularization_losses
b	variables
єlayers
ctrainable_variables
Їmetrics
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
:	 (2
AdamW/iter
: (2AdamW/beta_1
: (2AdamW/beta_2
: (2AdamW/decay
: (2AdamW/learning_rate
: (2AdamW/weight_decay
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
15"
trackable_list_wrapper
0
ї0
Љ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
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

љtotal

Њcount
њ	variables
ј	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 50}
Т

Ѕtotal

¬count
√
_fn_kwargs
ƒ	variables
≈	keras_api"∆
_tf_keras_metricЂ{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 36}
:  (2total
:  (2count
0
љ0
Њ1"
trackable_list_wrapper
.
њ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ѕ0
¬1"
trackable_list_wrapper
.
ƒ	variables"
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
-:+2AdamW/conv_5/kernel/m
:2AdamW/conv_5/bias/m
-:+2AdamW/conv_6/kernel/m
:2AdamW/conv_6/bias/m
$:"	Р
22AdamW/FC_1/kernel/m
:22AdamW/FC_1/bias/m
#:!222AdamW/FC_2/kernel/m
:22AdamW/FC_2/bias/m
+:)22AdamW/output_layer/kernel/m
%:#2AdamW/output_layer/bias/m
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
-:+2AdamW/conv_5/kernel/v
:2AdamW/conv_5/bias/v
-:+2AdamW/conv_6/kernel/v
:2AdamW/conv_6/bias/v
$:"	Р
22AdamW/FC_1/kernel/v
:22AdamW/FC_1/bias/v
#:!222AdamW/FC_2/kernel/v
:22AdamW/FC_2/bias/v
+:)22AdamW/output_layer/kernel/v
%:#2AdamW/output_layer/bias/v
к2з
 __inference__wrapped_model_56482¬
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
annotations™ *2Ґ/
-К*
input_layer€€€€€€€€€dd
О2Л
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57248
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57314
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57076
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57131ј
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
Ґ2Я
5__inference_WheatClassifier_CNN_2_layer_call_fn_56735
5__inference_WheatClassifier_CNN_2_layer_call_fn_57355
5__inference_WheatClassifier_CNN_2_layer_call_fn_57396
5__inference_WheatClassifier_CNN_2_layer_call_fn_57021ј
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
л2и
A__inference_conv_1_layer_call_and_return_conditional_losses_57406Ґ
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
&__inference_conv_1_layer_call_fn_57415Ґ
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
A__inference_conv_2_layer_call_and_return_conditional_losses_57425Ґ
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
&__inference_conv_2_layer_call_fn_57434Ґ
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
ђ2©
D__inference_maxpool_1_layer_call_and_return_conditional_losses_56488а
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
С2О
)__inference_maxpool_1_layer_call_fn_56494а
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
л2и
A__inference_conv_3_layer_call_and_return_conditional_losses_57444Ґ
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
&__inference_conv_3_layer_call_fn_57453Ґ
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
A__inference_conv_4_layer_call_and_return_conditional_losses_57463Ґ
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
&__inference_conv_4_layer_call_fn_57472Ґ
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
ђ2©
D__inference_maxpool_2_layer_call_and_return_conditional_losses_56500а
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
С2О
)__inference_maxpool_2_layer_call_fn_56506а
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
л2и
A__inference_conv_5_layer_call_and_return_conditional_losses_57482Ґ
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
&__inference_conv_5_layer_call_fn_57491Ґ
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
A__inference_conv_6_layer_call_and_return_conditional_losses_57501Ґ
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
&__inference_conv_6_layer_call_fn_57510Ґ
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
ђ2©
D__inference_maxpool_3_layer_call_and_return_conditional_losses_56512а
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
С2О
)__inference_maxpool_3_layer_call_fn_56518а
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
т2п
H__inference_flatten_layer_layer_call_and_return_conditional_losses_57516Ґ
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
-__inference_flatten_layer_layer_call_fn_57521Ґ
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
й2ж
?__inference_FC_1_layer_call_and_return_conditional_losses_57531Ґ
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
$__inference_FC_1_layer_call_fn_57540Ґ
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
с2о
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_57545Ґ
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
÷2”
,__inference_leaky_ReLu_1_layer_call_fn_57550Ґ
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
й2ж
?__inference_FC_2_layer_call_and_return_conditional_losses_57560Ґ
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
$__inference_FC_2_layer_call_fn_57569Ґ
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
с2о
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_57574Ґ
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
÷2”
,__inference_leaky_ReLu_2_layer_call_fn_57579Ґ
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
с2о
G__inference_output_layer_layer_call_and_return_conditional_losses_57590Ґ
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
÷2”
,__inference_output_layer_layer_call_fn_57599Ґ
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
ќBЋ
#__inference_signature_wrapper_57182input_layer"Ф
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
 †
?__inference_FC_1_layer_call_and_return_conditional_losses_57531]KL0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р

™ "%Ґ"
К
0€€€€€€€€€2
Ъ x
$__inference_FC_1_layer_call_fn_57540PKL0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Р

™ "К€€€€€€€€€2Я
?__inference_FC_2_layer_call_and_return_conditional_losses_57560\UV/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€2
Ъ w
$__inference_FC_2_layer_call_fn_57569OUV/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€2÷
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57076Б'(-.78=>KLUV_`DҐA
:Ґ7
-К*
input_layer€€€€€€€€€dd
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ÷
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57131Б'(-.78=>KLUV_`DҐA
:Ґ7
-К*
input_layer€€€€€€€€€dd
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ –
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57248|'(-.78=>KLUV_`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€dd
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ –
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_57314|'(-.78=>KLUV_`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€dd
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ≠
5__inference_WheatClassifier_CNN_2_layer_call_fn_56735t'(-.78=>KLUV_`DҐA
:Ґ7
-К*
input_layer€€€€€€€€€dd
p 

 
™ "К€€€€€€€€€≠
5__inference_WheatClassifier_CNN_2_layer_call_fn_57021t'(-.78=>KLUV_`DҐA
:Ґ7
-К*
input_layer€€€€€€€€€dd
p

 
™ "К€€€€€€€€€®
5__inference_WheatClassifier_CNN_2_layer_call_fn_57355o'(-.78=>KLUV_`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€dd
p 

 
™ "К€€€€€€€€€®
5__inference_WheatClassifier_CNN_2_layer_call_fn_57396o'(-.78=>KLUV_`?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€dd
p

 
™ "К€€€€€€€€€і
 __inference__wrapped_model_56482П'(-.78=>KLUV_`<Ґ9
2Ґ/
-К*
input_layer€€€€€€€€€dd
™ ";™8
6
output_layer&К#
output_layer€€€€€€€€€±
A__inference_conv_1_layer_call_and_return_conditional_losses_57406l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€dd
™ "-Ґ*
#К 
0€€€€€€€€€bb

Ъ Й
&__inference_conv_1_layer_call_fn_57415_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€dd
™ " К€€€€€€€€€bb
±
A__inference_conv_2_layer_call_and_return_conditional_losses_57425l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€bb

™ "-Ґ*
#К 
0€€€€€€€€€``

Ъ Й
&__inference_conv_2_layer_call_fn_57434_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€bb

™ " К€€€€€€€€€``
±
A__inference_conv_3_layer_call_and_return_conditional_losses_57444l'(7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€00

™ "-Ґ*
#К 
0€€€€€€€€€..
Ъ Й
&__inference_conv_3_layer_call_fn_57453_'(7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€00

™ " К€€€€€€€€€..±
A__inference_conv_4_layer_call_and_return_conditional_losses_57463l-.7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€..
™ "-Ґ*
#К 
0€€€€€€€€€,,
Ъ Й
&__inference_conv_4_layer_call_fn_57472_-.7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€..
™ " К€€€€€€€€€,,±
A__inference_conv_5_layer_call_and_return_conditional_losses_57482l787Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Й
&__inference_conv_5_layer_call_fn_57491_787Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€±
A__inference_conv_6_layer_call_and_return_conditional_losses_57501l=>7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ Й
&__inference_conv_6_layer_call_fn_57510_=>7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ " К€€€€€€€€€≠
H__inference_flatten_layer_layer_call_and_return_conditional_losses_57516a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€		
™ "&Ґ#
К
0€€€€€€€€€Р

Ъ Е
-__inference_flatten_layer_layer_call_fn_57521T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€		
™ "К€€€€€€€€€Р
£
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_57545X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€2
Ъ {
,__inference_leaky_ReLu_1_layer_call_fn_57550K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€2£
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_57574X/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€2
Ъ {
,__inference_leaky_ReLu_2_layer_call_fn_57579K/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€2з
D__inference_maxpool_1_layer_call_and_return_conditional_losses_56488ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ њ
)__inference_maxpool_1_layer_call_fn_56494СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€з
D__inference_maxpool_2_layer_call_and_return_conditional_losses_56500ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ њ
)__inference_maxpool_2_layer_call_fn_56506СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€з
D__inference_maxpool_3_layer_call_and_return_conditional_losses_56512ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ њ
)__inference_maxpool_3_layer_call_fn_56518СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€І
G__inference_output_layer_layer_call_and_return_conditional_losses_57590\_`/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "%Ґ"
К
0€€€€€€€€€
Ъ 
,__inference_output_layer_layer_call_fn_57599O_`/Ґ,
%Ґ"
 К
inputs€€€€€€€€€2
™ "К€€€€€€€€€∆
#__inference_signature_wrapper_57182Ю'(-.78=>KLUV_`KҐH
Ґ 
A™>
<
input_layer-К*
input_layer€€€€€€€€€dd";™8
6
output_layer&К#
output_layer€€€€€€€€€