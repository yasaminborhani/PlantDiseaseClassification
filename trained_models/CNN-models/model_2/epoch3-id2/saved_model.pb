эв
 Н
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
ѓ
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
delete_old_dirsbool(ѕ
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
dtypetypeѕ
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
list(type)(0ѕ
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.5.02v2.5.0-0-ga4dfb8d1a718ЏЯ
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
shape:	љ
2*
shared_nameFC_1/kernel
l
FC_1/kernel/Read/ReadVariableOpReadVariableOpFC_1/kernel*
_output_shapes
:	љ
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
ј
AdamW/conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_1/kernel/m
Є
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
ј
AdamW/conv_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_5/kernel/m
Є
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
ј
AdamW/conv_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_6/kernel/m
Є
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
Ѓ
AdamW/FC_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ
2*$
shared_nameAdamW/FC_1/kernel/m
|
'AdamW/FC_1/kernel/m/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/m*
_output_shapes
:	љ
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
ј
AdamW/conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdamW/conv_1/kernel/v
Є
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
ј
AdamW/conv_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_5/kernel/v
Є
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
ј
AdamW/conv_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdamW/conv_6/kernel/v
Є
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
Ѓ
AdamW/FC_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	љ
2*$
shared_nameAdamW/FC_1/kernel/v
|
'AdamW/FC_1/kernel/v/Read/ReadVariableOpReadVariableOpAdamW/FC_1/kernel/v*
_output_shapes
:	љ
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

NoOpNoOp
Хi
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ыh
valueуhBСh BПh
▒
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
║
eiter

fbeta_1

gbeta_2
	hdecay
ilearning_rate
jweight_decaymкmКm╚m╔'m╩(m╦-m╠.m═7m╬8m¤=mл>mЛKmмLmМUmнVmН_mо`mОvпv┘v┌v█'v▄(vП-vя.v▀7vЯ8vр=vР>vсKvСLvтUvТVvу_vУ`vж
 
є
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
є
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
Г
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
Г
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
Г
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
Г
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
▒
)regularization_losses
non_trainable_variables
ђlayer_metrics
 Ђlayer_regularization_losses
*	variables
ѓlayers
+trainable_variables
Ѓmetrics
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
▓
/regularization_losses
ёnon_trainable_variables
Ёlayer_metrics
 єlayer_regularization_losses
0	variables
Єlayers
1trainable_variables
ѕmetrics
 
 
 
▓
3regularization_losses
Ѕnon_trainable_variables
іlayer_metrics
 Іlayer_regularization_losses
4	variables
їlayers
5trainable_variables
Їmetrics
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
▓
9regularization_losses
јnon_trainable_variables
Јlayer_metrics
 љlayer_regularization_losses
:	variables
Љlayers
;trainable_variables
њmetrics
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
▓
?regularization_losses
Њnon_trainable_variables
ћlayer_metrics
 Ћlayer_regularization_losses
@	variables
ќlayers
Atrainable_variables
Ќmetrics
 
 
 
▓
Cregularization_losses
ўnon_trainable_variables
Ўlayer_metrics
 џlayer_regularization_losses
D	variables
Џlayers
Etrainable_variables
юmetrics
 
 
 
▓
Gregularization_losses
Юnon_trainable_variables
ъlayer_metrics
 Ъlayer_regularization_losses
H	variables
аlayers
Itrainable_variables
Аmetrics
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
▓
Mregularization_losses
бnon_trainable_variables
Бlayer_metrics
 цlayer_regularization_losses
N	variables
Цlayers
Otrainable_variables
дmetrics
 
 
 
▓
Qregularization_losses
Дnon_trainable_variables
еlayer_metrics
 Еlayer_regularization_losses
R	variables
фlayers
Strainable_variables
Фmetrics
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
▓
Wregularization_losses
гnon_trainable_variables
Гlayer_metrics
 «layer_regularization_losses
X	variables
»layers
Ytrainable_variables
░metrics
 
 
 
▓
[regularization_losses
▒non_trainable_variables
▓layer_metrics
 │layer_regularization_losses
\	variables
┤layers
]trainable_variables
хmetrics
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
▓
aregularization_losses
Хnon_trainable_variables
иlayer_metrics
 Иlayer_regularization_losses
b	variables
╣layers
ctrainable_variables
║metrics
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
╗0
╝1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

йtotal

Йcount
┐	variables
└	keras_api
I

┴total

┬count
├
_fn_kwargs
─	variables
┼	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

й0
Й1

┐	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

┴0
┬1

─	variables
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
ёЂ
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
ёЂ
VARIABLE_VALUEAdamW/output_layer/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdamW/output_layer/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ј
serving_default_input_layerPlaceholder*/
_output_shapes
:         dd*
dtype0*$
shape:         dd
┘
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_layerconv_1/kernelconv_1/biasconv_2/kernelconv_2/biasconv_3/kernelconv_3/biasconv_4/kernelconv_4/biasconv_5/kernelconv_5/biasconv_6/kernelconv_6/biasFC_1/kernel	FC_1/biasFC_2/kernel	FC_2/biasoutput_layer/kerneloutput_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_10157
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Э
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
GPU 2J 8ѓ *'
f"R 
__inference__traced_save_10789
з
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
GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_10991ЈМ

Ь
§
#__inference_signature_wrapper_10157
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

unknown_11:	љ
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityѕбStatefulPartitionedCallФ
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
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__wrapped_model_94572
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameinput_layer
Г

щ
@__inference_conv_5_layer_call_and_return_conditional_losses_9576

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
:         *
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
:         2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
К
D
(__inference_maxpool_2_layer_call_fn_9481

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_2_layer_call_and_return_conditional_losses_94752
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╝
Џ
&__inference_conv_4_layer_call_fn_10447

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_4_layer_call_and_return_conditional_losses_95592
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         ,,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ..: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         ..
 
_user_specified_nameinputs
■A
№
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10051
input_layer%
conv_1_9999:

conv_1_10001:
&
conv_2_10004:


conv_2_10006:
&
conv_3_10010:

conv_3_10012:&
conv_4_10015:
conv_4_10017:&
conv_5_10021:
conv_5_10023:&
conv_6_10026:
conv_6_10028:

fc_1_10033:	љ
2

fc_1_10035:2

fc_2_10039:22

fc_2_10041:2$
output_layer_10045:2 
output_layer_10047:
identityѕбFC_1/StatefulPartitionedCallбFC_2/StatefulPartitionedCallбconv_1/StatefulPartitionedCallбconv_2/StatefulPartitionedCallбconv_3/StatefulPartitionedCallбconv_4/StatefulPartitionedCallбconv_5/StatefulPartitionedCallбconv_6/StatefulPartitionedCallб$output_layer/StatefulPartitionedCallњ
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_9999conv_1_10001*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_95102 
conv_1/StatefulPartitionedCall»
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_10004conv_2_10006*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_95262 
conv_2/StatefulPartitionedCall■
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_1_layer_call_and_return_conditional_losses_94632
maxpool_1/PartitionedCallф
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_10010conv_3_10012*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_95432 
conv_3/StatefulPartitionedCall»
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_10015conv_4_10017*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_4_layer_call_and_return_conditional_losses_95592 
conv_4/StatefulPartitionedCall■
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_2_layer_call_and_return_conditional_losses_94752
maxpool_2/PartitionedCallф
conv_5/StatefulPartitionedCallStatefulPartitionedCall"maxpool_2/PartitionedCall:output:0conv_5_10021conv_5_10023*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_5_layer_call_and_return_conditional_losses_95762 
conv_5/StatefulPartitionedCall»
conv_6/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0conv_6_10026conv_6_10028*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_6_layer_call_and_return_conditional_losses_95922 
conv_6/StatefulPartitionedCall■
maxpool_3/PartitionedCallPartitionedCall'conv_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_3_layer_call_and_return_conditional_losses_94872
maxpool_3/PartitionedCall■
flatten_layer/PartitionedCallPartitionedCall"maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_flatten_layer_layer_call_and_return_conditional_losses_96052
flatten_layer/PartitionedCallю
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0
fc_1_10033
fc_1_10035*
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_1_layer_call_and_return_conditional_losses_96172
FC_1/StatefulPartitionedCall§
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_96282
leaky_ReLu_1/PartitionedCallЏ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0
fc_2_10039
fc_2_10041*
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_2_layer_call_and_return_conditional_losses_96402
FC_2/StatefulPartitionedCall§
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_96512
leaky_ReLu_2/PartitionedCall├
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_10045output_layer_10047*
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
GPU 2J 8ѓ *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_96642&
$output_layer/StatefulPartitionedCallг
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 2<
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
:         dd
%
_user_specified_nameinput_layer
╝
Џ
&__inference_conv_2_layer_call_fn_10409

inputs!
unknown:


	unknown_0:

identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_95262
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         ``
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         bb
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         bb

 
_user_specified_nameinputs
и

Э
G__inference_output_layer_layer_call_and_return_conditional_losses_10565

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
Б
_
C__inference_maxpool_2_layer_call_and_return_conditional_losses_9475

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╝
Џ
&__inference_conv_1_layer_call_fn_10390

inputs!
unknown:

	unknown_0:

identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_95102
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         bb
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
»
ј
4__inference_WheatClassifier_CNN_2_layer_call_fn_9710
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

unknown_11:	љ
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityѕбStatefulPartitionedCall█
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
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_96712
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameinput_layer
ж
c
G__inference_flatten_layer_layer_call_and_return_conditional_losses_9605

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         љ
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         љ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         		:W S
/
_output_shapes
:         		
 
_user_specified_nameinputs
А
і
5__inference_WheatClassifier_CNN_2_layer_call_fn_10371

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

unknown_11:	љ
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityѕбStatefulPartitionedCallо
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
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_99162
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
╠A
п
O__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_9916

inputs%
conv_1_9864:

conv_1_9866:
%
conv_2_9869:


conv_2_9871:
%
conv_3_9875:

conv_3_9877:%
conv_4_9880:
conv_4_9882:%
conv_5_9886:
conv_5_9888:%
conv_6_9891:
conv_6_9893:
	fc_1_9898:	љ
2
	fc_1_9900:2
	fc_2_9904:22
	fc_2_9906:2#
output_layer_9910:2
output_layer_9912:
identityѕбFC_1/StatefulPartitionedCallбFC_2/StatefulPartitionedCallбconv_1/StatefulPartitionedCallбconv_2/StatefulPartitionedCallбconv_3/StatefulPartitionedCallбconv_4/StatefulPartitionedCallбconv_5/StatefulPartitionedCallбconv_6/StatefulPartitionedCallб$output_layer/StatefulPartitionedCallї
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_9864conv_1_9866*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_95102 
conv_1/StatefulPartitionedCallГ
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_9869conv_2_9871*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_95262 
conv_2/StatefulPartitionedCall■
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_1_layer_call_and_return_conditional_losses_94632
maxpool_1/PartitionedCallе
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_9875conv_3_9877*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_95432 
conv_3/StatefulPartitionedCallГ
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_9880conv_4_9882*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_4_layer_call_and_return_conditional_losses_95592 
conv_4/StatefulPartitionedCall■
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_2_layer_call_and_return_conditional_losses_94752
maxpool_2/PartitionedCallе
conv_5/StatefulPartitionedCallStatefulPartitionedCall"maxpool_2/PartitionedCall:output:0conv_5_9886conv_5_9888*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_5_layer_call_and_return_conditional_losses_95762 
conv_5/StatefulPartitionedCallГ
conv_6/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0conv_6_9891conv_6_9893*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_6_layer_call_and_return_conditional_losses_95922 
conv_6/StatefulPartitionedCall■
maxpool_3/PartitionedCallPartitionedCall'conv_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_3_layer_call_and_return_conditional_losses_94872
maxpool_3/PartitionedCall■
flatten_layer/PartitionedCallPartitionedCall"maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_flatten_layer_layer_call_and_return_conditional_losses_96052
flatten_layer/PartitionedCallџ
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0	fc_1_9898	fc_1_9900*
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_1_layer_call_and_return_conditional_losses_96172
FC_1/StatefulPartitionedCall§
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_96282
leaky_ReLu_1/PartitionedCallЎ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0	fc_2_9904	fc_2_9906*
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_2_layer_call_and_return_conditional_losses_96402
FC_2/StatefulPartitionedCall§
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_96512
leaky_ReLu_2/PartitionedCall┴
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_9910output_layer_9912*
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
GPU 2J 8ѓ *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_96642&
$output_layer/StatefulPartitionedCallг
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 2<
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
:         dd
 
_user_specified_nameinputs
К
D
(__inference_maxpool_3_layer_call_fn_9493

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_3_layer_call_and_return_conditional_losses_94872
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
щ
b
F__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_9651

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
Ж
d
H__inference_flatten_layer_layer_call_and_return_conditional_losses_10491

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         љ
2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         љ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         		:W S
/
_output_shapes
:         		
 
_user_specified_nameinputs
Щ
c
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_10520

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
╝
Џ
&__inference_conv_5_layer_call_fn_10466

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_5_layer_call_and_return_conditional_losses_95762
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
а
Ў
,__inference_output_layer_layer_call_fn_10574

inputs
unknown:2
	unknown_0:
identityѕбStatefulPartitionedCallШ
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
GPU 2J 8ѓ *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_96642
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
└љ
Ц'
!__inference__traced_restore_10991
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
assignvariableop_12_fc_1_kernel:	љ
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
'assignvariableop_40_adamw_fc_1_kernel_m:	љ
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
'assignvariableop_58_adamw_fc_1_kernel_v:	љ
23
%assignvariableop_59_adamw_fc_1_bias_v:29
'assignvariableop_60_adamw_fc_2_kernel_v:223
%assignvariableop_61_adamw_fc_2_bias_v:2A
/assignvariableop_62_adamw_output_layer_kernel_v:2;
-assignvariableop_63_adamw_output_layer_bias_v:
identity_65ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9А$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*Г#
valueБ#Bа#AB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЊ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*Ќ
valueЇBіAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesз
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*џ
_output_shapesЄ
ё:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_conv_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Б
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ц
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Б
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ц
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Б
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ц
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Б
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ц
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Б
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Е
AssignVariableOp_10AssignVariableOp!assignvariableop_10_conv_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Д
AssignVariableOp_11AssignVariableOpassignvariableop_11_conv_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Д
AssignVariableOp_12AssignVariableOpassignvariableop_12_fc_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ц
AssignVariableOp_13AssignVariableOpassignvariableop_13_fc_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Д
AssignVariableOp_14AssignVariableOpassignvariableop_14_fc_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ц
AssignVariableOp_15AssignVariableOpassignvariableop_15_fc_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16»
AssignVariableOp_16AssignVariableOp'assignvariableop_16_output_layer_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Г
AssignVariableOp_17AssignVariableOp%assignvariableop_17_output_layer_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18д
AssignVariableOp_18AssignVariableOpassignvariableop_18_adamw_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19е
AssignVariableOp_19AssignVariableOp assignvariableop_19_adamw_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20е
AssignVariableOp_20AssignVariableOp assignvariableop_20_adamw_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Д
AssignVariableOp_21AssignVariableOpassignvariableop_21_adamw_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22»
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adamw_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23«
AssignVariableOp_23AssignVariableOp&assignvariableop_23_adamw_weight_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24А
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25А
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Б
AssignVariableOp_26AssignVariableOpassignvariableop_26_total_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Б
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▒
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adamw_conv_1_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29»
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adamw_conv_1_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▒
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adamw_conv_2_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31»
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adamw_conv_2_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adamw_conv_3_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33»
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adamw_conv_3_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adamw_conv_4_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35»
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adamw_conv_4_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▒
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adamw_conv_5_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37»
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adamw_conv_5_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adamw_conv_6_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39»
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adamw_conv_6_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40»
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adamw_fc_1_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Г
AssignVariableOp_41AssignVariableOp%assignvariableop_41_adamw_fc_1_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42»
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adamw_fc_2_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Г
AssignVariableOp_43AssignVariableOp%assignvariableop_43_adamw_fc_2_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44и
AssignVariableOp_44AssignVariableOp/assignvariableop_44_adamw_output_layer_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45х
AssignVariableOp_45AssignVariableOp-assignvariableop_45_adamw_output_layer_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adamw_conv_1_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47»
AssignVariableOp_47AssignVariableOp'assignvariableop_47_adamw_conv_1_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adamw_conv_2_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49»
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adamw_conv_2_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50▒
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adamw_conv_3_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51»
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adamw_conv_3_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52▒
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adamw_conv_4_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53»
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adamw_conv_4_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54▒
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adamw_conv_5_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55»
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adamw_conv_5_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56▒
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adamw_conv_6_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57»
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adamw_conv_6_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58»
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adamw_fc_1_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Г
AssignVariableOp_59AssignVariableOp%assignvariableop_59_adamw_fc_1_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60»
AssignVariableOp_60AssignVariableOp'assignvariableop_60_adamw_fc_2_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Г
AssignVariableOp_61AssignVariableOp%assignvariableop_61_adamw_fc_2_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62и
AssignVariableOp_62AssignVariableOp/assignvariableop_62_adamw_output_layer_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63х
AssignVariableOp_63AssignVariableOp-assignvariableop_63_adamw_output_layer_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_639
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpя
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_64Л
Identity_65IdentityIdentity_64:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_65"#
identity_65Identity_65:output:0*Ќ
_input_shapesЁ
ѓ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
А
і
5__inference_WheatClassifier_CNN_2_layer_call_fn_10330

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

unknown_11:	љ
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityѕбStatefulPartitionedCallо
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
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_96712
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
╦	
­
?__inference_FC_2_layer_call_and_return_conditional_losses_10535

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
Г

щ
@__inference_conv_2_layer_call_and_return_conditional_losses_9526

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
:         ``
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
:         ``
2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         ``
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         bb
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         bb

 
_user_specified_nameinputs
¤	
ы
?__inference_FC_1_layer_call_and_return_conditional_losses_10506

inputs1
matmul_readvariableop_resource:	љ
2-
biasadd_readvariableop_resource:2
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	љ
2*
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
_construction_contextkEagerRuntime*+
_input_shapes
:         љ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         љ

 
_user_specified_nameinputs
Х

э
F__inference_output_layer_layer_call_and_return_conditional_losses_9664

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
┴
H
,__inference_leaky_ReLu_1_layer_call_fn_10525

inputs
identity─
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_96282
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
┬X
к
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10289

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
#fc_1_matmul_readvariableop_resource:	љ
22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identityѕбFC_1/BiasAdd/ReadVariableOpбFC_1/MatMul/ReadVariableOpбFC_2/BiasAdd/ReadVariableOpбFC_2/MatMul/ReadVariableOpбconv_1/BiasAdd/ReadVariableOpбconv_1/Conv2D/ReadVariableOpбconv_2/BiasAdd/ReadVariableOpбconv_2/Conv2D/ReadVariableOpбconv_3/BiasAdd/ReadVariableOpбconv_3/Conv2D/ReadVariableOpбconv_4/BiasAdd/ReadVariableOpбconv_4/Conv2D/ReadVariableOpбconv_5/BiasAdd/ReadVariableOpбconv_5/Conv2D/ReadVariableOpбconv_6/BiasAdd/ReadVariableOpбconv_6/Conv2D/ReadVariableOpб#output_layer/BiasAdd/ReadVariableOpб"output_layer/MatMul/ReadVariableOpф
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOp╣
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         bb
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
:         bb
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
:         ``
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
:         ``
2
conv_2/BiasAddи
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:         00
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPoolф
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_3/Conv2D/ReadVariableOp═
conv_3/Conv2DConv2Dmaxpool_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ..*
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
:         ..2
conv_3/BiasAddф
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOp╩
conv_4/Conv2DConv2Dconv_3/BiasAdd:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ,,*
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
:         ,,2
conv_4/BiasAddи
maxpool_2/MaxPoolMaxPoolconv_4/BiasAdd:output:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
maxpool_2/MaxPoolф
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_5/Conv2D/ReadVariableOp═
conv_5/Conv2DConv2Dmaxpool_2/MaxPool:output:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv_5/Conv2DА
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_5/BiasAdd/ReadVariableOpц
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv_5/BiasAddф
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_6/Conv2D/ReadVariableOp╩
conv_6/Conv2DConv2Dconv_5/BiasAdd:output:0$conv_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv_6/Conv2DА
conv_6/BiasAdd/ReadVariableOpReadVariableOp&conv_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_6/BiasAdd/ReadVariableOpц
conv_6/BiasAddBiasAddconv_6/Conv2D:output:0%conv_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv_6/BiasAddи
maxpool_3/MaxPoolMaxPoolconv_6/BiasAdd:output:0*/
_output_shapes
:         		*
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
valueB"      2
flatten_layer/Constд
flatten_layer/ReshapeReshapemaxpool_3/MaxPool:output:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:         љ
2
flatten_layer/ReshapeЮ
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	љ
2*
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
output_layer/SoftmaxГ
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^conv_5/BiasAdd/ReadVariableOp^conv_5/Conv2D/ReadVariableOp^conv_6/BiasAdd/ReadVariableOp^conv_6/Conv2D/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 2:
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
:         dd
 
_user_specified_nameinputs
»
ј
4__inference_WheatClassifier_CNN_2_layer_call_fn_9996
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

unknown_11:	љ
2

unknown_12:2

unknown_13:22

unknown_14:2

unknown_15:2

unknown_16:
identityѕбStatefulPartitionedCall█
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
:         *4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *X
fSRQ
O__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_99162
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
/
_output_shapes
:         dd
%
_user_specified_nameinput_layer
щ
b
F__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_9628

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
ђB
­
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10106
input_layer&
conv_1_10054:

conv_1_10056:
&
conv_2_10059:


conv_2_10061:
&
conv_3_10065:

conv_3_10067:&
conv_4_10070:
conv_4_10072:&
conv_5_10076:
conv_5_10078:&
conv_6_10081:
conv_6_10083:

fc_1_10088:	љ
2

fc_1_10090:2

fc_2_10094:22

fc_2_10096:2$
output_layer_10100:2 
output_layer_10102:
identityѕбFC_1/StatefulPartitionedCallбFC_2/StatefulPartitionedCallбconv_1/StatefulPartitionedCallбconv_2/StatefulPartitionedCallбconv_3/StatefulPartitionedCallбconv_4/StatefulPartitionedCallбconv_5/StatefulPartitionedCallбconv_6/StatefulPartitionedCallб$output_layer/StatefulPartitionedCallЊ
conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_layerconv_1_10054conv_1_10056*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_95102 
conv_1/StatefulPartitionedCall»
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_10059conv_2_10061*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_95262 
conv_2/StatefulPartitionedCall■
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_1_layer_call_and_return_conditional_losses_94632
maxpool_1/PartitionedCallф
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_10065conv_3_10067*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_95432 
conv_3/StatefulPartitionedCall»
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_10070conv_4_10072*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_4_layer_call_and_return_conditional_losses_95592 
conv_4/StatefulPartitionedCall■
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_2_layer_call_and_return_conditional_losses_94752
maxpool_2/PartitionedCallф
conv_5/StatefulPartitionedCallStatefulPartitionedCall"maxpool_2/PartitionedCall:output:0conv_5_10076conv_5_10078*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_5_layer_call_and_return_conditional_losses_95762 
conv_5/StatefulPartitionedCall»
conv_6/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0conv_6_10081conv_6_10083*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_6_layer_call_and_return_conditional_losses_95922 
conv_6/StatefulPartitionedCall■
maxpool_3/PartitionedCallPartitionedCall'conv_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_3_layer_call_and_return_conditional_losses_94872
maxpool_3/PartitionedCall■
flatten_layer/PartitionedCallPartitionedCall"maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_flatten_layer_layer_call_and_return_conditional_losses_96052
flatten_layer/PartitionedCallю
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0
fc_1_10088
fc_1_10090*
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_1_layer_call_and_return_conditional_losses_96172
FC_1/StatefulPartitionedCall§
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_96282
leaky_ReLu_1/PartitionedCallЏ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0
fc_2_10094
fc_2_10096*
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_2_layer_call_and_return_conditional_losses_96402
FC_2/StatefulPartitionedCall§
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_96512
leaky_ReLu_2/PartitionedCall├
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_10100output_layer_10102*
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
GPU 2J 8ѓ *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_96642&
$output_layer/StatefulPartitionedCallг
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 2<
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
:         dd
%
_user_specified_nameinput_layer
Г

щ
@__inference_conv_3_layer_call_and_return_conditional_losses_9543

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
:         ..*
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
:         ..2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         ..2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         00
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         00

 
_user_specified_nameinputs
«

Щ
A__inference_conv_2_layer_call_and_return_conditional_losses_10400

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
:         ``
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
:         ``
2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         ``
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         bb
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         bb

 
_user_specified_nameinputs
Щ
c
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_10549

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
«

Щ
A__inference_conv_1_layer_call_and_return_conditional_losses_10381

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         bb
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
:         bb
2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         bb
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
ѓ~
╔
__inference__traced_save_10789
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
ShardedFilenameЏ$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*Г#
valueБ#Bа#AB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/weight_decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЇ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*Ќ
valueЇBіAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesп
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv_1_kernel_read_readvariableop&savev2_conv_1_bias_read_readvariableop(savev2_conv_2_kernel_read_readvariableop&savev2_conv_2_bias_read_readvariableop(savev2_conv_3_kernel_read_readvariableop&savev2_conv_3_bias_read_readvariableop(savev2_conv_4_kernel_read_readvariableop&savev2_conv_4_bias_read_readvariableop(savev2_conv_5_kernel_read_readvariableop&savev2_conv_5_bias_read_readvariableop(savev2_conv_6_kernel_read_readvariableop&savev2_conv_6_bias_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop&savev2_fc_2_kernel_read_readvariableop$savev2_fc_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop%savev2_adamw_iter_read_readvariableop'savev2_adamw_beta_1_read_readvariableop'savev2_adamw_beta_2_read_readvariableop&savev2_adamw_decay_read_readvariableop.savev2_adamw_learning_rate_read_readvariableop-savev2_adamw_weight_decay_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adamw_conv_1_kernel_m_read_readvariableop.savev2_adamw_conv_1_bias_m_read_readvariableop0savev2_adamw_conv_2_kernel_m_read_readvariableop.savev2_adamw_conv_2_bias_m_read_readvariableop0savev2_adamw_conv_3_kernel_m_read_readvariableop.savev2_adamw_conv_3_bias_m_read_readvariableop0savev2_adamw_conv_4_kernel_m_read_readvariableop.savev2_adamw_conv_4_bias_m_read_readvariableop0savev2_adamw_conv_5_kernel_m_read_readvariableop.savev2_adamw_conv_5_bias_m_read_readvariableop0savev2_adamw_conv_6_kernel_m_read_readvariableop.savev2_adamw_conv_6_bias_m_read_readvariableop.savev2_adamw_fc_1_kernel_m_read_readvariableop,savev2_adamw_fc_1_bias_m_read_readvariableop.savev2_adamw_fc_2_kernel_m_read_readvariableop,savev2_adamw_fc_2_bias_m_read_readvariableop6savev2_adamw_output_layer_kernel_m_read_readvariableop4savev2_adamw_output_layer_bias_m_read_readvariableop0savev2_adamw_conv_1_kernel_v_read_readvariableop.savev2_adamw_conv_1_bias_v_read_readvariableop0savev2_adamw_conv_2_kernel_v_read_readvariableop.savev2_adamw_conv_2_bias_v_read_readvariableop0savev2_adamw_conv_3_kernel_v_read_readvariableop.savev2_adamw_conv_3_bias_v_read_readvariableop0savev2_adamw_conv_4_kernel_v_read_readvariableop.savev2_adamw_conv_4_bias_v_read_readvariableop0savev2_adamw_conv_5_kernel_v_read_readvariableop.savev2_adamw_conv_5_bias_v_read_readvariableop0savev2_adamw_conv_6_kernel_v_read_readvariableop.savev2_adamw_conv_6_bias_v_read_readvariableop.savev2_adamw_fc_1_kernel_v_read_readvariableop,savev2_adamw_fc_1_bias_v_read_readvariableop.savev2_adamw_fc_2_kernel_v_read_readvariableop,savev2_adamw_fc_2_bias_v_read_readvariableop6savev2_adamw_output_layer_kernel_v_read_readvariableop4savev2_adamw_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *O
dtypesE
C2A	2
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

identity_1Identity_1:output:0*­
_input_shapesя
█: :
:
:

:
:
::::::::	љ
2:2:22:2:2:: : : : : : : : : : :
:
:

:
:
::::::::	љ
2:2:22:2:2::
:
:

:
:
::::::::	љ
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
:	љ
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
:	љ
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
:	љ
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
«

Щ
A__inference_conv_6_layer_call_and_return_conditional_losses_10476

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
:         *
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
:         2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┴
H
,__inference_leaky_ReLu_2_layer_call_fn_10554

inputs
identity─
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_96512
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
«

Щ
A__inference_conv_3_layer_call_and_return_conditional_losses_10419

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
:         ..*
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
:         ..2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         ..2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         00
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         00

 
_user_specified_nameinputs
┬X
к
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10223

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
#fc_1_matmul_readvariableop_resource:	љ
22
$fc_1_biasadd_readvariableop_resource:25
#fc_2_matmul_readvariableop_resource:222
$fc_2_biasadd_readvariableop_resource:2=
+output_layer_matmul_readvariableop_resource:2:
,output_layer_biasadd_readvariableop_resource:
identityѕбFC_1/BiasAdd/ReadVariableOpбFC_1/MatMul/ReadVariableOpбFC_2/BiasAdd/ReadVariableOpбFC_2/MatMul/ReadVariableOpбconv_1/BiasAdd/ReadVariableOpбconv_1/Conv2D/ReadVariableOpбconv_2/BiasAdd/ReadVariableOpбconv_2/Conv2D/ReadVariableOpбconv_3/BiasAdd/ReadVariableOpбconv_3/Conv2D/ReadVariableOpбconv_4/BiasAdd/ReadVariableOpбconv_4/Conv2D/ReadVariableOpбconv_5/BiasAdd/ReadVariableOpбconv_5/Conv2D/ReadVariableOpбconv_6/BiasAdd/ReadVariableOpбconv_6/Conv2D/ReadVariableOpб#output_layer/BiasAdd/ReadVariableOpб"output_layer/MatMul/ReadVariableOpф
conv_1/Conv2D/ReadVariableOpReadVariableOp%conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_1/Conv2D/ReadVariableOp╣
conv_1/Conv2DConv2Dinputs$conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         bb
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
:         bb
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
:         ``
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
:         ``
2
conv_2/BiasAddи
maxpool_1/MaxPoolMaxPoolconv_2/BiasAdd:output:0*/
_output_shapes
:         00
*
ksize
*
paddingVALID*
strides
2
maxpool_1/MaxPoolф
conv_3/Conv2D/ReadVariableOpReadVariableOp%conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
conv_3/Conv2D/ReadVariableOp═
conv_3/Conv2DConv2Dmaxpool_1/MaxPool:output:0$conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ..*
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
:         ..2
conv_3/BiasAddф
conv_4/Conv2D/ReadVariableOpReadVariableOp%conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_4/Conv2D/ReadVariableOp╩
conv_4/Conv2DConv2Dconv_3/BiasAdd:output:0$conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ,,*
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
:         ,,2
conv_4/BiasAddи
maxpool_2/MaxPoolMaxPoolconv_4/BiasAdd:output:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2
maxpool_2/MaxPoolф
conv_5/Conv2D/ReadVariableOpReadVariableOp%conv_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_5/Conv2D/ReadVariableOp═
conv_5/Conv2DConv2Dmaxpool_2/MaxPool:output:0$conv_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv_5/Conv2DА
conv_5/BiasAdd/ReadVariableOpReadVariableOp&conv_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_5/BiasAdd/ReadVariableOpц
conv_5/BiasAddBiasAddconv_5/Conv2D:output:0%conv_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv_5/BiasAddф
conv_6/Conv2D/ReadVariableOpReadVariableOp%conv_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv_6/Conv2D/ReadVariableOp╩
conv_6/Conv2DConv2Dconv_5/BiasAdd:output:0$conv_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv_6/Conv2DА
conv_6/BiasAdd/ReadVariableOpReadVariableOp&conv_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv_6/BiasAdd/ReadVariableOpц
conv_6/BiasAddBiasAddconv_6/Conv2D:output:0%conv_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv_6/BiasAddи
maxpool_3/MaxPoolMaxPoolconv_6/BiasAdd:output:0*/
_output_shapes
:         		*
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
valueB"      2
flatten_layer/Constд
flatten_layer/ReshapeReshapemaxpool_3/MaxPool:output:0flatten_layer/Const:output:0*
T0*(
_output_shapes
:         љ
2
flatten_layer/ReshapeЮ
FC_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes
:	љ
2*
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
output_layer/SoftmaxГ
IdentityIdentityoutput_layer/Softmax:softmax:0^FC_1/BiasAdd/ReadVariableOp^FC_1/MatMul/ReadVariableOp^FC_2/BiasAdd/ReadVariableOp^FC_2/MatMul/ReadVariableOp^conv_1/BiasAdd/ReadVariableOp^conv_1/Conv2D/ReadVariableOp^conv_2/BiasAdd/ReadVariableOp^conv_2/Conv2D/ReadVariableOp^conv_3/BiasAdd/ReadVariableOp^conv_3/Conv2D/ReadVariableOp^conv_4/BiasAdd/ReadVariableOp^conv_4/Conv2D/ReadVariableOp^conv_5/BiasAdd/ReadVariableOp^conv_5/Conv2D/ReadVariableOp^conv_6/BiasAdd/ReadVariableOp^conv_6/Conv2D/ReadVariableOp$^output_layer/BiasAdd/ReadVariableOp#^output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 2:
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
:         dd
 
_user_specified_nameinputs
Г

щ
@__inference_conv_6_layer_call_and_return_conditional_losses_9592

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
:         *
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
:         2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
╠A
п
O__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_9671

inputs%
conv_1_9511:

conv_1_9513:
%
conv_2_9527:


conv_2_9529:
%
conv_3_9544:

conv_3_9546:%
conv_4_9560:
conv_4_9562:%
conv_5_9577:
conv_5_9579:%
conv_6_9593:
conv_6_9595:
	fc_1_9618:	љ
2
	fc_1_9620:2
	fc_2_9641:22
	fc_2_9643:2#
output_layer_9665:2
output_layer_9667:
identityѕбFC_1/StatefulPartitionedCallбFC_2/StatefulPartitionedCallбconv_1/StatefulPartitionedCallбconv_2/StatefulPartitionedCallбconv_3/StatefulPartitionedCallбconv_4/StatefulPartitionedCallбconv_5/StatefulPartitionedCallбconv_6/StatefulPartitionedCallб$output_layer/StatefulPartitionedCallї
conv_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv_1_9511conv_1_9513*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         bb
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_1_layer_call_and_return_conditional_losses_95102 
conv_1/StatefulPartitionedCallГ
conv_2/StatefulPartitionedCallStatefulPartitionedCall'conv_1/StatefulPartitionedCall:output:0conv_2_9527conv_2_9529*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ``
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_2_layer_call_and_return_conditional_losses_95262 
conv_2/StatefulPartitionedCall■
maxpool_1/PartitionedCallPartitionedCall'conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         00
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_1_layer_call_and_return_conditional_losses_94632
maxpool_1/PartitionedCallе
conv_3/StatefulPartitionedCallStatefulPartitionedCall"maxpool_1/PartitionedCall:output:0conv_3_9544conv_3_9546*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_95432 
conv_3/StatefulPartitionedCallГ
conv_4/StatefulPartitionedCallStatefulPartitionedCall'conv_3/StatefulPartitionedCall:output:0conv_4_9560conv_4_9562*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ,,*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_4_layer_call_and_return_conditional_losses_95592 
conv_4/StatefulPartitionedCall■
maxpool_2/PartitionedCallPartitionedCall'conv_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_2_layer_call_and_return_conditional_losses_94752
maxpool_2/PartitionedCallе
conv_5/StatefulPartitionedCallStatefulPartitionedCall"maxpool_2/PartitionedCall:output:0conv_5_9577conv_5_9579*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_5_layer_call_and_return_conditional_losses_95762 
conv_5/StatefulPartitionedCallГ
conv_6/StatefulPartitionedCallStatefulPartitionedCall'conv_5/StatefulPartitionedCall:output:0conv_6_9593conv_6_9595*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_6_layer_call_and_return_conditional_losses_95922 
conv_6/StatefulPartitionedCall■
maxpool_3/PartitionedCallPartitionedCall'conv_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         		* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_3_layer_call_and_return_conditional_losses_94872
maxpool_3/PartitionedCall■
flatten_layer/PartitionedCallPartitionedCall"maxpool_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_flatten_layer_layer_call_and_return_conditional_losses_96052
flatten_layer/PartitionedCallџ
FC_1/StatefulPartitionedCallStatefulPartitionedCall&flatten_layer/PartitionedCall:output:0	fc_1_9618	fc_1_9620*
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_1_layer_call_and_return_conditional_losses_96172
FC_1/StatefulPartitionedCall§
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_96282
leaky_ReLu_1/PartitionedCallЎ
FC_2/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_1/PartitionedCall:output:0	fc_2_9641	fc_2_9643*
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_2_layer_call_and_return_conditional_losses_96402
FC_2/StatefulPartitionedCall§
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
GPU 2J 8ѓ *O
fJRH
F__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_96512
leaky_ReLu_2/PartitionedCall┴
$output_layer/StatefulPartitionedCallStatefulPartitionedCall%leaky_ReLu_2/PartitionedCall:output:0output_layer_9665output_layer_9667*
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
GPU 2J 8ѓ *O
fJRH
F__inference_output_layer_layer_call_and_return_conditional_losses_96642&
$output_layer/StatefulPartitionedCallг
IdentityIdentity-output_layer/StatefulPartitionedCall:output:0^FC_1/StatefulPartitionedCall^FC_2/StatefulPartitionedCall^conv_1/StatefulPartitionedCall^conv_2/StatefulPartitionedCall^conv_3/StatefulPartitionedCall^conv_4/StatefulPartitionedCall^conv_5/StatefulPartitionedCall^conv_6/StatefulPartitionedCall%^output_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 2<
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
:         dd
 
_user_specified_nameinputs
Њ
њ
$__inference_FC_1_layer_call_fn_10515

inputs
unknown:	љ
2
	unknown_0:2
identityѕбStatefulPartitionedCallЬ
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_1_layer_call_and_return_conditional_losses_96172
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         22

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         љ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         љ

 
_user_specified_nameinputs
╩	
№
>__inference_FC_2_layer_call_and_return_conditional_losses_9640

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
Н
I
-__inference_flatten_layer_layer_call_fn_10496

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         љ
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *P
fKRI
G__inference_flatten_layer_layer_call_and_return_conditional_losses_96052
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         љ
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         		:W S
/
_output_shapes
:         		
 
_user_specified_nameinputs
╬	
­
>__inference_FC_1_layer_call_and_return_conditional_losses_9617

inputs1
matmul_readvariableop_resource:	љ
2-
biasadd_readvariableop_resource:2
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	љ
2*
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
_construction_contextkEagerRuntime*+
_input_shapes
:         љ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         љ

 
_user_specified_nameinputs
╝
Џ
&__inference_conv_3_layer_call_fn_10428

inputs!
unknown:

	unknown_0:
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         ..*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_3_layer_call_and_return_conditional_losses_95432
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         ..2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         00
: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         00

 
_user_specified_nameinputs
љ
Љ
$__inference_FC_2_layer_call_fn_10544

inputs
unknown:22
	unknown_0:2
identityѕбStatefulPartitionedCallЬ
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
GPU 2J 8ѓ *G
fBR@
>__inference_FC_2_layer_call_and_return_conditional_losses_96402
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
Г

щ
@__inference_conv_1_layer_call_and_return_conditional_losses_9510

inputs8
conv2d_readvariableop_resource:
-
biasadd_readvariableop_resource:

identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         bb
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
:         bb
2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         bb
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         dd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         dd
 
_user_specified_nameinputs
Б
_
C__inference_maxpool_3_layer_call_and_return_conditional_losses_9487

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Г

щ
@__inference_conv_4_layer_call_and_return_conditional_losses_9559

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
:         ,,*
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
:         ,,2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         ,,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ..: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         ..
 
_user_specified_nameinputs
╝
Џ
&__inference_conv_6_layer_call_fn_10485

inputs!
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_conv_6_layer_call_and_return_conditional_losses_95922
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Б
_
C__inference_maxpool_1_layer_call_and_return_conditional_losses_9463

inputs
identityГ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЄ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
«

Щ
A__inference_conv_5_layer_call_and_return_conditional_losses_10457

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
:         *
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
:         2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
┴Ђ
▓
__inference__wrapped_model_9457
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
9wheatclassifier_cnn_2_fc_1_matmul_readvariableop_resource:	љ
2H
:wheatclassifier_cnn_2_fc_1_biasadd_readvariableop_resource:2K
9wheatclassifier_cnn_2_fc_2_matmul_readvariableop_resource:22H
:wheatclassifier_cnn_2_fc_2_biasadd_readvariableop_resource:2S
Awheatclassifier_cnn_2_output_layer_matmul_readvariableop_resource:2P
Bwheatclassifier_cnn_2_output_layer_biasadd_readvariableop_resource:
identityѕб1WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOpб0WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOpб1WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOpб0WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOpб3WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOpб3WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOpб3WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOpб3WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOpб3WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOpб3WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOpб2WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOpб9WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOpб8WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOpВ
2WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype024
2WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOpђ
#WheatClassifier_CNN_2/conv_1/Conv2DConv2Dinput_layer:WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         bb
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_1/Conv2Dс
3WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_2/conv_1/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_1/Conv2D:output:0;WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         bb
2&
$WheatClassifier_CNN_2/conv_1/BiasAddВ
2WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:

*
dtype024
2WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOpб
#WheatClassifier_CNN_2/conv_2/Conv2DConv2D-WheatClassifier_CNN_2/conv_1/BiasAdd:output:0:WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ``
*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_2/Conv2Dс
3WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype025
3WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_2/conv_2/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_2/Conv2D:output:0;WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ``
2&
$WheatClassifier_CNN_2/conv_2/BiasAddщ
'WheatClassifier_CNN_2/maxpool_1/MaxPoolMaxPool-WheatClassifier_CNN_2/conv_2/BiasAdd:output:0*/
_output_shapes
:         00
*
ksize
*
paddingVALID*
strides
2)
'WheatClassifier_CNN_2/maxpool_1/MaxPoolВ
2WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_3_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype024
2WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOpЦ
#WheatClassifier_CNN_2/conv_3/Conv2DConv2D0WheatClassifier_CNN_2/maxpool_1/MaxPool:output:0:WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ..*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_3/Conv2Dс
3WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_2/conv_3/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_3/Conv2D:output:0;WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ..2&
$WheatClassifier_CNN_2/conv_3/BiasAddВ
2WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOpб
#WheatClassifier_CNN_2/conv_4/Conv2DConv2D-WheatClassifier_CNN_2/conv_3/BiasAdd:output:0:WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ,,*
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_4/Conv2Dс
3WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_2/conv_4/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_4/Conv2D:output:0;WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ,,2&
$WheatClassifier_CNN_2/conv_4/BiasAddщ
'WheatClassifier_CNN_2/maxpool_2/MaxPoolMaxPool-WheatClassifier_CNN_2/conv_4/BiasAdd:output:0*/
_output_shapes
:         *
ksize
*
paddingVALID*
strides
2)
'WheatClassifier_CNN_2/maxpool_2/MaxPoolВ
2WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOpЦ
#WheatClassifier_CNN_2/conv_5/Conv2DConv2D0WheatClassifier_CNN_2/maxpool_2/MaxPool:output:0:WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_5/Conv2Dс
3WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_2/conv_5/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_5/Conv2D:output:0;WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2&
$WheatClassifier_CNN_2/conv_5/BiasAddВ
2WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOpReadVariableOp;wheatclassifier_cnn_2_conv_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOpб
#WheatClassifier_CNN_2/conv_6/Conv2DConv2D-WheatClassifier_CNN_2/conv_5/BiasAdd:output:0:WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2%
#WheatClassifier_CNN_2/conv_6/Conv2Dс
3WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOpReadVariableOp<wheatclassifier_cnn_2_conv_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOpЧ
$WheatClassifier_CNN_2/conv_6/BiasAddBiasAdd,WheatClassifier_CNN_2/conv_6/Conv2D:output:0;WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2&
$WheatClassifier_CNN_2/conv_6/BiasAddщ
'WheatClassifier_CNN_2/maxpool_3/MaxPoolMaxPool-WheatClassifier_CNN_2/conv_6/BiasAdd:output:0*/
_output_shapes
:         		*
ksize
*
paddingVALID*
strides
2)
'WheatClassifier_CNN_2/maxpool_3/MaxPoolД
)WheatClassifier_CNN_2/flatten_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2+
)WheatClassifier_CNN_2/flatten_layer/Const■
+WheatClassifier_CNN_2/flatten_layer/ReshapeReshape0WheatClassifier_CNN_2/maxpool_3/MaxPool:output:02WheatClassifier_CNN_2/flatten_layer/Const:output:0*
T0*(
_output_shapes
:         љ
2-
+WheatClassifier_CNN_2/flatten_layer/Reshape▀
0WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_2_fc_1_matmul_readvariableop_resource*
_output_shapes
:	љ
2*
dtype022
0WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOpЫ
!WheatClassifier_CNN_2/FC_1/MatMulMatMul4WheatClassifier_CNN_2/flatten_layer/Reshape:output:08WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22#
!WheatClassifier_CNN_2/FC_1/MatMulП
1WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_2_fc_1_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOpь
"WheatClassifier_CNN_2/FC_1/BiasAddBiasAdd+WheatClassifier_CNN_2/FC_1/MatMul:product:09WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22$
"WheatClassifier_CNN_2/FC_1/BiasAdd¤
,WheatClassifier_CNN_2/leaky_ReLu_1/LeakyRelu	LeakyRelu+WheatClassifier_CNN_2/FC_1/BiasAdd:output:0*'
_output_shapes
:         2*
alpha%џЎЎ>2.
,WheatClassifier_CNN_2/leaky_ReLu_1/LeakyReluя
0WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOpReadVariableOp9wheatclassifier_cnn_2_fc_2_matmul_readvariableop_resource*
_output_shapes

:22*
dtype022
0WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOpЭ
!WheatClassifier_CNN_2/FC_2/MatMulMatMul:WheatClassifier_CNN_2/leaky_ReLu_1/LeakyRelu:activations:08WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22#
!WheatClassifier_CNN_2/FC_2/MatMulП
1WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOpReadVariableOp:wheatclassifier_cnn_2_fc_2_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype023
1WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOpь
"WheatClassifier_CNN_2/FC_2/BiasAddBiasAdd+WheatClassifier_CNN_2/FC_2/MatMul:product:09WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         22$
"WheatClassifier_CNN_2/FC_2/BiasAdd¤
,WheatClassifier_CNN_2/leaky_ReLu_2/LeakyRelu	LeakyRelu+WheatClassifier_CNN_2/FC_2/BiasAdd:output:0*'
_output_shapes
:         2*
alpha%џЎЎ>2.
,WheatClassifier_CNN_2/leaky_ReLu_2/LeakyReluШ
8WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOpReadVariableOpAwheatclassifier_cnn_2_output_layer_matmul_readvariableop_resource*
_output_shapes

:2*
dtype02:
8WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOpљ
)WheatClassifier_CNN_2/output_layer/MatMulMatMul:WheatClassifier_CNN_2/leaky_ReLu_2/LeakyRelu:activations:0@WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2+
)WheatClassifier_CNN_2/output_layer/MatMulш
9WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOpReadVariableOpBwheatclassifier_cnn_2_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOpЇ
*WheatClassifier_CNN_2/output_layer/BiasAddBiasAdd3WheatClassifier_CNN_2/output_layer/MatMul:product:0AWheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2,
*WheatClassifier_CNN_2/output_layer/BiasAdd╩
*WheatClassifier_CNN_2/output_layer/SoftmaxSoftmax3WheatClassifier_CNN_2/output_layer/BiasAdd:output:0*
T0*'
_output_shapes
:         2,
*WheatClassifier_CNN_2/output_layer/Softmax¤
IdentityIdentity4WheatClassifier_CNN_2/output_layer/Softmax:softmax:02^WheatClassifier_CNN_2/FC_1/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_2/FC_1/MatMul/ReadVariableOp2^WheatClassifier_CNN_2/FC_2/BiasAdd/ReadVariableOp1^WheatClassifier_CNN_2/FC_2/MatMul/ReadVariableOp4^WheatClassifier_CNN_2/conv_1/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_1/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_2/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_2/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_3/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_3/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_4/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_4/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_5/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_5/Conv2D/ReadVariableOp4^WheatClassifier_CNN_2/conv_6/BiasAdd/ReadVariableOp3^WheatClassifier_CNN_2/conv_6/Conv2D/ReadVariableOp:^WheatClassifier_CNN_2/output_layer/BiasAdd/ReadVariableOp9^WheatClassifier_CNN_2/output_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         dd: : : : : : : : : : : : : : : : : : 2f
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
:         dd
%
_user_specified_nameinput_layer
К
D
(__inference_maxpool_1_layer_call_fn_9469

inputs
identityС
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_maxpool_1_layer_call_and_return_conditional_losses_94632
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
«

Щ
A__inference_conv_4_layer_call_and_return_conditional_losses_10438

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
:         ,,*
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
:         ,,2	
BiasAddЮ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         ,,2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         ..: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         ..
 
_user_specified_nameinputs"╠L
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
StatefulPartitionedCall:0         tensorflow/serving/predict:рш
░Љ
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
Ж_default_save_signature
+в&call_and_return_all_conditional_losses
В__call__"Аї
_tf_keras_networkёї{"name": "WheatClassifier_CNN_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "WheatClassifier_CNN_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["conv_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["maxpool_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["conv_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_2", "inbound_nodes": [[["conv_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["maxpool_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_6", "inbound_nodes": [[["conv_5", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_3", "inbound_nodes": [[["conv_6", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_layer", "inbound_nodes": [[["maxpool_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1", "inbound_nodes": [[["flatten_layer", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_1", "inbound_nodes": [[["FC_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_2", "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_2", "inbound_nodes": [[["FC_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}, "shared_object_id": 34, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 100, 100, 3]}, "float32", "input_layer"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "WheatClassifier_CNN_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "name": "input_layer", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_1", "inbound_nodes": [[["input_layer", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_2", "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_1", "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_3", "inbound_nodes": [[["maxpool_1", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_4", "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_2", "inbound_nodes": [[["conv_4", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_5", "inbound_nodes": [[["maxpool_2", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv_6", "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 20}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "maxpool_3", "inbound_nodes": [[["conv_6", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_layer", "inbound_nodes": [[["maxpool_3", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1", "inbound_nodes": [[["flatten_layer", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_1", "inbound_nodes": [[["FC_1", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_2", "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]], "shared_object_id": 29}, {"class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "name": "leaky_ReLu_2", "inbound_nodes": [[["FC_2", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output_layer", "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]], "shared_object_id": 33}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["output_layer", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 36}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Addons>AdamW", "config": {"name": "AdamW", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false, "weight_decay": 9.999999747378752e-05}}}}
Ё"ѓ
_tf_keras_input_layerР{"class_name": "InputLayer", "name": "input_layer", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 100, 100, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}}
ђ

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+ь&call_and_return_all_conditional_losses
Ь__call__"┘	
_tf_keras_layer┐	{"name": "conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_layer", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 100, 100, 3]}}
ч


kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
+№&call_and_return_all_conditional_losses
­__call__"н	
_tf_keras_layer║	{"name": "conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_2", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_1", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 98, 98, 10]}}
¤
#regularization_losses
$	variables
%trainable_variables
&	keras_api
+ы&call_and_return_all_conditional_losses
Ы__call__"Й
_tf_keras_layerц{"name": "maxpool_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_2", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 39}}
 


'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
+з&call_and_return_all_conditional_losses
З__call__"п	
_tf_keras_layerЙ	{"name": "conv_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["maxpool_1", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 10}}, "shared_object_id": 40}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 10]}}
■


-kernel
.bias
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+ш&call_and_return_all_conditional_losses
Ш__call__"О	
_tf_keras_layerй	{"name": "conv_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_3", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 46, 46, 16]}}
л
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+э&call_and_return_all_conditional_losses
Э__call__"┐
_tf_keras_layerЦ{"name": "maxpool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_4", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 42}}
Ђ

7kernel
8bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+щ&call_and_return_all_conditional_losses
Щ__call__"┌	
_tf_keras_layer└	{"name": "conv_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_5", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["maxpool_2", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 22, 16]}}
■


=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
+ч&call_and_return_all_conditional_losses
Ч__call__"О	
_tf_keras_layerй	{"name": "conv_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv_6", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv_5", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 20, 16]}}
л
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
+§&call_and_return_all_conditional_losses
■__call__"┐
_tf_keras_layerЦ{"name": "maxpool_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "maxpool_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv_6", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
╬
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+ &call_and_return_all_conditional_losses
ђ__call__"й
_tf_keras_layerБ{"name": "flatten_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["maxpool_3", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 46}}
ё	

Kkernel
Lbias
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
+Ђ&call_and_return_all_conditional_losses
ѓ__call__"П
_tf_keras_layer├{"name": "FC_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_1", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_layer", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1296}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1296]}}
Ъ
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
+Ѓ&call_and_return_all_conditional_losses
ё__call__"ј
_tf_keras_layerЗ{"name": "leaky_ReLu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_1", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_1", 0, 0, {}]]], "shared_object_id": 26}
 

Ukernel
Vbias
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
+Ё&call_and_return_all_conditional_losses
є__call__"п
_tf_keras_layerЙ{"name": "FC_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "FC_2", "trainable": true, "dtype": "float32", "units": 50, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 27}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 28}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_1", 0, 0, {}]]], "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
Ъ
[regularization_losses
\	variables
]trainable_variables
^	keras_api
+Є&call_and_return_all_conditional_losses
ѕ__call__"ј
_tf_keras_layerЗ{"name": "leaky_ReLu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_ReLu_2", "trainable": true, "dtype": "float32", "alpha": 0.30000001192092896}, "inbound_nodes": [[["FC_2", 0, 0, {}]]], "shared_object_id": 30}
Ј	

_kernel
`bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+Ѕ&call_and_return_all_conditional_losses
і__call__"У
_tf_keras_layer╬{"name": "output_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output_layer", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["leaky_ReLu_2", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
═
eiter

fbeta_1

gbeta_2
	hdecay
ilearning_rate
jweight_decaymкmКm╚m╔'m╩(m╦-m╠.m═7m╬8m¤=mл>mЛKmмLmМUmнVmН_mо`mОvпv┘v┌v█'v▄(vП-vя.v▀7vЯ8vр=vР>vсKvСLvтUvТVvу_vУ`vж"
	optimizer
 "
trackable_list_wrapper
д
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
д
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
╬
regularization_losses
knon_trainable_variables
llayer_metrics
mlayer_regularization_losses
	variables

nlayers
trainable_variables
ometrics
В__call__
Ж_default_save_signature
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
-
Іserving_default"
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
░
regularization_losses
pnon_trainable_variables
qlayer_metrics
rlayer_regularization_losses
	variables

slayers
trainable_variables
tmetrics
Ь__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
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
░
regularization_losses
unon_trainable_variables
vlayer_metrics
wlayer_regularization_losses
 	variables

xlayers
!trainable_variables
ymetrics
­__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
#regularization_losses
znon_trainable_variables
{layer_metrics
|layer_regularization_losses
$	variables

}layers
%trainable_variables
~metrics
Ы__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
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
┤
)regularization_losses
non_trainable_variables
ђlayer_metrics
 Ђlayer_regularization_losses
*	variables
ѓlayers
+trainable_variables
Ѓmetrics
З__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
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
х
/regularization_losses
ёnon_trainable_variables
Ёlayer_metrics
 єlayer_regularization_losses
0	variables
Єlayers
1trainable_variables
ѕmetrics
Ш__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
3regularization_losses
Ѕnon_trainable_variables
іlayer_metrics
 Іlayer_regularization_losses
4	variables
їlayers
5trainable_variables
Їmetrics
Э__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
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
х
9regularization_losses
јnon_trainable_variables
Јlayer_metrics
 љlayer_regularization_losses
:	variables
Љlayers
;trainable_variables
њmetrics
Щ__call__
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
х
?regularization_losses
Њnon_trainable_variables
ћlayer_metrics
 Ћlayer_regularization_losses
@	variables
ќlayers
Atrainable_variables
Ќmetrics
Ч__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Cregularization_losses
ўnon_trainable_variables
Ўlayer_metrics
 џlayer_regularization_losses
D	variables
Џlayers
Etrainable_variables
юmetrics
■__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Gregularization_losses
Юnon_trainable_variables
ъlayer_metrics
 Ъlayer_regularization_losses
H	variables
аlayers
Itrainable_variables
Аmetrics
ђ__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
:	љ
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
х
Mregularization_losses
бnon_trainable_variables
Бlayer_metrics
 цlayer_regularization_losses
N	variables
Цlayers
Otrainable_variables
дmetrics
ѓ__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Qregularization_losses
Дnon_trainable_variables
еlayer_metrics
 Еlayer_regularization_losses
R	variables
фlayers
Strainable_variables
Фmetrics
ё__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
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
х
Wregularization_losses
гnon_trainable_variables
Гlayer_metrics
 «layer_regularization_losses
X	variables
»layers
Ytrainable_variables
░metrics
є__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
[regularization_losses
▒non_trainable_variables
▓layer_metrics
 │layer_regularization_losses
\	variables
┤layers
]trainable_variables
хmetrics
ѕ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
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
х
aregularization_losses
Хnon_trainable_variables
иlayer_metrics
 Иlayer_regularization_losses
b	variables
╣layers
ctrainable_variables
║metrics
і__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
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
ќ
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
╗0
╝1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
п

йtotal

Йcount
┐	variables
└	keras_api"Ю
_tf_keras_metricѓ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 50}
њ

┴total

┬count
├
_fn_kwargs
─	variables
┼	keras_api"к
_tf_keras_metricФ{"class_name": "MeanMetricWrapper", "name": "acc", "dtype": "float32", "config": {"name": "acc", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 36}
:  (2total
:  (2count
0
й0
Й1"
trackable_list_wrapper
.
┐	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
┴0
┬1"
trackable_list_wrapper
.
─	variables"
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
$:"	љ
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
$:"	љ
22AdamW/FC_1/kernel/v
:22AdamW/FC_1/bias/v
#:!222AdamW/FC_2/kernel/v
:22AdamW/FC_2/bias/v
+:)22AdamW/output_layer/kernel/v
%:#2AdamW/output_layer/bias/v
ж2Т
__inference__wrapped_model_9457┬
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
ј2І
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10223
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10289
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10051
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10106└
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
а2Ю
4__inference_WheatClassifier_CNN_2_layer_call_fn_9710
5__inference_WheatClassifier_CNN_2_layer_call_fn_10330
5__inference_WheatClassifier_CNN_2_layer_call_fn_10371
4__inference_WheatClassifier_CNN_2_layer_call_fn_9996└
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
A__inference_conv_1_layer_call_and_return_conditional_losses_10381б
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
&__inference_conv_1_layer_call_fn_10390б
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
A__inference_conv_2_layer_call_and_return_conditional_losses_10400б
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
&__inference_conv_2_layer_call_fn_10409б
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
Ф2е
C__inference_maxpool_1_layer_call_and_return_conditional_losses_9463Я
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
annotationsф *@б=
;і84                                    
љ2Ї
(__inference_maxpool_1_layer_call_fn_9469Я
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
annotationsф *@б=
;і84                                    
в2У
A__inference_conv_3_layer_call_and_return_conditional_losses_10419б
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
&__inference_conv_3_layer_call_fn_10428б
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
A__inference_conv_4_layer_call_and_return_conditional_losses_10438б
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
&__inference_conv_4_layer_call_fn_10447б
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
Ф2е
C__inference_maxpool_2_layer_call_and_return_conditional_losses_9475Я
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
annotationsф *@б=
;і84                                    
љ2Ї
(__inference_maxpool_2_layer_call_fn_9481Я
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
annotationsф *@б=
;і84                                    
в2У
A__inference_conv_5_layer_call_and_return_conditional_losses_10457б
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
&__inference_conv_5_layer_call_fn_10466б
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
A__inference_conv_6_layer_call_and_return_conditional_losses_10476б
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
&__inference_conv_6_layer_call_fn_10485б
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
Ф2е
C__inference_maxpool_3_layer_call_and_return_conditional_losses_9487Я
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
annotationsф *@б=
;і84                                    
љ2Ї
(__inference_maxpool_3_layer_call_fn_9493Я
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
annotationsф *@б=
;і84                                    
Ы2№
H__inference_flatten_layer_layer_call_and_return_conditional_losses_10491б
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
-__inference_flatten_layer_layer_call_fn_10496б
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
ж2Т
?__inference_FC_1_layer_call_and_return_conditional_losses_10506б
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
$__inference_FC_1_layer_call_fn_10515б
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
ы2Ь
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_10520б
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
о2М
,__inference_leaky_ReLu_1_layer_call_fn_10525б
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
ж2Т
?__inference_FC_2_layer_call_and_return_conditional_losses_10535б
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
$__inference_FC_2_layer_call_fn_10544б
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
ы2Ь
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_10549б
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
о2М
,__inference_leaky_ReLu_2_layer_call_fn_10554б
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
ы2Ь
G__inference_output_layer_layer_call_and_return_conditional_losses_10565б
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
о2М
,__inference_output_layer_layer_call_fn_10574б
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
╬B╦
#__inference_signature_wrapper_10157input_layer"ћ
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
 а
?__inference_FC_1_layer_call_and_return_conditional_losses_10506]KL0б-
&б#
!і
inputs         љ

ф "%б"
і
0         2
џ x
$__inference_FC_1_layer_call_fn_10515PKL0б-
&б#
!і
inputs         љ

ф "і         2Ъ
?__inference_FC_2_layer_call_and_return_conditional_losses_10535\UV/б,
%б"
 і
inputs         2
ф "%б"
і
0         2
џ w
$__inference_FC_2_layer_call_fn_10544OUV/б,
%б"
 і
inputs         2
ф "і         2о
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10051Ђ'(-.78=>KLUV_`DбA
:б7
-і*
input_layer         dd
p 

 
ф "%б"
і
0         
џ о
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10106Ђ'(-.78=>KLUV_`DбA
:б7
-і*
input_layer         dd
p

 
ф "%б"
і
0         
џ л
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10223|'(-.78=>KLUV_`?б<
5б2
(і%
inputs         dd
p 

 
ф "%б"
і
0         
џ л
P__inference_WheatClassifier_CNN_2_layer_call_and_return_conditional_losses_10289|'(-.78=>KLUV_`?б<
5б2
(і%
inputs         dd
p

 
ф "%б"
і
0         
џ е
5__inference_WheatClassifier_CNN_2_layer_call_fn_10330o'(-.78=>KLUV_`?б<
5б2
(і%
inputs         dd
p 

 
ф "і         е
5__inference_WheatClassifier_CNN_2_layer_call_fn_10371o'(-.78=>KLUV_`?б<
5б2
(і%
inputs         dd
p

 
ф "і         г
4__inference_WheatClassifier_CNN_2_layer_call_fn_9710t'(-.78=>KLUV_`DбA
:б7
-і*
input_layer         dd
p 

 
ф "і         г
4__inference_WheatClassifier_CNN_2_layer_call_fn_9996t'(-.78=>KLUV_`DбA
:б7
-і*
input_layer         dd
p

 
ф "і         │
__inference__wrapped_model_9457Ј'(-.78=>KLUV_`<б9
2б/
-і*
input_layer         dd
ф ";ф8
6
output_layer&і#
output_layer         ▒
A__inference_conv_1_layer_call_and_return_conditional_losses_10381l7б4
-б*
(і%
inputs         dd
ф "-б*
#і 
0         bb

џ Ѕ
&__inference_conv_1_layer_call_fn_10390_7б4
-б*
(і%
inputs         dd
ф " і         bb
▒
A__inference_conv_2_layer_call_and_return_conditional_losses_10400l7б4
-б*
(і%
inputs         bb

ф "-б*
#і 
0         ``

џ Ѕ
&__inference_conv_2_layer_call_fn_10409_7б4
-б*
(і%
inputs         bb

ф " і         ``
▒
A__inference_conv_3_layer_call_and_return_conditional_losses_10419l'(7б4
-б*
(і%
inputs         00

ф "-б*
#і 
0         ..
џ Ѕ
&__inference_conv_3_layer_call_fn_10428_'(7б4
-б*
(і%
inputs         00

ф " і         ..▒
A__inference_conv_4_layer_call_and_return_conditional_losses_10438l-.7б4
-б*
(і%
inputs         ..
ф "-б*
#і 
0         ,,
џ Ѕ
&__inference_conv_4_layer_call_fn_10447_-.7б4
-б*
(і%
inputs         ..
ф " і         ,,▒
A__inference_conv_5_layer_call_and_return_conditional_losses_10457l787б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ Ѕ
&__inference_conv_5_layer_call_fn_10466_787б4
-б*
(і%
inputs         
ф " і         ▒
A__inference_conv_6_layer_call_and_return_conditional_losses_10476l=>7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ Ѕ
&__inference_conv_6_layer_call_fn_10485_=>7б4
-б*
(і%
inputs         
ф " і         Г
H__inference_flatten_layer_layer_call_and_return_conditional_losses_10491a7б4
-б*
(і%
inputs         		
ф "&б#
і
0         љ

џ Ё
-__inference_flatten_layer_layer_call_fn_10496T7б4
-б*
(і%
inputs         		
ф "і         љ
Б
G__inference_leaky_ReLu_1_layer_call_and_return_conditional_losses_10520X/б,
%б"
 і
inputs         2
ф "%б"
і
0         2
џ {
,__inference_leaky_ReLu_1_layer_call_fn_10525K/б,
%б"
 і
inputs         2
ф "і         2Б
G__inference_leaky_ReLu_2_layer_call_and_return_conditional_losses_10549X/б,
%б"
 і
inputs         2
ф "%б"
і
0         2
џ {
,__inference_leaky_ReLu_2_layer_call_fn_10554K/б,
%б"
 і
inputs         2
ф "і         2Т
C__inference_maxpool_1_layer_call_and_return_conditional_losses_9463ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ Й
(__inference_maxpool_1_layer_call_fn_9469ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Т
C__inference_maxpool_2_layer_call_and_return_conditional_losses_9475ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ Й
(__inference_maxpool_2_layer_call_fn_9481ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Т
C__inference_maxpool_3_layer_call_and_return_conditional_losses_9487ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ Й
(__inference_maxpool_3_layer_call_fn_9493ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Д
G__inference_output_layer_layer_call_and_return_conditional_losses_10565\_`/б,
%б"
 і
inputs         2
ф "%б"
і
0         
џ 
,__inference_output_layer_layer_call_fn_10574O_`/б,
%б"
 і
inputs         2
ф "і         к
#__inference_signature_wrapper_10157ъ'(-.78=>KLUV_`KбH
б 
Aф>
<
input_layer-і*
input_layer         dd";ф8
6
output_layer&і#
output_layer         