Кй
х╗
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
·
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
╛
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
 И"serve*2.4.12v2.4.1-0-g85c8b2a817f8яч
О
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma
З
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
:*
dtype0
М
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta
Е
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
:*
dtype0
Ъ
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean
У
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
:*
dtype0
в
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance
Ы
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
:*
dtype0
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
: *
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
: *
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:  *
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
: *
dtype0
|
conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*
shared_nameconv3/kernel
u
 conv3/kernel/Read/ReadVariableOpReadVariableOpconv3/kernel*&
_output_shapes
: @*
dtype0
l

conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv3/bias
e
conv3/bias/Read/ReadVariableOpReadVariableOp
conv3/bias*
_output_shapes
:@*
dtype0
|
conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*
shared_nameconv4/kernel
u
 conv4/kernel/Read/ReadVariableOpReadVariableOpconv4/kernel*&
_output_shapes
:@@*
dtype0
l

conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
conv4/bias
e
conv4/bias/Read/ReadVariableOpReadVariableOp
conv4/bias*
_output_shapes
:@*
dtype0
x
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└ш*
shared_namedense1/kernel
q
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel* 
_output_shapes
:
└ш*
dtype0
o
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ш*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:ш*
dtype0
w
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ш
*
shared_namedense2/kernel
p
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes
:	ш
*
dtype0
n
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense2/bias
g
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes
:
*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
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
ж
'RMSprop/batch_normalization_8/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'RMSprop/batch_normalization_8/gamma/rms
Я
;RMSprop/batch_normalization_8/gamma/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_8/gamma/rms*
_output_shapes
:*
dtype0
д
&RMSprop/batch_normalization_8/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&RMSprop/batch_normalization_8/beta/rms
Э
:RMSprop/batch_normalization_8/beta/rms/Read/ReadVariableOpReadVariableOp&RMSprop/batch_normalization_8/beta/rms*
_output_shapes
:*
dtype0
Ф
RMSprop/conv1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameRMSprop/conv1/kernel/rms
Н
,RMSprop/conv1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1/kernel/rms*&
_output_shapes
: *
dtype0
Д
RMSprop/conv1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameRMSprop/conv1/bias/rms
}
*RMSprop/conv1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv1/bias/rms*
_output_shapes
: *
dtype0
Ф
RMSprop/conv2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *)
shared_nameRMSprop/conv2/kernel/rms
Н
,RMSprop/conv2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2/kernel/rms*&
_output_shapes
:  *
dtype0
Д
RMSprop/conv2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameRMSprop/conv2/bias/rms
}
*RMSprop/conv2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2/bias/rms*
_output_shapes
: *
dtype0
Ф
RMSprop/conv3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*)
shared_nameRMSprop/conv3/kernel/rms
Н
,RMSprop/conv3/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3/kernel/rms*&
_output_shapes
: @*
dtype0
Д
RMSprop/conv3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameRMSprop/conv3/bias/rms
}
*RMSprop/conv3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv3/bias/rms*
_output_shapes
:@*
dtype0
Ф
RMSprop/conv4/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameRMSprop/conv4/kernel/rms
Н
,RMSprop/conv4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv4/kernel/rms*&
_output_shapes
:@@*
dtype0
Д
RMSprop/conv4/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameRMSprop/conv4/bias/rms
}
*RMSprop/conv4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv4/bias/rms*
_output_shapes
:@*
dtype0
Р
RMSprop/dense1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
└ш**
shared_nameRMSprop/dense1/kernel/rms
Й
-RMSprop/dense1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense1/kernel/rms* 
_output_shapes
:
└ш*
dtype0
З
RMSprop/dense1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:ш*(
shared_nameRMSprop/dense1/bias/rms
А
+RMSprop/dense1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense1/bias/rms*
_output_shapes	
:ш*
dtype0
П
RMSprop/dense2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ш
**
shared_nameRMSprop/dense2/kernel/rms
И
-RMSprop/dense2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense2/kernel/rms*
_output_shapes
:	ш
*
dtype0
Ж
RMSprop/dense2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameRMSprop/dense2/bias/rms

+RMSprop/dense2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense2/bias/rms*
_output_shapes
:
*
dtype0

NoOpNoOp
щK
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*дK
valueЪKBЧK BРK
╙
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
Ч
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
h

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
R
)trainable_variables
*regularization_losses
+	variables
,	keras_api
R
-trainable_variables
.regularization_losses
/	variables
0	keras_api
h

1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
h

7kernel
8bias
9trainable_variables
:regularization_losses
;	variables
<	keras_api
R
=trainable_variables
>regularization_losses
?	variables
@	keras_api
R
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
R
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
h

Skernel
Tbias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
ч
Yiter
	Zdecay
[learning_rate
\momentum
]rho
rmsп
rms░
rms▒
rms▓
#rms│
$rms┤
1rms╡
2rms╢
7rms╖
8rms╕
Irms╣
Jrms║
Srms╗
Trms╝
f
0
1
2
3
#4
$5
16
27
78
89
I10
J11
S12
T13
 
v
0
1
2
3
4
5
#6
$7
18
29
710
811
I12
J13
S14
T15
н
trainable_variables
^non_trainable_variables

_layers
`layer_metrics
regularization_losses
ametrics
blayer_regularization_losses
	variables
 
 
fd
VARIABLE_VALUEbatch_normalization_8/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_8/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_8/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_8/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
2
3
н
trainable_variables
cnon_trainable_variables
dlayer_metrics
regularization_losses
emetrics

flayers
glayer_regularization_losses
	variables
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
н
trainable_variables
hnon_trainable_variables
ilayer_metrics
 regularization_losses
jmetrics

klayers
llayer_regularization_losses
!	variables
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
н
%trainable_variables
mnon_trainable_variables
nlayer_metrics
&regularization_losses
ometrics

players
qlayer_regularization_losses
'	variables
 
 
 
н
)trainable_variables
rnon_trainable_variables
slayer_metrics
*regularization_losses
tmetrics

ulayers
vlayer_regularization_losses
+	variables
 
 
 
н
-trainable_variables
wnon_trainable_variables
xlayer_metrics
.regularization_losses
ymetrics

zlayers
{layer_regularization_losses
/	variables
XV
VARIABLE_VALUEconv3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21
 

10
21
о
3trainable_variables
|non_trainable_variables
}layer_metrics
4regularization_losses
~metrics

layers
 Аlayer_regularization_losses
5	variables
XV
VARIABLE_VALUEconv4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
▓
9trainable_variables
Бnon_trainable_variables
Вlayer_metrics
:regularization_losses
Гmetrics
Дlayers
 Еlayer_regularization_losses
;	variables
 
 
 
▓
=trainable_variables
Жnon_trainable_variables
Зlayer_metrics
>regularization_losses
Иmetrics
Йlayers
 Кlayer_regularization_losses
?	variables
 
 
 
▓
Atrainable_variables
Лnon_trainable_variables
Мlayer_metrics
Bregularization_losses
Нmetrics
Оlayers
 Пlayer_regularization_losses
C	variables
 
 
 
▓
Etrainable_variables
Рnon_trainable_variables
Сlayer_metrics
Fregularization_losses
Тmetrics
Уlayers
 Фlayer_regularization_losses
G	variables
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 

I0
J1
▓
Ktrainable_variables
Хnon_trainable_variables
Цlayer_metrics
Lregularization_losses
Чmetrics
Шlayers
 Щlayer_regularization_losses
M	variables
 
 
 
▓
Otrainable_variables
Ъnon_trainable_variables
Ыlayer_metrics
Pregularization_losses
Ьmetrics
Эlayers
 Юlayer_regularization_losses
Q	variables
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1
 

S0
T1
▓
Utrainable_variables
Яnon_trainable_variables
аlayer_metrics
Vregularization_losses
бmetrics
вlayers
 гlayer_regularization_losses
W	variables
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE

0
1
^
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
 

д0
е1
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
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

жtotal

зcount
и	variables
й	keras_api
I

кtotal

лcount
м
_fn_kwargs
н	variables
о	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ж0
з1

и	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

к0
л1

н	variables
СО
VARIABLE_VALUE'RMSprop/batch_normalization_8/gamma/rmsSlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE&RMSprop/batch_normalization_8/beta/rmsRlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUERMSprop/conv1/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/conv1/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUERMSprop/conv2/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/conv2/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUERMSprop/conv3/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/conv3/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUERMSprop/conv4/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/conv4/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUERMSprop/dense1/kernel/rmsTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/dense1/bias/rmsRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUERMSprop/dense2/kernel/rmsTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUERMSprop/dense2/bias/rmsRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
К
serving_default_input_9Placeholder*/
_output_shapes
:           *
dtype0*$
shape:           
№
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_9batch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasdense1/kerneldense1/biasdense2/kerneldense2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *1J 8В */
f*R(
&__inference_signature_wrapper_38371402
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ь
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp conv3/kernel/Read/ReadVariableOpconv3/bias/Read/ReadVariableOp conv4/kernel/Read/ReadVariableOpconv4/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp;RMSprop/batch_normalization_8/gamma/rms/Read/ReadVariableOp:RMSprop/batch_normalization_8/beta/rms/Read/ReadVariableOp,RMSprop/conv1/kernel/rms/Read/ReadVariableOp*RMSprop/conv1/bias/rms/Read/ReadVariableOp,RMSprop/conv2/kernel/rms/Read/ReadVariableOp*RMSprop/conv2/bias/rms/Read/ReadVariableOp,RMSprop/conv3/kernel/rms/Read/ReadVariableOp*RMSprop/conv3/bias/rms/Read/ReadVariableOp,RMSprop/conv4/kernel/rms/Read/ReadVariableOp*RMSprop/conv4/bias/rms/Read/ReadVariableOp-RMSprop/dense1/kernel/rms/Read/ReadVariableOp+RMSprop/dense1/bias/rms/Read/ReadVariableOp-RMSprop/dense2/kernel/rms/Read/ReadVariableOp+RMSprop/dense2/bias/rms/Read/ReadVariableOpConst*4
Tin-
+2)	*
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
GPU2 *1J 8В **
f%R#
!__inference__traced_save_38372253
█
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv1/kernel
conv1/biasconv2/kernel
conv2/biasconv3/kernel
conv3/biasconv4/kernel
conv4/biasdense1/kerneldense1/biasdense2/kerneldense2/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1'RMSprop/batch_normalization_8/gamma/rms&RMSprop/batch_normalization_8/beta/rmsRMSprop/conv1/kernel/rmsRMSprop/conv1/bias/rmsRMSprop/conv2/kernel/rmsRMSprop/conv2/bias/rmsRMSprop/conv3/kernel/rmsRMSprop/conv3/bias/rmsRMSprop/conv4/kernel/rmsRMSprop/conv4/bias/rmsRMSprop/dense1/kernel/rmsRMSprop/dense1/bias/rmsRMSprop/dense2/kernel/rmsRMSprop/dense2/bias/rms*3
Tin,
*2(*
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
GPU2 *1J 8В *-
f(R&
$__inference__traced_restore_38372380·░
ў	
▌
D__inference_dense1_layer_call_and_return_conditional_losses_38372013

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ш2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ш2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
д
Н
C__inference_conv2_layer_call_and_return_conditional_losses_38370770

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв.conv2/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Relu╟
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype020
.conv2/kernel/Regularizer/Square/ReadVariableOp╡
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2!
conv2/kernel/Regularizer/SquareЩ
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv2/kernel/Regularizer/Const▓
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/SumЕ
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv2/kernel/Regularizer/mul/x┤
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/mul╨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
щ
d
F__inference_dropout1_layer_call_and_return_conditional_losses_38370804

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:          2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
∙	
▌
D__inference_dense2_layer_call_and_return_conditional_losses_38370996

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ш
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ш::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
├
Ў
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38370608

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
╦

▄
/__inference_sequential_8_layer_call_fn_38371331
input_9
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
identityИвStatefulPartitionedCall╗
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *1J 8В *S
fNRL
J__inference_sequential_8_layer_call_and_return_conditional_losses_383712962
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_9
√
Ў
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371783

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:           2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
┤Я
е
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371516

inputs1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource
identityИв$batch_normalization_8/AssignNewValueв&batch_normalization_8/AssignNewValue_1в5batch_normalization_8/FusedBatchNormV3/ReadVariableOpв7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_8/ReadVariableOpв&batch_normalization_8/ReadVariableOp_1вconv1/BiasAdd/ReadVariableOpвconv1/Conv2D/ReadVariableOpв.conv1/kernel/Regularizer/Square/ReadVariableOpвconv2/BiasAdd/ReadVariableOpвconv2/Conv2D/ReadVariableOpв.conv2/kernel/Regularizer/Square/ReadVariableOpвconv3/BiasAdd/ReadVariableOpвconv3/Conv2D/ReadVariableOpв.conv3/kernel/Regularizer/Square/ReadVariableOpвconv4/BiasAdd/ReadVariableOpвconv4/Conv2D/ReadVariableOpв.conv4/kernel/Regularizer/Square/ReadVariableOpвdense1/BiasAdd/ReadVariableOpвdense1/MatMul/ReadVariableOpвdense2/BiasAdd/ReadVariableOpвdense2/MatMul/ReadVariableOp╢
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp╝
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1щ
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1▄
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3inputs,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2(
&batch_normalization_8/FusedBatchNormV3▒
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*Q
_classG
ECloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValue┐
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*S
_classI
GEloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1з
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1/Conv2D/ReadVariableOp┌
conv1/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOpа
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2

conv1/Reluз
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2/Conv2D/ReadVariableOp╚
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOpа
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:          2

conv2/Relu╢
maxpool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
maxpool1/MaxPoolu
dropout1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout1/dropout/Constй
dropout1/dropout/MulMulmaxpool1/MaxPool:output:0dropout1/dropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout1/dropout/Muly
dropout1/dropout/ShapeShapemaxpool1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout1/dropout/Shape╫
-dropout1/dropout/random_uniform/RandomUniformRandomUniformdropout1/dropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype02/
-dropout1/dropout/random_uniform/RandomUniformЗ
dropout1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2!
dropout1/dropout/GreaterEqual/yъ
dropout1/dropout/GreaterEqualGreaterEqual6dropout1/dropout/random_uniform/RandomUniform:output:0(dropout1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2
dropout1/dropout/GreaterEqualв
dropout1/dropout/CastCast!dropout1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout1/dropout/Castж
dropout1/dropout/Mul_1Muldropout1/dropout/Mul:z:0dropout1/dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout1/dropout/Mul_1з
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv3/Conv2D/ReadVariableOp╩
conv3/Conv2DConv2Ddropout1/dropout/Mul_1:z:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
conv3/Conv2DЮ
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv3/BiasAdd/ReadVariableOpа
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2

conv3/Reluз
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv4/Conv2D/ReadVariableOp╚
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@*
paddingVALID*
strides
2
conv4/Conv2DЮ
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOpа
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:         

@2

conv4/Relu╢
maxpool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
maxpool2/MaxPoolu
dropout2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout2/dropout/Constй
dropout2/dropout/MulMulmaxpool2/MaxPool:output:0dropout2/dropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout2/dropout/Muly
dropout2/dropout/ShapeShapemaxpool2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout2/dropout/Shape╫
-dropout2/dropout/random_uniform/RandomUniformRandomUniformdropout2/dropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02/
-dropout2/dropout/random_uniform/RandomUniformЗ
dropout2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2!
dropout2/dropout/GreaterEqual/yъ
dropout2/dropout/GreaterEqualGreaterEqual6dropout2/dropout/random_uniform/RandomUniform:output:0(dropout2/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout2/dropout/GreaterEqualв
dropout2/dropout/CastCast!dropout2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout2/dropout/Castж
dropout2/dropout/Mul_1Muldropout2/dropout/Mul:z:0dropout2/dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout2/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
flatten/ConstФ
flatten/ReshapeReshapedropout2/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:         └2
flatten/Reshapeд
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
└ш*
dtype02
dense1/MatMul/ReadVariableOpЫ
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2
dense1/MatMulв
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02
dense1/BiasAdd/ReadVariableOpЮ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:         ш2
dense1/Reluu
dropout3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout3/dropout/Constв
dropout3/dropout/MulMuldense1/Relu:activations:0dropout3/dropout/Const:output:0*
T0*(
_output_shapes
:         ш2
dropout3/dropout/Muly
dropout3/dropout/ShapeShapedense1/Relu:activations:0*
T0*
_output_shapes
:2
dropout3/dropout/Shape╨
-dropout3/dropout/random_uniform/RandomUniformRandomUniformdropout3/dropout/Shape:output:0*
T0*(
_output_shapes
:         ш*
dtype02/
-dropout3/dropout/random_uniform/RandomUniformЗ
dropout3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2!
dropout3/dropout/GreaterEqual/yу
dropout3/dropout/GreaterEqualGreaterEqual6dropout3/dropout/random_uniform/RandomUniform:output:0(dropout3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ш2
dropout3/dropout/GreaterEqualЫ
dropout3/dropout/CastCast!dropout3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ш2
dropout3/dropout/CastЯ
dropout3/dropout/Mul_1Muldropout3/dropout/Mul:z:0dropout3/dropout/Cast:y:0*
T0*(
_output_shapes
:         ш2
dropout3/dropout/Mul_1г
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	ш
*
dtype02
dense2/MatMul/ReadVariableOpЬ
dense2/MatMulMatMuldropout3/dropout/Mul_1:z:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense2/MatMulб
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense2/BiasAdd/ReadVariableOpЭ
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense2/BiasAddv
dense2/SoftmaxSoftmaxdense2/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense2/Softmax═
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv1/kernel/Regularizer/Square/ReadVariableOp╡
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv1/kernel/Regularizer/SquareЩ
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv1/kernel/Regularizer/Const▓
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/SumЕ
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv1/kernel/Regularizer/mul/x┤
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/mul═
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype020
.conv2/kernel/Regularizer/Square/ReadVariableOp╡
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2!
conv2/kernel/Regularizer/SquareЩ
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv2/kernel/Regularizer/Const▓
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/SumЕ
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv2/kernel/Regularizer/mul/x┤
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/mul═
.conv3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv3/kernel/Regularizer/Square/ReadVariableOp╡
conv3/kernel/Regularizer/SquareSquare6conv3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv3/kernel/Regularizer/SquareЩ
conv3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv3/kernel/Regularizer/Const▓
conv3/kernel/Regularizer/SumSum#conv3/kernel/Regularizer/Square:y:0'conv3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/SumЕ
conv3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv3/kernel/Regularizer/mul/x┤
conv3/kernel/Regularizer/mulMul'conv3/kernel/Regularizer/mul/x:output:0%conv3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/mul═
.conv4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype020
.conv4/kernel/Regularizer/Square/ReadVariableOp╡
conv4/kernel/Regularizer/SquareSquare6conv4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2!
conv4/kernel/Regularizer/SquareЩ
conv4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv4/kernel/Regularizer/Const▓
conv4/kernel/Regularizer/SumSum#conv4/kernel/Regularizer/Square:y:0'conv4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/SumЕ
conv4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv4/kernel/Regularizer/mul/x┤
conv4/kernel/Regularizer/mulMul'conv4/kernel/Regularizer/mul/x:output:0%conv4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/mul┤
IdentityIdentitydense2/Softmax:softmax:0%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp/^conv2/kernel/Regularizer/Square/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp/^conv3/kernel/Regularizer/Square/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp/^conv4/kernel/Regularizer/Square/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2`
.conv3/kernel/Regularizer/Square/ReadVariableOp.conv3/kernel/Regularizer/Square/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2`
.conv4/kernel/Regularizer/Square/ReadVariableOp.conv4/kernel/Regularizer/Square/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
№
b
F__inference_maxpool1_layer_call_and_return_conditional_losses_38370625

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Я
G
+__inference_dropout3_layer_call_fn_38372049

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_383709722
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ш2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ш:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
■
Я
__inference_loss_fn_3_38372113;
7conv4_kernel_regularizer_square_readvariableop_resource
identityИв.conv4/kernel/Regularizer/Square/ReadVariableOpр
.conv4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv4_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype020
.conv4/kernel/Regularizer/Square/ReadVariableOp╡
conv4/kernel/Regularizer/SquareSquare6conv4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2!
conv4/kernel/Regularizer/SquareЩ
conv4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv4/kernel/Regularizer/Const▓
conv4/kernel/Regularizer/SumSum#conv4/kernel/Regularizer/Square:y:0'conv4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/SumЕ
conv4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv4/kernel/Regularizer/mul/x┤
conv4/kernel/Regularizer/mulMul'conv4/kernel/Regularizer/mul/x:output:0%conv4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/mulФ
IdentityIdentity conv4/kernel/Regularizer/mul:z:0/^conv4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.conv4/kernel/Regularizer/Square/ReadVariableOp.conv4/kernel/Regularizer/Square/ReadVariableOp
╜
a
E__inference_flatten_layer_call_and_return_conditional_losses_38371997

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╙д
╠
$__inference__traced_restore_38372380
file_prefix0
,assignvariableop_batch_normalization_8_gamma1
-assignvariableop_1_batch_normalization_8_beta8
4assignvariableop_2_batch_normalization_8_moving_mean<
8assignvariableop_3_batch_normalization_8_moving_variance#
assignvariableop_4_conv1_kernel!
assignvariableop_5_conv1_bias#
assignvariableop_6_conv2_kernel!
assignvariableop_7_conv2_bias#
assignvariableop_8_conv3_kernel!
assignvariableop_9_conv3_bias$
 assignvariableop_10_conv4_kernel"
assignvariableop_11_conv4_bias%
!assignvariableop_12_dense1_kernel#
assignvariableop_13_dense1_bias%
!assignvariableop_14_dense2_kernel#
assignvariableop_15_dense2_bias$
 assignvariableop_16_rmsprop_iter%
!assignvariableop_17_rmsprop_decay-
)assignvariableop_18_rmsprop_learning_rate(
$assignvariableop_19_rmsprop_momentum#
assignvariableop_20_rmsprop_rho
assignvariableop_21_total
assignvariableop_22_count
assignvariableop_23_total_1
assignvariableop_24_count_1?
;assignvariableop_25_rmsprop_batch_normalization_8_gamma_rms>
:assignvariableop_26_rmsprop_batch_normalization_8_beta_rms0
,assignvariableop_27_rmsprop_conv1_kernel_rms.
*assignvariableop_28_rmsprop_conv1_bias_rms0
,assignvariableop_29_rmsprop_conv2_kernel_rms.
*assignvariableop_30_rmsprop_conv2_bias_rms0
,assignvariableop_31_rmsprop_conv3_kernel_rms.
*assignvariableop_32_rmsprop_conv3_bias_rms0
,assignvariableop_33_rmsprop_conv4_kernel_rms.
*assignvariableop_34_rmsprop_conv4_bias_rms1
-assignvariableop_35_rmsprop_dense1_kernel_rms/
+assignvariableop_36_rmsprop_dense1_bias_rms1
-assignvariableop_37_rmsprop_dense2_kernel_rms/
+assignvariableop_38_rmsprop_dense2_bias_rms
identity_40ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9З
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*У
valueЙBЖ(B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names▐
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╢
_output_shapesг
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityл
AssignVariableOpAssignVariableOp,assignvariableop_batch_normalization_8_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1▓
AssignVariableOp_1AssignVariableOp-assignvariableop_1_batch_normalization_8_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2╣
AssignVariableOp_2AssignVariableOp4assignvariableop_2_batch_normalization_8_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3╜
AssignVariableOp_3AssignVariableOp8assignvariableop_3_batch_normalization_8_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4д
AssignVariableOp_4AssignVariableOpassignvariableop_4_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5в
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6д
AssignVariableOp_6AssignVariableOpassignvariableop_6_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7в
AssignVariableOp_7AssignVariableOpassignvariableop_7_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8д
AssignVariableOp_8AssignVariableOpassignvariableop_8_conv3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9в
AssignVariableOp_9AssignVariableOpassignvariableop_9_conv3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10и
AssignVariableOp_10AssignVariableOp assignvariableop_10_conv4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ж
AssignVariableOp_11AssignVariableOpassignvariableop_11_conv4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12й
AssignVariableOp_12AssignVariableOp!assignvariableop_12_dense1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13з
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14й
AssignVariableOp_14AssignVariableOp!assignvariableop_14_dense2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15з
AssignVariableOp_15AssignVariableOpassignvariableop_15_dense2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16и
AssignVariableOp_16AssignVariableOp assignvariableop_16_rmsprop_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17й
AssignVariableOp_17AssignVariableOp!assignvariableop_17_rmsprop_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18▒
AssignVariableOp_18AssignVariableOp)assignvariableop_18_rmsprop_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19м
AssignVariableOp_19AssignVariableOp$assignvariableop_19_rmsprop_momentumIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20з
AssignVariableOp_20AssignVariableOpassignvariableop_20_rmsprop_rhoIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21б
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22б
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23г
AssignVariableOp_23AssignVariableOpassignvariableop_23_total_1Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24г
AssignVariableOp_24AssignVariableOpassignvariableop_24_count_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25├
AssignVariableOp_25AssignVariableOp;assignvariableop_25_rmsprop_batch_normalization_8_gamma_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26┬
AssignVariableOp_26AssignVariableOp:assignvariableop_26_rmsprop_batch_normalization_8_beta_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27┤
AssignVariableOp_27AssignVariableOp,assignvariableop_27_rmsprop_conv1_kernel_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▓
AssignVariableOp_28AssignVariableOp*assignvariableop_28_rmsprop_conv1_bias_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29┤
AssignVariableOp_29AssignVariableOp,assignvariableop_29_rmsprop_conv2_kernel_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▓
AssignVariableOp_30AssignVariableOp*assignvariableop_30_rmsprop_conv2_bias_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31┤
AssignVariableOp_31AssignVariableOp,assignvariableop_31_rmsprop_conv3_kernel_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▓
AssignVariableOp_32AssignVariableOp*assignvariableop_32_rmsprop_conv3_bias_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33┤
AssignVariableOp_33AssignVariableOp,assignvariableop_33_rmsprop_conv4_kernel_rmsIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▓
AssignVariableOp_34AssignVariableOp*assignvariableop_34_rmsprop_conv4_bias_rmsIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35╡
AssignVariableOp_35AssignVariableOp-assignvariableop_35_rmsprop_dense1_kernel_rmsIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36│
AssignVariableOp_36AssignVariableOp+assignvariableop_36_rmsprop_dense1_bias_rmsIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37╡
AssignVariableOp_37AssignVariableOp-assignvariableop_37_rmsprop_dense2_kernel_rmsIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38│
AssignVariableOp_38AssignVariableOp+assignvariableop_38_rmsprop_dense2_bias_rmsIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_389
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╕
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39л
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*│
_input_shapesб
Ю: :::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
у
~
)__inference_dense2_layer_call_fn_38372069

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall∙
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_383709962
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ш::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
оi
Л
#__inference__wrapped_model_38370515
input_9>
:sequential_8_batch_normalization_8_readvariableop_resource@
<sequential_8_batch_normalization_8_readvariableop_1_resourceO
Ksequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resourceQ
Msequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource5
1sequential_8_conv1_conv2d_readvariableop_resource6
2sequential_8_conv1_biasadd_readvariableop_resource5
1sequential_8_conv2_conv2d_readvariableop_resource6
2sequential_8_conv2_biasadd_readvariableop_resource5
1sequential_8_conv3_conv2d_readvariableop_resource6
2sequential_8_conv3_biasadd_readvariableop_resource5
1sequential_8_conv4_conv2d_readvariableop_resource6
2sequential_8_conv4_biasadd_readvariableop_resource6
2sequential_8_dense1_matmul_readvariableop_resource7
3sequential_8_dense1_biasadd_readvariableop_resource6
2sequential_8_dense2_matmul_readvariableop_resource7
3sequential_8_dense2_biasadd_readvariableop_resource
identityИвBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpвDsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в1sequential_8/batch_normalization_8/ReadVariableOpв3sequential_8/batch_normalization_8/ReadVariableOp_1в)sequential_8/conv1/BiasAdd/ReadVariableOpв(sequential_8/conv1/Conv2D/ReadVariableOpв)sequential_8/conv2/BiasAdd/ReadVariableOpв(sequential_8/conv2/Conv2D/ReadVariableOpв)sequential_8/conv3/BiasAdd/ReadVariableOpв(sequential_8/conv3/Conv2D/ReadVariableOpв)sequential_8/conv4/BiasAdd/ReadVariableOpв(sequential_8/conv4/Conv2D/ReadVariableOpв*sequential_8/dense1/BiasAdd/ReadVariableOpв)sequential_8/dense1/MatMul/ReadVariableOpв*sequential_8/dense2/BiasAdd/ReadVariableOpв)sequential_8/dense2/MatMul/ReadVariableOp▌
1sequential_8/batch_normalization_8/ReadVariableOpReadVariableOp:sequential_8_batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential_8/batch_normalization_8/ReadVariableOpу
3sequential_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp<sequential_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype025
3sequential_8/batch_normalization_8/ReadVariableOp_1Р
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02D
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpЦ
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02F
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Э
3sequential_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3input_99sequential_8/batch_normalization_8/ReadVariableOp:value:0;sequential_8/batch_normalization_8/ReadVariableOp_1:value:0Jsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           :::::*
epsilon%oГ:*
is_training( 25
3sequential_8/batch_normalization_8/FusedBatchNormV3╬
(sequential_8/conv1/Conv2D/ReadVariableOpReadVariableOp1sequential_8_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(sequential_8/conv1/Conv2D/ReadVariableOpО
sequential_8/conv1/Conv2DConv2D7sequential_8/batch_normalization_8/FusedBatchNormV3:y:00sequential_8/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
sequential_8/conv1/Conv2D┼
)sequential_8/conv1/BiasAdd/ReadVariableOpReadVariableOp2sequential_8_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)sequential_8/conv1/BiasAdd/ReadVariableOp╘
sequential_8/conv1/BiasAddBiasAdd"sequential_8/conv1/Conv2D:output:01sequential_8/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
sequential_8/conv1/BiasAddЩ
sequential_8/conv1/ReluRelu#sequential_8/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_8/conv1/Relu╬
(sequential_8/conv2/Conv2D/ReadVariableOpReadVariableOp1sequential_8_conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02*
(sequential_8/conv2/Conv2D/ReadVariableOp№
sequential_8/conv2/Conv2DConv2D%sequential_8/conv1/Relu:activations:00sequential_8/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
sequential_8/conv2/Conv2D┼
)sequential_8/conv2/BiasAdd/ReadVariableOpReadVariableOp2sequential_8_conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)sequential_8/conv2/BiasAdd/ReadVariableOp╘
sequential_8/conv2/BiasAddBiasAdd"sequential_8/conv2/Conv2D:output:01sequential_8/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
sequential_8/conv2/BiasAddЩ
sequential_8/conv2/ReluRelu#sequential_8/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:          2
sequential_8/conv2/Relu▌
sequential_8/maxpool1/MaxPoolMaxPool%sequential_8/conv2/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
sequential_8/maxpool1/MaxPoolо
sequential_8/dropout1/IdentityIdentity&sequential_8/maxpool1/MaxPool:output:0*
T0*/
_output_shapes
:          2 
sequential_8/dropout1/Identity╬
(sequential_8/conv3/Conv2D/ReadVariableOpReadVariableOp1sequential_8_conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02*
(sequential_8/conv3/Conv2D/ReadVariableOp■
sequential_8/conv3/Conv2DConv2D'sequential_8/dropout1/Identity:output:00sequential_8/conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
sequential_8/conv3/Conv2D┼
)sequential_8/conv3/BiasAdd/ReadVariableOpReadVariableOp2sequential_8_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential_8/conv3/BiasAdd/ReadVariableOp╘
sequential_8/conv3/BiasAddBiasAdd"sequential_8/conv3/Conv2D:output:01sequential_8/conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
sequential_8/conv3/BiasAddЩ
sequential_8/conv3/ReluRelu#sequential_8/conv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
sequential_8/conv3/Relu╬
(sequential_8/conv4/Conv2D/ReadVariableOpReadVariableOp1sequential_8_conv4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(sequential_8/conv4/Conv2D/ReadVariableOp№
sequential_8/conv4/Conv2DConv2D%sequential_8/conv3/Relu:activations:00sequential_8/conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@*
paddingVALID*
strides
2
sequential_8/conv4/Conv2D┼
)sequential_8/conv4/BiasAdd/ReadVariableOpReadVariableOp2sequential_8_conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)sequential_8/conv4/BiasAdd/ReadVariableOp╘
sequential_8/conv4/BiasAddBiasAdd"sequential_8/conv4/Conv2D:output:01sequential_8/conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@2
sequential_8/conv4/BiasAddЩ
sequential_8/conv4/ReluRelu#sequential_8/conv4/BiasAdd:output:0*
T0*/
_output_shapes
:         

@2
sequential_8/conv4/Relu▌
sequential_8/maxpool2/MaxPoolMaxPool%sequential_8/conv4/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
sequential_8/maxpool2/MaxPoolо
sequential_8/dropout2/IdentityIdentity&sequential_8/maxpool2/MaxPool:output:0*
T0*/
_output_shapes
:         @2 
sequential_8/dropout2/IdentityЙ
sequential_8/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
sequential_8/flatten/Const╚
sequential_8/flatten/ReshapeReshape'sequential_8/dropout2/Identity:output:0#sequential_8/flatten/Const:output:0*
T0*(
_output_shapes
:         └2
sequential_8/flatten/Reshape╦
)sequential_8/dense1/MatMul/ReadVariableOpReadVariableOp2sequential_8_dense1_matmul_readvariableop_resource* 
_output_shapes
:
└ш*
dtype02+
)sequential_8/dense1/MatMul/ReadVariableOp╧
sequential_8/dense1/MatMulMatMul%sequential_8/flatten/Reshape:output:01sequential_8/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2
sequential_8/dense1/MatMul╔
*sequential_8/dense1/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_dense1_biasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02,
*sequential_8/dense1/BiasAdd/ReadVariableOp╥
sequential_8/dense1/BiasAddBiasAdd$sequential_8/dense1/MatMul:product:02sequential_8/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2
sequential_8/dense1/BiasAddХ
sequential_8/dense1/ReluRelu$sequential_8/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:         ш2
sequential_8/dense1/Reluз
sequential_8/dropout3/IdentityIdentity&sequential_8/dense1/Relu:activations:0*
T0*(
_output_shapes
:         ш2 
sequential_8/dropout3/Identity╩
)sequential_8/dense2/MatMul/ReadVariableOpReadVariableOp2sequential_8_dense2_matmul_readvariableop_resource*
_output_shapes
:	ш
*
dtype02+
)sequential_8/dense2/MatMul/ReadVariableOp╨
sequential_8/dense2/MatMulMatMul'sequential_8/dropout3/Identity:output:01sequential_8/dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential_8/dense2/MatMul╚
*sequential_8/dense2/BiasAdd/ReadVariableOpReadVariableOp3sequential_8_dense2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02,
*sequential_8/dense2/BiasAdd/ReadVariableOp╤
sequential_8/dense2/BiasAddBiasAdd$sequential_8/dense2/MatMul:product:02sequential_8/dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
sequential_8/dense2/BiasAddЭ
sequential_8/dense2/SoftmaxSoftmax$sequential_8/dense2/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
sequential_8/dense2/Softmax¤
IdentityIdentity%sequential_8/dense2/Softmax:softmax:0C^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpE^sequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12^sequential_8/batch_normalization_8/ReadVariableOp4^sequential_8/batch_normalization_8/ReadVariableOp_1*^sequential_8/conv1/BiasAdd/ReadVariableOp)^sequential_8/conv1/Conv2D/ReadVariableOp*^sequential_8/conv2/BiasAdd/ReadVariableOp)^sequential_8/conv2/Conv2D/ReadVariableOp*^sequential_8/conv3/BiasAdd/ReadVariableOp)^sequential_8/conv3/Conv2D/ReadVariableOp*^sequential_8/conv4/BiasAdd/ReadVariableOp)^sequential_8/conv4/Conv2D/ReadVariableOp+^sequential_8/dense1/BiasAdd/ReadVariableOp*^sequential_8/dense1/MatMul/ReadVariableOp+^sequential_8/dense2/BiasAdd/ReadVariableOp*^sequential_8/dense2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::2И
Bsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpBsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2М
Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1Dsequential_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12f
1sequential_8/batch_normalization_8/ReadVariableOp1sequential_8/batch_normalization_8/ReadVariableOp2j
3sequential_8/batch_normalization_8/ReadVariableOp_13sequential_8/batch_normalization_8/ReadVariableOp_12V
)sequential_8/conv1/BiasAdd/ReadVariableOp)sequential_8/conv1/BiasAdd/ReadVariableOp2T
(sequential_8/conv1/Conv2D/ReadVariableOp(sequential_8/conv1/Conv2D/ReadVariableOp2V
)sequential_8/conv2/BiasAdd/ReadVariableOp)sequential_8/conv2/BiasAdd/ReadVariableOp2T
(sequential_8/conv2/Conv2D/ReadVariableOp(sequential_8/conv2/Conv2D/ReadVariableOp2V
)sequential_8/conv3/BiasAdd/ReadVariableOp)sequential_8/conv3/BiasAdd/ReadVariableOp2T
(sequential_8/conv3/Conv2D/ReadVariableOp(sequential_8/conv3/Conv2D/ReadVariableOp2V
)sequential_8/conv4/BiasAdd/ReadVariableOp)sequential_8/conv4/BiasAdd/ReadVariableOp2T
(sequential_8/conv4/Conv2D/ReadVariableOp(sequential_8/conv4/Conv2D/ReadVariableOp2X
*sequential_8/dense1/BiasAdd/ReadVariableOp*sequential_8/dense1/BiasAdd/ReadVariableOp2V
)sequential_8/dense1/MatMul/ReadVariableOp)sequential_8/dense1/MatMul/ReadVariableOp2X
*sequential_8/dense2/BiasAdd/ReadVariableOp*sequential_8/dense2/BiasAdd/ReadVariableOp2V
)sequential_8/dense2/MatMul/ReadVariableOp)sequential_8/dense2/MatMul/ReadVariableOp:X T
/
_output_shapes
:           
!
_user_specified_name	input_9
 
}
(__inference_conv1_layer_call_fn_38371841

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_383707372
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:           ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╜
a
E__inference_flatten_layer_call_and_return_conditional_losses_38370920

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         └2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
д
Н
C__inference_conv3_layer_call_and_return_conditional_losses_38370834

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв.conv3/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Relu╟
.conv3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv3/kernel/Regularizer/Square/ReadVariableOp╡
conv3/kernel/Regularizer/SquareSquare6conv3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv3/kernel/Regularizer/SquareЩ
conv3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv3/kernel/Regularizer/Const▓
conv3/kernel/Regularizer/SumSum#conv3/kernel/Regularizer/Square:y:0'conv3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/SumЕ
conv3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv3/kernel/Regularizer/mul/x┤
conv3/kernel/Regularizer/mulMul'conv3/kernel/Regularizer/mul/x:output:0%conv3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/mul╨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv3/kernel/Regularizer/Square/ReadVariableOp.conv3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╟
d
+__inference_dropout1_layer_call_fn_38371895

inputs
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_383707992
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ы

╙
&__inference_signature_wrapper_38371402
input_9
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
identityИвStatefulPartitionedCallФ
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *1J 8В *,
f'R%
#__inference__wrapped_model_383705152
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_9
Л
e
F__inference_dropout3_layer_call_and_return_conditional_losses_38370967

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ш2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ш*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ш2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ш2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ш2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ш2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ш:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
ў	
▌
D__inference_dense1_layer_call_and_return_conditional_losses_38370939

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
└ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ш2
ReluШ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ш2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
ж
л
8__inference_batch_normalization_8_layer_call_fn_38371732

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_383705772
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
р
л
8__inference_batch_normalization_8_layer_call_fn_38371809

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_383706842
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:           2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
 
}
(__inference_conv4_layer_call_fn_38371964

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_383708672
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         

@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
л
F
*__inference_flatten_layer_call_fn_38372002

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_383709202
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         └2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╧
Ъ
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38370577

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
═
d
F__inference_dropout3_layer_call_and_return_conditional_losses_38370972

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ш2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ш2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ш:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
Тd
Н
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371037
input_9"
batch_normalization_8_38370711"
batch_normalization_8_38370713"
batch_normalization_8_38370715"
batch_normalization_8_38370717
conv1_38370748
conv1_38370750
conv2_38370781
conv2_38370783
conv3_38370845
conv3_38370847
conv4_38370878
conv4_38370880
dense1_38370950
dense1_38370952
dense2_38371007
dense2_38371009
identityИв-batch_normalization_8/StatefulPartitionedCallвconv1/StatefulPartitionedCallв.conv1/kernel/Regularizer/Square/ReadVariableOpвconv2/StatefulPartitionedCallв.conv2/kernel/Regularizer/Square/ReadVariableOpвconv3/StatefulPartitionedCallв.conv3/kernel/Regularizer/Square/ReadVariableOpвconv4/StatefulPartitionedCallв.conv4/kernel/Regularizer/Square/ReadVariableOpвdense1/StatefulPartitionedCallвdense2/StatefulPartitionedCallв dropout1/StatefulPartitionedCallв dropout2/StatefulPartitionedCallв dropout3/StatefulPartitionedCallл
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCallinput_9batch_normalization_8_38370711batch_normalization_8_38370713batch_normalization_8_38370715batch_normalization_8_38370717*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_383706662/
-batch_normalization_8/StatefulPartitionedCall╚
conv1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv1_38370748conv1_38370750*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_383707372
conv1/StatefulPartitionedCall╕
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_38370781conv2_38370783*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_383707702
conv2/StatefulPartitionedCallГ
maxpool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool1_layer_call_and_return_conditional_losses_383706252
maxpool1/PartitionedCallЦ
 dropout1/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_383707992"
 dropout1/StatefulPartitionedCall╗
conv3/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0conv3_38370845conv3_38370847*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_383708342
conv3/StatefulPartitionedCall╕
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0conv4_38370878conv4_38370880*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_383708672
conv4/StatefulPartitionedCallГ
maxpool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool2_layer_call_and_return_conditional_losses_383706372
maxpool2/PartitionedCall╣
 dropout2/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0!^dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_383708962"
 dropout2/StatefulPartitionedCall№
flatten/PartitionedCallPartitionedCall)dropout2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_383709202
flatten/PartitionedCall░
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_38370950dense1_38370952*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_383709392 
dense1/StatefulPartitionedCall╕
 dropout3/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0!^dropout2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_383709672"
 dropout3/StatefulPartitionedCall╕
dense2/StatefulPartitionedCallStatefulPartitionedCall)dropout3/StatefulPartitionedCall:output:0dense2_38371007dense2_38371009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_383709962 
dense2/StatefulPartitionedCall╖
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_38370748*&
_output_shapes
: *
dtype020
.conv1/kernel/Regularizer/Square/ReadVariableOp╡
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv1/kernel/Regularizer/SquareЩ
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv1/kernel/Regularizer/Const▓
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/SumЕ
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv1/kernel/Regularizer/mul/x┤
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/mul╖
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_38370781*&
_output_shapes
:  *
dtype020
.conv2/kernel/Regularizer/Square/ReadVariableOp╡
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2!
conv2/kernel/Regularizer/SquareЩ
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv2/kernel/Regularizer/Const▓
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/SumЕ
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv2/kernel/Regularizer/mul/x┤
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/mul╖
.conv3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_38370845*&
_output_shapes
: @*
dtype020
.conv3/kernel/Regularizer/Square/ReadVariableOp╡
conv3/kernel/Regularizer/SquareSquare6conv3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv3/kernel/Regularizer/SquareЩ
conv3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv3/kernel/Regularizer/Const▓
conv3/kernel/Regularizer/SumSum#conv3/kernel/Regularizer/Square:y:0'conv3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/SumЕ
conv3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv3/kernel/Regularizer/mul/x┤
conv3/kernel/Regularizer/mulMul'conv3/kernel/Regularizer/mul/x:output:0%conv3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/mul╖
.conv4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv4_38370878*&
_output_shapes
:@@*
dtype020
.conv4/kernel/Regularizer/Square/ReadVariableOp╡
conv4/kernel/Regularizer/SquareSquare6conv4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2!
conv4/kernel/Regularizer/SquareЩ
conv4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv4/kernel/Regularizer/Const▓
conv4/kernel/Regularizer/SumSum#conv4/kernel/Regularizer/Square:y:0'conv4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/SumЕ
conv4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv4/kernel/Regularizer/mul/x┤
conv4/kernel/Regularizer/mulMul'conv4/kernel/Regularizer/mul/x:output:0%conv4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/mulЪ
IdentityIdentity'dense2/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall^conv1/StatefulPartitionedCall/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/StatefulPartitionedCall/^conv2/kernel/Regularizer/Square/ReadVariableOp^conv3/StatefulPartitionedCall/^conv3/kernel/Regularizer/Square/ReadVariableOp^conv4/StatefulPartitionedCall/^conv4/kernel/Regularizer/Square/ReadVariableOp^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall!^dropout3/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2`
.conv3/kernel/Regularizer/Square/ReadVariableOp.conv3/kernel/Regularizer/Square/ReadVariableOp2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2`
.conv4/kernel/Regularizer/Square/ReadVariableOp.conv4/kernel/Regularizer/Square/ReadVariableOp2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2D
 dropout3/StatefulPartitionedCall dropout3/StatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_9
■
Я
__inference_loss_fn_0_38372080;
7conv1_kernel_regularizer_square_readvariableop_resource
identityИв.conv1/kernel/Regularizer/Square/ReadVariableOpр
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv1_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv1/kernel/Regularizer/Square/ReadVariableOp╡
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv1/kernel/Regularizer/SquareЩ
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv1/kernel/Regularizer/Const▓
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/SumЕ
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv1/kernel/Regularizer/mul/x┤
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/mulФ
IdentityIdentity conv1/kernel/Regularizer/mul:z:0/^conv1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp
л
d
+__inference_dropout3_layer_call_fn_38372044

inputs
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_383709672
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ш2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ш22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
─
e
F__inference_dropout1_layer_call_and_return_conditional_losses_38370799

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
щ
d
F__inference_dropout1_layer_call_and_return_conditional_losses_38371890

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:          2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:          2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╟
d
+__inference_dropout2_layer_call_fn_38371986

inputs
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_383708962
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
д
Н
C__inference_conv4_layer_call_and_return_conditional_losses_38370867

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв.conv4/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         

@2
Relu╟
.conv4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype020
.conv4/kernel/Regularizer/Square/ReadVariableOp╡
conv4/kernel/Regularizer/SquareSquare6conv4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2!
conv4/kernel/Regularizer/SquareЩ
conv4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv4/kernel/Regularizer/Const▓
conv4/kernel/Regularizer/SumSum#conv4/kernel/Regularizer/Square:y:0'conv4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/SumЕ
conv4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv4/kernel/Regularizer/mul/x┤
conv4/kernel/Regularizer/mulMul'conv4/kernel/Regularizer/mul/x:output:0%conv4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/mul╨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         

@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv4/kernel/Regularizer/Square/ReadVariableOp.conv4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
д
Н
C__inference_conv4_layer_call_and_return_conditional_losses_38371955

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв.conv4/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         

@2
Relu╟
.conv4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype020
.conv4/kernel/Regularizer/Square/ReadVariableOp╡
conv4/kernel/Regularizer/SquareSquare6conv4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2!
conv4/kernel/Regularizer/SquareЩ
conv4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv4/kernel/Regularizer/Const▓
conv4/kernel/Regularizer/SumSum#conv4/kernel/Regularizer/Square:y:0'conv4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/SumЕ
conv4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv4/kernel/Regularizer/mul/x┤
conv4/kernel/Regularizer/mulMul'conv4/kernel/Regularizer/mul/x:output:0%conv4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/mul╨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         

@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv4/kernel/Regularizer/Square/ReadVariableOp.conv4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
├
Ў
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371719

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3ь
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
─
e
F__inference_dropout2_layer_call_and_return_conditional_losses_38371976

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
щ
d
F__inference_dropout2_layer_call_and_return_conditional_losses_38370901

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╗
G
+__inference_dropout1_layer_call_fn_38371900

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_383708042
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
╞

█
/__inference_sequential_8_layer_call_fn_38371644

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
identityИвStatefulPartitionedCall╕
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *1J 8В *S
fNRL
J__inference_sequential_8_layer_call_and_return_conditional_losses_383711862
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
и
л
8__inference_batch_normalization_8_layer_call_fn_38371745

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_383706082
StatefulPartitionedCallи
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
и
G
+__inference_maxpool2_layer_call_fn_38370643

inputs
identityь
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
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool2_layer_call_and_return_conditional_losses_383706372
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╗
G
+__inference_dropout2_layer_call_fn_38371991

inputs
identity╤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_383709012
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
∙	
▌
D__inference_dense2_layer_call_and_return_conditional_losses_38372060

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ш
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         
2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ш::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
щ
d
F__inference_dropout2_layer_call_and_return_conditional_losses_38371981

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:         @2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         @2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
■
Я
__inference_loss_fn_1_38372091;
7conv2_kernel_regularizer_square_readvariableop_resource
identityИв.conv2/kernel/Regularizer/Square/ReadVariableOpр
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype020
.conv2/kernel/Regularizer/Square/ReadVariableOp╡
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2!
conv2/kernel/Regularizer/SquareЩ
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv2/kernel/Regularizer/Const▓
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/SumЕ
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv2/kernel/Regularizer/mul/x┤
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/mulФ
IdentityIdentity conv2/kernel/Regularizer/mul:z:0/^conv2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp
╩_
д
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371110
input_9"
batch_normalization_8_38371040"
batch_normalization_8_38371042"
batch_normalization_8_38371044"
batch_normalization_8_38371046
conv1_38371049
conv1_38371051
conv2_38371054
conv2_38371056
conv3_38371061
conv3_38371063
conv4_38371066
conv4_38371068
dense1_38371074
dense1_38371076
dense2_38371080
dense2_38371082
identityИв-batch_normalization_8/StatefulPartitionedCallвconv1/StatefulPartitionedCallв.conv1/kernel/Regularizer/Square/ReadVariableOpвconv2/StatefulPartitionedCallв.conv2/kernel/Regularizer/Square/ReadVariableOpвconv3/StatefulPartitionedCallв.conv3/kernel/Regularizer/Square/ReadVariableOpвconv4/StatefulPartitionedCallв.conv4/kernel/Regularizer/Square/ReadVariableOpвdense1/StatefulPartitionedCallвdense2/StatefulPartitionedCallн
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCallinput_9batch_normalization_8_38371040batch_normalization_8_38371042batch_normalization_8_38371044batch_normalization_8_38371046*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_383706842/
-batch_normalization_8/StatefulPartitionedCall╚
conv1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv1_38371049conv1_38371051*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_383707372
conv1/StatefulPartitionedCall╕
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_38371054conv2_38371056*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_383707702
conv2/StatefulPartitionedCallГ
maxpool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool1_layer_call_and_return_conditional_losses_383706252
maxpool1/PartitionedCall■
dropout1/PartitionedCallPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_383708042
dropout1/PartitionedCall│
conv3/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0conv3_38371061conv3_38371063*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_383708342
conv3/StatefulPartitionedCall╕
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0conv4_38371066conv4_38371068*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_383708672
conv4/StatefulPartitionedCallГ
maxpool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool2_layer_call_and_return_conditional_losses_383706372
maxpool2/PartitionedCall■
dropout2/PartitionedCallPartitionedCall!maxpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_383709012
dropout2/PartitionedCallЇ
flatten/PartitionedCallPartitionedCall!dropout2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_383709202
flatten/PartitionedCall░
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_38371074dense1_38371076*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_383709392 
dense1/StatefulPartitionedCall¤
dropout3/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_383709722
dropout3/PartitionedCall░
dense2/StatefulPartitionedCallStatefulPartitionedCall!dropout3/PartitionedCall:output:0dense2_38371080dense2_38371082*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_383709962 
dense2/StatefulPartitionedCall╖
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_38371049*&
_output_shapes
: *
dtype020
.conv1/kernel/Regularizer/Square/ReadVariableOp╡
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv1/kernel/Regularizer/SquareЩ
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv1/kernel/Regularizer/Const▓
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/SumЕ
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv1/kernel/Regularizer/mul/x┤
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/mul╖
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_38371054*&
_output_shapes
:  *
dtype020
.conv2/kernel/Regularizer/Square/ReadVariableOp╡
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2!
conv2/kernel/Regularizer/SquareЩ
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv2/kernel/Regularizer/Const▓
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/SumЕ
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv2/kernel/Regularizer/mul/x┤
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/mul╖
.conv3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_38371061*&
_output_shapes
: @*
dtype020
.conv3/kernel/Regularizer/Square/ReadVariableOp╡
conv3/kernel/Regularizer/SquareSquare6conv3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv3/kernel/Regularizer/SquareЩ
conv3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv3/kernel/Regularizer/Const▓
conv3/kernel/Regularizer/SumSum#conv3/kernel/Regularizer/Square:y:0'conv3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/SumЕ
conv3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv3/kernel/Regularizer/mul/x┤
conv3/kernel/Regularizer/mulMul'conv3/kernel/Regularizer/mul/x:output:0%conv3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/mul╖
.conv4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv4_38371066*&
_output_shapes
:@@*
dtype020
.conv4/kernel/Regularizer/Square/ReadVariableOp╡
conv4/kernel/Regularizer/SquareSquare6conv4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2!
conv4/kernel/Regularizer/SquareЩ
conv4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv4/kernel/Regularizer/Const▓
conv4/kernel/Regularizer/SumSum#conv4/kernel/Regularizer/Square:y:0'conv4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/SumЕ
conv4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv4/kernel/Regularizer/mul/x┤
conv4/kernel/Regularizer/mulMul'conv4/kernel/Regularizer/mul/x:output:0%conv4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/mul▒
IdentityIdentity'dense2/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall^conv1/StatefulPartitionedCall/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/StatefulPartitionedCall/^conv2/kernel/Regularizer/Square/ReadVariableOp^conv3/StatefulPartitionedCall/^conv3/kernel/Regularizer/Square/ReadVariableOp^conv4/StatefulPartitionedCall/^conv4/kernel/Regularizer/Square/ReadVariableOp^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2`
.conv3/kernel/Regularizer/Square/ReadVariableOp.conv3/kernel/Regularizer/Square/ReadVariableOp2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2`
.conv4/kernel/Regularizer/Square/ReadVariableOp.conv4/kernel/Regularizer/Square/ReadVariableOp2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_9
╔

▄
/__inference_sequential_8_layer_call_fn_38371221
input_9
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
identityИвStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *1J 8В *S
fNRL
J__inference_sequential_8_layer_call_and_return_conditional_losses_383711862
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:           
!
_user_specified_name	input_9
 
}
(__inference_conv3_layer_call_fn_38371932

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_383708342
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
д
Н
C__inference_conv1_layer_call_and_return_conditional_losses_38370737

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв.conv1/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Relu╟
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv1/kernel/Regularizer/Square/ReadVariableOp╡
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv1/kernel/Regularizer/SquareЩ
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv1/kernel/Regularizer/Const▓
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/SumЕ
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv1/kernel/Regularizer/mul/x┤
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/mul╨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
єy
╒
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371607

inputs1
-batch_normalization_8_readvariableop_resource3
/batch_normalization_8_readvariableop_1_resourceB
>batch_normalization_8_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource(
$conv1_conv2d_readvariableop_resource)
%conv1_biasadd_readvariableop_resource(
$conv2_conv2d_readvariableop_resource)
%conv2_biasadd_readvariableop_resource(
$conv3_conv2d_readvariableop_resource)
%conv3_biasadd_readvariableop_resource(
$conv4_conv2d_readvariableop_resource)
%conv4_biasadd_readvariableop_resource)
%dense1_matmul_readvariableop_resource*
&dense1_biasadd_readvariableop_resource)
%dense2_matmul_readvariableop_resource*
&dense2_biasadd_readvariableop_resource
identityИв5batch_normalization_8/FusedBatchNormV3/ReadVariableOpв7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1в$batch_normalization_8/ReadVariableOpв&batch_normalization_8/ReadVariableOp_1вconv1/BiasAdd/ReadVariableOpвconv1/Conv2D/ReadVariableOpв.conv1/kernel/Regularizer/Square/ReadVariableOpвconv2/BiasAdd/ReadVariableOpвconv2/Conv2D/ReadVariableOpв.conv2/kernel/Regularizer/Square/ReadVariableOpвconv3/BiasAdd/ReadVariableOpвconv3/Conv2D/ReadVariableOpв.conv3/kernel/Regularizer/Square/ReadVariableOpвconv4/BiasAdd/ReadVariableOpвconv4/Conv2D/ReadVariableOpв.conv4/kernel/Regularizer/Square/ReadVariableOpвdense1/BiasAdd/ReadVariableOpвdense1/MatMul/ReadVariableOpвdense2/BiasAdd/ReadVariableOpвdense2/MatMul/ReadVariableOp╢
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
:*
dtype02&
$batch_normalization_8/ReadVariableOp╝
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1щ
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpя
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1╬
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3inputs,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           :::::*
epsilon%oГ:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3з
conv1/Conv2D/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv1/Conv2D/ReadVariableOp┌
conv1/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp%conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv1/BiasAdd/ReadVariableOpа
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2

conv1/Reluз
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
conv2/Conv2D/ReadVariableOp╚
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp%conv2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2/BiasAdd/ReadVariableOpа
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:          2

conv2/Relu╢
maxpool1/MaxPoolMaxPoolconv2/Relu:activations:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
2
maxpool1/MaxPoolЗ
dropout1/IdentityIdentitymaxpool1/MaxPool:output:0*
T0*/
_output_shapes
:          2
dropout1/Identityз
conv3/Conv2D/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
conv3/Conv2D/ReadVariableOp╩
conv3/Conv2DConv2Ddropout1/Identity:output:0#conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
conv3/Conv2DЮ
conv3/BiasAdd/ReadVariableOpReadVariableOp%conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv3/BiasAdd/ReadVariableOpа
conv3/BiasAddBiasAddconv3/Conv2D:output:0$conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
conv3/BiasAddr

conv3/ReluReluconv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2

conv3/Reluз
conv4/Conv2D/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
conv4/Conv2D/ReadVariableOp╚
conv4/Conv2DConv2Dconv3/Relu:activations:0#conv4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@*
paddingVALID*
strides
2
conv4/Conv2DЮ
conv4/BiasAdd/ReadVariableOpReadVariableOp%conv4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
conv4/BiasAdd/ReadVariableOpа
conv4/BiasAddBiasAddconv4/Conv2D:output:0$conv4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         

@2
conv4/BiasAddr

conv4/ReluReluconv4/BiasAdd:output:0*
T0*/
_output_shapes
:         

@2

conv4/Relu╢
maxpool2/MaxPoolMaxPoolconv4/Relu:activations:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
2
maxpool2/MaxPoolЗ
dropout2/IdentityIdentitymaxpool2/MaxPool:output:0*
T0*/
_output_shapes
:         @2
dropout2/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @  2
flatten/ConstФ
flatten/ReshapeReshapedropout2/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:         └2
flatten/Reshapeд
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource* 
_output_shapes
:
└ш*
dtype02
dense1/MatMul/ReadVariableOpЫ
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2
dense1/MatMulв
dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes	
:ш*
dtype02
dense1/BiasAdd/ReadVariableOpЮ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ш2
dense1/BiasAddn
dense1/ReluReludense1/BiasAdd:output:0*
T0*(
_output_shapes
:         ш2
dense1/ReluА
dropout3/IdentityIdentitydense1/Relu:activations:0*
T0*(
_output_shapes
:         ш2
dropout3/Identityг
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes
:	ш
*
dtype02
dense2/MatMul/ReadVariableOpЬ
dense2/MatMulMatMuldropout3/Identity:output:0$dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense2/MatMulб
dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense2/BiasAdd/ReadVariableOpЭ
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         
2
dense2/BiasAddv
dense2/SoftmaxSoftmaxdense2/BiasAdd:output:0*
T0*'
_output_shapes
:         
2
dense2/Softmax═
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv1/kernel/Regularizer/Square/ReadVariableOp╡
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv1/kernel/Regularizer/SquareЩ
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv1/kernel/Regularizer/Const▓
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/SumЕ
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv1/kernel/Regularizer/mul/x┤
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/mul═
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype020
.conv2/kernel/Regularizer/Square/ReadVariableOp╡
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2!
conv2/kernel/Regularizer/SquareЩ
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv2/kernel/Regularizer/Const▓
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/SumЕ
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv2/kernel/Regularizer/mul/x┤
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/mul═
.conv3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv3_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv3/kernel/Regularizer/Square/ReadVariableOp╡
conv3/kernel/Regularizer/SquareSquare6conv3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv3/kernel/Regularizer/SquareЩ
conv3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv3/kernel/Regularizer/Const▓
conv3/kernel/Regularizer/SumSum#conv3/kernel/Regularizer/Square:y:0'conv3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/SumЕ
conv3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv3/kernel/Regularizer/mul/x┤
conv3/kernel/Regularizer/mulMul'conv3/kernel/Regularizer/mul/x:output:0%conv3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/mul═
.conv4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp$conv4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype020
.conv4/kernel/Regularizer/Square/ReadVariableOp╡
conv4/kernel/Regularizer/SquareSquare6conv4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2!
conv4/kernel/Regularizer/SquareЩ
conv4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv4/kernel/Regularizer/Const▓
conv4/kernel/Regularizer/SumSum#conv4/kernel/Regularizer/Square:y:0'conv4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/SumЕ
conv4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv4/kernel/Regularizer/mul/x┤
conv4/kernel/Regularizer/mulMul'conv4/kernel/Regularizer/mul/x:output:0%conv4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/mulф
IdentityIdentitydense2/Softmax:softmax:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp/^conv2/kernel/Regularizer/Square/ReadVariableOp^conv3/BiasAdd/ReadVariableOp^conv3/Conv2D/ReadVariableOp/^conv3/kernel/Regularizer/Square/ReadVariableOp^conv4/BiasAdd/ReadVariableOp^conv4/Conv2D/ReadVariableOp/^conv4/kernel/Regularizer/Square/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2<
conv3/BiasAdd/ReadVariableOpconv3/BiasAdd/ReadVariableOp2:
conv3/Conv2D/ReadVariableOpconv3/Conv2D/ReadVariableOp2`
.conv3/kernel/Regularizer/Square/ReadVariableOp.conv3/kernel/Regularizer/Square/ReadVariableOp2<
conv4/BiasAdd/ReadVariableOpconv4/BiasAdd/ReadVariableOp2:
conv4/Conv2D/ReadVariableOpconv4/Conv2D/ReadVariableOp2`
.conv4/kernel/Regularizer/Square/ReadVariableOp.conv4/kernel/Regularizer/Square/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
Пd
М
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371186

inputs"
batch_normalization_8_38371116"
batch_normalization_8_38371118"
batch_normalization_8_38371120"
batch_normalization_8_38371122
conv1_38371125
conv1_38371127
conv2_38371130
conv2_38371132
conv3_38371137
conv3_38371139
conv4_38371142
conv4_38371144
dense1_38371150
dense1_38371152
dense2_38371156
dense2_38371158
identityИв-batch_normalization_8/StatefulPartitionedCallвconv1/StatefulPartitionedCallв.conv1/kernel/Regularizer/Square/ReadVariableOpвconv2/StatefulPartitionedCallв.conv2/kernel/Regularizer/Square/ReadVariableOpвconv3/StatefulPartitionedCallв.conv3/kernel/Regularizer/Square/ReadVariableOpвconv4/StatefulPartitionedCallв.conv4/kernel/Regularizer/Square/ReadVariableOpвdense1/StatefulPartitionedCallвdense2/StatefulPartitionedCallв dropout1/StatefulPartitionedCallв dropout2/StatefulPartitionedCallв dropout3/StatefulPartitionedCallк
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_38371116batch_normalization_8_38371118batch_normalization_8_38371120batch_normalization_8_38371122*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_383706662/
-batch_normalization_8/StatefulPartitionedCall╚
conv1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv1_38371125conv1_38371127*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_383707372
conv1/StatefulPartitionedCall╕
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_38371130conv2_38371132*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_383707702
conv2/StatefulPartitionedCallГ
maxpool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool1_layer_call_and_return_conditional_losses_383706252
maxpool1/PartitionedCallЦ
 dropout1/StatefulPartitionedCallStatefulPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_383707992"
 dropout1/StatefulPartitionedCall╗
conv3/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0conv3_38371137conv3_38371139*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_383708342
conv3/StatefulPartitionedCall╕
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0conv4_38371142conv4_38371144*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_383708672
conv4/StatefulPartitionedCallГ
maxpool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool2_layer_call_and_return_conditional_losses_383706372
maxpool2/PartitionedCall╣
 dropout2/StatefulPartitionedCallStatefulPartitionedCall!maxpool2/PartitionedCall:output:0!^dropout1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_383708962"
 dropout2/StatefulPartitionedCall№
flatten/PartitionedCallPartitionedCall)dropout2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_383709202
flatten/PartitionedCall░
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_38371150dense1_38371152*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_383709392 
dense1/StatefulPartitionedCall╕
 dropout3/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0!^dropout2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_383709672"
 dropout3/StatefulPartitionedCall╕
dense2/StatefulPartitionedCallStatefulPartitionedCall)dropout3/StatefulPartitionedCall:output:0dense2_38371156dense2_38371158*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_383709962 
dense2/StatefulPartitionedCall╖
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_38371125*&
_output_shapes
: *
dtype020
.conv1/kernel/Regularizer/Square/ReadVariableOp╡
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv1/kernel/Regularizer/SquareЩ
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv1/kernel/Regularizer/Const▓
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/SumЕ
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv1/kernel/Regularizer/mul/x┤
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/mul╖
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_38371130*&
_output_shapes
:  *
dtype020
.conv2/kernel/Regularizer/Square/ReadVariableOp╡
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2!
conv2/kernel/Regularizer/SquareЩ
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv2/kernel/Regularizer/Const▓
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/SumЕ
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv2/kernel/Regularizer/mul/x┤
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/mul╖
.conv3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_38371137*&
_output_shapes
: @*
dtype020
.conv3/kernel/Regularizer/Square/ReadVariableOp╡
conv3/kernel/Regularizer/SquareSquare6conv3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv3/kernel/Regularizer/SquareЩ
conv3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv3/kernel/Regularizer/Const▓
conv3/kernel/Regularizer/SumSum#conv3/kernel/Regularizer/Square:y:0'conv3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/SumЕ
conv3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv3/kernel/Regularizer/mul/x┤
conv3/kernel/Regularizer/mulMul'conv3/kernel/Regularizer/mul/x:output:0%conv3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/mul╖
.conv4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv4_38371142*&
_output_shapes
:@@*
dtype020
.conv4/kernel/Regularizer/Square/ReadVariableOp╡
conv4/kernel/Regularizer/SquareSquare6conv4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2!
conv4/kernel/Regularizer/SquareЩ
conv4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv4/kernel/Regularizer/Const▓
conv4/kernel/Regularizer/SumSum#conv4/kernel/Regularizer/Square:y:0'conv4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/SumЕ
conv4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv4/kernel/Regularizer/mul/x┤
conv4/kernel/Regularizer/mulMul'conv4/kernel/Regularizer/mul/x:output:0%conv4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/mulЪ
IdentityIdentity'dense2/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall^conv1/StatefulPartitionedCall/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/StatefulPartitionedCall/^conv2/kernel/Regularizer/Square/ReadVariableOp^conv3/StatefulPartitionedCall/^conv3/kernel/Regularizer/Square/ReadVariableOp^conv4/StatefulPartitionedCall/^conv4/kernel/Regularizer/Square/ReadVariableOp^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall!^dropout2/StatefulPartitionedCall!^dropout3/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2`
.conv3/kernel/Regularizer/Square/ReadVariableOp.conv3/kernel/Regularizer/Square/ReadVariableOp2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2`
.conv4/kernel/Regularizer/Square/ReadVariableOp.conv4/kernel/Regularizer/Square/ReadVariableOp2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2D
 dropout2/StatefulPartitionedCall dropout2/StatefulPartitionedCall2D
 dropout3/StatefulPartitionedCall dropout3/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
─
e
F__inference_dropout2_layer_call_and_return_conditional_losses_38370896

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:         @2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:         @*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:         @2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:         @2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         @2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*.
_input_shapes
:         @:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
х
~
)__inference_dense1_layer_call_fn_38372022

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_383709392
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ш2

Identity"
identityIdentity:output:0*/
_input_shapes
:         └::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         └
 
_user_specified_nameinputs
Л
e
F__inference_dropout3_layer_call_and_return_conditional_losses_38372034

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ш2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╡
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ш*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ш2
dropout/GreaterEqualА
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ш2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ш2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ш2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ш:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
╟_
г
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371296

inputs"
batch_normalization_8_38371226"
batch_normalization_8_38371228"
batch_normalization_8_38371230"
batch_normalization_8_38371232
conv1_38371235
conv1_38371237
conv2_38371240
conv2_38371242
conv3_38371247
conv3_38371249
conv4_38371252
conv4_38371254
dense1_38371260
dense1_38371262
dense2_38371266
dense2_38371268
identityИв-batch_normalization_8/StatefulPartitionedCallвconv1/StatefulPartitionedCallв.conv1/kernel/Regularizer/Square/ReadVariableOpвconv2/StatefulPartitionedCallв.conv2/kernel/Regularizer/Square/ReadVariableOpвconv3/StatefulPartitionedCallв.conv3/kernel/Regularizer/Square/ReadVariableOpвconv4/StatefulPartitionedCallв.conv4/kernel/Regularizer/Square/ReadVariableOpвdense1/StatefulPartitionedCallвdense2/StatefulPartitionedCallм
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_38371226batch_normalization_8_38371228batch_normalization_8_38371230batch_normalization_8_38371232*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_383706842/
-batch_normalization_8/StatefulPartitionedCall╚
conv1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv1_38371235conv1_38371237*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv1_layer_call_and_return_conditional_losses_383707372
conv1/StatefulPartitionedCall╕
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_38371240conv2_38371242*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_383707702
conv2/StatefulPartitionedCallГ
maxpool1/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool1_layer_call_and_return_conditional_losses_383706252
maxpool1/PartitionedCall■
dropout1/PartitionedCallPartitionedCall!maxpool1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout1_layer_call_and_return_conditional_losses_383708042
dropout1/PartitionedCall│
conv3/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0conv3_38371247conv3_38371249*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv3_layer_call_and_return_conditional_losses_383708342
conv3/StatefulPartitionedCall╕
conv4/StatefulPartitionedCallStatefulPartitionedCall&conv3/StatefulPartitionedCall:output:0conv4_38371252conv4_38371254*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         

@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv4_layer_call_and_return_conditional_losses_383708672
conv4/StatefulPartitionedCallГ
maxpool2/PartitionedCallPartitionedCall&conv4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool2_layer_call_and_return_conditional_losses_383706372
maxpool2/PartitionedCall■
dropout2/PartitionedCallPartitionedCall!maxpool2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout2_layer_call_and_return_conditional_losses_383709012
dropout2/PartitionedCallЇ
flatten/PartitionedCallPartitionedCall!dropout2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         └* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_383709202
flatten/PartitionedCall░
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_38371260dense1_38371262*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense1_layer_call_and_return_conditional_losses_383709392 
dense1/StatefulPartitionedCall¤
dropout3/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ш* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_dropout3_layer_call_and_return_conditional_losses_383709722
dropout3/PartitionedCall░
dense2/StatefulPartitionedCallStatefulPartitionedCall!dropout3/PartitionedCall:output:0dense2_38371266dense2_38371268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *M
fHRF
D__inference_dense2_layer_call_and_return_conditional_losses_383709962 
dense2/StatefulPartitionedCall╖
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv1_38371235*&
_output_shapes
: *
dtype020
.conv1/kernel/Regularizer/Square/ReadVariableOp╡
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv1/kernel/Regularizer/SquareЩ
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv1/kernel/Regularizer/Const▓
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/SumЕ
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv1/kernel/Regularizer/mul/x┤
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/mul╖
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2_38371240*&
_output_shapes
:  *
dtype020
.conv2/kernel/Regularizer/Square/ReadVariableOp╡
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2!
conv2/kernel/Regularizer/SquareЩ
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv2/kernel/Regularizer/Const▓
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/SumЕ
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv2/kernel/Regularizer/mul/x┤
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/mul╖
.conv3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv3_38371247*&
_output_shapes
: @*
dtype020
.conv3/kernel/Regularizer/Square/ReadVariableOp╡
conv3/kernel/Regularizer/SquareSquare6conv3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv3/kernel/Regularizer/SquareЩ
conv3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv3/kernel/Regularizer/Const▓
conv3/kernel/Regularizer/SumSum#conv3/kernel/Regularizer/Square:y:0'conv3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/SumЕ
conv3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv3/kernel/Regularizer/mul/x┤
conv3/kernel/Regularizer/mulMul'conv3/kernel/Regularizer/mul/x:output:0%conv3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/mul╖
.conv4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv4_38371252*&
_output_shapes
:@@*
dtype020
.conv4/kernel/Regularizer/Square/ReadVariableOp╡
conv4/kernel/Regularizer/SquareSquare6conv4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2!
conv4/kernel/Regularizer/SquareЩ
conv4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv4/kernel/Regularizer/Const▓
conv4/kernel/Regularizer/SumSum#conv4/kernel/Regularizer/Square:y:0'conv4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/SumЕ
conv4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv4/kernel/Regularizer/mul/x┤
conv4/kernel/Regularizer/mulMul'conv4/kernel/Regularizer/mul/x:output:0%conv4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv4/kernel/Regularizer/mul▒
IdentityIdentity'dense2/StatefulPartitionedCall:output:0.^batch_normalization_8/StatefulPartitionedCall^conv1/StatefulPartitionedCall/^conv1/kernel/Regularizer/Square/ReadVariableOp^conv2/StatefulPartitionedCall/^conv2/kernel/Regularizer/Square/ReadVariableOp^conv3/StatefulPartitionedCall/^conv3/kernel/Regularizer/Square/ReadVariableOp^conv4/StatefulPartitionedCall/^conv4/kernel/Regularizer/Square/ReadVariableOp^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp2>
conv3/StatefulPartitionedCallconv3/StatefulPartitionedCall2`
.conv3/kernel/Regularizer/Square/ReadVariableOp.conv3/kernel/Regularizer/Square/ReadVariableOp2>
conv4/StatefulPartitionedCallconv4/StatefulPartitionedCall2`
.conv4/kernel/Regularizer/Square/ReadVariableOp.conv4/kernel/Regularizer/Square/ReadVariableOp2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
√
Ў
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38370684

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╩
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           :::::*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3┌
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:           2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
д
Н
C__inference_conv1_layer_call_and_return_conditional_losses_38371832

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв.conv1/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Relu╟
.conv1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv1/kernel/Regularizer/Square/ReadVariableOp╡
conv1/kernel/Regularizer/SquareSquare6conv1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv1/kernel/Regularizer/SquareЩ
conv1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv1/kernel/Regularizer/Const▓
conv1/kernel/Regularizer/SumSum#conv1/kernel/Regularizer/Square:y:0'conv1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/SumЕ
conv1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv1/kernel/Regularizer/mul/x┤
conv1/kernel/Regularizer/mulMul'conv1/kernel/Regularizer/mul/x:output:0%conv1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv1/kernel/Regularizer/mul╨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv1/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:           ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv1/kernel/Regularizer/Square/ReadVariableOp.conv1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
д
Н
C__inference_conv2_layer_call_and_return_conditional_losses_38371864

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв.conv2/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:          2
Relu╟
.conv2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype020
.conv2/kernel/Regularizer/Square/ReadVariableOp╡
conv2/kernel/Regularizer/SquareSquare6conv2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2!
conv2/kernel/Regularizer/SquareЩ
conv2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv2/kernel/Regularizer/Const▓
conv2/kernel/Regularizer/SumSum#conv2/kernel/Regularizer/Square:y:0'conv2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/SumЕ
conv2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv2/kernel/Regularizer/mul/x┤
conv2/kernel/Regularizer/mulMul'conv2/kernel/Regularizer/mul/x:output:0%conv2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2/kernel/Regularizer/mul╨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2/kernel/Regularizer/Square/ReadVariableOp.conv2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
З
Ъ
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371765

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:           2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
■
Я
__inference_loss_fn_2_38372102;
7conv3_kernel_regularizer_square_readvariableop_resource
identityИв.conv3/kernel/Regularizer/Square/ReadVariableOpр
.conv3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv3_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv3/kernel/Regularizer/Square/ReadVariableOp╡
conv3/kernel/Regularizer/SquareSquare6conv3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv3/kernel/Regularizer/SquareЩ
conv3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv3/kernel/Regularizer/Const▓
conv3/kernel/Regularizer/SumSum#conv3/kernel/Regularizer/Square:y:0'conv3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/SumЕ
conv3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv3/kernel/Regularizer/mul/x┤
conv3/kernel/Regularizer/mulMul'conv3/kernel/Regularizer/mul/x:output:0%conv3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/mulФ
IdentityIdentity conv3/kernel/Regularizer/mul:z:0/^conv3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2`
.conv3/kernel/Regularizer/Square/ReadVariableOp.conv3/kernel/Regularizer/Square/ReadVariableOp
и
G
+__inference_maxpool1_layer_call_fn_38370631

inputs
identityь
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
 *2
config_proto" 

CPU

GPU2 *1J 8В *O
fJRH
F__inference_maxpool1_layer_call_and_return_conditional_losses_383706252
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
▐
л
8__inference_batch_normalization_8_layer_call_fn_38371796

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:           *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *\
fWRU
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_383706662
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:           2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╧
Ъ
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371701

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1Р
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+                           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           
 
_user_specified_nameinputs
─
e
F__inference_dropout1_layer_call_and_return_conditional_losses_38371885

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *лкк?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:          2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╝
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:          *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2
dropout/GreaterEqual/y╞
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:          2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:          2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:          2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          :W S
/
_output_shapes
:          
 
_user_specified_nameinputs
З
Ъ
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38370666

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1╪
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:           :::::*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3н
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue╗
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1",/job:localhost/replica:0/task:0/device:GPU:0*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1■
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:           2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:           ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
╚

█
/__inference_sequential_8_layer_call_fn_38371681

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
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*2
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *1J 8В *S
fNRL
J__inference_sequential_8_layer_call_and_return_conditional_losses_383712962
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:           ::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:           
 
_user_specified_nameinputs
 
}
(__inference_conv2_layer_call_fn_38371873

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *1J 8В *L
fGRE
C__inference_conv2_layer_call_and_return_conditional_losses_383707702
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:          2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
д
Н
C__inference_conv3_layer_call_and_return_conditional_losses_38371923

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpв.conv3/kernel/Regularizer/Square/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOpд
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @2
Relu╟
.conv3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv3/kernel/Regularizer/Square/ReadVariableOp╡
conv3/kernel/Regularizer/SquareSquare6conv3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv3/kernel/Regularizer/SquareЩ
conv3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2 
conv3/kernel/Regularizer/Const▓
conv3/kernel/Regularizer/SumSum#conv3/kernel/Regularizer/Square:y:0'conv3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/SumЕ
conv3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *╖╤82 
conv3/kernel/Regularizer/mul/x┤
conv3/kernel/Regularizer/mulMul'conv3/kernel/Regularizer/mul/x:output:0%conv3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv3/kernel/Regularizer/mul╨
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv3/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:         @2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv3/kernel/Regularizer/Square/ReadVariableOp.conv3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
№
b
F__inference_maxpool2_layer_call_and_return_conditional_losses_38370637

inputs
identityн
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
═
d
F__inference_dropout3_layer_call_and_return_conditional_losses_38372039

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ш2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ш2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ш:P L
(
_output_shapes
:         ш
 
_user_specified_nameinputs
ўR
╛
!__inference__traced_save_38372253
file_prefix:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop+
'savev2_conv3_kernel_read_readvariableop)
%savev2_conv3_bias_read_readvariableop+
'savev2_conv4_kernel_read_readvariableop)
%savev2_conv4_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopF
Bsavev2_rmsprop_batch_normalization_8_gamma_rms_read_readvariableopE
Asavev2_rmsprop_batch_normalization_8_beta_rms_read_readvariableop7
3savev2_rmsprop_conv1_kernel_rms_read_readvariableop5
1savev2_rmsprop_conv1_bias_rms_read_readvariableop7
3savev2_rmsprop_conv2_kernel_rms_read_readvariableop5
1savev2_rmsprop_conv2_bias_rms_read_readvariableop7
3savev2_rmsprop_conv3_kernel_rms_read_readvariableop5
1savev2_rmsprop_conv3_bias_rms_read_readvariableop7
3savev2_rmsprop_conv4_kernel_rms_read_readvariableop5
1savev2_rmsprop_conv4_bias_rms_read_readvariableop8
4savev2_rmsprop_dense1_kernel_rms_read_readvariableop6
2savev2_rmsprop_dense1_bias_rms_read_readvariableop8
4savev2_rmsprop_dense2_kernel_rms_read_readvariableop6
2savev2_rmsprop_dense2_bias_rms_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameБ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*У
valueЙBЖ(B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╪
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:(*
dtype0*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesХ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop'savev2_conv3_kernel_read_readvariableop%savev2_conv3_bias_read_readvariableop'savev2_conv4_kernel_read_readvariableop%savev2_conv4_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopBsavev2_rmsprop_batch_normalization_8_gamma_rms_read_readvariableopAsavev2_rmsprop_batch_normalization_8_beta_rms_read_readvariableop3savev2_rmsprop_conv1_kernel_rms_read_readvariableop1savev2_rmsprop_conv1_bias_rms_read_readvariableop3savev2_rmsprop_conv2_kernel_rms_read_readvariableop1savev2_rmsprop_conv2_bias_rms_read_readvariableop3savev2_rmsprop_conv3_kernel_rms_read_readvariableop1savev2_rmsprop_conv3_bias_rms_read_readvariableop3savev2_rmsprop_conv4_kernel_rms_read_readvariableop1savev2_rmsprop_conv4_bias_rms_read_readvariableop4savev2_rmsprop_dense1_kernel_rms_read_readvariableop2savev2_rmsprop_dense1_bias_rms_read_readvariableop4savev2_rmsprop_dense2_kernel_rms_read_readvariableop2savev2_rmsprop_dense2_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *6
dtypes,
*2(	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*╫
_input_shapes┼
┬: ::::: : :  : : @:@:@@:@:
└ш:ш:	ш
:
: : : : : : : : : ::: : :  : : @:@:@@:@:
└ш:ш:	ш
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
: @: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
└ш:!

_output_shapes	
:ш:%!

_output_shapes
:	ш
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :
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
: : 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :, (
&
_output_shapes
: @: !

_output_shapes
:@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@:&$"
 
_output_shapes
:
└ш:!%

_output_shapes	
:ш:%&!

_output_shapes
:	ш
: '

_output_shapes
:
:(

_output_shapes
: "▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▒
serving_defaultЭ
C
input_98
serving_default_input_9:0           :
dense20
StatefulPartitionedCall:0         
tensorflow/serving/predict:╚и
їe
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
╜_default_save_signature
╛__call__
+┐&call_and_return_all_conditional_losses"┼a
_tf_keras_sequentialжa{"class_name": "Sequential", "name": "sequential_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout3", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_8", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_9"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Conv2D", "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout3", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}}, "metrics": [[{"class_name": "SparseCategoricalAccuracy", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
║	
axis
	gamma
beta
moving_mean
moving_variance
trainable_variables
regularization_losses
	variables
	keras_api
└__call__
+┴&call_and_return_all_conditional_losses"ф
_tf_keras_layer╩{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
ж


kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
┬__call__
+├&call_and_return_all_conditional_losses" 
_tf_keras_layerх{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
и


#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
─__call__
+┼&call_and_return_all_conditional_losses"Б	
_tf_keras_layerч{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 30, 30, 32]}}
є
)trainable_variables
*regularization_losses
+	variables
,	keras_api
╞__call__
+╟&call_and_return_all_conditional_losses"т
_tf_keras_layer╚{"class_name": "MaxPooling2D", "name": "maxpool1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ц
-trainable_variables
.regularization_losses
/	variables
0	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses"╒
_tf_keras_layer╗{"class_name": "Dropout", "name": "dropout1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout1", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
и


1kernel
2bias
3trainable_variables
4regularization_losses
5	variables
6	keras_api
╩__call__
+╦&call_and_return_all_conditional_losses"Б	
_tf_keras_layerч{"class_name": "Conv2D", "name": "conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
и


7kernel
8bias
9trainable_variables
:regularization_losses
;	variables
<	keras_api
╠__call__
+═&call_and_return_all_conditional_losses"Б	
_tf_keras_layerч{"class_name": "Conv2D", "name": "conv4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 64]}}
є
=trainable_variables
>regularization_losses
?	variables
@	keras_api
╬__call__
+╧&call_and_return_all_conditional_losses"т
_tf_keras_layer╚{"class_name": "MaxPooling2D", "name": "maxpool2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "maxpool2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ц
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
╨__call__
+╤&call_and_return_all_conditional_losses"╒
_tf_keras_layer╗{"class_name": "Dropout", "name": "dropout2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout2", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ф
Etrainable_variables
Fregularization_losses
G	variables
H	keras_api
╥__call__
+╙&call_and_return_all_conditional_losses"╙
_tf_keras_layer╣{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ў

Ikernel
Jbias
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
╘__call__
+╒&call_and_return_all_conditional_losses"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 1000, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1600}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1600]}}
ц
Otrainable_variables
Pregularization_losses
Q	variables
R	keras_api
╓__call__
+╫&call_and_return_all_conditional_losses"╒
_tf_keras_layer╗{"class_name": "Dropout", "name": "dropout3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout3", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}
ў

Skernel
Tbias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
╪__call__
+┘&call_and_return_all_conditional_losses"╨
_tf_keras_layer╢{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
·
Yiter
	Zdecay
[learning_rate
\momentum
]rho
rmsп
rms░
rms▒
rms▓
#rms│
$rms┤
1rms╡
2rms╢
7rms╖
8rms╕
Irms╣
Jrms║
Srms╗
Trms╝"
	optimizer
Ж
0
1
2
3
#4
$5
16
27
78
89
I10
J11
S12
T13"
trackable_list_wrapper
@
┌0
█1
▄2
▌3"
trackable_list_wrapper
Ц
0
1
2
3
4
5
#6
$7
18
29
710
811
I12
J13
S14
T15"
trackable_list_wrapper
╬
trainable_variables
^non_trainable_variables

_layers
`layer_metrics
regularization_losses
ametrics
blayer_regularization_losses
	variables
╛__call__
╜_default_save_signature
+┐&call_and_return_all_conditional_losses
'┐"call_and_return_conditional_losses"
_generic_user_object
-
▐serving_default"
signature_map
 "
trackable_list_wrapper
):'2batch_normalization_8/gamma
(:&2batch_normalization_8/beta
1:/ (2!batch_normalization_8/moving_mean
5:3 (2%batch_normalization_8/moving_variance
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
░
trainable_variables
cnon_trainable_variables
dlayer_metrics
regularization_losses
emetrics

flayers
glayer_regularization_losses
	variables
└__call__
+┴&call_and_return_all_conditional_losses
'┴"call_and_return_conditional_losses"
_generic_user_object
&:$ 2conv1/kernel
: 2
conv1/bias
.
0
1"
trackable_list_wrapper
(
┌0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
trainable_variables
hnon_trainable_variables
ilayer_metrics
 regularization_losses
jmetrics

klayers
llayer_regularization_losses
!	variables
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
&:$  2conv2/kernel
: 2
conv2/bias
.
#0
$1"
trackable_list_wrapper
(
█0"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
░
%trainable_variables
mnon_trainable_variables
nlayer_metrics
&regularization_losses
ometrics

players
qlayer_regularization_losses
'	variables
─__call__
+┼&call_and_return_all_conditional_losses
'┼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
)trainable_variables
rnon_trainable_variables
slayer_metrics
*regularization_losses
tmetrics

ulayers
vlayer_regularization_losses
+	variables
╞__call__
+╟&call_and_return_all_conditional_losses
'╟"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
-trainable_variables
wnon_trainable_variables
xlayer_metrics
.regularization_losses
ymetrics

zlayers
{layer_regularization_losses
/	variables
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
&:$ @2conv3/kernel
:@2
conv3/bias
.
10
21"
trackable_list_wrapper
(
▄0"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
▒
3trainable_variables
|non_trainable_variables
}layer_metrics
4regularization_losses
~metrics

layers
 Аlayer_regularization_losses
5	variables
╩__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
&:$@@2conv4/kernel
:@2
conv4/bias
.
70
81"
trackable_list_wrapper
(
▌0"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
╡
9trainable_variables
Бnon_trainable_variables
Вlayer_metrics
:regularization_losses
Гmetrics
Дlayers
 Еlayer_regularization_losses
;	variables
╠__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
=trainable_variables
Жnon_trainable_variables
Зlayer_metrics
>regularization_losses
Иmetrics
Йlayers
 Кlayer_regularization_losses
?	variables
╬__call__
+╧&call_and_return_all_conditional_losses
'╧"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Atrainable_variables
Лnon_trainable_variables
Мlayer_metrics
Bregularization_losses
Нmetrics
Оlayers
 Пlayer_regularization_losses
C	variables
╨__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Etrainable_variables
Рnon_trainable_variables
Сlayer_metrics
Fregularization_losses
Тmetrics
Уlayers
 Фlayer_regularization_losses
G	variables
╥__call__
+╙&call_and_return_all_conditional_losses
'╙"call_and_return_conditional_losses"
_generic_user_object
!:
└ш2dense1/kernel
:ш2dense1/bias
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
╡
Ktrainable_variables
Хnon_trainable_variables
Цlayer_metrics
Lregularization_losses
Чmetrics
Шlayers
 Щlayer_regularization_losses
M	variables
╘__call__
+╒&call_and_return_all_conditional_losses
'╒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Otrainable_variables
Ъnon_trainable_variables
Ыlayer_metrics
Pregularization_losses
Ьmetrics
Эlayers
 Юlayer_regularization_losses
Q	variables
╓__call__
+╫&call_and_return_all_conditional_losses
'╫"call_and_return_conditional_losses"
_generic_user_object
 :	ш
2dense2/kernel
:
2dense2/bias
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
╡
Utrainable_variables
Яnon_trainable_variables
аlayer_metrics
Vregularization_losses
бmetrics
вlayers
 гlayer_regularization_losses
W	variables
╪__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
.
0
1"
trackable_list_wrapper
~
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
12"
trackable_list_wrapper
 "
trackable_dict_wrapper
0
д0
е1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
┌0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
█0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
(
▄0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
▌0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
┐

жtotal

зcount
и	variables
й	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ф

кtotal

лcount
м
_fn_kwargs
н	variables
о	keras_api"╚
_tf_keras_metricн{"class_name": "SparseCategoricalAccuracy", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
:  (2total
:  (2count
0
ж0
з1"
trackable_list_wrapper
.
и	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
к0
л1"
trackable_list_wrapper
.
н	variables"
_generic_user_object
3:12'RMSprop/batch_normalization_8/gamma/rms
2:02&RMSprop/batch_normalization_8/beta/rms
0:. 2RMSprop/conv1/kernel/rms
":  2RMSprop/conv1/bias/rms
0:.  2RMSprop/conv2/kernel/rms
":  2RMSprop/conv2/bias/rms
0:. @2RMSprop/conv3/kernel/rms
": @2RMSprop/conv3/bias/rms
0:.@@2RMSprop/conv4/kernel/rms
": @2RMSprop/conv4/bias/rms
+:)
└ш2RMSprop/dense1/kernel/rms
$:"ш2RMSprop/dense1/bias/rms
*:(	ш
2RMSprop/dense2/kernel/rms
#:!
2RMSprop/dense2/bias/rms
щ2ц
#__inference__wrapped_model_38370515╛
Л▓З
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
annotationsк *.в+
)К&
input_9           
К2З
/__inference_sequential_8_layer_call_fn_38371221
/__inference_sequential_8_layer_call_fn_38371644
/__inference_sequential_8_layer_call_fn_38371681
/__inference_sequential_8_layer_call_fn_38371331└
╖▓│
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
kwonlydefaultsк 
annotationsк *
 
Ў2є
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371037
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371516
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371110
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371607└
╖▓│
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
kwonlydefaultsк 
annotationsк *
 
в2Я
8__inference_batch_normalization_8_layer_call_fn_38371745
8__inference_batch_normalization_8_layer_call_fn_38371809
8__inference_batch_normalization_8_layer_call_fn_38371732
8__inference_batch_normalization_8_layer_call_fn_38371796┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
О2Л
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371701
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371719
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371783
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371765┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_conv1_layer_call_fn_38371841в
Щ▓Х
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
annotationsк *
 
э2ъ
C__inference_conv1_layer_call_and_return_conditional_losses_38371832в
Щ▓Х
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
annotationsк *
 
╥2╧
(__inference_conv2_layer_call_fn_38371873в
Щ▓Х
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
annotationsк *
 
э2ъ
C__inference_conv2_layer_call_and_return_conditional_losses_38371864в
Щ▓Х
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
annotationsк *
 
У2Р
+__inference_maxpool1_layer_call_fn_38370631р
Щ▓Х
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
annotationsк *@в=
;К84                                    
о2л
F__inference_maxpool1_layer_call_and_return_conditional_losses_38370625р
Щ▓Х
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
annotationsк *@в=
;К84                                    
Ф2С
+__inference_dropout1_layer_call_fn_38371895
+__inference_dropout1_layer_call_fn_38371900┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout1_layer_call_and_return_conditional_losses_38371890
F__inference_dropout1_layer_call_and_return_conditional_losses_38371885┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_conv3_layer_call_fn_38371932в
Щ▓Х
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
annotationsк *
 
э2ъ
C__inference_conv3_layer_call_and_return_conditional_losses_38371923в
Щ▓Х
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
annotationsк *
 
╥2╧
(__inference_conv4_layer_call_fn_38371964в
Щ▓Х
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
annotationsк *
 
э2ъ
C__inference_conv4_layer_call_and_return_conditional_losses_38371955в
Щ▓Х
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
annotationsк *
 
У2Р
+__inference_maxpool2_layer_call_fn_38370643р
Щ▓Х
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
annotationsк *@в=
;К84                                    
о2л
F__inference_maxpool2_layer_call_and_return_conditional_losses_38370637р
Щ▓Х
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
annotationsк *@в=
;К84                                    
Ф2С
+__inference_dropout2_layer_call_fn_38371986
+__inference_dropout2_layer_call_fn_38371991┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout2_layer_call_and_return_conditional_losses_38371976
F__inference_dropout2_layer_call_and_return_conditional_losses_38371981┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╘2╤
*__inference_flatten_layer_call_fn_38372002в
Щ▓Х
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
annotationsк *
 
я2ь
E__inference_flatten_layer_call_and_return_conditional_losses_38371997в
Щ▓Х
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
annotationsк *
 
╙2╨
)__inference_dense1_layer_call_fn_38372022в
Щ▓Х
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
annotationsк *
 
ю2ы
D__inference_dense1_layer_call_and_return_conditional_losses_38372013в
Щ▓Х
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
annotationsк *
 
Ф2С
+__inference_dropout3_layer_call_fn_38372049
+__inference_dropout3_layer_call_fn_38372044┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╩2╟
F__inference_dropout3_layer_call_and_return_conditional_losses_38372039
F__inference_dropout3_layer_call_and_return_conditional_losses_38372034┤
л▓з
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
kwonlydefaultsк 
annotationsк *
 
╙2╨
)__inference_dense2_layer_call_fn_38372069в
Щ▓Х
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
annotationsк *
 
ю2ы
D__inference_dense2_layer_call_and_return_conditional_losses_38372060в
Щ▓Х
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
annotationsк *
 
╡2▓
__inference_loss_fn_0_38372080П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡2▓
__inference_loss_fn_1_38372091П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡2▓
__inference_loss_fn_2_38372102П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╡2▓
__inference_loss_fn_3_38372113П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
═B╩
&__inference_signature_wrapper_38371402input_9"Ф
Н▓Й
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
annotationsк *
 д
#__inference__wrapped_model_38370515}#$1278IJST8в5
.в+
)К&
input_9           
к "/к,
*
dense2 К
dense2         
ю
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371701ЦMвJ
Cв@
:К7
inputs+                           
p
к "?в<
5К2
0+                           
Ъ ю
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371719ЦMвJ
Cв@
:К7
inputs+                           
p 
к "?в<
5К2
0+                           
Ъ ╔
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371765r;в8
1в.
(К%
inputs           
p
к "-в*
#К 
0           
Ъ ╔
S__inference_batch_normalization_8_layer_call_and_return_conditional_losses_38371783r;в8
1в.
(К%
inputs           
p 
к "-в*
#К 
0           
Ъ ╞
8__inference_batch_normalization_8_layer_call_fn_38371732ЙMвJ
Cв@
:К7
inputs+                           
p
к "2К/+                           ╞
8__inference_batch_normalization_8_layer_call_fn_38371745ЙMвJ
Cв@
:К7
inputs+                           
p 
к "2К/+                           б
8__inference_batch_normalization_8_layer_call_fn_38371796e;в8
1в.
(К%
inputs           
p
к " К           б
8__inference_batch_normalization_8_layer_call_fn_38371809e;в8
1в.
(К%
inputs           
p 
к " К           │
C__inference_conv1_layer_call_and_return_conditional_losses_38371832l7в4
-в*
(К%
inputs           
к "-в*
#К 
0          
Ъ Л
(__inference_conv1_layer_call_fn_38371841_7в4
-в*
(К%
inputs           
к " К          │
C__inference_conv2_layer_call_and_return_conditional_losses_38371864l#$7в4
-в*
(К%
inputs          
к "-в*
#К 
0          
Ъ Л
(__inference_conv2_layer_call_fn_38371873_#$7в4
-в*
(К%
inputs          
к " К          │
C__inference_conv3_layer_call_and_return_conditional_losses_38371923l127в4
-в*
(К%
inputs          
к "-в*
#К 
0         @
Ъ Л
(__inference_conv3_layer_call_fn_38371932_127в4
-в*
(К%
inputs          
к " К         @│
C__inference_conv4_layer_call_and_return_conditional_losses_38371955l787в4
-в*
(К%
inputs         @
к "-в*
#К 
0         

@
Ъ Л
(__inference_conv4_layer_call_fn_38371964_787в4
-в*
(К%
inputs         @
к " К         

@ж
D__inference_dense1_layer_call_and_return_conditional_losses_38372013^IJ0в-
&в#
!К
inputs         └
к "&в#
К
0         ш
Ъ ~
)__inference_dense1_layer_call_fn_38372022QIJ0в-
&в#
!К
inputs         └
к "К         ше
D__inference_dense2_layer_call_and_return_conditional_losses_38372060]ST0в-
&в#
!К
inputs         ш
к "%в"
К
0         

Ъ }
)__inference_dense2_layer_call_fn_38372069PST0в-
&в#
!К
inputs         ш
к "К         
╢
F__inference_dropout1_layer_call_and_return_conditional_losses_38371885l;в8
1в.
(К%
inputs          
p
к "-в*
#К 
0          
Ъ ╢
F__inference_dropout1_layer_call_and_return_conditional_losses_38371890l;в8
1в.
(К%
inputs          
p 
к "-в*
#К 
0          
Ъ О
+__inference_dropout1_layer_call_fn_38371895_;в8
1в.
(К%
inputs          
p
к " К          О
+__inference_dropout1_layer_call_fn_38371900_;в8
1в.
(К%
inputs          
p 
к " К          ╢
F__inference_dropout2_layer_call_and_return_conditional_losses_38371976l;в8
1в.
(К%
inputs         @
p
к "-в*
#К 
0         @
Ъ ╢
F__inference_dropout2_layer_call_and_return_conditional_losses_38371981l;в8
1в.
(К%
inputs         @
p 
к "-в*
#К 
0         @
Ъ О
+__inference_dropout2_layer_call_fn_38371986_;в8
1в.
(К%
inputs         @
p
к " К         @О
+__inference_dropout2_layer_call_fn_38371991_;в8
1в.
(К%
inputs         @
p 
к " К         @и
F__inference_dropout3_layer_call_and_return_conditional_losses_38372034^4в1
*в'
!К
inputs         ш
p
к "&в#
К
0         ш
Ъ и
F__inference_dropout3_layer_call_and_return_conditional_losses_38372039^4в1
*в'
!К
inputs         ш
p 
к "&в#
К
0         ш
Ъ А
+__inference_dropout3_layer_call_fn_38372044Q4в1
*в'
!К
inputs         ш
p
к "К         шА
+__inference_dropout3_layer_call_fn_38372049Q4в1
*в'
!К
inputs         ш
p 
к "К         шк
E__inference_flatten_layer_call_and_return_conditional_losses_38371997a7в4
-в*
(К%
inputs         @
к "&в#
К
0         └
Ъ В
*__inference_flatten_layer_call_fn_38372002T7в4
-в*
(К%
inputs         @
к "К         └=
__inference_loss_fn_0_38372080в

в 
к "К =
__inference_loss_fn_1_38372091#в

в 
к "К =
__inference_loss_fn_2_383721021в

в 
к "К =
__inference_loss_fn_3_383721137в

в 
к "К щ
F__inference_maxpool1_layer_call_and_return_conditional_losses_38370625ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┴
+__inference_maxpool1_layer_call_fn_38370631СRвO
HвE
CК@
inputs4                                    
к ";К84                                    щ
F__inference_maxpool2_layer_call_and_return_conditional_losses_38370637ЮRвO
HвE
CК@
inputs4                                    
к "HвE
>К;
04                                    
Ъ ┴
+__inference_maxpool2_layer_call_fn_38370643СRвO
HвE
CК@
inputs4                                    
к ";К84                                    ╔
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371037{#$1278IJST@в=
6в3
)К&
input_9           
p

 
к "%в"
К
0         

Ъ ╔
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371110{#$1278IJST@в=
6в3
)К&
input_9           
p 

 
к "%в"
К
0         

Ъ ╚
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371516z#$1278IJST?в<
5в2
(К%
inputs           
p

 
к "%в"
К
0         

Ъ ╚
J__inference_sequential_8_layer_call_and_return_conditional_losses_38371607z#$1278IJST?в<
5в2
(К%
inputs           
p 

 
к "%в"
К
0         

Ъ б
/__inference_sequential_8_layer_call_fn_38371221n#$1278IJST@в=
6в3
)К&
input_9           
p

 
к "К         
б
/__inference_sequential_8_layer_call_fn_38371331n#$1278IJST@в=
6в3
)К&
input_9           
p 

 
к "К         
а
/__inference_sequential_8_layer_call_fn_38371644m#$1278IJST?в<
5в2
(К%
inputs           
p

 
к "К         
а
/__inference_sequential_8_layer_call_fn_38371681m#$1278IJST?в<
5в2
(К%
inputs           
p 

 
к "К         
│
&__inference_signature_wrapper_38371402И#$1278IJSTCв@
в 
9к6
4
input_9)К&
input_9           "/к,
*
dense2 К
dense2         
