σά
Ο!΅!
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	
ξ
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
k
BatchMatMulV2
x"T
y"T
output"T"
Ttype:

2	"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
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
8
Maximum
x"T
y"T
z"T"
Ttype:

2	

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
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
1
Square
x"T
y"T"
Ttype:

2	
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

StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
j
SymbolicGradient
input2Tin
output2Tout"
Tin
list(type)(0"
Tout
list(type)(0"	
ffunc
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*2.0.02unknown?
j
input_XPlaceholder*
shape:?????????*
dtype0*'
_output_shapes
:?????????
j
input_TPlaceholder*'
_output_shapes
:?????????*
shape:?????????*
dtype0
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:?????????*
shape:?????????
]
strided_slice/stackConst*
valueB: *
_output_shapes
:*
dtype0
`
strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
_
strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

strided_sliceStridedSliceinput_Xstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
ellipsis_mask *
T0*
end_mask *'
_output_shapes
:?????????*
Index0*
shrink_axis_mask *

begin_mask *
new_axis_mask 
_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
b
strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
a
strided_slice_1/stack_2Const*
valueB:*
_output_shapes
:*
dtype0

strided_slice_1StridedSliceinput_Tstrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*'
_output_shapes
:?????????*
shrink_axis_mask *
T0*
end_mask *
ellipsis_mask *

begin_mask *
Index0*
new_axis_mask 
_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
b
strided_slice_2/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
a
strided_slice_2/stack_2Const*
dtype0*
valueB:*
_output_shapes
:

strided_slice_2StridedSlicePlaceholderstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*

begin_mask *
new_axis_mask *
end_mask *
shrink_axis_mask *
ellipsis_mask *
Index0*'
_output_shapes
:?????????*
T0
©
3NIN/dense/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*#
_class
loc:@NIN/dense/kernel*
dtype0*
valueB"      

2NIN/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@NIN/dense/kernel*
valueB
 *    *
_output_shapes
: 

4NIN/dense/kernel/Initializer/truncated_normal/stddevConst*#
_class
loc:@NIN/dense/kernel*
valueB
 *ΝΜΜ=*
dtype0*
_output_shapes
: 
ω
=NIN/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3NIN/dense/kernel/Initializer/truncated_normal/shape*#
_class
loc:@NIN/dense/kernel*
T0*

seed *
seed2 *
dtype0*
_output_shapes

:
ϋ
1NIN/dense/kernel/Initializer/truncated_normal/mulMul=NIN/dense/kernel/Initializer/truncated_normal/TruncatedNormal4NIN/dense/kernel/Initializer/truncated_normal/stddev*#
_class
loc:@NIN/dense/kernel*
_output_shapes

:*
T0
ι
-NIN/dense/kernel/Initializer/truncated_normalAdd1NIN/dense/kernel/Initializer/truncated_normal/mul2NIN/dense/kernel/Initializer/truncated_normal/mean*#
_class
loc:@NIN/dense/kernel*
_output_shapes

:*
T0
©
NIN/dense/kernel
VariableV2*
shared_name *
dtype0*
	container *
_output_shapes

:*#
_class
loc:@NIN/dense/kernel*
shape
:
Ω
NIN/dense/kernel/AssignAssignNIN/dense/kernel-NIN/dense/kernel/Initializer/truncated_normal*
use_locking(*
_output_shapes

:*#
_class
loc:@NIN/dense/kernel*
T0*
validate_shape(

NIN/dense/kernel/readIdentityNIN/dense/kernel*
_output_shapes

:*#
_class
loc:@NIN/dense/kernel*
T0

1NIN/dense/bias/Initializer/truncated_normal/shapeConst*
dtype0*!
_class
loc:@NIN/dense/bias*
_output_shapes
:*
valueB:

0NIN/dense/bias/Initializer/truncated_normal/meanConst*!
_class
loc:@NIN/dense/bias*
dtype0*
_output_shapes
: *
valueB
 *    

2NIN/dense/bias/Initializer/truncated_normal/stddevConst*!
_class
loc:@NIN/dense/bias*
dtype0*
_output_shapes
: *
valueB
 *ΝΜΜ=
ο
;NIN/dense/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal1NIN/dense/bias/Initializer/truncated_normal/shape*

seed *
_output_shapes
:*
T0*!
_class
loc:@NIN/dense/bias*
seed2 *
dtype0
ο
/NIN/dense/bias/Initializer/truncated_normal/mulMul;NIN/dense/bias/Initializer/truncated_normal/TruncatedNormal2NIN/dense/bias/Initializer/truncated_normal/stddev*!
_class
loc:@NIN/dense/bias*
_output_shapes
:*
T0
έ
+NIN/dense/bias/Initializer/truncated_normalAdd/NIN/dense/bias/Initializer/truncated_normal/mul0NIN/dense/bias/Initializer/truncated_normal/mean*
_output_shapes
:*!
_class
loc:@NIN/dense/bias*
T0

NIN/dense/bias
VariableV2*
shape:*
	container *
_output_shapes
:*
shared_name *
dtype0*!
_class
loc:@NIN/dense/bias
Ν
NIN/dense/bias/AssignAssignNIN/dense/bias+NIN/dense/bias/Initializer/truncated_normal*
validate_shape(*
use_locking(*!
_class
loc:@NIN/dense/bias*
_output_shapes
:*
T0
w
NIN/dense/bias/readIdentityNIN/dense/bias*
T0*!
_class
loc:@NIN/dense/bias*
_output_shapes
:

NIN/dense/MatMulMatMulstrided_slice_1NIN/dense/kernel/read*'
_output_shapes
:?????????*
T0*
transpose_b( *
transpose_a( 

NIN/dense/BiasAddBiasAddNIN/dense/MatMulNIN/dense/bias/read*'
_output_shapes
:?????????*
T0*
data_formatNHWC

NIN/dense/swish_f32	swish_f32NIN/dense/BiasAdd*#
_disable_call_shape_inference(*'
_output_shapes
:?????????
­
5NIN/dense_1/kernel/Initializer/truncated_normal/shapeConst*
dtype0*%
_class
loc:@NIN/dense_1/kernel*
_output_shapes
:*
valueB"      
 
4NIN/dense_1/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    *%
_class
loc:@NIN/dense_1/kernel
’
6NIN/dense_1/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *%
_class
loc:@NIN/dense_1/kernel*
valueB
 *ΝΜΜ=*
dtype0
?
?NIN/dense_1/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5NIN/dense_1/kernel/Initializer/truncated_normal/shape*
dtype0*
_output_shapes

:*%
_class
loc:@NIN/dense_1/kernel*

seed *
seed2 *
T0

3NIN/dense_1/kernel/Initializer/truncated_normal/mulMul?NIN/dense_1/kernel/Initializer/truncated_normal/TruncatedNormal6NIN/dense_1/kernel/Initializer/truncated_normal/stddev*
T0*%
_class
loc:@NIN/dense_1/kernel*
_output_shapes

:
ρ
/NIN/dense_1/kernel/Initializer/truncated_normalAdd3NIN/dense_1/kernel/Initializer/truncated_normal/mul4NIN/dense_1/kernel/Initializer/truncated_normal/mean*
_output_shapes

:*%
_class
loc:@NIN/dense_1/kernel*
T0
­
NIN/dense_1/kernel
VariableV2*
	container *
dtype0*
shared_name *
_output_shapes

:*
shape
:*%
_class
loc:@NIN/dense_1/kernel
α
NIN/dense_1/kernel/AssignAssignNIN/dense_1/kernel/NIN/dense_1/kernel/Initializer/truncated_normal*
T0*
_output_shapes

:*%
_class
loc:@NIN/dense_1/kernel*
validate_shape(*
use_locking(

NIN/dense_1/kernel/readIdentityNIN/dense_1/kernel*%
_class
loc:@NIN/dense_1/kernel*
T0*
_output_shapes

:
’
3NIN/dense_1/bias/Initializer/truncated_normal/shapeConst*
dtype0*#
_class
loc:@NIN/dense_1/bias*
valueB:*
_output_shapes
:

2NIN/dense_1/bias/Initializer/truncated_normal/meanConst*#
_class
loc:@NIN/dense_1/bias*
dtype0*
_output_shapes
: *
valueB
 *    

4NIN/dense_1/bias/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *#
_class
loc:@NIN/dense_1/bias*
valueB
 *ΝΜΜ=
υ
=NIN/dense_1/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3NIN/dense_1/bias/Initializer/truncated_normal/shape*
T0*
dtype0*#
_class
loc:@NIN/dense_1/bias*
_output_shapes
:*

seed *
seed2 
χ
1NIN/dense_1/bias/Initializer/truncated_normal/mulMul=NIN/dense_1/bias/Initializer/truncated_normal/TruncatedNormal4NIN/dense_1/bias/Initializer/truncated_normal/stddev*
_output_shapes
:*
T0*#
_class
loc:@NIN/dense_1/bias
ε
-NIN/dense_1/bias/Initializer/truncated_normalAdd1NIN/dense_1/bias/Initializer/truncated_normal/mul2NIN/dense_1/bias/Initializer/truncated_normal/mean*
T0*#
_class
loc:@NIN/dense_1/bias*
_output_shapes
:
‘
NIN/dense_1/bias
VariableV2*
dtype0*
shape:*#
_class
loc:@NIN/dense_1/bias*
	container *
shared_name *
_output_shapes
:
Υ
NIN/dense_1/bias/AssignAssignNIN/dense_1/bias-NIN/dense_1/bias/Initializer/truncated_normal*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*#
_class
loc:@NIN/dense_1/bias
}
NIN/dense_1/bias/readIdentityNIN/dense_1/bias*
_output_shapes
:*
T0*#
_class
loc:@NIN/dense_1/bias
’
NIN/dense_1/MatMulMatMulNIN/dense/swish_f32NIN/dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:?????????

NIN/dense_1/BiasAddBiasAddNIN/dense_1/MatMulNIN/dense_1/bias/read*
data_formatNHWC*'
_output_shapes
:?????????*
T0

NIN/dense_1/swish_f32	swish_f32NIN/dense_1/BiasAdd*'
_output_shapes
:?????????*#
_disable_call_shape_inference(
­
5NIN/dense_2/kernel/Initializer/truncated_normal/shapeConst*%
_class
loc:@NIN/dense_2/kernel*
_output_shapes
:*
valueB"      *
dtype0
 
4NIN/dense_2/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
dtype0*%
_class
loc:@NIN/dense_2/kernel*
valueB
 *    
’
6NIN/dense_2/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ΝΜΜ=*
dtype0*%
_class
loc:@NIN/dense_2/kernel
?
?NIN/dense_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5NIN/dense_2/kernel/Initializer/truncated_normal/shape*

seed *
T0*
seed2 *%
_class
loc:@NIN/dense_2/kernel*
dtype0*
_output_shapes

:

3NIN/dense_2/kernel/Initializer/truncated_normal/mulMul?NIN/dense_2/kernel/Initializer/truncated_normal/TruncatedNormal6NIN/dense_2/kernel/Initializer/truncated_normal/stddev*%
_class
loc:@NIN/dense_2/kernel*
T0*
_output_shapes

:
ρ
/NIN/dense_2/kernel/Initializer/truncated_normalAdd3NIN/dense_2/kernel/Initializer/truncated_normal/mul4NIN/dense_2/kernel/Initializer/truncated_normal/mean*
T0*%
_class
loc:@NIN/dense_2/kernel*
_output_shapes

:
­
NIN/dense_2/kernel
VariableV2*
_output_shapes

:*
shared_name *
	container *
shape
:*%
_class
loc:@NIN/dense_2/kernel*
dtype0
α
NIN/dense_2/kernel/AssignAssignNIN/dense_2/kernel/NIN/dense_2/kernel/Initializer/truncated_normal*
T0*
_output_shapes

:*
validate_shape(*%
_class
loc:@NIN/dense_2/kernel*
use_locking(

NIN/dense_2/kernel/readIdentityNIN/dense_2/kernel*%
_class
loc:@NIN/dense_2/kernel*
_output_shapes

:*
T0
’
3NIN/dense_2/bias/Initializer/truncated_normal/shapeConst*
valueB:*#
_class
loc:@NIN/dense_2/bias*
dtype0*
_output_shapes
:

2NIN/dense_2/bias/Initializer/truncated_normal/meanConst*
dtype0*#
_class
loc:@NIN/dense_2/bias*
valueB
 *    *
_output_shapes
: 

4NIN/dense_2/bias/Initializer/truncated_normal/stddevConst*
valueB
 *ΝΜΜ=*
dtype0*
_output_shapes
: *#
_class
loc:@NIN/dense_2/bias
υ
=NIN/dense_2/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3NIN/dense_2/bias/Initializer/truncated_normal/shape*

seed *
dtype0*#
_class
loc:@NIN/dense_2/bias*
T0*
_output_shapes
:*
seed2 
χ
1NIN/dense_2/bias/Initializer/truncated_normal/mulMul=NIN/dense_2/bias/Initializer/truncated_normal/TruncatedNormal4NIN/dense_2/bias/Initializer/truncated_normal/stddev*#
_class
loc:@NIN/dense_2/bias*
T0*
_output_shapes
:
ε
-NIN/dense_2/bias/Initializer/truncated_normalAdd1NIN/dense_2/bias/Initializer/truncated_normal/mul2NIN/dense_2/bias/Initializer/truncated_normal/mean*
T0*#
_class
loc:@NIN/dense_2/bias*
_output_shapes
:
‘
NIN/dense_2/bias
VariableV2*
shape:*#
_class
loc:@NIN/dense_2/bias*
	container *
shared_name *
dtype0*
_output_shapes
:
Υ
NIN/dense_2/bias/AssignAssignNIN/dense_2/bias-NIN/dense_2/bias/Initializer/truncated_normal*
_output_shapes
:*
T0*
use_locking(*#
_class
loc:@NIN/dense_2/bias*
validate_shape(
}
NIN/dense_2/bias/readIdentityNIN/dense_2/bias*#
_class
loc:@NIN/dense_2/bias*
_output_shapes
:*
T0
€
NIN/dense_2/MatMulMatMulNIN/dense_1/swish_f32NIN/dense_2/kernel/read*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 

NIN/dense_2/BiasAddBiasAddNIN/dense_2/MatMulNIN/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:?????????
­
5NIN/dense_3/kernel/Initializer/truncated_normal/shapeConst*
dtype0*%
_class
loc:@NIN/dense_3/kernel*
valueB"      *
_output_shapes
:
 
4NIN/dense_3/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *%
_class
loc:@NIN/dense_3/kernel*
valueB
 *    
’
6NIN/dense_3/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *ΝΜΜ=*
dtype0*
_output_shapes
: *%
_class
loc:@NIN/dense_3/kernel
?
?NIN/dense_3/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5NIN/dense_3/kernel/Initializer/truncated_normal/shape*
dtype0*
T0*%
_class
loc:@NIN/dense_3/kernel*
seed2 *
_output_shapes

:*

seed 

3NIN/dense_3/kernel/Initializer/truncated_normal/mulMul?NIN/dense_3/kernel/Initializer/truncated_normal/TruncatedNormal6NIN/dense_3/kernel/Initializer/truncated_normal/stddev*%
_class
loc:@NIN/dense_3/kernel*
T0*
_output_shapes

:
ρ
/NIN/dense_3/kernel/Initializer/truncated_normalAdd3NIN/dense_3/kernel/Initializer/truncated_normal/mul4NIN/dense_3/kernel/Initializer/truncated_normal/mean*%
_class
loc:@NIN/dense_3/kernel*
T0*
_output_shapes

:
­
NIN/dense_3/kernel
VariableV2*
shape
:*
_output_shapes

:*
dtype0*
shared_name *%
_class
loc:@NIN/dense_3/kernel*
	container 
α
NIN/dense_3/kernel/AssignAssignNIN/dense_3/kernel/NIN/dense_3/kernel/Initializer/truncated_normal*
T0*
_output_shapes

:*
validate_shape(*%
_class
loc:@NIN/dense_3/kernel*
use_locking(

NIN/dense_3/kernel/readIdentityNIN/dense_3/kernel*
T0*%
_class
loc:@NIN/dense_3/kernel*
_output_shapes

:
’
3NIN/dense_3/bias/Initializer/truncated_normal/shapeConst*#
_class
loc:@NIN/dense_3/bias*
dtype0*
_output_shapes
:*
valueB:

2NIN/dense_3/bias/Initializer/truncated_normal/meanConst*#
_class
loc:@NIN/dense_3/bias*
_output_shapes
: *
valueB
 *    *
dtype0

4NIN/dense_3/bias/Initializer/truncated_normal/stddevConst*
valueB
 *ΝΜΜ=*
_output_shapes
: *
dtype0*#
_class
loc:@NIN/dense_3/bias
υ
=NIN/dense_3/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3NIN/dense_3/bias/Initializer/truncated_normal/shape*
dtype0*

seed *#
_class
loc:@NIN/dense_3/bias*
seed2 *
T0*
_output_shapes
:
χ
1NIN/dense_3/bias/Initializer/truncated_normal/mulMul=NIN/dense_3/bias/Initializer/truncated_normal/TruncatedNormal4NIN/dense_3/bias/Initializer/truncated_normal/stddev*
T0*#
_class
loc:@NIN/dense_3/bias*
_output_shapes
:
ε
-NIN/dense_3/bias/Initializer/truncated_normalAdd1NIN/dense_3/bias/Initializer/truncated_normal/mul2NIN/dense_3/bias/Initializer/truncated_normal/mean*
_output_shapes
:*#
_class
loc:@NIN/dense_3/bias*
T0
‘
NIN/dense_3/bias
VariableV2*
dtype0*
shape:*#
_class
loc:@NIN/dense_3/bias*
	container *
shared_name *
_output_shapes
:
Υ
NIN/dense_3/bias/AssignAssignNIN/dense_3/bias-NIN/dense_3/bias/Initializer/truncated_normal*#
_class
loc:@NIN/dense_3/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
}
NIN/dense_3/bias/readIdentityNIN/dense_3/bias*
_output_shapes
:*#
_class
loc:@NIN/dense_3/bias*
T0

NIN/dense_3/MatMulMatMulstrided_sliceNIN/dense_3/kernel/read*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:?????????

NIN/dense_3/BiasAddBiasAddNIN/dense_3/MatMulNIN/dense_3/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:?????????

NIN/dense_3/swish_f32	swish_f32NIN/dense_3/BiasAdd*#
_disable_call_shape_inference(*'
_output_shapes
:?????????
­
5NIN/dense_4/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *%
_class
loc:@NIN/dense_4/kernel*
dtype0*
_output_shapes
:
 
4NIN/dense_4/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *
_output_shapes
: *%
_class
loc:@NIN/dense_4/kernel*
dtype0
’
6NIN/dense_4/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *ΝΜΜ=*
dtype0*%
_class
loc:@NIN/dense_4/kernel*
_output_shapes
: 
?
?NIN/dense_4/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5NIN/dense_4/kernel/Initializer/truncated_normal/shape*

seed *
dtype0*%
_class
loc:@NIN/dense_4/kernel*
seed2 *
_output_shapes

:*
T0

3NIN/dense_4/kernel/Initializer/truncated_normal/mulMul?NIN/dense_4/kernel/Initializer/truncated_normal/TruncatedNormal6NIN/dense_4/kernel/Initializer/truncated_normal/stddev*
T0*%
_class
loc:@NIN/dense_4/kernel*
_output_shapes

:
ρ
/NIN/dense_4/kernel/Initializer/truncated_normalAdd3NIN/dense_4/kernel/Initializer/truncated_normal/mul4NIN/dense_4/kernel/Initializer/truncated_normal/mean*
_output_shapes

:*
T0*%
_class
loc:@NIN/dense_4/kernel
­
NIN/dense_4/kernel
VariableV2*
_output_shapes

:*
shape
:*
	container *
dtype0*
shared_name *%
_class
loc:@NIN/dense_4/kernel
α
NIN/dense_4/kernel/AssignAssignNIN/dense_4/kernel/NIN/dense_4/kernel/Initializer/truncated_normal*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*%
_class
loc:@NIN/dense_4/kernel

NIN/dense_4/kernel/readIdentityNIN/dense_4/kernel*
_output_shapes

:*%
_class
loc:@NIN/dense_4/kernel*
T0
’
3NIN/dense_4/bias/Initializer/truncated_normal/shapeConst*#
_class
loc:@NIN/dense_4/bias*
dtype0*
valueB:*
_output_shapes
:

2NIN/dense_4/bias/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *
dtype0*#
_class
loc:@NIN/dense_4/bias

4NIN/dense_4/bias/Initializer/truncated_normal/stddevConst*
dtype0*
valueB
 *ΝΜΜ=*#
_class
loc:@NIN/dense_4/bias*
_output_shapes
: 
υ
=NIN/dense_4/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3NIN/dense_4/bias/Initializer/truncated_normal/shape*

seed *#
_class
loc:@NIN/dense_4/bias*
seed2 *
dtype0*
_output_shapes
:*
T0
χ
1NIN/dense_4/bias/Initializer/truncated_normal/mulMul=NIN/dense_4/bias/Initializer/truncated_normal/TruncatedNormal4NIN/dense_4/bias/Initializer/truncated_normal/stddev*
_output_shapes
:*#
_class
loc:@NIN/dense_4/bias*
T0
ε
-NIN/dense_4/bias/Initializer/truncated_normalAdd1NIN/dense_4/bias/Initializer/truncated_normal/mul2NIN/dense_4/bias/Initializer/truncated_normal/mean*
_output_shapes
:*#
_class
loc:@NIN/dense_4/bias*
T0
‘
NIN/dense_4/bias
VariableV2*
dtype0*
shared_name *#
_class
loc:@NIN/dense_4/bias*
shape:*
	container *
_output_shapes
:
Υ
NIN/dense_4/bias/AssignAssignNIN/dense_4/bias-NIN/dense_4/bias/Initializer/truncated_normal*
_output_shapes
:*
use_locking(*
validate_shape(*
T0*#
_class
loc:@NIN/dense_4/bias
}
NIN/dense_4/bias/readIdentityNIN/dense_4/bias*
T0*#
_class
loc:@NIN/dense_4/bias*
_output_shapes
:
€
NIN/dense_4/MatMulMatMulNIN/dense_3/swish_f32NIN/dense_4/kernel/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:?????????

NIN/dense_4/BiasAddBiasAddNIN/dense_4/MatMulNIN/dense_4/bias/read*'
_output_shapes
:?????????*
T0*
data_formatNHWC

NIN/dense_4/swish_f32	swish_f32NIN/dense_4/BiasAdd*'
_output_shapes
:?????????*#
_disable_call_shape_inference(
p
NIN/addAddV2NIN/dense_3/swish_f32NIN/dense_4/swish_f32*'
_output_shapes
:?????????*
T0
­
5NIN/dense_5/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
dtype0*
valueB"      *%
_class
loc:@NIN/dense_5/kernel
 
4NIN/dense_5/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *%
_class
loc:@NIN/dense_5/kernel*
dtype0*
_output_shapes
: 
’
6NIN/dense_5/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *ΝΜΜ=*
dtype0*%
_class
loc:@NIN/dense_5/kernel
?
?NIN/dense_5/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal5NIN/dense_5/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0*
_output_shapes

:*
T0*%
_class
loc:@NIN/dense_5/kernel*

seed 

3NIN/dense_5/kernel/Initializer/truncated_normal/mulMul?NIN/dense_5/kernel/Initializer/truncated_normal/TruncatedNormal6NIN/dense_5/kernel/Initializer/truncated_normal/stddev*
T0*%
_class
loc:@NIN/dense_5/kernel*
_output_shapes

:
ρ
/NIN/dense_5/kernel/Initializer/truncated_normalAdd3NIN/dense_5/kernel/Initializer/truncated_normal/mul4NIN/dense_5/kernel/Initializer/truncated_normal/mean*%
_class
loc:@NIN/dense_5/kernel*
_output_shapes

:*
T0
­
NIN/dense_5/kernel
VariableV2*
shape
:*%
_class
loc:@NIN/dense_5/kernel*
dtype0*
shared_name *
_output_shapes

:*
	container 
α
NIN/dense_5/kernel/AssignAssignNIN/dense_5/kernel/NIN/dense_5/kernel/Initializer/truncated_normal*%
_class
loc:@NIN/dense_5/kernel*
_output_shapes

:*
T0*
validate_shape(*
use_locking(

NIN/dense_5/kernel/readIdentityNIN/dense_5/kernel*
_output_shapes

:*
T0*%
_class
loc:@NIN/dense_5/kernel
’
3NIN/dense_5/bias/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB:*#
_class
loc:@NIN/dense_5/bias*
dtype0

2NIN/dense_5/bias/Initializer/truncated_normal/meanConst*#
_class
loc:@NIN/dense_5/bias*
dtype0*
_output_shapes
: *
valueB
 *    

4NIN/dense_5/bias/Initializer/truncated_normal/stddevConst*
_output_shapes
: *#
_class
loc:@NIN/dense_5/bias*
valueB
 *ΝΜΜ=*
dtype0
υ
=NIN/dense_5/bias/Initializer/truncated_normal/TruncatedNormalTruncatedNormal3NIN/dense_5/bias/Initializer/truncated_normal/shape*
seed2 *
_output_shapes
:*#
_class
loc:@NIN/dense_5/bias*

seed *
dtype0*
T0
χ
1NIN/dense_5/bias/Initializer/truncated_normal/mulMul=NIN/dense_5/bias/Initializer/truncated_normal/TruncatedNormal4NIN/dense_5/bias/Initializer/truncated_normal/stddev*#
_class
loc:@NIN/dense_5/bias*
_output_shapes
:*
T0
ε
-NIN/dense_5/bias/Initializer/truncated_normalAdd1NIN/dense_5/bias/Initializer/truncated_normal/mul2NIN/dense_5/bias/Initializer/truncated_normal/mean*
T0*
_output_shapes
:*#
_class
loc:@NIN/dense_5/bias
‘
NIN/dense_5/bias
VariableV2*
_output_shapes
:*
shape:*
	container *
shared_name *
dtype0*#
_class
loc:@NIN/dense_5/bias
Υ
NIN/dense_5/bias/AssignAssignNIN/dense_5/bias-NIN/dense_5/bias/Initializer/truncated_normal*
T0*
use_locking(*#
_class
loc:@NIN/dense_5/bias*
_output_shapes
:*
validate_shape(
}
NIN/dense_5/bias/readIdentityNIN/dense_5/bias*
T0*#
_class
loc:@NIN/dense_5/bias*
_output_shapes
:

NIN/dense_5/MatMulMatMulNIN/addNIN/dense_5/kernel/read*
transpose_b( *'
_output_shapes
:?????????*
T0*
transpose_a( 

NIN/dense_5/BiasAddBiasAddNIN/dense_5/MatMulNIN/dense_5/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:?????????

NIN/dense_5/swish_f32	swish_f32NIN/dense_5/BiasAdd*'
_output_shapes
:?????????*#
_disable_call_shape_inference(
h
NIN/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
j
NIN/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
j
NIN/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
¨
NIN/strided_sliceStridedSliceNIN/dense_2/BiasAddNIN/strided_slice/stackNIN/strided_slice/stack_1NIN/strided_slice/stack_2*
ellipsis_mask *
Index0*
new_axis_mask *
end_mask*
T0*'
_output_shapes
:?????????*

begin_mask*
shrink_axis_mask 
f
NIN/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"????      

NIN/ReshapeReshapeNIN/strided_sliceNIN/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:?????????
j
NIN/strided_slice_1/stackConst*
_output_shapes
:*
valueB"    ????*
dtype0
l
NIN/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
l
NIN/strided_slice_1/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
¬
NIN/strided_slice_1StridedSliceNIN/dense_2/BiasAddNIN/strided_slice_1/stackNIN/strided_slice_1/stack_1NIN/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
ellipsis_mask *
end_mask*#
_output_shapes
:?????????*

begin_mask*
new_axis_mask 
d
NIN/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   

NIN/Reshape_1ReshapeNIN/strided_slice_1NIN/Reshape_1/shape*
Tshape0*'
_output_shapes
:?????????*
T0
e
NIN/einsum/ShapeShapeNIN/dense_5/swish_f32*
_output_shapes
:*
T0*
out_type0
h
NIN/einsum/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
j
 NIN/einsum/strided_slice/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
j
 NIN/einsum/strided_slice/stack_2Const*
valueB:*
_output_shapes
:*
dtype0
°
NIN/einsum/strided_sliceStridedSliceNIN/einsum/ShapeNIN/einsum/strided_slice/stack NIN/einsum/strided_slice/stack_1 NIN/einsum/strided_slice/stack_2*
Index0*
shrink_axis_mask*
new_axis_mask *
ellipsis_mask *
T0*

begin_mask *
end_mask *
_output_shapes
: 
\
NIN/einsum/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
\
NIN/einsum/Reshape/shape/2Const*
_output_shapes
: *
value	B :*
dtype0
¬
NIN/einsum/Reshape/shapePackNIN/einsum/strided_sliceNIN/einsum/Reshape/shape/1NIN/einsum/Reshape/shape/2*
T0*

axis *
_output_shapes
:*
N

NIN/einsum/ReshapeReshapeNIN/dense_5/swish_f32NIN/einsum/Reshape/shape*
T0*+
_output_shapes
:?????????*
Tshape0
]
NIN/einsum/Shape_1ShapeNIN/Reshape*
T0*
_output_shapes
:*
out_type0
j
 NIN/einsum/strided_slice_1/stackConst*
valueB: *
_output_shapes
:*
dtype0
l
"NIN/einsum/strided_slice_1/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
l
"NIN/einsum/strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
Ί
NIN/einsum/strided_slice_1StridedSliceNIN/einsum/Shape_1 NIN/einsum/strided_slice_1/stack"NIN/einsum/strided_slice_1/stack_1"NIN/einsum/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
ellipsis_mask 
^
NIN/einsum/Reshape_1/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
^
NIN/einsum/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
΄
NIN/einsum/Reshape_1/shapePackNIN/einsum/strided_slice_1NIN/einsum/Reshape_1/shape/1NIN/einsum/Reshape_1/shape/2*

axis *
N*
T0*
_output_shapes
:

NIN/einsum/Reshape_1ReshapeNIN/ReshapeNIN/einsum/Reshape_1/shape*
T0*+
_output_shapes
:?????????*
Tshape0

NIN/einsum/MatMulBatchMatMulV2NIN/einsum/ReshapeNIN/einsum/Reshape_1*
adj_y( *
T0*+
_output_shapes
:?????????*
adj_x( 
^
NIN/einsum/Reshape_2/shape/1Const*
dtype0*
value	B :*
_output_shapes
: 

NIN/einsum/Reshape_2/shapePackNIN/einsum/strided_sliceNIN/einsum/Reshape_2/shape/1*
_output_shapes
:*

axis *
N*
T0

NIN/einsum/Reshape_2ReshapeNIN/einsum/MatMulNIN/einsum/Reshape_2/shape*
Tshape0*
T0*'
_output_shapes
:?????????
i
	NIN/add_1AddV2NIN/einsum/Reshape_2NIN/Reshape_1*
T0*'
_output_shapes
:?????????
X
subSub	NIN/add_1strided_slice_2*'
_output_shapes
:?????????*
T0
G
SquareSquaresub*'
_output_shapes
:?????????*
T0
V
ConstConst*
valueB"       *
_output_shapes
:*
dtype0
Y
MeanMeanSquareConst*
_output_shapes
: *
	keep_dims( *
T0*

Tidx0
R
gradients/ShapeConst*
dtype0*
valueB *
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
_output_shapes
: *
dtype0
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *

index_type0*
T0
r
!gradients/Mean_grad/Reshape/shapeConst*
dtype0*
valueB"      *
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
_output_shapes

:*
T0*
Tshape0
_
gradients/Mean_grad/ShapeShapeSquare*
_output_shapes
:*
T0*
out_type0

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*

Tmultiples0*'
_output_shapes
:?????????
a
gradients/Mean_grad/Shape_1ShapeSquare*
_output_shapes
:*
T0*
out_type0
^
gradients/Mean_grad/Shape_2Const*
dtype0*
_output_shapes
: *
valueB 
c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
valueB: *
dtype0

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*'
_output_shapes
:?????????*
T0
~
gradients/Square_grad/ConstConst^gradients/Mean_grad/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @
t
gradients/Square_grad/MulMulsubgradients/Square_grad/Const*'
_output_shapes
:?????????*
T0

gradients/Square_grad/Mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/Mul*'
_output_shapes
:?????????*
T0
a
gradients/sub_grad/ShapeShape	NIN/add_1*
T0*
_output_shapes
:*
out_type0
i
gradients/sub_grad/Shape_1Shapestrided_slice_2*
out_type0*
T0*
_output_shapes
:
΄
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
€
gradients/sub_grad/SumSumgradients/Square_grad/Mul_1(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0

gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
l
gradients/sub_grad/NegNeggradients/Square_grad/Mul_1*'
_output_shapes
:?????????*
T0
£
gradients/sub_grad/Sum_1Sumgradients/sub_grad/Neg*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Sum_1gradients/sub_grad/Shape_1*
T0*'
_output_shapes
:?????????*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
Ϊ
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:?????????*-
_class#
!loc:@gradients/sub_grad/Reshape*
T0
ΰ
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*'
_output_shapes
:?????????*/
_class%
#!loc:@gradients/sub_grad/Reshape_1*
T0
r
gradients/NIN/add_1_grad/ShapeShapeNIN/einsum/Reshape_2*
T0*
_output_shapes
:*
out_type0
m
 gradients/NIN/add_1_grad/Shape_1ShapeNIN/Reshape_1*
out_type0*
T0*
_output_shapes
:
Ζ
.gradients/NIN/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/NIN/add_1_grad/Shape gradients/NIN/add_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
ΐ
gradients/NIN/add_1_grad/SumSum+gradients/sub_grad/tuple/control_dependency.gradients/NIN/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
©
 gradients/NIN/add_1_grad/ReshapeReshapegradients/NIN/add_1_grad/Sumgradients/NIN/add_1_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
Δ
gradients/NIN/add_1_grad/Sum_1Sum+gradients/sub_grad/tuple/control_dependency0gradients/NIN/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
―
"gradients/NIN/add_1_grad/Reshape_1Reshapegradients/NIN/add_1_grad/Sum_1 gradients/NIN/add_1_grad/Shape_1*
T0*'
_output_shapes
:?????????*
Tshape0
y
)gradients/NIN/add_1_grad/tuple/group_depsNoOp!^gradients/NIN/add_1_grad/Reshape#^gradients/NIN/add_1_grad/Reshape_1
ς
1gradients/NIN/add_1_grad/tuple/control_dependencyIdentity gradients/NIN/add_1_grad/Reshape*^gradients/NIN/add_1_grad/tuple/group_deps*3
_class)
'%loc:@gradients/NIN/add_1_grad/Reshape*'
_output_shapes
:?????????*
T0
ψ
3gradients/NIN/add_1_grad/tuple/control_dependency_1Identity"gradients/NIN/add_1_grad/Reshape_1*^gradients/NIN/add_1_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/NIN/add_1_grad/Reshape_1*'
_output_shapes
:?????????
z
)gradients/NIN/einsum/Reshape_2_grad/ShapeShapeNIN/einsum/MatMul*
out_type0*
T0*
_output_shapes
:
Ψ
+gradients/NIN/einsum/Reshape_2_grad/ReshapeReshape1gradients/NIN/add_1_grad/tuple/control_dependency)gradients/NIN/einsum/Reshape_2_grad/Shape*+
_output_shapes
:?????????*
Tshape0*
T0
u
"gradients/NIN/Reshape_1_grad/ShapeShapeNIN/strided_slice_1*
T0*
out_type0*
_output_shapes
:
Δ
$gradients/NIN/Reshape_1_grad/ReshapeReshape3gradients/NIN/add_1_grad/tuple/control_dependency_1"gradients/NIN/Reshape_1_grad/Shape*#
_output_shapes
:?????????*
T0*
Tshape0
Λ
'gradients/NIN/einsum/MatMul_grad/MatMulBatchMatMulV2+gradients/NIN/einsum/Reshape_2_grad/ReshapeNIN/einsum/Reshape_1*
adj_x( *+
_output_shapes
:?????????*
adj_y(*
T0
Λ
)gradients/NIN/einsum/MatMul_grad/MatMul_1BatchMatMulV2NIN/einsum/Reshape+gradients/NIN/einsum/Reshape_2_grad/Reshape*
T0*
adj_y( *
adj_x(*+
_output_shapes
:?????????
x
&gradients/NIN/einsum/MatMul_grad/ShapeShapeNIN/einsum/Reshape*
_output_shapes
:*
out_type0*
T0
|
(gradients/NIN/einsum/MatMul_grad/Shape_1ShapeNIN/einsum/Reshape_1*
_output_shapes
:*
T0*
out_type0
~
4gradients/NIN/einsum/MatMul_grad/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 

6gradients/NIN/einsum/MatMul_grad/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
ώ????????

6gradients/NIN/einsum/MatMul_grad/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
’
.gradients/NIN/einsum/MatMul_grad/strided_sliceStridedSlice&gradients/NIN/einsum/MatMul_grad/Shape4gradients/NIN/einsum/MatMul_grad/strided_slice/stack6gradients/NIN/einsum/MatMul_grad/strided_slice/stack_16gradients/NIN/einsum/MatMul_grad/strided_slice/stack_2*
_output_shapes
:*
ellipsis_mask *
T0*
new_axis_mask *

begin_mask*
end_mask *
shrink_axis_mask *
Index0

6gradients/NIN/einsum/MatMul_grad/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:

8gradients/NIN/einsum/MatMul_grad/strided_slice_1/stack_1Const*
valueB:
ώ????????*
dtype0*
_output_shapes
:

8gradients/NIN/einsum/MatMul_grad/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¬
0gradients/NIN/einsum/MatMul_grad/strided_slice_1StridedSlice(gradients/NIN/einsum/MatMul_grad/Shape_16gradients/NIN/einsum/MatMul_grad/strided_slice_1/stack8gradients/NIN/einsum/MatMul_grad/strided_slice_1/stack_18gradients/NIN/einsum/MatMul_grad/strided_slice_1/stack_2*

begin_mask*
T0*
_output_shapes
:*
shrink_axis_mask *
ellipsis_mask *
new_axis_mask *
Index0*
end_mask 
ξ
6gradients/NIN/einsum/MatMul_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/NIN/einsum/MatMul_grad/strided_slice0gradients/NIN/einsum/MatMul_grad/strided_slice_1*
T0*2
_output_shapes 
:?????????:?????????
Μ
$gradients/NIN/einsum/MatMul_grad/SumSum'gradients/NIN/einsum/MatMul_grad/MatMul6gradients/NIN/einsum/MatMul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
Ε
(gradients/NIN/einsum/MatMul_grad/ReshapeReshape$gradients/NIN/einsum/MatMul_grad/Sum&gradients/NIN/einsum/MatMul_grad/Shape*
T0*
Tshape0*+
_output_shapes
:?????????
?
&gradients/NIN/einsum/MatMul_grad/Sum_1Sum)gradients/NIN/einsum/MatMul_grad/MatMul_18gradients/NIN/einsum/MatMul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
Λ
*gradients/NIN/einsum/MatMul_grad/Reshape_1Reshape&gradients/NIN/einsum/MatMul_grad/Sum_1(gradients/NIN/einsum/MatMul_grad/Shape_1*
T0*+
_output_shapes
:?????????*
Tshape0

1gradients/NIN/einsum/MatMul_grad/tuple/group_depsNoOp)^gradients/NIN/einsum/MatMul_grad/Reshape+^gradients/NIN/einsum/MatMul_grad/Reshape_1

9gradients/NIN/einsum/MatMul_grad/tuple/control_dependencyIdentity(gradients/NIN/einsum/MatMul_grad/Reshape2^gradients/NIN/einsum/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients/NIN/einsum/MatMul_grad/Reshape*
T0*+
_output_shapes
:?????????

;gradients/NIN/einsum/MatMul_grad/tuple/control_dependency_1Identity*gradients/NIN/einsum/MatMul_grad/Reshape_12^gradients/NIN/einsum/MatMul_grad/tuple/group_deps*+
_output_shapes
:?????????*
T0*=
_class3
1/loc:@gradients/NIN/einsum/MatMul_grad/Reshape_1
{
(gradients/NIN/strided_slice_1_grad/ShapeShapeNIN/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0

3gradients/NIN/strided_slice_1_grad/StridedSliceGradStridedSliceGrad(gradients/NIN/strided_slice_1_grad/ShapeNIN/strided_slice_1/stackNIN/strided_slice_1/stack_1NIN/strided_slice_1/stack_2$gradients/NIN/Reshape_1_grad/Reshape*

begin_mask*'
_output_shapes
:?????????*
new_axis_mask *
Index0*
shrink_axis_mask*
ellipsis_mask *
T0*
end_mask
|
'gradients/NIN/einsum/Reshape_grad/ShapeShapeNIN/dense_5/swish_f32*
out_type0*
T0*
_output_shapes
:
Ψ
)gradients/NIN/einsum/Reshape_grad/ReshapeReshape9gradients/NIN/einsum/MatMul_grad/tuple/control_dependency'gradients/NIN/einsum/Reshape_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
t
)gradients/NIN/einsum/Reshape_1_grad/ShapeShapeNIN/Reshape*
out_type0*
_output_shapes
:*
T0
β
+gradients/NIN/einsum/Reshape_1_grad/ReshapeReshape;gradients/NIN/einsum/MatMul_grad/tuple/control_dependency_1)gradients/NIN/einsum/Reshape_1_grad/Shape*
Tshape0*
T0*+
_output_shapes
:?????????

5gradients/NIN/dense_5/swish_f32_grad/SymbolicGradientSymbolicGradientNIN/dense_5/BiasAdd)gradients/NIN/einsum/Reshape_grad/Reshape*7
f2R0
	swish_f32#
_disable_call_shape_inference(*'
_output_shapes
:?????????*
Tout
2*
Tin
2
q
 gradients/NIN/Reshape_grad/ShapeShapeNIN/strided_slice*
T0*
out_type0*
_output_shapes
:
Ό
"gradients/NIN/Reshape_grad/ReshapeReshape+gradients/NIN/einsum/Reshape_1_grad/Reshape gradients/NIN/Reshape_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
°
.gradients/NIN/dense_5/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/NIN/dense_5/swish_f32_grad/SymbolicGradient*
data_formatNHWC*
_output_shapes
:*
T0
€
3gradients/NIN/dense_5/BiasAdd_grad/tuple/group_depsNoOp/^gradients/NIN/dense_5/BiasAdd_grad/BiasAddGrad6^gradients/NIN/dense_5/swish_f32_grad/SymbolicGradient
°
;gradients/NIN/dense_5/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/NIN/dense_5/swish_f32_grad/SymbolicGradient4^gradients/NIN/dense_5/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*H
_class>
<:loc:@gradients/NIN/dense_5/swish_f32_grad/SymbolicGradient

=gradients/NIN/dense_5/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/NIN/dense_5/BiasAdd_grad/BiasAddGrad4^gradients/NIN/dense_5/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:*A
_class7
53loc:@gradients/NIN/dense_5/BiasAdd_grad/BiasAddGrad
y
&gradients/NIN/strided_slice_grad/ShapeShapeNIN/dense_2/BiasAdd*
out_type0*
T0*
_output_shapes
:

1gradients/NIN/strided_slice_grad/StridedSliceGradStridedSliceGrad&gradients/NIN/strided_slice_grad/ShapeNIN/strided_slice/stackNIN/strided_slice/stack_1NIN/strided_slice/stack_2"gradients/NIN/Reshape_grad/Reshape*
T0*
ellipsis_mask *'
_output_shapes
:?????????*
shrink_axis_mask *

begin_mask*
Index0*
new_axis_mask *
end_mask
ΰ
(gradients/NIN/dense_5/MatMul_grad/MatMulMatMul;gradients/NIN/dense_5/BiasAdd_grad/tuple/control_dependencyNIN/dense_5/kernel/read*
transpose_a( *'
_output_shapes
:?????????*
T0*
transpose_b(
Ι
*gradients/NIN/dense_5/MatMul_grad/MatMul_1MatMulNIN/add;gradients/NIN/dense_5/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_b( *
T0*
transpose_a(

2gradients/NIN/dense_5/MatMul_grad/tuple/group_depsNoOp)^gradients/NIN/dense_5/MatMul_grad/MatMul+^gradients/NIN/dense_5/MatMul_grad/MatMul_1

:gradients/NIN/dense_5/MatMul_grad/tuple/control_dependencyIdentity(gradients/NIN/dense_5/MatMul_grad/MatMul3^gradients/NIN/dense_5/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients/NIN/dense_5/MatMul_grad/MatMul*'
_output_shapes
:?????????*
T0

<gradients/NIN/dense_5/MatMul_grad/tuple/control_dependency_1Identity*gradients/NIN/dense_5/MatMul_grad/MatMul_13^gradients/NIN/dense_5/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/NIN/dense_5/MatMul_grad/MatMul_1*
_output_shapes

:

gradients/AddNAddN3gradients/NIN/strided_slice_1_grad/StridedSliceGrad1gradients/NIN/strided_slice_grad/StridedSliceGrad*
N*
T0*'
_output_shapes
:?????????*F
_class<
:8loc:@gradients/NIN/strided_slice_1_grad/StridedSliceGrad

.gradients/NIN/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN*
T0*
data_formatNHWC*
_output_shapes
:
}
3gradients/NIN/dense_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN/^gradients/NIN/dense_2/BiasAdd_grad/BiasAddGrad

;gradients/NIN/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN4^gradients/NIN/dense_2/BiasAdd_grad/tuple/group_deps*F
_class<
:8loc:@gradients/NIN/strided_slice_1_grad/StridedSliceGrad*'
_output_shapes
:?????????*
T0

=gradients/NIN/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/NIN/dense_2/BiasAdd_grad/BiasAddGrad4^gradients/NIN/dense_2/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients/NIN/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
q
gradients/NIN/add_grad/ShapeShapeNIN/dense_3/swish_f32*
out_type0*
_output_shapes
:*
T0
s
gradients/NIN/add_grad/Shape_1ShapeNIN/dense_4/swish_f32*
out_type0*
_output_shapes
:*
T0
ΐ
,gradients/NIN/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/NIN/add_grad/Shapegradients/NIN/add_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
Λ
gradients/NIN/add_grad/SumSum:gradients/NIN/dense_5/MatMul_grad/tuple/control_dependency,gradients/NIN/add_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
£
gradients/NIN/add_grad/ReshapeReshapegradients/NIN/add_grad/Sumgradients/NIN/add_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
Ο
gradients/NIN/add_grad/Sum_1Sum:gradients/NIN/dense_5/MatMul_grad/tuple/control_dependency.gradients/NIN/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
©
 gradients/NIN/add_grad/Reshape_1Reshapegradients/NIN/add_grad/Sum_1gradients/NIN/add_grad/Shape_1*
T0*'
_output_shapes
:?????????*
Tshape0
s
'gradients/NIN/add_grad/tuple/group_depsNoOp^gradients/NIN/add_grad/Reshape!^gradients/NIN/add_grad/Reshape_1
κ
/gradients/NIN/add_grad/tuple/control_dependencyIdentitygradients/NIN/add_grad/Reshape(^gradients/NIN/add_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*1
_class'
%#loc:@gradients/NIN/add_grad/Reshape
π
1gradients/NIN/add_grad/tuple/control_dependency_1Identity gradients/NIN/add_grad/Reshape_1(^gradients/NIN/add_grad/tuple/group_deps*3
_class)
'%loc:@gradients/NIN/add_grad/Reshape_1*'
_output_shapes
:?????????*
T0
ΰ
(gradients/NIN/dense_2/MatMul_grad/MatMulMatMul;gradients/NIN/dense_2/BiasAdd_grad/tuple/control_dependencyNIN/dense_2/kernel/read*
transpose_b(*'
_output_shapes
:?????????*
transpose_a( *
T0
Χ
*gradients/NIN/dense_2/MatMul_grad/MatMul_1MatMulNIN/dense_1/swish_f32;gradients/NIN/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_b( *
transpose_a(*
T0

2gradients/NIN/dense_2/MatMul_grad/tuple/group_depsNoOp)^gradients/NIN/dense_2/MatMul_grad/MatMul+^gradients/NIN/dense_2/MatMul_grad/MatMul_1

:gradients/NIN/dense_2/MatMul_grad/tuple/control_dependencyIdentity(gradients/NIN/dense_2/MatMul_grad/MatMul3^gradients/NIN/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????*;
_class1
/-loc:@gradients/NIN/dense_2/MatMul_grad/MatMul*
T0

<gradients/NIN/dense_2/MatMul_grad/tuple/control_dependency_1Identity*gradients/NIN/dense_2/MatMul_grad/MatMul_13^gradients/NIN/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:*
T0*=
_class3
1/loc:@gradients/NIN/dense_2/MatMul_grad/MatMul_1

5gradients/NIN/dense_4/swish_f32_grad/SymbolicGradientSymbolicGradientNIN/dense_4/BiasAdd1gradients/NIN/add_grad/tuple/control_dependency_1*7
f2R0
	swish_f32#
_disable_call_shape_inference(*
Tin
2*'
_output_shapes
:?????????*
Tout
2

5gradients/NIN/dense_1/swish_f32_grad/SymbolicGradientSymbolicGradientNIN/dense_1/BiasAdd:gradients/NIN/dense_2/MatMul_grad/tuple/control_dependency*
Tout
2*'
_output_shapes
:?????????*
Tin
2*7
f2R0
	swish_f32#
_disable_call_shape_inference(
°
.gradients/NIN/dense_4/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/NIN/dense_4/swish_f32_grad/SymbolicGradient*
_output_shapes
:*
T0*
data_formatNHWC
€
3gradients/NIN/dense_4/BiasAdd_grad/tuple/group_depsNoOp/^gradients/NIN/dense_4/BiasAdd_grad/BiasAddGrad6^gradients/NIN/dense_4/swish_f32_grad/SymbolicGradient
°
;gradients/NIN/dense_4/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/NIN/dense_4/swish_f32_grad/SymbolicGradient4^gradients/NIN/dense_4/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*H
_class>
<:loc:@gradients/NIN/dense_4/swish_f32_grad/SymbolicGradient

=gradients/NIN/dense_4/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/NIN/dense_4/BiasAdd_grad/BiasAddGrad4^gradients/NIN/dense_4/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients/NIN/dense_4/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
°
.gradients/NIN/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/NIN/dense_1/swish_f32_grad/SymbolicGradient*
_output_shapes
:*
data_formatNHWC*
T0
€
3gradients/NIN/dense_1/BiasAdd_grad/tuple/group_depsNoOp/^gradients/NIN/dense_1/BiasAdd_grad/BiasAddGrad6^gradients/NIN/dense_1/swish_f32_grad/SymbolicGradient
°
;gradients/NIN/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/NIN/dense_1/swish_f32_grad/SymbolicGradient4^gradients/NIN/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:?????????*H
_class>
<:loc:@gradients/NIN/dense_1/swish_f32_grad/SymbolicGradient*
T0

=gradients/NIN/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/NIN/dense_1/BiasAdd_grad/BiasAddGrad4^gradients/NIN/dense_1/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*A
_class7
53loc:@gradients/NIN/dense_1/BiasAdd_grad/BiasAddGrad
ΰ
(gradients/NIN/dense_4/MatMul_grad/MatMulMatMul;gradients/NIN/dense_4/BiasAdd_grad/tuple/control_dependencyNIN/dense_4/kernel/read*
transpose_a( *'
_output_shapes
:?????????*
T0*
transpose_b(
Χ
*gradients/NIN/dense_4/MatMul_grad/MatMul_1MatMulNIN/dense_3/swish_f32;gradients/NIN/dense_4/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( *
_output_shapes

:

2gradients/NIN/dense_4/MatMul_grad/tuple/group_depsNoOp)^gradients/NIN/dense_4/MatMul_grad/MatMul+^gradients/NIN/dense_4/MatMul_grad/MatMul_1

:gradients/NIN/dense_4/MatMul_grad/tuple/control_dependencyIdentity(gradients/NIN/dense_4/MatMul_grad/MatMul3^gradients/NIN/dense_4/MatMul_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*;
_class1
/-loc:@gradients/NIN/dense_4/MatMul_grad/MatMul

<gradients/NIN/dense_4/MatMul_grad/tuple/control_dependency_1Identity*gradients/NIN/dense_4/MatMul_grad/MatMul_13^gradients/NIN/dense_4/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/NIN/dense_4/MatMul_grad/MatMul_1*
_output_shapes

:
ΰ
(gradients/NIN/dense_1/MatMul_grad/MatMulMatMul;gradients/NIN/dense_1/BiasAdd_grad/tuple/control_dependencyNIN/dense_1/kernel/read*
T0*
transpose_a( *'
_output_shapes
:?????????*
transpose_b(
Υ
*gradients/NIN/dense_1/MatMul_grad/MatMul_1MatMulNIN/dense/swish_f32;gradients/NIN/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
_output_shapes

:*
transpose_a(*
T0

2gradients/NIN/dense_1/MatMul_grad/tuple/group_depsNoOp)^gradients/NIN/dense_1/MatMul_grad/MatMul+^gradients/NIN/dense_1/MatMul_grad/MatMul_1

:gradients/NIN/dense_1/MatMul_grad/tuple/control_dependencyIdentity(gradients/NIN/dense_1/MatMul_grad/MatMul3^gradients/NIN/dense_1/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients/NIN/dense_1/MatMul_grad/MatMul*'
_output_shapes
:?????????*
T0

<gradients/NIN/dense_1/MatMul_grad/tuple/control_dependency_1Identity*gradients/NIN/dense_1/MatMul_grad/MatMul_13^gradients/NIN/dense_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/NIN/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:
σ
gradients/AddN_1AddN/gradients/NIN/add_grad/tuple/control_dependency:gradients/NIN/dense_4/MatMul_grad/tuple/control_dependency*'
_output_shapes
:?????????*
N*1
_class'
%#loc:@gradients/NIN/add_grad/Reshape*
T0
π
5gradients/NIN/dense_3/swish_f32_grad/SymbolicGradientSymbolicGradientNIN/dense_3/BiasAddgradients/AddN_1*
Tin
2*7
f2R0
	swish_f32#
_disable_call_shape_inference(*'
_output_shapes
:?????????*
Tout
2

3gradients/NIN/dense/swish_f32_grad/SymbolicGradientSymbolicGradientNIN/dense/BiasAdd:gradients/NIN/dense_1/MatMul_grad/tuple/control_dependency*
Tin
2*'
_output_shapes
:?????????*7
f2R0
	swish_f32#
_disable_call_shape_inference(*
Tout
2
°
.gradients/NIN/dense_3/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/NIN/dense_3/swish_f32_grad/SymbolicGradient*
T0*
data_formatNHWC*
_output_shapes
:
€
3gradients/NIN/dense_3/BiasAdd_grad/tuple/group_depsNoOp/^gradients/NIN/dense_3/BiasAdd_grad/BiasAddGrad6^gradients/NIN/dense_3/swish_f32_grad/SymbolicGradient
°
;gradients/NIN/dense_3/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/NIN/dense_3/swish_f32_grad/SymbolicGradient4^gradients/NIN/dense_3/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*H
_class>
<:loc:@gradients/NIN/dense_3/swish_f32_grad/SymbolicGradient

=gradients/NIN/dense_3/BiasAdd_grad/tuple/control_dependency_1Identity.gradients/NIN/dense_3/BiasAdd_grad/BiasAddGrad4^gradients/NIN/dense_3/BiasAdd_grad/tuple/group_deps*A
_class7
53loc:@gradients/NIN/dense_3/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
¬
,gradients/NIN/dense/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/NIN/dense/swish_f32_grad/SymbolicGradient*
data_formatNHWC*
T0*
_output_shapes
:

1gradients/NIN/dense/BiasAdd_grad/tuple/group_depsNoOp-^gradients/NIN/dense/BiasAdd_grad/BiasAddGrad4^gradients/NIN/dense/swish_f32_grad/SymbolicGradient
¨
9gradients/NIN/dense/BiasAdd_grad/tuple/control_dependencyIdentity3gradients/NIN/dense/swish_f32_grad/SymbolicGradient2^gradients/NIN/dense/BiasAdd_grad/tuple/group_deps*
T0*'
_output_shapes
:?????????*F
_class<
:8loc:@gradients/NIN/dense/swish_f32_grad/SymbolicGradient

;gradients/NIN/dense/BiasAdd_grad/tuple/control_dependency_1Identity,gradients/NIN/dense/BiasAdd_grad/BiasAddGrad2^gradients/NIN/dense/BiasAdd_grad/tuple/group_deps*?
_class5
31loc:@gradients/NIN/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
ΰ
(gradients/NIN/dense_3/MatMul_grad/MatMulMatMul;gradients/NIN/dense_3/BiasAdd_grad/tuple/control_dependencyNIN/dense_3/kernel/read*
transpose_a( *
T0*'
_output_shapes
:?????????*
transpose_b(
Ο
*gradients/NIN/dense_3/MatMul_grad/MatMul_1MatMulstrided_slice;gradients/NIN/dense_3/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
T0*
transpose_a(*
transpose_b( 

2gradients/NIN/dense_3/MatMul_grad/tuple/group_depsNoOp)^gradients/NIN/dense_3/MatMul_grad/MatMul+^gradients/NIN/dense_3/MatMul_grad/MatMul_1

:gradients/NIN/dense_3/MatMul_grad/tuple/control_dependencyIdentity(gradients/NIN/dense_3/MatMul_grad/MatMul3^gradients/NIN/dense_3/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*;
_class1
/-loc:@gradients/NIN/dense_3/MatMul_grad/MatMul

<gradients/NIN/dense_3/MatMul_grad/tuple/control_dependency_1Identity*gradients/NIN/dense_3/MatMul_grad/MatMul_13^gradients/NIN/dense_3/MatMul_grad/tuple/group_deps*=
_class3
1/loc:@gradients/NIN/dense_3/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
Ϊ
&gradients/NIN/dense/MatMul_grad/MatMulMatMul9gradients/NIN/dense/BiasAdd_grad/tuple/control_dependencyNIN/dense/kernel/read*
T0*
transpose_b(*'
_output_shapes
:?????????*
transpose_a( 
Ν
(gradients/NIN/dense/MatMul_grad/MatMul_1MatMulstrided_slice_19gradients/NIN/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:*
transpose_a(*
T0*
transpose_b( 

0gradients/NIN/dense/MatMul_grad/tuple/group_depsNoOp'^gradients/NIN/dense/MatMul_grad/MatMul)^gradients/NIN/dense/MatMul_grad/MatMul_1

8gradients/NIN/dense/MatMul_grad/tuple/control_dependencyIdentity&gradients/NIN/dense/MatMul_grad/MatMul1^gradients/NIN/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????*9
_class/
-+loc:@gradients/NIN/dense/MatMul_grad/MatMul*
T0

:gradients/NIN/dense/MatMul_grad/tuple/control_dependency_1Identity(gradients/NIN/dense/MatMul_grad/MatMul_11^gradients/NIN/dense/MatMul_grad/tuple/group_deps*;
_class1
/-loc:@gradients/NIN/dense/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 


ExpandDims
ExpandDims:gradients/NIN/dense/MatMul_grad/tuple/control_dependency_1ExpandDims/dim*
T0*"
_output_shapes
:*

Tdim0
S
concat/concat_dimConst*
value	B : *
_output_shapes
: *
dtype0
R
concat/concatIdentity
ExpandDims*"
_output_shapes
:*
T0
Z
Mean_1/reduction_indicesConst*
dtype0*
value	B : *
_output_shapes
: 
}
Mean_1Meanconcat/concatMean_1/reduction_indices*
_output_shapes

:*
T0*
	keep_dims( *

Tidx0
R
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 

ExpandDims_1
ExpandDims;gradients/NIN/dense/BiasAdd_grad/tuple/control_dependency_1ExpandDims_1/dim*

Tdim0*
_output_shapes

:*
T0
U
concat_1/concat_dimConst*
dtype0*
value	B : *
_output_shapes
: 
R
concat_1/concatIdentityExpandDims_1*
T0*
_output_shapes

:
Z
Mean_2/reduction_indicesConst*
value	B : *
dtype0*
_output_shapes
: 
{
Mean_2Meanconcat_1/concatMean_2/reduction_indices*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
R
ExpandDims_2/dimConst*
dtype0*
_output_shapes
: *
value	B : 
£
ExpandDims_2
ExpandDims<gradients/NIN/dense_1/MatMul_grad/tuple/control_dependency_1ExpandDims_2/dim*"
_output_shapes
:*

Tdim0*
T0
U
concat_2/concat_dimConst*
_output_shapes
: *
value	B : *
dtype0
V
concat_2/concatIdentityExpandDims_2*
T0*"
_output_shapes
:
Z
Mean_3/reduction_indicesConst*
value	B : *
dtype0*
_output_shapes
: 

Mean_3Meanconcat_2/concatMean_3/reduction_indices*
	keep_dims( *
_output_shapes

:*

Tidx0*
T0
R
ExpandDims_3/dimConst*
dtype0*
value	B : *
_output_shapes
: 
 
ExpandDims_3
ExpandDims=gradients/NIN/dense_1/BiasAdd_grad/tuple/control_dependency_1ExpandDims_3/dim*

Tdim0*
T0*
_output_shapes

:
U
concat_3/concat_dimConst*
value	B : *
_output_shapes
: *
dtype0
R
concat_3/concatIdentityExpandDims_3*
_output_shapes

:*
T0
Z
Mean_4/reduction_indicesConst*
dtype0*
value	B : *
_output_shapes
: 
{
Mean_4Meanconcat_3/concatMean_4/reduction_indices*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
R
ExpandDims_4/dimConst*
dtype0*
_output_shapes
: *
value	B : 
£
ExpandDims_4
ExpandDims<gradients/NIN/dense_2/MatMul_grad/tuple/control_dependency_1ExpandDims_4/dim*

Tdim0*
T0*"
_output_shapes
:
U
concat_4/concat_dimConst*
value	B : *
dtype0*
_output_shapes
: 
V
concat_4/concatIdentityExpandDims_4*
T0*"
_output_shapes
:
Z
Mean_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 

Mean_5Meanconcat_4/concatMean_5/reduction_indices*
_output_shapes

:*
	keep_dims( *
T0*

Tidx0
R
ExpandDims_5/dimConst*
_output_shapes
: *
value	B : *
dtype0
 
ExpandDims_5
ExpandDims=gradients/NIN/dense_2/BiasAdd_grad/tuple/control_dependency_1ExpandDims_5/dim*
_output_shapes

:*

Tdim0*
T0
U
concat_5/concat_dimConst*
value	B : *
dtype0*
_output_shapes
: 
R
concat_5/concatIdentityExpandDims_5*
_output_shapes

:*
T0
Z
Mean_6/reduction_indicesConst*
dtype0*
value	B : *
_output_shapes
: 
{
Mean_6Meanconcat_5/concatMean_6/reduction_indices*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
R
ExpandDims_6/dimConst*
value	B : *
dtype0*
_output_shapes
: 
£
ExpandDims_6
ExpandDims<gradients/NIN/dense_3/MatMul_grad/tuple/control_dependency_1ExpandDims_6/dim*"
_output_shapes
:*

Tdim0*
T0
U
concat_6/concat_dimConst*
dtype0*
_output_shapes
: *
value	B : 
V
concat_6/concatIdentityExpandDims_6*
T0*"
_output_shapes
:
Z
Mean_7/reduction_indicesConst*
value	B : *
_output_shapes
: *
dtype0

Mean_7Meanconcat_6/concatMean_7/reduction_indices*
_output_shapes

:*
	keep_dims( *

Tidx0*
T0
R
ExpandDims_7/dimConst*
value	B : *
_output_shapes
: *
dtype0
 
ExpandDims_7
ExpandDims=gradients/NIN/dense_3/BiasAdd_grad/tuple/control_dependency_1ExpandDims_7/dim*

Tdim0*
T0*
_output_shapes

:
U
concat_7/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 
R
concat_7/concatIdentityExpandDims_7*
T0*
_output_shapes

:
Z
Mean_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
{
Mean_8Meanconcat_7/concatMean_8/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
R
ExpandDims_8/dimConst*
dtype0*
_output_shapes
: *
value	B : 
£
ExpandDims_8
ExpandDims<gradients/NIN/dense_4/MatMul_grad/tuple/control_dependency_1ExpandDims_8/dim*"
_output_shapes
:*
T0*

Tdim0
U
concat_8/concat_dimConst*
_output_shapes
: *
value	B : *
dtype0
V
concat_8/concatIdentityExpandDims_8*
T0*"
_output_shapes
:
Z
Mean_9/reduction_indicesConst*
value	B : *
_output_shapes
: *
dtype0

Mean_9Meanconcat_8/concatMean_9/reduction_indices*
_output_shapes

:*
T0*
	keep_dims( *

Tidx0
R
ExpandDims_9/dimConst*
_output_shapes
: *
value	B : *
dtype0
 
ExpandDims_9
ExpandDims=gradients/NIN/dense_4/BiasAdd_grad/tuple/control_dependency_1ExpandDims_9/dim*
_output_shapes

:*

Tdim0*
T0
U
concat_9/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 
R
concat_9/concatIdentityExpandDims_9*
T0*
_output_shapes

:
[
Mean_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 
}
Mean_10Meanconcat_9/concatMean_10/reduction_indices*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
S
ExpandDims_10/dimConst*
_output_shapes
: *
value	B : *
dtype0
₯
ExpandDims_10
ExpandDims<gradients/NIN/dense_5/MatMul_grad/tuple/control_dependency_1ExpandDims_10/dim*"
_output_shapes
:*

Tdim0*
T0
V
concat_10/concat_dimConst*
_output_shapes
: *
dtype0*
value	B : 
X
concat_10/concatIdentityExpandDims_10*"
_output_shapes
:*
T0
[
Mean_11/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B : 

Mean_11Meanconcat_10/concatMean_11/reduction_indices*
	keep_dims( *

Tidx0*
_output_shapes

:*
T0
S
ExpandDims_11/dimConst*
_output_shapes
: *
dtype0*
value	B : 
’
ExpandDims_11
ExpandDims=gradients/NIN/dense_5/BiasAdd_grad/tuple/control_dependency_1ExpandDims_11/dim*

Tdim0*
T0*
_output_shapes

:
V
concat_11/concat_dimConst*
_output_shapes
: *
value	B : *
dtype0
T
concat_11/concatIdentityExpandDims_11*
_output_shapes

:*
T0
[
Mean_12/reduction_indicesConst*
value	B : *
_output_shapes
: *
dtype0
~
Mean_12Meanconcat_11/concatMean_12/reduction_indices*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 

beta1_power/initial_valueConst*
dtype0*
_output_shapes
: *!
_class
loc:@NIN/dense/bias*
valueB
 *fff?

beta1_power
VariableV2*
shape: *
dtype0*!
_class
loc:@NIN/dense/bias*
_output_shapes
: *
	container *
shared_name 
±
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*
_output_shapes
: *!
_class
loc:@NIN/dense/bias*
validate_shape(*
use_locking(
m
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*!
_class
loc:@NIN/dense/bias

beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wΎ?*!
_class
loc:@NIN/dense/bias

beta2_power
VariableV2*!
_class
loc:@NIN/dense/bias*
_output_shapes
: *
dtype0*
shared_name *
	container *
shape: 
±
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*!
_class
loc:@NIN/dense/bias*
T0*
_output_shapes
: *
use_locking(
m
beta2_power/readIdentitybeta2_power*
T0*!
_class
loc:@NIN/dense/bias*
_output_shapes
: 
‘
'NIN/dense/kernel/Adam/Initializer/zerosConst*
_output_shapes

:*#
_class
loc:@NIN/dense/kernel*
valueB*    *
dtype0
?
NIN/dense/kernel/Adam
VariableV2*
shared_name *#
_class
loc:@NIN/dense/kernel*
_output_shapes

:*
dtype0*
	container *
shape
:
έ
NIN/dense/kernel/Adam/AssignAssignNIN/dense/kernel/Adam'NIN/dense/kernel/Adam/Initializer/zeros*
_output_shapes

:*
T0*
validate_shape(*
use_locking(*#
_class
loc:@NIN/dense/kernel

NIN/dense/kernel/Adam/readIdentityNIN/dense/kernel/Adam*
T0*
_output_shapes

:*#
_class
loc:@NIN/dense/kernel
£
)NIN/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:*
valueB*    *#
_class
loc:@NIN/dense/kernel*
dtype0
°
NIN/dense/kernel/Adam_1
VariableV2*#
_class
loc:@NIN/dense/kernel*
shared_name *
	container *
dtype0*
shape
:*
_output_shapes

:
γ
NIN/dense/kernel/Adam_1/AssignAssignNIN/dense/kernel/Adam_1)NIN/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
T0*
use_locking(*#
_class
loc:@NIN/dense/kernel*
validate_shape(

NIN/dense/kernel/Adam_1/readIdentityNIN/dense/kernel/Adam_1*#
_class
loc:@NIN/dense/kernel*
T0*
_output_shapes

:

%NIN/dense/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *
dtype0*!
_class
loc:@NIN/dense/bias
’
NIN/dense/bias/Adam
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:*!
_class
loc:@NIN/dense/bias
Ρ
NIN/dense/bias/Adam/AssignAssignNIN/dense/bias/Adam%NIN/dense/bias/Adam/Initializer/zeros*!
_class
loc:@NIN/dense/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0

NIN/dense/bias/Adam/readIdentityNIN/dense/bias/Adam*
T0*!
_class
loc:@NIN/dense/bias*
_output_shapes
:

'NIN/dense/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@NIN/dense/bias*
dtype0*
_output_shapes
:*
valueB*    
€
NIN/dense/bias/Adam_1
VariableV2*
dtype0*
shape:*
	container *
shared_name *!
_class
loc:@NIN/dense/bias*
_output_shapes
:
Χ
NIN/dense/bias/Adam_1/AssignAssignNIN/dense/bias/Adam_1'NIN/dense/bias/Adam_1/Initializer/zeros*!
_class
loc:@NIN/dense/bias*
use_locking(*
_output_shapes
:*
T0*
validate_shape(

NIN/dense/bias/Adam_1/readIdentityNIN/dense/bias/Adam_1*
T0*
_output_shapes
:*!
_class
loc:@NIN/dense/bias
₯
)NIN/dense_1/kernel/Adam/Initializer/zerosConst*
dtype0*
valueB*    *%
_class
loc:@NIN/dense_1/kernel*
_output_shapes

:
²
NIN/dense_1/kernel/Adam
VariableV2*
shape
:*%
_class
loc:@NIN/dense_1/kernel*
shared_name *
dtype0*
_output_shapes

:*
	container 
ε
NIN/dense_1/kernel/Adam/AssignAssignNIN/dense_1/kernel/Adam)NIN/dense_1/kernel/Adam/Initializer/zeros*%
_class
loc:@NIN/dense_1/kernel*
_output_shapes

:*
T0*
validate_shape(*
use_locking(

NIN/dense_1/kernel/Adam/readIdentityNIN/dense_1/kernel/Adam*%
_class
loc:@NIN/dense_1/kernel*
_output_shapes

:*
T0
§
+NIN/dense_1/kernel/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*%
_class
loc:@NIN/dense_1/kernel*
_output_shapes

:
΄
NIN/dense_1/kernel/Adam_1
VariableV2*
_output_shapes

:*
shape
:*
dtype0*
shared_name *
	container *%
_class
loc:@NIN/dense_1/kernel
λ
 NIN/dense_1/kernel/Adam_1/AssignAssignNIN/dense_1/kernel/Adam_1+NIN/dense_1/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:*%
_class
loc:@NIN/dense_1/kernel*
use_locking(*
T0

NIN/dense_1/kernel/Adam_1/readIdentityNIN/dense_1/kernel/Adam_1*%
_class
loc:@NIN/dense_1/kernel*
_output_shapes

:*
T0

'NIN/dense_1/bias/Adam/Initializer/zerosConst*
_output_shapes
:*
valueB*    *#
_class
loc:@NIN/dense_1/bias*
dtype0
¦
NIN/dense_1/bias/Adam
VariableV2*
shared_name *
_output_shapes
:*#
_class
loc:@NIN/dense_1/bias*
shape:*
dtype0*
	container 
Ω
NIN/dense_1/bias/Adam/AssignAssignNIN/dense_1/bias/Adam'NIN/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
T0*#
_class
loc:@NIN/dense_1/bias*
use_locking(*
_output_shapes
:

NIN/dense_1/bias/Adam/readIdentityNIN/dense_1/bias/Adam*
T0*#
_class
loc:@NIN/dense_1/bias*
_output_shapes
:

)NIN/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB*    *
dtype0*#
_class
loc:@NIN/dense_1/bias*
_output_shapes
:
¨
NIN/dense_1/bias/Adam_1
VariableV2*
	container *
shared_name *
dtype0*#
_class
loc:@NIN/dense_1/bias*
_output_shapes
:*
shape:
ί
NIN/dense_1/bias/Adam_1/AssignAssignNIN/dense_1/bias/Adam_1)NIN/dense_1/bias/Adam_1/Initializer/zeros*#
_class
loc:@NIN/dense_1/bias*
_output_shapes
:*
use_locking(*
T0*
validate_shape(

NIN/dense_1/bias/Adam_1/readIdentityNIN/dense_1/bias/Adam_1*
_output_shapes
:*#
_class
loc:@NIN/dense_1/bias*
T0
₯
)NIN/dense_2/kernel/Adam/Initializer/zerosConst*%
_class
loc:@NIN/dense_2/kernel*
_output_shapes

:*
valueB*    *
dtype0
²
NIN/dense_2/kernel/Adam
VariableV2*
shape
:*
dtype0*
	container *%
_class
loc:@NIN/dense_2/kernel*
_output_shapes

:*
shared_name 
ε
NIN/dense_2/kernel/Adam/AssignAssignNIN/dense_2/kernel/Adam)NIN/dense_2/kernel/Adam/Initializer/zeros*%
_class
loc:@NIN/dense_2/kernel*
use_locking(*
validate_shape(*
T0*
_output_shapes

:

NIN/dense_2/kernel/Adam/readIdentityNIN/dense_2/kernel/Adam*
T0*
_output_shapes

:*%
_class
loc:@NIN/dense_2/kernel
§
+NIN/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:*
dtype0*%
_class
loc:@NIN/dense_2/kernel*
valueB*    
΄
NIN/dense_2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:*%
_class
loc:@NIN/dense_2/kernel*
shape
:*
	container *
shared_name 
λ
 NIN/dense_2/kernel/Adam_1/AssignAssignNIN/dense_2/kernel/Adam_1+NIN/dense_2/kernel/Adam_1/Initializer/zeros*
_output_shapes

:*
use_locking(*
validate_shape(*
T0*%
_class
loc:@NIN/dense_2/kernel

NIN/dense_2/kernel/Adam_1/readIdentityNIN/dense_2/kernel/Adam_1*%
_class
loc:@NIN/dense_2/kernel*
_output_shapes

:*
T0

'NIN/dense_2/bias/Adam/Initializer/zerosConst*
dtype0*
valueB*    *#
_class
loc:@NIN/dense_2/bias*
_output_shapes
:
¦
NIN/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
shape:*#
_class
loc:@NIN/dense_2/bias*
shared_name *
	container *
dtype0
Ω
NIN/dense_2/bias/Adam/AssignAssignNIN/dense_2/bias/Adam'NIN/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
use_locking(*#
_class
loc:@NIN/dense_2/bias*
_output_shapes
:*
T0

NIN/dense_2/bias/Adam/readIdentityNIN/dense_2/bias/Adam*
T0*
_output_shapes
:*#
_class
loc:@NIN/dense_2/bias

)NIN/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
valueB*    *#
_class
loc:@NIN/dense_2/bias*
dtype0
¨
NIN/dense_2/bias/Adam_1
VariableV2*
dtype0*
shared_name *#
_class
loc:@NIN/dense_2/bias*
_output_shapes
:*
	container *
shape:
ί
NIN/dense_2/bias/Adam_1/AssignAssignNIN/dense_2/bias/Adam_1)NIN/dense_2/bias/Adam_1/Initializer/zeros*
T0*
validate_shape(*
use_locking(*#
_class
loc:@NIN/dense_2/bias*
_output_shapes
:

NIN/dense_2/bias/Adam_1/readIdentityNIN/dense_2/bias/Adam_1*#
_class
loc:@NIN/dense_2/bias*
T0*
_output_shapes
:
₯
)NIN/dense_3/kernel/Adam/Initializer/zerosConst*%
_class
loc:@NIN/dense_3/kernel*
valueB*    *
_output_shapes

:*
dtype0
²
NIN/dense_3/kernel/Adam
VariableV2*
shape
:*
dtype0*%
_class
loc:@NIN/dense_3/kernel*
_output_shapes

:*
	container *
shared_name 
ε
NIN/dense_3/kernel/Adam/AssignAssignNIN/dense_3/kernel/Adam)NIN/dense_3/kernel/Adam/Initializer/zeros*
_output_shapes

:*
T0*
use_locking(*%
_class
loc:@NIN/dense_3/kernel*
validate_shape(

NIN/dense_3/kernel/Adam/readIdentityNIN/dense_3/kernel/Adam*%
_class
loc:@NIN/dense_3/kernel*
T0*
_output_shapes

:
§
+NIN/dense_3/kernel/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes

:*
dtype0*%
_class
loc:@NIN/dense_3/kernel
΄
NIN/dense_3/kernel/Adam_1
VariableV2*
dtype0*
shape
:*
_output_shapes

:*%
_class
loc:@NIN/dense_3/kernel*
	container *
shared_name 
λ
 NIN/dense_3/kernel/Adam_1/AssignAssignNIN/dense_3/kernel/Adam_1+NIN/dense_3/kernel/Adam_1/Initializer/zeros*%
_class
loc:@NIN/dense_3/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes

:

NIN/dense_3/kernel/Adam_1/readIdentityNIN/dense_3/kernel/Adam_1*%
_class
loc:@NIN/dense_3/kernel*
_output_shapes

:*
T0

'NIN/dense_3/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *#
_class
loc:@NIN/dense_3/bias
¦
NIN/dense_3/bias/Adam
VariableV2*
dtype0*#
_class
loc:@NIN/dense_3/bias*
shared_name *
shape:*
_output_shapes
:*
	container 
Ω
NIN/dense_3/bias/Adam/AssignAssignNIN/dense_3/bias/Adam'NIN/dense_3/bias/Adam/Initializer/zeros*
validate_shape(*#
_class
loc:@NIN/dense_3/bias*
use_locking(*
T0*
_output_shapes
:

NIN/dense_3/bias/Adam/readIdentityNIN/dense_3/bias/Adam*
T0*
_output_shapes
:*#
_class
loc:@NIN/dense_3/bias

)NIN/dense_3/bias/Adam_1/Initializer/zerosConst*#
_class
loc:@NIN/dense_3/bias*
dtype0*
_output_shapes
:*
valueB*    
¨
NIN/dense_3/bias/Adam_1
VariableV2*#
_class
loc:@NIN/dense_3/bias*
shape:*
shared_name *
	container *
_output_shapes
:*
dtype0
ί
NIN/dense_3/bias/Adam_1/AssignAssignNIN/dense_3/bias/Adam_1)NIN/dense_3/bias/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*#
_class
loc:@NIN/dense_3/bias*
T0*
_output_shapes
:

NIN/dense_3/bias/Adam_1/readIdentityNIN/dense_3/bias/Adam_1*
T0*#
_class
loc:@NIN/dense_3/bias*
_output_shapes
:
₯
)NIN/dense_4/kernel/Adam/Initializer/zerosConst*%
_class
loc:@NIN/dense_4/kernel*
valueB*    *
dtype0*
_output_shapes

:
²
NIN/dense_4/kernel/Adam
VariableV2*
_output_shapes

:*
shape
:*%
_class
loc:@NIN/dense_4/kernel*
	container *
dtype0*
shared_name 
ε
NIN/dense_4/kernel/Adam/AssignAssignNIN/dense_4/kernel/Adam)NIN/dense_4/kernel/Adam/Initializer/zeros*
use_locking(*
_output_shapes

:*
T0*
validate_shape(*%
_class
loc:@NIN/dense_4/kernel

NIN/dense_4/kernel/Adam/readIdentityNIN/dense_4/kernel/Adam*%
_class
loc:@NIN/dense_4/kernel*
_output_shapes

:*
T0
§
+NIN/dense_4/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:*
valueB*    *%
_class
loc:@NIN/dense_4/kernel*
dtype0
΄
NIN/dense_4/kernel/Adam_1
VariableV2*
dtype0*
	container *%
_class
loc:@NIN/dense_4/kernel*
shape
:*
_output_shapes

:*
shared_name 
λ
 NIN/dense_4/kernel/Adam_1/AssignAssignNIN/dense_4/kernel/Adam_1+NIN/dense_4/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*%
_class
loc:@NIN/dense_4/kernel*
T0*
_output_shapes

:

NIN/dense_4/kernel/Adam_1/readIdentityNIN/dense_4/kernel/Adam_1*
_output_shapes

:*%
_class
loc:@NIN/dense_4/kernel*
T0

'NIN/dense_4/bias/Adam/Initializer/zerosConst*
valueB*    *#
_class
loc:@NIN/dense_4/bias*
dtype0*
_output_shapes
:
¦
NIN/dense_4/bias/Adam
VariableV2*
shape:*
	container *
_output_shapes
:*
shared_name *#
_class
loc:@NIN/dense_4/bias*
dtype0
Ω
NIN/dense_4/bias/Adam/AssignAssignNIN/dense_4/bias/Adam'NIN/dense_4/bias/Adam/Initializer/zeros*
_output_shapes
:*#
_class
loc:@NIN/dense_4/bias*
use_locking(*
T0*
validate_shape(

NIN/dense_4/bias/Adam/readIdentityNIN/dense_4/bias/Adam*
T0*#
_class
loc:@NIN/dense_4/bias*
_output_shapes
:

)NIN/dense_4/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*
dtype0*#
_class
loc:@NIN/dense_4/bias*
valueB*    
¨
NIN/dense_4/bias/Adam_1
VariableV2*#
_class
loc:@NIN/dense_4/bias*
_output_shapes
:*
shared_name *
dtype0*
shape:*
	container 
ί
NIN/dense_4/bias/Adam_1/AssignAssignNIN/dense_4/bias/Adam_1)NIN/dense_4/bias/Adam_1/Initializer/zeros*#
_class
loc:@NIN/dense_4/bias*
_output_shapes
:*
validate_shape(*
use_locking(*
T0

NIN/dense_4/bias/Adam_1/readIdentityNIN/dense_4/bias/Adam_1*#
_class
loc:@NIN/dense_4/bias*
_output_shapes
:*
T0
₯
)NIN/dense_5/kernel/Adam/Initializer/zerosConst*
dtype0*
valueB*    *
_output_shapes

:*%
_class
loc:@NIN/dense_5/kernel
²
NIN/dense_5/kernel/Adam
VariableV2*
shared_name *
shape
:*
_output_shapes

:*
dtype0*
	container *%
_class
loc:@NIN/dense_5/kernel
ε
NIN/dense_5/kernel/Adam/AssignAssignNIN/dense_5/kernel/Adam)NIN/dense_5/kernel/Adam/Initializer/zeros*
validate_shape(*
T0*%
_class
loc:@NIN/dense_5/kernel*
_output_shapes

:*
use_locking(

NIN/dense_5/kernel/Adam/readIdentityNIN/dense_5/kernel/Adam*%
_class
loc:@NIN/dense_5/kernel*
T0*
_output_shapes

:
§
+NIN/dense_5/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:*%
_class
loc:@NIN/dense_5/kernel*
valueB*    
΄
NIN/dense_5/kernel/Adam_1
VariableV2*%
_class
loc:@NIN/dense_5/kernel*
shared_name *
	container *
shape
:*
dtype0*
_output_shapes

:
λ
 NIN/dense_5/kernel/Adam_1/AssignAssignNIN/dense_5/kernel/Adam_1+NIN/dense_5/kernel/Adam_1/Initializer/zeros*
validate_shape(*
use_locking(*
T0*
_output_shapes

:*%
_class
loc:@NIN/dense_5/kernel

NIN/dense_5/kernel/Adam_1/readIdentityNIN/dense_5/kernel/Adam_1*%
_class
loc:@NIN/dense_5/kernel*
_output_shapes

:*
T0

'NIN/dense_5/bias/Adam/Initializer/zerosConst*#
_class
loc:@NIN/dense_5/bias*
dtype0*
valueB*    *
_output_shapes
:
¦
NIN/dense_5/bias/Adam
VariableV2*
_output_shapes
:*#
_class
loc:@NIN/dense_5/bias*
dtype0*
shared_name *
shape:*
	container 
Ω
NIN/dense_5/bias/Adam/AssignAssignNIN/dense_5/bias/Adam'NIN/dense_5/bias/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@NIN/dense_5/bias*
_output_shapes
:*
validate_shape(

NIN/dense_5/bias/Adam/readIdentityNIN/dense_5/bias/Adam*#
_class
loc:@NIN/dense_5/bias*
T0*
_output_shapes
:

)NIN/dense_5/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes
:*#
_class
loc:@NIN/dense_5/bias*
dtype0
¨
NIN/dense_5/bias/Adam_1
VariableV2*#
_class
loc:@NIN/dense_5/bias*
	container *
_output_shapes
:*
dtype0*
shape:*
shared_name 
ί
NIN/dense_5/bias/Adam_1/AssignAssignNIN/dense_5/bias/Adam_1)NIN/dense_5/bias/Adam_1/Initializer/zeros*#
_class
loc:@NIN/dense_5/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(

NIN/dense_5/bias/Adam_1/readIdentityNIN/dense_5/bias/Adam_1*
T0*
_output_shapes
:*#
_class
loc:@NIN/dense_5/bias
W
Adam/learning_rateConst*
valueB
 *o:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wΎ?
Q
Adam/epsilonConst*
dtype0*
valueB
 *wΜ+2*
_output_shapes
: 
Π
&Adam/update_NIN/dense/kernel/ApplyAdam	ApplyAdamNIN/dense/kernelNIN/dense/kernel/AdamNIN/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_1*
use_locking( *
use_nesterov( *
T0*#
_class
loc:@NIN/dense/kernel*
_output_shapes

:
Β
$Adam/update_NIN/dense/bias/ApplyAdam	ApplyAdamNIN/dense/biasNIN/dense/bias/AdamNIN/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_2*!
_class
loc:@NIN/dense/bias*
T0*
use_locking( *
use_nesterov( *
_output_shapes
:
Ϊ
(Adam/update_NIN/dense_1/kernel/ApplyAdam	ApplyAdamNIN/dense_1/kernelNIN/dense_1/kernel/AdamNIN/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_3*
use_nesterov( *%
_class
loc:@NIN/dense_1/kernel*
_output_shapes

:*
T0*
use_locking( 
Μ
&Adam/update_NIN/dense_1/bias/ApplyAdam	ApplyAdamNIN/dense_1/biasNIN/dense_1/bias/AdamNIN/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_4*
_output_shapes
:*
T0*
use_locking( *
use_nesterov( *#
_class
loc:@NIN/dense_1/bias
Ϊ
(Adam/update_NIN/dense_2/kernel/ApplyAdam	ApplyAdamNIN/dense_2/kernelNIN/dense_2/kernel/AdamNIN/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_5*
_output_shapes

:*
T0*%
_class
loc:@NIN/dense_2/kernel*
use_locking( *
use_nesterov( 
Μ
&Adam/update_NIN/dense_2/bias/ApplyAdam	ApplyAdamNIN/dense_2/biasNIN/dense_2/bias/AdamNIN/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_6*
use_nesterov( *
T0*
use_locking( *#
_class
loc:@NIN/dense_2/bias*
_output_shapes
:
Ϊ
(Adam/update_NIN/dense_3/kernel/ApplyAdam	ApplyAdamNIN/dense_3/kernelNIN/dense_3/kernel/AdamNIN/dense_3/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_7*
use_locking( *%
_class
loc:@NIN/dense_3/kernel*
use_nesterov( *
T0*
_output_shapes

:
Μ
&Adam/update_NIN/dense_3/bias/ApplyAdam	ApplyAdamNIN/dense_3/biasNIN/dense_3/bias/AdamNIN/dense_3/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_8*
use_nesterov( *
T0*#
_class
loc:@NIN/dense_3/bias*
use_locking( *
_output_shapes
:
Ϊ
(Adam/update_NIN/dense_4/kernel/ApplyAdam	ApplyAdamNIN/dense_4/kernelNIN/dense_4/kernel/AdamNIN/dense_4/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_9*
T0*%
_class
loc:@NIN/dense_4/kernel*
use_locking( *
use_nesterov( *
_output_shapes

:
Ν
&Adam/update_NIN/dense_4/bias/ApplyAdam	ApplyAdamNIN/dense_4/biasNIN/dense_4/bias/AdamNIN/dense_4/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_10*
T0*
_output_shapes
:*#
_class
loc:@NIN/dense_4/bias*
use_nesterov( *
use_locking( 
Ϋ
(Adam/update_NIN/dense_5/kernel/ApplyAdam	ApplyAdamNIN/dense_5/kernelNIN/dense_5/kernel/AdamNIN/dense_5/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_11*
use_locking( *
T0*
use_nesterov( *%
_class
loc:@NIN/dense_5/kernel*
_output_shapes

:
Ν
&Adam/update_NIN/dense_5/bias/ApplyAdam	ApplyAdamNIN/dense_5/biasNIN/dense_5/bias/AdamNIN/dense_5/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonMean_12*
use_nesterov( *
T0*
_output_shapes
:*#
_class
loc:@NIN/dense_5/bias*
use_locking( 
ε
Adam/mulMulbeta1_power/read
Adam/beta1%^Adam/update_NIN/dense/bias/ApplyAdam'^Adam/update_NIN/dense/kernel/ApplyAdam'^Adam/update_NIN/dense_1/bias/ApplyAdam)^Adam/update_NIN/dense_1/kernel/ApplyAdam'^Adam/update_NIN/dense_2/bias/ApplyAdam)^Adam/update_NIN/dense_2/kernel/ApplyAdam'^Adam/update_NIN/dense_3/bias/ApplyAdam)^Adam/update_NIN/dense_3/kernel/ApplyAdam'^Adam/update_NIN/dense_4/bias/ApplyAdam)^Adam/update_NIN/dense_4/kernel/ApplyAdam'^Adam/update_NIN/dense_5/bias/ApplyAdam)^Adam/update_NIN/dense_5/kernel/ApplyAdam*!
_class
loc:@NIN/dense/bias*
T0*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_output_shapes
: *!
_class
loc:@NIN/dense/bias*
validate_shape(*
use_locking( 
η

Adam/mul_1Mulbeta2_power/read
Adam/beta2%^Adam/update_NIN/dense/bias/ApplyAdam'^Adam/update_NIN/dense/kernel/ApplyAdam'^Adam/update_NIN/dense_1/bias/ApplyAdam)^Adam/update_NIN/dense_1/kernel/ApplyAdam'^Adam/update_NIN/dense_2/bias/ApplyAdam)^Adam/update_NIN/dense_2/kernel/ApplyAdam'^Adam/update_NIN/dense_3/bias/ApplyAdam)^Adam/update_NIN/dense_3/kernel/ApplyAdam'^Adam/update_NIN/dense_4/bias/ApplyAdam)^Adam/update_NIN/dense_4/kernel/ApplyAdam'^Adam/update_NIN/dense_5/bias/ApplyAdam)^Adam/update_NIN/dense_5/kernel/ApplyAdam*!
_class
loc:@NIN/dense/bias*
T0*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
_output_shapes
: *
validate_shape(*!
_class
loc:@NIN/dense/bias*
T0

AdamNoOp^Adam/Assign^Adam/Assign_1%^Adam/update_NIN/dense/bias/ApplyAdam'^Adam/update_NIN/dense/kernel/ApplyAdam'^Adam/update_NIN/dense_1/bias/ApplyAdam)^Adam/update_NIN/dense_1/kernel/ApplyAdam'^Adam/update_NIN/dense_2/bias/ApplyAdam)^Adam/update_NIN/dense_2/kernel/ApplyAdam'^Adam/update_NIN/dense_3/bias/ApplyAdam)^Adam/update_NIN/dense_3/kernel/ApplyAdam'^Adam/update_NIN/dense_4/bias/ApplyAdam)^Adam/update_NIN/dense_4/kernel/ApplyAdam'^Adam/update_NIN/dense_5/bias/ApplyAdam)^Adam/update_NIN/dense_5/kernel/ApplyAdam

NIN_1/dense/MatMulMatMulinput_TNIN/dense/kernel/read*'
_output_shapes
:?????????*
T0*
transpose_a( *
transpose_b( 

NIN_1/dense/BiasAddBiasAddNIN_1/dense/MatMulNIN/dense/bias/read*
T0*'
_output_shapes
:?????????*
data_formatNHWC

NIN_1/dense/swish_f32	swish_f32NIN_1/dense/BiasAdd*#
_disable_call_shape_inference(*'
_output_shapes
:?????????
¦
NIN_1/dense_1/MatMulMatMulNIN_1/dense/swish_f32NIN/dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:?????????

NIN_1/dense_1/BiasAddBiasAddNIN_1/dense_1/MatMulNIN/dense_1/bias/read*
data_formatNHWC*'
_output_shapes
:?????????*
T0

NIN_1/dense_1/swish_f32	swish_f32NIN_1/dense_1/BiasAdd*#
_disable_call_shape_inference(*'
_output_shapes
:?????????
¨
NIN_1/dense_2/MatMulMatMulNIN_1/dense_1/swish_f32NIN/dense_2/kernel/read*
transpose_a( *
T0*'
_output_shapes
:?????????*
transpose_b( 

NIN_1/dense_2/BiasAddBiasAddNIN_1/dense_2/MatMulNIN/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:?????????*
T0

NIN_1/dense_3/MatMulMatMulinput_XNIN/dense_3/kernel/read*'
_output_shapes
:?????????*
T0*
transpose_b( *
transpose_a( 

NIN_1/dense_3/BiasAddBiasAddNIN_1/dense_3/MatMulNIN/dense_3/bias/read*
T0*'
_output_shapes
:?????????*
data_formatNHWC

NIN_1/dense_3/swish_f32	swish_f32NIN_1/dense_3/BiasAdd*#
_disable_call_shape_inference(*'
_output_shapes
:?????????
¨
NIN_1/dense_4/MatMulMatMulNIN_1/dense_3/swish_f32NIN/dense_4/kernel/read*
transpose_b( *
T0*'
_output_shapes
:?????????*
transpose_a( 

NIN_1/dense_4/BiasAddBiasAddNIN_1/dense_4/MatMulNIN/dense_4/bias/read*'
_output_shapes
:?????????*
T0*
data_formatNHWC

NIN_1/dense_4/swish_f32	swish_f32NIN_1/dense_4/BiasAdd*#
_disable_call_shape_inference(*'
_output_shapes
:?????????
v
	NIN_1/addAddV2NIN_1/dense_3/swish_f32NIN_1/dense_4/swish_f32*'
_output_shapes
:?????????*
T0

NIN_1/dense_5/MatMulMatMul	NIN_1/addNIN/dense_5/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:?????????

NIN_1/dense_5/BiasAddBiasAddNIN_1/dense_5/MatMulNIN/dense_5/bias/read*
T0*'
_output_shapes
:?????????*
data_formatNHWC

NIN_1/dense_5/swish_f32	swish_f32NIN_1/dense_5/BiasAdd*'
_output_shapes
:?????????*#
_disable_call_shape_inference(
j
NIN_1/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
l
NIN_1/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
l
NIN_1/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
²
NIN_1/strided_sliceStridedSliceNIN_1/dense_2/BiasAddNIN_1/strided_slice/stackNIN_1/strided_slice/stack_1NIN_1/strided_slice/stack_2*
ellipsis_mask *'
_output_shapes
:?????????*
shrink_axis_mask *
new_axis_mask *
T0*
Index0*
end_mask*

begin_mask
h
NIN_1/Reshape/shapeConst*
dtype0*!
valueB"????      *
_output_shapes
:

NIN_1/ReshapeReshapeNIN_1/strided_sliceNIN_1/Reshape/shape*+
_output_shapes
:?????????*
Tshape0*
T0
l
NIN_1/strided_slice_1/stackConst*
dtype0*
valueB"    ????*
_output_shapes
:
n
NIN_1/strided_slice_1/stack_1Const*
valueB"        *
dtype0*
_output_shapes
:
n
NIN_1/strided_slice_1/stack_2Const*
valueB"      *
_output_shapes
:*
dtype0
Ά
NIN_1/strided_slice_1StridedSliceNIN_1/dense_2/BiasAddNIN_1/strided_slice_1/stackNIN_1/strided_slice_1/stack_1NIN_1/strided_slice_1/stack_2*
T0*
ellipsis_mask *
Index0*
new_axis_mask *
end_mask*#
_output_shapes
:?????????*
shrink_axis_mask*

begin_mask
f
NIN_1/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   

NIN_1/Reshape_1ReshapeNIN_1/strided_slice_1NIN_1/Reshape_1/shape*
Tshape0*
T0*'
_output_shapes
:?????????
i
NIN_1/einsum/ShapeShapeNIN_1/dense_5/swish_f32*
T0*
_output_shapes
:*
out_type0
j
 NIN_1/einsum/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
l
"NIN_1/einsum/strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0
l
"NIN_1/einsum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ί
NIN_1/einsum/strided_sliceStridedSliceNIN_1/einsum/Shape NIN_1/einsum/strided_slice/stack"NIN_1/einsum/strided_slice/stack_1"NIN_1/einsum/strided_slice/stack_2*
end_mask *
ellipsis_mask *

begin_mask *
T0*
Index0*
new_axis_mask *
_output_shapes
: *
shrink_axis_mask
^
NIN_1/einsum/Reshape/shape/1Const*
_output_shapes
: *
value	B :*
dtype0
^
NIN_1/einsum/Reshape/shape/2Const*
value	B :*
_output_shapes
: *
dtype0
΄
NIN_1/einsum/Reshape/shapePackNIN_1/einsum/strided_sliceNIN_1/einsum/Reshape/shape/1NIN_1/einsum/Reshape/shape/2*
N*
T0*

axis *
_output_shapes
:

NIN_1/einsum/ReshapeReshapeNIN_1/dense_5/swish_f32NIN_1/einsum/Reshape/shape*
T0*
Tshape0*+
_output_shapes
:?????????
a
NIN_1/einsum/Shape_1ShapeNIN_1/Reshape*
out_type0*
_output_shapes
:*
T0
l
"NIN_1/einsum/strided_slice_1/stackConst*
valueB: *
_output_shapes
:*
dtype0
n
$NIN_1/einsum/strided_slice_1/stack_1Const*
valueB:*
_output_shapes
:*
dtype0
n
$NIN_1/einsum/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Δ
NIN_1/einsum/strided_slice_1StridedSliceNIN_1/einsum/Shape_1"NIN_1/einsum/strided_slice_1/stack$NIN_1/einsum/strided_slice_1/stack_1$NIN_1/einsum/strided_slice_1/stack_2*
ellipsis_mask *
Index0*
end_mask *
new_axis_mask *
_output_shapes
: *

begin_mask *
shrink_axis_mask*
T0
`
NIN_1/einsum/Reshape_1/shape/1Const*
value	B :*
_output_shapes
: *
dtype0
`
NIN_1/einsum/Reshape_1/shape/2Const*
_output_shapes
: *
value	B :*
dtype0
Ό
NIN_1/einsum/Reshape_1/shapePackNIN_1/einsum/strided_slice_1NIN_1/einsum/Reshape_1/shape/1NIN_1/einsum/Reshape_1/shape/2*
_output_shapes
:*
N*

axis *
T0

NIN_1/einsum/Reshape_1ReshapeNIN_1/ReshapeNIN_1/einsum/Reshape_1/shape*
T0*
Tshape0*+
_output_shapes
:?????????
’
NIN_1/einsum/MatMulBatchMatMulV2NIN_1/einsum/ReshapeNIN_1/einsum/Reshape_1*+
_output_shapes
:?????????*
adj_x( *
T0*
adj_y( 
`
NIN_1/einsum/Reshape_2/shape/1Const*
value	B :*
_output_shapes
: *
dtype0

NIN_1/einsum/Reshape_2/shapePackNIN_1/einsum/strided_sliceNIN_1/einsum/Reshape_2/shape/1*
_output_shapes
:*

axis *
T0*
N

NIN_1/einsum/Reshape_2ReshapeNIN_1/einsum/MatMulNIN_1/einsum/Reshape_2/shape*'
_output_shapes
:?????????*
Tshape0*
T0
o
NIN_1/add_1AddV2NIN_1/einsum/Reshape_2NIN_1/Reshape_1*'
_output_shapes
:?????????*
T0
S
output_uIdentityNIN_1/add_1*'
_output_shapes
:?????????*
T0
U
sub_1Suboutput_uPlaceholder*
T0*'
_output_shapes
:?????????
K
Square_1Squaresub_1*
T0*'
_output_shapes
:?????????
X
Const_1Const*
dtype0*
valueB"       *
_output_shapes
:
`
Mean_13MeanSquare_1Const_1*
T0*
	keep_dims( *

Tidx0*
_output_shapes
: 
	
initNoOp^NIN/dense/bias/Adam/Assign^NIN/dense/bias/Adam_1/Assign^NIN/dense/bias/Assign^NIN/dense/kernel/Adam/Assign^NIN/dense/kernel/Adam_1/Assign^NIN/dense/kernel/Assign^NIN/dense_1/bias/Adam/Assign^NIN/dense_1/bias/Adam_1/Assign^NIN/dense_1/bias/Assign^NIN/dense_1/kernel/Adam/Assign!^NIN/dense_1/kernel/Adam_1/Assign^NIN/dense_1/kernel/Assign^NIN/dense_2/bias/Adam/Assign^NIN/dense_2/bias/Adam_1/Assign^NIN/dense_2/bias/Assign^NIN/dense_2/kernel/Adam/Assign!^NIN/dense_2/kernel/Adam_1/Assign^NIN/dense_2/kernel/Assign^NIN/dense_3/bias/Adam/Assign^NIN/dense_3/bias/Adam_1/Assign^NIN/dense_3/bias/Assign^NIN/dense_3/kernel/Adam/Assign!^NIN/dense_3/kernel/Adam_1/Assign^NIN/dense_3/kernel/Assign^NIN/dense_4/bias/Adam/Assign^NIN/dense_4/bias/Adam_1/Assign^NIN/dense_4/bias/Assign^NIN/dense_4/kernel/Adam/Assign!^NIN/dense_4/kernel/Adam_1/Assign^NIN/dense_4/kernel/Assign^NIN/dense_5/bias/Adam/Assign^NIN/dense_5/bias/Adam_1/Assign^NIN/dense_5/bias/Assign^NIN/dense_5/kernel/Adam/Assign!^NIN/dense_5/kernel/Adam_1/Assign^NIN/dense_5/kernel/Assign^beta1_power/Assign^beta2_power/Assign
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
shape: *
dtype0
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 

save/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_959144a428974934ae7536444935026b/part*
dtype0
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
\
save/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
¬
save/SaveV2/tensor_namesConst*ί
valueΥB?&BNIN/dense/biasBNIN/dense/bias/AdamBNIN/dense/bias/Adam_1BNIN/dense/kernelBNIN/dense/kernel/AdamBNIN/dense/kernel/Adam_1BNIN/dense_1/biasBNIN/dense_1/bias/AdamBNIN/dense_1/bias/Adam_1BNIN/dense_1/kernelBNIN/dense_1/kernel/AdamBNIN/dense_1/kernel/Adam_1BNIN/dense_2/biasBNIN/dense_2/bias/AdamBNIN/dense_2/bias/Adam_1BNIN/dense_2/kernelBNIN/dense_2/kernel/AdamBNIN/dense_2/kernel/Adam_1BNIN/dense_3/biasBNIN/dense_3/bias/AdamBNIN/dense_3/bias/Adam_1BNIN/dense_3/kernelBNIN/dense_3/kernel/AdamBNIN/dense_3/kernel/Adam_1BNIN/dense_4/biasBNIN/dense_4/bias/AdamBNIN/dense_4/bias/Adam_1BNIN/dense_4/kernelBNIN/dense_4/kernel/AdamBNIN/dense_4/kernel/Adam_1BNIN/dense_5/biasBNIN/dense_5/bias/AdamBNIN/dense_5/bias/Adam_1BNIN/dense_5/kernelBNIN/dense_5/kernel/AdamBNIN/dense_5/kernel/Adam_1Bbeta1_powerBbeta2_power*
_output_shapes
:&*
dtype0
―
save/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:&*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
γ
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesNIN/dense/biasNIN/dense/bias/AdamNIN/dense/bias/Adam_1NIN/dense/kernelNIN/dense/kernel/AdamNIN/dense/kernel/Adam_1NIN/dense_1/biasNIN/dense_1/bias/AdamNIN/dense_1/bias/Adam_1NIN/dense_1/kernelNIN/dense_1/kernel/AdamNIN/dense_1/kernel/Adam_1NIN/dense_2/biasNIN/dense_2/bias/AdamNIN/dense_2/bias/Adam_1NIN/dense_2/kernelNIN/dense_2/kernel/AdamNIN/dense_2/kernel/Adam_1NIN/dense_3/biasNIN/dense_3/bias/AdamNIN/dense_3/bias/Adam_1NIN/dense_3/kernelNIN/dense_3/kernel/AdamNIN/dense_3/kernel/Adam_1NIN/dense_4/biasNIN/dense_4/bias/AdamNIN/dense_4/bias/Adam_1NIN/dense_4/kernelNIN/dense_4/kernel/AdamNIN/dense_4/kernel/Adam_1NIN/dense_5/biasNIN/dense_5/bias/AdamNIN/dense_5/bias/Adam_1NIN/dense_5/kernelNIN/dense_5/kernel/AdamNIN/dense_5/kernel/Adam_1beta1_powerbeta2_power*4
dtypes*
(2&

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*
N*

axis *
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
―
save/RestoreV2/tensor_namesConst*ί
valueΥB?&BNIN/dense/biasBNIN/dense/bias/AdamBNIN/dense/bias/Adam_1BNIN/dense/kernelBNIN/dense/kernel/AdamBNIN/dense/kernel/Adam_1BNIN/dense_1/biasBNIN/dense_1/bias/AdamBNIN/dense_1/bias/Adam_1BNIN/dense_1/kernelBNIN/dense_1/kernel/AdamBNIN/dense_1/kernel/Adam_1BNIN/dense_2/biasBNIN/dense_2/bias/AdamBNIN/dense_2/bias/Adam_1BNIN/dense_2/kernelBNIN/dense_2/kernel/AdamBNIN/dense_2/kernel/Adam_1BNIN/dense_3/biasBNIN/dense_3/bias/AdamBNIN/dense_3/bias/Adam_1BNIN/dense_3/kernelBNIN/dense_3/kernel/AdamBNIN/dense_3/kernel/Adam_1BNIN/dense_4/biasBNIN/dense_4/bias/AdamBNIN/dense_4/bias/Adam_1BNIN/dense_4/kernelBNIN/dense_4/kernel/AdamBNIN/dense_4/kernel/Adam_1BNIN/dense_5/biasBNIN/dense_5/bias/AdamBNIN/dense_5/bias/Adam_1BNIN/dense_5/kernelBNIN/dense_5/kernel/AdamBNIN/dense_5/kernel/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:&
²
save/RestoreV2/shape_and_slicesConst*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Μ
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*?
_output_shapes
::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&
¦
save/AssignAssignNIN/dense/biassave/RestoreV2*
T0*!
_class
loc:@NIN/dense/bias*
use_locking(*
validate_shape(*
_output_shapes
:
―
save/Assign_1AssignNIN/dense/bias/Adamsave/RestoreV2:1*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*!
_class
loc:@NIN/dense/bias
±
save/Assign_2AssignNIN/dense/bias/Adam_1save/RestoreV2:2*
use_locking(*
validate_shape(*
T0*!
_class
loc:@NIN/dense/bias*
_output_shapes
:
²
save/Assign_3AssignNIN/dense/kernelsave/RestoreV2:3*
T0*
_output_shapes

:*#
_class
loc:@NIN/dense/kernel*
validate_shape(*
use_locking(
·
save/Assign_4AssignNIN/dense/kernel/Adamsave/RestoreV2:4*
validate_shape(*
use_locking(*
_output_shapes

:*#
_class
loc:@NIN/dense/kernel*
T0
Ή
save/Assign_5AssignNIN/dense/kernel/Adam_1save/RestoreV2:5*#
_class
loc:@NIN/dense/kernel*
use_locking(*
validate_shape(*
_output_shapes

:*
T0
?
save/Assign_6AssignNIN/dense_1/biassave/RestoreV2:6*
T0*
_output_shapes
:*
use_locking(*#
_class
loc:@NIN/dense_1/bias*
validate_shape(
³
save/Assign_7AssignNIN/dense_1/bias/Adamsave/RestoreV2:7*
T0*
_output_shapes
:*
validate_shape(*
use_locking(*#
_class
loc:@NIN/dense_1/bias
΅
save/Assign_8AssignNIN/dense_1/bias/Adam_1save/RestoreV2:8*#
_class
loc:@NIN/dense_1/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
Ά
save/Assign_9AssignNIN/dense_1/kernelsave/RestoreV2:9*
T0*%
_class
loc:@NIN/dense_1/kernel*
validate_shape(*
use_locking(*
_output_shapes

:
½
save/Assign_10AssignNIN/dense_1/kernel/Adamsave/RestoreV2:10*
validate_shape(*%
_class
loc:@NIN/dense_1/kernel*
_output_shapes

:*
use_locking(*
T0
Ώ
save/Assign_11AssignNIN/dense_1/kernel/Adam_1save/RestoreV2:11*
use_locking(*
validate_shape(*
T0*
_output_shapes

:*%
_class
loc:@NIN/dense_1/kernel
°
save/Assign_12AssignNIN/dense_2/biassave/RestoreV2:12*
validate_shape(*#
_class
loc:@NIN/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
΅
save/Assign_13AssignNIN/dense_2/bias/Adamsave/RestoreV2:13*
validate_shape(*#
_class
loc:@NIN/dense_2/bias*
T0*
_output_shapes
:*
use_locking(
·
save/Assign_14AssignNIN/dense_2/bias/Adam_1save/RestoreV2:14*
validate_shape(*#
_class
loc:@NIN/dense_2/bias*
use_locking(*
_output_shapes
:*
T0
Έ
save/Assign_15AssignNIN/dense_2/kernelsave/RestoreV2:15*
validate_shape(*
use_locking(*%
_class
loc:@NIN/dense_2/kernel*
T0*
_output_shapes

:
½
save/Assign_16AssignNIN/dense_2/kernel/Adamsave/RestoreV2:16*
T0*
use_locking(*%
_class
loc:@NIN/dense_2/kernel*
validate_shape(*
_output_shapes

:
Ώ
save/Assign_17AssignNIN/dense_2/kernel/Adam_1save/RestoreV2:17*
use_locking(*%
_class
loc:@NIN/dense_2/kernel*
_output_shapes

:*
T0*
validate_shape(
°
save/Assign_18AssignNIN/dense_3/biassave/RestoreV2:18*#
_class
loc:@NIN/dense_3/bias*
use_locking(*
T0*
_output_shapes
:*
validate_shape(
΅
save/Assign_19AssignNIN/dense_3/bias/Adamsave/RestoreV2:19*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*#
_class
loc:@NIN/dense_3/bias
·
save/Assign_20AssignNIN/dense_3/bias/Adam_1save/RestoreV2:20*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*#
_class
loc:@NIN/dense_3/bias
Έ
save/Assign_21AssignNIN/dense_3/kernelsave/RestoreV2:21*%
_class
loc:@NIN/dense_3/kernel*
T0*
use_locking(*
validate_shape(*
_output_shapes

:
½
save/Assign_22AssignNIN/dense_3/kernel/Adamsave/RestoreV2:22*
use_locking(*%
_class
loc:@NIN/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0
Ώ
save/Assign_23AssignNIN/dense_3/kernel/Adam_1save/RestoreV2:23*
use_locking(*%
_class
loc:@NIN/dense_3/kernel*
_output_shapes

:*
validate_shape(*
T0
°
save/Assign_24AssignNIN/dense_4/biassave/RestoreV2:24*
_output_shapes
:*#
_class
loc:@NIN/dense_4/bias*
T0*
validate_shape(*
use_locking(
΅
save/Assign_25AssignNIN/dense_4/bias/Adamsave/RestoreV2:25*
T0*
use_locking(*#
_class
loc:@NIN/dense_4/bias*
validate_shape(*
_output_shapes
:
·
save/Assign_26AssignNIN/dense_4/bias/Adam_1save/RestoreV2:26*#
_class
loc:@NIN/dense_4/bias*
_output_shapes
:*
validate_shape(*
T0*
use_locking(
Έ
save/Assign_27AssignNIN/dense_4/kernelsave/RestoreV2:27*
validate_shape(*%
_class
loc:@NIN/dense_4/kernel*
T0*
_output_shapes

:*
use_locking(
½
save/Assign_28AssignNIN/dense_4/kernel/Adamsave/RestoreV2:28*
validate_shape(*
use_locking(*
T0*%
_class
loc:@NIN/dense_4/kernel*
_output_shapes

:
Ώ
save/Assign_29AssignNIN/dense_4/kernel/Adam_1save/RestoreV2:29*%
_class
loc:@NIN/dense_4/kernel*
validate_shape(*
_output_shapes

:*
T0*
use_locking(
°
save/Assign_30AssignNIN/dense_5/biassave/RestoreV2:30*
validate_shape(*
_output_shapes
:*
use_locking(*#
_class
loc:@NIN/dense_5/bias*
T0
΅
save/Assign_31AssignNIN/dense_5/bias/Adamsave/RestoreV2:31*#
_class
loc:@NIN/dense_5/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
:
·
save/Assign_32AssignNIN/dense_5/bias/Adam_1save/RestoreV2:32*
T0*
validate_shape(*
use_locking(*
_output_shapes
:*#
_class
loc:@NIN/dense_5/bias
Έ
save/Assign_33AssignNIN/dense_5/kernelsave/RestoreV2:33*
use_locking(*
validate_shape(*%
_class
loc:@NIN/dense_5/kernel*
T0*
_output_shapes

:
½
save/Assign_34AssignNIN/dense_5/kernel/Adamsave/RestoreV2:34*%
_class
loc:@NIN/dense_5/kernel*
_output_shapes

:*
T0*
use_locking(*
validate_shape(
Ώ
save/Assign_35AssignNIN/dense_5/kernel/Adam_1save/RestoreV2:35*%
_class
loc:@NIN/dense_5/kernel*
use_locking(*
_output_shapes

:*
T0*
validate_shape(
₯
save/Assign_36Assignbeta1_powersave/RestoreV2:36*
_output_shapes
: *
T0*
use_locking(*
validate_shape(*!
_class
loc:@NIN/dense/bias
₯
save/Assign_37Assignbeta2_powersave/RestoreV2:37*
_output_shapes
: *
validate_shape(*
use_locking(*
T0*!
_class
loc:@NIN/dense/bias

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard°	
λ
a
swish_grad_f32f32
features
grad	
mul_22)Gradient of Swish function defined below.-
SigmoidSigmoidfeatures"/gpu:0*
T0:
sub/xConst"/gpu:0*
valueB
 *  ?*
dtype08
subSubsub/x:output:0Sigmoid:y:0"/gpu:0*
T0.
mulMulfeaturessub:z:0"/gpu:0*
T0:
add/xConst"/gpu:0*
valueB
 *  ?*
dtype06
addAddV2add/x:output:0mul:z:0"/gpu:0*
T03
mul_1MulSigmoid:y:0add:z:0"/gpu:0*
T00
mul_2_0Mulgrad	mul_1:z:0"/gpu:0*
T0"
mul_2mul_2_0:z:0*
	_noinline(*#
_disable_call_shape_inference(:  : 

ι
	swish_f32
features
mul2ΔComputes the Swish activation function: `x * sigmoid(x)`.

  Source: "Searching for Activation Functions" (Ramachandran et al. 2017)
  https://arxiv.org/abs/1710.05941

  Args:
    features: A `Tensor` representing preactivation values.
    name: A name for the operation (optional).

  Returns:
    The activation value.
  -
SigmoidSigmoidfeatures"/gpu:0*
T04
mul_0MulfeaturesSigmoid:y:0"/gpu:0*
T0"
mul	mul_0:z:0*#
_disable_call_shape_inference(*
	_noinline(:  
	swish_f32swish_grad_f32f32"w<
save/Const:0save/Identity:0save/restore_all (5 @F8"'
	variables?&ό&
y
NIN/dense/kernel:0NIN/dense/kernel/AssignNIN/dense/kernel/read:02/NIN/dense/kernel/Initializer/truncated_normal:08
q
NIN/dense/bias:0NIN/dense/bias/AssignNIN/dense/bias/read:02-NIN/dense/bias/Initializer/truncated_normal:08

NIN/dense_1/kernel:0NIN/dense_1/kernel/AssignNIN/dense_1/kernel/read:021NIN/dense_1/kernel/Initializer/truncated_normal:08
y
NIN/dense_1/bias:0NIN/dense_1/bias/AssignNIN/dense_1/bias/read:02/NIN/dense_1/bias/Initializer/truncated_normal:08

NIN/dense_2/kernel:0NIN/dense_2/kernel/AssignNIN/dense_2/kernel/read:021NIN/dense_2/kernel/Initializer/truncated_normal:08
y
NIN/dense_2/bias:0NIN/dense_2/bias/AssignNIN/dense_2/bias/read:02/NIN/dense_2/bias/Initializer/truncated_normal:08

NIN/dense_3/kernel:0NIN/dense_3/kernel/AssignNIN/dense_3/kernel/read:021NIN/dense_3/kernel/Initializer/truncated_normal:08
y
NIN/dense_3/bias:0NIN/dense_3/bias/AssignNIN/dense_3/bias/read:02/NIN/dense_3/bias/Initializer/truncated_normal:08

NIN/dense_4/kernel:0NIN/dense_4/kernel/AssignNIN/dense_4/kernel/read:021NIN/dense_4/kernel/Initializer/truncated_normal:08
y
NIN/dense_4/bias:0NIN/dense_4/bias/AssignNIN/dense_4/bias/read:02/NIN/dense_4/bias/Initializer/truncated_normal:08

NIN/dense_5/kernel:0NIN/dense_5/kernel/AssignNIN/dense_5/kernel/read:021NIN/dense_5/kernel/Initializer/truncated_normal:08
y
NIN/dense_5/bias:0NIN/dense_5/bias/AssignNIN/dense_5/bias/read:02/NIN/dense_5/bias/Initializer/truncated_normal:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

NIN/dense/kernel/Adam:0NIN/dense/kernel/Adam/AssignNIN/dense/kernel/Adam/read:02)NIN/dense/kernel/Adam/Initializer/zeros:0

NIN/dense/kernel/Adam_1:0NIN/dense/kernel/Adam_1/AssignNIN/dense/kernel/Adam_1/read:02+NIN/dense/kernel/Adam_1/Initializer/zeros:0
x
NIN/dense/bias/Adam:0NIN/dense/bias/Adam/AssignNIN/dense/bias/Adam/read:02'NIN/dense/bias/Adam/Initializer/zeros:0

NIN/dense/bias/Adam_1:0NIN/dense/bias/Adam_1/AssignNIN/dense/bias/Adam_1/read:02)NIN/dense/bias/Adam_1/Initializer/zeros:0

NIN/dense_1/kernel/Adam:0NIN/dense_1/kernel/Adam/AssignNIN/dense_1/kernel/Adam/read:02+NIN/dense_1/kernel/Adam/Initializer/zeros:0

NIN/dense_1/kernel/Adam_1:0 NIN/dense_1/kernel/Adam_1/Assign NIN/dense_1/kernel/Adam_1/read:02-NIN/dense_1/kernel/Adam_1/Initializer/zeros:0

NIN/dense_1/bias/Adam:0NIN/dense_1/bias/Adam/AssignNIN/dense_1/bias/Adam/read:02)NIN/dense_1/bias/Adam/Initializer/zeros:0

NIN/dense_1/bias/Adam_1:0NIN/dense_1/bias/Adam_1/AssignNIN/dense_1/bias/Adam_1/read:02+NIN/dense_1/bias/Adam_1/Initializer/zeros:0

NIN/dense_2/kernel/Adam:0NIN/dense_2/kernel/Adam/AssignNIN/dense_2/kernel/Adam/read:02+NIN/dense_2/kernel/Adam/Initializer/zeros:0

NIN/dense_2/kernel/Adam_1:0 NIN/dense_2/kernel/Adam_1/Assign NIN/dense_2/kernel/Adam_1/read:02-NIN/dense_2/kernel/Adam_1/Initializer/zeros:0

NIN/dense_2/bias/Adam:0NIN/dense_2/bias/Adam/AssignNIN/dense_2/bias/Adam/read:02)NIN/dense_2/bias/Adam/Initializer/zeros:0

NIN/dense_2/bias/Adam_1:0NIN/dense_2/bias/Adam_1/AssignNIN/dense_2/bias/Adam_1/read:02+NIN/dense_2/bias/Adam_1/Initializer/zeros:0

NIN/dense_3/kernel/Adam:0NIN/dense_3/kernel/Adam/AssignNIN/dense_3/kernel/Adam/read:02+NIN/dense_3/kernel/Adam/Initializer/zeros:0

NIN/dense_3/kernel/Adam_1:0 NIN/dense_3/kernel/Adam_1/Assign NIN/dense_3/kernel/Adam_1/read:02-NIN/dense_3/kernel/Adam_1/Initializer/zeros:0

NIN/dense_3/bias/Adam:0NIN/dense_3/bias/Adam/AssignNIN/dense_3/bias/Adam/read:02)NIN/dense_3/bias/Adam/Initializer/zeros:0

NIN/dense_3/bias/Adam_1:0NIN/dense_3/bias/Adam_1/AssignNIN/dense_3/bias/Adam_1/read:02+NIN/dense_3/bias/Adam_1/Initializer/zeros:0

NIN/dense_4/kernel/Adam:0NIN/dense_4/kernel/Adam/AssignNIN/dense_4/kernel/Adam/read:02+NIN/dense_4/kernel/Adam/Initializer/zeros:0

NIN/dense_4/kernel/Adam_1:0 NIN/dense_4/kernel/Adam_1/Assign NIN/dense_4/kernel/Adam_1/read:02-NIN/dense_4/kernel/Adam_1/Initializer/zeros:0

NIN/dense_4/bias/Adam:0NIN/dense_4/bias/Adam/AssignNIN/dense_4/bias/Adam/read:02)NIN/dense_4/bias/Adam/Initializer/zeros:0

NIN/dense_4/bias/Adam_1:0NIN/dense_4/bias/Adam_1/AssignNIN/dense_4/bias/Adam_1/read:02+NIN/dense_4/bias/Adam_1/Initializer/zeros:0

NIN/dense_5/kernel/Adam:0NIN/dense_5/kernel/Adam/AssignNIN/dense_5/kernel/Adam/read:02+NIN/dense_5/kernel/Adam/Initializer/zeros:0

NIN/dense_5/kernel/Adam_1:0 NIN/dense_5/kernel/Adam_1/Assign NIN/dense_5/kernel/Adam_1/read:02-NIN/dense_5/kernel/Adam_1/Initializer/zeros:0

NIN/dense_5/bias/Adam:0NIN/dense_5/bias/Adam/AssignNIN/dense_5/bias/Adam/read:02)NIN/dense_5/bias/Adam/Initializer/zeros:0

NIN/dense_5/bias/Adam_1:0NIN/dense_5/bias/Adam_1/AssignNIN/dense_5/bias/Adam_1/read:02+NIN/dense_5/bias/Adam_1/Initializer/zeros:0"
trainable_variablesμι
y
NIN/dense/kernel:0NIN/dense/kernel/AssignNIN/dense/kernel/read:02/NIN/dense/kernel/Initializer/truncated_normal:08
q
NIN/dense/bias:0NIN/dense/bias/AssignNIN/dense/bias/read:02-NIN/dense/bias/Initializer/truncated_normal:08

NIN/dense_1/kernel:0NIN/dense_1/kernel/AssignNIN/dense_1/kernel/read:021NIN/dense_1/kernel/Initializer/truncated_normal:08
y
NIN/dense_1/bias:0NIN/dense_1/bias/AssignNIN/dense_1/bias/read:02/NIN/dense_1/bias/Initializer/truncated_normal:08

NIN/dense_2/kernel:0NIN/dense_2/kernel/AssignNIN/dense_2/kernel/read:021NIN/dense_2/kernel/Initializer/truncated_normal:08
y
NIN/dense_2/bias:0NIN/dense_2/bias/AssignNIN/dense_2/bias/read:02/NIN/dense_2/bias/Initializer/truncated_normal:08

NIN/dense_3/kernel:0NIN/dense_3/kernel/AssignNIN/dense_3/kernel/read:021NIN/dense_3/kernel/Initializer/truncated_normal:08
y
NIN/dense_3/bias:0NIN/dense_3/bias/AssignNIN/dense_3/bias/read:02/NIN/dense_3/bias/Initializer/truncated_normal:08

NIN/dense_4/kernel:0NIN/dense_4/kernel/AssignNIN/dense_4/kernel/read:021NIN/dense_4/kernel/Initializer/truncated_normal:08
y
NIN/dense_4/bias:0NIN/dense_4/bias/AssignNIN/dense_4/bias/read:02/NIN/dense_4/bias/Initializer/truncated_normal:08

NIN/dense_5/kernel:0NIN/dense_5/kernel/AssignNIN/dense_5/kernel/read:021NIN/dense_5/kernel/Initializer/truncated_normal:08
y
NIN/dense_5/bias:0NIN/dense_5/bias/AssignNIN/dense_5/bias/read:02/NIN/dense_5/bias/Initializer/truncated_normal:08"
train_op

Adam*έ
serving_defaultΙ
)
u$
Placeholder:0?????????
%
t 
	input_T:0?????????
%
x 
	input_X:0?????????2
eval_output_u!

output_u:0?????????tensorflow/serving/predict