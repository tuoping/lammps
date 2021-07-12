DEEPMD_ROOT=/home/pint/soft/deepmd-1.2.0_root-debug
TENSORFLOW_INCLUDE_DIRS="/home/pint/soft/1.14_cuda10.1_cuDNN7/include;/home/pint/soft/1.14_cuda10.1_cuDNN7/include"
TENSORFLOW_LIBRARY_PATH="/home/pint/soft/1.14_cuda10.1_cuDNN7/lib;/home/pint/soft/1.14_cuda10.1_cuDNN7/lib"

TF_INCLUDE_DIRS=`echo $TENSORFLOW_INCLUDE_DIRS | sed "s/;/ -I/g"`
TF_LIBRARY_PATH=`echo $TENSORFLOW_LIBRARY_PATH | sed "s/;/ -L/g"`
TF_RPATH=`echo $TENSORFLOW_LIBRARY_PATH | sed "s/;/ -Wl,-rpath=/g"`

NNP_INC=" -std=c++11 -DHIGH_PREC   -I$TF_INCLUDE_DIRS -I$DEEPMD_ROOT/include/deepmd "
NNP_PATH=" -L$TF_LIBRARY_PATH -L$DEEPMD_ROOT/lib"
NNP_LIB=" -Wl,--no-as-needed -ldeepmd_op -ldeepmd_op -ldeepmd -ltensorflow_cc -ltensorflow_framework -Wl,-rpath=$TF_RPATH -Wl,-rpath=$DEEPMD_ROOT/lib"
