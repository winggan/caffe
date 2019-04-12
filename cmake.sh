export CMAKE_PREFIX_PATH=$MY_LIBRARY_PATH:$CMAKE_PREFIX_PATH
export CMAKE_EXE=/home/wing/exe/bin/cmake
export PYTHON_EXE=~/python36/bin/python3.6m
export PYTHON_LIB=~/python36/lib/libpython3.6m.so
export PYTHON_INC=~/python36/include/python3.6m
$CMAKE_EXE -DBUILD_SHARED_LIBS=ON \
  -DUSE_LMDB=ON \
  -DBLAS=open \
  -DUSE_LEVELDB=OFF \
  -DUSE_NCCL=ON \
  -DALLOW_LMDB_NOLOCK=ON \
  -Dpython_version=3 \
  -DPYTHON_EXECUTABLE=$PYTHON_EXE \
  -DPYTHON_LIBRARIES=$PYTHON_LIB \
  -DPYTHON_INCLUDE_DIR=$PYTHON_INC ..

#to build with python3, need boost with python3 support:
#1. install python3-devel
#2. build boost with python3
#  here take python3.4 as example
#  sh bootstrap.sh  --with-python=python3.4m
 
