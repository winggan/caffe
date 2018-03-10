export CMAKE_PREFIX_PATH=$MY_LIBRARY_PATH:/home/wing/intel/mkl:$CMAKE_PREFIX_PATH
export CMAKE_EXE=/home/wing/exe/bin/cmake
$CMAKE_EXE -DBUILD_SHARED_LIBS=ON -DUSE_LMDB=ON -DBLAS=MKL -DUSE_LEVELDB=OFF -DUSE_NCCL=ON -DALLOW_LMDB_NOLOCK=ON ..

#to build with python3:
#1. install python3-devel
#2. build boost with python3
#  here take python3.4 as example
#  sh bootstrap.sh  --with-python=python3.4m 
#3. add "-Dpython_version=3" to the cmake command above