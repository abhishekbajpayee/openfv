OpenFV
======

Berlin's work on OpenFv. Build3 branch is for OpenFV with OpenCV3, Build2 branch is for OpenFV with OpenCV2.

Both builds are curretly built without CUDA, which means their std_include,h has WITHOUT_CUDA defined. Some of the code runs CUDA commands so having this header variable defined helps us keep the actual code clean while not breaking anything.
