#!/bin/sh


# Old parallelised compilation commands
#ifort -c  -fast -ipo -parallel -par-num-threads=4 -xSSSE3 -I/usr/matlab7.8/extern/include -I/usr/matlab7.8/simulink/include -fexceptions -fPIC  -DMX_COMPAT_32 -O3  "mxmart_large.F90"
#ifort -O3 -fast -ipo -parallel -par-num-threads=4 -xSSSE3 -shared -fPIC -Wl,--version-script,/usr/matlab7.8/extern/lib/glnxa64/fexport.map -Wl,--no-undefined -Wl,-Bstatic,-lsvml,-Bdynamic -o  "mxmart_large.mexa64"  mxmart_large.o  -Wl,-rpath-link,/usr/matlab7.8/bin/glnxa64 -L/usr/matlab7.8/bin/glnxa64 -lmx -lmex -lmat -lm -Wl,-Bstatic,-lsvml,-Bdynamic



# Set up root directories
TMWROOT="/usr/local/MATLAB/R2015a"
MKLROOT="/opt/intel/mkl"

# Compilation FLags (debugging or optimised run alternative)
#COMPILEFLAGS="-c -g -debug -check bounds -heap-arrays -fpstkchk -inline_debug_info -debug extended -fexceptions -fPIC -DMX_COMPAT_32"
#COMPILEFLAGS="-c -g -heap-arrays -O3 -fast -unroll32 -fexceptions -ipo -fPIC -DMX_COMPAT_32 -xSSSE3"
COMPILEFLAGS="-c -O3 -heap-arrays -fast -fexceptions -ipo -fPIC -DMX_COMPAT_32 -xSSSE3"

# Linker Flags (debugging, optimised profiling or optimised run alternative)
#LINKFLAGS="-g -shared -fPIC"
#LINKFLAGS=" -g -O3 -fast -fPIC -ipo -shared -xSSSE3"
LINKFLAGS="-O3 -fast -fPIC -ipo -shared -xSSSE3"

# Compile
ifort $COMPILEFLAGS -I$TMWROOT/extern/include -I$MKLROOT/include -I$MKLROOT/examples/dftf/source "mxmart_large.F90"

# Link
ifort $LINKFLAGS -Wl,--version-script,$TMWROOT/extern/lib/glnxa64/fexport.map -Wl,--no-undefined -Wl,-Bstatic,-lsvml,-lirc,-Bdynamic -o  "mxmart_large.mexa64"  mxmart_large.o  -Wl,-rpath-link,$TMWROOT/bin/glnxa64,-rpath,/opt/intel/composerxe-2011.1.107/compiler/lib/intel64/ -L$TMWROOT/bin/glnxa64 -lmx -lmex -lmat -lm -Wl,-Bstatic,-lsvml,-lirc,-Bdynamic
#ifort $LINKFLAGS -Wl,--version-script,$TMWROOT/extern/lib/glnxa64/fexport.map -Wl,--no-undefined -o  "mxmart_large.mexa64"  mxmart_large.o  -Wl,-rpath-link,$TMWROOT/bin/glnxa64 -L$TMWROOT/bin/glnxa64 -lmx -lmex -lmat -lm
rm mxmart_large.o
rm weights_mod.mod






