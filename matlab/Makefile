CC=ifort
CFLAGS1=-c -O3 -heap-arrays -ipo -xHost -fexceptions -fPIC -qopenmp -qopt-report
CFLAGS2=-O3 -shared -fPIC -static-intel -heap-arrays -qopenmp -qopt-report -fast -ipo -xHost

MEXFLAGS=-c -O3 -heap-arrays -fast -fexceptions -ipo -qopenmp -qopt-report -fPIC -DMATLAB_MEX_FILE -DMX_COMPAT_32 -xHost

MKLROOT=/opt/intel/mkl
INTELROOT=/opt/intel/lib/intel64
TMWROOT=/usr/local/MATLAB/R2015a
PIV3DROOT=$(shell pwd)/piv/pivAlgorithms/libpiv3d

MKLFLAGS=-Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread

.cte: $(PIV3DROOT)/cte3d.F90
	cd $(PIV3DROOT) && $(CC) $(CFLAGS1) -I$(MKLROOT)/include -I$(MKLROOT)/examples/dftf/source cte3d.F90

.libpiv3d: .cte
	cd $(PIV3DROOT) && $(CC) $(CFLAGS2) -Wl,-Bdynamic,-liomp5,-lpthread,-Bstatic -o libpiv3d.so cte3d.o $(INTELROOT)/libifcoremt_pic.a -liomp5 $(MKLFLAGS) -Wl,--version-script=libpiv3d.version

.mex: .libpiv3d
	cd $(PIV3DROOT) && $(CC) $(MEXFLAGS) -I$(TMWROOT)/extern/include mex_vodim3d.F90
	cd $(PIV3DROOT) && $(CC) $(MEXFLAGS) -I$(TMWROOT)/extern/include mex_cte3d.F90

.link: .mex
	cd $(PIV3DROOT) && $(CC) $(CFLAGS2) -Wl,--version-script,$(TMWROOT)/extern/lib/glnxa64/fexport.map -Wl,--no-undefined -o mex_vodim3d.mexa64 mex_vodim3d.o $(INTELROOT)/libifcoremt_pic.a -L$(PIV3DROOT) -lpiv3d -Wl,-rpath-link,$(TMWROOT)/bin/glnxa64 -L$(TMWROOT)/bin/glnxa64 -lm -lmx -lmex -lmat -liomp5
	cd $(PIV3DROOT) && $(CC) $(CFLAGS2) -Wl,--version-script,$(TMWROOT)/extern/lib/glnxa64/fexport.map -Wl,--no-undefined -o mex_cte3d.mexa64 mex_cte3d.o $(INTELROOT)/libifcoremt_pic.a -L$(PIV3DROOT) -lpiv3d -Wl,-rpath-link,$(TMWROOT)/bin/glnxa64 -L$(TMWROOT)/bin/glnxa64 -lm -lmx -lmex -lmat -liomp5

piv: .link
	cd $(PIV3DROOT) && rm *.mod && rm *.o && rm *.optrpt
