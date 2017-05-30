!mex_vodim3d.F90   Performs windowed cross correlation of two 
!              volumetric scalar fields
!
!
! Matlab Syntax: 
!		[ur uc up snr] = mex_vodim3d(fieldA, fieldB, c0, c1, c2, c3, c4, c5, c6, c7, wSize); 
!
!
! Inputs:
!
!		fieldA   	[nvox_Y x nvox_X x nvox_Z] single
!								Intensity distribution in the first
!								reconstructed volume. Should be 
!								non-negative. See note on ordering.
!
!		fieldB   	[nvox_Y x nvox_X x nvox_Z] single
!								Intensity distribution in the second
!								reconstructed volume. Should be 
!								non-negative. See note on ordering.
!
!
!
!
! Outputs:
!		
! 
! Other files required:     fintrf.h (matlab MEX interface header file)
!                           libpiv3d.so (piv3d library file)
!                           mkl_dfti.F90 (Intel Math Kernel Library file)
!
! Subroutines:              none
!
! MAT-files required:       none
!
! Future Improvements:      none foreseen
!
! Platform Independance
!	mwpointer, mwsize and mwindex types have been used, along with the 
!	fintrf.h preprocessor directives file to ensure that this mex file 
!	compiles on both 32 and 64 bit systems.
!
!	Apple-Mac compatibility: The Difference Between .f and .F Files.
!	Fortran compilers assume source files using a lowercase .f file 
!	extension have been preprocessed. On most platforms, mex makes sure 
!	the file is preprocessed regardless of the file extension. However, 
!	on Apple® Macintosh® platforms, mex cannot force preprocessing. Use 
!	an uppercase .F or .F90 file extension to ensure your Fortran 
!	MEX-file is platform independent.
!
! Compilation Commands
!
!	See the compile_fMexPIV.m compiation script for compiler commands.
!	Note - no attempt has been made to generalise these. They are the 
!   ones that work for my system. General users should either adapt 
!   these, or use the 'mex' command in MATLAB. Adapting these might be a
!	good idea, as they contain speed optimisations that MATLAB normally 
!	won't apply with the mex command.
!
!
! Useful References:
!	[1] Raffel M. Willert C. Wereley S and Kompenhans J. 
!		"Particle Image Velocimetry (A Practical Guide)" 
!		2nd ed. Springer, ISBN 978-3-540-72307-3
!
!   [2] Elsinga G.E. Scarano F. Wienke B. and van Oudheusden B.W. (2006)
!       Tomographic Particle Image Velocimetry, Exp. Fluids 41:933-947
!
!	[3] Commentary by the authors of Numerical Recipes 3rd Edition
!		(Numerical Recipes for MATLAB) http://www.nr.com/nr3_matlab.html
!
!
! Author:               T.H. Clark
! Work address:         Fluids Laboratory
!                       Cambridge University Engineering Department
!                       2 Trumpington Street
!                       Cambridge
!                       CB21PZ
! Email:                t.clark@cantab.net
! Website:              http://www2.eng.cam.ac.uk/~thc29
!
! Revision History:     02 July 2011        Created

! ____________________________ INCLUSIONS ______________________________

! Inclusion of the preprocessor header file, allowing the fortran 
! compiler to correctly interpret what integer type mwSize, mwIndex and 
! mwPointer are (i.e. it automatically determines between integer*4 and 
! integer*8 depending on whether you're using 32 or 64 bit system. Thus
! using this helps make mex files platform-independant)
! Important part for reminder:
!#if defined(__x86_64__)
!# define mwPointer integer*8
!#else
!# define mwPointer integer*4
!#endif
!#if defined(MX_COMPAT_32)
!# define mwSize  integer*4
!# define mwIndex integer*4
!# define mwSignedIndex integer*4
!#else
!# define mwSize  mwPointer
!# define mwIndex mwPointer
!# define mwSignedIndex mwPointer
!#endif
#include "fintrf.h"

! Inclusion of James Tursa's Interface functions
#include "MatlabAPImex.f"
#include "MatlabAPImx.f"



!_______________________________________________________________________
!
! GATEWAY FUNCTION FOR MEX FILE
!_______________________________________________________________________

      SUBROUTINE mexFunction(nlhs, plhs, nrhs, prhs)

!     Use James Tursa's excellent MATLAB API for FORTRAN 95 to provide a 
!     proper interface to the MATLAB API functions
      USE MatlabAPImex
      USE MatlabAPImx
      

!     Safer to have no implicit variables
      IMPLICIT NONE
      
! 	  The number of left and right hand side arguments
      INTEGER*4 nlhs, nrhs

!	  Arrays containing pointers to the locations in memory of left and 
!	  right hand side arguments
      mwPointer plhs(*), prhs(*)

!	  Internal variables used to administer the gateway
      INTEGER*4 :: clsid
	  INTEGER*4 :: complexFlag
	  mwSize :: dims(2)
	  INTEGER*4 :: fieldDims(3)
      INTEGER*4, DIMENSION(1) :: ISC
      REAL*4 , DIMENSION(1) :: RSC
      INTEGER*4 :: ERRCODE

!     Declare pointers to scalar input arguments and their corresponding fortran local scalar variables
      mwPointer :: xptrWSIZE, xptrIWMAXDISP, xptrFETCHTYPE
      INTEGER*4 :: WSIZE, FETCHTYPE
      REAL*4 :: IWMAXDISP
     
!     FORTRAN pointers to input/output arrays
      REAL*4, POINTER :: fieldAptr(:,:,:)
      REAL*4, POINTER :: fieldBptr(:,:,:)
      REAL*4, POINTER :: c0ptr(:,:)
      REAL*4, POINTER :: c1ptr(:,:)
      REAL*4, POINTER :: c2ptr(:,:)
      REAL*4, POINTER :: c3ptr(:,:)
      REAL*4, POINTER :: c4ptr(:,:)
      REAL*4, POINTER :: c5ptr(:,:)
      REAL*4, POINTER :: c6ptr(:,:)
      REAL*4, POINTER :: c7ptr(:,:)
      REAL*4, POINTER :: urptr(:,:)
      REAL*4, POINTER :: ucptr(:,:)
      REAL*4, POINTER :: upptr(:,:)
      REAL*4, POINTER :: snrptr(:,:)

!     OTHER (admin variables)
      INTEGER*4 :: NWINDOWS, NCORRS

!     GATEWAY CODE (INPUT ARGUMENTS)

!     Get pointers to the scalar input arguments out of the prhs array   
      xptrWSIZE     = mxGetPr(prhs(11))
      xptrIWMAXDISP = mxGetPr(prhs(12))
	  xptrFETCHTYPE = mxGetPr(prhs(13))
	  
!     Get single precision pointers to the input arrays
      fieldAptr => fpGetPr3Single(prhs(1))
      fieldBptr => fpGetPr3Single(prhs(2))
      c0ptr => fpGetPr2Single(prhs(3))
      c1ptr => fpGetPr2Single(prhs(4))
      c2ptr => fpGetPr2Single(prhs(5))
      c3ptr => fpGetPr2Single(prhs(6))
      c4ptr => fpGetPr2Single(prhs(7))
      c5ptr => fpGetPr2Single(prhs(8))
      c6ptr => fpGetPr2Single(prhs(9))
      c7ptr => fpGetPr2Single(prhs(10))


!     Make local copies of scalar input arguments. NB because we're
!     using James Tursa's routines (reducing memory copy overhead for 
!     arrays) we have to deal with the weird behaviour when coping with 
!     scalars by copying via a single element array.
      CALL mxCopyPtrToInteger4(xptrWSIZE,     ISC, 1)
      WSIZE     = ISC(1)
      CALL mxCopyPtrToReal4(   xptrIWMAXDISP, RSC, 1)
      IWMAXDISP = RSC(1)
      CALL mxCopyPtrToInteger4(xptrFETCHTYPE, ISC, 1)
      FETCHTYPE = ISC(1)
      

!     Determine sizes of input arguments where required
	  NWINDOWS = mxGetM(prhs(5))
      NCORRS = NWINDOWS/2
      CALL mxCopyPtrToInteger4(mxGetDimensions(prhs(1)),fieldDims,&
                                      mxGetNumberOfDimensions(prhs(1)))



!     GATEWAY CODE (OUTPUT ARGUMENTS)

!     Create arrays (in the MATLAB workspace) for the return arguments. 
      clsid = mxClassIDFromClassName('single')
      dims(1) = NCORRS
      dims(2) = 2
	  complexFlag = 0
      plhs(1) = mxCreateNumericMatrix(dims(1), dims(2), &
									           clsid, complexFlag)
      plhs(2) = mxCreateNumericMatrix(dims(1), dims(2), &
									           clsid, complexFlag)
      plhs(3) = mxCreateNumericMatrix(dims(1), dims(2), &
									           clsid, complexFlag)
      plhs(4) = mxCreateNumericMatrix(dims(1), dims(2), &
									           clsid, complexFlag)

!     Get pointer for the return argument
      urptr => fpGetPr2Single(plhs(1))
      ucptr => fpGetPr2Single(plhs(2))
      upptr => fpGetPr2Single(plhs(3))
      snrptr => fpGetPr2Single(plhs(4))



!     CALL COMPUTATIONAL SUBROUTINE  

      CALL vodim3d(WSIZE, NWINDOWS, NCORRS, &
                         fieldDims(1), fieldDims(2), fieldDims(3), &
                         0, IWMAXDISP, &
                         fieldAptr, fieldBptr, FETCHTYPE, &
                         c0ptr, c1ptr, c2ptr, c3ptr, c4ptr, c5ptr, &
                         c6ptr, c7ptr, urptr, ucptr, upptr, snrptr, &
                         ERRCODE)




! End the main routine (mexFunction)
	  RETURN
      END SUBROUTINE mexFunction


    


