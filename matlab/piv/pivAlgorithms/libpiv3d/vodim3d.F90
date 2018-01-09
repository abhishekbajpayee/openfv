!_______________________________________________________________________
!
!	  				SUBROUTINE vodim3d
!
!   Fetch and cross correlate window pairs - this performs a 
!   single pass of the VODIM technique. No window convection is done.
!
!   Constitutes part of the libpiv3d library.
!
!
!	Inputs:
!
!       WSIZE           [1 x 1] integer*4
!                               dimension of the correlation cube array
!
!       NCORRS        [1 x 1] integer*4
!                               number of window pairs to fetch
!
!       fd*             [1 x 1] integer*4 
!                               Dimensions of the FIELD* arrays in 
!                               row (fd1), col (fd2), page (fd3) 
!                               directions 
!
!       CC_SMOOTH_FLAG  [1 x 1] integer*4
!                               * NOT CURRENTLY IMPLEMENTED *
!                               A value of 1 causes the correlation 
!                               plane to be smoothed using a gaussian 
!                               kernel filter. Default 0 = off.
!
!		IW_MAX_D		    [1 x 1]	real*4
!								Maximum displacement of an interrogation
!								window during the cross correlation, 
!                               as a percentage of window size.
!                               For example, a value of 50.0 means that
!                               any displacement over 50% 
!                               of the interrogation window size is 
!                               considered impossible.
!                               This is useful, since edge effects 
!                               caused by the FFT can result in 
!                               erroneously high elements at the corners
!                               of the correlation volume, resulting in 
!                               false vectors.
!
!       WEIGHT          [WSIZE x WSIZE x WSIZE] real*4 array
!                               The debiasing array for the cross 
!                               correlation plane. This is initialised 
!                               using the GetWeights subfunction
!
!		FIELDX   	    [nvox_X x nvox_Y x nvox_Z] real*4 array
!								Intensity distribution in the
!								reconstructed volumes. Must be 
!								non-negative. See note on ordering.
!                               X= B,C for different snapshots 
!                               separated by time dt (B -> t, C -> t+dt)
!
!       FETCHTYPE       [1 x 1] integer*4
!                               1 causes direct fetch of the window 
!                               (based on the nearest integer to the 
!                               first 'C0' corner of the array and the 
!                               window size)
!                               2 Causes fetch by linear interpolation 
!                               from the image
!                               3 Causes fetch by 5^3 extent Whittaker 
!                               (aka Cardinal, aka sinc) interpolation
!                               from the image.
!                               4 Causes fetch by 7^3 extent Whittaker 
!                               (aka Cardinal, aka sinc) interpolation
!                               from the image.
!                               All fetch types except for No. 1 use 
!                               linear interpolation between the window 
!                               corners to ascertain locations to 
!                               interpolate from the 
!                               images/reconstructions
!
!       CX              [NCORRS*2 x 3] real*4 array
!                               Where X is 0,1,2,3,4,5,6,7 and the 
!                               numbers correspond to corners of the 
!                               interrogation window. See 
!                                   windowOrdering.jpg 
!                               for details of window numbering
!                               arrangement. IMPORTANT: each row 
!                               of CX contains a 3 element (noninteger)
!                               index into the intensity array. The 
!                               triplet is ordered as [col, row, page] 
!                               indices, NOT [row, col,..] as might be 
!                               expected.
!                               CX is NCORRS*2 in length: The first 
!                               NCORRS rows describe windows from 
!                               intensity array B, the second half of 
!                               the rows point into field C.
!
!       UR, UC, UP      [NCORRS x 2] real*4 array
!                               Each row contains the X, Y or Z , NWINDOWS
!                               component of velocity for the window in 
!                               the CX arrays corresponding to that row.
!                               The first column contains the primary
!                               (strongest) peak in the correlation
!                               array, the second column contains the
!                               secondary peak.
!
!       SNR             [NCORRS x 2] real*4 array
!                               Each row contains the signal to noise
!                               ratio peakValue/meanValue in the
!                               correlation volume. Column 1 contains 
!                               the SNR for the primary peak, column 2 
!                               contains SNR for secondary peak.
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
! Revision History:     17 September 2011   Created as part of the 
!                                           libpiv3d library



! Inclusion of header file required for Intel Math Kernel Library FFT
! libraries
! #include "mkl_dfti.f90"
! #include "mkl_service.f90"

! Inclusion of the module. This module includes all subroutines and 
! declares some shared arrays.
! #include "mod_piv3d.F90"
! #include "cte3d.F90"

SUBROUTINE vodim3d(WSIZE, NWINDOWS, NCORRS, fd1, fd2, fd3, &
     CC_SMOOTH_FLAG, IWMAXDISP, &
     FIELDA, FIELDB, FETCHTYPE, &
     C0, C1, C2, C3, C4, C5, C6, C7, &
     UR, UC, UP, SNR, ERRCODE)
  
  !	  Using Intel MKL for the FFTs
  USE MKL_DFTI
  USE MKL_SERVICE
  
  !     Use the OMP library so we can call functions to set max numbers of
  !     threads etc
  USE omp_lib
  
  !     Use the piv3d module which contains all our subroutines and a 
  !     nice explicit inteface to them :)
  USE mod_piv3d


  IMPLICIT NONE

  !     Include the Intel Math Kernel Library examples file, which
  !     contains parameters and interfaces useful for FFT manipulation
  INCLUDE 'mkl_dfti_examples.fi'

  !     Inputs
  INTEGER*4, INTENT(in) :: WSIZE, NWINDOWS, NCORRS, fd1, fd2, fd3
  INTEGER*4, INTENT(in) :: CC_SMOOTH_FLAG
  REAL*4,    INTENT(in) :: IWMAXDISP
  REAL*4, DIMENSION(fd1, fd2, fd3), INTENT(in) :: FIELDA, FIELDB
  INTEGER*4, INTENT(in) :: FETCHTYPE
  REAL*4, DIMENSION(NWINDOWS, 3), INTENT(in) :: C0, C1, C2, C3,&
       C4, C5, C6, C7
  
  !     Outputs
  REAL*4, DIMENSION(NCORRS,2), INTENT(out) :: UR, UC, UP, SNR
  INTEGER*4, INTENT(out) :: ERRCODE

  !     Internal Variables                                                     
  TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandle    ! DFT plan for this window size
  TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleInv ! IDFT plan for this window size
  INTEGER*4 :: WCTR                               ! Counter
  REAL*4, DIMENSION(WSIZE,WSIZE,WSIZE) :: WA,WB   ! Local arrays for the windows
  INTEGER*4 :: THREADERRCODE
  REAL*4, DIMENSION(WSIZE, WSIZE, WSIZE) :: WEIGHT
  INTEGER :: stcksz

  !     Admin variables for FFT descriptors
  INTEGER*4 :: stat
  REAL*4    :: fftScale
  INTEGER*4 :: lengths(3)
  INTEGER*4 :: strides_in(4)
  INTEGER*4 :: maxThreads

  !     Initialise
  ERRCODE = 0
  THREADERRCODE = 0
  UR = 0.0
  UC = 0.0
  UP = 0.0
  SNR = 0.0

  !     Get the maximum number of threads
  IF (FETCHTYPE.GT.2) THEN  
     ! Alter this if adding parallelism to other loops some parts and not others
     !       maxThreads = omp_get_max_threads()
     !       Temp debug: Handle the weird thing on our 12 core nehalem pc
     !       IF (maxThreads.EQ.0) THEN
     !           maxThreads = 1
     !       ENDIF
     ! IT IS KEY TO MAKE THIS NUMBER LESS THAN THE TOTAL CORES
     ! ON THE MACHINE OTHERWISE CRAZY SHIT WILL LIKELY ENSUE
     maxThreads = 16
  ELSE
     maxThreads = 1
  ENDIF
  
      
  !     Calculate the weightings array to debias the correlation peak
  CALL getWeight(WSIZE, WEIGHT)


  !     Set the stack size - doing this using environment variables 
  !     doesn't cut it - MATLAB fucks about with things, even when you use
  !     MATLAB's own setenv command to set them - just don't.  Its a 
  !     nightmare (literally - writing this 3am 7 days from PhD deadline. 
  !     SO fucked off with MATLAB and OMP right now). 
  !     So we hard-set it here. If you start getting weird segmentation 
  !     violations, the first thing to do is to try raising this...
  CALL kmp_set_stacksize_s(536870912)  ! 512 MB

  !     If you want, you can check it with the following...
  !        stcksz = kmp_get_stacksize_s()


  !_______________________________________________________________________
  !
  !     SETUP FFTS
  !     This is ugly but for some reason I can't make it work in a subroutine
  !_______________________________________________________________________


  !     Set up the FFT descriptors for the cross correlations
  !      CALL setupCrossCorr(WSIZE, DescHandle, DescHandleInv)


  !     Setup lengths and strides for correlation FFTs  (always cubes)
  lengths(1) = WSIZE
  lengths(2) = WSIZE
  lengths(3) = WSIZE
  strides_in(1) = 0
  strides_in(2) = 1
  strides_in(3) = WSIZE
  strides_in(4) = WSIZE*WSIZE

  

  !     Set the number of threads
  CALL omp_set_num_threads(maxThreads)

  !     Set MKL to single threaded (I'm explicitly parallelising it - although the hyperthreading thing might allow us to  do 2 threads....
  stat = 0;
  ! stat = mkl_domain_set_num_threads(1, MKL_FFT)
  stat = mkl_domain_set_num_threads(1, MKL_DOMAIN_FFT)

  !     Create DFTI descriptors for forward and inverse window transforms
  stat = DftiCreateDescriptor( DescHandle, DFTI_SINGLE, DFTI_COMPLEX, 3, lengths)
  !      IF (HandleFFTError(stat).EQ.1) THEN CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Error in creation of Forward Window FFT descriptor') ENDIF
  stat = DftiCreateDescriptor( DescHandleInv, DFTI_SINGLE, DFTI_COMPLEX, 3, lengths)
  !      IF (HandleFFTError(stat).EQ.1) THEN CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Error in creation of Inverse Window FFT descriptor') ENDIF
  !     Set the stride pattern for DFTIs
  stat = DftiSetValue( DescHandle, DFTI_INPUT_STRIDES, strides_in)
  !      IF (HandleFFTError(stat).EQ.1) THEN CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Cannot set stride pattern for Forward Window FFT') ENDIF
  !     Set the maximum number of threads
  stat = DftiSetValue (DescHandle, DFTI_NUMBER_OF_USER_THREADS, maxThreads)
  !     Commit Dfti descriptors
  stat = DftiCommitDescriptor( DescHandle )
  !      IF (HandleFFTError(stat).EQ.1) THEN CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Cannot commit Forward Window FFT descriptor') ENDIF 
  !     Set fft scale for inverse transforms
  fftScale = 1.0/real(WSIZE*WSIZE*WSIZE, kind=4)
  stat     = DftiSetValue(DescHandleInv, DFTI_BACKWARD_SCALE, fftScale)
  !      IF (HandleFFTError(stat).EQ.1) THEN CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Cannot set backward scale value for Inverse Window FFT') ENDIF
  !     Set the maximum number of threads
  stat = DftiSetValue (DescHandleInv, DFTI_NUMBER_OF_USER_THREADS, maxThreads)
  !     Commit Dfti descriptor for backwards transform
  stat = DftiCommitDescriptor( DescHandleInv )
  !      IF (HandleFFTError(stat).EQ.1) THEN CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Cannot commit Inverse Window FFT descriptor') ENDIF
  
  
  
  
  !_______________________________________________________________________
  !
  !     PERFORM PARALLELISED CROSS CORRELATION
  !_______________________________________________________________________
  
  
  !     Outer IF statement allows us to use different parallel constructs
  !     for different fetch types. We loop for each window pair to be 
  !     cross correlated
  IF (FETCHTYPE.EQ.1) THEN

     ! OMP PARALLEL DO DEFAULT(NONE) SHARED(NCORRS, NWINDOWS, WSIZE, CC_SMOOTH_FLAG, IWMAXDISP, WEIGHT, fd1, fd2, fd3, DescHandle, DescHandleInv, C0, C1, C2, C3, C4, C5, C6, C7, FIELDA, FIELDB, UR, UC, UP, SNR, ERRCODE) PRIVATE(WA,WB, THREADERRCODE) SCHEDULE(DYNAMIC,10)
     DO WCTR = 1,NCORRS

        ! Retrieve Windows A and B using direct indexing
        CALL indexedFetch( WCTR,        WSIZE, NWINDOWS, fd1, fd2, fd3, C0, C1, C2, C3, C4, C5, C6, C7, FIELDA, WA)
        CALL indexedFetch( WCTR+NCORRS, WSIZE, NWINDOWS, fd1, fd2, fd3, C0, C1, C2, C3, C4, C5, C6, C7, FIELDB, WB)

        ! Cross Correlate Window A with Window B
        CALL crossCorr(WCTR, WSIZE, NCORRS, CC_SMOOTH_FLAG, IWMAXDISP, WEIGHT, DescHandle, DescHandleInv, WA, WB, UR, UC, UP, SNR)

     ENDDO
     ! OMP END PARALLEL DO


  ELSEIF (FETCHTYPE.EQ.2) THEN


     ! OMP PARALLEL DO DEFAULT(NONE) SHARED(NCORRS, NWINDOWS, WSIZE, CC_SMOOTH_FLAG, IWMAXDISP, WEIGHT, fd1, fd2, fd3, DescHandle, DescHandleInv, C0, C1, C2, C3, C4, C5, C6, C7, FIELDA, FIELDB, UR, UC, UP, SNR, ERRCODE) PRIVATE(WA,WB, THREADERRCODE) SCHEDULE(DYNAMIC,50)
     DO WCTR = 1,NCORRS

        ! Retrieve Windows A and B using linear interpolation
        CALL triLInterp( WCTR,        WSIZE, NWINDOWS, fd1, fd2, fd3, C0, C1, C2, C3, C4, C5, C6, C7, FIELDA, WA)
        CALL triLInterp( WCTR+NCORRS, WSIZE, NWINDOWS, fd1, fd2, fd3, C0, C1, C2, C3, C4, C5, C6, C7, FIELDB, WB)

        ! Cross Correlate Window A with Window B
        CALL crossCorr(WCTR, WSIZE, NCORRS, CC_SMOOTH_FLAG, IWMAXDISP, WEIGHT, DescHandle, DescHandleInv, WA, WB, UR, UC, UP, SNR)

     ENDDO
     ! OMP END PARALLEL DO


  ELSEIF (FETCHTYPE.EQ.3) THEN


     !$OMP PARALLEL DO DEFAULT(NONE) SHARED(NCORRS, NWINDOWS, WSIZE, CC_SMOOTH_FLAG, IWMAXDISP, WEIGHT, fd1, fd2, fd3, DescHandle, DescHandleInv, C0, C1, C2, C3, C4, C5, C6, C7, FIELDA, FIELDB, UR, UC, UP, SNR, ERRCODE) PRIVATE(WA,WB, THREADERRCODE) SCHEDULE(STATIC,50)
     DO WCTR = 1,NCORRS

        ! Retrieve Windows A and B using the 7^3 Whittaker Cardinal Function
        CALL whitCardInterp5( WCTR,        WSIZE, NWINDOWS, fd1, fd2, fd3, C0, C1, C2, C3, C4, C5, C6, C7, FIELDA, WA)
        CALL whitCardInterp5( WCTR+NCORRS, WSIZE, NWINDOWS, fd1, fd2, fd3, C0, C1, C2, C3, C4, C5, C6, C7, FIELDB, WB)

        ! Cross Correlate Window A with Window B
        CALL crossCorr(WCTR, WSIZE, NCORRS, CC_SMOOTH_FLAG, IWMAXDISP, WEIGHT, DescHandle, DescHandleInv, WA, WB, UR, UC, UP, SNR)

     ENDDO
     !$OMP END PARALLEL DO


  ELSEIF (FETCHTYPE.EQ.4) THEN


     !$OMP PARALLEL DO DEFAULT(NONE) SHARED(NCORRS, NWINDOWS, WSIZE, CC_SMOOTH_FLAG, IWMAXDISP, WEIGHT, fd1, fd2, fd3, DescHandle, DescHandleInv, C0, C1, C2, C3, C4, C5, C6, C7, FIELDA, FIELDB, UR, UC, UP, SNR, ERRCODE) PRIVATE(WA,WB, THREADERRCODE) SCHEDULE(STATIC,50)
     DO WCTR = 1,NCORRS

        ! Retrieve Windows A and B using the 7^3 Whittaker Cardinal Function
        CALL whitCardInterp7( WCTR,        WSIZE, NWINDOWS, fd1, fd2, fd3, C0, C1, C2, C3, C4, C5, C6, C7, FIELDA, WA)
        CALL whitCardInterp7( WCTR+NCORRS, WSIZE, NWINDOWS, fd1, fd2, fd3, C0, C1, C2, C3, C4, C5, C6, C7, FIELDB, WB)

        ! Cross Correlate Window A with Window B
        CALL crossCorr(WCTR, WSIZE, NCORRS, CC_SMOOTH_FLAG, IWMAXDISP, WEIGHT, DescHandle, DescHandleInv, WA, WB, UR, UC, UP, SNR)

     ENDDO
     !$OMP END PARALLEL DO


  ENDIF

  !     Free the MKL FFT descriptors
  stat = DftiFreeDescriptor(DescHandle)
  stat = DftiFreeDescriptor(DescHandleInv)

  !     Free the Intel MKL buffers to prevent memory leak from MEX file
  CALL mkl_free_buffers


END SUBROUTINE vodim3d




