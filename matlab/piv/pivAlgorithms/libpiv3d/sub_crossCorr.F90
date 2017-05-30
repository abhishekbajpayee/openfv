!_______________________________________________________________________
!
!	SUBROUTINE setupCrossCorr
!
!	Sets up the FFTs ready for cross correlation to take place.
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs are as defined in vodim3d.F90
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
! Revision History:     04 July 2011        Created (split from 
!                                           subroutine CrossCorr)
!                       17 September 2011   Updated interface for
!                                           libpiv3d


!_______________________________________________________________________
!
!                   DECLARATIONS AND INCLUSIONS
!_______________________________________________________________________

!     SUBROUTINE setupCrossCorr( WSIZE, DescHandle, DescHandleInv)


!	  Using Intel Fortran Compiler FFT libraries
!      USE MKL_DFTI

!     Safer to have no implicit variables
!      IMPLICIT NONE

!     Include the Intel Math Kernel Library examples file, which
!     contains parameters and interfaces useful for FFT manipulation
!	  INCLUDE 'mkl_dfti_examples.fi'

!	  Input
!      INTEGER*4, INTENT(in) :: WSIZE

!     Output      
!      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandle    ! IDFT plan for this window size
!      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleInv ! IDFT plan for this window size


!     Admin variables for FFTs
!      INTEGER*4 :: stat
!      REAL*4    :: fftScale
!      INTEGER*4 :: lengths(3)
!      INTEGER*4 :: strides_in(4)
!      INTEGER*4 :: n, m, p



!_______________________________________________________________________
!
!                   SETUP FFTS
!_______________________________________________________________________

!	  Set transform parameters (always cubes)
!      m = WSIZE
!      n = WSIZE
!      p = WSIZE

!     Setup lengths and strides for correlation FFTs
!      lengths(1) = m
!      lengths(2) = n
!      lengths(3) = p
!      strides_in(1) = 0
!     strides_in(2) = 1
!      strides_in(3) = m
!      strides_in(4) = m*n

!     Create DFTI descriptors for forward and inverse window transforms
!     stat          = DftiCreateDescriptor( DescHandle,             &
!                                           DFTI_SINGLE,            &
!                                            DFTI_COMPLEX, 3, lengths)
!      IF (HandleFFTError(stat).EQ.1) THEN
!        CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Error in creation of Forward Window FFT descriptor')
!      ENDIF

!      stat          = DftiCreateDescriptor( DescHandleInv,          &
!                                            DFTI_SINGLE,            &
!                                            DFTI_COMPLEX, 3, lengths)
!      IF (HandleFFTError(stat).EQ.1) THEN
!        CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Error in creation of Inverse Window FFT descriptor')
!      ENDIF

!     Set the stride pattern for DFTIs
!      stat          = DftiSetValue( DescHandle,                     &
!                                    DFTI_INPUT_STRIDES, strides_in)  
!      IF (HandleFFTError(stat).EQ.1) THEN
!        CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Cannot set stride pattern for Forward Window FFT')
!      ENDIF    

!     Commit Dfti descriptors
!      stat          = DftiCommitDescriptor( DescHandle )
!      IF (HandleFFTError(stat).EQ.1) THEN
!        CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Cannot commit Forward Window FFT descriptor')
!      ENDIF    

!     Set fft scale for inverse transforms
!      fftScale      = 1.0/real(m*n*p, kind=4)
!      stat          = DftiSetValue(DescHandleInv,                    &
!                                    DFTI_BACKWARD_SCALE, fftScale)
!      IF (HandleFFTError(stat).EQ.1) THEN
!        CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Cannot set backward scale value for Inverse Window FFT')
!      ENDIF

!     Commit Dfti descriptor for backwards transform
!      stat          = DftiCommitDescriptor( DescHandleInv )
!      IF (HandleFFTError(stat).EQ.1) THEN
!        CALL mexErrMsgTxt('fMexPIV:SetupCrossCorr: Cannot commit Inverse Window FFT descriptor')
!      ENDIF

!	  End of subroutine
!      END SUBROUTINE setupCrossCorr









!_______________________________________________________________________
!
!	  				SUBROUTINE crossCorr
!
!	Performs a straightforward cross-correlation on two 3D scalar 
!   fields. Windowing is done by other subfunctions - this routine 
!   simply cross correlates WA with WB, where both those
!   variables are 3D arrays shared with this subfuncion via a module 
!   interfacesecond order
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs and outputs are as defined in vodim3d.F90
!
!		WA   	    [WSIZE x WSIZE x WSIZE] real*4 array
!								Intensity distribution in a window of
!                               the FIELDA intensity array. The window
!                               corners are defined by the locations in 
!                               row WCTR of the CX arrays.
!
!		WB   	    [WSIZE x WSIZE x WSIZE] real*4 array
!								Intensity distribution in a window of
!                               the FIELDB intensity array. The window
!                               corners are defined by the locations in 
!                               row WCTR of the CX arrays.
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
!                       05 July 2011        Added window displacement
!                                           limit code and fixed various
!                                           bugs
!                       17 September 2011   Updated interface for
!                                           libpiv3d


!_______________________________________________________________________
!
!                   DECLARATIONS AND INCLUSIONS
!_______________________________________________________________________

SUBROUTINE crossCorr(WCTR, WSIZE, NCORRS, CC_SMOOTH_FLAG, &
     IWMAXDISP, WEIGHT, DescHandle, &
     DescHandleInv, WA, WB, UR, UC, UP, SNR)


  !	  Using Intel Fortran Compiler FFT libraries
  USE MKL_DFTI

  !     Safer to have no implicit variables
  IMPLICIT NONE

  !     Include the Intel Math Kernel Library examples file, which
  !     contains parameters and interfaces useful for FFT manipulation
  INCLUDE 'mkl_dfti_examples.fi'

  !	  Inputs (explicit shape dummy arrays)
  INTEGER*4, INTENT(in) :: WCTR, WSIZE, CC_SMOOTH_FLAG, NCORRS
  REAL*4,    INTENT(in) :: IWMAXDISP
  REAL*4, DIMENSION(WSIZE, WSIZE, WSIZE), INTENT(in):: WEIGHT
  TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandle    ! DFT plan for this window size
  TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleInv ! IDFT plan for this window size
  REAL*4, DIMENSION(WSIZE,WSIZE,WSIZE):: WA, WB ! no intention as we use them as internal arrays too

  !     Outputs
  REAL*4, DIMENSION(NCORRS,2) :: UR, UC, UP, SNR ! no intention as elements of these are affected by other threads

  !     Admin variables for FFTs
  INTEGER*4 :: stat
  INTEGER*4 :: n, m, p

  !     Admin variables for setting up cross correlation array of 
  !     interrogation volumes
  INTEGER*4 :: aCtr
  INTEGER*4, DIMENSION(2) :: IWlims

  !	  Intermediate variables used for peak location
  INTEGER*4               :: peakCtr
  REAL*4                  :: peak, peak1, peak2, u_r, u_c, u_p
  INTEGER*4, DIMENSION(3) :: pl,   pl1,   pl2
  REAL*4,    DIMENSION(3) :: plr,  pl1r,  pl2r, denom
  REAL*4,    DIMENSION(3) :: rArr, cArr, pArr

  !     Complex and real type arrays for storing intermediate corr vol
  !        These are automatic arrays: explicit shape.
  COMPLEX,   DIMENSION(WSIZE*WSIZE*WSIZE) :: XA, XB
  REAL*4,    DIMENSION(WSIZE,WSIZE,WSIZE) :: X_out


  !_______________________________________________________________________
  !
  !                   PERFORM TRANSFORMS
  !_______________________________________________________________________

  !	  Set transform parameters (always cubes)
  m = WSIZE
  n = WSIZE
  p = WSIZE

  !	  FFT functions need to read data in the 3D arrays as if it were 
  !     a vector. The EQUIVALENCE statement used in the examples to 
  !     achieve this without a memory copy is now deprecated by IFC as it 
  !     is incompatible with module arrangements and variable size arrays,
  !     so here we reshape the arrays instead of assigning equivalence:
  XA = CMPLX(RESHAPE(WA,   SHAPE(XA)))
  XB = CMPLX(RESHAPE(WB,   SHAPE(XB)))

  !     Compute forward transform of window A
  !      stat = DftiComputeForward(DescHandle,XA, XAcmplx)
  stat = DftiComputeForward(DescHandle,XA)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt("fMexPIV:CrossCorr: Cannot compute Forward Window A FFT")
  !      ENDIF

  !     Compute forward transform of window B
  !      stat = DftiComputeForward(DescHandle,XB, XBcmplx)
  stat = DftiComputeForward(DescHandle,XB)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt("fMexPIV:CrossCorr: Cannot compute Forward Window B FFT")
  !      ENDIF

  !     Multiply the first fft with the complex conjugate of the second
  !      XAcmplx = XAcmplx * conjg(XBcmplx)
  XA = XA * conjg(XB)

  !     Smooth the correlation plane
  !      IF (CC_SMOOTH_FLAG.EQ.1) THEN
  !                           
  !        CALL thc_padreplicate(IW,u_pad3D_1,X_IN_3D)
  !        stat_smooth_1 = DftiComputeForward(DescHandleSmooth, u_pad2D_1)
  !        DO smii = 1,ms*ns*ks
  !          u_pad2D_1(smii) = u_pad2D_1(smii)*kernel_pad2D_1(smii)
  !        ENDDO
  !        stat_smooth_2 = DftiComputeBackward(       &
  !                                          Desc_Handle_smoo2, u_pad2D_1)
  !        X_IN_3D = u_pad3D_1(2:ms-1,2:ns-1,2:ks-1);

  !     Compute backwards transformation
  !      stat = DftiComputeBackward(DescHandleInv, XAcmplx, XA)
  stat = DftiComputeBackward(DescHandleInv, XA)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt('fMexPIV:CrossCorr: Cannot compute Inverse Window FFT')
  !      ENDIF

  !     Reshape and cast back to 3D real array
  X_out = RESHAPE(real(XA, kind=4), SHAPE(X_out))

  !     Perform FFTSHIFT on real component of the correlation volume,
  !     storing in the shared WA array. This is the same as MATLAB's
  !     fftShift() command.
  WA( 1:m/2,       1:n/2,       1:p/2)         =           &
       X_out( ((m/2)+1):m, ((n/2)+1):n, ((p/2)+1):p)
  WA( ((m/2)+1):m, 1:n/2,       1:p/2)         =           &
       X_out( 1:m/2,       ((n/2)+1):n, ((p/2)+1):p)
  WA( 1:m/2,       ((n/2)+1):n, 1:p/2)         =           &
       X_out( ((m/2)+1):m, 1:n/2,       ((p/2)+1):p)
  WA( ((m/2)+1):m, ((n/2)+1):n, 1:(p/2))       =           &
       X_out( 1:(m/2),     1:(n/2),     ((p/2)+1):p)
  WA( ((m/2)+1):m, ((n/2)+1):n, ((p/2)+1):p)   =           &
       X_out( 1:(m/2),     1:(n/2),     1:(p/2))
  WA( 1:(m/2),     ((n/2)+1):n, ((p/2)+1):p)   =           &
       X_out( ((m/2)+1):m, 1:(n/2),     1:(p/2))
  WA( ((m/2)+1):m, 1:(n/2),     ((p/2)+1):p)   =           &
       X_out( 1:(m/2),     ((n/2)+1):n, 1:(p/2))
  WA( 1:(m/2),     1:(n/2),     ((p/2)+1):p)   =		   &
       X_out( ((m/2)+1):m, ((n/2)+1):n, 1:(p/2))


  !_______________________________________________________________________
  !
  !                   LOCATE PEAKS
  !_______________________________________________________________________


  !     Elementary divide by the weighting matrix
  WA = WA / WEIGHT

  !     Interrogation window peak location limit (avoids finding peak in 
  !     erroneous corner data caused by ringing of FFTs)
  IWlims(1)=(WSIZE/2)-nint(real(WSIZE)/2.*IWMAXDISP/100.)
  IWlims(2)=(WSIZE/2)+nint(real(WSIZE)/2.*IWMAXDISP/100.)

  !     Find primary peak location
  peak1 = MAXVAL( WA(  IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2)) )
  pl1   = MAXLOC( WA(  IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2)) )

  !     Correct for lower limit
  pl1(1) = pl1(1) + IWlims(1) - 1
  pl1(2) = pl1(2) + IWlims(1) - 1
  pl1(3) = pl1(3) + IWlims(1) - 1
  pl1r   = real(pl1)

  !     Set primary peak to zero, then find secondary peak
  WA( pl1(1), pl1(2), pl1(3) ) = 0	  
  peak2 = MAXVAL( WA(  IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2)) )
  pl2   = MAXLOC(WA(   IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2)) )

  !     Correct for lower limit
  pl2(1) = pl2(1) + IWlims(1) - 1
  pl2(2) = pl2(2) + IWlims(1) - 1
  pl2(3) = pl2(3) + IWlims(1) - 1
  pl2r   = real(pl2)

  !     Restore primary peak location
  WA( pl1(1), pl1(2), pl1(3) ) = peak1

  !     Find the velocities for each peak to sub voxel accuracy
  DO peakCtr = 1,2
     IF (peakCtr.EQ.1) THEN
        pl   = pl1
        plr  = pl1r
        peak = peak1
     ELSEIF (peakCtr.EQ.2) THEN
        pl   = pl2
        plr  = pl2r
        peak = peak2
     ENDIF

     !       Store points either side of peak in 3 dimensions	  
     rArr(1:3) = [ WA(pl(1)-1, pl(2),   pl(3)  ),   &
          WA(pl(1),   pl(2),   pl(3)  ),   &
          WA(pl(1)+1, pl(2),   pl(3)  ) ]
     cArr(1:3) = [ WA(pl(1),   pl(2)-1, pl(3)  ),   &
          WA(pl(1),   pl(2),   pl(3)  ),   &
          WA(pl(1),   pl(2)+1, pl(3)  ) ]
     pArr(1:3) = [ WA(pl(1),   pl(2),   pl(3)-1),   &
          WA(pl(1),   pl(2),   pl(3)  ),   &
          WA(pl(1),   pl(2),   pl(3)+1) ]

     !       If one of these values is <= zero, use the
     !       others and find through linear interpolation
     DO aCtr = 1,3
        IF (rArr(aCtr).LE.0) THEN
           rArr(aCtr) = 0.0000001
        ENDIF
        IF (cArr(aCtr).LE.0) THEN
           cArr(aCtr) = 0.0000001
        ENDIF
        IF (pArr(aCtr).LE.0) THEN
           pArr(aCtr) = 0.0000001
        ENDIF
     ENDDO

     !		Use a Gaussian fit. Calculate denominators first, setting 
     !       velocity to NaN where the denominator is zero
     denom(1) = 2.*log(rArr(1)) - 4.*log(rArr(2)) + 2.*log(rArr(3))
     denom(2) = 2.*log(cArr(1)) - 4.*log(cArr(2)) + 2.*log(cArr(3))
     denom(3) = 2.*log(pArr(1)) - 4.*log(pArr(2)) + 2.*log(pArr(3))										

     IF (ANY(denom.EQ.(0.0))) THEN
        ! Set velocities to NaN
        u_r = sqrt(-1.0)
        u_c = sqrt(-1.0)
        u_p = sqrt(-1.0)
     ELSE
        plr(1) = plr(1) + (log(rArr(1))-log(rArr(3)))/denom(1)  
        plr(2) = plr(2) + (log(cArr(1))-log(cArr(3)))/denom(2)
        plr(3) = plr(3) + (log(pArr(1))-log(pArr(3)))/denom(3)                
        u_r = (real(m)/2.) + 1. - plr(1)
        u_c = (real(n)/2.) + 1. - plr(2)
        u_p = (real(p)/2.) + 1. - plr(3)
     ENDIF

     !       Store results to the correct location in the thread-shared arrays for output
     UR(WCTR,peakCtr) = u_r
     UC(WCTR,peakCtr) = u_c
     UP(WCTR,peakCtr) = u_p
     SNR(WCTR,peakCtr) = peak * real(WSIZE*WSIZE*WSIZE, kind = 4)  &
          / SUM(WA)


  ENDDO



  !	  End of subroutine
END SUBROUTINE crossCorr



!_______________________________________________________________________
!
!	  				SUBROUTINE crossCorrCTE
!
!	Performs a CTE cross-correlation on four 3D scalar 
!   fields. Windowing is done by other subfunctions - this routine 
!   simply performs cross correlation of windows, whiche are 3D arrays 
!   shared with this subfuncion via a module 
!   interface
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs and outputs are as defined in cte3d.F90
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
! Revision History:     11 July 2011        Created from first order
!                                           equivalent CrossCorr
!                       17 September 2011   Updated interface for
!                                           libpiv3d


!_______________________________________________________________________
!
!                   DECLARATIONS AND INCLUSIONS
!_______________________________________________________________________

SUBROUTINE crossCorrCTE(WCTR, WSIZE, NCORRS, CC_SMOOTH_FLAG, &
     IWMAXDISP, WEIGHT, DescHandle, &
     DescHandleInv, WA, WB, WC, WD, &
     UR, UC, UP, SNR)


  !	  Using Intel Fortran Compiler FFT libraries
  USE MKL_DFTI

  !     Safer to have no implicit variables
  IMPLICIT NONE

  !     Include the Intel Math Kernel Library examples file, which
  !     contains parameters and interfaces useful for FFT manipulation
  ! INCLUDE 'mkl_dfti_examples.fi'

  !	  Declare inputs as explicit-shape dummy arrays
  INTEGER*4, INTENT(in) :: WCTR, WSIZE, CC_SMOOTH_FLAG, NCORRS
  REAL*4,    INTENT(in) :: IWMAXDISP
  REAL*4, DIMENSION(WSIZE, WSIZE, WSIZE), INTENT(in):: WEIGHT
  TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandle    ! DFT plan for this window size
  TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleInv ! IDFT plan for this window size
  REAL*4, DIMENSION(WSIZE,WSIZE,WSIZE):: WA, WB, WC, WD ! no intention as we use them as internal arrays too

  !     Outputs
  REAL*4, DIMENSION(NCORRS,2) :: UR, UC, UP, SNR ! no intention as elements of these are affected by other threads


  !     Admin variables for FFTs
  INTEGER*4 :: stat
  INTEGER*4 :: n, m, p

  !     Admin variables for setting up cross correlation array of 
  !     interrogation volumes
  INTEGER*4 :: aCtr
  INTEGER*4, DIMENSION(2) :: IWlims

  !	  Intermediate variables used for peak location
  INTEGER*4               :: peakCtr
  REAL*4                  :: peak, peak1, peak2, u_r, u_c, u_p
  INTEGER*4, DIMENSION(3) :: pl,   pl1,   pl2
  REAL*4,    DIMENSION(3) :: plr,  pl1r,  pl2r, denom
  REAL*4,    DIMENSION(3) :: rArr, cArr, pArr

  !     Complex and real type arrays for storing intermediate corr vol
  !        These are automatic arrays: explicit shape, nonconstant bounds.
  COMPLEX,   DIMENSION(WSIZE*WSIZE*WSIZE) :: XA, XB, XC, XD
  REAL*4,    DIMENSION(WSIZE,WSIZE,WSIZE) :: X_out

  character*120 line

  !_______________________________________________________________________
  !
  !                   PERFORM TRANSFORMS
  !_______________________________________________________________________

  !	  Set transform parameters (always cubes)
  m = WSIZE
  n = WSIZE
  p = WSIZE

  !	  FFT functions need to read data in the 3D arrays as if it were 
  !     a vector. The EQUIVALENCE statement used in the examples to 
  !     achieve this without a memory copy is now deprecated by IFC as it 
  !     is incompatible with module arrangements and variable size arrays,
  !     so here we reshape the arrays instead of assigning equivalence:
  XA = CMPLX(RESHAPE(WA,   SHAPE(XA)))
  XB = CMPLX(RESHAPE(WB,   SHAPE(XB)))
  XC = CMPLX(RESHAPE(WC,   SHAPE(XC)))
  XD = CMPLX(RESHAPE(WD,   SHAPE(XD)))

  !     Compute forward transform of window A
  stat = DftiComputeForward(DescHandle,XA)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt("fMexPIV:CrossCorr: Cannot compute Forward Window A FFT")
  !      ENDIF

  !     Compute forward transform of window B
  stat = DftiComputeForward(DescHandle,XB)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt("fMexPIV:CrossCorr: Cannot compute Forward Window B FFT")
  !      ENDIF

  !     Compute forward transform of window C
  stat = DftiComputeForward(DescHandle,XC)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt("fMexPIV:CrossCorr: Cannot compute Forward Window C FFT")
  !      ENDIF

  !     Compute forward transform of window D
  stat = DftiComputeForward(DescHandle,XD)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt("fMexPIV:CrossCorr: Cannot compute Forward Window D FFT")
  !      ENDIF

  !     Multiply the first fft with the complex conjugate of the second
  XA = XA * conjg(XB)

  !     Multiply the second fft with the complex conjugate of the third
  XB = XB * conjg(XC)

  !     Multiply the third fft with the complex conjugate of the fourth
  XC = XC * conjg(XD)


  !     Compute backwards transformation
  stat = DftiComputeBackward(DescHandleInv, XA)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt('fMexPIV:CrossCorr: Cannot compute Inverse Window FFT XAB')
  !      ENDIF                      
  !     Compute backwards transformation
  stat = DftiComputeBackward(DescHandleInv, XB)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt('fMexPIV:CrossCorr: Cannot compute Inverse Window FFT XBC')
  !      ENDIF                      
  !     Compute backwards transformation
  stat = DftiComputeBackward(DescHandleInv, XC)
  !      IF (HandleFFTError(stat).EQ.1) THEN
  !        CALL mexErrMsgTxt('fMexPIV:CrossCorr: Cannot compute Inverse Window FFT XCD')
  !      ENDIF


  !_______________________________________________________________________
  !
  !                   CTE CROSS CORRELATION PEAK FINDER
  !_______________________________________________________________________


  !     Multiply to enhance signal to noise ratio
  XA = XA * XB * XC ! xa + 2xb +xc

  !     Reshape and cast back to 3D real array
  X_out = RESHAPE(real(XA, kind=4), SHAPE(X_out))

  !     Perform FFTSHIFT on real component of the correlation volume,
  !     storing in the shared WA array. This is the same as MATLAB's
  !     fftShift() command.
  WA( 1:m/2,       1:n/2,       1:p/2)         =           &
       X_out( ((m/2)+1):m, ((n/2)+1):n, ((p/2)+1):p)
  WA( ((m/2)+1):m, 1:n/2,       1:p/2)         =           &
       X_out( 1:m/2,       ((n/2)+1):n, ((p/2)+1):p)
  WA( 1:m/2,       ((n/2)+1):n, 1:p/2)         =           &
       X_out( ((m/2)+1):m, 1:n/2,       ((p/2)+1):p)
  WA( ((m/2)+1):m, ((n/2)+1):n, 1:(p/2))       =           &
       X_out( 1:(m/2),     1:(n/2),     ((p/2)+1):p)
  WA( ((m/2)+1):m, ((n/2)+1):n, ((p/2)+1):p)   =           &
       X_out( 1:(m/2),     1:(n/2),     1:(p/2))
  WA( 1:(m/2),     ((n/2)+1):n, ((p/2)+1):p)   =           &
       X_out( ((m/2)+1):m, 1:(n/2),     1:(p/2))
  WA( ((m/2)+1):m, 1:(n/2),     ((p/2)+1):p)   =           &
       X_out( 1:(m/2),     ((n/2)+1):n, 1:(p/2))
  WA( 1:(m/2),     1:(n/2),     ((p/2)+1):p)   =		   &
       X_out( ((m/2)+1):m, ((n/2)+1):n, 1:(p/2))



  !     Elementary divide by the weighting matrix
  WA = WA / WEIGHT

  !     Interrogation window peak location limit (avoids finding peak in 
  !     erroneous corner data caused by ringing of FFTs)
  IWlims(1)=(WSIZE/2)-nint(real(WSIZE)/2.*IWMAXDISP/100.)
  IWlims(2)=(WSIZE/2)+nint(real(WSIZE)/2.*IWMAXDISP/100.)

  !     Find primary peak location
  peak1 = MAXVAL( WA(  IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2)) )
  pl1   = MAXLOC( WA(  IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2)) )

  !     Correct for lower limit
  pl1(1) = pl1(1) + IWlims(1) - 1
  pl1(2) = pl1(2) + IWlims(1) - 1
  pl1(3) = pl1(3) + IWlims(1) - 1
  pl1r   = real(pl1)

  !     Set primary peak to zero, then find secondary peak
  WA( pl1(1), pl1(2), pl1(3) ) = 0	  
  peak2 = MAXVAL( WA(  IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2)) )
  pl2   = MAXLOC(WA(   IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2),            &
       IWlims(1):IWlims(2)) )

  !     Correct for lower limit
  pl2(1) = pl2(1) + IWlims(1) - 1
  pl2(2) = pl2(2) + IWlims(1) - 1
  pl2(3) = pl2(3) + IWlims(1) - 1
  pl2r   = real(pl2)

  !     Restore primary peak location
  WA( pl1(1), pl1(2), pl1(3) ) = peak1

  !     Find the velocities for each peak to sub voxel accuracy
  DO peakCtr = 1,2
     IF (peakCtr.EQ.1) THEN
        pl   = pl1
        plr  = pl1r
        peak = peak1
     ELSEIF (peakCtr.EQ.2) THEN
        pl   = pl2
        plr  = pl2r
        peak = peak2
     ENDIF

     !       Store points either side of peak in 3 dimensions	  
     rArr(1:3) = [ WA(pl(1)-1, pl(2),   pl(3)  ),   &
          WA(pl(1),   pl(2),   pl(3)  ),   &
          WA(pl(1)+1, pl(2),   pl(3)  ) ]
     cArr(1:3) = [ WA(pl(1),   pl(2)-1, pl(3)  ),   &
          WA(pl(1),   pl(2),   pl(3)  ),   &
          WA(pl(1),   pl(2)+1, pl(3)  ) ]
     pArr(1:3) = [ WA(pl(1),   pl(2),   pl(3)-1),   &
          WA(pl(1),   pl(2),   pl(3)  ),   &
          WA(pl(1),   pl(2),   pl(3)+1) ]

     !       If one of these values is <= zero, use the
     !       others and find through linear interpolation
     DO aCtr = 1,3
        IF (rArr(aCtr).LE.0) THEN
           rArr(aCtr) = 0.0000001
        ENDIF
        IF (cArr(aCtr).LE.0) THEN
           cArr(aCtr) = 0.0000001
        ENDIF
        IF (pArr(aCtr).LE.0) THEN
           pArr(aCtr) = 0.0000001
        ENDIF
     ENDDO

     !		Use a Gaussian fit. Calculate denominators first, setting 
     !       velocity to NaN where the denominator is zero
     denom(1) = 2.*log(rArr(1)) - 4.*log(rArr(2)) + 2.*log(rArr(3))
     denom(2) = 2.*log(cArr(1)) - 4.*log(cArr(2)) + 2.*log(cArr(3))
     denom(3) = 2.*log(pArr(1)) - 4.*log(pArr(2)) + 2.*log(pArr(3))										

     IF (ANY(denom.EQ.(0.0))) THEN
        ! Set velocities to NaN
        u_r = sqrt(-1.0)
        u_c = sqrt(-1.0)
        u_p = sqrt(-1.0)
     ELSE
        plr(1) = plr(1) + (log(rArr(1))-log(rArr(3)))/denom(1)  
        plr(2) = plr(2) + (log(cArr(1))-log(cArr(3)))/denom(2)
        plr(3) = plr(3) + (log(pArr(1))-log(pArr(3)))/denom(3)                
        u_r = (real(m)/2.) + 1. - plr(1)
        u_c = (real(n)/2.) + 1. - plr(2)
        u_p = (real(p)/2.) + 1. - plr(3)
     ENDIF

     !       Store results to the shared arrays for output
     UR(WCTR,peakCtr) = u_r
     UC(WCTR,peakCtr) = u_c
     UP(WCTR,peakCtr) = u_p
     SNR(WCTR,peakCtr) = peak * real(WSIZE*WSIZE*WSIZE, kind = 4) &
          / SUM(WA)


  ENDDO



  !_______________________________________________________________________
  !
  !               VODIM (NON-CTE) CROSS CORRELATION PEAK FINDER
  !_______________________________________________________________________

  !     It's hardly any more work to return standard PIV results for 
  !     comparison so we might as well compute them here. We use XB which
  !     contains the cross correlation between B and C, i.e. represents 
  !     the same point in time as the CTE cross correlation

  !     Reshape and cast back to 3D real array
  !      X_out = RESHAPE(real(XB, kind=4), SHAPE(X_out))

  !     Perform FFTSHIFT on real component of the correlation volume,
  !     storing in the shared WA array. This is the same as MATLAB's
  !     fftShift() command.
  !      WA( 1:m/2,       1:n/2,       1:p/2)         =           &
  !		X_out( ((m/2)+1):m, ((n/2)+1):n, ((p/2)+1):p)
  !      WA( ((m/2)+1):m, 1:n/2,       1:p/2)         =           &
  !		X_out( 1:m/2,       ((n/2)+1):n, ((p/2)+1):p)
  !      WA( 1:m/2,       ((n/2)+1):n, 1:p/2)         =           &
  !		X_out( ((m/2)+1):m, 1:n/2,       ((p/2)+1):p)
  !      WA( ((m/2)+1):m, ((n/2)+1):n, 1:(p/2))       =           &
  !		X_out( 1:(m/2),     1:(n/2),     ((p/2)+1):p)
  !      WA( ((m/2)+1):m, ((n/2)+1):n, ((p/2)+1):p)   =           &
  !		X_out( 1:(m/2),     1:(n/2),     1:(p/2))
  !      WA( 1:(m/2),     ((n/2)+1):n, ((p/2)+1):p)   =           &
  !		X_out( ((m/2)+1):m, 1:(n/2),     1:(p/2))
  !      WA( ((m/2)+1):m, 1:(n/2),     ((p/2)+1):p)   =           &
  !		X_out( 1:(m/2),     ((n/2)+1):n, 1:(p/2))
  !      WA( 1:(m/2),     1:(n/2),     ((p/2)+1):p)   =		   &
  !		X_out( ((m/2)+1):m, ((n/2)+1):n, 1:(p/2))



  !     Elementary divide by the weighting matrix
  !      WA = WA / WEIGHT

  !     Interrogation window peak location limit computed above, 
  !     doesn't change

  !     Find primary peak location
  !      peak1 = MAXVAL( WA(  IWlims(1):IWlims(2),            &
  !                                IWlims(1):IWlims(2),            &
  !                                IWlims(1):IWlims(2)) )
  !      pl1   = MAXLOC( WA(  IWlims(1):IWlims(2),            &
  !                                IWlims(1):IWlims(2),            &
  !                                IWlims(1):IWlims(2)) )

  !     Correct for lower limit
  !      pl1(1) = pl1(1) + IWlims(1) - 1
  !      pl1(2) = pl1(2) + IWlims(1) - 1
  !      pl1(3) = pl1(3) + IWlims(1) - 1
  !      pl1r   = real(pl1)

  !!     Set primary peak to zero, then find secondary peak
  !      WA( pl1(1), pl1(2), pl1(3) ) = 0	  
  ! !     peak2 = MAXVAL( WA(  IWlims(1):IWlims(2),            &
  !                                IWlims(1):IWlims(2),            &
  !!                                IWlims(1):IWlims(2)) )
  !      pl2   = MAXLOC(WA(   IWlims(1):IWlims(2),            &
  !                                IWlims(1):IWlims(2),            &
  !                                IWlims(1):IWlims(2)) )

  !     Correct for lower limit
  !      pl2(1) = pl2(1) + IWlims(1) - 1
  !      pl2(2) = pl2(2) + IWlims(1) - 1
  !      pl2(3) = pl2(3) + IWlims(1) - 1
  !      pl2r   = real(pl2)

  !     Restore primary peak location
  !      WA( pl1(1), pl1(2), pl1(3) ) = peak1

  !     Find the velocities for each peak to sub voxel accuracy
  !      DO peakCtr = 1,2
  !        IF (peakCtr.EQ.1) THEN
  !            pl   = pl1
  !            plr  = pl1r
  !            peak = peak1
  !		ELSEIF (peakCtr.EQ.2) THEN
  !            pl   = pl2
  !            plr  = pl2r
  !            peak = peak2
  !        ENDIF

  !       Store points either side of peak in 3 dimensions	  
  !        rArr(1:3) = [ WA(pl(1)-1, pl(2),   pl(3)  ),   &
  !                      WA(pl(1),   pl(2),   pl(3)  ),   &
  !					  WA(pl(1)+1, pl(2),   pl(3)  ) ]
  !		cArr(1:3) = [ WA(pl(1),   pl(2)-1, pl(3)  ),   &
  !					  WA(pl(1),   pl(2),   pl(3)  ),   &
  !					  WA(pl(1),   pl(2)+1, pl(3)  ) ]
  !        pArr(1:3) = [ WA(pl(1),   pl(2),   pl(3)-1),   &
  !					  WA(pl(1),   pl(2),   pl(3)  ),   &
  !					  WA(pl(1),   pl(2),   pl(3)+1) ]

  !       If one of these values is <= zero, use the
  !       others and find through linear interpolation
  !        DO aCtr = 1,3
  !            IF (rArr(aCtr).LE.0) THEN
  !			    rArr(aCtr) = 0.0000001
  !!		    ENDIF					
  !		    IF (cArr(aCtr).LE.0) THEN
  !			    cArr(aCtr) = 0.0000001
  !		    ENDIF
  !			IF (pArr(aCtr).LE.0) THEN
  !			    pArr(aCtr) = 0.0000001
  !		    ENDIF								          
  !        ENDDO

  !		Use a Gaussian fit. Calculate denominators first, setting 
  !       velocity to NaN where the denominator is zero
  !        denom(1) = 2.*log(rArr(1)) - 4.*log(rArr(2)) + 2.*log(rArr(3))
  !        denom(2) = 2.*log(cArr(1)) - 4.*log(cArr(2)) + 2.*log(cArr(3))
  !        denom(3) = 2.*log(pArr(1)) - 4.*log(pArr(2)) + 2.*log(pArr(3))										

  !        IF (ANY(denom.EQ.(0.0))) THEN
  !            ! Set velocities to NaN
  !            u_r = sqrt(-1.0)
  !            u_c = sqrt(-1.0)
  !		    u_p = sqrt(-1.0)
  !        ELSE
  !            plr(1) = plr(1) + (log(rArr(1))-log(rArr(3)))/denom(1)  
  !            plr(2) = plr(2) + (log(cArr(1))-log(cArr(3)))/denom(2)
  !            plr(3) = plr(3) + (log(pArr(1))-log(pArr(3)))/denom(3)                
  !            u_r = (real(m)/2.) + 1. - plr(1)
  !		    u_c = (real(n)/2.) + 1. - plr(2)
  !			u_p = (real(p)/2.) + 1. - plr(3)
  !        ENDIF

  !       Store VODIM results to the shared arrays for output
  !        VUR(WCTR,peakCtr) = u_r
  !        VUC(WCTR,peakCtr) = u_c
  !        VUP(WCTR,peakCtr) = u_p
  !        VSNR(WCTR,peakCtr) = peak * real(WSIZE*WSIZE*WSIZE, kind = 4) &
  !                                 / SUM(WA)


  !      ENDDO






  !	  End of subroutine
END SUBROUTINE crossCorrCTE
