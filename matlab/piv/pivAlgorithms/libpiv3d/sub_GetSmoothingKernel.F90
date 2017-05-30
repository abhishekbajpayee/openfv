



!_______________________________________________________________________
!
!	  				SUBROUTINE CrossCorr
!
!	Performs a straightforward cross-correlation on two 3D scalar 
!   fields. Windowing is done by other subfunctions - this routine 
!   simply cross correlates WINDOWA with WINDOWB, where both those
!   variables are 3D arrays shared with this subfuncion via a module 
!   interface
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs are as follows:
!
!       CC_SMOOTH_FLAG  [1 x 1] integer*4
!                               A value of 1 causes the correlation 
!                               plane to be smoothed using a gaussian 
!                               kernel filter.
!
!		IW_MAX_D		    [1 x 1]	real*4
!								Maximum displacement of an interrogation
!								window during the cross correlation, 
!                               as a fraction of window size.
!                               For example, a value of 0.5 means that
!                               any displacement over 50% 
!                               of the interrogation window size is 
!                               considered impossible.
!                               This is useful, since edge effects 
!                               caused by the FFT can result in 
!                               erroneously high elements at the corners
!                               of the correlation volume, resulting in 
!                               false vectors
!       
!       WCTR            [1 x 1] integer *4
!                               Row index into the UX, UY, UZ, SNR 
!                               arrays (also shared in the module) for 
!                               the current window.


      SUBROUTINE CrossCorr( WCTR, WSIZE, CC_SMOOTH_FLAG, IW_MAX_D)


!	  Using Intel Fortran Compiler FFT libraries
      USE MKL_DFTI

!     Safer to have no implicit variables
      IMPLICIT NONE

!     Include the Intel Math Kernel Library examples file, which
!     contains parameters and interfaces useful for FFT manipulation
	  INCLUDE 'mkl_dfti_examples.fi'


!	  Declare inputs as explicit-shape dummy arrays
      INTEGER*4, INTENT(in) :: CC_SMOOTH_FLAG, WCTR, WSIZE
	  REAL*4,    INTENT(in) :: IW_MAX_D
      



!     Descriptors (constructor objects) for FFTs
      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandle
      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleInv
      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleSmooth
      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleInvSmooth


!     Admin variables for FFTs
      INTEGER :: statFwd, statInv, statFwdSmooth, statInvSmooth
      REAL    :: fftScale, fftScaleSmooth
      INTEGER :: lengths(3), lengths_s(3)
      INTEGER :: strides_in(4), strides_in_s(4)
      INTEGER :: i, smii


!     Admin variables for setting up cross correlation array of 
!     interrogation volumes
      INTEGER*4 :: ii, jj, kk, i2, j2, k2, c, isw, jsw, ksw, csw, iri, &
				   jri, kri, CTR_I, CTR_J, CTR_K, VOX_CTR, testcount
	  REAL    :: wdisttocen, wdistfromcen
      INTEGER*4 :: IWmid(3), IWlims(2,3), IWdisp(3), IWtr(3),          &
				   ptrlim(3,2), trlim(3,2)
 	  INTEGER*4, DIMENSION(:), ALLOCATABLE :: IWx, IWy, IWz
	  REAL*4    :: OVERLAP(3)
	  REAL*4    :: ir, jr, kr, temp
      REAL, DIMENSION(:,:,:), ALLOCATABLE :: WINDOWA, weight
      INTEGER*4 :: n, m, k, ms, ns, ks

!     Storage of the gaussian smoothing kernel
      REAL, DIMENSION(3,3,3) :: GX
	  
!	  Intermediate variables used for peak location
	  REAL    :: peak, peak1, peak2, peak3
	  INTEGER :: pl(3),  pl1(3),  pl2(3),  peakcyc
	  REAL    :: plr(3), pl1r(3), pl2r(3), denom(3)
	  REAL    :: xarr(3), yarr(3), zarr(3), u, v, w

!     Complex and real type arrays for storing intermediate corr vol
!        These are automatic arrays: explicit shape, nonconstant bounds.
	  COMPLEX, DIMENSION(WSIZE,WSIZE,WSIZE) :: XA_3D, XB_3D 
	  COMPLEX, DIMENSION(WSIZE*WSIZE*WSIZE) :: XA, XB
	  REAL,    DIMENSION(WSIZE,WSIZE,WSIZE) :: X_out

!     Smoothing windows for this pass. 
!        These are automatic arrays: explicit shape, nonconstant bounds.
      COMPLEX, DIMENSION(SW1(1),SW1(2),SW1(3)) :: kernelPad3D
      COMPLEX, DIMENSION(SW1(1)*SW1(2)*SW1(3)) :: kernelPad2D

      COMPLEX, DIMENSION(SW1(1),SW1(2),SW1(3)) :: u_pad3D_1
      COMPLEX, DIMENSION(SW1(1)*SW1(2)*SW1(3)) :: u_pad2D_1

!	  Debugging variables
      character*120 line




!     Get the gaussian smoothing kernel
      CALL GetSmoothingKernel(GX)



!	  Set transform parameters (always cubes)
      m = WSIZE;     ms = SW1(1)
      n = WSIZE;     ns = SW1(2)
      k = WSIZE;     ks = SW1(3)

!     Allocate variables the right size for the current loop
      ALLOCATE(WINDOWA(1:m,1:n,1:k))
      ALLOCATE(weight		(1:m,1:n,1:k))

!     SETUP FFTs 

!     Setup lengths and strides for correlation FFTs
      lengths(1) = m
      lengths(2) = n
      lengths(3) = k
      strides_in(1) = 0
      strides_in(2) = 1
      strides_in(3) = m
      strides_in(4) = m*n

!     Setup lengths and strides for smoothing FFTs
      lengths_s(1) = ms
      lengths_s(2) = ns
      lengths_s(3) = ks
      strides_in_s(1) = 0
      strides_in_s(2) = 1
      strides_in_s(3) = ms
      strides_in_s(4) = ms*ns

!     Create DFTI descriptors for forward and inverse window transforms
      statFwd       = DftiCreateDescriptor( DescHandle,             &
                                            DFTI_SINGLE,            &
                                            DFTI_COMPLEX, 3, lengths)

      statInv       = DftiCreateDescriptor( DescHandleInv,          &
                                            DFTI_SINGLE,            &
                                            DFTI_COMPLEX, 3, lengths)


!     Create DFTI descriptors for forward and inverse smoothing kernel 
!     transforms
      statFwdSmooth = DftiCreateDescriptor( DescHandleSmooth,       &
                                            DFTI_SINGLE,            &
                                            DFTI_COMPLEX, 3, lengths_s)

      statInvSmooth = DftiCreateDescriptor( DescHandleInvSmooth,    &
                                            DFTI_SINGLE,            &
                                            DFTI_COMPLEX, 3, lengths_s)

!     Check for errors in creation of the descriptors
! TODO: Sort this...
!      IF (.NOT. DftiErrorClass(statFwd, DFTI_NO_ERROR)) THEN
!        GOTO 101 
!      ENDIF

!     Set the stride pattern for DFTIs
      statFwd       = DftiSetValue( DescHandle,                     &
                                    DFTI_INPUT_STRIDES, strides_in)      

      statFwdSmooth = DftiSetValue( DescHandleSmooth,               &
                                    DFTI_INPUT_STRIDES, strides_in_s)

!     Commit Dfti descriptors
      statFwd       = DftiCommitDescriptor( DescHandle       )
      statFwdSmooth = DftiCommitDescriptor( DescHandleSmooth )


!     Set fftScale number for inverse transforms
      fftScale       = 1.0/real(m*n*k, KIND=4)
      fftScaleSmooth = 1.0/real(ms*ns*ks, KIND=4)
      
      statInv        = DftiSetValue(DescHandle_2,                   &
                                    DFTI_BACKWARD_SCALE, fftScale)

      statInvSmooth  = DftiSetValue(DescHandleInvSmooth,            &
                                    DFTI_BACKWARD_SCALE, fftScaleSmooth)


!     Commit Dfti descriptor for backwards transform
      statInv       = DftiCommitDescriptor( DescHandleInv )
      statInvSmooth = DftiCommitDescriptor( DescHandleInvSmooth )

!     Compute forward transform of padded convolution kernel
!     (done once and reused for each window)
      kernelPad3D = 0
      kernelPad3D(1:3,1:3,1:3) = GX;
      kernelPad2D = RESHAPE(kernelPad3D,   SHAPE(kernelPad2D))
      statFwdSmooth = DftiComputeForward(   DescHandleSmooth,       &
                                            kernelPad2D)


!		CALCULATE THE WEIGHTING MATRIX
		wdisttocen = (	(((REAL(m)+1)/2)**2)	+					   &
						(((REAL(n)+1)/2)**2)	+					   &
						(((REAL(k)+1)/2)**2)		)**0.5 
		DO ii=1,m
			DO jj=1,n
				DO kk=1,k
					wdistfromcen = ( ((REAL(ii)-((REAL(m)+1)/2))**2) + &
									 ((REAL(jj)-((REAL(n)+1)/2))**2) + &
									 ((REAL(kk)-((REAL(k)+1)/2))**2) ) &
																   **0.5
					weight(ii,jj,kk) = 1 - (wdistfromcen/wdisttocen)
					IF (weight(ii,jj,kk).LE.0) THEN
						weight(ii,jj,kk) = 0.01
					ENDIF
				ENDDO
			ENDDO
		ENDDO



!		write(line,*) 'LOOP_CTR = ',LOOP_CTR
!		CALL mexPrintf(line//achar(13)) 
!		write(line,*) 'IWnums(1) = ',IWnums(1)
!		CALL mexPrintf(line//achar(13)) 
!		write(line,*) 'IWnums(2) = ',IWnums(2)
!		CALL mexPrintf(line//achar(13)) 
!		write(line,*) 'IWnums(3) = ',IWnums(3)
!		CALL mexPrintf(line//achar(13))




!     EXECUTE CROSS-CORRELATION


      XA_3D(:,:,:) = CMPLX(WINDOWA)
	  XB_3D(:,:,:) = CMPLX(WINDOWB)
	  
!	  The equivalence statements have been scratched out
!	  to give more robust programmatic structure. However,
!	  the FFT functions need to read data in the 3D arrays
!	  as if it were a vector. The EQUIVALENCE statement is
!	  replaced with this slight hack:
      XA = RESHAPE(XA_3D,   SHAPE(XA))
      XB = RESHAPE(XB_3D,   SHAPE(XB))

!     Compute forward transform of window A
      statFwd = DftiComputeForward(DescHandle,XA)
	  
!     Compute forward transform of window B
      statFwd = DftiComputeForward(DescHandle,XB)

!     Multiply the first fft with the complex conjugate of the second
      XA = XA * conjg(XB)
      
!     Smooth the correlation plane before the inverse transformation
!      IF (CC_SMOOTH_FLAG.EQ.1) THEN
!      
!        CALL thc_padreplicate(IW,u_pad3D_1,X_IN_3D)
!        statFwdSmooth = DftiComputeForward(DescHandleSmooth, u_pad2D_1)
!        DO smii = 1,ms*ns*ks
!            u_pad2D_1(smii) = u_pad2D_1(smii)*kernel_pad2D_1(smii)
!        ENDDO
!        statInvSmooth = DftiComputeBackward( DescHandleInvSmooth,   &
!                                             u_pad2D_1)
!        ! NB THIS ISN'T RIGHT - relied on the old equivalence statements X_IN_3D = u_pad3D_1(2:ms-1,2:ns-1,2:ks-1);
!      ENDIF
      
!     Compute backwards transformation
      statInv = DftiComputeBackward(DescHandleInv, XA)

!     Reshape back to 3D array
      XA_3D = RESHAPE(XA, SHAPE(XA_3D))

!     Perform FFTSHIFT on real component of the correlation volume,
!     storing in the shared WINDOWA array. This is the same as MATLAB's
!     fftShift() command.
      X_out = REAL(XA_3D)
      WINDOWA( 1:m/2,       1:n/2,       1:k/2)         =           &
		X_out( ((m/2)+1):m, ((n/2)+1):n, ((k/2)+1):k)
      WINDOWA( ((m/2)+1):m, 1:n/2,       1:k/2)         =           &
		X_out( 1:m/2,       ((n/2)+1):n, ((k/2)+1):k)
      WINDOWA( 1:m/2,       ((n/2)+1):n, 1:k/2)         =           &
		X_out( ((m/2)+1):m, 1:n/2,       ((k/2)+1):k)
      WINDOWA( ((m/2)+1):m, ((n/2)+1):n, 1:(k/2))       =           &
		X_out( 1:(m/2),     1:(n/2),     ((k/2)+1):k)
      WINDOWA( ((m/2)+1):m, ((n/2)+1):n, ((k/2)+1):k)   =           &
		X_out( 1:(m/2),     1:(n/2),     1:(k/2))
      WINDOWA( 1:(m/2),     ((n/2)+1):n, ((k/2)+1):k)   =           &
		X_out( ((m/2)+1):m, 1:(n/2),     1:(k/2))
      WINDOWA( ((m/2)+1):m, 1:(n/2),     ((k/2)+1):k)   =           &
		X_out( 1:(m/2),     ((n/2)+1):n, 1:(k/2))
      WINDOWA( 1:(m/2),     1:(n/2),     ((k/2)+1):k)   =		   &
		X_out( ((m/2)+1):m, ((n/2)+1):n, 1:(k/2))

!     Elementary divide by the weighting matrix
      WINDOWA = WINDOWA / weight

!     Find primary peak location
      peak1 = MAXVAL( WINDOWA(  IWlims(1,1):IWlims(2,1),            &
                                IWlims(1,2):IWlims(2,2),            &
                                IWlims(1,3):IWlims(2,3)) )
      pl1   = MAXLOC( WINDOWA(  IWlims(1,1):IWlims(2,1),            &
                                IWlims(1,2):IWlims(2,2),            &
                                IWlims(1,3):IWlims(2,3)) )

!     Correct for lower limit
      pl1(1) = pl1(1) + IWlims(1,1) - 1
      pl1(2) = pl1(2) + IWlims(1,2) - 1
      pl1(3) = pl1(3) + IWlims(1,3) - 1
      pl1r   = real(pl1)

!     Set primary peak to zero, then find secondary peak
      WINDOWA( pl1(1), pl1(2), pl1(3) ) = 0	  
      peak2 = MAXVAL( WINDOWA(  IWlims(1,1):IWlims(2,1),            &
                                IWlims(1,2):IWlims(2,2),            &
                                IWlims(1,3):IWlims(2,3)) )
      pl2   = MAXLOC(WINDOWA(   IWlims(1,1):IWlims(2,1),            &
                                IWlims(1,2):IWlims(2,2),            &
                                IWlims(1,3):IWlims(2,3)) )

!     Correct for lower limit
      pl2(1) = pl2(1) + IWlims(1,1) - 1
      pl2(2) = pl2(2) + IWlims(1,2) - 1
      pl2(3) = pl2(3) + IWlims(1,3) - 1
      pl2r   = real(pl2)

!     Restore primary peak location
      WINDOWA( pl1(1), pl1(2), pl1(3) ) = peak1

!     Find the velocities for each peak to sub voxel accuracy
      DO peakcyc = 1,2
        IF (peakcyc.EQ.1) THEN
            pl   = pl1
            plr  = pl1r
            peak = peak1
		ELSEIF (peakcyc.EQ.2) THEN
            pl   = pl2
            plr  = pl2r
            peak = peak2
        ENDIF
		  		  
!       Store points either side of peak in 3 dimensions	  
        xarr(1:3) = [ WINDOWA(pl(1)-1, pl(2),   pl(3)  ),   &
                      WINDOWA(pl(1),   pl(2),   pl(3)  ),   &
					  WINDOWA(pl(1)+1, pl(2),   pl(3)  ) ]
		yarr(1:3) = [ WINDOWA(pl(1),   pl(2)-1, pl(3)  ),   &
					  WINDOWA(pl(1),   pl(2),   pl(3)  ),   &
					  WINDOWA(pl(1),   pl(2)+1, pl(3)  ) ]
        zarr(1:3) = [ WINDOWA(pl(1),   pl(2),   pl(3)-1),   &
					  WINDOWA(pl(1),   pl(2),   pl(3)  ),   &
					  WINDOWA(pl(1),   pl(2),   pl(3)+1) ]

!       If one of these values is close to zero, use the
!       others and find through linear interpolation
        DO i2 = 1,3
            IF (xarr(i2).LE.0) THEN
			    xarr(i2) = 0.0000001
		    ENDIF					
		    IF (yarr(i2).LE.0) THEN
			    yarr(i2) = 0.0000001
		    ENDIF
			IF (zarr(i2).LE.0) THEN
			    zarr(i2) = 0.0000001
		    ENDIF								          
        ENDDO
	  
!		Use a Gaussian fit. Calculate denominators first, setting 
!       velocity to zero where the denominator is zero
        denom(1) = 2*log(xarr(1)) - 4*log(xarr(2)) + 2*log(xarr(3))
        denom(2) = 2*log(yarr(1)) - 4*log(yarr(2)) + 2*log(yarr(3))
        denom(3) = 2*log(zarr(1)) - 4*log(zarr(2)) + 2*log(zarr(3))										

        IF (denom(1).EQ.0) THEN
            u = 0
        ELSE
            plr(1) = plr(1) + (log(xarr(1))-log(xarr(3)))/denom(1)                  
            u = (m/2) + 1 - plr(1)
        ENDIF
        IF (denom(2).EQ.0) THEN
            v = 0
        ELSE
            plr(2) = plr(2) + (log(yarr(1))-log(yarr(3)))/denom(2)
		    v = (n/2) + 1 - plr(2)
	    ENDIF		 	 
		IF (denom(3).EQ.0) THEN
		    w = 0
		ELSE
            plr(3) = plr(3) + (log(zarr(1))-log(zarr(3)))/denom(3)
			w = (k/2) + 1 - plr(3)
        ENDIF	


!	  Deallocate arrays
      DEALLOCATE(WINDOWA)
      DEALLOCATE(weight)



!	  ERROR HANDLING

!	  If there is a problem this frees the descriptor, and exits 
!	  subroutine. 
!	  For the forward transform descriptor:
 100  CONTINUE
      statFwd = DftiFreeDescriptor(DescHandle)
      IF (.NOT. DftiErrorClass(statFwd, DFTI_NO_ERROR) ) THEN
!          CALL dfti_example_status_print(statFwd)
      ENDIF

!	  For the backward transform descriptor
 110  CONTINUE
      statInv = DftiFreeDescriptor(DescHandle_2)
      IF (.NOT. DftiErrorClass(statInv, DFTI_NO_ERROR) ) THEN
!          CALL dfti_example_status_print(statInv)
      ENDIF

 101  CONTINUE

!	  End of subroutine thc_crosscorr
      END SUBROUTINE thc_crosscorr





 







!_______________________________________________________________________
!
!	  				SUBROUTINE CrossCorr
!
!	Performs a straightforward cross-correlation on two 3D scalar 
!   fields. Windowing is done by other subfunctions - this routine 
!   simply cross correlates WINDOWA with WINDOWB, where both those
!   variables are 3D arrays shared with this subfuncion via a module 
!   interface
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs are as follows:
!
!       CC_SMOOTH_FLAG  [1 x 1] integer*4
!                               A value of 1 causes the correlation 
!                               plane to be smoothed using a gaussian 
!                               kernel filter.
!
!		IW_MAX_D		    [1 x 1]	real*4
!								Maximum displacement of an interrogation
!								window during the cross correlation, 
!                               as a fraction of window size.
!                               For example, a value of 0.5 means that
!                               any displacement over 50% 
!                               of the interrogation window size is 
!                               considered impossible.
!                               This is useful, since edge effects 
!                               caused by the FFT can result in 
!                               erroneously high elements at the corners
!                               of the correlation volume, resulting in 
!                               false vectors
!       
!       WCTR            [1 x 1] integer *4
!                               Row index into the UX, UY, UZ, SNR 
!                               arrays (also shared in the module) for 
!                               the current window.
!
!       WSIZE           [1 x 1] integer *1
!                               dimension of the correlation cube array
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



      SUBROUTINE CrossCorr( WCTR, WSIZE, CC_SMOOTH_FLAG, IW_MAX_D)


!	  Using Intel Fortran Compiler FFT libraries
      USE MKL_DFTI

!     Safer to have no implicit variables
      IMPLICIT NONE

!     Include the Intel Math Kernel Library examples file, which
!     contains parameters and interfaces useful for FFT manipulation
	  INCLUDE 'mkl_dfti_examples.fi'


!	  Declare inputs as explicit-shape dummy arrays
      INTEGER*4, INTENT(in) :: CC_SMOOTH_FLAG, WCTR, WSIZE
	  REAL*4,    INTENT(in) :: IW_MAX_D
      



!     Descriptors (constructor objects) for FFTs
      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandle
      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleInv
      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleSmooth
      TYPE(DFTI_DESCRIPTOR), POINTER :: DescHandleInvSmooth


!     Admin variables for FFTs
      INTEGER*4 :: statFwd, statInv!, statFwdSmooth, statInvSmooth
      REAL*4    :: fftScale!, fftScaleSmooth
      INTEGER*4 :: lengths(3)!, lengths_s(3)
      INTEGER*4 :: strides_in(4)!, strides_in_s(4)
!      INTEGER*4 :: smii


!     Admin variables for setting up cross correlation array of 
!     interrogation volumes
      INTEGER*4 :: aCtr
      INTEGER*4 :: IWlims(2,3)
      INTEGER*4 :: n, m, k!, ms, ns, ks

!     Storage of the gaussian smoothing kernel
!      REAL,     DIMENSION(3,3,3) :: GX
	  
!	  Intermediate variables used for peak location
	  REAL*4    :: peak,    peak1,   peak2
	  INTEGER*4 :: pl(3),   pl1(3),  pl2(3),  peakCtr
	  REAL*4    :: plr(3),  pl1r(3), pl2r(3), denom(3)
	  REAL*4    :: xArr(3), yArr(3), zArr(3), u, v, w

!     Complex and real type arrays for storing intermediate corr vol
!        These are automatic arrays: explicit shape, nonconstant bounds.
	  COMPLEX,  DIMENSION(WSIZE,WSIZE,WSIZE) :: XA_3D, XB_3D 
	  COMPLEX,  DIMENSION(WSIZE*WSIZE*WSIZE) :: XA, XB
	  REAL*4,   DIMENSION(WSIZE,WSIZE,WSIZE) :: X_out

!     Smoothing windows for this pass. 
!        These are automatic arrays: explicit shape, nonconstant bounds.
!      COMPLEX,  DIMENSION(SW1(1),SW1(2),SW1(3)) :: kernelPad3D
!      COMPLEX,  DIMENSION(SW1(1)*SW1(2)*SW1(3)) :: kernelPad2D

!      COMPLEX,  DIMENSION(SW1(1),SW1(2),SW1(3)) :: u_pad3D_1
!      COMPLEX,  DIMENSION(SW1(1)*SW1(2)*SW1(3)) :: u_pad2D_1

!	  Debugging variables
!      character*120 line




!     Get the gaussian smoothing kernel
!      CALL GetSmoothingKernel(GX)



!	  Set transform parameters (always cubes)
      m = WSIZE!;     ms = SW1(1)
      n = WSIZE!;     ns = SW1(2)
      k = WSIZE!;     ks = SW1(3)


!     SETUP FFTs 

!     Setup lengths and strides for correlation FFTs
      lengths(1) = m
      lengths(2) = n
      lengths(3) = k
      strides_in(1) = 0
      strides_in(2) = 1
      strides_in(3) = m
      strides_in(4) = m*n

!     Setup lengths and strides for smoothing FFTs
      lengths_s(1) = ms
      lengths_s(2) = ns
      lengths_s(3) = ks
      strides_in_s(1) = 0
      strides_in_s(2) = 1
      strides_in_s(3) = ms
      strides_in_s(4) = ms*ns

!     Create DFTI descriptors for forward and inverse window transforms
      statFwd       = DftiCreateDescriptor( DescHandle,             &
                                            DFTI_SINGLE,            &
                                            DFTI_COMPLEX, 3, lengths)

      statInv       = DftiCreateDescriptor( DescHandleInv,          &
                                            DFTI_SINGLE,            &
                                            DFTI_COMPLEX, 3, lengths)


!     Create DFTI descriptors for forward and inverse smoothing kernel 
!     transforms
!      statFwdSmooth = DftiCreateDescriptor( DescHandleSmooth,       &
!                                            DFTI_SINGLE,            &
!                                            DFTI_COMPLEX, 3, lengths_s)

!      statInvSmooth = DftiCreateDescriptor( DescHandleInvSmooth,    &
!                                            DFTI_SINGLE,            &
!                                            DFTI_COMPLEX, 3, lengths_s)

!     Check for errors in creation of the descriptors
! TODO: Sort this...
!      IF (.NOT. DftiErrorClass(statFwd, DFTI_NO_ERROR)) THEN
!        GOTO 101 
!      ENDIF

!     Set the stride pattern for DFTIs
      statFwd       = DftiSetValue( DescHandle,                     &
                                    DFTI_INPUT_STRIDES, strides_in)      

!      statFwdSmooth = DftiSetValue( DescHandleSmooth,               &
!                                    DFTI_INPUT_STRIDES, strides_in_s)

!     Commit Dfti descriptors
      statFwd       = DftiCommitDescriptor( DescHandle       )
!      statFwdSmooth = DftiCommitDescriptor( DescHandleSmooth )


!     Set fftScale number for inverse transforms
      fftScale       = 1.0/real(m*n*k, KIND=4)
!      fftScaleSmooth = 1.0/real(ms*ns*ks, KIND=4)
      
      statInv        = DftiSetValue(DescHandle_2,                   &
                                    DFTI_BACKWARD_SCALE, fftScale)

!      statInvSmooth  = DftiSetValue(DescHandleInvSmooth,            &
!                                    DFTI_BACKWARD_SCALE, fftScaleSmooth)


!     Commit Dfti descriptor for backwards transform
      statInv       = DftiCommitDescriptor( DescHandleInv )
!      statInvSmooth = DftiCommitDescriptor( DescHandleInvSmooth )

!     Compute forward transform of padded convolution kernel
!     (done once and reused for each window)
!      kernelPad3D = 0
!      kernelPad3D(1:3,1:3,1:3) = GX;
!      kernelPad2D = RESHAPE(kernelPad3D,   SHAPE(kernelPad2D))
!      statFwdSmooth = DftiComputeForward(   DescHandleSmooth,       &
!                                            kernelPad2D)




!     EXECUTE CROSS-CORRELATION
      XA_3D(:,:,:) = CMPLX(WINDOWA)
	  XB_3D(:,:,:) = CMPLX(WINDOWB)
	  
!	  FFT functions need to read data in the 3D arrays as if it were 
!     a vector. The EQUIVALENCE statement used in the examples to 
!     achieve this without a memory copy is now deprecated by IFC as it 
!     is incompatible with module arrangements and variable size arrays,
!     so here we reshape the arrays instead of assigning equivalence:
      XA = RESHAPE(XA_3D,   SHAPE(XA))
      XB = RESHAPE(XB_3D,   SHAPE(XB))

!     Compute forward transform of window A
      statFwd = DftiComputeForward(DescHandle,XA)
	  
!     Compute forward transform of window B
      statFwd = DftiComputeForward(DescHandle,XB)

!     Multiply the first fft with the complex conjugate of the second
      XA = XA * conjg(XB)
      
!     Smooth the correlation plane before the inverse transformation
!      IF (CC_SMOOTH_FLAG.EQ.1) THEN
!      
!        CALL thc_padreplicate(IW,u_pad3D_1,X_IN_3D)
!        statFwdSmooth = DftiComputeForward(DescHandleSmooth, u_pad2D_1)
!        DO smii = 1,ms*ns*ks
!            u_pad2D_1(smii) = u_pad2D_1(smii)*kernelPad2D(smii)
!        ENDDO
!        statInvSmooth = DftiComputeBackward( DescHandleInvSmooth,   &
!                                             u_pad2D_1)
!        ! NB THIS ISN'T RIGHT - relied on the old equivalence statements X_IN_3D = u_pad3D_1(2:ms-1,2:ns-1,2:ks-1);
!      ENDIF
      
!     Compute backwards transformation
      statInv = DftiComputeBackward(DescHandleInv, XA)

!     Reshape back to 3D array
      XA_3D = RESHAPE(XA, SHAPE(XA_3D))

!     Perform FFTSHIFT on real component of the correlation volume,
!     storing in the shared WINDOWA array. This is the same as MATLAB's
!     fftShift() command.
      X_out = REAL(XA_3D)
      WINDOWA( 1:m/2,       1:n/2,       1:k/2)         =           &
		X_out( ((m/2)+1):m, ((n/2)+1):n, ((k/2)+1):k)
      WINDOWA( ((m/2)+1):m, 1:n/2,       1:k/2)         =           &
		X_out( 1:m/2,       ((n/2)+1):n, ((k/2)+1):k)
      WINDOWA( 1:m/2,       ((n/2)+1):n, 1:k/2)         =           &
		X_out( ((m/2)+1):m, 1:n/2,       ((k/2)+1):k)
      WINDOWA( ((m/2)+1):m, ((n/2)+1):n, 1:(k/2))       =           &
		X_out( 1:(m/2),     1:(n/2),     ((k/2)+1):k)
      WINDOWA( ((m/2)+1):m, ((n/2)+1):n, ((k/2)+1):k)   =           &
		X_out( 1:(m/2),     1:(n/2),     1:(k/2))
      WINDOWA( 1:(m/2),     ((n/2)+1):n, ((k/2)+1):k)   =           &
		X_out( ((m/2)+1):m, 1:(n/2),     1:(k/2))
      WINDOWA( ((m/2)+1):m, 1:(n/2),     ((k/2)+1):k)   =           &
		X_out( 1:(m/2),     ((n/2)+1):n, 1:(k/2))
      WINDOWA( 1:(m/2),     1:(n/2),     ((k/2)+1):k)   =		   &
		X_out( ((m/2)+1):m, ((n/2)+1):n, 1:(k/2))

!     Elementary divide by the WEIGHTing matrix
      WINDOWA = WINDOWA / WEIGHT

!     Find primary peak location
      peak1 = MAXVAL( WINDOWA(  IWlims(1,1):IWlims(2,1),            &
                                IWlims(1,2):IWlims(2,2),            &
                                IWlims(1,3):IWlims(2,3)) )
      pl1   = MAXLOC( WINDOWA(  IWlims(1,1):IWlims(2,1),            &
                                IWlims(1,2):IWlims(2,2),            &
                                IWlims(1,3):IWlims(2,3)) )

!     Correct for lower limit
      pl1(1) = pl1(1) + IWlims(1,1) - 1
      pl1(2) = pl1(2) + IWlims(1,2) - 1
      pl1(3) = pl1(3) + IWlims(1,3) - 1
      pl1r   = real(pl1)

!     Set primary peak to zero, then find secondary peak
      WINDOWA( pl1(1), pl1(2), pl1(3) ) = 0	  
      peak2 = MAXVAL( WINDOWA(  IWlims(1,1):IWlims(2,1),            &
                                IWlims(1,2):IWlims(2,2),            &
                                IWlims(1,3):IWlims(2,3)) )
      pl2   = MAXLOC(WINDOWA(   IWlims(1,1):IWlims(2,1),            &
                                IWlims(1,2):IWlims(2,2),            &
                                IWlims(1,3):IWlims(2,3)) )

!     Correct for lower limit
      pl2(1) = pl2(1) + IWlims(1,1) - 1
      pl2(2) = pl2(2) + IWlims(1,2) - 1
      pl2(3) = pl2(3) + IWlims(1,3) - 1
      pl2r   = real(pl2)

!     Restore primary peak location
      WINDOWA( pl1(1), pl1(2), pl1(3) ) = peak1

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
        xArr(1:3) = [ WINDOWA(pl(1)-1, pl(2),   pl(3)  ),   &
                      WINDOWA(pl(1),   pl(2),   pl(3)  ),   &
					  WINDOWA(pl(1)+1, pl(2),   pl(3)  ) ]
		yArr(1:3) = [ WINDOWA(pl(1),   pl(2)-1, pl(3)  ),   &
					  WINDOWA(pl(1),   pl(2),   pl(3)  ),   &
					  WINDOWA(pl(1),   pl(2)+1, pl(3)  ) ]
        zArr(1:3) = [ WINDOWA(pl(1),   pl(2),   pl(3)-1),   &
					  WINDOWA(pl(1),   pl(2),   pl(3)  ),   &
					  WINDOWA(pl(1),   pl(2),   pl(3)+1) ]

!       If one of these values is <= zero, use the
!       others and find through linear interpolation
        DO aCtr = 1,3
            IF (xArr(aCtr).LE.0) THEN
			    xArr(aCtr) = 0.0000001
		    ENDIF					
		    IF (yArr(aCtr).LE.0) THEN
			    yArr(aCtr) = 0.0000001
		    ENDIF
			IF (zArr(aCtr).LE.0) THEN
			    zArr(aCtr) = 0.0000001
		    ENDIF								          
        ENDDO
	  
!		Use a Gaussian fit. Calculate denominators first, setting 
!       velocity to NaN where the denominator is zero
        denom(1) = 2*log(xArr(1)) - 4*log(xArr(2)) + 2*log(xArr(3))
        denom(2) = 2*log(yArr(1)) - 4*log(yArr(2)) + 2*log(yArr(3))
        denom(3) = 2*log(zArr(1)) - 4*log(zArr(2)) + 2*log(zArr(3))										

        IF (denom(1).EQ.0) THEN
            u = sqrt(-1.0)!NaN
        ELSE
            plr(1) = plr(1) + (log(xArr(1))-log(xArr(3)))/denom(1)                  
            u = (m/2) + 1 - plr(1)
        ENDIF
        IF (denom(2).EQ.0) THEN
            v = sqrt(-1.0)!NaN
        ELSE
            plr(2) = plr(2) + (log(yArr(1))-log(yArr(3)))/denom(2)
		    v = (n/2) + 1 - plr(2)
	    ENDIF		 	 
		IF (denom(3).EQ.0) THEN
		    w = sqrt(-1.0)!NaN
		ELSE
            plr(3) = plr(3) + (log(zArr(1))-log(zArr(3)))/denom(3)
			w = (k/2) + 1 - plr(3)
        ENDIF	

!       Store results to the shared arrays for output
        IF (peakCtr.EQ.1) THEN
            UX(WCTR,1) = u
            UY(WCTR,1) = v
            UZ(WCTR,1) = w
            SNR(WCTR,1) = peak/(SUM(WINDOWA)/(real(WSIZE*WSIZE*WSIZE, kind = 4))
        ELSE
            UX(WCTR,2) = u
            UY(WCTR,2) = v
            UZ(WCTR,2) = w
            SNR(WCTR,2) = peak/(SUM(WINDOWA)/(real(WSIZE*WSIZE*WSIZE, kind = 4))
        END
            
            
      ENDDO



!	  ERROR HANDLING

!	  If there is a problem this frees the descriptor, and exits 
!	  subroutine. 
!	  For the forward transform descriptor:
 100  CONTINUE
      statFwd = DftiFreeDescriptor(DescHandle)
      IF (.NOT. DftiErrorClass(statFwd, DFTI_NO_ERROR) ) THEN
!          CALL dfti_example_status_print(statFwd)
      ENDIF

!	  For the backward transform descriptor
 110  CONTINUE
      statInv = DftiFreeDescriptor(DescHandle_2)
      IF (.NOT. DftiErrorClass(statInv, DFTI_NO_ERROR) ) THEN
!          CALL dfti_example_status_print(statInv)
      ENDIF

 101  CONTINUE

!	  End of subroutine
      END SUBROUTINE CrossCorr
 
