
!_______________________________________________________________________
!
! WEIGHTINGS MODULE AND SUBROUTINE
!_______________________________________________________________________
!
! Definition of SHARED ARRAYS declared within module:
!
!		PIXELS		[npix x 1]	Single precision, real, non-negative.
!								Contains the intensities of the pixels 
!								from which the reconstruction is made, 
!								in a column vector format.
!
!		VOXELS		[nvox x 1]	Single precision, real, non-negative.
!								Contains the intensity field of the 
!								reconstructed volume. See notes below on 
!								storage order.
!
!		C			[npix x 3]	Single precision, real.
!								Contains the first coefficients of the 
!								line of sight of each pixel 
!								(the A of S = A + B*lamda). This is 
!								denoted C here, as the coefficient must 
!								be such that the result is in the VOXEL 
!								frame, NOT the global mm frame. For each
!								coefficient C there are 3 values, 
!								representing the X,Y and Z directions 
!								respectively (hence 3 columns)
!
!		D			[npix x 3]	Single precision, real.
!								Contains the second coefficients of the 
!								line of sight of each pixel 
!								(the B of S = A + B*lamda). This is 
!								denoted D here, as the coefficient must 
!								be such that the result is in the VOXEL 
!								frame, NOT the global mm frame. For each
!								coefficient D there are 3 values, 
!								representing the X,Y and Z directions 
!								respectively (hence 3 columns)
!
!		LOS_FACTOR	[2 x npix]	Single precision, real
!								0 <= LOS_FACTOR(i,j) <= 1.
!								Where a pixel LoS passes through a plane
!								of constant Z, it makes an angle with 
!								that plane. When calculating the 
!								distance between the LoS and a voxel in 
!								that plane, the distance is calculated 
!								by determining the position where the 
!								line intersects the plane (then 
!								subtracting the voxel coordinate). This 
!								gives the hypotenuse of a right-angled 
!								triangle, since if the LoS is at an 
!								angle, the LoS will always pass slightly 
!								closer to the voxel than the 
!								intersection point. The LOS factor is 
!								passed in for each pixel to compensate 
!								for this. Factors are applied 
!								independantly in the x and y directions,
!								so row 1 contains a factor for X, row 2 
!								contains a factor for Y.
!								The orientation of the array is made 
!								such that sequential retrieval from 
!								memory is achievable (faster).
!
!
! Author:           T.H. Clark
! Work address:     Fluids / CFD Lab
!                   Cambridge University Engineering Department
!                   2 Trumpington Street
!                   Cambridge
!                   CB21PZ
! Email:            t.clark@cantab.net
! Website:          http://www2.eng.cam.ac.uk/~thc29
!
! Created:          01 June 2009 
! Last revised:     12 June 2009




!_______________________________________________________________________
!
!					       MODULE MFG_MOD
!_______________________________________________________________________

	  MODULE mfg_mod
	  IMPLICIT NONE
	  SAVE

!	  SAVE
!	  Unnecessary to SAVE as only one program unit ever uses the module.

! 	  Here, we declare variables which are: 
!		- Used by more than one program unit
!		- Are of a variable size (are allocatable)
	  REAL*4, DIMENSION(:,:), ALLOCATABLE :: C, D, LOS_FACTOR
	  INTEGER*4, DIMENSION(:), ALLOCATABLE :: INDICES 
	  REAL*4,    DIMENSION(:), ALLOCATABLE :: WEIGHTS, VOX_TEMP
	  REAL*4, DIMENSION(:), ALLOCATABLE :: PIXELS
	  REAL*4, DIMENSION(:), ALLOCATABLE :: VOXELS



	  CONTAINS



!_______________________________________________________________________







!_______________________________________________________________________

!                       SUBROUTINE MFG
!_______________________________________________________________________


	  SUBROUTINE mfg(NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9, NCAMS, &
                        CAM_OFFSETS, LOOKUP_WTS, PIX_THRESH)

!	  NOTES ON THEORY
!	  
!	  Multiplicative first guess for tomographic reconstruction of a 
!	  particle field. See:
!	  	Worth N.A. and Nickels T.B. (2008) 'Acceleration of Tomo-PIV by 
!		estimating the initial volume intensity distribution', 
!		Exp. Fluids, DOI 10.1007/s00348-008-0504-6
!
!	  For straightforward application of the MART algorithm (such as is 
!	  found in mxmart_large), the zero-level pixels must be submitted to
!	  the get_weights subroutine during the first iteration. Thus the 
!	  voxels whose intensity is zero can be ascertained. For subsequent 
!	  iterations, the zero level voxels need not be computed. The first 
!	  iteration, then, is the longest to execute.
!
!	  Here, a computationally efficient way of determining the zero 
!	  level voxels is used to reduce the size of the problem. 
!	  For each camera, a logical array is built representing those 
!	  voxels which are weighted with a nonzero pixel. This array is of 
!	  dimension NVOX_Y*NVOX_X*NVOX_Z ie contains one element for each 
!	  voxel.
!
!	  There are two potential benefits: 
!			1. 	Reduce the duration of the initial iteration by only 
!				processing NZ pixels
!			2.  Improve the first guess in the regions where voxels 
!				contain nonzero values
!	  		2.  Use of the resultant logical mask to speed up the 
!				get_weights_mfg subroutine in subsequent iterations
!
!	Temp note- thought occurs that it might be quicker to totally rejig 
!	the calculation - run through each voxel, map it to pixels in each
!	camera, and see whether NZ or Z intensity. If NZ, then multiply by 
!	the value in each camera. This is the same as the MFG technique, but 
!	the other way around - could be quite efficient.


	
!_____________________ DECLARE INTERNAL VARIABLES ______________________

	  IMPLICIT NONE

! 	  Input variables
	  INTEGER*4, INTENT(in) :: NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9, NCAMS
	  INTEGER*4, INTENT(in) :: CAM_OFFSETS(NCAMS)
	  REAL*4,    INTENT(in) :: LOOKUP_WTS(227)
	  REAL*4,    INTENT(in) :: PIX_THRESH
	  
!	  Output variables
!	  	- none. Output made via module shared variables

!	  Internal Variables
	  INTEGER*4 :: PIXCTR, CAMCTR, N_WTS, START_IND, FIN_IND, NVOX
	  LOGICAL*1 :: MFG_MODE
	  REAL*4, DIMENSION(:), ALLOCATABLE :: VOX_MULT

!	  Module Variables
!	  Used, but do not declare in subroutines:
!	  INTEGER*4, DIMENSION(:), ALLOCATABLE :: INDICES 
!	  LOGICAL*1, DIMENSION(:,:), ALLOCATABLE :: VOX_MASK
!	  REAL*4, DIMENSION(:), ALLOCATABLE :: PIXELS
!	  REAL*4, DIMENSION(:), ALLOCATABLE :: VOXELS
	  
! Temporary debugging symbols
       character*120 line



!____________________ INITIALISE INTERNAL VARIABLES ____________________

!	  VOX_MULT array used to store a multiplication array for each 
!	  camera
	  NVOX = NVOX_X*NVOX_Y*NVOX_Z
	  ALLOCATE(VOX_MULT(NVOX))



!______________________ FIRST CAMERA PREMASKING ________________________


!     We handle the first camera as a special case (duplicates code 
!	  somewhat, but simpler to understand). Basically, just store 
!	  results straight into the VOXELS array as this saves us a matrix 
!	  multiplication. This requires that VOXELS is initialised to 0, 
!     rather than some nonzero value
	  CAMCTR = 1;

!	  For first camera, MFG_MODE = .TRUE. to avoid zeroing all voxels
	  MFG_MODE = .TRUE.

!	  Loop through the pixels from the current camera
	  DO PIXCTR = 1, CAM_OFFSETS(1)

!		If pixel has nonzero intensity, calculate its weightings
		IF (PIXELS(PIXCTR) .GE. PIX_THRESH) THEN

!     		Call the get_preweight subroutine to determine the 
!			indices which are weighted
			CALL get_weight_mfg(NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9,   &
							   PIXCTR, LOOKUP_WTS, N_WTS, MFG_MODE)

!	  		Multiply the voxel values by the pixel intensity
			VOXELS(INDICES(1:N_WTS)) = PIXELS(PIXCTR)

		ENDIF
	  ENDDO


!______________________ OTHER CAMERAS PREMASKING _______________________

!	  For cameras other than the first one
	  MFG_MODE = .FALSE.


	  DO CAMCTR = 2,NCAMS

!		Determine the starting and finishing indices into the pixels 
!		array for this camera
		START_IND = CAM_OFFSETS(CAMCTR-1) + 1
		FIN_IND = CAM_OFFSETS(CAMCTR)

!		Loop through the pixels from the current camera
	  	DO PIXCTR = START_IND, FIN_IND

!	    	If pixel has nonzero intensity, calculate its weightings
			IF (PIXELS(PIXCTR) .GE. 0.000001) THEN

!     			Call the get_preweight subroutine to determine the 
!				indices which are weighted
				CALL get_weight_mfg(NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9,   &
							   PIXCTR, LOOKUP_WTS, N_WTS, MFG_MODE)

!	  			Multiply the voxel values by the pixel intensity
				VOX_MULT(INDICES(1:N_WTS)) = PIXELS(PIXCTR)

			ENDIF

		ENDDO

!		Multiplication of the projected field from the current camera 
!		with the voxels field.
		VOXELS = VOXELS * VOX_MULT
		VOX_MULT = 0.0

!	  End the CAMCTR loop
	  ENDDO

!	  Normalise by the number of cameras to ensure relative field 
!	  magnitude continuity between MFG and MART
	  VOXELS = VOXELS**(1.0/REAL(NCAMS))



!	  End the premask subroutine
	  END SUBROUTINE mfg













!_______________________________________________________________________

!                       SUBROUTINE GET_WEIGHT_MFG
!_______________________________________________________________________

      SUBROUTINE get_weight_mfg(NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9, 	   &
							   PIXCTR, LOOKUP_WTS, N_WTS, MFG_MODE)


!_______________________________ NOTES _________________________________

!	  The get_weight_mfg subroutine is almost the same as the heavily 
!	  commented get_weight subroutine in mxmart_large. Where variables 
!	  have the same specifications they are named the same as in 
!	  get_weight. Comments are mostly omitted (rather than repeat 
!	  them from get_weights).



!___________________________ DECLARATIONS ______________________________

!     Declare Input Arguments (unchanged by this subroutine)
	  INTEGER*4, INTENT(in) :: NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9, 	   &
							   PIXCTR
      REAL*4,    INTENT(in) :: LOOKUP_WTS(227)
      LOGICAL*1, INTENT(in) :: MFG_MODE

!     Declare Output Arguments (set or updated by this subroutine)
	  INTEGER*4, INTENT(out):: N_WTS
	  
!	  Declare Internal Arguments (used only in this subroutine)
	  INTEGER*4 :: ZCTR, DPC, PACKIND, PACKCTR, N_VALID
	  INTEGER*4 :: X_C_IND(NVOX_Z), Y_C_IND(NVOX_Z)
	  INTEGER*4 :: X_INDS(NVOX_Z9), Y_INDS(NVOX_Z9), Z_INDS(NVOX_Z9)
	  INTEGER*4, DIMENSION(:), ALLOCATABLE :: LOOKUP_IND

	  REAL*4 :: C_VALUE(3), D_VALUE(3)
	  REAL*4 :: ZVEC(NVOX_Z), P1(NVOX_Z), P2(NVOX_Z)
	  REAL*4 :: P1_LARGE(NVOX_Z9), P2_LARGE(NVOX_Z9)
	  REAL, DIMENSION(:), ALLOCATABLE :: DX, DY, DIST_SQD

      LOGICAL*1 :: LOG_SMALL(NVOX_Z9)

! Temporary debugging symbols
       character*120 line


!____________________ LoS INTERSECTION WITH Z PLANE ____________________

!	  Determine LoS of current pixel.
	  C_VALUE = C(PIXCTR,:)
	  D_VALUE = D(PIXCTR,:)

!	  Determine, at each plane of voxels, the intersection 
!	  position between the LoS and that plane.
	  DO ZCTR = 1,NVOX_Z
		ZVEC(ZCTR) = REAL(ZCTR)
	  ENDDO
!	  Now find the X,Y positions at which the pixel LoSs intersect 
!	  each Z plane
	  P1 = (ZVEC*D_VALUE(1)) + C_VALUE(1)
	  P2 = (ZVEC*D_VALUE(2)) + C_VALUE(2)


!___________________ FIND INDICES OF AFFECTED VOXELS ___________________

	  X_C_IND = NINT(P1)
	  Y_C_IND = NINT(P2)
	  DPC = 1
	  DO ZCTR = 1, NVOX_Z

    	X_INDS(DPC)   = X_C_IND(ZCTR) - 1
    	X_INDS(1+DPC) = X_C_IND(ZCTR) - 1
    	X_INDS(2+DPC) = X_C_IND(ZCTR) - 1
   		X_INDS(3+DPC) = X_C_IND(ZCTR)
    	X_INDS(4+DPC) = X_C_IND(ZCTR)
    	X_INDS(5+DPC) = X_C_IND(ZCTR)
    	X_INDS(6+DPC) = X_C_IND(ZCTR) + 1
    	X_INDS(7+DPC) = X_C_IND(ZCTR) + 1
    	X_INDS(8+DPC) = X_C_IND(ZCTR) + 1

    	Y_INDS(DPC)   = Y_C_IND(ZCTR) - 1
    	Y_INDS(1+DPC) = Y_C_IND(ZCTR)
    	Y_INDS(2+DPC) = Y_C_IND(ZCTR) + 1
    	Y_INDS(3+DPC) = Y_C_IND(ZCTR) - 1
    	Y_INDS(4+DPC) = Y_C_IND(ZCTR)
    	Y_INDS(5+DPC) = Y_C_IND(ZCTR) + 1
    	Y_INDS(6+DPC) = Y_C_IND(ZCTR) - 1
    	Y_INDS(7+DPC) = Y_C_IND(ZCTR)
    	Y_INDS(8+DPC) = Y_C_IND(ZCTR) + 1

!		Easily indexed, but unwrapping like this is slightly quicker 
!		with ifort, for no apparent reason
		Z_INDS(DPC)   = ZCTR
		Z_INDS(DPC+1) = ZCTR
		Z_INDS(DPC+2) = ZCTR
		Z_INDS(DPC+3) = ZCTR
		Z_INDS(DPC+4) = ZCTR
		Z_INDS(DPC+5) = ZCTR
		Z_INDS(DPC+6) = ZCTR
		Z_INDS(DPC+7) = ZCTR
		Z_INDS(DPC+8) = ZCTR

		P1_LARGE(DPC)   = P1(ZCTR)
		P1_LARGE(DPC+1) = P1(ZCTR)
		P1_LARGE(DPC+2) = P1(ZCTR)
		P1_LARGE(DPC+3) = P1(ZCTR)
		P1_LARGE(DPC+4) = P1(ZCTR)
		P1_LARGE(DPC+5) = P1(ZCTR)
		P1_LARGE(DPC+6) = P1(ZCTR)
		P1_LARGE(DPC+7) = P1(ZCTR)
		P1_LARGE(DPC+8) = P1(ZCTR)

		P2_LARGE(DPC)   = P2(ZCTR)
		P2_LARGE(DPC+1) = P2(ZCTR)
		P2_LARGE(DPC+2) = P2(ZCTR)
		P2_LARGE(DPC+3) = P2(ZCTR)
		P2_LARGE(DPC+4) = P2(ZCTR)
		P2_LARGE(DPC+5) = P2(ZCTR)
		P2_LARGE(DPC+6) = P2(ZCTR)
		P2_LARGE(DPC+7) = P2(ZCTR)
		P2_LARGE(DPC+8) = P2(ZCTR)

    	DPC = DPC + 9;
	  ENDDO


!	  Some of these indices will pertain to voxels which are not within 
!	  the volume being reconstructed (e.g. where the LoS passes close to
!	  the edge of the volume). Determine which ones are valid and pack 
!	  into consecutive locations:
	  PACKIND = 1
	  DO PACKCTR = 1,NVOX_Z9
		IF ( (X_INDS(PACKCTR).GT.0) .AND. (Y_INDS(PACKCTR).GT.0) 	   &
									.AND. (X_INDS(PACKCTR).LE.NVOX_X)  &
									.AND. (Y_INDS(PACKCTR).LE.NVOX_Y)) &
		THEN
			X_INDS(PACKIND) = X_INDS(PACKCTR)
			Y_INDS(PACKIND) = Y_INDS(PACKCTR)
			Z_INDS(PACKIND) = Z_INDS(PACKCTR)
			P1_LARGE(PACKIND) = P1_LARGE(PACKCTR)
			P2_LARGE(PACKIND) = P2_LARGE(PACKCTR)
			PACKIND = PACKIND +1
		ENDIF
	  ENDDO
	  N_VALID = PACKIND-1

!	  Calculate single-element index from subscripts
	  INDICES(1:N_VALID) = (NVOX_X*NVOX_Y)*(Z_INDS(1:N_VALID)-1) +     &
									NVOX_Y*(X_INDS(1:N_VALID)-1) + 	   &
									Y_INDS(1:N_VALID)



!________________ MASK WHERE VOXEL INTENSITY ALREADY = 0 _______________


!	  From the existing VOXELS array, we can determine which indices 
!	  point to voxels which already have zero intensity.
	  VOX_TEMP(1:N_VALID)  = VOXELS(INDICES(1:N_VALID))

!	  Mask and pack again to remove indices to voxels already masked, 
!	  unless operating in MFG mode, in which case doing this would 
!	  cause no voxels to be updated.

	  IF (.NOT.MFG_MODE) THEN
	  	PACKIND = 1
	  	DO PACKCTR = 1,N_VALID
			IF (VOX_TEMP(PACKCTR) .GE. 0.0000001) THEN
				VOX_TEMP(PACKIND) = VOX_TEMP(PACKCTR)
				INDICES(PACKIND)  = INDICES(PACKCTR)
    			X_INDS(PACKIND)   = X_INDS(PACKCTR)
				Y_INDS(PACKIND)   = Y_INDS(PACKCTR)
				P1_LARGE(PACKIND) = P1_LARGE(PACKCTR)
				P2_LARGE(PACKIND) = P2_LARGE(PACKCTR)
				PACKIND = PACKIND+1
			ENDIF
	  	ENDDO
!	    Update the number of valid weights
	    N_WTS = PACKIND - 1
	  ELSE
		N_WTS = N_VALID
	  ENDIF



! 	  Often, the number of weighted pixels is zero. Performing this 
!	  logical test reduces execution time
	  IF (N_WTS .EQ. 0) THEN 
		RETURN
	  ENDIF


!_____________ CALCULATE DISTANCE BETWEEN LOS AND EACH VOXEL ___________

!     Allocate temporary arrays to hold calculations
	  ALLOCATE(DX(N_WTS))
	  ALLOCATE(DY(N_WTS))
	  ALLOCATE(DIST_SQD(N_WTS))
	  ALLOCATE(LOOKUP_IND(N_WTS))

!	  Find the distance, in voxels, in the X and Y directions between 
!	  voxels and intersection points. Also correct for the Lines of 
!     Sight not being perpendicular (10% change for a 25 degree angle)
	  DX = X_INDS(1:N_WTS) - P1_LARGE(1:N_WTS)
	  DY = Y_INDS(1:N_WTS) - P2_LARGE(1:N_WTS)
	  DX = DX*LOS_FACTOR(1,PIXCTR)
	  DY = DY*LOS_FACTOR(2,PIXCTR)
	  DIST_SQD = (DX**2) + (DY**2)


!______________________ DETERMINE WEIGHTINGS (LOOKUP) __________________

	  LOOKUP_IND = NINT(DIST_SQD/0.02) + 1
	  WEIGHTS(1:N_WTS) = LOOKUP_WTS(LOOKUP_IND)

!     We need a further pack to account for any voxels in the 3x3 zone 
!	  which have a zero weighting
  	  PACKIND = 1
  	  DO PACKCTR = 1,N_WTS
		IF (WEIGHTS(PACKCTR) .GE. 0.000001) THEN
			VOX_TEMP(PACKIND) = VOX_TEMP(PACKCTR)
  			INDICES(PACKIND)  = INDICES(PACKCTR)
  			WEIGHTS(PACKIND)  = WEIGHTS(PACKCTR)
			PACKIND = PACKIND+1
		ENDIF
  	  ENDDO
	  N_WTS = PACKIND - 1;

!	  Deallocate the temporary variables
	  DEALLOCATE(DX)
	  DEALLOCATE(DY)
	  DEALLOCATE(DIST_SQD)
	  DEALLOCATE(LOOKUP_IND)


! End of subroutine (get_preweight)
      END SUBROUTINE get_weight_mfg





! End of module (mfg_mod)
	  END MODULE mfg_mod














