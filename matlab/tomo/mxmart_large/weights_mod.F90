
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





	  MODULE weights_mod


	  IMPLICIT NONE
!	  Unnecessary to SAVE as only one program unit ever uses the module.

! 	  Here, we declare variables which are: 
!		- Shared between mexFunction and get_weights
!		- Are of a variable size (are allocatable)
	  REAL*4, DIMENSION(:), ALLOCATABLE :: PIXELS
	  REAL*4, DIMENSION(:,:), ALLOCATABLE :: C, D, LOS_FACTOR
	  REAL*4, DIMENSION(:), ALLOCATABLE :: VOXELS
	  INTEGER*4, DIMENSION(:), ALLOCATABLE :: INDICES 
	  REAL*4,    DIMENSION(:), ALLOCATABLE :: VOX_TEMP, WEIGHTS



!_______________________________________________________________________

	  CONTAINS




      SUBROUTINE get_weights(NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9, NVOX,    &
							 PIXCTR, PIXZERO, LOOKUP_WTS,    		   &
							 N_WTS, VOX_THRESH)

! 	  NOTES ON THEORY
!
!	  We have made the assumption that camera LoSs are within about 25 
!	  degrees of the normal Z direction. Looking down a line of sight, 
!	  we see that at each 'layer' of voxels in constant Z, the LoS 
!	  intersects one of these voxels. With P1 and P2, we have calculated
!	  the position of this intersection at each Z layer.
!
!	  We say that the voxel at each Z plane which is intersected by the 
!	  LoS is weighted, i.e. the current pixel can 'see' those voxels. 
!	  We also make the assumption that the neighbouring voxels may also 
!	  be weighted - so we take a 3x3 square of voxels, at each z plane, 
!	  (centred on the voxel which is intersected by the LoS) and 
!	  calculate the weightings for those voxels. Outside that 3x3 square
!	  we assume that there is no weighting thus do not invest the 
!	  computational effort of calculating weights.
!
!	  The use of a 3x3 square has been suggested by Nick Worth, who has 
!	  validated the assumption (results were not significant using 5x5 
!	  squares or other shapes which calculated weightings for more 
!	  voxels).

!	  This explains also the function of the 'shortcut' flag. Typically, 
!	  we would just calculate the weightings in the 3x3 squares. 
!	  However, what if the pixel has zero intensity (i.e. the PIXZERO 
!	  flag is set)? In that case, we do not need to know what the 
!	  weighting is; we just need to know that there is a weighting. 
!	  To save the effort of calculating the weighting, the shortcut flag
!	  is turned on - in that case, we simply assume that all of the 
!	  voxels in the 3x3 square are weighted by some amount, and so we 
!	  can return their indices without calculating any weights.
!
!
!     NOTES ON CODE STRUCTURE (use of a subroutine for this function):
!
!     1: Computational Efficiency
!			There is potential for using a graphics-card based system 
!			for calculating these weightings, which is the most 
!			computationally intensive part of Tomographic PIV. So, 
!			although we are likely ultimately to code the entire 
!			reconstruction for a graphics card, we could just swap out 
!			this subroutine for one written in C for a graphics card, 
!			then include it from the main fortran function.
!			There is an overhead (usually ~100 flops) associated with 
!			calling a subroutine. For this reason, best to use an inline 
!			flag when compiling to remove the overhead.
!     2: Clarity of Code
!			Calculation of the pixel-voxel weightings is irritating (the
!			code isn't very concise for reasons of computational 
!			efficiency). So, it's better to put this into a subroutine, 
!			allowing anyone working on the code at a later date to 
!			simply pay attention to the MART algorithm itself.
!     3: Flexibility of Code
!			If you want to use a different weightings calculation just 
!			swap the subroutine.
!			Also, this routine is used in different ways by the main 
!			routine. To achieve that without a subroutine, code would 
!			be repetitive.
!	  
!
!	  NOTES ON STORAGE ORDER OF VOXELS ARRAY:
!
!			VOXELS is a vector, but it represents a multidimensional 
!			array. The dimensions are by -Y (row), then X (col) then -Z 
!			(depth).
!			The reason for this slightly bizarre choice of direction is
!			(a) To give efficient indexing into the array and (b) so 
!			that the matrix is arranged as viewed in a typical 
!			calibration shot: i.e. the first (NVOX_X*NVOX_Y) elements 
!			represent the layer of voxels at the face plane with the 
!			first element at the top left, NVOX_Yth element at the 
!			bottom left, ((NVOX_X-1)*NVOX_Y)+1 th element at the top 
!			right, and (NVOX_X*NVOX_Y)th element at the bottom right.
!			------------------------------------------------------------
!			In summary, the first element contains the value at
!			[Xmin, Ymax, Zmax]. The second element contains the value at 
!			[Xmin, (Ymax-Vox_Size), Zmax]. The (NVOX_X+1)th element 
!			contains the value at [(Xmin+vox_size), Ymax, Zmax]. 
!			The (NVOX_X*NVOX_Y + 1)th element contains the value at 
!			[Xmin, Ymax, Zmax-vox_size]
!
!
!	  FUTURE IMPROVEMENTS
!
!	  The subroutine is structured such that the intersection line is 
!	  worked out (variables LAMDA, P1, P2) then these are used to create
!	  longer variables containing the 3x3 voxel grids at each Z layer. 
!	  It may well be more efficient if Z_INDS is calculated right from 
!	  the start, and LAMDA is calculated for all of those (even though 
!	  the calculation is effectively repeated 9 times) - it depends on 
!	  how much time it takes to pack and repack the P1, P2 variables.
!
!	  Integer variables are set as INTEGER*4 type - several of them 
!	  could be INTEGER*2 or even INTEGER*1 for lower memory 
!	  requirements. Would this make casting to REAL*4 more time 
!     consuming or less?
!
!	  It is possible that we could add a logical variable, the same size
!	  as the VOXELS array, which carries a TRUE where the voxel 
!	  intensity in the corresponding place is nonzero. 
!	  Using that, we could reduce the computational demand of the line 
!	  to check the voxel intensities...
!	  		VALID_IND_2 = VOX_TEMP .GE. 0.0000001
!	  ...somewhat reduce the demand of the line to get VOX_TEMP out of 
!	  VOXELS...
!	  		VOX_TEMP(1:N_VALID) = VOXELS(INDICES(1:N_VALID))
!	  ... and eliminate the line which packs the VOX_TEMP array...
!			VOX_TEMP(1:N_WTS) = PACK(VOX_TEMP,VALID_IND_2)
!	  The penalty for this would be an additional variable in memory 
!	  (just under 100Mb, 0.1bn element logical*1) and the addition of a 
!	  line like...
!			MASK = VOXLOGIC(INDICES(1:N_VALID))
!	  ... to retrieve information from the logical array.
       character*120 line
       integer*4 k

!___________________________ DECLARATIONS ______________________________

!     Declare Input Arguments (unchanged by this subroutine)
	  INTEGER*4, INTENT(in) :: NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9, NVOX
	  INTEGER*4, INTENT(in) :: PIXCTR
	  LOGICAL*1, INTENT(in) :: PIXZERO
	  REAL*4,    INTENT(in) :: VOX_THRESH
      REAL*4,    INTENT(in) :: LOOKUP_WTS(227)

!     Declare Output Arguments (set or updated by this subroutine)
	  INTEGER*4, INTENT(out):: N_WTS
	  
!	  Declare Internal Arguments (used only in this subroutine)
	  INTEGER*4 :: ZCTR, OFF, N_VALID, DPC, LAMDACTR
	  INTEGER*4 :: X_C_IND(NVOX_Z), Y_C_IND(NVOX_Z)
	  INTEGER*4 :: X_INDS(NVOX_Z9), Y_INDS(NVOX_Z9), Z_INDS(NVOX_Z9)
      INTEGER*4, DIMENSION(:), ALLOCATABLE :: LOOKUP_IND

	  REAL*4 :: LAMDA(NVOX_Z), P1(NVOX_Z), P2(NVOX_Z)
	  REAL*4 :: P1_LARGE(NVOX_Z9), P2_LARGE(NVOX_Z9)
	  REAL*4 :: C_VALUE(3), D_VALUE(3)
	  REAL, DIMENSION(:), ALLOCATABLE :: DX, DY, DIST_SQD
	  LOGICAL*1 :: VALID_IND(NVOX_Z9), VALID_IND_2(NVOX_Z9)
	  
	  
	  INTEGER*4 :: PACKIND, PACKCTR, TEMPCTR
	  REAL*4 :: ZVEC(NVOX_Z)

		INTEGER*4, DIMENSION(:), ALLOCATABLE :: INDCHECK


!____________________ LoS INTERSECTION WITH Z PLANE ____________________

!	  Determine LoS of current pixel.
!		C_VALUE and D_VALUE are single precision triplets, in the 
!		voxel index coordinate frame (NOT in mm!) and are the 
!		coefficients of the line of sight of the current pixel.
	  C_VALUE = C(PIXCTR,:)
	  D_VALUE = D(PIXCTR,:)

!	  Making assumption that LoS goes from front to back plane 
!	  (i.e. camera angles all within about 25 degrees of the Z or -Z 
!	  direction) determine, at each plane of voxels, the intersection 
!	  position between the LoS and that plane.
	  DO ZCTR = 1,NVOX_Z
		ZVEC(ZCTR) = REAL(ZCTR)
	  ENDDO
!	  Now find the X,Y positions at which the pixel LoSs intersect 
!	  each Z plane
	  P1 = (ZVEC*D_VALUE(1)) + C_VALUE(1)
	  P2 = (ZVEC*D_VALUE(2)) + C_VALUE(2)

!	  NOTE: The specification of the D array (and D_VALUE) has changed.
!	  Originally, it was a normalised direction vector. Now, the 
!	  magnitude has significance: THe magnitude is that required for the
!	  D vector to connect one Z plane with another (i.e. D_VALUE(3) = 1 
!	  voxel). This allows us to simply multiply by the Z position at 
!	  each layer of voxels to determine the LoS line coefficient.
!	  The original code was as follows:
!	  				LAMDACTR = NVOX_Z
!	  					DO ZCTR = 1, NVOX_Z
!						LAMDA(ZCTR) = (ZCTR - C_VALUE(3)) / D_VALUE(3)
!						LAMDACTR = LAMDACTR - 1
!	  				ENDDO
!					P1 = (LAMDA*D_VALUE(1)) + C_VALUE(1)
!	  				P2 = (LAMDA*D_VALUE(2)) + C_VALUE(2)



!___________________ FIND INDICES OF AFFECTED VOXELS ___________________

! 	  Find the indices into the VOXELS array of the voxels which fall 
!	  within these 3x3 squares for the current pixel. 
!	  Note that C and D are already in the voxels indices frame (not in 
!	  absolute mm) so can simply be rounded:
!	  NOTE the order of these has been selected carefully, so that when
!	  the VOXELS array is accessed using these indices (in the main
!	  routine) there is a sequential recall of memory from the stored
!	  array (much quicker than nonsequential).
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
!	  the edge of the volume). Determine which ones:
!	  VALID_IND = (X_INDS.GT.0)
!	  VALID_IND = VALID_IND .AND. (Y_INDS.GT.0)
!	  VALID_IND = VALID_IND .AND. (X_INDS.LE.NVOX_X)
!      VALID_IND = VALID_IND .AND. (Y_INDS.LE.NVOX_Y)
!	  Strictly we should check in Z, too - but we've only assigned valid
!	  Z inds so save the computational effort

!     We need to pack all the valid indices into a consecutive locations
!	  in the indices arrays. The pack command is slow at this,
!	  (external c library - the calling process and internal checks 
!	  slow it down) so here we loop to do it ourselves:
	  PACKIND = 1
	  DO PACKCTR = 1,NVOX_Z9
!		IF (VALID_IND(PACKCTR)) THEN
		IF ( (X_INDS(PACKCTR).GT.0) .AND. (Y_INDS(PACKCTR).GT.0) .AND. &
			 (X_INDS(PACKCTR).LE.NVOX_X) .AND. 						   &
			 (Y_INDS(PACKCTR).LE.NVOX_Y) ) THEN
			X_INDS(PACKIND) = X_INDS(PACKCTR)
			Y_INDS(PACKIND) = Y_INDS(PACKCTR)
			Z_INDS(PACKIND) = Z_INDS(PACKCTR)
			P1_LARGE(PACKIND) = P1_LARGE(PACKCTR)
			P2_LARGE(PACKIND) = P2_LARGE(PACKCTR)
			PACKIND = PACKIND +1
		ENDIF
	  ENDDO
	  N_VALID = PACKIND-1
!	  The following code is a simplified version of the packing loop 
!	  above. It takes about 14% longer to execute according to vtune, 
!	  but the code is easier to understand (does the same thing):
!	  N_VALID = COUNT(VALID_IND)
!	  X_INDS(1:N_VALID)  = PACK(X_INDS,VALID_IND)
!	  Y_INDS(1:N_VALID)  = PACK(Y_INDS,VALID_IND)
!	  Z_INDS(1:N_VALID)  = PACK(Z_INDS,VALID_IND)
	  INDICES(1:N_VALID) = (NVOX_X*NVOX_Y)*(Z_INDS(1:N_VALID)-1) +     &
									NVOX_Y*(X_INDS(1:N_VALID)-1) + 	   &
									Y_INDS(1:N_VALID)



!________________ MASK WHERE VOXEL INTENSITY ALREADY = 0 _______________

!	  We next need to retrieve data from the large VOXELS array, which
!	  we will store in a smaller local variable, VOX_TEMP.
!	  Indexing in this fashion is notoriously inefficient if indices 
!	  are out of order. We have the advantage that the elements required
!	  are in ascending order of index, so we only have to cycle through 
!	  the VOXELS array once.
	  VOX_TEMP(1:N_VALID) = VOXELS(INDICES(1:N_VALID))


!	  Now, if voxels already contain an intensity of zero, we do not 
!	  want to calculate weightings for them. So we again mask and pack 
!	  the indices vector, this time along with the subindices vectors, 
!	  to reduce the number of weightings calculated. We also must update
!	  the corresponding N_WTS value.
	  PACKIND = 1
	  DO PACKCTR = 1,N_VALID
		IF (VOX_TEMP(PACKCTR) .GE. VOX_THRESH) THEN
	  		VOX_TEMP(PACKIND) = VOX_TEMP(PACKCTR)
	  		INDICES(PACKIND)  = INDICES(PACKCTR)
      		X_INDS(PACKIND)   = X_INDS(PACKCTR)
	  		Y_INDS(PACKIND)   = Y_INDS(PACKCTR)
			P1_LARGE(PACKIND) = P1_LARGE(PACKCTR)
			P2_LARGE(PACKIND) = P2_LARGE(PACKCTR)
			PACKIND = PACKIND+1
		ENDIF
	  ENDDO
			
!	  Clearer but slower code to do the same as above
!	  VALID_IND_2 = .FALSE.
!	  VALID_IND_2(1:N_VALID) = VOX_TEMP(1:N_VALID) .GE. 0.0000001
!	  N_WTS = COUNT(VALID_IND_2)
	  N_WTS = PACKIND - 1

! 	  Often, the number of weighted pixels is zero. Performing this 
!	  logical test reduces execution time
	  IF (N_WTS .EQ. 0) THEN 
		RETURN
	  ENDIF

!	  Possible slight speedup here by indexing the mask 1:N_VALID.
!	  VOX_TEMP(1:N_WTS) = PACK(VOX_TEMP,VALID_IND_2)
!	  INDICES(1:N_WTS)  = PACK(INDICES,VALID_IND_2)
!      X_INDS(1:N_WTS)   = PACK(X_INDS,VALID_IND_2)
!	  Y_INDS(1:N_WTS)   = PACK(Y_INDS,VALID_IND_2)

!	  If we're not shortcutting, or the pixel intensity is nonzero, 
!	  then we need to calculate the weights for these positions. There 
!	  are three steps to this:
!		1. Define a lookup table relating distance to weight
!		2. Calc distance between voxel centres and LoS intersection
!		3. Use lookup table to get weight from distance
!	  We have already defined a lookup table which is passed in to this 
!	  subroutine (no point redefining it every time this subroutine is 
!	  called).

!	  Note that we use distance in the voxel frame rather than in global
!	  mm - this saves us converting to the global frame. The distance 
!	  criterion was already modified in the initialisation stage of the 
!	  main routine to take this into account.
!	  Here, we copy out the P1 and P2 arrays to larger variables (same 
!	  size as the original indices array) then mask them in the same way
!	  that the INDICES array has been packed and re-packed.
!	  OFF = 1
!	  DO ZCTR = 1, NVOX_Z
!		P1_LARGE(OFF:OFF+8) = P1(ZCTR)
!		P2_LARGE(OFF:OFF+8) = P2(ZCTR)
!	    OFF = OFF+9
!	  ENDDO
!	  P1_LARGE(1:N_VALID) = PACK(P1_LARGE,VALID_IND)
!	  P2_LARGE(1:N_VALID) = PACK(P2_LARGE,VALID_IND)
!      P1_LARGE(1:N_WTS)   = PACK(P1_LARGE(1:N_VALID), VALID_IND_2(1:N_VALID))
!      P2_LARGE(1:N_WTS)   = PACK(P2_LARGE(1:N_VALID), VALID_IND_2(1:N_VALID))


!_____________ CALCULATE DISTANCE BETWEEN LOS AND EACH VOXEL ___________

!     Allocate temporary arrays to hold calculations
	  ALLOCATE(DX(N_WTS))
	  ALLOCATE(DY(N_WTS))
	  ALLOCATE(DIST_SQD(N_WTS))
	  ALLOCATE(LOOKUP_IND(N_WTS))

!	  Data typing from INTEGER to REAL is done implicitly in the 
!	  following expression, which determines the distance, in voxels, 
!	  in the X and Y directions between the voxels we care about and the
!	  intersection points. Note the small correction for the Lines of 
!     Sight not being perpendicular (10% change for a 25 degree angle)
	  DX = X_INDS(1:N_WTS) - P1_LARGE(1:N_WTS)
	  DY = Y_INDS(1:N_WTS) - P2_LARGE(1:N_WTS)
	  DX = DX*LOS_FACTOR(1,PIXCTR)
	  DY = DY*LOS_FACTOR(2,PIXCTR)
	  DIST_SQD = (DX**2) + (DY**2)

!	  If the code were ordered so that it was known which pixels came 
!	  from which camera, then a further simplifying assumption could be 
!	  made here - we could pass in the mean LoS factor of each camera, 
!	  then you're only passing in a few (NUMCAMS) values rather than an 
!	  entire distribution of LoS factors (one value for each pixel). 
!     The factors would be somewhat less accurate, but I suspect that the 
!	  computation is insensitive for LOS angles < 30 degrees due to the 
!	  COS term in the factor calculation.



!______________________ DETERMINE WEIGHTINGS (LOOKUP) __________________

!	  That gives us a distance^2 in voxel units. Now, for a 3x3 grid, 
!	  the maximum possible distance between a node of the grid and the 
!	  line intersection must be 3/sqrt(2) voxels. So, the maximum 
!	  distance squared is 4.5 voxels.
!	  The lookup table is designed to give a relationship 
!	  between distance and weighting. It is given for a range of values 
!	  of distance, where distance^2 varies between 0 and 4.52
!	  It has been designed to have 227 values, each corresponding to a 
!	  step of 0.02 in distance squared. So if we divide DIST_SQD by 
!	  0.02, then take the nearest integer value, we can directly derive 
!	  an index into the lookup table - saves us interpolating.
	  LOOKUP_IND = NINT(DIST_SQD/0.02) + 1
	  DIST_SQD = DIST_SQD/0.02

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



! End of subroutine (get_weights) - 92 lines of code
      END SUBROUTINE get_weights
	  END MODULE weights_mod
