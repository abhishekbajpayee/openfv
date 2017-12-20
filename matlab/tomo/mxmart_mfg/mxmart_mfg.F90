!MXMART_MFG.F90 Performs tomographic reconstruction using MFG+MART
!   Performs Multiplicative First Guess (MFG) then applies the MART
!   reconstruction algorithm.
!
!   See also mxmart_large for a non-MFG approach
!
!	This MEX file does not include rigorous internal checks. For safe 
!	operation, do not call directly, but use the mart_mfg() function, 
!	which is a MATLAB m-file designed specifically to call this mex 
!	file with rigorous input checking and type classification.
!
! Syntax:
!		[voxels] = mxmart_mfg(  numiters, pixels, nvox_X, ...
!								nvox_Y, nvox_Z, c, d, lookup_wts, ...
!								mu, los_factor, cam_offsets);
!
! Inputs:
!
!		numiters	[1 x 1]		int32, 0 < numiters
!								Number of iterations of the MART 
!								algorithm to perform (recommended 5). 
!								Note that application of the MFG 
!                               algorithm is excluded from this number.
!
!		pixels		[npix x 1]	Single precision, real, non-negative.
!								Contains the intensities of the pixels 
!								from which the reconstruction is made, 
!								in a column vector format.
!
!		nvox_X		[1 x 1]		int32, 0 < nvox_X
!								The number of voxels in the X direction 
!								of the cuboid reconstruction domain.
!
!		nvox_Y		[1 x 1]     int32, 0 < nvox_Y
!								The number of voxels in the Y direction 
!								of the cuboid reconstruction domain.
!
!		nvox_Z		[1 x 1]     int32, 0 < nvox_Z
!								The number of voxels in the Z direction 
!								of the cuboid reconstruction domain.
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
!		lookup_wts  [227 x 1] 	Single precision, real, non-negative
!								Contains the relationship between ds^2 
!								and weight, where ds is the distance 
!								between any voxel and the LoS of a given
!								pixel. The values of the relationship 
!								should be given in intervals of ds^2 
!								increasing by 0.02 from 0 to 4.52. 
!								Typically, the relationship will either 
!								be gaussian (between ds and weight) or 
!								be derived using a circle approximation 
!								- i.e. approximate a voxel to be a 
!								circle, and a pixel to be a circle of 
!								equal diameter, then determine the 
!								intersecting area between the two 
!								circles which are some radius ds apart.
!
!		mu 			[1 x 1]		real positive single, 0 < mu < 1
!								relaxation constant of the mart 
!								iteration (see ref [1]).
!
!		shortcut_flag [1 x 1]	int32, 0 <= shortcut_flag <= 1
!								Determines whether or not to take a 
!								shortcut during the first MART 
!								iteration. Typically, to reduce 
!								computational time during subsequent 
!								iterations, the first iteration zeroes 
!								all voxels which can be seen by any 
!								pixels whose intensity is zero. To do 
!								this, the entire weightings matrix must 
!								be calculated. 
!								To avoid such a hefty computation, this 
!								shortcut simply zeros the 9 voxels (on 
!								each Z=constant plane) which surround 
!								the LoS of each zero-intensity pixel. It
!								is recommended that this is turned off 
!								until validity of this approximation is 
!								studied somewhat.
!
!		los_factor	[2 x npix]	Single precision, real
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
!		cam_offsets [ncams]		int32
!								Contains an array of indices into the 
!								pixel, C and D arrays. Each value is the
!								index of the final pixel associated with
!								each different camera. So say the pixel
!								array contains 80 pixels from the first 
!								camera, 90 pixels from the second 
!								camera, and 100 pixels from the third:
!									cam_offsets = [80 170 270]
!								mxmart_premask can then access the
!								from a specific camera only.
!
! Outputs:
!		
!		voxels		[nvox x 1]	single
!								Reconstructed intensity of each voxel.
! 
! Other files required:   	mart_large.m (wrapper function)
!							fintrf.h (matlab MEX interface header file)
! Subroutines:              get_weights
! MAT-files required:       none
!
! Future Improvements:
!
!   1.  Alter to use -largeArrayDims flag in compilation (64 bit array 
!       dims). This is a _really_ annoying job.
!
!	2.  Utilise the fortran 95 interface to MATLAB (with some 
!       modification for single precision) to create a mxArray contining
!       the VOXELS variable, and point to it. Similarly we can avoid 
!       the copy overhead of the input variables.
!		
!	3.	L.o.S. Assumptions:
!		To reduce computational requirement, we only look at voxels in a
!		'zone' around the line of sight of each pixel. 
!		This zone is a 3x3 voxel square (at each level in Z). 
!		This causes problems for experimental setups where the voxel 
!		size is somewhat smaller than the pixel size (i.e. the pixel to 
!		voxel ratio is low, or a pixel 'sees' more than 9 voxels in 
!		each Z plane). We should require the pixel to voxel ratios of 
!		each camera to be passed in - then check them to ensure that the
!		assumption of a 3x3 square is valid. Note that Nick Worth has 
!		checked the assumption using a 5x5 and larger grids; and found 
!		little difference from using a 3x3.
!
!	SEE ALSO 'Future Improvements' in the get_weights subroutine.
!
! Platform Independance
!	mwpointer, mwsize and mwindex types have been used, along with the 
!	fintrf.h preprocessor directives file to ensure that this mex file 
!	compiles on both 32 and 64 bit systems, although there is some 
!   concern about compatibility when using 64 bit addressing 
!   (the -largeArrayDims flag). See improvement #1.
!
!	Apple-Mac compatibility: The Difference Between .f and .F Files.
!	Fortran compilers assume source files using a lowercase .f file 
!	extension have been preprocessed. On most platforms, mex makes sure 
!	the file is preprocessed regardless of the file extension. However, 
!	on Apple® Macintosh® platforms, mex cannot force preprocessing. Use 
!	an uppercase .F or .F90 file extension to ensure your Fortran 
!	MEX-file is platform independent.
!
!
! References:
!   [1] Elsinga G.E. Scarano F. Wienke B. and van Oudheusden B.W. (2006)
!       Tomographic Particle Image Velocimetry, Exp. Fluids 41:933-947
!
!	[2] Commentary by the authors of Numerical Recipes 3rd Edition
!		(Numerical Recipes for MATLAB) http://www.nr.com/nr3_matlab.html
!
!	[3]	Worth N.A. and Nickels T.B. (2008) 'Acceleration of Tomo-PIV by 
!		estimating the initial volume intensity distribution', 
!		Exp. Fluids, DOI 10.1007/s00348-008-0504-6
!
!   [4] Fortran 95 interface:
!       http://www.mathworks.co.uk/matlabcentral/fileexchange/25934-fortran-95-interface-to-matlab-api-with-extras
!
! Author:               T.H. Clark
! Work address:         Fluids / CFD Lab
!                       Cambridge University Engineering Department
!                       2 Trumpington Street
!                       Cambridge
!                       CB21PZ
! Email:                t.clark@cantab.net
! Website:              http://www2.eng.cam.ac.uk/~thc29
!
! Revision History:     01 June 2009        Created
!                       10 June 2009        Testing and comparison
!                                           + debug
!                       30 July 2011        Updated documentation, cut 
!                                           out a couple of lines that 
!                                           weren't used, checked that 
!                                           64 bit system compilation 
!                                           was possible.    
!_______________________________________________________________________




! ____________________________ INCLUSIONS ______________________________

! Inclusion of the preprocessor header file, allowing the fortran 
! compiler to correctly interpret what integer type mwSize, mwIndex and 
! mwPointer are (i.e. it automatically determines between integer*4 and 
! integer*8 depending on whether you're using 32 or 64 bit system. Thus
! using this helps make mex files platform-independant)
#include "fintrf.h"

! Inclusion of the weightings module. This module includes definition of
! a subroutine to determine weightings. It also declares allocatable 
! arrays which are shared between mexFunction and the get_weights 
! subroutine.
#include "mfg_mod.F90"


!_______________________________________________________________________
!
! GATEWAY FUNCTION FOR MEX FILE
!_______________________________________________________________________

      SUBROUTINE mexFunction(nlhs, plhs, nrhs, prhs)

! Use the module weights_mod (below). This contains a subroutine to 
! get weightings for voxels (given a pixel index). It also declares data 
! which is shared between the get_weights subroutine and the 
! mexFunction. This has a number of benefits:
!	  1. Explicit interface between mexFunction and get_weights gives 
!		 rise to better compile-time error detection
!	  2. Allows use of assumed-shape allocatable arrays which are shared 
!		 between the program units
!	  3. Potential speed improvement in the interface between the 
!		 program units, as compiler explicitly relates variables rather 
!	     than performing run-time checks.
	  USE mfg_mod

! Safer to have no implicit variables
      IMPLICIT NONE


 
! ______________________ GATEWAY DECLARATIONS __________________________

! Here we declare the mx function interfaces that we will use (e.g. to 
! interface with MATLAB). We'll need:
      
! 	  The number of left and right hand side arguments
      INTEGER*4 nlhs, nrhs

!	  Arrays containing pointers to the locations in memory of left and 
!	  right hand side arguments
      mwPointer plhs(*), prhs(*)

!	  MATLAB wrapper functions
	  mwPointer mxGetPr
	  mwPointer mxClassIDFromClassName
	  mwPointer mxCreateNumericMatrix
	  mwSize    mxGetM

!	  Internal variables used solely to administer the wrapper functions
      INTEGER*4 clsid
	  INTEGER*4 complexFlag
	  mwSize dims(2)



! ___________________ DECLARE ARGUMENT POINTERS ________________________

! Declare pointers to input arguments
      mwPointer xptrINITIAL, xptrNUMITERS, xptrPIXELS, xptrNVOX_X, 	   &
				xptrNVOX_Y, xptrNVOX_Z, xptrC, xptrD, xptrLOOKUP_WTS,  &
				xptrMU, xptrLOS_FACTOR, xptrCAM_OFFSETS,               &
                xptrPIX_THRESH

! Declare pointers to output arguments
      mwPointer yptrVOXELS
     


! _______________ DECLARE FORTRAN INTERNAL VARIABLES ___________________
      
! Variables/Arrays to hold input arguments:
	  INTEGER*4 :: NUMITERS, NVOX_X, NVOX_Y, NVOX_Z
	  INTEGER*4, DIMENSION(:), ALLOCATABLE :: CAM_OFFSETS
	  REAL*4    :: MU, PIX_THRESH, VOX_THRESH
	  REAL*4, DIMENSION(227) :: LOOKUP_WTS

! Internal Variables/Arrays (counters etc):
	  LOGICAL*1 :: PIXZERO, MFG_MODE
	  INTEGER*4 :: MARTCTR, PIXCTR, LINECTR, VOXCTR, NPIX, NVOX, 	   &
				    N_WTS, NVOX_Z9, CAMCTR, NCAMS
	  REAL*4    :: LINESUM

! Temporary debugging symbols
      character*120 :: line


! _______________________ OTHER DECLARATIONS ___________________________
!
! Note that some arrays have already been declared in the shared_weights
! module. These are:
!
!	  REAL*4, 	 DIMENSION(:), 	 ALLOCATABLE :: PIXELS
!	  REAL*4, 	 DIMENSION(:,:), ALLOCATABLE :: C, D, LOS_FACTOR
!	  REAL*4, 	 DIMENSION(:),   ALLOCATABLE :: VOXELS
!	  INTEGER*4, DIMENSION(:), 	 ALLOCATABLE :: INDICES 
!	  REAL*4,    DIMENSION(:), 	 ALLOCATABLE :: WEIGHTS, VOX_TEMP
! 	  LOGICAL*1, DIMENSION(:,:), ALLOCATABLE :: VOX_MASK




! __________________ GATEWAY CODE (INPUT ARGUMENTS) ____________________

! Get pointers to the input arguments out of the prhs array
      xptrNUMITERS      = mxGetPr(prhs(1))
      xptrPIXELS		= mxGetPr(prhs(2))
	  xptrNVOX_X		= mxGetPr(prhs(3))
	  xptrNVOX_Y     	= mxGetPr(prhs(4))
      xptrNVOX_Z     	= mxGetPr(prhs(5))
      xptrC   			= mxGetPr(prhs(6))
      xptrD         	= mxGetPr(prhs(7))
      xptrLOOKUP_WTS    = mxGetPr(prhs(8))
      xptrMU         	= mxGetPr(prhs(9))
	  xptrLOS_FACTOR    = mxGetPr(prhs(10))
	  xptrCAM_OFFSETS   = mxGetPr(prhs(11))
	  xptrPIX_THRESH    = mxGetPr(prhs(12))

! Make local copies of the input arguments
      CALL mxCopyPtrToInteger4(xptrNUMITERS,    NUMITERS,   1)
      CALL mxCopyPtrToInteger4(xptrNVOX_X,      NVOX_X,     1)
      CALL mxCopyPtrToInteger4(xptrNVOX_Y,      NVOX_Y,     1)
      CALL mxCopyPtrToInteger4(xptrNVOX_Z,      NVOX_Z,     1)
      CALL mxCopyPtrToReal4(xptrLOOKUP_WTS,     LOOKUP_WTS, 227)
      CALL mxCopyPtrToReal4(xptrMU,             MU,         1)
      CALL mxCopyPtrToReal4(xptrPIX_THRESH,     PIX_THRESH, 1)

! Determine sizes of input arguments where required
	  NPIX 	  = mxGetM(prhs(2))
	  NVOX 	  = NVOX_X*NVOX_Y*NVOX_Z
	  NVOX_Z9 = NVOX_Z*9
	  NCAMS   = mxGetM(prhs(11))

! Allocate arrays declared in the premask_mod module definition:
!	  Inputs
      ALLOCATE(PIXELS(NPIX))
      ALLOCATE(C(NPIX,3))
      ALLOCATE(D(NPIX,3))
      ALLOCATE(LOS_FACTOR(2,NPIX))
	  ALLOCATE(CAM_OFFSETS(NCAMS))
!	  Internal
	  ALLOCATE(INDICES(NVOX_Z9))
	  ALLOCATE(WEIGHTS(NVOX_Z9))
	  ALLOCATE(VOX_TEMP(NVOX_Z9))
!	  Outputs
	  ALLOCATE(VOXELS(NVOX))


! Copy input data to newly allocated shared arrays
      CALL mxCopyPtrToReal4(xptrPIXELS, PIXELS, NPIX)
      CALL mxCopyPtrToReal4(xptrC, C, NPIX*3)
      CALL mxCopyPtrToReal4(xptrD, D, NPIX*3)
      CALL mxCopyPtrToReal4(xptrLOS_FACTOR, LOS_FACTOR, 2*NPIX)
      CALL mxCopyPtrToInteger4(xptrCAM_OFFSETS, CAM_OFFSETS, NCAMS)




! ____________________ INITIALISE VARIABLES ____________________________

! 	  Set the initial guess of the voxel intensity. To use MFG, this 
!     should be set to zero.
	  VOXELS = 0.0
	  VOX_TEMP = 0.0
      INDICES = 0
      WEIGHTS = 0.0



! ____________________ MULTIPLICATIVE FIRST GUESS ______________________



	  CALL mfg( NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9, NCAMS, CAM_OFFSETS, &
				LOOKUP_WTS, PIX_THRESH)

!	  Turn off MFG_MODE for future calls to the weightings subroutine, 
!     to improve it's efficiency for MART
	  MFG_MODE = .FALSE.

! ______________________ MAKE MART CALCULATION _________________________


! For however many MART iterations you want
	  DO MARTCTR = 1,NUMITERS

! 		For each Pixel
		DO PIXCTR = 1,NPIX
			
!			For this pixel, we must get an array containing weightings 
!			between this pixel and each voxel. This represents a row of 
!			the wij matrix. We do not store an entire row, as this would
!			create another variable the size of the voxels array. 
!			Instead, we store it in a sparse form, with the indices 
!			vector containing column indices, and the weights vector 
!			containing the weightings which are at the corresponding 
!			columns in wij.
!
!			The output/updated variables contain:
!				N_WTS      [1x1] 	integer
!									Contains the number of non-zero 
!									weightings in wij(pixctr,:). i.e. 
!									this is the number of voxels which 
!									can be seen by this pixel**.
!
!				INDICES 	[Nx1] 	integer array
!									Contains the column indices of the 
!									non-zero elements in wij(pixctr,:).
!									Note that N_WTS <= N <= NVOX
!									The reasoning for this is explained 
!									in the get_weights subroutine. 
!									The key implication is that only the
!									first N_WTS elements of 
!									INDICES should be used in the MART 
!									calculation.
!									Note also that, unlike in the 
!									alternative mxmart routine, these 
!									indices are ONE-BASED, rather than 
!									zero-based.
!
!				WEIGHTS		[Nx1]	real (single) array 
!									Contains the nonzero weighting 
!									values from wij(pixctr,:). Note that
!									the size of WEIGHT is same size as 
!									INDICES for the same reason.
!!
!
!			Logical value, TRUE if pixel has zero intensity.
			PIXZERO = (PIXELS(PIXCTR) .LE. PIX_THRESH)

			IF (.NOT.(PIXZERO)) THEN

!				The current pixel has a non-zero intensity. Calculate 
!				the weightings and apply the MART algorithm

!				Calculate weightings
				CALL get_weight_mfg(NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9,  &
							   PIXCTR, LOOKUP_WTS, N_WTS, MFG_MODE)

!				Calculate line integral of vox_weight * vox_intensity,
!				down the line of sight of the current pixel
				LINESUM = 0;
				DO LINECTR = 1, N_WTS
					LINESUM = LINESUM + 							   &
							  (WEIGHTS(LINECTR) *  VOX_TEMP(LINECTR))
				ENDDO

! 				Loop for each Voxel. 
!				Note that in a simple expression of the problem we 
!				should have DO VOXCTR = 1,NVOX (i.e. apply the MART 
!				algorithm for all the voxels). For computational 
!				efficiency, we only apply the algorithm to the N_WTS 
!				affected voxels currently stored in VOX_TEMP. 
!				All other voxels: 
!				EITHER have zero weighting (thus the MART correction 
!						factor, raised to power 0, tends to unity) 
!				OR have zero intensity (i.e. can be 'seen' by a pixel 
!						with zero intensity, therefore do not contain 
!						a particle)
!				OR both,
!				THEREFORE do not need to be calculated

! 				Here, we rely on the correct conditioning of algorithm 
!				inputs: if entries in WEIGHTS are zero, OR if the
!				initial guess value is zero, then the linesum value may
!				be zero. This causes a /0 error in the voxel 
!				calculation. Such a situation is avoided implicitly in 
!				the correction for zero intensity voxels made during the
!				get_weights subroutine (see notes in the subroutine)

				DO VOXCTR = 1,N_WTS

!           		Apply MART algorithm to this pixel-voxel combination
!					  power = mart_mu*wij(i,j)
!					  factor = (intensity(j)/line_integral) ^ power
!  					  voxels(j) = voxels(j) * factor
					VOX_TEMP(VOXCTR) = VOX_TEMP(VOXCTR) *  &
     			  	( (PIXELS(PIXCTR)/LINESUM) ** (MU*WEIGHTS(VOXCTR)) )

!				End voxctr loop
				ENDDO

!				Store results back in the larger voxels array
				VOXELS(INDICES(1:N_WTS)) = VOX_TEMP(1:N_WTS)

!			End different behaviour for zero-intensity pixels
			ENDIF
	


!		End pixctr loop
		ENDDO
!	  End martctr loop
	  ENDDO


! _________________ GATEWAY CODE (OUTPUT ARGUMENTS) ____________________

! Create arrays (in the MATLAB workspace) for the return argument(s). 
      clsid = mxClassIDFromClassName('single')
      dims(1) = NVOX
      dims(2) = 1
	  complexFlag = 0
      plhs(1) = mxCreateNumericMatrix(dims(1), dims(2), &
									           clsid, complexFlag)

! Get output pointers for the return argument(s)
!  	  NB Call mxGetPr to determine the starting address of array to 
!	  which you wish to copy the output data
      yptrVOXELS = mxGetPr(plhs(1))

! Copy the output data 
!	  NB this copies from the fortran internal 'VOXELS' variable to the
!	  array in the MATLAB workspace which was created by the call to 
!	  mxCreateNumericMatrix, above.
      CALL mxCopyReal4ToPtr(VOXELS, yptrVOXELS, dims(1)) 

! You have the choice of copying your Fortran variable into the mxArray
! plhs variable, OR you can create your mxArray plhs variable up front 
! and use the %VAL() construct to pass the mxGetPr(plhs(1)) result to 
! your routine. That way the results automatically get stored in the 
! return plhs variable and you will avoid the extra unnecessary copy.
! This would, however, require you to lodge the main code into a 
! subroutine.



! __________________________ THE END GAME ______________________________    

! Deallocate variables which have been dynamically allocated here in 
! fortran (i.e. not by Matlab)
      DEALLOCATE (INDICES)
      DEALLOCATE (PIXELS)
      DEALLOCATE (WEIGHTS)
      DEALLOCATE (C)
      DEALLOCATE (D)
      DEALLOCATE (VOX_TEMP)
      DEALLOCATE (VOXELS)
      DEALLOCATE (LOS_FACTOR)


! End the main routine (mexFunction) - 97 lines of code
	  RETURN
      END SUBROUTINE mexFunction





