!MXMART_LARGE.F90 Performs MART tomographic reconstruction 
!	mxmart_large.F90 should be compiled as a MEX file and used with 
!	MATLAB.
!
!	This is part of a measurement technique for fluid dynamics research; 
!	Tomographic Particle Image Velocimetry [ref 1].
!	Particle fields are captured with multiple cameras, reconstructed in 
!	3D using this routine, then cross-correlated to measure a fully 3D 
!	velocity field within a fluid region.
!
!   This is a companion function to mx_wrs, which uses a more complex 
!	process to reduce the size of the weightings matrix, and thereby 
!	avoid having to compute it several times. The benefit of that 
!	process is clear, as solution of the reconstruction problem can be 
!	made using least squares analysis. 
!	However, a comparison case which uses solely the MART approach is
!	highly useful, since to date, all studies of tomographic PIV have
!	the MART algorithm.
!
!	In addition, although we waste some effort re-computing 
!	weightings at each MART iteration, we have clearer code and the 
!	ability to handle computations without a spike in memory use 
!	during the least squares calculation - thus this approach is ideal 
!	for running in an explicitly parallel mode, or with fields which do
!	not have a high degree of sparsity (which would limit mx_wrs).
!
!	This MEX file does not include rigorous internal checks. For safe 
!	operation, do not call directly, but use the mart_large() function, 
!	which is a MATLAB m-file designed specifically to call this mex 
!	file with rigorous input checking and type classification.
!
! Syntax:  
!       [voxels] = mxmart_large(initial, numiters, pixels, nvox_X, ...
!								nvox_Y, nvox_Z, C, D, lookup_wts, ...
!								mu, los_factor)
!
!		Solves the classical Ax = b matrix problem...
!					w_sparse --> A
!					voxels   --> x
!					pixels   --> b
!		 ...using the Multiplicative Algebraic Reconstruction Technique.
!
! Inputs:
!
!		initial 	[1 x 1]		real positive single, 0 <= initial
!								Sets an initial guess for the intensity 
!								of the voxels. Suggest: 
!								mean_pixel_intensity * num_pix / num_vox
!
!
!		numiters	[1 x 1]		int32, 0 < numiters
!								Number of iterations of the MART 
!								algorithm to perform (recommended 5). 
!								The first iteration will take 
!								considerably longer than subsequent ones
!								as zero-level voxels are assigned.
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
!	1.  Alteration of the initial intensity to allow input of an array 
!		of size [NVOX x 1], which would contain a starting distribution
!		for voxel intensity. If a sensible guess can be made, this will 
!		accelerate convergence of the MART algorithm (see also #5).
!
!   2.  Maximum number of voxels which can be processed is 2,147,483,647
!		(the maximum value of NVOX as  a 32-bit integer*4). This could 
!		be increased by switching NVOX and associated counters etc to 
!		64 bit integers.
!
!	3.  Restructure to put the mart computation loop into a subroutine. 
!		The plhs pointer for the output voxels array could be created 
!		before the routine starts, and passed in using the %VAL 
!		construct. This eliminates the unnecessary copy of the VOXELS 
!		array to the output mxArray.
!
!	4.  Implementation of 'mxmart_basic' which throws out the 
!		assumptions and does a long but more certain calculation as a 
!		comparison case.
!
!	5.  Implementation of a Multiplicative First Guess (we're halfway 
!		there with the first iteration zeroing already) to speed up 
!		convergence time. This has some significant disadvantages in
!		that the process, although quicker, gives rise to a high number 
!		of 'ghost particles' in the field. The effect of ghost particles
!		on the cross-correlation is unclear.
!
!	6.  Pixel to voxel ratios. Currently, the PVR is forced to unity. 
!		This means that (say):
!			A pixel from camera 1 has an LoS which passes a distance S
!			from a voxel. The lookup table gives a pixel-voxel weighting
!			of W1. A second pixel, this time from a different camera 
!			(or a different region of camera 1's CCD) passes the same 
!			distance S from a voxel. Where the PVR is forced to 1, the 
!			weighting applied to the second pixel-voxel combination is 
!			equal to W1 (i.e. weighting is solely a function of distance
!			between LoS and voxel position)
!		The problem that occurs is where (for example) one camera is 
!		positioned oddly, or different camera types/resolutions are 
!		used. In such a case:
!			The pixel to voxel ratio can vary between cameras. Cameras 
!			with 'larger' pixels relative to the voxel size then tend to
!			contribute more to the intensity distribution (larger pixels
!			'capture' more light) than the other cameras. This biases 
!			the reconstruction toward the cameras with larger pixels.
!		Consequent Limitation:
!			For the assumption of PVR = 1 to hold, all cameras must be 
!			positioned (and lenses chosen etc) such that a distance on 
!			the image of one pixel represents a distance in real space 
!			of approximately the same amount. A good way of achieving 
!			this whilst setting up is to ensure that the same number of 
!			calibration dots appear on the calibration image from each 
!			camera.
!		
!	7.	L.o.S. Assumptions:
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
!		(see 'CHECKS' below for checks of correlatable distance 
!		explanation)
!
!	SEE ALSO 'Future Improvements' in the get_weights subroutine.
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
!	an uppercase .F or .FOR file extension to ensure your Fortran 
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
! Last revised:     10 June 2009
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
#include "weights_mod.F90"


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
	  USE weights_mod

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
				xptrMU, xptrNPIX, xptrNVOX, 		   &
				xptrLOS_FACTOR, xptrPIX_THRESH, xptrVOX_THRESH

! Declare pointers to output arguments
      mwPointer yptrVOXELS
     


! _______________ DECLARE FORTRAN INTERNAL VARIABLES ___________________
      
! Variables/Arrays to hold input arguments:
	  INTEGER*4 :: NUMITERS, NVOX_X, NVOX_Y, NVOX_Z
	  REAL*4    :: INITIAL, MU, PIX_THRESH, VOX_THRESH
	  REAL*4, DIMENSION(227) :: LOOKUP_WTS

! Internal Variables/Arrays (counters etc):
	  LOGICAL*1 :: PIXZERO
	  INTEGER*4 :: MARTCTR, PIXCTR, LINECTR, VOXCTR, NPIX, NVOX, N_WTS
	  INTEGER*4 :: NVOX_Z9
	  REAL*4    :: LINESUM



! _______________________ OTHER DECLARATIONS ___________________________
!
! Note that some arrays have already been declared in the shared_weights
! module. These are:
!
!		REAL*4, 	DIMENSION(:), 	ALLOCATABLE :: PIXELS
!		REAL*4, 	DIMENSION(:,:), ALLOCATABLE :: C, D, LOS_FACTOR
!		REAL*4, 	DIMENSION(:),   ALLOCATABLE :: VOXELS
!		INTEGER*4, 	DIMENSION(:), 	ALLOCATABLE :: INDICES 
!		REAL*4,    	DIMENSION(:), 	ALLOCATABLE :: VOX_TEMP, WEIGHTS



! __________________ GATEWAY CODE (INPUT ARGUMENTS) ____________________

! Get pointers to the input arguments out of the prhs array   
      xptrINITIAL       = mxGetPr(prhs(1))
      xptrNUMITERS      = mxGetPr(prhs(2))
      xptrPIXELS		= mxGetPr(prhs(3))
	  xptrNVOX_X		= mxGetPr(prhs(4))
	  xptrNVOX_Y     	= mxGetPr(prhs(5))
      xptrNVOX_Z     	= mxGetPr(prhs(6))
      xptrC   			= mxGetPr(prhs(7))
      xptrD         	= mxGetPr(prhs(8))
      xptrLOOKUP_WTS    = mxGetPr(prhs(9))
      xptrMU         	= mxGetPr(prhs(10))
	  xptrLOS_FACTOR    = mxGetPr(prhs(11))
	  xptrPIX_THRESH    = mxGetPr(prhs(12))
	  xptrVOX_THRESH    = mxGetPr(prhs(13))

! Make local copies of the input arguments
      CALL mxCopyPtrToInteger4(xptrNUMITERS, 	  NUMITERS, 	 1)
      CALL mxCopyPtrToInteger4(xptrNVOX_X, 		  NVOX_X, 		 1)
      CALL mxCopyPtrToInteger4(xptrNVOX_Y, 		  NVOX_Y, 		 1)
      CALL mxCopyPtrToInteger4(xptrNVOX_Z, 		  NVOX_Z, 		 1)
      CALL mxCopyPtrToReal4(   xptrINITIAL, 	  INITIAL,    	 1)
      CALL mxCopyPtrToReal4(   xptrLOOKUP_WTS,    LOOKUP_WTS, 	 227)
      CALL mxCopyPtrToReal4(   xptrMU, 	 		  MU, 	    	 1)
      CALL mxCopyPtrToReal4(   xptrPIX_THRESH, 	  PIX_THRESH, 	 1)
      CALL mxCopyPtrToReal4(   xptrVOX_THRESH, 	  VOX_THRESH, 	 1)

! Determine sizes of input arguments where required
	  NPIX 	  = mxGetM(prhs(3))
	  NVOX 	  = NVOX_X*NVOX_Y*NVOX_Z
	  NVOX_Z9 = NVOX_Z*9

! Allocate arrays declared in the shared_weights module:
!	  Inputs
      ALLOCATE(PIXELS(NPIX))
      ALLOCATE(C(NPIX,3))
      ALLOCATE(D(NPIX,3))
      ALLOCATE(LOS_FACTOR(2,NPIX))
!	  Internal (temporary)
	  ALLOCATE (INDICES(NVOX_Z9))
	  ALLOCATE (WEIGHTS(NVOX_Z9))
	  ALLOCATE (VOX_TEMP(NVOX_Z9))
!	  Outputs
	  ALLOCATE(VOXELS(NVOX))

! Copy input data to newly allocated shared arrays
      CALL mxCopyPtrToReal4(xptrPIXELS, PIXELS, NPIX)
      CALL mxCopyPtrToReal4(xptrC, C, NPIX*3)
      CALL mxCopyPtrToReal4(xptrD, D, NPIX*3)
      CALL mxCopyPtrToReal4(xptrLOS_FACTOR, LOS_FACTOR, 2*NPIX)



! _____________________________ CHECKS _________________________________      

! Check maximum correlatable distance (i.e. we assume that, in a plane 
! of constant Z, a pixel can 'see' a square of nine voxels around the 
! point where the LoS intersects the Z  plane. If this assumption is 
! invalid, then warn the user.
! Check also the minimum correlatable distance. If less than a voxel, it
! is possible that the initial iteration (where voxels are blanked based
! on an assumed weighting due to a shortcut in get_weights), then it is 
! possible we are blanking some voxels which actually contain weight. 
! We must eliminate the shortcut and do the full weights calculation.



! ____________________ INITIALISE VARIABLES ____________________________

! Set the initial guess of the voxel intensity.
	  VOXELS = INITIAL
	  


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
!			The implementation of get_weights is carefully considered 
!			(see notes in the heading of the subroutine itself). Here 
!			we simply note what the output/updated variables contain:
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
!
!			On the first MART iteration, we determine the weightings for
!			all the pixels. Where pixels have an intensity of zero, the 
!			weighted voxels for those pixels are assigned the value 0, 
!			and the MART algorithm is not applied.
!
!			On subsequent MART iterations, we only determine weightings 
!			for pixels with a non-zero intensity value. The weighted 
!			voxels for these non-zero pixels are then checked: Where 
!			their intensity is equal to zero, we know that a pixel other
!			than the current one can 'see' this voxel and has zero 
!			intensity. Thus we do not need to calculate the weightings 
!			for those voxels or assign a new value to them.
!
!			Logical value, TRUE if pixel has zero intensity.
			PIXZERO = (PIXELS(PIXCTR) .LE. PIX_THRESH)

			IF ((MARTCTR .EQ. 1).AND.(PIXZERO)) THEN

!				The current pixel has zero intensity, and we're in the 
!				first MART iteration. Determine which voxels are 'seen' 
!				by this pixel, but do not perform MART.
				CALL get_weights(NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9, NVOX,&
							 PIXCTR, PIXZERO, LOOKUP_WTS,    &
							 N_WTS, VOX_THRESH)

!				We now simply assign a zero value to those voxels which 
!				are weighted ('seen') by this pixel.
				VOXELS(INDICES(1:N_WTS)) = 0.0

			ELSEIF (.NOT.(PIXZERO)) THEN
				
!				The current pixel has a non-zero intensity. Calculate 
!				the weightings and apply the MART algorithm

!				Calculate weightings
				CALL get_weights(NVOX_X, NVOX_Y, NVOX_Z, NVOX_Z9, NVOX,&
							 PIXCTR, PIXZERO, LOOKUP_WTS,    &
							 N_WTS, VOX_THRESH)

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





