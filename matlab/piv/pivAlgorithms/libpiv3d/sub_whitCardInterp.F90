



!_______________________________________________________________________
!
!	  				SUBROUTINE whitCardInterp5
!
!	Extracts windows from larger intensity arrays. Image deformation is 
!   done using the Whittaker Cardinal (sinc) Reconstruction, i.e. input
!   window corners describe a hexahedron, but not necessaily a cuboid.
!
!   The Whittaker Reconstruction is a method for ensuring a bandlimited
!   interpolation (i.e. limited to the bandwidth of the original signal 
!   sampling) of data on a regular grid. The interpolated hypersurface 
!   always passes through the original sampled datapoints.
!
!   The technique is exposed very well by Ref [3] Stearns and Hush but
!   not so well by the other references, who use confusing notation. 
!   However, Ref [1] extends the technique to 2D (from Stearns' 1D) and 
!   Ref [2] provides a good exposition of the benefits of Whittaker 
!   reconstruction for use in image-deformed PIV.
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs are as follows:
!
!       WCTR            [1 x 1] integer *4
!                               Row index into the UX, UY, UZ, SNR 
!                               arrays (also shared in the module) for 
!                               the current window pair.
!
!       WSIZE           [1 x 1] integer *1
!                               dimension of the correlation cube array
!
! References:
!
!   [1] Lourenco L. and Krothapalli A. (1995) On the accuracy of velocity and
!       vorticity measurements with PIV. Experiments in Fluids 18 pp. 421-428
!
!   [2] Raffel M. Willert C. Wereley S and Kompenhans J. 
!		"Particle Image Velocimetry (A Practical Guide)" 
!		2nd ed. Springer, ISBN 978-3-540-72307-3
!
!   [3] Stearns S.D. and Hush D. (1990) Digital Signal Analysis. Second Edition.
!       Prentice-Hall pp. 75-84
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
! Revision History:     09 August 2011      Created from fMexPIVpar
!                       17 September 2011   Altered interface for 
!                                           libpiv3d.F90


!_______________________________________________________________________
!
!     DECLARATIONS AND INCLUSIONS
!_______________________________________________________________________


      SUBROUTINE whitCardInterp5( WCTR, WSIZE, NWINDOWS, &
                                    fd1, fd2, fd3,&
                        C0, C1, C2, C3, C4, C5, C6, C7, FIELD, WINDOW)


!     Safer to have no implicit variables
      IMPLICIT NONE

!     PI as a parameter
      REAL*4, PARAMETER :: pi = 3.141592653589793

!     CARDINAL CUBE SIZES (either 5,2 or 7,3)
      INTEGER*4, PARAMETER :: cardN = 5
      INTEGER*4, PARAMETER :: cardM = 2

!	  Inputs (explicit-shape dummy arrays)
      INTEGER*4, INTENT(in) :: WCTR, WSIZE, fd1, fd2, fd3, NWINDOWS
      REAL*4, DIMENSION(NWINDOWS, 3), INTENT(in)  :: C0, C1, C2, C3,&
                                                       C4, C5, C6, C7
	  REAL*4,   DIMENSION(fd1,fd2,fd3), INTENT(in) :: FIELD

!     Outputs
      REAL*4, DIMENSION(WSIZE, WSIZE, WSIZE), INTENT(out) :: WINDOW

!     Counters etc
      INTEGER*4 :: iElem, iRow, iCol, iPage, row, col, dep, NEl
      INTEGER*4 :: dbCtr1, dbCtr2, minR, minC, minP,                &
                                    nR,   nC,   nP,                  &
                                    maxR, maxC, maxP
      character*120 line

!     Local Automatic arrays containing rounded locations and terms
      REAL*4,    DIMENSION(WSIZE*WSIZE*WSIZE,3) :: pMap
      INTEGER*4, DIMENSION(WSIZE*WSIZE*WSIZE,3):: pMapRound
      REAL*4,    DIMENSION(WSIZE*WSIZE*WSIZE) :: tMapSinTerm


      

!     VARIABLES FOR SINGLE POINT CARDINAL INTERPOLATION KERNEL
      
!	  Declare inputs as explicit-size dummy arrays
      REAL*4, DIMENSION(1,3) :: tPoint
      REAL*4                  :: tSinOnPiCub
      REAL*4, DIMENSION(-cardM:cardM,-cardM:cardM,-cardM:cardM) ::f_t

!     Declare period, number of samples and index arrays, initialising
!     at compile time where possible for maximum speed at runtime
      INTEGER*4 :: i,       j,       k
      REAL*4    :: rMinusi, cMinusj, pMinusk
      REAL*4    :: m1powi,  m1powj,  m1powk
      REAL*4    :: rMult,   cMult,   pMult, cpMult
      REAL*4    :: fLoc
      REAL*4    :: ampl      
!     Outputs
      REAL*4  :: amplitude



!_______________________________________________________________________
!
!     RETRIEVE WINDOW ELEMENT INDICES
!_______________________________________________________________________


!     Update window element indices in list form (variable pMap in 
!     shared module)
      CALL windowCoords(WCTR, WSIZE, NWINDOWS, C0, C1, C2, C3,C4,C5,&
                                        C6, C7, pMap)


!_______________________________________________________________________
!
!     PREPROCESS AND SET-UP VARIABLES
!_______________________________________________________________________

!     Number of elements in the WSIZE^3 array
      NEl = WSIZE*WSIZE*WSIZE

!     Get their integer equivalents
      pMapRound = nint(pMap)

!     And their noninteger components (location in local 7x7x7
!     frame which is centred on pMapRound)
      pMap = pMap - real(pMapRound, kind = 4)

!     If the values of pMap are approaching zero, we must correct for 
!     the singularity which occurs (/0). We do this where values are 
!     below 10*eps('single')
      DO dbCtr2 = 1,3
        DO dbCtr1 = 1,NEl
            IF (ABS(pMap(dbCtr1,dbCtr2)).LE.0.0000011920929 ) THEN
                pMap(dbCtr1,dbCtr2) = 0.0000011920929
            ENDIF
        ENDDO
      ENDDO

!     We take the sin term outside of the cardinal interpolation to
!     improve processing speed. We get a speed advantage
!     by multiplying across rows and storing as a single column.
      tMapSinTerm(1:NEl) =  SIN(pi*pMap(1:NEl,1))   *   &
                            SIN(pi*pMap(1:NEl,2))   *   &
                            SIN(pi*pMap(1:NEl,3))   /   (pi**3)



!_______________________________________________________________________
!
!     INTERPOLATE FOR AMPLITUDE OF EACH WINDOW ELEMENT
!_______________________________________________________________________

!     For each element
      DO iPage = 1,WSIZE
        DO iCol = 1,WSIZE
            DO iRow = 1,WSIZE

!               Determine the subscript indices into the windows
                iElem = (iPage-1)*WSIZE*WSIZE + (iCol-1)*WSIZE + iRow

!               Retrieve the 5^3 array surrounding the interpolation point: 
!               Amplitude at sampling locations is contained in f_t
!               NB have to be careful here - if the rounded corner points are 
!               within 2 voxels of the edge of the array, we index out of bounds
!               and get a segmentation fault. Check for this in the calling code!
!                row = pMapRound(iElem,1) - minR + 1
!                col = pMapRound(iElem,2) - minC + 1
!                dep = pMapRound(iElem,3) - minP + 1
                row = pMapRound(iElem,1)
                col = pMapRound(iElem,2)
                dep = pMapRound(iElem,3)


!               Index the local array out of field
                f_t = FIELD(   row-cardM:row+cardM,    &
                               col-cardM:col+cardM,    &
                               dep-cardM:dep+cardM)

!_______________________________________________________________________
!
!               SINGLE POINT 3D CARDINAL INTERPOLATION
!
!               This used to be a subroutine - now forced inline to 
!               eliminate memory allocation overheads
!_______________________________________________________________________

                ! Get current values for this step of the loop
                tPoint(1,1:3) = pMap(iElem,1:3)
                tSinOnPiCub = tMapSinTerm(iElem)

!               Initialise amplitude
                ampl = 0.0

!               Loop for each of the 7x7x7 or 5x5x5 surrounding points
                DO k = -cardM,cardM
      
!                   Local position of interpolant
                    pMinusk = tPoint(1,3) - real(k, kind=4)
        
!                   Get the oscillating negation multiplier
                    m1powk = (-1)**k

!                   Get the multiplication factor
                    pMult = m1powk/pMinusk
        
                    DO j = -cardM,cardM
        
!                       Local position of interpolant
                        cMinusj = tPoint(1,2) - real(j, kind=4)
            
!                       Get the oscillating negation multiplier
                        m1powj = (-1)**j
            
!                       Get the multiplication factor
                        cMult = m1powj/cMinusj
                        cpMult = cMult*pMult
                
                        DO i = -cardM,cardM

!                           Local position of interpolant
                            rMinusi = tPoint(1,1) - real(i, kind=4)
    
!                           Get the oscillating negation multiplier
                            m1powi = (-1)**i
                
!                           Get the multiplication factor
                            rMult = m1powi/rMinusi

!                           Add the amplitude contribution
                            fLoc = f_t(i,j,k)
                            ampl = ampl + (rMult*cpMult*fLoc)

                        ENDDO !i
                    ENDDO !j
                ENDDO !k

!               Outside the loop, we multiply by the sin terms
                amplitude = tSinOnPiCub*ampl

!_______________________________________________________________________
!
!               END SINGLE POINT 3D CARDINAL INTERPOLATION
!_______________________________________________________________________



!               Store the interpolated result in WINDOW
                WINDOW(iRow, iCol, iPage) = amplitude
            

!               End the nested loops
            ENDDO
          ENDDO
      ENDDO



!	  End of subroutine
      RETURN
      END SUBROUTINE whitCardInterp5
 





!_______________________________________________________________________
!
!	  				SUBROUTINE whitCardInterp7
!
!	Extracts windows from larger intensity arrays. Image deformation is 
!   done using the Whittaker Cardinal (sinc) Reconstruction, i.e. input
!   window corners describe a hexahedron, but not necessaily a cuboid.
!
!   The Whittaker Reconstruction is a method for ensuring a bandlimited
!   interpolation (i.e. limited to the bandwidth of the original signal 
!   sampling) of data on a regular grid. The interpolated hypersurface 
!   always passes through the original sampled datapoints.
!
!   The technique is exposed very well by Ref [3] Stearns and Hush but
!   not so well by the other references, who use confusing notation. 
!   However, Ref [1] extends the technique to 2D (from Stearns' 1D) and 
!   Ref [2] provides a good exposition of the benefits of Whittaker 
!   reconstruction for use in image-deformed PIV.
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs are as follows:
!
!       WCTR            [1 x 1] integer *4
!                               Row index into the UX, UY, UZ, SNR 
!                               arrays (also shared in the module) for 
!                               the current window pair.
!
!       WSIZE           [1 x 1] integer *1
!                               dimension of the correlation cube array
!
! References:
!
!   [1] Lourenco L. and Krothapalli A. (1995) On the accuracy of velocity and
!       vorticity measurements with PIV. Experiments in Fluids 18 pp. 421-428
!
!   [2] Raffel M. Willert C. Wereley S and Kompenhans J. 
!		"Particle Image Velocimetry (A Practical Guide)" 
!		2nd ed. Springer, ISBN 978-3-540-72307-3
!
!   [3] Stearns S.D. and Hush D. (1990) Digital Signal Analysis. Second Edition.
!       Prentice-Hall pp. 75-84
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
! Revision History:     09 August 2011      Created from fMexPIVpar
!                       17 September 2011   Updated interface for
!                                           libpiv3d



!_______________________________________________________________________
!
!     DECLARATIONS AND INCLUSIONS
!_______________________________________________________________________


      SUBROUTINE whitCardInterp7( WCTR, WSIZE, NWINDOWS, &
                                    fd1, fd2, fd3,&
                        C0, C1, C2, C3, C4, C5, C6, C7, FIELD, WINDOW)


!     Safer to have no implicit variables
      IMPLICIT NONE

!     PI as a parameter
      REAL*4, PARAMETER :: pi = 3.141592653589793

!     CARDINAL CUBE SIZES (either 5,2 or 7,3)
      INTEGER*4, PARAMETER :: cardN = 7
      INTEGER*4, PARAMETER :: cardM = 3

!	  Inputs (explicit-shape dummy arrays)
      INTEGER*4, INTENT(in) :: WCTR, WSIZE, fd1, fd2, fd3, NWINDOWS
      REAL*4, DIMENSION(NWINDOWS, 3), INTENT(in) :: C0, C1, C2, C3,&
                                                       C4, C5, C6, C7
	  REAL*4,   DIMENSION(fd1,fd2,fd3), INTENT(in) :: FIELD

!     Outputs
      REAL*4, DIMENSION(WSIZE, WSIZE, WSIZE), INTENT(out) :: WINDOW

!     Counters etc
      INTEGER*4 :: iElem, iRow, iCol, iPage, row, col, dep, NEl
      INTEGER*4 :: dbCtr1, dbCtr2, minR, minC, minP,                &
                                    nR,   nC,   nP,                  &
                                    maxR, maxC, maxP
      character*120 line

!     Local Automatic arrays containing rounded locations and terms
      REAL*4,    DIMENSION(WSIZE*WSIZE*WSIZE,3) :: pMap
      INTEGER*4, DIMENSION(WSIZE*WSIZE*WSIZE,3):: pMapRound
      REAL*4,    DIMENSION(WSIZE*WSIZE*WSIZE) :: tMapSinTerm

      

!     VARIABLES FOR SINGLE POINT CARDINAL INTERPOLATION KERNEL
      
!	  Declare inputs as explicit-size dummy arrays
      REAL*4, DIMENSION(1,3) :: tPoint
      REAL*4                  :: tSinOnPiCub
      REAL*4, DIMENSION(-cardM:cardM,-cardM:cardM,-cardM:cardM) :: f_t

!     Declare period, number of samples and index arrays, initialising
!     at compile time where possible for maximum speed at runtime
      INTEGER*4 :: i,       j,       k
      REAL*4    :: rMinusi, cMinusj, pMinusk
      REAL*4    :: m1powi,  m1powj,  m1powk
      REAL*4    :: rMult,   cMult,   pMult, cpMult
      REAL*4    :: fLoc
      REAL*4    :: ampl      
!     Outputs
      REAL*4  :: amplitude



!_______________________________________________________________________
!
!     RETRIEVE WINDOW ELEMENT INDICES
!_______________________________________________________________________


!     Update window element indices in list form (variable pMap in 
!     shared module)
      CALL windowCoords(WCTR, WSIZE, NWINDOWS, C0, C1, C2, C3,C4,C5,&
                                        C6, C7, pMap)

!_______________________________________________________________________
!
!     PREPROCESS AND SET-UP VARIABLES
!_______________________________________________________________________

!     Number of elements in the WSIZE^3 array
      NEl = WSIZE*WSIZE*WSIZE

!     Get their integer equivalents
      pMapRound = nint(pMap)

!     And their noninteger components (location in local 7x7x7
!     frame which is centred on pMapRound)
      pMap = pMap - real(pMapRound, kind = 4)

!     If the values of pMap are approaching zero, we must correct for 
!     the singularity which occurs (/0). We do this where values are 
!     below 10*eps('single')
      DO dbCtr2 = 1,3
        DO dbCtr1 = 1,NEl
            IF (ABS(pMap(dbCtr1,dbCtr2)).LE.0.0000011920929 ) THEN
                pMap(dbCtr1,dbCtr2) = 0.0000011920929
            ENDIF
        ENDDO
      ENDDO

!     We take the sin term outside of the cardinal interpolation to
!     improve processing speed. We get a speed advantage
!     by multiplying across rows and storing as a single column.
      tMapSinTerm(1:NEl) =  SIN(pi*pMap(1:NEl,1))   *   &
                            SIN(pi*pMap(1:NEl,2))   *   &
                            SIN(pi*pMap(1:NEl,3))   /   (pi**3)



!_______________________________________________________________________
!
!     INTERPOLATE FOR AMPLITUDE OF EACH WINDOW ELEMENT
!_______________________________________________________________________

!     For each element
      DO iPage = 1,WSIZE
        DO iCol = 1,WSIZE
            DO iRow = 1,WSIZE

!               Determine the subscript indices into the windows
                iElem = (iPage-1)*WSIZE*WSIZE + (iCol-1)*WSIZE + iRow

!               Retrieve the 7^3 array surrounding the interpolation point: 
!               Amplitude at sampling locations is contained in f_t
!               NB have to be careful here - if the rounded corner points are 
!               within 3 voxels of the edge of the array, we index out of bounds
!               and get a segmentation fault. Check for this in the calling code!
!                row = pMapRound(iElem,1) - minR + 1
!                col = pMapRound(iElem,2) - minC + 1
!                dep = pMapRound(iElem,3) - minP + 1
                row = pMapRound(iElem,1)
                col = pMapRound(iElem,2)
                dep = pMapRound(iElem,3)


!               Index the local array out of field
                f_t = FIELD(   row-cardM:row+cardM,    &
                               col-cardM:col+cardM,    &
                               dep-cardM:dep+cardM)

!_______________________________________________________________________
!
!               SINGLE POINT 3D CARDINAL INTERPOLATION
!
!               This used to be a subroutine - now forced inline to 
!               eliminate memory allocation overheads
!_______________________________________________________________________

                ! Get current values for this step of the loop
                tPoint(1,1:3) = pMap(iElem,1:3)
                tSinOnPiCub = tMapSinTerm(iElem)

!               Initialise amplitude
                ampl = 0.0

!               Loop for each of the 7x7x7 or 5x5x5 surrounding points
                DO k = -cardM,cardM
      
!                   Local position of interpolant
                    pMinusk = tPoint(1,3) - real(k, kind=4)
        
!                   Get the oscillating negation multiplier
                    m1powk = (-1)**k

!                   Get the multiplication factor
                    pMult = m1powk/pMinusk
        
                    DO j = -cardM,cardM
        
!                       Local position of interpolant
                        cMinusj = tPoint(1,2) - real(j, kind=4)
            
!                       Get the oscillating negation multiplier
                        m1powj = (-1)**j
            
!                       Get the multiplication factor
                        cMult = m1powj/cMinusj
                        cpMult = cMult*pMult
                
                        DO i = -cardM,cardM

!                           Local position of interpolant
                            rMinusi = tPoint(1,1) - real(i, kind=4)
    
!                           Get the oscillating negation multiplier
                            m1powi = (-1)**i
                
!                           Get the multiplication factor
                            rMult = m1powi/rMinusi

!                           Add the amplitude contribution
                            fLoc = f_t(i,j,k)
                            ampl = ampl + (rMult*cpMult*fLoc)

                        ENDDO !i
                    ENDDO !j
                ENDDO !k

!               Outside the loop, we multiply by the sin terms
                amplitude = tSinOnPiCub*ampl

!_______________________________________________________________________
!
!               END SINGLE POINT 3D CARDINAL INTERPOLATION
!_______________________________________________________________________



!               Store the interpolated result in WINDOW
                WINDOW(iRow, iCol, iPage) = amplitude
            

!               End the nested loops
            ENDDO
          ENDDO
      ENDDO



!	  End of subroutine
      RETURN
      END SUBROUTINE whitCardInterp7
 







 
!_______________________________________________________________________
!
!	  				SUBROUTINE windowCoords
!
!	Returns coordinates of window elements in list form, derived from 
!   corner coordinates. 
!
!   This subroutine should be included in the source code for the 
!   WhitCardInterp subfunction, since it is called by that function and
!   uses variables in the fMexPIV module.
!
!	Inputs:
!       WCTR            [1 x 1] integer*4
!                               Index of the current window in the CX 
!                               arrays
!
!       WSIZE           [1 x 1] integer*4
!                               Assumption: Windows are cubes. WSIZE 
!                               contains the number of voxels in a 
!                               window in each direction. e.g for 
!                               WSIZE = 32, element positions are 
!                               returned for a 32x32x32 window.
!
!   Outputs:
!       pMap            [NELEMENTS x 3] real*4
!                               Positions (as noninterger index triplets
!                               into the FIELD arrays) of each element 
!                               (voxel) of the current deformed and 
!                               shifted window (which is specified by 
!                               the window corner locations). Note that
!                               the order of the for loops is such that 
!                               each column of the list is effectively 
!                               reshaped from a [row,col,dep] array.
!                ** NB pMap is already allocated in the module
!
! References:
!
!   [1] Lourenco L. and Krothapalli A. (1995) On the accuracy of velocity and
!       vorticity measurements with PIV. Experiments in Fluids 18 pp. 421-428
!
!   [2] Raffel M. Willert C. Wereley S and Kompenhans J. 
!		"Particle Image Velocimetry (A Practical Guide)" 
!		2nd ed. Springer, ISBN 978-3-540-72307-3
!
!   [3] Stearns S.D. and Hush D. (1990) Digital Signal Analysis. Second Edition.
!       Prentice-Hall pp. 75-84
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
! Revision History:     09 August 2011      Created from fMexPIVpar
!                       17 September 2011   Updated interface for
!                                           libpiv3d


!_______________________________________________________________________
!
!                   DECLARATIONS AND INCLUSIONS
!_______________________________________________________________________


      SUBROUTINE windowCoords(WCTR, WSIZE, NWINDOWS, &
                                C0, C1, C2, C3, C4, C5, C6, C7, pMap)


!     Safer to have no implicit variables
      IMPLICIT NONE

!     Inputs
      INTEGER*4, INTENT(in) :: WCTR, WSIZE, NWINDOWS
      REAL*4, DIMENSION(NWINDOWS,3), INTENT(in) :: C0, C1, C2, C3, &
                                                       C4, C5, C6, C7
!     Counters
      INTEGER*4 :: eleCtr, p, c, r

!     Outputs 
      REAL*4, DIMENSION(WSIZE*WSIZE*WSIZE,3), INTENT(out) :: pMap

!     Positions and vectors
      REAL*4, DIMENSION(1,3) :: f1, f2, f3,                         &
                                  p0, p1, p2, p3, p4, p5, p6, p7,   &
                                  p31, p32, p33, p34,               &
                                  p41, p42, pOut

!     Other
      REAL*4 :: realWSizeM1
      

!_______________________________________________________________________
!
!                   LINEAR INTERPOLATION FOR LOCATION OF EACH ELEMENT
!_______________________________________________________________________


!     Initialise element counter
      eleCtr = 1

!     Initialise pMap
      pMap = 0.0;

!     Extract window corner indices from shared arrays in the module
      p0(1,1:3) = C0(WCTR,1:3)
      p1(1,1:3) = C1(WCTR,1:3)
      p2(1,1:3) = C2(WCTR,1:3)
      p3(1,1:3) = C3(WCTR,1:3)
      p4(1,1:3) = C4(WCTR,1:3)
      p5(1,1:3) = C5(WCTR,1:3)
      p6(1,1:3) = C6(WCTR,1:3)
      p7(1,1:3) = C7(WCTR,1:3)
      
!     Precompute outside loops
      realWSizeM1 = real((WSIZE-1),kind=4)

!     Loop. Note the order chosen to put values into the output array in
!     row, column, page form.
      DO p = 1,WSIZE

!       Determine the fractional position in the x,y,z 
!       directions within the window (float, varies 
!       between 0 and 1). These are given as 3 element row 
!       vectors for use later
        f3(1,1) =  real((p-1), kind=4) / realWSizeM1   
        f3(1,2) =  real((p-1), kind=4) / realWSizeM1      
        f3(1,3) =  real((p-1), kind=4) / realWSizeM1

        DO c = 1,WSIZE

!           Determine the fractional position in the x,y,z 
!           directions within the window
            f1(1,1) =  real((c-1), kind=4) / realWSizeM1 
            f1(1,2) =  real((c-1), kind=4) / realWSizeM1        
            f1(1,3) =  real((c-1), kind=4) / realWSizeM1      

            DO r = 1,WSIZE

!               Determine the fractional position in the x,y,z 
!               directions within the window
                f2(1,1) =  real((r-1), kind=4) / realWSizeM1 
                f2(1,2) =  real((r-1), kind=4) / realWSizeM1        
                f2(1,3) =  real((r-1), kind=4) / realWSizeM1   

!               The fairly crude code above is a fast replacement for
!               the following commands:
!               Determine the fractional position in the x,y,z 
!               directions within the window (float, varies 
!               between 0 and 1). These are given as 3 element row 
!               vectors for use later
                !f1 = SPREAD([real((c-1), kind=4) / real((WSIZE-1), kind=4)], 2, 3)
                !f2 = SPREAD([real((r-1), kind=4) / real((WSIZE-1), kind=4)], 2, 3)
                !f3 = SPREAD([real((p-1), kind=4) / real((WSIZE-1), kind=4)], 2, 3)        



!               Using the fractional positions, we map the output 
!               position r,c,p to the row, column and page in the 
!               texture.

!               This requires knowledge of the ORDER in which corner 
!               indices are stored in the wCorner vectors. Order is as
!               follows:
!                 0       low row, low col, low page
!                 1       high row, low col, low page
!                 2       high row, high col, low page
!                 3       low row, high col, low page
!                 4       low row, low col, high page
!                 5       high row, low col, high page
!                 6       high row, high col, high page
!                 7       low row, high col, high page
!                 8       repeat for next window in same order
 
!               The photo 'windowOrdering.jpg' shows the diagram I've 
!               drawn to illustrate this (in the source code folder)

                p31 = f3*(p4 - p0) + p0
                p32 = f3*(p5 - p1) + p1
                p33 = f3*(p7 - p3) + p3
                p34 = f3*(p6 - p2) + p2

                p41  = f1*(p33 - p31) + p31
                p42  = f1*(p34 - p32) + p32
                pOut(1,1:3) = f2(1,1:3)*(p42(1,1:3) - p41(1,1:3)) + &
                                                            p41(1,1:3)

!               Output in Row, Column, Page index form
                pMap(eleCtr,1) = pOut(1,2)
                pMap(eleCtr,2) = pOut(1,1)
                pMap(eleCtr,3) = pOut(1,3)


!               Increment the counter
                eleCtr = eleCtr + 1;
            

            ENDDO !r
        ENDDO !c
      ENDDO !p

      


!	  End of subroutine
      RETURN
      END SUBROUTINE windowCoords


