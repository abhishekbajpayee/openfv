



!_______________________________________________________________________
!
!	  				SUBROUTINE indexedFetch
!
!	Extracts windows from larger intensity arrays. No image deformation 
!   is done. Input window corner indices must be integer values and 
!   describe a cube array of the same size as the windows.
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs are as defined in vodim3d.F90
!
! Definitions
!
! References:
!   [1] Raffel M. Willert C. Wereley S and Kompenhans J. 
!		"Particle Image Velocimetry (A Practical Guide)" 
!		2nd ed. Springer, ISBN 978-3-540-72307-3
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
! Revision History:     16 July 2011        Created from WhitCardInterp
!                       09 August 2011      Updated to work with the 
!                                           parallel implementation
!                       17 September 2011   Altered interface for
!                                           libpiv3d

! NOTE***************
!   THIS ROUTINE REQUIRES UPDATING TO FETCH WINDOWS BASED ON THE C0 LOCATION ONLY AND TO PROPERLY CHECK BOUNDS



!_______________________________________________________________________
!
!                   DECLARATIONS AND INCLUSIONS
!_______________________________________________________________________

      SUBROUTINE indexedFetch( WCTR, WSIZE, NWINDOWS, fd1, fd2, fd3, C0, C1, C2, C3, C4, C5, C6, C7, FIELD, WINDOW)

!     Safer to have no implicit variables
      IMPLICIT NONE

!	  Declare inputs as explicit-shape dummy arrays
      INTEGER*4, INTENT(in) :: WCTR, WSIZE, fd1, fd2, fd3, NWINDOWS
      REAL*4, DIMENSION(NWINDOWS, 3), INTENT(in)  :: C0, C1, C2, C3,&
                                                       C4, C5, C6, C7
	  REAL*4, DIMENSION(fd1, fd2, fd3), INTENT(in) :: FIELD

!     Outpupts
      REAL*4, DIMENSION(WSIZE, WSIZE, WSIZE), INTENT(out) :: WINDOW

!     Internals
      INTEGER*4 :: rowMin, rowMax, colMin, colMax, pageMin, pageMax
      INTEGER*4, DIMENSION(WSIZE*WSIZE*WSIZE,3):: pMapRound
      REAL*4,    DIMENSION(WSIZE*WSIZE*WSIZE,3) :: pMap
      character*120 line
!_______________________________________________________________________
!
!                   RETRIEVE WINDOW ELEMENT INDICES
!_______________________________________________________________________


!     Update window element indices in list form (variable pMap in 
!     shared module)
      CALL windowCoords(WCTR, WSIZE, NWINDOWS, C0, C1, C2, C3, C4, C5, &
                                        C6, C7, pMap)

!     Get their integer equivalents. NB they're input as reals to be 
!     compatible with the other window fetch functions (which deform, 
!     therefore require noninteger locations). NB Windows can be the 
!     correct WSIZE but input as running from (say) 1.5:64.5. To allow 
!     for that, we round, always flooring the value:
      pMapRound = nint(pMap)

!     Get the cube ordinates
      rowMin  = MINVAL(pMapRound(:,1))
      rowMax  = MAXVAL(pMapRound(:,1))
      colMin  = MINVAL(pMapRound(:,2))
      colMax  = MAXVAL(pMapRound(:,2))
      pageMin = MINVAL(pMapRound(:,3))
      pageMax = MAXVAL(pMapRound(:,3))

!     Checks on the cube ordinates
!      IF (rowMin.LE.0) THEN
!        write(line,*) 'rowMin = ', rowMin
!        CALL mexPrintf(line//achar(13))
!        CALL FreeAll
!        CALL mexErrMsgTxt('Window fetch ordinates out of bounds')
!      ENDIF
!      IF (colMin.LE.0) THEN
!        write(line,*) 'colMin = ', colMin
!        CALL mexPrintf(line//achar(13))
!        CALL FreeAll
!        CALL mexErrMsgTxt('Window fetch ordinates out of bounds')
!      ENDIF
!      IF (pageMin.LE.0) THEN
!        write(line,*) 'pageMin = ', pageMin
!        CALL mexPrintf(line//achar(13))
!        CALL FreeAll
!        CALL mexErrMsgTxt('Window fetch ordinates out of bounds')
!      ENDIF
!      IF ((rowMax - rowMin + 1).NE.WSIZE) THEN
!        write(line,*) 'rowMin = ', rowMin, ',     rowMax = ', rowMax
!        CALL mexPrintf(line//achar(13))
!        CALL FreeAll
!        CALL mexErrMsgTxt('Row-wise window size does not match WSIZE parameter')
!      ENDIF
!      IF ((colMax - colMin + 1).NE.WSIZE) THEN
!        write(line,*) 'colMin = ', colMin, ',     colMax = ', colMax
!        CALL mexPrintf(line//achar(13))
!        CALL FreeAll
!        CALL mexErrMsgTxt('Column-wise window size does not match WSIZE parameter')
!      ENDIF
!      IF ((pageMax - pageMin + 1).NE.WSIZE) THEN
!        write(line,*) 'pageMin = ', pageMin, ',     pageMax = ', pageMax
!        CALL mexPrintf(line//achar(13))
!        CALL FreeAll
!        CALL mexErrMsgTxt('Page-wise window size does not match WSIZE parameter')
!      ENDIF


!_______________________________________________________________________
!
!                   FETCH AMPLITUDE OF EACH WINDOW ELEMENT
!_______________________________________________________________________


!     Index the local array out of field
      WINDOW = FIELD(rowMin:rowMax,colMin:colMax, pageMin:pageMax)



!	  End of subroutine
      END SUBROUTINE indexedFetch
 
 


