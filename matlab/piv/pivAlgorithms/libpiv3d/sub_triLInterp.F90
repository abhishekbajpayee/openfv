



!_______________________________________________________________________
!
!	  				SUBROUTINE triLInterp
!
!	Extracts windows from larger intensity arrays using trilinear 
!   interpolation
!
!   This subroutine should be included in the module fMexPIV_mod.F90, 
!   since it uses arrays declared in that module.
!
!	Inputs are as defined in vodim3d.F90
!
! Definitions
!
! References:
!
!   [2] Raffel M. Willert C. Wereley S and Kompenhans J. 
!		"Particle Image Velocimetry (A Practical Guide)" 
!		2nd ed. Springer, ISBN 978-3-540-72307-3
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
!                       09 August 2011      Updated for parallel 
!                                           implementation
!                       17 September 2011   Updated interface for
!                                           libpiv3d



!_______________________________________________________________________
!
!                   DECLARATIONS AND INCLUSIONS
!_______________________________________________________________________

      SUBROUTINE triLInterp( WCTR, WSIZE, NWINDOWS, fd1, fd2, fd3, &
                         C0, C1, C2, C3, C4, C5, C6, C7, FIELD, WINDOW)


!     Safer to have no implicit variables
      IMPLICIT NONE

!	  Declare inputs as explicit-shape dummy arrays
      INTEGER*4, INTENT(in) :: WCTR, WSIZE, fd1, fd2, fd3, NWINDOWS
      REAL*4, DIMENSION(NWINDOWS, 3), INTENT(in)  :: C0, C1, C2, C3,&
                                                       C4, C5, C6, C7
	  REAL*4, DIMENSION(fd1, fd2, fd3), INTENT(in) :: FIELD

!     Outpupts
      REAL*4, DIMENSION(WSIZE, WSIZE, WSIZE), INTENT(out) :: WINDOW

!     Counters etc
      INTEGER*4 :: NEl, rOff, cOff, pOff
      INTEGER*4, DIMENSION(1) :: vecShape
      INTEGER*4, DIMENSION(3) :: winShape
      INTEGER*4, DIMENSION(WSIZE*WSIZE*WSIZE,3):: pMapRound
      REAL*4,    DIMENSION(WSIZE, WSIZE, WSIZE) :: VCORNER
      REAL*4,    DIMENSION(WSIZE*WSIZE*WSIZE,3) :: pMap


!_______________________________________________________________________
!
!                   RETRIEVE WINDOW ELEMENT INDICES
!_______________________________________________________________________


!     Update window element indices in list form (variable pMap in 
!     shared module)
      CALL windowCoords(WCTR, WSIZE, NWINDOWS, C0, C1, C2, C3, C4, C5, &
                                        C6, C7, pMap)

!_______________________________________________________________________
!
!                   PREPROCESS AND SET-UP VARIABLES
!_______________________________________________________________________

!     Number of elements in the WSIZE^3 array
      NEl = WSIZE*WSIZE*WSIZE

!     Get the lower integer equivalents
      pMapRound = floor(pMap)

!     And their noninteger components
      pMap = pMap - real(pMapRound, kind = 4)

!     Shapes of things ready for reshaping!
      vecShape = shape(pMap(1:NEl,1))
      winShape = shape(WINDOW)

!_______________________________________________________________________
!
!                   INTERPOLATE FOR AMPLITUDE OF EACH WINDOW ELEMENT
!_______________________________________________________________________


!       V000
        rOff = 0
        cOff = 0
        pOff = 0
        VCORNER = FIELD(pMapRound(:,1)+rOff,&
                        pMapRound(:,2)+cOff,&
                        pMapRound(:,3)+pOff)
        WINDOW = reshape( reshape(VCORNER, vecShape) * (1.-pMap(:,1)) &
                            * (1.-pMap(:,2)) * (1.-pMap(:,3)), winShape)

!       V100
        rOff = 1
        cOff = 0
        pOff = 0
        VCORNER = FIELD(pMapRound(:,1)+rOff,&
                        pMapRound(:,2)+cOff,&
                        pMapRound(:,3)+pOff)
        WINDOW = WINDOW + reshape( reshape(VCORNER, vecShape) * &
                pMap(:,1) * (1.-pMap(:,2)) * (1.-pMap(:,3)), winShape)
      
!       V010
        rOff = 0
        cOff = 1
        pOff = 0
        VCORNER = FIELD(pMapRound(:,1)+rOff,&
                        pMapRound(:,2)+cOff,&
                        pMapRound(:,3)+pOff)
        WINDOW = WINDOW + reshape( reshape(VCORNER, vecShape) * &
                (1.-pMap(:,1)) * pMap(:,2) * (1.-pMap(:,3)), winShape)

!       V110
        rOff = 1
        cOff = 1
        pOff = 0
        VCORNER = FIELD(pMapRound(:,1)+rOff,&
                        pMapRound(:,2)+cOff,&
                        pMapRound(:,3)+pOff)
        WINDOW = WINDOW + reshape( reshape(VCORNER, vecShape) * &
                    pMap(:,1) * pMap(:,2) * (1.-pMap(:,3)), winShape)

!       V001
        rOff = 0
        cOff = 0
        pOff = 1
        VCORNER = FIELD(pMapRound(:,1)+rOff,&
                        pMapRound(:,2)+cOff,&
                        pMapRound(:,3)+pOff)
        WINDOW = WINDOW + reshape( reshape(VCORNER, vecShape) * &
                (1.-pMap(:,1)) * (1.-pMap(:,2)) * pMap(:,3), winShape)

!       V101
        rOff = 1
        cOff = 0
        pOff = 1
        VCORNER = FIELD(pMapRound(:,1)+rOff,&
                        pMapRound(:,2)+cOff,&
                        pMapRound(:,3)+pOff)
        WINDOW = WINDOW + reshape( reshape(VCORNER, vecShape) * &
                    pMap(:,1) * (1.-pMap(:,2)) * pMap(:,3), winShape)

!       V011
        rOff = 0
        cOff = 1
        pOff = 1
        VCORNER = FIELD(pMapRound(:,1)+rOff,&
                        pMapRound(:,2)+cOff,&
                        pMapRound(:,3)+pOff)
        WINDOW = WINDOW + reshape( reshape(VCORNER, vecShape) * &
                    (1.-pMap(:,1)) * pMap(:,2) * pMap(:,3), winShape)

!       V111
        rOff = 1
        cOff = 1
        pOff = 1
        VCORNER = FIELD(pMapRound(:,1)+rOff,&
                        pMapRound(:,2)+cOff,&
                        pMapRound(:,3)+pOff)
        WINDOW = WINDOW + reshape( reshape(VCORNER, vecShape) * &
                        pMap(:,1) * pMap(:,2) * pMap(:,3), winShape)



!	  End of subroutine
      END SUBROUTINE triLInterp


 
