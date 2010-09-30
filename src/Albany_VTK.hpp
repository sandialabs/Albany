/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_VTK
#define ALBANY_VTK

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"
#include "Albany_AbstractDiscretization.hpp"

#ifdef HAVE_VTK

#include "iMesh_interface.hpp"

#include "vtkITAPSUnstructuredSource.hpp"
#include <vtkMPIController.h>
#include <vtkGeometryFilter.h>
#include <vtkLinearExtrusionFilter.h>
#include <vtkWarpScalar.h>
#include <vtkTransform.h>
#include <vtkTransformFilter.h>
#include <vtkColorTransferFunction.h>
#include <vtkDataSetMapper.h>
#include <vtkActor.h>
#include <vtkScalarBarActor.h>
#include <vtkCompositeRenderManager.h>
#include <vtkRenderer.h>
#include <vtkLightKit.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCamera.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkMPEG2Writer.h>

#include <vtkSmartPointer.h>
#define VTK_CREATE(type, name) \
  vtkSmartPointer<type> name = vtkSmartPointer<type>::New ();

#endif /* HAVE_VTK */

class Albany_VTK
{
public:
   Albany_VTK ();
   Albany_VTK (Teuchos::ParameterList& vtkParams);
   ~Albany_VTK ();

   void updateGeometry (Teuchos::RCP<Albany::AbstractDiscretization> disc);
   void updatePipeline ();
   void visualizeField (const Epetra_Vector& soln,
                        Teuchos::RCP<Albany::AbstractDiscretization> disc);
private:
   Teuchos::RCP<const Teuchos::ParameterList> getValidVTKParameters() const;

   bool mPipelineUpdated;
   bool mCameraSet;
   bool mCameraStatic;
   int mIterations;
   double mscalefactor;

#ifdef HAVE_VTK
   vtkMPIController* mMPI;
   vtkSmartPointer<vtkITAPSUnstructuredSource> mSource;
   vtkSmartPointer<vtkWarpScalar> mWarp;
   vtkSmartPointer<vtkGeometryFilter> mGridToPoly;
   vtkSmartPointer<vtkLinearExtrusionFilter> mExtrude;
   vtkSmartPointer<vtkTransformFilter> mTrans;
   vtkSmartPointer<vtkColorTransferFunction> mLUT;
   vtkSmartPointer<vtkDataSetMapper> mMap;
   vtkSmartPointer<vtkActor> mActor;
   vtkSmartPointer<vtkScalarBarActor> mBar;
   vtkSmartPointer<vtkCompositeRenderManager> mRM;
   vtkSmartPointer<vtkRenderer> mRenderer;
   vtkSmartPointer<vtkLightKit> mLightKit;
   vtkSmartPointer<vtkRenderWindow> mRenderWindow;
   vtkSmartPointer<vtkWindowToImageFilter> mWinToImage;
   vtkSmartPointer<vtkMPEG2Writer> mMPEGWriter;
   vtkSmartPointer<vtkPNGWriter> mPNGWriter;
#endif
   Albany_VTK (const Albany_VTK&);
   void operator= (const Albany_VTK&);
};

/*
void Albany_VTK(const Epetra_Vector& soln,
                Teuchos::RCP<Albany::AbstractDiscretization> disc,
                Teuchos::ParameterList& vtkParams);
*/

#endif //ALBANY_VTK
