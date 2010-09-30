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


#include "Albany_VTK.hpp"

#ifdef HAVE_VTK
#include "vtkUnstructuredGrid.h"
#endif

using namespace std;

Albany_VTK::Albany_VTK (Teuchos::ParameterList& vtkParams) 
  : mPipelineUpdated (false), mCameraSet (false)
{
#ifdef HAVE_VTK
   vtkParams.validateParametersAndSetDefaults(*getValidVTKParameters(),0);

   mIterations = 0;

   mMPI = vtkMPIController::New ();
   mMPI->Initialize (0, 0, 1);

   // Construct the pipeline
   mSource = vtkSmartPointer<vtkITAPSUnstructuredSource>::New ();

   mWarp = vtkSmartPointer<vtkWarpScalar>::New ();
   mWarp->SetInputConnection (mSource->GetOutputPort ());
   mWarp->SetNormal (0, 0, -1);
   mscalefactor = vtkParams.get<double>("Scale Factor");
   mWarp->SetScaleFactor (mscalefactor);
   
   VTK_CREATE (vtkTransform, t);
   t->RotateX (vtkParams.get<double>("RotateX"));
   t->RotateZ (vtkParams.get<double>("RotateZ"));

   mCameraStatic = vtkParams.get<bool>("Static Camera");

   mTrans = vtkSmartPointer<vtkTransformFilter>::New ();
   mTrans->SetInputConnection (mWarp->GetOutputPort ());
   mTrans->SetTransform (t);

   mLUT = vtkSmartPointer<vtkColorTransferFunction>::New ();
   mLUT->AddRGBPoint (0.0, 0.1381, 0.2411, 0.7091);
   mLUT->AddRGBPoint (1.0, 0.6728, 0.1408, 0.1266);
   mLUT->SetColorSpaceToDiverging ();
   mLUT->SetVectorModeToMagnitude ();

   mMap = vtkSmartPointer<vtkDataSetMapper>::New ();
   mMap->SetInputConnection (mTrans->GetOutputPort ());
   mMap->SetLookupTable (mLUT);

   mActor = vtkSmartPointer<vtkActor>::New ();
   mActor->SetMapper (mMap);

   mBar = vtkSmartPointer<vtkScalarBarActor>::New ();
   mBar->SetLookupTable (mLUT);
   mBar->SetNumberOfLabels (6);

   mRM = vtkSmartPointer<vtkCompositeRenderManager>::New ();
   mRM->SetController (mMPI);

   mRenderer.TakeReference (mRM->MakeRenderer ());
   mRenderer->AddActor (mActor);
   mRenderer->AddActor (mBar);
   mRenderer->SetBackground (0,0,0);

   mLightKit = vtkSmartPointer<vtkLightKit>::New ();
   mRenderer->RemoveAllLights ();
   mLightKit->AddLightsToRenderer (mRenderer);

   mRenderWindow.TakeReference (mRM->MakeRenderWindow ());
   mRenderWindow->AddRenderer (mRenderer);
   mRenderWindow->SetSize (
     vtkParams.get<int>("Window X"),
     vtkParams.get<int>("Window Y"));

   mRM->SetRenderWindow (mRenderWindow);
   mRM->InitializePieces ();
   mRM->InitializeOffScreen ();

   mRenderWindow->OffScreenRenderingOn ();
   mRenderWindow->SwapBuffersOff ();

   if (vtkParams.get<bool>("Stagger Windows") == true) {
     int id = mMPI->GetLocalProcessId ();
     int numproc = mMPI->GetNumberOfProcesses ();
     int width = static_cast<int>(sqrtf (numproc));
     int x = id % width;
     int y = id / width;
     int *size = mRenderWindow->GetSize ();
     mRenderWindow->SetPosition (x * size[0], y * size[1]);
   }

   mWinToImage = vtkSmartPointer<vtkWindowToImageFilter>::New ();
   mWinToImage->SetInput (mRenderWindow);
   mWinToImage->ReadFrontBufferOff ();
   mWinToImage->ShouldRerenderOff ();

   mPNGWriter = vtkSmartPointer<vtkPNGWriter>::New ();
   mPNGWriter->SetInputConnection (mWinToImage->GetOutputPort ());

#endif /* HAVE_VTK */
}

Albany_VTK::~Albany_VTK ()
{
#ifdef HAVE_VTK
/*
   if (mMPI->GetLocalProcessId () == 0) {
      mMPEGWriter->End ();
   }
*/
   mMPI->Finalize (1);
   mMPI->Delete ();
#endif /* HAVE_VTK */
}

void Albany_VTK::updateGeometry (
                Teuchos::RCP<Albany::AbstractDiscretization> disc)
{
#ifdef HAVE_VTK
   iMesh_Instance instance = (iMesh_Instance) disc.get();
   mSource->SetMesh (instance);
   mRM->InitializePieces ();
   mPipelineUpdated = false;
#endif /* HAVE_VTK */
}

void Albany_VTK::updatePipeline ()
{
#ifdef HAVE_VTK
   mSource->Update ();
   vtkUnstructuredGrid *out = mSource->GetOutput ();

   out->ComputeBounds ();
   double *bounds = out->GetBounds ();
   double dx = bounds[1] - bounds[0];
   double dy = bounds[3] - bounds[2];

   if (dx == 0 && dy == 0) {
     cerr << "Both x and y axis are 0!\n";
   }
   else if (dx == 0 || dy == 0) {
     mGridToPoly = vtkSmartPointer<vtkGeometryFilter>::New ();
     mGridToPoly->SetInputConnection (mSource->GetOutputPort ());

     mExtrude = vtkSmartPointer<vtkLinearExtrusionFilter>::New ();
     mExtrude->SetInputConnection (mGridToPoly->GetOutputPort ());
     mExtrude->SetExtrusionTypeToVectorExtrusion ();
     if (dx == 0) {
       mExtrude->SetVector (1, 0, 0);
       mExtrude->SetScaleFactor (mscalefactor*dy);
     }
     else {
       mExtrude->SetVector (0, 1, 0);
       mExtrude->SetScaleFactor (mscalefactor*dx);
     }
     mWarp->SetInputConnection (mExtrude->GetOutputPort ());
  }
  else {
     mWarp->SetInputConnection (mSource->GetOutputPort ());
  }
#endif
}

void Albany_VTK::visualizeField (const Epetra_Vector& soln,
                Teuchos::RCP<Albany::AbstractDiscretization> disc)
{
#ifdef HAVE_VTK
   
 
   iMesh_Instance instance = (iMesh_Instance) disc.get();

   // Scatter soln vector to overlap map -- which is element based
   Teuchos::RCP<Epetra_Vector> soln_overlap =
     Teuchos::rcp( new Epetra_Vector(*(disc->getOverlapMap())) );
   Epetra_Import importer( *(disc->getOverlapMap()), soln.Map());
   soln_overlap->Import(soln, importer, Insert);
   
   soln_overlap->Scale(mscalefactor);
   iMesh_setFields(soln_overlap);

   mSource->Modified ();

   if (mPipelineUpdated == false) {
      updatePipeline ();
      mPipelineUpdated = true;
   }

#if 0
     VTK_CREATE (vtkRenderer, renderer);
     renderer->AddActor (mActor);
     renderer->AddActor (mBar);
     renderer->SetBackground (0,0,0);
     renderer->ResetCamera ();

     VTK_CREATE (vtkRenderWindow, renderWindow);
     renderWindow->AddRenderer (renderer);

     vtkCamera *cam = renderer->GetActiveCamera ();
     cam->UpdateViewport (renderer);
     renderWindow->Render ();

     VTK_CREATE (vtkWindowToImageFilter, winToImage);
     winToImage->SetInput (renderWindow);
     winToImage->ReadFrontBufferOff ();
     winToImage->ShouldRerenderOff ();
     winToImage->Update ();

     VTK_CREATE (vtkPNGWriter, png);
     char filename[64];
     sprintf (filename, "vis%04d.%04d.png", mIterations, mMPI->GetLocalProcessId ());
     png->SetFileName (filename);
     png->SetInputConnection (winToImage->GetOutputPort ());
     png->Write ();
#endif

   if (mMPI->GetLocalProcessId () == 0) {
     if (!mCameraSet || !mCameraStatic) {
       mRM->ResetAllCameras ();
       mCameraSet = true;
     } 
     vtkCamera *cam = mRenderer->GetActiveCamera ();
     //cam->UpdateViewport (mRenderer);

     mRenderWindow->Render ();
     mWinToImage->Modified ();
     // mMPEGWriter->Write ();

     char filename[64];
     sprintf (filename, "vis%04d.png", mIterations);
     mPNGWriter->SetFileName (filename);
     mPNGWriter->Write ();

     mRM->StopServices ();
   } 
   else {
     mRM->StartServices ();
   }

   mIterations ++;
#endif /* HAVE_VTK */
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany_VTK::getValidVTKParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     rcp(new Teuchos::ParameterList("ValidVTKParams"));;

  validPL->set<bool>("Do Visualization", false);
  validPL->set<bool>("Visualize Sensitivities", false);
  validPL->set<double>("Scale Factor", 1.0);
  validPL->set<double>("RotateX", 0.0);
  validPL->set<double>("RotateZ", 0.0);
  validPL->set<int>("Window X", 256);
  validPL->set<int>("Window Y", 256);
  validPL->set<bool>("Static Camera", true);
  validPL->set<bool>("Stagger Windows", false);

  return validPL;
}

