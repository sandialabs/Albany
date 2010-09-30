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


#ifndef __vtkITAPSUnstructuredSource_h
#define __vtkITAPSUnstructuredSource_h

#ifdef HAVE_VTK

#include <vtkUnstructuredGridAlgorithm.h>

#include "iMesh_interface.hpp"

class vtkITAPSUnstructuredSource : public vtkUnstructuredGridAlgorithm
{
public:
  static vtkITAPSUnstructuredSource* New ();
  vtkTypeRevisionMacro (vtkITAPSUnstructuredSource, vtkUnstructuredGridAlgorithm);
  void PrintSelf (ostream& os, vtkIndent indent);

  // Description:
  // Specify the Mesh instance to turn into UnstructuredGrid
  virtual void SetMesh (iMesh_Instance mesh);
  vtkGetMacro (Mesh, iMesh_Instance);

protected:
  vtkITAPSUnstructuredSource ();
  ~vtkITAPSUnstructuredSource ();

  iMesh_Instance Mesh;
  bool MeshModified;

  void ExecuteData (vtkDataObject *);

private:
  vtkITAPSUnstructuredSource (const vtkITAPSUnstructuredSource&); // Not Implemented
  void operator= (const vtkITAPSUnstructuredSource&); // Not Implemented
};

#endif /* HAVE_VTK */

#endif /* __vtkITAPSUnstructuredSource_h */
