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


#ifdef HAVE_VTK

#include "vtkITAPSUnstructuredSource.hpp"
#include "vtkObjectFactory.h"
#include "vtkUnstructuredGrid.h"
#include "vtkPoints.h"
#include "vtkPointData.h"
#include "vtkDoubleArray.h"
#include "vtkCellType.h"
#include "vtkIdList.h"

vtkStandardNewMacro (vtkITAPSUnstructuredSource);
vtkCxxRevisionMacro (vtkITAPSUnstructuredSource, "$Revision$");

vtkITAPSUnstructuredSource::vtkITAPSUnstructuredSource ()
{
  this->Mesh = 0;
  this->MeshModified = false;
  this->SetNumberOfInputPorts (0);
}

vtkITAPSUnstructuredSource::~vtkITAPSUnstructuredSource ()
{
}

void vtkITAPSUnstructuredSource::PrintSelf (ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf (os, indent);
  os << indent << "Mesh " << Mesh << endl;
}

void vtkITAPSUnstructuredSource::SetMesh (iMesh_Instance mesh)
{
  if (this->Mesh != mesh)
    {
    this->Mesh = mesh;
    this->MeshModified = true;
    this->Modified ();
    }
}

//----------------------------------------------------------------------------
void vtkITAPSUnstructuredSource::ExecuteData(vtkDataObject* obj)
{
  if (!this->Mesh)
    {
    vtkErrorMacro("No iMesh given to use with GenericDataSet");
    return;
    }

  vtkUnstructuredGrid *output = vtkUnstructuredGrid::SafeDownCast (obj);

  int result;
  iBase_EntitySetHandle rootset;
  iMesh_getRootSet (this->Mesh, &rootset, &result);

  if (this->MeshModified)
    {
    double* coords = 0;
    int coords_alloc = 0, coords_size;
    int* in_entity_set = 0;
    int in_entity_set_alloc = 0, in_entity_set_size;
    int order = iBase_INTERLEAVED;

    iMesh_getAllVtxCoords (this->Mesh, rootset, 
                           &coords, &coords_alloc, &coords_size,
                           &in_entity_set, &in_entity_set_alloc, &in_entity_set_size,
                           &order, &result);
    if (in_entity_set_alloc)
      {
      delete [] in_entity_set;
      }

    vtkPoints* points = vtkPoints::New ();
    for (int i = 0; i < coords_size; i += 3)
      {
      points->InsertNextPoint (coords[i], coords[i+1], coords[i+2]);
      }
    output->SetPoints (points);
    points->Delete ();
  
    delete [] coords;
  
    int* offset = 0, offset_alloc = 0, offset_size;
    int* index = 0, index_alloc = 0, index_size;
    int* topo = 0, topo_alloc = 0, topo_size;
    iMesh_getVtxCoordIndex (this->Mesh, rootset, 
                            iBase_REGION, iMesh_ALL_TOPOLOGIES, iBase_VERTEX,
                            &offset, &offset_alloc, &offset_size,
                            &index, &index_alloc, &index_size,
                            &topo, &topo_alloc, &topo_size,
                            &result);
    output->Allocate (topo_size);
    for (int i = 0; i < topo_size; i ++) 
      {
      int nodes_per_element;
      if (i == offset_size - 1)
        {
        nodes_per_element = index_size - offset[i];
        }
      else
        {
        nodes_per_element = offset[i+1] - offset[i];
        }
      switch (nodes_per_element)
        {
        case 2:
          topo[i] = VTK_LINE;
          break;
        case 3:
          topo[i] = VTK_TRIANGLE;
          break;
        case 4:
          if (topo[i] == iMesh_QUADRILATERAL)
            {
            topo[i] = VTK_QUAD;
            }
          else
            {
            topo[i] = VTK_TETRA;
            }
          break;
        case 5:
          topo[i] = VTK_PYRAMID;
          break;
        case 6:
          topo[i] = VTK_WEDGE;
          break;
        case 8:
          topo[i] = VTK_HEXAHEDRON;
          break;
        default:
          vtkErrorMacro ("Unsupported cell type");
          continue;
        }
/*
      switch (topo[i])
        {
        case iMesh_TRIANGLE:
          topo[i] = VTK_TRIANGLE;
          break;
        case iMesh_QUADRILATERAL: 
          // topo[i] = VTK_QUAD;
          topo[i] = VTK_LINE;
          break;
        case iMesh_POLYGON:
          topo[i] = VTK_POLYGON;
          break;
        case iMesh_TETRAHEDRON:
          topo[i] = VTK_TETRA;
          break;
        case iMesh_HEXAHEDRON:
          topo[i] = VTK_HEXAHEDRON;
          break;
        case iMesh_PRISM:
          topo[i] = VTK_WEDGE;
          break;
        case iMesh_PYRAMID:
          topo[i] = VTK_PYRAMID;
          break;
        default:
          vtkErrorMacro ("Unsupported cell type");
          continue;
        }
*/
        vtkIdList* list = vtkIdList::New ();
        if (i == offset_size - 1)
          {
          for (int j = offset[i]; j < index_size; j ++)
            {
            list->InsertNextId (index[j]);
            }
          }
        else
          {
          for (int j = offset[i]; j < offset[i+1]; j ++)
            {
            list->InsertNextId (index[j]);
            }
          }
        output->InsertNextCell (topo[i], list);
        list->Delete ();
      }
  
    delete [] offset;
    delete [] index;
    delete [] topo;
    // TODO why can't I assume this?
    // this->MeshModified = false;
    }

  double* fields = 0;
  int fields_allocated = 0, fields_size, fields_per_vertex;
  int order = iBase_INTERLEAVED;
  iMesh_getAllVtxFields (this->Mesh, rootset,
    &fields, &fields_allocated, &fields_size, &fields_per_vertex,
    &order, &result);   

  vtkPointData *pd = output->GetPointData ();
  pd->Initialize ();
  for (int i = 0; i < fields_per_vertex; i ++)
    {
    vtkDoubleArray *da = vtkDoubleArray::New ();
    char name[64];
    sprintf (name, "%d", i+1);
    da->SetName (name);
    da->SetNumberOfTuples (fields_size/fields_per_vertex);
    da->SetNumberOfComponents (1);
    int index = 0;
    for (int j = i; j < fields_size; j += fields_per_vertex)
      {
      da->SetComponent (index ++, 0, fields[j]);
      }
    pd->AddArray (da);
    da->Delete ();
    }

  if (fields_per_vertex > 0) 
    {
    pd->SetActiveAttribute (0, vtkDataSetAttributes::SCALARS);
    delete [] fields;
    }

}

#endif
