//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
ElementBarycenter<EvalT, Traits>::
ElementBarycenter(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
 : coords (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector )
 , bary      (p.get<std::string>  ("Barycenter Name"), dl->cell_gradient)
 , bary_x    (p.get<std::string>  ("Barycenter X Name"), dl->cell_scalar2)
{
  this->addDependentField(coords.fieldTag());
  this->addEvaluatedField(bary);
  this->addEvaluatedField(bary_x);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->vertices_vector->dimensions(dim);
  numVertices = dim[1];
  numDims = dim[2];

  // In higher dims, add y and z coords of baricenter
  if (numDims>1) {
    bary_y = decltype(bary_y)(p.get<std::string>  ("Barycenter Y Name"), dl->cell_scalar2);
    this->addEvaluatedField(bary_y);
    if (numDims>2) {
      bary_z = decltype(bary_z)(p.get<std::string>  ("Barycenter Z Name"), dl->cell_scalar2);
      this->addEvaluatedField(bary_z);
    }
  }

  this->setName("ElementBarycenter"+PHX::print<EvalT>());
}

template<typename EvalT, typename Traits>
void ElementBarycenter<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coords,fm);
  this->utils.setFieldData(bary,fm);
  this->utils.setFieldData(bary_x,fm);

  if (numDims>1) {
    this->utils.setFieldData(bary_y,fm);
    if (numDims>2) {
      this->utils.setFieldData(bary_z,fm);
    }
  }

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits>
void ElementBarycenter<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  for (int icell=0; icell<workset.numCells; ++icell) {
    for (int idim=0; idim<numDims; ++idim) {
      bary(icell,idim) = coords(icell,0,idim);
      for (int iv=1; iv<numVertices; ++iv) {
        bary(icell,idim) += coords(icell,iv,idim);
      }
      bary(icell,idim) /= numVertices;
    }

    bary_x(icell) = bary(icell,0);
    if (numDims>1) {
      bary_y(icell) = bary(icell,1);
      if (numDims>2) {
        bary_z(icell) = bary(icell,2);
      }
    }
  }
}

} // namespace PHAL
