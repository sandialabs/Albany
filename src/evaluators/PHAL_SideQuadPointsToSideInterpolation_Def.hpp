//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
SideQuadPointsToSideInterpolationBase<EvalT, Traits, ScalarT>::
SideQuadPointsToSideInterpolationBase (const Teuchos::ParameterList& p,
                                       const Teuchos::RCP<Albany::Layouts>& dl_side) :
  w_measure (p.get<std::string>("Weighted Measure Name"), dl_side->qp_scalar)
{
  fieldDim = p.isParameter("Field Dimension") ? p.get<int>("Field Dimension") : 0;

  sideSetName = p.get<std::string>("Side Set Name");
  if (fieldDim==0)
  {
    field_qp   = decltype(field_qp)(p.get<std::string> ("Field QP Name"), dl_side->qp_scalar);
    field_side = decltype(field_side)(p.get<std::string> ("Field Side Name"), dl_side->cell_scalar2);
  }
  else if (fieldDim==1)
  {
    field_qp   = decltype(field_qp)(p.get<std::string> ("Field QP Name"), dl_side->qp_vector);
    field_side = decltype(field_side)(p.get<std::string> ("Field Side Name"), dl_side->cell_vector);
  }
  else if (fieldDim==2)
  {
    field_qp   = decltype(field_qp)(p.get<std::string> ("Field QP Name"), dl_side->qp_tensor);
    field_side = decltype(field_side)(p.get<std::string> ("Field Side Name"), dl_side->cell_tensor);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Field dimension not supported.\n");
  }

  field_qp.dimensions(dims);

  this->addDependentField (field_qp);
  this->addDependentField (w_measure);
  this->addEvaluatedField (field_side);

  this->setName("SideQuadPointsToSideInterpolation"+PHX::typeAsString<EvalT>());
}

template<typename EvalT, typename Traits, typename ScalarT>
void SideQuadPointsToSideInterpolationBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData (field_qp,fm);
  this->utils.setFieldData (w_measure,fm);

  this->utils.setFieldData (field_side,fm);
}

template<typename EvalT, typename Traits, typename ScalarT>
void SideQuadPointsToSideInterpolationBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  // Note: since only required sides are processed by the evaluator,
  //       if we don't zero out the values from the previous workset
  //       we may save this field using old values and make a mess!

  ScalarT zero = 0.;
  field_side.deep_copy (zero);

  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    MeshScalarT meas = 0.0;
    for (int qp(0); qp<dims[2]; ++qp)
    {
      meas += w_measure(cell,side,qp);
    }

    switch (fieldDim)
    {
      case 0:
        field_side(cell,side) = 0.0;
        for (int qp(0); qp<dims[2]; ++qp)
          field_side(cell,side) += field_qp(cell,side,qp)*w_measure(cell,side,qp);
        field_side(cell,side) /= meas;
        break;

      case 1:
        for (int i(0); i<dims[3]; ++i)
        {
          field_side(cell,side,i) = 0;
          for (int qp(0); qp<dims[2]; ++qp)
            field_side(cell,side,i) += field_qp(cell,side,qp,i)*w_measure(cell,side,qp);
          field_side(cell,side,i) /= meas;
        }
        break;

      case 2:
        for (int i(0); i<dims[3]; ++i)
        {
          for (int j(0); j<dims[4]; ++j)
          {
            field_side(cell,side,i,j) = 0;
            for (int qp(0); qp<dims[2]; ++qp)
              field_side(cell,side,i,j) += field_qp(cell,side,qp,i,j)*w_measure(cell,side,qp);
            field_side(cell,side,i,j) /= meas;
          }
        }
        break;

      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Field dimension not supported (this error should have already appeared).\n");
    }
  }
}

} // Namespace PHAL
