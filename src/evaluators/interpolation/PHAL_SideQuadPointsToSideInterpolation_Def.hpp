//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "PHAL_SideQuadPointsToSideInterpolation.hpp"
#include "Albany_DiscretizationUtils.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
SideQuadPointsToSideInterpolationBase<EvalT, Traits, ScalarT>::
SideQuadPointsToSideInterpolationBase (const Teuchos::ParameterList& p,
                                       const Teuchos::RCP<Albany::Layouts>& dl_side) :
  w_measure (p.get<std::string>("Weighted Measure Name"), dl_side->useCollapsedSidesets ? dl_side->qp_scalar_sideset : dl_side->qp_scalar)
{
  fieldDim = p.isParameter("Field Dimension") ? p.get<int>("Field Dimension") : 0;

  useCollapsedSidesets = dl_side->useCollapsedSidesets;

  sideSetName = p.get<std::string>("Side Set Name");
  if (fieldDim==0)
  {
    field_qp   = decltype(field_qp)(p.get<std::string> ("Field QP Name"), useCollapsedSidesets ? dl_side->qp_scalar_sideset : dl_side->qp_scalar);
    field_side = decltype(field_side)(p.get<std::string> ("Field Side Name"), useCollapsedSidesets ? dl_side->cell_scalar2_sideset : dl_side->cell_scalar2);
  }
  else if (fieldDim==1)
  {
    field_qp   = decltype(field_qp)(p.get<std::string> ("Field QP Name"), useCollapsedSidesets ? dl_side->qp_vector_sideset : dl_side->qp_vector);
    field_side = decltype(field_side)(p.get<std::string> ("Field Side Name"), useCollapsedSidesets ? dl_side->cell_vector_sideset : dl_side->cell_vector);
  }
  else if (fieldDim==2)
  {
    field_qp   = decltype(field_qp)(p.get<std::string> ("Field QP Name"), useCollapsedSidesets ? dl_side->qp_tensor_sideset : dl_side->qp_tensor);
    field_side = decltype(field_side)(p.get<std::string> ("Field Side Name"), useCollapsedSidesets ? dl_side->cell_tensor_sideset : dl_side->cell_tensor);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Field dimension not supported.\n");
  }

  this->addDependentField (field_qp.fieldTag());
  this->addDependentField (w_measure.fieldTag());
  this->addEvaluatedField (field_side);

  this->setName("SideQuadPointsToSideInterpolation"+PHX::print<EvalT>());
}

template<typename EvalT, typename Traits, typename ScalarT>
void SideQuadPointsToSideInterpolationBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData (field_qp,fm);
  this->utils.setFieldData (w_measure,fm);

  this->utils.setFieldData (field_side,fm);
  field_qp.dimensions(dims);

  TEUCHOS_TEST_FOR_EXCEPTION (dims.size() > 5, Teuchos::Exceptions::InvalidParameter, "Error! val_side has more dimensions than expected.\n");

  for (unsigned int i = 0; i < dims.size(); ++i)
    dimsArray[i] = dims[i];

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void SideQuadPointsToSideInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Dim0_Tag& tag, const int& sideSet_idx) const {

  MeshScalarT meas = 0.0;
      
  for (unsigned int qp(0); qp<dimsArray[1]; ++qp)
    meas += w_measure(sideSet_idx,qp);

  field_side(sideSet_idx) = 0.0;
  for (unsigned int qp(0); qp<dimsArray[1]; ++qp) {
    field_side(sideSet_idx) += field_qp(sideSet_idx,qp)*w_measure(sideSet_idx,qp);
  }
  field_side(sideSet_idx) /= meas;

}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void SideQuadPointsToSideInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Dim1_Tag& tag, const int& sideSet_idx) const {

  MeshScalarT meas = 0.0;
      
  for (unsigned int qp(0); qp<dimsArray[1]; ++qp)
    meas += w_measure(sideSet_idx,qp);

  for (unsigned int i(0); i<dimsArray[2]; ++i)
  {
    field_side(sideSet_idx,i) = 0;
    for (unsigned int qp(0); qp<dimsArray[1]; ++qp)
      field_side(sideSet_idx,i) += field_qp(sideSet_idx,qp,i)*w_measure(sideSet_idx,qp);
    field_side(sideSet_idx,i) /= meas;
  }

}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void SideQuadPointsToSideInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Dim2_Tag& tag, const int& sideSet_idx) const {

  MeshScalarT meas = 0.0;
      
  for (unsigned int qp(0); qp<dimsArray[1]; ++qp)
    meas += w_measure(sideSet_idx,qp);

  for (unsigned int i(0); i<dimsArray[2]; ++i)
  {
    for (unsigned int j(0); j<dimsArray[3]; ++j)
    {
      field_side(sideSet_idx,i,j) = 0;
      for (unsigned int qp(0); qp<dimsArray[1]; ++qp)
        field_side(sideSet_idx,i,j) += field_qp(sideSet_idx,qp,i,j)*w_measure(sideSet_idx,qp);
      field_side(sideSet_idx,i,j) /= meas;
    }
  }

}

template<typename EvalT, typename Traits, typename ScalarT>
void SideQuadPointsToSideInterpolationBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return;
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  sideSet = workset.sideSetViews->at(sideSetName);

  if (useCollapsedSidesets) {
    switch (fieldDim)
    {
      case 0:
        Kokkos::parallel_for(Dim0_Policy(0, sideSet.size), *this);
        break;
      case 1:
        Kokkos::parallel_for(Dim1_Policy(0, sideSet.size), *this);
        break;
      case 2:
        Kokkos::parallel_for(Dim2_Policy(0, sideSet.size), *this);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Field dimension not supported (this error should have already appeared).\n");
    }
  } else {
    switch (fieldDim)
    {
      case 0:
        for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
        {
          // Get the local data of side and cell
          const int cell = sideSet.elem_LID(sideSet_idx);
          const int side = sideSet.side_local_id(sideSet_idx);

          MeshScalarT meas = 0.0;
          
          for (int qp(0); qp<dims[2]; ++qp)
            meas += w_measure(cell,side,qp);

          field_side(cell,side) = 0.0;
          for (int qp(0); qp<dims[2]; ++qp) {
            field_side(cell,side) += field_qp(cell,side,qp)*w_measure(cell,side,qp);
          }
          field_side(cell,side) /= meas;
        }
        break;
      case 1:
        for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
        {
          // Get the local data of side and cell
          const int cell = sideSet.elem_LID(sideSet_idx);
          const int side = sideSet.side_local_id(sideSet_idx);

          MeshScalarT meas = 0.0;
          
          for (int qp(0); qp<dims[2]; ++qp)
            meas += w_measure(cell,side,qp);

          for (int i(0); i<dims[3]; ++i)
          {
            field_side(cell,side,i) = 0;
            for (int qp(0); qp<dims[2]; ++qp)
              field_side(cell,side,i) += field_qp(cell,side,qp,i)*w_measure(cell,side,qp);
            field_side(cell,side,i) /= meas;
          }
        }
        break;
      case 2:
        for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
        {
          // Get the local data of side and cell
          const int cell = sideSet.elem_LID(sideSet_idx);
          const int side = sideSet.side_local_id(sideSet_idx);

          MeshScalarT meas = 0.0;
          
          for (int qp(0); qp<dims[2]; ++qp)
            meas += w_measure(cell,side,qp);

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
        }
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Field dimension not supported (this error should have already appeared).\n");
    }
  }

}

} // Namespace PHAL
