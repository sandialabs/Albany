//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"

#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits, typename ThicknessScalarT>
LandIce::FluxDiv<EvalT, Traits, ThicknessScalarT>::
FluxDiv (const Teuchos::ParameterList& p,
         const Teuchos::RCP<Albany::Layouts>& dl_basal)
{
  // get and validate Response parameter list
  std::string fieldName = p.get<std::string> ("Field Name");

  const std::string& averaged_velocity_name     = p.get<std::string>("Averaged Velocity Side QP Variable Name");
  const std::string& div_averaged_velocity_name = p.get<std::string>("Averaged Velocity Side QP Divergence Name");
  const std::string& thickness_name             = p.get<std::string>("Thickness Side QP Variable Name");
  const std::string& grad_thickness_name        = p.get<std::string>("Thickness Gradient Name");
  const std::string& side_tangents_name         = p.get<std::string>("Side Tangents Name");

  averaged_velocity     = decltype(averaged_velocity)(averaged_velocity_name, dl_basal->qp_vector);
  div_averaged_velocity = decltype(div_averaged_velocity)(div_averaged_velocity_name, dl_basal->qp_scalar);
  thickness             = decltype(thickness)(thickness_name, dl_basal->qp_scalar);
  grad_thickness        = decltype(grad_thickness)(grad_thickness_name, dl_basal->qp_gradient);
  side_tangents         = decltype(side_tangents)(side_tangents_name, dl_basal->qp_tensor_cd_sd);

  flux_div              = decltype(flux_div)(fieldName, dl_basal->qp_scalar);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl_basal->qp_vector->dimensions(dims);
  numSideQPs = dims[1];
  numSideDims  = dims[2];

  sideSetName = p.get<std::string> ("Side Set Name");

  // add dependent fields
  this->addDependentField(averaged_velocity);
  this->addDependentField(div_averaged_velocity);
  this->addDependentField(thickness);
  this->addDependentField(grad_thickness);
  this->addDependentField(side_tangents);

  this->addEvaluatedField(flux_div);

  this->setName(PHX::print<FluxDiv<EvalT,Traits,ThicknessScalarT>>());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarT>
void LandIce::FluxDiv<EvalT, Traits, ThicknessScalarT>::postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(averaged_velocity, fm);
  this->utils.setFieldData(div_averaged_velocity, fm);
  this->utils.setFieldData(thickness, fm);
  this->utils.setFieldData(grad_thickness, fm);
  this->utils.setFieldData(side_tangents, fm);
  this->utils.setFieldData(flux_div, fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename ThicknessScalarT>
KOKKOS_INLINE_FUNCTION
void LandIce::FluxDiv<EvalT, Traits, ThicknessScalarT>::
operator() (const FluxDiv_Tag& tag, const int& sideSet_idx) const {

  for (unsigned int qp=0; qp<numSideQPs; ++qp)
  {
    ScalarT grad_thickness_tmp[2] = {0.0, 0.0};
    for (std::size_t dim = 0; dim < numSideDims; ++dim)
    {
      grad_thickness_tmp[0] += side_tangents(sideSet_idx,qp,0,dim)*grad_thickness(sideSet_idx,qp,dim);
      grad_thickness_tmp[1] += side_tangents(sideSet_idx,qp,1,dim)*grad_thickness(sideSet_idx,qp,dim);
    }

    ScalarT divHV = div_averaged_velocity(sideSet_idx,qp)* thickness(sideSet_idx,qp);
    for (std::size_t dim = 0; dim < numSideDims; ++dim) {
      divHV += grad_thickness_tmp[dim]*averaged_velocity(sideSet_idx,qp,dim);
    }
    flux_div(sideSet_idx,qp) = divHV;
  }

}

// **********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarT>
void LandIce::FluxDiv<EvalT, Traits, ThicknessScalarT>::evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::runtime_error,
                              "Side sets defined in input file but not properly specified on the mesh.\n");

  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return;
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  sideSet = workset.sideSetViews->at(sideSetName);
  
  Kokkos::parallel_for(FluxDiv_Policy(0, sideSet.size), *this);  
}
