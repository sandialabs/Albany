//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_Layouts.hpp"

#include "LandIce_BasalFrictionCoefficientGradient.hpp"
#include "LandIce_ParamEnum.hpp"

#include <string.hpp> // for 'upper_case' (comes from src/utility; not to be confused with <string>)
//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

template<typename EvalT, typename Traits>
BasalFrictionCoefficientGradient<EvalT, Traits>::
BasalFrictionCoefficientGradient (const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl) :
  grad_beta (p.get<std::string> ("Basal Friction Coefficient Gradient Name"), dl->qp_gradient_sideset)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
#endif

  Teuchos::ParameterList& beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");

  std::string betaType = util::upper_case((beta_list.isParameter("Type") ? beta_list.get<std::string>("Type") : "Given Field"));

  numSideQPs = dl->qp_gradient->extent(2);
  sideDim    = dl->qp_gradient->extent(3);

  basalSideName = p.get<std::string>("Side Set Name");
  if (betaType == "GIVEN CONSTANT")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Constant and uniform beta, loaded from xml input file.\n";
#endif
    beta_type = GIVEN_CONSTANT;
  }
  else if ((betaType == "GIVEN FIELD") || (betaType == "EXPONENT OF GIVEN FIELD") || (betaType == "GALERKIN PROJECTION OF EXPONENT OF GIVEN FIELD"))
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Constant beta, loaded from file.\n";
#endif
    beta_type = GIVEN_FIELD;

    auto is_dist_param = p.isParameter("Dist Param Query Map") ? p.get<Teuchos::RCP<std::map<std::string,bool>>>("Dist Param Query Map") : Teuchos::null;
    std::string given_field_name = beta_list.get<std::string> ("Given Field Variable Name");
    is_given_field_param = is_dist_param.is_null() ? false : (*is_dist_param)[given_field_name];

    given_field_name += "_" + basalSideName;;

    if (is_given_field_param) {
      given_field_param = PHX::MDField<const ParamScalarT>(given_field_name, dl->node_scalar_sideset);
      this->addDependentField (given_field_param);
    } else {
      given_field = PHX::MDField<const RealType>(given_field_name, dl->node_scalar_sideset);
      this->addDependentField (given_field);
    }
    GradBF     = PHX::MDField<MeshScalarT>(p.get<std::string> ("Gradient BF Side Variable Name"), dl->node_qp_gradient_sideset);
    this->addDependentField (GradBF);

    numSideNodes = dl->node_qp_gradient->extent(2);
  }
  else if ((betaType == "GALERKIN PROJECTION OF EXPONENT OF GIVEN FIELD"))
  {
    // This is not supported. However, this evaluator may be created 'just in case we need' it,
    // and then be thrown away during the PHX DAG reorganization. We will throw in postRegistrationSetup
    beta_type = INVALID;
  }
  else if (betaType == "REGULARIZED COULOMB")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Regularized Coulomb (Schoof's sliding law).\n";
#endif
    beta_type = REGULARIZED_COULOMB;

    N      = PHX::MDField<ParamScalarT>(p.get<std::string> ("Effective Pressure QP Name"), dl->qp_scalar_sideset);
    U      = PHX::MDField<ScalarT>(p.get<std::string> ("Basal Velocity QP Name"), dl->qp_vector_sideset);
    gradN  = PHX::MDField<ScalarT>(p.get<std::string> ("Effective Pressure Gradient QP Name"), dl->qp_gradient_sideset);
    gradU  = PHX::MDField<ScalarT>(p.get<std::string> ("Basal Velocity Gradient QP Name"), dl->qp_vecgradient_sideset);
    u_norm = PHX::MDField<ScalarT>(p.get<std::string> ("Sliding Velocity QP Name"), dl->qp_scalar_sideset);

    muParam        = PHX::MDField<ScalarT,Dim>("Coulomb Friction Coefficient", dl->shared_param);
    lambdaParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
    powerParam     = PHX::MDField<ScalarT,Dim>("Power Exponent", dl->shared_param);

    this->addDependentField (N);
    this->addDependentField (U);
    this->addDependentField (gradN);
    this->addDependentField (gradU);
    this->addDependentField (u_norm);

    this->addDependentField (muParam);
    this->addDependentField (lambdaParam);
    this->addDependentField (powerParam);

    vecDim = dl->qp_vecgradient->extent(3);

    A = beta_list.get<double>("Constant Flow Factor A");

    // A*N^{1/q} is dimensionally correct only for q=1/3. To fix this, we modify A
    // so that the formula becomes (A_mod*N)^{1/q}. This means that A_mod = A^{1/3}
    A = std::cbrt(A);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Unrecognized basal friction condition coefficient type.\n");
    beta_type = INVALID;
  }

  this->addEvaluatedField(grad_beta);

  auto& stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if(use_stereographic_map)
  {
    coordVec = PHX::MDField<const MeshScalarT>(p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_coords_sideset);

    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    this->addDependentField(coordVec);
  }

  this->setName("BasalFrictionCoefficientGradient"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficientGradient<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficientGradient<EvalT, Traits>::
operator() (const GivenFieldParam_Tag& tag, const int& sideSet_idx) const {

  for (unsigned int qp=0; qp<numSideQPs; ++qp) {
    for (unsigned int dim=0; dim<sideDim; ++dim) {
      grad_beta(sideSet_idx,qp,dim) = 0.;
      for (unsigned int node=0; node<numSideNodes; ++node) {
        grad_beta(sideSet_idx,qp,dim) += GradBF(sideSet_idx,node,qp,dim)*given_field_param(sideSet_idx,node);
      }
    }
  }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficientGradient<EvalT, Traits>::
operator() (const GivenField_Tag& tag, const int& sideSet_idx) const {

  for (unsigned int qp=0; qp<numSideQPs; ++qp) {
    for (unsigned int dim=0; dim<sideDim; ++dim) {
      grad_beta(sideSet_idx,qp,dim) = 0.;
      for (unsigned int node=0; node<numSideNodes; ++node) {
        grad_beta(sideSet_idx,qp,dim) += GradBF(sideSet_idx,node,qp,dim)*given_field(sideSet_idx,node);
      }
    }
  }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficientGradient<EvalT, Traits>::
operator() (const RegularizedCoulomb_Tag& tag, const int& sideSet_idx) const {

  for (unsigned int qp=0; qp<numSideQPs; ++qp)
  {
    ScalarT u_val      = u_norm(sideSet_idx,qp);
    ParamScalarT N_val = N(sideSet_idx,qp);
    ScalarT den        = u_val+lambda*std::pow(A*N_val,1./power);

    ScalarT f_u = (power-1)*mu*N_val*std::pow(u_val,power-2)/std::pow(u_val+lambda*std::pow(A*N_val,1./power), power)
                - power*mu*N_val*std::pow(u_val,power-1)/std::pow(den, power+1);
    ScalarT f_N = mu*std::pow(u_val,power-1)/std::pow(u_val+lambda*std::pow(A*N_val,1./power), power)
                - mu*N_val*std::pow(u_val,power-1)/std::pow(den, power+1)*lambda*std::pow(A*N_val,1./power-1)*A;
    for (unsigned int dim=0; dim<sideDim; ++dim)
    {
      grad_beta(sideSet_idx,qp,dim) = f_N*gradN(sideSet_idx,qp,dim);
      for (unsigned int comp=0; comp<vecDim; ++comp)
        grad_beta(sideSet_idx,qp,dim) += f_u * (U(sideSet_idx,qp,comp)/u_val)*gradU(sideSet_idx,qp,vecDim,dim);
    }
  }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficientGradient<EvalT, Traits>::
operator() (const StereographicMapCorrection_Tag& tag, const int& sideSet_idx) const {

  for (unsigned int qp=0; qp<numSideQPs; ++qp)
  {
    MeshScalarT x = coordVec(sideSet_idx,qp,0) - x_0;
    MeshScalarT y = coordVec(sideSet_idx,qp,1) - y_0;
    MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
    for (unsigned int dim=0; dim<sideDim; ++dim)
      grad_beta(sideSet_idx,qp,dim) *= h*h;
  }

}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficientGradient<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (beta_type==INVALID, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error in LandIce::BasalFrictionCoefficientGradient: cannot compute the gradient of this type of beta.");

  if (workset.sideSetViews->find(basalSideName)==workset.sideSetViews->end()) return;
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  if (beta_type==REGULARIZED_COULOMB)
  {
    lambda = lambdaParam(0);
    mu     = muParam(0);
    power  = powerParam(0);
  }

  sideSet = workset.sideSetViews->at(basalSideName);

  switch (beta_type)
  {
    case GIVEN_CONSTANT:
      grad_beta.deep_copy(0);
      break;
    case GIVEN_FIELD:
      if (is_given_field_param) {
        Kokkos::parallel_for(GivenFieldParam_Policy(0, sideSet.size), *this);
      } else {
        Kokkos::parallel_for(GivenField_Policy(0, sideSet.size), *this);
      }
      break;
    case REGULARIZED_COULOMB:
      Kokkos::parallel_for(RegularizedCoulomb_Policy(0, sideSet.size), *this);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
            std::endl << "Error in LandIce::BasalFrictionCoefficientGradient: cannot compute the gradient of this type of beta.");
  }
  if (use_stereographic_map) {
    Kokkos::parallel_for(StereographicMapCorrection_Policy(0, sideSet.size), *this);
  }
}

} // Namespace LandIce
