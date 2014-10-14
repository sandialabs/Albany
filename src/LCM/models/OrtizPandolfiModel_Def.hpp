//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
//#include <typeinfo>

namespace LCM
{

//------------------------------------------------------------------------------
// See Ortiz and Pandolfi, IJNME (1999)
// Finite-deformation irreversible cohesive elements for 3-D crack
// propagation analysis
//------------------------------------------------------------------------------

template<typename EvalT, typename Traits>
OrtizPandolfiModel<EvalT, Traits>::
OrtizPandolfiModel(Teuchos::ParameterList* p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
  delta_c(p->get<RealType>("delta_c", 1.0)),
  sigma_c(p->get<RealType>("sigma_c", 1.0)),
  beta(p->get<RealType>("beta", 1.0)),
  stiff_c(p->get<RealType>("stiff_c", 1.0))
{

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair("Vector Jump", dl->qp_vector));
  this->dep_field_map_.insert(std::make_pair("Current Basis", dl->qp_tensor));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair("Cohesive_Traction", dl->qp_vector));
  this->eval_field_map_.insert(std::make_pair("Normal_Traction", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Shear_Traction", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Normal_Jump", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Shear_Jump", dl->qp_scalar));
  this->eval_field_map_.insert(std::make_pair("Max_Jump", dl->qp_scalar));

  // define the state variables
  //
  // cohesive traction
  this->num_state_variables_++;
  this->state_var_names_.push_back("Cohesive_Traction");
  this->state_var_layouts_.push_back(dl->qp_vector);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Cohesive Traction", false));
  //
  // normal traction
  this->num_state_variables_++;
  this->state_var_names_.push_back("Normal_Traction");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Normal Traction", false));
  //
  // shear traction
  this->num_state_variables_++;
  this->state_var_names_.push_back("Shear_Traction");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Shear Traction", false));
  //
  // normal jump
  this->num_state_variables_++;
  this->state_var_names_.push_back("Normal_Jump");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Normal Jump", false));
  //
  // shear jump
  this->num_state_variables_++;
  this->state_var_names_.push_back("Shear_Jump");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Shear Jump", false));
  //
  // max jump
  this->num_state_variables_++;
  this->state_var_names_.push_back("Max_Jump");
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Max Jump", false));
  //
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void OrtizPandolfiModel<EvalT, Traits>::
computeState(typename Traits::EvalData workset,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
    std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
{

  // extract dependent MDFields
  PHX::MDField<ScalarT> jump = *dep_fields["Vector Jump"];
  PHX::MDField<ScalarT> basis = *dep_fields["Current Basis"];

  // extract evaluated MDFields
  PHX::MDField<ScalarT> traction = *eval_fields["Cohesive_Traction"];
  PHX::MDField<ScalarT> tractionNormal = *eval_fields["Normal_Traction"];
  PHX::MDField<ScalarT> tractionShear = *eval_fields["Shear_Traction"];
  PHX::MDField<ScalarT> jumpNormal = *eval_fields["Normal_Jump"];
  PHX::MDField<ScalarT> jumpShear = *eval_fields["Shear_Jump"];
  PHX::MDField<ScalarT> jumpMax = *eval_fields["Max_Jump"];

  // get state variable
  Albany::MDArray jumpMaxOld = (*workset.stateArrayPtr)["Max_Jump_old"];

  //bool print = false;
  //  if (typeid(ScalarT) == typeid(RealType)) print = true;
  //  std::cout.precision(15);

  for (std::size_t cell(0); cell < workset.numCells; ++cell) {
    for (std::size_t pt(0); pt < num_pts_; ++pt) {
      
      //current basis vector
      Intrepid::Vector<ScalarT> g_0(3, &basis(cell, pt, 0, 0));
      Intrepid::Vector<ScalarT> g_1(3, &basis(cell, pt, 1, 0));
      Intrepid::Vector<ScalarT> n(3, &basis(cell, pt, 2, 0));

      //current jump vector - move PHX::MDField into Intrepid::Vector
      Intrepid::Vector<ScalarT> jumpPt(3, &jump(cell, pt, 0));

      //construct Identity tensor (2nd order) and tensor product of normal
      Intrepid::Tensor<ScalarT> I(Intrepid::eye<ScalarT>(3));
      Intrepid::Tensor<ScalarT> Fn(Intrepid::bun(n,n));

      // define components of the jump
      // jumpN is the normal component
      // jumpS is the shear component
      // jumpM is the maximum effective jump from prior converged iteration
      // vecJumpS is the shear vector

      ScalarT jumpN, jumpS, jumpM;
      Intrepid::Vector<ScalarT> vecJumpS(3);

      jumpM = jumpMaxOld(cell,pt);
      jumpN = Intrepid::dot(jumpPt,n);
      vecJumpS = Intrepid::dot(I - Fn,jumpPt);
      jumpS = sqrt(Intrepid::dot(vecJumpS,vecJumpS));

      // define the effective jump
      // for intepenetration, only employ shear component

      ScalarT jumpEff;
      if (jumpN >= 0.0)
          jumpEff = sqrt(beta*beta*jumpS*jumpS + jumpN*jumpN);
      else
    	  jumpEff = beta*jumpS;

      // Debugging - print kinematics
      //if (print) {
      //        std::cout << "jump for cell " << cell << " integration point " << pt << std::endl;
      //        std::cout << jumpPt << std::endl;
      //        std::cout << "normal jump for cell " << cell << " integration point " << pt << std::endl;
      //        std::cout << jumpN << std::endl;
      //        std::cout << "shear jump for cell " << cell << " integration point " << pt << std::endl;
      //        std::cout << jumpS << std::endl;
      //        std::cout << "effective jump for cell " << cell << " integration point " << pt << std::endl;
      //        std::cout << jumpEff << std::endl;
      //     }


      // define the constitutive response through an effective traction

      ScalarT tEff;
      if (jumpEff < jumpM && jumpEff < delta_c) // linear unloading toward origin
          tEff = sigma_c/jumpM*(1.0 - jumpM/delta_c)*jumpEff;
      else if (jumpEff >= jumpM && jumpEff <= delta_c) // linear unloading toward delta_c
          tEff = sigma_c*(1.0 - jumpEff/delta_c);
      else  // completely unloaded
    	  tEff = 0.0;

      // calculate the global traction
      // penalize interpentration through stiff_c
      Intrepid::Vector<ScalarT> tVec(3);
      if (jumpN == 0.0 & jumpEff == 0.0)        // no interpenetration, no effective jump
    	  tVec = 0.0*n;
      else if (jumpN < 0.0 && jumpEff == 0.0)   // interpenetration, no effective jump
          tVec = stiff_c*jumpN*n;
      else if (jumpN < 0.0 && jumpEff > 0.0)    //  interpenetration, effective jump
    	  tVec = tEff/jumpEff*beta*beta*vecJumpS + stiff_c*jumpN*n;
      else
          tVec = tEff/jumpEff*(beta*beta*vecJumpS + jumpN*n);

      // Debugging - print tractions
      //      if (print) {
      //              std::cout << "traction for cell " << cell << " integration point " << pt << std::endl;
      //              std::cout << tVec << std::endl;
      //              std::cout << "effective traction for cell " << cell << " integration point " << pt << std::endl;
      //              std::cout << tEff << std::endl;
      //           }

      // update global traction
      traction(cell,pt,0) = tVec(0);
      traction(cell,pt,1) = tVec(1);
      traction(cell,pt,2) = tVec(2);

      // update state variables 
      if (jumpN < 0.0)
          tractionNormal(cell,pt) = stiff_c*jumpN;
      else
          tractionNormal(cell,pt) = tEff*jumpN/jumpEff;

      tractionShear(cell,pt) = tEff*jumpS/jumpEff*beta*beta;
      jumpNormal(cell,pt) = jumpN;
      jumpShear(cell,pt) = jumpS;

      // only true state variable is jumpMax
      if (jumpEff > jumpM)
          jumpMax(cell,pt) = jumpEff;

    }
  }
}
//------------------------------------------------------------------------------
}

