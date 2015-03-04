//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Intrepid_FunctionSpaceTools.hpp>
#include <Intrepid_MiniTensor.h>
#include <Teuchos_TestForException.hpp>
#include <Phalanx_DataLayout.hpp>
#include <typeinfo>

#include "LocalNonlinearSolver.hpp"


namespace LCM
{

//**********************************************************************
  template<typename EvalT, typename Traits>
  CapImplicitModel<EvalT, Traits>::
  CapImplicitModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
    A(p->get<RealType>("A")),
    B(p->get<RealType>("B")),
    C(p->get<RealType>("C")),
    theta(p->get<RealType>("theta")),
    R(p->get<RealType>("R")),
    kappa0(p->get<RealType>("kappa0")),
    W(p->get<RealType>("W")),
    D1(p->get<RealType>("D1")),
    D2(p->get<RealType>("D2")),
    calpha(p->get<RealType>("calpha")),
    psi(p->get<RealType>("psi")),
    N(p->get<RealType>("N")),
    L(p->get<RealType>("L")),
    phi(p->get<RealType>("phi")),
    Q(p->get<RealType>("Q"))
  {
      // define the dependent fields
      this->dep_field_map_.insert(std::make_pair("Strain", dl->qp_tensor));
      this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
      this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));
      
      // retrieve appropriate field name strings
      std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
      std::string strain_string = (*field_name_map_)["Strain"];
      std::string backStress_string = (*field_name_map_)["Back_Stress"];
      std::string capParameter_string = (*field_name_map_)["Cap_Parameter"];
      std::string eqps_string = (*field_name_map_)["eqps"];
      std::string volPlasticStrain_string = (*field_name_map_)["volPlastic_Strain"];
      
      // define the evaluated fields
      this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
      this->eval_field_map_.insert(std::make_pair(backStress_string, dl->qp_tensor));
      this->eval_field_map_.insert(std::make_pair(capParameter_string, dl->qp_scalar));
      this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
      this->eval_field_map_.insert(std::make_pair(volPlasticStrain_string, dl->qp_scalar));
      this->eval_field_map_.insert(
                                   std::make_pair("Material Tangent", dl->qp_tensor4));
      
      // define the state variables
      //
      // strain
      this->num_state_variables_++;
      this->state_var_names_.push_back(strain_string);
      this->state_var_layouts_.push_back(dl->qp_tensor);
      this->state_var_init_types_.push_back("scalar");
      this->state_var_init_values_.push_back(0.0);
      this->state_var_old_state_flags_.push_back(true);
      this->state_var_output_flags_.push_back(true);
      //
      // stress
      this->num_state_variables_++;
      this->state_var_names_.push_back(cauchy_string);
      this->state_var_layouts_.push_back(dl->qp_tensor);
      this->state_var_init_types_.push_back("scalar");
      this->state_var_init_values_.push_back(0.0);
      this->state_var_old_state_flags_.push_back(true);
      this->state_var_output_flags_.push_back(true);
      //
      // backStress
      this->num_state_variables_++;
      this->state_var_names_.push_back(backStress_string);
      this->state_var_layouts_.push_back(dl->qp_tensor);
      this->state_var_init_types_.push_back("scalar");
      this->state_var_init_values_.push_back(0.0);
      this->state_var_old_state_flags_.push_back(true);
      this->state_var_output_flags_.push_back(true);
      //
      // capParameter
      this->num_state_variables_++;
      this->state_var_names_.push_back(capParameter_string);
      this->state_var_layouts_.push_back(dl->qp_scalar);
      this->state_var_init_types_.push_back("scalar");
      this->state_var_init_values_.push_back(kappa0);
      this->state_var_old_state_flags_.push_back(true);
      this->state_var_output_flags_.push_back(true);
      //
      // eqps
      this->num_state_variables_++;
      this->state_var_names_.push_back(eqps_string);
      this->state_var_layouts_.push_back(dl->qp_scalar);
      this->state_var_init_types_.push_back("scalar");
      this->state_var_init_values_.push_back(0.0);
      this->state_var_old_state_flags_.push_back(true);
      this->state_var_output_flags_.push_back(true);
      //
      // volPlasticStrain
      this->num_state_variables_++;
      this->state_var_names_.push_back(volPlasticStrain_string);
      this->state_var_layouts_.push_back(dl->qp_scalar);
      this->state_var_init_types_.push_back("scalar");
      this->state_var_init_values_.push_back(0.0);
      this->state_var_old_state_flags_.push_back(true);
      this->state_var_output_flags_.push_back(true);
     
  }

//**********************************************************************
  template<typename EvalT, typename Traits>
    void CapImplicitModel<EvalT, Traits>::
    computeState(typename Traits::EvalData workset,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > dep_fields,
      std::map<std::string, Teuchos::RCP<PHX::MDField<ScalarT> > > eval_fields)
  {
      
      // extract dependent MDFields
      PHX::MDField<ScalarT> strain = *dep_fields["Strain"];
      PHX::MDField<ScalarT> poissons_ratio = *dep_fields["Poissons Ratio"];
      PHX::MDField<ScalarT> elastic_modulus = *dep_fields["Elastic Modulus"];
      
      // retrieve appropriate field name strings
      std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
      std::string strain_string = (*field_name_map_)["Strain"];
      std::string backStress_string = (*field_name_map_)["Back_Stress"];
      std::string capParameter_string = (*field_name_map_)["Cap_Parameter"];
      std::string eqps_string = (*field_name_map_)["eqps"];
      std::string volPlasticStrain_string = (*field_name_map_)["volPlastic_Strain"];
      
      // extract evaluated MDFields
      PHX::MDField<ScalarT> stress = *eval_fields[cauchy_string];
      PHX::MDField<ScalarT> backStress = *eval_fields[backStress_string];
      PHX::MDField<ScalarT> capParameter = *eval_fields[capParameter_string];
      PHX::MDField<ScalarT> eqps = *eval_fields[eqps_string];
      PHX::MDField<ScalarT> volPlasticStrain = *eval_fields[volPlasticStrain_string];
      PHX::MDField<ScalarT> tangent = *eval_fields["Material Tangent"];
      
      // get State Variables
      Albany::MDArray strainold = (*workset.stateArrayPtr)[strain_string + "_old"];
      Albany::MDArray stressold = (*workset.stateArrayPtr)[cauchy_string + "_old"];
      Albany::MDArray backStressold = (*workset.stateArrayPtr)[backStress_string + "_old"];
      Albany::MDArray capParameterold = (*workset.stateArrayPtr)[capParameter_string + "_old"];
      Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqps_string + "_old"];
      Albany::MDArray volPlasticStrainold =
      (*workset.stateArrayPtr)[volPlasticStrain_string + "_old"];
      
      
      for (int cell = 0; cell < workset.numCells; ++cell) {
          for (int qp = 0; qp < num_pts_; ++qp) {
              // local parameters
              ScalarT lame = elastic_modulus(cell, qp) * poissons_ratio(cell, qp)
              / (1.0 + poissons_ratio(cell, qp))
              / (1.0 - 2.0 * poissons_ratio(cell, qp));
              ScalarT mu = elastic_modulus(cell, qp) / 2.0
              / (1.0 + poissons_ratio(cell, qp));
              ScalarT bulkModulus = lame + (2. / 3.) * mu;
              
              // elastic matrix
              Intrepid::Tensor4<ScalarT> Celastic = lame
              * Intrepid::identity_3<ScalarT>(3)
              + mu
              * (Intrepid::identity_1<ScalarT>(3)
                 + Intrepid::identity_2<ScalarT>(3));
              
              // elastic compliance tangent matrix
              Intrepid::Tensor4<ScalarT> compliance = (1. / bulkModulus / 9.)
              * Intrepid::identity_3<ScalarT>(3)
              + (1. / mu / 2.)
              * (0.5
                 * (Intrepid::identity_1<ScalarT>(3)
                    + Intrepid::identity_2<ScalarT>(3))
                 - (1. / 3.) * Intrepid::identity_3<ScalarT>(3));
              
              // previous state
              Intrepid::Tensor<ScalarT>
              sigmaN(3, Intrepid::ZEROS),
              alphaN(3, Intrepid::ZEROS),
              strainN(3, Intrepid::ZEROS);
              
              // incremental strain tensor
              Intrepid::Tensor<ScalarT> depsilon(3);
              for (int i = 0; i < num_dims_; ++i) {
                  for (int j = 0; j < num_dims_; ++j) {
                      depsilon(i, j) = strain(cell, qp, i, j) - strainold(cell, qp, i, j);
                      strainN(i, j) = strainold(cell, qp, i, j);
                  }
              }
              
              // trial state
              Intrepid::Tensor<ScalarT> sigmaVal = Intrepid::dotdot(Celastic,
                                                                    depsilon);
              Intrepid::Tensor<ScalarT> alphaVal(3, Intrepid::ZEROS);
              
              for (int i = 0; i < num_dims_; ++i) {
                  for (int j = 0; j < num_dims_; ++j) {
                      sigmaVal(i, j) = sigmaVal(i, j) + stressold(cell, qp, i, j);
                      alphaVal(i, j) = backStressold(cell, qp, i, j);
                      sigmaN(i, j) = stressold(cell, qp, i, j);
                      alphaN(i, j) = backStressold(cell, qp, i, j);
                  }
              }
              
              ScalarT kappaVal = capParameterold(cell, qp);
              ScalarT dgammaVal = 0.0;
              
              // used in defining generalized hardening modulus
              ScalarT Htan(0.0);
              
              // define plastic strain increment, its two invariants: dev, and vol
              Intrepid::Tensor<ScalarT> deps_plastic(3, Intrepid::ZEROS);
              ScalarT deqps(0.0), devolps(0.0);
              
              // define temporary trial stress, used in computing plastic strain
              Intrepid::Tensor<ScalarT> sigmaTr = sigmaVal;
              
              std::vector<ScalarT> XXVal(13);
              
              // check yielding
              ScalarT f = compute_f(sigmaVal, alphaVal, kappaVal);
              XXVal = initialize(sigmaVal, alphaVal, kappaVal, dgammaVal);
              
              // local Newton loop
              if (f > 1.e-11) { // plastic yielding
                  
                  ScalarT normR, normR0, conv;
                  bool kappa_flag = false;
                  bool converged = false;
                  int iter = 0;
                  
                  std::vector<ScalarT> R(13);
                  std::vector<ScalarT> dRdX(13 * 13);
                  LocalNonlinearSolver<EvalT, Traits> solver;
                  
                  while (!converged) {
                      
                      // assemble residual vector and local Jacobian
                      compute_ResidJacobian(XXVal, R, dRdX, sigmaVal, alphaVal, kappaVal,
                                            Celastic, kappa_flag);
                      
                      normR = 0.0;
                      for (int i = 0; i < 13; i++)
                          normR += R[i] * R[i];
                      
                      normR = std::sqrt(normR);
                      
                      if (iter == 0)
                          normR0 = normR;
                      if (normR0 != 0)
                          conv = normR / normR0;
                      else
                          conv = normR0;
                      
                      if (conv < 1.e-11 || normR < 1.e-11)
                          break;
                      
                      if(iter > 20)
                          break;
                      
                      //TEUCHOS_TEST_FOR_EXCEPTION( iter > 20, std::runtime_error,
                      // std::endl << "Error in return mapping, iter = " << iter << "\nres = " << normR << "\nrelres = " << conv << std::endl);
                      
                      std::vector<ScalarT> XXValK = XXVal;
                      solver.solve(dRdX, XXValK, R);
                      
                      // put restrictions on kappa: only allows monotonic decreasing (cap hardening)
                      if (XXValK[11] > XXVal[11]) {
                          kappa_flag = true;
                      }
                      else {
                          XXVal = XXValK;
                          kappa_flag = false;
                      }
                      
                      // debugging
                      //XXVal = XXValK;
                      
                      iter++;
                  } //end local NR
                  
                  // compute sensitivity information, and pack back to X.
                  solver.computeFadInfo(dRdX, XXVal, R);
                  
              } // end of plasticity
              
              // update
              sigmaVal(0, 0) = XXVal[0];
              sigmaVal(0, 1) = XXVal[5];
              sigmaVal(0, 2) = XXVal[4];
              sigmaVal(1, 0) = XXVal[5];
              sigmaVal(1, 1) = XXVal[1];
              sigmaVal(1, 2) = XXVal[3];
              sigmaVal(2, 0) = XXVal[4];
              sigmaVal(2, 1) = XXVal[3];
              sigmaVal(2, 2) = XXVal[2];
              
              alphaVal(0, 0) = XXVal[6];
              alphaVal(0, 1) = XXVal[10];
              alphaVal(0, 2) = XXVal[9];
              alphaVal(1, 0) = XXVal[10];
              alphaVal(1, 1) = XXVal[7];
              alphaVal(1, 2) = XXVal[8];
              alphaVal(2, 0) = XXVal[9];
              alphaVal(2, 1) = XXVal[8];
              alphaVal(2, 2) = -XXVal[6] - XXVal[7];
              
              kappaVal = XXVal[11];
              
              //dgammaVal = XXVal[12];
              
              //compute plastic strain increment deps_plastic = compliance ( sigma_tr - sigma_(n+1));
              Intrepid::Tensor<ScalarT> dsigma = sigmaTr - sigmaVal;
              deps_plastic = Intrepid::dotdot(compliance, dsigma);
              
              // compute its two invariants: devolps (volumetric) and deqps (deviatoric)
              devolps = Intrepid::trace(deps_plastic);
              Intrepid::Tensor<ScalarT> dev_plastic = deps_plastic
              - (1.0 / 3.0) * devolps * Intrepid::identity<ScalarT>(3);
              //deqps = std::sqrt(2./3.) * Intrepid::norm(dev_plastic);
              // use altenative definition, just differ by constants
              deqps = std::sqrt(2) * Intrepid::norm(dev_plastic);
  
              // stress and back stress
              for (int i = 0; i < num_dims_; ++i) {
                  for (int j = 0; j < num_dims_; ++j) {
                      stress(cell, qp, i, j) = sigmaVal(i, j);
                      backStress(cell, qp, i, j) = alphaVal(i, j);
                  }
              }
              
              capParameter(cell, qp) = kappaVal;
              eqps(cell, qp) = eqpsold(cell, qp) + deqps;
              volPlasticStrain(cell, qp) = volPlasticStrainold(cell, qp) + devolps;
              
          } //loop over qps
          
      } //loop over cell

  } // end of evaluateFields

//**********************************************************************
// all local functions
template<typename EvalT, typename Traits>
typename CapImplicitModel<EvalT, Traits>::ScalarT
//typename EvalT::ScalarT
CapImplicitModel<EvalT, Traits>::compute_f(Intrepid::Tensor<ScalarT> & sigma, Intrepid::Tensor<ScalarT> & alpha, ScalarT & kappa)
    {
        
        Intrepid::Tensor<ScalarT> xi = sigma - alpha;
        
        ScalarT I1 = Intrepid::trace(xi), p = I1 / 3.;
        
        Intrepid::Tensor<ScalarT> s = xi - p * Intrepid::identity<ScalarT>(3);
        
        ScalarT J2 = 0.5 * Intrepid::dotdot(s, s);
        
        ScalarT J3 = Intrepid::det(s);
        
        ScalarT Gamma = 1.0;
        if (psi != 0 && J2 != 0)
            Gamma =
            0.5
            * (1. - 3. * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)
               + (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5))
               / psi);
        
        ScalarT Ff_I1 = A - C * std::exp(B * I1) - theta * I1;
        
        ScalarT Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;
        
        ScalarT X = kappa - R * Ff_kappa;
        
        ScalarT Fc = 1.0;
        
        if ((kappa - I1) > 0 && ((X - kappa) != 0))
            Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);
        
        return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
    }
    
template<typename EvalT, typename Traits>
typename CapImplicitModel<EvalT, Traits>::DFadType
//typename EvalT::DFadType
CapImplicitModel<EvalT, Traits>::compute_f(Intrepid::Tensor<DFadType> & sigma, Intrepid::Tensor<DFadType> & alpha, DFadType & kappa)
    {
        
        Intrepid::Tensor<DFadType> xi = sigma - alpha;
        
        DFadType I1 = Intrepid::trace(xi), p = I1 / 3.;
        
        Intrepid::Tensor<DFadType> s = xi - p * Intrepid::identity<DFadType>(3);
        
        DFadType J2 = 0.5 * Intrepid::dotdot(s, s);
        
        DFadType J3 = Intrepid::det(s);
        
        DFadType Gamma = 1.0;
        if (psi != 0 && J2 != 0)
            Gamma =
            0.5
            * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)
               + (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5))
               / psi);
        
        DFadType Ff_I1 = A - C * std::exp(B * I1) - theta * I1;
        
        DFadType Ff_kappa = A - C * std::exp(B * kappa) - theta * kappa;
        
        DFadType X = kappa - R * Ff_kappa;
        
        DFadType Fc = 1.0;
        
        if ((kappa - I1) > 0 && ((X - kappa) != 0))
            Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);
        
        return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
    }
    
template<typename EvalT, typename Traits>
typename CapImplicitModel<EvalT, Traits>::D2FadType
//typename EValT::D2FadType
CapImplicitModel<EvalT, Traits>::compute_g(Intrepid::Tensor<D2FadType> & sigma, Intrepid::Tensor<D2FadType> & alpha, D2FadType & kappa)
    {
        
        Intrepid::Tensor<D2FadType> xi = sigma - alpha;
        
        D2FadType I1 = Intrepid::trace(xi), p = I1 / 3.;
        
        Intrepid::Tensor<D2FadType> s = xi - p * Intrepid::identity<D2FadType>(3);
        
        D2FadType J2 = 0.5 * Intrepid::dotdot(s, s);
        
        D2FadType J3 = Intrepid::det(s);
        
        D2FadType Gamma = 1.0;
        if (psi != 0 && J2 != 0)
            Gamma =
            0.5
            * (1. - 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5)
               + (1. + 3.0 * std::sqrt(3.0) * J3 / 2. / std::pow(J2, 1.5))
               / psi);
        
        D2FadType Ff_I1 = A - C * std::exp(L * I1) - phi * I1;
        
        D2FadType Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;
        
        D2FadType X = kappa - Q * Ff_kappa;
        
        D2FadType Fc = 1.0;
        
        if ((kappa - I1) > 0 && ((X - kappa) != 0))
            Fc = 1.0 - (I1 - kappa) * (I1 - kappa) / (X - kappa) / (X - kappa);
        
        return Gamma * Gamma * J2 - Fc * (Ff_I1 - N) * (Ff_I1 - N);
    }
    
template<typename EvalT, typename Traits>
std::vector<typename CapImplicitModel<EvalT, Traits>::ScalarT>
//std::vector<typename EvalT::ScalarT>
CapImplicitModel<EvalT,Traits>::initialize(Intrepid::Tensor<ScalarT> & sigmaVal, Intrepid::Tensor<ScalarT> & alphaVal, ScalarT & kappaVal, ScalarT & dgammaVal)
    {
        std::vector<ScalarT> XX(13);
        
        XX[0] = sigmaVal(0, 0);
        XX[1] = sigmaVal(1, 1);
        XX[2] = sigmaVal(2, 2);
        XX[3] = sigmaVal(1, 2);
        XX[4] = sigmaVal(0, 2);
        XX[5] = sigmaVal(0, 1);
        XX[6] = alphaVal(0, 0);
        XX[7] = alphaVal(1, 1);
        XX[8] = alphaVal(1, 2);
        XX[9] = alphaVal(0, 2);
        XX[10] = alphaVal(0, 1);
        XX[11] = kappaVal;
        XX[12] = dgammaVal;
        
        return XX;
    }
    
template<typename EvalT, typename Traits>
Intrepid::Tensor<typename CapImplicitModel<EvalT, Traits>::DFadType>
//Intrepid::Tensor<typename EvalT::DFadType>
CapImplicitModel<EvalT, Traits>::compute_dgdsigma(std::vector<DFadType> const & XX)
    {
        std::vector<D2FadType> D2XX(13);
        
        for (int i = 0; i < 13; ++i) {
            D2XX[i] = D2FadType(13, i, XX[i]);
        }
        
        Intrepid::Tensor<D2FadType> sigma(3), alpha(3);
        
        sigma(0, 0) = D2XX[0];
        sigma(0, 1) = D2XX[5];
        sigma(0, 2) = D2XX[4];
        sigma(1, 0) = D2XX[5];
        sigma(1, 1) = D2XX[1];
        sigma(1, 2) = D2XX[3];
        sigma(2, 0) = D2XX[4];
        sigma(2, 1) = D2XX[3];
        sigma(2, 2) = D2XX[2];
        
        alpha(0, 0) = D2XX[6];
        alpha(0, 1) = D2XX[10];
        alpha(0, 2) = D2XX[9];
        alpha(1, 0) = D2XX[10];
        alpha(1, 1) = D2XX[7];
        alpha(1, 2) = D2XX[8];
        alpha(2, 0) = D2XX[9];
        alpha(2, 1) = D2XX[8];
        alpha(2, 2) = -D2XX[6] - D2XX[7];
        
        D2FadType kappa = D2XX[11];
        
        D2FadType g = compute_g(sigma, alpha, kappa);
        
        Intrepid::Tensor<DFadType> dgdsigma(3);
        
        dgdsigma(0, 0) = g.dx(0);
        dgdsigma(0, 1) = g.dx(5);
        dgdsigma(0, 2) = g.dx(4);
        dgdsigma(1, 0) = g.dx(5);
        dgdsigma(1, 1) = g.dx(1);
        dgdsigma(1, 2) = g.dx(3);
        dgdsigma(2, 0) = g.dx(4);
        dgdsigma(2, 1) = g.dx(3);
        dgdsigma(2, 2) = g.dx(2);
        
        return dgdsigma;
    }
    
template<typename EvalT, typename Traits>
typename CapImplicitModel<EvalT, Traits>::DFadType
//typename EvalT::DFadType
CapImplicitModel<EvalT, Traits>::compute_Galpha(DFadType J2_alpha)
    {
        if (N != 0)
            return 1.0 - pow(J2_alpha, 0.5) / N;
        else
            return 0.0;
    }
    
template<typename EvalT, typename Traits>
Intrepid::Tensor<typename CapImplicitModel<EvalT, Traits>::DFadType>
//Intrepid::Tensor<typename EvalT::DFadType>
CapImplicitModel<EvalT, Traits>::compute_halpha(Intrepid::Tensor<DFadType> const & dgdsigma, DFadType const J2_alpha)
    {
        
        DFadType Galpha = compute_Galpha(J2_alpha);
        
        DFadType I1 = Intrepid::trace(dgdsigma), p = I1 / 3.0;
        
        Intrepid::Tensor<DFadType> s = dgdsigma
        - p * Intrepid::identity<DFadType>(3);
        
        //Intrepid::Tensor<DFadType, 3> halpha = calpha * Galpha * s; // * operator not defined;
        Intrepid::Tensor<DFadType> halpha(3);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                halpha(i, j) = calpha * Galpha * s(i, j);
            }
        }
        
        return halpha;
    }
    
template<typename EvalT, typename Traits>
typename CapImplicitModel<EvalT, Traits>::DFadType
//typename EvalT::DFadType
CapImplicitModel<EvalT, Traits>::compute_dedkappa(DFadType const kappa)
    {
        
        //******** use analytical expression
        ScalarT Ff_kappa0 = A - C * std::exp(L * kappa0) - phi * kappa0;
        
        ScalarT X0 = kappa0 - Q * Ff_kappa0;
        
        DFadType Ff_kappa = A - C * std::exp(L * kappa) - phi * kappa;
        
        DFadType X = kappa - Q * Ff_kappa;
        
        DFadType dedX = (D1 - 2. * D2 * (X - X0))
        * std::exp((D1 - D2 * (X - X0)) * (X - X0)) * W;
        
        DFadType dXdkappa = 1. + Q * C * L * exp(L * kappa) + Q * phi;
        
        return dedX * dXdkappa;
    }
    
template<typename EvalT, typename Traits>
typename CapImplicitModel<EvalT, Traits>::DFadType
//typename EvalT::DFadType
CapImplicitModel<EvalT, Traits>::compute_hkappa(DFadType const I1_dgdsigma, DFadType const dedkappa)
    {
        if (dedkappa != 0)
            return I1_dgdsigma / dedkappa;
        else
            return 0;
    }
    
template<typename EvalT, typename Traits>
void
CapImplicitModel<EvalT, Traits>::compute_ResidJacobian(std::vector<ScalarT> const & XXVal, std::vector<ScalarT> & R,std::vector<ScalarT> & dRdX, const Intrepid::Tensor<ScalarT> & sigmaVal,const Intrepid::Tensor<ScalarT> & alphaVal, const ScalarT & kappaVal,Intrepid::Tensor4<ScalarT> const & Celastic, bool kappa_flag)
    {
        
        std::vector<DFadType> Rfad(13);
        std::vector<DFadType> XX(13);
        std::vector<ScalarT> XXtmp(13);
        
        // initialize DFadType local unknown vector Xfad
        // Note that since Xfad is a temporary variable that gets changed within local iterations
        // when we initialize Xfad, we only pass in the values of X, NOT the system sensitivity information
        for (int i = 0; i < 13; ++i) {
            XXtmp[i] = Sacado::ScalarValue<ScalarT>::eval(XXVal[i]);
            XX[i] = DFadType(13, i, XXtmp[i]);
        }
        
        Intrepid::Tensor<DFadType> sigma(3), alpha(3);
        
        sigma(0, 0) = XX[0];
        sigma(0, 1) = XX[5];
        sigma(0, 2) = XX[4];
        sigma(1, 0) = XX[5];
        sigma(1, 1) = XX[1];
        sigma(1, 2) = XX[3];
        sigma(2, 0) = XX[4];
        sigma(2, 1) = XX[3];
        sigma(2, 2) = XX[2];
        
        alpha(0, 0) = XX[6];
        alpha(0, 1) = XX[10];
        alpha(0, 2) = XX[9];
        alpha(1, 0) = XX[10];
        alpha(1, 1) = XX[7];
        alpha(1, 2) = XX[8];
        alpha(2, 0) = XX[9];
        alpha(2, 1) = XX[8];
        alpha(2, 2) = -XX[6] - XX[7];
        
        DFadType kappa = XX[11];
        
        DFadType dgamma = XX[12];
        
        DFadType f = compute_f(sigma, alpha, kappa);
        
        Intrepid::Tensor<DFadType> dgdsigma = compute_dgdsigma(XX);
        
        DFadType J2_alpha = 0.5 * Intrepid::dotdot(alpha, alpha);
        
        Intrepid::Tensor<DFadType> halpha = compute_halpha(dgdsigma, J2_alpha);
        
        DFadType I1_dgdsigma = Intrepid::trace(dgdsigma);
        
        DFadType dedkappa = compute_dedkappa(kappa);
        
        DFadType hkappa = compute_hkappa(I1_dgdsigma, dedkappa);
        
        DFadType t;
        
        t = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                t = t + Celastic(0, 0, i, j) * dgdsigma(i, j);
            }
        }
        Rfad[0] = dgamma * t + sigma(0, 0) - sigmaVal(0, 0);
        
        t = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                t = t + Celastic(1, 1, i, j) * dgdsigma(i, j);
            }
        }
        Rfad[1] = dgamma * t + sigma(1, 1) - sigmaVal(1, 1);
        
        t = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                t = t + Celastic(2, 2, i, j) * dgdsigma(i, j);
            }
        }
        Rfad[2] = dgamma * t + sigma(2, 2) - sigmaVal(2, 2);
        
        t = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                t = t + Celastic(1, 2, i, j) * dgdsigma(i, j);
            }
        }
        Rfad[3] = dgamma * t + sigma(1, 2) - sigmaVal(1, 2);
        
        t = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                t = t + Celastic(0, 2, i, j) * dgdsigma(i, j);
            }
        }
        Rfad[4] = dgamma * t + sigma(0, 2) - sigmaVal(0, 2);
        
        t = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                t = t + Celastic(0, 1, i, j) * dgdsigma(i, j);
            }
        }
        Rfad[5] = dgamma * t + sigma(0, 1) - sigmaVal(0, 1);
        
        Rfad[6] = dgamma * halpha(0, 0) - alpha(0, 0) + alphaVal(0, 0);
        
        Rfad[7] = dgamma * halpha(1, 1) - alpha(1, 1) + alphaVal(1, 1);
        
        Rfad[8] = dgamma * halpha(1, 2) - alpha(1, 2) + alphaVal(1, 2);
        
        Rfad[9] = dgamma * halpha(0, 2) - alpha(0, 2) + alphaVal(0, 2);
        
        Rfad[10] = dgamma * halpha(0, 1) - alpha(0, 1) + alphaVal(0, 1);
        
        if (kappa_flag == false)
            Rfad[11] = dgamma * hkappa - kappa + kappaVal;
        else
            Rfad[11] = 0;
        
        // debugging
        //	if(kappa_flag == false)Rfad[11] = -dgamma * hkappa - kappa + kappaVal;
        //	else Rfad[11] = 0;
        
        Rfad[12] = f;
        
        // get ScalarT Residual
        for (int i = 0; i < 13; i++)
            R[i] = Rfad[i].val();
        
        //std::cout << "in assemble_Resid, R= " << R[0] << " " << R[1] << " " << R[2] << " " << R[3]<< std::endl;
        
        // get Jacobian
        for (int i = 0; i < 13; i++)
            for (int j = 0; j < 13; j++)
                dRdX[i + 13 * j] = Rfad[i].dx(j);
        
        if (kappa_flag == true) {
            for (int j = 0; j < 13; j++)
                dRdX[11 + 13 * j] = 0.0;
            
            dRdX[11 + 13 * 11] = 1.0;
        }
        
    }
    
} // end LCM
