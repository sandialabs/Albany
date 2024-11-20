//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_RANDOMPHYSICALPARAMETER_HPP
#define PHAL_RANDOMPHYSICALPARAMETER_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_SharedParameter.hpp"
#include "Albany_UnivariateDistribution.hpp"

namespace PHAL {
///
/// RandomPhysicalParameterBase
///
template<typename EvalT, typename Traits>
class RandomPhysicalParameterBase : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT   ScalarT;
  //typedef ParamNameEnum             EnumType;

  RandomPhysicalParameterBase (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
  {
    std::string param_name   = p.get<std::string>("Parameter Name");
    std::string theta_name   = p.get<std::string>("Random Parameter Name");
    param_as_field = PHX::MDField<ScalarT,Dim>(param_name,dl->shared_param);
    theta_as_field = PHX::MDField<ScalarT,Dim>(theta_name, dl->shared_param);

    // Never actually evaluated, but creates the evaluation tag
    this->addEvaluatedField(param_as_field);
    this->addDependentField(theta_as_field);
    this->setName("Random Parameter " + param_name + PHX::print<EvalT>());

    const Teuchos::ParameterList* distributionList = p.get<const Teuchos::ParameterList*>("Distribution");

    if (distributionList->get<std::string>("Name") == "Normal")
      distribution = Teuchos::rcp<Albany::UnivariatDistribution>(new Albany::NormalDistribution(*distributionList));
    if (distributionList->get<std::string>("Name") == "LogNormal")
      distribution = Teuchos::rcp<Albany::UnivariatDistribution>(new Albany::LogNormalDistribution(*distributionList));
    if (distributionList->get<std::string>("Name") == "Uniform")
      distribution = Teuchos::rcp<Albany::UnivariatDistribution>(new Albany::UniformDistribution(*distributionList));
  }

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(param_as_field,fm);
    d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
  }

protected:
  PHX::MDField<ScalarT,Dim>   param_as_field;
  PHX::MDField<const ScalarT,Dim>   theta_as_field;
  Teuchos::RCP<Albany::UnivariatDistribution> distribution;
};

template<typename EvalT, typename Traits> class RandomPhysicalParameter;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class RandomPhysicalParameter<PHAL::AlbanyTraits::Residual,Traits>
  : public RandomPhysicalParameterBase<PHAL::AlbanyTraits::Residual, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT   ScalarT;

    RandomPhysicalParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::Residual, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::Residual, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData /* d */)
    {
      this->param_as_field(0) = this->distribution->fromNormalMapping(this->theta_as_field(0));

      std::cout << " RandomPhysicalParameter<PHAL::AlbanyTraits::Residual>::evaluateFields() "
        << this->param_as_field(0) << " theta " << this->theta_as_field(0) << std::endl;
    }
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class RandomPhysicalParameter<PHAL::AlbanyTraits::Jacobian,Traits>
  : public RandomPhysicalParameterBase<PHAL::AlbanyTraits::Jacobian, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT   ScalarT;

    RandomPhysicalParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::Jacobian, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::Jacobian, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData /* d */)
    {
      double theta_val = this->theta_as_field(0).val();
      this->param_as_field(0).val() = this->distribution->fromNormalMapping(theta_val);

      std::cout << " RandomPhysicalParameter<PHAL::AlbanyTraits::Jacobian>::evaluateFields() "
        << this->param_as_field(0) << " theta " << this->theta_as_field(0) << std::endl;
    }
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class RandomPhysicalParameter<PHAL::AlbanyTraits::Tangent,Traits>
  : public RandomPhysicalParameterBase<PHAL::AlbanyTraits::Tangent, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT   ScalarT;

    RandomPhysicalParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::Tangent, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::Tangent, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData /* d */)
    {
      double theta_val = this->theta_as_field(0).val();
      this->param_as_field(0).val() = this->distribution->fromNormalMapping(theta_val);
      double v_dx = this->distribution->fromNormalMapping_dx(theta_val);

      int size = this->theta_as_field(0).size();

      for (int i = 0; i < size; ++i)
        this->param_as_field(0).fastAccessDx(i) = this->theta_as_field(0).fastAccessDx(i) * v_dx;

      std::cout << " RandomPhysicalParameter<PHAL::AlbanyTraits::Tangent>::evaluateFields() "
        << this->param_as_field(0) << " theta " << this->theta_as_field(0) << std::endl;
    }
};

// **************************************************************
// DistParamDeriv
// **************************************************************
template<typename Traits>
class RandomPhysicalParameter<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public RandomPhysicalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT   ScalarT;

    RandomPhysicalParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData /* d */)
    {
      double theta_val = this->theta_as_field(0).val();
      this->param_as_field(0).val() = this->distribution->fromNormalMapping(theta_val);
      double v_dx = this->distribution->fromNormalMapping_dx(theta_val);

      int size = this->theta_as_field(0).size();

      for (int i = 0; i < size; ++i)
        this->param_as_field(0).fastAccessDx(i) = this->theta_as_field(0).fastAccessDx(i) * v_dx;

      std::cout << " RandomPhysicalParameter<PHAL::AlbanyTraits::DistParamDeriv>::evaluateFields() "
        << this->param_as_field(0) << " theta " << this->theta_as_field(0) << std::endl;
    }
};

// **************************************************************
// HessianVec
// **************************************************************
template<typename Traits>
class RandomPhysicalParameter<PHAL::AlbanyTraits::HessianVec,Traits>
  : public RandomPhysicalParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>  {

  public:
    typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT   ScalarT;
 
    RandomPhysicalParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>(p, dl)
     {
     }

    void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
    {
      RandomPhysicalParameterBase<PHAL::AlbanyTraits::HessianVec, Traits>::
        postRegistrationSetup(d, fm);
    }

    void evaluateFields(typename Traits::EvalData /* d */)
    {
      double theta_val = this->theta_as_field(0).val().val();
      this->param_as_field(0).val().val() = this->distribution->fromNormalMapping(theta_val);
      double v_dx = this->distribution->fromNormalMapping_dx(theta_val);
      double v_dx_dx = this->distribution->fromNormalMapping_dx_dx(theta_val);

      int size_1 = this->theta_as_field(0).size();
      int size_2 = this->theta_as_field(0).val().size();

      for (int i_1 = 0; i_1 < size_1; ++i_1) {
        this->param_as_field(0).fastAccessDx(i_1).val() = this->theta_as_field(0).fastAccessDx(i_1).val() * v_dx;
        for (int i_2 = 0; i_2 < size_2; ++i_2)
          this->param_as_field(0).fastAccessDx(i_1).fastAccessDx(i_2) = this->theta_as_field(0).fastAccessDx(i_1).fastAccessDx(i_2) * v_dx 
                                                                        + this->theta_as_field(0).fastAccessDx(i_1).val() * v_dx_dx;
      }

      for (int i_2 = 0; i_2 < size_2; ++i_2)
        this->param_as_field(0).val().fastAccessDx(i_2) = this->theta_as_field(0).val().fastAccessDx(i_2) * v_dx;
      

      std::cout << " RandomPhysicalParameter<PHAL::AlbanyTraits::HessianVec>::evaluateFields() "
        << this->param_as_field(0) << " theta " << this->theta_as_field(0) << std::endl;
    }
};

}  // Namespace PHAL

#endif  // PHAL_RANDOMPHYSICALPARAMETER_HPP
