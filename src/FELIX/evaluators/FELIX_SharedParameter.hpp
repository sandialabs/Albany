#ifndef FELIX_SHARED_PARAMETER_HPP
#define FELIX_SHARED_PARAMETER_HPP 1

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace FELIX
{

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
class SharedParameter : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits>,
                        public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
public:

  typedef typename EvalT::ScalarT   ScalarT;
  typedef ParamNameEnum             EnumType;

  SharedParameter (const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
  {
    param_name   = p.get<std::string>("Parameter Name");
    param_as_field = PHX::MDField<ScalarT,Dim>(param_name,dl->shared_param);

    // Never actually evaluated, but creates the evaluation tag
    this->addEvaluatedField(param_as_field);

    // Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
    this->registerSacadoParameter(param_name, paramLib);
    this->setName("Shared Parameter " + param_name + PHX::typeAsString<EvalT>());
  }

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(param_as_field,fm);
  }

  static void setNominalValue (double val)
  {
    value = val;
    dummy = 0;
  }

  static ScalarT getValue ()
  {
    return value;
  }

  ScalarT& getValue(const std::string &n)
  {
    if (n==param_name)
      return value;
    return dummy;
  }

  void evaluateFields(typename Traits::EvalData /*d*/)
  {
    param_as_field(0) = value;
  }

protected:

  static ScalarT              value;
  static ScalarT              dummy;

  std::string                 param_name;
  PHX::MDField<ScalarT,Dim>   param_as_field;
};

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
typename EvalT::ScalarT SharedParameter<EvalT,Traits,ParamNameEnum,ParamName>::value;

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
typename EvalT::ScalarT SharedParameter<EvalT,Traits,ParamNameEnum,ParamName>::dummy;
} // Namespace FELIX

#endif // FELIX_SHARED_PARAMETER_HPP
