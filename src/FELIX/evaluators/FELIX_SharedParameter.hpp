#ifndef FELIX_SHARED_PARAMETER_HPP
#define FELIX_SHARED_PARAMETER_HPP 1

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Sacado_ParameterAccessor.hpp"

#include "Albany_Utils.hpp"

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

  static void setNominalValue (const Teuchos::ParameterList& p, double default_value);

  static ScalarT getValue ()
  {
    return value;
  }

  ScalarT& getValue(const std::string &n)
  {
	//std::cout << "we! " << value << std::endl;
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
  static std::string          param_name;

  PHX::MDField<ScalarT,Dim>   param_as_field;
};

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
void SharedParameter<EvalT,Traits,ParamNameEnum,ParamName>::
setNominalValue (const Teuchos::ParameterList& p, double default_value)
{
  // First we scan the Parameter list to see if this parameter is listed in it,
  // in which case we use the nominal value.
  bool found = false;
  if (p.isParameter("Number of Parameter Vectors"))
  {
    int n = p.get<int>("Number of Parameter Vectors");
    for (int i=0; (found==false) && i<n; ++i)
    {
      const Teuchos::ParameterList& pvi = p.sublist(Albany::strint("Parameter Vector",i));
      if (!pvi.isParameter("Nominal Values"))
        continue; // Pointless to check the parameter names, since we don't have nominal values

      int m = pvi.get<int>("Number");
      for (int j=0; j<m; ++j)
      {
        if (pvi.get<std::string>(Albany::strint("Parameter",j))==param_name)
        {
          Teuchos::Array<double> nom_vals = pvi.get<Teuchos::Array<double>>("Nominal Values");
          value = nom_vals[j];
          found = true;
          break;
        }
      }
    }
  }
  else if (p.isParameter("Number") && !p.isParameter("Nominal Values"))
  {
    int m = p.get<int>("Number");
    for (int j=0; j<m; ++j)
    {
      if (p.get<std::string>(Albany::strint("Parameter",j))==param_name)
      {
        Teuchos::Array<double> nom_vals = p.get<Teuchos::Array<double>>("Nominal Values");
        value = nom_vals[j];
        found = true;
        break;
      }
    }
  }

  if (!found)
  {
    value = default_value;
  }

  dummy = 0;
}

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
typename EvalT::ScalarT SharedParameter<EvalT,Traits,ParamNameEnum,ParamName>::value;

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
typename EvalT::ScalarT SharedParameter<EvalT,Traits,ParamNameEnum,ParamName>::dummy;

template<typename EvalT, typename Traits, typename ParamNameEnum, ParamNameEnum ParamName>
std::string SharedParameter<EvalT,Traits,ParamNameEnum,ParamName>::param_name;

} // Namespace FELIX

#endif // FELIX_SHARED_PARAMETER_HPP
