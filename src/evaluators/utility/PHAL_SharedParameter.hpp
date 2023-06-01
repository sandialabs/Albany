#ifndef PHAL_SHARED_PARAMETER_HPP
#define PHAL_SHARED_PARAMETER_HPP 1

#include "PHAL_Dimension.hpp"
#include "Albany_SacadoTypes.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"
#include "Albany_ScalarParameterAccessors.hpp"
#include "Albany_Layouts.hpp"

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Sacado_ParameterAccessor.hpp"

#include <memory>

namespace PHAL
{

template<typename EvalT, typename Traits>
class SharedParameter : public PHX::EvaluatorWithBaseImpl<Traits>,
                        public PHX::EvaluatorDerived<EvalT, Traits>,
                        public Sacado::ParameterAccessor<EvalT, SPL_Traits>
{
private:
  std::shared_ptr<Albany::ScalarParameterAccessor<typename EvalT::ScalarT>> accessor;

public:

  typedef typename EvalT::ScalarT   ScalarT;

  SharedParameter (Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
  {
    param_name   = p.get<std::string>("Parameter Name");
    param_as_field = PHX::MDField<ScalarT,Dim>(param_name,dl->shared_param);

    Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>> accessors =
      p.get<Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>>>("Accessors");
    if (accessors->accessors.count(param_name)==0)
      accessors->accessors[param_name] = std::make_shared<Albany::ScalarParameterAccessor<ScalarT>>();

    accessor = accessors->accessors.at(param_name);

    // Never actually evaluated, but creates the evaluation tag
    this->addEvaluatedField(param_as_field);

    // Sacado-ized parameter
    Teuchos::RCP<ParamLib> paramLib = p.get<Teuchos::RCP<ParamLib>>("Parameter Library");
    this->setName("Shared Parameter " + param_name + PHX::print<EvalT>());

    const Teuchos::ParameterList* paramsList = p.get<const Teuchos::ParameterList*>("Parameters List");

    // Find the parameter in the Parameter list,
    // register as a Sacado Parameter and set the Nominal value
    bool nominalValueSet = false;
    log_parameter = false;
    if((paramsList != NULL) && paramsList->isParameter("Number Of Parameters"))
    {
      int n = paramsList->get<int>("Number Of Parameters");
      for (int i=0; (nominalValueSet==false) && i<n; ++i)
      {
        const Teuchos::ParameterList& pvi = paramsList->sublist(util::strint("Parameter",i));
        std::string parameterType = "Scalar";
        if(pvi.isParameter("Type"))
          parameterType = pvi.get<std::string>("Type");
        if (parameterType == "Distributed")
          break; // Pointless to check the remaining parameters as they are all distributed

        if (parameterType == "Scalar") {
          if (pvi.get<std::string>("Name")==param_name)
          {
            this->registerSacadoParameter(param_name, paramLib);
            if (pvi.isParameter("Nominal Value")) {
              double nom_val = pvi.get<double>("Nominal Value");
              accessor->getValue() = nom_val;
              nominalValueSet = true;
            }
            if (pvi.isParameter("Log Of Physical Parameter")) {
              log_parameter = pvi.get<bool>("Log Of Physical Parameter");
            }
          break;
          }
        }
        else { //"Vector"
          int m = pvi.get<int>("Dimension");
          for (int j=0; j<m; ++j)
          {
            const Teuchos::ParameterList& pj = pvi.sublist(util::strint("Scalar",j));
            if (pj.get<std::string>("Name")==param_name)
            {
              this->registerSacadoParameter(param_name, paramLib);
              if (pj.isParameter("Nominal Value")) {
                double nom_val = pj.get<double>("Nominal Value");
                accessor->getValue() = nom_val;
                nominalValueSet = true;
              }
              if (pj.isParameter("Log Of Physical Parameter")) {
                log_parameter = pj.get<bool>("Log Of Physical Parameter");
              }
              break;
            }
          }
        }
      }
    }

    if(!nominalValueSet) 
      accessor->getValue() = p.get<double>("Default Nominal Value");

    dummy = 0;
  }

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(param_as_field,fm);
    d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
  }

  ScalarT& getValue ()
  {
    return accessor->getValue();
  }

  ScalarT& getValue(const std::string &n)
  {
    if (n==param_name)
      return accessor->getValue();

    return dummy;
  }

  void evaluateFields(typename Traits::EvalData /*d*/)
  {
    if (log_parameter) {
      param_as_field(0) = std::exp(accessor->getValue());
    } else {
      param_as_field(0) = accessor->getValue();
    }
  }

protected:

  ScalarT              dummy;
  std::string          param_name;
  bool                 log_parameter;

  PHX::MDField<ScalarT,Dim>   param_as_field;
};

} // Namespace PHAL

#endif // PHAL_SHARED_PARAMETER_HPP
