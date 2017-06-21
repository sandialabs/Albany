#ifndef CTM_TEMPERATURE_EVALUATOR_HPP
#define CTM_TEMPERATURE_EVALUATOR_HPP

#include <Phalanx_config.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Albany_APFDiscretization.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Array.hpp>

namespace CTM {

template<typename EvalT, typename Traits>
class Temperature
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits> {
  public:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    Temperature(const Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(
        typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

    ScalarT& getValue(const std::string &n);

  private:

    PHX::MDField<ScalarT, Cell, QuadPoint> T_;

    unsigned int num_qps_;
    unsigned int num_dims_;
    unsigned int num_nodes_;
    unsigned int workset_size_;

    std::string Temperature_Name_;
};

} // namespace CTM

#endif
