#ifndef LANDICE_HYDROLOGY_SURFACE_WATER_INPUT_HPP
#define LANDICE_HYDROLOGY_SURFACE_WATER_INPUT_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace LandIce
{

/** \brief Hydrology water source

    This evaluator computes the subglacial hydrology water source
    from measurements of the Surface Mass Balance
*/

template<typename EvalT, typename Traits, bool OnSide>
class HydrologySurfaceWaterInput: public PHX::EvaluatorWithBaseImpl<Traits>,
                                  public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ParamScalarT ParamScalarT;

  HydrologySurfaceWaterInput (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  enum class InputType {
    GIVEN_VALUE,
    GIVEN_FIELD,
    SMB_APPROX
  };

  void evaluateFieldsCell(typename Traits::EvalData d);
  void evaluateFieldsSide(typename Traits::EvalData d);

  // Input(s):
  PHX::MDField<const ParamScalarT>  smb;          // surface mass balance; only for SMB_APPROX
  PHX::MDField<const ParamScalarT>  sh;           // surface height: only for MATH_EXPR

  // Output:
  PHX::MDField<ParamScalarT>        omega;

  InputType input_type;

  unsigned int numNodes;

  double omega_val;

  std::string sideSetName;  // Needed only if OnSide=true
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_SURFACE_WATER_INPUT_HPP
