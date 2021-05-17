#ifndef LANDICE_HYDROLOGY_SURFACE_WATER_INPUT_HPP
#define LANDICE_HYDROLOGY_SURFACE_WATER_INPUT_HPP

#include "Albany_Layouts.hpp"

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"

namespace LandIce
{

/** \brief Hydrology water source

    This evaluator computes the subglacial hydrology water source
    from measurements of the Surface Mass Balance
*/

template<typename EvalT, typename Traits>
class HydrologySurfaceWaterInput: public PHX::EvaluatorWithBaseImpl<Traits>
{
public:

  typedef typename EvalT::ParamScalarT ParamScalarT;

  HydrologySurfaceWaterInput (const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData,
                              PHX::FieldManager<Traits>&) {}

  void evaluateFields(typename Traits::EvalData d);

private:

  enum class InputType {
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

  bool eval_on_side;

  std::string sideSetName;  // Needed only if eval_on_side=true
  Albany::LocalSideSetInfo sideSet; // Needed only if eval_on_side=true
};

} // Namespace LandIce

#endif // LANDICE_HYDROLOGY_SURFACE_WATER_INPUT_HPP
