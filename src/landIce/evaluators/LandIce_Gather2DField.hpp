//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_GATHER_2D_FIELD_HPP
#define LANDICE_GATHER_2D_FIELD_HPP

#include "Albany_Layouts.hpp"

#include "PHAL_AlbanyTraits.hpp"

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Shards_CellTopology.hpp"

namespace LandIce {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class Gather2DField : public PHX::EvaluatorWithBaseImpl<Traits>,
		                  public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  Gather2DField (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~Gather2DField () = default;

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields (typename Traits::EvalData /* d */) {
    TEUCHOS_TEST_FOR_EXCEPTION (false, std::runtime_error,
        "[Gather2DField::evaluateFields] Not yet implemented for EvalT=" + PHX::print<EvalT>() + "\n");
  }


protected:
  void check_topology (const shards::CellTopology& topo);

  using ScalarT = typename EvalT::ScalarT;
  using ref_t   = typename PHAL::Ref<ScalarT>::type;

  // Output:
  PHX::MDField<ScalarT,Cell,Node>  field2D;

  int offset; // Offset of first DOF being gathered

  int fieldLevel;
  std::string meshPart;

  bool extruded;
};

} // namespace LandIce

#endif // LANDICE_GATHER_2D_FIELD_HPP
