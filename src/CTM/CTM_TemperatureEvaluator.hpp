//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef CTM_TEMPERATUREEVALUATOR_HPP
#define CTM_TEMPERATUREEVALUATOR_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"
//
#include <Albany_DiscretizationFactory.hpp>
#include "Albany_APFDiscretization.hpp"
#include <Albany_AbstractDiscretization.hpp>

namespace CTM {
    ///
    /// \brief evaluates the current temperature
    ///

    template<typename EvalT, typename Traits>
    class Temperature :
    public PHX::EvaluatorWithBaseImpl<Traits>,
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
        
        // temperature name
        std::string Temperature_Name_;
    };

}

#endif
