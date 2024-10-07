//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_FIELD_SOURCE_HPP
#define PHAL_FIELD_SOURCE_HPP

#include <string>
#include <vector>

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_Dimension.hpp"
#include "Sacado_Traits.hpp"

#include "Albany_Layouts.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Sacado_ParameterRegistration.hpp"

namespace PHAL
{

    namespace Source_Functions
    {

        // This needs to be templated to get mesh derivatives
        template <typename EvalT, typename Traits>
        class Spatial_Base
        {
        public:
            virtual ~Spatial_Base() {}
            typedef typename EvalT::MeshScalarT MeshScalarT;
            typedef typename EvalT::ScalarT ScalarT;
            virtual ScalarT evaluateFields(const std::vector<MeshScalarT> &coords) = 0;
        };

        template <typename EvalT, typename Traits>
        class Gaussian : public Spatial_Base<EvalT, Traits>,
                         public PHX::EvaluatorWithBaseImpl<Traits>,
                         public PHX::EvaluatorDerived<EvalT, Traits>
        {
        public:
            static bool check_for_existance(Teuchos::ParameterList &source_list);
            typedef typename EvalT::MeshScalarT MeshScalarT;
            typedef typename EvalT::ScalarT ScalarT;
            Gaussian(Teuchos::ParameterList &source_list, Teuchos::ParameterList &scalarParam_list, std::size_t num, PHX::FieldManager<Traits> &fm, const Teuchos::RCP<Albany::Layouts> &dl);
            virtual ~Gaussian() {}
            virtual ScalarT evaluateFields(const std::vector<MeshScalarT> &coords);

            void evaluateFields(typename Traits::EvalData /* d */){};

        private:
            ScalarT m_amplitude;
            ScalarT m_radius;
            PHX::MDField<const ScalarT> mdf_amplitude;
            PHX::MDField<const ScalarT> mdf_radius;
            int m_num;
            Teuchos::Array<double> m_centroid;
            virtual ScalarT &getValue(const std::string &n);
        };

    } // namespace Source_Functions

} // namespace PHAL
#endif
