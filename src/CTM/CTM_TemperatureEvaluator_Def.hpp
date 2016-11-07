//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"
//
#include <Albany_DiscretizationFactory.hpp>
#include "Albany_APFDiscretization.hpp"
#include <Albany_AbstractDiscretization.hpp>


namespace CTM {

    //**********************************************************************

    template<typename EvalT, typename Traits>
    Temperature<EvalT, Traits>::
    Temperature(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
    T_(p.get<std::string>("Temperature Name"),
    dl->qp_scalar){

        // evaluated field
        this->addEvaluatedField(T_);

        std::vector<PHX::Device::size_type> dims;
        Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
        scalar_dl->dimensions(dims);
        workset_size_ = dims[0];
        num_qps_ = dims[1];

        // get temperature variable name
        Temperature_Name_ = p.get<std::string>("Temperature Name");

        this->setName("Temperature" + PHX::typeAsString<EvalT>());

    }

    //**********************************************************************

    template<typename EvalT, typename Traits>
    void Temperature<EvalT, Traits>::
    postRegistrationSetup(typename Traits::SetupData d,
            PHX::FieldManager<Traits>& fm) {
        this->utils.setFieldData(T_, fm);
    }

    //**********************************************************************

    template<typename EvalT, typename Traits>
    void Temperature<EvalT, Traits>::
    evaluateFields(typename Traits::EvalData workset) {
        
        Albany::MDArray T_qp = (*workset.stateArrayPtr)[Temperature_Name_];
        
        for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
            for (std::size_t qp = 0; qp < num_qps_; ++qp) {
                // fill Temperature
                T_(cell, qp) = T_qp(cell,qp);
            }
        }
    }

    //**********************************************************************


}

