//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "PHAL_Utilities.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

namespace AMP {

    //**********************************************************************

    template<typename EvalT, typename Traits>
    EnergyDot<EvalT, Traits>::
    EnergyDot(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
    T_              (p.get<std::string>("Temperature Name"),
                    dl->qp_scalar),
    T_dot_          (p.get<std::string>("Temperature Time Derivative Name"),
                    dl->qp_scalar),
    time_           (p.get<std::string>("Time Name"),
                    dl->workset_scalar),
    psi_            (p.get<std::string>("Psi Name"),
                    dl->qp_scalar),
    phi_            (p.get<std::string>("Phi Name"),
                    dl->qp_scalar),
    phi_dot_        (p.get<std::string>("Phi Dot Name"),
		     dl->qp_scalar),
    rho_Cp_         (p.get<std::string>("Rho Cp Name"),
                    dl->qp_scalar),
    deltaTime_      (p.get<std::string>("Delta Time Name"),
                    dl->workset_scalar),
    energyDot_      (p.get<std::string>("Energy Rate Name"),
                    dl->qp_scalar) {
        
      // dependent field
        this->addDependentField(T_);
        this->addDependentField(T_dot_);
        this->addDependentField(phi_);
        this->addDependentField(psi_);
        this->addDependentField(rho_Cp_);
        this->addDependentField(time_);
        this->addDependentField(deltaTime_);

	// evaluated field
        this->addEvaluatedField(energyDot_);
	this->addEvaluatedField(phi_dot_);

        std::vector<PHX::Device::size_type> dims;
        Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
        scalar_dl->dimensions(dims);
        workset_size_ = dims[0];
        num_qps_ = dims[1];

	// get temperature old variable name
        Temperature_Name_ = p.get<std::string>("Temperature Name") + "_old";
	// Get phi old variable name
	Phi_old_name_ =  p.get<std::string>("Phi Name") + "_old";


        // Only verify Change Phase Parameter list because initial Phi already
        // verified inside Phi evaluator (I hope so)
        Teuchos::ParameterList* cond_list =
                p.get<Teuchos::ParameterList*>("Phase Change Parameter List");

        Teuchos::RCP<const Teuchos::ParameterList> reflist =
                this->getValidEnergyDotParameters();

        cond_list->validateParameters(*reflist, 0,
                Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);


        // Get volumetric heat capacity
        Cl_ = cond_list->get<double>("Volumetric Heat Capacity Liquid Value", 5.95e6);
        // Get latent heat value
        L_ = cond_list->get<double>("Latent Heat Value", 2.18e9);


        // later we need to verify that user have input proper values. Not done
        // now! But do it soon!


        cond_list = p.get<Teuchos::ParameterList*>("Initial Phi Parameter List");

        // Get melting temperature
        Tm_ = cond_list->get("Melting Temperature Value", 1700.0);

        // Get delta temperature value
        Tc_ = cond_list->get("delta Temperature Value", 50.0); 
        
        Temperature_Name_ = p.get<std::string>("Temperature Name") + "_old";

        this->setName("EnergyDot" + PHX::typeAsString<EvalT>());

    }

    //**********************************************************************

    template<typename EvalT, typename Traits>
    void EnergyDot<EvalT, Traits>::
    postRegistrationSetup(typename Traits::SetupData d,
            PHX::FieldManager<Traits>& fm) {
        this->utils.setFieldData(T_, fm);
        this->utils.setFieldData(T_dot_, fm);
        this->utils.setFieldData(time_, fm);
        this->utils.setFieldData(deltaTime_, fm);
        this->utils.setFieldData(phi_, fm);
	this->utils.setFieldData(phi_dot_, fm);
        this->utils.setFieldData(psi_, fm);
        this->utils.setFieldData(rho_Cp_, fm);
        this->utils.setFieldData(energyDot_, fm);
    }

    //**********************************************************************
template<typename EvalT, typename Traits>
void EnergyDot<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
    //current time
    // time step
    ScalarT dt = deltaTime_(0);

    typedef Intrepid2::FunctionSpaceTools<PHX::Device> FST;

    if (dt == 0.0) dt = 1.0e-15;
    //grab old temperature
    Albany::MDArray T_old = (*workset.stateArrayPtr)[Temperature_Name_];

    // grab old value opf phi
     Albany::MDArray phi_old = (*workset.stateArrayPtr)[Phi_old_name_];

    // Compute Temp rate

    // temporal variables
    // store value of phi at Gauss point, i.e., phi = phi_(cell,qp)
    ScalarT phi;
    // Store value of volumetric heat capacity at Gauss point, i.e., Cs = Rhp_Cp_(cell,qp)
    ScalarT Cs;
    // Volumetric heat capacity of solid. For now same as powder.
    ScalarT Cd;
    // Variable used to store dp/dphi = 30*phi^2*(1-2*phi+phi^2)
    ScalarT dpdphi;
    // Variable used to store time derivative of phi
    //ScalarT phi_dot;
    // Variable used to compute p = phi^3 * (10-15*phi+6*phi^2)
    ScalarT p;

    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
    {
        for (std::size_t qp = 0; qp < num_qps_; ++qp)
        {

            // compute dT/dt using finite difference
            T_dot_(cell, qp) = (T_(cell, qp) - T_old(cell, qp)) / dt;

            // compute dp/dphi
            phi = phi_(cell, qp);
            dpdphi = 30.0 * phi * phi * (1.0 - 2.0 * phi + phi * phi);

            // compute phi_dot
            //phi_dot = (1.0 / (2.0 * Tc_)) * std::pow(std::cosh((T_(cell, qp) - Tm_) / Tc_), -2.0) * T_dot_(cell, qp);
	    phi_dot_(cell,qp) = ( phi_(cell,qp) - phi_old(cell,qp) ) / dt;

            // compute energy dot
            Cs = rho_Cp_(cell, qp);
            Cd = Cs;
            // p
            p = phi * phi * phi * (10.0 - 15.0 * phi + 6.0 * phi * phi);
            energyDot_(cell, qp) = (Cs + p * (Cl_ - Cs)) * T_dot_(cell, qp) +
	      dpdphi * (L_ + (Cl_ - Cs) * (T_(cell, qp) - Tm_)) * phi_dot_(cell,qp);
		
        }
    }
}

    //**********************************************************************

    //**********************************************************************

    template<typename EvalT, typename Traits>
    Teuchos::RCP<const Teuchos::ParameterList>
    EnergyDot<EvalT, Traits>::
    getValidEnergyDotParameters() const {
        Teuchos::RCP<Teuchos::ParameterList> valid_pl =
                rcp(new Teuchos::ParameterList("Valid Energy Dot Params"));

        valid_pl->set<double>("Volumetric Heat Capacity Liquid Value", 5.95e6);

        valid_pl->set<double>("Latent Heat Value", 2.18e9);

        return valid_pl;
    }

    //**********************************************************************

}
