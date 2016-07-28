/*
 * FELIX_BasalMeltRate_Def.hpp
 *
 *  Created on: Jun 16, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{


template<typename EvalT, typename Traits, typename VelocityType>
BasalMeltRate<EvalT,Traits,VelocityType>::
BasalMeltRate(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl_basal):
	omega				(p.get<std::string> ("Omega Side Variable Name"),dl_basal->node_scalar),
	geoFluxHeat			(p.get<std::string> ("Geotermal Flux Side Variable Name"),dl_basal->node_scalar),
	velocity			(p.get<std::string> ("Velocity Side Variable Name"),dl_basal->node_vector),
	beta				(p.get<std::string> ("Basal Friction Coefficient Side Variable Name"),dl_basal->node_scalar),
	EnthalpyHs			(p.get<std::string> ("Enthalpy Hs Side Variable Name"),dl_basal->node_scalar),
	Enthalpy			(p.get<std::string> ("Enthalpy Side Variable Name"),dl_basal->node_scalar),
	basalMeltRate		(p.get<std::string> ("Basal Melt Rate Variable Name"),dl_basal->node_scalar),
    homotopy			(p.get<std::string> ("Continuation Parameter Name"),dl_basal->shared_param)
{
	this->addDependentField(omega);
	this->addDependentField(geoFluxHeat);
	this->addDependentField(velocity);
	this->addDependentField(beta);
	this->addDependentField(EnthalpyHs);
	this->addDependentField(Enthalpy);
	this->addDependentField(homotopy);

	this->addEvaluatedField(basalMeltRate);
	this->setName("Basal Melt Rate");

	std::vector<PHX::DataLayout::size_type> dims;
	dl_basal->node_qp_gradient->dimensions(dims);
	int numSides = dims[1];
	numSideNodes = dims[2];
	sideDim      = dims[4];
	numCellNodes = basalMeltRate.fieldTag().dataLayout().dimension(1);

	basalSideName = p.get<std::string> ("Side Set Name");

	Teuchos::ParameterList* physics_list = p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
	rho_w = physics_list->get("Water Density", 1000.0);
	rho_i = physics_list->get("Ice Density", 910.0);
	L = physics_list->get("Latent heat of fusion", 3e5);

	k_0 = physics_list->get("Permeability factor", 0.0);
	eta_w = physics_list->get("Viscosity of water", 0.0018);
	g = physics_list->get("Gravity Acceleration", 9.8);
	alpha_om = physics_list->get("Omega exponent alpha", 2.0);
}

template<typename EvalT, typename Traits, typename VelocityType>
void BasalMeltRate<EvalT,Traits,VelocityType>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
	this->utils.setFieldData(omega,fm);
	this->utils.setFieldData(geoFluxHeat,fm);
	this->utils.setFieldData(velocity,fm);
	this->utils.setFieldData(beta,fm);
	this->utils.setFieldData(EnthalpyHs,fm);
	this->utils.setFieldData(Enthalpy,fm);
	this->utils.setFieldData(homotopy,fm);
	this->utils.setFieldData(basalMeltRate,fm);
}

template<typename EvalT, typename Traits, typename VelocityType>
void BasalMeltRate<EvalT,Traits,VelocityType>::
evaluateFields(typename Traits::EvalData d)
{
	TEUCHOS_TEST_FOR_EXCEPTION (d.sideSets==Teuchos::null, std::runtime_error,
	                            "Side sets defined in input file but not properly specified on the mesh.\n");
	int vecDimFO = 2;
	double pi = atan(1.) * 4.;
	ScalarT hom = homotopy(0);
	ScalarT alpha = pow(10.0, -8.0 + hom*10);
	double scaling = pow(10.0,8.0) / 3.171;
	ScalarT omegaExp;

	if (d.sideSets->find(basalSideName) != d.sideSets->end())
	{
		const std::vector<Albany::SideStruct>& sideSet = d.sideSets->at(basalSideName);
	    for (auto const& it_side : sideSet)
	    {
	    	// Get the local data of side and cell
	    	const int cell = it_side.elem_LID;
	    	const int side = it_side.side_local_id;

	    	for (int node = 0; node < numSideNodes; ++node)
	    	{
    			ScalarT scale = - atan(alpha * (Enthalpy(cell,side,node) - EnthalpyHs(cell,side,node)))/pi + 0.5;
    			ScalarT basalHeat = 0.;
    			for (int dim = 0; dim < vecDimFO; dim++)
    				basalHeat += (1./(3.154*pow(10.0,4.0))) * beta(cell,side,node) * velocity(cell,side,node,dim) * velocity(cell,side,node,dim);  // check dimensions

    			//std::cout << "omega = " << omega(cell,side,node) << "\n";

    			omegaExp = pow(omega(cell,side,node),alpha_om);

    			basalMeltRate(cell,side,node) = scaling*( ((1 - scale)*( basalHeat + geoFluxHeat(cell,side,node) ) / ((1 - rho_w/rho_i*omega(cell,side,node))*L*rho_w)) -
    											k_0 * (rho_w - rho_i) * g / eta_w * omegaExp );
	    	}
	    }
	}
}


} //namespace FELIX


