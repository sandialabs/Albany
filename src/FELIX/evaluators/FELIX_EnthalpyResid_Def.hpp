/*
 * FELIX_EnthalpyResid_Def.hpp
 *
 *  Created on: May 11, 2016
 *      Author: abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX
{

double distance (const double& x0, const double& x1, const double& x2,
				 const double& y0, const double& y1, const double& y2)
{
	const double d = std::sqrt((x0-y0)*(x0-y0) +
                               (x1-y1)*(x1-y1) +
                               (x2-y2)*(x2-y2));

	return d;
}


template<typename EvalT, typename Traits, typename VelocityType>
EnthalpyResid<EvalT,Traits,VelocityType>::
EnthalpyResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	wBF      		(p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
	wGradBF  		(p.get<std::string> ("Weighted Gradient BF Variable Name"),dl->node_qp_gradient),
	Enthalpy        (p.get<std::string> ("Enthalpy QP Variable Name"), dl->qp_scalar),
	EnthalpyGrad    (p.get<std::string> ("Enthalpy Gradient QP Variable Name"), dl->qp_gradient),
	EnthalpyHs		(p.get<std::string> ("Enthalpy Hs QP Variable Name"), dl->qp_scalar ),
	Velocity		(p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector),
	verticalVel		(p.get<std::string> ("Vertical Velocity QP Variable Name"),  dl->qp_scalar),
    coordVec 		(p.get<std::string> ("Coordinate Vector Name"),dl->vertices_vector),
	meltTempGrad	(p.get<std::string> ("Melting Temperature Gradient QP Variable Name"), dl->qp_gradient),
	omega			(p.get<std::string> ("Liquid Water Fraction QP Variable Name"), dl->qp_scalar ),
	omegaGrad		(p.get<std::string> ("Liquid Water Fraction Gradient QP Variable Name"), dl->qp_gradient ),
	Residual 		(p.get<std::string> ("Residual Variable Name"), dl->node_scalar),
    homotopy		(p.get<std::string> ("Continuation Parameter Name"), dl->shared_param)
{
	Teuchos::RCP<PHX::DataLayout> vector_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
	std::vector<PHX::Device::size_type> dims;
	vector_dl->dimensions(dims);
	numNodes = dims[1];
	numQPs   = dims[2];
	vecDimFO = 2;

	Teuchos::ParameterList* SUPG_list = p.get<Teuchos::ParameterList*>("SUPG Settings");
	haveSUPG = SUPG_list->get("Have SUPG Stabilization", false);
	delta = SUPG_list->get("Parameter Delta", 0.1);

	needsDiss = p.get<bool>("Needs Dissipation");
	needsBasFric = p.get<bool>("Needs Basal Friction");

	this->addDependentField(Enthalpy);
	this->addDependentField(EnthalpyGrad);
	this->addDependentField(EnthalpyHs);
	this->addDependentField(wBF);
	this->addDependentField(wGradBF);
	this->addDependentField(Velocity);
	this->addDependentField(verticalVel);
	this->addDependentField(coordVec);
	this->addDependentField(meltTempGrad);
	this->addDependentField(omega);
	this->addDependentField(omegaGrad);
	this->addDependentField(homotopy);

	if (needsDiss)
	{
		diss = PHX::MDField<ScalarT,Cell,QuadPoint>(p.get<std::string> ("Dissipation QP Variable Name"),dl->qp_scalar);
		this->addDependentField(diss);
	}

	if (needsBasFric)
	{
		basalFricHeat = PHX::MDField<ScalarT,Cell,QuadPoint>(p.get<std::string> ("Basal Friction Heat QP Variable Name"),dl->qp_scalar);
		this->addDependentField(basalFricHeat);

		if(haveSUPG)
		{
			basalFricHeatSUPG = PHX::MDField<ScalarT,Cell,QuadPoint>(p.get<std::string> ("Basal Friction Heat QP SUPG Variable Name"),dl->qp_scalar);
			this->addDependentField(basalFricHeatSUPG);
		}
	}

	geoFluxHeat = PHX::MDField<ScalarT,Cell,QuadPoint>(p.get<std::string> ("Geotermal Flux Heat QP Variable Name"),dl->qp_scalar);
	this->addDependentField(geoFluxHeat);

	if(haveSUPG)
	{
		geoFluxHeatSUPG = PHX::MDField<ScalarT,Cell,QuadPoint>(p.get<std::string> ("Geotermal Flux Heat QP SUPG Variable Name"),dl->qp_scalar);
		this->addDependentField(geoFluxHeatSUPG);
	}

	this->addEvaluatedField(Residual);
	this->setName("EnthalpyResid");

	Teuchos::ParameterList* physics_list = p.get<Teuchos::ParameterList*>("FELIX Physical Parameters");
	rho_i = physics_list->get("Ice Density", 910.0);
	rho_w = physics_list->get("Water Density", 1000.0);

	k = physics_list->get("Conductivity of ice", 1.0);
	k *= 0.001; //scaling needs to be done to match dimensions
	c = physics_list->get("Heat capacity of ice", 2000.0);
	K_i = k / (rho_i * c);

	nu = physics_list->get("Diffusivity temperate ice", 0.000000011);
	nu *= 0.001; //scaling needs to be done to match dimensions

	k_0 = physics_list->get("Permeability factor", 0.1);
	eta_w = physics_list->get("Viscosity of water", 0.0018);
	g = physics_list->get("Gravity Acceleration", 9.8);
	L = physics_list->get("Latent heat of fusion", 3e5);
	alpha_om = physics_list->get("Omega exponent alpha", 2.0);

	drainage_coeff = g * rho_w * L * k_0 * (rho_w - rho_i) / eta_w;

	/*
	std::cout << "PARAMETERS:" << "\n" << "rho_i = " << rho_i << "\n"
			<< "rho_w = " << rho_w << "\n"
			<< "k = " << k << "\n"
			<< "c = " << c << "\n"
			<< "K_i = " << K_i << "\n"
			<< "nu = " << nu << "\n"
			<< "k_0 = " << k_0 << "\n"
			<< "eta_w = " << eta_w << "\n"
			<< "g = " << g << "\n"
			<< "L = " << L << "\n"
			<< "drainage_coeff = " << drainage_coeff << "\n"

	 */

	//alpha = physics_list->get("alpha", 1.0);

	printedAlpha = -1.0;
}

template<typename EvalT, typename Traits, typename VelocityType>
void EnthalpyResid<EvalT,Traits,VelocityType>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Enthalpy,fm);
  this->utils.setFieldData(EnthalpyGrad,fm);
  this->utils.setFieldData(EnthalpyHs,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(Velocity,fm);
  this->utils.setFieldData(verticalVel,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(meltTempGrad,fm);
  this->utils.setFieldData(omega,fm);
  this->utils.setFieldData(omegaGrad,fm);
  this->utils.setFieldData(homotopy,fm);

  if (needsDiss)
	  this->utils.setFieldData(diss,fm);

  if (needsBasFric)
  {
	  this->utils.setFieldData(basalFricHeat,fm);
	  if(haveSUPG)
		  this->utils.setFieldData(basalFricHeatSUPG,fm);
  }

  this->utils.setFieldData(geoFluxHeat,fm);
  if(haveSUPG)
	  this->utils.setFieldData(geoFluxHeatSUPG,fm);

  this->utils.setFieldData(Residual,fm);
}

template<typename EvalT, typename Traits, typename VelocityType>
void EnthalpyResid<EvalT,Traits,VelocityType>::
evaluateFields(typename Traits::EvalData d)
{
	double scaling = 0.317057705 * pow(10.0,-7.0);
	ScalarT K;
	double pi = atan(1.) * 4.;
	ScalarT hom = homotopy(0);
	ScalarT alpha = pow(10.0, -8.0 + hom*10);

    if (std::fabs(printedAlpha - alpha) > 0.0001*alpha)
    {
        std::cout << "[Diffusivity] alpha = " << alpha << "\n";
        //std::cout << "[Homotopy param] h = " << hom << "\n";
        printedAlpha = alpha;
    }

	for (std::size_t cell = 0; cell < d.numCells; ++cell)
    	for (std::size_t node = 0; node < numNodes; ++node)
    		Residual(cell,node) = 0.0;

    if (needsDiss)	// this term is always in no matter the bc at the base is
    {
        for (std::size_t cell = 0; cell < d.numCells; ++cell)
        {
        	for (std::size_t node = 0; node < numNodes; ++node)
        	{
        		for (std::size_t qp = 0; qp < numQPs; ++qp)
        		{
        			Residual(cell,node) -= diss(cell,qp)*wBF(cell,node,qp);
        		}
        	}
        }
    }

    if (needsBasFric)
    {
    	for (std::size_t cell = 0; cell < d.numCells; ++cell)
    	{
    		for (std::size_t node = 0; node < numNodes; ++node)
    		{
        		for (std::size_t qp = 0; qp < numQPs; ++qp)
        		{
        			// Modify here if you want to impose different basal BC. NB: in case of temperate ice, we disregard the extra boundary term related to the gradient of the T_m. You might want to reconsider this in the future
        			ScalarT scale = - atan(alpha * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)))/pi + 0.5;
        			Residual(cell,node) -= ( basalFricHeat(cell,qp) + geoFluxHeat(cell,qp) ) * scale;  //go to zero in temperate region
        		}
        	}
    	}
    }
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
    	for (std::size_t node = 0; node < numNodes; ++node)
    	{
    		for (std::size_t qp = 0; qp < numQPs; ++qp)
    		{
    			K = - (K_i - nu)/pi * atan(alpha * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp))) + (K_i + nu)/2;
				Residual(cell,node) += K * (EnthalpyGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
									   EnthalpyGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
									   EnthalpyGrad(cell,qp,2)*wGradBF(cell,node,qp,2)) +
					 				   scaling * (Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
									   Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1) + verticalVel(cell,qp)*EnthalpyGrad(cell,qp,2))*wBF(cell,node,qp);

				//if ( Enthalpy(cell,qp) >= EnthalpyHs(cell,qp) ) // if the ice is temperate
    				ScalarT scale = - atan(alpha * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)))/pi + 0.5;
					Residual(cell,node) += (1 - scale) * 0.001*(k - rho_i*c*nu) * (meltTempGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
										   meltTempGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
										   meltTempGrad(cell,qp,2)*wGradBF(cell,node,qp,2)) -
										   (1 - scale) * drainage_coeff*alpha_om*pow(omega(cell,qp),alpha_om-1)*omegaGrad(cell,qp,2)*wBF(cell,node,qp);
				//std::cout << "omega = " << omega(cell,qp) << "\n";
			}
        }
    }

    if (haveSUPG)
    {
    	VelocityType vmax = 0.0;
    	ParamScalarT diam = 0.0;
    	ScalarT wSUPG = 0.0;
    	double pow10 = pow(10.0,10.0);

		// compute the max norm of the velocity
    	for (std::size_t cell=0; cell < d.numCells; ++cell)
		{
			for (std::size_t qp = 0; qp < numQPs; ++qp)
			{
				for (std::size_t i = 0; i < vecDimFO; i++)
				{
					vmax = std::max(vmax,std::fabs(Velocity(cell,qp,i)));

					if (vmax == 0)
						vmax = 1e-3;
				}
			}

    		for (std::size_t i = 0; i < numNodes-1; ++i)
    		{
        		for (std::size_t j = i + 1; j < numNodes; ++j)
        		{
					diam = std::max(diam,distance(coordVec(cell,i,0),coordVec(cell,i,1),coordVec(cell,i,2),
												  coordVec(cell,j,0),coordVec(cell,j,1),coordVec(cell,j,2)));
        		}
        	}
        	for (std::size_t node=0; node < numNodes; ++node)
        	{
				for (std::size_t qp=0; qp < numQPs; ++qp)
				{
					wSUPG = (1/(3.154 * pow10)) *
							(Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) + Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) + verticalVel(cell,qp) * wGradBF(cell,node,qp,2));
					//std::cout << " vmax = " << vmax << std::endl;
					/*
					Residual(cell,node) += (delta*diam/vmax*(3.154 * pow10))*(scaling * Velocity(cell,qp,0) * EnthalpyGrad(cell,qp,0) * (1/(3.154 * pow10)) * Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +
	    			   					  	scaling * Velocity(cell,qp,1) * EnthalpyGrad(cell,qp,1) * (1/(3.154 * pow10)) * Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) +
										    scaling * verticalVel(cell,qp) * EnthalpyGrad(cell,qp,2) * (1/(3.154 * pow10)) * verticalVel(cell,qp) * wGradBF(cell,node,qp,2));
					*/
					Residual(cell,node) += (delta*diam/vmax*(3.154 * pow10)) * scaling * (Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0) +
							   	   	   	    Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1) + verticalVel(cell,qp)*EnthalpyGrad(cell,qp,2)) * wSUPG;

					//if ( Enthalpy(cell,qp) >= EnthalpyHs(cell,qp) )
					//{
						/*
						Residual(cell,node) += 0.001*(k - rho_i*c*nu) * meltTempGrad(cell,qp,0) * (1/(3.154 * pow10)) * Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +
											   0.001*(k - rho_i*c*nu) * meltTempGrad(cell,qp,1) * (1/(3.154 * pow10)) * Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) +
											   0.001*(k - rho_i*c*nu) * meltTempGrad(cell,qp,2) * (1/(3.154 * pow10)) * verticalVel(cell,qp) * wGradBF(cell,node,qp,2) -
											   drainage_coeff*2*omega(cell,qp)*omegaGrad(cell,qp,2) * (1/(3.154 * pow10)) * (Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +
											   Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) + verticalVel(cell,qp) * wGradBF(cell,node,qp,2));
						*/
	    				ScalarT scale = - atan(alpha * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)))/pi + 0.5;
						Residual(cell,node) -= (1-scale) * drainage_coeff*alpha_om*pow(omega(cell,qp),alpha_om-1)*omegaGrad(cell,qp,2) * wSUPG;  // check whether is plus or minus
					//}
				}
      	  	}
    	}

    	// additional contributions of dissipation, basal friction heat and geothermal flux
    	if (needsDiss && needsBasFric)
    	{
			for (std::size_t cell=0; cell < d.numCells; ++cell)
			{
				for (std::size_t qp = 0; qp < numQPs; ++qp)
				{
					for (std::size_t i = 0; i < vecDimFO; i++)
					{
						vmax = std::max(vmax,std::fabs(Velocity(cell,qp,i)));
					}
				}

				for (std::size_t i = 0; i < numNodes-1; ++i)
				{
					for (std::size_t j = i + 1; j < numNodes; ++j)
					{
						diam = std::max(diam,distance(coordVec(cell,i,0),coordVec(cell,i,1),coordVec(cell,i,2),
													  coordVec(cell,j,0),coordVec(cell,j,1),coordVec(cell,j,2)));
					}
				}

				for (std::size_t node=0; node < numNodes; ++node)
				{
					for (std::size_t qp=0; qp < numQPs; ++qp)
					{
	        			// Modify here if you want to impose different basal BC
						ScalarT scale = - atan(alpha * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)))/pi + 0.5;
						Residual(cell,node) -= scale*(delta*diam/vmax*(3.154 * pow10))*( basalFricHeatSUPG(cell,qp) + geoFluxHeatSUPG(cell,qp) );

						Residual(cell,node) -= (delta*diam/vmax*(3.154 * pow10))*
											    diss(cell,qp) * (1./(3.154 * pow10)) * (Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) + Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) +
												verticalVel(cell,qp) * wGradBF(cell,node,qp,2));
					}
				}
			}
    	}
    	else if (needsBasFric)
    	{
			for (std::size_t cell=0; cell < d.numCells; ++cell)
			{
				for (std::size_t qp = 0; qp < numQPs; ++qp)
				{
					for (std::size_t i = 0; i < vecDimFO; i++)
					{
						vmax = std::max(vmax,std::fabs(Velocity(cell,qp,i)));
					}
				}

				for (std::size_t i = 0; i < numNodes-1; ++i)
				{
					for (std::size_t j = i + 1; j < numNodes; ++j)
					{
						diam = std::max(diam,distance(coordVec(cell,i,0),coordVec(cell,i,1),coordVec(cell,i,2),
													  coordVec(cell,j,0),coordVec(cell,j,1),coordVec(cell,j,2)));
					}
				}

				for (std::size_t node=0; node < numNodes; ++node)
				{
					for (std::size_t qp = 0; qp < numQPs; ++qp)
					{
	        			// Modify here if you want to impose different basal BC
						ScalarT scale = - atan(alpha * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp)))/pi + 0.5;
						Residual(cell,node) -= scale*( delta*diam/vmax*(3.154 * pow10))*( basalFricHeatSUPG(cell,qp) + geoFluxHeatSUPG(cell,qp) );
					}
				}
			}
    	}
    	else if (needsDiss)
    	{
			for (std::size_t cell=0; cell < d.numCells; ++cell)
			{
				for (std::size_t qp = 0; qp < numQPs; ++qp)
				{
					for (std::size_t i = 0; i < vecDimFO; i++)
					{
						vmax = std::max(vmax,std::fabs(Velocity(cell,qp,i)));
					}
				}

				for (std::size_t i = 0; i < numNodes-1; ++i)
				{
					for (std::size_t j = i + 1; j < numNodes; ++j)
					{
						diam = std::max(diam,distance(coordVec(cell,i,0),coordVec(cell,i,1),coordVec(cell,i,2),
													  coordVec(cell,j,0),coordVec(cell,j,1),coordVec(cell,j,2)));
					}
				}
				for (std::size_t node=0; node < numNodes; ++node)
				{
					for (std::size_t qp=0; qp < numQPs; ++qp)
					{
						Residual(cell,node) -= (delta*diam/vmax*(3.154 * pow10))*
										        diss(cell,qp) * (1./(3.154 * pow10)) * (Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) + Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) +
												verticalVel(cell,qp) * wGradBF(cell,node,qp,2));
					}
				}
			}
    	}
    }
}

}
