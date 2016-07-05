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
	Residual 		(p.get<std::string> ("Residual Variable Name"), dl->node_scalar)
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
	k_i = physics_list->get("Conductivity of ice", 1.0);
	k_i *= 0.001; //scaling needs to be done to match dimensions
	c_i = physics_list->get("Heat capacity of ice",2000.0);
	K_i = k_i / c_i;

	K_0 = physics_list->get("Diffusivity temperate ice", 0.001);
	K_0 *= 0.001; //scaling needs to be done to match dimensions
	rho = physics_list->get("Ice Density", 910.0);

	alpha = physics_list->get("alpha", 1.0);
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
        			//if ( Enthalpy(cell,qp) < EnthalpyHs(cell,qp) )	// if the base is cold, we consider the basal friction and the geothermal flux, otherwise we impose homogeneous Neumann bc
        				Residual(cell,node) -= ( basalFricHeat(cell,qp) + geoFluxHeat(cell,qp) );
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
    			K = - (K_i - K_0)/pi * atan(alpha * (Enthalpy(cell,qp) - EnthalpyHs(cell,qp))) + (K_i + K_0)/2;
				//if ( Enthalpy(cell,qp) < EnthalpyHs(cell,qp) )
				//{	//cold ice
					Residual(cell,node) += K*EnthalpyGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
										   K*EnthalpyGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
										   K*EnthalpyGrad(cell,qp,2)*wGradBF(cell,node,qp,2) +
										   scaling * rho * Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0)*wBF(cell,node,qp) +
										   scaling * rho * Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1)*wBF(cell,node,qp) +
										   scaling * rho * verticalVel(cell,qp)*EnthalpyGrad(cell,qp,2)*wBF(cell,node,qp);
				//}
				/*
				else
				{	//temperate ice
				Residual(cell,node) += K_0*EnthalpyGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
									   K_0*EnthalpyGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
									   K_0*EnthalpyGrad(cell,qp,2)*wGradBF(cell,node,qp,2) +
									   scaling * rho * Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0)*wBF(cell,node,qp) +
									   scaling * rho * Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1)*wBF(cell,node,qp) +
									   k_i*meltTempGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
									   k_i*meltTempGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
									   k_i*meltTempGrad(cell,qp,2)*wGradBF(cell,node,qp,2);
				}
				*/
				/*
				if ( abs(K-K_0) < 1e-3 ) // if the ice is approximatively temperate
					Residual(cell,node) += k_i*meltTempGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
					   	   	   	   	   	   k_i*meltTempGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
										   k_i*meltTempGrad(cell,qp,2)*wGradBF(cell,node,qp,2);
				*/
			}
        }
    }

    if (haveSUPG)
    {
    	VelocityType vmax = 0.0;
    	ParamScalarT diam = 0.0;

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
					//std::cout << " vmax = " << vmax << std::endl;
					Residual(cell,node) += (delta*diam/vmax*(3.154 * pow10))*(scaling * rho * Velocity(cell,qp,0) * EnthalpyGrad(cell,qp,0) * (1/(3.154 * pow10)) * Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +
	    			   					  scaling * rho * Velocity(cell,qp,1) * EnthalpyGrad(cell,qp,1) * (1/(3.154 * pow10)) * Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) +
										  scaling * rho * verticalVel(cell,qp) * EnthalpyGrad(cell,qp,2) * (1/(3.154 * pow10)) * verticalVel(cell,qp) * wGradBF(cell,node,qp,2));
					/*
					if ( Enthalpy(cell,qp) >= EnthalpyHs(cell,qp) )
						Residual(cell,node) += k_i * meltTempGrad(cell,qp,0) * (1/(3.154 * pow10)) * Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +
											   k_i * meltTempGrad(cell,qp,1) * (1/(3.154 * pow10)) * Velocity(cell,qp,1) * wGradBF(cell,node,qp,1);
					 */
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
						//if ( Enthalpy(cell,qp) < EnthalpyHs(cell,qp) )
							Residual(cell,node) -= (delta*diam/vmax*(3.154 * pow10))*( basalFricHeatSUPG(cell,qp) + geoFluxHeatSUPG(cell,qp) );

						Residual(cell,node) -= (delta*diam/vmax*(3.154 * pow10))*
											   (diss(cell,qp) * (1./(3.154 * pow10)) * Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +
												diss(cell,qp) * (1./(3.154 * pow10)) * Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) +
												diss(cell,qp) * (1./(3.154 * pow10)) * verticalVel(cell,qp) * wGradBF(cell,node,qp,2));
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
						if ( Enthalpy(cell,qp) < EnthalpyHs(cell,qp) )
							Residual(cell,node) -= ( delta*diam/vmax*(3.154 * pow10))*( basalFricHeatSUPG(cell,qp) + geoFluxHeatSUPG(cell,qp) );
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
												   (diss(cell,qp) * (1./(3.154 * pow10)) * Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +
													diss(cell,qp) * (1./(3.154 * pow10)) * Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) +
													diss(cell,qp) * (1./(3.154 * pow10)) * verticalVel(cell,qp) * wGradBF(cell,node,qp,2));
					}
				}
			}
    	}
    }
}

}
