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
//#include "Intrepid2_FunctionSpaceTools.hpp"

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


template<typename EvalT, typename Traits>
EnthalpyResid<EvalT,Traits>::
EnthalpyResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
	wBF      		(p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
	wGradBF  		(p.get<std::string> ("Weighted Gradient BF Variable Name"),dl->node_qp_gradient),
	Enthalpy        (p.get<std::string> ("Enthalpy QP Variable Name"), dl->qp_scalar),
	EnthalpyGrad    (p.get<std::string> ("Enthalpy Gradient QP Variable Name"), dl->qp_gradient),
	Velocity		(p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector),
    coordVec 		(p.get<std::string> ("Coordinate Vector Name"),dl->vertices_vector),
    diss 			(p.get<std::string> ("Dissipation QP Variable Name"),dl->qp_scalar),
	Residual 		(p.get<std::string> ("Residual Variable Name"), dl->node_scalar)
{
	Teuchos::RCP<PHX::DataLayout> vector_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
	std::vector<PHX::Device::size_type> dims;
	vector_dl->dimensions(dims);
	numNodes = dims[1];
	numQPs   = dims[2];
	numDims  = dims[3];

	this->addDependentField(Enthalpy);
	this->addDependentField(EnthalpyGrad);
	this->addDependentField(wBF);
	this->addDependentField(wGradBF);
	this->addDependentField(Velocity);
	this->addDependentField(coordVec);
	this->addDependentField(diss);

	this->addEvaluatedField(Residual);
	this->setName("EnthalpyResid");

	Teuchos::ParameterList* option_list = p.get<Teuchos::ParameterList*>("Options");
	k = option_list->get("Ice thermal conductivity k", 1.0);
	k *= 0.001;
	Teuchos::ParameterList* SUPG_list = p.get<Teuchos::ParameterList*>("SUPG Settings");
	haveSUPG = SUPG_list->get("Have SUPG Stabilization", false);
	delta = SUPG_list->get("Parameter Delta", 0.1);
}

template<typename EvalT, typename Traits>
void EnthalpyResid<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Enthalpy,fm);
  this->utils.setFieldData(EnthalpyGrad,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(Velocity,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(diss,fm);

  this->utils.setFieldData(Residual,fm);
}

template<typename EvalT, typename Traits>
void EnthalpyResid<EvalT,Traits>::
evaluateFields(typename Traits::EvalData d)
{
    for (std::size_t cell = 0; cell < d.numCells; ++cell)
    {
    	for (std::size_t node = 0; node < numNodes; ++node)
    	{
    		Residual(cell,node) = 0.0;

    		for (std::size_t qp = 0; qp < numQPs; ++qp)
    		{ //mu*du/dx + mu*du/dy + mu*du/dz + vel_x*du/dx + vel_y*du/dy = 0
            Residual(cell,node) += k*EnthalpyGrad(cell,qp,0)*wGradBF(cell,node,qp,0) +
            					   k*EnthalpyGrad(cell,qp,1)*wGradBF(cell,node,qp,1) +
								   k*EnthalpyGrad(cell,qp,2)*wGradBF(cell,node,qp,2) +
								   0.057964172 * Velocity(cell,qp,0)*EnthalpyGrad(cell,qp,0)*wBF(cell,node,qp) +
								   0.057964172 * Velocity(cell,qp,1)*EnthalpyGrad(cell,qp,1)*wBF(cell,node,qp) -
								   1.0/(3.154*pow(10.0,4.0)) * diss(cell,qp)*wBF(cell,node,qp);
    		}
        }
    }

    if (haveSUPG)
    {
    	ParamScalarT vmax = 0.0;
    	ParamScalarT diam = 0.0;

		// compute the max norm of the velocity
    	for (std::size_t cell=0; cell < d.numCells; ++cell)
		{
			for (std::size_t qp = 0; qp < numQPs; ++qp)
			{
				for (std::size_t i = 0; i < numDims; i++)
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
	    				Residual(cell,node) += (delta*diam/vmax*(3.154 * pow(10.0,10.0)))*(0.057964172 * Velocity(cell,qp,0) * EnthalpyGrad(cell,qp,0) * (1/(3.154 * pow(10.0,10.0))) * Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) +
	    														  0.057964172 * Velocity(cell,qp,1) * EnthalpyGrad(cell,qp,1) * (1/(3.154 * pow(10.0,10.0))) * Velocity(cell,qp,1) * wGradBF(cell,node,qp,1) -
																  1.0/(3.154*pow(10.0,4.0)) * diss(cell,qp) * (1/(3.154 * pow(10.0,10.0))) * Velocity(cell,qp,0) * wGradBF(cell,node,qp,0) -
																  1.0/(3.154*pow(10.0,4.0)) * diss(cell,qp) * (1/(3.154 * pow(10.0,10.0))) * Velocity(cell,qp,1) * wGradBF(cell,node,qp,1));
				}
      	  	}
    	}
    }
}

}
