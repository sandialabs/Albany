//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  PoroElasticityResidMass<EvalT, Traits>::
  PoroElasticityResidMass(const Teuchos::ParameterList& p) :
    wBF         (p.get<std::string>                   ("Weighted BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout") ),
    porePressure (p.get<std::string>                   ("QP Pore Pressure Name"),
		  p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
    Tdot        (p.get<std::string>                   ("QP Time Derivative Variable Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
	stabParameter  (p.get<std::string>                   ("Material Property Name"),
		 		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
    ThermalCond (p.get<std::string>                   ("Thermal Conductivity Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
    kcPermeability (p.get<std::string>            ("Kozeny-Carman Permeability Name"),
		    p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
    porosity (p.get<std::string>                   ("Porosity Name"),
	      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
    biotCoefficient (p.get<std::string>           ("Biot Coefficient Name"),
		     p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
    elasticModulus (p.get<std::string>                   ("Elastic Modulus Name"),
		  p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
	pRatio (p.get<std::string>                   ("Poissons Ratio Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
	biotModulus (p.get<std::string>                   ("Biot Modulus Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
    wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout") ),
    TGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout") ),
    Source      (p.get<std::string>                   ("Source Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
    strain      (p.get<std::string>                   ("Strain Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
	coordVec      (p.get<std::string>                   ("Coordinate Vector Name"),
				 p.get<Teuchos::RCP<PHX::DataLayout>>("Coordinate Data Layout") ),
    cubature      (p.get<Teuchos::RCP <Intrepid::Cubature<RealType>> >("Cubature")),
	cellType      (p.get<Teuchos::RCP <shards::CellTopology>> ("Cell Type")),
	weights       (p.get<std::string>                   ("Weights Name"),
		         p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
	deltaTime (p.get<std::string>("Delta Time Name"),
		       p.get<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout")),
    TResidual   (p.get<std::string>                   ("Residual Name"),
		 p.get<Teuchos::RCP<PHX::DataLayout>>("Node Scalar Data Layout") ),
    haveSource  (p.get<bool>("Have Source")),
    haveConvection(false),
    haveAbsorption  (p.get<bool>("Have Absorption")),
    haverhoCp(false)
  {
    if (p.isType<bool>("Disable Transient"))
      enableTransient = !p.get<bool>("Disable Transient");
    else enableTransient = true;

    this->addDependentField(stabParameter);
    this->addDependentField(deltaTime);
    this->addDependentField(weights);
    this->addDependentField(coordVec);
    this->addDependentField(wBF);
    this->addDependentField(porePressure);
    this->addDependentField(ThermalCond);
    this->addDependentField(kcPermeability);
    this->addDependentField(porosity);
    this->addDependentField(biotCoefficient);
    this->addDependentField(biotModulus);
    this->addDependentField(elasticModulus);
    this->addDependentField(pRatio);
    if (enableTransient) this->addDependentField(Tdot);
    this->addDependentField(TGrad);
    this->addDependentField(wGradBF);
    if (haveSource) this->addDependentField(Source);
    if (haveAbsorption) {
      Absorption = PHX::MDField<ScalarT,Cell,QuadPoint>(
							p.get<std::string>("Absorption Name"),
							p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"));
      this->addDependentField(Absorption);
    }



    this->addDependentField(strain);
    this->addEvaluatedField(TResidual);

    Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get< Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    vector_dl->dimensions(dims);

    // Get data from previous converged time step
    strainName = p.get<std::string>("Strain Name")+"_old";
    porosityName = p.get<std::string>("Porosity Name")+"_old";
    porePressureName = p.get<std::string>("QP Pore Pressure Name")+"_old";



    worksetSize = dims[0];
    numNodes = dims[1];
    numQPs  = dims[2];
    numDims = dims[3];



    // Allocate workspace
    flux.resize(dims[0], numQPs, numDims);
    fluxdt.resize(dims[0], numQPs, numDims);
    pterm.resize(dims[0], numQPs);
    tpterm.resize(dims[0], numNodes, numQPs);



    if (haveAbsorption)  aterm.resize(dims[0], numQPs);

    convectionVels = Teuchos::getArrayFromStringParameter<double> (p,
								   "Convection Velocity", numDims, false);
    if (p.isType<std::string>("Convection Velocity")) {
      convectionVels = Teuchos::getArrayFromStringParameter<double> (p,
								     "Convection Velocity", numDims, false);
    }
    if (convectionVels.size()>0) {
      haveConvection = true;
      if (p.isType<bool>("Have Rho Cp"))
	haverhoCp = p.get<bool>("Have Rho Cp");
      if (haverhoCp) {
	PHX::MDField<ScalarT,Cell,QuadPoint> tmp(p.get<std::string>("Rho Cp Name"),
						 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"));
	rhoCp = tmp;
	this->addDependentField(rhoCp);
      }
    }

    this->setName("PoroElasticityResidMass"+PHX::typeAsString<EvalT>());

  }

  //**********************************************************************
  template<typename EvalT, typename Traits>
  void PoroElasticityResidMass<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm)
  {
	this->utils.setFieldData(stabParameter,fm);
	this->utils.setFieldData(deltaTime,fm);
	this->utils.setFieldData(weights,fm);
    this->utils.setFieldData(coordVec,fm);
    this->utils.setFieldData(wBF,fm);
    this->utils.setFieldData(porePressure,fm);
    this->utils.setFieldData(ThermalCond,fm);
    this->utils.setFieldData(kcPermeability,fm);
    this->utils.setFieldData(porosity,fm);
    this->utils.setFieldData(biotCoefficient,fm);
    this->utils.setFieldData(biotModulus,fm);
    this->utils.setFieldData(elasticModulus,fm);
    this->utils.setFieldData(pRatio,fm);
    this->utils.setFieldData(TGrad,fm);
    this->utils.setFieldData(wGradBF,fm);
    if (haveSource)  this->utils.setFieldData(Source,fm);
    if (enableTransient) this->utils.setFieldData(Tdot,fm);
    if (haveAbsorption)  this->utils.setFieldData(Absorption,fm);
    if (haveConvection && haverhoCp)  this->utils.setFieldData(rhoCp,fm);
    this->utils.setFieldData(strain,fm);
    this->utils.setFieldData(TResidual,fm);
  }

//**********************************************************************
template<typename EvalT, typename Traits>
void PoroElasticityResidMass<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;


  Albany::MDArray strainold = (*workset.stateArrayPtr)[strainName];
  Albany::MDArray porosityold = (*workset.stateArrayPtr)[porosityName];
  Albany::MDArray porePressureold = (*workset.stateArrayPtr)[porePressureName];


  switch (numDims) {
  case 3:

	  // Pore-fluid diffusion coupling.
	  for (int cell=0; cell < workset.numCells; ++cell) {

		  for (int node=0; node < numNodes; ++node) {
			  TResidual(cell,node)=0.0;
			  for (int qp=0; qp < numQPs; ++qp) {

				  // Transient partial saturated flow
				  ScalarT trstrain = 0.0;
				  for (int i(0); i < numDims; ++i){
					  trstrain += strainold(cell,qp,i,i);
				  }
				  // Volumetric Constraint Term
				  TResidual(cell,node) += -biotCoefficient(cell, qp)
						                                    *(strain(cell,qp,0,0)
						                                    + strain(cell,qp,1,1)
						                                    +strain(cell,qp,2,2)
						                                    - trstrain)*wBF(cell, node, qp)  ;

				  // Pore-fluid Resistance Term
				  TResidual(cell,node) +=  -(porePressure(cell, qp)
						                   -porePressureold(cell, qp))
						                  /biotModulus(cell, qp)*wBF(cell, node, qp);

			  }
		  }
	  }
	  break;
  case 2:
	  // Pore-fluid diffusion coupling.
	  for (int cell=0; cell < workset.numCells; ++cell) {

		  for (int node=0; node < numNodes; ++node) {
			  TResidual(cell,node)=0.0;
			  for (int qp=0; qp < numQPs; ++qp) {

				  // Transient partial saturated flow
				  ScalarT trstrain = 0.0;
				  for (int i(0); i < numDims; ++i){
					  trstrain += strainold(cell,qp,i,i);
				  }
				  // Volumetric Constraint Term
				  TResidual(cell,node) += -biotCoefficient(cell, qp)*(
					              	  strain(cell,qp,0,0) + strain(cell,qp,1,1) - trstrain
						              )*wBF(cell, node, qp)  ;

				  // Pore-fluid Resistance Term
				  TResidual(cell,node) +=  -(porePressure(cell, qp)
						                   -porePressureold(cell, qp))
            		                    		/biotModulus(cell, qp)*
            		                    		wBF(cell, node, qp);
			  }
		  }
	  }
	  break;
  case 1:
	  // Pore-fluid diffusion coupling.
	  	  for (int cell=0; cell < workset.numCells; ++cell) {
	  		  for (int node=0; node < numNodes; ++node) {
	  			  TResidual(cell,node)=0.0;
	  			  for (int qp=0; qp < numQPs; ++qp) {

	  				  // Transient partial saturated flow
	  				  ScalarT trstrain = 0.0;
	  				  for (int i(0); i < numDims; ++i){
	  					  trstrain += strainold(cell,qp,i,i);
	  				  }
	  				  // Volumetric Constraint Term
	  				  TResidual(cell,node) += -biotCoefficient(cell, qp)*(
	  					              	  strain(cell,qp,0,0) - trstrain
	  						              )*wBF(cell, node, qp)  ;

	  				  // Pore-fluid Resistance Term
	  				  TResidual(cell,node) +=  -(porePressure(cell, qp)
	  						                                     -porePressureold(cell, qp))
	  						                                     /biotModulus(cell, qp)
	              		                    		            *wBF(cell, node, qp);
	  			  }
	  		  }
	  	  }
	  	  break;
   }


  // Pore-Fluid Diffusion Term

   ScalarT dt = deltaTime(0);

    FST::scalarMultiplyDataData<ScalarT> (flux, kcPermeability, TGrad); // flux_i = k I_ij p_j

   for (int cell=0; cell < workset.numCells; ++cell){
      for (int qp=0; qp < numQPs; ++qp) {
    	  for (int dim=0; dim <numDims; ++dim){
    		  fluxdt(cell, qp, dim) = -flux(cell,qp,dim)*dt; // should replace the number with dt
    	  }
      }
  }

   FST::integrate<ScalarT>(TResidual, fluxdt, wGradBF, Intrepid::COMP_CPP, true); // "true" sums into

  //---------------------------------------------------------------------------//
  // Stabilization Term (only 2D and 3D problem need stabilizer)

// Penalty Term

  ScalarT Mp(0);  // 1/M' = 1/M + 1/(4/3G + K)

  for (int cell=0; cell < workset.numCells; ++cell){

   porePbar = 0.0;

   vol = 0.0;
   for (int qp=0; qp < numQPs; ++qp) {
	porePbar += weights(cell,qp)*(porePressure(cell,qp)
			                     -porePressureold(cell, qp)
			                      );

	vol  += weights(cell,qp);
   }
   porePbar  /= vol;

   for (int qp=0; qp < numQPs; ++qp) {
	   pterm(cell,qp) = porePbar;
   }

   for (int node=0; node < numNodes; ++node) {
	     trialPbar = 0.0;
 		 for (int qp=0; qp < numQPs; ++qp) {
 			  trialPbar += wBF(cell,node,qp);
 		 }
 		 trialPbar /= vol;
 		 for (int qp=0; qp < numQPs; ++qp) {
 		 		   tpterm(cell,node,qp) = trialPbar;
		 }

   }

 }

  for (int cell=0; cell < workset.numCells; ++cell) {
	  for (int node=0; node < numNodes; ++node) {
		  for (int qp=0; qp < numQPs; ++qp) {


			      Mp = elasticModulus(cell,qp)
			    		  *(1-pRatio(cell,qp))
			    		  /(1+pRatio(cell,qp))
			    		  /(1-pRatio(cell,qp)-pRatio(cell,qp));

 				  TResidual(cell,node) -= (porePressure(cell, qp)
 						                  -porePressureold(cell, qp) )
                    		                    		*stabParameter(cell, qp)
                    		                    		*(1/biotModulus(cell, qp)+1/Mp)*
                    		                    		( wBF(cell, node, qp)
                    		                    				);
 				  TResidual(cell,node) += pterm(cell,qp)*stabParameter(cell, qp)
		                                 *(1/biotModulus(cell, qp)+1/Mp)*
 						                  ( wBF(cell, node, qp) );

		  }
	  }
  }


}
//**********************************************************************
}


