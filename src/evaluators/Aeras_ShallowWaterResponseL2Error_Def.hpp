//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Aeras_ShallowWaterConstants.hpp"

namespace Aeras {

}


template<typename EvalT, typename Traits>
Aeras::ShallowWaterResponseL2Error<EvalT, Traits>::
ShallowWaterResponseL2Error(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl) :
  sphere_coord("Lat-Long", dl->qp_gradient),
  weighted_measure("Weights", dl->qp_scalar),
  flow_state_field("Flow State", dl->node_vector), 
  BF("BF",dl->node_qp_scalar)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  // get and validate Response parameter list
  Teuchos::ParameterList* plist = 
    p.get<Teuchos::ParameterList*>("Parameter List");
  Teuchos::RCP<const Teuchos::ParameterList> reflist = 
    this->getValidResponseParameters();
  plist->validateParameters(*reflist,0);

  std::string fieldName = "Flow Field"; //field to integral is the flow field 

  // coordinate dimensions
  std::vector<PHX::DataLayout::size_type> coord_dims;
  dl->qp_vector->dimensions(coord_dims);
  numQPs = coord_dims[1]; //# quad points
  numDims = coord_dims[2]; //# spatial dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  flow_state_field.fieldTag().dataLayout().dimensions(dims);
  vecDim = dims[2]; //# dofs per node
  numNodes =  dims[1]; //# nodes per element

 
  // User-specified parameters
  refSolName = plist->get<std::string>("Reference Solution Name"); //no reference solution by default.
  *out << "Reference Solution Name for Aeras::ShallowWaterResponseL2Error response: " << refSolName << std::endl; 
  inputData = plist->get<double>("Reference Solution Data", 0.0);
   
  if (refSolName == "Zero")
    ref_sol_name = ZERO;
  else if (refSolName == "TC2")
    ref_sol_name  = TC2;
  //Add other test case reference solutions here...
  else if (refSolName == "TC4"){
    ref_sol_name = TC4;
    
    myPi = Aeras::ShallowWaterConstants::self().pi;
    earthRadius = Aeras::ShallowWaterConstants::self().earthRadius;
    gravity = Aeras::ShallowWaterConstants::self().gravity;
    
    Omega = 2.0*myPi/(24.*3600.); //this should be sitting in SW Constants class
    
    rlon0 = 0.;
    rlat0 = myPi/4.;
    npwr = 14.;
    
    su0 = 20.;
    phi0 = 1.0e5;
    alfa = -0.03*(phi0/(2.*Omega*sin(myPi/4.)));
    sigma = (2.*earthRadius/1.0e6)*(2.*earthRadius/1.0e6);
    
  }
  else { 
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown reference solution name " << ref_sol_name <<
      "!" << std::endl;);
  }

  // add dependent fields
  this->addDependentField(sphere_coord);
  this->addDependentField(flow_state_field);
  this->addDependentField(weighted_measure);
  this->addDependentField(BF);
  this->setName(fieldName+" Aeras Shallow Water L2 Error"+PHX::TypeString<EvalT>::value);
  
  using PHX::MDALayout;

  // Setup scatter evaluator
  p.set("Stand-alone Evaluator", false);
  std::string local_response_name = fieldName + " Local Response Aeras Shallow Water L2 Error";
  std::string global_response_name = fieldName + " Global Response Aeras Shallow Water L2 Error";
  int worksetSize = dl->qp_scalar->dimension(0);
  //There are three components of the response returned by this function: 
  //1.) The absolute error in the solution
  //2.) The norm of the reference solution.
  //3.) The relative error in the solution w.r.t. the reference solution.
  responseSize = 3; 
  Teuchos::RCP<PHX::DataLayout> local_response_layout = Teuchos::rcp(new MDALayout<Cell, Dim>(worksetSize, responseSize));
  Teuchos::RCP<PHX::DataLayout> global_response_layout = Teuchos::rcp(new MDALayout<Dim>(responseSize));
  PHX::Tag<ScalarT> local_response_tag(local_response_name, local_response_layout);
  PHX::Tag<ScalarT> global_response_tag(global_response_name, global_response_layout);
  p.set("Local Response Field Tag", local_response_tag);
  p.set("Global Response Field Tag", global_response_tag);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::setup(p,dl);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ShallowWaterResponseL2Error<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(sphere_coord,fm);
  this->utils.setFieldData(flow_state_field,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(BF,fm);
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postRegistrationSetup(d,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ShallowWaterResponseL2Error<EvalT, Traits>::
preEvaluate(typename Traits::PreEvalData workset)
{
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->global_response.size(); i++)
    this->global_response[i] = 0.0;

  // Do global initialization
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::preEvaluate(workset);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ShallowWaterResponseL2Error<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{   
  // Zero out local response
  for (typename PHX::MDField<ScalarT>::size_type i=0; 
       i<this->local_response.size(); i++)
    this->local_response[i] = 0.0;


  Intrepid::FieldContainer<ScalarT> flow_state_field_qp(workset.numCells, numQPs, vecDim); //flow_state_field at quad points
  Intrepid::FieldContainer<ScalarT> flow_state_field_ref_qp(workset.numCells, numQPs, vecDim); //flow_state_field_ref (exact solution) at quad points
  Intrepid::FieldContainer<ScalarT> err_qp(workset.numCells, numQPs, vecDim); //error at quadrature points

  //Interpolate flow_state_field from nodes -> quadrature points.  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      for (std::size_t i=0; i<vecDim; i++) {
        // Zero out for node==0; then += for node = 1 to numNodes
        flow_state_field_qp(cell,qp,i) = 0.0;
        flow_state_field_qp(cell,qp,i) = flow_state_field(cell, 0, i) * BF(cell, 0, qp); 
        for (std::size_t node=1; node < numNodes; ++node) {
          flow_state_field_qp(cell,qp,i) += flow_state_field(cell,node,i)*BF(cell,node,qp); 
        }
       }
     }
    }

  //Get final time from workset.  This is for setting time-dependent exact solution.  
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  const RealType final_time  = workset.current_time;
  *out << "final time = " << final_time << std::endl; 
 
  //Set reference solution at quadrature points
  if (ref_sol_name == ZERO) { //zero reference solution 
    for (std::size_t cell=0; cell < workset.numCells; ++cell) 
      for (std::size_t qp=0; qp < numQPs; ++qp) 
        for (std::size_t dim=0; dim < vecDim; dim++)
          flow_state_field_ref_qp(cell,qp,dim) = 0.0;  
  }
  else if (ref_sol_name == TC2) { //reference solution for TC2 

    const double myPi = Aeras::ShallowWaterConstants::self().pi;
    const double gravity = Aeras::ShallowWaterConstants::self().gravity;
    const double Omega = 2.0*myPi/(24.*3600.);
    const double a = Aeras::ShallowWaterConstants::self().earthRadius;
    const double u0 = 2.*myPi*a/(12*24*3600.);  // magnitude of wind
    const double h0g = inputData;
    const double alpha = 0; /* must match value in ShallowWaterResidDef
                             don't know how to get data from input into this class and that one. */
    const double cosAlpha = std::cos(alpha);
    const double sinAlpha = std::sin(alpha);
    const double h0     = h0g/gravity;

    static const double DIST_THRESHOLD = Aeras::ShallowWaterConstants::self().distanceThreshold;

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        MeshScalarT lambda = sphere_coord(cell, qp, 0);//lambda 
        MeshScalarT theta = sphere_coord(cell, qp, 1); //theta
        if (std::abs(std::abs(theta)-myPi/2) < DIST_THRESHOLD) lambda = 0.0;
        else if (lambda < 0) lambda += 2*myPi;
        const MeshScalarT cosLambda = std::cos(lambda); //cos(lambda)
        const MeshScalarT sinLambda = std::sin(lambda); //sin(lambda)
        const MeshScalarT cosTheta = std::cos(theta); //cos(theta)
        const MeshScalarT sinTheta = std::sin(theta); //sin(theta)
        flow_state_field_ref_qp(cell,qp,0) =  h0 - 1.0/gravity * (a*Omega*u0 + u0*u0/2.0)*(-cosLambda*cosTheta*sinAlpha + sinTheta*cosAlpha)*
           (-cosLambda*cosTheta*sinAlpha + sinTheta*cosAlpha); //h
        flow_state_field_ref_qp(cell,qp,1) = u0*(cosTheta*cosAlpha + sinTheta*cosLambda*sinAlpha); //u
        flow_state_field_ref_qp(cell,qp,2) = -u0*(sinLambda*sinAlpha); //v
       }
     }
   }else if (ref_sol_name == TC4) { //reference solution for TC4
    
    ScalarT a = earthRadius;
     
    ScalarT tol = 1.e-10;
     
    ScalarT ai = 1./a;
    ScalarT a2i = ai*ai;
     
    ////og: this is a patch to get rid of conversion error message
    const double myPi_local = Aeras::ShallowWaterConstants::self().pi;
     
    //repeated code
    static const double DIST_THRESHOLD = Aeras::ShallowWaterConstants::self().distanceThreshold;
    
    const RealType time = workset.current_time; //current time from workset
     
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        
        /////////repeated code
        MeshScalarT lambda = sphere_coord(cell, qp, 0);//lambda
        MeshScalarT theta = sphere_coord(cell, qp, 1); //theta
        if (std::abs(std::abs(theta)-myPi_local/2) < DIST_THRESHOLD) lambda = 0.0;
        else if (lambda < 0) lambda += 2*myPi_local;
        /////////
        
        ///time shift
        ScalarT TMSHFT = su0*time/a;
        
        ScalarT snj = std::sin(theta);
        ScalarT csj = std::cos(theta)*std::cos(theta);
        ScalarT srcsj = std::cos(theta);
        ScalarT tmpry = std::tan(theta);
        ScalarT tmpry2 = tmpry*tmpry;
        ScalarT den = 1./std::cos(theta);
        ScalarT aacsji = 1./(a*a*csj);
        ScalarT corr = 2.*Omega*snj;
     
        ScalarT ucon = bubfnc(theta);
        ScalarT bigubr = ucon*srcsj; ///
        ScalarT dbub = dbubf(theta); ///
     
        ScalarT c = std::sin(rlat0)*snj + std::cos(rlat0)*srcsj*
                    std::cos(lambda - TMSHFT - rlon0);
     
        //if-statements about ~fabs(c+1) is due to singularities ~1/(c+1)
        //in derivatives. However, they are overtaken by the presence of
        //multipliers ~exp(-1/(c+1)).
        ScalarT psib = 0.;
        if(fabs(c+1.)>tol)
          psib = alfa*std::exp(-sigma*((1.-c)/(1.+c)));
     
        ScalarT dcdm = std::sin(rlat0)-std::cos(lambda - TMSHFT -rlon0)*
                       std::cos(rlat0)*tmpry;
        ScalarT dcdl = -std::cos(rlat0)*srcsj*std::sin(lambda - TMSHFT -rlon0);
        ScalarT d2cdm = -std::cos(rlat0)*std::cos(lambda - TMSHFT -rlon0)*(1.+tmpry2)/srcsj;
        ScalarT d2cdl = -std::cos(rlat0)*srcsj*std::cos(lambda - TMSHFT -rlon0);
     
        ScalarT tmp1 = 0.;
        if(fabs(c+1.)>tol)
          tmp1 = 2.*sigma*psib/((1.+c)*(1.+c));
        ScalarT tmp2 = 0.;
        if(fabs(c+1.)>tol)
          tmp2 = (sigma - (1.0+c))/((1.+c)*(1.+c));
        ScalarT dkdm = tmp1*dcdm;
        ScalarT dkdl = tmp1*dcdl;
     
        ScalarT d2kdm  = tmp1*(d2cdm + 2.0*(dcdm*dcdm)*tmp2);
        ScalarT d2kdl  = tmp1*(d2cdl + 2.0*(dcdl*dcdl)*tmp2);
     
        ScalarT u, v, h;
     
        u = bigubr*den - srcsj*ai*dkdm;
        v = (dkdl*ai)*den;
        h = phicon(theta)+corr*psib/gravity;
        

        flow_state_field_ref_qp(cell,qp,0) = h; //h
        flow_state_field_ref_qp(cell,qp,1) = u; //u
        flow_state_field_ref_qp(cell,qp,2) = v; //v
      }
    }
  }


  //Calculate L2 error at all the quad points 
   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t dim=0; dim < vecDim; ++dim) {
          err_qp(cell,qp,dim) = flow_state_field_qp(cell,qp,dim) - flow_state_field_ref_qp(cell,qp,dim);
        }
        //debug print statements
        /*std::cout << "cell, qp: " << cell << ", " << qp << std::endl;
        std::cout << "error h: " << err_qp(cell,qp,0) << std::endl;   
        std::cout << "h calc, h ref: " << flow_state_field_qp(cell,qp,0) << ", " << flow_state_field_ref_qp(cell,qp,0) << std::endl; 
        std::cout << "error u: " << err_qp(cell,qp,1) << std::endl;  
        std::cout << "u calc, u ref: " << flow_state_field_qp(cell,qp,1) << ", " << flow_state_field_ref_qp(cell,qp,1) << std::endl; 
        std::cout << "error v: " << err_qp(cell,qp,2) << std::endl;  
        */
      }
    }

  //Calculate absolute L2 error squared and norm of reference solution squared
   ScalarT err_sq = 0.0;
    ScalarT norm_ref_sq = 0.0;
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      this->local_response(cell,2) = 0.0;  
      this->global_response(2) = 0.0; 
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        for (std::size_t dim=0; dim < vecDim; ++dim) {
           //L^2 error squared w.r.t. flow_state_field_ref -- first component of global_response
            err_sq = err_qp(cell,qp,dim)*err_qp(cell,qp,dim);
           this->local_response(cell,0) += err_sq;
           this->global_response(0) += err_sq;
           //L^2 norm squared of flow_state_field_ref, the exact solution  -- second component of global_response
            norm_ref_sq = flow_state_field_ref_qp(cell,qp,dim)*flow_state_field_ref_qp(cell,qp,dim);
           this->local_response(cell,1) += norm_ref_sq;
           this->global_response(1) += norm_ref_sq;
        }
      }
    }
//  }

  // Do any local-scattering necessary
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::evaluateFields(workset);
}

//***********************************************************************
//***********************************************************************
//
template<typename EvalT, typename Traits>
typename Aeras::ShallowWaterResponseL2Error<EvalT,Traits>::ScalarT
Aeras::ShallowWaterResponseL2Error<EvalT, Traits>::
dbubf(ScalarT lat){
  ScalarT rmu = sin(lat);
  ScalarT coslat = cos(lat);
  return 2.*su0*std::pow(2.*rmu*coslat,npwr-1.)
  *(npwr-(2.*npwr+1)*rmu*rmu);
}

template<typename EvalT, typename Traits>
typename Aeras::ShallowWaterResponseL2Error<EvalT,Traits>::ScalarT
Aeras::ShallowWaterResponseL2Error<EvalT, Traits>::
bubfnc(ScalarT lat){
  return su0*std::pow((2.*sin(lat)*cos(lat)), npwr);
}

template<typename EvalT, typename Traits>
typename Aeras::ShallowWaterResponseL2Error<EvalT,Traits>::ScalarT
Aeras::ShallowWaterResponseL2Error<EvalT, Traits>::
phicon(ScalarT lat){
  
  ScalarT a = earthRadius;
  
  const int integration_steps = 1000;
  
  ScalarT h = 0.;
  
  ScalarT deltat = (lat+myPi/2.0)/integration_steps;
  for(int i=0; i<integration_steps; i++){
    
    ScalarT midpoint1 = -myPi/2.0 + (i-1)*deltat;
    ScalarT midpoint2 = -myPi/2.0 + i*deltat;
    
    ScalarT loc_u = bubfnc(midpoint1);
    
    h -= a*deltat*(2*Omega*sin(midpoint1)+loc_u*tan(midpoint1)/a)*loc_u/2.;
    
    loc_u = bubfnc(midpoint2);
    
    h -= a*deltat*(2*Omega*sin(midpoint2)+loc_u*tan(midpoint2)/a)*loc_u/2.;
    
  }
  
  h = (phi0 + h)/gravity;
  
  return h;
  
}


//***********************************************************************
// **********************************************************************
template<typename EvalT, typename Traits>
void Aeras::ShallowWaterResponseL2Error<EvalT, Traits>::
postEvaluate(typename Traits::PostEvalData workset)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  // Add contributions across processors
  Teuchos::RCP< Teuchos::ValueTypeSerializer<int,ScalarT> > serializer =
    workset.serializerManager.template getValue<EvalT>();

  // we cannot pass the same object for both the send and receive buffers in reduceAll call
  // creating a copy of the global_response, not a view
  std::vector<ScalarT> partial_vector(&this->global_response[0],&this->global_response[0]+this->global_response.size()); //needed for allocating new storage
  PHX::MDField<ScalarT> partial_response(this->global_response);
  partial_response.setFieldData(Teuchos::ArrayRCP<ScalarT>(partial_vector.data(),0,partial_vector.size(),false));


  //perform reduction for each of the components of the response
  Teuchos::reduceAll(
      *workset.comm, *serializer, Teuchos::REDUCE_SUM,
      this->global_response.size(), &partial_response[0],
      &this->global_response[0]);
  
  ScalarT abs_err_sq = this->global_response[0];
  ScalarT norm_ref_sq = this->global_response[1];
  this-> global_response[0] = sqrt(abs_err_sq); //absolute error in solution w.r.t. reference solution.
  this-> global_response[1] = sqrt(norm_ref_sq); //norm of reference solution
  this-> global_response[2] = sqrt(abs_err_sq/norm_ref_sq); //relative error in solution w.r.t. reference solution.
  if (norm_ref_sq == 0)  {
    *out << "Aeras::ShallowWaterResponseL2Error::postEvaluate WARNING: norm of reference solution is 0.  Aeras Shallow Water L2 Error response" <<
            "will report 'nan' or 'inf' for the relative error, so please look at the absolute error." << std::endl;
  }

  // Do global scattering
  PHAL::SeparableScatterScalarResponse<EvalT,Traits>::postEvaluate(workset);
}

// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Aeras::ShallowWaterResponseL2Error<EvalT,Traits>::
getValidResponseParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid ShallowWaterResponseL2Error Params"));
  Teuchos::RCP<const Teuchos::ParameterList> baseValidPL =
    PHAL::SeparableScatterScalarResponse<EvalT,Traits>::getValidResponseParameters();
  validPL->setParameters(*baseValidPL);

  validPL->set<std::string>("Name", "", "Name of response function");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0, "Make dot file to visualize phalanx graph");
  validPL->set<std::string>("Reference Solution Name", "", "Name of reference solution");
  validPL->set<double>("Reference Solution Data", 0.0, "Data needed to specifying reference solution");

  return validPL;
}

// **********************************************************************

