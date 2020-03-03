//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <fstream>
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace TDM {

//**********************************************************************
template<typename EvalT, typename Traits>
Laser_Source<EvalT, Traits>::
Laser_Source(Teuchos::ParameterList& p,
                         const Teuchos::RCP<Albany::Layouts>& dl) :
  coord_        	(p.get<std::string>("Coordinate Name"), dl->qp_vector),
  time        		(p.get<std::string>("Time Name"), dl->workset_scalar),
  deltaTime   		(p.get<std::string>("Delta Time Name"), dl->workset_scalar),
  laser_source_ 	(p.get<std::string>("Laser Source Name"), dl->qp_scalar),
  psi1_	    			(p.get<std::string>("Psi1 Name"), dl->qp_scalar),
  psi2_        		(p.get<std::string>("Psi2 Name"), dl->qp_scalar),
  depth_          (p.get<std::string>("Depth Name"), dl->qp_scalar),
  Subtractive_          (p.get<bool>("Subtractive"))
{

  this->addDependentField(coord_);
  this->addDependentField(time);
  this->addDependentField(deltaTime);
  this->addDependentField(psi1_);	
  this->addDependentField(psi2_);
  this->addDependentField(depth_);        
  this->addEvaluatedField(laser_source_);
 
  Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  workset_size_ = dims[0];
  num_qps_      = dims[1];

  Teuchos::ParameterList* cond_list =
    p.get<Teuchos::ParameterList*>("Parameter List");
	
  Teuchos::ParameterList* porosity_list =
    p.get<Teuchos::ParameterList*>("InitialPorosity Parameter List");
  /*
  Teuchos::RCP<const Teuchos::ParameterList> reflist =
    this->getValidLaser_SourceParameters();

  cond_list->validateParameters(*reflist, 0,
      Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);
  */
    
  //From materials file
  absortivity = cond_list->get("Absortivity", 1.0);
  reflectivity = cond_list->get("Reflectivity", 1.0);

  powder_diameter = cond_list->get("Powder Diameter", 1.0);
  powder_layer_thickness = cond_list->get("Powder Layer Thickness", 1.0);

  laser_beam_radius = cond_list->get("Laser Beam Radius", 1.0);
  average_laser_power = cond_list->get("Average Laser Power",1.0);

  initial_porosity = porosity_list->get("Value", 1.0);

  //sim_type = input_list->get<std::string>("Simulation Type");
  
  //Import the laser path data from the specified file
  Teuchos::ParameterList* input_list = p.get<Teuchos::ParameterList*>("Input List");
  laser_path_filename = input_list->get<std::string>("Laser Path Input Filename");
  //std::cout<<"Inside LaserSource_Def, input file name is: "<<laser_path_filename<<"\n";
  Laser_object.Import_Laser_Path_Data(laser_path_filename); 
  
  this->setName("Laser_Source"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Laser_Source<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coord_,fm);
  this->utils.setFieldData(time,fm);
  this->utils.setFieldData(deltaTime,fm);
  this->utils.setFieldData(psi1_,fm);	
  this->utils.setFieldData(psi2_,fm);
  this->utils.setFieldData(depth_,fm);
  this->utils.setFieldData(laser_source_,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Laser_Source<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // current time
  const RealType t = workset.current_time;
  // time step
  ScalarT dt = deltaTime(0);
  if (dt == 0.0) {dt = 1.0e-6;} //Not good! Must be changed
  
  //Create throwaway lasercenter object and set it's time to the current workset time
  TDM::LaserCenter Val;
  Val.t = t; 
  //Create storage variables
  RealType x, y, power;
  //Use the current time to interpolate between the laser input datapoints
  // and store the data in the x,y,power variables created above
  Laser_object.getLaserPosition(t,Val,x,y,power);
  ScalarT Laser_center_x = x;
  ScalarT Laser_center_y = y;
  ScalarT Laser_power = power * average_laser_power;
  
  ScalarT pi = 3.1415926535897932;

  //initialization variables for later calculation
  ScalarT beta_p ;
  ScalarT lambda ;
  ScalarT a ;
  ScalarT A ;
  ScalarT B ;
  ScalarT b1 ;
  ScalarT b2 ;
  ScalarT c1 ;
  ScalarT c2 ;
  ScalarT C ;
  ScalarT E ;
	  
  //Absortivity for the dense material
  ScalarT beta_d = absortivity;
  
  // if there's powder defined
  if ( powder_layer_thickness != 0 && powder_diameter !=0 && initial_porosity !=0 ) {
  //Compute the powder layer source if there is an additive component
  //if (sim_type == "SLM Additive"){
    ScalarT beta_p = 1.5*(1.0 - initial_porosity)/(initial_porosity*powder_diameter);
    ScalarT lambda = powder_layer_thickness*beta_p;
	    
    ScalarT a = sqrt(1.0 - reflectivity);
    ScalarT A = (1.0 - pow(reflectivity,2))*exp(-lambda);
	  
    ScalarT B = 3.0 + reflectivity*exp(-2*lambda);
    ScalarT b1 = 1 - a;
    ScalarT b2 = 1 + a;
	  
    ScalarT c1 = b2 - reflectivity*b1;
    ScalarT c2 = b1 - reflectivity*b2;
	  
    ScalarT C = b1*c2*exp(-2*a*lambda) - b2*c1*exp(2*a*lambda);
	  
    ScalarT E = 1.0/(3.0 - 4.0*reflectivity);
	  
  //Absortivity for the dense material
    ScalarT beta_d = absortivity;
  //-----------------------------------------------------------------------------------------------
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < num_qps_; ++qp) {
        MeshScalarT X = coord_(cell,qp,0);
        MeshScalarT Y = coord_(cell,qp,1);
        MeshScalarT Z = coord_(cell,qp,2);

        ScalarT radius = sqrt(((X - Laser_center_x)*(X - Laser_center_x)) + ((Y - Laser_center_y)*(Y - Laser_center_y)));
        if (radius < (2*laser_beam_radius)){
          ScalarT Q =(2.0*Laser_power/(pi*laser_beam_radius*laser_beam_radius))*exp(-2*radius*radius/(laser_beam_radius*laser_beam_radius));  // horizontal aenergy distribution
          //Dense Material Power Input
          ScalarT g_d = (1.0 - reflectivity); //fraction of energy absorbed
          ScalarT H_l = initial_porosity*std::min(Z,powder_layer_thickness);  //location where dense material starts
          ScalarT h_d = g_d*exp(-absortivity*(Z - H_l)); //dense(solid) depth profile
          ScalarT U_d = Q*beta_d*h_d;   
      
          ScalarT U_p;
          ScalarT xi = beta_p*Z;
          ScalarT term1 = A*(b2*exp(2*a*xi) - b1*exp(-2*a*xi));
          ScalarT term2 = B*(c2*exp(-2*a*(lambda-xi)) - c1*exp(2*a*(lambda-xi)));
          ScalarT term3 = 3*(a*a)*(exp(-xi) + reflectivity*exp(xi-(2*lambda))); 
          ScalarT h_p = E * ( ((2*reflectivity*a*a)/C)*(term1-term2) + term3);
          U_p = Q*beta_p*h_p; 
          laser_source_(cell,qp) =  (U_p*(1-psi1_(cell,qp)) + U_d*psi1_(cell,qp));
        }
        else   laser_source_(cell,qp) = 0.0;
      }
    }
  
  }
  
  else{

    //Absortivity for the dense material
    beta_d = absortivity;
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t qp = 0; qp < num_qps_; ++qp) {
        MeshScalarT X = coord_(cell,qp,0);
	MeshScalarT Y = coord_(cell,qp,1);
	ScalarT Z = depth_(cell,qp);
	 
	ScalarT radius = sqrt(((X - Laser_center_x)*(X - Laser_center_x)) + ((Y - Laser_center_y)*(Y - Laser_center_y)));

	if( Z>=0 ){  
	  if (radius < (2*laser_beam_radius)){
	    ScalarT Q =(2.0*Laser_power/(pi*laser_beam_radius*laser_beam_radius))*exp(-2*radius*radius/(laser_beam_radius*laser_beam_radius));  // horizontal aenergy distribution
	    //Dense Material Power Input
	    ScalarT g_d = (1.0 - reflectivity); //fraction of energy absorbed
	    ScalarT h_d = g_d*exp(-absortivity*Z); //dense(solid) depth profile
	    ScalarT U_d = Q*beta_d*h_d;   
	    laser_source_(cell,qp) = U_d;
	  }
	  else   laser_source_(cell,qp) = 0.0;
	}
	else{
	  laser_source_(cell,qp) = 0.0;
	}

	if (psi2_(cell,qp)==1.0){
	  laser_source_(cell,qp) = 0.0;
	}

       }
     }
   }
 }

//**********************************************************************
template<typename EvalT, typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
Laser_Source<EvalT, Traits>::
getValidLaser_SourceParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> valid_pl =
    rcp(new Teuchos::ParameterList("Valid Laser Source Params"));;
 
  valid_pl->set<std::string>("Absortivity Type", "Constant");
  valid_pl->set<double>("Absortivity", 1.0);
  valid_pl->set<std::string>("Reflectivity Type", "Constant");
  valid_pl->set<double>("Reflectivity", 1.0);
  valid_pl->set<std::string>("Laser Beam Radius Type", "Constant");
  valid_pl->set<double>("Laser Beam Radius", 1.0);
  valid_pl->set<std::string>("Powder Diameter Type", "Constant");
  valid_pl->set<double>("Powder Diameter", 1.0);
  valid_pl->set<std::string>("Powder Layer Thickness Type", "Constant");
  valid_pl->set<double>("Powder Layer Thickness", 1.0);

  return valid_pl;
}
//**********************************************************************

}
