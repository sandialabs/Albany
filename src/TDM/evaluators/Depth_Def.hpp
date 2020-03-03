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
#include <cmath>
#include <string>

#include <PCU.h>
#include <pumi.h>
#include <apf.h>
#include <pcu_util.h>
#include "apfShape.h"
#include "SurfaceUpdate.hpp"
#include "lionPrint.h"
#include "apfDynamicArray.h"
#include "apfMesh2.h"


#include "Albany_Application.hpp"
#include "Albany_APFMeshStruct.hpp"
#include "Albany_APFDiscretization.hpp"

namespace TDM {

  //**********************************************************************
  template<typename EvalT, typename Traits>
  Depth<EvalT, Traits>::
  Depth(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
    coord_        (p.get<std::string>("Coordinate Name"), dl->qp_vector),
    psi2_         (p.get<std::string>("Psi2 Name"),dl->qp_scalar),
    depth_        (p.get<std::string>("Depth Name"),dl->qp_scalar)
  {

    this->addDependentField(coord_);
    this->addDependentField(psi2_);
    this->addEvaluatedField(depth_);

    Teuchos::RCP<PHX::DataLayout> scalar_dl = dl->qp_scalar;
    std::vector<PHX::Device::size_type> dims;
    scalar_dl->dimensions(dims);
    workset_size_ = dims[0];
    num_qps_      = dims[1];

    Teuchos::ParameterList* cond_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

    Teuchos::RCP<const Teuchos::ParameterList> reflist =
      this->getValidDepthParameters();
    
    paramLib_ = p.get<Teuchos::RCP<ParamLib>>("paramLib");

    cond_list->validateParameters(*reflist, 0,
				  Teuchos::VALIDATE_USED_ENABLED, Teuchos::VALIDATE_DEFAULTS_DISABLED);

    frequency_ = cond_list->get("Frequency", 200.0);

    depth_Name_ = p.get<std::string>("Depth Name")+"_old";
    this->setName("Depth"+PHX::print<EvalT>());
  }

  //**********************************************************************



  //**********************************************************************
  template<typename EvalT, typename Traits>
  void Depth<EvalT, Traits>::
  postRegistrationSetup(typename Traits::SetupData d,
			PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(coord_,fm);    
    this->utils.setFieldData(psi2_,fm);
    this->utils.setFieldData(depth_,fm);
  }

  //**********************************************************************

  template<typename EvalT, typename Traits>
  void Depth<EvalT, Traits>::
  evaluateFields(typename Traits::EvalData workset)
  {
    // get Abstract Discretization

    // below returns a Teuchos::RCP<const Thyra_VectorSpace>,  ( return Thyra::tepetraVectorSpace(ST)(getMapT()) ) 
    //auto vector_space = workset.disc->getVectorSpace();
    //Teuchos::RCP<Thyra_Vector> my_field = Thyra::createMember(vector_space);
    //Teuchos::RCP<Albany::AbstractDiscretization> discPtr = workset.disc;
    //auto APFDisc_ptr = Teuchos::rcp_dynamic_cast<Albany::APFDiscretization>(discPtr);
    // apf::Mesh2* m = APFDisc_ptr -> getMesh();


    //grab old depth value
    Albany::MDArray depth_old = (*workset.stateArrayPtr)[depth_Name_];
    // current time
    const RealType t = workset.current_time;

    // check if update is needed 
    double update;
    update = fmod( t , ( 1000.0 /frequency_) );

    if (t == 0.0)  // if this is the overall time = 0
    {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell){
        for (std::size_t qp = 0; qp < num_qps_; ++qp){
          MeshScalarT Z = coord_(cell,qp,2);
                depth_(cell, qp) = Z;
        }
      }
    }
    else if ( (update > 0)&&(update < 0.000125) ){	// if this is the time = 0 at each pulse

      for (std::size_t cell = 0; cell < workset.numCells; ++cell){
        for (std::size_t qp = 0; qp < num_qps_; ++qp){
             depth_(cell, qp) = depth_old(cell,qp);
        }
      }
	/*
      lion_set_verbosity(1);
      apf::FieldShape* IPShape = apf::getIPShape(3,1);
      apf::Field* f;
      APFDisc_ptr -> copyQPStatesToAPF( f, IPShape, false);
      std::cout<<"copy QP states to APF successfully\n";
      SurfaceUpdate(m);
      std::cout<<"Back to Albany\n";
      std::string Prefix="SurfaceOut_"+std::to_string(t);
      char OutName[Prefix.size()+1];
      strcpy(OutName,Prefix.c_str());
      apf::writeASCIIVtkFiles(OutName,m);
      //APFDisc_ptr -> updateMesh(false, paramLib_);
      APFDisc_ptr -> copyQPStatesFromAPF();
      std::cout<<"All APF data copied into Albany\n";
      APFDisc_ptr -> removeQPStatesFromAPF();
	*/

    }
    else {
      for (std::size_t cell = 0; cell < workset.numCells; ++cell){
        for (std::size_t qp = 0; qp < num_qps_; ++qp){
          depth_(cell, qp) = depth_old(cell,qp);
        }
      }
    }


  }


  //**********************************************************************

  template<typename EvalT, typename Traits>
  Teuchos::RCP<const Teuchos::ParameterList>
  Depth<EvalT, Traits>::
  getValidDepthParameters() const
  {
    Teuchos::RCP<Teuchos::ParameterList> valid_pl =
      rcp(new Teuchos::ParameterList("Valid Depth Params"));
  
    valid_pl->set<double>("Frequency", 1.0);
    valid_pl->set<double>("Absortivity", 1.0);
    valid_pl->set<double>("Reflectivity", 1.0);
    valid_pl->set<double>("Powder Diameter", 1.0);
    valid_pl->set<double>("Laser Beam Radius", 1.0);
    valid_pl->set<double>("Powder Layer Thickness", 1.0);
    valid_pl->set<double>("Average Laser Power", 1.0);


  
    return valid_pl;
  }

  //**********************************************************************

}
