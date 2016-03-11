//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

#include "Aeras_Eta.hpp"
#include "Aeras_ShallowWaterConstants.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_SurfaceGeopotential<EvalT, Traits>::
XZHydrostatic_SurfaceGeopotential(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  PhiSurf       (p.get<std::string> ("SurfaceGeopotential"), dl->node_scalar),
  numNodes ( dl->node_scalar          ->dimension(1)),
  numParam(0)

{

  Teuchos::ParameterList* xzhydrostatic_list = p.get<Teuchos::ParameterList*>("XZHydrostatic Problem");

  //std::string topoTypeString = surfGeopList->get<std::string>("Topography Type", "None");
  
  std::string topoTypeString = xzhydrostatic_list->get<std::string> ("Topography Type", "None");
  
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  
  const bool invalidString = (topoTypeString != "None" && topoTypeString != "Mountain1") && (topoTypeString != "SphereMountain1");
  
  TEUCHOS_TEST_FOR_EXCEPTION( invalidString,
                             std::logic_error,
                             "Unknown topography type string of " << topoTypeString
                             << " encountered. " << std::endl);
  

  
  
  
  if (topoTypeString == "None"){
    topoType = NONE;
  }
  else if (topoTypeString == "Mountain1") {
    
    topoType = MOUNTAIN1;
    
    numParam = 3;
    
    Teuchos::Array<double> defaultData(numParam);
    
    defaultData[0] = 150.0;//center
    defaultData[1] = 50.0;//width
    defaultData[2] = 1000.0;//height
    
    topoData = xzhydrostatic_list->get("Topography Data", defaultData);
    
    TEUCHOS_TEST_FOR_EXCEPTION((topoData.size() != numParam),
                               std::logic_error,
                               "Error! Invalid specification of params for Mountain1: incorrect length of " <<
                               "Topography Data ; required numParam = " << numParam <<
                               ", provided data.size() = " << topoData.size()
                               <<  std::endl) ;
  }
  else if ( topoTypeString == "SphereMountain1") {
  	topoType = SPHERE_MOUNTAIN1;
  	
  	numParam = 3;
  	Teuchos::Array<double> defaultData(numParam);
  	
  	defaultData[0] = 2000.0; // height
  	defaultData[1] = 2.356194490192345; // width = 3 * pi / 4 (radians)
  	defaultData[2] = 0.196349540849362; // halfWidth = pi / 16.0 (radians)
  	
  	topoData = xzhydrostatic_list->get("Topography Data", defaultData);
    
    TEUCHOS_TEST_FOR_EXCEPTION((topoData.size() != numParam),
                               std::logic_error,
                               "Error! Invalid specification of params for SphereMountain1: incorrect length of " <<
                               "Topography Data ; required numParam = " << numParam <<
                               ", provided data.size() = " << topoData.size()
                               <<  std::endl) ;
  }
  
  
  this->addEvaluatedField(PhiSurf);

  this->setName("Aeras::XZHydrostatic_SurfaceGeopotential"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_SurfaceGeopotential<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(PhiSurf      , fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_SurfaceGeopotential<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  //these will eventually be grabbed from a constants class
  //it is not clear if in XZHydro gravity is used
  double local_pi = 3.141592653589793;
  double local_gravity = 9.80616;
  
  if (topoType == NONE){
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
      
        //How to get x coordinate:
        //workset.wsCoords[cell][node][0]
        PhiSurf(cell,node) = 0.0;
      
        //std::cout << "topotype = none" <<std::endl;
        
        //std::cout << "cell="<<cell<<", node="<<node<<", coord x="<<
        //workset.wsCoords[cell][node][0] << std::endl;
      
      }
    }
  }
  else if (topoType == MOUNTAIN1) {
  
    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int node=0; node < numNodes; ++node) {
        
        double center = topoData[0];
        double width = topoData[1];
        double height = topoData[2];
        
        //How to get x coordinate:
        //workset.wsCoords[cell][node][0]
        
        double xcoord = workset.wsCoords[cell][node][0];
        if (std::abs(xcoord - center) <= (width/2.)) {
          PhiSurf(cell,node) =
                 (std::cos( (xcoord - center)*local_pi*2./width ) + 1.)
          *height/2. ;//*local_gravity;
        }else
          PhiSurf(cell,node) = 0.0;
        
        //std::cout << "topotype = mountain1"<<std::endl;
        //std::cout << "center width height"<<center<<" "<<width<<" "<<height<<std::endl;
        //std::cout << "cell="<<cell<<", node="<<node<<", coord x="<<
        //xcoord <<" and f_s="<<PhiSurf(cell,node) << std::endl;
        
      }
    }
    
  }
  else if ( topoType == SPHERE_MOUNTAIN1 ){
  	const double cntrLat = 0.0;
	const double cntrLon = 4.712388980384690; // 3 * pi / 2

	const double mtnHeight = topoData[0];
	const double mtnWidth = topoData[1];
	const double mtnHalfWidth = topoData[2];
	
	const double PI = 3.141592653589793;
	
	const double G = Aeras::ShallowWaterConstants::self().gravity;
  
  	for ( int cell = 0; cell < workset.numCells; ++cell ) {
  		for ( int node = 0; node < numNodes; ++node ) {
			const double x = workset.wsCoords[cell][node][0];
			const double y = workset.wsCoords[cell][node][1];
			const double z = workset.wsCoords[cell][node][2];
			
			const double theta = std::atan2( z, std::sqrt( x*x + y*y ) );
			const double lambda = std::atan2( y, x );
			
			const double radialDist = std::acos( std::sin( cntrLat ) * std::sin( theta ) + 
					std::cos( cntrLon ) * std::cos( theta ) * std::cos( cntrLon - lambda ) );
					
			const double zsurf = radialDist < mtnWidth ? 0.5 * mtnHeight * ( 1.0 + std::cos ( PI * radialDist / mtnWidth ) ) *
				std::cos( PI * radialDist / mtnHalfWidth ) * std::cos( PI * radialDist / mtnHalfWidth ) : 0.0;
		
			PhiSurf(cell, node) = G * zsurf;
  		}
  	}
  }
      
}
}
