//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
//#include "Teuchos_CommHelpers.hpp"

template<typename EvalT, typename Traits>
QCAD::MeshRegion<EvalT, Traits>::
MeshRegion(std::string coordVecName, std::string weightsName,
	   Teuchos::ParameterList& p, 
	   const Teuchos::RCP<QCAD::MaterialDatabase> matDB,
	   const Teuchos::RCP<Albany::Layouts>& dl_ )
{
  materialDB = matDB;
  coordVecFieldname = coordVecName;
  weightsFieldname = weightsName;

  // number of quad points per cell and dimension of space
  std::vector<PHX::DataLayout::size_type> dims;
  dl = dl_;
  dl->qp_vector->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Restriction to element blocks
  std::string ebNameStr = p.get<std::string>("Element Block Name","");
  if(ebNameStr.length() == 0) ebNameStr = p.get<std::string>("Element Block Names","");
  if(ebNameStr.length() > 0) Albany::splitStringOnDelim(ebNameStr,',',ebNames);
  bQuantumEBsOnly = p.get<bool>("Quantum Element Blocks Only",false);

  // Restriction to coordinate ranges
  if( p.isParameter("x min") && p.isParameter("x max") ) {
    limitX = true; TEUCHOS_TEST_FOR_EXCEPT(numDims <= 0);
    xmin = p.get<double>("x min");
    xmax = p.get<double>("x max");
  }
  if( p.isParameter("y min") && p.isParameter("y max") ) {
    limitY = true; TEUCHOS_TEST_FOR_EXCEPT(numDims <= 1);
    ymin = p.get<double>("y min");
    ymax = p.get<double>("y max");
  }
  if( p.isParameter("z min") && p.isParameter("z max") ) {
    limitZ = true; TEUCHOS_TEST_FOR_EXCEPT(numDims <= 2);
    zmin = p.get<double>("z min");
    zmax = p.get<double>("z max");
  }

  // Restriction to a level set of a field
  levelSetFieldname = p.get<std::string>("Level Set Field Name","");
  levelSetFieldMin = p.get<double>("Level Set Field Minimum", -1e100);
  levelSetFieldMax = p.get<double>("Level Set Field Maximum", +1e100);
  bRestrictToLevelSet = (levelSetFieldname.length() > 0);

}



// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::MeshRegion<EvalT, Traits>::
addDependentFields(PHX::EvaluatorWithBaseImpl<Traits>* evaluator)
{
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> f(coordVecFieldname, dl->qp_vector); 
  coordVec = f;
  evaluator->addDependentField(coordVec);

  if(bRestrictToLevelSet) {
    PHX::MDField<MeshScalarT,Cell,QuadPoint> g(weightsFieldname, dl->qp_scalar); 
    weights = g;
    evaluator->addDependentField(weights);

    PHX::MDField<ScalarT> h(levelSetFieldname, dl->qp_scalar); 
    levelSetField = h;
    evaluator->addDependentField(levelSetField);
  }
}


// **********************************************************************
template<typename EvalT, typename Traits>
void QCAD::MeshRegion<EvalT, Traits>::
postRegistrationSetup(PHX::FieldManager<Traits>& fm)
{
  utils.setFieldData(coordVec,fm);
  if(bRestrictToLevelSet) {
    utils.setFieldData(weights,fm);
    utils.setFieldData(levelSetField,fm);
  }
}



// **********************************************************************
template<typename EvalT, typename Traits>
bool QCAD::MeshRegion<EvalT, Traits>::
elementBlockIsInRegion(std::string ebName) const
{
  bool bQuantumEB = false; //assume eb's aren't "quantum" unless we're told they are by the material DB

  if(materialDB != Teuchos::null)
    bQuantumEB = materialDB->getElementBlockParam<bool>(ebName,"quantum",false);
    
  if(bQuantumEBsOnly == true && bQuantumEB == false)
    return false;

  if(ebNames.size() > 0 && std::find(ebNames.begin(), ebNames.end(), ebName) == ebNames.end()) 
    return false;

  return true;
}

template<typename EvalT, typename Traits>
bool QCAD::MeshRegion<EvalT, Traits>::
cellIsInRegion(std::size_t cell)
{
  //Check that cell lies *entirely* in box
  for (std::size_t qp=0; qp < numQPs; ++qp) {
    if( (limitX && (coordVec(cell,qp,0) < xmin || coordVec(cell,qp,0) > xmax)) ||
	(limitY && (coordVec(cell,qp,1) < ymin || coordVec(cell,qp,1) > ymax)) ||
	(limitZ && (coordVec(cell,qp,2) < zmin || coordVec(cell,qp,2) > zmax)) )
      return false;
  }

  if(bRestrictToLevelSet) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

      // Get the average value for the cell (integral of field over cell divided by cell volume)
      ScalarT cellVol = 0.0, avgCellVal = 0.0;;
      for (std::size_t qp=0; qp < numQPs; ++qp) {
	avgCellVal += levelSetField(cell,qp) * weights(cell,qp);
	cellVol += weights(cell,qp);
      }
      avgCellVal /= cellVol;
      
      if( avgCellVal > levelSetFieldMax || avgCellVal < levelSetFieldMin)
	return false;
    }
  }	
	 
  return true;
}


// **********************************************************************
template<typename EvalT,typename Traits>
Teuchos::RCP<const Teuchos::ParameterList>
QCAD::MeshRegion<EvalT, Traits>::getValidParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     	rcp(new Teuchos::ParameterList("Valid MeshRegion Params"));;

  validPL->set<string>("Operation Domain", "", "Deprecated - does nothing"); //TODO: remove?

  validPL->set<string>("Element Block Name", "", "Element block name to restrict region to");
  validPL->set<string>("Element Block Names", "", "Element block names to restrict region to");
  validPL->set<bool>("Quantum Element Blocks Only", false, "Restricts region to quantum element blocks");

  validPL->set<double>("x min", 0.0, "Box domain minimum x coordinate");
  validPL->set<double>("x max", 0.0, "Box domain maximum x coordinate");
  validPL->set<double>("y min", 0.0, "Box domain minimum y coordinate");
  validPL->set<double>("y max", 0.0, "Box domain maximum y coordinate");
  validPL->set<double>("z min", 0.0, "Box domain minimum z coordinate");
  validPL->set<double>("z max", 0.0, "Box domain maximum z coordinate");

  validPL->set<string>("Level Set Field Name", "<field name>","Scalar Field to use for level set region");
  validPL->set<double>("Level Set Field Minimum", 0.0, "Minimum value of field to include in region");
  validPL->set<double>("Level Set Field Maximum", 0.0, "Maximum value of field to include in region");

  return validPL;
}
