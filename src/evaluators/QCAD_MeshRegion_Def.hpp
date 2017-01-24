//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
  limitX = limitY = limitZ = false;
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

  //Restriction to xy polygon
  const int polyPtDim = 2; //expect (x,y) coordinates
  int nPts = 0; 
  if( p.isSublist("XY Polygon") ) {
    Teuchos::Array<double> ar;
    Teuchos::ParameterList& polyList = p.sublist("XY Polygon");
    nPts = polyList.get<int>("Number of Points",0);
    xyPolygon.resize(nPts);

    for(int i=0; i<nPts; i++) {
      xyPolygon[i].resize(numDims);
      ar = polyList.get<Teuchos::Array<double> >( Albany::strint("Point",i) );
      TEUCHOS_TEST_FOR_EXCEPTION (ar.size() != (int)polyPtDim, Teuchos::Exceptions::InvalidParameter, std::endl 
				  << "XY-polygon point does not have " << polyPtDim << " elements" << std::endl); 
      for(std::size_t k=0; k<polyPtDim; k++) xyPolygon[i][k] = ar[k];
    }
  }
  bRestrictToXYPolygon = (nPts >= 3); //need at least three points to restrict to a polygon

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
  evaluator->addDependentField(coordVec.fieldTag());

  if(bRestrictToLevelSet) {
    PHX::MDField<MeshScalarT,Cell,QuadPoint> g(weightsFieldname, dl->qp_scalar); 
    weights = g;
    evaluator->addDependentField(weights.fieldTag());

    PHX::MDField<ScalarT> h(levelSetFieldname, dl->qp_scalar); 
    levelSetField = h;
    evaluator->addDependentField(levelSetField.fieldTag());
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

    if(bRestrictToXYPolygon) {
      RealType pt[2]; //MP: when MeshScalarT != double we take ADValue of that, is this correct?
      pt[0] = Albany::ADValue(coordVec(cell,qp,0));
      pt[1] = Albany::ADValue(coordVec(cell,qp,1));
      if( !QCAD::ptInPolygon(xyPolygon, pt) ) return false;
    }
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
