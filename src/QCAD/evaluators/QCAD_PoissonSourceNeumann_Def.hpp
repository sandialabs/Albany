//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include <string>

#include "Intrepid_FunctionSpaceTools.hpp"
//#include "Sacado_ParameterRegistration.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN


namespace QCAD {
const double pi = 3.1415926535897932385;

//**********************************************************************
template<typename EvalT, typename Traits>
PoissonSourceNeumannBase<EvalT, Traits>::
PoissonSourceNeumannBase(const Teuchos::ParameterList& p) :

  dl             (p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  meshSpecs      (p.get<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct")),
  offset         (p.get<Teuchos::Array<int> >("Equation Offset")),
  sideSetIDs     (p.get<Teuchos::Array<std::string> >("Side Set IDs")),
  coordVec       (p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector),
  surfaceElectronDensity("Surface Electron Density",dl->node_scalar)
{
  // The DOF offsets are contained in the Equation Offset array. The length of this array are the
  // number of DOFs we will set each call
  numDOFsSet = offset.size();

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> > ("Parameter Library");
  TEUCHOS_ASSERT( p.isType<Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB") );

  responseOnly = p.get< bool >("Response Only"); 
    //when true, just compute output fields and don't try to change the workset's residual
    // vector (as per a usual Neumann BC) since this memory hasn't been allocated.

  //! Material database - holds the scaling we need
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

  //! Energy unit of phi in eV
  energy_unit_in_eV = p.get<double>("Energy unit in eV");
  temperature = p.get<double>("Temperature");
  double length_unit_in_meters = p.get<double>("Length unit in meters");

  V0 = kbBoltz*temperature / energy_unit_in_eV; // kb*T in desired energy unit ( or kb*T/q in desired voltage unit) [myV]

  //! Constant energy reference for heterogeneous structures
  ScalarT qPhiRef = getReferencePotential(); // in [myV]

  //! Get prefactor and exponent offset term for each sideset
  std::size_t nSidesets = this->sideSetIDs.size();
  prefactors.resize(nSidesets);
  phiOffsets.resize(nSidesets);

  for(std::size_t i=0; i < nSidesets; ++i) {
    std::string ebName = materialDB->getSideSetParam<std::string>(this->sideSetIDs[i],"elementBlock","");
    double mdn = materialDB->getElementBlockParam<double>(ebName,"Electron DOS Effective Mass");
    double Tref = materialDB->getElementBlockParam<double>(ebName,"Reference Temperature"); // in [K]
    double Chi = materialDB->getElementBlockParam<double>(ebName,"Electron Affinity") / energy_unit_in_eV; // in [myV]

    //! constant prefactor in calculating Nc and Nv in [cm-2]
    double NcvFactor2D = (m0*kbBoltz*eleQ*Tref)/(pi*pow(hbar,2)) * 1e-4;  // m*kbT/pi*hbar^2
    // eleQ converts kbBoltz in [eV/K] to [J/K], 1e-4 converts [m-2] to [cm-2]
    // NcvFactor2D * cm^2 = [kg * J / (J*s)^2 * m^2/cm^2] * cm^2 = kg m^2 / (J*s^2) = 1 (unitless),
    //  so NcvFactor2D has units of [cm-2]

    double Nc = NcvFactor2D*mdn*temperature/Tref;  // in [cm-2]

    //Flux Scale factor, which converts an area electron density [cm-2] to slope units [myV]/[myL]
    double convert_densityToDeltaSlope = 1.0 / eps0 * eleQ * (length_unit_in_meters*1e2) / energy_unit_in_eV;
    //  (1 e/cm^-2) / [ C/(V*cm) ] = [eV/(C*cm)] => * [C/e] = [V/cm] => * [cm/myL] * [myV/V] = [myV/myL]
    
    prefactors[i] = convert_densityToDeltaSlope / 2 * Nc;  // [myV/myL] 
    // divide by 2 because we assume there are 2 sideset contributions for each patch of surface, and we 
    //  want the sum of these two contributions to equal the change in solution slope.
    
    double kbT = kbBoltz*temperature / energy_unit_in_eV; // in [myV] -- same as V0
    ScalarT eArgOffset = (-qPhiRef+Chi)/kbT; //unitless

    //HACK -- need to get fermiE from DBCs in the case when element block is in electrical contact with a nodeset (see QCAD_PoissonSource_Def.hpp)
    ScalarT fermiE = 0.0; 

    phiOffsets[i] = eArgOffset + fermiE/kbT; //unitless

    //Note on "Flux Scale" in materials.xml files:  areal density is typically given in [1e11 cm-2] units and the
    //  mesh given in [myL] = um.  The "Flux Scale" parameter converts between density units and 
    //  [V]/[myL], with an additional division by 2 b/c there are 2 sideset faces for each patch of surface,
    //  which in the typical case is:
    //  1e11 e/cm^-2 => 1e15 e/m^-2 then / e0 = 8.85e-12 C/(V*m) => 0.113e27 e*V/(C*m) * 1.602e-19 C/e
    //  => 0.181e8 V/m * 1m/1e6 um => 0.181e8e2 V/um / 2 (for the two sides?) ~= 9.05 V/um
  }

  //currently this evaluator doesn't support the case when the DOF is a vector
  TEUCHOS_TEST_FOR_EXCEPTION(p.get<bool>("Vector Field") == true,
                             Teuchos::Exceptions::InvalidParameter,
                             std::endl << "Error: PoissonSource Neumann boundary conditions "
                             << "only supported when the DOF is not a vector" << std::endl);

  PHX::MDField<ScalarT,Cell,Node> tmp(p.get<std::string>("DOF Name"),
				      p.get<Teuchos::RCP<PHX::DataLayout> >("DOF Data Layout"));
  dof = tmp;
  this->addDependentField(dof);
  this->addDependentField(coordVec);

  std::string name = "Neumann Poisson Source Evaluator";
  PHX::Tag<ScalarT> fieldTag(name, dl->dummy);

  this->addEvaluatedField(fieldTag);
  this->addEvaluatedField(surfaceElectronDensity);

  // Build element and side integration support (Copied from PHAL_Neumann_Def.hpp)

  const CellTopologyData * const elem_top = &meshSpecs->ctd;

  intrepidBasis = Albany::getIntrepidBasis(*elem_top);

  cellType = Teuchos::rcp(new shards::CellTopology (elem_top));

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  cubatureCell = cubFactory.create(*cellType, meshSpecs->cubatureDegree);

  int cubatureDegree = (p.get<int>("Cubature Degree") > 0 ) ? p.get<int>("Cubature Degree") : meshSpecs->cubatureDegree;

  int numSidesOnElem = elem_top->side_count;
  sideType.resize(numSidesOnElem);
  cubatureSide.resize(numSidesOnElem);
  side_type.resize(numSidesOnElem);

  // Build containers that depend on side topology
  int maxSideDim(0), maxNumQpSide(0);
  const char* sideTypeName;

  for(int i=0; i<numSidesOnElem; ++i) {
    sideType[i] = Teuchos::rcp(new shards::CellTopology(elem_top->side[i].topology));
    cubatureSide[i] = cubFactory.create(*sideType[i], cubatureDegree);
    maxSideDim = std::max( maxSideDim, (int)sideType[i]->getDimension());
    maxNumQpSide = std::max(maxNumQpSide, (int)cubatureSide[i]->getNumPoints());
    sideTypeName = sideType[i]->getName();
    if(strncasecmp(sideTypeName, "Line", 4) == 0)
      side_type[i] = LINE;
    else if(strncasecmp(sideTypeName, "Tri", 3) == 0)
      side_type[i] = TRI;
    else if(strncasecmp(sideTypeName, "Quad", 4) == 0)
      side_type[i] = QUAD;
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "QCAD_PoissonSourceNeumann: side type : " << sideTypeName << " is not supported." << std::endl);
  }

  numNodes = intrepidBasis->getCardinality();

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->qp_tensor->dimensions(dim);
  int containerSize = dim[0];
  numQPs = dim[1];
  cellDims = dim[2];

  // Allocate Temporary FieldContainers
  physPointsCell.resize(1, numNodes, cellDims);
  dofCell.resize(1, numNodes);
  dofCellVec.resize(1, numNodes, numDOFsSet);
  neumann.resize(containerSize, numNodes, numDOFsSet);

  // Allocate Temporary FieldContainers based on max sizes of sides. Need to be resized later for each side.
  cubPointsSide.resize(maxNumQpSide, maxSideDim);
  refPointsSide.resize(maxNumQpSide, cellDims);
  cubWeightsSide.resize(maxNumQpSide);
  physPointsSide.resize(1, maxNumQpSide, cellDims);
  dofSide.resize(1, maxNumQpSide);
  dofSideVec.resize(1, maxNumQpSide, numDOFsSet);

  // Do the BC one side at a time for now
  jacobianSide.resize(1, maxNumQpSide, cellDims, cellDims);
  jacobianSide_det.resize(1, maxNumQpSide);

  weighted_measure.resize(1, maxNumQpSide);
  basis_refPointsSide.resize(numNodes, maxNumQpSide);
  trans_basis_refPointsSide.resize(1, numNodes, maxNumQpSide);
  weighted_trans_basis_refPointsSide.resize(1, numNodes, maxNumQpSide);

  data.resize(1, maxNumQpSide, numDOFsSet);


  this->setName(name);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void PoissonSourceNeumannBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(dof,fm);
  this->utils.setFieldData(surfaceElectronDensity,fm);
  // Note, we do not need to add dependent field to fm here for output - that is done
  // by PoissonSourceNeumann Aggregator
}

template<typename EvalT, typename Traits>
void PoissonSourceNeumannBase<EvalT, Traits>::
evaluateNeumannContribution(typename Traits::EvalData workset)
{

  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!

  // GAH: Note that this loosely follows from
  // $TRILINOS_DIR/packages/intrepid/test/Discretization/Basis/HGRAD_QUAD_C1_FEM/test_02.cpp

  //Zero out outputs
  std::cout << "DEBUG: PSN eval neumann contribution: zeroing out fields" << std::endl;
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t node=0; node < numNodes; ++node)
      for (std::size_t dim=0; dim < numDOFsSet; ++dim) 
	neumann(cell, node, dim) = 0.0;

  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t node=0; node < numNodes; ++node)
      surfaceElectronDensity(cell, node) = 0.0;


  // Loop over each sideset
  for(std::size_t i = 0; i < this->sideSetIDs.size(); i++) {

    if(workset.sideSets == Teuchos::null || this->sideSetIDs[i].length() == 0)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
	 "Side sets defined in input file but not properly specified on the mesh" << std::endl);


    const Albany::SideSetList& ssList = *(workset.sideSets);
    Albany::SideSetList::const_iterator it = ssList.find(this->sideSetIDs[i]);
    if(it == ssList.end()) 
      std::cout << "DEBUG: Poisson Source Neumann: sideset " << this->sideSetIDs[i] << " does not exist in this workset!" << std::endl;
    else
      std::cout << "DEBUG: Poisson Source Neumann filling sideset " << this->sideSetIDs[i] << " for current workset" << std::endl;

    if(it == ssList.end()) return; // This sideset does not exist in this workset (GAH - this can go away
                                   // once we move logic to BCUtils


    Intrepid::FieldContainer<ScalarT> betaOnSide;
    Intrepid::FieldContainer<ScalarT> thicknessOnSide;
    Intrepid::FieldContainer<ScalarT> elevationOnSide;

    const std::vector<Albany::SideStruct>& sideSet = it->second;

    // Loop over the sides that form the boundary condition
    for (std::size_t side=0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side

      const int elem_GID = sideSet[side].elem_GID;
      const int elem_LID = sideSet[side].elem_LID;
      const int elem_side = sideSet[side].side_local_id;

      int sideDims = sideType[elem_side]->getDimension();
      int numQPsSide = cubatureSide[elem_side]->getNumPoints();


      //need to resize containers because they depend on side topology
      cubPointsSide.resize(numQPsSide, sideDims);
      refPointsSide.resize(numQPsSide, cellDims);
      cubWeightsSide.resize(numQPsSide);
      physPointsSide.resize(1, numQPsSide, cellDims);
      dofSide.resize(1, numQPsSide);
      dofSideVec.resize(1, numQPsSide, numDOFsSet);
      
      // Do the BC one side at a time for now
      jacobianSide.resize(1, numQPsSide, cellDims, cellDims);
      jacobianSide_det.resize(1, numQPsSide);
      
      weighted_measure.resize(1, numQPsSide);
      basis_refPointsSide.resize(numNodes, numQPsSide);
      trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);
      weighted_trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);
      data.resize(1, numQPsSide, numDOFsSet);
      
      betaOnSide.resize(1,numQPsSide);
      thicknessOnSide.resize(1,numQPsSide);
      elevationOnSide.resize(1,numQPsSide);
      
      cubatureSide[elem_side]->getCubature(cubPointsSide, cubWeightsSide);
      
      // Copy the coordinate data over to a temp container
      
      for (std::size_t node=0; node < numNodes; ++node)
	for (std::size_t dim=0; dim < cellDims; ++dim)
	  physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
      

      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      Intrepid::CellTools<RealType>::mapToReferenceSubcell
	(refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

      // Calculate side geometry
      Intrepid::CellTools<MeshScalarT>::setJacobian
	(jacobianSide, refPointsSide, physPointsCell, *cellType);

      Intrepid::CellTools<MeshScalarT>::setJacobianDet(jacobianSide_det, jacobianSide);
      
      if (sideDims < 2) { //for 1 and 2D, get weighted edge measure
	Intrepid::FunctionSpaceTools::computeEdgeMeasure<MeshScalarT>
	  (weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      }
      else { //for 3D, get weighted face measure
	Intrepid::FunctionSpaceTools::computeFaceMeasure<MeshScalarT>
	  (weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      }

      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid::OPERATOR_VALUE);

      // Transform values of the basis functions
      Intrepid::FunctionSpaceTools::HGRADtransformVALUE<MeshScalarT>
	(trans_basis_refPointsSide, basis_refPointsSide);

      // Multiply with weighted measure
      Intrepid::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
	(weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);
      
#ifdef ALBANY_USE_PUBLICTRILINOS
      // Map cell (reference) cubature points to the appropriate side (elem_side) in physical space
      Intrepid::CellTools<MeshScalarT>::mapToPhysicalFrame
	(physPointsSide, refPointsSide, physPointsCell, *cellType);
#else
      Intrepid::CellTools<MeshScalarT>::mapToPhysicalFrame
	(physPointsSide, refPointsSide, physPointsCell, intrepidBasis);
#endif
      
      // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
      for (std::size_t node=0; node < numNodes; ++node)
	dofCell(0, node) = dof(elem_LID, node);

      // This is needed, since evaluate currently sums into
      for (int k=0; k < numQPsSide ; k++) dofSide(0,k) = 0.0;

      // Get dof at cubature points of appropriate side (see DOFInterpolation evaluator)
      Intrepid::FunctionSpaceTools::
	evaluate<ScalarT>(dofSide, dofCell, trans_basis_refPointsSide);
      
      // Get dof at cubature points of appropriate side (see DOFVecInterpolation evaluator)
      //Intrepid::FunctionSpaceTools::
      //evaluate<ScalarT>(dofSide, dofCell, trans_basis_refPointsSide);

      // Transform the given BC data to the physical space QPs in each side (elem_side)
      calc_dudn_2DThomasFermi(data, physPointsSide, dofSide, jacobianSide, *cellType, cellDims, elem_side, i);

      // Put this side's contribution into the vector
      for (std::size_t node=0; node < numNodes; ++node) {
	for (std::size_t qp=0; qp < numQPsSide; ++qp) {
	  for (std::size_t dim=0; dim < numDOFsSet; ++dim) {
	    neumann(elem_LID, node, dim) +=
	      data(0, qp, dim) * weighted_trans_basis_refPointsSide(0, node, qp);
	  }
	}
	if(responseOnly) {
	  surfaceElectronDensity(elem_LID, node) = neumann(elem_LID, node, 0);
	  //std::cout << "DEBUG: Setting Poisson Source Neumann (" << elem_LID << "," << node << ") = " << surfaceElectronDensity(elem_LID, node) << std::endl;
	}  
      }

    }
  }
}

template<typename EvalT, typename Traits>
void PoissonSourceNeumannBase<EvalT, Traits>::
calc_dudn_2DThomasFermi(Intrepid::FieldContainer<ScalarT> & qp_data_returned,
			const Intrepid::FieldContainer<MeshScalarT>& phys_side_cub_points,
			const Intrepid::FieldContainer<ScalarT>& dof_side,
			const Intrepid::FieldContainer<MeshScalarT>& jacobian_side_refcell,
			const shards::CellTopology & celltopo,
			const int cellDims,
			int local_side_id, int iSideset){

  int numCells = qp_data_returned.dimension(0); // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.dimension(1); // How many QPs per cell?
  int numDOFs = qp_data_returned.dimension(2); // How many DOFs per node to calculate?

  for(int pt = 0; pt < numPoints; pt++) {
    for(int dim = 0; dim < numDOFsSet; dim++) {
      //ScalarT ten = 10.0;
      //qp_data_returned(0, pt, dim) = ten; //for DEBUG

      const ScalarT& unscaled_phi = dof_side(0,pt); // [myV]
      ScalarT phi = unscaled_phi / V0; // unitless, as V0 == kbT in [myV]
      qp_data_returned(0, pt, dim) = prefactors[iSideset] * log (1.0 + exp(phi + phiOffsets[iSideset]) ); // [myV/myL] (for delta(slope) of soln)
    }
  }
}

template<typename EvalT, typename Traits>
typename QCAD::PoissonSourceNeumannBase<EvalT,Traits>::ScalarT 
PoissonSourceNeumannBase<EvalT, Traits>::getReferencePotential()
{
  //! Constant energy reference for heterogeneous structures
  ScalarT qPhiRef;

  std::string refMtrlName, category;
  refMtrlName = materialDB->getParam<std::string>("Reference Material");
  category = materialDB->getMaterialParam<std::string>(refMtrlName,"Category");
  if (category == "Semiconductor") {
 
    // Get quantities in desired energy (voltage) units, which we denote "[myV]"
 
    // Same qPhiRef needs to be used for the entire structure
    double mdn = materialDB->getMaterialParam<double>(refMtrlName,"Electron DOS Effective Mass");
    double mdp = materialDB->getMaterialParam<double>(refMtrlName,"Hole DOS Effective Mass");
    double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity") / energy_unit_in_eV; // in [myV]
    double Eg0 = materialDB->getMaterialParam<double>(refMtrlName,"Zero Temperature Band Gap") / energy_unit_in_eV; // in [myV]
    double alpha = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Alpha Coefficient"); // in [eV/K]
    double beta = materialDB->getMaterialParam<double>(refMtrlName,"Band Gap Beta Coefficient");  // in [K]
    ScalarT Eg = Eg0-(alpha*pow(temperature,2.0)/(beta+temperature)) / energy_unit_in_eV; // in [myV]
  
    ScalarT kbT = kbBoltz*temperature / energy_unit_in_eV; // in [myV] (desired voltage unit)
    ScalarT Eic = -Eg/2. + 3./4.*kbT*log(mdp/mdn);  // (Ei-Ec) in [myV]
    qPhiRef = Chi - Eic;  // (Evac-Ei) in [myV] where Evac = vacuum level
  }
  else if (category == "Insulator") {
    double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");
    qPhiRef = Chi / energy_unit_in_eV; // in [myV]
  }
  else if (category == "Metal") {
    double workFn = materialDB->getMaterialParam<double>(refMtrlName,"Work Function"); 
    qPhiRef = workFn / energy_unit_in_eV; // in [myV]
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, std::endl 
 			  << "Error!  Invalid category " << category 
 			  << " for reference material !" << std::endl);
  }

  return qPhiRef;
}
  


// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::Residual,Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::Residual,Traits>(p)
{
}



template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill in "neumann" array
  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();
  ScalarT *valptr;

  // Place it at the appropriate offset into F
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

      valptr = &(this->neumann)(cell, node, dim);
     fT_nonconstView[nodeID[node][this->offset[dim]]] += *valptr;

    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::Jacobian, Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::Jacobian,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill in "neumann" array
  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;

  ScalarT *valptr;

  int lcol;
  Teuchos::Array<LO> rowT(1);
  Teuchos::Array<LO> colT(1);
  Teuchos::Array<ST> value(1);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

      rowT[0] = nodeID[node][this->offset[dim]];

      int neq = nodeID[node].size();

      if (fT != Teuchos::null) {
         fT->sumIntoLocalValue(rowT[0], valptr->val());
      }

        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

            // Global column
            colT[0] =  nodeID[node_col][eq_col];
            value[0] = valptr->fastAccessDx(lcol);   
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              JacT->sumIntoLocalValues(colT[0], rowT(), value());
            }
            else {
              // Sum Jacobian
            JacT->sumIntoLocalValues(rowT[0], colT(), value());
            }
          } // column equations
        } // column nodes
      } // has fast access
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::Tangent, Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::Tangent,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = workset.JVT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = workset.fpT;
  
   ScalarT *valptr;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        int row = nodeID[node][this->offset[dim]];

        if (fT != Teuchos::null)
          fT->sumIntoLocalValue(row, valptr->val());

        if (JVT != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)

            JVT->sumIntoLocalValue(row, col, valptr->dx(col));

        if (fpT != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            fpT->sumIntoLocalValue(row, col, valptr->dx(col+workset.param_offset));
      }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP<Tpetra_MultiVector> fpVT = workset.fpVT;
  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.VpT->getNumVectors();
  ScalarT *valptr;

  if (trans) {
    int neq = workset.numEqs;
    const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = local_Vp.size()/neq;
      for (int i=0; i<num_deriv; i++) {
        for (int col=0; col<num_cols; col++) {
          double val = 0.0;
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){
              valptr = &(this->neumann)(cell, node, dim);
              int eq = this->offset[dim];
              val += valptr->dx(i)*local_Vp[node*neq+eq][col];
            }
          }
          const LO row = wsElDofs((int)cell,i,0);
          fpVT->sumIntoLocalValue(row, col, val);
        }
      }
    }

  }

  else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  =
        workset.wsElNodeEqID[cell];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = local_Vp.size();

      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){
          valptr = &(this->neumann)(cell, node, dim);
          const int row = nodeID[node][this->offset[dim]];
          for (int col=0; col<num_cols; col++) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i)
              val += valptr->dx(i)*local_Vp[i][col];
            fpVT->sumIntoLocalValue(row, col, val);
          }
        }
    }

  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::SGResidual, Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::SGResidual,Traits>(p)
{
}


// **********************************************************************
template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  ScalarT *valptr;

  int nblock = f->size();


  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][this->offset[dim]]] += valptr->coeff(block);

    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::SGJacobian, Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::SGJacobian,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > Jac =
    workset.sg_Jac;
  ScalarT *valptr;


  int row, lcol, col;
  int nblock = 0;

  if (f != Teuchos::null)
    nblock = f->size();

  int nblock_jac = Jac->size();
  double c; // use double since it goes into CrsMatrix

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        row = nodeID[node][this->offset[dim]];
        int neq = nodeID[node].size();

        if (f != Teuchos::null) {

          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));

        }

        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID[node_col][eq_col];

              // Sum Jacobian
              for (int block=0; block<nblock_jac; block++) {

                c = valptr->fastAccessDx(lcol).coeff(block);
                if (workset.is_adjoint) {

                  (*Jac)[block].SumIntoMyValues(col, 1, &c, &row);

                }
                else {

                  (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);

                }
              }
            } // column equations
          } // column nodes
        } // has fast access
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::SGTangent, Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::SGTangent,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > JV = workset.sg_JV;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > fp = workset.sg_fp;
  ScalarT *valptr;


  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  else if (JV != Teuchos::null)
    nblock = JV->size();
  else if (fp != Teuchos::null)
    nblock = fp->size();
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                       "One of sg_f, sg_JV, or sg_fp must be non-null! " <<
                       std::endl);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {

    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        int row = nodeID[node][this->offset[dim]];

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, valptr->dx(col).coeff(block));

          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, valptr->dx(col+workset.param_offset).coeff(block));
    }
  }
}
#endif 
#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::MPResidual, Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::MPResidual,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  ScalarT *valptr;


  int nblock = f->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        for (int block=0; block<nblock; block++)
          (*f)[block][nodeID[node][this->offset[dim]]] += valptr->coeff(block);

    }
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::MPJacobian, Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > Jac =
    workset.mp_Jac;
  ScalarT *valptr;


  int row, lcol, col;
  int nblock = 0;

  if (f != Teuchos::null)
    nblock = f->size();

  int nblock_jac = Jac->size();
  double c; // use double since it goes into CrsMatrix

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        row = nodeID[node][this->offset[dim]];
        int neq = nodeID[node].size();

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));


        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID[node_col][eq_col];

              // Sum Jacobian
              for (int block=0; block<nblock_jac; block++) {

                c = valptr->fastAccessDx(lcol).coeff(block);
               (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);

             }
            } // column equations
          } // column nodes
        } // has fast access
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
PoissonSourceNeumann<PHAL::AlbanyTraits::MPTangent, Traits>::
PoissonSourceNeumann(Teuchos::ParameterList& p)
  : PoissonSourceNeumannBase<PHAL::AlbanyTraits::MPTangent,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceNeumann<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateNeumannContribution(workset);

  if(this->responseOnly) return; //short-circuit for "response only" mode

  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > JV = workset.mp_JV;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > fp = workset.mp_fp;
  ScalarT *valptr;


  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  else if (JV != Teuchos::null)
    nblock = JV->size();
  else if (fp != Teuchos::null)
    nblock = fp->size();
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                       "One of mp_f, mp_JV, or mp_fp must be non-null! " <<
                       std::endl);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node)
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim){

        valptr = &(this->neumann)(cell, node, dim);

        int row = nodeID[node][this->offset[dim]];

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, valptr->dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, valptr->dx(col+workset.param_offset).coeff(block));

    }
  }
}
#endif


}
