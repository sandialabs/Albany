//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include <string>

#include "Intrepid2_FunctionSpaceTools.hpp"
//#include "Sacado_ParameterRegistration.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN


namespace QCAD {

//**********************************************************************
template<typename EvalT, typename Traits>
PoissonSourceInterfaceBase<EvalT, Traits>::
PoissonSourceInterfaceBase(const Teuchos::ParameterList& p) :

  dl             (p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  meshSpecs      (p.get<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct")),
  offset         (p.get<Teuchos::Array<int> >("Equation Offset")),
  sideSetIDs     (p.get<Teuchos::Array<std::string> >("Side Set IDs")),
  coordVec       (p.get<std::string>("Coordinate Vector Name"), dl->vertices_vector)
{
  // The DOF offsets are contained in the Equation Offset array. The length of this array are the
  // number of DOFs we will set each call
  numDOFsSet = offset.size();  // always 1 for Poisson problems

  TEUCHOS_ASSERT(p.isType<Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB") );

  //! Material database - holds the scaling we need
  materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

  //! Energy unit of phi in eV
  energy_unit_in_eV = p.get<double>("Energy unit in eV");
  length_unit_in_meters = p.get<double>("Length unit in meters");
  temperature = p.get<double>("Temperature");
  
  // length scaling to get to [cm] (structure dimension in [um] usually)
  X0 = length_unit_in_meters / 1e-2; 
  
  // kb*T in desired energy unit ( or kb*T/q in desired voltage unit) of [eV]
  kbT = kbBoltz*temperature / energy_unit_in_eV;  
  
  //! Get the "Interface Traps" parameter list
  Teuchos::ParameterList& trapsPList = *p.get<Teuchos::ParameterList*>("Interface Traps Parameter List"); 
  
  //! Constant energy reference for heterogeneous structures
  qPhiRef = getReferencePotential(); // in [eV]

  //! Get electron affinity, band gap, fermi energy, and trap params for each sideset
  std::size_t nSidesets = this->sideSetIDs.size();
  elecAffinity.resize(nSidesets);
  bandGap.resize(nSidesets);
  fermiEnergy.resize(nSidesets); 

  trapSpectrum.resize(nSidesets);
  trapType.resize(nSidesets); 
  trapDensity.resize(nSidesets);
  acceptorDegFac.resize(nSidesets);
  donorDegFac.resize(nSidesets); 

  for(std::size_t i=0; i < nSidesets; ++i) 
  {
    std::string ebName = materialDB->getSideSetParam<std::string>(this->sideSetIDs[i],"elementBlock","");
    
    double Chi = materialDB->getElementBlockParam<double>(ebName,"Electron Affinity") / energy_unit_in_eV; // in [myV]
    double Eg0 = materialDB->getElementBlockParam<double>(ebName,"Zero Temperature Band Gap") / energy_unit_in_eV; // in [myV];
    double alpha = materialDB->getElementBlockParam<double>(ebName,"Band Gap Alpha Coefficient");
    double beta = materialDB->getElementBlockParam<double>(ebName,"Band Gap Beta Coefficient");

    elecAffinity[i] = Chi; 
    bandGap[i] = Eg0 - (alpha*std::pow(temperature,2.0)/(beta+temperature)) / energy_unit_in_eV; // in [myV]
    
    //HACK -- need to get fermiE from DBCs in the case when element block is in electrical contact with a nodeset (see QCAD_PoissonSource_Def.hpp)
    fermiEnergy[i] = 0.0; 
    
    // Obtain trap parameters for each sideset
    std::stringstream intrapsstrm;
    intrapsstrm << "Interface Trap for SS " << sideSetIDs[i];
    std::string intrapstr = intrapsstrm.str();
    
    if (!trapsPList.isSublist(intrapstr))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, std::endl << "Error: " 
        << intrapstr << " parameterlist is NOT found !" << std::endl);
    
    Teuchos::ParameterList& trapSublist = trapsPList.sublist(intrapstr); 
    trapSpectrum[i] = trapSublist.get<std::string>("Energy Spectrum");
    trapType[i] = trapSublist.get<std::string>("Trap Type");      
    trapDensity[i] = trapSublist.get<double>("Trap Density"); 
    
    acceptorDegFac[i] = 1; // by default; 
    if (trapSublist.isParameter("Acceptor Degeneracy Factor"))
      acceptorDegFac[i] = trapSublist.get<double>("Acceptor Degeneracy Factor");
    
    donorDegFac[i] = 1; // by default  
    if (trapSublist.isParameter("Donor Degeneracy Factor"))
      donorDegFac[i] = trapSublist.get<double>("Donor Degeneracy Factor");
  }

  //currently this evaluator doesn't support the case when the DOF is a vector
  TEUCHOS_TEST_FOR_EXCEPTION(p.get<bool>("Vector Field") == true,
                             Teuchos::Exceptions::InvalidParameter,
                             std::endl << "Error: PoissonSource Interface boundary conditions "
                             << "are only supported when the DOF is not a vector" << std::endl);

  //PHX::MDField<ScalarT,Cell,Node> tmp(p.get<std::string>("DOF Name"),
  //p.get<Teuchos::RCP<PHX::DataLayout> >("DOF Data Layout"));
  //dof = tmp;
  
  dof = PHX::MDField<ScalarT,Cell,Node>(p.get<std::string>("DOF Name"),
				     p.get<Teuchos::RCP<PHX::DataLayout> >("DOF Data Layout"));
  this->addDependentField(dof);
  this->addDependentField(coordVec);

  std::string name = "Poisson Source Interface Evaluator";
  PHX::Tag<ScalarT> fieldTag(name, dl->dummy);

  this->addEvaluatedField(fieldTag);

  // Build element and side integration support (Copied from PHAL_Neumann_Def.hpp)

  const CellTopologyData * const elem_top = &meshSpecs->ctd;

  intrepidBasis = Albany::getIntrepid2Basis(*elem_top);

  cellType = Teuchos::rcp(new shards::CellTopology (elem_top));

  Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
  cubatureCell = cubFactory.create(*cellType, meshSpecs->cubatureDegree);

  int cubatureDegree = (p.get<int>("Cubature Degree") > 0 ) ? p.get<int>("Cubature Degree") : meshSpecs->cubatureDegree;

  int numSidesOnElem = elem_top->side_count;
  sideType.resize(numSidesOnElem);
  cubatureSide.resize(numSidesOnElem);
  side_type.resize(numSidesOnElem);

  // Build containers that depend on side topology
  int maxSideDim(0), maxNumQpSide(0);
  const char* sideTypeName;

  for(int i = 0; i < numSidesOnElem; ++i) 
  {
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
        "QCAD_PoissonSourceInterface: side type : " << sideTypeName << " is not supported." << std::endl);
  }

  numNodes = intrepidBasis->getCardinality();  // number of nodes in a volume cell

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->qp_tensor->dimensions(dim);
  int containerSize = dim[0];  // number of volume cells
  numQPs = dim[1];             // number of QPs in a cell
  cellDims = dim[2];           // number of spatial dimensions in a cell

  // Allocate Temporary FieldContainers
  physPointsCell.resize(1, numNodes, cellDims);
  dofCell.resize(1, numNodes);
  neumann.resize(containerSize, numNodes, numDOFsSet);

  // Allocate Temporary FieldContainers based on max sizes of sides. Need to be resized later for each side.
  cubPointsSide.resize(maxNumQpSide, maxSideDim);
  refPointsSide.resize(maxNumQpSide, cellDims);
  cubWeightsSide.resize(maxNumQpSide);
  physPointsSide.resize(1, maxNumQpSide, cellDims);
  dofSide.resize(1, maxNumQpSide);

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
void PoissonSourceInterfaceBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(dof,fm);
}

template<typename EvalT, typename Traits>
void PoissonSourceInterfaceBase<EvalT, Traits>::
evaluateInterfaceContribution(typename Traits::EvalData workset)
{
  // setJacobian only needs to be RealType since the data type is only
  //  used internally for Basis Fns on reference elements, which are
  //  not functions of coordinates. This save 18min of compile time!!!

  // GAH: Note that this loosely follows from
  // $TRILINOS_DIR/packages/intrepid/test/Discretization/Basis/HGRAD_QUAD_C1_FEM/test_02.cpp

  //Zero out outputs
  // std::cout << "DEBUG: PSI eval interface contribution: zeroing out fields" << std::endl;
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t node=0; node < numNodes; ++node)
      for (std::size_t dim=0; dim < numDOFsSet; ++dim) 
        neumann(cell, node, dim) = 0.0;

  // Loop over each sideset
  for(std::size_t i = 0; i < this->sideSetIDs.size(); i++) 
  {
    if(workset.sideSets == Teuchos::null || this->sideSetIDs[i].length() == 0)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
	      "Side sets defined in input file but not properly specified on the mesh" << std::endl);

    const Albany::SideSetList& ssList = *(workset.sideSets);
    Albany::SideSetList::const_iterator it = ssList.find(this->sideSetIDs[i]);
    //if(it == ssList.end()) 
    //  std::cout << "DEBUG: Poisson Source Interface: sideset " << this->sideSetIDs[i] << " does not exist in this workset!" << std::endl;
    //else
    //  std::cout << "DEBUG: Poisson Source Interface filling sideset " << this->sideSetIDs[i] << " for current workset" << std::endl;

    if(it == ssList.end()) return; // This sideset does not exist in this workset (GAH - this can go away
                                   // once we move logic to BCUtils

    const std::vector<Albany::SideStruct>& sideSet = it->second;

    // Loop over the side edges/faces that are a part of a given sideset and belong to the current workset
    for (std::size_t side = 0; side < sideSet.size(); ++side) 
    { 
      // Get the data that correspond to the side
      const int elem_GID = sideSet[side].elem_GID;       // the global id of the element containing the side 
      const int elem_LID = sideSet[side].elem_LID;       // the local id of the element containing the side
      const int elem_side = sideSet[side].side_local_id; // the local id of the side relative to the owning element

      int sideDims = sideType[elem_side]->getDimension();       // spatial dimensions of a side (1 for edge, 2 for face)
      int numQPsSide = cubatureSide[elem_side]->getNumPoints(); // ??? number of QPs on a side or element ???

      // Need to resize containers because they depend on side topology
      cubPointsSide.resize(numQPsSide, sideDims);
      refPointsSide.resize(numQPsSide, cellDims);
      cubWeightsSide.resize(numQPsSide);
      physPointsSide.resize(1, numQPsSide, cellDims);
      dofSide.resize(1, numQPsSide);
      
      // Do the BC one side edge/face at a time for now
      jacobianSide.resize(1, numQPsSide, cellDims, cellDims);
      jacobianSide_det.resize(1, numQPsSide);
      
      weighted_measure.resize(1, numQPsSide);
      basis_refPointsSide.resize(numNodes, numQPsSide);
      trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);
      weighted_trans_basis_refPointsSide.resize(1, numNodes, numQPsSide);
      data.resize(1, numQPsSide, numDOFsSet);
      
      // Return cubature points and weights for a side in a reference frame
      cubatureSide[elem_side]->getCubature(cubPointsSide, cubWeightsSide);  
      
      // Copy the coordinate data of the element to a temp container
      for (std::size_t node = 0; node < numNodes; ++node)
        for (std::size_t dim = 0; dim < cellDims; ++dim)
	        physPointsCell(0, node, dim) = coordVec(elem_LID, node, dim);
      
      // Map side cubature points to the reference parent cell based on the appropriate side (elem_side)
      // refPointsSide is the QPs of the side in form of (numQPsSide, cellDims), 
      // while cubPointsSide is the QPs of the side in form of (numQPsSide, sideDims).
      Intrepid2::CellTools<RealType>::mapToReferenceSubcell
        (refPointsSide, cubPointsSide, sideDims, elem_side, *cellType);

      // Calculate Jacobian for refPointsSide on the side
      Intrepid2::CellTools<MeshScalarT>::setJacobian(jacobianSide, refPointsSide, physPointsCell, *cellType);

      Intrepid2::CellTools<MeshScalarT>::setJacobianDet(jacobianSide_det, jacobianSide);
      
      if (sideDims < 2)  //for 1 and 2D, get weighted edge measure
      {
        Intrepid2::FunctionSpaceTools::computeEdgeMeasure<MeshScalarT>
          (weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      }
      else  //for 3D, get weighted face measure for the side
      {
        Intrepid2::FunctionSpaceTools::computeFaceMeasure<MeshScalarT>
          (weighted_measure, jacobianSide, cubWeightsSide, elem_side, *cellType);
      }

      // Values of the basis functions at side cubature points, in the reference parent cell domain
      intrepidBasis->getValues(basis_refPointsSide, refPointsSide, Intrepid2::OPERATOR_VALUE);

      // Transform values of the basis functions to physical frame
      Intrepid2::FunctionSpaceTools::HGRADtransformVALUE<MeshScalarT>
        (trans_basis_refPointsSide, basis_refPointsSide);

      // Multiply with weighted measure
      Intrepid2::FunctionSpaceTools::multiplyMeasure<MeshScalarT>
        (weighted_trans_basis_refPointsSide, weighted_measure, trans_basis_refPointsSide);
      
      // Map the side cubature points in reference frame to physical frame
      Intrepid2::CellTools<MeshScalarT>::mapToPhysicalFrame
        (physPointsSide, refPointsSide, physPointsCell, intrepidBasis);
      
      // Map cell (reference) degree of freedom points to the appropriate side (elem_side)
      for (std::size_t node=0; node < numNodes; ++node)
        dofCell(0, node) = dof(elem_LID, node);

      // This is needed, since evaluate currently sums into
      for (int k=0; k < numQPsSide ; k++) dofSide(0,k) = 0.0;

      // Get dof at cubature points of appropriate side (see DOFInterpolation evaluator)
      Intrepid2::FunctionSpaceTools::evaluate<ScalarT>(dofSide, dofCell, trans_basis_refPointsSide);
      
      // Transform the given BC data to the physical space QPs in each side (elem_side)
      calcInterfaceTrapChargDensity(data, dofSide, i); 
      
      // Put this side's contribution into the vector
      for (std::size_t node=0; node < numNodes; ++node) 
        for (std::size_t qp=0; qp < numQPsSide; ++qp) 
          for (std::size_t dim=0; dim < numDOFsSet; ++dim) 
            neumann(elem_LID, node, dim) += data(0, qp, dim) * weighted_trans_basis_refPointsSide(0, node, qp);
	  
    }  // end of loop over sides for a given sideset
  }  // end of loop over sidesets
}


template<typename EvalT, typename Traits>
void PoissonSourceInterfaceBase<EvalT, Traits>::
calcInterfaceTrapChargDensity(Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> & qp_data_returned,
			const Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device>& dof_side, int iSideset) 
{			
  int numCells = qp_data_returned.dimension(0);  // How many cell's worth of data is being computed?
  int numPoints = qp_data_returned.dimension(1); // How many QPs per cell?
  int numDOFs = qp_data_returned.dimension(2);   // How many DOFs per node to calculate?

 if(trapSpectrum[iSideset] == "Uniform")  // for Uniform energy spectrum
 { 
  for(int pt = 0; pt < numPoints; pt++) 
  {
    for(int dim = 0; dim < numDOFsSet; dim++) 
    {
      //ScalarT ten = 10.0;
      //qp_data_returned(0, pt, dim) = ten; //for DEBUG

      const ScalarT& us_phi = dof_side(0,pt); // unscaled phi in unit of [V]
      
      // compute conduction band (Ec), valence band (Ev), and intrinsic Fermi level (Ei) 
      ScalarT Ec = qPhiRef - elecAffinity[iSideset] - us_phi*1.0/energy_unit_in_eV;  // in [eV]
      ScalarT Ev = Ec - bandGap[iSideset]; 
      ScalarT Ei = (Ec + Ev) / 2.0; 
      ScalarT Ef = fermiEnergy[iSideset]; 
      
      double Dit = trapDensity[iSideset];  // in unit of [#/(eV.cm^2)]
      double aDegFac = acceptorDegFac[iSideset]; // degeneracy factor for acceptor traps
      double dDegFac = donorDegFac[iSideset];    // degeneracy factor for donor traps
      std::string type = trapType[iSideset];     // Trap Type = Acceptor, Donor, or Both
      
      ScalarT aUpperLimit = (Ef - Ec) / kbT;  // for Trap Type = Both
      ScalarT aLowerLimit = (Ef - Ei) / kbT; 
      ScalarT dUpperLimit = (Ei - Ef) / kbT; 
      ScalarT dLowerLimit = (Ev - Ef) / kbT; 
      
      if (type == "Acceptor")
        aLowerLimit = (Ef - Ev) / kbT; 
      else if (type == "Donor")
        dUpperLimit = (Ec - Ef) / kbT; 
      
      ScalarT kbT_eV = kbT * energy_unit_in_eV;   // make sure kbT in unit of [eV]
      ScalarT atmp1 = 0.0, atmp2 = 0.0, dtmp1 = 0.0, dtmp2 = 0.0;

      if ((type == "Acceptor") || (type == "Both"))
      { 
        if (aUpperLimit > MAX_EXPONENT)  
          atmp1 = aUpperLimit;  // log(degfac+exp(x)) = x when x > MAX_EXPONENT
        else
          atmp1 = std::log(aDegFac + std::exp(aUpperLimit)); 
        
        if (aLowerLimit > MAX_EXPONENT)
          atmp2 = aLowerLimit; 
        else
          atmp2 = std::log(aDegFac + std::exp(aLowerLimit));   
      }
      
      if ((type == "Donor") || (type == "Both"))
      {
        if (dUpperLimit > MAX_EXPONENT)
          dtmp1 = dUpperLimit; 
        else
          dtmp1 = std::log(dDegFac + std::exp(dUpperLimit));   

        if (dLowerLimit > MAX_EXPONENT)
          dtmp2 = dLowerLimit; 
        else
          dtmp2 = std::log(dDegFac + std::exp(dLowerLimit));   
      }
      
      // compute interface trap charge density in [#/cm^2]
      // when Dit = constant, the integration over energy for Nit can be carried out analytically
      ScalarT Nit = kbT_eV * Dit * (atmp1 - atmp2 + dtmp1 - dtmp2 );  
      
      // compute -q*X0/eps0*Nit, which is in unit of [V]
      qp_data_returned(0, pt, dim) = -eleQ * X0 / eps0 * Nit; 
      
      // std::cout << "pt=" << pt << ", dofset=" << dim << ", Nit=" << Nit << ", data=" << qp_data_returned(0,pt,dim) << std::endl; 
    }
  }
 }  // end of  if(trapSpectrum == "Uniform")
 
 
}

template<typename EvalT, typename Traits>
typename QCAD::PoissonSourceInterfaceBase<EvalT,Traits>::ScalarT 
PoissonSourceInterfaceBase<EvalT, Traits>::getReferencePotential()
{
  //! Constant energy reference for heterogeneous structures
  ScalarT qPhiRef;

  std::string refMtrlName, category;
  refMtrlName = materialDB->getParam<std::string>("Reference Material");
  category = materialDB->getMaterialParam<std::string>(refMtrlName,"Category");
  if (category == "Semiconductor") 
  {
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
  else if (category == "Insulator") 
  {
    double Chi = materialDB->getMaterialParam<double>(refMtrlName,"Electron Affinity");
    qPhiRef = Chi / energy_unit_in_eV; // in [myV]
  }
  else if (category == "Metal") 
  {
    double workFn = materialDB->getMaterialParam<double>(refMtrlName,"Work Function"); 
    qPhiRef = workFn / energy_unit_in_eV; // in [myV]
  }
  else 
  {
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
PoissonSourceInterface<PHAL::AlbanyTraits::Residual,Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::Residual,Traits>(p)
{
}



template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill in "neumann" array
  this->evaluateInterfaceContribution(workset);

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();
  ScalarT *valptr;

  // Place it at the appropriate offset into F
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) 
  {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node)
    {
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim)
      {
        valptr = &(this->neumann)(cell, node, dim);
        fT_nonconstView[nodeID[node][this->offset[dim]]] += *valptr;
      }
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
PoissonSourceInterface<PHAL::AlbanyTraits::Jacobian, Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::Jacobian,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill in "neumann" array
  this->evaluateInterfaceContribution(workset);

  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::ArrayRCP<ST> fT_nonconstView; 
  if (Teuchos::nonnull(fT)) 
    fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;

  ScalarT *valptr;

  int lcol;
  Teuchos::Array<LO> rowT(1);
  Teuchos::Array<LO> colT(1);
  Teuchos::Array<ST> value(1);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) 
  {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node)
    {
      for (std::size_t dim = 0; dim < this->numDOFsSet; ++dim)
      {
        valptr = &(this->neumann)(cell, node, dim);

        rowT[0] = nodeID[node][this->offset[dim]];

        int neq = nodeID[node].size();

        if (fT != Teuchos::null) 
        {
          fT->sumIntoLocalValue(rowT[0], valptr->val());
        }

        // Check derivative array is nonzero
        if (valptr->hasFastAccess()) 
        {
          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++)
          {
            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) 
            {
              lcol = neq * node_col + eq_col;

              // Global column
              colT[0] =  nodeID[node_col][eq_col];
              value[0] = valptr->fastAccessDx(lcol);   
              if (workset.is_adjoint) 
              {
                // Sum Jacobian transposed
                JacT->sumIntoLocalValues(colT[0], rowT(), value());
              }
              else 
              {
                // Sum Jacobian
                JacT->sumIntoLocalValues(rowT[0], colT(), value());
              }
            } // column equations
          } // column nodes
        } // has fast access
      }  // loop over numDOFsSet  
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
PoissonSourceInterface<PHAL::AlbanyTraits::Tangent, Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::Tangent,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateInterfaceContribution(workset);

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
PoissonSourceInterface<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateInterfaceContribution(workset);

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
PoissonSourceInterface<PHAL::AlbanyTraits::SGResidual, Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::SGResidual,Traits>(p)
{
}


// **********************************************************************
template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateInterfaceContribution(workset);

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
PoissonSourceInterface<PHAL::AlbanyTraits::SGJacobian, Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::SGJacobian,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateInterfaceContribution(workset);

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
PoissonSourceInterface<PHAL::AlbanyTraits::SGTangent, Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::SGTangent,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateInterfaceContribution(workset);

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
PoissonSourceInterface<PHAL::AlbanyTraits::MPResidual, Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::MPResidual,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateInterfaceContribution(workset);

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
PoissonSourceInterface<PHAL::AlbanyTraits::MPJacobian, Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateInterfaceContribution(workset);

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
PoissonSourceInterface<PHAL::AlbanyTraits::MPTangent, Traits>::
PoissonSourceInterface(Teuchos::ParameterList& p)
  : PoissonSourceInterfaceBase<PHAL::AlbanyTraits::MPTangent,Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void PoissonSourceInterface<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Fill the local "neumann" array with cell contributions

  this->evaluateInterfaceContribution(workset);

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
