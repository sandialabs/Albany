
#include <iostream>
#include <fstream>
#include "felix_driver.H"

using namespace std; 
using std::string;

static int maxStep;
static double maxTime;


extern "C" void felix_driver_();

int
felix_store(int obj_index, Felix ** felix_object, int mode)
{
  static Felix * felix_store_ptr_arr[DYCORE_MODEL_COUNT];

  switch (mode) {
  case 0: felix_store_ptr_arr[obj_index] = *felix_object;
    cout << "In felix_store, mode = 0 -- Storing Felix Object # " 
	 << obj_index << ", Address = " << *felix_object << endl;
    break;
  case 1: *felix_object = felix_store_ptr_arr[obj_index];
    cout << "In felix_store, mode = 1 -- Retrieving Felix Object # " 
	 << obj_index << ", Address = " << *felix_object << endl;
    break;
  default: ;
  }
  return 0;
}

void felix_driver_run(int argc, int exec_mode);
 
/// types of basal friction (beta) distributions
/** SinusoidalBeta is the one for exp C in Pattyn et al (2008)
    guassianBump is used for the MISMIP3D perturbations tests.
 */
enum basalFrictionTypes {constantBeta = 0,
                         sinusoidalBeta,
                         sinusoidalBetay,
                         twistyStreamx,
			 gaussianBump,
                         NUM_BETA_TYPES};

void felix_driver_init(int argc, int exec_mode,FelixToGlimmer * btg_ptr, const char * input_fname)
{ 

  char *argv[3];
  //char argv1[] = "/home/ranken/util/BISICLES/code/interface/inputs32.glimmer";
  // this one assumed we were running in gc1/parallel/src/fortran
  //char argv1[] = "../../../..//BISICLES/code/interface/inputs.glimmer";
  // this one assumes we're running in gc1/parallel/bin
  char argv1[] = "../../..//BISICLES/code/interface/inputs.glimmer";
  //char argv1[] = "inputs.glimmer";
  char argv0[] = "felix_driver";

  //  argv[0] = 
  argv[0] = argv0;
  argv[1] = argv1;

  cout << "In felix_driver..." << endl;
  cout << "Printing this from Albany...  This worked!  Yay!  Nov. 4 2013" << endl; 


  { // Begin nested scope
    
  cout << "Beginning nested scope..." << endl;
     

//ifdef CH_MPI
//    MPI_Barrier(Chombo_MPI::comm);
//#endif
    int rank, number_procs;
//#ifdef CH_MPI
//    MPI_Comm_rank(Chombo_MPI::comm, &rank);
//    MPI_Comm_size(Chombo_MPI::comm, &number_procs);
//#else
//    rank=0;
//    number_procs=1;
//#endif


    long cism_communicator, cism_process_count, my_cism_rank;

    cism_communicator = *(btg_ptr -> getLongVar("communicator","mpi_vars"));
    cism_process_count = *(btg_ptr -> getLongVar("process_count","mpi_vars"));
    my_cism_rank = *(btg_ptr -> getLongVar("my_rank","mpi_vars"));
    cout << "In felix_driver, CISM comm, count, my_rank = " << cism_communicator << "  "
         << cism_process_count << "  " << my_cism_rank << endl;

    bool verbose = true;
    if (verbose)
      {
	cout << "rank " << rank << " of " << number_procs << endl;
      }


    Felix* felixPtr = new Felix();

    //bikePtr->amrIce = new AmrIce;
    //AmrIce* amrObjectPtr = bikePtr->amrIce;
    //if(argc < 2) 
    //  { std::cerr << " usage: " << argv[0] << " <input_file>\n"; exit(0); }
    //    char* in_file = argv[1];
    //const char* in_file = input_fname;
    //cout << "Parsing: " << in_file << endl;
    //ParmParse pp(argc-2,argv+2,NULL,in_file);
    //bikePtr->parmParse = new ParmParse(argc-2,argv+2,NULL,in_file);

    /*if (verbose)
      {
	cout << "...done reading file..." << endl;
      }
    RealVect domainSize;

    ParmParse pp2("main");

    ParmParse interfacePP("glimmerInterface");

    if (verbose)
      {
	cout << "... done" << endl;
	
	cout << "setting geometry IBC..." << endl;
      }

    ParmParse geomPP("geometry");
    */
    double dew, dns;
    long * dimInfo;        
    int * dimInfoVelo;
    //IntVect ghostVect = IntVect::Zero;
    bool nodalGeom;

    // ---------------------------------------------
    // set IBC -- this includes initial ice thickness, 
    // and basal geometry
    // ---------------------------------------------
   
    
    //IceThicknessIBC* thicknessIBC = NULL;
    //std::string problem_type; 
    
    cout << "Getting geometry info from CISM..." << endl; 
    // geometry info from CISM
    int i, reg_index;      
    dimInfo = btg_ptr -> getLongVar("dimInfo","geometry");
    
    //dew = 1000.;
    //dns = 1000.;
    
    dew = *(btg_ptr -> getDoubleVar("dew","numerics"));
    dns = *(btg_ptr -> getDoubleVar("dns","numerics"));
    cout << "In felix_driver: dew, dns = " << dew << "  " << dns << endl;
    
    int * dimInfoGeom = new int[dimInfo[0]+1];    
    
    for (i=0;i<=dimInfo[0];i++) dimInfoGeom[i] = dimInfo[i];   
    cout << "DimInfoGeom  in felix_driver: " << endl;
    for (i=0;i<=dimInfoGeom[0];i++) cout << dimInfoGeom[i] << " ";
    cout << endl;
    
    long ewlb, ewub, nslb, nsub;
    
    ewlb = *(btg_ptr -> getLongVar("ewlb","geometry"));
    ewub = *(btg_ptr -> getLongVar("ewub","geometry"));
    nslb = *(btg_ptr -> getLongVar("nslb","geometry"));
    nsub = *(btg_ptr -> getLongVar("nsub","geometry"));
    cout << "In felix_driver: ewlb, ewub = " << ewlb << "  " << ewub <<  endl;
    cout << "In felix_driver: nslb, nsub = " << nslb << "  " << nsub <<  endl;

    //IK, 11/13/13: check that connectivity derived types are transfered over from CISM to Albany/FELIX    
    long nCellsActive; 
    nCellsActive = *(btg_ptr -> getLongVar("nCellsActive","connectivity")); 
    cout << "In felix_driver: nCellsActive = " << nCellsActive <<  endl;

 
    //int lb[SpaceDim];
    //int ub[SpaceDim];
    
    /*D_TERM(lb[0] = ewlb;
           ub[0] = ewub;,
           lb[1] = nslb;
           ub[1] = nsub;,
           lb[2] = 0;
           ub[2] = numCells[2]-1;)
      */
      
      
  /*    // bit of a hack, since we need to have periodicity info here,
      // which is normally taken care of inside AmrIce
      ParmParse ppAmr("amr");
    // default is that domains are not periodic
    bool is_periodic[SpaceDim];
    for (int dir=0; dir<SpaceDim; dir++)
      is_periodic[dir] = false;
    Vector<int> is_periodic_int(SpaceDim, 0);
    
    ppAmr.getarr("is_periodic", is_periodic_int, 0, SpaceDim);
    for (int dir=0; dir<SpaceDim; dir++) 
      {
        is_periodic[dir] = (is_periodic_int[dir] == 1);
      }

        nodalGeom = true;
        interfacePP.query("nodalInitialData", nodalGeom);
        
	if (verbose)
	  {
	    cout << "nodal initial data = " << nodalGeom << endl;
	  }
    */
    
    // define domain using dim_info
    int ewn = dimInfoGeom[2];
    int nsn = dimInfoGeom[3];

    /*// convert to 0->n-1 ordering to suit Chombo's preferences
    IntVect domLo = IntVect::Zero;
    IntVect domHi = IntVect(D_DECL(ewn-1, nsn-1, dimInfoGeom[1]));

    // convert from node->cells if necessary (domain is always 
    // cell-centered 
    if (nodalGeom)
      {
        domHi -= IntVect::Unit;
      }

    Box domainBox(domLo, domHi);
    ProblemDomain baseDomain(domainBox);
    for (int dir=0; dir<SpaceDim; dir++)
      {
        baseDomain.setPeriodic(dir, is_periodic[dir]);
      }
    
    if (verbose)
      {
        pout() << "Base Domain = " << baseDomain << endl;
      }

    // this is to convert Fortran indexing to C indexing.
  
    IntVect offset = IntVect::Unit;

    geomPP.get("problem_type", problem_type);
    if (problem_type =="fortran")
      {
        FortranInterfaceIBC* ibcPtr = new FortranInterfaceIBC;
        // need to set thickness and topography

        // default is that CISM is using 2 ghost cells 
        Vector<int> nGhost(SpaceDim, 2);
        interfacePP.queryarr("numGhost", nGhost, 0, SpaceDim);
        {
          ghostVect = IntVect(D_DECL(nGhost[0], nGhost[1], nGhost[2]));
        }

        nodalGeom = true;
        interfacePP.query("nodalInitialData", nodalGeom);
        
	if (verbose)
	  {
	    cout << "nodal initial data = " << nodalGeom << endl;
	  }
	
        // this is about removing ice from regions which
        // don't affect the dynamics of the region, but which 
        // can cause our solvers problems. Siple Island comes to mind here
        // for now, store these regions in a separate file. There's probably 
        // a better way to do this.
        
        // this will contain the boxes in the index space of the 
        // original data in which the thickness will be cleared.
        Vector<Box> clearBoxes;

        bool clearThicknessRegions = false;
        if (interfacePP.contains("clearThicknessRegionsFile"))
          {
            clearThicknessRegions = true;
            std::string clearFile;
            interfacePP.get("clearThicknessRegionsFile", clearFile);

            if (procID() == uniqueProc(SerialTask::compute))
              {
                ifstream is(clearFile.c_str(), ios::in);
                if (is.fail())
                  {
                    MayDay::Error("Cannot open file with regions for thickness clearing");
                  }
                // format of file: number of boxes, then list of boxes.
                int numRegions;
                is >> numRegions;
                // advance pointer in file
                while (is.get() != '\n');

                clearBoxes.resize(numRegions);

                for (int i=0; i<numRegions; i++)
                  {
                    Box bx;
                    is >> bx;
                    while (is.get() != '\n');

                    clearBoxes[i] = bx;
                  }
                    
              } // end if serial proc
            // broadcast results
            broadcast(clearBoxes, uniqueProc(SerialTask::compute));
            
            ibcPtr->setThicknessClearRegions(clearBoxes);
          }
        
	if (verbose)
	  {
	    cout << "...done" << endl;
	  }
*/
        // constants
        double* seconds_per_year_ptr;
        double* gravity_ptr;
        double* rho_ice_ptr;
        double* rho_seawater_ptr;

        seconds_per_year_ptr = btg_ptr -> getDoubleVar("seconds_per_year","constants");
        gravity_ptr = btg_ptr -> getDoubleVar("gravity","constants");
        rho_ice_ptr = btg_ptr -> getDoubleVar("rho_ice","constants");
        rho_seawater_ptr = btg_ptr -> getDoubleVar("rho_seawater","constants");

        double* thicknessDataPtr, *topographyDataPtr;
        double* upperSurfaceDataPtr;
        double* lowerSurfaceDataPtr;
        double* floating_maskDataPtr;
        double* ice_maskDataPtr;
        double* lower_cell_locDataPtr;

        thicknessDataPtr = btg_ptr -> getDoubleVar("thck","geometry");
        topographyDataPtr = btg_ptr -> getDoubleVar("topg","geometry");
        upperSurfaceDataPtr = btg_ptr -> getDoubleVar("usrf","geometry");
        lowerSurfaceDataPtr = btg_ptr -> getDoubleVar("lsrf","geometry");
        floating_maskDataPtr = btg_ptr -> getDoubleVar("floating_mask","geometry");
        ice_maskDataPtr = btg_ptr -> getDoubleVar("ice_mask","geometry");
        lower_cell_locDataPtr = btg_ptr -> getDoubleVar("lower_cell_loc","geometry");

        //IK, 11/13/13: 
        //connectivity arrays
        cout << "In felix_driver: grabbing connectivity array pointers from CISM..." << endl; 
        double* xyz_at_nodes_Ptr, *surf_height_at_nodes_Ptr, *beta_at_nodes_Ptr;
        xyz_at_nodes_Ptr = btg_ptr -> getDoubleVar("xyz_at_nodes","connectivity"); 
        surf_height_at_nodes_Ptr = btg_ptr -> getDoubleVar("surf_height_at_nodes","connectivity"); 
        beta_at_nodes_Ptr = btg_ptr -> getDoubleVar("beta_at_nodes","connectivity");
        double *flwa_at_active_elements_Ptr; 
        flwa_at_active_elements_Ptr = btg_ptr -> getDoubleVar("flwa_at_active_elements","connectivity"); 
        long int* global_node_id_owned_map_Ptr; 
        global_node_id_owned_map_Ptr = btg_ptr -> getLongVar("global_node_id_owned_map","connectivity");  
        long int* global_element_conn_active_Ptr; 
        global_element_conn_active_Ptr = btg_ptr -> getLongVar("global_element_conn_active","connectivity");  
        long int* global_element_id_active_owned_map_Ptr; 
        global_element_id_active_owned_map_Ptr = btg_ptr -> getLongVar("global_element_id_active_owned_map","connectivity");  
        long int* global_basal_face_conn_active_Ptr; 
        global_basal_face_conn_active_Ptr = btg_ptr -> getLongVar("global_basal_face_conn_active","connectivity");  
        long int* global_basal_face_id_active_owned_map_Ptr; 
        global_basal_face_id_active_owned_map_Ptr = btg_ptr -> getLongVar("global_basal_face_id_active_owned_map","connectivity");  
        global_basal_face_id_active_owned_map_Ptr = btg_ptr -> getLongVar("global_basal_face_id_active_owned_map","connectivity");  
        cout << "...done!" << endl; 

  
	// this is mainly to get the ProblemDomain info into the mix
/*	ibcPtr->define(baseDomain, dew);

        CH_assert(SpaceDim == 2);
#if 1
        if (nodalGeom)
          {
            // slc : thck and topg are defined at cell nodes on the glimmer grid,
            //       while bisicles needs them at cell centers. To get round this,
            //       decrement the grid dimensions by 1, then interpolate to the
            //       cell centers. This means  extra work, but is required
            //       if we think the domain boundaries should be  in the same place.
            dimInfoGeom[2] -= 1; dimInfoGeom[3] -=1;
          }
        domainSize[0] = dew*(dimInfoGeom[2]); 
        domainSize[1] = dns*(dimInfoGeom[3]);     
                 
        ibcPtr->setThickness(thicknessDataPtr, dimInfoGeom, lb,ub,
                             &dew, &dns, 
                             offset, ghostVect, nodalGeom);                             
        ibcPtr->setTopography(topographyDataPtr, dimInfoGeom, lb, ub, 
                              &dew, &dns, 
                              offset, ghostVect, nodalGeom);

        ibcPtr->setSurface(upperSurfaceDataPtr, dimInfoGeom, lb, ub, 
                           &dew, &dns, 
                           offset, ghostVect, nodalGeom);


        ibcPtr->setLowerSurface(lowerSurfaceDataPtr, dimInfoGeom, lb, ub, 
                                &dew, &dns, 
                                offset, ghostVect, nodalGeom);

        ibcPtr->setFloatingMask(floating_maskDataPtr, dimInfoGeom, lb, ub, 
                                &dew, &dns, 
                                offset, ghostVect, nodalGeom);

        ibcPtr->setIceMask(ice_maskDataPtr, dimInfoGeom, lb, ub, 
                           &dew, &dns, 
                           offset, ghostVect, nodalGeom);
        

        ibcPtr->setLowerCellCenterZ(lower_cell_locDataPtr, dimInfoGeom, lb, ub, 
                                    &dew, &dns, 
                                    offset, ghostVect, nodalGeom);
*/
#if 0
  //  reg_index = dycore_registry(0,1,&model_index,&dtg_ptr,-1);
  amrObjectPtr->getIceThickness(thicknessPtr, dim_info,
			    dew, dns);
#endif
/*#else
	domainSize[0] = dew*(dimInfoGeom[2]); 
	domainSize[1] = dns*(dimInfoGeom[3]);	
        ibcPtr->setThickness(thicknessDataPtr, dimInfoGeom, lb,ub,
                             &dew, &dns, offset, ghostVect);
        ibcPtr->setTopography(topographyDataPtr, dimInfoGeom, lb, ub,
                              &dew, &dns, offset, ghostVect);
#endif
        // if desired, smooth out topography to fill in holes
        bool fillTopographyHoles = false;
        geomPP.query("fill_topography_holes", fillTopographyHoles);
        if (fillTopographyHoles)
          {
            Real holeVal = 0.0;
            geomPP.query("holeFillValue", holeVal);
            int numPasses = 1;
            geomPP.query("num_fill_passes", numPasses);

            for (int pass=0; pass<numPasses; pass++)
              {
                ibcPtr->fillTopographyHoles(holeVal);
              }
          }

        thicknessIBC = static_cast<IceThicknessIBC*>(ibcPtr);
*/
	dimInfo = btg_ptr -> getLongVar("dimInfo","velocity");
        
        dimInfoVelo = new int[dimInfo[0]+1];
    
        for (i=0;i<=dimInfo[0];i++) dimInfoVelo[i] = dimInfo[i];      
        
        cout << "DimInfoVelo in felix_driver: " << endl;
        for (i=0;i<=dimInfoVelo[0];i++) cout << dimInfoVelo[i] << " ";
        cout << endl; 

        // get Glimmer surface mass balance
        double * surfMassBalPtr;

        dimInfo = btg_ptr -> getLongVar("dimInfo","climate"); 

        int * dimInfoClim = new int[dimInfo[0]+1];

        for (i=0;i<=dimInfo[0];i++) dimInfoClim[i] = dimInfo[i];      
        surfMassBalPtr = btg_ptr -> getDoubleVar("acab","climate");
        cout << "DimInfoClim in felix_driver: " << endl;
        for (i=0;i<=dimInfoClim[0];i++) cout << dimInfoClim[i] << " ";
        cout << endl;      
        
     // }
    //else 
    //  {
        //MayDay::Error("bad problem type");
    //  }




    // ---------------------------------------------
    // set constitutive relation & rate factor
    // ---------------------------------------------
    if (verbose)
      {
	cout << "initializing constRel" << endl;
      }

   /* std::string constRelType;
    pp2.get("constitutiveRelation", constRelType);
    ConstitutiveRelation* constRelPtr = NULL;
    GlensFlowRelation* gfrPtr = NULL;
    if (constRelType == "constMu")
      {
        constMuRelation* newPtr = new constMuRelation;
        ParmParse crPP("constMu");
        Real muVal;
        crPP.get("mu", muVal);
        newPtr->setConstVal(muVal);
        constRelPtr = static_cast<ConstitutiveRelation*>(newPtr);
      }
    else if (constRelType == "GlensLaw")
      {
        constRelPtr = new GlensFlowRelation;
	gfrPtr = dynamic_cast<GlensFlowRelation*>(constRelPtr);
      }
    else if (constRelType == "L1L2")
      {
        L1L2ConstitutiveRelation* l1l2Ptr = new L1L2ConstitutiveRelation;
        ParmParse ppL1L2("l1l2");

        // set L1L2 internal solver tolerance
        // default matches original hardwired value
        Real tol = 1.0e-6;
        ppL1L2.query("solverTolerance", tol);
        l1l2Ptr->solverTolerance(tol);
	gfrPtr = l1l2Ptr->getGlensFlowRelationPtr();
        constRelPtr = l1l2Ptr;

      }
    else 
      {
        MayDay::Error("bad Constitutive relation type");
      }

    Real epsSqr0 = 1.0e-9;
    std::string rateFactorType = "constRate";
    pp2.query("rateFactor", rateFactorType);
    if (rateFactorType == "constRate")
      {
	ParmParse crPP("constRate");
	Real A = 9.2e-18;
	crPP.query("A", A);
	ConstantRateFactor rateFactor(A);

	crPP.query("epsSqr0", epsSqr0);
	amrObjectPtr->setRateFactor(&rateFactor);
	if (gfrPtr) 
	  {
	    gfrPtr->setParameters(3.0 , &rateFactor, epsSqr0);
	  }
      }
    else if (rateFactorType == "arrheniusRate")
      {
	ArrheniusRateFactor rateFactor;
	ParmParse arPP("ArrheniusRate");
	arPP.query("epsSqr0", epsSqr0);
	amrObjectPtr->setRateFactor(&rateFactor);
	if (gfrPtr) 
	  {
	    gfrPtr->setParameters(3.0 , &rateFactor, epsSqr0);
	  }
      }


    amrObjectPtr->setConstitutiveRelation(constRelPtr);  
    */
    if (verbose)
      {
	cout << "... done" << endl;
	
	cout << "setting surface flux... " << endl;
      }

    // ---------------------------------------------
    // set (upper) surface flux. 
    // ---------------------------------------------

    /*SurfaceFlux* surf_flux_ptr = SurfaceFlux::parseSurfaceFlux("surfaceFlux");


    if (surf_flux_ptr == NULL)
      {
	// chunk for compatiblity with older input files
	MayDay::Warning("trying to parse old style surface_flux_type");
	surf_flux_ptr = NULL;
	std::string surfaceFluxType = "zeroFlux";
	pp2.query("surface_flux_type", surfaceFluxType);
	
	if (surfaceFluxType == "zeroFlux")
	  {
            surf_flux_ptr = new zeroFlux;
	  }
	else if (surfaceFluxType == "constantFlux")
	  {
	    constantFlux* constFluxPtr = new constantFlux;
	    Real fluxVal;
	    ParmParse ppFlux("constFlux");
	    ppFlux.get("flux_value", fluxVal);
	    constFluxPtr->setFluxVal(fluxVal);
	    
	    surf_flux_ptr = static_cast<SurfaceFlux*>(constFluxPtr);
	  }
      }
    else
      {
        // we allocated a SurfaceFlux, but we need to test to see if
        // it was a  fortranInterfaceFlux, in which case we didn't have 
        // the context to actually set things up. Do that here...
        ParmParse surfaceFluxPP("surfaceFlux");
        std::string type = "";
        
        surfaceFluxPP.query("type",type);

        if (type == "fortran")
          {
            // set things up here
            fortranInterfaceFlux* fluxPtr = dynamic_cast<fortranInterfaceFlux*>(surf_flux_ptr);
            int nghostFlux = 2;
            IntVect ghostVect = nghostFlux*IntVect::Unit;
            // flux has same centering as thickness and topography
            bool nodalFlux = nodalGeom;
            interfacePP.query("nodalSurfaceFlux", nodalFlux);
            
            if (verbose)
              {
                cout << "nodal surfaceFlux = " << nodalFlux << endl;
              }

            long int* dimInfoFluxPtr = (btg_ptr -> getLongVar("dimInfo","climate"));
    
            double* fluxDataPtr = btg_ptr -> getDoubleVar("acab","climate");

            int* dimInfoFlux = new int[dimInfoFluxPtr[0]+1];
            if (verbose)
              {
                cout << "DimInfoFlux in bike_driver: ";
              }

            for (i=0; i<= dimInfoFluxPtr[0]; i++)
              {
                dimInfoFlux[i] = dimInfoFluxPtr[i];
                if (verbose) cout << dimInfoFlux[i] << "  ";
              }
            if (verbose) cout << endl;

            
            fluxPtr->setFluxVal(fluxDataPtr, dimInfoFlux, lb,ub,
                                &dew, &dns, offset, ghostVect, baseDomain,
                                nodalGeom);
            
          }
        
      }
        
    if (surf_flux_ptr == NULL)
      {
	MayDay::Error("invalid surface flux type");
      }

    amrObjectPtr->setSurfaceFlux(surf_flux_ptr);
    */
    if (verbose)
      {
	cout << "... done" << endl;
	
	cout << "setting basal flux... " << endl;
      }

    // ---------------------------------------------
    // set basal (lower surface) flux. 
    // ---------------------------------------------
    
  /*  SurfaceFlux* basal_flux_ptr = SurfaceFlux::parseSurfaceFlux("basalFlux");
    
    if (basal_flux_ptr == NULL)
      {
	//chunk for compatiblity with older input files
	MayDay::Warning("trying to parse old style basal_flux_type");
	std::string basalFluxType = "zeroFlux";
	pp2.query("basal_flux_type", basalFluxType);

	if (basalFluxType == "zeroFlux")
	  {
	    basal_flux_ptr = new zeroFlux;
	  }
	else if (basalFluxType == "constantFlux")
	  {
	    constantFlux* constFluxPtr = new constantFlux;
	    Real fluxVal;
	    ParmParse ppFlux("basalConstFlux");
	    ppFlux.get("flux_value", fluxVal);
	    constFluxPtr->setFluxVal(fluxVal);
	    basal_flux_ptr = static_cast<SurfaceFlux*>(constFluxPtr);
	    
	  }
	else if (basalFluxType == "maskedFlux")
	  {
	    SurfaceFlux* grounded_basal_flux_ptr = NULL;
	    std::string groundedBasalFluxType = "zeroFlux";
	    ParmParse ppbmFlux("basalMaskedFlux");
	    ppbmFlux.query("grounded_flux_type",groundedBasalFluxType);
	    if (groundedBasalFluxType == "zeroFlux")
	      {
		grounded_basal_flux_ptr = new zeroFlux;
	      }
	    else if (groundedBasalFluxType == "constantFlux")
	      {
		constantFlux* constFluxPtr = new constantFlux;
		Real fluxVal;
		ParmParse ppgFlux("groundedBasalConstFlux");
		ppgFlux.get("flux_value", fluxVal);
		constFluxPtr->setFluxVal(fluxVal);
		grounded_basal_flux_ptr = static_cast<SurfaceFlux*>(constFluxPtr);
	      }
	    else
	      {
		MayDay::Error("invalid grounded basal flux type");
	      }
	    
	    SurfaceFlux* floating_basal_flux_ptr = NULL;
	    std::string floatingBasalFluxType = "zeroFlux";
	    
	    ppbmFlux.query("floating_flux_type",floatingBasalFluxType);
	    if (floatingBasalFluxType == "zeroFlux")
	      {
		floating_basal_flux_ptr = new zeroFlux;
	      }
	    else if (floatingBasalFluxType == "constantFlux")
	      {
		constantFlux* constFluxPtr = new constantFlux;
		Real fluxVal;
		ParmParse ppgFlux("groundedBasalConstFlux");
		ppgFlux.get("flux_value", fluxVal);
		constFluxPtr->setFluxVal(fluxVal);
		grounded_basal_flux_ptr = static_cast<SurfaceFlux*>(constFluxPtr);
	      }
	    else
	      {
		MayDay::Error("invalid grounded basal flux type");
	      }
	    
	    SurfaceFlux* floating_basal_flux_ptr = NULL;
	    std::string floatingBasalFluxType = "zeroFlux";
	    
	    ppbmFlux.query("floating_flux_type",floatingBasalFluxType);
	    if (floatingBasalFluxType == "zeroFlux")
	      {
		floating_basal_flux_ptr = new zeroFlux;
	      }
	    else if (floatingBasalFluxType == "constantFlux")
	      {
		constantFlux* constFluxPtr = new constantFlux;
		Real fluxVal;
		ParmParse ppfFlux("floatingBasalConstFlux");
		ppfFlux.get("flux_value", fluxVal);
		constFluxPtr->setFluxVal(fluxVal);
		floating_basal_flux_ptr = static_cast<SurfaceFlux*>(constFluxPtr);
	      }
	    else if (floatingBasalFluxType == "piecewiseLinearFlux")
	      {
		ParmParse pwlFlux("floatingBasalPWLFlux");
		int n = 1;  
		pwlFlux.query("n",n);
		Vector<Real> vabs(n,0.0);
		Vector<Real> vord(n,0.0);
		pwlFlux.getarr("abscissae",vabs,0,n);
		pwlFlux.getarr("ordinates",vord,0,n);
		PiecewiseLinearFlux* ptr = new PiecewiseLinearFlux(vabs,vord);
		floating_basal_flux_ptr = static_cast<SurfaceFlux*>(ptr);
	      }
	    else
	      {
		MayDay::Error("invalid floating basal flux type");
	      }

	    SurfaceFlux* openland_basal_flux_ptr = new zeroFlux;
	    SurfaceFlux* opensea_basal_flux_ptr = new zeroFlux;
	    
	    basal_flux_ptr = static_cast<SurfaceFlux*>
	      (new MaskedFlux(grounded_basal_flux_ptr->new_surfaceFlux(), 
			      floating_basal_flux_ptr->new_surfaceFlux(),
			      opensea_basal_flux_ptr->new_surfaceFlux(),
			      openland_basal_flux_ptr->new_surfaceFlux()));
	    
	    delete grounded_basal_flux_ptr;
	    delete floating_basal_flux_ptr;
	    delete opensea_basal_flux_ptr;
	    delete openland_basal_flux_ptr;
       	
	  }
      }
    
    if (basal_flux_ptr == NULL)
      {
	MayDay::Error("invalid basal flux type");
      }
    
    amrObjectPtr->setBasalFlux(basal_flux_ptr); 
    */
    if (verbose)
      {
	cout << "... done" << endl;
	cout << "setting mu..." << endl;
      }

    // ---------------------------------------------
    // set mu coefficient
    // ---------------------------------------------
    /*ParmParse muPP("muCoefficient");
    std::string muCoefType = "unit";
    muPP.query("type",muCoefType );
    if (muCoefType == "unit")
      {
	MuCoefficient* ptr = static_cast<MuCoefficient*>(new UnitMuCoefficient());
	amrObjectPtr->setMuCoefficient(ptr);
	delete ptr;
      }
    else if (muCoefType == "LevelData")
      {
	//read a one level muCoef from an AMR Hierarchy, and  store it in a LevelDataMuCoeffcient
	 ParmParse ildPP("inputLevelData");
	 std::string infile;
	 ildPP.get("muCoefFile",infile);
	 std::string frictionName = "muCoef";
	 ildPP.query("muCoefName",frictionName);
	 RefCountedPtr<LevelData<FArrayBox> > levelMuCoef (new LevelData<FArrayBox>());
	 Vector<RefCountedPtr<LevelData<FArrayBox> > > vectMuCoef;
	 vectMuCoef.push_back(levelMuCoef);
	 Vector<std::string> names(1);
	 names[0] = frictionName;
	 Real dx;
	 readLevelData(vectMuCoef,dx,infile,names,1);
	 RealVect levelDx = RealVect::Unit * dx;
	 MuCoefficient* ptr = static_cast<MuCoefficient*>
	   (new LevelDataMuCoefficient(levelMuCoef,levelDx));
	 amrObjectPtr->setMuCoefficient(ptr);
	 delete ptr;
      }
    else
      {
	MayDay::Error("undefined MuCoefficient in inputs");
      }



    // this lets us over-ride the Glimmer domain size
    // for the case where we're not using the entire glimmer domain
    // (normally due to the fact that they tend to choose really 
    //  odd (and un-coarsenable) domain sizes, so we often want 
    // to throw away a row or two of cells in each direction in 
    // order to make Multigrid have a chance of converging
    
    if (pp2.contains("domain_size"))
      {
        Vector<Real> domSize(SpaceDim);
        pp2.getarr("domain_size", domSize, 0, SpaceDim);
        domainSize = RealVect(D_DECL(domSize[0], domSize[1], domSize[2]));
      }

    amrObjectPtr -> setDomainSize(domainSize);
            
    // amrObjectPtr->setDomainSize(domainSize);
    amrObjectPtr -> setThicknessBC(thicknessIBC);


    // ---------------------------------------------
    // set basal friction coefficient and relation
    // ---------------------------------------------
*/
    if (verbose)
      {
	cout << "setting basal friction..." << endl;
      }
  /*  
    BasalFriction* basalFrictionPtr = NULL;

    std::string beta_type;
    geomPP.get("beta_type", beta_type);
    // read in type of beta^2 distribution
    
    if (beta_type == "constantBeta")
      {
        Real betaVal;
        geomPP.get("betaValue", betaVal);
        basalFrictionPtr = static_cast<BasalFriction*>(new constantFriction(betaVal));
      }
    else if (beta_type == "sinusoidalBeta")
      {
        Real betaVal, eps;
        RealVect omega(RealVect::Unit);
        Vector<Real> omegaVect(SpaceDim);
        geomPP.get("betaValue", betaVal);
        if (geomPP.contains("omega"))
          {
            geomPP.getarr("omega", omegaVect, 0, SpaceDim);
            omega = RealVect(D_DECL(omegaVect[0], omegaVect[1], omegaVect[2]));
          }
        geomPP.get("betaEps", eps);
        // fix this later, if we need to...
        MayDay::Error("trying to define basal Friction with undefined DomainSize");
        basalFrictionPtr = static_cast<BasalFriction*>(new sinusoidalFriction(betaVal, 
                                                                              omega, 
                                                                              eps,
                                                                              domainSize));
      }
    // keep this one around for backward compatibility, even if it
    // is a special case of sinusoidalBeta
    else if (beta_type == "sinusoidalBetay")
      {
        Real betaVal, eps, omegaVal;
        RealVect omega(RealVect::Zero);
        omega[1] = 1;
        
        geomPP.get("betaValue", betaVal);
        if (geomPP.contains("omega"))
          {
            geomPP.get("omega", omegaVal);
            omega[1] = omegaVal;
          }
        geomPP.get("betaEps", eps);
        // fix this later, if we need to...
        MayDay::Error("trying to define basal Friction with undefined DomainSize");
        basalFrictionPtr = static_cast<BasalFriction*>(new sinusoidalFriction(betaVal, 
                                                                              omega, 
                                                                              eps,
                                                                              domainSize));

        }
    else if (beta_type == "twistyStreamx")
      {
        Real betaVal, eps, magOffset;
        magOffset = 0.25;
        RealVect omega(RealVect::Unit);
        Vector<Real> omegaVect(SpaceDim);
        geomPP.get("betaValue", betaVal);
        if (geomPP.contains("omega"))
          {
            geomPP.getarr("omega", omegaVect, 0, SpaceDim);
            omega = RealVect(D_DECL(omegaVect[0], omegaVect[1], omegaVect[2]));
          }
        geomPP.query("magOffset", magOffset);
        geomPP.get("betaEps", eps);
        // fix this later, if we need to...
        MayDay::Error("trying to define basal Friction with undefined DomainSize");
        basalFrictionPtr = static_cast<BasalFriction*>(new twistyStreamFriction(betaVal, 
                                                                                omega, 
                                                                                magOffset, 
                                                                                eps,
                                                                                domainSize));          
      }
     else if (beta_type == "gaussianBump")
      {
	int nt;
	geomPP.get("gaussianBump_nt", nt);
	Vector<Real> t(nt-1);
	Vector<Real> C0(nt),a(nt);
	Vector<RealVect> b(nt),c(nt);

	geomPP.getarr("gaussianBump_t", t, 0, nt-1);
	geomPP.getarr("gaussianBump_C", C0, 0, nt);
	geomPP.getarr("gaussianBump_a", a, 0, nt);
       
#if CH_SPACEDIM == 2
	Vector<Real> xb(nt),yb(nt),xc(nt),yc(nt);
	geomPP.getarr("gaussianBump_xb", xb, 0, nt);
	geomPP.getarr("gaussianBump_xc", xc, 0, nt);
	geomPP.getarr("gaussianBump_yb", yb, 0, nt);
	geomPP.getarr("gaussianBump_yc", yc, 0, nt);
	for (int i = 0; i < nt; ++i)
	  {
	    b[i][0] = xb[i];
	    b[i][1] = yb[i];
	    c[i][0] = xc[i];
	    c[i][1] = yc[i];
	  }
#else
	       MayDay::Error("beta_type = gaussianBump not implemented for CH_SPACEDIM > 2")
#endif
        basalFrictionPtr = static_cast<BasalFriction*>
	  (new GaussianBumpFriction(t, C0, a, b, c));
      }
     else if (beta_type == "fortran")
       {
         cout << "setting up basal friction coefficient " << endl;
         double * basalTractionCoeffPtr = btg_ptr -> getDoubleVar("btrc","velocity"); 
         FortranInterfaceBasalFriction* fibfPtr = new FortranInterfaceBasalFriction();
         RealVect dx;
         dx[0] = dew;
         dx[1] = dns;
         
         // presumption is that basal friction centering is opposite
         // of thickness (can over-ride in inputs file if otherwise)
         bool bfnodal = !nodalGeom;
         interfacePP.query("nodalBasalFrictionData", bfnodal);
         
         if (!bfnodal)
           {
             // friction is cell-centered...
             cout << "cell-centered basal friction data" << endl;
             fibfPtr->setReferenceFAB(basalTractionCoeffPtr, dimInfoVelo, dx, 
                                      ghostVect, false);
           }
         else
           {
             // (SLC -- 11/25/11) -- if we are taking glimmer's thickness to
             // be cell-centered, then its friction must be node centered 
             cout << "nodal basal friction data will be averaged to cell-centers" << endl;
             fibfPtr->setReferenceFAB(basalTractionCoeffPtr, dimInfoVelo, dx, 
                                      ghostVect, bfnodal);
           }
         basalFrictionPtr = static_cast<BasalFriction*>(fibfPtr);
       }
    
     else if (beta_type == "LevelData")
       {
	 //read a one level beta^2 from an AMR Hierarchy, and  store it in a LevelDataBasalFriction
	 ParmParse ildPP("inputLevelData");
	 std::string infile;
	 ildPP.get("frictionFile",infile);
	 std::string frictionName = "btrc";
	 ildPP.query("frictionName",frictionName);

	 RefCountedPtr<LevelData<FArrayBox> > levelC (new LevelData<FArrayBox>());

	 Real dx;

	 Vector<RefCountedPtr<LevelData<FArrayBox> > > vectC;
	 vectC.push_back(levelC);

	 Vector<std::string> names(1);
	 names[0] = frictionName;


	 readLevelData(vectC,dx,infile,names,1);
	   
	 RealVect levelDx = RealVect::Unit * dx;
	 basalFrictionPtr = static_cast<BasalFriction*>
	   (new LevelDataBasalFriction(levelC,levelDx));
       }

    else 
      {
        MayDay::Error("undefined beta_type in inputs");
      }

    amrObjectPtr->setBasalFriction(basalFrictionPtr);
    
    BasalFrictionRelation* basalFrictionRelationPtr;
    std::string basalFrictionRelType = "powerLaw";
    pp2.query("basalFrictionRelation", basalFrictionRelType);
    
    if (basalFrictionRelType == "powerLaw")
      {
	ParmParse plPP("BasalFrictionPowerLaw");

	Real m = 1.0;
	plPP.query("m",m);
	bool includeEffectivePressure = false;
	plPP.query("includeEffectivePressure",includeEffectivePressure);

	BasalFrictionPowerLaw*  pl = new BasalFrictionPowerLaw(m,includeEffectivePressure);
	basalFrictionRelationPtr = static_cast<BasalFrictionRelation*>(pl);
      }
    else
      {
	MayDay::Error("undefined basalFrictionRelation in inputs");
      }

    amrObjectPtr->setBasalFrictionRelation(basalFrictionRelationPtr);
*/
    if (verbose)
      {
	cout << "... done" << endl;
      }

    // ---------------------------------------------
    // now set temperature BC's
    // ---------------------------------------------
/*
    IceTemperatureIBC* temperatureIBC = NULL;
    ParmParse tempPP("temperature");
    std::string tempType("constant");
    tempPP.query("type",tempType);
    if (tempType == "constant")
      {
	Real T = 258.0;
	tempPP.query("value",T);
	ConstantIceTemperatureIBC* ptr = new ConstantIceTemperatureIBC(T);
	temperatureIBC  = static_cast<IceTemperatureIBC*>(ptr);
      }
    else if (tempType == "LevelData")
      {
	ParmParse ildPP("inputLevelData");
	LevelDataTemperatureIBC* ptr = NULL;
	CH_assert( (ptr = LevelDataTemperatureIBC::parse(ildPP)) != NULL);
	temperatureIBC  = static_cast<IceTemperatureIBC*>(ptr);
      }
    else 
      {
	MayDay::Error("bad temperature type");
      }	
	
    amrObjectPtr->setTemperatureBC(temperatureIBC);
 
   
    bike_store(btg_ptr -> getDyCoreIndex(), &bikePtr,0);

    // set up initial grids, initialize data, etc.
    cout << "Calling initialize..." << endl;  
    amrObjectPtr -> initialize();
    cout << "AMR object initialized." << endl;

    //Real startTime;

    // maxTime is passed in from CISM
    //pp2.get("maxTime", maxTime);
    pp2.get("maxStep", maxStep);
    
    // final thing to do -- return flattened states back to CISM
    // (including velocity)

    // first, return ice geometry (in case something was changed)
    if (problem_type =="fortran")
      {
        // can use existing IBC
        FortranInterfaceIBC* fibcPtr = dynamic_cast<FortranInterfaceIBC*>(thicknessIBC);
        const Vector<RefCountedPtr<LevelSigmaCS> >& amrGeometry = amrObjectPtr->amrGeometry();
        fibcPtr->flattenIceGeometry(amrGeometry);

        // now velocity
        // velocity should have the same centering as basal friction
        // (which should be opposite of thickness) -- can be over-ridden
        // in inputs file if otherwise
        bool nodalVel = !nodalGeom;
        // first over-ride to friction centering if specified
        interfacePP.query("nodalBasalFrictionData", nodalVel);
        // then override the override if we want to specify velocity as 
        // different from friction
        interfacePP.query("nodalVelocityData", nodalVel);
        
        long ewlb, ewub, nslb, nsub;

        ewlb = *(btg_ptr -> getLongVar("ewlb","geometry"));
        ewub = *(btg_ptr -> getLongVar("ewub","geometry"));
        nslb = *(btg_ptr -> getLongVar("nslb","geometry"));
        nsub = *(btg_ptr -> getLongVar("nsub","geometry"));
        cout << "In bike_driver: ewlb, ewub = " << ewlb << "  " << ewub <<  endl;
        cout << "In bike_driver: nslb, nsub = " << nslb << "  " << nsub <<  endl;

        int lb[SpaceDim];
        int ub[SpaceDim];

        D_TERM(lb[0] = ewlb;
               ub[0] = ewub;,
               lb[1] = nslb;
               ub[1] = nsub;,
               lb[2] = 0;
               ub[2] = numCells[2]-1;)

        int lbvel[SpaceDim];
        int ubvel[SpaceDim];

        IntVect velGhost = ghostVect;
        IntVect velOffset = IntVect::Unit;

        if (nodalVel)
          {
            D_TERM(lbvel[0] = ewlb;
                   ubvel[0] = ewub-1;,
                   lbvel[1] = nslb;
                   ubvel[1] = nsub-1;,
                   lbvel[2] = 0;
                   ubvel[2] = numCells[2]-1;)

              // because of the enclosed-nodes->ghostcell combination,
              // velocity appears to have one less ghost than the CC variables
              velGhost = ghostVect;
            //velGhost -= IntVect::Unit;
            
            // also, node-centering apparently doesn't require an offset
            velOffset = IntVect::Zero;
            
          }
        else
          {
            // cell-centered velocity
            D_TERM(lbvel[0] = ewlb;
                   ubvel[0] = ewub-1;,
                   lbvel[1] = nslb;
                   ubvel[1] = nsub-1;,
                   lbvel[2] = 0;
                   ubvel[2] = numCells[2]-1;)
              
              // because of the enclosed-nodes->ghostcell combination,
              // velocity appears to have one less ghost than the CC variables
              velGhost = ghostVect;
            
            // also, node-centering requires an offset
            velOffset = IntVect::Unit;
            
            
          }
        // this is to convert between C- and Fortran ordering
        //IntVect offset(IntVect::Unit);
        

        // velocity is node-centered
        pout() << "flattening velocity data" << endl;
        
        const Vector<LevelData<FArrayBox>* >& amrVel = amrObjectPtr->amrVelocity();
        const Vector<int>& refRatio = amrObjectPtr->refRatios();
        
        // get uvel and vvel from registry
        double* uVelPtr = btg_ptr->getDoubleVar("uvel", "velocity");
        double* vVelPtr = btg_ptr->getDoubleVar("vvel", "velocity");
        
        const Vector<Real>& sigmaLevels = amrObjectPtr->getFaceSigma();
        const Vector<Real>& amrDx = amrObjectPtr->amrDx();
        

        if (!nodalVel)
          {
            //            velOffset = IntVect::Unit;
          }
        
        if (verbose) 
          {
            pout () << "entering flattenVelocity: lbvel = (" 
                    << lbvel[0] << ", " << lbvel[1] << "), ubvel = (" 
                    << ubvel[0] << ", " << ubvel[1] << ")" << endl;
          }
        
        // first cut, just flatten basal velocity and ignore vertical shear
        fibcPtr->flattenVelocity(uVelPtr, vVelPtr, dimInfoVelo,
                                 lbvel, ubvel,
                                 &dew, &dns, velOffset,
                                 amrVel, refRatio, amrDx, velGhost,
                                 nodalVel);
              
        
        // flatten temperatures -- only want temperature at lower cell
        const Vector<LevelData<FArrayBox>* >& fullTemperatures=amrObjectPtr->getTemperature();
        const Vector<Real>& sigma = amrGeometry[0]->getSigma();
        int numLayers = sigma.size();
        double* lower_cell_tempDataPtr;
        lower_cell_tempDataPtr = btg_ptr -> getDoubleVar("lower_cell_temp","geometry");
        
        // same centering as thickness, etc, so use nodalGeom
        fibcPtr->flattenData(lower_cell_tempDataPtr,
                             dimInfoGeom,
                             lb, ub,
                             &dew, &dns,
                             offset,
                             fullTemperatures,
                             refRatio, amrDx, 
                             numLayers-1, 0, 1,
                             ghostVect, 
                             nodalGeom);
        
      }
    else
      {
        // create a FortranInterfaceIBC for returning data to CISM
        MayDay::Error("Non-FortranIBC geometry not implemented yet");
      }

    
    // now, return velocities
    */  

    // return temperatures


    // clean up
    cout << "exec mode = " << exec_mode << endl;
    /*if (exec_mode == 0) {
    if (constRelPtr != NULL)
      {
        delete constRelPtr;
        constRelPtr = NULL;
      }
            

    if (surf_flux_ptr != NULL)
      {
        delete surf_flux_ptr;
        surf_flux_ptr = NULL;
      }

    if (basal_flux_ptr != NULL)
      {
        delete basal_flux_ptr;
        basal_flux_ptr = NULL;
      }
       

    if (thicknessIBC != NULL)
      {
	delete thicknessIBC;
        thicknessIBC=NULL;
      }    
    if (basalFrictionPtr != NULL)
      {
        delete basalFrictionPtr;
        basalFrictionPtr = NULL;
      }
    if (basalFrictionRelationPtr != NULL)
      {
        delete basalFrictionRelationPtr;
        basalFrictionRelationPtr = NULL;
      }
    } // if exec_mode == 2)*/

  cout << "End of nested scope." << endl; 
  }  

 

}


// updates cur_time_yr as solution is advanced
void felix_driver_run(FelixToGlimmer * btg_ptr, float& cur_time_yr, float time_inc_yr)
{
  Felix *felixPtr;
  
  cout << "In felix_driver_run, cur_time, time_inc = " 
       << cur_time_yr << "   " << time_inc_yr << endl;
 
  felix_store(btg_ptr -> getDyCoreIndex(), &felixPtr ,1);



}
  

void felix_driver_finalize(int amr_obj_index)
{
  Felix* felixPtr;

  cout << "In felix_driver_finalize..." << endl;

  felix_store(amr_obj_index, &felixPtr, 1);
  
  if (felixPtr != NULL)
    {
      //delete felixPtr; 
      felixPtr = NULL;
    }
  cout << "Felix Object deleted." << endl << endl; 
//#ifdef CH_MPI
  //  MPI_Finalize();
//#endif
  
    //return 0;
}

