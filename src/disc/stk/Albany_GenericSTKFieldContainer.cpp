//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <iostream>

#include "Albany_GenericSTKFieldContainer.hpp"

#include "Albany_Utils.hpp"

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

Albany::GenericSTKFieldContainer::GenericSTKFieldContainer(
   const Teuchos::RCP<Teuchos::ParameterList>& params_,
   stk::mesh::fem::FEMMetaData* metaData_,
   const int neq, 
   const AbstractFieldContainer::FieldContainerRequirements& req,
   const int numDim,
   const Teuchos::RCP<Albany::StateInfoStruct>& sis)
    : metaData(metaData_),
      params(params_),
      buildSurfaceHeight(false)
{

  if(std::find(req.begin(), req.end(), "Surface Height") != req.end()){
      buildSurfaceHeight = true;
  }
  else {
      surfaceHeight_field = NULL;
  }

  //Start STK stuff
  coordinates_field = & metaData->declare_field< VectorFieldType >( "coordinates" );
  solution_field = & metaData->declare_field< VectorFieldType >(
    params->get<string>("Exodus Solution Name", "solution"));

#ifdef ALBANY_LCM
  residual_field = & metaData->declare_field< VectorFieldType >(
    params->get<string>("Exodus Residual Name", "residual"));
#endif

#ifdef ALBANY_FELIX
  if(buildSurfaceHeight)
    surfaceHeight_field = & metaData->declare_field< ScalarFieldType >("surface_height");
#endif

  stk::mesh::put_field( *coordinates_field , metaData->node_rank() , metaData->universal_part(), numDim );
  stk::mesh::put_field( *solution_field , metaData->node_rank() , metaData->universal_part(), neq );

#ifdef ALBANY_LCM
  stk::mesh::put_field( *residual_field , metaData->node_rank() , metaData->universal_part() , neq );
#endif

#ifdef ALBANY_FELIX
  if(buildSurfaceHeight)
    stk::mesh::put_field( *surfaceHeight_field , metaData->node_rank() , metaData->universal_part());
#endif
  
#ifdef ALBANY_SEACAS
  stk::io::set_field_role(*coordinates_field, Ioss::Field::MESH);
  stk::io::set_field_role(*solution_field, Ioss::Field::TRANSIENT);
#ifdef ALBANY_LCM
  stk::io::set_field_role(*residual_field, Ioss::Field::TRANSIENT);
#endif

#ifdef ALBANY_FELIX
  // ATTRIBUTE writes only once per file, but somehow did not work on restart.
  //stk::io::set_field_role(*surfaceHeight_field, Ioss::Field::ATTRIBUTE);
  if(buildSurfaceHeight)
     stk::io::set_field_role(*surfaceHeight_field, Ioss::Field::TRANSIENT);
#endif
#endif

  // Code to parse the vector of StateStructs and create STK fields
  for (std::size_t i=0; i<sis->size(); i++) {
    Albany::StateStruct& st = *((*sis)[i]);
    std::vector<int>& dim = st.dim;
    if (dim.size() == 2 && st.entity=="QuadPoint") {
      qpscalar_states.push_back(& metaData->declare_field< QPScalarFieldType >( st.name) );
      stk::mesh::put_field( *qpscalar_states.back() , metaData->element_rank(),
			    metaData->universal_part(), dim[1]);
      cout << "NNNN qps field name " << qpscalar_states.back()->name() << endl;
#ifdef ALBANY_SEACAS
      if (st.output) stk::io::set_field_role(*qpscalar_states.back(), Ioss::Field::TRANSIENT);
#endif
    }
    else if (dim.size() == 3 && st.entity=="QuadPoint") {
      qpvector_states.push_back(& metaData->declare_field< QPVectorFieldType >( st.name) );
      // Multi-dim order is Fortran Ordering, so reversed here
      stk::mesh::put_field( *qpvector_states.back() , metaData->element_rank(),
			    metaData->universal_part(), dim[2], dim[1]);
      cout << "NNNN qpv field name " << qpvector_states.back()->name() << endl;
#ifdef ALBANY_SEACAS
      if (st.output) stk::io::set_field_role(*qpvector_states.back(), Ioss::Field::TRANSIENT);
#endif
    }
    else if (dim.size() == 4 && st.entity=="QuadPoint") {
      qptensor_states.push_back(& metaData->declare_field< QPTensorFieldType >( st.name) );
      // Multi-dim order is Fortran Ordering, so reversed here
      stk::mesh::put_field( *qptensor_states.back() , metaData->element_rank(),
			    metaData->universal_part(), dim[3], dim[2], dim[1]);
      cout << "NNNN qpt field name " << qptensor_states.back()->name() << endl;
#ifdef ALBANY_SEACAS
      if (st.output) stk::io::set_field_role(*qptensor_states.back(), Ioss::Field::TRANSIENT);
#endif
    }
    else if ( dim.size() == 1 && st.entity=="ScalarValue" ) {
      scalarValue_states.push_back(st.name);
    }
    else TEUCHOS_TEST_FOR_EXCEPT(dim.size() < 2 || dim.size()>4 || st.entity!="QuadPoint");

  }
  
  initializeSTKAdaptation();

}

Albany::GenericSTKFieldContainer::~GenericSTKFieldContainer(){
}

void Albany::GenericSTKFieldContainer::initializeSTKAdaptation(){

    proc_rank_field = & metaData->declare_field< IntScalarFieldType >( "proc_rank" );
    // Processor rank field, a scalar
    stk::mesh::put_field( *proc_rank_field , metaData->element_rank() , metaData->universal_part());
#ifdef ALBANY_SEACAS
    stk::io::set_field_role(*proc_rank_field, Ioss::Field::MESH);
#endif

}

