//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_GENERICSTKFIELDCONT_HPP
#define ALBANY_GENERICSTKFIELDCONT_HPP

#include "Albany_AbstractSTKFieldContainer.hpp"

// Start of STK stuff
#include <stk_util/parallel/Parallel.hpp>
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/fem/FEMMetaData.hpp>


namespace Albany {


  class GenericSTKFieldContainer : public AbstractSTKFieldContainer {

    public:

    GenericSTKFieldContainer(const Teuchos::RCP<Teuchos::ParameterList>& params_,
      stk::mesh::fem::FEMMetaData* metaData_,
      const int neq_, 
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const int numDim_,
      const Teuchos::RCP<Albany::StateInfoStruct>& sis);

    ~GenericSTKFieldContainer();

    double *getSolutionFieldData(const stk::mesh::Entity& ent){ 
        return stk::mesh::field_data(*solution_field, ent); }

    double *getResidualFieldData(const stk::mesh::Entity& ent){
        return stk::mesh::field_data(*residual_field, ent); }



    private:

       void initializeSTKAdaptation();
       stk::mesh::fem::FEMMetaData* metaData;
       Teuchos::RCP<Teuchos::ParameterList> params;

       bool buildSurfaceHeight;

  };

}

#endif
