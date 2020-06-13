//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_GenericSTKFieldContainer.hpp"
#include "Albany_STKNodeFieldContainer.hpp"

#include "Albany_Utils.hpp"
#include "Albany_StateInfoStruct.hpp"

// Start of STK stuff
#include <stk_mesh/base/Types.hpp>
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/MetaData.hpp>
#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#endif

namespace Albany {

template<bool Interleaved>
GenericSTKFieldContainer<Interleaved>::GenericSTKFieldContainer(
  const Teuchos::RCP<Teuchos::ParameterList>& params_,
  const Teuchos::RCP<stk::mesh::MetaData>& metaData_,
  const Teuchos::RCP<stk::mesh::BulkData>& bulkData_,
  const int neq_,
  const int numDim_)
  : metaData(metaData_),
    bulkData(bulkData_),
    params(params_),
    neq(neq_),
    numDim(numDim_) {
}

#ifdef ALBANY_SEACAS
namespace {
//amb 13 Nov 2014. After new STK was integrated, fields with output set to false
// were nonetheless being written to Exodus output files. As a possibly
// temporary but also possibly permanent fix, set the role of such fields to
// INFORMATION rather than TRANSIENT. The enum RoleType is defined in
// seacas/libraries/ioss/src/Ioss_Field.h. Grepping around there suggests that
// fields having the role INFORMATION are not written to file: first,
// INFORMATION is never actually used; second, I/O behavior is based on chained
// 'else if's with no trailing 'else'; hence, any role type not explicitly
// handled is not acted on.
// It appears that the output boolean is used only in this file in the context
// of role type, so for now I'm applying this fix only to this file.
//
// IKT, 5/9/2020: INFORMATION has gone away in Ioss_Field.h; replaced it with MESH_REDUCTION,
// per Greg Sjaardema's suggestion.  INFORMATION was an alias for MESH_REDUCTION.

inline Ioss::Field::RoleType role_type(const bool output) {
  return output ? Ioss::Field::TRANSIENT : Ioss::Field::MESH_REDUCTION;
}
}
#endif

template<bool Interleaved>
void GenericSTKFieldContainer<Interleaved>::
addStateStructs(const Teuchos::RCP<Albany::StateInfoStruct>& sis)
{
  if (sis==Teuchos::null)
    return;

  using namespace Albany;

  // QuadPoint fields
  // dim[0] = nCells, dim[1] = nQP, dim[2] = nVec dim[3] = nVec dim[4] = nVec
  typedef typename AbstractSTKFieldContainer::QPScalarFieldType QPSFT;
  typedef typename AbstractSTKFieldContainer::QPVectorFieldType QPVFT;
  typedef typename AbstractSTKFieldContainer::QPTensorFieldType QPTFT;
  typedef typename AbstractSTKFieldContainer::ScalarFieldType SFT;
  typedef typename AbstractSTKFieldContainer::VectorFieldType VFT;
  typedef typename AbstractSTKFieldContainer::TensorFieldType TFT;

  // Code to parse the vector of StateStructs and create STK fields
  for(std::size_t i = 0; i < sis->size(); i++) {
    StateStruct& st = *((*sis)[i]);
    StateStruct::FieldDims& dim = st.dim;

    if(st.entity == StateStruct::ElemData){

      if (dim.size()==1)
      {
        // Scalar on cell
        cell_scalar_states.push_back(& metaData->declare_field< SFT >(stk::topology::ELEMENT_RANK, st.name));
        stk::mesh::put_field_on_mesh(*cell_scalar_states.back(), metaData->universal_part(), 1, nullptr);
#ifdef ALBANY_SEACAS
        stk::io::set_field_role(*cell_scalar_states.back(), role_type(st.output));
#endif
      }
      else if (dim.size()==2)
      {
        // Vector on cell
        cell_vector_states.push_back(& metaData->declare_field< VFT >(stk::topology::ELEMENT_RANK, st.name));
        stk::mesh::put_field_on_mesh(*cell_vector_states.back(), metaData->universal_part(), dim[1], nullptr);
#ifdef ALBANY_SEACAS
        stk::io::set_field_role(*cell_vector_states.back(), role_type(st.output));
#endif
      }
      else if (dim.size()==3)
      {
        // 2nd order tensor on cell
        cell_tensor_states.push_back(& metaData->declare_field< TFT >(stk::topology::ELEMENT_RANK, st.name));
        stk::mesh::put_field_on_mesh(*cell_tensor_states.back(), metaData->universal_part(), dim[2], dim[1], nullptr);
#ifdef ALBANY_SEACAS
        stk::io::set_field_role(*cell_tensor_states.back(), role_type(st.output));
#endif
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Unexpected state rank.\n");
      }
      //Debug
      //      cout << "Allocating qps field name " << qpscalar_states.back()->name() <<
      //            " size: (" << dim[0] << ", " << dim[1] << ")" <<endl;

    } else if(st.entity == StateStruct::QuadPoint || st.entity == StateStruct::ElemNode){

        if(dim.size() == 2){ // Scalar at QPs
          qpscalar_states.push_back(& metaData->declare_field< QPSFT >(stk::topology::ELEMENT_RANK, st.name));
          stk::mesh::put_field_on_mesh(*qpscalar_states.back(), metaData->universal_part(), dim[1], nullptr);
        //Debug
        //      cout << "Allocating qps field name " << qpscalar_states.back()->name() <<
        //            " size: (" << dim[0] << ", " << dim[1] << ")" <<endl;
#ifdef ALBANY_SEACAS
          stk::io::set_field_role(*qpscalar_states.back(), role_type(st.output));
#endif
        }
        else if(dim.size() == 3){ // Vector at QPs
          qpvector_states.push_back(& metaData->declare_field< QPVFT >(stk::topology::ELEMENT_RANK, st.name));
          // Multi-dim order is Fortran Ordering, so reversed here
          stk::mesh::put_field_on_mesh(*qpvector_states.back(), metaData->universal_part(), dim[2], dim[1], nullptr);
          //Debug
          //      cout << "Allocating qpv field name " << qpvector_states.back()->name() <<
          //            " size: (" << dim[0] << ", " << dim[1] << ", " << dim[2] << ")" <<endl;
#ifdef ALBANY_SEACAS
          stk::io::set_field_role(*qpvector_states.back(), role_type(st.output));
#endif
        }
        else if(dim.size() == 4){ // Tensor at QPs
          qptensor_states.push_back(& metaData->declare_field< QPTFT >(stk::topology::ELEMENT_RANK, st.name));
          // Multi-dim order is Fortran Ordering, so reversed here
#ifdef IKT_DEBUG 
          //Debug
          std::cout << "Allocating qpt field name " << qptensor_states.back()->name() <<
                      " size: (" << dim[0] << ", " << dim[1] << ", " << dim[2] << ", " << dim[3] << ")" << std::endl;
#endif
          if (dim[1] == 4) {
            stk::mesh::put_field_on_mesh(*qptensor_states.back() ,
                           metaData->universal_part(), dim[3], dim[2], dim[1], nullptr);
          }
          else {
            //IKT, 12/20/18: this changes the way the qp_tensor field 
            //for 1D and 3D problems appears in the output exodus field.
            //Fields appear like: Cauchy_Stress_1_1, ...  Cauchy_Stress_8_9,
            //instead of Cauchy_Stress_1_01 .. Cauchy_Stress_3_24 to make it 
            //more clear which entry corresponds to which component/quad point.
            //I believe for 2D problems the original layout is correct, hence
            //the if statement above here.  
            stk::mesh::put_field_on_mesh(*qptensor_states.back() ,
                             metaData->universal_part(), dim[1], dim[2], dim[3], nullptr);
          }
#ifdef ALBANY_SEACAS
          stk::io::set_field_role(*qptensor_states.back(), role_type(st.output));
#endif
        }
        // Something other than a scalar, vector, or tensor
        else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
            "Error: GenericSTKFieldContainer - cannot match QPData");
    } // end QuadPoint
    // Single scalar at center of the workset
    else if(dim.size() == 1 && st.entity == StateStruct::WorksetValue) { // A single value that applies over the entire workset (time)
      scalarValue_states.push_back(&st.name); // Just save a pointer to the name allocated in st
    } // End scalar at center of element
    else if((st.entity == StateStruct::NodalData) ||(st.entity == StateStruct::NodalDataToElemNode) || (st.entity == StateStruct::NodalDistParameter))
    { // Data at the node points
        const Teuchos::RCP<Albany::NodeFieldContainer>& nodeContainer
               = sis->getNodalDataBase()->getNodeContainer();
      // const Teuchos::RCP<Albany::AbstractNodeFieldContainer>& nodeContainer
      //         = sis->getNodalDataBlock()->getNodeContainer();

        if(st.entity == StateStruct::NodalDataToElemNode) {
          nodal_sis.push_back((*sis)[i]);
          StateStruct::FieldDims nodalFieldDim;
          //convert ElemNode dims to NodalData dims.
          nodalFieldDim.insert(nodalFieldDim.begin(), dim.begin()+1,dim.end());
          (*nodeContainer)[st.name] = Albany::buildSTKNodeField(st.name, nodalFieldDim, metaData, st.output);
        }
        else if(st.entity == StateStruct::NodalDistParameter) {
          nodal_parameter_sis.push_back((*sis)[i]);
          StateStruct::FieldDims nodalFieldDim;
          //convert ElemNode dims to NodalData dims.
          nodalFieldDim.insert(nodalFieldDim.begin(), dim.begin()+1,dim.end());
          (*nodeContainer)[st.name] = Albany::buildSTKNodeField(st.name, nodalFieldDim, metaData, st.output);
        }
        else
          (*nodeContainer)[st.name] = Albany::buildSTKNodeField(st.name, dim, metaData, st.output);

    } // end Node class - anything else is an error
    else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
            "Error: GenericSTKFieldContainer - cannot match unknown entity : " << st.entity << std::endl);

    // Checking if the field is layered, in which case the normalized layer coordinates need to be stored in the meta data
    if (st.layered)
    {
      std::string tmp_str = st.name + "_NLC";

      TEUCHOS_TEST_FOR_EXCEPTION (mesh_vector_states.find(tmp_str)!=mesh_vector_states.end(), std::logic_error,
                                  "Error! Another layered state with the same name already exists.\n");
      TEUCHOS_TEST_FOR_EXCEPTION (dim.back()<=0, std::logic_error, 
                                  "Error! Invalid layer dimension for state " + st.name + ".\n");
      mesh_vector_states[tmp_str] = std::vector<double>(dim.back());
    }
  }
}

} // namespace Albany
