//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <Intrepid_MiniTensor.h>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_RealSpaceTools.hpp"

#include <typeinfo>

namespace LCM {

  template<typename EvalT, typename Traits>
  FaceFractureCriteria<EvalT, Traits>::FaceFractureCriteria(
      const Teuchos::ParameterList& p) :
      coord(p.get<std::string>("Coordinate Vector Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Vertex Vector Data Layout")), faceAve(
          p.get<std::string>("Face Average Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Face Vector Data Layout")), yieldStrength(
          p.get<RealType>("Yield Name")), fractureLimit(
          p.get<RealType>("Fracture Limit Name")), criterion(
          p.get<std::string>("Insertion Criteria Name")), cellType(
          p.get<Teuchos::RCP<shards::CellTopology>>("Cell Type")), criteriaMet(
          p.get<std::string>("Criteria Met Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Face Scalar Data Layout")), temp(
          p.get<std::string>("Temp2 Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Cell Scalar Data Layout"))
  {
    this->addDependentField(coord);
    this->addDependentField(faceAve);

    this->addEvaluatedField(criteriaMet);
    this->addEvaluatedField(temp); // temp for testing

    // Get Dimensions
    Teuchos::RCP<PHX::DataLayout> vec_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("Face Vector Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    vec_dl->dimensions(dims);

    worksetSize = dims[0];
    numFaces = dims[1];
    numComp = dims[2];

    /* As the vector length for this problem is not equal to the number
     * of spatial dimensions (as in the normal mechanics problems), we
     * get the spatial dimension from the coordinate vector.
     */
    Teuchos::RCP<PHX::DataLayout> vertex_dl = p.get<
        Teuchos::RCP<PHX::DataLayout>>("Vertex Vector Data Layout");
    vertex_dl->dimensions(dims);
    numDims = dims[2];

    sides = cellType->getCellTopologyData()->side;

    this->setName("FaceFractureCriteria" + PHX::typeAsString<EvalT>());
  }

  template<typename EvalT, typename Traits>
  void FaceFractureCriteria<EvalT, Traits>::postRegistrationSetup(
      typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
  {
    this->utils.setFieldData(coord, fm);
    this->utils.setFieldData(faceAve, fm);
    this->utils.setFieldData(criteriaMet, fm);

    this->utils.setFieldData(temp, fm); // temp for testing

  }

  template<typename EvalT, typename Traits>
  void FaceFractureCriteria<EvalT, Traits>::evaluateFields(
      typename Traits::EvalData workset)
  {
    std::cout << "Insertion Criterion: " << criterion << std::endl;
    if (criterion == "Test Fracture")
      testFracture();
    else if (criterion == "Traction")
      tractionCriterion();
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Fracture Criterion Not Recognized");

  }

  // Test fracture criterion
  template<typename EvalT, typename Traits>
  void FaceFractureCriteria<EvalT, Traits>::testFracture()
  {
    TEUCHOS_TEST_FOR_EXCEPTION(fractureLimit<=0.0, std::logic_error,
        "Fracture Limit Must Be > 0");

    for (int cell = 0; cell < worksetSize; ++cell) {
      for (int face = 0; face < numFaces; ++face) {
        ScalarT max_comp = 0.0;
        criteriaMet(cell, face) = 0;
        for (int comp = 0; comp < numComp; ++comp) {
          max_comp = std::max(faceAve(cell, face, comp), max_comp);
        }
        if (max_comp > fractureLimit) {
          criteriaMet(cell, face) = 1;
          // for debug
          std::cout << "Fracture Criteria met for (cell, face): " << cell << ","
                    << face << " Max Stress: " << max_comp << std::endl;
        }
      }

      // hack to force evaluation
      //amb Something's not right here. Probably a dummy evaluated field is
      // intended.
      temp(cell,0) = 0.0;
    }

  }

  // Traction based criterion
  template<typename EvalT, typename Traits>
  void FaceFractureCriteria<EvalT, Traits>::tractionCriterion()
  {
    TEUCHOS_TEST_FOR_EXCEPTION(fractureLimit<=0.0, std::logic_error,
        "Fracture Limit Must Be > 0");

    // local variables to create the traction vector
    Intrepid::Vector<ScalarT> p0(3);
    Intrepid::Vector<ScalarT> p1(3);
    Intrepid::Vector<ScalarT> p2(3);
    Intrepid::Vector<ScalarT> normal(3); // face normal
    Intrepid::Vector<ScalarT> traction(3);
    Intrepid::Tensor<ScalarT> stress(3);

    for (int cell = 0; cell < worksetSize; ++cell) {
      //hack to force evaluation
      temp(cell,0) = 0.0;

      for (int face = 0; face < numFaces; ++face) {
        criteriaMet(cell, face) = 0;

        // Get the face normal
        // First get the (local) IDs of three independent nodes on the face
        int n0 = sides[face].node[0];
        int n1 = sides[face].node[1];
        int n2 = sides[face].node[2];

        // Then create a vector of the nodal points of each
        for (int i = 0; i < numDims; ++i) {
          p0(i) = coord(cell, n0, i);
          p1(i) = coord(cell, n1, i);
          p2(i) = coord(cell, n2, i);
        }

        normal = Intrepid::normal(p0, p1, p2);

        // fill the stress tensor
        for (int i = 0; i < numComp; ++i) {
          stress(i / numDims, i % numDims) = faceAve(cell, face, i);
        }

        // Get the traction
        traction = Intrepid::dot(stress, normal);
        ScalarT traction_norm = Intrepid::norm(traction);

        if (traction_norm > fractureLimit) {
          criteriaMet(cell, face) = 1;
          // for debug
          std::cout << "Fracture Criteria met for (cell, face): " << cell << ","
                    << face << " Max Stress: " << traction_norm << std::endl;
        }
        //else
        //    cout << "Fracture Criteria NOT met for (cell, face): " << cell << ","
        //         << face << " Max Stress: " << traction_norm << std::endl;
      }
    }
  }

} // namespace LCM

