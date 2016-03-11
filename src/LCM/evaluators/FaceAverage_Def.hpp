//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"

#include <typeinfo>

namespace LCM {

template<typename EvalT, typename Traits>
FaceAverage<EvalT, Traits>::
FaceAverage(const Teuchos::ParameterList& p) :
  coordinates(p.get<std::string>("Coordinate Vector Name"),
    p.get<Teuchos::RCP<PHX::DataLayout>>("Vertex Vector Data Layout")),
  projected(p.get<std::string>("Projected Field Name"),
    p.get<Teuchos::RCP<PHX::DataLayout>>("Node Vector Data Layout")),
  cubature(p.get<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> >>>("Face Cubature")),
  intrepidBasis(p.get<Teuchos::RCP<Intrepid2::Basis
    <RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>>("Face Intrepid2 Basis")),
  cellType(p.get<Teuchos::RCP<shards::CellTopology>>("Cell Type")),
  faceAve(p.get<std::string>("Face Average Name"),
    p.get<Teuchos::RCP<PHX::DataLayout>>("Face Vector Data Layout")),
  temp(p.get<std::string>("Temp Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("Cell Scalar Data Layout"))
{
    this->addDependentField(coordinates);
    this->addDependentField(projected);
    this->addEvaluatedField(faceAve);

    this->addEvaluatedField(temp); // temp for testing

    // Get Dimensions
    Teuchos::RCP<PHX::DataLayout> vec_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Node Vector Data Layout");
    std::vector<PHX::DataLayout::size_type> dims;
    vec_dl->dimensions(dims);

    worksetSize = dims[0];
    numNodes    = dims[1];
    numComp     = dims[2];

    /* As the vector length for this problem is not equal to the number
     * of spatial dimensions (as in the normal mechanics problems), we
     * get the spatial dimension from the coordinate vector.
     */
    Teuchos::RCP<PHX::DataLayout> vertex_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("Vertex Vector Data Layout");
    vertex_dl->dimensions(dims);
    numDims = dims[2];

    // The number of quadrature points for the face
    numQPs = cubature->getNumPoints();

    numFaces = cellType->getFaceCount();
    faceDim = numDims - 1; // see note below for an explanation why this may be suboptimal

    /*  Get the number of face nodes from the cellType object.
     *  Note: Assumes that the face rank is one less than the
     *  spatial rank of the system. Will be incorrect if lower rank
     *  elements are embedded in a higher order problem(e.g. plate
     *  elements in a 3D problem).
     *  It might be better to replace this with a call to the topology
     *  of the cell face itself with
     *  numFaceNodes = sides->getCellTopology->num_nodes
     */
    numFaceNodes = cellType->getNodeCount(numDims-1,0);

    // Need the local ordering of the nodes on the faces
    sides = cellType->getCellTopologyData()->side;

    // Get the quadrature weights and basis functions
    refPoints.resize(numQPs,faceDim);
    refWeights.resize(numQPs);
    refValues.resize(numFaceNodes,numQPs);

    cubature->getCubature(refPoints,refWeights);
    intrepidBasis->getValues(refValues, refPoints, Intrepid2::OPERATOR_VALUE);


    this->setName("FaceAverage"+PHX::typeAsString<EvalT>());

}

template<typename EvalT, typename Traits>
void FaceAverage<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& fm)
{
    this->utils.setFieldData(coordinates,fm);
    this->utils.setFieldData(projected, fm);
    this->utils.setFieldData(faceAve, fm);

    this->utils.setFieldData(temp,fm);  // temp for testing

}

template<typename EvalT, typename Traits>
void FaceAverage<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset)
{


    // test output of the side ordering
    /*
    for (int i=0; i < cellType->getSideCount(); ++i){
      cout << "Side " << i << ":" << std::endl;
      for (int j=0; j < 4; ++j){
         cout << sides[i].node[j] << ",";
      }
      cout << std::endl;
    }
    */

    for (int cell=0; cell < workset.numCells; ++cell)
    {
      for (int face=0; face<numFaces; ++face)
      {
        for (int comp=0; comp<numComp; ++comp)
        {
          faceAve(cell,face,comp) = 0.0;
          double area = 0.0;

          for (int qp=0; qp<numQPs; ++qp)
          {
            area += refWeights(qp);

            for (int np=0; np<numFaceNodes; ++np)
            {
              // map from the local face node numbering to the local
              // element node numbering
              int node = sides[face].node[np];
              faceAve(cell,face,comp) += refValues(np,qp)*projected(cell,node,comp)*refWeights(qp);
              // A more naive averaging scheme
              //faceAve(cell,face,i) += projected(cell,node,i);
            }  // node
          }  // quadrature point

          faceAve(cell,face,comp) = faceAve(cell,face,comp) / area;
          // A more naive averaging scheme
          //faceAve(cell,face,i) = faceAve(cell,face,i)/static_cast<ScalarT>(numFaceNodes);

          // For debug purposes
          //cout << "Face Average (cell,face): (" << cell << "," << face << "): "
          //                    << faceAve(cell,face,comp) << std::endl;
        } // vector component
      } // face

      // temp variable to trick the code into running the evaluator
      //amb, during kokkos conversion. What's going on here? Why does the
      // evaluator need to be tricked? Why not use a dummy evaluated field like
      // usual?
      temp(cell,0) = cell;
    } // cell
}

} // namespace LCM

