#include "LandIce_ProblemUtils.hpp"

#include "Phalanx_DataLayout_MDALayout.hpp"

namespace LandIce
{

Teuchos::RCP<PHX::DataLayout>
extrudeSideLayout (const Teuchos::RCP<PHX::DataLayout>& in, const int numLayers) {
  Teuchos::RCP<PHX::DataLayout> out;

  std::vector<std::string> names;
  in->names(names);
  TEUCHOS_TEST_FOR_EXCEPTION(names[0]!="Cell" && names[1]!="Side", std::runtime_error,
    "Error! A side layout must begin with <Cell,Side,...>.\n");

  std::vector<PHX::DataLayout::size_type> dims;
  in->dimensions(dims);
  const int rank = in->rank();
  switch (rank) {
    case 2:
      out = Teuchos::rcp(new PHX::MDALayout<Cell,Side,LayerDim>(dims[0],dims[1],numLayers));
        break;
    case 3:
      if (names[2]=="Node") {
        out = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Node,LayerDim>(dims[0],dims[1],dims[2],numLayers));
      } else if (names[2]=="Dim") {
        out = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Dim,LayerDim>(dims[0],dims[1],dims[2],numLayers));
      } else if (names[2]=="VecDim") {
        out = Teuchos::rcp(new PHX::MDALayout<Cell,Side,VecDim,LayerDim>(dims[0],dims[1],dims[2],numLayers));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          "Error! Unexpected value for names[2]: " + names[2] + ".\n");
      }
      break;
    case 4:
      if (names[2]=="Dim" && names[3]=="Dim") {
        out = Teuchos::rcp(new PHX::MDALayout<Cell,Side,Dim,Dim,LayerDim>(dims[0],dims[1],dims[2],dims[3],numLayers));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          "Error! Unexpected value for names[2],names[3]: " + names[2] + ", " + names[3] + ".\n");
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "Error! Unexpected rank for input layout (" + std::to_string(rank) + ").\n");
  }

  return out;
}

Teuchos::RCP<PHX::DataLayout>
extrudeCollapsedSideLayout (const Teuchos::RCP<PHX::DataLayout>& in, const int numLayers) {
  Teuchos::RCP<PHX::DataLayout> out;

  std::vector<std::string> names;
  in->names(names);
  TEUCHOS_TEST_FOR_EXCEPTION(names[0]!="Side", std::runtime_error,
    "Error! A collapsed side layout must begin with <Side,...>.\n");

  std::vector<PHX::DataLayout::size_type> dims;
  in->dimensions(dims);
  const int rank = in->rank();
  switch (rank) {
    case 1:
      out = Teuchos::rcp(new PHX::MDALayout<Side,LayerDim>(dims[0],numLayers));
        break;
    case 2:
      if (names[1]=="Node") {
        out = Teuchos::rcp(new PHX::MDALayout<Side,Node,LayerDim>(dims[0],dims[1],numLayers));
      } else if (names[1]=="Dim") {
        out = Teuchos::rcp(new PHX::MDALayout<Side,Dim,LayerDim>(dims[0],dims[1],numLayers));
      } else if (names[1]=="VecDim") {
        out = Teuchos::rcp(new PHX::MDALayout<Side,VecDim,LayerDim>(dims[0],dims[1],numLayers));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          "Error! Unexpected value for names[1]: " + names[1] + ".\n");
      }
      break;
    case 3:
      if (names[1]=="Dim" && names[2]=="Dim") {
        out = Teuchos::rcp(new PHX::MDALayout<Side,Dim,Dim,LayerDim>(dims[0],dims[1],dims[2],numLayers));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          "Error! Unexpected value for names[1],names[2]: " + names[1] + ", " + names[2] + ".\n");
      }
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "Error! Unexpected rank for input layout (" + std::to_string(rank) + ").\n");
  }

  return out;
}

} // namespace LandIce
