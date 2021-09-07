#include "LandIce_ProblemUtils.hpp"

#include "Phalanx_DataLayout_MDALayout.hpp"

namespace LandIce
{

Teuchos::RCP<PHX::DataLayout>
extrudeSideLayout (const Teuchos::RCP<PHX::DataLayout>& in, const int numLayers) {
  Teuchos::RCP<PHX::DataLayout> out;

  std::vector<std::string> names;
  in->names(names);
  TEUCHOS_TEST_FOR_EXCEPTION(names[0]!=PHX::print<Side>(), std::runtime_error,
    "Error! A side layout must begin with <Side,...>.\n");

  std::vector<PHX::DataLayout::size_type> dims;
  in->dimensions(dims);
  const int rank = in->rank();
  switch (rank) {
    case 1:
      out = Teuchos::rcp(new PHX::MDALayout<Side,LayerDim>(dims[0],numLayers));
        break;
    case 2:
      if (names[1]==PHX::print<Node>()) {
        out = Teuchos::rcp(new PHX::MDALayout<Side,Node,LayerDim>(dims[0],dims[1],numLayers));
      } else if (names[1]==PHX::print<Dim>()) {
        out = Teuchos::rcp(new PHX::MDALayout<Side,Dim,LayerDim>(dims[0],dims[1],numLayers));
      } else if (names[1]==PHX::print<VecDim>()) {
        out = Teuchos::rcp(new PHX::MDALayout<Side,VecDim,LayerDim>(dims[0],dims[1],numLayers));
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
          "Error! Unexpected value for names[1]: " + names[1] + ".\n");
      }
      break;
    case 3:
      if (names[1]==PHX::print<Node>() && names[2]==PHX::print<Dim>()) {
        out = Teuchos::rcp(new PHX::MDALayout<Side,Node,Dim,LayerDim>(dims[0],dims[1],dims[2],numLayers));
      } else if (names[1]==PHX::print<Node>() && names[2]==PHX::print<VecDim>()) {
        out = Teuchos::rcp(new PHX::MDALayout<Side,Node,VecDim,LayerDim>(dims[0],dims[1],dims[2],numLayers));
      } else if (names[1]==PHX::print<Dim>() && names[2]==PHX::print<Dim>()) {
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
