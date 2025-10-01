//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_LAYERED_MESH_NUMBERING_HPP
#define ALBANY_LAYERED_MESH_NUMBERING_HPP

namespace Albany {

enum class LayeredMeshOrdering
{
  LAYER  = 0,
  COLUMN = 1
};

template <typename T>
struct LayeredMeshNumbering {
  // Position of top/bot side in the local element
  int bot_side_pos = -1;
  int top_side_pos = -1;
  bool layerOrd;

  // T stride;
  T numHorizEntities;

  LayeredMeshOrdering ordering;
  int numLayers;

  LayeredMeshNumbering(const int _numLayers,
                       const LayeredMeshOrdering _ordering)
  {
    numHorizEntities = -1;
    ordering = _ordering;
    layerOrd = ordering == LayeredMeshOrdering::LAYER;
    numLayers = _numLayers;
  }

  LayeredMeshNumbering(const T _numHorizEntities,
                       const int _numLayers,
                       const LayeredMeshOrdering _ordering)
   : LayeredMeshNumbering(_numLayers,_ordering)
  {
    numHorizEntities = _numHorizEntities;
  }

  T getId(const T column_id, const T level_index) const {
      return layerOrd ? column_id + level_index*numHorizEntities :
                        column_id * numLayers + level_index;
  }

  void getIndices(const T id, T& column_id, T& level_index) const {
    if(layerOrd) {
      level_index = id / numHorizEntities;
      column_id   = id % numHorizEntities;
    } else {
      level_index = id % numLayers;
      column_id   = id / numLayers;
    }
  }

  // Shift between adjacent entities in the same column
  T getColumnShift () const {
    return layerOrd ? numHorizEntities : 1;
  }
  // Shift between adjacent entities in the same layer
  T getLayerShift () const {
    return layerOrd ? 1 : numLayers;
  }

  T getColumnId (const T id) const {
    return layerOrd ? id % numHorizEntities : id / numLayers;
  }

  T getLayerId (const T id) const {
    return layerOrd ? id / numHorizEntities : id % numLayers;
  }
};

} // Namespace Albany

#endif // ALBANY_LAYERED_MESH_NUMBERING_HPP
