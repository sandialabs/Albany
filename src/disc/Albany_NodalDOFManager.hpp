//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_NODALDOFMANAGER_HPP
#define ALBANY_NODALDOFMANAGER_HPP

namespace Albany {


class NodalDOFManager {
  public:
//    NodalDOFManager(stk::mesh::BulkData const * bulkData = 0, int numComponents=0, int numLocalDOF=0, long long int numGlobalDOF=0, bool interleaved=true) :
//      _bulkData(bulkData),_numComponents(numComponents), _numLocalDOF(numLocalDOF), _numGlobalDOF(numGlobalDOF), _interleaved(interleaved){};
    NodalDOFManager() :
      _numComponents(0), _numLocalDOF(0), _numGlobalDOF(0), _interleaved(true){};

    void setup(int numComponents, int numLocalDOF, long long int numGlobalDOF, bool interleaved=true) {
      _numComponents = numComponents;
      _numLocalDOF = numLocalDOF;
      _numGlobalDOF = numGlobalDOF;
      _interleaved = interleaved;
    }

    inline int getLocalDOF(int inode, int icomp) const
      { return (_interleaved) ? inode*_numComponents + icomp : inode + _numLocalDOF*icomp; }
//    inline long long int getGlobalDOF(stk::mesh::Entity node, int icomp) const
//      { return (_interleaved) ? (_bulkData->identifier(node)-1) *_numComponents + icomp : (_bulkData->identifier(node)-1) + _numGlobalDOF*icomp; }
    inline uint64_t getGlobalDOF(uint64_t node, int icomp) const
      { return (_interleaved) ? node *_numComponents + icomp : node + _numGlobalDOF*icomp; }
    inline uint64_t getGlobalDOF(stk::mesh::Entity node, int icomp) const
      { return (_interleaved) ? (_bulkData->identifier(node)-1) *_numComponents + icomp : (_bulkData->identifier(node)-1) + _numGlobalDOF*icomp; }
    int numComponents() const {return _numComponents;}


  private:
    int _numComponents;
    int _numLocalDOF;
    uint64_t _numGlobalDOF;
    bool _interleaved;
  };

}

#endif
