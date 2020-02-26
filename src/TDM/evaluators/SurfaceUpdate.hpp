//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef _TDM_SURFACEUPDATE_HPP_
#define _TDM_SURFACEUPDATE_HPP_

#include <PCU.h>
#include <pumi.h>
#include <iostream>
#include <vector>
#include <assert.h> 
#include <apf.h>
#include <math.h>
#include <spr.h>
#include <apfMDS.h>
#include <apfConvert.h>
#include <parma.h>

namespace TDM
{
        
          class GroupCode : public Parma_GroupCode {
	    apf::Mesh2* mesh;
	    void run(int){
	      //mesh->writeNative("OutFile.smb");
	    }
	  };

	// define a surface class, congtaining three point (triangle)
	class Surface{
  	public:

	  Surface(double *A, double *B, double *C); // constructor

	  // Accessor
	  double getAx() const;
	  double getAy() const;
	  double getAz() const;
	  double getBx() const;
	  double getBy() const;
	  double getBz() const;
	  double getCx() const;
	  double getCy() const;
	  double getCz() const;

	  double* getAptr() ;
	  double* getBptr() ;
	  double* getCptr() ;

 	 // Print
 	 void print();
	
	private:

	  double A_[3];
	  double B_[3];
	  double C_[3];

	};

	Surface::Surface(double *A, double *B, double *C){
  		for (int i=0; i<3; i++){
	    A_[i]=*(A+i);
	    B_[i]=*(B+i);
	    C_[i]=*(C+i);
	  }
	}

	  double Surface::getAx() const{ return A_[0]; }
	  double Surface::getAy() const{ return A_[1]; }
	  double Surface::getAz() const{ return A_[2]; }
	  double Surface::getBx() const{ return B_[0]; }
	  double Surface::getBy() const{ return B_[1]; }
	  double Surface::getBz() const{ return B_[2]; }
	  double Surface::getCx() const{ return C_[0]; }
	  double Surface::getCy() const{ return C_[1]; }
	  double Surface::getCz() const{ return C_[2]; }

	  double* Surface::getAptr() { return A_; }
	  double* Surface::getBptr() { return B_; }
	  double* Surface::getCptr() { return C_; }

	  void Surface::print(){
	    //std::cout<<"This surface contains ("<<A_[0]<<","<<A_[1]<<","<<A_[2]<<"), ("<<B_[0]<<","<<B_[1]<<","<<B_[2]<<"), ("<<C_[0]<<","<<C_[1]<<","<<C_[2]<<").\n";
	  }	 


	//********************************************************
	// Extra Functions

	// This function switch the order of two nodes
	void exchange(int *first, int *second, int size){   
	  int storage[size];
	  for (int i=0; i<size; i++){
	    storage[i]=*(first+i);
	    *(first+i)=*(second+i);
	    *(second+i)=*(storage+i);
	  }
	}

	  // this fuction sort a [0,1][1,2] into p[0,1,2]
	void sort(int *recorder1, int* recorder2, int*result ){
	  int a,b,c,d;
	  a=*recorder1;
	  b=*(recorder1+1);
	  c=*recorder2;
	  d=*(recorder2+1);
	  //std::cout<<"enter sort loop, abcd is "<<a<<b<<c<<d<<"\n";
	  if( (a==c)|(b==c) ){
	    *result=a;
	    *(result+1)=b;
	    *(result+2)=d;
	  }
	  else if( (a==d)|(b==d) ) {
	    *result=a;
	    *(result+1)=b;
	    *(result+2)=c;
	  }
	  else{
	    std::cerr<<"!!!!Can not sort 3 out of 4 since all four vertices are different\n";
	  }
	  //std::cout<<"after sorting, value is "<< *result <<", " << *(result+1) << ", " <<*(result+2)<<"\n";
	}

//*****************************************************************************************************
// main surface update function
	void SurfaceUpdate( pMesh mesh ){
		std::cout<<"Import mesh successfuly\n";
		//pumi_start();
  		if ( pumi_mesh_getDim( mesh ) == 3) { std::cout << "original mesh file is a 3d mesh\n"; }
  		
//*****************************************************************************************************
// get to the Psi2 elemental field, and recover it to a nodal field using SPR
		apf::MeshIterator* it ;		// loop over mesh element	
  		pMeshEnt e;
		/*
		std::cout<<"I'm "<<PCU_Comm_Self()<<",before spr with "<< mesh->count(3) <<"m esh entities\n";
		int count = 0;
		it = mesh->begin(3);
		while( (e=mesh->iterate(it))){
		  std::cout<<"I'm "<<PCU_Comm_Self()<<", before hand"<<count<<"th element is "<<e<<"\n";
		  count++;
		}
		*/
  		pField PSI2 = mesh -> findField("Psi2");
	 	ALBANY_ASSERT(PSI2, "\nExpected field Psi2 doesn't exist!\n");
		/*
		apf::DynamicArray<apf::MeshTag*> tags;
		mesh->getTags(tags);
		apf::MeshIterator* it = mesh->begin(3);
		std::cout<<"iteration begins\n";
		while (pMeshEnt e =mesh->iterate(it)){
		  void* data;
		  std::cout<<"This is processes "<<PCU_Comm_Self()<<", this element is owned by "<<mesh->getOwner(e) <<"\n";
		  if(PCU_Comm_Self()!=(mesh->getOwner(e))){
		    std::cout<<"rank and owner are different\n";
		  }
		  for(unsigned int i=0; i<tags.getSize(); i++){
		  if( mesh->hasTag(e,tags[i]) ){
		    std::cout<<"before recover, has tag"<< mesh->getTagName(tags[i])<<"\n";
		  }
		  else{std::cout<<"before recover tag not there, name is "<<mesh->getTagName(tags[i])<<"\n";}
		}
		}
		
		std::cout<<"I'm "<<PCU_Comm_Self() <<", Before SPR, I have "<<mesh->count(3)<<" entities\n";
		*/
		pField NodalPSI2 = spr::recoverField(PSI2);
		/*

		// checking field stored in the mesh
		int fieldNum = mesh -> countFields();
		std::cout<<"Inside surface update, number of fields is "<<fieldNum<<"\n";

		for ( unsigned int i=0; i<fieldNum; i++ ){
		    apf::Field* field = mesh -> getField(i);
		    const char * name = getName(field);
		    std::cout<<"Inside surface update, field "<<i<< "has name "<< name << std::endl;
		    std::cout<<"\n";
		}
		*/

  		std::vector<Surface> NewSurface;				// vector that store all new triangular surface element
  		std::vector<Surface>:: iterator SurfacePtr;
  		int NumNew=0;
  		std::vector<pMeshEnt> edges;			// store 6 edges of a element
  		std::vector<pMeshEnt> nodes;			// store 2 nodes of an edge
  		std::vector<pMeshEnt> vertices;   // store 4 vertices of an element
  		std::vector<pMeshEnt> flipEdges;  // store edges that has a flip, used later for order
  		int flipCounter;                  // count how many sign flip for current vertices
  		Vector3 v1,v2,v3,v4;						// 2 Vector3 object to store nodes' coordinate for an edge, then 4 of them store element for qp location
  		double SurfaceNode[4][3];         // store sign flip location before order
  		double SurfaceNodeOrdered[4][3];  // store sign flip location after order
  		int NumFlip;							// counter for # of sign flip in this element
  		double critical=0.95;           // critical value to be checked
  		double *value1;
  		double *value2;           // !!!!!pointers to field values of two nodes
  		double t;                 // interpolation value
  		pMeshEnt New3Vertices[3];             // new vertices if NumFlip=3
  		pMeshEnt New4Vertices[4];             // new vertices if NumFlip=4
  		double ThreeTriNodes[3][3];                 // Save Vector 3 if NumFlip=3
  		double FourTriNodes[4][3];                 

  		double weight[4]={0.25,0.25,0.25,0.25}; // the weighting function for an order 1 tetrahedral 
  		double qp[3];                           // coordinate of the quadrature point
  		double a[3];                            // value of shape function of triangle vertex at integration point;
  		Vector3 x;                              // store values for triangular coordinate shape function calculation
  		Vector3 y;
  		Vector3 z;
  		double Area;
  		double UpdateValue=0;					// container for the value to be updated into the integration point field   
  		apf::Downward adjacent;					// target container for downward adjacent
  		apf::Adjacent UPadjacent;						// target container for up adjacent
  		int numAdj;				

//*****************************************************************************************************
		// surface extraction

		
  		pField DEPTH = mesh -> findField("Depth");
	 	ALBANY_ASSERT(DEPTH, "\nExpected field Depth doesn't exist!\n");
  		pField NodalDEPTH = spr::recoverField(DEPTH);
  		std::cout<<"Nodal DEPTH field is recovered from Elemental Field\n";	 	

		it = mesh->begin(3);
  		while( (e = mesh->iterate(it) ) ){
  			NumFlip = 0;
  			edges.clear();
  			flipEdges.clear();
      		numAdj = mesh->getDownward(e,1,adjacent); 				// get adjacent edges of current mesh element
      		for (int j = 0; j < numAdj; ++j) {
        		edges.push_back(adjacent[j]);
      		}
      		
	  		//std::cout<<"element has "<<edges.size()<<" edges\n";
	  		for (int i=0; i<edges.size(); i++){							// loop over edges
	  			nodes.clear();											// get adjacent nodes of current edge
      			numAdj = mesh->getDownward(edges[i],0,adjacent); 				
      			for (int j = 0; j < numAdj; ++j) {
        			nodes.push_back(adjacent[j]);
      			}
      			double value[2];
		  		//std::cout<<"this edge has "<<nodes.size()<<" nodes\n";
      			apf::getComponents(NodalPSI2, nodes[0], 0, &value[0]);
      			apf::getComponents(NodalPSI2, nodes[1], 0, &value[1]);
      			
      			if (value[0]<value[1]){									// if first vertex has smaller Psi2 value
      				if( value[0]<critical & critical<value[1] ){		// sign flip pattern found
      					flipEdges.push_back(edges[i]);					// put this edge into vectors indicating sign flip edges	
      					NumFlip += 1;									// record number of sign flip within this element, to determin if triangulation		
      					mesh->getPoint( nodes[0], 0, v1);				// save those two vertices 
      					mesh->getPoint( nodes[1], 0, v2);
      					t = (value[1]-critical)/(value[1]-value[0] );	// interpolation index for the zeroth location
      					for (int d=0; d<3;d++){							// calculate xyz coordinate the zeroth location
      						SurfaceNode[NumFlip-1][d]= v2[d]+(v1[d]*t-v2[d]*t);
      					}
          				Vector3 forOut( SurfaceNode[NumFlip-1][0], SurfaceNode[NumFlip-1][1], SurfaceNode[NumFlip-1][2] );
          				//std::cout<<"value[0]="<<value[0]<<", value[1]="<<value[1]<<", v1="<<v1<<", v2="<<v2<<", t="<<t<<", critical node="<<forOut<<"\n";

      				}
      			}

      			else{													 // if first vertex has larger Psi2 value				
      				if ( value[0]>critical & critical>value[1] ){		// if sign flip pattern found
      					flipEdges.push_back(edges[i]);
      					NumFlip += 1;
      					mesh->getPoint( nodes[0], 0, v1);
      					mesh->getPoint( nodes[1], 0, v2);
      					t = (value[0]-critical) / (value[0]-value[1]);
      					for (int d=0; d<3; d++){
      						SurfaceNode[NumFlip-1][d] = v1[d]+(v2[d]*t-v1[d]*t);
      					}
          				Vector3 forOut( SurfaceNode[NumFlip-1][0], SurfaceNode[NumFlip-1][1], SurfaceNode[NumFlip-1][2] );
          				//std::cout<<"value[0]="<<value[0]<<", value[1]="<<value[1]<<", v1="<<v1<<", v2="<<v2<<", t="<<t<<", critical node="<<forOut<<"\n" ;     					
      				}
      			}
	  		}															// finish loop over edges of current mesh elements

	  		if ( 3==NumFlip ){											// if 3 points, then simply create a single element
	  			for (int l=0; l<NumFlip; l++){							// save 3 zeros location into target container
          			ThreeTriNodes[l][0]=SurfaceNode[l][0];
          			ThreeTriNodes[l][1]=SurfaceNode[l][1];
          			ThreeTriNodes[l][2]=SurfaceNode[l][2]; 	  				
	  			}
	  			NewSurface.push_back( Surface( ThreeTriNodes[0], ThreeTriNodes[1], ThreeTriNodes[2] ) );
        		NumNew+=1;
        		//std::cout<<"new single surface triangle added\n";
        		NewSurface[NumNew-1].print();   
	  		}
	  		else if ( 4==NumFlip ){										// if 4 points, then create two single triangle
	  			vertices.clear();
	  			numAdj = mesh->getDownward(e,0,adjacent); 				// get vertices within current element			
      			for (int j = 0; j < numAdj; ++j) {
        			vertices.push_back(adjacent[j]);
      			}
      			int recorder[4][2]={-1,-1,-1,-1,-1,-1,-1,-1};      
      			for ( unsigned int j=0; j<vertices.size(); j++){		// loop over vertices to get their adjacent edges
      				flipCounter = 0;
      				edges.clear();
      				
      				mesh->getAdjacent(vertices[j],1,UPadjacent); 		// get all edges of current node, check if in the sign flip list			
      				for (int k = 0; k<UPadjacent.getSize(); ++k) {
        				edges.push_back(UPadjacent[k]);
      				}

      				for ( unsigned int m=0; (m<NumFlip)&(flipCounter<2) ; m++){
      					for ( unsigned int n=0; n<edges.size(); n++){
      						if ( edges[n]==flipEdges[m] ){				// if this edge is in the sign flip list
      							flipCounter += 1;
      							recorder[j][flipCounter-1] = m;
      						}
      					}
      				}
      			}														// finish loop over vertices within current element

      			int order[2][3] = {0,0,0,0,0,0};
      			if ( (recorder[0][0]==recorder[1][0]) | (recorder[0][0]==recorder[1][1]) | (recorder[0][1]==recorder[1][0]) | (recorder[0][1]==recorder[1][1])  ){  // test if the first two recorder has overlapped vertex
          			sort(recorder[0], recorder[1], order[0]);        	// if overlapped, then thses are correct 3 vertices for a triangle
          			sort(recorder[2], recorder[3], order[1]);
      			}
      			else{
        			exchange(recorder[1],recorder[2],2);                // if not overlapped, change recorder[1] and [2]
        			sort(recorder[0], recorder[1], order[0]); 
        			sort(recorder[2], recorder[3], order[1]);        
      			} 
      			
      			for ( int l=0; l<NumFlip; l++){							// get all four nodes of zeros location							
          			FourTriNodes[l][0]=SurfaceNode[l][0];
          			FourTriNodes[l][1]=SurfaceNode[l][1];
          			FourTriNodes[l][2]=SurfaceNode[l][2]; 
      			}

      			for ( int l=0; l<3; l++){     // using 3 of the above 4 point, namely 123, create a new surface
          			ThreeTriNodes[l][0]=FourTriNodes[ order[0][l] ][0];
          			ThreeTriNodes[l][1]=FourTriNodes[ order[0][l] ][1];
          			ThreeTriNodes[l][2]=FourTriNodes[ order[0][l] ][2];
      			}
      			NewSurface.push_back( Surface( ThreeTriNodes[0], ThreeTriNodes[1], ThreeTriNodes[2] ) );
        		NumNew+=1;
        		//std::cout<<"new 1 of 2 surface triangle added\n";
        		NewSurface[NumNew-1].print(); 

       			for ( int l=0; l<3; l++){     // using 3 of the above 4 point, namely 134, create another new surface 
          			ThreeTriNodes[l][0]=FourTriNodes[ order[1][l] ][0];
          			ThreeTriNodes[l][1]=FourTriNodes[ order[1][l] ][1];
          			ThreeTriNodes[l][2]=FourTriNodes[ order[1][l] ][2];
      			}

        		NewSurface.push_back( Surface( ThreeTriNodes[0], ThreeTriNodes[1], ThreeTriNodes[2] ) );
        		NumNew+=1;
        		//std::cout<<"new 2 of 2 surface triangle added\n";
        		NewSurface[NumNew-1].print(); 
	  		}
	  		
  		}

  		mesh->end(it);
  		std::cout<<"Surface extraction finished\n";
			
//*****************************************************************************************************

		// Depth interogation
  		it = mesh->begin(3);												// loop over elements to get its 1st order quadrature point
  		while (e=mesh->iterate(it)){
		  //std::cout<<"Enter a new mesh for integrrogation\n";
  			nodes.clear();
  			int NumProjection = 0;
      		numAdj = mesh->getDownward(e,0,adjacent); 						// get 4 nodes of current mesh element to determine quadrature point
      		for (int j = 0; j < numAdj; ++j) {
        		nodes.push_back(adjacent[j]);
      		}
      		mesh -> getPoint_( nodes[0], 0, v1);
      		mesh -> getPoint_( nodes[1], 0, v2);
      		mesh -> getPoint_( nodes[2], 0, v3);
      		mesh -> getPoint_( nodes[3], 0, v4);

    		for (int i=0; i<3; i++){                              			// calculate x,y,z of the quadrature point
      		qp[i]= weight[0]*v1[i]+weight[1]*v2[i]+weight[2]*v3[i]+weight[3]*v4[i];
      		//std::cout<<"Enter new mesh, equation is "<< weight[0]<< "x" <<v1[i]<< "+"<< weight[1]<< "x" <<v2[i]<< "+" <<weight[2]<< "x" <<v3[i]<< "+" <<weight[3]<< "x" <<v4[i]<<"\n";
    		}
		//std::cout<<"Get QP point finished, NewSurface.size()="<<NewSurface.size() <<"\n";

    		UpdateValue = 0;
		
    		// Loop over 2d triangular elements, determine projection and calculate weight
    		for (unsigned int j=0; (j<NewSurface.size())&(UpdateValue==0); j++){
		  //std::cout<<"into loop over surface elements\n";
    			for (unsigned int k=0; k<3; k++){							// get the correct information of current surface element, x y z indicates three vertices while 0 1 2 indicates three components 
        			x[k]=*( NewSurface[j].getAptr()+k );
        			y[k]=*( NewSurface[j].getBptr()+k );
        			z[k]=*( NewSurface[j].getCptr()+k );    				
    			}
    			Vector3 xy=x-y;
    			Vector3 yz=z-y;
    			xy[2]=0;
    			yz[2]=0;													// assume implicit 2d surface element
    			Vector3 Cross = apf::cross( xy, yz );
    			//std::cout<<"x is "<<x<<" y is "<<y <<"z is" <<z<<"xy and yz are"<<xy<<yz<<"crosss is "<<Cross<<"\n";
      			Area= fabs( Cross[2]) / 2;
      			a[0]=fabs( ( ( y[0]*z[1] - z[0]*y[1] ) + ( y[1]-z[1] )*qp[0] + ( z[0]-y[0] )*qp[1] )/2/Area ) ;
      			a[1]=fabs( ( ( z[0]*x[1] - x[0]*z[1] ) + ( z[1]-x[1] )*qp[0] + ( x[0]-z[0] )*qp[1] )/2/Area );
      			a[2]=fabs( ( ( x[0]*y[1] - y[0]*x[1] ) + ( x[1]-y[1] )*qp[0] + ( y[0]-x[0] )*qp[1] )/2/Area );
      			//std::cout<<"For this element, with respect to "<<j<<"th NewSurface, coefficients are "<< a[0] <<", " << a[1] <<", " << a[2] <<", area is "<< Area << "\n";
			
      			if ( (0<a[0]) & (a[0]<1) & (0<a[1]) & (a[1]<1) & (0<a[2]) & (a[2]<1) & ( (a[0]+a[1]+a[2] ) <1.01 ) & ( 0.99<(a[0]+a[1]+a[2] ) )  ){  // if 0<a1<1, 0<a2<1, 0<a3<1, and they sum up less than 1.01 (tolerance), then the qp point falls into this NewSurface[j]
			        UpdateValue = qp[2] - (a[0]*x[2] + a[1]*y[2] + a[2]*z[2]);// new depth value
        			
        			if (UpdateValue>0){										// if the element is below the surface then continue as normal
				  double depth_old;
				  apf::getComponents( DEPTH, e, 0, &depth_old);
				        apf::setComponents( DEPTH, e, 0, &UpdateValue);
					Vector3 forOut(qp[0],qp[1],qp[2]);
					double printout;
					apf::getComponents( DEPTH, e, 0, &printout);
					//apf::MeshTag* tag = mesh -> findTag("global_id");
					//int global_id;
					//mesh->getIntTag(e,tag,&global_id);
          				//std::cout<<"For this elemen, QP is "<<forOut<<",depth_old is"<< depth_old <<",UpdateValue is "<<UpdateValue <<", Element is above so depth is updated to be "<<printout<<"\n";
          				NumProjection+=1;
   					 	ALBANY_ASSERT( (NumProjection<2) , "!Multiple projection found in this element\n");
		        	//std::cout<<"Updated value is"<<UpdateValue<<"\n";
          			}

          			else{
          				UpdateValue = -1;									// if element is above the surface, then set the Depth to be -1
        				double depth_old;
					apf::getComponents( DEPTH, e, 0, &depth_old);
					apf::setComponents( DEPTH, e, 0, &UpdateValue);
					Vector3 forOut(qp[0],qp[1],qp[2]);
					double printout;
					apf::getComponents( DEPTH, e, 0, &printout); 
					//apf::MeshTag* tag = mesh -> findTag("global_id");
					//int global_id;
					//mesh->getIntTag(e,tag,&global_id);
          				//std::cout<<"For this elemen, QP is "<<forOut<<",depth_old is"<< depth_old <<",UpdateValue is "<<UpdateValue <<", Element is above so depth is updated to be "<<printout<<"\n";
          				NumProjection+=1;
   					 	ALBANY_ASSERT( (NumProjection<2) , "!Multiple projection found in this element\n");         				
          			}
        		}
			
    		} 
  		}	// end of looping over elements
  		mesh->end(it);
  		std::cout<<"Depth interogation finished\n";
	      		
		//		pumi_field_print (DEPTH);
  		
  		//apf::writeASCIIVtkFiles("SurfaceOut",mesh);
				
		apf::destroyField(NodalPSI2);
  		apf::destroyField(NodalDEPTH);
		GroupCode code;
		//apf::Unmodulo outMap(PCU_Comm_Self(), PCU_Comm_Peers());
		//apf::writeASCIIVtkFiles("BeforeShrink",mesh);
		//Parma_ShrinkPartition(mesh, PCU_Comm_Peers(), code);
		//apf::writeASCIIVtkFiles("AfterShrink",mesh);
		/*
		std::cout<<"I'm "<<PCU_Comm_Self()<<",after repartition with "<< mesh->count(3) <<"m esh entities\n";		
		count = 0;
		it = mesh->begin(3);
		while( (e=mesh->iterate(it))){
		  std::cout<<"I'm "<<PCU_Comm_Self()<<", after hand"<<count<<"th element is "<<e<<"\n";
		  count++;
		}
		*/
  		std::cout<<"Surface update has been finished\n";

	}




}

#endif
