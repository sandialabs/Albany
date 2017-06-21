NAME = notch2D

all: $(NAME).smb

$(NAME).geo:
	mpicxx -std=c++11 $(NAME).cpp -o $(NAME) -I${GMODEL_DIR}/include -L${GMODEL_DIR}/lib -lgmodel
	./$(NAME)

$(NAME).msh: $(NAME).geo
	gmsh -2 $(NAME).geo -o $(NAME).msh	

$(NAME)-serial.smb: $(NAME).msh
	from_gmsh $(NAME).dmg $(NAME).msh $(NAME)-serial.smb

$(NAME).smb: $(NAME)-serial.smb
	mpirun -n 4 split $(NAME).dmg $(NAME)-serial.smb $(NAME).smb 4

clean:
	rm -rf $(NAME) $(NAME).geo $(NAME).dmg $(NAME).msh $(NAME)*.smb out*
