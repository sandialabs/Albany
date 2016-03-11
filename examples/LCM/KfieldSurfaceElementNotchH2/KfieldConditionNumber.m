% grab Irina's mmread function (thank you)
addpath('~/LCM/Albany/matlab')

% read in tangents from Albany and AlbanyT
albany_tangent = mmread('jac0Epetra.mm');
albanyT_tangent = mmread('jac0Tpetra.mm');

% find the largest and smallest eigenvalues of the sparse matrices
albany_large_eigs = eigs(albany_tangent)
albanyT_large_eigs = eigs(albanyT_tangent)
albany_small_eigs = eigs(albany_tangent,6,'sm')
albanyT_small_eigs = eigs(albanyT_tangent,6,'sm')

% calculate the condition numbers
albany_condition = max(albany_large_eigs)/min(albany_small_eigs)
albanyT_condition = max(albanyT_large_eigs)/min(albanyT_small_eigs)





