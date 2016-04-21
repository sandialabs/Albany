close all; 
clear all; 
addpath('../');


lapl = mmread('laplaceSW.mm');  
lh = mmread('laplaceH.mm');
N = 294/3; %for ne6
nn=N;


%reshape Mass and Laplace to make them block 
%M = sparse(3*N, 3*N); 
Lsw = sparse(3*N, 3*N);
Lh = sparse(7*N, 7*N);

%M(1:nn, 1:nn)               = mass(1:3:end,1:3:end);
%M(nn+1:2*nn, nn+1:2*nn)     = mass(2:3:end,2:3:end);
%M(2*nn+1:3*nn, 2*nn+1:3*nn) = mass(3:3:end,3:3:end);

%M(1:nn, nn+1:2*nn)           = mass(1:3:end,2:3:end); 
%M(1:nn, 2*nn+1:3*nn)         = mass(1:3:end,3:3:end); 

%M(nn+1:2*nn, 1:nn)           = mass(2:3:end,1:3:end); 
%M(nn+1:2*nn, 2*nn+1:3*nn)    = mass(2:3:end,3:3:end); 

%M(2*nn+1:3*nn, 1:nn)         = mass(3:3:end,1:3:end); 
%M(2*nn+1:3*nn, nn+1:2*nn)    = mass(3:3:end,2:3:end); 


%%%%%%%%%%%%%%%%%%% sorting SW laplace
Lsw(1:nn, 1:nn)               = lapl(1:3:end,1:3:end);
Lsw(nn+1:2*nn, nn+1:2*nn)     = lapl(2:3:end,2:3:end);
Lsw(2*nn+1:3*nn, 2*nn+1:3*nn) = lapl(3:3:end,3:3:end);

Lsw(1:nn, nn+1:2*nn)           = lapl(1:3:end,2:3:end); 
Lsw(1:nn, 2*nn+1:3*nn)         = lapl(1:3:end,3:3:end); 

Lsw(nn+1:2*nn, 1:nn)           = lapl(2:3:end,1:3:end); 
Lsw(nn+1:2*nn, 2*nn+1:3*nn)    = lapl(2:3:end,3:3:end); 

Lsw(2*nn+1:3*nn, 1:nn)         = lapl(3:3:end,1:3:end); 
Lsw(2*nn+1:3*nn, nn+1:2*nn)    = lapl(3:3:end,2:3:end);



%%%%%%%%%%%%%%%%%%% sorting H laplace
r1 = [1:nn];
r2 = [nn+1:2*nn];
r3 = [2*nn+1:3*nn];
r4 = [3*nn+1:4*nn];
r5 = [4*nn+1:5*nn];
r6 = [5*nn+1:6*nn];
r7 = [6*nn+1:7*nn];

d1 = [1:7:7*nn];
d2 = [2:7:7*nn];
d3 = [3:7:7*nn];
d4 = [4:7:7*nn];
d5 = [5:7:7*nn];
d6 = [6:7:7*nn];
d7 = [7:7:7*nn];
Lh(r1,r1) = lh(1:7:end,1:7:end);
Lh(r2,r2) = lh(2:7:end,2:7:end);
Lh(r3,r3) = lh(3:7:end,3:7:end);
Lh(r4,r4) = lh(4:7:end,4:7:end);
Lh(r5,r5) = lh(5:7:end,5:7:end);
Lh(r6,r6) = lh(6:7:end,6:7:end);
Lh(r7,r7) = lh(7:7:end,7:7:end);

%%%%row 1
Lh(r1,r2) = lh(d1,d2);
Lh(r1,r3) = lh(d1,d3);
Lh(r1,r4) = lh(d1,d4);
Lh(r1,r5) = lh(d1,d5);
Lh(r1,r6) = lh(d1,d6);
Lh(r1,r7) = lh(d1,d7);
%%%%%row 2
Lh(r2,r1) = lh(d2,d1);
Lh(r2,r3) = lh(d2,d3);
Lh(r2,r4) = lh(d2,d4);
Lh(r2,r5) = lh(d2,d5);
Lh(r2,r6) = lh(d2,d6);
Lh(r2,r7) = lh(d2,d7);
%%%%% row3
Lh(r3,r1) = lh(d3,d1);
Lh(r3,r2) = lh(d3,d2);
Lh(r3,r4) = lh(d3,d4);
Lh(r3,r5) = lh(d3,d5);
Lh(r3,r6) = lh(d3,d6);
Lh(r3,r7) = lh(d3,d7);
%%%%% row4
Lh(r4,r2) = lh(d4,d2);
Lh(r4,r3) = lh(d4,d3);
Lh(r4,r1) = lh(d4,d1);
Lh(r4,r5) = lh(d4,d5);
Lh(r4,r6) = lh(d4,d6);
Lh(r4,r7) = lh(d4,d7);
%%% row 5
Lh(r5,r2) = lh(d5,d2);
Lh(r5,r3) = lh(d5,d3);
Lh(r5,r4) = lh(d5,d4);
Lh(r5,r1) = lh(d5,d1);
Lh(r5,r6) = lh(d5,d6);
Lh(r5,r7) = lh(d5,d7);
%%%row 6
Lh(r6,r2) = lh(d6,d2);
Lh(r6,r3) = lh(d6,d3);
Lh(r6,r4) = lh(d6,d4);
Lh(r6,r5) = lh(d6,d5);
Lh(r6,r1) = lh(d6,d1);
Lh(r6,r7) = lh(d6,d7);
%%%% row 7
Lh(r7,r2) = lh(d7,d2);
Lh(r7,r3) = lh(d7,d3);
Lh(r7,r4) = lh(d7,d4);
Lh(r7,r5) = lh(d7,d5);
Lh(r7,r6) = lh(d7,d6);
Lh(r7,r1) = lh(d7,d1);

max(max(abs(Lh - Lh')))
%%%%%%%
%%%%% Lh is ordered this way: ps, u1,v1, T1, u2, v2, T2
%%%% so the block for ps should be zero
%%%% u1v1 should be the same as u2v2 block and second block in SW laplace


Lscalar = Lsw(1:nn,1:nn);
Lvec = Lsw(nn+1:end,nn+1:end);

Lhps = Lh(1:nn,1:nn);
Lhuv1 = Lh(nn+1:3*nn,nn+1:3*nn);
Lht1 = Lh(3*nn+1:4*nn,3*nn+1:4*nn);
Lhuv2 = Lh(4*nn+1:6*nn,4*nn+1:6*nn);
Lht2 = Lh(6*nn+1:end,6*nn+1:end);


disp('verifying hydrostatic');

max(max(abs(zeros(nn) - Lhps)))
max(max(abs(Lscalar - Lht1)))
max(max(abs(Lscalar - Lht2)))

max(max(abs(Lvec - Lhuv1)))
max(max(abs(Lvec - Lhuv2)))



%max(max(    abs(Lsw-Lsw')    ))
%L11 = Lsw(1:nn, 1:nn);
%max(max(    abs(L11-L11')    ))




aa = 1;











