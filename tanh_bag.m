
randn('seed',10);
rand('seed',10);



addpath('/Users/rsalakhu/research/Matlab/minFunc_2012/minFunc');
addpath('/Users/rsalakhu/research/Matlab/minFunc_2012/');
addpath('/Users/rsalakhu/research/Matlab/minFunc_2012/autoDif');
addpath('/Users/rsalakhu/research/Matlab/minFunc_2012/logisticExample');

%%% Test
%nInst = 250;
%nVars = 10;
%order = 1;
%X = randn(nInst,nVars);
%w = randn(nVars,1);
%y = sign(X*w + randn(nInst,1));
%wTest = randn(nVars,1);
%fprintf('Testing gradient using central-differencing...\n');
%derivativeCheck(@LogisticLoss,wTest,order,2,X,y);

T = 100, dd = 4;
nh=5;
w1 = 0.001*randn(dd,nh);
w2 = 0.001*randn(nh,1);

%xx = randn(dd,T);
%xxd = randn(dd,T);
load earth_xy
xx = traj; 
xxd = F; 
T = length(xx); 

cc = max(xx')';
cc2 = max(xxd')';

%xx = xx ./ 1.0e+10; 
%xxd = xxd ./ 1.0e+10; 

%ss = std(xx'); 
%xx = xx ./ (ss'*ones(1,T)); 

w1 = 0.01*randn(dd,nh);
%w1 = 0.000000000001*randn(dd,nh);
w1 = w1 .* (1./cc*ones(1,nh)); 
w2 = 0.1*randn(nh,1);


w1probs = 1./(1 + exp(-w1'*xx));
temp = ((w2*ones(1,T)).*(w1probs.*(1-w1probs)))'*w1' .* xxd'; 
temp1 = sum(temp,2); 
f1_old = sum(temp1.*temp1)

%temp = ((w2*ones(1,T)).*(w1probs.*(1-w1probs))).*(w1'*xxd); 
%f = sum(sum(temp))

%temp = (w1probs.*(1-w1probs)).*(w1'*xxd);
%temp2 = (w2*ones(1,T)).*temp; 
%f3 = sum(sum(temp2))


options = [];
options.display = 'iter';
options.Method = 'scg';
options.maxIter = 100;
%options.maxFunEvals = 10;

VV = [w1(:)' w2(:)']';
Dim = [dd, nh]; 

[X, fX] = minFunc(@CG_tanh_l2,VV,options,Dim,xx,xxd); 
%order=1; 
%derivativeCheck(@CG_tanh_junk,VV,order,2,Dim,xx,xxd);
%x = minFunc(@CG_tanh,[0 0]',options);
  w1 = reshape(X(1:dd*nh),dd,nh);
  xxx = dd*nh;
  w2 = reshape(X(xxx+1:xxx+nh),nh,1);


%VV = [w1(:)' w2(:)']';
%[X, fX] = minFunc(@CG_tanh_junk,VV,options,Dim,xx,xxd);     
%  w1 = reshape(X(1:dd*nh),dd,nh);
%  xxx = dd*nh;
%  w2 = reshape(X(xxx+1:xxx+nh),nh,1);


w1probs = 1./(1 + exp(-w1'*xx));
temp = ((w2*ones(1,T)).*(w1probs.*(1-w1probs)))'*w1' .* xxd';
temp1 = sum(temp,2); 
f1 = sum(temp1.*temp1)

E = w2'*w1probs;  





