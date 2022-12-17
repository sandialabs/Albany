function [field]=extend_field(field, n)

f = zeros(size(field)+2); I = zeros(size(field)+2);

%sigma = 4;
%h = d2gauss(1+4*ceil(sigma),sigma);
%thk = filter2(h,thk);

for i=1:n
  f(2:end-1,2:end-1) = field;
  mask = field~=0;
  I(2:end-1,2:end-1) = mask;
  %fext = filter2(h,f);%./(filter2(h,I)+eps);
  
  fext =  (f(1:end-2, 2:end-1)+f(3:end, 2:end-1)+f(2:end-1, 1:end-2)+f(2:end-1,3:end) ...
      + (f(1:end-2, 1:end-2)+f(3:end, 1:end-2)+f(1:end-2, 3:end)+f(3:end,3:end))/sqrt(2))./...
      (I(1:end-2, 2:end-1)+I(3:end, 2:end-1)+I(2:end-1, 1:end-2)+I(2:end-1,3:end)+eps...
      +(I(1:end-2, 1:end-2)+I(3:end, 1:end-2)+I(1:end-2, 3:end)+I(3:end,3:end))/sqrt(2));
  %fext = filter2(h,f);
  field(~mask) = fext(~mask);
  %field(mask)=0;
end  