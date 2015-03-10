function o = so (o, fld, def_val, chk_fn)
% Struct option parser.
  if (nargin < 4) chk_fn = []; end
  if (nargin < 3) def_val = []; end
  if (~isfield(o, fld)) o.(fld) = def_val; end
  if (~isempty(chk_fn))
    if (~chk_fn(o.(fld)))
      error(sprintf('Field ''%s'' with value ''%s'' failed chk_fn ''%s''.', ...
                    fld, varstr(o.(fld)), varstr(chk_fn)));
    end
  end
end
