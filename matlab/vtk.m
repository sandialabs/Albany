function varargout = vtk (varargin)
% Code for interacting with the tet meshes and solutions saved to VTK files
% by FMDB.
  [varargout{1:nargout}] = feval(varargin{:});
end

function ds = read_vtks (fn_base, nbrs, o)
  iso;
  o = so(o, 'draw', 1);
  ds = {};
  pr('%d:', numel(nbrs));
  for (i = 1:numel(nbrs))
    pr(' %d', i);
    try
      ds{i} = read_vtk_parallel([fn_base '_' num2str(nbrs(i))], o);
    catch
      break;
    end
    if (o.draw) clf; draw(ds(i), struct('skin', 1)); ic; drawnow; end
  end
  pr('\n');
end

function draw (ds, o)
  iso;
  o = so(o, 'lines', 1);
  o = so(o, 'fld', 'xd');
  o = so(o, 'skin', 1);
  o = so(o, 'bb', repmat([-inf inf], 3, 1));
  if (~iscell(ds)) ds = {ds}; end
  pats = '.ovx+*<p';
  clrs = 'grcywmb';
  for (i = 1:numel(ds))
    pat = select(pats, i);
    clr = select(clrs, i);
    c = ds{i}.c;
    if (o.skin) c = get_skin(c); end
    [x y z] = get_lines(ds{i}.(o.fld), c);
    lns = {x, y, z};
    if (any(~isinf(o.bb(:)))) lns = bb_cull_lines(o.bb, lns); end
    set(line(lns{1}, lns{2}, lns{3}), 'color', clr);
  end
  xlabel('x'); ylabel('y'); zlabel('z');
  axis equal; axis tight; hold off; view(3);
end

function draw_projz (ds)
  if (~iscell(ds)) ds = {ds}; end
  pat = '.ovx+*<p'; pat = [pat pat pat];
  clr = 'grcywmb'; clr = [clr clr clr];
  for (i = 1:numel(ds))
    plot(ds{i}.xd(:,1), ds{i}.xd(:,2), [pat(i) clr(i)]);
    hold all;
  end
  axis equal; axis tight; hold off;
end

function myplot3 (d, fld, pat)
  plot3(d.(fld)(:,1), d.(fld)(:,2), d.(fld)(:,3), pat);
end

function h = draw_elems (x, c, eis)
  [x y z] = get_lines(x, c, eis);
  h = line(x, y, z);
  set(h, 'color', 'g');
  xlabel('x'); ylabel('y'); zlabel('z');
end

function eis = get_elems_in_bb (x, c, bb)
  assert(all(size(bb) == [3 2]));
  mx = logical(ones(size(x,1),1));
  for (i = 1:size(x,2)) mx = mx & x(:,i) >= bb(i,1) & x(:,i) <= bb(i,2); end
  eis = get_elems_having_x(size(x, 1), c, find(mx));
end

function eis = get_elems_having_x (nx, c, ix)
  mx = logical(zeros(nx, 1));
  mx(ix) = true;
  m = logical(zeros(size(c,1),1));
  for (i = 1:size(c,2)) m = m | mx(c(:,i)); end
  eis = find(m);
end

% s> private

function d = read_vtk_parallel (fn_base, o)
  [path, fn_base, ~] = fileparts(fn_base);
  if (isempty(path)) path = '.'; end
  fns = dir([path filesep fn_base '_*.vtu']);
  if (isempty(fns))
    % Try w/o _.
    fns = dir([path filesep fn_base '*.vtu']);
  end
  
  for (i = 1:numel(fns))
    di = read_vtk([path filesep fns(i).name], o);
    if (i == 1)
      d = di;
      flds = fieldnames(d);
    else
      for (j = 1:numel(flds))
        f = flds{j};
        d.(f) = [d.(f); di.(f)];
      end
    end
  end
end

function d = read_vtk (fn, o)
  fid = fopen(fn);
  raw = fread(fid);
  fclose(fid);
  raw = char(raw.');
  
  da_beg = strfind(raw, '<DataArray');
  da_end = strfind(raw, '</DataArray');
  da_n = numel(da_beg);
  assert(numel(da_end) == da_n);
  
  d = struct;
  name = '';
  for (i = 1:da_n)
    chunk = raw(da_beg(i) : da_end(i));
    k = strfind(chunk(1:100), 10); k = k(1); % \n
    name = regexp(chunk(1:k), 'Name="([^"]*)"', 'tokens'); name = name{1}{1};
    ndim = regexp(chunk(1:k), 'NumberOfComponents="([^"]*)"', 'tokens');
    have_noc = ~isempty(ndim);
    if (have_noc)
      % This is a field like 'coordinates' or 'solution'.
      ndim = str2num(ndim{1}{1});
      d.(name) = sscanf(chunk(k+1 : end-1), '%f');
      d.(name) = reshape(d.(name), ndim, numel(d.(name))/ndim).';
    else
      % Looking for 'connectivity'.
      if (~strcmp(name, 'connectivity')) continue; end
      %assume All tets, so 4 ints per line.
      ndim = 4;
      d.(name) = sscanf(chunk(k+1 : end-1), '%d');
      d.(name) = reshape(d.(name), ndim, numel(d.(name))/ndim).';
      % Add 1 because of Matlab.
      d.(name) = d.(name) + 1;
    end
  end
  
  % Some shorter names.
  subs = {{'coordinates', 'x'}, {'residual', 'r'}, {'solution', 'd'}, ...
          {'connectivity' 'c'}};
  for (i = 1:numel(subs))
    if (isfield(d, subs{i}{1}))
      d.(subs{i}{2}) = d.(subs{i}{1});
      d = rmfield(d, subs{i}{1});
    end
  end

  % For convenience.
  try
    d.xd = d.x + d.d;
  catch, end
end

function [x y z] = get_lines (v, c, eis)
% v are the vertices. c is the connectivity. eis are optional element indices.
  if (nargin < 3) eis = []; end
  % Lines connect from c1(i) to c2(i).
  switch (size(c, 2))
    case 4 % tets
      is = [1 2 3 1 2 3
            2 3 1 4 4 4];
    case 3 % tris
      is = [1 2 3
            2 3 1];
  end
  [x y z] = get_lines_for_shape(v, c, eis, is);
end

function [x y z] = get_lines_for_shape (v, c, eis, pis)
  assert(size(v, 2) == 3); % 3D points
  assert(size(pis,1) == 2);
  ne = size(pis,2);

  if (isempty(eis)) eis = 1:size(c,1); end
  
  if (0)
    % Lines for all polygons, with redundancy.
    lc = deal(zeros(2, ne*numel(eis)));
    for (k = 1:numel(eis))
      is = ne*(k-1)+1 : ne*k;
      ei = eis(k);
      vl = c(ei,:);
      polygon = c(ei,:);
      lc(:,is) = [polygon(pis(1,:)); polygon(pis(2,:))];
    end
  else
    lc = vtkmex('get_unique_edges', c(eis,:), pis)';
  end
    
  x = [v(lc(1,:),1) v(lc(2,:),1)]';
  y = [v(lc(1,:),2) v(lc(2,:),2)]';
  z = [v(lc(1,:),3) v(lc(2,:),3)]';
end

function stris = get_skin (tets)
% The skin is the set of tris belonging to just one tet.
  stris = vtkmex('get_skin', tets); return;
  % Dumb pure Matlab implementation. The mex alg is better, so there are two
  % sources of speedup.
  ntet = size(tets, 1);
  ntri = 4*ntet;
  % Gather all triangles.
  tris = zeros(ntri, 3);
  for (k = 1:ntet)
    is = 4*(k-1)+1 : 4*k;
    tet = tets(k,:);
    tris(is,:) = tet([1 2 3; 1 2 4; 2 3 4; 3 1 4]);
  end
  % Determine the triangle indices of tris belonging to the skin.
  tris_sorted = sort(tris, 2);
  sis = [];
  for (i = 1:ntri)
    fnd = 0;
    ti = tris_sorted(i,:);
    for (j = [1:i-1, i+1:ntri])
      if (all(ti == tris_sorted(j,:)))
        fnd = 1;
        break;
      end
    end
    if (~fnd) sis(end+1) = i; end
  end
  stris = tris(sis,:);
end

function h = draw_skin (x, stris)
  [x y z] = get_lines(x, stris);
  h = line(x, y, z);
  set(h, 'color', 'g');
  xlabel('x'); ylabel('y'); zlabel('z');
  axis equal; axis tight;
end

function lns = bb_cull_lines (bb, lns)
  keep = logical(ones(1, size(lns{1}, 2)));
  for (i = 1:numel(lns))
    keep = keep & lns{i}(1,:) >= bb(i,1) & lns{i}(2,:) <= bb(i,2);
  end
  for (i = 1:numel(lns))
    lns{i} = lns{i}(:, keep);
  end
end

function s = select (a, i)
  s = a(mod(i-1, numel(a)) + 1);
end

function varargout = pr (varargin)
  fprintf(1, varargin{:});
end
