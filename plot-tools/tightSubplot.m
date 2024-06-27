function varargout = tightSubplot(nrows,ncols,N,dx,dy,pads,xsize,ysize)
% [ax] = tightSubplot(nrows,ncols,N,dx,dy,pads,xsize,ysize);
%   Customized subplotting routine for cleaner/tighter/highly customizable
%   axes compared to a standard MATLAB subplot.
%   - Basic functionality is largely the same as MATLAB's built in subplot,
%   however this function makes it very easy to precisely control axes
%   locations, e.g. for making print-quality plots. This replicates some of
%   the more customizable behaviour in some python plotting libraries, for
%   example.
%   - The default behaviour here will make tightly paneled plots with no 
%   spacing between (good for tiled plots). Most often, you will specify at
%   least one of the optional arguments to set axes spacing/location.
%
% REQUIRED ARGS:
%           nrows = Number of panel rows (same as in subplot)
%           ncols = Number of panel columns (same as in subplot)
%           N     = Plot number (same as in subplot)
% OPTIONAL ARGS:
%           dx    = normalized x spacing between axes, 0 to 1 [default = 0]
%           dy    = normalized y spacing between axes, 0 to 1 [default = 0]
%           pads  = normalized (to figure size) border space vector 
%                   [left right bottom top] --> Default: [0.11 0.04 0.1 0.06]
%           xsize = vector that generates non-uniform axes widths,
%                   normalized to each other
%           ysize = vector that generates non-uniform axes heights,
%                   normalized to each other
%      
%           Setting any optional argument to 0 will turn off
%           ticks/labels on the corresponding axis. 
%           Eg 1a: dx = 0 will turn off axis labels and tick labels for the y
%           axis, EXCEPT for the left most axis, since the plots will have
%           no spacing between them.
%           Eg 1b: pads = [0 0.1 0.1 0.1]  would ALSO turn off the left most
%           axis, since you are leaving no border space on the left side of
%           the figure.
%   
%           Eg 2. xsize = [2 1] will tile columns of subplots with RELATIVE
%           width ratios of 2 to 1, after filling space allotted by pads and dx.
%           
%               tightSubplot(4,2,N,[],[],[],[2 1]) for N = 1:8 will ;00[                                     create
%           a plot that looks like this:
%                   
%                       |     |   |     |
%                       |     |   |     |
%                       |---- |-- |---- |--
%
%                       |     |   |     |
%                       |     |   |     |
%                       |---- |-- |---- |--
%
%   OUTPUT:  ax = axes handle
%
% C Rowell, 
%   - V1.0 Jun 2017
%   - V1.1 Adjusted dx,dy defaults to 0, since other default behaviors are
%   easily handled by standard subplot.
%
%   
%     NOTES:  Might implement variable dx,dy spacing, adaptive pads based
%     on aspect ratio
%% Set defaults - make adjustments here according to your taste
% MATLAB DEFAULTS
% pads_def = [0.13 0.095 0.11 0.075]; % Matlab defaults
% dx_X_ratio = 0.3158; % Ratio of horizontal subplot spacing to axes width
% dx_X_ratio = 0.3889; % Ratio of vertical subplot spacing to axes height
nargoutchk(0,1)

% My defaults
pads_def = [0.11 0.04 0.1 0.06];
dx_X_ratio = 0; %0.25; % Changed these to 0 for now, makes more sense as the default here
dx_Y_ratio = 0; %0.3;
% Aspect ratio thresholds - experimental
aspR_hi = 1.25; % If higher, use xpads as ypads
aspR_lo = 0.5;  % If lower, use ypads as xpads

%% Parse input
if nargin<4
    dx = [];
end
if nargin<5
    dy = [];
end
if nargin<6
    pads = [];
end
if nargin<7
    xsize = [];
end
if nargin<8
    ysize = [];
end

%% Get aspect ratio, calculate spacings
% This will initiate a figure if one does not exist
figpos = get(gcf,'position');
aspRatio = figpos(4)/figpos(3);

% Assign pads if necessary
if isempty(pads)
    pads = pads_def;
end

%% Calc spacings where necessary
% Number of spaces needed
ndx = ncols-1;
ndy = nrows-1;

% Calc total space available, and average space used by axes
x_space = 1-sum(pads(1:2));
y_space = 1-sum(pads(3:4));

if isempty(dx)
    dx = (x_space)/ncols*dx_X_ratio; % Calc dx assuming evenly sized axes
end
if isempty(dy)
    dy = (y_space)/nrows*dx_Y_ratio; % Calc dy assuming evenly sized axes
end

%% Calc axis size and position based on pads,dx,dy, and size vectors
% Assign current axes position from index
iy = ceil(N/ncols);
ix = N - (iy-1)*ncols;

% Create normalized length vector for axes
if ~isempty(xsize)
    xsize = repmat(xsize,[1 ceil(ncols/numel(xsize))]);
    xsize = xsize(1:ncols)/sum(xsize(1:ncols))*(x_space-ndx*dx);
else
    xsize = ones(1,ncols)*((x_space - (ndx)*dx)/ncols);
end

if ~isempty(ysize)
    ysize = repmat(ysize,[1 ceil(nrows/numel(ysize))]);
    ysize = ysize(1:nrows)/sum(ysize(1:nrows))*(y_space-ndy*dy);
else
    ysize = ones(1,nrows)*((y_space - (nrows-1)*dy)/nrows);
end

% Calc position vector
xw = xsize(ix); %(x_space - (ncols-1)*dx)/ncols; % Axes width
yh = ysize(iy); %(y_space - (nrows-1)*dy)/nrows; % Axes height
x = pads(1) + (ix-1)*dx + sum(xsize(1:ix-1));
y = pads(3)+y_space-(iy-1)*dy-sum(ysize(1:iy));   % pads(3) + (iy-1)*(yh+dy);
pos = [x y xw yh];

%% Add axes
ax=axes('position',pos);
% Text to check position as a useful trick
% text(0.5,0.5,sprintf('%i, %i',ix,iy))
ax_stat = get(gca);

%% Turn off/adjust axes labels when spacing is 0
if dx==0
    box on
    if ix>1
        set(ax,'YTickLabel',[])
    end
    if ix~=ncols && ax_stat.XTick(end)==ax_stat.XLim(2);
        set(ax,'XTick',ax_stat.XTick(1:end-1))
    end
end
if dy==0
    box on
    if iy<nrows
        set(ax,'XTickLabel',[])
    end
    if iy>1
        set(ax,'YTick',ax_stat.YTick(1:end-1))
    end
end

if nargout==1
    varargout{1} = ax;
end
