function [handles,sVals] = getScatterSizeLegend(mSize,inHandle,sizeVals,ax)
% [handles,sVals] = getScatterSizeLegend(mSize,inHandle,sizeVals,ax)
% ARGS:
%   mSize    = marker size plot vector
%   sizeVals = optional actual size values to label points with (otherwise
%               assumed equal to mSize)
%   inHandle = handle of actual scatter plot
%   ax       = optional axes handle of scatter plot
%
% OUTPUT:
%   handles = handles of dummy plots that match desired legend properties
%   sVals   = size values used to match legend (3 are assumed)

if nargin<4
    ax = gca;
end
if nargin<3 || isempty(sizeVals)
    sizeVals = mSize;
end
    
    lcol = [0.5 0.5 0.5]; % Default replacement color for color scaled points

    mSz = [min(mSize) mean([min(mSize) max(mSize)]) max(mSize)]; % Get low, mid, high size range
    sVals = [min(sizeVals) mean([min(sizeVals) max(sizeVals)]) max(sizeVals)];
    
    matchfields = {'MarkerEdgeColor','MarkerFaceColor','LineWidth'};

    for ii=1:3
%         handles(ii) = scatter(ax,nan,nan,mSz(ii),[0.5 0.5 0.5]);    
        handles(ii) = plot(ax,nan,nan,'Marker',inHandle.Marker,'MarkerSize',sqrt(mSz(ii)),'LineStyle','none');    
%         handles(ii) = copyobj(inHandle,ax);
%         set(handles(ii),'XData', NaN', 'YData', NaN,'CData',[0.5 0.5 0.5],'SizeData',mSz(ii))
%         handles(ii).MarkerSize = mSz(ii);

        if strcmp(inHandle.MarkerEdgeColor,'flat')
            handles(ii).MarkerEdgeColor = lcol; % Not bothering with alpha for now
        else
            handles(ii).MarkerEdgeColor = inHandle.MarkerEdgeColor;
        end
        if strcmp(inHandle.MarkerFaceColor,'flat')
            handles(ii).MarkerFaceColor = lcol; % Not bothering with alpha for now
        else
            handles(ii).MarkerFaceColor = inHandle.MarkerFaceColor;
        end
        handles(ii).LineWidth = inHandle.LineWidth;
    end

%     fn = fieldnames(inHandle);
%     matchfields = {'Marker','Line'};
%     
%     % Match visual properties except color and size
%     for fi = 1:length(fn)
%         
%         if contains(fn{fi},'Color') && strcmp(inHandle.(fn{fi}),'flat')
%             handles(1).(fn{fi}) = [0.5 0.5 0.5];
%             handles(2).(fn{fi}) = [0.5 0.5 0.5];
%             handles(3).(fn{fi}) = [0.5 0.5 0.5];
%         elseif contains(fn{fi},matchfields)
%             handles(1).(fn{fi}) = inHandle.(fn{fi});
%             handles(2).(fn{fi}) = inHandle.(fn{fi});
%             handles(3).(fn{fi}) = inHandle.(fn{fi});
%         end
%     end
    
%     [~,b] = legend('foo','bar','baz')
%     set(findobj(b,'-property','MarkerSize'),'MarkerSize',10)
    

end