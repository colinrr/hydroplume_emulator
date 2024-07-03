function [dataTable,scaledDataTable,fixedVars, MCvars] = getMCoutputTable(mainOutputFile, summaryOutputFile, dat, QScale, makePlots, printPlots)
%  mainOutputFile = saved MAIN output file with "dat" struct
%  summaryOutputFile = saved summary file with "qA" struct
%  Qscale = 'log' or 'linear'. Generally defaults to 'log', unless MC input
%           used a linear Q

    if nargin<3 || isempty(dat)
        load(mainOutputFile,'MC','cI','pI','randPars','dat');
    else
        load(mainOutputFile,'MC','cI','pI','randPars');
    end
    load(summaryOutputFile,'qA');
    
    if nargin<5 || isempty(makePlots)
        makePlots = false;
    end

    if nargin<4 || isempty(QScale)
        if ismember('logQ',fieldnames(MC.cI))
            QScale = 'log';
        elseif ismember('Q',fieldnames(MC.cI)) && ismember(MC.cI.Q.dist,{'uniform','normal'})
            QScale = 'linear';
        else
            QScale = 'log';
        end
    end
    
    if nargin<6 || isempty(printPlots)
        printPlots = false;
    end
    if printPlots
        fs = 9;
        mSizeScale = 0.1;
    else
        fs = 15;
        mSizeScale = 1;
    end
    
%% A Few tuned/tunable user params to consider for inputs
    
  
Ze_over_Rv_min      = 3.5; % Min value of Ze/conduit radius to use for steam plume scaling filter
    
% KNN SEARCH KEY INPUTS
% distance_weights and n_neighbours could be a more clever function of
% variances between Q and Ze, but performance is good for now
n_neighbours        = 50;  
distance_weights    = [0.3; 1]; % Q vs Z weights for chi-square nearest neighbour distance metric
regime_weights      = [0.45 0.45 0.2]; % Relative weights among regimes


%% Get summary vars
% Get fixed I/O vars for Monte Carlo run
    fn = fieldnames(pI);
    for fi = 1:length(fn)
        fixedVars.(fn{fi}) = pI.(fn{fi}); 
    end
    fn = fieldnames(cI);
    for fi = 1:length(fn)
        fixedVars.(fn{fi}) = cI.(fn{fi}); 
    end

    % Get summary MC input variables
    MCvarnames = [fieldnames(MC.cI); fieldnames(MC.pI)];
    nMC = length(MCvarnames);
    vardist = cell(nMC,1);
    varval1 = zeros(nMC,1);
    varval2 = zeros(nMC,1);
    fnci = fieldnames(MC.cI);
    nci  = length(fnci);
    for fi = 1:nci
        varval1(fi) = MC.cI.(fnci{fi}).range(1);
        varval2(fi) = MC.cI.(fnci{fi}).range(2);
        vardist{fi} = MC.cI.(fnci{fi}).dist;
    end
    fnpi = fieldnames(MC.pI);
    for fi = 1:length(fnpi)
        varval1(fi+nci) = MC.pI.(fnpi{fi}).range(1);
        varval2(fi+nci) = MC.pI.(fnpi{fi}).range(2);
        vardist{fi+nci} = MC.pI.(fnpi{fi}).dist;
    end

    MCvars = table(string(MCvarnames),varval1,varval2,vardist,...
        'VariableNames',{'Variable','Value 1','Value 2','Distribution Type'});


%% Get main model outputs - NOT ACTUALLY CLEAR IF I NEED MOST OF THESE...

    u_crit = 15;  % Critical minimum velocity at which to define collapse radius

    qkeep = find(qA.QClevel(:,1)~=0); % Cut out failed runs, can be used to filter more later if needed
    nn = length(qkeep);    
    rawDataTable = randPars(qkeep,:); % Get main input table, exclude failed runs (generally all at extreme low conduit radius)
    rawDataTable = renamevars(rawDataTable,'Zw','Ze'); % Renaming for consistency with original paper
    
    plumeFlux.QClevel = qA.QClevel(qkeep,:);
    plumeFlux.clps = logical(qA.pO.collapse(qkeep));  % Collapse
    plumeFlux.Qp    = qA.cI.Q(qkeep);
    plumeFlux.Ze    = qA.cI.Zw(qkeep);
    plumeFlux.u_0   = qA.pI.u_0(qkeep);
    plumeFlux.r_v   = qA.cI.conduit_radius(qkeep);
    plumeFlux.r_p   = qA.pI.r_0(qkeep);
    plumeFlux.rhop  = qA.pI.rho_B0(qkeep);
    plumeFlux.hm    = qA.pO.hm(qkeep) + qA.cI.vh0(qkeep); % Max height
    plumeFlux.hb    = qA.pO.hb(qkeep) + qA.cI.vh0(qkeep); % LNB height
    plumeFlux.zC   = zeros(nn,1); % Collapse height
    plumeFlux.rC   = zeros(nn,1); % Collapse radius
    plumeFlux.qs0  = zeros(nn,1); % starting subaerial particle flux
    plumeFlux.qw0  = zeros(nn,1); % starting subaerial water flux
    plumeFlux.qsC  = zeros(nn,1); % collapse particle flux
    plumeFlux.qwC  = zeros(nn,1); % collapse water flux


    for ii=1:nn
            zeFilt = qkeep(ii);
            plumeFlux.clps  = logical(qA.pO.collapse(qkeep));

            if qA.QClevel(zeFilt,2) == 1 % Water breach successful
                plumeFlux.qs0(ii) = dat(zeFilt).pO.m_s(1);
                plumeFlux.qw0(ii) = dat(zeFilt).pO.m_l(1) + dat(zeFilt).pO.m_v(1) - dat(zeFilt).cI.Q.*dat(zeFilt).cI.n_0; % Discount magmatic water
%                 if plumeFlux.clps(ii)
                   ri = find( (dat(zeFilt).pO.u.*sin(dat(zeFilt).pO.angle)) <= u_crit,1,'first');
                   plumeFlux.zC(ii) = dat(zeFilt).pO.z(ri);  % Height at collapse. 
                   plumeFlux.rC(ii) = dat(zeFilt).pO.r(ri);

                   plumeFlux.qsC(ii) = dat(zeFilt).pO.m_s(ri);
                   plumeFlux.qwC(ii) = dat(zeFilt).pO.m_l(ri) + dat(zeFilt).pO.m_v(ri);
%                 end
            else
                % Add magmatic water for steam plume cases, but ignore n_ec
                % as assumed to come from cauldron
                plumeFlux.qw0(ii) =  - dat(zeFilt).cI.Q.*dat(zeFilt).cI.n_0; 
            end
    end

    if ismember('Q',fieldnames(MC.cI))
        plumeFlux.Qmax  = max(MC.cI.Q.range); 
        plumeFlux.Qmin  = min(MC.cI.Q.range);
    elseif ismember('logQ',fieldnames(MC.cI))
        plumeFlux.Qmax  = max(10.^MC.cI.logQ.range); 
        plumeFlux.Qmin  = min(10.^MC.cI.logQ.range);
    end
    plumeFlux.Zemax = max(MC.cI.Zw.range);
    plumeFlux.Zemin = min(MC.cI.Zw.range);
    plumeFlux.Aclps = pi.*plumeFlux.rC.^2;
    plumeFlux.u_crit = u_crit;

    % Total mass flux at water breach
    plumeFlux.QpBreach = plumeFlux.rhop.*plumeFlux.u_0.*pi.*plumeFlux.r_p.^2;

    % Behavior, all runs
    plumeFlux.clps_regime = zeros(nn,1);  % Default buoyant plumes
    plumeFlux.clps_regime(logical(plumeFlux.clps)) = 1; % Total collapse
    plumeFlux.clps_regime(~plumeFlux.QClevel(:,2)) = 2; % Steam plume


%% Get un-processed key output params for emulation
% -  hm, hb, Qw, Qp, collapse regime (bouyant, partial, total, steam), collapse area
 %  --> Infer smooth hm at
 %          - collapse transition
 %          - beyond steam plume regime with scaling
 %  --> Infer smooth A_collapse

 % USES ONLY RAW OUTPUT AT PRESENT
 varNames = {'clps_regime','hm','qs0','qsC','rC','qw0','qwC'}; % Plot each of these as well
 
 % Plot parameters corresponding to the variables
plotLabels  = { 'Clps flag'
                'h_m (m)'
                'Q_{s0} (kg/s)'
                'Q_{sC} (kg/s)'
                'r_{clps} (m)'
                'Q_{w0} (kg/s)'
                'Q_{wC} (kg/s)'};
            
plotNames  = {  'Collapse Regime'
                'Max. height'
                'Particle flux (at breach)'
                'Particle flux (at collapse)'
                'Collapse radius'
                'Water flux (at breach)'
                'Water flux (at collapse)'};
coindex = [7 4 2 2 5 1 1];
axpos = [4 1 2 3 5 6 7];

 % Build raw data table
 rawDataTable = addvars(rawDataTable, ...
     plumeFlux.clps_regime,...
     plumeFlux.hm, ...
     plumeFlux.qs0, ...
     plumeFlux.qsC, ...
     plumeFlux.rC, ...
     plumeFlux.qw0, ...
     plumeFlux.qwC, ...
     'NewVariableNames',varNames);
 
 dataTable = rawDataTable; % create copy for processing into smooth fields
%% Get hm scaling for steam plume regime

% Get some key non-dimensional quantities
n_total = sum(dataTable{:,{'n_0','n_ec'}},2);
Ze_over_Rv       = plumeFlux.Ze./extrapVentRadius(plumeFlux.Qp);
QpBreach_over_Qp = plumeFlux.QpBreach./plumeFlux.Qp;

% 1) Get polynomial model for u_0 at breach for steam/failed plume regime
    % - A cool filter for linear drop off in u_0
    
    zeFilt = (Ze_over_Rv)>=1.6 & plumeFlux.clps_regime~=2;
    
    ft = fittype( 'poly12' ); % Fit linear in Ze, quadratic in n_total
    [xData, yData, zData] = prepareSurfaceData( Ze_over_Rv(zeFilt), n_total(zeFilt), plumeFlux.u_0(zeFilt) );
    % Set up fittype and options.

    % Fit model to data.
    [u0Mdl, gof(1)] = fit( [xData, yData], zData, ft );

% 2) Get linear model for r_0 (jet radius @ breach) for "
    % - Get total collapse indices, filtered to relatively large Ze/r_v
    r_0 = plumeFlux.r_p;
    u_0 = plumeFlux.u_0;
    Fr0 = plumeFlux.u_0./(9.81 * plumeFlux.r_p).^(1/2); % Calculate Froude # at water breach
    
    rFilt = zeFilt & plumeFlux.clps_regime>0; % Filter for collapse regime
    ft = fittype( 'poly11' ); % Bi_linear
    % Fit Ze/Rv vs log10(Q) vs r_0/r_vent (extrapolated)
    [xData, yData, zData] = prepareSurfaceData( Ze_over_Rv(rFilt),log10(plumeFlux.Qp(rFilt)),...
        r_0(rFilt)./extrapVentRadius(plumeFlux.Qp(rFilt)) );
    
    [r0Mdl, gof(2)] = fit( [xData, yData], zData, ft );
    

% 3) Get predictions for u_0, r_0, and Froude number for steam plume regime

    clp3 = plumeFlux.clps_regime==2;
    u_0_predict = u0Mdl(Ze_over_Rv(clp3), n_total(clp3));
    u_0_predict(u_0_predict<0) = 0;
    r_0_predict = r0Mdl(Ze_over_Rv(clp3), log10(plumeFlux.Qp(clp3))) .* extrapVentRadius(plumeFlux.Qp(clp3)); % Dimensionalize
    r_0_predict(r_0_predict<0) = NaN;
    
    Fr_0_predict = u_0_predict ./ (9.81 * r_0_predict).^(1/2);
    
% 4) Finally, get h_m for steam plume regime
    h_m_predict = 0.9 .* u_0_predict.^2 / 9.81;
    
    rawDataTable.hm(clp3) = h_m_predict;
    dataTable.hm(clp3)    = h_m_predict;

    r_0(clp3) = r_0_predict;
    Fr0_final = Fr0; Fr0_final(clp3) = Fr_0_predict;

 %% Calculate collapse regimes, proxy collapse frac

% Test some metrics for getting a smoothed collapse regime function

% Get a scaled marker size for plots based on total mass flux
mrng = 60;
msz1 = 30;
if strcmp(QScale,'log')
    msz = (log10(plumeFlux.Qp)-log10(min(plumeFlux.Qp)))./range(log10(plumeFlux.Qp)).*mrng + msz1;
else
    msz = (plumeFlux.Qp-min(plumeFlux.Qp))./range(plumeFlux.Qp).*mrng + msz1;
end
msz = msz.*mSizeScale;
co = get(0,'DefaultAxesColorOrder');


% Get and normalize data
x = table2array(rawDataTable(:,{'Q','Ze'}));
if strcmp(QScale,'log')
    x(:,1) = log10(x(:,1));
end
x(:,1) = normalize(x(:,1),'range');
x(:,2) = normalize(x(:,2),'range');

% 1 - get neighbourhood for regime to produce a "smoothed" regime curve
%      -> these will be used as weights for each point?
% 2 - get mean neighbourhood values for:
%       - regime
%       - hm, Qw0, Qs0, QwC, QsC
%       - A_clps?

% Get all neighbourhoods for all points
chiSqrDist = @(x,Z) sqrt((bsxfun(@minus,x,Z).^2)*distance_weights);  % chi-square distance metric
Idx = knnsearch(x,x,'Distance',chiSqrDist,'K',n_neighbours);

% Use matrix array of weights and indices to build weighted mean for all
% points based on their neighbourhood
regime_neighbourhood = rawDataTable.clps_regime(Idx); % Which regimes are in each point neighbourhood?
weights = regime_weights(regime_neighbourhood+1);
num_regimes = sum(diff(sort(regime_neighbourhood,2),1,2)~=0,2)+1; % How many regimes represented in each neighbourhood?
blend_regimes = num_regimes>1; % Take the neighbourhood mean only where more than one regime are present
for vi=1:length(varNames)
    % Take the arithmetic mean of the neighbourhood
    dataTable.(varNames{vi})(blend_regimes) = sum(weights(blend_regimes,:) ...
        .* dataTable.(varNames{vi})(Idx(blend_regimes,:)), 2, 'omitnan')...
        ./ sum(weights(blend_regimes,:), 2, 'omitnan');
end

%% Make a scaled data table to better highlight physical relationships
scaledDataTable = dataTable;

newVars = {
    'Q'
    'logQ'
    'Ze_over_Rv'
    'T'
    'n_total'
    'Rv'
    'a_over_Rv'
    'D'
    'clps_regime'
    'hm_over_Q14'
    'r0'
    'rC_over_r0'
    'rC_over_Rv'
    'Fr0'
    'qs0_over_Q'
    'qsC_over_Q'
    'qw0_over_Q'
    'qwC_over_Q'
    };

scaledDataTable = table(...
                    dataTable.Q, ...
                    log10(dataTable.Q), ...
                    Ze_over_Rv,...
                    dataTable.T,...
                    n_total,...
                    extrapVentRadius(dataTable.Q),...
                    dataTable.conduit_radius./extrapVentRadius(dataTable.Q),...
                    dataTable.D,...
                    dataTable.clps_regime,...
                    dataTable.hm./dataTable.Q.^(1/4),...
                    r_0,...
                    dataTable.rC./r_0,...
                    dataTable.rC./extrapVentRadius(dataTable.Q),...
                    Fr0_final,...
                    dataTable.qs0./dataTable.Q,...
                    dataTable.qsC./dataTable.Q,...
                    dataTable.qw0./dataTable.Q,...
                    dataTable.qwC./dataTable.Q,...
                    'VariableNames',newVars);

scaledRawDataTable = table(...
                    rawDataTable.Q, ...
                    log10(rawDataTable.Q), ...
                    Ze_over_Rv,...
                    rawDataTable.T,...
                    n_total,...
                    extrapVentRadius(rawDataTable.Q),...
                    rawDataTable.conduit_radius./extrapVentRadius(rawDataTable.Q),...
                    rawDataTable.D,...
                    rawDataTable.clps_regime,...
                    rawDataTable.hm./rawDataTable.Q.^(1/4),...
                    r_0,...
                    rawDataTable.rC./r_0,...
                    rawDataTable.rC./extrapVentRadius(rawDataTable.Q),...
                    Fr0_final,...
                    rawDataTable.qs0./rawDataTable.Q,...
                    rawDataTable.qsC./rawDataTable.Q,...
                    rawDataTable.qw0./rawDataTable.Q,...
                    rawDataTable.qwC./rawDataTable.Q,...
                    'VariableNames',newVars);

% ----- END OF PROCESSING -----
%% ========================================================================
%                               PLOTTING
%  ========================================================================
%% Plot imputed h_m scaling for steam plume/fountain regime
if makePlots
    nr = 2;
    nc = 2;
    
    figure
    % Showing scaling fit for collapsing regime
    scatter(0.9*Fr0(rFilt).^2, plumeFlux.hm(rFilt)./r_0(rFilt), 30, n_total(rFilt) )
    hold on
    plot([0 max(0.9*Fr0(rFilt).^2)], [0 max(0.9*Fr0(rFilt).^2)], '--k')
    scatter(0.9*Fr_0_predict.^2, h_m_predict./r_0_predict, 30, 'x')
    xlabel('0.9 Fr_0^2')
    ylabel('h_m/r_0')
    
    figure

    % Getting u_0
    subplot(nr,nc,1)
    scatter3(Ze_over_Rv(zeFilt), n_total(zeFilt), plumeFlux.u_0(zeFilt) ,20, dataTable.T(zeFilt), 'filled'); colormap(pasteljet);
    hold on
    scatter3(Ze_over_Rv(clp3), n_total(clp3), u_0_predict ,20, dataTable.T(clp3),'x')
    xlabel('Z_w/R_v'); ylabel('u_0'); zlabel('u_0 (predicted)')
    cb=colorbar;cb.Label.String='T_0 (K)';
    set(gca,'FontSize',fs)
    
    % Getting r_0
    subplot(nr,nc,2)
    scatter3(Ze_over_Rv(rFilt), log10(plumeFlux.Qp(rFilt)), r_0(rFilt),...
        20, n_total(rFilt), 'filled' );
    hold on
    scatter3(Ze_over_Rv(clp3), log10(plumeFlux.Qp(clp3)), r_0_predict ,20,  n_total(clp3), 'x')
    xlabel('Z_e/extrap(R_v)'); ylabel('log_{10}(Q_p)'); zlabel('r_0 (predicted)')
    cb=colorbar;cb.Label.String='n_0 + n_{ec}';
    title('Linear fit for mass flux at breach')
    set(gca,'FontSize',fs)
    
    subplot(nr,nc,3)
    scatter3(u_0, r_0, Fr0, 30, plumeFlux.Qp, 'filled')
    hold on
    scatter3(u_0_predict, r_0_predict, Fr_0_predict, 30 , plumeFlux.Qp(clp3), 'x')
    xlabel('u_0'); ylabel('r_0'); zlabel('Fr_0')
    cb=colorbar;cb.Label.String='Q_p (kg/s)';
    set(gca,'FontSize',fs)
    
    subplot(nr,nc,4)
    scatter(0.9.*plumeFlux.u_0(rFilt).^2./9.81, plumeFlux.hm(rFilt), 30, 'filled')
    hold on
    scatter(0.9.*u_0_predict.^2./9.81, h_m_predict, 30, 'x')
    set(gca,'FontSize',fs)

    
    % Getting fountain scaling constant C_f...
end

%% KNN weighting plots
if makePlots
% ------ Make a sample knn search plot to show representative neighbourhoods ------
    nsamps = 20;
    samp_idx = randsample(1:nn,nsamps);

    
    y = x(samp_idx,:);
    Idx = knnsearch(x,y,'Distance',chiSqrDist,'K',n_neighbours);
    
    figure
    hold(gca,'on')
    scatter3ByRegime(x(:,1), x(:,2), plumeFlux.clps_regime,...
                msz, plumeFlux.clps_regime, [0.5 0.5 0.5]);
    view([0 0 1])
    plot3(y(:,1),y(:,2),plumeFlux.clps_regime(samp_idx),'ok','MarkerFaceColor','k')
    Qnorm = x(:,1); ZeNorm = x(:,2);
    plot3(Qnorm(Idx)',ZeNorm(Idx)',plumeFlux.clps_regime(Idx)','o')
    xlabel('Normalized Q')
    ylabel('Normalized Z_w')
    zlabel('Collapse regime')
    grid on
    set(gca,'FontSize',fs)
    
% ------ Show knn weighted mean output -------

    regimes = [0 1 2];
    regNames = {'Buoyant','Collapse','No breach'};
    symbols = ['o','x','s'];
    faceAlpha = [1 0 0];
    calpha = [0.7 0.95 0.5];
    
    
    nr = 2;
    nc = 4;
    dx = 0.04;
    dy = 0.09;
    xsz = [1 1 1 1.5];
    ysz = [1 1];
    figpos = [50 50 1600 800];
    ppads = [0.08 0.03 0.08 0.03];
    ppads2 = [0.08 0.03 0.25 0.25];
%     viewangle = [0 0 1]; 
    viewangle = [-1 0.4 0.6];

    figure('position',figpos)
    for ai = 1:length(varNames)
        if strcmp(varNames{ai},'clps_regime')
            ax(ai)=tightSubplot(1,nc,axpos(ai),dx,dy,ppads2,xsz,ysz);
            set(gca,'YDir','reverse')
            set(gca,'ZTick',[0 1 2],'ZTickLabel',{'Buoyant','Collapsing','Steam Plume'})
        else
            ax(ai)=tightSubplot(nr,nc,axpos(ai),dx,dy,ppads,xsz,ysz);
        end
        hold on
        
        handles = scatter3ByRegime(rawDataTable.Q, rawDataTable.Ze, rawDataTable.(varNames{ai}),...
            msz, rawDataTable.clps_regime, [0.7 0.7 0.7] );

        handles2 = scatter3ByRegime(dataTable.Q, dataTable.Ze, dataTable.(varNames{ai}),...
            msz, rawDataTable.clps_regime, co(coindex(ai),:) );
        for fi=1:length(handles)
            handles(fi).DisplayName = [handles(fi).DisplayName ': raw'];
            handles2(fi).DisplayName = [handles2(fi).DisplayName ': KNN-weighted'];
        end
        
        title(plotNames{ai})
        if strcmp(varNames{ai},'clps_regime')
            legend('location','southeast')
        end
        if strcmp(QScale,'log')
            set(gca,'XScale','log')
        end
        ylabel('Z_w (m)')
        xlabel('Q (kg/s)')
        view(viewangle)
        grid on
        zlabel(plotLabels{ai})
        set(gca,'FontSize',fs)
    end
end

%% Scaled KNN weighting plots
if makePlots

% ------ Show knn weighted mean output -------
    plotVarNames = { 
        'clps_regime'
        'hm_over_Q14'
        'qs0_over_Q'
        'qsC_over_Q'
        'rC_over_Rv'
        'qw0_over_Q'
        'qwC_over_Q'
        };

    scaledLabels  = { 'Collapse flag'
                'h_m/Q^{1/4}'
                'Q_{s0}/Q'
                'Q_{sC}/Q'
                'r_{clps}/r_v'
                'Q_{w0}/Q'
                'Q_{wC}/Q'};
    
    regimes = [0 1 2];
    regNames = {'Buoyant','Collapse','No breach'};
    symbols = ['o','x','s'];
    faceAlpha = [1 0 0];
    calpha = [0.7 0.95 0.5];
    
    
    nr = 2;
    nc = 4;
    dx = 0.04;
    dy = 0.09;
    xsz = [1 1 1 1.5];
    ysz = [1 1];
    figpos = [50 50 1600 800];
    ppads = [0.08 0.03 0.08 0.03];
    ppads2 = [0.08 0.03 0.25 0.25];
%     viewangle = [0 0 1]; 
    viewangle = [-1 0.4 0.6];

    figure('position',figpos)
    for ai = 1:length(plotVarNames)
        if strcmp(plotVarNames{ai},'clps_regime')
            ax(ai)=tightSubplot(1,nc,axpos(ai),dx,dy,ppads2,xsz,ysz);
            set(gca,'YDir','reverse')
            set(gca,'ZTick',[0 1 2],'ZTickLabel',{'Buoyant','Collapsing','Steam Plume'})
        else
            ax(ai)=tightSubplot(nr,nc,axpos(ai),dx,dy,ppads,xsz,ysz);
        end
        hold on
        
        handles = scatterByRegime(scaledRawDataTable.Ze_over_Rv, scaledRawDataTable.(plotVarNames{ai}),...
            msz, scaledRawDataTable.clps_regime, [0.7 0.7 0.7] );

        handles2 = scatterByRegime(scaledDataTable.Ze_over_Rv, scaledDataTable.(plotVarNames{ai}),...
            msz, scaledRawDataTable.clps_regime, co(coindex(ai),:) );
        for fi=1:length(handles)
            handles(fi).DisplayName = [handles(fi).DisplayName ': raw'];
            handles2(fi).DisplayName = [handles2(fi).DisplayName ': KNN-weighted'];
        end
        
        title(plotNames{ai})
        if strcmp(plotVarNames{ai},'clps_regime')
            legend('location','southeast')
        end

        xlabel('Z_w/r_v')
        grid on
        ylabel(scaledLabels{ai})
        set(gca,'FontSize',fs)
        xlim([0 8])
    end
end
%% Show raw and transformed data with and without scaling - hm and qw0 only
if makePlots
    figpos = [50 50 1500 600];
    cbpos = [0.83 0.015];
    nr = 1;
    nc = 2;
    dx = 0.09;
    ppads = [0.07 0.18 0.2 0.07];
    cmap = redblue;

    % -------- H_m ---------
    figure('position',figpos)
    
    % Unscaled
    sax(1) = tightSubplot(nr,nc,1,dx,[],ppads);
    % Raw output
    sp(1) = scatter(rawDataTable.Ze(~clp3), rawDataTable.hm(~clp3)/1e3,msz(~clp3), ...
        dataTable.clps_regime(~clp3),'MarkerEdgeColor','k','MarkerEdgeAlpha',0.4);
    hold on
    sp(2) = scatter(rawDataTable.Ze(clp3), rawDataTable.hm(clp3)/1e3,msz(clp3), ...
        'x','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4);
    sp(3) = scatter(dataTable.Ze, dataTable.hm/1e3,msz, ...
        dataTable.clps_regime,'filled','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4,...
        'MarkerFaceAlpha',0.5);
    xlabel('$\displaystyle Z_e$ (m)','interpreter','latex')
    ylabel('$\displaystyle h_m$ (km)','interpreter','latex')
    title('Plume height')
    set(gca,'FontSize',fs)
    
    % Legend setup
    if strcmp(QScale,'log')
        [legh,legSz] = getScatterSizeLegend(msz,sp(3),log10(plumeFlux.Qp),sax(1));
        szLabel = '$Q_p = 10^%.0f$ kg/s';
    else
        [legh,legSz] = getScatterSizeLegend(msz,sp(3),plumeFlux.Qp,sax(1));
        szLabel = '$Q_p = %.1e$ kg/s';
    end
    for sn = 1:length(legSz)
        leglabels{sn} = sprintf(szLabel,legSz(sn));
    end
    lg = legend([sp legh],['Model raw output' 'Imputed plume heights' 'KNN-weighted' leglabels],'Location','northeast','FontSize',fs-1,'Interpreter','Latex');
    lg.FontSize = fs-1;
    
    % Scaledƒ
    sax(2) = tightSubplot(nr,nc,2,dx,[],ppads);
%     sp(1) = scatter(Ze_over_Rv(~clp3), rawDataTable.hm(~clp3)./sqrt(r0(~clp3)),msz(~clp3), ...
%         dataTable.clps_regime(~clp3),'MarkerEdgeColor','k','MarkerEdgeAlpha',0.4);
%     hold on
%     sp(2) = scatter(Ze_over_Rv(clp3), rawDataTable.hm(clp3)./sqrt(r0(clp3)),msz(clp3), ...
%         dataTable.clps_regime(clp3),'x','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4);

    sp(1) = scatter(Ze_over_Rv(~clp3), rawDataTable.hm(~clp3)./dataTable.Q(~clp3).^(1/4),msz(~clp3), ...
        dataTable.clps_regime(~clp3),'MarkerEdgeColor','k','MarkerEdgeAlpha',0.4);
    hold on
    sp(2) = scatter(Ze_over_Rv(clp3), rawDataTable.hm(clp3)./dataTable.Q(clp3).^(1/4),msz(clp3), ...
        dataTable.clps_regime(clp3),'x','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4);
    sp(3) = scatter(Ze_over_Rv, dataTable.hm./dataTable.Q.^(1/4),msz, ...
        dataTable.clps_regime,'filled','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4,...
        'MarkerFaceAlpha',0.5);
    title('Plume height: Scaled')
    xlabel('$\displaystyle\frac{Z_e}{R_v}$','interpreter','latex')
    ylabel('$\displaystyle\frac{h_m}{Q_p^{1/4}}$','interpreter','latex','rotation',0,'HorizontalAlignment','right')
    cb = colorbar('location','eastoutside');
    cb.Label.String = 'Continuous Collapse Regime';
    cb.Ticks = [0 1 2];
    cb.TickLabels = {'Buoyant','Total Collapse','Steam Plume'};
    cb.Position([1 3]) = cbpos;
    set(gca,'FontSize',fs)
    colormap(cmap)

    % -------- Qw_0 ---------
    figure('position',figpos)
    
    clear sp
    % Unscaled
    sax(1) = tightSubplot(nr,nc,1,dx,[],ppads);
    % Raw output
    sp(1) = scatter(rawDataTable.Ze, rawDataTable.qw0,msz, ...
        dataTable.clps_regime,'x','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4);
    hold on
    sp(2) = scatter(dataTable.Ze, dataTable.qw0,msz, ...
        dataTable.clps_regime,'filled','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4,...
        'MarkerFaceAlpha',0.5);
    xlabel('$\displaystyle Z_e$ (m)','interpreter','latex')
    ylabel('$\displaystyle Q_{w0}$ (kg/s)','interpreter','latex')
    title('Water flux at water surface')
    set(gca,'FontSize',fs)

    % Legend setup
    if strcmp(QScale,'log')
        [legh,legSz] = getScatterSizeLegend(msz,sp(2),log10(plumeFlux.Qp),sax(1));
        szLabel = '$Q_p = 10^%.0f$ kg/s';
    else
        [legh,legSz] = getScatterSizeLegend(msz,sp(2),plumeFlux.Qp,sax(1));
        szLabel = '$Q_p = %.1e$ kg/s';
    end
    for sn = 1:length(legSz)
        leglabels{sn} = sprintf(szLabel,legSz(sn));
    end
    lg = legend([sp legh],['Model raw output' 'KNN-weighted' leglabels],'Location','northeast','FontSize',fs-1,'Interpreter','Latex');
    lg.FontSize = fs-1;

    % Scaled
    sax(2) = tightSubplot(nr,nc,2,dx,[],ppads);
    sp(1) = scatter(Ze_over_Rv(~clp3), rawDataTable.qw0(~clp3)./rawDataTable.Q(~clp3), msz(~clp3), ...
        'x','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4);

    hold on
    sp(2) = scatter(Ze_over_Rv(clp3), rawDataTable.qw0(clp3)./dataTable.Q(clp3),msz(clp3), ...
        dataTable.clps_regime(clp3),'x','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4);
    sp(3) = scatter(Ze_over_Rv, dataTable.qw0./dataTable.Q,msz, ...
        dataTable.clps_regime,'filled','MarkerEdgeColor','k','MarkerEdgeAlpha',0.4,...
        'MarkerFaceAlpha',0.5);
    title('Water flux at water surface: Non-dimensionalized')
    xlabel('$\displaystyle\frac{Z_e}{R_v}$','interpreter','latex')
    ylabel('$\displaystyle\frac{Q_{w0}}{Q_p}$','interpreter','latex')
    xlim([0 8])
    cb = colorbar('location','eastoutside');
    cb.Label.String = 'Continuous Collapse Regime';
    cb.Ticks = [0 1 2];
    cb.TickLabels = {'Buoyant','Total Collapse','Steam Plume'};
    cb.Position([1 3]) = cbpos;
    set(gca,'FontSize',fs)
    colormap(cmap)
    
    
end
%% Showing just the raw data - plot no longer used but kept for reference
if false %makePlots
    
% ----------- The first set plots raw output of key params: ----------
    plotLabels  = {'Clps flag','h_m (m)','Q_{s0} (kg/s)','Q_{sC} (kg/s)','A_{clps} (m^2)','Q_{w0} (kg/s)','Q_{wC} (kg/s)'};
    plotNames  = {'Collapse Regime', 'Max. height','Particle flux (at breach)',...
        'Particle flux (at collapse)','Collapse area','Water flux (at breach)',...
        'Water flux (at collapse)'};
    coindex = [7 4 2 2 5 1 1];
    axpos = [4 1 2 3 5 6 7];

    regimes = [0 1 2];
    regNames = {'Buoyant','Collapse','No breach'};
    symbols = ['o','x','s'];
    faceAlpha = [1 0 0];
    calpha = [0.7 0.95 0.5];
    
    
    nr = 2;
    nc = 4;
    dx = 0.04;
    dy = 0.09;
    xsz = [1 1 1 1.5];
    ysz = [1 1];
    figpos = [50 50 1600 800];
    ppads = [0.08 0.03 0.08 0.03];
    ppads2 = [0.08 0.03 0.25 0.25];
%     viewangle = [0 0 1]; 
    viewangle = [-1 -0.8 0.5];


    figure('position',figpos)
    for ai = 1:length(varNames)
        if strcmp(varNames{ai},'clps_regime')
            ax(ai)=tightSubplot(1,nc,axpos(ai),dx,dy,ppads2,xsz,ysz);
        else
            ax(ai)=tightSubplot(nr,nc,axpos(ai),dx,dy,ppads,xsz,ysz);
        end
        hold on
        
        scatter3ByRegime(rawDataTable.Q, rawDataTable.Ze, plumeFlux.(varNames{ai}),...
            msz, plumeFlux.clps_regime, co(coindex(ai),:) )

        title(plotNames{ai})
        legend
        ylabel('Z_w (m)')
        xlabel('Q (kg/s)')
        view(viewangle)
        grid on
        zlabel(plotLabels{ai})
    end


end

end

function fi = scatter3ByRegime(x,y,z,msz,regime,color)
    regimes = [0 1 2];
    regNames = {'Buoyant','Collapse','No breach'};
    symbols = ['o','x','s'];
    faceAlpha = [1 0 0];
    calpha = [0.7 0.95 0.5];
    hold(gca,'on')

    for ri = 1:length(regimes)
        X = x(regime==regimes(ri));
        Y = y(regime==regimes(ri));
        Z = z(regime==regimes(ri));
        fi(ri) = scatter3(X,Y,Z,msz(regime==regimes(ri)),...
            rgba2rgb(color,calpha(ri)),symbols(ri),...
            'MarkerFaceAlpha',faceAlpha(ri),'DisplayName',regNames{ri});
    end
end

function fi = scatterByRegime(x,y,msz,regime,color)
    regimes = [0 1 2];
    regNames = {'Buoyant','Collapse','No breach'};
    symbols = ['o','x','s'];
    faceAlpha = [1 0 0];
    calpha = [0.7 0.95 0.5];
    hold(gca,'on')

    for ri = 1:length(regimes)
        X = x(regime==regimes(ri));
        Y = y(regime==regimes(ri));
        fi(ri) = scatter(X,Y,msz(regime==regimes(ri)),...
            rgba2rgb(color,calpha(ri)),symbols(ri),...
            'MarkerFaceAlpha',faceAlpha(ri),'DisplayName',regNames{ri});
    end
end