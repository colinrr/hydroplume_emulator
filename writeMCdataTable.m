
% This script takes the katla Monte Carlo sim set and outputs a clean data table
% for statistical use and model emulation
clearvars -except dat

dDir = '/Users/crrowell/Kahuna/data/gvolc-meghan/MonteCarloOutput/';

MCfiles = {
%            'KatlaHydro_v8_noLd_2022-10-01_N10000.mat'
           'KatlaHydro_v8_noLd_2024-06-22_N10000.mat'
          };
      
figDir = '~/Kahuna/phd-docs/research/katla/manuscript_2024/mat-figs/';
    makePlots = true;
    writePlots = true;
%%

mainMCfile = fullfile(dDir,MCfiles{1});
summaryMCfile = fullfile(dDir,['outputSummary_' MCfiles{1}]);
loadif(mainMCfile,'dat');

[dataTable,fixedVars, MCvars] = getMCoutputTable(mainMCfile, summaryMCfile, dat, [], makePlots, writePlots);

% Setting units tside teh function for now since they are generally variable in MC
% input
 dataTable.Properties.VariableUnits = {'K' 'm' '-' '-' 'kg/s' 'm' '-' '-' 'm' 'kg/s' 'kg/s' 'm' 'kg/s' 'kg/s'  };


[~,oname,~] = fileparts(mainMCfile);


% parquetwrite(fullfile(dDir,[oname '.parquet']),dataTable)

%% Figure output

if false %writePlots
    figure(5)
    printpdf('Plume-height-scaled',figDir,[18 10])
    
    figure(6)
    printpdf('Water-flux-scaled',figDir,[18 10])
end