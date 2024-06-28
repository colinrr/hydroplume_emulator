
% This script takes the katla Monte Carlo sim set and outputs a clean data table
% for statistical use and model emulation
clearvars -except dat

dDir = '/Users/crrowell/Kahuna/data/gvolc-meghan/MonteCarloOutput/';
outputDir = '/Users/crrowell/code/research-projects/katla/katlaPlumePhysicsEmulator/data/';

MCfiles = {
%            'KatlaHydro_v8_noLd_2022-10-01_N10000.mat'
           'KatlaHydro_v8_noLd_2024-06-22_N10000.mat'
          };
      
writeData = false;

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

%% Write output files
if writeData
    [~,oname,~] = fileparts(mainMCfile);
    parquetwrite(fullfile(outputDir,[oname '.parquet']),dataTable)
    save(fullfile(outputDir,'fixed_MC_vars'),'fixedVars')
    parquetwrite(fullfile(outputDir,'rand_MC_vars'),MCvars)
end
%% Figure output

if writePlots
    figure(5)
    printpdf('Plume-height-scaled',figDir,[18 8])
    
    figure(6)
    printpdf('Water-flux-scaled',figDir,[18 8])
end

%% Test dummy struct for python

testStruct = struct('scalar',5,'vector',1:5,'matrix',eye(5),'char','faafo');
testStruct.cell = {'faafo' ,2};
save(fullfile(outputDir,'dummy_struct'),'testStruct')