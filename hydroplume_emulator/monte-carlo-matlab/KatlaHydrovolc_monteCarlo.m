% ========= Control Script for Katla Monte Carlo simulations ==============
% NOTE: This script is included here for reference in I/O parameters. 
%       Running it requires the hydroVolc plume model dependency.
%        -> Model citation: (Rowell et al 2022, Frontiers in Earth Science)

clear all; close all

% ==== Directories and flags =====
% C
% dDir = '/Users/crrowell/Kahuna/data/gvolc-meghan/';
% codeDir = '/Users/crrowell/code/research-projects/glaciovolc/glaciovolc-dev/';

% FB
% codeDir = 'C:\Users\crowell\Documents\GitHub\glaciovolc-dev\';
% dDir = 'D:\Kahuna\data\gvolc-meghan\';

% SJ
codeDir = 'C:\Users\crowell\Documents\GitHub\glaciovolc\glaciovolc-dev\';
dDir = 'C:\Users\crowell\Kahuna\data\gvolc-meghan\';

% ------------------
oDir = fullfile(dDir,'MonteCarloOutput/');


runControlMC    = false;
runHydroMC      = true;
writeOutput     = true;

addpath(genpath(codeDir))


%% ==================== HYDRO RUN SETUP ======================
% oNameH = 'KatlaHydro_v8_noLd_%s_N%i'; % Original Oct-2022 BullVolc submission set

% --------- 2024 revisions - new run set ------------
oNameH = 'KatlaHydro_v8_noLd_%s_N%i'; % 2024 Run 1 with new parameter ranges

% LAT LON
% -20.545186, -175.390147

N = 10000; % Number of runs (use for all Q)
% N = 1000; % Use for fixed Q
ncores = 8; % SJ

% ============ Constant params ============
% ------ CONDUIT ------

cI.proxy    = true;
cI.T_ec     = 273.15; %
cI.rho_rock = 2800;

cI.atmo     = fullfile(dDir,'ERA5/Katla_avg_atmo_Oct12_1979-2005.mat');
cI.vh0      = 850; % Eyeball from Magnusson et al 2021 DEM
cI.rho_melt = 2600;

% ------ MWI/PLUME ------
pI.Tw0      = cI.T_ec;
pI.useDecompressLength = false; % Turn off Ld to be a bit closer to a basalt fountain type

% Glass transition data:
% Giordano et al 2005, basalt assuming ~0.5wt% residual H20
pI.T_g       = 870; % K
pI.T_g_rng   = 50;
pI.phiFrag_mu   = 3; % Rough output sizes for second mode of Jonsdottir 2015
pI.phiFrag_sig  = 1.25;

% ========== Monte carlo params ===========
% ------ CONDUIT ------
MC.cI.T.dist  = 'uniform';
MC.cI.T.range = [1375 1525]; % Basalt, 1200-1270 C

MC.cI.Zw.dist  = 'uniform';
MC.cI.Zw.range = [0 500];

MC.cI.n_0.dist  = 'uniform';
MC.cI.n_0.range = [0.005 0.025];

MC.cI.n_ec.dist  = 'uniform';
MC.cI.n_ec.range = [0.0 0.2]; % Allow high conduit water influx

% MC.cI.Q.dist = 'uniform';
% MC.cI.Q.range = [3e7 9e7];

% ---- Special conduit inputs. Q always first -----
% Log mass flux variation - calculates a uniform dist'n in log space
MC.cI.logQ.dist  = 'uniform';
MC.cI.logQ.range = [6 8];

% Conduit radius variation (fraction around lookup value) (use AREA??)
MC.cI.a_var.dist  = 'normal';
MC.cI.a_var.range = [1 0.15];


% ------ MWI/PLUME ------
MC.pI.D.dist      = 'discrete';
MC.pI.D.range     = [2.7 2.8]; % A bit coarser for basalt?


%% =================== HYDRO: Do the thing =====================

[summ,randPars,dat] = hydroVolcMC(cI,pI,MC,N,ncores);

%%
if writeOutput
    oFile = fullfile(oDir,sprintf(oNameH,datestr(now,'yyyy-mm-dd'),N));
    fprintf('Saving output:\n\t%s\n',oFile)
    save(oFile,'pI','cI','MC','summ','randPars','dat','-v7.3')
    [qA,~,oFile] = getHydroVolcSweepSummary(oFile,writeOutput);  % Hydro
end

%% =================== SUMMARY GATHERS =====================
% [qA,~,oFile] = getHydroVolcSweepSummary(oFileC,true); % Control
